"""Online adaptation loop tying replay, builder and promotion together.

Order for a transition ending at t+1 (spec 12.2):
  1. predict with the versions available before the target arrives, and save it,
  2. receive and validate the target, then score the saved predictions,
  3. admit the causal windows to replay,
  4. train a builder later, under the credit budget,
  5. consider promotion only on separate future data and anchors.

The robot is unaffected unless shadow=False, and even then a new version is installed
only at a planning boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from ..models.adapted_predictor import AdaptedPredictor
from .losses import token_metric
from .online_builder import OnlineBuilder, OnlineConfig
from .promotion import (
    GateDecision,
    GateInput,
    ModelHub,
    PromotionConfig,
    RollbackMonitor,
    action_probe,
    adapters_finite,
    evaluate_gate,
)
from .replay import CreditLedger, FeatureCache, ReplayBuffer, collate_windows
from .windows import Episode, WindowSpec, dense_starts, window_is_valid


def live_batch(
    episode: Episode,
    cache: FeatureCache,
    spec: WindowSpec,
    end_tick: int,
    context: int,
    planned_commands: np.ndarray,
    view_ids: list[int],
    device,
) -> dict:
    """Context ending at end_tick plus the command block about to be issued.

    Reads nothing after end_tick except the commands themselves, so the prediction is
    causal by construction. planned_commands: [stride, 6].
    """
    start = end_tick - (context - 1) * spec.stride
    if start < 0 or planned_commands.shape != (spec.stride, 6):
        raise ValueError("context or command block does not fit")
    keyframes = np.arange(start, end_tick + 1, spec.stride)
    first = max(start - spec.history_ticks, 0)
    while first < start and episode.segment[first] != episode.segment[end_tick]:
        first += 1
    pad = spec.history_ticks - (start - first)
    issued = np.concatenate(
        [np.repeat(episode.position[start : start + 1], pad, axis=0), episode.issued[first:end_tick], planned_commands]
    )
    valid = np.concatenate(
        [np.zeros(pad, dtype=bool), episode.command_valid[first:end_tick], np.ones(spec.stride, dtype=bool)]
    )
    dt = np.full(issued.shape[0], spec.tick_s)
    dt[pad : pad + end_tick - first] = episode.time_s[first + 1 : end_tick + 1] - episode.time_s[first:end_tick]
    features = [cache.get(episode.encoder_hash, episode.episode_id, int(t)) for t in keyframes]
    return {
        "features": torch.stack(features)[None].to(device),
        "position": torch.as_tensor(episode.position[keyframes], dtype=torch.float32, device=device)[None],
        "issued": torch.as_tensor(issued, dtype=torch.float32, device=device)[None],
        "dt": torch.as_tensor(dt, dtype=torch.float32, device=device)[None],
        "command_valid": torch.as_tensor(valid, device=device)[None],
        "view_ids": torch.as_tensor([view_ids], dtype=torch.long, device=device),
    }


def context_is_valid(episode: Episode, spec: WindowSpec, end_tick: int, context: int) -> bool:
    """A live context, like a replay window, must not cross a reset or an invalid frame."""
    start = end_tick - (context - 1) * spec.stride
    if start < 0 or end_tick >= episode.length:
        return False
    keyframes = np.arange(start, end_tick + 1, spec.stride)
    return bool(
        (episode.segment[keyframes] == episode.segment[end_tick]).all()
        and episode.position_valid[keyframes].all()
        and episode.frame_valid[keyframes].all()
    )


@dataclass
class PendingPrediction:
    target_tick: int
    target_segment: int
    predictions: dict[str, torch.Tensor]
    joints: np.ndarray
    version_keys: dict[str, str]


@dataclass
class Scores:
    serving: list[float] = field(default_factory=list)
    challenger: list[float] = field(default_factory=list)
    serving_joint: list[float] = field(default_factory=list)


class OnlineRunner:
    def __init__(
        self,
        hub: ModelHub,
        replay: ReplayBuffer,
        credits: CreditLedger,
        cache: FeatureCache,
        spec: WindowSpec,
        view_ids: dict[str, list[int]],
        online: OnlineConfig,
        promotion: PromotionConfig,
        batch_size: int = 16,
        microbatch: Optional[int] = None,
        regime: int = 0,
        device="cpu",
        clock: Callable[[], float] = time.monotonic,
        seed: int = 0,
    ):
        self.hub = hub
        self.replay = replay
        self.credits = credits
        self.cache = cache
        self.spec = spec
        self.view_ids = view_ids
        self.promotion = promotion
        self.batch_size = batch_size
        self.microbatch = microbatch
        self.regime = regime
        self.device = device
        self.clock = clock
        self.builder = OnlineBuilder(hub.builder, hub.reference, online, seed)
        self.episodes: dict[str, Episode] = {}
        self.next_start: dict[str, int] = {}
        self.admitted_total = 0
        self.updates_at_snapshot = 0
        self.scored_total = 0
        self.rollback_due: Optional[str] = None
        self.pending: list[PendingPrediction] = []
        self.scores = Scores()
        self.rollback_monitor = RollbackMonitor(promotion)
        self.anchor_groups: Optional[Callable[[AdaptedPredictor, AdaptedPredictor], dict]] = None
        self.faults: list[str] = []
        self.history: list[dict] = []

    def register_episode(self, episode: Episode) -> None:
        self.episodes[episode.episode_id] = episode
        self.next_start.setdefault(episode.episode_id, 0)

    @torch.no_grad()
    def predict(self, episode: Episode, end_tick: int, context: int, planned_commands: np.ndarray) -> bool:
        """Step 1: save predictions for frame end_tick + stride before it exists.

        Returns False when the context is invalid, as just after a reset.
        """
        if not context_is_valid(episode, self.spec, end_tick, context):
            return False
        batch = live_batch(
            episode, self.cache, self.spec, end_tick, context, planned_commands, self.view_ids[episode.episode_id], self.device
        )
        predictions = {"serving": self.hub.serving.rollout(batch, context, 1).cpu()}
        keys = {"serving": self.hub.serving.version}
        if self.hub.challenger is not None:
            predictions["challenger"] = self.hub.challenger.rollout(batch, context, 1).cpu()
            keys["challenger"] = self.hub.challenger.version
        if self.hub.previous is not None:
            predictions["previous"] = self.hub.previous.rollout(batch, context, 1).cpu()
        joints = self.hub.serving.predicted_positions(batch, context, 1)[0, 0].cpu().numpy()
        self.pending.append(
            PendingPrediction(end_tick + self.spec.stride, int(episode.segment[end_tick]), predictions, joints, keys)
        )
        return True

    def resolve(self, episode: Episode, tick: int) -> None:
        """Step 2: score predictions whose target frame has arrived and passed checks.

        A boundary or invalid target drops the prediction, since a reset is not a
        dynamics outcome. A rollback trigger is held for the next planning boundary.
        """
        remaining = []
        for item in self.pending:
            if item.target_tick > tick:
                remaining.append(item)
                continue
            usable = (
                episode.segment[item.target_tick] == item.target_segment
                and episode.frame_valid[item.target_tick].all()
                and episode.position_valid[item.target_tick]
            )
            if not usable:
                continue
            target = self.cache.get(episode.encoder_hash, episode.episode_id, item.target_tick)[None, None]
            losses = {k: float(token_metric(v, target.cpu())) for k, v in item.predictions.items()}
            self.scored_total += 1
            self.scores.serving.append(losses["serving"])
            self.scores.serving_joint.append(float(np.abs(item.joints - episode.position[item.target_tick]).mean()))
            challenger_live = self.hub.challenger is not None and item.version_keys.get("challenger") == self.hub.challenger.version
            if challenger_live:
                self.scores.challenger.append(losses["challenger"])
            if "previous" in losses:
                if self.rollback_monitor.observe(float(episode.time_s[item.target_tick]), losses["serving"], losses["previous"]):
                    self.rollback_due = "sustained prequential regression"
        self.pending = remaining

    def admit_windows(self, episode: Episode) -> int:
        """Step 3: admit every window that has become fully valid."""
        count = 0
        first = self.next_start[episode.episode_id]
        last = episode.length - self.spec.span_ticks
        for start in dense_starts(episode, self.spec, first):
            self.replay.add((episode.episode_id, start), self.regime)
            count += 1
        self.next_start[episode.episode_id] = max(first, last)
        self.credits.admit(count)
        self.admitted_total += count
        return count

    def maybe_train(self) -> bool:
        """Step 4: one builder update if the credit, rate and duty budgets allow it."""
        now = self.clock()
        if not self.credits.can_update(now):
            return False
        windows, flags = self.replay.sample(self.batch_size, self.regime)
        batch = collate_windows(windows, self.episodes, self.cache, self.spec, self.view_ids, self.device)
        self.credits.spend(now)
        started = time.perf_counter()
        result = self.builder.update(batch, flags, microbatch=self.microbatch)
        self.credits.record_busy(time.perf_counter() - started)
        if result.status != "ok":
            self.credits.refund()
            self.faults.append(f"nonfinite update at credit {self.credits.spent}")
            return False
        self.history.append({"loss": result.loss, "cmd_tf": result.cmd_tf, "rollout": result.rollout, "grad": result.grad_norm})
        return True

    def maybe_snapshot(self) -> bool:
        if self.builder.updates - self.updates_at_snapshot < self.promotion.minimum_updates_per_snapshot:
            return False
        started = self.hub.maybe_snapshot(self.clock(), self.admitted_total)
        if started:
            self.updates_at_snapshot = self.builder.updates
            self.scores.challenger.clear()
            self.scores.serving.clear()
            self.scores.serving_joint.clear()
        return started

    def gate_ready(self) -> bool:
        return self.hub.challenger is not None and len(self.scores.challenger) >= self.promotion.future_macro_transitions

    @torch.no_grad()
    def _anchor_scores(self) -> tuple[dict, Optional[tuple[float, float]]]:
        """Serving and challenger loss on anchors, and the action probe on one batch.

        Anchors are grouped by episode, the finest grouping available without labels.
        A caller with task or condition labels should set anchor_groups instead.
        """
        anchors = self.replay.anchors
        if not anchors:
            return {}, None
        rng = np.random.default_rng(0)
        count = min(self.promotion.anchor_windows, len(anchors))
        chosen = [anchors[i] for i in rng.choice(len(anchors), count, replace=False)]
        context = min(self.builder.config.max_context, self.spec.frames - 1)
        totals: dict[str, list[list[float]]] = {}
        probe = None
        for first in range(0, count, self.batch_size):
            part = chosen[first : first + self.batch_size]
            batch = collate_windows(part, self.episodes, self.cache, self.spec, self.view_ids, self.device)
            if probe is None:
                probe = action_probe(self.hub.serving, self.hub.challenger, batch, context, 0.02)
            target = batch["features"][:, context : context + 1]
            old = self.hub.serving.rollout(batch, context, 1)
            new = self.hub.challenger.rollout(batch, context, 1)
            for i, (episode_id, _) in enumerate(part):
                pair = [float(token_metric(old[i : i + 1], target[i : i + 1])), float(token_metric(new[i : i + 1], target[i : i + 1]))]
                totals.setdefault(episode_id, []).append(pair)
        groups = {k: tuple(np.mean(v, axis=0)) for k, v in totals.items()}
        return groups, probe

    def evaluate_challenger(self, seed: int = 0) -> GateDecision:
        """Step 5: gate on future transitions, anchors and action response.

        The joint term of a visual-only update is equal by construction, because B1
        joint prediction comes from the same servo model in every version.
        """
        n = len(self.scores.challenger)
        anchors, probe = self._anchor_scores()
        if self.anchor_groups:
            anchors = self.anchor_groups(self.hub.serving, self.hub.challenger)
        data = GateInput(
            serving_loss=np.asarray(self.scores.serving[-n:]),
            challenger_loss=np.asarray(self.scores.challenger),
            serving_joint_error=float(np.mean(self.scores.serving_joint[-n:])),
            challenger_joint_error=float(np.mean(self.scores.serving_joint[-n:])),
            anchor_groups=anchors,
            sensitivity_ratio=probe[0] if probe else None,
            response_cosine=probe[1] if probe else None,
            numerics_ok=adapters_finite(self.hub.challenger),
        )
        decision = evaluate_gate(data, self.promotion, seed)
        self.hub.log.append({"event": "gate", "version": self.hub.challenger.version, "accept": decision.accept, "reasons": decision.reasons})
        return decision

    def rollback(self, reason: str) -> str:
        """Restore serving and the builder together (spec 13.4).

        Restoring serving alone would leave the builder training from the weights
        that were just rejected. The builder returns to the validated parameters and
        keeps that version's optimizer moments when they were saved. Replay stays:
        new experience keeps its provenance and physical actions cannot be undone.
        """
        target = self.hub.rollback(reason)
        saved = self.hub.store.load(target)
        if saved["builder"] is not None:
            self.builder.load_state_dict(saved["builder"])
        else:
            self.builder.reset_optimizer()
        self.builder.model.load_adapter_state(saved["snapshot"].state, target)
        self.builder.model.set_trainable("online")
        self.updates_at_snapshot = self.builder.updates
        self.rollback_monitor.reset()
        self.rollback_due = None
        self.pending.clear()
        return target

    def planning_boundary(self, decision: Optional[GateDecision]) -> Optional[str]:
        """Act on a pending rollback or a gate decision. Call only between plans."""
        if self.rollback_due is not None:
            return self.rollback(self.rollback_due)
        if decision is None:
            return None
        if decision.accept and not self.hub.shadow:
            version = self.hub.promote(self.builder.state_dict())
            self.rollback_monitor.reset()
            return version
        self.hub.discard_challenger()
        return None
