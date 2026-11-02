"""Serving, builder, challenger roles, the promotion gate and rollback (spec 13).

A challenger is scored only on transitions after its snapshot, on exactly the same
commands and observations as serving. Thresholds are engineering choices from the
spec. The gate reduces risk but is no statistical guarantee under adaptive reuse. An
inconclusive result retains serving. Adaptation is optional, and with no real shift
the correct outcome is usually rejection.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..models.adapted_predictor import AdaptedPredictor, AdapterSnapshot


@dataclass(frozen=True)
class PromotionConfig:
    minimum_snapshot_interval_s: float = 30.0
    minimum_updates_per_snapshot: int = 1  # not in the spec: an untrained snapshot equals serving
    future_macro_transitions: int = 128
    temporal_block_s: float = 2.0
    macro_hz: float = 5.0
    anchor_windows: int = 512
    minimum_relative_improvement: float = 0.02
    bootstrap_confidence: float = 0.9
    bootstrap_replicates: int = 2000
    maximum_joint_regression: float = 0.02
    maximum_anchor_regression: float = 0.05
    maximum_task_regression: float = 0.05
    rollback_regression: float = 0.10
    rollback_window_s: float = 60.0
    retain_validated: int = 3
    # Not in the spec, which names the check but gives no numbers.
    probe_minimum_sensitivity_ratio: float = 0.5
    probe_minimum_cosine: float = 0.0


@dataclass
class GateInput:
    serving_loss: np.ndarray  # per future transition, chronological
    challenger_loss: np.ndarray
    serving_joint_error: float
    challenger_joint_error: float
    anchor_groups: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (serving, challenger)
    task_probe: Optional[tuple[float, float]] = None
    sensitivity_ratio: Optional[float] = None
    response_cosine: Optional[float] = None
    numerics_ok: bool = True
    deadlines_ok: bool = True


@dataclass
class GateDecision:
    accept: bool
    reasons: list[str]
    relative_improvement: float
    upper_bound: float


def block_bootstrap_upper(differences: np.ndarray, block: int, replicates: int, confidence: float, seed: int) -> float:
    """One-sided upper bound on the mean of challenger minus serving.

    Non-overlapping blocks are the resampling unit, so overlapping temporal data is
    not treated as independent.
    """
    count = len(differences) // block
    if count < 2:
        return float("inf")
    means = differences[: count * block].reshape(count, block).mean(axis=1)
    rng = np.random.default_rng(seed)
    resampled = means[rng.integers(0, count, size=(replicates, count))].mean(axis=1)
    return float(np.quantile(resampled, confidence))


def evaluate_gate(data: GateInput, config: PromotionConfig, seed: int = 0) -> GateDecision:
    serving, challenger = np.asarray(data.serving_loss), np.asarray(data.challenger_loss)
    reasons: list[str] = []
    if serving.shape != challenger.shape:
        raise ValueError("gate needs losses on exactly the same transitions")
    if len(serving) < config.future_macro_transitions:
        reasons.append(f"only {len(serving)} future transitions")
    if not data.numerics_ok:
        reasons.append("numerical check failed")
    if not data.deadlines_ok:
        reasons.append("deadline check failed")

    mean_serving = float(serving.mean()) if len(serving) else float("nan")
    mean_challenger = float(challenger.mean()) if len(challenger) else float("nan")
    relative = (mean_serving - mean_challenger) / mean_serving if mean_serving > 0 else float("-inf")
    block = max(int(round(config.temporal_block_s * config.macro_hz)), 1)
    upper = block_bootstrap_upper(
        challenger - serving, block, config.bootstrap_replicates, config.bootstrap_confidence, seed
    )
    if not relative >= config.minimum_relative_improvement:
        reasons.append(f"improvement {relative:.4f} below threshold")
    if not upper < 0:
        reasons.append(f"upper bound {upper:.6f} is not below zero")
    if data.challenger_joint_error > data.serving_joint_error * (1 + config.maximum_joint_regression):
        reasons.append("joint prediction regressed")
    for name, (old, new) in data.anchor_groups.items():
        if new > old * (1 + config.maximum_anchor_regression):
            reasons.append(f"anchor group {name} regressed")
    if data.task_probe is not None and data.task_probe[1] > data.task_probe[0] * (1 + config.maximum_task_regression):
        reasons.append("task probe regressed")
    if data.sensitivity_ratio is not None and data.sensitivity_ratio < config.probe_minimum_sensitivity_ratio:
        reasons.append("action response flattened")
    if data.response_cosine is not None and data.response_cosine < config.probe_minimum_cosine:
        reasons.append("action response reversed")
    return GateDecision(not reasons, reasons, relative, upper)


@torch.no_grad()
def action_probe(serving: AdaptedPredictor, challenger: AdaptedPredictor, batch: dict, context: int, delta: float):
    """Compare token responses to a small command change between two versions.

    Returns (sensitivity ratio, cosine). Near-zero ratio means the challenger has
    begun ignoring actions, and negative cosine means its response has reversed.
    """
    shifted = dict(batch)
    shifted["issued"] = batch["issued"] + delta

    def response(model):
        return (model.rollout(shifted, context, 1) - model.rollout(batch, context, 1)).float().flatten(1)

    old, new = response(serving), response(challenger)
    ratio = (new.norm(dim=1) / old.norm(dim=1).clamp_min(1e-12)).mean()
    cosine = torch.nn.functional.cosine_similarity(old, new, dim=1).mean()
    return float(ratio), float(cosine)


def adapters_finite(model: AdaptedPredictor) -> bool:
    return all(torch.isfinite(value).all() for value in model.adapter_state().values())


class VersionStore:
    """Validated adapter versions on disk, written atomically.

    The offline reference and the newest validated versions stay available for
    rollback. Older files stay on disk. Pruning them is a human decision.
    """

    def __init__(self, directory: str | os.PathLike, retain: int = 3):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.retain = retain
        self.reference: Optional[str] = None
        self.validated: list[str] = []

    def _path(self, version: str) -> Path:
        return self.directory / f"{version}.pt"

    def save(self, snapshot: AdapterSnapshot, builder_state: Optional[dict] = None, reference: bool = False) -> Path:
        path = self._path(snapshot.version)
        if path.exists():
            raise FileExistsError(f"version {snapshot.version} already exists and versions are immutable")
        temporary = path.with_suffix(".tmp")
        torch.save({"snapshot": snapshot, "builder": builder_state}, temporary)
        os.replace(temporary, path)
        if reference:
            self.reference = snapshot.version
        else:
            self.validated.append(snapshot.version)
        with open(self.directory / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump({"reference": self.reference, "validated": self.validated}, handle, indent=2)
        return path

    def load(self, version: str) -> dict:
        return torch.load(self._path(version), map_location="cpu", weights_only=False)

    def rollback_candidates(self) -> list[str]:
        """Newest first, ending with the offline reference."""
        recent = self.validated[-self.retain :][::-1]
        return recent + ([self.reference] if self.reference else [])


class RollbackMonitor:
    """Sustained prequential degradation over a window of valid data (spec 13.4).

    Compares the active version with the version it replaced, both scored on the same
    live transitions, and needs a mostly full window before it can trigger.
    """

    def __init__(self, config: PromotionConfig):
        self.config = config
        self.samples: deque[tuple[float, float, float]] = deque()

    def reset(self) -> None:
        self.samples.clear()

    def observe(self, time_s: float, active_loss: float, previous_loss: float) -> bool:
        if not (np.isfinite(active_loss)):
            return True
        self.samples.append((time_s, active_loss, previous_loss))
        horizon = time_s - self.config.rollback_window_s
        while self.samples and self.samples[0][0] < horizon:
            self.samples.popleft()
        needed = int(0.8 * self.config.rollback_window_s * self.config.macro_hz)
        if len(self.samples) < needed:
            return False
        active = np.mean([s[1] for s in self.samples])
        previous = np.mean([s[2] for s in self.samples])
        return bool(active > previous * (1 + self.config.rollback_regression))


class ModelHub:
    """Owns the roles. The serving object is replaced, never mutated.

    A plan holds a reference to its serving model, so a promotion cannot change the
    weights underneath a running plan. New caches carry the new version, and a stale
    cache raises StaleCacheError instead of silently mixing versions.
    """

    def __init__(self, base: AdaptedPredictor, store: VersionStore, config: PromotionConfig, shadow: bool = True):
        self.config = config
        self.store = store
        self.shadow = shadow
        self.serving = base
        self.serving.set_trainable("none")
        self.builder = base.clone_role()
        self.reference = base.clone_role()
        self.previous: Optional[AdaptedPredictor] = None
        self.challenger: Optional[AdaptedPredictor] = None
        self.challenger_snapshot: Optional[AdapterSnapshot] = None
        self.last_snapshot_s = -1e18
        self.counter = 0
        self.log: list[dict] = []
        store.save(base.snapshot("offline-reference", data_cutoff=-1), reference=True)
        base.version = "offline-reference"
        self.builder.version = "offline-reference"
        self.reference.version = "offline-reference"

    def maybe_snapshot(self, now_s: float, data_cutoff: int) -> bool:
        if self.challenger is not None:
            return False
        if now_s - self.last_snapshot_s < self.config.minimum_snapshot_interval_s:
            return False
        self.counter += 1
        version = f"candidate-{self.counter:05d}"
        self.challenger_snapshot = self.builder.snapshot(version, parent=self.serving.version, data_cutoff=data_cutoff)
        challenger = self.serving.clone_role()
        challenger.load_adapter_state(self.challenger_snapshot.state, version)
        challenger.set_trainable("none")
        self.challenger = challenger
        self.last_snapshot_s = now_s
        return True

    def discard_challenger(self) -> None:
        self.challenger = None
        self.challenger_snapshot = None

    def promote(self, builder_state: Optional[dict] = None) -> str:
        """Install the challenger as a new immutable serving version.

        Call only at a planning boundary, after any active plan has finished. The
        caller must rebuild contexts and caches, which is enforced by version checks.
        """
        if self.challenger is None or self.challenger_snapshot is None:
            raise RuntimeError("no challenger to promote")
        if self.shadow:
            raise RuntimeError("shadow mode never changes the serving model")
        self.store.save(self.challenger_snapshot, builder_state)
        self.previous, self.serving = self.serving, self.challenger
        self.log.append({"event": "promote", "version": self.serving.version, "time": time.time()})
        self.challenger = None
        self.challenger_snapshot = None
        return self.serving.version

    def rollback(self, reason: str) -> str:
        """Restore the newest earlier version. Physical actions and replay are unchanged."""
        candidates = [v for v in self.store.rollback_candidates() if v != self.serving.version]
        if not candidates:
            raise RuntimeError("no earlier version to roll back to")
        target = candidates[0]
        saved = self.store.load(target)
        restored = self.serving.clone_role()
        restored.load_adapter_state(saved["snapshot"].state, target)
        restored.set_trainable("none")
        self.previous, self.serving = None, restored
        self.discard_challenger()
        self.log.append({"event": "rollback", "version": target, "reason": reason, "time": time.time()})
        return target
