"""Stage 3: train adapters offline on logged robot data (spec 9.7, 10.6).

    L_V = 0.25 L_realized + 0.5 L_cmdTF + 0.5 L_AR + 0.05 L_cons

L_cons needs the compact model, which B1 does not have, so it is absent here and the
weights are not renormalised. This run produces the offline reference for L_anchor.

Differences from RoboJEPA training, from the spec: only new modules train, commands
replace realized actions in the deployable path, and rollouts use predicted joints.
Same as RoboJEPA: L1 in LN0 space, detached rollout feedback, random context prefix,
AdamW betas (0.9, 0.95), global clip 1.0, bf16 network computation.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch

from ..models.adapted_predictor import AdaptedPredictor
from .distributed import SINGLE, DistContext, all_reduce_gradients, all_reduce_sum, broadcast_object, broadcast_parameters
from .losses import discounted_rollout_loss, token_metric
from .online_builder import build_optimizer


@dataclass(frozen=True)
class OfflineConfig:
    sequence_frames: int = 16
    effective_batch: int = 16
    microbatch: int = 1
    max_updates: int = 20_000
    warmup_updates: int = 500
    curriculum: tuple[tuple[int, int], ...] = ((5000, 2), (10000, 5), (5000, 10))
    gamma: float = 0.95
    lora_lr: float = 1e-4
    residual_lr: float = 3e-4
    final_lr_fraction: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    weight_realized: float = 0.25
    weight_cmd_tf: float = 0.5
    weight_rollout: float = 0.5
    max_context: int = 8
    validation_interval: int = 500
    patience_evaluations: int = 8
    seed: int = 101


def lr_factor(update: int, config: OfflineConfig) -> float:
    if update < config.warmup_updates:
        return (update + 1) / config.warmup_updates
    span = max(config.max_updates - config.warmup_updates, 1)
    progress = min((update - config.warmup_updates) / span, 1.0)
    return config.final_lr_fraction + (1 - config.final_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))


def horizon_at(update: int, config: OfflineConfig) -> int:
    reached = 0
    for updates, horizon in config.curriculum:
        reached += updates
        if update < reached:
            return horizon
    return config.curriculum[-1][1]


def visual_loss(
    model: AdaptedPredictor, batch: dict, context: int, horizon: int, config: OfflineConfig, use_realized: bool = True
):
    """Return the weighted loss and its unweighted parts for logging."""
    target = batch["features"][:, context : context + horizon]
    predicted = model.rollout(batch, context, horizon)
    steps = [token_metric(predicted[:, k : k + 1], target[:, k : k + 1]) for k in range(horizon)]
    cmd_tf, rollout = steps[0], discounted_rollout_loss(steps, config.gamma)
    loss = config.weight_cmd_tf * cmd_tf + config.weight_rollout * rollout
    parts = {"cmd_tf": cmd_tf.item(), "rollout": rollout.item()}
    if use_realized:
        frames = batch["features"].shape[1]
        predicted_realized = model.realized_forward(batch, frames)
        realized = token_metric(predicted_realized, batch["features"][:, 1:])
        loss = loss + config.weight_realized * realized
        parts["realized"] = realized.item()
    return loss, parts


class OfflineTrainer:
    """Stage-3 trainer. With a multi-rank DistContext it is data parallel.

    The effective batch of sequences is split evenly over ranks, so the objective and
    the optimizer step are the same as one process accumulating every microbatch.
    Only rank 0 validates, saves and logs; its decisions are broadcast so every rank
    stops at the same update.
    """

    def __init__(
        self,
        model: AdaptedPredictor,
        config: OfflineConfig,
        directory: str | os.PathLike,
        dist: DistContext = SINGLE,
    ):
        micro_steps = config.effective_batch // config.microbatch
        if config.effective_batch % config.microbatch or micro_steps % dist.world_size:
            raise ValueError(
                f"effective batch {config.effective_batch} must split into microbatches of {config.microbatch} "
                f"shared evenly by {dist.world_size} ranks"
            )
        self.model = model
        self.config = config
        self.dist = dist
        self.directory = Path(directory)
        if dist.is_main:
            self.directory.mkdir(parents=True, exist_ok=True)
        # Ranks draw different random contexts; data shards are the caller's job.
        self.rng = random.Random(config.seed + dist.rank)
        groups = model.set_trainable("offline")
        # Rank 0's random LoRA A and residual initialisation become everyone's.
        broadcast_parameters(model.trainable_named().values(), dist)
        self.optimizer = build_optimizer(
            groups,
            {"lora": config.lora_lr, "residual": config.residual_lr},
            config.weight_decay,
            config.betas,
            config.eps,
        )
        self.update = 0
        self.best = float("inf")
        self.stale = 0
        self.log: list[dict] = []

    def _context(self, frames: int, horizon: int) -> int:
        limit = min(self.config.max_context, frames - horizon)
        if limit < 1:
            raise ValueError(f"sequence of {frames} frames cannot hold horizon {horizon}")
        return self.rng.randint(1, limit)

    def _save(self, name: str) -> None:
        if not self.dist.is_main:
            return
        path = self.directory / name
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "adapters": self.model.adapter_state(),
                "optimizer": self.optimizer.state_dict(),
                "update": self.update,
                "best": self.best,
                "stale": self.stale,
                "rng": self.rng.getstate(),
            },
            temporary,
        )
        os.replace(temporary, path)

    def resume(self, name: str = "latest.pt") -> bool:
        path = self.directory / name
        if not path.exists():
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_adapter_state(state["adapters"], f"offline-{state['update']}")
        self.model.set_trainable("offline")
        self.optimizer.load_state_dict(state["optimizer"])
        self.update, self.best, self.stale = state["update"], state["best"], state["stale"]
        if self.dist.is_main:
            self.rng.setstate(state["rng"])
        else:
            # Only rank 0's stream is saved; keep the others distinct and reproducible.
            self.rng = random.Random(self.config.seed + self.dist.rank + 7919 * self.update)
        return True

    def train_step(self, next_batch: Callable[[], dict]) -> dict:
        cfg = self.config
        factor = lr_factor(self.update, cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        horizon = horizon_at(self.update, cfg)
        micro_steps = cfg.effective_batch // cfg.microbatch  # over all ranks
        local_steps = micro_steps // self.dist.world_size
        self.optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {"loss": 0.0}
        for _ in range(local_steps):
            batch = next_batch()
            context = self._context(batch["features"].shape[1], horizon)
            loss, parts = visual_loss(self.model, batch, context, horizon, cfg)
            (loss / micro_steps).backward()
            totals["loss"] += loss.item() / micro_steps
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value / micro_steps
        parameters = list(self.model.trainable_named().values())
        all_reduce_gradients(parameters, self.dist)
        totals = all_reduce_sum(totals, self.dist, parameters[0].device)
        total_loss = totals.pop("loss")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip_norm)
        if not math.isfinite(total_loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite training step at update {self.update}: loss {total_loss}")
        self.optimizer.step()
        self.update += 1
        return {"update": self.update, "loss": total_loss, "grad_norm": float(grad_norm), "horizon": horizon, **totals}

    def fit(
        self,
        next_batch: Callable[[], dict],
        validate: Callable[[AdaptedPredictor], float],
        log: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Train with early stopping on validation command prediction.

        Stops after patience_evaluations without improvement and does not run the
        remaining curriculum just to match a schedule (spec 10.6).

        Only rank 0 calls validate; the score is broadcast so all ranks agree on the
        best checkpoint and on when to stop.
        """
        started = time.time()
        while self.update < self.config.max_updates:
            record = self.train_step(next_batch)
            if self.update % self.config.validation_interval == 0:
                score = validate(self.model) if self.dist.is_main else None
                record["validation"] = broadcast_object(score, self.dist)
                if record["validation"] < self.best:
                    self.best = record["validation"]
                    self.stale = 0
                    self._save("best.pt")
                else:
                    self.stale += 1
                self._save("latest.pt")
            self.log.append(record)
            if log is not None and self.dist.is_main:
                log(record)
            if self.stale >= self.config.patience_evaluations:
                break
        return {"updates": self.update, "best": self.best, "seconds": time.time() - started}
