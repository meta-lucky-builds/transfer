"""Offline training of the compact belief by differentiable filtering (spec 9.3, 9.4, 10.5).

    L_C = L_NLL + 0.5 L_p + 1e-4 L_regularity   (+ 0.1 regime CE when labels exist)

L_NLL (eq. 26) is the predictive marginal negative log likelihood of each observation
under the filter BEFORE assimilating it, divided by its number of valid coordinates.
Dimension normalisation balances training only; the runtime posterior is untouched.
L_p (eq. 27) is a Huber loss on the prior joint mean against the measured joints,
scaled per joint by s_loss. L_regularity is the mean squared weight of the transition
output rows that produce the fast residual.

Gradients flow through the Jacobians and covariance recursion (the reference variant,
which needs second-order autodiff). differentiable_jacobian=False is the surrogate
ablation of spec 9.3 and must be reported by that name.

Averaging follows spec 9.1: valid steps are averaged within a sequence, then
sequences are averaged, so length differences do not reweight the objective.

The first step of every sequence initialises the belief and has no prediction, so
it is never scored. Burn-in steps update the state without loss, and the state is
detached once at each sequence's burn-in boundary (truncated BPTT).
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from ..learning.online_builder import build_optimizer
from .compact import BeliefState, CompactBelief, StepInput


@dataclass(frozen=True)
class CompactTrainConfig:
    effective_batch: int = 32
    microbatch: int = 32
    max_updates: int = 40_000
    warmup_updates: int = 1000
    lr: float = 3e-4
    final_lr: float = 3e-5
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    validation_interval: int = 500
    patience_evaluations: int = 8
    weight_nll: float = 1.0
    weight_joint: float = 0.5
    weight_regularity: float = 1e-4
    weight_regime_ce: float = 0.1
    regimes: tuple[int, ...] = (0,)
    differentiable_jacobian: bool = True
    seed: int = 101


def lr_factor(update: int, config: CompactTrainConfig) -> float:
    if update < config.warmup_updates:
        return (update + 1) / config.warmup_updates
    final = config.final_lr / config.lr
    span = max(config.max_updates - config.warmup_updates, 1)
    progress = min((update - config.warmup_updates) / span, 1.0)
    return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * progress))


def _step_input(batch: dict, t: int) -> StepInput:
    return StepInput(
        observation=batch["observation"][:, t],
        obs_mask=batch["obs_mask"][:, t],
        issued=batch["issued"][:, t],
        tick_dt=batch["tick_dt"][:, t],
        command_valid=batch["command_valid"][:, t],
        dt=batch["dt"][:, t],
        step_valid=batch["step_valid"][:, t],
    )


def _sequence_mean(values: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
    """values, mask [B, T] -> mean over valid steps per sequence, then over sequences."""
    count = mask.sum(1)
    keep = count > 0
    if not bool(keep.any()):
        return None
    per_sequence = (values * mask).sum(1)[keep] / count[keep]
    return per_sequence.mean()


def sequence_loss(
    model: CompactBelief,
    batch: dict,
    joint_scale: torch.Tensor,
    config: CompactTrainConfig,
    collect: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Run the filter over a padded batch and return (loss, parts).

    batch keys as built by BeliefSequences.batch. Parameters receive gradients only
    from scored steps; burn-in steps shape the state but are detached at the boundary.
    """
    B, T = batch["observation"].shape[:2]
    dtype = next(model.parameters()).dtype
    batch = {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in batch.items()}
    joints0 = batch["observation"][:, 0, :6]
    state = model.initial_state(joints0, batch["obs_mask"][:, 0, :6].all(-1), config.regimes)
    nll = torch.zeros(B, T, dtype=dtype, device=joints0.device)
    joint = torch.zeros_like(nll)
    ce = torch.zeros_like(nll)
    nll_mask = torch.zeros(B, T, dtype=torch.bool, device=joints0.device)
    joint_mask = torch.zeros_like(nll_mask)
    stats = {"nis_joint": [], "nis_visual": []} if collect else None
    for t in range(T):
        step = _step_input(batch, t)
        state, out = model.step(state, step, predict=t > 0, differentiable_jacobian=config.differentiable_jacobian)
        if t > 0:
            scored = batch["score"][:, t]
            has_obs = out.valid_count > 0
            nll[:, t] = -out.log_predictive / out.valid_count.clamp_min(1).to(dtype)
            nll_mask[:, t] = scored & has_obs
            joint_ok = step.obs_mask[:, :6].all(-1) & scored
            error = (out.prior_joint_mean - step.observation[:, :6]) / joint_scale.to(dtype)
            joint[:, t] = F.huber_loss(error, torch.zeros_like(error), delta=1.0, reduction="none").mean(-1)
            joint_mask[:, t] = joint_ok
            if "regime" in batch:
                ce[:, t] = -out.log_posterior.gather(1, batch["regime"][:, t : t + 1]).squeeze(1)
        # Truncated BPTT: cut the graph once, after each sequence's burn-in boundary.
        boundary = batch["burn_in"] == t
        if bool(boundary.any()):
            state = state.detach().select(boundary, state)
    parts: dict[str, float] = {}
    terms = []
    value = _sequence_mean(nll, nll_mask)
    if value is not None:
        terms.append(config.weight_nll * value)
        parts["nll"] = float(value)
    value = _sequence_mean(joint, joint_mask)
    if value is not None:
        terms.append(config.weight_joint * value)
        parts["joint"] = float(value)
    if "regime" in batch:
        value = _sequence_mean(ce, nll_mask)
        if value is not None:
            terms.append(config.weight_regime_ce * value)
            parts["regime_ce"] = float(value)
    regularity = model.fast_output_weights().pow(2).mean()
    terms.append(config.weight_regularity * regularity)
    parts["regularity"] = float(regularity)
    if len(terms) == 1:
        raise ValueError("batch has no scored step with any valid observation")
    loss = sum(terms)
    parts["scored_steps"] = int(nll_mask.sum())
    return loss, parts


class CompactTrainer:
    """AdamW on every offline parameter, warmup then cosine, early stopping on validation NLL."""

    def __init__(self, model: CompactBelief, config: CompactTrainConfig, joint_scale: torch.Tensor, directory):
        if config.effective_batch % config.microbatch:
            raise ValueError("effective batch must be a multiple of the microbatch")
        self.model = model
        self.config = config
        self.joint_scale = joint_scale
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(config.seed)
        for p in model.parameters():
            p.requires_grad_(False)
        params = model.offline_parameters()
        for p in params:
            p.requires_grad_(True)
        self.optimizer = build_optimizer({"compact": params}, {"compact": config.lr}, config.weight_decay, config.betas, config.eps)
        self.update = 0
        self.best = float("inf")
        self.stale = 0
        self.log: list[dict] = []

    def train_step(self, next_batch: Callable[[], dict]) -> dict:
        cfg = self.config
        factor = lr_factor(self.update, cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        micro = cfg.effective_batch // cfg.microbatch
        self.optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        loss_total = 0.0
        for _ in range(micro):
            loss, parts = sequence_loss(self.model, next_batch(), self.joint_scale, cfg)
            (loss / micro).backward()
            loss_total += float(loss) / micro
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v / micro
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
        if not math.isfinite(loss_total) or not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite compact update at {self.update}: loss {loss_total}")
        self.optimizer.step()
        self.update += 1
        return {"update": self.update, "loss": loss_total, "grad_norm": float(grad_norm), "repairs": self.model.repairs.count, **totals}

    @torch.no_grad()
    def evaluate(self, batches: list[dict]) -> float:
        """Mean scored predictive NLL per coordinate over a fixed set (no gradient)."""
        values, weights = [], []
        for batch in batches:
            _, parts = sequence_loss(self.model, batch, self.joint_scale, self.config)
            if "nll" in parts:
                values.append(parts["nll"])
                weights.append(batch["observation"].shape[0])
        if not values:
            raise ValueError("validation set has no scored observations")
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)

    def _save(self, name: str) -> None:
        path = self.directory / name
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update": self.update,
                "best": self.best,
                "stale": self.stale,
                "rng": self.rng.getstate(),
                "joint_scale": self.joint_scale,
                "config": self.model.config,
            },
            temporary,
        )
        os.replace(temporary, path)

    def resume(self, name: str = "latest.pt") -> bool:
        path = self.directory / name
        if not path.exists():
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update, self.best, self.stale = state["update"], state["best"], state["stale"]
        self.rng.setstate(state["rng"])
        return True

    def fit(self, next_batch: Callable[[], dict], validation: list[dict], log: Optional[Callable[[dict], None]] = None) -> dict:
        started = time.time()
        while self.update < self.config.max_updates:
            record = self.train_step(next_batch)
            if self.update % self.config.validation_interval == 0:
                record["validation_nll"] = self.evaluate(validation)
                if record["validation_nll"] < self.best:
                    self.best, self.stale = record["validation_nll"], 0
                    self._save("best.pt")
                else:
                    self.stale += 1
                self._save("latest.pt")
            self.log.append(record)
            if log is not None:
                log(record)
            if self.stale >= self.config.patience_evaluations:
                break
        return {"updates": self.update, "best": self.best, "seconds": time.time() - started}
