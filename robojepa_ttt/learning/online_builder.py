"""Persistent online optimisation of adapters (spec 9.7, 12).

    L_onlineV = L_cmdTF + 0.5 L_AR + 0.1 L_anchor + 1e-4 L_prox

Only the builder receives gradients. Parameters and AdamW moments persist across
ordinary episodes, as in JEPA-TTT, whose matched control found persistence decisive.
Differences from JEPA-TTT, all from the spec: adapters instead of the full predictor,
L1 in LN0 space instead of MSE, an autonomous rollout term, and functional anchors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from ..models.adapted_predictor import AdaptedPredictor
from .losses import discounted_rollout_loss, proximal_penalty, token_metric


@dataclass(frozen=True)
class OnlineConfig:
    lora_lr: float = 1e-5
    residual_lr: float = 3e-5
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    warmup_updates: int = 100
    horizon: int = 2
    gamma: float = 0.95
    max_context: int = 8
    weight_ar: float = 0.5
    weight_anchor: float = 0.1
    weight_prox: float = 1e-4


@dataclass
class UpdateResult:
    status: str  # "ok" or "nonfinite"
    loss: float
    cmd_tf: float
    rollout: float
    anchor: Optional[float]
    grad_norm: float


def build_optimizer(groups: dict[str, list], lrs: dict[str, float], weight_decay: float, betas, eps):
    """AdamW groups with the decay rule of spec 12.1: none on biases or scalars."""
    param_groups = []
    for name, parameters in groups.items():
        decay = [p for p in parameters if p.ndim >= 2]
        plain = [p for p in parameters if p.ndim < 2]
        if decay:
            param_groups.append({"params": decay, "lr": lrs[name], "base_lr": lrs[name], "weight_decay": weight_decay})
        if plain:
            param_groups.append({"params": plain, "lr": lrs[name], "base_lr": lrs[name], "weight_decay": 0.0})
    return torch.optim.AdamW(param_groups, betas=betas, eps=eps)


class OnlineBuilder:
    def __init__(self, model: AdaptedPredictor, reference: AdaptedPredictor, config: OnlineConfig, seed: int = 0):
        self.model = model
        self.reference = reference
        self.config = config
        self.rng = random.Random(seed)
        self.updates = 0
        groups = model.set_trainable("online")
        reference.set_trainable("none")
        self.optimizer = build_optimizer(
            groups, {"lora": config.lora_lr, "residual": config.residual_lr}, config.weight_decay, config.betas, config.eps
        )
        self.reference_state = {n: p.detach().clone() for n, p in model.trainable_named().items()}

    def reset_optimizer(self) -> None:
        """Spec 12.1: clear moments only after an incompatible change, and log it."""
        groups = self.model.set_trainable("online")
        self.optimizer = build_optimizer(
            groups,
            {"lora": self.config.lora_lr, "residual": self.config.residual_lr},
            self.config.weight_decay,
            self.config.betas,
            self.config.eps,
        )
        self.updates = 0

    def _set_lr(self) -> None:
        factor = min(1.0, (self.updates + 1) / self.config.warmup_updates)
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor

    def sample_context(self, frames: int) -> int:
        return self.rng.randint(1, min(self.config.max_context, frames - self.config.horizon))

    def _chunk_loss(self, batch: dict, flags: torch.Tensor, context: int):
        """Loss terms of one microbatch: (cmd_tf, rollout, anchor or None)."""
        cfg = self.config
        target = batch["features"][:, context : context + cfg.horizon]
        predicted = self.model.rollout(batch, context, cfg.horizon)
        steps = [token_metric(predicted[:, k : k + 1], target[:, k : k + 1]) for k in range(cfg.horizon)]
        anchor = None
        if flags.any():
            subset = {k: v[flags] for k, v in batch.items()}
            with torch.no_grad():
                reference_prediction = self.reference.rollout(subset, context, cfg.horizon)
            anchor = token_metric(predicted[flags], reference_prediction.detach())
        return steps[0], discounted_rollout_loss(steps, cfg.gamma), anchor

    def update(
        self,
        batch: dict,
        anchor_flags: Sequence[bool],
        context: Optional[int] = None,
        microbatch: Optional[int] = None,
    ) -> UpdateResult:
        """One optimizer step. Microbatches give the same objective as one full batch.

        Each sequence term is weighted by its share of the batch and each anchor term
        by its share of the anchors, so splitting changes memory use, not the result
        (spec 11.4). The proximal term depends only on parameters and is added once.
        """
        cfg = self.config
        size = batch["features"].shape[0]
        frames = batch["features"].shape[1]
        context = context if context is not None else self.sample_context(frames)
        microbatch = microbatch or size
        device = batch["features"].device
        flags = torch.as_tensor(list(anchor_flags), dtype=torch.bool, device=device)
        anchors_total = max(int(flags.sum()), 1)

        self._set_lr()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "cmd_tf": 0.0, "rollout": 0.0, "anchor": 0.0}
        for start in range(0, size, microbatch):
            index = range(start, min(start + microbatch, size))
            part = {k: v[index.start : index.stop] for k, v in batch.items()}
            part_flags = flags[index.start : index.stop]
            cmd_tf, rollout, anchor = self._chunk_loss(part, part_flags, context)
            share = len(index) / size
            loss = share * (cmd_tf + cfg.weight_ar * rollout)
            if anchor is not None:
                anchor_share = int(part_flags.sum()) / anchors_total
                loss = loss + cfg.weight_anchor * anchor_share * anchor
                totals["anchor"] += anchor_share * anchor.item()
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                return UpdateResult("nonfinite", float("nan"), float("nan"), float("nan"), None, float("nan"))
            loss.backward()
            totals["loss"] += loss.item()
            totals["cmd_tf"] += share * cmd_tf.item()
            totals["rollout"] += share * rollout.item()

        prox = proximal_penalty(dict(self.model.trainable_named()), self.reference_state)
        if prox is not None and cfg.weight_prox:
            (cfg.weight_prox * prox).backward()
            totals["loss"] += cfg.weight_prox * prox.item()

        parameters = list(self.model.trainable_named().values())
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            return UpdateResult("nonfinite", totals["loss"], totals["cmd_tf"], totals["rollout"], None, float("nan"))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        anchor_value = totals["anchor"] if flags.any() else None
        return UpdateResult("ok", totals["loss"], totals["cmd_tf"], totals["rollout"], anchor_value, grad_norm.item())

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "adapters": self.model.adapter_state(),
            "version": self.model.version,
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_adapter_state(state["adapters"], state["version"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
