"""Calibration of the compact likelihood on episode-disjoint calibration data (spec 7.5).

Covariance multipliers scale R per block (joints, each view). They are fitted after
training on calibration episodes, by direct search on predictive NLL, never by
gradient, and calibration episodes never provide training gradients.

innovation_summary reports normalised innovations per block. Under a correct
Gaussian model the mean NIS per valid coordinate is 1; that is a diagnostic, not
proof (spec 7.5). Pre-inflation residuals stay in the report, so failures are
classified rather than hidden by inflating R until every error disappears.
"""

from __future__ import annotations

import math

import torch

from .compact import CompactBelief
from .train import CompactTrainConfig, _step_input


@torch.no_grad()
def innovation_summary(model: CompactBelief, batches: list[dict], regimes=(0,)) -> dict[str, dict]:
    totals: dict[str, list[float]] = {}
    for batch in batches:
        dtype = next(model.parameters()).dtype
        batch = {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in batch.items()}
        state = model.initial_state(batch["observation"][:, 0, :6], batch["obs_mask"][:, 0, :6].all(-1), regimes)
        for t in range(batch["observation"].shape[1]):
            state, out = model.step(state, _step_input(batch, t), predict=t > 0, compute_nis=t > 0)
            if t == 0:
                continue
            scored = batch["score"][:, t]
            for name, (nis, count) in out.nis.items():
                keep = scored & (count > 0)
                entry = totals.setdefault(name, [0.0, 0.0, 0])
                entry[0] += float(nis[keep].sum())
                entry[1] += float(count[keep].sum())
                entry[2] += int(keep.sum())
    return {
        name: {"nis_per_coordinate": s / c if c else float("nan"), "coordinates": c, "steps": n}
        for name, (s, c, n) in totals.items()
    }


@torch.no_grad()
def scored_nll(model: CompactBelief, batches: list[dict], joint_scale: torch.Tensor, regimes=(0,)) -> float:
    from .train import sequence_loss

    config = CompactTrainConfig(regimes=regimes)
    values, weights = [], []
    for batch in batches:
        _, parts = sequence_loss(model, batch, joint_scale, config)
        values.append(parts["nll"])
        weights.append(parts["scored_steps"])
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


@torch.no_grad()
def fit_calibration(
    model: CompactBelief,
    batches: list[dict],
    joint_scale: torch.Tensor,
    regimes=(0,),
    grid: tuple[float, ...] = tuple(2.0 ** k for k in range(-3, 6)),
    passes: int = 2,
) -> dict:
    """Coordinate search over per-block R multipliers, minimising calibration NLL.

    Writes the result into model.calibration and returns before/after NLL and the
    multipliers. Starts from the current multipliers, so it can refine a previous fit.
    """
    before = scored_nll(model, batches, joint_scale, regimes)
    best = before
    for _ in range(passes):
        for block in range(model.calibration.numel()):
            current = float(model.calibration[block])
            choice = current
            for factor in grid:
                model.calibration[block] = factor
                value = scored_nll(model, batches, joint_scale, regimes)
                if value < best - 1e-9:
                    best, choice = value, factor
            model.calibration[block] = choice
    if not math.isfinite(best):
        raise FloatingPointError("calibration produced a non-finite NLL")
    return {"nll_before": before, "nll_after": best, "multipliers": model.calibration.tolist()}
