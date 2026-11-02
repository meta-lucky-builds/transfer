"""Prediction metrics and the B0 / B1 comparison (spec 19).

Per-episode means are the unit of uncertainty, because windows from one episode are
correlated. Prediction gains are necessary, not sufficient: spec 14.7 and eq. 42 both
explain how lower latent error can coexist with worse control, so gate G4 still
requires physical task outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from ..learning.losses import token_metric
from ..models.adapted_predictor import AdaptedPredictor


@torch.no_grad()
def rollout_errors(model: AdaptedPredictor, batch: dict, context: int, horizon: int) -> np.ndarray:
    """Per-sequence L1 at each horizon, [B, H]."""
    predicted = model.rollout(batch, context, horizon)
    target = batch["features"][:, context : context + horizon]
    errors = np.zeros((predicted.shape[0], horizon))
    for b in range(predicted.shape[0]):
        for k in range(horizon):
            errors[b, k] = float(token_metric(predicted[b : b + 1, k : k + 1], target[b : b + 1, k : k + 1]))
    return errors


@torch.no_grad()
def stationary_errors(batch: dict, context: int, horizon: int) -> np.ndarray:
    """Baseline for gate G1: repeat the last observed frame."""
    last = batch["features"][:, context - 1 : context]
    target = batch["features"][:, context : context + horizon]
    errors = np.zeros((last.shape[0], horizon))
    for b in range(last.shape[0]):
        for k in range(horizon):
            errors[b, k] = float(token_metric(last[b : b + 1], target[b : b + 1, k : k + 1]))
    return errors


@dataclass
class Comparison:
    baseline_mean: float
    adapted_mean: float
    relative_improvement: float
    ci_low: float
    ci_high: float
    episodes: int


def compare_by_episode(baseline: dict[str, float], adapted: dict[str, float], replicates: int = 2000, seed: int = 0):
    """Paired episode bootstrap of (baseline - adapted) / baseline, 90 percent interval."""
    episodes = sorted(set(baseline) & set(adapted))
    if len(episodes) < 2:
        raise ValueError("need at least two shared episodes")
    a = np.array([baseline[e] for e in episodes])
    b = np.array([adapted[e] for e in episodes])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(episodes), size=(replicates, len(episodes)))
    relative = (a[index].mean(1) - b[index].mean(1)) / a[index].mean(1)
    return Comparison(
        float(a.mean()),
        float(b.mean()),
        float((a.mean() - b.mean()) / a.mean()),
        float(np.quantile(relative, 0.05)),
        float(np.quantile(relative, 0.95)),
        len(episodes),
    )
