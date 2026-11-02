"""Nominal servo model, the command-to-motion bridge of baseline B1 (spec 3.3, 6.4).

Eq. 8, per joint, integrated over the actual tick durations:

    p[n+1] = p[n] + (1 - exp(-dt[n] / tau)) * (u[n - delay] - p[n])

The pretrained predictor conditions on realized displacement, which is unknown
when commands are chosen. This model predicts that displacement from issued
commands alone. It never reads a measured position after the rollout start, so
it cannot leak the future.

Fitted by least squares for the gain at each candidate delay, then the delay is
chosen by open-loop simulation error, because one-step error barely separates
delays when the signal is smooth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class ServoParams:
    """tau_s and delay_ticks hold one entry per joint in JOINT_ORDER."""

    tau_s: tuple[float, ...]
    delay_ticks: tuple[int, ...]
    fit_rmse: tuple[float, ...] = field(default=())

    def __post_init__(self):
        if len(self.tau_s) != 6 or len(self.delay_ticks) != 6:
            raise ValueError("servo parameters need six joints")
        if min(self.tau_s) <= 0 or min(self.delay_ticks) < 0:
            raise ValueError("tau must be positive and delay non-negative")

    @property
    def max_delay(self) -> int:
        return max(self.delay_ticks)


def rollout_servo(
    start: torch.Tensor,
    commands: torch.Tensor,
    dt: torch.Tensor,
    params: ServoParams,
    start_index: int,
) -> torch.Tensor:
    """Simulate from `start` through issued commands.

    start: [..., 6] physical position at tick start_index.
    commands: [..., N, 6] issued targets for ticks 0..N-1 of the same timeline,
        which must cover start_index - max_delay. History before start_index is
        what makes a delay usable at the first predicted tick.
    dt: [..., N] durations, tick n lasting from tick n to tick n + 1.
    Returns [..., N - start_index, 6], the positions at ticks start_index+1..N.
    """
    if start_index < params.max_delay:
        raise ValueError(f"need {params.max_delay} ticks of command history, got {start_index}")
    tau = commands.new_tensor(params.tau_s)
    delays = torch.as_tensor(params.delay_ticks, device=commands.device)
    joints = torch.arange(6, device=commands.device)
    position = start
    outputs = []
    for n in range(start_index, commands.shape[-2]):
        delayed = commands[..., n - delays, joints]
        gain = 1.0 - torch.exp(-dt[..., n, None] / tau)
        position = position + gain * (delayed - position)
        outputs.append(position)
    return torch.stack(outputs, dim=-2)


def _simulate_joint(p0: float, u: np.ndarray, gain: float) -> np.ndarray:
    out = np.empty(len(u))
    p = p0
    for i, target in enumerate(u):
        p = p + gain * (target - p)
        out[i] = p
    return out


def fit_servo(
    positions: list[np.ndarray],
    commands: list[np.ndarray],
    tick_s: float,
    max_delay_s: float = 0.3,
) -> ServoParams:
    """Fit tau and delay per joint from segments at a constant tick.

    positions[k]: [N, 6] measured. commands[k]: [N, 6] issued, aligned so
    commands[k][n] was issued at tick n.
    A nonpositive fitted gain means the data holds no usable excitation for that
    joint (eq. 40), which raises rather than returning a confident guess.
    """
    if len(positions) != len(commands) or not positions:
        raise ValueError("need matching non-empty position and command segments")
    max_delay = int(round(max_delay_s / tick_s))
    taus, delays, rmses = [], [], []
    for joint in range(6):
        best = None
        for delay in range(max_delay + 1):
            xs, ys = [], []
            for p, u in zip(positions, commands):
                if len(p) <= delay + 1:
                    continue
                xs.append(u[: len(p) - delay - 1, joint] - p[delay:-1, joint])
                ys.append(p[delay + 1 :, joint] - p[delay:-1, joint])
            if not xs:
                continue
            x, y = np.concatenate(xs), np.concatenate(ys)
            energy = float(x @ x)
            if energy <= 1e-12:
                continue
            gain = float(x @ y) / energy
            if not 0.0 < gain < 1.0:
                continue
            squared, count = 0.0, 0
            for p, u in zip(positions, commands):
                if len(p) <= delay + 1:
                    continue
                simulated = _simulate_joint(p[delay, joint], u[: len(p) - delay - 1, joint], gain)
                squared += float(((simulated - p[delay + 1 :, joint]) ** 2).sum())
                count += len(simulated)
            rmse = (squared / count) ** 0.5
            if best is None or rmse < best[0]:
                best = (rmse, delay, gain)
        if best is None:
            raise ValueError(f"no identifiable servo response for joint {joint}; collect varied motion")
        rmse, delay, gain = best
        taus.append(-tick_s / float(np.log1p(-gain)))
        delays.append(delay)
        rmses.append(rmse)
    return ServoParams(tuple(taus), tuple(delays), tuple(rmses))
