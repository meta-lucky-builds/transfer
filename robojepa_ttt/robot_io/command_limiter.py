"""Reference limiter shared by the local supervisor and the planner.

Spec 14.3: the simulator inside the planner must apply the same limiter as the
robot, otherwise CEM optimises commands that are never executed. One function
serves both, so the two cannot drift apart. Limits come from measured
commissioning values; there are no defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LimiterConfig:
    """All values are physical units per second, shape (6,)."""

    position_min: tuple[float, ...]
    position_max: tuple[float, ...]
    velocity_max: tuple[float, ...]
    acceleration_max: tuple[float, ...]

    def __post_init__(self):
        for name in ("position_min", "position_max", "velocity_max", "acceleration_max"):
            value = getattr(self, name)
            if value is None or len(value) != 6:
                raise ValueError(f"{name} must hold six measured values")
        if any(v <= 0 for v in self.velocity_max) or any(a <= 0 for a in self.acceleration_max):
            raise ValueError("velocity and acceleration limits must be positive")
        if any(lo >= hi for lo, hi in zip(self.position_min, self.position_max)):
            raise ValueError("position_min must be below position_max")


def limit_reference(
    start: torch.Tensor,
    start_velocity: torch.Tensor,
    targets: torch.Tensor,
    dt: float,
    config: LimiterConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interpolate targets at raw ticks under position, rate and acceleration limits.

    start, start_velocity: [..., 6] state of the reference at the first tick.
    targets: [..., N, 6] requested references, one per raw tick.
    Returns (references [..., N, 6], final velocity [..., 6], clipped [..., N] bool).
    """
    lo = targets.new_tensor(config.position_min)
    hi = targets.new_tensor(config.position_max)
    v_max = targets.new_tensor(config.velocity_max)
    a_max = targets.new_tensor(config.acceleration_max)
    # Clipping an out-of-limits reference would be a step of arbitrary size, and its
    # implied velocity would then drive the rate limits. Recovery is an operator
    # decision, so this refuses rather than moving.
    if (start < lo - 1e-6).any() or (start > hi + 1e-6).any():
        raise ValueError("reference starts outside the validated position limits")

    reference, velocity = start, start_velocity
    outputs, clipped = [], []
    for n in range(targets.shape[-2]):
        goal = targets[..., n, :].clamp(lo, hi)
        wanted_velocity = (goal - reference) / dt
        limited = wanted_velocity.clamp(-v_max, v_max)
        limited = limited.clamp(velocity - a_max * dt, velocity + a_max * dt)
        next_reference = (reference + limited * dt).clamp(lo, hi)
        # Recover the velocity actually applied, including position clipping.
        velocity = (next_reference - reference) / dt
        reference = next_reference
        outputs.append(reference)
        clipped.append(((reference - targets[..., n, :]).abs() > 1e-6).any(-1))
    return torch.stack(outputs, dim=-2), velocity, torch.stack(clipped, dim=-1)
