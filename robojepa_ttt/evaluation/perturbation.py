"""Software dynamics shifts and a hardware-free arm (spec 19.2 families 2 and 8).

CommandShift changes what the arm receives while the model still sees the unperturbed
command, which is what an unmodelled actuator change looks like. Bounds are enforced.
Shifts must not be used to create hazards, and a real arm still needs its limiter.

SimulatedArm is a first-order arm with a synthetic camera. Its features are a fixed
smooth function of joint angles, not images. It exercises the machinery only, and its
results say nothing about real manipulation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CommandShift:
    gain: float = 1.0
    delay_ticks: int = 0

    def __post_init__(self):
        if not 0.3 <= self.gain <= 1.0:
            raise ValueError("gain shift is limited to [0.3, 1.0] so that motion can only shrink")
        if not 0 <= self.delay_ticks <= 15:
            raise ValueError("delay shift is limited to half a second")


class ShiftedChannel:
    """Applies a CommandShift to a stream of commands."""

    def __init__(self, shift: CommandShift, start: np.ndarray):
        self.shift = shift
        self.queue = deque([np.asarray(start, dtype=float)] * (shift.delay_ticks + 1), maxlen=shift.delay_ticks + 1)

    def apply(self, target: np.ndarray, position: np.ndarray) -> np.ndarray:
        self.queue.append(np.asarray(target, dtype=float))
        return position + self.shift.gain * (self.queue[0] - position)


class SimulatedArm:
    def __init__(self, tau_s=0.1, tick_s=1.0 / 30.0, noise=1e-4, seed=0):
        self.gain = 1.0 - np.exp(-tick_s / tau_s)
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.position = np.zeros(6)

    def step(self, target: np.ndarray) -> np.ndarray:
        self.position = self.position + self.gain * (target - self.position)
        return self.position + self.rng.normal(0.0, self.noise, 6)


class SyntheticCamera:
    """Maps joints to token grids. Changes are smooth, like a slowly moving scene."""

    def __init__(self, views=2, height=3, width=4, channels=32, seed=7):
        generator = torch.Generator().manual_seed(seed)
        self.projection = torch.randn(views, height * width, channels, 6, generator=generator) * 2.0
        self.phase = torch.randn(views, height * width, channels, generator=generator)
        self.shape = (views, height, width, channels)

    def features(self, position: np.ndarray) -> torch.Tensor:
        p = torch.as_tensor(position, dtype=torch.float32)
        tokens = torch.sin((self.projection * p).sum(-1) + self.phase)
        return F.layer_norm(tokens.reshape(self.shape), (self.shape[-1],))
