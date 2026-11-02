"""Episode storage and dense causal windows (spec 4, 10.1, 11.1, 11.2, eq. 32).

Raw records are stored once and windows are only (episode, start) references.
A window W_i holds keyframes i, i+K, ..., i+LK and the raw commands between them.

Admission differs from the JEPA-TTT paper on purpose. The paper admits a start
once (C+1)K raw rows exist, which is longer than the keyframes need. This code
admits a window exactly when every indexed observation, command and timing
alignment exists (spec 11.1). An exact paper reproduction must not use this rule.

Dense windows raise temporal coverage, not the number of independent experiments.
Overlapping windows from one episode must never be counted as independent data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class WindowSpec:
    frames: int = 10  # keyframes per window: up to 8 context plus 2 rollout targets
    stride: int = 6
    tick_s: float = 1.0 / 30.0
    timing_tolerance: float = 0.5  # allowed |dt - tick| as a fraction of a tick
    history_ticks: int = 30  # one second of preceding commands

    @property
    def span_ticks(self) -> int:
        return (self.frames - 1) * self.stride


class Episode:
    """Growing per-tick record of one episode, safe to extend while windows read it.

    position[n] is measured at tick n. issued[n] was sent at tick n. segment[n]
    increases at any physical reset or calibration change. Windows never cross a
    segment boundary, and command history stops at it too.
    """

    def __init__(self, episode_id: str, calibration_hash: str, encoder_hash: str, views: int, capacity: int = 1024):
        self.episode_id = episode_id
        self.calibration_hash = calibration_hash
        self.encoder_hash = encoder_hash
        self.views = views
        self.length = 0
        self.time_s = np.zeros(capacity)
        self.position = np.zeros((capacity, 6))
        self.issued = np.zeros((capacity, 6))
        self.position_valid = np.zeros(capacity, dtype=bool)
        self.command_valid = np.zeros(capacity, dtype=bool)
        self.frame_valid = np.zeros((capacity, views), dtype=bool)
        self.segment = np.zeros(capacity, dtype=np.int64)
        self._segment = 0

    def _grow(self):
        for name in ("time_s", "position", "issued", "position_valid", "command_valid", "frame_valid", "segment"):
            array = getattr(self, name)
            bigger = np.zeros((array.shape[0] * 2, *array.shape[1:]), dtype=array.dtype)
            bigger[: self.length] = array[: self.length]
            setattr(self, name, bigger)

    def mark_boundary(self):
        """Call on a physical reset, camera move, power cycle or unlogged gap."""
        self._segment += 1

    def append(self, time_s, position, position_valid, issued, command_valid, frame_valid) -> int:
        if self.length and time_s <= self.time_s[self.length - 1]:
            raise ValueError("tick times must increase strictly; repair ordering before appending")
        if self.length == self.time_s.shape[0]:
            self._grow()
        n = self.length
        self.time_s[n] = time_s
        self.position[n] = position
        self.position_valid[n] = position_valid
        self.issued[n] = issued
        self.command_valid[n] = command_valid
        self.frame_valid[n] = frame_valid
        self.segment[n] = self._segment
        self.length = n + 1
        return n


def window_is_valid(episode: Episode, start: int, spec: WindowSpec) -> bool:
    end = start + spec.span_ticks
    if start < 0 or end >= episode.length:
        return False
    if episode.segment[start] != episode.segment[end]:
        return False
    keyframes = np.arange(start, end + 1, spec.stride)
    if not episode.position_valid[keyframes].all() or not episode.frame_valid[keyframes].all():
        return False
    if not episode.command_valid[start:end].all():
        return False
    durations = np.diff(episode.time_s[start : end + 1])
    return bool((np.abs(durations - spec.tick_s) <= spec.timing_tolerance * spec.tick_s).all())


def dense_starts(episode: Episode, spec: WindowSpec, first: int = 0) -> Iterator[int]:
    """Valid starts at every raw offset from `first`, one for each tick of phase."""
    for start in range(first, episode.length - spec.span_ticks):
        if window_is_valid(episode, start, spec):
            yield start


def sparse_starts(episode: Episode, spec: WindowSpec) -> Iterator[int]:
    """Ablation B7: starts at keyframe boundaries only."""
    for start in range(0, episode.length - spec.span_ticks, spec.stride):
        if window_is_valid(episode, start, spec):
            yield start


def extract_window(episode: Episode, start: int, spec: WindowSpec) -> dict[str, np.ndarray]:
    """Copy one window with its causal command history.

    Ticks before the episode or segment start have no commands. There the target
    is held at the first measured position and validity is 0: no command had been
    issued, so holding is the only assumption that invents no motion.
    Keys: keyframes, position [F, 6], ticks of issued/dt/valid of length
    history + span, and start_tick, the index of `start` inside those arrays.
    """
    keyframes = np.arange(start, start + spec.span_ticks + 1, spec.stride)
    segment_start = start
    limit = max(start - spec.history_ticks, 0)
    while segment_start > limit and episode.segment[segment_start - 1] == episode.segment[start]:
        segment_start -= 1

    pad = spec.history_ticks - (start - segment_start)
    span = start + spec.span_ticks
    issued = np.concatenate([np.repeat(episode.position[start : start + 1], pad, axis=0), episode.issued[segment_start:span]])
    valid = np.concatenate([np.zeros(pad, dtype=bool), episode.command_valid[segment_start:span]])
    dt = np.full(issued.shape[0], spec.tick_s)
    # time_s[n + 1] - time_s[n] is how long the command of tick n was in force.
    dt[pad:] = episode.time_s[segment_start + 1 : span + 1] - episode.time_s[segment_start:span]
    return {
        "keyframes": keyframes,
        "position": episode.position[keyframes].copy(),
        "issued": issued,
        "dt": dt,
        "command_valid": valid,
        "start_tick": np.asarray(spec.history_ticks),
    }
