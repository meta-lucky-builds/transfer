"""Training sequences for the compact belief (spec 10.5, 11.5).

A sequence starts at keyframe s, where the belief is initialised from the measured
joints, runs a burn-in up to keyframe a, then scores `scored` macro steps:

    s = a - burn_in * stride (or the segment start, whichever is later)
    steps:  s (initialise) .. a (burn-in, no loss) .. a + scored * stride (scored)

Burn-in is shortened, not invented, when the segment starts later; such a sequence
is padded at the end and the padding marked invalid. Truncated BPTT detaches the
state after step `burn_in_steps`, per sequence.

Admission. Every keyframe from s to the last scored one lies in one segment (no
reset, calibration change or unlogged gap), every raw tick duration in that span is
within tolerance, every command inside the span is valid, and the joint reading at s
is valid. Frames and joints inside the span may be invalid: they become masked
coordinates, which is what the filter must learn to handle.

Units: joints in the observation are compact coordinates, issued commands physical
(the model converts them), times in seconds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import torch

from ..learning.windows import Episode
from ..robot_io.so101_units import So101Units
from .sketch import ObservationSketch


@dataclass(frozen=True)
class SequenceSpec:
    scored: int = 32
    burn_in: int = 16
    stride: int = 6
    history_ticks: int = 30
    tick_s: float = 1.0 / 30.0
    timing_tolerance: float = 0.5

    @property
    def max_steps(self) -> int:
        return self.burn_in + 1 + self.scored


def command_block(episode: Episode, end_tick: int, spec: SequenceSpec) -> dict[str, np.ndarray]:
    """Issued targets for the block ending at end_tick, with its causal history.

    Ticks [end_tick - stride - history, end_tick). Ticks before the segment start, or
    before the episode, hold the first in-segment measured position with validity 0
    and nominal duration: no command was logged there, and holding invents no motion.
    """
    length = spec.history_ticks + spec.stride
    first = end_tick - length
    segment = episode.segment[end_tick]
    start = max(first, 0)
    while start < end_tick and episode.segment[start] != segment:
        start += 1
    pad = start - first
    issued = np.empty((length, 6))
    issued[:pad] = episode.position[start]
    issued[pad:] = episode.issued[start:end_tick]
    valid = np.zeros(length, dtype=bool)
    valid[pad:] = episode.command_valid[start:end_tick]
    dt = np.full(length, spec.tick_s)
    dt[pad:] = episode.time_s[start + 1 : end_tick + 1] - episode.time_s[start:end_tick]
    return {"issued": issued, "tick_dt": dt, "command_valid": valid}


def sequence_admissible(episode: Episode, start: int, last: int, spec: SequenceSpec) -> bool:
    if start < 0 or last >= episode.length:
        return False
    if episode.segment[start] != episode.segment[last]:
        return False
    if not episode.position_valid[start]:
        return False
    if not episode.command_valid[start:last].all():
        return False
    durations = np.diff(episode.time_s[start : last + 1])
    return bool((np.abs(durations - spec.tick_s) <= spec.timing_tolerance * spec.tick_s).all())


def sequence_starts(episode: Episode, spec: SequenceSpec, step_every: Optional[int] = None) -> Iterator[tuple[int, int]]:
    """Yield (s, a) pairs on keyframe ticks (multiples of stride).

    step_every spaces scored starts, default every keyframe. Overlapping sequences
    from one episode are not independent data (spec 11.1).
    """
    stride = spec.stride
    step_every = step_every or stride
    for a in range(0, episode.length, step_every):
        if a % stride:
            continue
        last = a + spec.scored * stride
        if last >= episode.length:
            break
        s = a
        while s - stride >= 0 and (a - (s - stride)) // stride <= spec.burn_in:
            if not sequence_admissible(episode, s - stride, last, spec):
                break
            s -= stride
        if sequence_admissible(episode, s, last, spec):
            yield s, a


class BeliefSequences:
    """Builds padded sequence batches from episodes and their keyframe features."""

    def __init__(
        self,
        episodes: Sequence[Episode],
        view_ids: dict[str, list[int]],
        features: Callable[[Episode, int], torch.Tensor],
        sketch: ObservationSketch,
        units: So101Units,
        spec: SequenceSpec,
        step_every: Optional[int] = None,
        regime_labels: Optional[Callable[[Episode, int], int]] = None,
    ):
        self.episodes = {e.episode_id: e for e in episodes}
        self.view_ids = view_ids
        self.features = features
        self.sketch = sketch
        self.units = units
        self.spec = spec
        self.regime_labels = regime_labels
        self.index = [(e.episode_id, s, a) for e in episodes for s, a in sequence_starts(e, spec, step_every)]

    def __len__(self) -> int:
        return len(self.index)

    def _one(self, episode_id: str, s: int, a: int) -> dict:
        episode = self.episodes[episode_id]
        spec, stride = self.spec, self.spec.stride
        last = a + spec.scored * stride
        keyframes = np.arange(s, last + 1, stride)
        burn = (a - s) // stride
        joints = self.units.physical_to_compact(torch.as_tensor(episode.position[keyframes], dtype=torch.float32))
        tokens = torch.stack([self.features(episode, int(k)) for k in keyframes]).float()
        visual, view_valid = self.sketch.observe(
            tokens, self.view_ids[episode_id], torch.as_tensor(episode.frame_valid[keyframes])
        )
        blocks = [command_block(episode, int(k), spec) for k in keyframes]
        dt = np.empty(len(keyframes))
        dt[0] = stride * spec.tick_s
        dt[1:] = np.diff(episode.time_s[keyframes])
        per_view = self.sketch.components
        joint_valid = torch.as_tensor(episode.position_valid[keyframes])
        mask = torch.cat([joint_valid[:, None].expand(-1, 6), view_valid.repeat_interleave(per_view, 1)], 1)
        item = {
            "observation": torch.cat([joints, visual], 1),
            "obs_mask": mask,
            "issued": torch.as_tensor(np.stack([b["issued"] for b in blocks]), dtype=torch.float32),
            "tick_dt": torch.as_tensor(np.stack([b["tick_dt"] for b in blocks]), dtype=torch.float32),
            "command_valid": torch.as_tensor(np.stack([b["command_valid"] for b in blocks])),
            "dt": torch.as_tensor(dt, dtype=torch.float32),
            "burn_in": burn,
        }
        if self.regime_labels is not None:
            item["regime"] = torch.as_tensor([self.regime_labels(episode, int(k)) for k in keyframes])
        return item

    def batch(self, entries: Sequence[tuple[str, int, int]], device="cpu") -> dict:
        """Pad to the longest sequence; padding steps are invalid and never scored."""
        items = [self._one(*e) for e in entries]
        T = max(item["observation"].shape[0] for item in items)
        out: dict = {}
        for key in ("observation", "obs_mask", "issued", "tick_dt", "command_valid", "dt", "regime"):
            if key not in items[0]:
                continue
            padded = []
            for item in items:
                value = item[key]
                missing = T - value.shape[0]
                if missing:
                    filler = value[-1:].expand(missing, *value.shape[1:]).clone()
                    if key in ("obs_mask", "command_valid"):
                        filler = torch.zeros_like(filler)
                    value = torch.cat([value, filler])
                padded.append(value)
            out[key] = torch.stack(padded).to(device)
        lengths = torch.tensor([item["observation"].shape[0] for item in items], device=device)
        steps = torch.arange(T, device=device)[None]
        out["step_valid"] = steps < lengths[:, None]
        out["burn_in"] = torch.tensor([item["burn_in"] for item in items], device=device)
        # Scored: after the burn-in boundary, inside the sequence, never step 0.
        out["score"] = (steps > out["burn_in"][:, None]) & out["step_valid"]
        return out

    def sample(self, batch_size: int, rng: random.Random, device="cpu") -> dict:
        return self.batch(rng.choices(self.index, k=batch_size), device)

    def fixed(self, limit: int, batch_size: int, seed: int = 0, device="cpu") -> list[dict]:
        """A fixed evaluation set, so scores are comparable between evaluations."""
        chosen = random.Random(seed).sample(self.index, min(limit, len(self.index)))
        return [self.batch(chosen[i : i + batch_size], device) for i in range(0, len(chosen), batch_size)]


def joint_loss_scale(episodes: Sequence[Episode], units: So101Units, stride: int, floor: float = 1e-3) -> torch.Tensor:
    """s_loss of eq. 27: per-joint median absolute one-macro-step movement, floored."""
    moves = []
    for e in episodes:
        p = units.physical_to_compact(torch.as_tensor(e.position[: e.length : stride], dtype=torch.float64))
        ok = torch.as_tensor(e.position_valid[: e.length : stride])
        same = torch.as_tensor(e.segment[: e.length : stride])
        d = (p[1:] - p[:-1]).abs()
        keep = ok[1:] & ok[:-1] & (same[1:] == same[:-1])
        moves.append(d[keep])
    moves = torch.cat(moves)
    if moves.shape[0] == 0:
        raise ValueError("no valid one-step movements to set the joint loss scale")
    return moves.median(0).values.clamp_min(floor).float()


def residual_bound_from_data(
    episodes: Sequence[Episode], units: So101Units, stride: int, quantile: float = 0.995, floor: float = 1e-3
) -> tuple[float, ...]:
    """a_max of spec 6.4 from validated movement per interval.

    The residual corrects the nominal servo prediction, so its size cannot exceed the
    largest real movement in one interval. This takes the given quantile of absolute
    compact movement per macro step, a deliberately generous bound. Inspect it.
    """
    moves = []
    for e in episodes:
        p = units.physical_to_compact(torch.as_tensor(e.position[: e.length : stride], dtype=torch.float64))
        ok = torch.as_tensor(e.position_valid[: e.length : stride])
        same = torch.as_tensor(e.segment[: e.length : stride])
        keep = ok[1:] & ok[:-1] & (same[1:] == same[:-1])
        moves.append((p[1:] - p[:-1]).abs()[keep])
    moves = torch.cat(moves)
    return tuple(float(v) for v in torch.quantile(moves, quantile, dim=0).clamp_min(floor))
