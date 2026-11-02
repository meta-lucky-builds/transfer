"""Bounded replay partitions, feature cache and update credits (spec 11.2 to 11.4)."""

from __future__ import annotations

import random
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Hashable, Optional

import numpy as np
import torch

from .windows import Episode, WindowSpec, extract_window

WindowRef = tuple[str, int]  # (episode_id, raw start tick)


class FeatureCache:
    """LRU feature cache with a byte budget. Keys include the encoder hash.

    Spec 11.2 costs one two-view frame at about 2 MB, so the configured 20 GB holds
    about 10,000 frames, or under six minutes at 30 Hz. Dense replay therefore
    re-encodes often and encoder throughput is a real limit, not a detail.
    """

    def __init__(self, encode: Callable[[str, int], torch.Tensor], byte_budget: int):
        self.encode = encode
        self.byte_budget = byte_budget
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self._items: OrderedDict[Hashable, torch.Tensor] = OrderedDict()

    def get(self, encoder_hash: str, episode_id: str, tick: int) -> torch.Tensor:
        key = (encoder_hash, episode_id, tick)
        if key in self._items:
            self.hits += 1
            self._items.move_to_end(key)
            return self._items[key]
        self.misses += 1
        value = self.encode(episode_id, tick)
        self._items[key] = value
        self.bytes += value.numel() * value.element_size()
        while self.bytes > self.byte_budget and len(self._items) > 1:
            _, evicted = self._items.popitem(last=False)
            self.bytes -= evicted.numel() * evicted.element_size()
        return value


class Reservoir:
    """Uniform reservoir (algorithm R) over everything ever offered to it."""

    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.rng = rng
        self.items: list[WindowRef] = []
        self.seen = 0

    def add(self, item: WindowRef) -> Optional[WindowRef]:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return None
        slot = self.rng.randrange(self.seen)
        if slot < self.capacity:
            evicted, self.items[slot] = self.items[slot], item
            return evicted
        return item


@dataclass(frozen=True)
class ReplayConfig:
    recent_windows: int = 10_000
    reservoir_windows: int = 20_000
    anchor_windows: int = 5_000
    quarantine_windows: int = 5_000
    recent_fraction: float = 0.5
    regime_fraction: float = 0.25
    anchor_fraction: float = 0.25


class ReplayBuffer:
    """Recent FIFO, per-regime reservoirs, immutable anchors, quarantine.

    Windows are references only. Reference counts tell the owner when no partition
    still needs an episode's raw data.
    """

    def __init__(self, config: ReplayConfig, regimes: int = 1, seed: int = 0):
        self.config = config
        self.rng = random.Random(seed)
        self.recent: deque[WindowRef] = deque()
        self.reservoirs = [Reservoir(config.reservoir_windows, self.rng) for _ in range(regimes)]
        self.anchors: list[WindowRef] = []
        self.quarantine: deque[WindowRef] = deque()
        self.references: dict[str, int] = {}

    def _retain(self, window: WindowRef):
        self.references[window[0]] = self.references.get(window[0], 0) + 1

    def _release(self, window: WindowRef):
        self.references[window[0]] -= 1
        if self.references[window[0]] == 0:
            del self.references[window[0]]

    def add(self, window: WindowRef, regime: Optional[int]) -> None:
        """regime None sends the window to quarantine, not to any regime's data."""
        if regime is None:
            self.quarantine.append(window)
            self._retain(window)
            if len(self.quarantine) > self.config.quarantine_windows:
                self._release(self.quarantine.popleft())
            return
        self.recent.append(window)
        self._retain(window)
        if len(self.recent) > self.config.recent_windows:
            self._release(self.recent.popleft())
        self._retain(window)
        evicted = self.reservoirs[regime].add(window)
        if evicted is not None:
            self._release(evicted)

    def add_anchor(self, window: WindowRef) -> bool:
        if len(self.anchors) >= self.config.anchor_windows:
            return False
        self.anchors.append(window)
        self._retain(window)
        return True

    def sample(self, batch_size: int, regime: int) -> tuple[list[WindowRef], list[bool]]:
        """Draw without replacement within the batch, as JEPA-TTT does.

        Returns windows and an anchor flag per window. If a partition is too small,
        its share moves to the recent partition. With no recent data the draw fails
        rather than returning a smaller batch, since microbatch normalisation assumes
        the full size.
        """
        reservoir = self.reservoirs[regime].items
        want_anchor = round(batch_size * self.config.anchor_fraction)
        want_regime = round(batch_size * self.config.regime_fraction)
        chosen: dict[WindowRef, bool] = {}
        for item in self.rng.sample(self.anchors, min(want_anchor, len(self.anchors))):
            chosen[item] = True
        for item in self.rng.sample(reservoir, min(want_regime, len(reservoir))):
            chosen.setdefault(item, False)
        # Keep the anchors' fair share but fill every shortfall from recent.
        pool = list(self.recent)
        self.rng.shuffle(pool)
        for item in pool:
            if len(chosen) >= batch_size:
                break
            chosen.setdefault(item, False)
        if len(chosen) < batch_size:
            raise ValueError(f"only {len(chosen)} unique windows available for a batch of {batch_size}")
        items = list(chosen.items())
        self.rng.shuffle(items)
        return [w for w, _ in items], [a for _, a in items]


@dataclass
class CreditLedger:
    """Update budget of spec 11.4.

    One credit per windows_per_credit new unique windows, nothing before
    minimum_windows, a backlog cap, at most max_updates_per_s, and a compute duty
    limit. Credit is spent only after a batch exists, so a failed update can be
    refunded. Stale credit above the cap is counted, never silently lost.
    """

    windows_per_credit: int = 16
    minimum_windows: int = 256
    max_backlog: int = 100
    max_updates_per_s: float = 1.0
    max_duty: float = 0.1
    windows_seen: int = 0
    earned: int = 0
    spent: int = 0
    discarded: int = 0
    last_update_s: float = -1e18
    busy_s: float = 0.0
    start_s: Optional[float] = None

    @property
    def available(self) -> int:
        return self.earned - self.spent - self.discarded

    def admit(self, count: int = 1) -> None:
        self.windows_seen += count
        total = self.windows_seen // self.windows_per_credit
        self.earned = max(self.earned, total)
        overflow = self.available - self.max_backlog
        if overflow > 0:
            self.discarded += overflow

    def can_update(self, now_s: float) -> bool:
        if self.start_s is None:
            self.start_s = now_s
        if self.windows_seen < self.minimum_windows or self.available <= 0:
            return False
        if now_s - self.last_update_s < 1.0 / self.max_updates_per_s:
            return False
        elapsed = max(now_s - self.start_s, 1e-9)
        return self.busy_s / elapsed < self.max_duty or self.busy_s == 0.0

    def spend(self, now_s: float) -> None:
        self.spent += 1
        self.last_update_s = now_s

    def refund(self) -> None:
        self.spent -= 1

    def record_busy(self, seconds: float) -> None:
        self.busy_s += seconds


def collate_windows(
    windows: list[WindowRef],
    episodes: dict[str, Episode],
    cache: FeatureCache,
    spec: WindowSpec,
    view_ids: dict[str, list[int]],
    device: str | torch.device = "cpu",
) -> dict:
    """Assemble tensors for model.rollout. Raw records stay the authoritative source."""
    extracted, features = [], []
    for episode_id, start in windows:
        episode = episodes[episode_id]
        window = extract_window(episode, start, spec)
        extracted.append(window)
        features.append(
            torch.stack([cache.get(episode.encoder_hash, episode_id, int(t)) for t in window["keyframes"]])
        )

    def stack(key, dtype=torch.float32):
        return torch.as_tensor(np.stack([w[key] for w in extracted]), dtype=dtype, device=device)

    return {
        "features": torch.stack(features).to(device),
        "position": stack("position"),
        "issued": stack("issued"),
        "dt": stack("dt"),
        "command_valid": stack("command_valid", torch.bool),
        "view_ids": torch.as_tensor([view_ids[e] for e, _ in windows], dtype=torch.long, device=device),
    }
