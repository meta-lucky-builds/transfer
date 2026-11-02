"""Episodes on disk and keyframe-start sampling for offline training."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

from ..learning.replay import FeatureCache, collate_windows
from ..learning.windows import Episode, WindowSpec, sparse_starts

FIELDS = ("time_s", "position", "issued", "position_valid", "command_valid", "frame_valid", "segment")


def save_episode(episode: Episode, directory: str | os.PathLike, view_ids: list[int]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{len(list(directory.glob('*.npz'))):08d}.npz"
    arrays = {name: getattr(episode, name)[: episode.length] for name in FIELDS}
    meta = {
        "episode_id": episode.episode_id,
        "calibration_hash": episode.calibration_hash,
        "encoder_hash": episode.encoder_hash,
        "views": episode.views,
        "view_ids": view_ids,
    }
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    return path


def load_episode(path: str | os.PathLike) -> tuple[Episode, list[int]]:
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        length = len(data["time_s"])
        episode = Episode(meta["episode_id"], meta["calibration_hash"], meta["encoder_hash"], meta["views"], max(length, 1))
        for name in FIELDS:
            getattr(episode, name)[:length] = data[name]
        episode.length = length
    return episode, meta["view_ids"]


class KeyframeSampler:
    """Random keyframe-aligned windows drawn from a fixed set of episodes.

    Episodes differing in views cannot share a tensor, so batches stay within one view
    count, chosen in proportion to its windows.
    """

    def __init__(self, episodes, view_ids, spec: WindowSpec, cache: FeatureCache, batch: int, device, seed: int):
        self.episodes = {e.episode_id: e for e in episodes}
        self.view_ids = view_ids
        self.spec = spec
        self.cache = cache
        self.batch = batch
        self.device = device
        self.rng = random.Random(seed)
        self.by_views: dict[int, list] = {}
        for episode in episodes:
            for start in sparse_starts(episode, spec):
                self.by_views.setdefault(episode.views, []).append((episode.episode_id, start))
        self.total = sum(len(v) for v in self.by_views.values())
        if self.total == 0:
            raise ValueError("no valid windows; check timing tolerance and episode length")

    def next_batch(self) -> dict:
        views = self.rng.choices(list(self.by_views), [len(v) for v in self.by_views.values()])[0]
        windows = self.rng.choices(self.by_views[views], k=self.batch)
        return collate_windows(windows, self.episodes, self.cache, self.spec, self.view_ids, self.device)

    def ordered_batches(self, limit: int, seed: int = 0):
        """A fixed validation set, so that scores are comparable between evaluations."""
        rng = random.Random(seed)
        for views, windows in self.by_views.items():
            chosen = rng.sample(windows, min(limit, len(windows)))
            for start in range(0, len(chosen), self.batch):
                yield collate_windows(
                    chosen[start : start + self.batch], self.episodes, self.cache, self.spec, self.view_ids, self.device
                )
