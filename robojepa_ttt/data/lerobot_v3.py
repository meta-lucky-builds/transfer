"""Read LeRobot v3.0 datasets without the lerobot package.

Why: lerobot 0.6.x pins torch < 2.12, which conflicts with the torch 2.14 build in
NVIDIA's NGC containers (checked with pip dry runs, 2026-11-03). This reader needs only
huggingface_hub, pandas/pyarrow and PyAV, and downloads only the files that hold the
episodes asked for.

v3.0 layout, all paths taken from meta/info.json rather than hard-coded:

    meta/info.json             fps, features, data_path and video_path templates
    meta/tasks.parquet         task string (index) -> task_index
    meta/episodes/**.parquet   per episode: length, tasks, data/chunk_index, data/file_index,
                               videos/<key>/{chunk_index, file_index, from_timestamp, to_timestamp}
    data_path template         parquet rows of many episodes (episode_index, timestamp, state, action)
    video_path template        one mp4 per (camera, chunk, file) holding many episodes back to back

A frame of episode e at in-episode timestamp t sits at file time from_timestamp(e) + t.
Decoding seeks near from_timestamp and keeps the decoded frame nearest each wanted time,
refusing any frame further than half a frame period away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class EpisodeRows:
    episode_index: int
    state: np.ndarray  # [N, 6] device units
    action: np.ndarray  # [N, 6] device units (the recorded, requested action)
    timestamp: np.ndarray  # [N] seconds from episode start
    task_index: Optional[int]


class LeRobotV3Source:
    def __init__(self, repo_id: str, root: Optional[str | Path] = None, revision: Optional[str] = None):
        """root: a local copy of the dataset (tests, or an existing download); else the Hub."""
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else None
        self.revision = revision
        if self.root is None:
            from huggingface_hub import snapshot_download

            self.meta_root = Path(
                snapshot_download(repo_id, repo_type="dataset", revision=revision, allow_patterns=["meta/**"])
            )
        else:
            self.meta_root = self.root
        self.info = json.loads((self.meta_root / "meta" / "info.json").read_text(encoding="utf-8"))
        if not str(self.info.get("codebase_version", "")).startswith("v3"):
            raise ValueError(f"this reader handles LeRobot v3 datasets, got {self.info.get('codebase_version')}")
        self.fps = float(self.info["fps"])
        self.data_path = self.info["data_path"]
        self.video_path = self.info.get("video_path")
        import pandas as pd

        files = sorted((self.meta_root / "meta" / "episodes").glob("**/*.parquet"))
        if not files:
            raise FileNotFoundError("meta/episodes has no parquet files")
        self.episodes = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).set_index("episode_index", drop=False).sort_index()
        tasks_file = self.meta_root / "meta" / "tasks.parquet"
        self.task_index_of: dict[str, int] = {}
        if tasks_file.exists():
            tasks = pd.read_parquet(tasks_file)
            column = "task_index"
            for task, row in tasks.iterrows():
                self.task_index_of[str(task)] = int(row[column])
        self._data_cache: dict[str, "pd.DataFrame"] = {}

    # ------------------------------------------------------------- files

    def local(self, relative: str) -> Path:
        """Path of a dataset file, downloading it when reading from the Hub."""
        if self.root is not None:
            path = self.root / relative
            if not path.exists():
                raise FileNotFoundError(path)
            return path
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.repo_id, relative, repo_type="dataset", revision=self.revision))

    def camera_keys(self) -> list[str]:
        return [k.removeprefix("observation.images.") for k in self.info["features"] if k.startswith("observation.images.")]

    def episode_task(self, episode: int) -> Optional[int]:
        row = self.episodes.loc[episode]
        if "tasks" in row and len(self.task_index_of):
            tasks = row["tasks"]
            first = tasks[0] if not isinstance(tasks, str) else tasks
            return self.task_index_of.get(str(first))
        return None

    def data_file(self, episode: int) -> str:
        row = self.episodes.loc[episode]
        return self.data_path.format(chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))

    def video_file(self, episode: int, camera: str) -> str:
        if self.video_path is None:
            raise ValueError("dataset has no video_path template")
        key = f"observation.images.{camera}"
        row = self.episodes.loc[episode]
        return self.video_path.format(
            video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]), file_index=int(row[f"videos/{key}/file_index"])
        )

    # -------------------------------------------------------------- rows

    def rows(self, episode: int) -> EpisodeRows:
        import pandas as pd

        relative = self.data_file(episode)
        if relative not in self._data_cache:
            if len(self._data_cache) > 4:
                self._data_cache.pop(next(iter(self._data_cache)))
            self._data_cache[relative] = pd.read_parquet(self.local(relative))
        table = self._data_cache[relative]
        part = table[table["episode_index"] == episode]
        if "frame_index" in part:
            part = part.sort_values("frame_index")
        expected = int(self.episodes.loc[episode]["length"])
        if len(part) != expected:
            raise ValueError(f"episode {episode}: {len(part)} rows in {relative}, metadata says {expected}")
        task = int(part["task_index"].iloc[0]) if "task_index" in part else self.episode_task(episode)
        return EpisodeRows(
            episode,
            np.stack(part["observation.state"].to_numpy()).astype(np.float64),
            np.stack(part["action"].to_numpy()).astype(np.float64),
            part["timestamp"].to_numpy(dtype=np.float64).ravel(),
            task,
        )

    # ------------------------------------------------------------ frames

    def frames(self, episode: int, camera: str, timestamps: Sequence[float]) -> np.ndarray:
        """RGB uint8 [K, H, W, 3] for the given in-episode timestamps of one camera."""
        import av

        key = f"observation.images.{camera}"
        row = self.episodes.loc[episode]
        start = float(row[f"videos/{key}/from_timestamp"])
        wanted = start + np.asarray(timestamps, dtype=np.float64)
        tolerance = 0.5 / self.fps
        best: list[Optional[np.ndarray]] = [None] * len(wanted)
        error = np.full(len(wanted), np.inf)
        with av.open(str(self.local(self.video_file(episode, camera)))) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            container.seek(max(int((wanted.min() - 1.0) / stream.time_base), 0), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = float(frame.pts * stream.time_base)
                if t > wanted.max() + tolerance:
                    break
                distance = np.abs(wanted - t)
                closer = (distance < error) & (distance <= tolerance)
                if closer.any():
                    image = frame.to_ndarray(format="rgb24")
                    for i in np.flatnonzero(closer):
                        best[i], error[i] = image, distance[i]
        missing = [i for i, image in enumerate(best) if image is None]
        if missing:
            raise ValueError(f"episode {episode} {camera}: no frame within {tolerance:.3f} s of {len(missing)} timestamps, e.g. {wanted[missing[0]]:.3f}")
        return np.stack(best)


def select_blocks(
    source: LeRobotV3Source, episodes_per_task: int, block: int = 10, seed: int = 101
) -> list[int]:
    """Task-balanced episodes taken as runs of consecutive episodes.

    Consecutive episodes share data and video files, so blocks keep downloads small,
    and they map naturally onto pseudo-session splits. Blocks start at deterministic
    random offsets and never overlap. Identical on every shard.
    """
    import random

    rng = random.Random(seed)
    by_task: dict[Optional[int], list[int]] = {}
    for episode in source.episodes.index:
        by_task.setdefault(source.episode_task(int(episode)), []).append(int(episode))
    chosen: list[int] = []
    for task in sorted(by_task, key=lambda t: (t is None, t)):
        members = sorted(by_task[task])
        want = min(episodes_per_task, len(members))
        starts = list(range(0, len(members), block))
        rng.shuffle(starts)
        picked: list[int] = []
        for s in starts:
            if len(picked) >= want:
                break
            picked += members[s : s + block][: want - len(picked)]
        chosen += picked
    return sorted(chosen)
