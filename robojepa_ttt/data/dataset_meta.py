"""Per-dataset metadata written by scripts/prepare_lerobot.py and read by every trainer.

Timing belongs to the data, not to config_so101.yaml. RoboJEPA predicts in 0.2 s macro
steps, so a 30 fps recording uses 6 raw ticks per step and a 20 fps one uses 4. Using
the wrong stride would feed the checkpoint 0.3 s or 0.13 s motions as if they were
0.2 s ones. The servo model, command history and window lengths are all in ticks, so
models trained at one rate must run at that rate, or be refit (spec 4).

Layout of a prepared dataset:

    <root>/dataset.json      this metadata
    <root>/episodes/*.npz    one Episode per file
    <root>/features/*.pt     encoded keyframes per episode
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

META_NAME = "dataset.json"
WRIST_VIEW_ID = 2  # RoboJEPA view_map: left 0, right 1, wrist 2, top 5, front 6


@dataclass(frozen=True)
class DatasetTiming:
    fps: float
    stride: int  # raw ticks per macro step
    tick_s: float
    history_ticks: int  # one second of command history

    @property
    def macro_dt_s(self) -> float:
        return self.stride * self.tick_s


def timing_for(fps: float, macro_dt_s: float = 0.2, history_s: float = 1.0) -> DatasetTiming:
    """Exact whole ticks per macro step and per history window, or a refusal.

    30 fps -> 6 ticks, 20 fps -> 4, 25 fps -> 5; 7 fps has no whole-tick 0.2 s step.
    """
    ticks = fps * macro_dt_s
    if abs(ticks - round(ticks)) > 1e-6 or round(ticks) < 1:
        raise ValueError(f"{fps} fps does not give a whole number of ticks per {macro_dt_s} s step")
    history = fps * history_s
    if abs(history - round(history)) > 1e-6:
        raise ValueError(f"{fps} fps does not give a whole number of ticks in {history_s} s of history")
    return DatasetTiming(float(fps), int(round(ticks)), 1.0 / fps, int(round(history)))


@dataclass
class DatasetMeta:
    repo_id: str
    fps: float
    cameras: dict[str, int]  # dataset camera name -> semantic view id, in view order
    use_degrees: bool
    gripper_closed: float
    gripper_open: float
    encoder_hash: str
    episodes: list[dict] = field(default_factory=list)  # episode_id, source index, task, length
    notes: list[str] = field(default_factory=list)

    @property
    def view_ids(self) -> list[int]:
        return list(self.cameras.values())

    @property
    def timing(self) -> DatasetTiming:
        return timing_for(self.fps)

    def save(self, root: str | Path) -> Path:
        path = Path(root) / META_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temporary.replace(path)
        return path


def load_meta(path: str | Path) -> Optional[DatasetMeta]:
    """Find dataset.json for a dataset root or its episodes directory. None if absent."""
    path = Path(path)
    for candidate in (path / META_NAME, path.parent / META_NAME):
        if candidate.exists():
            return DatasetMeta(**json.loads(candidate.read_text(encoding="utf-8")))
    return None


def common_timing(*directories: str | Path, default: Optional[DatasetTiming] = None) -> DatasetTiming:
    """One timing for several datasets used together, or a refusal.

    Datasets without metadata use `default` (the YAML timing), so older prepared
    data keeps working.
    """
    timings = set()
    for directory in directories:
        meta = load_meta(directory)
        if meta is not None:
            timings.add(meta.timing)
        elif default is not None:
            timings.add(default)
        else:
            raise ValueError(f"{directory} has no {META_NAME} and no default timing was given")
    if len(timings) != 1:
        raise ValueError(f"datasets disagree on timing, they cannot be mixed: {sorted(timings, key=lambda t: t.fps)}")
    return timings.pop()


def parse_cameras(specs: list[str], allow_wrist: bool = False) -> dict[str, int]:
    """'front=6 top=5' -> {'front': 6, 'top': 5}, refusing the wrist view by default.

    The deployment rig has fixed external phone cameras and no wrist camera, so
    training must not learn to rely on a wrist view it will never get.
    """
    cameras: dict[str, int] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"camera '{spec}' must be name=semantic_id")
        name, value = spec.split("=", 1)
        view = int(value)
        if view == WRIST_VIEW_ID and not allow_wrist:
            raise ValueError(f"camera '{name}' maps to the wrist view {WRIST_VIEW_ID}; this rig has none (pass --allow-wrist to override)")
        if name in cameras or view in cameras.values():
            raise ValueError(f"camera names and view ids must be unique: {specs}")
        cameras[name] = view
    if not cameras:
        raise ValueError("at least one camera is needed")
    return cameras
