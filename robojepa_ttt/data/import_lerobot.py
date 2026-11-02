"""Import Hugging Face LeRobot SO-101 episodes into Episode records.

What imported data cannot provide, and what this importer does about it:

* The saved action is the requested action, not the sent one, so `max_relative_target`
  clipping is invisible. It is imported as issued and the source is marked.
* Timing is the dataset's stored timestamp. Treat it as nominal.
* Body units are degrees or a calibrated percentage, and the metadata may not say
  which. The caller must state it, because the wrong guess is silently destructive.
* Each dataset is its own arm, camera rig and calibration. Episodes from different
  uploads must never share a calibration hash.

The recorded action leads the state by about 4 raw frames (so101.md), which the servo
fit absorbs as delay. Fit servo parameters per dataset, then refit on your own arm.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..learning.windows import Episode
from ..robot_io.so101_units import JOINT_ORDER, So101Units


@dataclass(frozen=True)
class DatasetCheck:
    robot_type: str
    fps: float
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]


def check_dataset_info(info_path: str | Path, expected_fps: float = 30.0) -> DatasetCheck:
    """Reject datasets whose joints or rate do not match, before any encoding cost."""
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    features = info["features"]
    state_names = tuple(n.removesuffix(".pos") for n in _names(features["observation.state"]))
    action_names = tuple(n.removesuffix(".pos") for n in _names(features["action"]))
    if state_names != JOINT_ORDER or action_names != JOINT_ORDER:
        raise ValueError(f"joint order differs: state {state_names}, action {action_names}")
    if float(info["fps"]) != expected_fps:
        raise ValueError(f"expected {expected_fps} fps, dataset has {info['fps']}")
    return DatasetCheck(str(info.get("robot_type")), float(info["fps"]), state_names, action_names)


def _names(feature: dict) -> list[str]:
    names = feature.get("names")
    if isinstance(names, dict):
        names = next(iter(names.values()))
    if not names:
        raise ValueError("dataset does not name its joints, so order cannot be verified")
    return list(names)


def episode_from_arrays(
    episode_id: str,
    dataset_id: str,
    units: So101Units,
    state_device: np.ndarray,
    action_device: np.ndarray,
    timestamps: np.ndarray,
    encoder_hash: str,
    views: int,
) -> Episode:
    """Build one Episode. Device arrays are [N, 6] in JOINT_ORDER."""
    position = units.device_to_physical(np.asarray(state_device, dtype=np.float64))
    issued = units.device_to_physical(np.asarray(action_device, dtype=np.float64))
    calibration = hashlib.sha256(dataset_id.encode()).hexdigest()[:16]
    episode = Episode(episode_id, calibration, encoder_hash, views, capacity=max(len(position), 8))
    frames_ok = np.ones(views, dtype=bool)
    for n in range(len(position)):
        ok = bool(np.isfinite(position[n]).all() and np.isfinite(issued[n]).all())
        episode.append(float(timestamps[n]), position[n], ok, issued[n], ok, frames_ok)
    return episode


def split_episodes(
    groups: dict[str, list[str]], seed: int = 101, fractions: tuple[float, float, float, float] = (0.7, 0.1, 0.1, 0.1)
) -> dict[str, list[str]]:
    """Split whole sessions into train, validation, calibration and test (spec 10.2).

    groups maps a session to its episodes. Splitting by session stops near-identical
    neighbouring clips from reaching both sides. With too few sessions to fill every
    split this raises, as a fake test set is worse than none.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    sessions = sorted(groups)
    random.Random(seed).shuffle(sessions)
    if len(sessions) < 4:
        raise ValueError("need at least four sessions for four disjoint splits")
    edges, total = [], 0.0
    for fraction in fractions[:-1]:
        total += fraction
        edges.append(round(total * len(sessions)))
    # Every split keeps at least one session.
    edges[0] = max(edges[0], 1)
    edges[1] = max(edges[1], edges[0] + 1)
    edges[2] = min(max(edges[2], edges[1] + 1), len(sessions) - 1)
    names = ("train", "validation", "calibration", "test")
    bounds = [0, *edges, len(sessions)]
    return {
        name: [e for s in sessions[bounds[i] : bounds[i + 1]] for e in groups[s]] for i, name in enumerate(names)
    }
