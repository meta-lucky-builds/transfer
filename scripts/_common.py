"""Shared setup for command-line scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robojepa_ttt.data.dataset_meta import DatasetTiming, common_timing, timing_for  # noqa: E402
from robojepa_ttt.data.feature_store import DiskFeatures  # noqa: E402
from robojepa_ttt.data.offline_dataset import load_episode  # noqa: E402
from robojepa_ttt.learning.replay import FeatureCache  # noqa: E402
from robojepa_ttt.models.adapted_predictor import AdaptedPredictor  # noqa: E402
from robojepa_ttt.models.servo import ServoParams  # noqa: E402
from robojepa_ttt.models.upstream import (  # noqa: E402
    build_predictor,
    load_json_config,
    load_predictor_weights,
    resolve_embodiment_id,
)
from robojepa_ttt.robot_io.so101_units import So101Units  # noqa: E402


def load_units(path) -> So101Units:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return So101Units(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})


def load_servo(path) -> ServoParams:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return ServoParams(tuple(values["tau_s"]), tuple(values["delay_ticks"]), tuple(values.get("fit_rmse", ())))


def load_episodes(directory):
    episodes, view_ids = [], {}
    for path in sorted(Path(directory).glob("*.npz")):
        if path.name.endswith(".tmp.npz"):  # an interrupted write, never a complete episode
            continue
        episode, views = load_episode(path)
        episodes.append(episode)
        view_ids[episode.episode_id] = views
    if not episodes:
        raise FileNotFoundError(f"no episodes in {directory}")
    return episodes, view_ids


def feature_cache(directory, stride: int, byte_budget: int) -> FeatureCache:
    disk = DiskFeatures(directory, stride)
    return FeatureCache(lambda episode, tick: disk.get(episode, tick), byte_budget)


def yaml_timing(settings) -> DatasetTiming:
    timing = settings.raw["timing"]
    return timing_for(timing["io_target_hz"], timing["nominal_macro_dt_s"], timing["command_history_s"])


def dataset_timing(settings, *directories) -> DatasetTiming:
    """Timing from the datasets' dataset.json, falling back to the YAML for older data.

    Refuses datasets with different rates: a model trained at one rate cannot be
    validated or tested at another.
    """
    timing = common_timing(*directories, default=yaml_timing(settings))
    # stderr: scripts print JSON results on stdout, which must stay parseable.
    print(f"timing: {timing.fps:g} fps, {timing.stride} ticks per macro step, {timing.history_ticks} history ticks", file=sys.stderr)
    return timing


def build_model(config_path, weights_path, units, servo, settings, device, checkpointing=False, timing=None) -> AdaptedPredictor:
    """Load weights first: wrapping attention renames the keys they load into."""
    config = load_json_config(config_path)
    config["use_activation_checkpointing"] = checkpointing
    predictor = build_predictor(config)
    load_predictor_weights(predictor, weights_path)
    lora = settings.raw["predictor"]["lora"]
    timing = timing or yaml_timing(settings)
    model = AdaptedPredictor(
        predictor,
        units,
        servo,
        resolve_embodiment_id(config),
        last_blocks=lora["last_blocks"],
        lora_rank=lora["rank"],
        lora_alpha=lora["alpha"],
        stride=timing.stride,
        tick_s=timing.tick_s,
        history_ticks=timing.history_ticks,
        amp_dtype=torch.bfloat16 if torch.device(device).type == "cuda" else None,
    )
    return model.to(device)


__all__ = ["ROOT", "build_model", "dataset_timing", "feature_cache", "load_episodes", "load_servo", "load_units", "np", "yaml_timing"]
