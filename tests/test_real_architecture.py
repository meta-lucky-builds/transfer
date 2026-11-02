"""Released configurations, built at full size without weights, and settings sync."""

import json
import sys
from pathlib import Path

import pytest
import torch

from robojepa_ttt.config import load_settings
from robojepa_ttt.data.offline_dataset import KeyframeSampler, load_episode, save_episode
from robojepa_ttt.evaluation.prediction import compare_by_episode
from robojepa_ttt.learning.replay import FeatureCache
from robojepa_ttt.learning.windows import Episode, WindowSpec
from robojepa_ttt.models.lora import attach_lora, lora_parameters
from robojepa_ttt.models.upstream import build_predictor, load_json_config, load_predictor_weights, resolve_embodiment_id

import numpy as np

SOURCES = Path(__file__).resolve().parents[1] / "sources"


@pytest.mark.parametrize(
    "name,layout,kv_width",
    [("wm_300m_config.json", "fused", None), ("wm_2b_config.json", "split", 8 * 128 * 2)],
)
def test_released_config_accepts_lora_on_last_eight_blocks(name, layout, kv_width):
    config = load_json_config(SOURCES / name)
    config["use_activation_checkpointing"] = False
    # Attention masks are prebuilt per frame count; two frame counts are enough here.
    config["num_frames"] = 2
    predictor = build_predictor(config)
    assert resolve_embodiment_id(config) == 25
    wrapped = attach_lora(predictor, 8, rank=8, alpha=16)
    attn = predictor.predictor_blocks[-1].attn
    assert hasattr(attn, "qkv") == (layout == "fused")
    assert len(wrapped) == 8 * (2 if layout == "fused" else 3)
    if kv_width:
        assert attn.kv.base.out_features == kv_width
    trainable = sum(p.numel() for p in lora_parameters(predictor, 0))
    total = sum(p.numel() for p in predictor.parameters()) - trainable
    assert trainable / total < 0.01, "adapters must stay a small fraction of the model"


def test_unknown_embodiment_is_refused():
    with pytest.raises(ValueError, match="not in this checkpoint"):
        resolve_embodiment_id({"embodiment_list": ["franka"]})


def test_weight_loading_reports_mismatches(tmp_path):
    from conftest import tiny_config

    predictor = build_predictor(tiny_config())
    state = predictor.state_dict()
    path = tmp_path / "w.pt"
    torch.save({"model": {f"module.{k}": v for k, v in state.items()}}, path)
    load_predictor_weights(predictor, path)
    state.pop(next(iter(state)))
    torch.save({"predictor": state}, path)
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        load_predictor_weights(predictor, path)


def test_settings_match_config_file():
    settings = load_settings()
    assert settings.online.lora_lr == 1e-5 and settings.online.residual_lr == 3e-5
    assert settings.offline.curriculum == ((5000, 2), (10000, 5), (5000, 10))
    assert settings.promotion.future_macro_transitions == 128
    assert settings.replay.recent_windows == 10_000
    assert settings.cem.candidates == 128 and settings.cem.view_weights == (0.5, 0.5)
    assert settings.cem.microbatch == 1
    assert settings.raw["predictor"]["profile"] == "wm_2b"
    assert settings.credits().windows_per_credit == 16


def test_episode_files_round_trip_and_sampler_uses_keyframes(tmp_path):
    episode = Episode("session/1", "cal", "enc", 2, capacity=4)
    for n in range(60):
        episode.append(n / 30, np.full(6, n / 100), True, np.full(6, n / 90), True, np.ones(2, bool))
    path = save_episode(episode, tmp_path, [0, 2])
    loaded, views = load_episode(path)
    assert loaded.length == 60 and views == [0, 2] and loaded.episode_id == "session/1"
    assert np.array_equal(loaded.position[:60], episode.position[:60])

    seen = []

    def encode(episode_id, tick):
        seen.append(tick)
        return torch.zeros(2, 3, 4, 32)

    spec = WindowSpec(frames=4)
    sampler = KeyframeSampler([loaded], {"session/1": views}, spec, FeatureCache(encode, 10**9), 2, "cpu", 0)
    batch = sampler.next_batch()
    assert batch["features"].shape == (2, 4, 2, 3, 4, 32)
    assert all(t % 6 == 0 for t in seen)


def test_episode_comparison_uses_episodes_as_the_unit():
    baseline = {f"e{i}": 1.0 + 0.01 * i for i in range(10)}
    adapted = {k: v * 0.9 for k, v in baseline.items()}
    result = compare_by_episode(baseline, adapted)
    assert result.relative_improvement == pytest.approx(0.1)
    assert result.episodes == 10 and result.ci_low <= 0.1 <= result.ci_high
