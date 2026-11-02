"""Shared fixtures: the real upstream predictor at tiny size, on CPU, sdpa attention."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robojepa_ttt.models.adapted_predictor import AdaptedPredictor  # noqa: E402
from robojepa_ttt.models.servo import ServoParams  # noqa: E402
from robojepa_ttt.models.token_residual import TokenResidual  # noqa: E402
from robojepa_ttt.models.upstream import build_predictor  # noqa: E402
from robojepa_ttt.robot_io.so101_units import So101Units  # noqa: E402

EMBODIMENTS = ["franka", "so-101", "so-101_joints"]


def tiny_config(num_heads=4, num_kv_heads=4, frames=16, views=2):
    """Mirrors wm_300m_config.json except for size. Head dim stays 32."""
    return dict(
        action_conditioning="token",
        action_dim=64,
        action_tokens=1,
        attn_block_size=64,
        attn_impl="sdpa",
        camera_extrinsics_dim=6,
        camera_extrinsics_tokens=0,
        decode_state=False,
        depth=4,
        embed_dim=32,
        embodiment_list=EMBODIMENTS,
        encoder_layernorm=True,
        grid_height=3,
        grid_width=4,
        local_attn_window=[8, -1, -1],
        norm_type="rmsnorm",
        num_frames=frames,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_views=views,
        patch_size=1,
        predictor_embed_dim=32 * num_heads,
        qk_norm_type="rmsnorm",
        state_dim=64,
        state_tokens=1,
        tubelet_size=1,
        uniform_power=True,
        use_activation_checkpointing=False,
        use_alibi=False,
        use_dt=False,
        use_multiembodiment_proj=True,
        use_qk_norm=True,
        use_rope=True,
        use_silu=False,
        use_strong_view_conditioning=True,
        use_view_tokens=False,
        view_groups_v=[1, 2],
        view_map=[0, 1, 2, 3, 4, 5, 6],
    )


UNITS = So101Units(gripper_direction_verified=True)
SERVO = ServoParams(tau_s=(0.1,) * 6, delay_ticks=(2,) * 6)


LAYOUTS = {"fused-300m": (4, 4), "split-2b": (4, 2)}
ACTIVE = {"heads": (4, 2)}


@pytest.fixture(params=sorted(LAYOUTS), autouse=True)
def attention_layout(request):
    """Every test runs on both attention layouts. 2B uses the split one."""
    ACTIVE["heads"] = LAYOUTS[request.param]
    yield request.param


def make_model(seed=0, num_heads=None, num_kv_heads=None, last_blocks=2, **kwargs):
    num_heads = ACTIVE["heads"][0] if num_heads is None else num_heads
    num_kv_heads = ACTIVE["heads"][1] if num_kv_heads is None else num_kv_heads
    torch.manual_seed(seed)
    predictor = build_predictor(tiny_config(num_heads, num_kv_heads))
    predictor.eval()
    residual = TokenResidual(token_dim=32, command_dim=16, hidden=(32, 32), regime_dim=4, view_dim=4, bottleneck=8)
    model = AdaptedPredictor(
        predictor,
        UNITS,
        SERVO,
        embodiment_id=2,
        last_blocks=last_blocks,
        lora_rank=4,
        lora_alpha=8.0,
        command_hidden=16,
        residual=residual,
        **kwargs,
    )
    return model


def make_batch(batch=2, frames=10, seed=1, views=2, history=30, stride=6):
    generator = torch.Generator().manual_seed(seed)
    ticks = history + (frames - 1) * stride
    features = torch.nn.functional.layer_norm(
        torch.randn(batch, frames, views, 3, 4, 32, generator=generator), (32,)
    )
    issued = torch.randn(batch, ticks, 6, generator=generator).cumsum(1) * 0.01
    return {
        "features": features,
        "position": torch.randn(batch, frames, 6, generator=generator).cumsum(1) * 0.02,
        "issued": issued,
        "dt": torch.full((batch, ticks), 1.0 / 30.0),
        "command_valid": torch.ones(batch, ticks, dtype=torch.bool),
        "view_ids": torch.tensor([[0, 2]] * batch)[:, :views],
    }


@pytest.fixture
def model():
    return make_model()


@pytest.fixture
def batch():
    return make_batch()


def set_adapters_nonzero(model, seed=3):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_b") or "b_o" in name:
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)


__all__ = ["make_model", "make_batch", "tiny_config", "set_adapters_nonzero", "UNITS", "SERVO", "np"]
