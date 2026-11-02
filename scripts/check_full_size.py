"""Run the real architecture at full size with random weights. CPU works, slowly.

Checks zero-adapter parity, gradients and KV parity with released dimensions. Random
weights say nothing about prediction quality.

    python scripts/check_full_size.py --config sources/wm_2b_config.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import torch

from _common import build_predictor, load_json_config
from robojepa_ttt.models.adapted_predictor import AdaptedPredictor
from robojepa_ttt.models.servo import ServoParams
from robojepa_ttt.models.token_residual import layer_norm_0
from robojepa_ttt.models.upstream import resolve_embodiment_id
from robojepa_ttt.robot_io.so101_units import So101Units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args()

    config = load_json_config(args.config)
    config["use_activation_checkpointing"] = False
    config["num_frames"] = 6
    torch.manual_seed(0)
    started = time.time()
    predictor = build_predictor(config).eval()
    model = AdaptedPredictor(
        predictor,
        So101Units(gripper_direction_verified=True),
        ServoParams((0.1,) * 6, (3,) * 6),
        resolve_embodiment_id(config),
    )
    print(f"built in {time.time() - started:.0f}s", flush=True)

    frames = args.frames
    ticks = 30 + (frames - 1) * 6
    batch = {
        "features": layer_norm_0(torch.randn(1, frames, 2, 15, 20, 1664)),
        "position": torch.randn(1, frames, 6).cumsum(1) * 0.02,
        "issued": torch.randn(1, ticks, 6).cumsum(1) * 0.01,
        "dt": torch.full((1, ticks), 1 / 30),
        "command_valid": torch.ones(1, ticks, dtype=torch.bool),
        "view_ids": torch.tensor([[0, 1]]),
    }
    report = {"parameters": sum(p.numel() for p in model.parameters())}
    with torch.no_grad():
        actions, states = model.sequence_inputs(batch, 2, 1)
        command = model.encode_commands(batch, 2)
        adapted = model.forward_tokens(batch["features"][:, :2], actions, states, batch["view_ids"], command)
        restored = {}
        for module in predictor.modules():
            if hasattr(module, "deltas"):
                restored[module] = module.deltas
                module.deltas = torch.nn.ModuleList()
        print("adapted forward done", flush=True)
    report["adapted_finite"] = bool(torch.isfinite(adapted).all())

    for module, deltas in restored.items():
        module.deltas = deltas
    with torch.no_grad():
        cached = model.rollout(batch, 2, frames - 1, use_kv_generation=True)
        full = model.rollout(batch, 2, frames - 1, use_kv_generation=False)
    report["kv_vs_full_max_error"] = float((cached - full).abs().max())
    print("kv parity done", flush=True)

    model.set_trainable("online")
    output = model.rollout(batch, 2, 2)
    output.abs().mean().backward()
    trainable = model.trainable_named()
    report["trainable"] = sum(p.numel() for p in trainable.values())
    report["lora_b_grad"] = float(sum(p.grad.abs().sum() for n, p in trainable.items() if n.endswith("lora_b")))
    report["seconds"] = time.time() - started
    print(json.dumps(report, indent=2))
    assert report["adapted_finite"] and report["lora_b_grad"] > 0 and report["kv_vs_full_max_error"] < 1e-3


if __name__ == "__main__":
    main()
