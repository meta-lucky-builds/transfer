"""First script to run on the GPU. No robot, no dataset, only the released checkpoint.

Checks what a CPU at tiny size cannot prove, and measures the numbers the spec leaves
empirical. Writes gpu_acceptance.json.

    python scripts/gpu_acceptance.py --predictor-config sources/wm_300m_config.json \
        --weights /path/to/world_model.pth.tar

On a 24 GB card add --microbatch 1 (or 2). The update still covers the full batch of
16 sequences, split into smaller pieces, so it gives the same result with less memory.
Timing numbers only describe the card the script ran on.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from _common import build_model
from robojepa_ttt.config import load_settings
from robojepa_ttt.learning.online_builder import OnlineBuilder
from robojepa_ttt.models.servo import ServoParams
from robojepa_ttt.models.token_residual import layer_norm_0
from robojepa_ttt.robot_io.so101_units import So101Units


def fake_batch(batch, frames, device, history=30, stride=6, views=2):
    ticks = history + (frames - 1) * stride
    return {
        "features": layer_norm_0(torch.randn(batch, frames, views, 15, 20, 1664, device=device)),
        "position": torch.randn(batch, frames, 6, device=device).cumsum(1) * 0.02,
        "issued": torch.randn(batch, ticks, 6, device=device).cumsum(1) * 0.01,
        "dt": torch.full((batch, ticks), 1 / 30, device=device),
        "command_valid": torch.ones(batch, ticks, dtype=torch.bool, device=device),
        "view_ids": torch.tensor([[0, 2]] * batch, device=device),
    }


def lora_gradient(model, batch, clear_cache: bool) -> float:
    original = torch.clear_autocast_cache
    if not clear_cache:
        torch.clear_autocast_cache = lambda: None
    try:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.rollout(batch, 4, 3)
        output.float().abs().mean().backward()
    finally:
        torch.clear_autocast_cache = original
    grads = [p.grad for n, p in model.trainable_named().items() if n.endswith("lora_b")]
    return float(sum(g.abs().sum() for g in grads if g is not None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor-config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="gpu_acceptance.json")
    parser.add_argument("--microbatch", type=int, default=None, help="sequences per pass in the update test")
    args = parser.parse_args()

    settings = load_settings()
    units = So101Units(gripper_direction_verified=True)
    servo = ServoParams((0.1,) * 6, (3,) * 6)
    report = {"gpu": torch.cuda.get_device_name(0), "microbatch": args.microbatch}

    torch.cuda.reset_peak_memory_stats()
    model = build_model(args.predictor_config, args.weights, units, servo, settings, args.device, checkpointing=True)
    report["parameters_total"] = sum(p.numel() for p in model.parameters())
    report["lora_modules"] = len(model.lora_modules)
    report["memory_loaded_gb"] = torch.cuda.memory_allocated() / 1e9

    batch = fake_batch(2, 10, args.device)
    model.set_trainable("online")
    report["trainable_online"] = sum(p.numel() for p in model.trainable_named().values())
    report["lora_grad_with_guard"] = lora_gradient(model, batch, True)
    report["lora_grad_without_guard"] = lora_gradient(model, batch, False)
    assert report["lora_grad_with_guard"] > 0, "LoRA received no gradient"

    with torch.no_grad():
        cached = model.rollout(batch, 2, 6, use_kv_generation=True).float()
        full = model.rollout(batch, 2, 6, use_kv_generation=False).float()
    report["kv_vs_full_max_error"] = float((cached - full).abs().max())

    builder = OnlineBuilder(model, model.clone_role(), settings.online)
    update_batch = fake_batch(settings.raw["online"]["effective_batch"], 10, args.device)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = builder.update(
            update_batch, [False] * update_batch["features"].shape[0], context=4, microbatch=args.microbatch
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        assert result.status == "ok"
    report["online_update_s_median"] = float(np.median(times[1:]))
    report["online_update_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    report["achievable_updates_per_s_at_10pct_duty"] = 0.1 / report["online_update_s_median"]

    model.set_trainable("none")
    single = fake_batch(1, 3, args.device)
    with torch.no_grad():
        started = time.perf_counter()
        for _ in range(10):
            model.rollout(single, 2, 1)
        torch.cuda.synchronize()
    report["live_prediction_s"] = (time.perf_counter() - started) / 10

    print(json.dumps(report, indent=2))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
