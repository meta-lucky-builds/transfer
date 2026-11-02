"""Locate a KV-cache versus full-forward mismatch on the GPU.

gpu_acceptance.py reports one max error over a whole 6-frame rollout. This splits it:

* per generated step, so compounding shows up as error growing with the step,
* fp32 versus bf16, so precision effects separate from logic differences,
* full forward repeated twice, the GPU's own run-to-run noise floor,
* the planner's cached single step against an uncached forward on the same input,
  which has no feedback and so cannot compound.

    python scripts/kv_parity_debug.py --predictor-config sources/wm_2b_config.json \
        --weights /workspace/ckpt/cooldown_2b_450k.pth.tar

Reading the result: if fp32 step-1 errors are tiny and errors grow with the step in
bf16 only, it is precision compounding through feedback on random inputs. If fp32
step 1, or the single cached step, is already large, the cached path computes
something different and must be fixed before KV generation is used.
"""

from __future__ import annotations

import argparse
import json

import torch

from _common import build_model
from gpu_acceptance import fake_batch
from robojepa_ttt.config import load_settings
from robojepa_ttt.models.adapted_predictor import repeat_cache
from robojepa_ttt.models.servo import ServoParams
from robojepa_ttt.robot_io.so101_units import So101Units


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a.float() - b.float()).abs()
    scale = b.float().abs().mean().clamp_min(1e-12)
    return {"max": float(diff.max()), "mean": float(diff.mean()), "relative_mean": float(diff.mean() / scale)}


@torch.no_grad()
def generation_errors(model, batch, context, horizon):
    """Compare per-step frames from cached and uncached generation."""
    actions, states = model.sequence_inputs(batch, context, horizon)
    command = model.encode_commands(batch, context + horizon - 1)
    observed = batch["features"][:, :context]
    with model.amp():
        cached = model._generate(observed, actions, states, batch["view_ids"], command, context, horizon, True)
        full = model._generate(observed, actions, states, batch["view_ids"], command, context, horizon, False)
        full_again = model._generate(observed, actions, states, batch["view_ids"], command, context, horizon, False)
    return {
        "cached_vs_full_per_step": [stats(c, f) for c, f in zip(cached, full)],
        "full_vs_full_per_step": [stats(a, b) for a, b in zip(full_again, full)],
    }


@torch.no_grad()
def single_cached_step(model, batch, context):
    """The planner's path: shared prefix cache, then one step, versus one full forward."""
    actions, states = model.sequence_inputs(batch, context, 1)
    command = model.encode_commands(batch, context)
    prefix = model.prime_context(batch["features"][:1, : context - 1], actions[:1, : context - 1], states[:1, : context - 1], batch["view_ids"][:1])
    prefix.cache = repeat_cache(prefix.cache, 1)
    cached, _ = model.step_cached(
        batch["features"][:1, context - 1 : context], actions[:1, context - 1 : context], states[:1, context - 1 : context],
        batch["view_ids"][:1], command[:1, context - 1 : context], prefix, context - 1,
    )
    with model.amp():
        full = model.forward_tokens(batch["features"][:1, :context], actions[:1], states[:1], batch["view_ids"][:1], command[:1])
    return stats(cached, full[:, -1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor-config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="kv_parity_debug.json")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = build_model(
        args.predictor_config, args.weights, So101Units(gripper_direction_verified=True), ServoParams((0.1,) * 6, (3,) * 6),
        load_settings(), args.device, checkpointing=False,
    )
    model.set_trainable("none")
    batch = fake_batch(2, 10, args.device)
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    for label, dtype in (("fp32", None), ("bf16", torch.bfloat16)):
        model.amp_dtype = dtype
        report[label] = {
            "planner_single_cached_step": single_cached_step(model, batch, 4),
            "context2_horizon6": generation_errors(model, batch, 2, 6),
            "context4_horizon3": generation_errors(model, batch, 4, 3),
        }
        print(label, json.dumps(report[label]["planner_single_cached_step"]), flush=True)
    print(json.dumps(report, indent=2))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
