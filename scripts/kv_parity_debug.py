"""Locate a KV-cache versus full-forward mismatch on the GPU.

gpu_acceptance.py reports one max error over a whole 6-frame rollout. This splits it:

* per generated step, so compounding shows up as error growing with the step,
* fp32 versus bf16, so precision effects separate from logic differences,
* full forward repeated twice, the GPU's own run-to-run noise floor,
* the planner's cached single step against an uncached forward on the same input,
  which has no feedback and so cannot compound,
* one bf16 forward against an fp32 forward on identical input: the precision noise
  of a single pass, the yardstick for every other number.

Upstream sdpa allows only the Flash and memory-efficient kernels. The cached step has
16 query heads against 8 key/value heads and no mask, which only Flash supports, and
Flash needs bf16. For an fp32 reference this script also allows PyTorch's math
kernel, by patching torch.nn.attention.sdpa_kernel inside this process only. The
upstream code and the training path are untouched.

    python scripts/kv_parity_debug.py --predictor-config sources/wm_2b_config.json \
        --weights /workspace/ckpt/cooldown_2b_450k.pth.tar

Reading the result: if fp32 step-1 errors are tiny and errors grow with the step in
bf16 only, it is precision compounding through feedback on random inputs. If fp32
step 1, or the single cached step, is already large, the cached path computes
something different and must be fixed before KV generation is used.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import traceback

import torch
from torch.nn.attention import SDPBackend

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


_original_sdpa_kernel = torch.nn.attention.sdpa_kernel


@contextlib.contextmanager
def math_allowed():
    """Add the math kernel to every backend list upstream requests."""
    def patched(backends, *args, **kwargs):
        backends = list(backends) if isinstance(backends, (list, tuple)) else [backends]
        if SDPBackend.MATH not in backends:
            backends.append(SDPBackend.MATH)
        return _original_sdpa_kernel(backends, *args, **kwargs)

    torch.nn.attention.sdpa_kernel = patched
    try:
        yield
    finally:
        torch.nn.attention.sdpa_kernel = _original_sdpa_kernel


def guarded(function, *args):
    try:
        return function(*args)
    except Exception as error:  # report and continue: one failure must not hide the rest
        return {"error": f"{type(error).__name__}: {error}", "trace": traceback.format_exc(limit=3)}


@torch.no_grad()
def precision_single_forward(model, batch, context):
    """Same input, one forward, bf16 versus fp32 (math kernel allowed)."""
    actions, states = model.sequence_inputs(batch, context, 1)
    command = model.encode_commands(batch, context)
    args = (batch["features"][:, :context], actions, states, batch["view_ids"], command)
    model.amp_dtype = torch.bfloat16
    with model.amp():
        low = model.forward_tokens(*args)
    model.amp_dtype = None
    with math_allowed():
        high = model.forward_tokens(*args)
    return stats(low, high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor-config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="kv_parity_debug.json")
    parser.add_argument("--strict-fp32", action="store_true",
                        help="disable TF32, which NGC containers enable by default; run with NVIDIA_TF32_OVERRIDE=0")
    args = parser.parse_args()
    if args.strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    torch.manual_seed(0)
    model = build_model(
        args.predictor_config, args.weights, So101Units(gripper_direction_verified=True), ServoParams((0.1,) * 6, (3,) * 6),
        load_settings(), args.device, checkpointing=False,
    )
    model.set_trainable("none")
    batch = fake_batch(2, 10, args.device)
    import os

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "env_NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "env_TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
    }
    print("precision settings:", json.dumps(report), flush=True)
    report["bf16_vs_fp32_single_forward"] = guarded(precision_single_forward, model, batch, 4)
    print("bf16 vs fp32, one forward:", json.dumps(report["bf16_vs_fp32_single_forward"])[:300], flush=True)
    for label, dtype, context_manager in (
        ("bf16", torch.bfloat16, contextlib.nullcontext),
        ("fp32_math_allowed", None, math_allowed),
    ):
        model.amp_dtype = dtype
        with context_manager():
            report[label] = {
                "planner_single_cached_step": guarded(single_cached_step, model, batch, 4),
                "context2_horizon6": guarded(generation_errors, model, batch, 2, 6),
                "context4_horizon3": guarded(generation_errors, model, batch, 4, 3),
            }
        print(label, "single cached step:", json.dumps(report[label]["planner_single_cached_step"])[:300], flush=True)
    print(json.dumps(report, indent=2))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
