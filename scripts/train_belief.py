"""Stage 2: fit the observation sketch, then train the compact recurrent belief.

    python scripts/train_belief.py --units data/X/units.json --servo data/X/servo.json \
        --train data/train/episodes --validation data/val/episodes \
        --features data/all/features --views 6 1 --joint-sensor-std 0.002 \
        --out runs/belief

Optional --calibration data/cal/episodes refits the R multipliers on calibration
episodes after training (spec 7.5). Those episodes never give training gradients.

joint_sensor_std is a measured property of the arm in compact units: measure joint
repeatability while the arm holds still, independent of tracking error (spec 6.5).
Public datasets cannot measure it for your arm; a small value such as 0.002 is a
placeholder there, and the reported calibration multiplier for joints shows how
wrong it was.

Resumes from runs/belief/latest.pt when it exists. Single GPU: the compact model is
small, so its cost is the sequential filter, not the batch.
"""

from __future__ import annotations

import os

# The filter needs true fp32 (spec 7.3). NGC containers turn TF32 on through these
# variables, which cuBLAS reads when CUDA starts, so they are set before torch loads.
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import argparse
import json
import random
from pathlib import Path

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

from _common import dataset_timing, feature_cache, load_episodes, load_servo, load_units
from robojepa_ttt.belief.calibration import fit_calibration, innovation_summary
from robojepa_ttt.belief.compact import CompactBelief
from robojepa_ttt.belief.data import BeliefSequences, SequenceSpec, joint_loss_scale, residual_bound_from_data
from robojepa_ttt.belief.sketch import ObservationSketch
from robojepa_ttt.belief.train import CompactTrainConfig, CompactTrainer
from robojepa_ttt.config import compact_config, load_settings


def sketch_frames(episodes, view_ids, cache, view, stride, limit, seed):
    """Yield chunks of training keyframe tokens for one semantic view, valid frames only."""
    candidates = []
    for e in episodes:
        views = view_ids[e.episode_id]
        if view not in views:
            continue
        index = views.index(view)
        for k in range(0, e.length, stride):
            if e.frame_valid[k, index]:
                candidates.append((e, k, index))
    random.Random(seed).shuffle(candidates)
    candidates = candidates[:limit]
    if not candidates:
        raise ValueError(f"no training frames for view {view}")
    for start in range(0, len(candidates), 256):
        chunk = candidates[start : start + 256]
        yield torch.stack([cache.get(e.encoder_hash, e.episode_id, k)[i] for e, k, i in chunk]).float()


def main():
    parser = argparse.ArgumentParser()
    for name in ("units", "servo", "train", "validation", "features", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--views", type=int, nargs="+", required=True, help="contract view ids, e.g. 6 1")
    parser.add_argument("--joint-sensor-std", type=float, nargs="+", default=None)
    parser.add_argument("--residual-bound", type=float, nargs="+", default=None, help="default: from training motion")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sketch-frames", type=int, default=20_000)
    parser.add_argument("--microbatch", type=int, default=None)
    parser.add_argument("--surrogate-jacobian", action="store_true", help="named ablation of spec 9.3")
    parser.add_argument("--feature-cache-gb", type=float, default=20.0)
    parser.add_argument("--max-updates", type=int, default=None, help="override, e.g. for a smoke run")
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="model init and sampling; default: the YAML seed")
    parser.add_argument("--sketch-path", default=None, help="shared sketch file; fitted and saved here if missing")
    parser.add_argument("--sketch-only", action="store_true", help="fit and save the sketch, then exit")
    args = parser.parse_args()

    settings = load_settings(args.settings) if args.settings else load_settings()
    raw = settings.raw
    compact_raw = raw["offline"]["compact"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    units, servo = load_units(args.units), load_servo(args.servo)
    timing = dataset_timing(settings, *[d for d in (args.train, args.validation, args.calibration) if d])
    stride, tick_s, history = timing.stride, timing.tick_s, timing.history_ticks

    train_eps, train_views = load_episodes(args.train)
    val_eps, val_views = load_episodes(args.validation)
    if {e.episode_id for e in train_eps} & {e.episode_id for e in val_eps}:
        raise ValueError("an episode is in both train and validation")
    views = {**train_views, **val_views}
    missing = sum(1 for e in train_eps if not set(args.views) <= set(views[e.episode_id]))
    print(f"contract views {args.views}; {missing}/{len(train_eps)} training episodes lack one (those slots are masked)")
    cache = feature_cache(args.features, stride, int(args.feature_cache_gb * 1e9))

    seed = raw["seed"] if args.seed is None else args.seed
    torch.manual_seed(seed)  # model initialisation; the sketch fit keeps the YAML seed so seeds share it
    sketch_path = Path(args.sketch_path) if args.sketch_path else out / "sketch.pt"
    first = train_eps[0]
    token_dim = cache.get(first.encoder_hash, first.episode_id, 0).shape[-1]
    sketch = ObservationSketch(args.views, token_dim=token_dim, components=raw["representation"]["sketch"]["pca_dims_per_view"])
    if sketch_path.exists():
        sketch.load_state_dict(torch.load(sketch_path, map_location="cpu"))
        print(f"loaded sketch {sketch.digest()}")
    else:
        report = sketch.fit(
            {v: sketch_frames(train_eps, train_views, cache, v, stride, args.sketch_frames, raw["seed"]) for v in args.views}
        )
        temporary = sketch_path.with_suffix(".tmp")
        torch.save(sketch.state_dict(), temporary)
        temporary.replace(sketch_path)  # atomic: a parallel reader never sees half a file
        print(f"fitted sketch {sketch.digest()} on training frames only: {json.dumps(report)}")
    if args.sketch_only:
        return

    bound = args.residual_bound or residual_bound_from_data(train_eps, units, stride)
    config = compact_config(settings, args.joint_sensor_std, bound, args.views)
    print(f"residual bound per joint {['%.4f' % b for b in config.residual_bound]}")
    model = CompactBelief(config, units, servo, stride, tick_s, history).to(args.device)
    joint_scale = joint_loss_scale(train_eps, units, stride).to(args.device)

    spec = SequenceSpec(compact_raw["sequence_steps"], compact_raw["burn_in_steps"], stride, history, tick_s)
    features = lambda e, k: cache.get(e.encoder_hash, e.episode_id, k)
    train = BeliefSequences(train_eps, views, features, sketch, units, spec)
    validation = BeliefSequences(val_eps, views, features, sketch, units, spec, step_every=spec.scored * stride)
    print(f"sequences: train {len(train)}, validation {len(validation)}")
    fixed_validation = validation.fixed(256, 32, device=args.device)

    train_config = CompactTrainConfig(
        effective_batch=compact_raw["effective_batch"],
        microbatch=args.microbatch or compact_raw["effective_batch"],
        max_updates=args.max_updates or compact_raw["max_updates"],
        warmup_updates=compact_raw["warmup_updates"],
        lr=compact_raw["lr"],
        final_lr=compact_raw["final_lr"],
        betas=tuple(compact_raw["betas"]),
        eps=compact_raw["eps"],
        weight_decay=compact_raw["weight_decay"],
        grad_clip_norm=compact_raw["grad_clip_norm"],
        validation_interval=args.validation_interval or compact_raw["validation_interval_updates"],
        patience_evaluations=compact_raw["early_stop_patience_evaluations"],
        weight_nll=compact_raw["loss_weights"]["predictive_nll"],
        weight_joint=compact_raw["loss_weights"]["joint_huber"],
        weight_regularity=compact_raw["loss_weights"]["regularity"],
        weight_regime_ce=compact_raw["loss_weights"]["supervised_regime_ce_when_available"],
        differentiable_jacobian=not args.surrogate_jacobian,
        seed=seed,
    )
    if args.surrogate_jacobian:
        print("SURROGATE GRADIENT ABLATION: Jacobians detached (spec 9.3). Report it under that name.")
    trainer = CompactTrainer(model, train_config, joint_scale, out)
    if trainer.resume():
        print(f"resumed at update {trainer.update}")
    rng = random.Random(seed)

    def log(record):
        if record["update"] % 20 == 0 or "validation_nll" in record:
            print(json.dumps(record), flush=True)

    print(trainer.fit(lambda: train.sample(train_config.microbatch, rng, args.device), fixed_validation, log))
    best = torch.load(out / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(best["model"])
    report = {"validation_nll": trainer.evaluate(fixed_validation), "innovations": innovation_summary(model, fixed_validation)}
    if args.calibration:
        cal_eps, cal_views = load_episodes(args.calibration)
        cal = BeliefSequences(cal_eps, {**views, **cal_views}, features, sketch, units, spec, step_every=spec.scored * stride)
        cal_batches = cal.fixed(256, 32, device=args.device)
        report["calibration"] = fit_calibration(model, cal_batches, joint_scale)
        report["innovations_after_calibration"] = innovation_summary(model, cal_batches)
    torch.save({"model": model.state_dict(), "config": config, "sketch_digest": sketch.digest(), "joint_scale": joint_scale.cpu(), "seed": seed}, out / "belief.pt")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
