"""Stage 3 offline adapter training, and the offline reference for online anchors.

    python scripts/train_offline.py --predictor-config sources/wm_300m_config.json \
        --weights /path/to/world_model.pth.tar --units data/X/units.json \
        --servo data/X/servo.json --train data/X/episodes --validation data/Y/episodes \
        --features data/X/features --out runs/stage3

Resumes from runs/stage3/latest.pt when it exists.

Multi-GPU, data parallel: launch the same command with torchrun. The GPU count must
divide effective_batch / microbatch (16 sequences by default, so 1, 2, 4, 8 or 16).

    torchrun --nproc_per_node 8 scripts/train_offline.py ...same arguments...

Several machines: add --nnodes, --node_rank and --rdzv_endpoint as torchrun documents.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from _common import build_model, dataset_timing, feature_cache, load_episodes, load_servo, load_units
from robojepa_ttt.config import load_settings
from robojepa_ttt.data.offline_dataset import KeyframeSampler
from robojepa_ttt.evaluation.prediction import rollout_errors
from robojepa_ttt.learning.distributed import cleanup, init_distributed
from robojepa_ttt.learning.offline_train import OfflineTrainer
from robojepa_ttt.learning.windows import WindowSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor-config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--servo", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--validation-features", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch", type=int, default=None)
    parser.add_argument("--feature-cache-gb", type=float, default=20.0)
    args = parser.parse_args()

    dist = init_distributed()
    device = f"cuda:{dist.local_rank}" if dist.enabled and torch.cuda.is_available() else args.device
    settings = load_settings(args.settings) if args.settings else load_settings()
    config = settings.offline
    if args.microbatch:
        config = replace(config, microbatch=args.microbatch)
    timing = dataset_timing(settings, args.train, args.validation)
    spec = WindowSpec(
        frames=config.sequence_frames,
        stride=timing.stride,
        tick_s=timing.tick_s,
        history_ticks=timing.history_ticks,
    )
    budget = int(args.feature_cache_gb * 1e9)
    train_episodes, train_views = load_episodes(args.train)
    validation_episodes, validation_views = load_episodes(args.validation)
    overlap = {e.episode_id for e in train_episodes} & {e.episode_id for e in validation_episodes}
    if overlap:
        raise ValueError(f"episodes in both splits: {sorted(overlap)[:5]}")

    train = KeyframeSampler(
        train_episodes,
        train_views,
        spec,
        feature_cache(args.features, spec.stride, budget),
        config.microbatch,
        device,
        config.seed + 1000 * dist.rank,  # each rank draws its own windows
    )
    fixed_validation = []
    if dist.is_main:  # only rank 0 validates
        validation = KeyframeSampler(
            validation_episodes,
            validation_views,
            spec,
            feature_cache(args.validation_features or args.features, spec.stride, budget // 4),
            config.microbatch,
            device,
            config.seed,
        )
        fixed_validation = list(validation.ordered_batches(256))
        print(
            f"ranks {dist.world_size}, windows: train {train.total}, validation {validation.total}, "
            f"fixed validation batches {len(fixed_validation)}"
        )

    model = build_model(
        args.predictor_config, args.weights, load_units(args.units), load_servo(args.servo), settings, device, checkpointing=True,
        timing=timing,
    )
    trainer = OfflineTrainer(model, config, args.out, dist)
    if trainer.resume() and dist.is_main:
        print(f"resumed at update {trainer.update}")

    def validate(current) -> float:
        errors = [rollout_errors(current, batch, 2, 5).mean() for batch in fixed_validation]
        return float(np.mean(errors))

    def log(record):
        if record["update"] % 20 == 0 or "validation" in record:
            print(json.dumps(record), flush=True)

    result = trainer.fit(train.next_batch, validate, log)
    if dist.is_main:
        print(result)
        Path(args.out, "log.json").write_text(json.dumps(trainer.log), encoding="utf-8")
    cleanup(dist)


if __name__ == "__main__":
    main()
