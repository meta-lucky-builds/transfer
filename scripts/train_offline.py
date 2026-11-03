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
import queue
import threading
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


class Prefetcher:
    """Builds the next training batches in a background thread while the GPU computes.

    One producer thread calls the sampler in order, so batches and their order are
    exactly those of calling it directly; only disk reads and collation overlap with
    compute. CUDA batches are pinned in the thread so the copy is asynchronous.
    """

    def __init__(self, produce, device, depth: int = 2):
        self.device = torch.device(device)
        self.items: queue.Queue = queue.Queue(maxsize=depth)
        pin = self.device.type == "cuda"

        def work():
            while True:
                try:
                    batch = produce()
                    if pin:
                        batch = {k: v.pin_memory() for k, v in batch.items()}
                    self.items.put(batch)
                except BaseException as error:  # surface it in the training thread
                    self.items.put(error)
                    return

        threading.Thread(target=work, daemon=True).start()

    def __call__(self) -> dict:
        item = self.items.get()
        if isinstance(item, BaseException):
            raise item
        return {k: v.to(self.device, non_blocking=True) for k, v in item.items()}


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
    parser.add_argument("--max-updates", type=int, default=None, help="override; with early stopping this is a ceiling")
    parser.add_argument("--warmup-updates", type=int, default=None)
    parser.add_argument("--curriculum", default=None, help="updates:horizon,... e.g. 250:2,400:5,200:10")
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None, help="evaluations without improvement before stopping")
    parser.add_argument("--validation-windows", type=int, default=64, help="fixed validation windows (kept on CPU)")
    parser.add_argument("--no-checkpointing", action="store_true",
                        help="keep activations instead of recomputing them: ~25-30%% faster, more memory")
    args = parser.parse_args()

    dist = init_distributed()
    device = f"cuda:{dist.local_rank}" if dist.enabled and torch.cuda.is_available() else args.device
    settings = load_settings(args.settings) if args.settings else load_settings()
    config = settings.offline
    if args.microbatch:
        config = replace(config, microbatch=args.microbatch)
    overrides = {}
    if args.max_updates:
        overrides["max_updates"] = args.max_updates
    if args.warmup_updates is not None:
        overrides["warmup_updates"] = args.warmup_updates
    if args.curriculum:
        overrides["curriculum"] = tuple(tuple(int(v) for v in part.split(":")) for part in args.curriculum.split(","))
    if args.validation_interval:
        overrides["validation_interval"] = args.validation_interval
    if args.patience:
        overrides["patience_evaluations"] = args.patience
    config = replace(config, **overrides)
    if max(h for _, h in config.curriculum) + 1 > config.sequence_frames:
        raise ValueError("the longest curriculum horizon needs a longer sequence")
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
        "cpu",  # assembled on the CPU by the prefetch thread, then copied asynchronously
        config.seed + 1000 * dist.rank,  # each rank draws its own windows
    )
    # Every rank builds the same fixed set and scores its own share (rank::world_size).
    validation = KeyframeSampler(
        validation_episodes,
        validation_views,
        spec,
        feature_cache(args.validation_features or args.features, spec.stride, budget // 4),
        config.microbatch,
        "cpu",  # kept off the GPU until used: at 2B one window is ~64 MB of features
        config.seed,
    )
    fixed_validation = list(validation.ordered_batches(args.validation_windows))[dist.rank :: dist.world_size]
    if dist.is_main:
        print(
            f"ranks {dist.world_size}, windows: train {train.total}, validation {validation.total}, "
            f"fixed validation batches per rank ~{len(fixed_validation)}"
        )

    model = build_model(
        args.predictor_config, args.weights, load_units(args.units), load_servo(args.servo), settings, device, checkpointing=not args.no_checkpointing,
        timing=timing,
    )
    trainer = OfflineTrainer(model, config, args.out, dist)
    if trainer.resume() and dist.is_main:
        print(f"resumed at update {trainer.update}")

    def validate(current):
        """(sum of per-window errors, window count) over this rank's share."""
        total, count = 0.0, 0
        for batch in fixed_validation:
            on_device = {k: v.to(device) for k, v in batch.items()}
            errors = rollout_errors(current, on_device, 2, 5).mean(axis=1)  # per window
            total += float(errors.sum())
            count += len(errors)
        return total, count

    if dist.is_main:
        print(json.dumps({"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(config).items()}}), flush=True)

    def log(record):
        if record["update"] % 20 == 0 or "validation" in record:
            print(json.dumps(record), flush=True)

    result = trainer.fit(Prefetcher(train.next_batch, device), validate, log, validate_all_ranks=True)
    if torch.cuda.is_available() and dist.is_main:
        print(json.dumps({"peak_gpu_gb_rank0": torch.cuda.max_memory_allocated() / 1e9}), flush=True)
    if dist.is_main:
        print(result)
        Path(args.out, "log.json").write_text(json.dumps(trainer.log), encoding="utf-8")
    cleanup(dist)


if __name__ == "__main__":
    main()
