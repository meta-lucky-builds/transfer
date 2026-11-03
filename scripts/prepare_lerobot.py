"""Import one LeRobot v3 dataset: write episodes, encode keyframes, record dataset.json.

Uses robojepa_ttt/data/lerobot_v3.py, not the lerobot package (which pins torch < 2.12
and so cannot share an NGC container's torch 2.14). Only the data and video files that
hold the selected episodes are downloaded. Run scripts/inspect_lerobot.py first and read
the joint ranges: unit choices are required arguments on purpose.

    HF_HOME=/workspace/hf_cache python scripts/prepare_lerobot.py \
        --repo-id armnet/armnetbench_v01_lerobot_so101 \
        --use-degrees --gripper-closed 0 --gripper-open 100 --gripper-verified \
        --cameras front=6 top=5 --episodes-per-task 50 \
        --encoder /workspace/ckpt/vjepa2_1_vitG_384.pt --out /workspace/data/armnet

Frame rate comes from the dataset (20 fps -> 4 ticks per 0.2 s step). Cameras map to
RoboJEPA's view_map (left 0, right 1, top 5, front 6); the wrist id 2 is refused, because
the target rig has two fixed phone cameras and no wrist camera. Unlisted cameras are
never downloaded.

Several GPUs: one process per GPU on disjoint shards of the same deterministic selection.
Episodes are sharded by data file so shards rarely share downloads.

    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/prepare_lerobot.py ... \
        --shard $i --num-shards 4 > prep_$i.log 2>&1 & done; wait
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import _common  # noqa: F401  sets the import path
from robojepa_ttt.data.dataset_meta import DatasetMeta, parse_cameras, timing_for
from robojepa_ttt.data.feature_store import DiskFeatures, encode_frames, encoder_hash
from robojepa_ttt.data.import_lerobot import check_dataset_info, episode_from_arrays
from robojepa_ttt.data.lerobot_v3 import LeRobotV3Source, select_blocks
from robojepa_ttt.data.offline_dataset import save_episode
from robojepa_ttt.robot_io.so101_units import So101Units


def shard_of(source: LeRobotV3Source, selected: list[int], num_shards: int) -> dict[int, int]:
    """Assign episodes to shards by data file, balancing episode counts greedily."""
    groups: dict[str, list[int]] = {}
    for episode in selected:
        groups.setdefault(source.data_file(episode), []).append(episode)
    load = [0] * num_shards
    assignment: dict[int, int] = {}
    for _, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        target = min(range(num_shards), key=lambda s: (load[s], s))
        for episode in members:
            assignment[episode] = target
        load[target] += len(members)
    return assignment


def prepare(
    source: LeRobotV3Source,
    episodes: list[int],
    cameras: dict[str, int],
    units: So101Units,
    encode: Callable[[np.ndarray], torch.Tensor],
    hash_id: str,
    out: Path,
    shard: int = 0,
    num_shards: int = 1,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Write episodes and keyframe features for `episodes`; return the manifest rows.

    encode maps uint8 frames [K, H, W, 3] to features [K, rows, cols, C].
    """
    timing = timing_for(source.fps)
    store = DiskFeatures(out / "features", timing.stride)
    manifest = []
    started = time.time()
    for done, value in enumerate(episodes, 1):
        rows = source.rows(value)
        episode_id = f"{source.repo_id}/{value}"
        episode = episode_from_arrays(
            episode_id, source.repo_id, units, rows.state, rows.action, rows.timestamp, hash_id, len(cameras)
        )
        keyframes = np.arange(0, len(rows.timestamp), timing.stride)
        views = [encode(source.frames(value, name, rows.timestamp[keyframes])) for name in cameras]
        store.save(episode_id, torch.stack(views, dim=1))
        save_episode(episode, out / "episodes", list(cameras.values()))
        manifest.append({"episode_id": episode_id, "source_index": int(value), "task_index": rows.task_index, "ticks": len(rows.timestamp)})
        rate = done / max(time.time() - started, 1e-9)
        log(f"[shard {shard}/{num_shards}] {done}/{len(episodes)} {episode_id}: {len(rows.timestamp)} ticks, {len(keyframes)} keyframes, {rate * 60:.1f} episodes/min")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--use-degrees", action="store_true", required=True, help="assert body joints are degrees")
    parser.add_argument("--gripper-closed", type=float, required=True)
    parser.add_argument("--gripper-open", type=float, required=True)
    parser.add_argument("--gripper-verified", action="store_true", required=True)
    parser.add_argument("--cameras", nargs="+", required=True, help="name=semantic_id; order sets the view order")
    parser.add_argument("--allow-wrist", action="store_true", help="permit the wrist view id 2")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect-fps", type=float, default=None, help="fail unless the dataset has this rate")
    parser.add_argument("--episodes-per-task", type=int, default=None, help="task-balanced subset; default all")
    parser.add_argument("--block", type=int, default=10, help="consecutive episodes per selected block")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=32, help="frames per encoder call")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("--shard must be in [0, num_shards)")

    cameras = parse_cameras(args.cameras, args.allow_wrist)
    units = So101Units(
        use_degrees=True,
        gripper_device_closed=args.gripper_closed,
        gripper_device_open=args.gripper_open,
        gripper_direction_verified=True,
    )
    source = LeRobotV3Source(args.repo_id)
    if args.expect_fps is not None and source.fps != args.expect_fps:
        raise ValueError(f"dataset has {source.fps} fps, expected {args.expect_fps}")
    timing = timing_for(source.fps)
    check_dataset_info(source.meta_root / "meta" / "info.json", expected_fps=source.fps)
    available = source.camera_keys()
    for name in cameras:
        if name not in available:
            raise ValueError(f"camera '{name}' is not in this dataset: {available}")

    if args.episodes_per_task is None:
        selected = sorted(int(e) for e in source.episodes.index)
    else:
        selected = select_blocks(source, args.episodes_per_task, args.block, args.seed)
    assignment = shard_of(source, selected, args.num_shards)
    mine = [e for e in selected if assignment[e] == args.shard]
    print(f"{source.fps:g} fps -> stride {timing.stride}; {len(selected)} episodes selected; shard {args.shard} has {len(mine)}", flush=True)

    from robojepa_ttt.models.upstream import add_upstream_to_path

    add_upstream_to_path()
    from app.robo_jepa.utils import build_encoder

    encoder = build_encoder(version="v21", device=args.device, filename=args.encoder, use_proj=False, max_num_frames=64)
    hash_id = encoder_hash(args.encoder)

    def encode(frames: np.ndarray) -> torch.Tensor:
        return encode_frames(encoder, torch.from_numpy(frames), args.device, batch=args.batch)

    out = Path(args.out)
    meta = DatasetMeta(args.repo_id, source.fps, cameras, True, args.gripper_closed, args.gripper_open, hash_id)
    meta.notes.append("wrist view excluded" if 2 not in cameras.values() else "wrist view included")
    meta.notes.append("read with robojepa_ttt.data.lerobot_v3, not the lerobot package")
    if args.shard == 0:
        meta.save(out)
    manifest = prepare(source, mine, cameras, units, encode, hash_id, out, args.shard, args.num_shards, lambda m: print(m, flush=True))
    (out / f"manifest-{args.shard}-of-{args.num_shards}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"shard {args.shard} done: {len(manifest)} episodes", flush=True)


if __name__ == "__main__":
    main()
