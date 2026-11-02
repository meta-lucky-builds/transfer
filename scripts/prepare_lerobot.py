"""Import one LeRobot dataset: write episodes, encode keyframes, record dataset.json.

Unit choices are required arguments on purpose: run scripts/inspect_lerobot.py first
and read the joint ranges. See robojepa_ttt/data/import_lerobot.py for what imported
data cannot provide.

    python scripts/prepare_lerobot.py --repo-id armnet/armnetbench_v01_lerobot_so101 \
        --use-degrees --gripper-closed 0 --gripper-open 100 --gripper-verified \
        --cameras front=6 top=5 --max-episodes 400 \
        --encoder /workspace/ckpt/vjepa2_1_vitG_384.pt --out /workspace/data/armnet

Frame rate comes from the dataset (20 fps -> 4 ticks per 0.2 s step, 30 fps -> 6).
Cameras: name=semantic_id with RoboJEPA's view_map (left 0, right 1, top 5, front 6).
The wrist id 2 is refused: the target rig has two fixed phone cameras and no wrist
camera, so training must not lean on a wrist view. Unlisted cameras are skipped.

Several GPUs: run one process per GPU on disjoint shards of the same selection.

    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/prepare_lerobot.py ... \
        --shard $i --num-shards 4 > prep_$i.log 2>&1 & done; wait
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import _common  # noqa: F401  sets the import path
from robojepa_ttt.data.dataset_meta import DatasetMeta, parse_cameras, timing_for
from robojepa_ttt.data.feature_store import DiskFeatures, encode_frames, encoder_hash
from robojepa_ttt.data.import_lerobot import check_dataset_info, episode_from_arrays
from robojepa_ttt.data.offline_dataset import save_episode
from robojepa_ttt.models.upstream import add_upstream_to_path
from robojepa_ttt.robot_io.so101_units import So101Units


def open_dataset(repo_id: str):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id)


def select_episodes(episode_index: np.ndarray, task_index: np.ndarray | None, limit: int | None, seed: int) -> list[int]:
    """Deterministic, task-balanced choice of source episodes (identical on every shard).

    Round-robin over tasks in shuffled order, so each task contributes as evenly as its
    episode count allows. Returned sorted, which keeps neighbouring episodes together
    for session-like splitting.
    """
    episodes = sorted(int(e) for e in np.unique(episode_index))
    if limit is None or limit >= len(episodes):
        return episodes
    rng = random.Random(seed)
    by_task: dict[int, list[int]] = defaultdict(list)
    if task_index is None:
        by_task[0] = episodes
    else:
        first_row = {}
        for row, e in enumerate(episode_index):
            first_row.setdefault(int(e), row)
        for e in episodes:
            by_task[int(task_index[first_row[e]])].append(e)
    for members in by_task.values():
        rng.shuffle(members)
    chosen, tasks = [], sorted(by_task)
    while len(chosen) < limit:
        progressed = False
        for t in tasks:
            if by_task[t] and len(chosen) < limit:
                chosen.append(by_task[t].pop())
                progressed = True
        if not progressed:
            break
    return sorted(chosen)


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
    parser.add_argument("--max-episodes", type=int, default=None, help="task-balanced subset")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
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

    dataset = open_dataset(args.repo_id)
    info_path = Path(dataset.root) / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info["fps"])
    if args.expect_fps is not None and fps != args.expect_fps:
        raise ValueError(f"dataset has {fps} fps, expected {args.expect_fps}")
    timing = timing_for(fps)
    check_dataset_info(info_path, expected_fps=fps)
    for name in cameras:
        if f"observation.images.{name}" not in info["features"]:
            raise ValueError(f"camera '{name}' is not in this dataset: {[k for k in info['features'] if k.startswith('observation.images.')]}")

    add_upstream_to_path()
    from app.robo_jepa.utils import build_encoder

    encoder = build_encoder(version="v21", device=args.device, filename=args.encoder, use_proj=False, max_num_frames=64)
    hash_id = encoder_hash(args.encoder)

    out = Path(args.out)
    store = DiskFeatures(out / "features", timing.stride)
    table = dataset.hf_dataset.with_format("numpy")
    episode_index = np.asarray(table["episode_index"])
    task_index = np.asarray(table["task_index"]) if "task_index" in table.column_names else None
    selected = select_episodes(episode_index, task_index, args.max_episodes, args.seed)
    mine = selected[args.shard :: args.num_shards]
    print(f"{fps:g} fps -> stride {timing.stride}; {len(selected)} episodes selected, shard {args.shard} has {len(mine)}", flush=True)

    meta = DatasetMeta(args.repo_id, fps, cameras, True, args.gripper_closed, args.gripper_open, hash_id)
    meta.notes.append("wrist view excluded" if 2 not in cameras.values() else "wrist view included")
    if args.shard == 0:
        meta.save(out)  # identical on every shard; written once
    manifest = []
    for value in mine:
        rows = np.flatnonzero(episode_index == value)
        part = table.select(rows.tolist())
        state = np.stack(part["observation.state"])
        action = np.stack(part["action"])
        timestamps = np.asarray(part["timestamp"], dtype=float).ravel()
        episode_id = f"{args.repo_id}/{int(value)}"
        episode = episode_from_arrays(episode_id, args.repo_id, units, state, action, timestamps, hash_id, len(cameras))

        keyframes = rows[:: timing.stride]
        images = {name: [] for name in cameras}
        for i in keyframes:
            item = dataset[int(i)]  # decodes every camera of this frame once
            for name in cameras:
                images[name].append(item[f"observation.images.{name}"])
        views = []
        for name in cameras:
            frames = (torch.stack(images[name]).permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8)
            views.append(encode_frames(encoder, frames, args.device))
        store.save(episode_id, torch.stack(views, dim=1))
        save_episode(episode, out / "episodes", list(cameras.values()))
        task = int(task_index[rows[0]]) if task_index is not None else None
        manifest.append({"episode_id": episode_id, "source_index": int(value), "task_index": task, "ticks": len(rows)})
        print(f"{episode_id}: {len(rows)} ticks, {len(keyframes)} keyframes", flush=True)
    (out / f"manifest-{args.shard}-of-{args.num_shards}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
