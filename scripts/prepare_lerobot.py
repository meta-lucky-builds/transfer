"""Import one LeRobot dataset: write episodes and encode keyframes.

Unit choices are required arguments on purpose. See robojepa_ttt/data/import_lerobot.py
for what imported data cannot provide.

    python scripts/prepare_lerobot.py --repo-id USER/DATASET --use-degrees \
        --gripper-closed 0 --gripper-open 100 --gripper-verified \
        --cameras front=6 wrist=2 \
        --encoder /path/to/vjepa2_1_vitG_384.pt --out data/USER_DATASET
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import _common  # noqa: F401  sets the import path
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--use-degrees", action="store_true", required=True, help="assert body joints are degrees")
    parser.add_argument("--gripper-closed", type=float, required=True)
    parser.add_argument("--gripper-open", type=float, required=True)
    parser.add_argument("--gripper-verified", action="store_true", required=True)
    parser.add_argument("--cameras", nargs="+", required=True, help="name=semantic_id, order sets the view order")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cameras = [c.split("=") for c in args.cameras]
    view_ids = [int(i) for _, i in cameras]
    units = So101Units(
        use_degrees=True,
        gripper_device_closed=args.gripper_closed,
        gripper_device_open=args.gripper_open,
        gripper_direction_verified=True,
    )

    add_upstream_to_path()
    from app.robo_jepa.utils import build_encoder

    encoder = build_encoder(version="v21", device=args.device, filename=args.encoder, use_proj=False, max_num_frames=64)
    hash_id = encoder_hash(args.encoder)

    dataset = open_dataset(args.repo_id)
    check_dataset_info(Path(dataset.root) / "meta" / "info.json")
    out = Path(args.out)
    store = DiskFeatures(out / "features", args.stride)

    table = dataset.hf_dataset.with_format("numpy")
    episode_index = np.asarray(table["episode_index"])
    for value in np.unique(episode_index):
        rows = np.flatnonzero(episode_index == value)
        state = np.stack([table[int(i)]["observation.state"] for i in rows])
        action = np.stack([table[int(i)]["action"] for i in rows])
        timestamps = np.asarray([table[int(i)]["timestamp"] for i in rows], dtype=float).ravel()
        episode_id = f"{args.repo_id}/{int(value)}"
        episode = episode_from_arrays(episode_id, args.repo_id, units, state, action, timestamps, hash_id, len(cameras))

        keyframes = rows[:: args.stride]
        views = []
        for name, _ in cameras:
            images = [dataset[int(i)][f"observation.images.{name}"] for i in keyframes]
            frames = (torch.stack(images).permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8)
            views.append(encode_frames(encoder, frames, args.device))
        store.save(episode_id, torch.stack(views, dim=1))
        save_episode(episode, out / "episodes", view_ids)
        print(f"{episode_id}: {len(rows)} ticks, {len(keyframes)} keyframes", flush=True)


if __name__ == "__main__":
    main()
