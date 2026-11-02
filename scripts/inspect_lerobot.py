"""Inspect a LeRobot dataset on the Hub before preparing it. Downloads metadata and the
non-video data files only (joint states and actions, no videos).

    python scripts/inspect_lerobot.py --repo-id armnet/armnetbench_v01_lerobot_so101

Prints frame rate, cameras and resolution, joint names, per-joint ranges of state and
action, gripper range, episode lengths and the task mix. It does not decide units:
degrees and LeRobot's normalised RANGE_M100_100 can both lie inside +-100, so read
the ranges, the dataset card and the robot calibration, then pass --use-degrees and
the gripper values to prepare_lerobot.py only when they are confirmed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--max-data-files", type=int, default=4, help="parquet files of joint data to read")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    import pandas as pd

    root = Path(snapshot_download(args.repo_id, repo_type="dataset", allow_patterns=["meta/**", "README.md"]))
    info = json.loads((root / "meta" / "info.json").read_text())
    features = info["features"]
    print(f"repo {args.repo_id}  codebase {info.get('codebase_version')}  robot {info.get('robot_type')}")
    print(f"fps {info['fps']}  episodes {info.get('total_episodes')}  frames {info.get('total_frames')}  tasks {info.get('total_tasks')}")
    fps = float(info["fps"])
    ticks = fps * 0.2
    print(f"0.2 s macro step = {ticks:g} ticks -> {'OK' if abs(ticks - round(ticks)) < 1e-6 else 'NOT a whole number: cannot use'}")
    print("\ncameras:")
    for key, value in features.items():
        if key.startswith("observation.images."):
            video = value.get("info", {}) or value.get("video_info", {})
            print(f"  {key.removeprefix('observation.images.'):<12} shape {value.get('shape')}  {json.dumps(video)[:160]}")
    names = {}
    for key in ("observation.state", "action"):
        raw = features[key].get("names")
        if isinstance(raw, dict):
            raw = next(iter(raw.values()))
        names[key] = raw
        print(f"\n{key} names: {raw}")

    from huggingface_hub import hf_hub_download, list_repo_files

    remote = sorted(f for f in list_repo_files(args.repo_id, repo_type="dataset") if f.startswith("data/") and f.endswith(".parquet"))
    print(f"\n{len(remote)} data files on the Hub; reading the first {min(args.max_data_files, len(remote))}")
    data_files = [Path(hf_hub_download(args.repo_id, f, repo_type="dataset")) for f in remote[: args.max_data_files]]
    if not data_files:
        print("\nno data parquet files found")
        return
    frame = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    print(f"\nread {len(frame)} frames from {len(data_files)} data files")
    for key in ("observation.state", "action"):
        values = np.stack(frame[key].to_numpy())
        print(f"\n{key}: per joint  min / p1 / median / p99 / max")
        for j, name in enumerate(names[key] or range(values.shape[1])):
            v = values[:, j]
            print(f"  {str(name):<20} {v.min():9.2f} {np.percentile(v, 1):9.2f} {np.median(v):9.2f} {np.percentile(v, 99):9.2f} {v.max():9.2f}")
        body = np.abs(values[:, :5])
        if body.max() <= 3.3:
            print("  body joints within +-pi: probably radians, which prepare_lerobot.py does not accept as degrees")
        elif body.max() > 100.5:
            print("  body joints exceed 100: not RANGE_M100_100, consistent with degrees")
        else:
            print("  body joints within +-100: degrees or RANGE_M100_100, CHECK the dataset card and calibration")
    if "episode_index" in frame:
        lengths = frame.groupby("episode_index").size().to_numpy()
        print(f"\nepisode length in frames (sampled episodes): min {lengths.min()} median {int(np.median(lengths))} max {lengths.max()}"
              f"  = median {np.median(lengths) / fps:.1f} s")
        total = info.get("total_frames")
        if total:
            keyframes = total / ticks
            print(f"whole dataset: about {keyframes:,.0f} keyframes, about {keyframes * 2e6 / 1e9:,.0f} GB of features for two views")
    if "task_index" in frame:
        print(f"\ntask mix in sampled frames: {frame['task_index'].value_counts().sort_index().to_dict()}")
    tasks = root / "meta" / "tasks.jsonl"
    if tasks.exists():
        print("\ntasks:")
        for line in tasks.read_text().splitlines()[:20]:
            print(" ", line)


if __name__ == "__main__":
    main()
