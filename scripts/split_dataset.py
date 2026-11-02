"""Split a prepared dataset by pseudo-session into train / validation / calibration / test.

    python scripts/split_dataset.py --root /workspace/data/armnet --group-size 10

Public LeRobot datasets do not record sessions, and neighbouring episodes are often
near-duplicates (same scene, same object placement). Consecutive source episodes are
therefore grouped into pseudo-sessions of --group-size, and whole groups go to one
split (spec 10.2). Split folders hold symbolic links to the episode files plus a copy
of dataset.json; features stay in <root>/features.

    <root>/splits/train/episodes      --train
    <root>/splits/validation/episodes --validation
    <root>/splits/calibration/episodes  belief calibration only, never training
    <root>/splits/test/episodes       final evaluation, untouched until then
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import _common  # noqa: F401
from robojepa_ttt.data.dataset_meta import META_NAME
from robojepa_ttt.data.import_lerobot import split_episodes
from robojepa_ttt.data.offline_dataset import episode_filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    root = Path(args.root)

    entries = []
    for manifest in sorted(root.glob("manifest-*-of-*.json")):
        entries += json.loads(manifest.read_text())
    if not entries:
        raise FileNotFoundError(f"no manifest-*.json in {root}; run prepare_lerobot.py first")
    entries.sort(key=lambda e: e["source_index"])
    missing = [e["episode_id"] for e in entries if not (root / "episodes" / episode_filename(e["episode_id"])).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest episodes have no file, e.g. {missing[:3]}")

    groups: dict[str, list[str]] = {}
    for position, entry in enumerate(entries):
        groups.setdefault(f"group{position // args.group_size:05d}", []).append(entry["episode_id"])
    split = split_episodes(groups, seed=args.seed)

    tasks = {e["episode_id"]: e.get("task_index") for e in entries}
    out = root / "splits"
    if out.exists():
        shutil.rmtree(out)
    for name, members in split.items():
        folder = out / name / "episodes"
        folder.mkdir(parents=True)
        for episode_id in members:
            filename = episode_filename(episode_id)
            (folder / filename).symlink_to((root / "episodes" / filename).resolve())
        shutil.copy(root / META_NAME, out / name / META_NAME)
        mix = dict(sorted(Counter(tasks[e] for e in members).items(), key=lambda kv: (kv[0] is None, kv[0])))
        print(f"{name:<12} {len(members):5d} episodes in {len(members) and -(-len(members) // args.group_size)} groups  tasks {mix}")
    print(f"features stay in {root / 'features'}")


if __name__ == "__main__":
    main()
