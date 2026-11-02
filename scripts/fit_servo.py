"""Fit nominal servo parameters from prepared episodes and write them to JSON.

    python scripts/fit_servo.py --episodes data/X/episodes --out data/X/servo.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from _common import load_episodes
from robojepa_ttt.data.dataset_meta import load_meta
from robojepa_ttt.models.servo import fit_servo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tick", type=float, default=None, help="default: 1 / fps from dataset.json")
    parser.add_argument("--max-delay", type=float, default=0.3)
    args = parser.parse_args()

    if args.tick is None:
        meta = load_meta(args.episodes)
        if meta is None:
            raise ValueError("no dataset.json next to the episodes: pass --tick explicitly")
        args.tick = meta.timing.tick_s
    print(f"tick {args.tick:.5f} s")
    episodes, _ = load_episodes(args.episodes)
    positions, commands = [], []
    for episode in episodes:
        usable = episode.position_valid[: episode.length] & episode.command_valid[: episode.length]
        if usable.all():
            positions.append(episode.position[: episode.length].copy())
            commands.append(episode.issued[: episode.length].copy())
    params = fit_servo(positions, commands, args.tick, args.max_delay)
    Path(args.out).write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
    print(params)
    print("Fit RMSE is open-loop on the fitting data. Check it on held-out episodes too.")
    motion = np.mean([np.abs(np.diff(p, axis=0)).mean() for p in positions])
    print(f"mean per-tick motion in the data: {motion:.5f}")


if __name__ == "__main__":
    main()
