"""Write units.json for a prepared dataset: device conventions plus compact normalisation.

    python scripts/make_units.py --episodes /workspace/data/armnet/splits/train/episodes \
        --out /workspace/data/armnet/units.json

Device conventions (degrees, gripper closed/open values) come from dataset.json, which
prepare_lerobot.py wrote from the arguments it was given. Compact coordinates (spec 3.1)
are (physical - center) / scale with the arm's allowed-range midpoint and half-range,
and 0.5 / 0.5 for the gripper. For public data the allowed range is not known, so this
uses the [low, high] percentiles of TRAINING positions with a margin. On your own arm,
replace them with the measured joint limits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import load_episodes
from robojepa_ttt.data.dataset_meta import load_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, help="TRAINING episodes only")
    parser.add_argument("--out", required=True)
    parser.add_argument("--low", type=float, default=0.5, help="percentile")
    parser.add_argument("--high", type=float, default=99.5, help="percentile")
    parser.add_argument("--margin", type=float, default=0.1, help="fraction of the range added on both sides")
    parser.add_argument("--min-scale", type=float, default=0.05, help="radians, floor on half-range")
    args = parser.parse_args()

    meta = load_meta(args.episodes)
    if meta is None:
        raise ValueError("no dataset.json next to the episodes")
    episodes, _ = load_episodes(args.episodes)
    positions = np.concatenate([e.position[: e.length][e.position_valid[: e.length]] for e in episodes])
    low = np.percentile(positions[:, :5], args.low, axis=0)
    high = np.percentile(positions[:, :5], args.high, axis=0)
    span = high - low
    low, high = low - args.margin * span, high + args.margin * span
    center = (low + high) / 2
    scale = np.maximum((high - low) / 2, args.min_scale)
    units = {
        "use_degrees": bool(meta.use_degrees),
        "gripper_device_closed": meta.gripper_closed,
        "gripper_device_open": meta.gripper_open,
        "gripper_direction_verified": True,
        "center": [float(v) for v in center] + [0.5],
        "scale": [float(v) for v in scale] + [0.5],
    }
    Path(args.out).write_text(json.dumps(units, indent=2), encoding="utf-8")
    compact = (positions[:, :5] - center) / scale
    print(json.dumps(units, indent=2))
    print(f"{len(positions)} training positions; compact body range {compact.min(0).round(2).tolist()} .. {compact.max(0).round(2).tolist()}")


if __name__ == "__main__":
    main()
