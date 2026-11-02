"""Compare B0 (zero adapters) with adapted weights on held-out episodes (gates G1, G3).

    python scripts/evaluate_b0_b1.py --predictor-config sources/wm_300m_config.json \
        --weights W --units U --servo S --episodes data/test/episodes \
        --features data/test/features --adapters runs/stage3/best.pt

Prediction only. Gate G4 needs task outcomes on the robot.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch

from _common import build_model, dataset_timing, feature_cache, load_episodes, load_servo, load_units
from robojepa_ttt.config import load_settings
from robojepa_ttt.evaluation.prediction import compare_by_episode, rollout_errors, stationary_errors
from robojepa_ttt.learning.replay import collate_windows
from robojepa_ttt.learning.windows import WindowSpec, sparse_starts


def main():
    parser = argparse.ArgumentParser()
    for name in ("predictor-config", "weights", "units", "servo", "episodes", "features", "adapters"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    settings = load_settings()
    timing = dataset_timing(settings, args.episodes)
    spec = WindowSpec(frames=args.context + args.horizon, stride=timing.stride, tick_s=timing.tick_s, history_ticks=timing.history_ticks)
    episodes, view_ids = load_episodes(args.episodes)
    lookup = {e.episode_id: e for e in episodes}
    cache = feature_cache(args.features, spec.stride, int(10e9))
    model = build_model(
        args.predictor_config, args.weights, load_units(args.units), load_servo(args.servo), settings, args.device, timing=timing
    )
    baseline_state = model.adapter_state()
    adapted_state = torch.load(args.adapters, map_location="cpu", weights_only=False)["adapters"]

    def errors_for(state, version):
        model.load_adapter_state(state, version)
        per_episode = defaultdict(list)
        for episode in episodes:
            starts = list(sparse_starts(episode, spec))
            for first in range(0, len(starts), args.batch):
                windows = [(episode.episode_id, s) for s in starts[first : first + args.batch]]
                batch = collate_windows(windows, lookup, cache, spec, view_ids, args.device)
                per_episode[episode.episode_id].append(rollout_errors(model, batch, args.context, args.horizon))
        return {k: np.concatenate(v).mean(0) for k, v in per_episode.items()}

    b0 = errors_for(baseline_state, "b0")
    b1 = errors_for(adapted_state, "b1")
    report = {}
    for label, k in (("h1", 0), ("h2", 1), ("h5", 4), ("h10", 9)):
        if k < args.horizon:
            comparison = compare_by_episode({e: float(v[k]) for e, v in b0.items()}, {e: float(v[k]) for e, v in b1.items()})
            report[label] = asdict(comparison)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
