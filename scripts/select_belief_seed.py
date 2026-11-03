"""Pick the belief seed with the lowest VALIDATION NLL and summarise the belief results.

    python scripts/select_belief_seed.py --run /workspace/runs/armnet --seeds 7
    python scripts/select_belief_seed.py --run /workspace/runs/armnet --summary

Selection uses validation data only; the test split never chooses a model. The spread
across seeds is reported because one seed's score is not a measure of the method
(spec 19.4).
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)

    if args.seeds:
        scores = {}
        for k in range(1, args.seeds + 1):
            report = run / f"belief_seed{k}" / "report.json"
            if not report.exists():
                raise FileNotFoundError(f"seed {k} wrote no report: {report}")
            scores[k] = json.loads(report.read_text())["validation_nll"]
        best = min(scores, key=scores.get)
        if (run / "belief").exists():
            shutil.rmtree(run / "belief")
        shutil.copytree(run / f"belief_seed{best}", run / "belief")
        spread = statistics.pstdev(scores.values()) if len(scores) > 1 else 0.0
        summary = {"validation_nll_by_seed": scores, "mean": statistics.mean(scores.values()), "spread": spread, "selected_seed": best}
        (run / "belief_seeds.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))

    if args.summary:
        report = json.loads((run / "belief" / "report.json").read_text())
        print("belief validation NLL per coordinate:", round(report["validation_nll"], 4))
        seeds = run / "belief_seeds.json"
        if seeds.exists():
            data = json.loads(seeds.read_text())
            print(f"  across seeds: mean {data['mean']:.4f}, spread {data['spread']:.4f}, selected seed {data['selected_seed']} by validation")
        for name, block in report.get("innovations_after_calibration", report.get("innovations", {})).items():
            print(f"  {name}: NIS per coordinate {block['nis_per_coordinate']:.3f} (1 is calibrated)")
        if "calibration" in report:
            print("  calibration multipliers:", [round(m, 3) for m in report["calibration"]["multipliers"]])


if __name__ == "__main__":
    main()
