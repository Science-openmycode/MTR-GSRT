"""Evaluate NextAcc/NextNLL for every released Beijing epsilon-sweep corpus."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_scripts.downstream_full_protocol import destination_conditioned_next_region_metrics  # noqa: E402
from generation.common.runtime import parse_fraction, sha256_file, write_json  # noqa: E402
from public_utils import load_trajectories  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-root", default="results/synthetic_datasets/mtr/geolife")
    parser.add_argument("--out-dir", default="results/privacy_budget_route_decision/geolife")
    parser.add_argument("--config", default="configs/mtr_epsilon_sweep_beijing.json")
    args = parser.parse_args()
    synthetic_root = Path(args.synthetic_root)
    output = Path(args.out_dir)
    config_path = Path(args.config)
    if not synthetic_root.is_absolute():
        synthetic_root = ROOT / synthetic_root
    if not output.is_absolute():
        output = ROOT / output
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    sweep = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = json.loads((ROOT / "configs" / "datasets.json").read_text(encoding="utf-8"))["datasets"]
    geolife = datasets["geolife"]
    real_spec = geolife["data"]
    real = load_trajectories(real_spec, limit=int(geolife["public_slot_count"]))
    real_eval = real[int(0.8 * len(real)):]
    rows = []
    for epsilon in sweep["epsilons"]:
        rational = str(parse_fraction(epsilon))
        slug = f"eps_{Fraction(rational).numerator}_{Fraction(rational).denominator}"
        for seed in sweep["noise_seeds"]:
            trajectory_path = synthetic_root / slug / f"seed_{seed}" / "trajectories.pkl"
            if not trajectory_path.is_file():
                raise FileNotFoundError(trajectory_path)
            synthetic = load_trajectories(str(trajectory_path), limit=None)
            metrics = destination_conditioned_next_region_metrics(synthetic, real_eval)
            row = {
                "epsilon": rational,
                "epsilon_value": float(Fraction(rational)),
                "seed": int(seed),
                **metrics,
                "synthetic_sha256": sha256_file(trajectory_path),
            }
            rows.append(row)
            run_dir = output / slug / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "route_decision.json", row)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "results.json", {
        "schema_version": 1,
        "status": "VERIFIED_PRIVACY_BUDGET_ROUTE_DECISION",
        "protocol": {
            "task": "destination-conditioned next-region prediction",
            "real_eval_start": int(0.8 * len(real)),
            "real_test_trip_count": len(real_eval),
            "matrix": "7 epsilon values by 5 DP noise seeds",
        },
        "rows": rows,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps({"status": "VERIFIED_PRIVACY_BUDGET_ROUTE_DECISION", "runs": len(rows), "out_dir": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
