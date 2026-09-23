"""Evaluate saved external releases on the retained route-choice task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.common.runtime import dataset_config, public_path
from public_utils import load_trajectories
from check.route_metric_replacement_20260724.screen_replacement import evaluate_trips, train_model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--synthetic", nargs=2, action="append", required=True, metavar=("NAME", "PATH"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    config = dataset_config(args.dataset_config)
    registered = {config["name"], *config.get("aliases", [])}
    real_path = public_path(config["data"] if args.real.lower() in registered else args.real)
    real = load_trajectories(str(real_path), limit=int(config["public_slot_count"]))
    test = real[int(0.8 * len(real)):]
    bbox = tuple(float(value) for value in config["bbox"])
    rows = []
    inputs = {"real": {"path": str(real_path), "sha256": sha256(real_path)}}
    for name, value in args.synthetic:
        path = public_path(value)
        synthetic = load_trajectories(str(path), limit=None)
        outcomes = evaluate_trips(train_model(synthetic, bbox), test, bbox)
        inputs[name] = {"path": str(path), "sha256": sha256(path), "count": len(synthetic)}
        for metric in ("NextRegionAcc", "NextRegionNLL"):
            rows.append({"method": name, "metric": metric, "mean": float(np.mean(outcomes[metric])), "trip_count": len(test)})

    out = public_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out / "metrics.json").write_text(json.dumps({
        "protocol": "retrospective destination-conditioned next-region task",
        "interface_note": "Saved external releases are evaluated as published; no model training or trajectory synthesis is rerun.",
        "inputs": inputs,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "classification": "EXTERNAL_INTERFACE_ROUTE_DECISION_DIAGNOSTIC",
        "inputs": inputs,
        "outputs": {
            "metrics.json": sha256(out / "metrics.json"),
            "metrics.csv": sha256(out / "metrics.csv"),
        },
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
