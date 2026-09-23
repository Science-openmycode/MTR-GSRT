from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def named_path(text: str) -> tuple[str, Path]:
    try:
        name, value = text.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from exc
    path = Path(value)
    return name, (path if path.is_absolute() else ROOT / path).resolve()


def similarity(value: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(value)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the unified evaluator for every synthetic corpus and aggregate a fresh profile."
    )
    parser.add_argument("--real", required=True)
    parser.add_argument("--synthetic", action="append", type=named_path, required=True)
    parser.add_argument("--witness", action="append", type=named_path, default=[])
    parser.add_argument("--dataset-config")
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--public-slot-count", type=int)
    parser.add_argument("--osm-cache", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    witnesses = dict(args.witness)
    output = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, synthetic in args.synthetic:
        raw = output / "raw" / name.replace(" ", "_")
        command = [
            sys.executable, str(ROOT / "evaluation" / "evaluation" / "evaluate_all.py"),
            "--real", str(Path(args.real).resolve()), "--synthetic", str(synthetic),
            "--osm-cache", str(Path(args.osm_cache).resolve()), "--out-dir", str(raw),
        ]
        if args.dataset_config:
            command += ["--dataset-config", args.dataset_config]
        if args.bbox:
            command += ["--bbox", *(str(value) for value in args.bbox)]
        if args.public_slot_count:
            command += ["--public-slot-count", str(args.public_slot_count)]
        if name in witnesses:
            command += ["--witness", str(witnesses[name])]
        print("RUN:", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        metrics = json.loads((raw / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        values = {
            "RoadYield": metrics.get("route_compatible_yield", 0.0),
            "DirValid": metrics.get("directed_road_validity", 0.0),
            "WitnessValid": metrics.get("witness_valid", 0.0),
            "DemandFid": similarity(metrics.get("trip_error", 1.0)),
            "GridSim": similarity(metrics.get("grid_density_jsd", 1.0)),
            "LengthSim": similarity(metrics.get("path_length_jsd", 1.0)),
            "NextRoadAcc": metrics.get("B2_next_region_accuracy", 0.0),
            "RouteMRR": metrics.get("B2_grid_route_mrr", 0.0),
            "RouteBest5F1": metrics.get("B2_grid_route_best5_transition_f1", 0.0),
        }
        for metric, value in values.items():
            rows.append({"algorithm": name, "layer": "executed_unified_evaluation", "metric": metric,
                         "value": value, "status": "VALID"})

    result = output / "results.csv"
    with result.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({
        "protocol": "executed_full_population_profile_v1",
        "corpora": [name for name, _ in args.synthetic],
        "result": str(result.resolve()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} freshly evaluated metric rows to {result}")


if __name__ == "__main__":
    main()
