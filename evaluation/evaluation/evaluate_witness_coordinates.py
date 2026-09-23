"""Evaluate the geometric and ordering agreement of an MTR witness release."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (_candidate / "configs" / "datasets.json").is_file():
            sys.path.insert(0, str(_candidate))
            break

from generation.common.runtime import dataset_config, public_path, sha256_file, write_json  # noqa: E402
from metric_suites.road_route import load_witness_records, witness_coordinate_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--witness", required=True)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--distance-m", type=float, default=200.0)
    parser.add_argument("--monotone-share", type=float, default=0.95)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from public_utils import filter_osm_ways_by_bbox, filter_osm_ways_by_highway, load_osm_ways, load_trajectories

    started = time.time()
    config = dataset_config(args.dataset_config)
    bbox = tuple(map(float, config["bbox"]))
    osm_path = public_path(config["osm_cache"])
    synthetic_path = public_path(args.synthetic)
    witness_path = public_path(args.witness)
    trajectories = load_trajectories(str(synthetic_path), limit=None)
    witnesses = load_witness_records(witness_path)
    ways = filter_osm_ways_by_bbox(load_osm_ways(str(osm_path)), bbox)
    ways = filter_osm_ways_by_highway(ways, list(config.get("osm_highway_classes", [])))
    final_dir = public_path("src/mtr/DP_GSRT/final")
    sys.path.insert(0, str(final_dir))
    import route_structure_potential_experiment as route

    coordinates, _ = route.prepare_graph([], bbox=bbox, osm_ways=ways, raw_graph=False)
    metrics = witness_coordinate_metrics(
        trajectories,
        witnesses,
        np.asarray(coordinates, dtype=float),
        args.max_points,
        args.distance_m,
        args.monotone_share,
    )
    out_dir = public_path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "dataset": config["name"],
        "parameters": {
            "max_points": args.max_points,
            "distance_m": args.distance_m,
            "monotone_share": args.monotone_share,
        },
        "metrics": metrics,
    }
    write_json(out_dir / "metrics.json", payload)
    manifest = {
        "inputs": {
            "synthetic": {"path": str(synthetic_path), "sha256": sha256_file(synthetic_path)},
            "witness": {"path": str(witness_path), "sha256": sha256_file(witness_path)},
            "osm": {"path": str(osm_path), "sha256": sha256_file(osm_path)},
        },
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "metric_module_sha256": sha256_file(Path(__file__).resolve().parent / "metric_suites" / "road_route.py"),
        "metrics_sha256": sha256_file(out_dir / "metrics.json"),
        "elapsed_sec": time.time() - started,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
