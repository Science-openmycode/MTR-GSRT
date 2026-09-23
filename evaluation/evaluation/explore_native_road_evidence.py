"""Measure raw coordinate road evidence without map matching or path completion.

Every score is computed from released coordinates and a fixed public directed
OSM graph.  No shortest path, FMM, or hidden intermediate road segment is
inserted between observations.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "configs" / "datasets.json").is_file():
            sys.path.insert(0, str(candidate))
            break

from generation.common.runtime import dataset_config, public_path, write_json  # noqa: E402


def _named_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"expected NAME=PATH, got {specification!r}")
    name, path = specification.split("=", 1)
    return name, public_path(path)


def _meters_between(points: np.ndarray, nearest: np.ndarray) -> np.ndarray:
    latitude = np.deg2rad((points[:, 0] + nearest[:, 0]) / 2.0)
    dy = np.deg2rad(points[:, 0] - nearest[:, 0]) * 6_371_008.8
    dx = np.deg2rad(points[:, 1] - nearest[:, 1]) * 6_371_008.8 * np.cos(latitude)
    return np.hypot(dx, dy)


def _step_meters(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.empty(0, dtype=float)
    return _meters_between(points[:-1], points[1:])


def _evaluate(corpus, coordinates, tree, edges, max_points, proximity_tau_m, jump_m) -> dict:
    point_scores = []
    edge_valid = edge_total = 0
    long_steps = step_total = 0
    route_evidence = []
    for trajectory in corpus:
        raw = np.asarray(trajectory, dtype=float)
        if len(raw) < 2:
            route_evidence.append(0.0)
            continue
        sample = raw if len(raw) <= max_points else raw[np.linspace(0, len(raw) - 1, max_points).round().astype(int)]
        node_ids = tree.query(sample, k=1)[1].astype(int)
        nearest = coordinates[node_ids]
        distances = _meters_between(sample, nearest)
        proximity = float(np.mean(np.exp(-distances / proximity_tau_m)))
        point_scores.append(proximity)
        nodes = node_ids[np.r_[True, np.diff(node_ids) != 0]]
        local_valid = local_total = 0
        for u, v in zip(nodes[:-1], nodes[1:]):
            local_total += 1
            edge_total += 1
            if (int(u), int(v)) in edges:
                local_valid += 1
                edge_valid += 1
        steps = _step_meters(raw)
        local_jump = float(np.mean(steps <= jump_m)) if len(steps) else 0.0
        step_total += len(steps)
        long_steps += int(np.sum(steps > jump_m))
        local_continuity = local_valid / local_total if local_total else 0.0
        # This is deliberately reported only as a diagnostic combination.  It
        # contains no path completion: all three factors must be evidenced by
        # released coordinates themselves.
        route_evidence.append(proximity * local_continuity * local_jump)
    point_proximity = float(np.mean(point_scores)) if point_scores else 0.0
    continuity = edge_valid / edge_total if edge_total else 0.0
    short_step_share = 1.0 - long_steps / step_total if step_total else 0.0
    return {
        "road_proximity_score": point_proximity,
        "directed_edge_continuity": continuity,
        "long_jump_rate": 1.0 - short_step_share,
        "route_evidence_yield": float(np.mean([value > 0.95 for value in route_evidence])) if route_evidence else 0.0,
        "native_road_evidence_mean": float(np.mean(route_evidence)) if route_evidence else 0.0,
        "native_road_evidence_median": float(np.median(route_evidence)) if route_evidence else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--method", action="append", required=True, help="Repeat NAME=RELEASE_PATH")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--proximity-tau-m", type=float, default=50.0)
    parser.add_argument("--jump-m", type=float, default=200.0)
    args = parser.parse_args()
    if args.max_points < 2 or args.proximity_tau_m <= 0 or args.jump_m <= 0:
        raise ValueError("metric parameters must be positive")
    from public_utils import filter_osm_ways_by_bbox, filter_osm_ways_by_highway, load_osm_ways, load_trajectories
    from pipeline.evaluate_kdd_revised_statistics import graph_context
    config = dataset_config(args.dataset_config)
    bbox = tuple(map(float, config["bbox"]))
    ways = filter_osm_ways_by_bbox(load_osm_ways(str(public_path(config["osm_cache"]))), bbox)
    ways = filter_osm_ways_by_highway(ways, list(config.get("osm_highway_classes", [])))
    coordinates, edges, tree, _ = graph_context(ways, bbox)
    methods = dict(_named_path(item) for item in args.method)
    results = {}
    for name, path in methods.items():
        corpus = load_trajectories(str(path), limit=args.limit)
        results[name] = _evaluate(
            corpus, coordinates, tree, edges, args.max_points,
            args.proximity_tau_m, args.jump_m,
        )
        results[name]["record_count"] = len(corpus)
    payload = {
        "schema_version": 1,
        "classification": "EXPLORATORY_RAW_COORDINATE_ROAD_EVIDENCE_NO_FMM_NO_PATH_COMPLETION",
        "definition": {
            "road_proximity_score": "mean exp(-nearest_public_road_node_distance_m/tau_m)",
            "directed_edge_continuity": "share of consecutive distinct nearest public nodes joined by a directed public edge",
            "long_jump_rate": "share of released consecutive coordinate gaps greater than jump_m",
            "native_road_evidence_mean": "per-record proximity * direct-edge-continuity * short-step-share, then averaged",
        },
        "parameters": {
            "limit": args.limit, "max_points": args.max_points,
            "proximity_tau_m": args.proximity_tau_m, "jump_m": args.jump_m,
        },
        "methods": {name: str(path) for name, path in methods.items()},
        "results": results,
    }
    out_dir = public_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "native_road_evidence.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
