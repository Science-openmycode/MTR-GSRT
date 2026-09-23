"""Evaluate road choice from native released evidence, without FMM completion.

Real reference events come from a fixed, precomputed real-only map-matching
measurement.  MTR events come from its released witnesses.  Coordinate-only
methods contribute only directly evidenced adjacent directed OSM edges; no
route is inserted between observations.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

if __package__ in {None, ""}:
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "configs" / "datasets.json").is_file():
            sys.path.insert(0, str(candidate))
            break

from generation.common.runtime import dataset_config, public_path, write_json  # noqa: E402
from metric_suites.road_choice import (  # noqa: E402
    matched_choice_trips,
    next_road_trip_scores,
    public_crossing_options,
    road_choice_fidelity,
)


def _named_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"expected NAME=PATH, got {specification!r}")
    name, value = specification.split("=", 1)
    if not name:
        raise ValueError("method name must not be empty")
    return name, public_path(value)


def _load_records(path: Path) -> list[dict]:
    with gzip.open(path, "rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise ValueError("real match cache must be a list")
    return records


def _sample(trajectory: np.ndarray, max_points: int) -> np.ndarray:
    if len(trajectory) <= max_points:
        return trajectory
    return trajectory[np.linspace(0, len(trajectory) - 1, max_points).round().astype(int)]


def _direct_coordinate_trips(
    trajectories: list[np.ndarray],
    coordinates: np.ndarray,
    tree: cKDTree,
    directed_edges: set[tuple[int, int]],
    option_context: dict[tuple[int, int], object],
    max_points: int,
) -> tuple[list[list[tuple[object, tuple[int, int]]]], dict]:
    trips = []
    records_with_any_edge = 0
    records_with_event = 0
    valid = total = 0
    for trajectory in trajectories:
        points = _sample(np.asarray(trajectory, dtype=float), max_points)
        nodes = tree.query(points, k=1)[1].astype(int)
        nodes = nodes[np.r_[True, np.diff(nodes) != 0]]
        events = []
        has_edge = False
        for u, v in zip(nodes[:-1], nodes[1:]):
            total += 1
            pair = (int(u), int(v))
            if pair not in directed_edges:
                continue
            valid += 1
            has_edge = True
            context = option_context.get(pair)
            if context is not None:
                events.append((context, pair))
        records_with_any_edge += int(has_edge)
        records_with_event += int(bool(events))
        trips.append(events)
    return trips, {
        "record_count": len(trips),
        "native_directed_edge_evidence_yield": records_with_any_edge / max(len(trips), 1),
        "native_choice_event_yield": records_with_event / max(len(trips), 1),
        "native_directed_edge_continuity": valid / total if total else 0.0,
    }


def _witness_trips(
    path: Path,
    option_context: dict[tuple[int, int], object],
) -> tuple[list[list[tuple[object, tuple[int, int]]]], dict]:
    with path.open("rb") as handle:
        witnesses = pickle.load(handle)
    trips = []
    records_with_event = 0
    for index, witness in enumerate(witnesses):
        if not isinstance(witness, dict) or int(witness.get("slot", -1)) != index:
            raise ValueError("invalid witness slot")
        events = []
        for edge in witness.get("directed_edges", ()):
            pair = tuple(map(int, edge))
            context = option_context.get(pair)
            if context is not None:
                events.append((context, pair))
        records_with_event += int(bool(events))
        trips.append(events)
    return trips, {
        "record_count": len(trips),
        "native_directed_edge_evidence_yield": 1.0,
        "native_choice_event_yield": records_with_event / max(len(trips), 1),
        "native_directed_edge_continuity": 1.0,
        "source": "released_witness",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--real-match-cache", required=True)
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--witness", action="append", default=[])
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--real-test-fraction", type=float, default=0.2)
    parser.add_argument("--quotient-regions", type=int, choices=(24, 96, 384), default=384)
    parser.add_argument("--mixture-weight", type=float, default=0.5)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if not 0 < args.real_test_fraction < 1:
        raise ValueError("--real-test-fraction must be in (0, 1)")
    from public_utils import filter_osm_ways_by_bbox, filter_osm_ways_by_highway, load_osm_ways, load_trajectories
    from evaluate_road_choice import _edge_endpoint_regions
    config = dataset_config(args.dataset_config)
    network = public_path(args.network)
    _, edge_nodes, carrier_regions, quotient = _edge_endpoint_regions(network, config, args.quotient_regions)
    carrier_ids = {index: pair for index, pair in enumerate(carrier_regions)}
    carrier_edge_regions = {
        index: carrier_regions[pair] for index, pair in carrier_ids.items()
    }
    public_options, _ = public_crossing_options(carrier_edge_regions, carrier_ids)
    option_context = {
        option: context
        for context, options in public_options.items()
        for option in options
    }
    edge_context = {edge_id: option_context[pair] for edge_id, pair in edge_nodes.items() if pair in option_context}

    real_records = _load_records(public_path(args.real_match_cache))
    real_trips = matched_choice_trips(real_records, edge_context, edge_nodes)
    split = int(len(real_trips) * (1.0 - args.real_test_fraction))
    real_test = real_trips[split:]

    bbox = tuple(map(float, config["bbox"]))
    ways = filter_osm_ways_by_bbox(load_osm_ways(str(public_path(config["osm_cache"]))), bbox)
    ways = filter_osm_ways_by_highway(ways, list(config.get("osm_highway_classes", [])))
    final_dir = public_path("src/mtr/DP_GSRT/final")
    sys.path.insert(0, str(final_dir))
    import route_structure_potential_experiment as route
    coordinates, graph = route.prepare_graph([], bbox=bbox, osm_ways=ways, raw_graph=False)
    coordinates = np.asarray(coordinates, dtype=float)
    directed_edges = {(int(u), int(v)) for u, adjacency in graph.items() for v, _ in adjacency}
    tree = cKDTree(coordinates)

    methods = dict(_named_path(value) for value in args.method)
    witnesses = dict(_named_path(value) for value in args.witness)
    if set(methods) & set(witnesses):
        raise ValueError("a method cannot have both coordinate and witness inputs")
    all_trips = {}
    evidence = {}
    for name, path in methods.items():
        trajectories = load_trajectories(str(path), limit=None)
        all_trips[name], evidence[name] = _direct_coordinate_trips(
            trajectories, coordinates, tree, directed_edges, option_context, args.max_points
        )
    for name, path in witnesses.items():
        all_trips[name], evidence[name] = _witness_trips(path, option_context)

    results = {}
    for name, trips in all_trips.items():
        fidelity = road_choice_fidelity(trips, real_test, public_options)
        scores = next_road_trip_scores(trips, real_test, public_options, mixture_weight=args.mixture_weight)
        coverage = evidence[name]["native_directed_edge_evidence_yield"]
        results[name] = {
            **evidence[name], **fidelity,
            "next_road_accuracy": float(np.mean(scores["next_road_accuracy"])),
            "next_road_nll": float(np.mean(scores["next_road_nll"])),
            "coverage_adjusted_rc_cpc": float(fidelity["road_choice_cpc"] * coverage),
            "coverage_adjusted_rc_ndcg": float(fidelity["road_choice_ndcg"] * coverage),
            "coverage_adjusted_next_road_accuracy": float(np.mean(scores["next_road_accuracy"]) * coverage),
        }
    payload = {
        "schema_version": 1,
        "classification": "NATIVE_ROAD_CHOICE_EVALUATION_NO_SYNTHETIC_FMM_COMPLETION",
        "definition": {
            "real_reference": "fixed real-only FMM measurement cache",
            "mtr": "released directed witness",
            "coordinate_baseline": "direct adjacent nearest-node edge only; non-edges produce no event",
            "coverage_adjusted": "conditional road-choice metric multiplied by native directed-edge evidence yield",
        },
        "parameters": {"max_points": args.max_points, "quotient_regions": args.quotient_regions, "real_test_fraction": args.real_test_fraction},
        "quotient": quotient,
        "results": results,
    }
    out_dir = public_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "native_road_choice_metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
