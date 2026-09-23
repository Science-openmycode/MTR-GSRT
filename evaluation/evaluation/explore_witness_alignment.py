"""Explore a continuous coordinate-to-road-witness alignment diagnostic.

This is an exploratory diagnostic, not a replacement for WitnessValid.  The
latter remains a binary release-object certificate.  WitnessAlignment measures
how faithfully a coordinate trajectory follows its supplied public witness.
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
from metric_suites.road_route import (  # noqa: E402
    _point_segment_projection,
    _xy_m,
    load_witness_records,
    sample_trajectory,
)


def _score_records(
    trajectories: list[np.ndarray],
    witnesses: list[dict],
    node_coordinates: np.ndarray,
    max_points: int,
    scale_m: float,
) -> dict:
    """Return per-slot and aggregate threshold-free alignment scores."""
    scores: list[float] = []
    p95_values: list[float] = []
    endpoint_values: list[float] = []
    monotone_values: list[float] = []
    structural_valid: list[bool] = []
    for slot, (trajectory, witness) in enumerate(zip(trajectories, witnesses)):
        nodes = np.asarray(witness.get("node_sequence", []), dtype=int) if isinstance(witness, dict) else np.asarray([], dtype=int)
        pairs = witness.get("directed_edges", []) if isinstance(witness, dict) else []
        structurally_valid = (
            isinstance(witness, dict)
            and int(witness.get("slot", -1)) == slot
            and len(nodes) >= 2
            and len(pairs) == len(nodes) - 1
            and np.all((0 <= nodes) & (nodes < len(node_coordinates)))
            and all(tuple(map(int, pair)) == (int(nodes[index]), int(nodes[index + 1])) for index, pair in enumerate(pairs))
        )
        structural_valid.append(structurally_valid)
        if not structurally_valid:
            scores.append(0.0)
            p95_values.append(math.inf)
            endpoint_values.append(math.inf)
            monotone_values.append(0.0)
            continue
        sampled = sample_trajectory(np.asarray(trajectory, dtype=float), max_points)
        latitude0 = float(np.mean(sampled[:, 0]))
        distances, positions = _point_segment_projection(
            _xy_m(sampled, latitude0), _xy_m(node_coordinates[nodes], latitude0)
        )
        p95 = float(np.quantile(distances, 0.95))
        endpoint = float(max(distances[0], distances[-1]))
        monotone = float(np.mean(np.diff(positions) >= -1e-6)) if len(positions) > 1 else 1.0
        # A score of one requires coincident coordinates and a forward traversal.
        # Residual penalties are smooth; scale_m is a public reporting scale.
        alignment = math.exp(-(p95 + endpoint) / scale_m) * monotone
        scores.append(alignment)
        p95_values.append(p95)
        endpoint_values.append(endpoint)
        monotone_values.append(monotone)
    finite = np.asarray([value for value in p95_values if math.isfinite(value)], dtype=float)
    return {
        "record_count": len(scores),
        "structural_valid_fraction": float(np.mean(structural_valid)) if structural_valid else 0.0,
        "witness_alignment_mean": float(np.mean(scores)) if scores else 0.0,
        "witness_alignment_median": float(np.median(scores)) if scores else 0.0,
        "witness_alignment_p05": float(np.quantile(scores, 0.05)) if scores else 0.0,
        "witness_alignment_p95": float(np.quantile(scores, 0.95)) if scores else 0.0,
        "p95_residual_median_m": float(np.median(finite)) if len(finite) else math.inf,
        "endpoint_residual_median_m": float(np.median([v for v in endpoint_values if math.isfinite(v)])) if finite.size else math.inf,
        "monotone_projection_mean": float(np.mean(monotone_values)) if monotone_values else 0.0,
    }


def _reverse_witnesses(witnesses: list[dict]) -> list[dict]:
    reversed_records = []
    for witness in witnesses:
        nodes = list(reversed(witness["node_sequence"]))
        record = dict(witness)
        record["node_sequence"] = nodes
        record["directed_edges"] = [[nodes[index], nodes[index + 1]] for index in range(len(nodes) - 1)]
        reversed_records.append(record)
    return reversed_records


def _shuffled_witnesses(witnesses: list[dict], seed: int) -> list[dict]:
    order = np.random.default_rng(seed).permutation(len(witnesses))
    shuffled = []
    for slot, source_index in enumerate(order):
        record = dict(witnesses[int(source_index)])
        record["slot"] = slot
        shuffled.append(record)
    return shuffled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--witness", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--scale-m", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--limit", type=int, default=None, help="Optional fixed prefix for an exploratory run.")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("paired", "slot-permuted", "reversed"),
        default=("paired", "slot-permuted", "reversed"),
        help="Paired is the release metric; the other variants are stress-test controls.",
    )
    args = parser.parse_args()

    from public_utils import filter_osm_ways_by_bbox, filter_osm_ways_by_highway, load_osm_ways, load_trajectories
    config = dataset_config(args.dataset_config)
    bbox = tuple(map(float, config["bbox"]))
    trajectories = load_trajectories(str(public_path(args.synthetic)), limit=None)
    witnesses = load_witness_records(public_path(args.witness))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        trajectories = trajectories[:args.limit]
        witnesses = witnesses[:args.limit]
    if len(trajectories) != len(witnesses):
        raise ValueError("synthetic and witness releases must have equal record counts")
    ways = filter_osm_ways_by_bbox(load_osm_ways(str(public_path(config["osm_cache"]))), bbox)
    ways = filter_osm_ways_by_highway(ways, list(config.get("osm_highway_classes", [])))
    final_dir = public_path("src/mtr/DP_GSRT/final")
    sys.path.insert(0, str(final_dir))
    import route_structure_potential_experiment as route
    coordinates, _ = route.prepare_graph([], bbox=bbox, osm_ways=ways, raw_graph=False)
    coordinates = np.asarray(coordinates, dtype=float)

    all_variants = {
        "paired_witness": witnesses,
        "slot_permuted_witness": _shuffled_witnesses(witnesses, args.seed),
        "reversed_witness": _reverse_witnesses(witnesses),
    }
    aliases = {
        "paired": "paired_witness",
        "slot-permuted": "slot_permuted_witness",
        "reversed": "reversed_witness",
    }
    variants = {aliases[name]: all_variants[aliases[name]] for name in args.variants}
    payload = {
        "schema_version": 1,
        "classification": "EXPLORATORY_CONTINUOUS_WITNESS_DIAGNOSTIC_NOT_PAPER_METRIC",
        "definition": "I_struct * exp(-(q95_coordinate_to_witness_m + endpoint_coordinate_to_witness_m)/tau_m) * monotone_projection_share",
        "interpretation": "one means a structurally valid, coincident, forward coordinate traversal; zero includes malformed or unbound witnesses",
        "parameters": {
            "max_points": args.max_points, "tau_m": args.scale_m, "seed": args.seed,
            "limit": args.limit, "variants": list(args.variants),
        },
        "variants": {
            name: _score_records(trajectories, variant, coordinates, args.max_points, args.scale_m)
            for name, variant in variants.items()
        },
    }
    out_dir = public_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "witness_alignment_exploration.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
