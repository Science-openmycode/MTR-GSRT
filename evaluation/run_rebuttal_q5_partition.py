"""Measured partition and quotient-granularity sensitivity for rebuttal Q5.

This is an internal utility experiment.  It recomputes exact q5 workloads only
in memory, applies the same exact discrete-Laplace mechanism and fixed-mass
projection as MTR-GSRT, and serializes only aggregate utility diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PUBLIC = ROOT / "generation" / "mtr_gsrt"
SEEDS = tuple(range(20260719, 20260724))
Q = 1_000_000
EPSILON_NUMERATOR = 1
EPSILON_DENOMINATOR = 5
SPECIFICATIONS = [
    *(('nested', regions) for regions in (
        48, 64, 80, 96, 112, 128, 160, 192, 224, 256,
        288, 320, 352, 384, 416, 448, 480, 512, 640, 768, 896, 1024,
    )),
    *(('flat', regions) for regions in (96, 192, 256, 320, 384, 448, 512, 768)),
]


def distribution(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float).ravel(), 0.0)
    total = float(values.sum())
    return values / total if total > 0 else np.full(values.size, 1.0 / values.size)


def cpc(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.minimum(distribution(first), distribution(second)).sum())


def jsd(first: np.ndarray, second: np.ndarray) -> float:
    p, q = distribution(first), distribution(second)
    m = 0.5 * (p + q)
    result = 0.0
    for values in (p, q):
        keep = values > 0
        result += 0.5 * float(np.sum(values[keep] * np.log2(values[keep] / m[keep])))
    return result


def ndcg(relevance: np.ndarray, ranking_score: np.ndarray) -> float:
    relevance = np.asarray(relevance, dtype=float)
    if float(relevance.sum()) <= 0:
        return math.nan
    order = np.argsort(-np.asarray(ranking_score, dtype=float), kind="stable")
    ideal = np.argsort(-relevance, kind="stable")
    discount = 1.0 / np.log2(np.arange(len(relevance), dtype=float) + 2.0)
    dcg = float(np.sum(relevance[order] * discount))
    idcg = float(np.sum(relevance[ideal] * discount))
    return dcg / idcg if idcg > 0 else math.nan


def projected_release(exact: dict[str, np.ndarray], masses: dict[str, int], seed: int):
    import certified_discrete_dp as certified
    import portal_fiber_route_release_development as portal

    names = tuple(exact)
    joined = np.concatenate([exact[name].ravel() for name in names])
    noisy, _ = certified.add_exact_discrete_laplace(
        joined,
        epsilon_numerator=EPSILON_NUMERATOR,
        epsilon_denominator=EPSILON_DENOMINATOR,
        sensitivity=Q,
        rng=random.Random(int(seed) + 770_027),
    )
    released, cursor = {}, 0
    for name in names:
        size = exact[name].size
        block = noisy[cursor:cursor + size].reshape(exact[name].shape)
        released[name] = portal.project_simplex(
            block.astype(float), float(masses[name]) * CAPACITY
        ) / float(Q)
        cursor += size
    return released


def portal_metrics(exact: np.ndarray, released: np.ndarray, portals, offsets) -> dict:
    weights, local_cpc, local_ndcg, local_top1 = [], [], [], []
    for key in sorted(portals):
        start = int(offsets[key])
        stop = start + len(portals[key])
        truth = np.asarray(exact[start:stop], dtype=float)
        estimate = np.asarray(released[start:stop], dtype=float)
        mass = float(truth.sum())
        if mass <= 0 or len(truth) == 0:
            continue
        weights.append(mass)
        local_cpc.append(cpc(truth, estimate))
        local_ndcg.append(ndcg(truth, estimate))
        local_top1.append(float(int(np.argmax(truth)) == int(np.argmax(estimate))))
    weights = np.asarray(weights, dtype=float)
    weights /= max(float(weights.sum()), 1e-15)
    expected_abs_noise = 2.0 * math.exp(-0.2 / Q) / (-math.expm1(-0.4 / Q))
    positive = exact > 0
    dominated = positive & (exact <= expected_abs_noise)
    return {
        "portal_global_cpc": cpc(exact, released),
        "portal_global_jsd": jsd(exact, released),
        "portal_context_cpc": float(np.sum(weights * np.asarray(local_cpc))),
        "portal_context_ndcg": float(np.sum(weights * np.asarray(local_ndcg))),
        "portal_context_top1": float(np.sum(weights * np.asarray(local_top1))),
        "positive_contexts": int(len(weights)),
        "portal_atoms": int(len(exact)),
        "noise_dominated_mass": float(exact[dominated].sum() / max(float(exact.sum()), 1.0)),
    }


def build_configuration(coords, graph, coarse_context, strategy: str, regions: int):
    import audit_two_level_semimarkov_dp as audit
    import nested_quotient_graph as nested
    import portal_fiber_route_release_development as portal

    if strategy == "nested":
        fine_context, _, _ = nested.build_nested_context(coords, graph, 24, regions, 4, 3)
    elif strategy == "flat":
        fine_context, _ = audit.build_context(coords, graph, 24, regions, 4, 3)
    else:
        raise ValueError(strategy)
    portals = portal.diverse_portals(graph, coords, fine_context.node_fine, 6)
    offsets, portal_atoms = portal.portal_layout(portals)
    exact = {
        "coarse24_occupancy": np.zeros(24, dtype=np.int64),
        "fine_occupancy": np.zeros(regions, dtype=np.int64),
        "fine96_flow": np.zeros(len(coarse_context.fine_edge_index) + 1, dtype=np.int64),
        "fine_flow": np.zeros(len(fine_context.fine_edge_index) + 1, dtype=np.int64),
        "portal_fiber_flow": np.zeros(portal_atoms + 1, dtype=np.int64),
    }
    return fine_context, portals, offsets, exact


def hierarchy_diagnostics(fine) -> dict:
    """Measure whether a fine public cell crosses the fixed coarse partition."""
    crossing_cells = 0
    crossing_nodes = 0
    max_parents = 1
    for region in range(int(fine.fine_regions)):
        nodes = np.flatnonzero(np.asarray(fine.node_fine, dtype=int) == region)
        parents = np.unique(np.asarray(fine.node_coarse, dtype=int)[nodes])
        max_parents = max(max_parents, int(len(parents)))
        if len(parents) > 1:
            crossing_cells += 1
            crossing_nodes += int(len(nodes))
    return {
        "crossing_fine_cells": crossing_cells,
        "crossing_fine_cell_fraction": crossing_cells / float(fine.fine_regions),
        "nodes_in_crossing_cells": crossing_nodes,
        "crossing_node_fraction": crossing_nodes / float(len(fine.node_fine)),
        "max_parents_per_fine": max_parents,
    }


def quantize_blocks(trajectory, context96, context_fine, portals, offsets, coords):
    import portal_fiber_route_release_development as portal
    import endpoint_coarse_route_release_development as endpoint_release

    array = np.asarray(trajectory, dtype=float)
    if len(array) < 2:
        return None
    midpoints = 0.5 * (array[:-1, :2] + array[1:, :2])
    lengths = np.linalg.norm(array[1:, :2] - array[:-1, :2], axis=1)
    if float(lengths.sum()) <= 0:
        return None
    nodes = np.asarray(context96.road_tree.query(midpoints)[1], dtype=int)
    labels24 = np.asarray(context96.node_coarse[nodes], dtype=int)
    labels96 = np.asarray(context96.node_fine[nodes], dtype=int)
    labels_fine = np.asarray(context_fine.node_fine[nodes], dtype=int)
    fractions = np.cumsum(lengths) / float(lengths.sum())
    phase_fraction = np.asarray([0.5] + fractions[:-1].tolist(), dtype=float)
    blocks = {
        "coarse24_occupancy": portal.query._occupancy(
            labels24, lengths, phase_fraction, family=0, families=1, phases=1,
            regions=24,
        ),
        "fine_occupancy": portal.query._occupancy(
            labels_fine, lengths, phase_fraction, family=0, families=1, phases=1,
            regions=context_fine.fine_regions,
        ),
        "fine96_flow": endpoint_release.transition_with_hold(
            labels96, context96.fine_adjacency, context96.fine_edge_index,
            family=0, families=1,
        ),
        "fine_flow": endpoint_release.transition_with_hold(
            labels_fine, context_fine.fine_adjacency, context_fine.fine_edge_index,
            family=0, families=1,
        ),
        "portal_fiber_flow": portal.portal_fiber_block(
            array, labels_fine, context_fine, portals, offsets, coords,
        ),
    }
    masses = {
        "coarse24_occupancy": 100_000,
        "fine_occupancy": 100_000,
        "fine96_flow": 150_000,
        "fine_flow": 300_000,
        "portal_fiber_flow": 350_000,
    }
    quantized = {}
    for name, values in blocks.items():
        values = np.maximum(np.asarray(values, dtype=float), 0.0)
        total = float(values.sum())
        if total <= 0:
            return None
        shape = values.shape
        scaled = (values / total * masses[name]).ravel()
        floor = np.floor(scaled).astype(np.int64)
        remainder = masses[name] - int(floor.sum())
        if remainder:
            order = np.argsort(-(scaled - floor), kind="stable")
            floor[order[:remainder]] += 1
        quantized[name] = floor.reshape(shape)
    if sum(int(value.sum()) for value in quantized.values()) != Q:
        raise RuntimeError("generalized q5 record does not contribute Q")
    return quantized


def main() -> None:
    global CAPACITY, SEEDS, SPECIFICATIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, help="Complete preprocessed real trajectory file")
    parser.add_argument("--osm-cache", required=True, help="Public OSM cache")
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--public-capacity", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-sha256", help="Optional expected complete real-file SHA-256")
    parser.add_argument("--osm-sha256", help="Optional expected public OSM-file SHA-256")
    parser.add_argument("--spec", action="append", help="Repeat strategy:K; omit for all 30 configurations")
    parser.add_argument("--seed", action="append", type=int, help="Repeat noise seed; omit for five fixed seeds")
    args = parser.parse_args()
    if args.public_capacity < 1:
        parser.error("--public-capacity must be positive")
    if args.spec:
        selected = []
        for item in args.spec:
            try:
                strategy, region_text = item.split(":", 1)
                pair = (strategy, int(region_text))
            except ValueError as exc:
                raise ValueError("--spec must be strategy:K") from exc
            if pair not in SPECIFICATIONS or pair in selected:
                raise ValueError(f"Unsupported or duplicate partition specification: {item}")
            selected.append(pair)
        SPECIFICATIONS = selected
    if args.seed:
        if len(args.seed) != len(set(args.seed)):
            parser.error("Duplicate noise seed")
        SEEDS = tuple(args.seed)
    if len(SEEDS) != 5:
        parser.error("Exactly five noise seeds are needed for the saved Student-t interval")
    def rooted(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else ROOT / path).resolve()

    data, osm_path, out = rooted(args.real), rooted(args.osm_cache), rooted(args.out_dir)
    if out.exists():
        parser.error(f"Output directory exists; use a new path: {out}")
    for path, expected in ((data, args.input_sha256), (osm_path, args.osm_sha256)):
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected and hashlib.sha256(path.read_bytes()).hexdigest().lower() != expected.lower():
            raise ValueError(f"Input SHA-256 mismatch: {path}")
    sys.path.insert(0, str(PUBLIC))
    from generation.common.runtime import add_runtime_paths
    from public_utils import filter_osm_ways_by_bbox, load_trajectories
    add_runtime_paths()
    import audit_two_level_semimarkov_dp as audit
    import graph_voronoi_doptimal_support_probe as support
    import route_structure_potential_experiment as route

    bbox = tuple(args.bbox)
    CAPACITY = int(args.public_capacity)
    trajectories = load_trajectories(str(data), limit=None)
    if len(trajectories) != CAPACITY:
        raise ValueError(f"Complete real input has {len(trajectories)} valid records, expected {CAPACITY}")
    with osm_path.open("rb") as handle:
        osm = filter_osm_ways_by_bbox(pickle.load(handle), bbox)
    support.BBOX = bbox
    coords, graph = route.prepare_graph([], bbox=bbox, osm_ways=osm, raw_graph=False)
    context96, _ = audit.build_context(coords, graph, 24, 96, 4, 3)

    out.mkdir(parents=True)
    detailed = out / "q5_partition_sensitivity_detailed.csv"
    existing: dict[tuple[str, int], list[dict]] = {}
    if detailed.is_file():
        integer_fields = {
            "fine_regions", "seed", "accepted_trajectories", "fine_edges",
            "portal_groups", "positive_contexts", "portal_atoms",
            "crossing_fine_cells", "nodes_in_crossing_cells", "max_parents_per_fine",
        }
        with detailed.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row = {
                    key: (int(value) if key in integer_fields else
                          float(value) if key != "strategy" else value)
                    for key, value in raw.items()
                }
                existing.setdefault((row["strategy"], row["fine_regions"]), []).append(row)

    rows = []
    for strategy, regions in SPECIFICATIONS:
        fine, portals, offsets, exact = build_configuration(
            coords, graph, context96, strategy, regions
        )
        hierarchy = hierarchy_diagnostics(fine)
        cached = existing.get((strategy, regions), [])
        if len(cached) == len(SEEDS) and {row["seed"] for row in cached} == set(SEEDS):
            print(f"[partition] reuse strategy={strategy} regions={regions}", flush=True)
            for row in cached:
                row.update(hierarchy)
            rows.extend(sorted(cached, key=lambda row: row["seed"]))
            continue
        print(f"[partition] build strategy={strategy} regions={regions}", flush=True)
        accepted = 0
        for index, trajectory in enumerate(trajectories, start=1):
            quantized = quantize_blocks(
                trajectory, context96, fine, portals, offsets, coords
            )
            if quantized is not None:
                accepted += 1
                for name in exact:
                    exact[name] += quantized[name].ravel()
            if index % 1000 == 0 or index == len(trajectories):
                print(
                    f"[partition {strategy}-{regions}] {index}/{len(trajectories)}",
                    flush=True,
                )
        for seed in SEEDS:
            released = projected_release(exact, {
                "coarse24_occupancy": 100_000,
                "fine_occupancy": 100_000,
                "fine96_flow": 150_000,
                "fine_flow": 300_000,
                "portal_fiber_flow": 350_000,
            }, seed)
            metrics = portal_metrics(
                exact["portal_fiber_flow"], released["portal_fiber_flow"], portals, offsets
            )
            rows.append({
                "strategy": strategy,
                "fine_regions": regions,
                "seed": seed,
                "accepted_trajectories": accepted,
                "fine_edges": len(fine.fine_edge_index),
                "portal_groups": len(portals),
                **hierarchy,
                **metrics,
            })

    with detailed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for strategy, regions in SPECIFICATIONS:
        current = [r for r in rows if r["strategy"] == strategy and r["fine_regions"] == regions]
        item = {key: current[0][key] for key in (
            "strategy", "fine_regions", "accepted_trajectories", "fine_edges",
            "portal_groups", "positive_contexts", "portal_atoms", "noise_dominated_mass",
            "crossing_fine_cells", "crossing_fine_cell_fraction",
            "nodes_in_crossing_cells", "crossing_node_fraction", "max_parents_per_fine",
        )}
        for metric in (
            "portal_global_cpc", "portal_global_jsd", "portal_context_cpc",
            "portal_context_ndcg", "portal_context_top1",
        ):
            values = np.asarray([row[metric] for row in current], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sd"] = float(values.std(ddof=1))
            half = 2.776 * float(values.std(ddof=1)) / math.sqrt(len(values))
            item[f"{metric}_ci95_low"] = float(values.mean() - half)
            item[f"{metric}_ci95_high"] = float(values.mean() + half)
        item["context_coverage"] = (
            float(item["positive_contexts"]) / max(float(item["portal_groups"]), 1.0)
        )
        item["atoms_per_positive_context"] = (
            float(item["portal_atoms"]) / max(float(item["positive_contexts"]), 1.0)
        )
        expected_abs_noise = 2.0 * math.exp(-0.2 / Q) / (-math.expm1(-0.4 / Q))
        item["mean_signal_noise_ratio"] = (
            350_000.0 * float(item["accepted_trajectories"])
            / max(float(item["portal_atoms"]), 1.0)
            / expected_abs_noise
        )
        summary.append(item)
    summary_path = out / "q5_partition_sensitivity_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    (out / "q5_partition_sensitivity_manifest.json").write_text(json.dumps({
        "classification": "INTERNAL_PRIVATE_UTILITY_DIAGNOSTIC_NOT_A_DP_RELEASE",
        "exact_workloads_serialized": False,
        "input_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "osm_sha256": hashlib.sha256(osm_path.read_bytes()).hexdigest(),
        "bbox": bbox,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "epsilon_q5": "1/5",
        "noise_seeds": list(SEEDS),
        "trajectory_count": len(trajectories),
        "configurations": [{"strategy": s, "fine_regions": k} for s, k in SPECIFICATIONS],
        "interpretation": "query-level route-choice fidelity under measured q5; not full-decoder utility",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
