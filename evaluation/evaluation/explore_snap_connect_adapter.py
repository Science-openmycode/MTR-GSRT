"""Fast public Snap-and-Connect adapter diagnostic (no HMM / no FMM).

The adapter snaps a fixed number of published coordinates to a fixed public
OSM graph and joins consecutive anchors only when a bounded public A* route
passes fixed admissibility gates.  Every rejected record remains INVALID;
there is no conditioning on successful completions.  This script is an
exploratory system-level diagnostic, not a native-release metric.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from public_utils import filter_osm_ways_by_bbox, filter_osm_ways_by_highway, load_osm_ways, load_trajectories
from generation.common.runtime import dataset_config, public_path
from metric_suites.road_choice import (
    matched_choice_trips,
    next_road_trip_scores,
    public_crossing_options,
    road_choice_fidelity,
)


def meters(a: np.ndarray, b: np.ndarray) -> float:
    """Equirectangular distance; accurate enough over one city."""
    lat = math.radians((float(a[0]) + float(b[0])) / 2.0)
    dy = math.radians(float(a[0] - b[0])) * 6_371_000.0
    dx = math.radians(float(a[1] - b[1])) * 6_371_000.0 * math.cos(lat)
    return math.hypot(dx, dy)


def sample_anchor_points(traj: np.ndarray, anchors: int) -> np.ndarray:
    arr = np.asarray(traj, dtype=float)
    if len(arr) <= anchors:
        return arr
    index = np.unique(np.linspace(0, len(arr) - 1, anchors).round().astype(int))
    return arr[index]


def astar(graph, coords: np.ndarray, src: int, dst: int, max_visits: int):
    import heapq
    if src == dst:
        return [src], 0.0
    def h(node: int) -> float:
        return float(np.linalg.norm(coords[node] - coords[dst]))
    best = {src: 0.0}
    prev: dict[int, int] = {}
    queue = [(h(src), 0.0, src)]
    visits = 0
    while queue and visits < max_visits:
        _, cost, node = heapq.heappop(queue)
        visits += 1
        if cost != best.get(node):
            continue
        if node == dst:
            path = [dst]
            while path[-1] != src:
                path.append(prev[path[-1]])
            path.reverse()
            return path, cost
        for nxt, weight in graph.get(node, ()):
            next_cost = cost + float(weight)
            if next_cost < best.get(nxt, float("inf")):
                best[nxt] = next_cost
                prev[nxt] = node
                heapq.heappush(queue, (next_cost + h(nxt), next_cost, nxt))
    return None, float("inf")


def complete_one(traj, tree, coords, graph, *, anchors, snap_m, segment_m, stretch, max_visits):
    points = sample_anchor_points(np.asarray(traj, dtype=float), anchors)
    if len(points) < 2:
        return None, "too_short"
    node_ids = tree.query(points, k=1)[1].astype(int)
    snapped = coords[node_ids]
    if any(meters(point, node) > snap_m for point, node in zip(points, snapped)):
        return None, "snap"
    nodes = node_ids[np.r_[True, np.diff(node_ids) != 0]]
    raw = points[np.r_[True, np.diff(node_ids) != 0]]
    if len(nodes) < 2:
        return None, "collapsed"
    output = [int(nodes[0])]
    for left, right, p_left, p_right in zip(nodes[:-1], nodes[1:], raw[:-1], raw[1:]):
        direct = meters(p_left, p_right)
        if direct <= 1.0 or direct > segment_m:
            return None, "segment_gate"
        route, cost = astar(graph, coords, int(left), int(right), max_visits)
        if route is None:
            return None, "no_route"
        # Graph weights are degree-scale coordinates. Convert for the stretch gate.
        route_m = sum(meters(coords[a], coords[b]) for a, b in zip(route[:-1], route[1:]))
        if route_m / direct > stretch:
            return None, "stretch_gate"
        output.extend(route[1:])
    return output, "accepted"


def jsd(real: Counter, synthetic: Counter) -> float | None:
    keys = sorted(set(real) | set(synthetic))
    if not keys or not synthetic:
        return None
    left = np.array([real[key] for key in keys], dtype=float)
    right = np.array([synthetic[key] for key in keys], dtype=float)
    left /= left.sum()
    right /= right.sum()
    return float(jensenshannon(left, right, base=2.0) ** 2)


def trajectory_statistics(real: list[np.ndarray], synthetic: list[np.ndarray], bbox: tuple[float, float, float, float]):
    """Three manuscript statistics on completed outputs under one fixed adapter.

    The reference is the real corpus passed through the same public completion
    operator.  Completion yield is reported separately and is never hidden by
    this conditional comparison.
    """
    import evaluate_kdd_revised_statistics as stats

    if not real or not synthetic:
        return {"grid_density_jsd": None, "trip_error": None, "path_length_jsd": None}
    lengths = np.asarray([
        stats.segment_lengths_km(traj).sum()
        for traj in [*real, *synthetic] if len(traj) >= 2
    ])
    edges = np.linspace(0, max(float(np.quantile(lengths, 0.99)), 1e-6), 33)
    return {
        "grid_density_jsd": stats.jsd(
            stats.grid_density(real, bbox, 64), stats.grid_density(synthetic, bbox, 64)
        ),
        "trip_error": 0.5 * (
            stats.jsd(stats.endpoint_density(real, bbox, 32, False), stats.endpoint_density(synthetic, bbox, 32, False))
            + stats.jsd(stats.endpoint_density(real, bbox, 32, True), stats.endpoint_density(synthetic, bbox, 32, True))
        ),
        "path_length_jsd": stats.jsd(stats.length_hist(real, edges), stats.length_hist(synthetic, edges)),
    }


def run_method(name, trajectories, tree, coords, graph, args, option_context=None):
    started = time.perf_counter()
    edge_counts: Counter = Counter()
    reasons: Counter = Counter()
    accepted = 0
    trips = []
    completed_coordinates: list[np.ndarray] = []
    for index, traj in enumerate(trajectories):
        route, reason = complete_one(
            traj, tree, coords, graph, anchors=args.anchors, snap_m=args.snap_m,
            segment_m=args.segment_m, stretch=args.stretch, max_visits=args.max_visits,
        )
        reasons[reason] += 1
        if route is None:
            trips.append([])
            continue
        accepted += 1
        completed_coordinates.append(np.asarray(coords[route], dtype=float))
        edge_counts.update(zip(route[:-1], route[1:]))
        trips.append([
            (option_context[pair], pair)
            for pair in zip(route[:-1], route[1:])
            if option_context is not None and pair in option_context
        ])
        if (index + 1) % 50 == 0:
            print(f"[{name}] {index + 1}/{len(trajectories)} accepted={accepted}", flush=True)
    return {
        "release_count": len(trajectories),
        "accepted_count": accepted,
        "invalid_mass": 1.0 - accepted / max(len(trajectories), 1),
        "edge_count": int(sum(edge_counts.values())),
        "reasons": dict(reasons),
        "elapsed_sec": time.perf_counter() - started,
    }, edge_counts, trips, completed_coordinates


def real_choice_reference(real_match_cache: Path, network: Path, config: dict, regions: int):
    """Frozen real-only reference.  Synthetic releases never enter FMM here."""
    from evaluate_road_choice import _edge_endpoint_regions
    with gzip.open(real_match_cache, "rb") as handle:
        records = pickle.load(handle)
    _, edge_nodes, carrier_regions, _ = _edge_endpoint_regions(network, config, regions)
    carrier_ids = {index: pair for index, pair in enumerate(carrier_regions)}
    carrier_edge_regions = {index: carrier_regions[pair] for index, pair in carrier_ids.items()}
    options, _ = public_crossing_options(carrier_edge_regions, carrier_ids)
    option_context = {
        pair: context for context, pairs in options.items() for pair in pairs
    }
    edge_context = {
        edge_id: option_context[pair] for edge_id, pair in edge_nodes.items() if pair in option_context
    }
    trips = matched_choice_trips(records, edge_context, edge_nodes)
    split = int(len(trips) * 0.8)
    return options, option_context, trips[split:]


def penalized_choice_metrics(trips, real_test, public_options, coverage: float):
    """Conditional road-choice utility plus an explicit completed-route penalty."""
    fidelity = road_choice_fidelity(trips, real_test, public_options)
    scores = next_road_trip_scores(trips, real_test, public_options, mixture_weight=0.5)
    conditional_acc = float(np.mean(scores["next_road_accuracy"]))
    conditional_nll = float(np.mean(scores["next_road_nll"]))
    return {
        "route_evidence_yield": coverage,
        "invalid_rate": 1.0 - coverage,
        "rc_cpc_conditional": float(fidelity["road_choice_cpc"]),
        "rc_ndcg_conditional": float(fidelity["road_choice_ndcg"]),
        "next_road_accuracy_conditional": conditional_acc,
        "next_road_nll_conditional": conditional_nll,
        "rc_cpc_penalized": float(fidelity["road_choice_cpc"]) * coverage,
        "rc_ndcg_penalized": float(fidelity["road_choice_ndcg"]) * coverage,
        "next_road_accuracy_penalized": conditional_acc * coverage,
        "next_road_nll_penalized": None if coverage == 0 else conditional_nll - math.log(coverage),
        "choice_contributing_records": int(fidelity["road_choice_synthetic_trips"]),
        "choice_events": int(fidelity["road_choice_synthetic_events"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/datasets.json")
    parser.add_argument("--dataset", default="geolife")
    parser.add_argument("--real", default="outputs/kdd_revised/split/real_full_frozen.pkl")
    parser.add_argument("--network", default="results/road_route_metrics/network/geolife/network.shp")
    parser.add_argument("--real-match-cache", default="results/road_choice_metrics/geolife_real_stmatch_v1/matched_paths/Real.pkl.gz")
    parser.add_argument("--quotient-regions", type=int, default=384)
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--snap-m", type=float, default=200.0)
    parser.add_argument("--segment-m", type=float, default=2000.0)
    parser.add_argument("--stretch", type=float, default=3.0)
    parser.add_argument("--max-visits", type=int, default=30000)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    config = dataset_config(args.dataset, args.dataset_config)
    bbox = tuple(map(float, config["bbox"]))
    ways = filter_osm_ways_by_bbox(load_osm_ways(str(public_path(config["osm_cache"]))), bbox)
    ways = filter_osm_ways_by_highway(ways, list(config.get("osm_highway_classes", [])))
    final = ROOT / "src" / "mtr" / "DP_GSRT" / "final"
    sys.path.insert(0, str(final))
    import route_structure_potential_experiment as route
    coords, graph = route.prepare_graph([], bbox=bbox, osm_ways=ways, raw_graph=False)
    coords = np.asarray(coords, dtype=float)
    tree = cKDTree(coords)
    real = load_trajectories(str(public_path(args.real)), limit=args.limit)
    methods = {spec.split("=", 1)[0]: load_trajectories(str(public_path(spec.split("=", 1)[1])), limit=args.limit) for spec in args.method}
    public_options, option_context, real_choice_test = real_choice_reference(
        public_path(args.real_match_cache), public_path(args.network), config, args.quotient_regions
    )
    real_summary, real_edges, _, real_completed = run_method("Real", real, tree, coords, graph, args, option_context)
    result = {"protocol": {"adapter": "public_snap_and_connect", "native_metric": False,
               "no_fmm": True, "invalid_mass_retained": True, "limit": args.limit,
               "anchors": args.anchors, "snap_m": args.snap_m, "segment_m": args.segment_m,
               "stretch": args.stretch, "max_visits": args.max_visits,
               "statistical_reference": "real_completed_by_the_same_fixed_public_adapter",
               "statistics_condition_on_completed_records": True}, "Real": real_summary}
    completed_artifacts = {"Real": real_completed}
    for name, trajectories in methods.items():
        summary, edges, trips, completed = run_method(name, trajectories, tree, coords, graph, args, option_context)
        summary["edge_jsd_to_real_completed"] = jsd(real_edges, edges)
        summary["road_choice_penalized"] = penalized_choice_metrics(
            trips, real_choice_test, public_options, 1.0 - summary["invalid_mass"]
        )
        summary["completed_output_statistics"] = trajectory_statistics(real_completed, completed, bbox)
        result[name] = summary
        completed_artifacts[name] = completed
    out = public_path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "snap_connect_diagnostic.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "completed_trajectories.pkl").open("wb") as handle:
        pickle.dump(completed_artifacts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (out / "README.md").write_text(
        "# Snap-and-Connect diagnostic\n\nThis is a fixed public adapter, not a native road-release metric. "
        "Each failed release record contributes to `invalid_mass`; edge JSD is reported only alongside that mass. "
        "`completed_output_statistics` compares completed synthetic paths with real paths completed by the same public operator; "
        "their conditional reference is accompanied by `route_evidence_yield` / `invalid_mass`. "
        "No synthetic trajectory receives FMM or HMM map matching.\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
