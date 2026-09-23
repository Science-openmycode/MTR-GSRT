"""Local privacy gate for the two-level semi-Markov route query."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ARA = Path(__file__).resolve().parents[1]
for path in [ARA / "dp_reward_exploration", ARA / "public_release", ARA / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dp_graph_flow_routing_probe as flow  # noqa: E402
import graph_voronoi_doptimal_support_probe as support  # noqa: E402
import route_structure_potential_experiment as route  # noqa: E402
from public_utils import load_osm_ways  # noqa: E402
from two_level_semimarkov_query import QueryContext, WEIGHTS, contribution_l1, trajectory_blocks  # noqa: E402


DEFAULT_OUT = ARA / "dp_reward_exploration" / "results" / "two_level_semimarkov_privacy_audit.json"


def graph_digest(coords: np.ndarray, graph: dict) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(coords, dtype=np.float64).tobytes())
    for source in sorted(graph):
        digest.update(np.asarray([int(source)], dtype=np.int64).tobytes())
        for target, weight in sorted(graph[source], key=lambda item: (int(item[0]), float(item[1]))):
            digest.update(np.asarray([int(target)], dtype=np.int64).tobytes())
            digest.update(np.asarray([float(weight)], dtype=np.float64).tobytes())
    return digest.hexdigest()


def graph_voronoi_labels(graph: dict, centers: list[int], node_count: int) -> np.ndarray:
    """Connected public graph-Voronoi cells from multi-source Dijkstra."""
    labels = np.full(int(node_count), -1, dtype=int)
    distances = np.full(int(node_count), np.inf, dtype=float)
    queue = []
    for label, node in enumerate(centers):
        node = int(node)
        labels[node] = int(label)
        distances[node] = 0.0
        heapq.heappush(queue, (0.0, int(label), node))
    while queue:
        distance, label, node = heapq.heappop(queue)
        if distance != distances[node] or label != labels[node]:
            continue
        for neighbour, weight in graph.get(node, []):
            neighbour = int(neighbour)
            candidate = distance + float(weight)
            if candidate < distances[neighbour] - 1e-15 or (
                abs(candidate - distances[neighbour]) <= 1e-15 and int(label) < int(labels[neighbour])
            ):
                distances[neighbour] = candidate
                labels[neighbour] = int(label)
                heapq.heappush(queue, (candidate, int(label), neighbour))
    if np.any(labels < 0):
        raise RuntimeError("Public road graph contains nodes unreachable from all Voronoi centers")
    return labels


def build_context(coords: np.ndarray, graph: dict, coarse_regions: int, fine_regions: int, macros: int, phases: int):
    coarse_nodes = support.farthest_point_landmarks(coords, int(coarse_regions))
    fine_nodes = support.farthest_point_landmarks(coords, int(fine_regions))
    coarse_coords = coords[np.asarray(coarse_nodes, dtype=int)]
    fine_coords = coords[np.asarray(fine_nodes, dtype=int)]
    coarse_tree, fine_tree = cKDTree(coarse_coords), cKDTree(fine_coords)
    node_coarse = graph_voronoi_labels(graph, coarse_nodes, len(coords))
    node_fine = graph_voronoi_labels(graph, fine_nodes, len(coords))
    coarse_edges, coarse_edge_index, _, _, coarse_adjacency = flow.region_graph_and_prior(
        graph, coords, node_coarse, int(coarse_regions)
    )
    fine_edges, fine_edge_index, fine_public_mass, _, fine_adjacency = flow.region_graph_and_prior(
        graph, coords, node_fine, int(fine_regions)
    )
    fine_importance = 1.0 / np.sqrt(np.maximum(np.asarray(fine_public_mass, dtype=float), 1e-15))
    fine_importance /= max(float(np.median(fine_importance)), 1e-15)
    fine_importance = np.clip(fine_importance, 0.25, 4.0)
    road_edge_midpoints = []
    seen_edges = set()
    for source, neighbours in graph.items():
        for target, _ in neighbours:
            edge = tuple(sorted((int(source), int(target))))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            road_edge_midpoints.append(0.5 * (coords[edge[0]] + coords[edge[1]]))
    road_edge_tree = cKDTree(np.asarray(road_edge_midpoints, dtype=float))
    macro_nodes = support.farthest_point_landmarks(coarse_coords, int(macros))
    macro_tree = cKDTree(coarse_coords[np.asarray(macro_nodes, dtype=int)])
    coarse_to_macro = np.asarray(macro_tree.query(coarse_coords)[1], dtype=int)
    context = QueryContext(
        road_tree=cKDTree(coords),
        road_edge_tree=road_edge_tree,
        node_coarse=node_coarse,
        node_fine=node_fine,
        coarse_tree=coarse_tree,
        fine_tree=fine_tree,
        coarse_adjacency=coarse_adjacency,
        fine_adjacency=fine_adjacency,
        coarse_edge_index=coarse_edge_index,
        fine_edge_index=fine_edge_index,
        coarse_to_macro=coarse_to_macro,
        fine_importance=fine_importance,
        phases=int(phases),
        coarse_regions=int(coarse_regions),
        fine_regions=int(fine_regions),
        macro_regions=int(macros),
    )
    public_diag = {
        "road_nodes": int(len(coords)),
        "coarse_regions": int(coarse_regions),
        "fine_regions": int(fine_regions),
        "coarse_directed_edges": int(len(coarse_edges)),
        "fine_directed_edges": int(len(fine_edges)),
        "macro_regions": int(macros),
        "phases": int(phases),
    }
    return context, public_diag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--coarse-regions", type=int, default=24)
    parser.add_argument("--fine-regions", type=int, default=96)
    parser.add_argument("--macros", type=int, default=4)
    parser.add_argument("--phases", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = support.load_trajectories(support.DEFAULT_REAL)
    osm = load_osm_ways("ara_final/evidence/tables/osm_cache_beijing.pkl")
    coords_a, graph_a = route.prepare_graph([], bbox=support.BBOX, osm_ways=osm, raw_graph=False)
    coords_b, graph_b = route.prepare_graph(trajectories[: min(100, len(trajectories))], bbox=support.BBOX, osm_ways=osm, raw_graph=False)
    digest_a, digest_b = graph_digest(coords_a, graph_a), graph_digest(coords_b, graph_b)
    context, public_diag = build_context(
        coords_a, graph_a, args.coarse_regions, args.fine_regions, args.macros, args.phases
    )

    maximum = 0.0
    minimum = float("inf")
    block_max = {name: 0.0 for name in WEIGHTS}
    violations = []
    for index, trajectory in enumerate(trajectories[: int(args.limit)]):
        blocks = trajectory_blocks(trajectory, context)
        total, norms = contribution_l1(blocks)
        maximum = max(maximum, total)
        minimum = min(minimum, total)
        for name, value in norms.items():
            block_max[name] = max(block_max[name], float(value))
        if total > 1.0 + 1e-9 or any(value > 1.0 + 1e-9 for value in norms.values()):
            violations.append({"index": int(index), "weighted_l1": float(total), "block_l1": norms})
        if (index + 1) % 200 == 0:
            print(f"[privacy-audit] checked {index + 1}/{min(len(trajectories), int(args.limit))}", flush=True)

    malformed = [np.empty((0, 2)), np.asarray([[39.9, 116.4]]), np.asarray([[np.nan, 116.4], [39.9, 116.5]])]
    malformed_norms = [contribution_l1(trajectory_blocks(item, context))[0] for item in malformed]
    checks = {
        "public_graph_independent_of_private_argument": bool(digest_a == digest_b),
        "weights_nonnegative": bool(all(value >= 0.0 for value in WEIGHTS.values())),
        "weights_sum_to_one": bool(abs(sum(WEIGHTS.values()) - 1.0) <= 1e-12),
        "all_sampled_block_norms_at_most_one": bool(all(value <= 1.0 + 1e-9 for value in block_max.values())),
        "all_sampled_joint_norms_at_most_one": bool(maximum <= 1.0 + 1e-9 and not violations),
        "malformed_records_contribute_zero": bool(all(abs(value) <= 1e-15 for value in malformed_norms)),
    }
    report = {
        "gate": "PASS" if all(checks.values()) else "FAIL",
        "scope": "query sensitivity and public graph boundary; not a finite-precision noise-sampler certification",
        "adjacency": "one complete trajectory add/remove against the null record in a fixed public capacity database",
        "protected_unit": "one complete trajectory",
        "theoretical_global_l1_sensitivity": 1.0,
        "weights": WEIGHTS,
        "checks": checks,
        "sampled_records": int(min(len(trajectories), int(args.limit))),
        "sampled_joint_l1_min": float(minimum),
        "sampled_joint_l1_max": float(maximum),
        "sampled_block_l1_max": block_max,
        "malformed_record_l1": malformed_norms,
        "public_graph_digest": digest_a,
        "public_graph": public_diag,
        "violations": violations[:20],
        "certified_release": False,
        "remaining_gate": "independent proof review and audited finite-precision DP noise sampler",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["gate"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
