"""Candidate-support upper-bound probe for the next DP-GSRT design.

This script is exploration-only.  It does not release a private model.  Raw
trajectories are used only to measure whether a public candidate family could
represent the observed route after its endpoints have been quantized.

The probe separates two bottlenecks:

1. endpoint support: noisy top-K private anchors versus public road-cover
   landmarks selected by farthest-point traversal;
2. route support: the current repeated-penalty family versus a union of
   near-shortest sidetrack/via families selected by a log-det objective.

The log-det selection uses public candidate features only.  Its feature matrix
contains path incidence, the fundamental-cycle residual relative to the
shortest path, and coarse directed cut crossings.  The objective

    log det(I + alpha * sum_{p in S} x_p x_p^T)

is monotone submodular, so greedy selection has the standard (1-1/e)
approximation guarantee for a fixed public pool and cardinality constraint.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ARA = Path(__file__).resolve().parents[1]
PUBLIC_RELEASE = ARA / "public_release"
EXPERIMENTS = PUBLIC_RELEASE / "src" / "experiments"
SHARED_GRAPH = PUBLIC_RELEASE / "src" / "mtr" / "shared_graph"
DP_GSRT = PUBLIC_RELEASE / "src" / "mtr" / "DP_GSRT"
SCRIPTS = ARA / "scripts"
for path in [PUBLIC_RELEASE, EXPERIMENTS, SHARED_GRAPH, DP_GSRT, SCRIPTS]:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from public_utils import load_osm_ways  # noqa: E402
import cross_dataset_ara_mode_benchmark as cdb  # noqa: E402
import route_structure_potential_experiment as rse  # noqa: E402
import reward_candidate_probe as probe  # noqa: E402
import route_choice_metric_probe as rcm  # noqa: E402


BBOX = (39.75, 40.15, 116.10, 116.65)
DEFAULT_REAL = ARA.parent / "ARA_codex_help" / "data" / "geolife_full_17123.pkl"
DEFAULT_OUT = ARA / "dp_reward_exploration" / "results" / "graph_voronoi_doptimal_support_probe"


def validate_raw_record_container(raw, capacity: int):
    """Enforce the public slot bound before data-dependent cleaning removes records."""
    if not isinstance(raw, (list, tuple)) or len(raw) > int(capacity):
        raise RuntimeError("Raw private input exceeds the fixed public slot capacity")
    return raw


def normalize_trajectory_records(raw) -> list[np.ndarray]:
    """Frozen record normalization shared by every production DP query."""
    out = []
    for trajectory in raw:
        try:
            arr = np.asarray(trajectory, dtype=float)
        except (TypeError, ValueError, OverflowError):
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        arr = arr[np.isfinite(arr[:, :2]).all(axis=1), :2]
        if len(arr) < 2:
            continue
        if np.any(np.abs(arr[:, 0]) > 90.0) or np.any(np.abs(arr[:, 1]) > 180.0):
            continue
        latitude = float(np.mean(arr[:, 0]))
        scale = np.asarray([111.32, 111.32 * np.cos(np.deg2rad(latitude))])
        lengths = np.linalg.norm(np.diff(arr, axis=0) * scale, axis=1)
        if np.all(np.isfinite(lengths)) and np.any(lengths > 1e-12):
            out.append(arr)
    return out


def load_trajectories(path: Path) -> list[np.ndarray]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    return normalize_trajectory_records(raw)


def standardized_xy(coords: np.ndarray) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = BBOX
    span = np.asarray([lat_max - lat_min, lon_max - lon_min], dtype=float)
    return (np.asarray(coords, dtype=float) - np.asarray([lat_min, lon_min])) / span


def farthest_point_landmarks(coords: np.ndarray, count: int) -> list[int]:
    """Public Euclidean k-center traversal on road nodes.

    Greedy farthest-point traversal is a 2-approximation for metric k-center.
    The deterministic first landmark is the road node nearest the public bbox
    centre, so the construction is independent of private trajectories.
    """
    if count <= 0:
        return []
    points = standardized_xy(coords)
    centre = np.asarray([0.5, 0.5], dtype=float)
    first = int(np.argmin(np.sum((points - centre) ** 2, axis=1)))
    chosen = [first]
    min_sq = np.sum((points - points[first]) ** 2, axis=1)
    min_sq[first] = -1.0
    while len(chosen) < min(int(count), len(points)):
        nxt = int(np.argmax(min_sq))
        chosen.append(nxt)
        dist_sq = np.sum((points - points[nxt]) ** 2, axis=1)
        min_sq = np.minimum(min_sq, dist_sq)
        min_sq[np.asarray(chosen, dtype=int)] = -1.0
    return chosen


def trajectory_region(arr: np.ndarray) -> str:
    norm = standardized_xy(arr)
    cheb = np.max(np.abs(norm - 0.5), axis=1)
    if np.any(norm[:, 0] < 0.16):
        return "south"
    if np.any(norm[:, 1] < 0.16):
        return "west"
    if np.any(cheb >= 0.40):
        return "outer"
    if np.any(cheb >= 0.32):
        return "peripheral"
    return "interior"


def stratified_indices(real: list[np.ndarray], n_eval: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {name: [] for name in ["south", "west", "outer", "peripheral", "interior"]}
    for idx, arr in enumerate(real):
        groups[trajectory_region(arr)].append(idx)
    order = ["south", "west", "outer", "peripheral", "interior"]
    quotas = {name: n_eval // len(order) for name in order}
    for name in order[: n_eval % len(order)]:
        quotas[name] += 1
    chosen: list[int] = []
    deficits = 0
    for name in order:
        values = np.asarray(groups[name], dtype=int)
        take = min(quotas[name], len(values))
        if take:
            chosen.extend(int(x) for x in rng.choice(values, size=take, replace=False))
        deficits += quotas[name] - take
    if deficits:
        remaining = np.asarray(sorted(set(range(len(real))) - set(chosen)), dtype=int)
        chosen.extend(int(x) for x in rng.choice(remaining, size=min(deficits, len(remaining)), replace=False))
    return chosen[:n_eval]


def current_dp_anchor_nodes(
    real: list[np.ndarray],
    road_coords: np.ndarray,
    *,
    count: int,
    epsilon: float,
    seed: int,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    cdb.K = int(count)
    rse.K = int(count)
    normalized, mn, span = rse.normalize_with_public_bbox(real, BBOX)
    rng = np.random.default_rng(seed)
    centres_norm = cdb.dp_anchors(normalized, float(epsilon), rng)
    centres = centres_norm * span + mn
    tree = cKDTree(road_coords)
    nodes = [int(tree.query(point)[1]) for point in centres]
    return nodes, centres, mn, span


def append_unique(dst: list[tuple[str, list[int]]], rows: list[tuple[str, list[int]]], prefix: str) -> None:
    seen = {tuple(int(x) for x in nodes) for _, nodes in dst}
    for label, nodes in rows:
        key = tuple(int(x) for x in nodes)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        dst.append((f"{prefix}:{label}", list(key)))


def diversified_pool(
    graph: dict,
    coords: np.ndarray,
    src: int,
    dst: int,
    tree: cKDTree,
    reverse_graph: dict,
    public_landmarks: list[int],
    *,
    pool_per_family: int,
    max_stretch: float,
    spur_trials: int,
    include_heavy: bool,
) -> list[tuple[str, list[int]]]:
    out: list[tuple[str, list[int]]] = []
    families = [
        ("reroute", None),
        ("hybrid_sidetrack", None),
        ("hybrid", None),
        ("hybrid_anchor", public_landmarks),
    ]
    if include_heavy:
        families.extend([("hybrid_replacement", None), ("hybrid_bridge", None)])
    for mode, anchors in families:
        rows = probe.generate_public_candidates(
            graph,
            coords,
            int(src),
            int(dst),
            max_candidates=int(pool_per_family),
            mode=mode,
            tree=tree,
            reverse_graph=reverse_graph,
            anchor_nodes=anchors,
            max_stretch=float(max_stretch),
            spur_trials=int(spur_trials),
        )
        append_unique(out, rows, mode)
    return out


def stable_edge_bin(u: int, v: int, dim: int, *, directed: bool) -> int:
    if not directed and u > v:
        u, v = v, u
    value = (int(u) * 1_000_003) ^ (int(v) * 97_409) ^ (0x9E3779B9 if directed else 0x85EBCA6B)
    return int(value % dim)


def path_feature(
    nodes: list[int],
    base_nodes: list[int],
    coords: np.ndarray,
    *,
    edge_dim: int,
    cut_grid: int,
    base_length: float,
    stretch_decay: float,
) -> np.ndarray:
    path_edges = {(int(a), int(b)) for a, b in zip(nodes[:-1], nodes[1:])}
    path_undirected = {tuple(sorted((a, b))) for a, b in path_edges}
    base_undirected = {tuple(sorted((int(a), int(b)))) for a, b in zip(base_nodes[:-1], base_nodes[1:])}
    cycle_edges = path_undirected.symmetric_difference(base_undirected)
    edge_vec = np.zeros(edge_dim, dtype=float)
    cycle_vec = np.zeros(edge_dim, dtype=float)
    for u, v in path_edges:
        edge_vec[stable_edge_bin(u, v, edge_dim, directed=True)] += 1.0
    for u, v in cycle_edges:
        cycle_vec[stable_edge_bin(u, v, edge_dim, directed=False)] += 1.0

    norm = standardized_xy(coords[np.asarray(nodes, dtype=int)])
    cells = np.clip((norm * cut_grid).astype(int), 0, cut_grid - 1)
    cut_dim = cut_grid * cut_grid
    cut_vec = np.zeros(cut_dim, dtype=float)
    for a, b in zip(cells[:-1], cells[1:]):
        ai = int(a[0]) * cut_grid + int(a[1])
        bi = int(b[0]) * cut_grid + int(b[1])
        if ai != bi:
            cut_vec[(ai * 131 + bi * 17) % cut_dim] += 1.0

    for vec in [edge_vec, cycle_vec, cut_vec]:
        length = float(np.linalg.norm(vec))
        if length > 0:
            vec /= length
    feature = np.concatenate([0.55 * edge_vec, 1.00 * cycle_vec, 0.70 * cut_vec])
    length = probe.path_length(coords, nodes)
    stretch = length / max(float(base_length), 1e-12)
    feature *= math.exp(-float(stretch_decay) * max(stretch - 1.0, 0.0))
    return feature


def greedy_logdet_select(
    candidates: list[tuple[str, list[int]]],
    coords: np.ndarray,
    k: int,
    *,
    edge_dim: int,
    cut_grid: int,
    alpha: float,
    stretch_decay: float,
) -> tuple[list[tuple[str, list[int]]], dict]:
    if not candidates or k <= 0:
        return [], {"pool_size": len(candidates), "selected": 0, "objective": 0.0}
    base_idx = next((i for i, (label, _) in enumerate(candidates) if "public_base" in label), 0)
    base = candidates[base_idx][1]
    base_length = probe.path_length(coords, base)
    directed_edges = []
    undirected_edges = []
    base_undirected = {tuple(sorted((int(a), int(b)))) for a, b in zip(base[:-1], base[1:])}
    for _, nodes in candidates:
        directed_edges.append({(int(a), int(b)) for a, b in zip(nodes[:-1], nodes[1:])})
        undirected_edges.append({tuple(sorted((int(a), int(b)))) for a, b in zip(nodes[:-1], nodes[1:])})
    edge_universe = {edge for edges in directed_edges for edge in edges}
    cycle_universe = {edge for edges in undirected_edges for edge in edges.symmetric_difference(base_undirected)}
    edge_index = {edge: idx for idx, edge in enumerate(sorted(edge_universe))}
    cycle_index = {edge: idx for idx, edge in enumerate(sorted(cycle_universe))}
    cut_dim = cut_grid * cut_grid
    features = np.zeros((len(candidates), len(edge_index) + len(cycle_index) + cut_dim), dtype=float)
    for row, ((_, nodes), path_edges, path_undirected) in enumerate(zip(candidates, directed_edges, undirected_edges)):
        edge_offset = 0
        cycle_offset = len(edge_index)
        cut_offset = cycle_offset + len(cycle_index)
        for edge in path_edges:
            features[row, edge_offset + edge_index[edge]] = 1.0
        cycle_edges = path_undirected.symmetric_difference(base_undirected)
        for edge in cycle_edges:
            features[row, cycle_offset + cycle_index[edge]] = 1.0
        norm = standardized_xy(coords[np.asarray(nodes, dtype=int)])
        cells = np.clip((norm * cut_grid).astype(int), 0, cut_grid - 1)
        for a, b in zip(cells[:-1], cells[1:]):
            ai = int(a[0]) * cut_grid + int(a[1])
            bi = int(b[0]) * cut_grid + int(b[1])
            if ai != bi:
                features[row, cut_offset + ((ai * 131 + bi * 17) % cut_dim)] += 1.0
        edge_slice = features[row, edge_offset:cycle_offset]
        cycle_slice = features[row, cycle_offset:cut_offset]
        cut_slice = features[row, cut_offset:]
        for vector, weight in [(edge_slice, 0.45), (cycle_slice, 1.25), (cut_slice, 0.80)]:
            vector_norm = float(np.linalg.norm(vector))
            if vector_norm > 0:
                vector *= weight / vector_norm
        stretch = probe.path_length(coords, nodes) / max(base_length, 1e-12)
        features[row] *= math.exp(-float(stretch_decay) * max(stretch - 1.0, 0.0))

    # The exact Gram matrix avoids hash collisions that can make visibly
    # overlapping roads appear orthogonal in the diversity objective.
    gram = features @ features.T
    selected = [base_idx]
    remaining = set(range(len(candidates))) - {base_idx}

    def objective(indices: list[int]) -> float:
        sub = gram[np.ix_(indices, indices)]
        sign, value = np.linalg.slogdet(np.eye(len(indices)) + float(alpha) * sub)
        return float(value) if sign > 0 else -math.inf

    current_objective = objective(selected)
    while remaining and len(selected) < min(int(k), len(candidates)):
        best_idx = None
        best_gain = -math.inf
        for index in sorted(remaining):
            gain = objective(selected + [index]) - current_objective
            if gain > best_gain:
                best_gain = gain
                best_idx = index
        if best_idx is None:
            break
        remaining.remove(best_idx)
        selected.append(best_idx)
        current_objective += best_gain
    return [candidates[i] for i in selected], {
        "pool_size": int(len(candidates)),
        "selected": int(len(selected)),
        "objective": float(current_objective),
        "exact_edge_features": True,
    }


def grid_cells(arr: np.ndarray, grid: int) -> set[int]:
    norm = standardized_xy(np.asarray(arr, dtype=float))
    cells = np.clip((norm * grid).astype(int), 0, grid - 1)
    return {int(r) * grid + int(c) for r, c in cells}


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def route_distance(real: np.ndarray, candidate: np.ndarray, mn: np.ndarray, span: np.ndarray, grid: int) -> float:
    real_flow, real_trans = rcm.path_cell_features(real, mn, span, grid)
    cand_flow, cand_trans = rcm.path_cell_features(candidate, mn, span, grid)
    d_trans = rcm.total_variation(real_trans, cand_trans)
    d_flow = rcm.total_variation(real_flow, cand_flow)
    real_len = max(rcm.polyline_len(real), 1e-12)
    cand_len = max(rcm.polyline_len(candidate), 1e-12)
    d_len = min(abs(math.log(cand_len / real_len)), 3.0)
    return float(0.65 * d_trans + 0.25 * d_flow + 0.10 * d_len)


def family_metrics(
    real: np.ndarray,
    candidates: list[tuple[str, list[int]]],
    coords: np.ndarray,
    mn: np.ndarray,
    span: np.ndarray,
    *,
    eval_grid: int,
) -> dict:
    if not candidates:
        return {
            "candidate_count": 0,
            "best_route_distance": None,
            "best_cell_jaccard": None,
            "mean_pairwise_jaccard": None,
            "union_cell_recall": None,
        }
    real_cells = grid_cells(real, eval_grid)
    candidate_cells = []
    distances = []
    similarities = []
    for _, nodes in candidates:
        arr = coords[np.asarray(nodes, dtype=int)]
        cells = grid_cells(arr, eval_grid)
        candidate_cells.append(cells)
        distances.append(route_distance(real, arr, mn, span, max(6, eval_grid // 4)))
        similarities.append(jaccard(real_cells, cells))
    pairwise = []
    for i in range(len(candidate_cells)):
        for j in range(i + 1, len(candidate_cells)):
            pairwise.append(jaccard(candidate_cells[i], candidate_cells[j]))
    union = set().union(*candidate_cells)
    return {
        "candidate_count": int(len(candidates)),
        "best_route_distance": float(min(distances)),
        "best_cell_jaccard": float(max(similarities)),
        "mean_pairwise_jaccard": float(np.mean(pairwise)) if pairwise else 1.0,
        "union_cell_recall": float(len(real_cells & union) / max(len(real_cells), 1)),
    }


def endpoint_error_km(real: np.ndarray, src: int, dst: int, coords: np.ndarray) -> float:
    errors = []
    for point, node in [(real[0], src), (real[-1], dst)]:
        lat_scale = 111.0
        lon_scale = 111.0 * math.cos(math.radians(float(point[0])))
        delta = np.asarray(point) - coords[int(node)]
        errors.append(math.hypot(float(delta[0]) * lat_scale, float(delta[1]) * lon_scale))
    return float(np.mean(errors))


def aggregate(rows: list[dict], family: str) -> dict:
    values: dict[str, list[float]] = {}
    for row in rows:
        metrics = row["families"][family]
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                values.setdefault(key, []).append(float(value))
    return {
        key: {
            "mean": float(np.mean(items)),
            "median": float(np.median(items)),
            "n": int(len(items)),
        }
        for key, items in values.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-eval", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--anchor-seed", type=int, default=20260709)
    parser.add_argument("--current-anchors", type=int, default=40)
    parser.add_argument("--public-landmarks", type=int, default=128)
    parser.add_argument("--candidate-k", type=int, default=24)
    parser.add_argument("--pool-per-family", type=int, default=20)
    parser.add_argument("--max-stretch", type=float, default=2.35)
    parser.add_argument("--spur-trials", type=int, default=24)
    parser.add_argument("--eval-grid", type=int, default=32)
    parser.add_argument("--include-heavy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    real = load_trajectories(args.real)
    osm_ways = load_osm_ways("ara_final/evidence/tables/osm_cache_beijing.pkl")
    road_coords, graph = rse.prepare_graph(real, bbox=BBOX, osm_ways=osm_ways, raw_graph=False)
    tree = cKDTree(road_coords)
    reverse_graph = probe.reverse_adjacency(graph)
    dp_anchor_nodes, _, mn, span = current_dp_anchor_nodes(
        real,
        road_coords,
        count=args.current_anchors,
        epsilon=0.20,
        seed=args.anchor_seed,
    )
    public_landmarks = farthest_point_landmarks(road_coords, args.public_landmarks)
    dp_anchor_tree = cKDTree(road_coords[np.asarray(dp_anchor_nodes, dtype=int)])
    public_landmark_tree = cKDTree(road_coords[np.asarray(public_landmarks, dtype=int)])
    indices = stratified_indices(real, args.n_eval, args.seed)

    rows = []
    for position, idx in enumerate(indices, start=1):
        trajectory = real[idx]
        _, dp_local = dp_anchor_tree.query(trajectory[[0, -1]])
        dp_src = int(dp_anchor_nodes[int(dp_local[0])])
        dp_dst = int(dp_anchor_nodes[int(dp_local[1])])
        _, public_local = public_landmark_tree.query(trajectory[[0, -1]])
        pub_src = int(public_landmarks[int(public_local[0])])
        pub_dst = int(public_landmarks[int(public_local[1])])

        current_candidates = probe.generate_public_candidates(
            graph,
            road_coords,
            dp_src,
            dp_dst,
            max_candidates=args.candidate_k,
            mode="hybrid_anchor",
            tree=tree,
            reverse_graph=reverse_graph,
            anchor_nodes=dp_anchor_nodes,
            max_stretch=args.max_stretch,
            spur_trials=args.spur_trials,
        ) if dp_src != dp_dst else []

        same_support_pool = diversified_pool(
            graph,
            road_coords,
            dp_src,
            dp_dst,
            tree,
            reverse_graph,
            public_landmarks,
            pool_per_family=args.pool_per_family,
            max_stretch=args.max_stretch,
            spur_trials=args.spur_trials,
            include_heavy=args.include_heavy,
        ) if dp_src != dp_dst else []
        same_support_selected, same_diag = greedy_logdet_select(
            same_support_pool,
            road_coords,
            args.candidate_k,
            edge_dim=192,
            cut_grid=8,
            alpha=2.0,
            stretch_decay=1.4,
        )

        graph_support_pool = diversified_pool(
            graph,
            road_coords,
            pub_src,
            pub_dst,
            tree,
            reverse_graph,
            public_landmarks,
            pool_per_family=args.pool_per_family,
            max_stretch=args.max_stretch,
            spur_trials=args.spur_trials,
            include_heavy=args.include_heavy,
        ) if pub_src != pub_dst else []
        graph_support_selected, graph_diag = greedy_logdet_select(
            graph_support_pool,
            road_coords,
            args.candidate_k,
            edge_dim=192,
            cut_grid=8,
            alpha=2.0,
            stretch_decay=1.4,
        )

        family_rows = {
            "current_anchor_current_pool": family_metrics(
                trajectory, current_candidates, road_coords, mn, span, eval_grid=args.eval_grid
            ),
            "current_anchor_doptimal_pool": family_metrics(
                trajectory, same_support_selected, road_coords, mn, span, eval_grid=args.eval_grid
            ),
            "public_cover_doptimal_pool": family_metrics(
                trajectory, graph_support_selected, road_coords, mn, span, eval_grid=args.eval_grid
            ),
        }
        family_rows["current_anchor_current_pool"]["endpoint_error_km"] = endpoint_error_km(
            trajectory, dp_src, dp_dst, road_coords
        )
        family_rows["current_anchor_doptimal_pool"]["endpoint_error_km"] = endpoint_error_km(
            trajectory, dp_src, dp_dst, road_coords
        )
        family_rows["public_cover_doptimal_pool"]["endpoint_error_km"] = endpoint_error_km(
            trajectory, pub_src, pub_dst, road_coords
        )
        rows.append(
            {
                "trajectory_index": int(idx),
                "region": trajectory_region(trajectory),
                "families": family_rows,
                "logdet": {
                    "current_anchor": same_diag,
                    "public_cover": graph_diag,
                },
            }
        )
        print(
            f"[support-probe] {position}/{len(indices)} region={rows[-1]['region']} "
            f"current={family_rows['current_anchor_current_pool']['best_route_distance']} "
            f"same={family_rows['current_anchor_doptimal_pool']['best_route_distance']} "
            f"cover={family_rows['public_cover_doptimal_pool']['best_route_distance']}",
            flush=True,
        )

    family_names = [
        "current_anchor_current_pool",
        "current_anchor_doptimal_pool",
        "public_cover_doptimal_pool",
    ]
    payload = {
        "experiment": "graph_voronoi_doptimal_candidate_support_upper_bound",
        "research_diagnostic_only": True,
        "privacy_note": (
            "Raw trajectories select oracle-best candidates only for offline evaluation. "
            "No oracle choice or per-record result is part of the proposed DP release."
        ),
        "configuration": {
            "n_real": len(real),
            "n_eval": len(indices),
            "seed": int(args.seed),
            "anchor_seed": int(args.anchor_seed),
            "current_anchors": int(args.current_anchors),
            "public_landmarks": int(args.public_landmarks),
            "candidate_k": int(args.candidate_k),
            "pool_per_family": int(args.pool_per_family),
            "max_stretch": float(args.max_stretch),
            "spur_trials": int(args.spur_trials),
            "eval_grid": int(args.eval_grid),
            "include_heavy": bool(args.include_heavy),
            "road_nodes": int(len(road_coords)),
            "runtime_sec_local_only": float(time.time() - started),
        },
        "aggregate": {name: aggregate(rows, name) for name in family_names},
        "by_region": {
            region: {
                name: aggregate([row for row in rows if row["region"] == region], name)
                for name in family_names
            }
            for region in sorted({row["region"] for row in rows})
        },
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "candidate_support_probe.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(out_path.resolve()), "aggregate": payload["aggregate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
