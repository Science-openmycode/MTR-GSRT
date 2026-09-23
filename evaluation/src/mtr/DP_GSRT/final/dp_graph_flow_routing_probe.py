"""OD/phase-conditioned DP graph-flow routing probe.

The mechanism releases a clipped trajectory contribution over a public road
partition.  Sixty percent of a trajectory's unit mass describes region
occupancy and forty percent describes directed cut crossings, both conditioned
on public OD-direction and route-phase buckets.  The concatenated contribution
has L1 norm at most one, so coordinatewise Laplace noise with scale 1/epsilon
is trajectory-level epsilon-DP under add/remove adjacency.

No private candidate set is constructed.  The noisy flow is projected onto
public probability simplices and converted to positive edge multipliers.  A*
search on the complete public road graph is therefore post-processing of the
DP release.  This script is an exploration diagnostic and uses a fixed seed;
certified releases must use fresh unpredictable randomness and must not publish
the local runtime fields recorded here.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ARA = Path(__file__).resolve().parents[1]
for path in [
    ARA / "dp_reward_exploration",
    ARA / "public_release",
    ARA / "public_release" / "src" / "experiments",
    ARA / "public_release" / "src" / "mtr" / "shared_graph",
    ARA / "public_release" / "src" / "mtr" / "DP_GSRT",
    ARA / "scripts",
]:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from public_utils import load_osm_ways  # noqa: E402
import route_structure_potential_experiment as rse  # noqa: E402
import graph_voronoi_doptimal_support_probe as support  # noqa: E402


BBOX = support.BBOX
DEFAULT_REAL = support.DEFAULT_REAL
DEFAULT_OUT = ARA / "dp_reward_exploration" / "results" / "dp_graph_flow_routing_probe"


def direction_family(start: np.ndarray, end: np.ndarray, sectors: int) -> int:
    delta = support.standardized_xy(np.asarray([start, end], dtype=float))[1] - support.standardized_xy(
        np.asarray([start, end], dtype=float)
    )[0]
    angle = math.atan2(float(delta[0]), float(delta[1]))
    return int(math.floor(((angle + math.pi) / (2.0 * math.pi)) * sectors)) % int(sectors)


def compress(values: np.ndarray) -> list[int]:
    out: list[int] = []
    for value in np.asarray(values, dtype=int).ravel():
        item = int(value)
        if not out or out[-1] != item:
            out.append(item)
    return out


def region_graph_and_prior(
    graph: dict,
    coords: np.ndarray,
    node_region: np.ndarray,
    regions: int,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], np.ndarray, np.ndarray, dict[int, set[int]]]:
    occupancy = np.zeros(int(regions), dtype=float)
    transition_weight: dict[tuple[int, int], float] = {}
    adjacency: dict[int, set[int]] = {region: set() for region in range(int(regions))}
    seen_undirected: set[tuple[int, int]] = set()
    for u, neighbours in graph.items():
        ru = int(node_region[int(u)])
        for v, weight in neighbours:
            v = int(v)
            rv = int(node_region[v])
            edge = tuple(sorted((int(u), v)))
            if edge not in seen_undirected:
                occupancy[ru] += 0.5 * float(weight)
                occupancy[rv] += 0.5 * float(weight)
                seen_undirected.add(edge)
            if ru != rv:
                transition_weight[(ru, rv)] = transition_weight.get((ru, rv), 0.0) + 1.0 / max(float(weight), 1e-12)
                adjacency[ru].add(rv)
    edges = sorted(transition_weight)
    edge_index = {edge: index for index, edge in enumerate(edges)}
    transition = np.asarray([transition_weight[edge] for edge in edges], dtype=float)
    occupancy = np.maximum(occupancy, 1e-12)
    occupancy /= occupancy.sum()
    transition = np.maximum(transition, 1e-12)
    transition /= transition.sum()
    return edges, edge_index, occupancy, transition, adjacency


def region_path(adjacency: dict[int, set[int]], source: int, target: int) -> list[int]:
    source, target = int(source), int(target)
    if source == target:
        return [source]
    queue = deque([source])
    parent = {source: -1}
    while queue:
        node = queue.popleft()
        for neighbour in sorted(adjacency.get(node, ())):
            if neighbour in parent:
                continue
            parent[neighbour] = node
            if neighbour == target:
                path = [target]
                while path[-1] != source:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append(neighbour)
    return [source, target]


def expanded_region_sequence(sequence: list[int], adjacency: dict[int, set[int]]) -> list[int]:
    if not sequence:
        return []
    out = [int(sequence[0])]
    for target in sequence[1:]:
        segment = region_path(adjacency, out[-1], int(target))
        out.extend(segment[1:])
    return compress(np.asarray(out, dtype=int))


def fit_flow_query(
    trajectories: list[np.ndarray],
    region_tree: cKDTree,
    adjacency: dict[int, set[int]],
    edge_index: dict[tuple[int, int], int],
    *,
    sectors: int,
    phases: int,
    regions: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    occupancy = np.zeros((int(sectors), int(phases), int(regions)), dtype=float)
    transitions = np.zeros((int(sectors), int(phases), len(edge_index)), dtype=float)
    maximum_l1 = 0.0
    for index, trajectory in enumerate(trajectories, start=1):
        family = direction_family(trajectory[0], trajectory[-1], sectors)
        raw_sequence = compress(region_tree.query(np.asarray(trajectory)[:, :2])[1])
        sequence = expanded_region_sequence(raw_sequence, adjacency)
        local_occ: dict[tuple[int, int], float] = {}
        local_trans: dict[tuple[int, int], float] = {}
        for position, region in enumerate(sequence):
            phase = min(int(phases) - 1, int(position * int(phases) / max(len(sequence), 1)))
            local_occ[(phase, int(region))] = local_occ.get((phase, int(region)), 0.0) + 1.0
        for position, (a, b) in enumerate(zip(sequence[:-1], sequence[1:])):
            edge = edge_index.get((int(a), int(b)))
            if edge is None:
                continue
            phase = min(int(phases) - 1, int(position * int(phases) / max(len(sequence) - 1, 1)))
            local_trans[(phase, edge)] = local_trans.get((phase, edge), 0.0) + 1.0
        occ_total = max(float(sum(local_occ.values())), 1.0)
        trans_total = max(float(sum(local_trans.values())), 1.0)
        contribution_l1 = 0.0
        for (phase, region), value in local_occ.items():
            mass = 0.60 * value / occ_total
            occupancy[family, phase, region] += mass
            contribution_l1 += mass
        for (phase, edge), value in local_trans.items():
            mass = 0.40 * value / trans_total
            transitions[family, phase, edge] += mass
            contribution_l1 += mass
        maximum_l1 = max(maximum_l1, contribution_l1)
        if index % 3000 == 0:
            print(f"[flow-query] processed {index}/{len(trajectories)}", flush=True)
    return occupancy, transitions, maximum_l1


def project_noisy_flow(
    exact_occupancy: np.ndarray,
    exact_transitions: np.ndarray,
    prior_occupancy: np.ndarray,
    prior_transitions: np.ndarray,
    *,
    epsilon: float | None,
    rng: np.random.Generator,
    pseudocount: float,
) -> tuple[np.ndarray, np.ndarray]:
    if epsilon is None:
        occupancy = exact_occupancy.copy()
        transitions = exact_transitions.copy()
        smoothing = float(pseudocount)
    else:
        scale = 1.0 / max(float(epsilon), 1e-12)
        occupancy = exact_occupancy + rng.laplace(0.0, scale, size=exact_occupancy.shape)
        transitions = exact_transitions + rng.laplace(0.0, scale, size=exact_transitions.shape)
        smoothing = float(pseudocount) / max(float(epsilon), 1e-12)
    occupancy = np.maximum(occupancy, 0.0)
    transitions = np.maximum(transitions, 0.0)
    for family in range(occupancy.shape[0]):
        for phase in range(occupancy.shape[1]):
            occ = occupancy[family, phase] + smoothing * prior_occupancy
            trans = transitions[family, phase] + smoothing * prior_transitions
            occupancy[family, phase] = occ / max(float(occ.sum()), 1e-12)
            transitions[family, phase] = trans / max(float(trans.sum()), 1e-12)
    return occupancy, transitions


def score_tables(
    occupancy: np.ndarray,
    transitions: np.ndarray,
    prior_occupancy: np.ndarray,
    prior_transitions: np.ndarray,
    clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    occ_score = np.log(np.maximum(occupancy, 1e-15) / np.maximum(prior_occupancy[None, None, :], 1e-15))
    trans_score = np.log(np.maximum(transitions, 1e-15) / np.maximum(prior_transitions[None, None, :], 1e-15))
    return np.clip(occ_score, -float(clip), float(clip)), np.clip(trans_score, -float(clip), float(clip))


def route_phase(coords: np.ndarray, source: int, target: int, u: int, v: int, phases: int) -> int:
    start, end = coords[int(source)], coords[int(target)]
    delta = end - start
    denom = max(float(delta @ delta), 1e-12)
    midpoint = 0.5 * (coords[int(u)] + coords[int(v)])
    fraction = float(np.clip(((midpoint - start) @ delta) / denom, 0.0, 1.0 - 1e-12))
    return min(int(phases) - 1, int(fraction * int(phases)))


def astar_flow_path(
    graph: dict,
    coords: np.ndarray,
    node_region: np.ndarray,
    edge_index: dict[tuple[int, int], int],
    source: int,
    target: int,
    occ_score: np.ndarray | None,
    trans_score: np.ndarray | None,
    *,
    sectors: int,
    phases: int,
    eta: float,
    max_expansions: int,
) -> list[int] | None:
    source, target = int(source), int(target)
    if source == target:
        return [source]
    family = direction_family(coords[source], coords[target], sectors)
    minimum_multiplier = math.exp(-float(eta) * 1.0)
    queue = [(float(np.linalg.norm(coords[source] - coords[target])) * minimum_multiplier, 0.0, source)]
    distance = {source: 0.0}
    parent: dict[int, int] = {}
    expansions = 0
    while queue and expansions < int(max_expansions):
        _, current_distance, u = heapq.heappop(queue)
        if current_distance != distance.get(u):
            continue
        if u == target:
            path = [u]
            while path[-1] != source:
                path.append(parent[path[-1]])
            return list(reversed(path))
        expansions += 1
        ru = int(node_region[u])
        for v, base_weight in graph.get(u, []):
            v = int(v)
            rv = int(node_region[v])
            score = 0.0
            if occ_score is not None:
                phase = route_phase(coords, source, target, u, v, phases)
                score = 0.70 * float(occ_score[family, phase, rv])
                edge = edge_index.get((ru, rv))
                if edge is not None and trans_score is not None:
                    score += 0.30 * float(trans_score[family, phase, edge])
            multiplier = math.exp(-float(eta) * float(np.clip(score, -1.0, 1.0)))
            candidate_distance = current_distance + float(base_weight) * multiplier
            if candidate_distance >= distance.get(v, math.inf):
                continue
            distance[v] = candidate_distance
            parent[v] = u
            heuristic = float(np.linalg.norm(coords[v] - coords[target])) * minimum_multiplier
            heapq.heappush(queue, (candidate_distance + heuristic, candidate_distance, v))
    return None


def evaluate_routes(
    real: list[np.ndarray],
    indices: list[int],
    graph: dict,
    coords: np.ndarray,
    road_tree: cKDTree,
    node_region: np.ndarray,
    edge_index: dict[tuple[int, int], int],
    models: dict[str, tuple[np.ndarray | None, np.ndarray | None]],
    *,
    sectors: int,
    phases: int,
    eta: float,
    eval_grid: int,
    max_expansions: int,
) -> dict:
    mn = np.asarray([BBOX[0], BBOX[2]], dtype=float)
    span = np.asarray([BBOX[1] - BBOX[0], BBOX[3] - BBOX[2]], dtype=float)
    rows = {name: [] for name in models}
    for position, index in enumerate(indices, start=1):
        trajectory = real[int(index)]
        src, dst = [int(x) for x in road_tree.query(trajectory[[0, -1], :2])[1]]
        if src == dst:
            continue
        real_cells = support.grid_cells(trajectory, eval_grid)
        for name, (occ_score, trans_score) in models.items():
            nodes = astar_flow_path(
                graph,
                coords,
                node_region,
                edge_index,
                src,
                dst,
                occ_score,
                trans_score,
                sectors=sectors,
                phases=phases,
                eta=eta,
                max_expansions=max_expansions,
            )
            if not nodes or len(nodes) < 2:
                continue
            candidate = coords[np.asarray(nodes, dtype=int)]
            rows[name].append(
                {
                    "trajectory_index": int(index),
                    "region": support.trajectory_region(trajectory),
                    "route_distance": support.route_distance(trajectory, candidate, mn, span, max(6, eval_grid // 4)),
                    "cell_jaccard": support.jaccard(real_cells, support.grid_cells(candidate, eval_grid)),
                    "path_nodes": int(len(nodes)),
                }
            )
        print(f"[flow-routing] {position}/{len(indices)}", flush=True)
    output = {}
    for name, values in rows.items():
        output[name] = {
            "n": len(values),
            "route_distance_mean": float(np.mean([row["route_distance"] for row in values])) if values else None,
            "route_distance_median": float(np.median([row["route_distance"] for row in values])) if values else None,
            "cell_jaccard_mean": float(np.mean([row["cell_jaccard"] for row in values])) if values else None,
            "by_region": {
                region: {
                    "n": sum(row["region"] == region for row in values),
                    "route_distance_mean": float(np.mean([row["route_distance"] for row in values if row["region"] == region]))
                    if any(row["region"] == region for row in values)
                    else None,
                }
                for region in sorted({row["region"] for row in values})
            },
            "rows": values,
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-eval", type=int, default=24)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--regions", type=int, default=16)
    parser.add_argument("--sectors", type=int, default=8)
    parser.add_argument("--phases", type=int, default=3)
    parser.add_argument("--epsilons", default="0.10,0.20,0.40,0.80")
    parser.add_argument("--eta", type=float, default=0.70)
    parser.add_argument("--score-clip", type=float, default=2.0)
    parser.add_argument("--pseudocount", type=float, default=1.0)
    parser.add_argument("--eval-grid", type=int, default=32)
    parser.add_argument("--max-expansions", type=int, default=250000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    rng = np.random.default_rng(args.seed)
    real = support.load_trajectories(args.real)
    permutation = rng.permutation(len(real))
    split = min(len(real) - 1, max(1, int(len(real) * float(args.train_fraction))))
    train = [real[int(index)] for index in permutation[:split]]
    heldout = [real[int(index)] for index in permutation[split:]]
    local_eval = support.stratified_indices(heldout, args.n_eval, args.seed + 1)
    eval_indices = [int(permutation[split + int(index)]) for index in local_eval]

    osm_ways = load_osm_ways("ara_final/evidence/tables/osm_cache_beijing.pkl")
    road_coords, graph = rse.prepare_graph(real, bbox=BBOX, osm_ways=osm_ways, raw_graph=False)
    road_tree = cKDTree(road_coords)
    landmark_nodes = support.farthest_point_landmarks(road_coords, args.regions)
    landmark_coords = road_coords[np.asarray(landmark_nodes, dtype=int)]
    region_tree = cKDTree(landmark_coords)
    node_region = np.asarray(region_tree.query(road_coords)[1], dtype=int)
    edges, edge_index, prior_occ, prior_trans, adjacency = region_graph_and_prior(
        graph, road_coords, node_region, args.regions
    )
    exact_occ, exact_trans, maximum_l1 = fit_flow_query(
        train,
        region_tree,
        adjacency,
        edge_index,
        sectors=args.sectors,
        phases=args.phases,
        regions=args.regions,
    )
    models: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {"public_shortest": (None, None)}
    nonprivate = project_noisy_flow(
        exact_occ,
        exact_trans,
        prior_occ,
        prior_trans,
        epsilon=None,
        rng=rng,
        pseudocount=args.pseudocount,
    )
    models["nonprivate_flow"] = score_tables(*nonprivate, prior_occ, prior_trans, args.score_clip)
    epsilons = [float(value.strip()) for value in args.epsilons.split(",") if value.strip()]
    dp_tables = {}
    for epsilon in epsilons:
        projected = project_noisy_flow(
            exact_occ,
            exact_trans,
            prior_occ,
            prior_trans,
            epsilon=epsilon,
            rng=rng,
            pseudocount=args.pseudocount,
        )
        scores = score_tables(*projected, prior_occ, prior_trans, args.score_clip)
        name = f"dp_flow_eps_{epsilon:.2f}"
        models[name] = scores
        dp_tables[name] = projected
    if "dp_flow_eps_0.20" in models:
        occ, trans = models["dp_flow_eps_0.20"]
        models["shuffled_flow_eps_0.20"] = (
            occ[:, :, rng.permutation(occ.shape[2])],
            trans[:, :, rng.permutation(trans.shape[2])],
        )
    evaluation = evaluate_routes(
        real,
        eval_indices,
        graph,
        road_coords,
        road_tree,
        node_region,
        edge_index,
        models,
        sectors=args.sectors,
        phases=args.phases,
        eta=args.eta,
        eval_grid=args.eval_grid,
        max_expansions=args.max_expansions,
    )
    public_mean = evaluation["public_shortest"]["route_distance_mean"]
    for name, result in evaluation.items():
        result["improvement_over_public_shortest"] = (
            float(public_mean - result["route_distance_mean"])
            if public_mean is not None and result["route_distance_mean"] is not None
            else None
        )
    payload = {
        "experiment": "od_phase_dp_graph_flow_routing",
        "research_diagnostic_only": True,
        "privacy_model": {
            "adjacency": "one complete trajectory add/remove",
            "single_record_l1_bound": 1.0,
            "observed_maximum_contribution_l1": float(maximum_l1),
            "mechanism": "coordinatewise Laplace on fixed occupancy/cut-flow vector",
            "postprocessing": "simplex projection, positive edge reweighting, and A* on public OSM graph",
            "development_seed_warning": "Fixed seed is for research only, not certified release randomness.",
        },
        "configuration": {
            "n_real": len(real),
            "n_train": len(train),
            "n_eval_requested": int(args.n_eval),
            "regions": int(args.regions),
            "region_edges": len(edges),
            "sectors": int(args.sectors),
            "phases": int(args.phases),
            "epsilons": epsilons,
            "eta": float(args.eta),
            "score_clip": float(args.score_clip),
            "pseudocount": float(args.pseudocount),
            "runtime_sec_local_only": float(time.time() - started),
        },
        "evaluation": evaluation,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "dp_graph_flow_routing_probe.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out_path.resolve()),
                "summary": {
                    name: {
                        key: value
                        for key, value in result.items()
                        if key in {"n", "route_distance_mean", "route_distance_median", "cell_jaccard_mean", "improvement_over_public_shortest"}
                    }
                    for name, result in evaluation.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
