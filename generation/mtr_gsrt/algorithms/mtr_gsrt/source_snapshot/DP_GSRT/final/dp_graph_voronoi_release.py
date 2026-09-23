"""Exploration-only DP graph-Voronoi trajectory release.

This prototype replaces private top-K hotspot anchors with a fixed public road
cover.  Private data only determine DP weights over that cover:

* oriented fine endpoint marginals (epsilon_endpoint);
* coarse OD demand (epsilon_od);
* OD-distance-conditioned point-count buckets (epsilon_joint);
* exact point-count distribution (epsilon_length);
* OD/phase-conditioned graph occupancy and cut flow (epsilon_route).

Every query has trajectory-level L1 sensitivity one under add/remove
adjacency.  All landmark construction, conditional sampling, fixed-slot
decoding, public shortest paths, and DP-flow A* paths are post-processing.

The script emits two releases from exactly the same DP requests:

1. graph_voronoi_public: public shortest-path control;
2. graph_voronoi_dp_flow: the DP graph-flow route reward.

Passing ``--seed`` creates a reproducible development artifact, not a
certified DP release.  The mathematical mechanism requires fresh,
unpredictable noise randomness for every actual release.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import secrets
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np
from scipy.spatial import cKDTree

from certified_discrete_dp import add_exact_discrete_laplace


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
import cross_dataset_ara_mode_benchmark as cdb  # noqa: E402
import route_structure_potential_experiment as rse  # noqa: E402
import graph_voronoi_doptimal_support_probe as support  # noqa: E402
import dp_graph_flow_routing_probe as flow  # noqa: E402


BBOX = support.BBOX
DEFAULT_REAL = support.DEFAULT_REAL
DEFAULT_OUT = ARA / "dp_reward_exploration" / "results" / "dp_graph_voronoi_release"
OD_DISTANCE_EDGES = np.asarray([0.0, 0.10, 0.20, 0.32, 0.48, math.inf], dtype=float)


def od_distance_bucket(a: int, b: int, coarse_coords_norm: np.ndarray) -> int:
    distance = float(np.linalg.norm(coarse_coords_norm[int(a)] - coarse_coords_norm[int(b)]))
    return min(max(int(np.searchsorted(OD_DISTANCE_EDGES, distance, side="right") - 1), 0), len(OD_DISTANCE_EDGES) - 2)


def project_simplex(values: np.ndarray, total: float) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum(x) = total}."""
    flat = np.asarray(values, dtype=float).ravel()
    target = max(float(total), 1e-9)
    order = np.sort(flat)[::-1]
    cumulative = np.cumsum(order) - target
    indices = np.arange(1, len(order) + 1, dtype=float)
    positive = order - cumulative / indices > 0
    rho = int(np.flatnonzero(positive)[-1]) if np.any(positive) else 0
    threshold = float(cumulative[rho] / (rho + 1))
    projected = np.maximum(flat - threshold, 0.0)
    projected *= target / max(float(projected.sum()), 1e-12)
    return projected.reshape(np.shape(values))


def noisy_probability(
    counts: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    *,
    noisy_total: float,
    certified_discrete: bool = False,
    integer_scale: int = 1,
    exact_rng=None,
) -> np.ndarray:
    if not np.isfinite(epsilon) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and strictly positive")
    if certified_discrete:
        scaled = np.asarray(counts, dtype=float) * int(integer_scale)
        integer = np.rint(scaled).astype(np.int64)
        if not np.allclose(scaled, integer, atol=1e-10, rtol=0.0):
            raise RuntimeError("Base query is not on its declared integer lattice")
        epsilon_fraction = Fraction(str(float(epsilon)))
        noisy_integer, _ = add_exact_discrete_laplace(
            integer,
            epsilon_numerator=epsilon_fraction.numerator,
            epsilon_denominator=epsilon_fraction.denominator,
            sensitivity=int(integer_scale),
            rng=exact_rng,
        )
        noisy = noisy_integer.astype(float) / int(integer_scale)
    else:
        noisy = np.asarray(counts, dtype=float) + rng.laplace(
            0.0, 1.0 / float(epsilon), size=np.shape(counts)
        )
    noisy = project_simplex(noisy, noisy_total)
    total = float(noisy.sum())
    if total <= 0.0:
        return np.full_like(noisy, 1.0 / max(noisy.size, 1), dtype=float)
    return noisy / total


def fit_base_measurements(
    real: list[np.ndarray],
    fine_tree: cKDTree,
    coarse_tree: cKDTree,
    coarse_coords_norm: np.ndarray,
    *,
    fine_count: int,
    coarse_count: int,
    eps_endpoint: float,
    eps_count: float,
    eps_od: float,
    eps_joint: float,
    eps_length: float,
    rng: np.random.Generator,
    certified_discrete: bool = False,
    exact_rng=None,
) -> dict:
    epsilons = np.asarray([eps_endpoint, eps_count, eps_od, eps_joint, eps_length], dtype=float)
    if not np.all(np.isfinite(epsilons)) or np.any(epsilons <= 0.0):
        raise ValueError("All base-mechanism epsilons must be finite and strictly positive")
    endpoint = np.zeros((2, int(fine_count)), dtype=float)
    od = np.zeros((int(coarse_count), int(coarse_count)), dtype=float)
    joint = np.zeros((len(OD_DISTANCE_EDGES) - 1, len(cdb.LENGTH_EDGES) - 1), dtype=float)
    exact_length = np.zeros(99, dtype=float)
    max_endpoint_l1 = 0.0
    for index, trajectory in enumerate(real, start=1):
        fine = fine_tree.query(trajectory[[0, -1], :2])[1]
        coarse = coarse_tree.query(trajectory[[0, -1], :2])[1]
        endpoint[0, int(fine[0])] += 0.5
        endpoint[1, int(fine[1])] += 0.5
        max_endpoint_l1 = max(max_endpoint_l1, 1.0)
        origin, destination = int(coarse[0]), int(coarse[1])
        od[origin, destination] += 1.0
        length_bucket = cdb.len_bucket(len(trajectory))
        joint[od_distance_bucket(origin, destination, coarse_coords_norm), length_bucket] += 1.0
        exact_length[min(max(len(trajectory), 2), 100) - 2] += 1.0
    sampler_reports = {}
    if certified_discrete:
        epsilon_fraction = Fraction(str(float(eps_count)))
        noisy_count, sampler_reports["count"] = add_exact_discrete_laplace(
            np.asarray([len(real)], dtype=np.int64),
            epsilon_numerator=epsilon_fraction.numerator,
            epsilon_denominator=epsilon_fraction.denominator,
            sensitivity=1,
            rng=exact_rng,
        )
        noisy_total = max(float(noisy_count[0]), 1.0)
    else:
        noisy_total = max(float(len(real)) + float(rng.laplace(0.0, 1.0 / float(eps_count))), 1.0)
    return {
        "endpoint": noisy_probability(
            endpoint, eps_endpoint, rng, noisy_total=noisy_total,
            certified_discrete=certified_discrete, integer_scale=2, exact_rng=exact_rng,
        ),
        "od": noisy_probability(
            od, eps_od, rng, noisy_total=noisy_total,
            certified_discrete=certified_discrete, exact_rng=exact_rng,
        ),
        "joint": noisy_probability(
            joint, eps_joint, rng, noisy_total=noisy_total,
            certified_discrete=certified_discrete, exact_rng=exact_rng,
        ),
        "length": noisy_probability(
            exact_length, eps_length, rng, noisy_total=noisy_total,
            certified_discrete=certified_discrete, exact_rng=exact_rng,
        ),
        "noise_sampler": {
            "type": "bit-exact two-sided geometric" if certified_discrete else "NumPy floating Laplace",
            **sampler_reports,
        },
        "sensitivity": {
            "count": 1.0,
            "endpoint": max_endpoint_l1,
            "od": 1.0,
            "joint": 1.0,
            "length": 1.0,
        },
    }


def sinkhorn_od_projection(
    noisy_od: np.ndarray,
    endpoint_probability: np.ndarray,
    fine_to_coarse: np.ndarray,
    coarse_count: int,
    *,
    iterations: int = 300,
) -> tuple[np.ndarray, dict]:
    """KL-project a noisy OD table onto DP endpoint marginals.

    All inputs are already DP outputs or public mappings.  With a strictly
    positive kernel and positive marginals, iterative proportional fitting has
    a unique information projection.
    """
    origin = np.zeros(int(coarse_count), dtype=float)
    destination = np.zeros(int(coarse_count), dtype=float)
    for fine, coarse in enumerate(np.asarray(fine_to_coarse, dtype=int)):
        origin[int(coarse)] += float(endpoint_probability[0, fine])
        destination[int(coarse)] += float(endpoint_probability[1, fine])
    origin = np.maximum(origin, 1e-12)
    destination = np.maximum(destination, 1e-12)
    origin /= origin.sum()
    destination /= destination.sum()
    coupling = np.maximum(np.asarray(noisy_od, dtype=float), 1e-12)
    coupling /= coupling.sum()
    for _ in range(int(iterations)):
        coupling *= (origin / np.maximum(coupling.sum(axis=1), 1e-15))[:, None]
        coupling *= (destination / np.maximum(coupling.sum(axis=0), 1e-15))[None, :]
    coupling /= coupling.sum()
    return coupling, {
        "row_l1_error": float(np.sum(np.abs(coupling.sum(axis=1) - origin))),
        "column_l1_error": float(np.sum(np.abs(coupling.sum(axis=0) - destination))),
        "iterations": int(iterations),
    }


def conditional_indices(fine_to_coarse: np.ndarray, coarse: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(fine_to_coarse, dtype=int) == int(coarse))


def sample_fine(
    endpoint_probability: np.ndarray,
    fine_to_coarse: np.ndarray,
    coarse: int,
    orientation: int,
    rng: np.random.Generator,
    *,
    exclude: int | None = None,
) -> int:
    indices = conditional_indices(fine_to_coarse, coarse)
    if exclude is not None:
        indices = indices[indices != int(exclude)]
    if indices.size == 0:
        indices = np.arange(endpoint_probability.shape[1], dtype=int)
        if exclude is not None and indices.size > 1:
            indices = indices[indices != int(exclude)]
    probabilities = np.asarray(endpoint_probability[int(orientation), indices], dtype=float)
    if float(probabilities.sum()) <= 0.0:
        probabilities = np.full(len(indices), 1.0 / max(len(indices), 1), dtype=float)
    else:
        probabilities /= probabilities.sum()
    return int(rng.choice(indices, p=probabilities))


def sample_exact_length(length_probability: np.ndarray, length_bucket: int, rng: np.random.Generator) -> int:
    values = np.arange(2, 101, dtype=int)
    low = cdb.LENGTH_EDGES[int(length_bucket)]
    high = cdb.LENGTH_EDGES[int(length_bucket) + 1]
    mask = (values >= low) & (values < high)
    probabilities = np.asarray(length_probability, dtype=float) * mask
    if float(probabilities.sum()) <= 0.0:
        probabilities = np.asarray(length_probability, dtype=float).copy()
    probabilities /= max(float(probabilities.sum()), 1e-12)
    return int(rng.choice(values, p=probabilities))


def sample_requests(
    measurements: dict,
    fine_to_coarse: np.ndarray,
    coarse_coords_norm: np.ndarray,
    fine_nodes: list[int],
    *,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    od_probability = np.asarray(measurements["od"], dtype=float).ravel()
    od_draws = rng.choice(od_probability.size, size=int(count), p=od_probability)
    requests = []
    for draw in od_draws:
        origin = int(draw // measurements["od"].shape[1])
        destination = int(draw % measurements["od"].shape[1])
        fine_origin = sample_fine(measurements["endpoint"], fine_to_coarse, origin, 0, rng)
        fine_destination = sample_fine(
            measurements["endpoint"],
            fine_to_coarse,
            destination,
            1,
            rng,
            exclude=fine_origin,
        )
        source = int(fine_nodes[fine_origin])
        target = int(fine_nodes[fine_destination])
        if source == target:
            alternatives = [int(node) for node in fine_nodes if int(node) != source]
            target = int(alternatives[int(rng.integers(0, len(alternatives)))])
        distance_bucket = od_distance_bucket(origin, destination, coarse_coords_norm)
        joint = np.asarray(measurements["joint"][distance_bucket], dtype=float)
        if float(joint.sum()) <= 0.0:
            joint = np.full_like(joint, 1.0 / max(len(joint), 1))
        else:
            joint /= joint.sum()
        length_bucket = int(rng.choice(len(joint), p=joint))
        exact_length = sample_exact_length(measurements["length"], length_bucket, rng)
        requests.append((source, target, exact_length))
    return requests


def synthesize_requests(
    requests: list[tuple[int, int, int]],
    graph: dict,
    road_coords: np.ndarray,
    node_region: np.ndarray,
    edge_index: dict[tuple[int, int], int],
    occ_score: np.ndarray | None,
    trans_score: np.ndarray | None,
    coarse_nodes: list[int],
    local_cache: dict[tuple[int, int], list[int] | None],
    *,
    sectors: int,
    phases: int,
    eta: float,
    max_expansions: int,
    label: str,
    hierarchical: bool,
    cache_path: Path | None,
) -> tuple[list[np.ndarray], dict]:
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as handle:
            cache = pickle.load(handle)
        print(f"[{label}] resumed route cache with {len(cache)} entries", flush=True)
    else:
        cache: dict[tuple[int, int], list[int] | None] = {}
    synthetic = []
    failed = 0
    for index, (source, target, length) in enumerate(requests, start=1):
        if not hierarchical:
            key = (int(source), int(target))
            if key not in cache:
                cache[key] = flow.astar_flow_path(
                    graph,
                    road_coords,
                    node_region,
                    edge_index,
                    source,
                    target,
                    occ_score,
                    trans_score,
                    sectors=sectors,
                    phases=phases,
                    eta=eta,
                    max_expansions=max_expansions,
                )
            nodes = cache[key]
        else:
            source_region = int(node_region[int(source)])
            target_region = int(node_region[int(target)])
            source_portal = int(coarse_nodes[source_region])
            target_portal = int(coarse_nodes[target_region])
            key = (source_portal, target_portal)
            if key not in cache:
                cache[key] = flow.astar_flow_path(
                    graph,
                    road_coords,
                    node_region,
                    edge_index,
                    source_portal,
                    target_portal,
                    occ_score,
                    trans_score,
                    sectors=sectors,
                    phases=phases,
                    eta=eta,
                    max_expansions=max_expansions,
                )
            for local_key in [(int(source), source_portal), (target_portal, int(target))]:
                if local_key not in local_cache:
                    local_cache[local_key] = flow.astar_flow_path(
                        graph,
                        road_coords,
                        node_region,
                        edge_index,
                        local_key[0],
                        local_key[1],
                        None,
                        None,
                        sectors=sectors,
                        phases=phases,
                        eta=0.0,
                        max_expansions=max_expansions,
                    )
            first = local_cache[(int(source), source_portal)]
            middle = cache[key]
            last = local_cache[(target_portal, int(target))]
            nodes = None
            if first and middle and last:
                nodes = [int(x) for x in first]
                nodes.extend(int(x) for x in middle[1:])
                nodes.extend(int(x) for x in last[1:])
                compact = []
                for node in nodes:
                    if not compact or compact[-1] != node:
                        compact.append(node)
                nodes = compact
        if not nodes or len(nodes) < 2:
            failed += 1
            synthetic.append(np.repeat(road_coords[[int(source)]], max(2, int(length)), axis=0))
        else:
            synthetic.append(rse.resample(road_coords[np.asarray(nodes, dtype=int)], max(2, int(length))))
        if index % 500 == 0 or index == len(requests):
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("wb") as handle:
                    pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"[{label}] generated {index}/{len(requests)} backbone_cache={len(cache)} "
                f"local_cache={len(local_cache)} failed={failed}",
                flush=True,
            )
    return synthetic, {
        "backbone_cache_size": len(cache),
        "shared_local_cache_size": len(local_cache),
        "failed_slots": int(failed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--public-slot-count", type=int, required=True)
    parser.add_argument("--n-gen", type=int, required=True)
    parser.add_argument("--fine-landmarks", type=int, default=256)
    parser.add_argument("--coarse-landmarks", type=int, default=24)
    parser.add_argument("--seed", type=int)
    # Public dimension-aware allocation: epsilon_i is proportional to d_i^(1/3),
    # the minimizer of summed Laplace coordinate variance under a fixed budget.
    parser.add_argument("--epsilon-endpoint", type=float, default=0.32)
    parser.add_argument("--epsilon-count", type=float, default=0.04)
    parser.add_argument("--epsilon-od", type=float, default=0.33)
    parser.add_argument("--epsilon-joint", type=float, default=0.12)
    parser.add_argument("--epsilon-length", type=float, default=0.19)
    parser.add_argument("--epsilon-route", type=float, default=0.20)
    parser.add_argument("--sectors", type=int, default=8)
    parser.add_argument("--phases", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.30)
    parser.add_argument("--score-clip", type=float, default=2.0)
    parser.add_argument("--pseudocount", type=float, default=1.0)
    parser.add_argument("--max-expansions", type=int, default=250000)
    parser.add_argument("--hierarchical-routing", action="store_true")
    parser.add_argument(
        "--skip-public",
        action="store_true",
        help="Skip the same-request public shortest-path control during parameter sweeps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.public_slot_count <= 0 or args.n_gen <= 0:
        raise SystemExit("public-slot-count and n-gen must be positive")
    seed = int(args.seed) if args.seed is not None else secrets.randbits(128)
    rng = np.random.default_rng(seed)
    development_mode = args.seed is not None
    real = support.load_trajectories(args.real)
    if len(real) > args.public_slot_count:
        raise RuntimeError("Private records exceed the fixed public database capacity")
    osm_ways = load_osm_ways("ara_final/evidence/tables/osm_cache_beijing.pkl")
    road_coords, graph = rse.prepare_graph(real, bbox=BBOX, osm_ways=osm_ways, raw_graph=False)
    fine_nodes = support.farthest_point_landmarks(road_coords, args.fine_landmarks)
    coarse_nodes = support.farthest_point_landmarks(road_coords, args.coarse_landmarks)
    fine_coords = road_coords[np.asarray(fine_nodes, dtype=int)]
    coarse_coords = road_coords[np.asarray(coarse_nodes, dtype=int)]
    fine_tree = cKDTree(fine_coords)
    coarse_tree = cKDTree(coarse_coords)
    fine_to_coarse = np.asarray(coarse_tree.query(fine_coords)[1], dtype=int)
    coarse_coords_norm = support.standardized_xy(coarse_coords)

    measurements = fit_base_measurements(
        real,
        fine_tree,
        coarse_tree,
        coarse_coords_norm,
        fine_count=args.fine_landmarks,
        coarse_count=args.coarse_landmarks,
        eps_endpoint=args.epsilon_endpoint,
        eps_count=args.epsilon_count,
        eps_od=args.epsilon_od,
        eps_joint=args.epsilon_joint,
        eps_length=args.epsilon_length,
        rng=rng,
    )
    measurements["od"], od_projection_diag = sinkhorn_od_projection(
        measurements["od"],
        measurements["endpoint"],
        fine_to_coarse,
        args.coarse_landmarks,
    )

    node_region = np.asarray(coarse_tree.query(road_coords)[1], dtype=int)
    edges, edge_index, prior_occ, prior_trans, adjacency = flow.region_graph_and_prior(
        graph, road_coords, node_region, args.coarse_landmarks
    )
    exact_occ, exact_trans, route_contribution_l1 = flow.fit_flow_query(
        real,
        coarse_tree,
        adjacency,
        edge_index,
        sectors=args.sectors,
        phases=args.phases,
        regions=args.coarse_landmarks,
    )
    if route_contribution_l1 > 1.0 + 1e-9:
        raise RuntimeError("Route query violated its public L1 clipping bound")
    projected = flow.project_noisy_flow(
        exact_occ,
        exact_trans,
        prior_occ,
        prior_trans,
        epsilon=args.epsilon_route,
        rng=rng,
        pseudocount=args.pseudocount,
    )
    occ_score, trans_score = flow.score_tables(
        *projected, prior_occ, prior_trans, args.score_clip
    )
    requests = sample_requests(
        measurements,
        fine_to_coarse,
        coarse_coords_norm,
        fine_nodes,
        count=args.n_gen,
        rng=rng,
    )
    local_cache: dict[tuple[int, int], list[int] | None] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    public_release = None
    public_diag = None
    if not args.skip_public:
        public_release, public_diag = synthesize_requests(
            requests,
            graph,
            road_coords,
            node_region,
            edge_index,
            None,
            None,
            coarse_nodes,
            local_cache,
            sectors=args.sectors,
            phases=args.phases,
            eta=0.0,
            max_expansions=args.max_expansions,
            label="voronoi-public",
            hierarchical=bool(args.hierarchical_routing),
            cache_path=args.out_dir / "public_route_cache.pkl",
        )
    dp_release, dp_diag = synthesize_requests(
        requests,
        graph,
        road_coords,
        node_region,
        edge_index,
        occ_score,
        trans_score,
        coarse_nodes,
        local_cache,
        sectors=args.sectors,
        phases=args.phases,
        eta=args.eta,
        max_expansions=args.max_expansions,
        label="voronoi-dp-flow",
        hierarchical=bool(args.hierarchical_routing),
        cache_path=args.out_dir / "dp_flow_route_cache.pkl",
    )

    public_path = args.out_dir / "graph_voronoi_public.pkl"
    dp_path = args.out_dir / "graph_voronoi_dp_flow.pkl"
    if public_release is not None:
        with public_path.open("wb") as handle:
            pickle.dump(public_release, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with dp_path.open("wb") as handle:
        pickle.dump(dp_release, handle, protocol=pickle.HIGHEST_PROTOCOL)
    protocol = {
        "algorithm": "DP-GSRT Graph-Voronoi Support Transport prototype",
        "development_artifact": bool(development_mode),
        "certified_release": False,
        "certification_note": (
            "The query/sensitivity/composition proof applies to fresh random mechanisms. "
            "A fixed-seed development artifact is deterministic and must not be claimed as a certified DP release."
            if development_mode
            else "NumPy's generator is not yet an audited exact pure-DP sampler; certification remains false."
        ),
        "adjacency": "one complete trajectory add/remove in a fixed public capacity database",
        "public_database_capacity": int(args.public_slot_count),
        "public_output_slots": int(args.n_gen),
        "public_objects": {
            "bbox": list(BBOX),
            "fine_landmarks": int(args.fine_landmarks),
            "coarse_landmarks": int(args.coarse_landmarks),
            "road_graph_source": "public OSM",
        },
        "epsilon": {
            "count": float(args.epsilon_count),
            "endpoint": float(args.epsilon_endpoint),
            "od": float(args.epsilon_od),
            "joint": float(args.epsilon_joint),
            "length": float(args.epsilon_length),
            "route": float(args.epsilon_route),
            "base_total": float(
                args.epsilon_count + args.epsilon_endpoint + args.epsilon_od + args.epsilon_joint + args.epsilon_length
            ),
            "total": float(
                args.epsilon_count
                + args.epsilon_endpoint
                + args.epsilon_od
                + args.epsilon_joint
                + args.epsilon_length
                + args.epsilon_route
            ),
            "delta": 0.0,
            "allocation_rule": "public dimension-aware rule epsilon_i proportional to query_dimension_i^(1/3)",
        },
        "sensitivity": {
            **measurements["sensitivity"],
            "route": 1.0,
        },
        "route_reward": {
            "od_direction_sectors": int(args.sectors),
            "route_phases": int(args.phases),
            "public_region_edges": int(len(edges)),
            "eta": float(args.eta),
            "score_clip": float(args.score_clip),
            "pseudocount": float(args.pseudocount),
        },
        "od_consistency_projection": od_projection_diag,
        "hierarchical_routing": bool(args.hierarchical_routing),
        "same_request_control": not bool(args.skip_public),
        "outputs": {
            "public_shortest": public_path.name if public_release is not None else None,
            "dp_flow": dp_path.name,
        },
        "postprocessed_diagnostics": {
            "public": public_diag,
            "dp_flow": dp_diag,
        },
        "not_released": ["runtime", "raw valid record count", "private query support counts"],
    }
    protocol_path = args.out_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "public": str(public_path.resolve()),
                "dp_flow": str(dp_path.resolve()),
                "protocol": str(protocol_path.resolve()),
                "development_artifact": development_mode,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
