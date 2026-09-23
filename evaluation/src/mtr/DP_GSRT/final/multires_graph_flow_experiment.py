"""Multiresolution and OD-conditioned quotient-graph DP reward experiment."""
from __future__ import annotations

import argparse
import json
import pickle
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

import candidate_adaptive_maxent_experiment as adaptive
import graph_voronoi_doptimal_support_probe as support
import route_structure_potential_experiment as route
import two_level_semimarkov_query as query
from audit_two_level_semimarkov_dp import build_context
from certified_discrete_dp import add_exact_discrete_laplace
from dp_graph_voronoi_release import project_simplex
from dp_two_level_semimarkov_release import ROUTE_LATTICE, hamilton_quantize_block
from public_utils import load_osm_ways
from two_level_maxent_projection import fit_projection, kl_trust_region


OD_DISTANCE_EDGES_KM = np.asarray([0.0, 2.0, 5.0, 10.0, np.inf], dtype=float)
OD_BEARING_SECTORS = 8
OD_FAMILIES = (len(OD_DISTANCE_EDGES_KM) - 1) * OD_BEARING_SECTORS
BLOCK_BUDGETS = {
    "coarse24_occupancy": 100_000,
    "fine384_occupancy": 100_000,
    "fine96_flow": 150_000,
    "fine384_flow": 300_000,
    "od_conditioned_fine96_flow": 350_000,
}
if sum(BLOCK_BUDGETS.values()) != ROUTE_LATTICE:
    raise RuntimeError("Multiresolution graph-flow budgets must sum to Q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--public-capacity", type=int, default=17_123)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--kl-radius", type=float, default=0.60)
    parser.add_argument("--max-iterations", type=int, default=180)
    parser.add_argument("--epsilon-route", choices=["0.10", "0.20"], default="0.20")
    parser.add_argument("--acknowledge-non-dp-development", action="store_true", required=True)
    return parser.parse_args()


def require_non_dp_output(path: Path) -> None:
    resolved = str(path.resolve()).lower()
    if "non_dp" not in path.name.lower():
        raise RuntimeError("Development output directory name must contain NON_DP")
    if "production_graph_cycle" in resolved or "certified_dp_bundle" in resolved:
        raise RuntimeError("Development output must be isolated from production and certified roots")


def od_family(trajectory: np.ndarray) -> int:
    array = np.asarray(trajectory, dtype=float)
    if len(array) < 2:
        return 0
    latitude = float(np.mean(array[[0, -1], 0]))
    delta_km = (array[-1] - array[0]) * np.asarray(
        [111.32, 111.32 * np.cos(np.deg2rad(latitude))]
    )
    distance = float(np.linalg.norm(delta_km))
    distance_bin = int(
        np.clip(np.searchsorted(OD_DISTANCE_EDGES_KM, distance, side="right") - 1, 0, 3)
    )
    angle = float(np.arctan2(delta_km[0], delta_km[1]) % (2.0 * np.pi))
    bearing_bin = min(int(angle / (2.0 * np.pi) * OD_BEARING_SECTORS), 7)
    return distance_bin * OD_BEARING_SECTORS + bearing_bin


def empty_blocks(context96, context384) -> dict[str, np.ndarray]:
    return {
        "coarse24_occupancy": np.zeros(context96.coarse_regions),
        "fine384_occupancy": np.zeros(context384.fine_regions),
        "fine96_flow": np.zeros(len(context96.fine_edge_index)),
        "fine384_flow": np.zeros(len(context384.fine_edge_index)),
        "od_conditioned_fine96_flow": np.zeros(
            OD_FAMILIES * len(context96.fine_edge_index)
        ),
    }


def graph_blocks(trajectory: np.ndarray, context96, context384) -> dict[str, np.ndarray]:
    midpoints, lengths, phase_fraction = query._segment_geometry(trajectory)
    if len(midpoints) == 0:
        return empty_blocks(context96, context384)
    road_nodes = np.asarray(context96.road_tree.query(midpoints)[1], dtype=int)
    labels96 = np.asarray(context96.node_fine[road_nodes], dtype=int)
    labels384 = np.asarray(context384.node_fine[road_nodes], dtype=int)
    labels24 = np.asarray(context96.node_coarse[road_nodes], dtype=int)
    family = od_family(trajectory)
    return {
        "coarse24_occupancy": query._occupancy(
            labels24,
            lengths,
            phase_fraction,
            family=0,
            families=1,
            phases=1,
            regions=context96.coarse_regions,
        ).ravel(),
        "fine384_occupancy": query._occupancy(
            labels384,
            lengths,
            phase_fraction,
            family=0,
            families=1,
            phases=1,
            regions=context384.fine_regions,
        ).ravel(),
        "fine96_flow": query._transitions(
            labels96,
            context96.fine_adjacency,
            context96.fine_edge_index,
            family=0,
            families=1,
            phases=1,
        ).ravel(),
        "fine384_flow": query._transitions(
            labels384,
            context384.fine_adjacency,
            context384.fine_edge_index,
            family=0,
            families=1,
            phases=1,
        ).ravel(),
        "od_conditioned_fine96_flow": query._transitions(
            labels96,
            context96.fine_adjacency,
            context96.fine_edge_index,
            family=family,
            families=OD_FAMILIES,
            phases=1,
        ).ravel(),
    }


def aggregate(real, context96, context384):
    totals = {
        name: np.zeros_like(values, dtype=np.int64)
        for name, values in empty_blocks(context96, context384).items()
    }
    for trajectory in real:
        blocks = graph_blocks(trajectory, context96, context384)
        for name, budget in BLOCK_BUDGETS.items():
            quantized = hamilton_quantize_block(blocks[name], budget)
            if quantized is not None:
                totals[name] += quantized
    return totals


def release(totals, capacity, exact_rng, epsilon: Fraction):
    exact = np.concatenate([totals[name].ravel() for name in BLOCK_BUDGETS])
    noisy, sampler = add_exact_discrete_laplace(
        exact,
        epsilon_numerator=epsilon.numerator,
        epsilon_denominator=epsilon.denominator,
        sensitivity=ROUTE_LATTICE,
        rng=exact_rng,
    )
    released, offset = {}, 0
    for name, budget in BLOCK_BUDGETS.items():
        size = totals[name].size
        block = noisy[offset : offset + size].reshape(totals[name].shape)
        released[name] = project_simplex(
            block.astype(float), float(capacity) * int(budget)
        ) / float(ROUTE_LATTICE)
        offset += size
    return released, sampler


def candidate_matrix(candidates, context96, context384):
    template = empty_blocks(context96, context384)
    offsets, dimension = {}, 0
    for name in BLOCK_BUDGETS:
        offsets[name] = dimension
        dimension += template[name].size
    groups, width = len(candidates[0]), len(candidates)
    indptr = np.zeros(groups * width + 1, dtype=np.int64)
    index_parts, value_parts = [], []
    row = 0
    for request in range(groups):
        for candidate in range(width):
            blocks = graph_blocks(candidates[candidate][request], context96, context384)
            indices, values = [], []
            for name, budget in BLOCK_BUDGETS.items():
                flat = np.asarray(blocks[name], dtype=float).ravel()
                nonzero = np.flatnonzero(flat)
                indices.append(offsets[name] + nonzero)
                values.append(float(budget) / ROUTE_LATTICE * flat[nonzero])
            current_indices = np.concatenate(indices).astype(np.int32, copy=False)
            current_values = np.concatenate(values)
            index_parts.append(current_indices)
            value_parts.append(current_values)
            row += 1
            indptr[row] = indptr[row - 1] + len(current_indices)
    return csr_matrix(
        (np.concatenate(value_parts), np.concatenate(index_parts), indptr),
        shape=(groups * width, dimension),
    )


def main() -> None:
    args = parse_args()
    require_non_dp_output(args.out_dir)
    candidates = adaptive.load_candidates(args.candidate_dir)
    groups, width = len(candidates[0]), len(candidates)
    real = support.load_trajectories(support.DEFAULT_REAL)
    osm = load_osm_ways("ara_final/evidence/tables/osm_cache_beijing.pkl")
    coords, graph = route.prepare_graph([], bbox=support.BBOX, osm_ways=osm, raw_graph=False)
    context96, graph96 = build_context(coords, graph, 24, 96, 4, 3)
    context384, graph384 = build_context(coords, graph, 24, 384, 4, 3)
    totals = aggregate(real, context96, context384)
    epsilon = Fraction(args.epsilon_route)
    released, sampler = release(
        totals,
        args.public_capacity,
        random.Random(int(args.seed) + 79_919),
        epsilon,
    )
    matrix = candidate_matrix(candidates, context96, context384)
    target = np.concatenate([released[name].ravel() for name in BLOCK_BUDGETS]) * (
        float(groups) / float(args.public_capacity)
    )
    probability, optimizer = fit_projection(
        matrix,
        target,
        groups=groups,
        candidates_per_group=width,
        noise_variance=0.25
        + 2.0 * groups / (float(epsilon) ** 2 * args.public_capacity**2),
        max_iterations=args.max_iterations,
    )
    probability, trust = kl_trust_region(probability, float(args.kl_radius))
    selection_seed = int(args.selection_seed if args.selection_seed is not None else args.seed)
    rng = np.random.default_rng(selection_seed + 50_001)
    projected, uniform, selected = [], [], []
    for request in range(groups):
        choice = int(rng.choice(width, p=probability[request]))
        control = int(rng.integers(0, width))
        projected.append(candidates[choice][request])
        uniform.append(candidates[control][request])
        selected.append(choice)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "DO_NOT_RELEASE.txt").write_text(
        "NON-DP fixed-seed development artifact. Never distribute as a private release.\n",
        encoding="ascii",
    )
    for name, values in (
        ("maxent_projected.pkl", projected),
        ("uniform_candidate_control.pkl", uniform),
    ):
        with (args.out_dir / name).open("wb") as handle:
            pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(args.out_dir / "selection_probability.npy", probability, allow_pickle=False)
    np.savez_compressed(args.out_dir / "multires_graph_flow_release.npz", **released)
    protocol = {
        "algorithm": "coarse-fine occupancy and OD-conditioned quotient-graph maximum entropy",
        "development_artifact": True,
        "certified_release": False,
        "warning": "fixed-seed development artifact; not a production DP release",
        "privacy_proof_key": (
            "Each nonempty nonnegative block is Hamilton-quantized to its fixed integer "
            "budget; block budgets sum to Q, so per-record L1 contribution is at most Q."
        ),
        "privacy_status": "NON_DP_FIXED_SEED_DO_NOT_RELEASE",
        "hypothetical_epsilon_only_for_fresh_random_one_run_replacement": {
            "base": 1.0,
            "route_rational": f"{epsilon.numerator}/{epsilon.denominator}",
            "total": 1.0 + float(epsilon),
            "delta": 0.0,
        },
        "integer_l1_sensitivity": ROUTE_LATTICE,
        "block_budgets": BLOCK_BUDGETS,
        "block_dimensions": {name: int(totals[name].size) for name in BLOCK_BUDGETS},
        "od_families": {
            "distance_edges_km": [0.0, 2.0, 5.0, 10.0, "inf"],
            "bearing_sectors": OD_BEARING_SECTORS,
        },
        "graph96": graph96,
        "graph384": graph384,
        "sampler": sampler,
        "optimizer": optimizer,
        "kl_trust_region": trust,
        "selected_candidate_counts": np.bincount(selected, minlength=width).tolist(),
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
