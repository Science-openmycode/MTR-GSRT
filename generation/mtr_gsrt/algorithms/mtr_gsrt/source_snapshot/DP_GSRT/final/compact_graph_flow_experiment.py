"""Low-dimensional quotient-graph flow reward for DP-GSRT development."""
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


BLOCK_BUDGETS = {
    "fine_occupancy": 300_000,
    "fine_flow": 600_000,
    "dwell": 100_000,
}
if sum(BLOCK_BUDGETS.values()) != ROUTE_LATTICE:
    raise RuntimeError("Compact graph-flow budgets must sum to Q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--public-capacity", type=int, default=17123)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--kl-radius", type=float, default=0.20)
    parser.add_argument("--max-iterations", type=int, default=160)
    parser.add_argument("--epsilon-route", choices=["0.05", "0.10", "0.20"], default="0.20")
    return parser.parse_args()


def compact_blocks(trajectory: np.ndarray, context):
    midpoints, lengths, phase_fraction = query._segment_geometry(trajectory)
    if len(midpoints) == 0:
        return {
            "fine_occupancy": np.zeros(context.fine_regions),
            "fine_flow": np.zeros(len(context.fine_edge_index)),
            "dwell": np.zeros(len(query.DWELL_EDGES) - 1),
        }
    road_nodes = np.asarray(context.road_tree.query(midpoints)[1], dtype=int)
    coarse_labels = np.asarray(context.node_coarse[road_nodes], dtype=int)
    fine_labels = np.asarray(context.node_fine[road_nodes], dtype=int)
    return {
        "fine_occupancy": query._occupancy(
            fine_labels,
            lengths,
            phase_fraction,
            family=0,
            families=1,
            phases=1,
            regions=context.fine_regions,
        ).ravel(),
        "fine_flow": query._transitions(
            fine_labels,
            context.fine_adjacency,
            context.fine_edge_index,
            family=0,
            families=1,
            phases=1,
        ).ravel(),
        "dwell": query._dwell(
            coarse_labels,
            lengths,
            phase_fraction,
            family=0,
            families=1,
            phases=1,
        ).ravel(),
    }


def aggregate(real, context):
    template = compact_blocks(np.empty((0, 2)), context)
    totals = {name: np.zeros_like(values, dtype=np.int64) for name, values in template.items()}
    for trajectory in real:
        blocks = compact_blocks(trajectory, context)
        for name, budget in BLOCK_BUDGETS.items():
            quantized = hamilton_quantize_block(blocks[name], budget)
            if quantized is not None:
                totals[name] += quantized
    return totals


def release(totals, capacity, exact_rng, epsilon: Fraction = Fraction(1, 5)):
    exact = np.concatenate([totals[name].ravel() for name in BLOCK_BUDGETS])
    noisy, sampler = add_exact_discrete_laplace(
        exact,
        epsilon_numerator=epsilon.numerator,
        epsilon_denominator=epsilon.denominator,
        sensitivity=ROUTE_LATTICE,
        rng=exact_rng,
    )
    released = {}
    offset = 0
    for name, budget in BLOCK_BUDGETS.items():
        size = totals[name].size
        block = noisy[offset : offset + size].reshape(totals[name].shape)
        released[name] = project_simplex(
            block.astype(float), float(capacity) * int(budget)
        ) / float(ROUTE_LATTICE)
        offset += size
    return released, sampler


def candidate_matrix(candidates, context):
    template = compact_blocks(np.empty((0, 2)), context)
    offsets = {}
    dimension = 0
    for name in BLOCK_BUDGETS:
        offsets[name] = dimension
        dimension += template[name].size
    groups, width = len(candidates[0]), len(candidates)
    indptr = np.zeros(groups * width + 1, dtype=np.int64)
    index_parts, value_parts = [], []
    row = 0
    for request in range(groups):
        for candidate in range(width):
            blocks = compact_blocks(candidates[candidate][request], context)
            indices, values = [], []
            for name, budget in BLOCK_BUDGETS.items():
                flat = np.asarray(blocks[name], dtype=float).ravel()
                nonzero = np.flatnonzero(flat)
                indices.append(offsets[name] + nonzero)
                values.append(float(budget) / ROUTE_LATTICE * flat[nonzero])
            index_parts.append(np.concatenate(indices).astype(np.int32, copy=False))
            value_parts.append(np.concatenate(values))
            row += 1
            indptr[row] = indptr[row - 1] + len(index_parts[-1])
    return csr_matrix(
        (np.concatenate(value_parts), np.concatenate(index_parts), indptr),
        shape=(groups * width, dimension),
    )


def main() -> None:
    args = parse_args()
    candidates = adaptive.load_candidates(args.candidate_dir)
    groups, width = len(candidates[0]), len(candidates)
    real = support.load_trajectories(support.DEFAULT_REAL)
    osm = load_osm_ways("ara_final/evidence/tables/osm_cache_beijing.pkl")
    coords, graph = route.prepare_graph([], bbox=support.BBOX, osm_ways=osm, raw_graph=False)
    context, graph_diag = build_context(coords, graph, 24, 96, 4, 3)
    totals = aggregate(real, context)
    epsilon_route = Fraction(args.epsilon_route)
    released, sampler = release(
        totals,
        args.public_capacity,
        random.Random(int(args.seed) + 7919),
        epsilon_route,
    )
    matrix = candidate_matrix(candidates, context)
    target = np.concatenate([released[name].ravel() for name in BLOCK_BUDGETS]) * (
        float(groups) / float(args.public_capacity)
    )
    probability, optimizer = fit_projection(
        matrix,
        target,
        groups=groups,
        candidates_per_group=width,
        noise_variance=0.25
        + 2.0 * groups / (float(epsilon_route) ** 2 * args.public_capacity**2),
        max_iterations=args.max_iterations,
    )
    probability, trust = kl_trust_region(probability, args.kl_radius)
    selection_seed = int(args.selection_seed if args.selection_seed is not None else args.seed)
    rng = np.random.default_rng(selection_seed + 5001)
    projected, uniform, selected = [], [], []
    for request in range(groups):
        choice = int(rng.choice(width, p=probability[request]))
        control = int(rng.integers(0, width))
        projected.append(candidates[choice][request])
        uniform.append(candidates[control][request])
        selected.append(choice)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in [("maxent_projected.pkl", projected), ("uniform_candidate_control.pkl", uniform)]:
        with (args.out_dir / name).open("wb") as handle:
            pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(args.out_dir / "selection_probability.npy", probability, allow_pickle=False)
    np.savez_compressed(args.out_dir / "compact_graph_flow_release.npz", **released)
    protocol = {
        "algorithm": "compact quotient-graph flow maximum entropy",
        "development_artifact": True,
        "certified_release": False,
        "warning": "fixed-seed development artifact; not a DP production release",
        "epsilon": {
            "base": 1.0,
            "route_rational": f"{epsilon_route.numerator}/{epsilon_route.denominator}",
            "total": 1.0 + float(epsilon_route),
            "delta": 0.0,
        },
        "integer_l1_sensitivity": ROUTE_LATTICE,
        "block_budgets": BLOCK_BUDGETS,
        "block_dimensions": {name: int(totals[name].size) for name in BLOCK_BUDGETS},
        "graph": graph_diag,
        "sampler": sampler,
        "optimizer": optimizer,
        "kl_trust_region": trust,
        "selected_candidate_counts": np.bincount(selected, minlength=width).tolist(),
    }
    (args.out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
