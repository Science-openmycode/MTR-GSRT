"""Candidate-adaptive exact-edge DP route moments on a finite route bank."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ARA = Path(__file__).resolve().parents[1]
for path in [ARA / "dp_reward_exploration", ARA / "public_release", ARA / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import graph_voronoi_doptimal_support_probe as support  # noqa: E402
import route_structure_potential_experiment as route  # noqa: E402
from audit_two_level_semimarkov_dp import build_context  # noqa: E402
from certified_discrete_dp import add_exact_discrete_laplace  # noqa: E402
from dp_graph_voronoi_release import project_simplex  # noqa: E402
from dp_two_level_semimarkov_release import ROUTE_LATTICE, hamilton_quantize_block  # noqa: E402
from public_utils import load_osm_ways  # noqa: E402
from two_level_maxent_projection import fit_projection, kl_trust_region  # noqa: E402
from two_level_semimarkov_query import trajectory_blocks  # noqa: E402


STRUCTURAL_BUDGETS = {
    "coarse_occupancy": 200_000,
    "coarse_transition": 180_000,
    "fine_occupancy": 150_000,
    "fine_transition": 150_000,
    "coarse_dwell": 50_000,
}
EDGE_OCCUPANCY_BUDGET = 135_000
EDGE_TRANSITION_BUDGET = 135_000
BLOCK_ORDER = [*STRUCTURAL_BUDGETS, "candidate_edge_occupancy", "candidate_edge_transition"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--public-capacity", type=int, default=17123)
    parser.add_argument("--epsilon-route", type=float, default=0.20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--kl-radius", type=float, default=0.20)
    parser.add_argument("--max-iterations", type=int, default=120)
    return parser.parse_args()


def load_candidates(directory: Path) -> list[list[np.ndarray]]:
    paths = sorted(directory.glob("public_candidate_*.pkl"))
    if len(paths) < 2:
        raise RuntimeError("Candidate-adaptive experiment requires at least two candidates")
    output = []
    for path in paths:
        with path.open("rb") as handle:
            output.append([np.asarray(item, dtype=float) for item in pickle.load(handle)])
    if len({len(items) for items in output}) != 1:
        raise RuntimeError("Candidate bank is not request aligned")
    return output


def edge_sequence(trajectory: np.ndarray, edge_tree) -> list[int]:
    array = np.asarray(trajectory, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2 or len(array) < 2:
        return []
    midpoints = 0.5 * (array[:-1, :2] + array[1:, :2])
    sequence = np.asarray(edge_tree.query(midpoints)[1], dtype=int).tolist()
    compact = []
    for edge in sequence:
        if not compact or compact[-1] != int(edge):
            compact.append(int(edge))
    return compact


def build_candidate_support(candidates: list[list[np.ndarray]], edge_tree):
    edges = set()
    transitions = set()
    sequences = []
    for candidate_list in candidates:
        candidate_sequences = []
        for trajectory in candidate_list:
            sequence = edge_sequence(trajectory, edge_tree)
            candidate_sequences.append(sequence)
            edges.update(sequence)
            transitions.update(zip(sequence[:-1], sequence[1:]))
        sequences.append(candidate_sequences)
    edge_values = sorted(int(value) for value in edges)
    transition_values = sorted((int(a), int(b)) for a, b in transitions)
    return (
        {value: index for index, value in enumerate(edge_values)},
        {value: index for index, value in enumerate(transition_values)},
        sequences,
        edge_values,
        transition_values,
    )


def integer_histogram(
    events,
    index: dict,
    budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(events)
    if not counts:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)
    supported = Counter()
    other = 0
    for event, count in counts.items():
        mapped = index.get(event)
        if mapped is None:
            other += int(count)
        else:
            supported[int(mapped)] += int(count)
    keys = sorted(supported)
    values = [supported[key] for key in keys]
    if other:
        keys.append(len(index))
        values.append(other)
    total = sum(values)
    divisions = [divmod(int(budget) * value, total) for value in values]
    allocations = [item[0] for item in divisions]
    residual = int(budget) - sum(allocations)
    order = sorted(range(len(keys)), key=lambda i: (-divisions[i][1], keys[i]))
    for position in order[:residual]:
        allocations[position] += 1
    return np.asarray(keys, dtype=np.int32), np.asarray(allocations, dtype=np.int64)


def aggregate_query(real, context, edge_tree, edge_index, transition_index):
    template = trajectory_blocks(np.empty((0, 2)), context)
    measurements = {
        name: np.zeros_like(template[name], dtype=np.int64) for name in STRUCTURAL_BUDGETS
    }
    measurements["candidate_edge_occupancy"] = np.zeros(len(edge_index) + 1, dtype=np.int64)
    measurements["candidate_edge_transition"] = np.zeros(len(transition_index) + 1, dtype=np.int64)
    maximum = 0
    for position, trajectory in enumerate(real, start=1):
        blocks = trajectory_blocks(trajectory, context)
        contribution = 0
        invalid = False
        quantized = {}
        for name, budget in STRUCTURAL_BUDGETS.items():
            values = hamilton_quantize_block(blocks[name], budget)
            if values is None:
                invalid = True
                break
            quantized[name] = values
            contribution += int(values.sum())
        if invalid:
            continue
        sequence = edge_sequence(trajectory, edge_tree)
        edge_ids, edge_values = integer_histogram(sequence, edge_index, EDGE_OCCUPANCY_BUDGET)
        pair_ids, pair_values = integer_histogram(
            list(zip(sequence[:-1], sequence[1:])), transition_index, EDGE_TRANSITION_BUDGET
        )
        for name, values in quantized.items():
            measurements[name] += values
        np.add.at(measurements["candidate_edge_occupancy"], edge_ids, edge_values)
        np.add.at(measurements["candidate_edge_transition"], pair_ids, pair_values)
        contribution += int(edge_values.sum()) + int(pair_values.sum())
        maximum = max(maximum, contribution)
        if contribution > ROUTE_LATTICE:
            raise RuntimeError("Candidate-adaptive contribution exceeded Q")
        if position % 3000 == 0:
            print(f"[candidate-adaptive-query] completed public batch marker {position // 3000}", flush=True)
    return measurements, maximum


def release_query(measurements, public_capacity: int, seed: int):
    shapes = {name: measurements[name].shape for name in BLOCK_ORDER}
    sizes = {name: int(np.prod(shapes[name])) for name in BLOCK_ORDER}
    exact = np.concatenate([measurements[name].ravel() for name in BLOCK_ORDER]).astype(np.int64)
    noisy, sampler = add_exact_discrete_laplace(
        exact,
        epsilon_numerator=1,
        epsilon_denominator=5,
        sensitivity=ROUTE_LATTICE,
        rng=random.Random(int(seed) + 7919),
    )
    projected = project_simplex(noisy.astype(float), float(public_capacity) * ROUTE_LATTICE) / ROUTE_LATTICE
    released = {}
    offset = 0
    for name in BLOCK_ORDER:
        released[name] = projected[offset : offset + sizes[name]].reshape(shapes[name])
        offset += sizes[name]
    return released, sampler


def continuous_histogram(events, index: dict, budget: int):
    counts = Counter(events)
    if not counts:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=float)
    ids, values = [], []
    total = sum(counts.values())
    for event, count in counts.items():
        ids.append(int(index.get(event, len(index))))
        values.append(float(budget) / ROUTE_LATTICE * float(count) / float(total))
    combined = Counter()
    for key, value in zip(ids, values):
        combined[key] += value
    keys = sorted(combined)
    return np.asarray(keys, dtype=np.int32), np.asarray([combined[key] for key in keys], dtype=float)


def candidate_matrix(candidates, sequences, context, edge_index, transition_index):
    template = trajectory_blocks(np.empty((0, 2)), context)
    offsets = {}
    dimension = 0
    for name in STRUCTURAL_BUDGETS:
        offsets[name] = dimension
        dimension += int(np.prod(template[name].shape))
    offsets["candidate_edge_occupancy"] = dimension
    dimension += len(edge_index) + 1
    offsets["candidate_edge_transition"] = dimension
    dimension += len(transition_index) + 1
    groups, width = len(candidates[0]), len(candidates)
    indptr = np.empty(groups * width + 1, dtype=np.int64)
    indptr[0] = 0
    index_parts, value_parts = [], []
    row = 0
    for request in range(groups):
        for candidate in range(width):
            blocks = trajectory_blocks(candidates[candidate][request], context)
            row_indices, row_values = [], []
            for name, budget in STRUCTURAL_BUDGETS.items():
                flat = np.asarray(blocks[name], dtype=float).ravel()
                nonzero = np.flatnonzero(flat)
                row_indices.append(offsets[name] + nonzero)
                row_values.append(float(budget) / ROUTE_LATTICE * flat[nonzero])
            sequence = sequences[candidate][request]
            edge_ids, edge_values = continuous_histogram(sequence, edge_index, EDGE_OCCUPANCY_BUDGET)
            pair_ids, pair_values = continuous_histogram(
                list(zip(sequence[:-1], sequence[1:])), transition_index, EDGE_TRANSITION_BUDGET
            )
            row_indices.extend(
                [offsets["candidate_edge_occupancy"] + edge_ids, offsets["candidate_edge_transition"] + pair_ids]
            )
            row_values.extend([edge_values, pair_values])
            indices = np.concatenate(row_indices).astype(np.int32, copy=False)
            values = np.concatenate(row_values).astype(float, copy=False)
            index_parts.append(indices)
            value_parts.append(values)
            row += 1
            indptr[row] = indptr[row - 1] + len(indices)
        if (request + 1) % 200 == 0:
            print(f"[candidate-adaptive-features] {request + 1}/{groups}", flush=True)
    return csr_matrix(
        (np.concatenate(value_parts), np.concatenate(index_parts), indptr),
        shape=(groups * width, dimension),
    )


def flatten_release(released):
    return np.concatenate([np.asarray(released[name], dtype=float).ravel() for name in BLOCK_ORDER])


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.candidate_dir)
    groups, width = len(candidates[0]), len(candidates)
    real = support.load_trajectories(support.DEFAULT_REAL)
    osm = load_osm_ways("ara_final/evidence/tables/osm_cache_beijing.pkl")
    coords, graph = route.prepare_graph([], bbox=support.BBOX, osm_ways=osm, raw_graph=False)
    context, graph_diag = build_context(coords, graph, 24, 96, 4, 3)
    edge_index, transition_index, sequences, edge_values, transition_values = build_candidate_support(
        candidates, context.road_edge_tree
    )
    print(
        f"[candidate-adaptive-support] edges={len(edge_index)} transitions={len(transition_index)}",
        flush=True,
    )
    exact, observed = aggregate_query(real, context, context.road_edge_tree, edge_index, transition_index)
    released, sampler = release_query(exact, args.public_capacity, args.seed)
    matrix = candidate_matrix(candidates, sequences, context, edge_index, transition_index)
    target = flatten_release(released) * (float(groups) / float(args.public_capacity))
    probability, optimizer = fit_projection(
        matrix,
        target,
        groups=groups,
        candidates_per_group=width,
        noise_variance=0.25 + 2.0 * groups / (args.epsilon_route**2 * args.public_capacity**2),
        max_iterations=args.max_iterations,
    )
    probability, trust = kl_trust_region(probability, args.kl_radius)
    selection_seed = int(args.selection_seed) if args.selection_seed is not None else int(args.seed)
    rng = np.random.default_rng(selection_seed + 5001)
    projected, uniform, selected = [], [], []
    for request in range(groups):
        choice = int(rng.choice(width, p=probability[request]))
        control = int(rng.integers(0, width))
        projected.append(candidates[choice][request])
        uniform.append(candidates[control][request])
        selected.append(choice)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "selection_probability.npy", probability, allow_pickle=False)
    for name, values in [("maxent_projected.pkl", projected), ("uniform_candidate_control.pkl", uniform)]:
        with (args.out_dir / name).open("wb") as handle:
            pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(args.out_dir / "candidate_adaptive_route_release.npz", **released)
    with (args.out_dir / "candidate_support.pkl").open("wb") as handle:
        pickle.dump({"edges": edge_values, "transitions": transition_values}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    protocol = {
        "algorithm": "candidate-adaptive finite-support exact-edge maximum entropy",
        "development_artifact": True,
        "certified_release": False,
        "adaptive_composition": "base DP release -> candidate support -> epsilon_route DP query",
        "epsilon": {"base": 1.0, "route": 0.2, "total": 1.2, "delta": 0.0},
        "integer_sensitivity": ROUTE_LATTICE,
        "budgets": {**STRUCTURAL_BUDGETS, "candidate_edge_occupancy": EDGE_OCCUPANCY_BUDGET,
                    "candidate_edge_transition": EDGE_TRANSITION_BUDGET},
        "support": {"edges": len(edge_index), "transitions": len(transition_index)},
        "graph": graph_diag,
        "sampler": sampler,
        "observed_private_norm_not_for_release": int(observed),
        "optimizer": optimizer,
        "kl_trust_region": trust,
        "selected_candidate_counts": np.bincount(selected, minlength=width).tolist(),
        "development_seeds": {"route_noise": int(args.seed), "selection": int(selection_seed)},
    }
    (args.out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({key: protocol[key] for key in ["support", "optimizer", "kl_trust_region"]}, indent=2))


if __name__ == "__main__":
    main()
