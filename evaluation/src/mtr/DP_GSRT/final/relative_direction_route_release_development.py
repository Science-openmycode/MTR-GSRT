"""Fixed-seed development release for distance-conditioned relative-direction moments."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from fractions import Fraction
from pathlib import Path

import numpy as np

import certified_discrete_dp as certified
import endpoint_coarse_route_release_development as endpoint_release
import graph_voronoi_doptimal_support_probe as support
import multires_graph_flow_experiment as legacy
import nested_quotient_graph as nested_graph
import route_structure_potential_experiment as route
import two_level_semimarkov_query as query
from audit_two_level_semimarkov_dp import build_context
from dp_graph_voronoi_release import project_simplex
from dp_two_level_semimarkov_release import ROUTE_LATTICE, hamilton_quantize_block
from public_utils import load_osm_ways


RELATIVE_DIRECTION_BINS = 8
BLOCK_BUDGETS = {
    "coarse24_occupancy": 100_000,
    "fine384_occupancy": 100_000,
    "fine96_flow": 150_000,
    "fine384_flow": 300_000,
    "distance_relative_direction": 350_000,
}
if sum(BLOCK_BUDGETS.values()) != ROUTE_LATTICE:
    raise RuntimeError("Relative-direction block budgets must sum to Q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=Path(support.DEFAULT_REAL))
    parser.add_argument("--exact-query", type=Path)
    parser.add_argument("--capacity", type=int, default=17_123)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilon-route", type=str, default="1/5")
    return parser.parse_args()


def empty_blocks(context96, context384) -> dict[str, np.ndarray]:
    return {
        "coarse24_occupancy": np.zeros(context96.coarse_regions),
        "fine384_occupancy": np.zeros(context384.fine_regions),
        "fine96_flow": np.zeros(len(context96.fine_edge_index) + 1),
        "fine384_flow": np.zeros(len(context384.fine_edge_index) + 1),
        "distance_relative_direction": np.zeros((4, RELATIVE_DIRECTION_BINS + 1)),
    }


def od_angle(trajectory: np.ndarray) -> float:
    values = np.asarray(trajectory, dtype=float)
    latitude = float(np.mean(values[[0, -1], 0]))
    delta = (values[-1, :2] - values[0, :2]) * np.asarray(
        [111.32, 111.32 * np.cos(np.deg2rad(latitude))]
    )
    return float(np.arctan2(delta[0], delta[1]) % (2.0 * np.pi))


def relative_direction_block(
    trajectory: np.ndarray,
    labels: np.ndarray,
    context384,
) -> np.ndarray:
    distance_family = int(legacy.od_family(trajectory) // legacy.OD_BEARING_SECTORS)
    result = np.zeros((4, RELATIVE_DIRECTION_BINS + 1), dtype=float)
    sequence, _ = query._expanded_with_phase(labels, context384.fine_adjacency)
    counts = np.zeros(RELATIVE_DIRECTION_BINS, dtype=float)
    direction = od_angle(trajectory)
    centers = np.asarray(context384.fine_tree.data, dtype=float)
    for source, target in zip(sequence[:-1], sequence[1:]):
        if context384.fine_edge_index.get((int(source), int(target))) is None:
            continue
        first, second = centers[int(source)], centers[int(target)]
        latitude = float(0.5 * (first[0] + second[0]))
        delta = (second - first) * np.asarray(
            [111.32, 111.32 * np.cos(np.deg2rad(latitude))]
        )
        edge_angle = float(np.arctan2(delta[0], delta[1]) % (2.0 * np.pi))
        relative = (edge_angle - direction) % (2.0 * np.pi)
        index = min(int(relative / (2.0 * np.pi) * RELATIVE_DIRECTION_BINS), 7)
        counts[index] += 1.0
    if float(counts.sum()) > 0.0:
        result[distance_family, :-1] = counts / float(counts.sum())
    else:
        result[distance_family, -1] = 1.0
    return result


def trajectory_blocks(trajectory, context96, context384) -> dict[str, np.ndarray]:
    midpoints, lengths, phase_fraction = query._segment_geometry(trajectory)
    if len(midpoints) == 0:
        return empty_blocks(context96, context384)
    road_nodes = np.asarray(context96.road_tree.query(midpoints)[1], dtype=int)
    labels24 = np.asarray(context96.node_coarse[road_nodes], dtype=int)
    labels96 = np.asarray(context96.node_fine[road_nodes], dtype=int)
    labels384 = np.asarray(context384.node_fine[road_nodes], dtype=int)
    return {
        "coarse24_occupancy": query._occupancy(
            labels24, lengths, phase_fraction, family=0, families=1, phases=1,
            regions=context96.coarse_regions,
        ).ravel(),
        "fine384_occupancy": query._occupancy(
            labels384, lengths, phase_fraction, family=0, families=1, phases=1,
            regions=context384.fine_regions,
        ).ravel(),
        "fine96_flow": endpoint_release.transition_with_hold(
            labels96, context96.fine_adjacency, context96.fine_edge_index,
            family=0, families=1,
        ).ravel(),
        "fine384_flow": endpoint_release.transition_with_hold(
            labels384, context384.fine_adjacency, context384.fine_edge_index,
            family=0, families=1,
        ).ravel(),
        "distance_relative_direction": relative_direction_block(
            np.asarray(trajectory, dtype=float), labels384, context384
        ),
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise RuntimeError("Development output directory must be new")
    numerator, denominator = (int(value) for value in args.epsilon_route.split("/"))
    epsilon = Fraction(numerator, denominator)
    if args.exact_query is not None:
        with np.load(args.exact_query, allow_pickle=False) as archive:
            if set(archive.files) != set(BLOCK_BUDGETS):
                raise RuntimeError("Exact-query block schema mismatch")
            totals = {name: np.asarray(archive[name], dtype=np.int64) for name in BLOCK_BUDGETS}
        capacity = int(args.capacity)
    else:
        with args.real.open("rb") as handle:
            real = pickle.load(handle)
        capacity = len(real)
        osm_path = (
            Path(__file__).resolve().parents[2] / "ara_final" / "evidence" / "tables"
            / "osm_cache_beijing.pkl"
        )
        coords, graph = route.prepare_graph(
            [], bbox=support.BBOX, osm_ways=load_osm_ways(osm_path), raw_graph=False
        )
        context96, _ = build_context(coords, graph, 24, 96, 4, 3)
        context384, _, _ = nested_graph.build_nested_context(coords, graph, 24, 384, 4, 3)
        totals = {
            name: np.zeros_like(value, dtype=np.int64)
            for name, value in empty_blocks(context96, context384).items()
        }
        for index, trajectory in enumerate(real):
            blocks = trajectory_blocks(trajectory, context96, context384)
            for name, budget in BLOCK_BUDGETS.items():
                quantized = hamilton_quantize_block(blocks[name], int(budget))
                if quantized is not None:
                    totals[name] += quantized
            if (index + 1) % 1000 == 0:
                print(f"[relative-direction-query] {index + 1}/{capacity}", flush=True)

    exact = np.concatenate([totals[name].ravel() for name in BLOCK_BUDGETS])
    noisy, sampler = certified.add_exact_discrete_laplace(
        exact,
        epsilon_numerator=epsilon.numerator,
        epsilon_denominator=epsilon.denominator,
        sensitivity=ROUTE_LATTICE,
        rng=random.Random(int(args.seed) + 660_023),
    )
    released, offset = {}, 0
    for name, budget in BLOCK_BUDGETS.items():
        size = int(totals[name].size)
        block = noisy[offset : offset + size].reshape(totals[name].shape)
        released[name] = (
            project_simplex(block.astype(float), float(capacity) * int(budget))
            / float(ROUTE_LATTICE)
        )
        offset += size
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with (args.out_dir / "multires_graph_flow_release.npz").open("wb") as handle:
        np.savez_compressed(handle, **released)
    with (args.out_dir / "exact_query.npz").open("wb") as handle:
        np.savez_compressed(handle, **totals)
    (args.out_dir / "DO_NOT_RELEASE.txt").write_text(
        "NON-DP fixed-seed private-query development artifact.\n", encoding="ascii"
    )
    report = {
        "classification": "NON_DP_FIXED_SEED_PRIVATE_QUERY_DO_NOT_RELEASE",
        "records": capacity,
        "dimension": int(exact.size),
        "integer_l1_sensitivity": ROUTE_LATTICE,
        "epsilon_route": str(epsilon),
        "block_budgets": BLOCK_BUDGETS,
        "block_dimensions": {name: int(value.size) for name, value in totals.items()},
        "sampler": sampler,
        "design": (
            "The 350000-unit route-choice block is four trip-distance slices by "
            "eight edge directions relative to the trajectory OD bearing plus one "
            "family-local hold atom."
        ),
    }
    (args.out_dir / "protocol.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
