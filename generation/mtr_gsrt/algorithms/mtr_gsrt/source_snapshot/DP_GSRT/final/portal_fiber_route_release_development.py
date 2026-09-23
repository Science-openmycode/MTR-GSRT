"""Fixed-seed development release for a DP portal-fiber route-choice query."""
from __future__ import annotations

import argparse
import json
import pickle
import random
from fractions import Fraction
from pathlib import Path

import numpy as np

import certified_discrete_dp as certified
import endpoint_coarse_route_release_development as endpoint_release
import graph_voronoi_doptimal_support_probe as support
import nested_quotient_graph as nested_graph
import route_structure_potential_experiment as route
import two_level_semimarkov_query as query
from audit_two_level_semimarkov_dp import build_context
from dp_graph_voronoi_release import project_simplex
from dp_two_level_semimarkov_release import ROUTE_LATTICE, hamilton_quantize_block
from public_utils import load_osm_ways


PORTALS_PER_FINE_EDGE = 6
BLOCK_BUDGETS = {
    "coarse24_occupancy": 100_000,
    "fine384_occupancy": 100_000,
    "fine96_flow": 150_000,
    "fine384_flow": 300_000,
    "portal_fiber_flow": 350_000,
}
if sum(BLOCK_BUDGETS.values()) != ROUTE_LATTICE:
    raise RuntimeError("Portal-fiber block budgets must sum to Q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=Path(support.DEFAULT_REAL))
    parser.add_argument("--exact-query", type=Path)
    parser.add_argument("--capacity", type=int, default=17_123)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilon-route", type=str, default="1/5")
    return parser.parse_args()


def diverse_portals(graph, coords, labels, count=PORTALS_PER_FINE_EDGE):
    crossings = {}
    for source, adjacency in graph.items():
        a = int(labels[int(source)])
        for target, _ in adjacency:
            b = int(labels[int(target)])
            if a != b:
                crossings.setdefault((a, b), []).append((int(source), int(target)))
    selected = {}
    for key, values in crossings.items():
        unique = list(dict.fromkeys(values))
        midpoints = np.asarray([0.5 * (coords[a] + coords[b]) for a, b in unique])
        chosen = [0]
        while len(chosen) < min(int(count), len(unique)):
            distance = np.min(
                np.sum(
                    (midpoints[:, None, :] - midpoints[np.asarray(chosen)][None, :, :]) ** 2,
                    axis=2,
                ),
                axis=1,
            )
            distance[np.asarray(chosen)] = -1.0
            chosen.append(int(np.argmax(distance)))
        selected[key] = [unique[index] for index in chosen]
    return selected


def portal_layout(portals):
    offsets, cursor = {}, 0
    for key in sorted(portals):
        offsets[key] = cursor
        cursor += len(portals[key])
    return offsets, cursor


def portal_fiber_block(trajectory, labels, context384, portals, offsets, coords):
    result = np.zeros(sum(len(values) for values in portals.values()) + 1, dtype=float)
    sequence, fractions = query._expanded_with_phase(labels, context384.fine_adjacency)
    values = np.asarray(trajectory, dtype=float)[:, :2]
    chosen = []
    for index, (source, target) in enumerate(zip(sequence[:-1], sequence[1:])):
        key = (int(source), int(target))
        options = portals.get(key, [])
        if not options:
            continue
        fraction = fractions[index]
        raw_index = min(int(fraction * len(values)), len(values) - 1)
        target_coord = values[raw_index]
        portal_coords = np.asarray([0.5 * (coords[u] + coords[v]) for u, v in options])
        option = int(np.argmin(np.sum((portal_coords - target_coord[None, :]) ** 2, axis=1)))
        chosen.append(int(offsets[key]) + option)
    if chosen:
        mass = 1.0 / len(chosen)
        for index in chosen:
            result[index] += mass
    else:
        result[-1] = 1.0
    return result


def trajectory_blocks(trajectory, context96, context384, portals, offsets, coords):
    midpoints, lengths, phase_fraction = query._segment_geometry(trajectory)
    if len(midpoints) == 0:
        return None
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
        "portal_fiber_flow": portal_fiber_block(
            np.asarray(trajectory, dtype=float), labels384, context384, portals, offsets, coords
        ),
    }


def quantize_trajectory_blocks(blocks):
    """Quantize one record atomically so malformed records contribute zero."""
    if blocks is None or set(blocks) != set(BLOCK_BUDGETS):
        return None
    quantized = {
        name: hamilton_quantize_block(blocks[name], int(budget))
        for name, budget in BLOCK_BUDGETS.items()
    }
    if any(value is None for value in quantized.values()):
        return None
    contribution = sum(int(value.sum()) for value in quantized.values())
    if contribution != ROUTE_LATTICE:
        raise RuntimeError("Valid portal-fiber record must contribute exactly Q")
    return quantized


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
        portal_count = int(totals["portal_fiber_flow"].size - 1)
    else:
        with args.real.open("rb") as handle:
            real = pickle.load(handle)
        if len(real) > int(args.capacity):
            raise RuntimeError("Private input exceeds the fixed public capacity")
        capacity = int(args.capacity)
        osm_path = (
            Path(__file__).resolve().parents[2] / "ara_final" / "evidence" / "tables"
            / "osm_cache_beijing.pkl"
        )
        coords, graph = route.prepare_graph(
            [], bbox=support.BBOX, osm_ways=load_osm_ways(osm_path), raw_graph=False
        )
        context96, _ = build_context(coords, graph, 24, 96, 4, 3)
        context384, _, _ = nested_graph.build_nested_context(coords, graph, 24, 384, 4, 3)
        portals = diverse_portals(graph, coords, context384.node_fine)
        offsets, portal_count = portal_layout(portals)
        templates = {
            "coarse24_occupancy": np.zeros(24),
            "fine384_occupancy": np.zeros(384),
            "fine96_flow": np.zeros(len(context96.fine_edge_index) + 1),
            "fine384_flow": np.zeros(len(context384.fine_edge_index) + 1),
            "portal_fiber_flow": np.zeros(portal_count + 1),
        }
        totals = {name: np.zeros_like(value, dtype=np.int64) for name, value in templates.items()}
        for index, trajectory in enumerate(real):
            blocks = trajectory_blocks(
                trajectory, context96, context384, portals, offsets, coords
            )
            quantized = quantize_trajectory_blocks(blocks)
            if quantized is None:
                continue
            for name in BLOCK_BUDGETS:
                totals[name] += quantized[name]
            if (index + 1) % 1000 == 0:
                print(f"[portal-fiber-query] {index + 1}/{capacity}", flush=True)

    exact = np.concatenate([totals[name].ravel() for name in BLOCK_BUDGETS])
    noisy, sampler = certified.add_exact_discrete_laplace(
        exact,
        epsilon_numerator=epsilon.numerator,
        epsilon_denominator=epsilon.denominator,
        sensitivity=ROUTE_LATTICE,
        rng=random.Random(int(args.seed) + 770_027),
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
        "portal_fiber_atoms": portal_count,
        "integer_l1_sensitivity": ROUTE_LATTICE,
        "epsilon_route": str(epsilon),
        "block_budgets": BLOCK_BUDGETS,
        "block_dimensions": {name: int(value.size) for name, value in totals.items()},
        "sampler": sampler,
        "design": "Public nested fine-edge portal fibers plus one global hold atom.",
    }
    (args.out_dir / "protocol.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
