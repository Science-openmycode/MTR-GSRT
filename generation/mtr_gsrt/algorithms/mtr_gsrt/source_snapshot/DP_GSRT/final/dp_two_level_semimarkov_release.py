"""Privacy-first two-level semi-Markov DP-GSRT prototype.

The complete route statistic is one nonnegative concatenated query with L1
sensitivity one.  Coarse and fine bridges, product-state conditioning, and
road lifting use only public OSM objects and the released DP statistic.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import pickle
import random
import secrets
import sys
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from certified_discrete_dp import add_exact_discrete_laplace


ARA = Path(__file__).resolve().parents[1]
for path in [ARA / "dp_reward_exploration", ARA / "public_release", ARA / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dp_graph_flow_routing_probe as flow  # noqa: E402
import dp_graph_voronoi_release as base  # noqa: E402
import graph_voronoi_doptimal_support_probe as support  # noqa: E402
import route_structure_potential_experiment as route  # noqa: E402
from audit_two_level_semimarkov_dp import build_context  # noqa: E402
from public_utils import load_osm_ways  # noqa: E402
from two_level_semimarkov_query import (  # noqa: E402
    DWELL_EDGES,
    QueryContext,
    WEIGHTS,
    add_to_measurements,
    contribution_l1,
    trajectory_blocks,
)


DEFAULT_OUT = ARA / "dp_reward_exploration" / "results" / "two_level_semimarkov"
BLOCK_ORDER = list(WEIGHTS)
ROUTE_LATTICE = 1_000_000
WEIGHT_UNITS = {name: int(round(ROUTE_LATTICE * value)) for name, value in WEIGHTS.items()}
if sum(WEIGHT_UNITS.values()) != ROUTE_LATTICE:
    raise RuntimeError("Route-query weights must lie on the declared integer lattice")


def atomic_pickle_dump(value, path: Path) -> None:
    """Write a pickle without exposing a partially written artifact."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def atomic_route_release_dump(released: dict[str, np.ndarray], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{name: released[name] for name in BLOCK_ORDER})
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_route_query(trajectories: list[np.ndarray], context: QueryContext) -> tuple[dict[str, np.ndarray], float]:
    template = trajectory_blocks(np.empty((0, 2)), context)
    measurements = {name: np.zeros_like(values, dtype=float) for name, values in template.items()}
    maximum = 0.0
    for index, trajectory in enumerate(trajectories, start=1):
        blocks = trajectory_blocks(trajectory, context)
        norm, _ = contribution_l1(blocks)
        if norm > 1.0 + 1e-9:
            raise RuntimeError("Route contribution exceeded the public sensitivity bound")
        maximum = max(maximum, norm)
        add_to_measurements(measurements, blocks)
    return measurements, maximum


def hamilton_quantize_block(values: np.ndarray, budget: int) -> np.ndarray | None:
    """Allocate an integer budget using exact dyadic remainders and public ties."""
    array = np.asarray(values, dtype=float)
    flat = array.ravel()
    if np.any(flat < 0.0) or not np.all(np.isfinite(flat)):
        return None
    nonzero = np.flatnonzero(flat > 0.0)
    integer = np.zeros(flat.size, dtype=np.int64)
    if nonzero.size == 0:
        return integer.reshape(array.shape)
    ratios = [float(flat[index]).as_integer_ratio() for index in nonzero]
    denominator = max(item[1] for item in ratios)
    numerators = [item[0] * (denominator // item[1]) for item in ratios]
    total = sum(numerators)
    divisions = [divmod(int(budget) * numerator, total) for numerator in numerators]
    allocations = [item[0] for item in divisions]
    residual = int(budget) - sum(allocations)
    order = sorted(
        range(len(nonzero)),
        key=lambda index: (-divisions[index][1], int(nonzero[index])),
    )
    for index in order[:residual]:
        allocations[index] += 1
    integer[nonzero] = np.asarray(allocations, dtype=np.int64)
    return integer.reshape(array.shape)


def aggregate_route_query_integer(
    trajectories: list[np.ndarray],
    context: QueryContext,
) -> tuple[dict[str, np.ndarray], int]:
    """Exact per-record Hamilton quantization with L1 norm <= ROUTE_LATTICE."""
    template = trajectory_blocks(np.empty((0, 2)), context)
    measurements = {name: np.zeros_like(values, dtype=np.int64) for name, values in template.items()}
    maximum = 0
    for trajectory in trajectories:
        blocks = trajectory_blocks(trajectory, context)
        quantized = {}
        contribution = 0
        for name in BLOCK_ORDER:
            integer = hamilton_quantize_block(blocks[name], WEIGHT_UNITS[name])
            if integer is None:
                quantized = None
                break
            quantized[name] = integer
            contribution += int(np.sum(integer))
        if quantized is None:
            continue
        for name in BLOCK_ORDER:
            measurements[name] += quantized[name]
        maximum = max(maximum, contribution)
        if contribution > ROUTE_LATTICE:
            raise RuntimeError("Integer route contribution exceeded the declared sensitivity")
    return measurements, maximum


def release_route_query(
    measurements: dict[str, np.ndarray],
    *,
    epsilon: float,
    public_capacity: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    if not np.isfinite(epsilon) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and strictly positive")
    if int(public_capacity) <= 0:
        raise ValueError("public_capacity must be strictly positive")
    shapes = {name: measurements[name].shape for name in BLOCK_ORDER}
    sizes = {name: int(np.prod(shapes[name])) for name in BLOCK_ORDER}
    exact = np.concatenate([measurements[name].ravel() for name in BLOCK_ORDER])
    noisy = exact + rng.laplace(0.0, 1.0 / max(float(epsilon), 1e-12), size=exact.shape)
    projected = base.project_simplex(noisy, float(public_capacity))
    released = {}
    offset = 0
    for name in BLOCK_ORDER:
        released[name] = projected[offset : offset + sizes[name]].reshape(shapes[name])
        offset += sizes[name]
    return released


def release_route_query_integer(
    measurements: dict[str, np.ndarray],
    *,
    epsilon: float,
    public_capacity: int,
    exact_rng=None,
) -> tuple[dict[str, np.ndarray], dict, dict[str, np.ndarray]]:
    if not np.isfinite(epsilon) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and strictly positive")
    if int(public_capacity) <= 0:
        raise ValueError("public_capacity must be strictly positive")
    shapes = {name: measurements[name].shape for name in BLOCK_ORDER}
    sizes = {name: int(np.prod(shapes[name])) for name in BLOCK_ORDER}
    exact = np.concatenate([np.asarray(measurements[name], dtype=np.int64).ravel() for name in BLOCK_ORDER])
    epsilon_fraction = Fraction(str(float(epsilon)))
    noisy, sampler = add_exact_discrete_laplace(
        exact,
        epsilon_numerator=epsilon_fraction.numerator,
        epsilon_denominator=epsilon_fraction.denominator,
        sensitivity=ROUTE_LATTICE,
        rng=exact_rng,
    )
    projected = (
        base.project_simplex(noisy.astype(float), float(public_capacity) * ROUTE_LATTICE)
        / ROUTE_LATTICE
    )
    released = {}
    noisy_blocks = {}
    offset = 0
    for name in BLOCK_ORDER:
        released[name] = projected[offset : offset + sizes[name]].reshape(shapes[name])
        noisy_blocks[name] = noisy[offset : offset + sizes[name]].reshape(shapes[name])
        offset += sizes[name]
    return released, sampler, noisy_blocks


def public_kernel(edges: list[tuple[int, int]], prior: np.ndarray, regions: int, stay: float) -> np.ndarray:
    kernel = np.zeros((int(regions), int(regions)), dtype=float)
    for value, (u, v) in zip(np.asarray(prior, dtype=float), edges):
        kernel[int(u), int(v)] += float(value)
    for region in range(int(regions)):
        if float(kernel[region].sum()) <= 0.0:
            kernel[region, region] = 1.0
        else:
            kernel[region] /= kernel[region].sum()
    kernel *= 1.0 - float(stay)
    kernel[np.arange(int(regions)), np.arange(int(regions))] += float(stay)
    return kernel


def expected_dwell_stay(dwell: np.ndarray) -> np.ndarray:
    midpoints = 0.5 * (DWELL_EDGES[:-1] + DWELL_EDGES[1:])
    probability = np.asarray(dwell, dtype=float)
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-15)
    expectation = np.sum(probability * midpoints[None, None, :], axis=-1)
    return np.clip(expectation, 0.05, 0.65)


def conditional_kernels(
    occupancy: np.ndarray,
    transitions: np.ndarray,
    edges: list[tuple[int, int]],
    public: np.ndarray,
    *,
    alpha: float | np.ndarray,
    stay: np.ndarray | float,
) -> np.ndarray:
    families, phases, regions = occupancy.shape
    result = np.zeros((families, phases, regions, regions), dtype=float)
    for family in range(families):
        for phase in range(phases):
            learned = np.zeros((regions, regions), dtype=float)
            for index, (u, v) in enumerate(edges):
                learned[int(u), int(v)] += float(transitions[family, phase, index])
            for region in range(regions):
                learned[region] += 1e-10 * public[region]
                learned[region] /= max(float(learned[region].sum()), 1e-15)
            local_alpha = float(alpha[family, phase]) if isinstance(alpha, np.ndarray) else float(alpha)
            kernel = (1.0 - local_alpha) * public + local_alpha * learned
            occ = np.asarray(occupancy[family, phase], dtype=float)
            occ /= max(float(occ.sum()), 1e-15)
            base_stay = float(stay[family, phase]) if isinstance(stay, np.ndarray) else float(stay)
            adaptive = base_stay * occ / max(float(occ.max()), 1e-15)
            kernel *= (1.0 - adaptive)[:, None]
            kernel[np.arange(regions), np.arange(regions)] += adaptive
            kernel /= np.maximum(kernel.sum(axis=1, keepdims=True), 1e-15)
            result[family, phase] = kernel
    return result


def snr_shrinkage_alpha(
    occupancy: np.ndarray,
    transitions: np.ndarray,
    *,
    epsilon: float,
    maximum_alpha: float,
) -> np.ndarray:
    """DP-safe reliability shrinkage using released mass and known noise."""
    signal = np.sum(np.asarray(occupancy, dtype=float), axis=-1) + np.sum(
        np.asarray(transitions, dtype=float), axis=-1
    )
    coordinate_count = int(occupancy.shape[-1] + transitions.shape[-1])
    expected_noise_l1 = coordinate_count / max(float(epsilon), 1e-12)
    reliability = signal / np.maximum(signal + expected_noise_l1, 1e-15)
    return np.clip(float(maximum_alpha) * reliability, 0.0, float(maximum_alpha))


def bridge_raw(
    kernels: np.ndarray,
    source: int,
    target: int,
    family: int,
    horizon: int,
    rng: np.random.Generator,
) -> list[int] | None:
    regions, phases = kernels.shape[-1], kernels.shape[1]
    backward = np.zeros((int(horizon) + 1, regions), dtype=float)
    backward[int(horizon), int(target)] = 1.0
    for time in range(int(horizon) - 1, -1, -1):
        phase = min(phases - 1, int(time * phases / max(int(horizon), 1)))
        backward[time] = kernels[int(family), phase] @ backward[time + 1]
        maximum = float(backward[time].max())
        if maximum > 0.0:
            backward[time] /= maximum
    if backward[0, int(source)] <= 0.0:
        return None
    sequence = [int(source)]
    current = int(source)
    for time in range(int(horizon)):
        phase = min(phases - 1, int(time * phases / max(int(horizon), 1)))
        weights = kernels[int(family), phase, current] * backward[time + 1]
        if float(weights.sum()) <= 0.0:
            return None
        current = int(rng.choice(regions, p=weights / weights.sum()))
        sequence.append(current)
    return sequence if sequence[-1] == int(target) else None


def fine_product_bridge(
    kernels: np.ndarray,
    source: int,
    target: int,
    family: int,
    coarse_sequence: list[int],
    fine_coarse_compatible: np.ndarray,
    fine_adjacency: dict[int, set[int]],
    *,
    slack: int,
    max_horizon: int,
    rng: np.random.Generator,
) -> list[int] | None:
    regions = kernels.shape[-1]
    phases = kernels.shape[1]
    progress_count = len(coarse_sequence)
    neighbours = []
    for region in range(regions):
        allowed = set(fine_adjacency.get(region, set()))
        allowed.add(region)
        neighbours.append(sorted(allowed))

    # The unconstrained fine shortest path can be much shorter than a path
    # that must realize every coarse bridge state. Compute the exact public
    # product-graph lower bound before choosing the bridge horizon.
    start_state = (0, int(source))
    goal_state = (progress_count - 1, int(target))
    queue = deque([start_state])
    product_distance = {start_state: 0}
    while queue and goal_state not in product_distance:
        progress, region = queue.popleft()
        parent_now = int(coarse_sequence[progress])
        parent_next = int(coarse_sequence[progress + 1]) if progress + 1 < progress_count else -1
        for next_region in neighbours[region]:
            options = []
            if bool(fine_coarse_compatible[next_region, parent_now]):
                options.append(progress)
            if progress + 1 < progress_count and bool(fine_coarse_compatible[next_region, parent_next]):
                options.append(progress + 1)
            for option in options:
                state = (int(option), int(next_region))
                if state not in product_distance:
                    product_distance[state] = product_distance[(progress, region)] + 1
                    queue.append(state)
    if goal_state not in product_distance:
        return None
    horizon = int(product_distance[goal_state]) + int(slack)
    if horizon > int(max_horizon):
        return None
    backward = np.zeros((horizon + 1, progress_count, regions), dtype=float)
    backward[horizon, progress_count - 1, int(target)] = 1.0
    for time in range(horizon - 1, -1, -1):
        phase = min(phases - 1, int(time * phases / max(horizon, 1)))
        kernel = kernels[int(family), phase]
        for progress in range(progress_count):
            parent_now = int(coarse_sequence[progress])
            parent_next = int(coarse_sequence[progress + 1]) if progress + 1 < progress_count else -1
            for region in range(regions):
                total = 0.0
                for next_region in neighbours[region]:
                    weight = float(kernel[region, next_region])
                    if weight <= 0.0:
                        continue
                    options = []
                    if bool(fine_coarse_compatible[next_region, parent_now]):
                        options.append(progress)
                    if progress + 1 < progress_count and bool(fine_coarse_compatible[next_region, parent_next]):
                        options.append(progress + 1)
                    if not options:
                        continue
                    share = weight / len(options)
                    total += sum(share * backward[time + 1, option, next_region] for option in options)
                backward[time, progress, region] = total
        maximum = float(backward[time].max())
        if maximum > 0.0:
            backward[time] /= maximum
    if backward[0, 0, int(source)] <= 0.0:
        return None
    region, progress = int(source), 0
    sequence = [region]
    for time in range(horizon):
        phase = min(phases - 1, int(time * phases / max(horizon, 1)))
        candidates = []
        weights = []
        parent_now = int(coarse_sequence[progress])
        parent_next = int(coarse_sequence[progress + 1]) if progress + 1 < progress_count else -1
        for next_region in neighbours[region]:
            kernel_weight = float(kernels[int(family), phase, region, next_region])
            options = []
            if bool(fine_coarse_compatible[next_region, parent_now]):
                options.append(progress)
            if progress + 1 < progress_count and bool(fine_coarse_compatible[next_region, parent_next]):
                options.append(progress + 1)
            for option in options:
                candidates.append((int(next_region), int(option)))
                weights.append(kernel_weight / max(len(options), 1) * backward[time + 1, option, next_region])
        probabilities = np.asarray(weights, dtype=float)
        if probabilities.size == 0 or float(probabilities.sum()) <= 0.0:
            return None
        choice = int(rng.choice(len(candidates), p=probabilities / probabilities.sum()))
        region, progress = candidates[choice]
        sequence.append(region)
    if region != int(target) or progress != progress_count - 1:
        return None
    return sequence


def lift_repeated_corridor(
    graph: dict,
    coords: np.ndarray,
    node_fine: np.ndarray,
    source: int,
    target: int,
    corridor: list[int],
    *,
    max_expansions: int,
) -> list[int] | None:
    if not corridor or int(node_fine[int(source)]) != int(corridor[0]) or int(node_fine[int(target)]) != int(corridor[-1]):
        return None
    goal = len(corridor) - 1
    start = (int(source), 0)
    queue = [(float(np.linalg.norm(coords[int(source)] - coords[int(target)])), 0.0, start)]
    distance = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    expansions = 0
    while queue and expansions < int(max_expansions):
        _, cost, state = heapq.heappop(queue)
        if cost != distance.get(state):
            continue
        node, progress = state
        if node == int(target) and progress == goal:
            states = [state]
            while states[-1] != start:
                states.append(parent[states[-1]])
            states.reverse()
            return [item[0] for item in states]
        expansions += 1
        for neighbour, weight in graph.get(node, []):
            neighbour = int(neighbour)
            region = int(node_fine[neighbour])
            next_progresses = []
            if region == int(corridor[progress]):
                next_progresses.append(progress)
            if progress < goal and region == int(corridor[progress + 1]):
                next_progresses.append(progress + 1)
            for next_progress in set(next_progresses):
                next_state = (neighbour, next_progress)
                next_cost = cost + float(weight)
                if next_cost >= distance.get(next_state, math.inf):
                    continue
                distance[next_state] = next_cost
                parent[next_state] = state
                heuristic = float(np.linalg.norm(coords[neighbour] - coords[int(target)]))
                heapq.heappush(queue, (next_cost + heuristic, next_cost, next_state))
    return None


def synthesize(
    requests: list[tuple[int, int, int]],
    coarse_kernels: np.ndarray,
    fine_kernels: np.ndarray,
    graph: dict,
    coords: np.ndarray,
    node_coarse: np.ndarray,
    node_fine: np.ndarray,
    fine_coarse_compatible: np.ndarray,
    coarse_to_macro: np.ndarray,
    coarse_adjacency: dict[int, set[int]],
    fine_adjacency: dict[int, set[int]],
    *,
    coarse_slack: int,
    fine_slack: int,
    max_coarse_horizon: int,
    max_fine_horizon: int,
    max_expansions: int,
    rng: np.random.Generator,
    label: str,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 250,
) -> tuple[list[np.ndarray], dict]:
    output = []
    coarse_fallback = fine_fallback = road_fallback = 0
    route_cache = {}
    start = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get("version") != 1 or checkpoint.get("request_count") != len(requests):
            raise RuntimeError(f"Incompatible synthesis checkpoint: {checkpoint_path}")
        output = checkpoint["output"]
        start = len(output)
        coarse_fallback = int(checkpoint["coarse_fallback"])
        fine_fallback = int(checkpoint["fine_fallback"])
        road_fallback = int(checkpoint["road_fallback"])
        route_cache = checkpoint["route_cache"]
        rng.bit_generator.state = checkpoint["rng_state"]
        print(f"[{label}] resumed checkpoint at {start}/{len(requests)}", flush=True)
    for index, (source, target, length) in enumerate(requests[start:], start=start + 1):
        source_fine, target_fine = int(node_fine[int(source)]), int(node_fine[int(target)])
        source_coarse, target_coarse = int(node_coarse[int(source)]), int(node_coarse[int(target)])
        origin_macro = int(coarse_to_macro[source_coarse])
        destination_macro = int(coarse_to_macro[target_coarse])
        coarse_family = origin_macro * len(set(coarse_to_macro.tolist())) + destination_macro
        coarse_shortest = flow.region_path(coarse_adjacency, source_coarse, target_coarse)
        coarse_horizon = min(len(coarse_shortest) - 1 + int(coarse_slack), int(max_coarse_horizon))
        coarse_sequence = bridge_raw(
            coarse_kernels, source_coarse, target_coarse, coarse_family, coarse_horizon, rng
        )
        if coarse_sequence is None:
            coarse_sequence = coarse_shortest
            coarse_fallback += 1
        fine_sequence = fine_product_bridge(
            fine_kernels,
            source_fine,
            target_fine,
            origin_macro,
            coarse_sequence,
            fine_coarse_compatible,
            fine_adjacency,
            slack=fine_slack,
            max_horizon=max_fine_horizon,
            rng=rng,
        )
        if fine_sequence is None:
            fine_sequence = flow.region_path(fine_adjacency, source_fine, target_fine)
            fine_fallback += 1
        key = (int(source), int(target), tuple(fine_sequence))
        if key not in route_cache:
            route_cache[key] = lift_repeated_corridor(
                graph,
                coords,
                node_fine,
                source,
                target,
                fine_sequence,
                max_expansions=max_expansions,
            )
        nodes = route_cache[key]
        if not nodes:
            road_fallback += 1
            nodes = flow.astar_flow_path(
                graph,
                coords,
                node_coarse,
                {},
                source,
                target,
                None,
                None,
                sectors=4,
                phases=3,
                eta=0.0,
                max_expansions=max_expansions,
            )
        if not nodes:
            nodes = [int(source), int(target)]
        output.append(route.resample(coords[np.asarray(nodes, dtype=int)], max(2, int(length))))
        if checkpoint_path is not None and (
            index % max(1, int(checkpoint_every)) == 0 or index == len(requests)
        ):
            atomic_pickle_dump(
                {
                    "version": 1,
                    "request_count": len(requests),
                    "output": output,
                    "coarse_fallback": coarse_fallback,
                    "fine_fallback": fine_fallback,
                    "road_fallback": road_fallback,
                    "route_cache": route_cache,
                    "rng_state": rng.bit_generator.state,
                },
                checkpoint_path,
            )
        if index % 100 == 0 or index == len(requests):
            print(
                f"[{label}] {index}/{len(requests)} coarse_fb={coarse_fallback} "
                f"fine_fb={fine_fallback} road_fb={road_fallback}",
                flush=True,
            )
    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)
    return output, {
        "coarse_fallback": int(coarse_fallback),
        "fine_fallback": int(fine_fallback),
        "road_fallback": int(road_fallback),
        "unique_routes": int(len(route_cache)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--public-slot-count", type=int, required=True)
    parser.add_argument("--n-gen", type=int, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--development-mode", action="store_true")
    parser.add_argument("--certified-discrete-noise", action="store_true")
    parser.add_argument("--production-release", action="store_true")
    parser.add_argument("--coarse-regions", type=int, default=24)
    parser.add_argument("--fine-regions", type=int, default=96)
    parser.add_argument("--macros", type=int, default=4)
    parser.add_argument("--phases", type=int, default=3)
    parser.add_argument("--fine-landmarks", type=int, default=256)
    parser.add_argument("--epsilon-route", type=float, default=0.20)
    parser.add_argument("--coarse-alpha", type=float, default=0.70)
    parser.add_argument("--fine-alpha", type=float, default=0.55)
    parser.add_argument("--coarse-slack", type=int, default=4)
    parser.add_argument("--fine-slack", type=int, default=5)
    parser.add_argument("--max-coarse-horizon", type=int, default=14)
    parser.add_argument("--max-fine-horizon", type=int, default=64)
    parser.add_argument("--max-expansions", type=int, default=250000)
    parser.add_argument("--public-bank-size", type=int, default=1)
    parser.add_argument("--bank-only", action="store_true")
    parser.add_argument("--candidate-bank-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.epsilon_route) or float(args.epsilon_route) <= 0.0:
        raise ValueError("--epsilon-route must be finite and strictly positive")
    if int(args.public_slot_count) <= 0 or int(args.n_gen) < 0:
        raise ValueError("public slot count must be positive and n-gen must be nonnegative")
    if args.seed is not None and not args.development_mode:
        raise ValueError("--seed is permitted only with --development-mode")
    if args.production_release:
        expected = {
            "public_slot_count": 17123,
            "n_gen": 17123,
            "coarse_regions": 24,
            "fine_regions": 96,
            "macros": 4,
            "phases": 3,
            "fine_landmarks": 256,
            "public_bank_size": 4,
            "coarse_slack": 4,
            "fine_slack": 5,
            "max_coarse_horizon": 14,
            "max_fine_horizon": 64,
            "max_expansions": 250000,
        }
        actual = {name: int(getattr(args, name)) for name in expected}
        if actual != expected:
            raise ValueError(f"Production configuration mismatch: expected {expected}, got {actual}")
        if Fraction(str(float(args.epsilon_route))) != Fraction(1, 5):
            raise ValueError("Production route epsilon must be the exact rational 1/5")
        if float(args.coarse_alpha) != 0.70 or float(args.fine_alpha) != 0.55:
            raise ValueError("Production shrinkage parameters must match the frozen specification")
        if args.development_mode or args.seed is not None or not args.certified_discrete_noise:
            raise ValueError("Production mode requires exact noise, no seed, and no development mode")
        if args.bank_only or not args.candidate_bank_only:
            raise ValueError("Production mode requires --candidate-bank-only and forbids --bank-only resume")
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise RuntimeError("Production output directory must be new and empty (fail-closed)")
    seed = int(args.seed) if args.seed is not None else secrets.randbits(128)
    rng = np.random.default_rng(seed)
    real = support.load_trajectories(support.DEFAULT_REAL)
    if len(real) > int(args.public_slot_count):
        raise RuntimeError("Private records exceed fixed public capacity")
    osm_path = ARA.parent / "ara_final" / "evidence" / "tables" / "osm_cache_beijing.pkl"
    osm = load_osm_ways(osm_path)
    coords, graph = route.prepare_graph([], bbox=support.BBOX, osm_ways=osm, raw_graph=False)
    context, public_diag = build_context(
        coords, graph, args.coarse_regions, args.fine_regions, args.macros, args.phases
    )
    node_coarse = np.asarray(context.node_coarse, dtype=int)
    node_fine = np.asarray(context.node_fine, dtype=int)
    fine_coarse_compatible = np.zeros((args.fine_regions, args.coarse_regions), dtype=bool)
    fine_coarse_compatible[node_fine, node_coarse] = True
    coarse_edges, _, _, coarse_prior, _ = flow.region_graph_and_prior(
        graph, coords, node_coarse, args.coarse_regions
    )
    fine_edges, _, _, fine_prior, _ = flow.region_graph_and_prior(
        graph, coords, node_fine, args.fine_regions
    )
    exact_rng = random.Random(seed + 7919) if args.development_mode else random.SystemRandom()
    route_sampler = {"mechanism": "NumPy floating Laplace"}
    noisy_route_integer = None
    if args.certified_discrete_noise:
        route_exact, observed_norm = aggregate_route_query_integer(real, context)
        released, route_sampler, noisy_route_integer = release_route_query_integer(
            route_exact,
            epsilon=args.epsilon_route,
            public_capacity=args.public_slot_count,
            exact_rng=exact_rng,
        )
    else:
        route_exact, observed_norm = aggregate_route_query(real, context)
        released = release_route_query(
            route_exact,
            epsilon=args.epsilon_route,
            public_capacity=args.public_slot_count,
            rng=rng,
        )
    coarse_public_base = public_kernel(coarse_edges, coarse_prior, args.coarse_regions, 0.15)
    fine_public_base = public_kernel(fine_edges, fine_prior, args.fine_regions, 0.10)
    coarse_families = args.macros * args.macros
    fine_families = args.macros
    public_coarse = np.broadcast_to(
        coarse_public_base, (coarse_families, args.phases, args.coarse_regions, args.coarse_regions)
    ).copy()
    public_fine = np.broadcast_to(
        fine_public_base, (fine_families, args.phases, args.fine_regions, args.fine_regions)
    ).copy()
    coarse_dp = conditional_kernels(
        released["coarse_occupancy"],
        released["coarse_transition"],
        coarse_edges,
        coarse_public_base,
        alpha=snr_shrinkage_alpha(
            released["coarse_occupancy"],
            released["coarse_transition"],
            epsilon=args.epsilon_route,
            maximum_alpha=args.coarse_alpha,
        ),
        stay=expected_dwell_stay(released["coarse_dwell"]),
    )
    fine_dp = conditional_kernels(
        released["fine_occupancy"],
        released["fine_transition"],
        fine_edges,
        fine_public_base,
        alpha=snr_shrinkage_alpha(
            released["fine_occupancy"],
            released["fine_transition"],
            epsilon=args.epsilon_route,
            maximum_alpha=args.fine_alpha,
        ),
        stay=0.10,
    )

    # Reuse the existing trajectory-level DP support mechanism and requests.
    fine_nodes = support.farthest_point_landmarks(coords, args.fine_landmarks)
    endpoint_coords = coords[np.asarray(fine_nodes, dtype=int)]
    endpoint_tree = cKDTree(endpoint_coords)
    coarse_landmark_nodes = support.farthest_point_landmarks(coords, args.coarse_regions)
    coarse_coords = coords[np.asarray(coarse_landmark_nodes, dtype=int)]
    coarse_tree = cKDTree(coarse_coords)
    fine_to_coarse = np.asarray(coarse_tree.query(endpoint_coords)[1], dtype=int)
    coarse_norm = support.standardized_xy(coarse_coords)
    base_measurements = base.fit_base_measurements(
        real,
        endpoint_tree,
        coarse_tree,
        coarse_norm,
        fine_count=args.fine_landmarks,
        coarse_count=args.coarse_regions,
        eps_endpoint=0.32,
        eps_count=0.04,
        eps_od=0.33,
        eps_joint=0.12,
        eps_length=0.19,
        rng=rng,
        certified_discrete=args.certified_discrete_noise,
        exact_rng=exact_rng,
    )
    base_measurements["od"], od_diag = base.sinkhorn_od_projection(
        base_measurements["od"], base_measurements["endpoint"], fine_to_coarse, args.coarse_regions
    )
    requests = base.sample_requests(
        base_measurements,
        fine_to_coarse,
        coarse_norm,
        fine_nodes,
        count=args.n_gen,
        rng=rng,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    route_release_path = args.out_dir / "route_release.npz"
    atomic_route_release_dump(released, route_release_path)
    integer_route_release_path = None
    if noisy_route_integer is not None:
        integer_route_release_path = args.out_dir / "route_release_integer_noisy.npz"
        atomic_route_release_dump(noisy_route_integer, integer_route_release_path)
    candidate_zero_path = args.out_dir / "public_candidate_00.pkl"
    if args.bank_only and candidate_zero_path.exists():
        with candidate_zero_path.open("rb") as handle:
            public_release = pickle.load(handle)
        public_run_diag = {"resumed": True, "candidate": 0}
        print(f"[public-bank-00] resumed {len(public_release)} trajectories", flush=True)
    else:
        public_release, public_run_diag = synthesize(
            requests,
            public_coarse,
            public_fine,
            graph,
            coords,
            node_coarse,
            node_fine,
            fine_coarse_compatible,
            context.coarse_to_macro,
            context.coarse_adjacency,
            context.fine_adjacency,
            coarse_slack=args.coarse_slack,
            fine_slack=args.fine_slack,
            max_coarse_horizon=args.max_coarse_horizon,
            max_fine_horizon=args.max_fine_horizon,
            max_expansions=args.max_expansions,
            rng=np.random.default_rng(seed + 2001),
            label="public-two-level",
            checkpoint_path=(
                None if args.production_release else args.out_dir / "public_candidate_00.checkpoint.pkl"
            ),
        )
        atomic_pickle_dump(public_release, candidate_zero_path)
    public_bank = [public_release]
    public_bank_diag = [public_run_diag]
    for replicate in range(1, int(args.public_bank_size)):
        candidate_path = args.out_dir / f"public_candidate_{replicate:02d}.pkl"
        if args.bank_only and candidate_path.exists():
            with candidate_path.open("rb") as handle:
                candidate = pickle.load(handle)
            candidate_diag = {"resumed": True, "candidate": int(replicate)}
            print(f"[public-bank-{replicate:02d}] resumed {len(candidate)} trajectories", flush=True)
        else:
            candidate, candidate_diag = synthesize(
                requests,
                public_coarse,
                public_fine,
                graph,
                coords,
                node_coarse,
                node_fine,
                fine_coarse_compatible,
                context.coarse_to_macro,
                context.coarse_adjacency,
                context.fine_adjacency,
                coarse_slack=args.coarse_slack,
                fine_slack=args.fine_slack,
                max_coarse_horizon=args.max_coarse_horizon,
                max_fine_horizon=args.max_fine_horizon,
                max_expansions=args.max_expansions,
                rng=np.random.default_rng(seed + 2001 + replicate),
                label=f"public-bank-{replicate:02d}",
                checkpoint_path=(
                    None
                    if args.production_release
                    else args.out_dir / f"public_candidate_{replicate:02d}.checkpoint.pkl"
                ),
            )
            atomic_pickle_dump(candidate, candidate_path)
        public_bank.append(candidate)
        public_bank_diag.append(candidate_diag)

    coarse_dp_release = fine_dp_release = dp_release = None
    coarse_dp_diag = fine_dp_diag = dp_run_diag = None
    candidate_only = bool(args.bank_only or args.candidate_bank_only)
    if not candidate_only:
        coarse_dp_release, coarse_dp_diag = synthesize(
        requests,
        coarse_dp,
        public_fine,
        graph,
        coords,
        node_coarse,
        node_fine,
        fine_coarse_compatible,
        context.coarse_to_macro,
        context.coarse_adjacency,
        context.fine_adjacency,
        coarse_slack=args.coarse_slack,
        fine_slack=args.fine_slack,
        max_coarse_horizon=args.max_coarse_horizon,
        max_fine_horizon=args.max_fine_horizon,
        max_expansions=args.max_expansions,
        rng=np.random.default_rng(seed + 2001),
        label="dp-coarse-public-fine",
        )
        fine_dp_release, fine_dp_diag = synthesize(
        requests,
        public_coarse,
        fine_dp,
        graph,
        coords,
        node_coarse,
        node_fine,
        fine_coarse_compatible,
        context.coarse_to_macro,
        context.coarse_adjacency,
        context.fine_adjacency,
        coarse_slack=args.coarse_slack,
        fine_slack=args.fine_slack,
        max_coarse_horizon=args.max_coarse_horizon,
        max_fine_horizon=args.max_fine_horizon,
        max_expansions=args.max_expansions,
        rng=np.random.default_rng(seed + 2001),
        label="public-coarse-dp-fine",
        )
        dp_release, dp_run_diag = synthesize(
        requests,
        coarse_dp,
        fine_dp,
        graph,
        coords,
        node_coarse,
        node_fine,
        fine_coarse_compatible,
        context.coarse_to_macro,
        context.coarse_adjacency,
        context.fine_adjacency,
        coarse_slack=args.coarse_slack,
        fine_slack=args.fine_slack,
        max_coarse_horizon=args.max_coarse_horizon,
        max_fine_horizon=args.max_fine_horizon,
        max_expansions=args.max_expansions,
        rng=np.random.default_rng(seed + 2001),
        label="dp-two-level",
        )
    paths = {"public": args.out_dir / "two_level_public.pkl"}
    values_to_write = [("public", public_release)]
    for replicate, candidate in enumerate(public_bank):
        name = f"public_candidate_{replicate:02d}"
        paths[name] = args.out_dir / f"{name}.pkl"
        values_to_write.append((name, candidate))
    if not candidate_only:
        paths.update(
            {
                "dp_coarse_public_fine": args.out_dir / "two_level_dp_coarse.pkl",
                "public_coarse_dp_fine": args.out_dir / "two_level_dp_fine.pkl",
                "dp": args.out_dir / "two_level_dp.pkl",
            }
        )
        values_to_write.extend(
            [
                ("dp_coarse_public_fine", coarse_dp_release),
                ("public_coarse_dp_fine", fine_dp_release),
                ("dp", dp_release),
            ]
        )
    for name, values in values_to_write:
        atomic_pickle_dump(values, paths[name])
    candidate_hashes = {
        path.name: sha256_file(path)
        for name, path in paths.items()
        if name.startswith("public_candidate_")
    }
    manifest = {
        "schema_version": 1,
        "certified_release": bool(args.production_release),
        "development_artifact": bool(args.development_mode),
        "mechanism": "DP-GSRT finite-support route mechanism",
        "candidate_provenance": "post-processing of the same base-DP requests using only fixed public OSM",
        "public_osm": {"path": str(osm_path.resolve()), "sha256": sha256_file(osm_path)},
        "privacy_lineage": {
            "adjacency": "one preprocessed complete trajectory add/remove against a null record",
            "base_epsilon": 1.0,
            "route_epsilon": float(args.epsilon_route),
            "delta": 0.0,
            "route_release_sha256": sha256_file(route_release_path),
            "integer_route_release_sha256": (
                sha256_file(integer_route_release_path) if integer_route_release_path is not None else None
            ),
            "noise_implementation": (
                "bit-exact two-sided geometric"
                if args.certified_discrete_noise
                else "NumPy floating Laplace development mechanism"
            ),
        },
        "configuration": {
            "public_capacity": int(args.public_slot_count),
            "output_slots": int(args.n_gen),
            "coarse_regions": int(args.coarse_regions),
            "fine_regions": int(args.fine_regions),
            "macros": int(args.macros),
            "phases": int(args.phases),
            "candidate_count": int(args.public_bank_size),
            "route_integer_lattice": int(ROUTE_LATTICE) if args.certified_discrete_noise else None,
            "route_block_integer_budgets": WEIGHT_UNITS if args.certified_discrete_noise else None,
            "epsilon_route_rational": "1/5" if args.certified_discrete_noise else None,
            "base_epsilon_rationals": {
                "endpoint": "8/25",
                "count": "1/25",
                "od": "33/100",
                "joint": "3/25",
                "length": "19/100",
            },
            "fine_landmarks": int(args.fine_landmarks),
            "coarse_alpha": float(args.coarse_alpha),
            "fine_alpha": float(args.fine_alpha),
            "coarse_slack": int(args.coarse_slack),
            "fine_slack": int(args.fine_slack),
            "max_coarse_horizon": int(args.max_coarse_horizon),
            "max_fine_horizon": int(args.max_fine_horizon),
            "max_expansions": int(args.max_expansions),
        },
        "candidate_sha256": candidate_hashes,
        "implementation_sha256": {
            "release": sha256_file(Path(__file__).resolve()),
            "base_query": sha256_file((Path(__file__).parent / "dp_graph_voronoi_release.py").resolve()),
            "route_query": sha256_file((Path(__file__).parent / "two_level_semimarkov_query.py").resolve()),
            "exact_sampler": sha256_file((Path(__file__).parent / "certified_discrete_dp.py").resolve()),
        },
    }
    (args.out_dir / "mechanism_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8"
    )
    protocol = {
        "algorithm": "two-level OD-conditioned semi-Markov Schrodinger bridge",
        "development_artifact": bool(args.development_mode),
        "certified_release": bool(args.production_release),
        "adjacency": "one complete trajectory add/remove against null in fixed public capacity",
        "protected_unit": "one complete trajectory",
        "public_capacity": int(args.public_slot_count),
        "public_output_slots": int(args.n_gen),
        "epsilon": {"base": 1.0, "route": float(args.epsilon_route), "total": 1.0 + float(args.epsilon_route), "delta": 0.0},
        "route_query": {"weights": WEIGHTS, "theoretical_l1_sensitivity": 1.0},
        "route_noise_sampler": route_sampler,
        "base_noise_sampler": base_measurements.get("noise_sampler"),
        "sanitized_route_release": route_release_path.name,
        "sanitized_integer_route_release": (
            integer_route_release_path.name if integer_route_release_path is not None else None
        ),
        "public_graph": public_diag,
        "od_projection": od_diag,
        "postprocessed_development_diagnostics": {
            "public": public_run_diag,
            "public_bank": public_bank_diag,
            "dp_coarse_public_fine": coarse_dp_diag,
            "public_coarse_dp_fine": fine_dp_diag,
            "dp": dp_run_diag,
        },
        "certification_blocker": (
            None
            if args.production_release
            else (
                "development mode uses deterministic pseudorandom bits"
                if args.certified_discrete_noise
                else "NumPy floating-point Laplace sampler is not production-certified; development mode also uses a fixed seed"
            )
        ),
        "not_released": ["raw valid count", "observed contribution norm", "runtime"],
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({name: str(path.resolve()) for name, path in paths.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
