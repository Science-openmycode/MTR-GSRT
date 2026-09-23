"""NON-DP development probe for a quotient randomized-shortest-path bridge."""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import pickle
import random
import shutil
from collections import Counter, OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

import block_shrinkage_residual_projection as ners
import compact_graph_flow_experiment as compact_graph_flow
import coarsefine_full_oracle_ceiling as oracle
import distance_coarse_route_release_development as distance_release
import endpoint_coarse_route_release_development as endpoint_release
import graph_cycle_dp_production_sanitizer as lineage
import graph_voronoi_doptimal_support_probe as support
import mass_calibrated_full_development as mass_runner
import multires_graph_flow_experiment as legacy
import nested_quotient_graph as nested_graph
import portal_fiber_qrsp_production_gate as production_gate
import portal_fiber_route_release_development as portal_release
import relative_direction_route_release_development as relative_release
import route_structure_potential_experiment as route
from assemble_graph_cycle_production_release import load_verified_candidates
from audit_two_level_semimarkov_dp import build_context
from public_utils import load_osm_ways


DEFAULT_BETAS = (0.03, 0.06, 0.12, 0.24, 0.48, 0.96)
proposal_logit_adjuster = None
candidate_path_augmenter = None
selected_path_acceptor = None
selected_path_context_initializer = None
GRAPH_FLOW_RELEASE = None
ACTIVE_DWELL_TARGET_COST_OFFSET = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path)
    parser.add_argument("--public-matrix-cache", type=Path)
    parser.add_argument("--dp-development-dir", type=Path)
    parser.add_argument("--portal-release-dir", type=Path)
    parser.add_argument("--fallback-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilon-route", type=float, default=0.20)
    parser.add_argument(
        "--route-release-schema",
        choices=(
            "od32-fine96", "distance4-coarse24", "endpoint8-coarse24",
            "relative32-nested384",
            "portal-fiber-nested384",
        ),
        default="od32-fine96",
    )
    parser.add_argument("--regions", type=int, choices=(96, 384), default=96)
    parser.add_argument(
        "--reference-source",
        choices=("dp-flow", "public-only"),
        default="dp-flow",
        help="Use the released graph flow or the same-support public OSM kernel.",
    )
    parser.add_argument(
        "--reference-conditioning",
        choices=(
            "global", "od-family", "od-factorized", "od-lifted", "endpoint-lifted",
            "relative-direction",
            "portal-fiber",
        ),
        default="global",
        help="Use one global kernel, one marginal OD kernel, or a distance-bearing factorized kernel.",
    )
    parser.add_argument(
        "--od-family-source",
        choices=("ners", "released"),
        default="ners",
        help="Use NERS-shrunk or directly sanitized nonnegative OD-family flow.",
    )
    parser.add_argument(
        "--od-family-granularity",
        choices=("distance-bearing", "bearing", "distance"),
        default="distance-bearing",
        help="Post-process 32 released OD families at the selected fixed granularity.",
    )
    parser.add_argument(
        "--od-likelihood-ratio-cap",
        type=float,
        default=100.0,
        help="Public information-geometric trust-region cap relative to the global kernel.",
    )
    parser.add_argument("--portal-count", type=int, default=6)
    parser.add_argument("--max-region-steps", type=int, default=192)
    parser.add_argument("--length-proposals", type=int, default=1)
    parser.add_argument("--length-log-penalty", type=float, default=8.0)
    parser.add_argument("--active-hierarchy", action="store_true")
    parser.add_argument("--epsilon-graph-flow", type=float)
    parser.add_argument("--occupancy-strength", type=float, default=1.0)
    parser.add_argument("--dwell-strength", type=float, default=1.0)
    parser.add_argument("--hierarchy-likelihood-ratio-cap", type=float, default=100.0)
    parser.add_argument(
        "--route-alignment",
        choices=("correct-dp", "shuffled-dp", "public-only"),
        default="correct-dp",
        help="Matched attribution arm; changes only the semantic alignment of the released q5 route block.",
    )
    parser.add_argument(
        "--decoder-seed",
        type=int,
        help="Data-independent shared decoder random stream for matched research evaluation.",
    )
    parser.add_argument(
        "--request-seed",
        type=int,
        help="Fixed post-processing request seed; required for matched research evaluation.",
    )
    parser.add_argument(
        "--public-fallback-cache",
        type=Path,
        help="Optional fixed public shortest-path cache shared only by matched arms.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--betas",
        type=str,
        default=",".join(str(value) for value in DEFAULT_BETAS),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--acknowledge-non-dp-development", action="store_true")
    mode.add_argument("--production-postprocess", action="store_true")
    return parser.parse_args()


def metric_distance_km(first: np.ndarray, second: np.ndarray) -> float:
    latitude = float(0.5 * (first[0] + second[0]))
    scale = np.asarray([111.32, 111.32 * math.cos(math.radians(latitude))])
    return float(np.linalg.norm((np.asarray(second) - np.asarray(first)) * scale))


def write_atomic_pickle(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def align_released_route_blocks(released: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray]:
    """Change only the separately released q5 Portal-Fiber coordinates."""
    aligned = {name: np.asarray(values, dtype=float).copy() for name, values in released.items()}
    if mode == "correct-dp":
        return aligned
    if mode not in {"shuffled-dp", "public-only"}:
        raise ValueError(f"Unknown route alignment mode: {mode}")
    if "portal_fiber_flow" not in aligned:
        raise ValueError("Matched attribution requires the q5 portal_fiber_flow block")
    alignment_rng = np.random.default_rng(0x5135)
    array = aligned["portal_fiber_flow"]
    flat = array.ravel()
    if mode == "shuffled-dp":
        changed = flat[alignment_rng.permutation(flat.size)]
    else:
        changed = np.full(flat.size, float(flat.mean()) if flat.size else 0.0)
    aligned["portal_fiber_flow"] = changed.reshape(array.shape)
    return aligned


def quotient_public_mass(graph: dict, labels: np.ndarray, edge_index: dict) -> np.ndarray:
    mass = np.zeros(len(edge_index), dtype=float)
    for source, adjacency in graph.items():
        a = int(labels[int(source)])
        for target, weight in adjacency:
            b = int(labels[int(target)])
            index = edge_index.get((a, b))
            if index is not None and a != b:
                mass[int(index)] += 1.0 / max(float(weight), 1e-12)
    positive = mass[mass > 0.0]
    if positive.size:
        mass /= float(np.median(positive))
    return mass


def reference_kernel(
    flow: np.ndarray,
    graph: dict,
    labels: np.ndarray,
    edge_index: dict,
    centers: np.ndarray,
    *,
    use_released_flow: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    regions = len(centers)
    public_mass = quotient_public_mass(graph, labels, edge_index)
    transition = np.zeros((regions, regions), dtype=float)
    cost = np.zeros((regions, regions), dtype=float)
    flow = np.maximum(np.asarray(flow, dtype=float).ravel(), 0.0)
    if len(flow) != len(edge_index):
        raise ValueError("Fine-flow block and quotient edge index disagree")
    positive_flow = flow[flow > 0.0]
    support_floor = (
        max(float(np.median(positive_flow)) * 1e-9, 1e-12)
        if positive_flow.size
        else 1e-12
    )
    for (source, target), index in edge_index.items():
        if int(source) == int(target):
            continue
        public_weight = max(float(public_mass[int(index)]), 1e-12)
        if use_released_flow:
            # The public term prevents support loss without inspecting the raw database.
            weight = float(flow[int(index)]) + support_floor * public_weight
        else:
            weight = public_weight
        transition[int(source), int(target)] = weight
        cost[int(source), int(target)] = max(
            metric_distance_km(centers[int(source)], centers[int(target)]), 1e-4
        )
    row_sum = transition.sum(axis=1)
    missing = np.flatnonzero(row_sum <= 0.0)
    if len(missing):
        raise RuntimeError(f"Quotient reference kernel has empty rows: {missing.tolist()}")
    transition /= row_sum[:, None]
    checks = {
        "regions": regions,
        "directed_edges": int(np.count_nonzero(transition)),
        "row_sum_max_abs_error": float(np.max(np.abs(transition.sum(axis=1) - 1.0))),
        "minimum_positive_probability": float(transition[transition > 0.0].min()),
        "minimum_positive_cost_km": float(cost[cost > 0.0].min()),
    }
    return transition, cost, checks


def _normalized_positive(values: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(values, dtype=float).ravel(), 0.0)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError("Released hierarchy block must be finite and nonempty")
    floor = max(float(result.sum()), 1.0) * 1e-12
    result += floor
    result /= float(result.sum())
    return result


def _lift_region_distribution(
    distribution: np.ndarray,
    fine_to_parent: np.ndarray,
    public_fine_mass: np.ndarray,
) -> np.ndarray:
    """Lift a parent distribution without inventing within-parent private detail."""
    parent = np.asarray(fine_to_parent, dtype=int)
    parent_distribution = _normalized_positive(distribution)
    public_mass = np.maximum(np.asarray(public_fine_mass, dtype=float).ravel(), 0.0)
    if (
        len(parent) != len(public_mass)
        or np.any(parent < 0)
        or np.any(parent >= len(parent_distribution))
    ):
        raise ValueError("Hierarchy parent map and released distribution disagree")
    result = np.zeros(len(parent), dtype=float)
    for group in range(len(parent_distribution)):
        children = np.flatnonzero(parent == group)
        if not len(children):
            continue
        weights = public_mass[children]
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(children), dtype=float)
        result[children] = float(parent_distribution[group]) * weights / float(weights.sum())
    return _normalized_positive(result)


def _node_information_projection(
    kernel: np.ndarray,
    fine_target: np.ndarray,
    lifted_targets: tuple[np.ndarray, ...],
    weights: tuple[float, ...],
    strength: float,
    likelihood_ratio_cap: float,
) -> tuple[np.ndarray, dict]:
    """Tilt a positive public-support kernel by multiscale released occupancy."""
    first = np.asarray(kernel, dtype=float)
    targets = (_normalized_positive(fine_target),) + tuple(
        _normalized_positive(value) for value in lifted_targets
    )
    coefficients = np.asarray(weights, dtype=float)
    cap = float(likelihood_ratio_cap)
    eta = float(strength)
    if (
        first.ndim != 2
        or first.shape[0] != first.shape[1]
        or any(len(value) != len(first) for value in targets)
        or len(coefficients) != len(targets)
        or np.any(coefficients < 0.0)
        or not np.isclose(float(coefficients.sum()), 1.0)
        or not np.isfinite(eta)
        or eta < 0.0
        or not np.isfinite(cap)
        or cap < 1.0
    ):
        raise ValueError("Multiscale occupancy projection inputs are invalid")
    public = np.maximum(np.asarray(first.sum(axis=0), dtype=float), 0.0)
    public = _normalized_positive(public)
    log_cap = math.log(cap)
    potential = np.zeros(len(first), dtype=float)
    for weight, target in zip(coefficients, targets):
        potential += float(weight) * np.clip(
            np.log(target) - np.log(public), -log_cap, log_cap
        )
    rows, columns = np.nonzero(first > 0.0)
    output = np.zeros_like(first)
    output[rows, columns] = first[rows, columns] * np.exp(eta * potential[columns])
    row_sum = output.sum(axis=1)
    if np.any(row_sum <= 0.0) or not np.all(np.isfinite(row_sum)):
        raise RuntimeError("Multiscale occupancy projection produced an empty row")
    output /= row_sum[:, None]
    return output, {
        "weights": [float(value) for value in coefficients],
        "strength": eta,
        "likelihood_ratio_cap": cap,
        "maximum_abs_node_potential": float(np.max(np.abs(potential))),
        "row_sum_max_abs_error": float(np.max(np.abs(output.sum(axis=1) - 1.0))),
    }


def active_hierarchy_projection(
    fine_kernel: np.ndarray,
    physical_cost: np.ndarray,
    graph_flow_release: dict[str, np.ndarray],
    q5_blocks: dict[str, np.ndarray],
    graph: dict,
    context96,
    context384,
    fine_to_96: np.ndarray,
    *,
    epsilon_graph_flow: float,
    occupancy_strength: float,
    dwell_strength: float,
    likelihood_ratio_cap: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reconstruct one coherent route law from every released hierarchy block."""
    required_graph = {"fine_occupancy", "fine_flow", "dwell"}
    required_q5 = {
        "coarse24_occupancy", "fine384_occupancy", "fine96_flow", "fine384_flow"
    }
    if set(graph_flow_release) != required_graph or not required_q5.issubset(q5_blocks):
        raise RuntimeError("Active hierarchy release schema mismatch")

    graph_values = {
        name: np.asarray(graph_flow_release[name], dtype=float).ravel()
        for name in compact_graph_flow.BLOCK_BUDGETS
    }
    public_graph_parts = {
        "fine_occupancy": np.bincount(
            context96.node_fine, minlength=int(context96.fine_regions)
        ).astype(float),
        "fine_flow": quotient_public_mass(
            graph, context96.node_fine, context96.fine_edge_index
        ),
        "dwell": np.ones(len(graph_values["dwell"]), dtype=float),
    }

    def scaled_public(values: np.ndarray, total: float) -> np.ndarray:
        normalized = _normalized_positive(values)
        return normalized * float(total)

    graph_target = np.concatenate(
        [graph_values[name] for name in compact_graph_flow.BLOCK_BUDGETS]
    )
    graph_public = np.concatenate(
        [
            scaled_public(public_graph_parts[name], graph_values[name].sum())
            for name in compact_graph_flow.BLOCK_BUDGETS
        ]
    )
    graph_variance = ners.discrete_laplace_variance_normalized(
        epsilon=float(epsilon_graph_flow), sensitivity=1_000_000
    )
    graph_shrunk, graph_shrinkage = ners.positive_part_block_shrinkage(
        graph_target,
        graph_public,
        {name: int(graph_values[name].size) for name in compact_graph_flow.BLOCK_BUDGETS},
        noise_variance=graph_variance,
    )
    graph_offsets, cursor = {}, 0
    for name in compact_graph_flow.BLOCK_BUDGETS:
        graph_offsets[name] = cursor
        cursor += int(graph_values[name].size)
    graph_blocks = {
        name: graph_shrunk[
            graph_offsets[name] : graph_offsets[name] + int(graph_values[name].size)
        ]
        for name in compact_graph_flow.BLOCK_BUDGETS
    }

    coarse_public, coarse_cost, _ = reference_kernel(
        np.zeros(len(context96.fine_edge_index)),
        graph,
        context96.node_fine,
        context96.fine_edge_index,
        np.asarray(context96.fine_tree.data, dtype=float),
        use_released_flow=False,
    )
    graph_kernel, graph_cost, _ = reference_kernel(
        graph_blocks["fine_flow"],
        graph,
        context96.node_fine,
        context96.fine_edge_index,
        np.asarray(context96.fine_tree.data, dtype=float),
        use_released_flow=True,
    )
    q5_coarse_kernel, q5_coarse_cost, _ = reference_kernel(
        q5_blocks["fine96_flow"],
        graph,
        context96.node_fine,
        context96.fine_edge_index,
        np.asarray(context96.fine_tree.data, dtype=float),
        use_released_flow=True,
    )
    if not (
        np.array_equal(coarse_cost, graph_cost)
        and np.array_equal(coarse_cost, q5_coarse_cost)
    ):
        raise RuntimeError("Coarse hierarchy costs disagree")
    graph_budget = float(compact_graph_flow.BLOCK_BUDGETS["fine_flow"])
    q5_budget = 150_000.0
    q5_precision_weight = q5_budget**2 / (graph_budget**2 + q5_budget**2)
    fused_coarse = rowwise_entropic_barycenter(
        graph_kernel,
        q5_coarse_kernel,
        q5_precision_weight,
        likelihood_ratio_cap=float(likelihood_ratio_cap),
    )
    fused_fine = lift_coarse_likelihood_projection(
        fine_kernel,
        coarse_public,
        fused_coarse,
        np.asarray(fine_to_96, dtype=int),
        conditioned_weight=1.0,
        likelihood_ratio_cap=float(likelihood_ratio_cap),
    )

    public_fine_mass = np.bincount(
        context384.node_fine, minlength=int(context384.fine_regions)
    ).astype(float)
    lifted_96 = _lift_region_distribution(
        graph_blocks["fine_occupancy"], fine_to_96, public_fine_mass
    )
    fine_to_24 = np.asarray(
        context96.coarse_tree.query(np.asarray(context384.fine_tree.data, dtype=float))[1],
        dtype=int,
    )
    lifted_24 = _lift_region_distribution(
        q5_blocks["coarse24_occupancy"], fine_to_24, public_fine_mass
    )
    occupancy_budgets = np.asarray(
        [100_000.0, float(compact_graph_flow.BLOCK_BUDGETS["fine_occupancy"]), 100_000.0]
    )
    occupancy_weights = occupancy_budgets / float(occupancy_budgets.sum())
    projected, occupancy_diagnostics = _node_information_projection(
        fused_fine,
        q5_blocks["fine384_occupancy"],
        (lifted_96, lifted_24),
        tuple(float(value) for value in occupancy_weights),
        float(occupancy_strength),
        float(likelihood_ratio_cap),
    )

    dwell_probability = _normalized_positive(graph_blocks["dwell"])
    dwell_edges = np.asarray([0.0, 0.05, 0.15, 0.35, 0.65, 1.0], dtype=float)
    dwell_midpoints = 0.5 * (dwell_edges[:-1] + dwell_edges[1:])
    expected_dwell = float(dwell_probability @ dwell_midpoints)
    target_region_count = float(np.clip(1.0 / max(expected_dwell, 1e-6), 2.0, 192.0))
    positive_cost = np.asarray(physical_cost, dtype=float)
    positive_cost = positive_cost[positive_cost > 0.0]
    median_cost = float(np.median(positive_cost)) if positive_cost.size else 0.0
    hop_penalty = float(dwell_strength) * median_cost * expected_dwell
    adjusted_cost = np.asarray(physical_cost, dtype=float).copy()
    adjusted_cost[adjusted_cost > 0.0] += hop_penalty
    diagnostics = {
        "graph_flow_shrinkage": graph_shrinkage,
        "coarse_flow_precision_weights": {
            "compact_graph_flow": 1.0 - q5_precision_weight,
            "q5_fine96_flow": q5_precision_weight,
        },
        "occupancy": occupancy_diagnostics,
        "dwell_probability": dwell_probability.tolist(),
        "expected_dwell_fraction": expected_dwell,
        "target_region_count": target_region_count,
        "hop_penalty": hop_penalty,
        "active_blocks": [
            "qg.fine_occupancy", "qg.fine_flow", "qg.dwell",
            "q5.coarse24_occupancy", "q5.fine384_occupancy",
            "q5.fine96_flow", "q5.fine384_flow", "q5.portal_fiber_flow",
        ],
        "row_sum_max_abs_error": float(np.max(np.abs(projected.sum(axis=1) - 1.0))),
        "minimum_positive_probability": float(projected[projected > 0.0].min()),
    }
    return projected, adjusted_cost, diagnostics


def rowwise_entropic_barycenter(
    global_kernel: np.ndarray,
    conditioned_kernel: np.ndarray,
    conditioned_weight: float,
    likelihood_ratio_cap: float | None = None,
) -> np.ndarray:
    """Unique KL projection under an optional bounded log-likelihood tilt."""
    first = np.asarray(global_kernel, dtype=float)
    second = np.asarray(conditioned_kernel, dtype=float)
    eta = float(conditioned_weight)
    if first.shape != second.shape or not 0.0 <= eta <= 1.0:
        raise ValueError("Entropic-barycenter kernels or weight are invalid")
    if not np.array_equal(first > 0.0, second > 0.0):
        raise ValueError("Entropic-barycenter kernels must have identical public support")
    output = np.zeros_like(first)
    support = first > 0.0
    log_ratio = np.log(second[support]) - np.log(first[support])
    if likelihood_ratio_cap is not None:
        cap = float(likelihood_ratio_cap)
        if not np.isfinite(cap) or cap < 1.0:
            raise ValueError("Likelihood-ratio cap must be finite and at least one")
        log_cap = math.log(cap)
        log_ratio = np.clip(log_ratio, -log_cap, log_cap)
    output[support] = first[support] * np.exp(eta * log_ratio)
    row_sum = output.sum(axis=1)
    if np.any(row_sum <= 0.0):
        raise RuntimeError("Entropic-barycenter kernel has an empty row")
    output /= row_sum[:, None]
    return output


def rowwise_factorized_information_projection(
    global_kernel: np.ndarray,
    conditioned_kernels: tuple[np.ndarray, ...],
    factor_weights: tuple[float, ...],
    conditioned_weight: float,
    likelihood_ratio_cap: float,
) -> np.ndarray:
    """KL projection under a convex combination of bounded log-likelihood factors."""
    first = np.asarray(global_kernel, dtype=float)
    factors = tuple(np.asarray(value, dtype=float) for value in conditioned_kernels)
    weights = np.asarray(factor_weights, dtype=float)
    eta = float(conditioned_weight)
    cap = float(likelihood_ratio_cap)
    if (
        not factors
        or len(factors) != len(weights)
        or np.any(weights < 0.0)
        or not np.isclose(float(weights.sum()), 1.0)
        or not 0.0 <= eta <= 1.0
        or not np.isfinite(cap)
        or cap < 1.0
    ):
        raise ValueError("Factorized information-projection parameters are invalid")
    support = first > 0.0
    combined = np.zeros(int(np.count_nonzero(support)), dtype=float)
    log_cap = math.log(cap)
    for weight, factor in zip(weights, factors):
        if factor.shape != first.shape or not np.array_equal(factor > 0.0, support):
            raise ValueError("Factorized kernels must have identical public support")
        log_ratio = np.log(factor[support]) - np.log(first[support])
        combined += float(weight) * np.clip(log_ratio, -log_cap, log_cap)
    output = np.zeros_like(first)
    output[support] = first[support] * np.exp(eta * combined)
    row_sum = output.sum(axis=1)
    if np.any(row_sum <= 0.0):
        raise RuntimeError("Factorized information-projection kernel has an empty row")
    output /= row_sum[:, None]
    return output


def relative_direction_information_projection(
    global_kernel: np.ndarray,
    centers: np.ndarray,
    conditioned_relative_mass: np.ndarray,
    global_relative_mass: np.ndarray,
    od_bearing_sector: int,
    conditioned_weight: float,
    likelihood_ratio_cap: float,
) -> np.ndarray:
    """Unique rowwise KL projection under a bounded relative-edge-direction tilt."""
    first = np.asarray(global_kernel, dtype=float)
    centers = np.asarray(centers, dtype=float)
    conditioned = np.maximum(np.asarray(conditioned_relative_mass, dtype=float).ravel(), 0.0)
    baseline = np.maximum(np.asarray(global_relative_mass, dtype=float).ravel(), 0.0)
    eta, cap = float(conditioned_weight), float(likelihood_ratio_cap)
    bins = len(conditioned)
    if (
        first.ndim != 2
        or first.shape[0] != first.shape[1]
        or centers.shape != (len(first), 2)
        or bins < 2
        or len(baseline) != bins
        or not 0 <= int(od_bearing_sector) < 8
        or not 0.0 <= eta <= 1.0
        or not np.isfinite(cap)
        or cap < 1.0
    ):
        raise ValueError("Relative-direction projection inputs are invalid")
    floor = 1e-12
    conditioned = (conditioned + floor) / float((conditioned + floor).sum())
    baseline = (baseline + floor) / float((baseline + floor).sum())
    reward_by_bin = np.clip(
        np.log(conditioned) - np.log(baseline), -math.log(cap), math.log(cap)
    )
    rows, columns = np.nonzero(first > 0.0)
    latitude = 0.5 * (centers[rows, 0] + centers[columns, 0])
    delta = centers[columns] - centers[rows]
    north = delta[:, 0] * 111.32
    east = delta[:, 1] * 111.32 * np.cos(np.deg2rad(latitude))
    edge_angle = np.mod(np.arctan2(north, east), 2.0 * np.pi)
    request_angle = (int(od_bearing_sector) + 0.5) / 8.0 * 2.0 * np.pi
    relative = np.mod(edge_angle - request_angle, 2.0 * np.pi)
    relative_bin = np.minimum((relative / (2.0 * np.pi) * bins).astype(int), bins - 1)
    output = np.zeros_like(first)
    output[rows, columns] = first[rows, columns] * np.exp(
        eta * reward_by_bin[relative_bin]
    )
    row_sum = output.sum(axis=1)
    if np.any(row_sum <= 0.0):
        raise RuntimeError("Relative-direction information projection has an empty row")
    output /= row_sum[:, None]
    return output


def portal_information_projection(
    public_probability: np.ndarray,
    released_portal_mass: np.ndarray,
    conditioned_weight: float,
    likelihood_ratio_cap: float,
) -> np.ndarray:
    """KL projection of a public portal prior under a bounded DP information tilt."""
    prior = np.asarray(public_probability, dtype=float).ravel()
    released = np.maximum(np.asarray(released_portal_mass, dtype=float).ravel(), 0.0)
    eta, cap = float(conditioned_weight), float(likelihood_ratio_cap)
    if (
        prior.size == 0
        or released.shape != prior.shape
        or np.any(prior <= 0.0)
        or not np.isfinite(prior).all()
        or not np.isfinite(released).all()
        or not 0.0 <= eta <= 1.0
        or not np.isfinite(cap)
        or cap < 1.0
    ):
        raise ValueError("Portal information-projection inputs are invalid")
    prior = prior / float(prior.sum())
    floor = max(float(released.sum()), 1.0) * 1e-12
    posterior = (released + floor) / float((released + floor).sum())
    uniform = np.full(prior.size, 1.0 / prior.size)
    reward = np.clip(
        np.log(posterior) - np.log(uniform), -math.log(cap), math.log(cap)
    )
    output = prior * np.exp(eta * reward)
    output /= float(output.sum())
    return output


def lift_coarse_likelihood_projection(
    fine_kernel: np.ndarray,
    coarse_global_kernel: np.ndarray,
    coarse_conditioned_kernel: np.ndarray,
    fine_to_coarse: np.ndarray,
    conditioned_weight: float,
    likelihood_ratio_cap: float,
) -> np.ndarray:
    """Pull a bounded coarse log-likelihood reward back to a fine graph."""
    fine = np.asarray(fine_kernel, dtype=float)
    coarse_global = np.asarray(coarse_global_kernel, dtype=float)
    coarse_conditioned = np.asarray(coarse_conditioned_kernel, dtype=float)
    parent = np.asarray(fine_to_coarse, dtype=int)
    eta, cap = float(conditioned_weight), float(likelihood_ratio_cap)
    if (
        fine.ndim != 2
        or fine.shape[0] != fine.shape[1]
        or coarse_global.shape != coarse_conditioned.shape
        or coarse_global.ndim != 2
        or coarse_global.shape[0] != coarse_global.shape[1]
        or len(parent) != len(fine)
        or np.any(parent < 0)
        or np.any(parent >= len(coarse_global))
        or not np.array_equal(coarse_global > 0.0, coarse_conditioned > 0.0)
        or not 0.0 <= eta <= 1.0
        or not np.isfinite(cap)
        or cap < 1.0
    ):
        raise ValueError("Coarse-to-fine information-projection inputs are invalid")
    coarse_support = coarse_global > 0.0
    coarse_reward = np.zeros_like(coarse_global)
    coarse_reward[coarse_support] = np.clip(
        np.log(coarse_conditioned[coarse_support])
        - np.log(coarse_global[coarse_support]),
        -math.log(cap),
        math.log(cap),
    )
    rows, columns = np.nonzero(fine > 0.0)
    reward = np.zeros(len(rows), dtype=float)
    parent_rows, parent_columns = parent[rows], parent[columns]
    cross = parent_rows != parent_columns
    valid = cross & coarse_support[parent_rows, parent_columns]
    reward[valid] = coarse_reward[parent_rows[valid], parent_columns[valid]]
    output = np.zeros_like(fine)
    output[rows, columns] = fine[rows, columns] * np.exp(eta * reward)
    row_sum = output.sum(axis=1)
    if np.any(row_sum <= 0.0):
        raise RuntimeError("Lifted information-projection kernel has an empty row")
    output /= row_sum[:, None]
    return output


def lift_factorized_coarse_likelihood_projection(
    fine_kernel: np.ndarray,
    coarse_global_kernel: np.ndarray,
    coarse_conditioned_kernels: tuple[np.ndarray, ...],
    fine_to_coarse: np.ndarray,
    factor_weights: tuple[float, ...],
    conditioned_weight: float,
    likelihood_ratio_cap: float,
) -> np.ndarray:
    """Pull a convex combination of bounded coarse rewards back to a fine graph."""
    fine = np.asarray(fine_kernel, dtype=float)
    coarse_global = np.asarray(coarse_global_kernel, dtype=float)
    factors = tuple(np.asarray(value, dtype=float) for value in coarse_conditioned_kernels)
    parent = np.asarray(fine_to_coarse, dtype=int)
    weights = np.asarray(factor_weights, dtype=float)
    eta, cap = float(conditioned_weight), float(likelihood_ratio_cap)
    coarse_support = coarse_global > 0.0
    if (
        not factors
        or len(factors) != len(weights)
        or np.any(weights < 0.0)
        or not np.isclose(float(weights.sum()), 1.0)
        or fine.ndim != 2
        or fine.shape[0] != fine.shape[1]
        or coarse_global.ndim != 2
        or coarse_global.shape[0] != coarse_global.shape[1]
        or len(parent) != len(fine)
        or np.any(parent < 0)
        or np.any(parent >= len(coarse_global))
        or not 0.0 <= eta <= 1.0
        or not np.isfinite(cap)
        or cap < 1.0
        or any(
            factor.shape != coarse_global.shape
            or not np.array_equal(factor > 0.0, coarse_support)
            for factor in factors
        )
    ):
        raise ValueError("Factorized coarse-to-fine projection inputs are invalid")
    coarse_reward = np.zeros_like(coarse_global)
    for weight, factor in zip(weights, factors):
        coarse_reward[coarse_support] += float(weight) * np.clip(
            np.log(factor[coarse_support]) - np.log(coarse_global[coarse_support]),
            -math.log(cap),
            math.log(cap),
        )
    rows, columns = np.nonzero(fine > 0.0)
    parent_rows, parent_columns = parent[rows], parent[columns]
    reward = np.zeros(len(rows), dtype=float)
    valid = (parent_rows != parent_columns) & coarse_support[parent_rows, parent_columns]
    reward[valid] = coarse_reward[parent_rows[valid], parent_columns[valid]]
    output = np.zeros_like(fine)
    output[rows, columns] = fine[rows, columns] * np.exp(eta * reward)
    row_sum = output.sum(axis=1)
    if np.any(row_sum <= 0.0):
        raise RuntimeError("Factorized coarse-to-fine projection has an empty row")
    output /= row_sum[:, None]
    return output


def conservative_factorized_flow_projection(
    fine_flow: np.ndarray,
    fine_edge_index: dict[tuple[int, int], int],
    fine_to_coarse: np.ndarray,
    coarse_edge_index: dict[tuple[int, int], int],
    coarse_global_flow: np.ndarray,
    coarse_conditioned_flows: tuple[np.ndarray, ...],
    factor_weights: tuple[float, ...],
    conditioned_weight: float,
    likelihood_ratio_cap: float,
) -> tuple[np.ndarray, dict]:
    """KL-project fine edge mass so its coarse cross-edge aggregate is exact."""
    fine = np.maximum(np.asarray(fine_flow, dtype=float).ravel(), 0.0)
    parent = np.asarray(fine_to_coarse, dtype=int)
    global_flow = np.maximum(np.asarray(coarse_global_flow, dtype=float).ravel(), 0.0)
    factors = tuple(np.maximum(np.asarray(value, dtype=float).ravel(), 0.0) for value in coarse_conditioned_flows)
    weights = np.asarray(factor_weights, dtype=float)
    eta, cap = float(conditioned_weight), float(likelihood_ratio_cap)
    if (
        len(fine) != len(fine_edge_index)
        or len(global_flow) != len(coarse_edge_index)
        or not factors
        or any(len(value) != len(global_flow) for value in factors)
        or len(weights) != len(factors)
        or np.any(weights < 0.0)
        or not np.isclose(float(weights.sum()), 1.0)
        or not 0.0 <= eta <= 1.0
        or not np.isfinite(cap)
        or cap < 1.0
    ):
        raise ValueError("Conservative factorized flow-projection inputs are invalid")
    positive_fine = fine[fine > 0.0]
    fine_floor = max(float(np.median(positive_fine)) * 1e-9, 1e-12) if positive_fine.size else 1e-12
    base = fine + fine_floor
    positive_coarse = global_flow[global_flow > 0.0]
    coarse_floor = (
        max(float(np.median(positive_coarse)) * 1e-9, 1e-12)
        if positive_coarse.size
        else 1e-12
    )
    global_positive = global_flow + coarse_floor
    reward = np.zeros_like(global_positive)
    log_cap = math.log(cap)
    for weight, factor in zip(weights, factors):
        reward += float(weight) * np.clip(
            np.log(factor + coarse_floor) - np.log(global_positive),
            -log_cap,
            log_cap,
        )
    coarse_target = global_positive * np.exp(eta * reward)
    rows_by_coarse: list[list[int]] = [[] for _ in range(len(global_flow))]
    intra_indices = []
    for (source, target), index in fine_edge_index.items():
        coarse_source, coarse_target_node = int(parent[int(source)]), int(parent[int(target)])
        if coarse_source == coarse_target_node:
            intra_indices.append(int(index))
            continue
        coarse_index = coarse_edge_index.get((coarse_source, coarse_target_node))
        if coarse_index is None:
            raise RuntimeError("Nested fine edge has no public coarse parent edge")
        rows_by_coarse[int(coarse_index)].append(int(index))
    cross_mass = float(sum(base[indices].sum() for indices in rows_by_coarse if indices))
    coarse_target *= cross_mass / max(float(coarse_target.sum()), 1e-15)
    projected = base.copy()
    for coarse_index, indices in enumerate(rows_by_coarse):
        if not indices:
            if coarse_target[coarse_index] > 1e-10:
                raise RuntimeError("Positive coarse target has no nested fine support")
            continue
        current = float(base[indices].sum())
        projected[indices] *= float(coarse_target[coarse_index]) / max(current, 1e-15)
    aggregate = np.asarray(
        [float(projected[indices].sum()) if indices else 0.0 for indices in rows_by_coarse]
    )
    return projected, {
        "cross_mass": cross_mass,
        "intra_fine_edges": int(len(intra_indices)),
        "cross_fine_edges": int(sum(len(indices) for indices in rows_by_coarse)),
        "coarse_aggregation_max_abs_error": float(np.max(np.abs(aggregate - coarse_target))),
    }


def aggregate_od_family_flow(values: np.ndarray, granularity: str) -> np.ndarray:
    full = np.asarray(values, dtype=float).reshape(4, 8, -1)
    if granularity == "distance-bearing":
        return full.reshape(32, -1)
    if granularity == "bearing":
        return full.sum(axis=0)
    if granularity == "distance":
        return full.sum(axis=1)
    raise ValueError(f"Unknown OD-family granularity: {granularity}")


def coarsen_od_family(full_family: int, granularity: str) -> int:
    distance, bearing = divmod(int(full_family), 8)
    if granularity == "distance-bearing":
        return int(full_family)
    if granularity == "bearing":
        return bearing
    if granularity == "distance":
        return distance
    raise ValueError(f"Unknown OD-family granularity: {granularity}")


@dataclass
class BridgeSolution:
    transition: np.ndarray
    expected_cost: np.ndarray
    desirability: np.ndarray
    residual: float


class RSPBridge:
    """Doob transform solving a KL-regularized hitting-path problem."""

    def __init__(
        self,
        reference: np.ndarray,
        cost: np.ndarray,
        betas: tuple[float, ...],
        max_cached_solutions: int = 512,
    ):
        self.reference = np.asarray(reference, dtype=float)
        self.cost = np.asarray(cost, dtype=float)
        self.betas = tuple(float(value) for value in betas)
        self.cache: OrderedDict[tuple[int, float], BridgeSolution] = OrderedDict()
        self.core_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.max_cached_solutions = int(max_cached_solutions)
        self.max_residual = 0.0

    def core(self, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        beta = float(beta)
        if beta in self.core_cache:
            return self.core_cache[beta]
        weighted = self.reference * np.exp(-beta * self.cost)
        fundamental = np.linalg.inv(np.eye(len(weighted)) - weighted)
        derivative_weighted = -self.cost * weighted
        derivative_fundamental = fundamental @ derivative_weighted @ fundamental
        expected = np.empty_like(fundamental)
        for destination in range(len(weighted)):
            column = fundamental[:, destination]
            derivative_column = derivative_fundamental[:, destination]
            derivative_ratio = np.divide(
                derivative_column,
                column,
                out=np.zeros_like(derivative_column),
                where=column != 0.0,
            )
            expected[:, destination] = -(
                derivative_ratio
                - derivative_fundamental[destination, destination]
                / fundamental[destination, destination]
            )
            expected[destination, destination] = 0.0
        expected = np.maximum(expected, 0.0)
        self.core_cache[beta] = weighted, fundamental, expected
        return self.core_cache[beta]

    def solve(self, destination: int, beta: float) -> BridgeSolution:
        key = (int(destination), float(beta))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        destination = int(destination)
        regions = len(self.reference)
        keep = np.asarray([index for index in range(regions) if index != destination])
        weighted, fundamental, expected_matrix = self.core(float(beta))
        desirability = fundamental[:, destination] / fundamental[destination, destination]
        if np.any(desirability[keep] <= 0.0) or not np.all(np.isfinite(desirability)):
            raise RuntimeError("RSP desirability is not finite and strictly positive")
        bridge = weighted * desirability[None, :] / desirability[:, None]
        bridge[destination, :] = 0.0
        row_sum = bridge.sum(axis=1)
        bridge[keep] /= row_sum[keep, None]
        immediate = np.sum(bridge * self.cost, axis=1)
        expected = np.asarray(expected_matrix[:, destination], dtype=float).copy()
        residual = float(
            np.max(
                np.abs(
                    desirability[keep]
                    - weighted[np.ix_(keep, np.arange(regions))] @ desirability
                )
            )
        )
        solution = BridgeSolution(bridge, expected, desirability, residual)
        self.cache[key] = solution
        self.cache.move_to_end(key)
        self.max_residual = max(self.max_residual, residual)
        while len(self.cache) > self.max_cached_solutions:
            self.cache.popitem(last=False)
        return solution

    def choose_beta(self, source: int, destination: int, target_km: float) -> float:
        effective_target = float(target_km) + float(ACTIVE_DWELL_TARGET_COST_OFFSET)
        errors = []
        for beta in self.betas:
            expected = float(self.solve(destination, beta).expected_cost[int(source)])
            errors.append(abs(expected - effective_target))
        return self.betas[int(np.argmin(errors))]

    def sample(
        self,
        source: int,
        destination: int,
        target_km: float,
        rng: np.random.Generator,
        max_steps: int,
        fixed_beta: float | None = None,
    ) -> tuple[list[int] | None, dict]:
        source, destination = int(source), int(destination)
        if source == destination:
            outgoing = np.flatnonzero(self.reference[source] > 0.0)
            if not len(outgoing):
                return None, {"failure": "same-region-without-neighbor"}
            first = int(rng.choice(outgoing, p=self.reference[source, outgoing] / self.reference[source, outgoing].sum()))
            beta = (
                float(fixed_beta)
                if fixed_beta is not None
                else self.choose_beta(first, destination, max(float(target_km), 0.1))
            )
            current, path = first, [source, first]
        else:
            beta = (
                float(fixed_beta)
                if fixed_beta is not None
                else self.choose_beta(source, destination, float(target_km))
            )
            current, path = source, [source]
        solution = self.solve(destination, beta)
        while current != destination and len(path) <= int(max_steps):
            probabilities = solution.transition[current]
            if probabilities.sum() <= 0.0:
                return None, {"failure": "zero-bridge-row", "beta": beta}
            current = int(rng.choice(len(probabilities), p=probabilities))
            path.append(current)
        if current != destination:
            return None, {"failure": "step-cap", "beta": beta, "steps": len(path) - 1}
        quotient_cost = float(
            sum(self.cost[a, b] for a, b in zip(path[:-1], path[1:]))
        )
        return path, {
            "beta": beta,
            "steps": len(path) - 1,
            "quotient_cost_km": quotient_cost,
            "expected_cost_km": float(solution.expected_cost[path[0]]),
            "desirability_residual": solution.residual,
        }


def diverse_portals(
    graph: dict,
    coords: np.ndarray,
    labels: np.ndarray,
    count: int,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    crossings: dict[tuple[int, int], list[tuple[int, int]]] = {}
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
                np.sum((midpoints[:, None, :] - midpoints[np.asarray(chosen)][None, :, :]) ** 2, axis=2),
                axis=1,
            )
            distance[np.asarray(chosen)] = -1.0
            chosen.append(int(np.argmax(distance)))
        selected[key] = [unique[index] for index in chosen]
    return selected


class RegionRouter:
    def __init__(self, graph: dict, labels: np.ndarray, max_cache_entries: int = 100_000):
        self.graph = graph
        self.labels = np.asarray(labels, dtype=int)
        self.cache: OrderedDict[tuple[int, int, int], list[int] | None] = OrderedDict()
        self.max_cache_entries = int(max_cache_entries)

    def _store(self, key: tuple[int, int, int], value: list[int] | None) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_cache_entries:
            self.cache.popitem(last=False)

    def shortest(self, region: int, source: int, destination: int) -> list[int] | None:
        key = (int(region), int(source), int(destination))
        reverse = (key[0], key[2], key[1])
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if source == destination:
            return [int(source)]
        distance = {int(source): 0.0}
        parent: dict[int, int] = {}
        queue = [(0.0, int(source))]
        while queue:
            current_distance, node = heapq.heappop(queue)
            if current_distance != distance.get(node):
                continue
            if node == int(destination):
                break
            for neighbor, weight in self.graph.get(node, []):
                neighbor = int(neighbor)
                if int(self.labels[neighbor]) != int(region):
                    continue
                candidate = current_distance + float(weight)
                if candidate < distance.get(neighbor, float("inf")):
                    distance[neighbor] = candidate
                    parent[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))
        if int(destination) not in distance:
            self._store(key, None)
            return None
        path = [int(destination)]
        while path[-1] != int(source):
            path.append(parent[path[-1]])
        path.reverse()
        self._store(key, path)
        self._store(reverse, list(reversed(path)))
        return path


def append_path(output: list[int], part: list[int]) -> None:
    if not output:
        output.extend(part)
    elif output[-1] == part[0]:
        output.extend(part[1:])
    else:
        output.extend(part)


def lift_quotient_path(
    regions: list[int],
    source: int,
    destination: int,
    graph: dict,
    coords: np.ndarray,
    labels: np.ndarray,
    portals: dict[tuple[int, int], list[tuple[int, int]]],
    router: RegionRouter,
    rng: np.random.Generator,
    portal_probabilities: dict[tuple[int, int], np.ndarray] | None = None,
    portal_weight: float = 7.0 / 13.0,
    portal_likelihood_ratio_cap: float = 100.0,
) -> tuple[list[int] | None, int]:
    current = int(source)
    output = [current]
    fallback_segments = 0
    total_steps = max(len(regions) - 1, 1)
    for step, (a, b) in enumerate(zip(regions[:-1], regions[1:])):
        options = portals.get((int(a), int(b)), [])
        if not options:
            return None, fallback_segments
        fraction = float(step + 1) / float(total_steps + 1)
        target_coord = (1.0 - fraction) * coords[int(source)] + fraction * coords[int(destination)]
        portal_coords = np.asarray([0.5 * (coords[u] + coords[v]) for u, v in options])
        distances = np.linalg.norm(portal_coords - target_coord[None, :], axis=1)
        scale = max(float(np.median(distances)), 1e-12)
        logits = -distances / scale
        probabilities = np.exp(logits - float(logits.max()))
        probabilities /= probabilities.sum()
        if portal_probabilities is not None:
            released_mass = portal_probabilities.get((int(a), int(b)))
            if released_mass is None:
                raise RuntimeError("Released portal layout is missing a public quotient edge")
            probabilities = portal_information_projection(
                probabilities,
                released_mass,
                portal_weight,
                portal_likelihood_ratio_cap,
            )
        u, v = options[int(rng.choice(len(options), p=probabilities))]
        inside = router.shortest(int(a), current, int(u))
        if inside is None:
            inside = route.shortest_path(graph, coords, current, int(u), max_visits=300_000)
            fallback_segments += 1
        if inside is None:
            return None, fallback_segments
        append_path(output, [int(node) for node in inside])
        append_path(output, [int(u), int(v)])
        current = int(v)
    inside = router.shortest(int(regions[-1]), current, int(destination))
    if inside is None:
        inside = route.shortest_path(graph, coords, current, int(destination), max_visits=300_000)
        fallback_segments += 1
    if inside is None:
        return None, fallback_segments
    append_path(output, [int(node) for node in inside])
    compact = [output[0]]
    for node in output[1:]:
        if int(node) != compact[-1]:
            compact.append(int(node))
    return compact, fallback_segments


def road_path_length_km(nodes: list[int], coords: np.ndarray) -> float:
    values = coords[np.asarray(nodes, dtype=int)]
    if len(values) < 2:
        return 0.0
    return float(
        sum(metric_distance_km(first, second) for first, second in zip(values[:-1], values[1:]))
    )


def public_shortest_path_fallbacks(
    source: np.ndarray,
    destination: np.ndarray,
    point_count: np.ndarray,
    graph: dict,
    coords: np.ndarray,
) -> tuple[list[np.ndarray], list[list[int]]]:
    """Build coordinate fallbacks and witnesses from DP requests/public graph."""
    fallbacks = []
    fallback_witnesses = []
    fixed_public_edge = next(
        ([int(origin), int(neighbors[0][0])] for origin, neighbors in sorted(graph.items()) if neighbors),
        None,
    )
    if fixed_public_edge is None:
        raise RuntimeError("Public graph contains no directed edge for a fixed fallback")
    for index, (origin, target, count) in enumerate(
        zip(source, destination, point_count), start=1
    ):
        origin, target = int(origin), int(target)
        nodes = route.shortest_path(
            graph, coords, origin, target, max_visits=300_000
        )
        if nodes is None or len(nodes) < 2:
            nodes = list(fixed_public_edge)
        nodes = [int(node) for node in nodes]
        values = coords[np.asarray(nodes, dtype=int)]
        fallbacks.append(route.resample(values, max(2, int(count))))
        fallback_witnesses.append(nodes)
        if index % 500 == 0:
            print(f"[quotient-rsp-public-fallback] {index}/{len(source)}", flush=True)
    return fallbacks, fallback_witnesses


def main() -> None:
    global ACTIVE_DWELL_TARGET_COST_OFFSET
    args = parse_args()
    ACTIVE_DWELL_TARGET_COST_OFFSET = 0.0
    production_mode = bool(args.production_postprocess)
    fixed_seed_mode = args.decoder_seed is not None or args.request_seed is not None
    if production_mode and ((args.decoder_seed is None) != (args.request_seed is None)):
        raise RuntimeError("Authenticated matched controls require both fixed seeds or neither")
    if production_mode and args.route_alignment != "correct-dp" and not fixed_seed_mode:
        raise RuntimeError("Non-correct matched arms require explicit fixed decoder and request seeds")
    formal_release = production_mode and not fixed_seed_mode and args.route_alignment == "correct-dp"
    resolved = str(args.out_dir.resolve()).lower()
    if not production_mode and (
        not args.out_dir.name.lower().startswith("non_dp") or "production" in resolved
    ):
        raise RuntimeError("RSP development output must be an isolated NON_DP directory")
    if args.out_dir.exists():
        raise RuntimeError("RSP output directory must be new")
    if production_mode:
        if (
            args.production_root is not None
            or args.dp_development_dir is not None
            or args.fallback_dir is not None
            or args.public_matrix_cache is not None
        ):
            raise RuntimeError("Production QRSP cannot read external request or fallback directories")
        if args.portal_release_dir is None:
            raise RuntimeError("Production QRSP requires the committed portal release directory")
        if (
            args.route_release_schema != "portal-fiber-nested384"
            or args.reference_source != "dp-flow"
            or args.reference_conditioning != "portal-fiber"
            or int(args.regions) != 384
        ):
            raise RuntimeError("Production QRSP is frozen to the portal-fiber nested-384 mechanism")
    elif (
        args.dp_development_dir is None
        or args.production_root is None
        or args.public_matrix_cache is None
    ):
        raise RuntimeError(
            "Development QRSP requires production requests, a public matrix, and a DP development dir"
        )
    betas = tuple(float(value) for value in args.betas.split(","))
    if not betas or any(value <= 0.0 for value in betas):
        raise ValueError("All inverse temperatures must be positive")
    if int(args.length_proposals) < 1:
        raise ValueError("Length proposal count must be positive")
    if float(args.length_log_penalty) < 0.0:
        raise ValueError("Length log penalty must be nonnegative")
    if args.active_hierarchy:
        if GRAPH_FLOW_RELEASE is None:
            raise RuntimeError("Active hierarchy requires the released compact graph-flow")
        if args.epsilon_graph_flow is None or float(args.epsilon_graph_flow) <= 0.0:
            raise ValueError("Active hierarchy requires a positive graph-flow epsilon")
        if float(args.occupancy_strength) < 0.0 or float(args.dwell_strength) < 0.0:
            raise ValueError("Active hierarchy strengths must be nonnegative")
        if (
            not np.isfinite(float(args.hierarchy_likelihood_ratio_cap))
            or float(args.hierarchy_likelihood_ratio_cap) < 1.0
        ):
            raise ValueError("Hierarchy likelihood-ratio cap must be finite and at least one")
    if int(args.checkpoint_every) < 1:
        raise ValueError("Checkpoint interval must be positive")
    if not np.isfinite(float(args.od_likelihood_ratio_cap)) or float(
        args.od_likelihood_ratio_cap
    ) < 1.0:
        raise ValueError("OD likelihood-ratio cap must be finite and at least one")
    if args.reference_conditioning in {"od-family", "od-factorized"} and int(args.regions) != 96:
        raise ValueError("OD-family conditioning currently uses the released 96-region family block")
    if args.reference_conditioning in {"od-lifted", "endpoint-lifted"} and int(args.regions) != 384:
        raise ValueError("Lifted OD conditioning requires the 384-region quotient graph")
    if args.route_release_schema == "distance4-coarse24" and (
        args.reference_conditioning != "od-lifted"
        or int(args.regions) != 384
        or args.od_family_granularity != "distance"
    ):
        raise ValueError(
            "The distance4-coarse24 release requires 384-region lifted distance conditioning"
        )
    if args.route_release_schema == "endpoint8-coarse24" and (
        args.reference_conditioning not in {"endpoint-lifted", "global"}
        or int(args.regions) != 384
    ):
        raise ValueError(
            "The endpoint8-coarse24 release requires 384-region global or endpoint-lifted conditioning"
        )
    if args.route_release_schema == "relative32-nested384" and (
        args.reference_conditioning not in {"relative-direction", "global"}
        or int(args.regions) != 384
    ):
        raise ValueError(
            "The relative32-nested384 release requires 384-region global or relative-direction conditioning"
        )
    if args.route_release_schema == "portal-fiber-nested384" and (
        args.reference_conditioning not in {"portal-fiber", "global"}
        or int(args.regions) != 384
    ):
        raise ValueError(
            "The portal-fiber-nested384 release requires 384-region global or portal-fiber conditioning"
        )

    osm_path = (
        Path(__file__).resolve().parents[2]
        / "ara_final"
        / "evidence"
        / "tables"
        / "osm_cache_beijing.pkl"
    )
    if production_mode:
        request_manifest, request_arrays, request_sha256, osm_ways = (
            production_gate.sample_committed_base_requests(
                osm_path, request_seed=args.request_seed
            )
        )
        request_path = None
    else:
        request_manifest = json.loads(
            (args.production_root / "requests" / "request_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        request_path = args.production_root / "requests" / "dp_requests.npz"
        if lineage.sha256_file(request_path) != request_manifest["request_sha256"]:
            raise RuntimeError("DP request hash mismatch")
        request_sha256 = request_manifest["request_sha256"]
        with np.load(request_path, allow_pickle=False) as archive:
            request_arrays = {name: np.asarray(archive[name]) for name in archive.files}
        osm_ways = load_osm_ways(osm_path)
    source = np.asarray(request_arrays["source"], dtype=int)
    destination = np.asarray(request_arrays["destination"], dtype=int)
    point_count = np.asarray(request_arrays["point_count"], dtype=int)
    target_length = np.asarray(request_arrays["target_length_km"], dtype=float)
    if production_mode:
        if args.limit is not None and int(args.limit) != len(source):
            raise RuntimeError("Production QRSP must synthesize the full DP request set")
        limit = len(source)
    else:
        limit = min(1000 if args.limit is None else int(args.limit), len(source))

    coords, graph = route.prepare_graph(
        [], bbox=support.BBOX, osm_ways=osm_ways, raw_graph=False
    )
    if production_mode and (
        np.any(source >= len(coords)) or np.any(destination >= len(coords))
    ):
        raise RuntimeError("Internally sampled DP request node is outside the public graph")
    context96, graph96 = build_context(coords, graph, 24, 96, 4, 3)
    nested_parent = None
    if args.route_release_schema in {
        "endpoint8-coarse24", "relative32-nested384", "portal-fiber-nested384"
    }:
        context384, graph384, nested_parent = nested_graph.build_nested_context(
            coords, graph, 24, 384, 4, 3
        )
    else:
        context384, graph384 = build_context(coords, graph, 24, 384, 4, 3)
    flow_name = "fine96_flow" if int(args.regions) == 96 else "fine384_flow"
    context = context96 if int(args.regions) == 96 else context384
    if selected_path_context_initializer is not None:
        selected_path_context_initializer(coords, graph, context)
    graph_diag = graph96 if int(args.regions) == 96 else graph384
    portal_manifest = None
    route_epsilon = None
    release_sha256 = None
    portal_manifest_sha256 = None
    released_snapshot = None
    if production_mode:
        production_portals = portal_release.diverse_portals(
            graph, coords, context384.node_fine, int(args.portal_count)
        )
        _, production_portal_atoms = portal_release.portal_layout(production_portals)
        expected_block_shapes = {
            "coarse24_occupancy": (24,),
            "fine384_occupancy": (384,),
            "fine96_flow": (len(context96.fine_edge_index) + 1,),
            "fine384_flow": (len(context384.fine_edge_index) + 1,),
            "portal_fiber_flow": (int(production_portal_atoms) + 1,),
        }
        (
            portal_manifest,
            released_snapshot,
            release_sha256,
            portal_manifest_sha256,
            route_epsilon,
        ) = (
            production_gate.verify_portal_release(
            args.portal_release_dir,
            request_manifest["public_osm_sha256"],
            capacity=int(request_manifest["request_count"]),
            coarse_occupancy_regions=24,
            coarse_transition_regions=96,
            fine_regions=384,
            macro_regions=4,
            phase_count=3,
            portals_per_fine_edge=int(args.portal_count),
            expected_block_shapes=expected_block_shapes,
            )
        )
        release_path = None
        args.epsilon_route = float(route_epsilon)
    else:
        release_path = args.dp_development_dir / "multires_graph_flow_release.npz"
    shrinkage = {}
    od_family_flow = None
    od_factorized_flow = None
    endpoint_factorized_flow = None
    relative_direction_flow = None
    portal_fiber_flow = None
    conservative_projection_checks = None
    if args.reference_source == "dp-flow":
        release_budgets = {
            "distance4-coarse24": distance_release.BLOCK_BUDGETS,
            "endpoint8-coarse24": endpoint_release.BLOCK_BUDGETS,
            "relative32-nested384": relative_release.BLOCK_BUDGETS,
            "portal-fiber-nested384": portal_release.BLOCK_BUDGETS,
        }.get(args.route_release_schema, legacy.BLOCK_BUDGETS)
        if production_mode:
            released = {
                name: np.asarray(released_snapshot[name], dtype=float)
                for name in release_budgets
            }
        else:
            with np.load(release_path, allow_pickle=False) as archive:
                released = {
                    name: np.asarray(archive[name], dtype=float) for name in release_budgets
                }
        # Matched attribution changes only how the already released q5
        # coordinates are aligned to their public feature coordinates.  The
        # transcript bytes, requests, graph, candidate support, decoder random
        # stream, fallback, output count, and serializer remain fixed.
        released = align_released_route_blocks(released, args.route_alignment)
        shrink_names = (
            tuple(list(release_budgets)[:4])
            if args.route_release_schema in {
                "distance4-coarse24", "endpoint8-coarse24", "relative32-nested384",
                "portal-fiber-nested384",
            }
            else tuple(legacy.BLOCK_BUDGETS)
        )
        shrink_values = {name: np.asarray(released[name], dtype=float).ravel() for name in shrink_names}
        if args.route_release_schema in {
            "endpoint8-coarse24", "relative32-nested384", "portal-fiber-nested384"
        }:
            shrink_values["fine96_flow"] = shrink_values["fine96_flow"][:-1]
            shrink_values["fine384_flow"] = shrink_values["fine384_flow"][:-1]
        target = np.concatenate([shrink_values[name] for name in shrink_names])
        if args.route_release_schema in {
            "endpoint8-coarse24", "relative32-nested384", "portal-fiber-nested384"
        }:
            def scaled_public(values: np.ndarray, total: float) -> np.ndarray:
                result = np.maximum(np.asarray(values, dtype=float).ravel(), 0.0)
                return result / max(float(result.sum()), 1e-15) * float(total)

            public_parts = {
                "coarse24_occupancy": scaled_public(
                    np.bincount(context96.node_coarse, minlength=24),
                    shrink_values["coarse24_occupancy"].sum(),
                ),
                "fine384_occupancy": scaled_public(
                    np.bincount(context384.node_fine, minlength=384),
                    shrink_values["fine384_occupancy"].sum(),
                ),
                "fine96_flow": scaled_public(
                    quotient_public_mass(graph, context96.node_fine, context96.fine_edge_index),
                    shrink_values["fine96_flow"].sum(),
                ),
                "fine384_flow": scaled_public(
                    quotient_public_mass(graph, context384.node_fine, context384.fine_edge_index),
                    shrink_values["fine384_flow"].sum(),
                ),
            }
            public_mean = np.concatenate([public_parts[name] for name in shrink_names])
        else:
            mass_matrix = load_npz(args.public_matrix_cache).tocsr()
            matrix = oracle.legacy_matrix_from_mass_cache(
                mass_matrix, mass_runner.component_slices(context96, context384)
            )
            public_mean_full = np.asarray(
                matrix.T @ np.full(matrix.shape[0], 1.0 / 8.0)
            ).ravel()
            legacy_template = legacy.empty_blocks(context96, context384)
            legacy_offsets, legacy_cursor = {}, 0
            for name in legacy.BLOCK_BUDGETS:
                legacy_offsets[name] = legacy_cursor
                legacy_cursor += int(legacy_template[name].size)
            public_mean = np.concatenate(
                [
                    public_mean_full[
                        legacy_offsets[name] : legacy_offsets[name]
                        + int(legacy_template[name].size)
                    ]
                    for name in shrink_names
                ]
            )
        variance = ners.discrete_laplace_variance_normalized(
            epsilon=float(args.epsilon_route), sensitivity=legacy.ROUTE_LATTICE
        )
        shrunk, shrinkage = ners.positive_part_block_shrinkage(
            target,
            public_mean,
            {name: int(shrink_values[name].size) for name in shrink_names},
            noise_variance=variance,
        )
        offsets, cursor = {}, 0
        for name in shrink_names:
            offsets[name] = cursor
            cursor += int(shrink_values[name].size)
        shrunk_blocks = {
            name: shrunk[offsets[name] : offsets[name] + int(shrink_values[name].size)]
            for name in shrink_names
        }
        offset = offsets[flow_name]
        fine_flow = shrunk[offset : offset + int(shrink_values[flow_name].size)]
        if args.route_release_schema == "distance4-coarse24":
            od_family_flow = np.asarray(
                released["distance_coarse24_flow"], dtype=float
            ).reshape(4, -1)
        elif args.route_release_schema == "endpoint8-coarse24":
            endpoint_factorized_flow = (
                np.asarray(released["origin_macro4_coarse24_flow"], dtype=float).reshape(4, -1)[:, :-1],
                np.asarray(released["destination_macro4_coarse24_flow"], dtype=float).reshape(4, -1)[:, :-1],
            )
        elif args.route_release_schema == "relative32-nested384":
            relative_direction_flow = np.asarray(
                released["distance_relative_direction"], dtype=float
            ).reshape(4, -1)[:, :-1]
        elif args.route_release_schema == "portal-fiber-nested384":
            portal_fiber_flow = np.asarray(released["portal_fiber_flow"], dtype=float).ravel()[:-1]
        else:
            od_family_values = (
                shrunk_blocks["od_conditioned_fine96_flow"]
                if args.od_family_source == "ners"
                else released["od_conditioned_fine96_flow"].ravel()
            )
            if args.reference_conditioning == "od-factorized":
                od_factorized_flow = (
                    aggregate_od_family_flow(od_family_values, "distance"),
                    aggregate_od_family_flow(od_family_values, "bearing"),
                )
            else:
                od_family_flow = aggregate_od_family_flow(
                    od_family_values, args.od_family_granularity
                )
        if not production_mode:
            release_sha256 = lineage.sha256_file(release_path)
    else:
        fine_flow = np.zeros(len(context.fine_edge_index), dtype=float)
    centers = np.asarray(context.fine_tree.data, dtype=float)
    reference, cost, kernel_checks = reference_kernel(
        fine_flow,
        graph,
        context.node_fine,
        context.fine_edge_index,
        centers,
        use_released_flow=(args.reference_source == "dp-flow"),
    )
    active_hierarchy_diagnostics = None
    if args.active_hierarchy:
        if (
            args.reference_source != "dp-flow"
            or args.route_release_schema != "portal-fiber-nested384"
            or int(args.regions) != 384
        ):
            raise RuntimeError(
                "Active hierarchy is defined only for the Portal-Fiber nested-384 DP decoder"
            )
        fine_to_96 = np.asarray(
            context96.fine_tree.query(np.asarray(context384.fine_tree.data, dtype=float))[1],
            dtype=int,
        )
        reference, cost, active_hierarchy_diagnostics = active_hierarchy_projection(
            reference,
            cost,
            GRAPH_FLOW_RELEASE,
            shrunk_blocks,
            graph,
            context96,
            context384,
            fine_to_96,
            epsilon_graph_flow=float(args.epsilon_graph_flow),
            occupancy_strength=float(args.occupancy_strength),
            dwell_strength=float(args.dwell_strength),
            likelihood_ratio_cap=float(args.hierarchy_likelihood_ratio_cap),
        )
        ACTIVE_DWELL_TARGET_COST_OFFSET = (
            float(active_hierarchy_diagnostics["hop_penalty"])
            * float(active_hierarchy_diagnostics["target_region_count"])
        )
        kernel_checks = dict(kernel_checks)
        kernel_checks["active_hierarchy"] = active_hierarchy_diagnostics
    bridge = RSPBridge(reference, cost, betas)
    bridges_by_family = None
    conditional_budget = (
        350_000
        if args.route_release_schema in {
            "distance4-coarse24", "endpoint8-coarse24", "relative32-nested384",
            "portal-fiber-nested384",
        }
        else legacy.BLOCK_BUDGETS["od_conditioned_fine96_flow"]
    )
    fine_budget = legacy.BLOCK_BUDGETS[
        "fine384_flow"
        if args.reference_conditioning in {
            "od-lifted", "endpoint-lifted", "relative-direction", "portal-fiber"
        }
        else "fine96_flow"
    ]
    conditioned_weight = conditional_budget / (conditional_budget + fine_budget)
    factor_weights = (1.0 / 3.0, 2.0 / 3.0)
    if args.reference_conditioning == "od-family":
        bridges_by_family = {}
        for family in range(len(od_family_flow) if od_family_flow is not None else {
            "distance-bearing": 32,
            "bearing": 8,
            "distance": 4,
        }[args.od_family_granularity]):
            if args.reference_source == "dp-flow":
                conditioned, conditioned_cost, _ = reference_kernel(
                    od_family_flow[family],
                    graph,
                    context.node_fine,
                    context.fine_edge_index,
                    centers,
                    use_released_flow=True,
                )
                if not np.array_equal(conditioned_cost, cost):
                    raise RuntimeError("Global and OD-conditioned public graph costs disagree")
                family_reference = rowwise_entropic_barycenter(
                    reference,
                    conditioned,
                    conditioned_weight,
                    likelihood_ratio_cap=float(args.od_likelihood_ratio_cap),
                )
            else:
                family_reference = reference
            bridges_by_family[family] = RSPBridge(family_reference, cost, betas)
    elif args.reference_conditioning == "od-factorized":
        if args.reference_source == "public-only":
            shared_public_bridge = RSPBridge(reference, cost, betas)
            bridges_by_family = {family: shared_public_bridge for family in range(32)}
        else:
            if od_factorized_flow is None:
                raise RuntimeError("Factorized OD flow is unavailable")
            distance_flow, bearing_flow = od_factorized_flow

            def conditioned_kernel(flow: np.ndarray) -> np.ndarray:
                result, result_cost, _ = reference_kernel(
                    flow,
                    graph,
                    context.node_fine,
                    context.fine_edge_index,
                    centers,
                    use_released_flow=True,
                )
                if not np.array_equal(result_cost, cost):
                    raise RuntimeError("Global and factorized OD public graph costs disagree")
                return result

            distance_kernels = tuple(conditioned_kernel(flow) for flow in distance_flow)
            bearing_kernels = tuple(conditioned_kernel(flow) for flow in bearing_flow)
            bridges_by_family = {}
            for family in range(32):
                distance, bearing = divmod(family, 8)
                family_reference = rowwise_factorized_information_projection(
                    reference,
                    (distance_kernels[distance], bearing_kernels[bearing]),
                    factor_weights,
                    conditioned_weight,
                    float(args.od_likelihood_ratio_cap),
                )
                bridges_by_family[family] = RSPBridge(family_reference, cost, betas)
    elif args.reference_conditioning == "od-lifted":
        family_count = {
            "distance-bearing": 32,
            "bearing": 8,
            "distance": 4,
        }[args.od_family_granularity]
        if args.reference_source == "public-only":
            shared_public_bridge = RSPBridge(reference, cost, betas)
            bridges_by_family = {
                family: shared_public_bridge for family in range(family_count)
            }
        else:
            if od_family_flow is None:
                raise RuntimeError("Lifted OD flow is unavailable")
            use_distance_coarse = args.route_release_schema == "distance4-coarse24"
            coarse_tree = context96.coarse_tree if use_distance_coarse else context96.fine_tree
            coarse_centers = np.asarray(coarse_tree.data, dtype=float)
            coarse_labels = (
                context96.node_coarse if use_distance_coarse else context96.node_fine
            )
            coarse_edge_index = (
                context96.coarse_edge_index
                if use_distance_coarse
                else context96.fine_edge_index
            )
            coarse_global_flow = (
                np.asarray(od_family_flow, dtype=float).sum(axis=0)
                if use_distance_coarse
                else shrunk_blocks["fine96_flow"]
            )
            coarse_reference, coarse_cost, _ = reference_kernel(
                coarse_global_flow,
                graph,
                coarse_labels,
                coarse_edge_index,
                coarse_centers,
                use_released_flow=True,
            )
            _, fine_to_coarse = coarse_tree.query(centers)
            fine_to_coarse = np.asarray(fine_to_coarse, dtype=int)
            bridges_by_family = {}
            for family in range(len(od_family_flow)):
                coarse_conditioned, conditioned_cost, _ = reference_kernel(
                    od_family_flow[family],
                    graph,
                    coarse_labels,
                    coarse_edge_index,
                    coarse_centers,
                    use_released_flow=True,
                )
                if not np.array_equal(conditioned_cost, coarse_cost):
                    raise RuntimeError("Global and lifted coarse graph costs disagree")
                family_reference = lift_coarse_likelihood_projection(
                    reference,
                    coarse_reference,
                    coarse_conditioned,
                    fine_to_coarse,
                    conditioned_weight,
                    float(args.od_likelihood_ratio_cap),
                )
                bridges_by_family[family] = RSPBridge(family_reference, cost, betas)
    elif args.reference_conditioning == "endpoint-lifted":
        if args.reference_source == "public-only":
            shared_public_bridge = RSPBridge(reference, cost, betas)
            bridges_by_family = {family: shared_public_bridge for family in range(16)}
        else:
            if endpoint_factorized_flow is None:
                raise RuntimeError("Endpoint-factorized coarse flow is unavailable")
            origin_flow, destination_flow = endpoint_factorized_flow
            coarse_tree = context96.coarse_tree
            coarse_centers = np.asarray(coarse_tree.data, dtype=float)
            coarse_labels = context96.node_coarse
            coarse_edge_index = context96.coarse_edge_index
            coarse_global_flow = 0.5 * (
                np.asarray(origin_flow, dtype=float).sum(axis=0)
                + np.asarray(destination_flow, dtype=float).sum(axis=0)
            )
            if nested_parent is None:
                raise RuntimeError("Endpoint-factorized decoder requires an exact nested parent map")
            fine_to_coarse = np.asarray(nested_parent, dtype=int)
            bridges_by_family = {}
            conservative_projection_checks = {}
            for origin_macro in range(4):
                for destination_macro in range(4):
                    family = origin_macro * 4 + destination_macro
                    projected_flow, projection_check = conservative_factorized_flow_projection(
                        fine_flow,
                        context.fine_edge_index,
                        fine_to_coarse,
                        coarse_edge_index,
                        coarse_global_flow,
                        (origin_flow[origin_macro], destination_flow[destination_macro]),
                        (0.5, 0.5),
                        conditioned_weight,
                        float(args.od_likelihood_ratio_cap),
                    )
                    family_reference, family_cost, _ = reference_kernel(
                        projected_flow,
                        graph,
                        context.node_fine,
                        context.fine_edge_index,
                        centers,
                        use_released_flow=True,
                    )
                    if not np.array_equal(family_cost, cost):
                        raise RuntimeError("Global and conservative endpoint fine costs disagree")
                    conservative_projection_checks[str(family)] = projection_check
                    bridges_by_family[family] = RSPBridge(family_reference, cost, betas)
    elif args.reference_conditioning == "relative-direction":
        if args.reference_source == "public-only":
            shared_public_bridge = RSPBridge(reference, cost, betas)
            bridges_by_family = {family: shared_public_bridge for family in range(32)}
        else:
            if relative_direction_flow is None:
                raise RuntimeError("Relative-direction route moments are unavailable")
            global_relative = np.asarray(relative_direction_flow, dtype=float).sum(axis=0)
            bridges_by_family = {}
            for distance_family in range(4):
                for bearing_sector in range(8):
                    family = distance_family * 8 + bearing_sector
                    family_reference = relative_direction_information_projection(
                        reference,
                        centers,
                        relative_direction_flow[distance_family],
                        global_relative,
                        bearing_sector,
                        conditioned_weight,
                        float(args.od_likelihood_ratio_cap),
                    )
                    bridges_by_family[family] = RSPBridge(family_reference, cost, betas)

    def maximum_bridge_residual() -> float:
        active = (
            list(bridges_by_family.values()) if bridges_by_family is not None else [bridge]
        )
        return float(max(item.max_residual for item in active))
    portals = diverse_portals(
        graph, coords, context.node_fine, int(args.portal_count)
    )
    portal_probabilities = None
    if args.reference_conditioning == "portal-fiber" and args.reference_source == "dp-flow":
        if portal_fiber_flow is None:
            raise RuntimeError("Released portal-fiber flow is unavailable")
        release_portals = portal_release.diverse_portals(
            graph, coords, context.node_fine, int(args.portal_count)
        )
        if release_portals != portals:
            raise RuntimeError("Query and decoder portal layouts disagree")
        portal_offsets, portal_atom_count = portal_release.portal_layout(portals)
        if int(portal_atom_count) != len(portal_fiber_flow):
            raise RuntimeError("Released portal-fiber dimension disagrees with public layout")
        portal_probabilities = {
            key: portal_fiber_flow[
                int(portal_offsets[key]) : int(portal_offsets[key]) + len(options)
            ]
            for key, options in portals.items()
        }
    region_router = RegionRouter(graph, context.node_fine)
    rng = np.random.default_rng(
        int(args.decoder_seed)
        if args.decoder_seed is not None
        else random.SystemRandom().getrandbits(128)
        if production_mode
        else int(args.seed) + 90_001
    )

    if production_mode:
        fallback_path = None
        cache_path = args.public_fallback_cache
        if cache_path is not None and cache_path.is_file():
            with cache_path.open("rb") as handle:
                fallback_payload = pickle.load(handle)
            if (
                not isinstance(fallback_payload, dict)
                or fallback_payload.get("schema_id") != "public-fallback-with-directed-witness-v1"
                or not isinstance(fallback_payload.get("coordinates"), list)
                or not isinstance(fallback_payload.get("node_sequences"), list)
            ):
                raise RuntimeError("Cached public fallback has the wrong schema")
            fallback = fallback_payload["coordinates"]
            fallback_witnesses = fallback_payload["node_sequences"]
            if len(fallback) != limit or len(fallback_witnesses) != limit:
                raise RuntimeError("Cached public fallback has the wrong slot count")
        else:
            fallback, fallback_witnesses = public_shortest_path_fallbacks(
                source, destination, point_count, graph, coords
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                write_atomic_pickle(
                    cache_path,
                    {
                        "schema_id": "public-fallback-with-directed-witness-v1",
                        "coordinates": fallback,
                        "node_sequences": fallback_witnesses,
                    },
                )
    else:
        fallback_root = args.fallback_dir or args.dp_development_dir
        fallback_path = fallback_root / (
            "maxent_projected.pkl"
            if args.reference_source == "dp-flow"
            else "uniform_candidate_control.pkl"
        )
        with fallback_path.open("rb") as handle:
            loaded_fallback = pickle.load(handle)
        fallback = loaded_fallback[:limit]
        fallback_witnesses = [None] * limit
        del loaded_fallback
    # This control uses only the same DP requests and public graph in production.
    public_control = fallback
    checkpoint_path = args.out_dir.parent / f".{args.out_dir.name}.checkpoint.pkl"
    checkpoint_config = {
        "schema_version": 1,
        "limit": limit,
        "randomness": (
            "SystemRandom-derived unrecorded seed"
            if formal_release
            else (
                f"matched fixed decoder seed {int(args.decoder_seed)} and request seed {int(args.request_seed)}"
                if production_mode
                else f"fixed development seed {int(args.seed)}"
            )
        ),
        "regions": int(args.regions),
        "route_release_schema": args.route_release_schema,
        "reference_source": args.reference_source,
        "reference_conditioning": args.reference_conditioning,
        "od_family_source": args.od_family_source,
        "od_family_granularity": args.od_family_granularity,
        "od_likelihood_ratio_cap": float(args.od_likelihood_ratio_cap),
        "portal_count": int(args.portal_count),
        "max_region_steps": int(args.max_region_steps),
        "length_proposals": int(args.length_proposals),
        "length_log_penalty": float(args.length_log_penalty),
        "route_alignment": str(args.route_alignment),
        "decoder_seed": int(args.decoder_seed) if args.decoder_seed is not None else None,
        "request_seed": int(args.request_seed) if args.request_seed is not None else None,
        "betas": list(betas),
        "request_sha256": request_sha256,
        "dp_route_release_sha256": release_sha256,
        "public_osm_sha256": (
            request_manifest["public_osm_sha256"]
            if production_mode
            else lineage.sha256_file(osm_path)
        ),
        "fallback_sha256": (
            lineage.sha256_file(fallback_path) if fallback_path is not None else None
        ),
    }
    checkpoint_enabled = (not production_mode) or ALLOW_AUTHENTICATED_ABLATION_CHECKPOINT
    if production_mode and checkpoint_path.exists() and not checkpoint_enabled:
        raise RuntimeError("Production QRSP forbids pre-existing checkpoint input")
    if checkpoint_enabled and checkpoint_path.exists():
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get("config") != checkpoint_config:
            raise RuntimeError("RSP checkpoint configuration or input lineage mismatch")
        trajectories = checkpoint["trajectories"]
        witnesses = checkpoint["witnesses"]
        diagnostics = checkpoint["diagnostics"]
        fallback_count = int(checkpoint["fallback_count"])
        restricted_fallback_segments = int(checkpoint["restricted_fallback_segments"])
        selected_length_errors = checkpoint["selected_length_errors"]
        proposal_length_errors = checkpoint["proposal_length_errors"]
        start_index = int(checkpoint["next_index"])
        residual = float(checkpoint["maximum_desirability_residual"])
        bridge.max_residual = residual
        if bridges_by_family is not None:
            for family_bridge in bridges_by_family.values():
                family_bridge.max_residual = residual
        rng.bit_generator.state = checkpoint["rng_state"]
        if (
            len(trajectories) != start_index
            or len(witnesses) != start_index
            or len(diagnostics) != start_index
        ):
            raise RuntimeError("RSP checkpoint prefix lengths are inconsistent")
        print(f"[quotient-rsp] resumed at {start_index}/{limit}", flush=True)
    else:
        trajectories, witnesses, diagnostics = [], [], []
        fallback_count = 0
        restricted_fallback_segments = 0
        selected_length_errors = []
        proposal_length_errors = []
        start_index = 0
    for index in range(start_index, limit):
        source_region = int(context.node_fine[int(source[index])])
        destination_region = int(context.node_fine[int(destination[index])])
        if args.reference_conditioning == "endpoint-lifted":
            source_coarse = int(context96.node_coarse[int(source[index])])
            destination_coarse = int(context96.node_coarse[int(destination[index])])
            source_macro = int(context96.coarse_to_macro[source_coarse])
            destination_macro = int(context96.coarse_to_macro[destination_coarse])
            family = source_macro * 4 + destination_macro
        else:
            full_family = legacy.od_family(
                np.vstack([coords[int(source[index])], coords[int(destination[index])]])
            )
            family = (
                int(full_family)
                if args.reference_conditioning in {"od-factorized", "relative-direction"}
                else coarsen_od_family(full_family, args.od_family_granularity)
            )
        active_bridge = (
            bridges_by_family[int(family)] if bridges_by_family is not None else bridge
        )
        proposals = []
        proposal_count = int(args.length_proposals)
        beta_order = list(betas)
        rng.shuffle(beta_order)
        for proposal_index in range(proposal_count):
            forced_beta = None
            if proposal_count > 1:
                forced_beta = beta_order[proposal_index % len(beta_order)]
            region_path, current = active_bridge.sample(
                source_region,
                destination_region,
                float(target_length[index]),
                rng,
                int(args.max_region_steps),
                fixed_beta=forced_beta,
            )
            nodes = None
            if region_path is not None:
                nodes, segment_fallbacks = lift_quotient_path(
                    region_path,
                    int(source[index]),
                    int(destination[index]),
                    graph,
                    coords,
                    context.node_fine,
                    portals,
                    region_router,
                    rng,
                    portal_probabilities=portal_probabilities,
                    portal_weight=conditioned_weight,
                    portal_likelihood_ratio_cap=float(args.od_likelihood_ratio_cap),
                )
                restricted_fallback_segments += int(segment_fallbacks)
            if nodes is not None and len(nodes) >= 2:
                length = road_path_length_km(nodes, coords)
                log_error = abs(
                    math.log((length + 0.1) / (float(target_length[index]) + 0.1))
                )
                proposal_length_errors.append(log_error)
                proposals.append((nodes, current, length, log_error))
        if candidate_path_augmenter is not None:
            extras = candidate_path_augmenter(
                int(source[index]),
                int(destination[index]),
                float(target_length[index]),
                proposals,
            )
            for extra_nodes, extra_state in extras:
                if extra_nodes is None or len(extra_nodes) < 2:
                    continue
                extra_length = road_path_length_km(extra_nodes, coords)
                extra_error = abs(
                    math.log((extra_length + 0.1) / (float(target_length[index]) + 0.1))
                )
                proposal_length_errors.append(extra_error)
                proposals.append((extra_nodes, extra_state, extra_length, extra_error))
        nodes = None
        current = {"failure": "no-valid-length-proposal"}
        if proposals:
            logits = -float(args.length_log_penalty) * np.asarray(
                [item[3] for item in proposals], dtype=float
            )
            if proposal_logit_adjuster is not None:
                adjustment = np.asarray(proposal_logit_adjuster(proposals), dtype=float)
                if adjustment.shape != logits.shape or not np.all(np.isfinite(adjustment)):
                    raise RuntimeError("Proposal logit adjustment must be finite and shape preserving")
                logits = logits + adjustment
            probabilities = np.exp(logits - float(logits.max()))
            probabilities /= probabilities.sum()
            chosen = proposals[int(rng.choice(len(proposals), p=probabilities))]
            nodes, current, chosen_length, chosen_error = chosen
            current = dict(current)
            current["od_family"] = int(family)
            current["lifted_road_length_km"] = float(chosen_length)
            current["length_log_error"] = float(chosen_error)
            current["length_proposal_count"] = len(proposals)
            selected_length_errors.append(float(chosen_error))
        if (
            nodes is not None
            and len(nodes) >= 2
            and selected_path_acceptor is not None
            and not selected_path_acceptor(nodes, current, fallback[index])
        ):
            current["candidate_source"] = "frozen-dp-gsrt-weak-evidence-fallback"
            current["failure"] = "selected-path-below-postprocessing-evidence-threshold"
            nodes = None
        if nodes is None or len(nodes) < 2:
            fallback_count += 1
            trajectory = np.asarray(fallback[index], dtype=float)
            current["lift_fallback"] = True
            witness_nodes = fallback_witnesses[index]
            if witness_nodes is None or len(witness_nodes) < 2:
                if production_mode:
                    raise RuntimeError("Every production fallback must retain a directed-edge witness")
                witness_nodes = None
        else:
            trajectory = route.resample(
                coords[np.asarray(nodes, dtype=int)], max(2, int(point_count[index]))
            )
            current["lift_fallback"] = False
            witness_nodes = [int(node) for node in nodes]
        trajectories.append(trajectory)
        witnesses.append(
            {
                "slot": int(index),
                "node_sequence": witness_nodes,
                "directed_edges": (
                    list(zip(witness_nodes[:-1], witness_nodes[1:]))
                    if witness_nodes is not None
                    else None
                ),
                "fallback": bool(current["lift_fallback"]),
            }
        )
        diagnostics.append(current)
        if (index + 1) % 250 == 0:
            print(f"[quotient-rsp] {index+1}/{limit}", flush=True)
        if (
            checkpoint_enabled
            and (index + 1) % int(args.checkpoint_every) == 0
            and index + 1 < limit
        ):
            write_atomic_pickle(
                checkpoint_path,
                {
                    "config": checkpoint_config,
                    "next_index": index + 1,
                    "trajectories": trajectories,
                    "witnesses": witnesses,
                    "diagnostics": diagnostics,
                    "fallback_count": fallback_count,
                    "restricted_fallback_segments": restricted_fallback_segments,
                    "selected_length_errors": selected_length_errors,
                    "proposal_length_errors": proposal_length_errors,
                    "maximum_desirability_residual": maximum_bridge_residual(),
                    "rng_state": rng.bit_generator.state,
                },
            )

    staging = args.out_dir.parent / f".{args.out_dir.name}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    if not production_mode:
        (staging / "DO_NOT_RELEASE.txt").write_text(
            "NON-DP fixed-seed quotient-RSP development artifact.\n", encoding="ascii"
        )
    elif not formal_release:
        (staging / "DO_NOT_RELEASE.txt").write_text(
            "MATCHED RESEARCH CONTROL; AUTHENTICATED DP POST-PROCESSING BUT NOT A FORMAL PRODUCTION RELEASE.\n",
            encoding="ascii",
        )
    synthetic_name = (
        "dp_gsrt_portal_qrsp.pkl" if production_mode else "maxent_projected.pkl"
    )
    control_name = (
        "dp_request_public_shortest_path_control.pkl"
        if production_mode
        else "uniform_candidate_control.pkl"
    )
    with (staging / synthetic_name).open("wb") as handle:
        pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (staging / "road_witnesses.pkl").open("wb") as handle:
        pickle.dump(witnesses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (staging / control_name).open("wb") as handle:
        pickle.dump(public_control, handle, protocol=pickle.HIGHEST_PROTOCOL)
    beta_counts = {
        str(beta): int(sum(item.get("beta") == beta for item in diagnostics)) for beta in betas
    }
    valid_steps = [item["steps"] for item in diagnostics if "steps" in item]
    privacy_accounting = None
    if production_mode:
        base_event = lineage.read_privacy_ledger()
        base_epsilon = Fraction(base_event["epsilon_rational"])
        total_epsilon = base_epsilon + route_epsilon
        privacy_accounting = {
            "adjacency": "trajectory-record add/remove with a fixed public slot capacity",
            "base_epsilon_rational": f"{base_epsilon.numerator}/{base_epsilon.denominator}",
            "portal_route_epsilon_rational": (
                f"{route_epsilon.numerator}/{route_epsilon.denominator}"
            ),
            "composed_epsilon_rational": (
                f"{total_epsilon.numerator}/{total_epsilon.denominator}"
            ),
            "delta": 0.0,
            "decoder_cost": "zero additional privacy loss by post-processing",
        }
    report = {
        "algorithm": "Information-geometric QRSP Doob bridge",
        "reference_source": args.reference_source,
        "reference_conditioning": args.reference_conditioning,
        "route_release_schema": args.route_release_schema,
        "route_alignment": str(args.route_alignment),
        "decoder_seed": int(args.decoder_seed) if args.decoder_seed is not None else None,
        "request_seed": int(args.request_seed) if args.request_seed is not None else None,
        "od_family_source": (
            args.od_family_source if args.reference_conditioning != "global" else None
        ),
        "od_family_granularity": (
            "distance-bearing-factorized"
            if args.reference_conditioning == "od-factorized"
            else f"coarse-to-fine-{args.od_family_granularity}"
            if args.reference_conditioning == "od-lifted"
            else "coarse-to-fine-origin-destination-macro-factorized"
            if args.reference_conditioning == "endpoint-lifted"
            else "distance-by-relative-edge-direction"
            if args.reference_conditioning == "relative-direction"
            else "nested-fine-edge-public-portal-fiber"
            if args.reference_conditioning == "portal-fiber"
            else args.od_family_granularity
            if args.reference_conditioning == "od-family"
            else None
        ),
        "classification": (
            "PRODUCTION_DP_POSTPROCESS"
            if formal_release
            else "MATCHED_RESEARCH_AUTHENTICATED_DP_POSTPROCESS_DO_NOT_RELEASE"
            if production_mode
            else "NON_DP_FIXED_SEED_DO_NOT_RELEASE"
        ),
        "certified_release": formal_release,
        "privacy_accounting": privacy_accounting,
        "privacy_boundary": (
            "Authenticated post-processing of the committed base sanitizer and portal-fiber DP "
            "transcripts; requests are sampled internally and fallback paths use only the public graph."
            if formal_release
            else "Matched research-only post-processing of authenticated DP transcripts with fixed, "
            "shared request and decoder randomness; not a formal production release."
            if production_mode
            else "Post-processing only if every request and graph-flow input is from an authenticated "
            "DP transcript and the road graph is public. This development run uses a fixed seed."
        ),
        "release_object": {
            "coordinate_polyline": synthetic_name,
            "road_witness_sidecar": "road_witnesses.pkl",
            "witness_schema": {
                "slot": "fixed output-slot index",
                "node_sequence": "ordered directed public-graph node sequence, including fixed public fallbacks",
                "directed_edges": "ordered adjacent node pairs witnessing the directed path, including fixed public fallbacks",
                "fallback": "whether the fixed public fallback was emitted",
            },
            "public_graph_binding": "public_osm_sha256",
        },
        "implementation_source_sha256": {
            "quotient_rsp_bridge_development.py": lineage.sha256_file(Path(__file__)),
            "portal_fiber_qrsp_production.py": lineage.sha256_file(
                Path(__file__).resolve().parent
                / "portal_fiber_qrsp_production.py"
            ),
            "portal_fiber_qrsp_production_gate.py": lineage.sha256_file(
                Path(__file__).resolve().parent
                / "portal_fiber_qrsp_production_gate.py"
            ),
        },
        "active_hierarchy": active_hierarchy_diagnostics,
        "mathematical_objective": (
            "For each OD family, minimize KL(q||P_global) minus the expected clipped "
            "OD log-likelihood tilt; then minimize expected graph cost plus beta^{-1} "
            "KL divergence from the resulting reference hitting-path law."
            if args.reference_conditioning in {
                "od-family", "od-factorized", "od-lifted", "endpoint-lifted",
                "relative-direction", "portal-fiber",
            }
            else "For each destination and beta, minimize expected graph cost plus beta^{-1} "
            "KL divergence from the released-flow reference hitting-path law."
        ),
        "limit": limit,
        "quotient_regions": int(args.regions),
        "released_flow_block": flow_name if args.reference_source == "dp-flow" else None,
        "od_conditioned_flow_block": (
            (
                "distance_coarse24_flow"
                if args.route_release_schema == "distance4-coarse24"
                else "origin_macro4_coarse24_flow + destination_macro4_coarse24_flow"
                if args.route_release_schema == "endpoint8-coarse24"
                else "distance_relative_direction"
                if args.route_release_schema == "relative32-nested384"
                else "portal_fiber_flow"
                if args.route_release_schema == "portal-fiber-nested384"
                else "od_conditioned_fine96_flow"
            )
            if args.reference_source == "dp-flow"
            and args.reference_conditioning != "global"
            else None
        ),
        "od_entropic_barycenter_weight": (
            float(conditioned_weight)
            if args.reference_conditioning != "global"
            else None
        ),
        "od_factor_weights_distance_bearing": (
            list(factor_weights)
            if args.reference_conditioning == "od-factorized"
            else None
        ),
        "coarse_to_fine_operator": (
            {
                "from_regions": (
                    24
                    if args.route_release_schema in {"distance4-coarse24", "endpoint8-coarse24"}
                    else 96
                ),
                "to_regions": 384,
                "type": (
                    "nested conservative edge-flow KL projection"
                    if args.reference_conditioning == "endpoint-lifted"
                    else "bounded likelihood-ratio pullback"
                ),
            }
            if args.reference_conditioning in {"od-lifted", "endpoint-lifted"}
            else None
        ),
        "relative_direction_operator": (
            "bounded edge-bearing likelihood-ratio information projection"
            if args.reference_conditioning == "relative-direction"
            else None
        ),
        "portal_fiber_operator": (
            "bounded portal-fiber likelihood-ratio information projection"
            if args.reference_conditioning == "portal-fiber"
            else None
        ),
        "od_likelihood_ratio_cap": (
            float(args.od_likelihood_ratio_cap)
            if args.reference_conditioning != "global"
            else None
        ),
        "betas_inverse_km": list(betas),
        "candidate_source_counts": dict(
            Counter(item.get("candidate_source", "qrsp") for item in diagnostics)
        ),
        "length_conditioning": {
            "proposal_count": int(args.length_proposals),
            "potential": "exp(-lambda * abs(log((road_length+0.1)/(target_length+0.1))))",
            "lambda": float(args.length_log_penalty),
            "selected_log_error_mean": (
                float(np.mean(selected_length_errors)) if selected_length_errors else None
            ),
            "selected_log_error_p90": (
                float(np.quantile(selected_length_errors, 0.9)) if selected_length_errors else None
            ),
            "all_proposal_log_error_mean": (
                float(np.mean(proposal_length_errors)) if proposal_length_errors else None
            ),
        },
        "beta_counts": beta_counts,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / max(limit, 1),
        "restricted_router_global_fallback_segments": restricted_fallback_segments,
        "region_steps_mean": float(np.mean(valid_steps)) if valid_steps else None,
        "region_steps_p90": float(np.quantile(valid_steps, 0.9)) if valid_steps else None,
        "kernel_checks": kernel_checks,
        "conservative_projection_checks": conservative_projection_checks,
        "graph": graph_diag,
        "shrinkage": shrinkage,
        "maximum_desirability_residual": maximum_bridge_residual(),
        "inputs": {
            "request_sha256": request_sha256,
            "dp_route_release_sha256": release_sha256,
            "base_sanitizer_manifest_sha256": (
                request_manifest["sanitizer_manifest_sha256"]
                if production_mode
                else None
            ),
            "development_request_manifest_sha256": (
                lineage.sha256_file(
                    args.production_root / "requests" / "request_manifest.json"
                )
                if not production_mode
                else None
            ),
            "portal_manifest_sha256": (
                portal_manifest_sha256
                if production_mode
                else None
            ),
            "public_candidate_matrix_sha256": (
                lineage.sha256_file(args.public_matrix_cache)
                if args.public_matrix_cache is not None
                else None
            ),
            "public_osm_sha256": (
                request_manifest["public_osm_sha256"]
                if production_mode
                else lineage.sha256_file(osm_path)
            ),
            "fallback_input_sha256": (
                lineage.sha256_file(fallback_path) if fallback_path is not None else None
            ),
            "public_fallback_cache_sha256": (
                lineage.sha256_file(args.public_fallback_cache)
                if args.public_fallback_cache is not None and args.public_fallback_cache.is_file()
                else None
            ),
            "production_fallback_source": (
                "fresh public-road shortest paths conditioned only on DP requests"
                if production_mode
                else None
            ),
            "candidate_count_used_only_for_public_NERS_baseline": 8,
        },
        "outputs": {
            synthetic_name: lineage.sha256_file(staging / synthetic_name),
            "road_witnesses.pkl": lineage.sha256_file(staging / "road_witnesses.pkl"),
            control_name: lineage.sha256_file(staging / control_name),
        },
    }
    (staging / "protocol.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(staging, args.out_dir)
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
ALLOW_AUTHENTICATED_ABLATION_CHECKPOINT = False
