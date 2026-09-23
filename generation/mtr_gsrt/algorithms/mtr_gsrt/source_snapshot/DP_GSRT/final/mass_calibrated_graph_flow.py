"""Mass-calibrated hierarchical graph-flow query and DP-safe post-processing."""
from __future__ import annotations

import random
from fractions import Fraction

import numpy as np

import multires_graph_flow_experiment as legacy
from certified_discrete_dp import add_exact_discrete_laplace
from dp_two_level_semimarkov_release import ROUTE_LATTICE, hamilton_quantize_block


PUBLIC_CAPACITY = 17_123
OD_FAMILIES = legacy.OD_FAMILIES


def activity_split(parent_budget: int, flow_coordinates_per_family: int) -> tuple[int, int]:
    """Dimension-balanced split minimizing F/gamma^2 + d/(1-gamma)^2."""
    if int(parent_budget) <= 0 or int(flow_coordinates_per_family) <= 0:
        raise ValueError("Budgets and dimensions must be positive")
    gamma = 1.0 / (1.0 + float(flow_coordinates_per_family) ** (1.0 / 3.0))
    activity = int(round(int(parent_budget) * gamma))
    return activity, int(parent_budget) - activity


FINE96_ACTIVITY_BUDGET, FINE96_FLOW_BUDGET = activity_split(
    legacy.BLOCK_BUDGETS["fine96_flow"], 438
)
FINE384_ACTIVITY_BUDGET, FINE384_FLOW_BUDGET = activity_split(
    legacy.BLOCK_BUDGETS["fine384_flow"], 1746
)
OD_ACTIVITY_BUDGET, OD_FLOW_BUDGET = activity_split(
    legacy.BLOCK_BUDGETS["od_conditioned_fine96_flow"], 438
)

COMPONENT_BUDGETS = {
    "coarse24_occupancy": legacy.BLOCK_BUDGETS["coarse24_occupancy"],
    "fine384_occupancy": legacy.BLOCK_BUDGETS["fine384_occupancy"],
    "fine96_activity": FINE96_ACTIVITY_BUDGET,
    "fine96_flow": FINE96_FLOW_BUDGET,
    "fine384_activity": FINE384_ACTIVITY_BUDGET,
    "fine384_flow": FINE384_FLOW_BUDGET,
    "od_fine96_activity": OD_ACTIVITY_BUDGET,
    "od_conditioned_fine96_flow": OD_FLOW_BUDGET,
}
if sum(COMPONENT_BUDGETS.values()) != ROUTE_LATTICE:
    raise RuntimeError("Mass-calibrated component budgets must sum to Q")


def empty_components(context96, context384) -> dict[str, np.ndarray]:
    return {
        "coarse24_occupancy": np.zeros(context96.coarse_regions),
        "fine384_occupancy": np.zeros(context384.fine_regions),
        "fine96_activity": np.zeros(1),
        "fine96_flow": np.zeros(len(context96.fine_edge_index)),
        "fine384_activity": np.zeros(1),
        "fine384_flow": np.zeros(len(context384.fine_edge_index)),
        "od_fine96_activity": np.zeros(OD_FAMILIES),
        "od_conditioned_fine96_flow": np.zeros(
            OD_FAMILIES * len(context96.fine_edge_index)
        ),
    }


def graph_components(
    trajectory: np.ndarray, context96, context384
) -> dict[str, np.ndarray]:
    blocks = legacy.graph_blocks(trajectory, context96, context384)
    fine96_active = float(np.sum(blocks["fine96_flow"]) > 0.0)
    fine384_active = float(np.sum(blocks["fine384_flow"]) > 0.0)
    family_activity = np.zeros(OD_FAMILIES, dtype=float)
    if fine96_active:
        family_activity[legacy.od_family(trajectory)] = 1.0
    return {
        "coarse24_occupancy": blocks["coarse24_occupancy"],
        "fine384_occupancy": blocks["fine384_occupancy"],
        "fine96_activity": np.asarray([fine96_active]),
        "fine96_flow": blocks["fine96_flow"],
        "fine384_activity": np.asarray([fine384_active]),
        "fine384_flow": blocks["fine384_flow"],
        "od_fine96_activity": family_activity,
        "od_conditioned_fine96_flow": blocks["od_conditioned_fine96_flow"],
    }


def quantized_contribution(
    trajectory: np.ndarray, context96, context384
) -> dict[str, np.ndarray]:
    components = graph_components(trajectory, context96, context384)
    result = {}
    for name, budget in COMPONENT_BUDGETS.items():
        quantized = hamilton_quantize_block(components[name], int(budget))
        result[name] = (
            np.zeros_like(components[name], dtype=np.int64)
            if quantized is None
            else np.asarray(quantized, dtype=np.int64)
        )
    if sum(int(value.sum()) for value in result.values()) > ROUTE_LATTICE:
        raise AssertionError("Per-record component mass exceeds Q")
    return result


def aggregate(
    trajectories: list[np.ndarray], context96, context384
) -> dict[str, np.ndarray]:
    totals = {
        name: np.zeros_like(value, dtype=np.int64)
        for name, value in empty_components(context96, context384).items()
    }
    for trajectory in trajectories:
        contribution = quantized_contribution(trajectory, context96, context384)
        for name in COMPONENT_BUDGETS:
            totals[name] += contribution[name]
    return totals


def project_simplex_exact(values: np.ndarray, total: float) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum(x) = total}, including total=0."""
    flat = np.asarray(values, dtype=float).ravel()
    target = float(total)
    if not np.isfinite(target) or target < 0.0:
        raise ValueError("Projection total must be finite and nonnegative")
    if target == 0.0:
        return np.zeros_like(values, dtype=float)
    order = np.sort(flat)[::-1]
    cumulative = np.cumsum(order) - target
    divisor = np.arange(1, len(order) + 1, dtype=float)
    positive = order - cumulative / divisor > 0.0
    rho = int(np.flatnonzero(positive)[-1]) if np.any(positive) else 0
    threshold = float(cumulative[rho] / (rho + 1))
    projected = np.maximum(flat - threshold, 0.0)
    projected *= target / max(float(projected.sum()), 1e-15)
    return projected.reshape(np.shape(values))


def _bounded_active_count(
    noisy_activity: np.ndarray, activity_budget: int, capacity: int
) -> tuple[np.ndarray, float]:
    maximum = float(capacity * int(activity_budget))
    released_integer = np.clip(np.asarray(noisy_activity, dtype=float), 0.0, maximum)
    count = float(released_integer.sum() / int(activity_budget))
    return released_integer, min(max(count, 0.0), float(capacity))


def postprocess_noisy(
    noisy: dict[str, np.ndarray], *, capacity: int
) -> tuple[dict[str, np.ndarray], dict]:
    """Restore activity-dependent flow mass using only the noisy transcript."""
    capacity = int(capacity)
    released: dict[str, np.ndarray] = {}
    for name in ("coarse24_occupancy", "fine384_occupancy"):
        total = float(capacity * COMPONENT_BUDGETS[name])
        released[name] = project_simplex_exact(noisy[name], total) / ROUTE_LATTICE

    fine96_activity, fine96_count = _bounded_active_count(
        noisy["fine96_activity"], FINE96_ACTIVITY_BUDGET, capacity
    )
    fine384_activity, fine384_count = _bounded_active_count(
        noisy["fine384_activity"], FINE384_ACTIVITY_BUDGET, capacity
    )
    released["fine96_activity"] = fine96_activity / ROUTE_LATTICE
    released["fine384_activity"] = fine384_activity / ROUTE_LATTICE
    released["fine96_flow"] = (
        project_simplex_exact(
            noisy["fine96_flow"], fine96_count * FINE96_FLOW_BUDGET
        )
        / ROUTE_LATTICE
    )
    released["fine384_flow"] = (
        project_simplex_exact(
            noisy["fine384_flow"], fine384_count * FINE384_FLOW_BUDGET
        )
        / ROUTE_LATTICE
    )

    # The OD-family activity vector and global fine96 activity scalar measure
    # the same public event, so enforce their exact aggregate consistency.
    family_activity_total = fine96_count * OD_ACTIVITY_BUDGET
    od_activity = project_simplex_exact(
        noisy["od_fine96_activity"], family_activity_total
    )
    released["od_fine96_activity"] = od_activity / ROUTE_LATTICE
    family_counts = od_activity / OD_ACTIVITY_BUDGET
    edge_count = int(np.asarray(noisy["od_conditioned_fine96_flow"]).size // OD_FAMILIES)
    noisy_family_flow = np.asarray(
        noisy["od_conditioned_fine96_flow"], dtype=float
    ).reshape(OD_FAMILIES, edge_count)
    family_flow = np.zeros_like(noisy_family_flow)
    for family in range(OD_FAMILIES):
        family_flow[family] = project_simplex_exact(
            noisy_family_flow[family],
            float(family_counts[family] * OD_FLOW_BUDGET),
        )
    released["od_conditioned_fine96_flow"] = family_flow.ravel() / ROUTE_LATTICE

    diagnostics = {
        "fine96_active_count": fine96_count,
        "fine384_active_count": fine384_count,
        "fine96_active_fraction": fine96_count / max(capacity, 1),
        "fine384_active_fraction": fine384_count / max(capacity, 1),
        "od_family_activity_sum": float(family_counts.sum()),
        "hierarchical_activity_residual": float(family_counts.sum() - fine96_count),
        "released_flow_mass": {
            "fine96_flow": float(released["fine96_flow"].sum()),
            "fine384_flow": float(released["fine384_flow"].sum()),
            "od_conditioned_fine96_flow": float(
                released["od_conditioned_fine96_flow"].sum()
            ),
        },
    }
    return released, diagnostics


def release(
    totals: dict[str, np.ndarray],
    *,
    capacity: int,
    exact_rng: random.Random,
    epsilon: Fraction,
    production_randomness: bool,
) -> tuple[dict[str, np.ndarray], dict, dict]:
    if production_randomness and not isinstance(exact_rng, random.SystemRandom):
        raise RuntimeError("Production release requires fresh random.SystemRandom")
    if not production_randomness and isinstance(exact_rng, random.SystemRandom):
        raise RuntimeError("Development release must be explicitly reproducible and NON_DP")
    exact = np.concatenate(
        [np.asarray(totals[name], dtype=np.int64).ravel() for name in COMPONENT_BUDGETS]
    )
    noisy_vector, sampler = add_exact_discrete_laplace(
        exact,
        epsilon_numerator=epsilon.numerator,
        epsilon_denominator=epsilon.denominator,
        sensitivity=ROUTE_LATTICE,
        rng=exact_rng,
    )
    noisy, offset = {}, 0
    for name in COMPONENT_BUDGETS:
        template = totals[name]
        size = int(np.asarray(template).size)
        noisy[name] = noisy_vector[offset : offset + size].reshape(np.shape(template))
        offset += size
    if offset != len(noisy_vector):
        raise AssertionError("Noisy transcript was not partitioned exactly")
    released, diagnostics = postprocess_noisy(noisy, capacity=int(capacity))
    return released, sampler, diagnostics
