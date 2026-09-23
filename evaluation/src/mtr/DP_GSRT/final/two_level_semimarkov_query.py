"""Privacy-critical query for the two-level semi-Markov DP bridge."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

import dp_graph_flow_routing_probe as flow


WEIGHTS = {
    "coarse_occupancy": 0.20,
    "coarse_transition": 0.18,
    "fine_occupancy": 0.15,
    "fine_transition": 0.15,
    "coarse_dwell": 0.05,
    "road_texture": 0.27,
}
DWELL_EDGES = np.asarray([0.0, 0.05, 0.15, 0.35, 0.65, 1.0 + 1e-12], dtype=float)
ROAD_DIRECTION_BINS = 8
ROAD_TEXTURE_BUCKETS = 1024


@dataclass(frozen=True)
class QueryContext:
    road_tree: cKDTree
    road_edge_tree: cKDTree
    node_coarse: np.ndarray
    node_fine: np.ndarray
    coarse_tree: cKDTree
    fine_tree: cKDTree
    coarse_adjacency: dict[int, set[int]]
    fine_adjacency: dict[int, set[int]]
    coarse_edge_index: dict[tuple[int, int], int]
    fine_edge_index: dict[tuple[int, int], int]
    coarse_to_macro: np.ndarray
    fine_importance: np.ndarray
    phases: int
    coarse_regions: int
    fine_regions: int
    macro_regions: int


def _segment_geometry(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        arr = np.asarray(trajectory, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return np.empty((0, 2)), np.empty(0), np.empty(0)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 2 or not np.all(np.isfinite(arr[:, :2])):
        return np.empty((0, 2)), np.empty(0), np.empty(0)
    points = arr[:, :2]
    if np.any(np.abs(points[:, 0]) > 90.0) or np.any(np.abs(points[:, 1]) > 180.0):
        return np.empty((0, 2)), np.empty(0), np.empty(0)
    latitude = float(np.mean(points[:, 0]))
    scale = np.asarray([111.32, 111.32 * np.cos(np.deg2rad(latitude))], dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.diff(points, axis=0) * scale
        lengths = np.linalg.norm(delta, axis=1)
    if not np.all(np.isfinite(lengths)):
        return np.empty((0, 2)), np.empty(0), np.empty(0)
    valid = lengths > 1e-12
    if not np.any(valid):
        return np.empty((0, 2)), np.empty(0), np.empty(0)
    midpoints = 0.5 * (points[:-1] + points[1:])
    midpoints = midpoints[valid]
    lengths = lengths[valid]
    cumulative = np.cumsum(lengths) - 0.5 * lengths
    phases = cumulative / max(float(lengths.sum()), 1e-15)
    return midpoints, lengths, phases


def _road_texture(
    midpoints: np.ndarray,
    phase_fraction: np.ndarray,
    edge_tree: cKDTree,
    *,
    family: int,
    families: int,
    phases: int,
) -> np.ndarray:
    result = np.zeros((int(families), int(phases), ROAD_TEXTURE_BUCKETS), dtype=float)
    if len(midpoints) < 2:
        return result
    edge_ids = np.asarray(edge_tree.query(midpoints)[1], dtype=int)
    compact = []
    compact_positions = []
    for index, edge in enumerate(edge_ids):
        if not compact or compact[-1] != int(edge):
            compact.append(int(edge))
            compact_positions.append(int(index))
    if len(compact) < 2:
        return result
    events = []
    for index, (incoming, outgoing) in enumerate(zip(compact[:-1], compact[1:])):
        position = compact_positions[index]
        phase = min(int(phases) - 1, int(float(phase_fraction[position]) * int(phases)))
        bucket = int((incoming * 2654435761 + outgoing * 2246822519) % ROAD_TEXTURE_BUCKETS)
        events.append((phase, bucket))
    mass = 1.0 / max(len(events), 1)
    for phase, bucket in events:
        result[int(family), int(phase), int(bucket)] += mass
    return result


def _phase_index(fractions: np.ndarray, phases: int) -> np.ndarray:
    return np.minimum((np.asarray(fractions, dtype=float) * int(phases)).astype(int), int(phases) - 1)


def _occupancy(
    labels: np.ndarray,
    lengths: np.ndarray,
    phase_fraction: np.ndarray,
    *,
    family: int,
    families: int,
    phases: int,
    regions: int,
    importance: np.ndarray | None = None,
) -> np.ndarray:
    result = np.zeros((int(families), int(phases), int(regions)), dtype=float)
    weights = np.asarray(lengths, dtype=float).copy()
    if importance is not None:
        weights *= np.asarray(importance, dtype=float)[np.asarray(labels, dtype=int)]
    total = float(np.sum(weights))
    if total <= 0.0:
        return result
    phase_ids = _phase_index(phase_fraction, phases)
    for region, phase, weight in zip(labels, phase_ids, weights):
        result[int(family), int(phase), int(region)] += float(weight) / total
    return result


def _expanded_with_phase(labels: np.ndarray, adjacency: dict[int, set[int]]) -> tuple[list[int], list[float]]:
    labels = np.asarray(labels, dtype=int).ravel()
    if labels.size == 0:
        return [], []
    compact = flow.compress(labels)
    expanded = flow.expanded_region_sequence(compact, adjacency)
    if len(expanded) < 2:
        return expanded, []
    phase_fraction = [(index + 0.5) / max(len(expanded) - 1, 1) for index in range(len(expanded) - 1)]
    return expanded, phase_fraction


def _transitions(
    labels: np.ndarray,
    adjacency: dict[int, set[int]],
    edge_index: dict[tuple[int, int], int],
    *,
    family: int,
    families: int,
    phases: int,
    importance: np.ndarray | None = None,
) -> np.ndarray:
    result = np.zeros((int(families), int(phases), len(edge_index)), dtype=float)
    sequence, phase_fraction = _expanded_with_phase(labels, adjacency)
    valid = []
    for index, (u, v) in enumerate(zip(sequence[:-1], sequence[1:])):
        edge = edge_index.get((int(u), int(v)))
        if edge is not None:
            weight = 1.0
            if importance is not None:
                weight = float(np.sqrt(float(importance[int(u)]) * float(importance[int(v)])))
            valid.append((edge, phase_fraction[index], weight))
    if not valid:
        return result
    normalizer = max(float(sum(item[2] for item in valid)), 1e-15)
    for edge, fraction, weight in valid:
        phase = min(int(phases) - 1, int(float(fraction) * int(phases)))
        result[int(family), phase, int(edge)] += float(weight) / normalizer
    return result


def _dwell(
    labels: np.ndarray,
    lengths: np.ndarray,
    phase_fraction: np.ndarray,
    *,
    family: int,
    families: int,
    phases: int,
) -> np.ndarray:
    result = np.zeros((int(families), int(phases), len(DWELL_EDGES) - 1), dtype=float)
    if len(labels) == 0:
        return result
    total = max(float(np.sum(lengths)), 1e-15)
    runs = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or int(labels[index]) != int(labels[start]):
            run_length = float(np.sum(lengths[start:index]))
            midpoint = float(np.sum(lengths[:start]) + 0.5 * run_length) / total
            runs.append((run_length / total, midpoint))
            start = index
    mass = 1.0 / max(len(runs), 1)
    for fraction, midpoint in runs:
        dwell_bin = min(
            max(int(np.searchsorted(DWELL_EDGES, fraction, side="right") - 1), 0),
            len(DWELL_EDGES) - 2,
        )
        phase = min(int(phases) - 1, int(midpoint * int(phases)))
        result[int(family), phase, dwell_bin] += mass
    return result


def trajectory_blocks(trajectory: np.ndarray, context: QueryContext) -> dict[str, np.ndarray]:
    midpoints, lengths, phase_fraction = _segment_geometry(trajectory)
    coarse_families = int(context.macro_regions) ** 2
    fine_families = int(context.macro_regions)
    if len(midpoints) == 0:
        return {
            "coarse_occupancy": np.zeros((coarse_families, context.phases, context.coarse_regions)),
            "coarse_transition": np.zeros((coarse_families, context.phases, len(context.coarse_edge_index))),
            "fine_occupancy": np.zeros((fine_families, context.phases, context.fine_regions)),
            "fine_transition": np.zeros((fine_families, context.phases, len(context.fine_edge_index))),
            "coarse_dwell": np.zeros((coarse_families, context.phases, len(DWELL_EDGES) - 1)),
            "road_texture": np.zeros((fine_families, context.phases, ROAD_TEXTURE_BUCKETS)),
        }
    road_nodes = np.asarray(context.road_tree.query(midpoints)[1], dtype=int)
    coarse_labels = np.asarray(context.node_coarse[road_nodes], dtype=int)
    fine_labels = np.asarray(context.node_fine[road_nodes], dtype=int)
    origin_macro = int(context.coarse_to_macro[int(coarse_labels[0])])
    destination_macro = int(context.coarse_to_macro[int(coarse_labels[-1])])
    coarse_family = origin_macro * int(context.macro_regions) + destination_macro
    return {
        "coarse_occupancy": _occupancy(
            coarse_labels,
            lengths,
            phase_fraction,
            family=coarse_family,
            families=coarse_families,
            phases=context.phases,
            regions=context.coarse_regions,
        ),
        "coarse_transition": _transitions(
            coarse_labels,
            context.coarse_adjacency,
            context.coarse_edge_index,
            family=coarse_family,
            families=coarse_families,
            phases=context.phases,
        ),
        "fine_occupancy": _occupancy(
            fine_labels,
            lengths,
            phase_fraction,
            family=origin_macro,
            families=fine_families,
            phases=context.phases,
            regions=context.fine_regions,
            importance=context.fine_importance,
        ),
        "fine_transition": _transitions(
            fine_labels,
            context.fine_adjacency,
            context.fine_edge_index,
            family=origin_macro,
            families=fine_families,
            phases=context.phases,
            importance=context.fine_importance,
        ),
        "coarse_dwell": _dwell(
            coarse_labels,
            lengths,
            phase_fraction,
            family=coarse_family,
            families=coarse_families,
            phases=context.phases,
        ),
        "road_texture": _road_texture(
            midpoints,
            phase_fraction,
            context.road_edge_tree,
            family=origin_macro,
            families=fine_families,
            phases=context.phases,
        ),
    }


def contribution_l1(blocks: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    norms = {name: float(np.sum(np.abs(values))) for name, values in blocks.items()}
    weighted = float(sum(float(WEIGHTS[name]) * norm for name, norm in norms.items()))
    return weighted, norms


def add_to_measurements(measurements: dict[str, np.ndarray], blocks: dict[str, np.ndarray]) -> None:
    for name, values in blocks.items():
        measurements[name] += float(WEIGHTS[name]) * values
