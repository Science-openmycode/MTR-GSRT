"""Candidate-fixed DP reward probe.

This prototype asks a narrower question than the full generator:

    If the public candidate family is held fixed, does a DP transition reward
    select routes whose route-choice statistics are closer to held-out real
    trajectories than public length-only selection?

The script is intentionally separate from production/paper code. It reuses the
repository loaders and graph helpers, but writes only exploration artifacts.
"""
from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
ARA_CODEX = ROOT / "ARA_codex"
SCRIPT_DIR = ARA_CODEX / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from route_structure_potential_experiment import (  # noqa: E402
    BEIJING_BBOX,
    DIVERSE_REROUTE_PENALTY,
    EXP_N_GEN,
    SEED,
    TRANS_GRID,
    cell_id,
    dp_anchors,
    load_geolife,
    load_osm_pickle,
    normalize_with_public_bbox,
    path_length,
    penalize_path_edges,
    prepare_graph,
    resample,
    sample_exact_len,
    sample_joint_draws,
    shortest_path,
    transition_node_cells,
)
from transition_bridge_experiment import edge_metric_suite, js_from_counts  # noqa: E402


OUT_DIR = ARA_CODEX / "dp_reward_exploration" / "results"
OD_BINS = np.asarray([0.0, 0.12, 0.25, 0.50, 0.80, np.inf])
LENGTH_BINS = np.asarray([0, 12, 24, 48, 96, np.inf])
OD_PAIR_GRID = 4
PHASE_BINS = 3
FEATURE_GRID = 8
EPS_SWEEP = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)
ALPHA_LENGTH = 0.30
BETA_REWARD = 1.35
SOFTMAX_TEMP = 0.65
SMOOTH_ETA = 1e-6
BACKOFF_TAU = 3.0
FACTOR_WEIGHTS = (0.40, 0.30, 0.30)  # OD, length, phase; sum <= 1 for sensitivity.
ROUTE_FEATURE_SWITCH_MARGIN = 0.08
ADAPTIVE_SWITCH_BASE = 0.02
ADAPTIVE_SWITCH_DIFF = 0.05
ADAPTIVE_SWITCH_LENGTH = 0.30
CONSENSUS_MIN_WEIGHT = 0.60
HYBRID_ROUTE_WEIGHT = 0.55
GRAPH_FLOW_PUBLIC_PRIOR = 5.0
GRAPH_FLOW_LCB_WEIGHT = 0.08
GRAPH_FLOW_LLR_CLIP = 4.0
CORRIDOR_BASIS_DIM = 12
CORRIDOR_SCORE_CLIP = 1.0
CYCLE_MOTIF_SCORE_CLIP = 1.0
CYCLE_MOTIF_LCB_Z = 1.0
HODGE_CYCLE_BASIS_DIM = 12
HODGE_CYCLE_SCORE_CLIP = 1.0
ELECTRICAL_CYCLE_BASIS_DIM = 12
ELECTRICAL_CYCLE_SCORE_CLIP = 1.0
LOCAL_CORRIDOR_BASIS_DIM = 4
LOCAL_CORRIDOR_SCORE_CLIP = 1.0
CLUSTER_CORRIDOR_BASIS_DIM = 3
CLUSTER_CORRIDOR_COUNT = 4
CLUSTER_CORRIDOR_SCORE_CLIP = 1.0
HIER_CLUSTER_WEIGHTS = (0.20, 0.30, 0.50)  # global, OD-distance, OD-cell; L1 sensitivity budget sums to one.
ANCHOR_CHOICE_WEIGHTS = (0.35, 0.65)  # global anchor vote, OD-conditioned anchor vote; sums to one.
ANCHOR_SHAPE_WEIGHTS = (0.30, 0.70)  # global OD-corridor shape vote, OD-conditioned shape vote.
ANCHOR_SHAPE_PHASE_BINS = np.asarray([0.0, 0.20, 0.40, 0.60, 0.80, 1.0000001])
ANCHOR_SHAPE_LATERAL_BINS = np.asarray([-np.inf, -0.45, -0.25, -0.10, 0.10, 0.25, 0.45, np.inf])
CUT_CORRIDOR_LAYERS = 4
CUT_CORRIDOR_SCORE_CLIP = 1.0
CUT_BAND_LATERAL_BUCKETS = 5
CUT_BAND_DIRECTION_BUCKETS = 4
OD_CUT_BAND_LAYERS = 4
OD_CUT_BAND_LATERAL_BUCKETS = 5
OD_CUT_BAND_DIRECTION_BUCKETS = 4
CUT_BAND_FAMILY_COUNT = 4
SKIP_EDGE_METRICS = False


def log_status(message: str) -> None:
    print(f"[reward-probe] {message}", flush=True)


def load_geolife_limited(limit: int) -> list[np.ndarray]:
    try:
        return load_geolife(limit=limit)
    except TypeError:
        load_geolife.__globals__["limit"] = limit
        return load_geolife()[:limit]


def od_bin_for_points(a: np.ndarray, b: np.ndarray, mn: np.ndarray, span: np.ndarray) -> int:
    na = (np.asarray(a, dtype=float) - mn) / span
    nb = (np.asarray(b, dtype=float) - mn) / span
    d = float(np.linalg.norm(na - nb))
    return min(max(int(np.searchsorted(OD_BINS, d, side="right") - 1), 0), len(OD_BINS) - 2)


def od_pair_cell_bin_for_points(a: np.ndarray, b: np.ndarray, mn: np.ndarray, span: np.ndarray, grid: int = OD_PAIR_GRID) -> int:
    grid = int(grid)
    orow, ocol = cell_id(np.asarray(a, dtype=float), mn, span, grid)
    drow, dcol = cell_id(np.asarray(b, dtype=float), mn, span, grid)
    ocell = int(orow) * grid + int(ocol)
    dcell = int(drow) * grid + int(dcol)
    return int(ocell * grid * grid + dcell)


def length_bin_for_len(length: int) -> int:
    return min(max(int(np.searchsorted(LENGTH_BINS, int(length), side="right") - 1), 0), len(LENGTH_BINS) - 2)


def compact_cell_sequence(traj: np.ndarray, mn: np.ndarray, span: np.ndarray, grid: int) -> list[int]:
    out: list[int] = []
    for p in np.asarray(traj, dtype=float):
        r, c = cell_id(p, mn, span, grid)
        idx = int(r * grid + c)
        if not out or out[-1] != idx:
            out.append(idx)
    return out


def dp_global_transition(train: list[np.ndarray], mn: np.ndarray, span: np.ndarray, rng: np.random.Generator, eps: float, grid: int) -> np.ndarray:
    n = grid * grid
    hist = np.zeros((n, n), dtype=float)
    for t in train:
        cells = compact_cell_sequence(np.asarray(t), mn, span, grid)
        trans = [(a, b) for a, b in zip(cells[:-1], cells[1:]) if a != b]
        if not trans:
            continue
        counts: dict[tuple[int, int], float] = {}
        for a, b in trans:
            counts[(a, b)] = counts.get((a, b), 0.0) + 1.0
        total = max(float(sum(counts.values())), 1e-12)
        for (a, b), v in counts.items():
            hist[a, b] += v / total
    if eps <= 0:
        noisy = np.zeros_like(hist)
    else:
        noisy = np.maximum(hist + rng.laplace(0.0, 1.0 / eps, size=hist.shape), 0.0)
    row = noisy + SMOOTH_ETA
    row = row / np.maximum(row.sum(axis=1, keepdims=True), 1e-12)
    col = row.mean(axis=0)
    col = col / max(float(col.sum()), 1e-12)
    return np.log(row) - np.log(col[None, :] + 1e-12)


def public_candidate_transition_flow(candidate_bank: list[dict], node_cells: np.ndarray, grid: int) -> np.ndarray:
    """Public baseline occupation over candidate-0 cell transitions."""
    n = grid * grid
    hist = np.zeros((n, n), dtype=float)
    for req_idx, item in enumerate(candidate_bank):
        cands = item.get("candidates", [])
        if not cands:
            continue
        cells = compact_node_cells(cands[0][1], node_cells)
        trans = [(a, b) for a, b in zip(cells[:-1], cells[1:]) if a != b]
        if not trans:
            continue
        total = max(float(len(trans)), 1e-12)
        for a, b in trans:
            hist[int(a), int(b)] += 1.0 / total
    return hist


def transition_occupation_from_cells(cells: list[int], grid: int) -> np.ndarray:
    n = grid * grid
    hist = np.zeros((n, n), dtype=float)
    trans = [(int(a), int(b)) for a, b in zip(cells[:-1], cells[1:]) if int(a) != int(b)]
    if not trans:
        return hist
    total = max(float(len(trans)), 1e-12)
    for a, b in trans:
        hist[a, b] += 1.0 / total
    return hist


def path_transition_occupation(nodes: list[int], node_cells: np.ndarray, grid: int) -> np.ndarray:
    return transition_occupation_from_cells(compact_node_cells(nodes, node_cells), grid)


def compact_cell_edges(cells: list[int]) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in zip(cells[:-1], cells[1:]) if int(a) != int(b)]


def path_edge_count_vector(cells: list[int], edge_index: dict[tuple[int, int], int]) -> np.ndarray:
    vec = np.zeros(len(edge_index), dtype=float)
    for edge in compact_cell_edges(cells):
        idx = edge_index.get(edge)
        if idx is not None:
            vec[int(idx)] += 1.0
    return vec


def build_incidence(edge_index: dict[tuple[int, int], int], n_nodes: int) -> np.ndarray:
    incidence = np.zeros((n_nodes, len(edge_index)), dtype=float)
    for (src, dst), idx in edge_index.items():
        incidence[int(src), int(idx)] -= 1.0
        incidence[int(dst), int(idx)] += 1.0
    return incidence


def electrical_flow_vector(src: int, dst: int, incidence: np.ndarray, lap_pinv: np.ndarray) -> np.ndarray:
    if src == dst or incidence.size == 0:
        return np.zeros(incidence.shape[1], dtype=float)
    b = np.zeros(incidence.shape[0], dtype=float)
    b[int(src)] -= 1.0
    b[int(dst)] += 1.0
    return incidence.T @ (lap_pinv @ b)


def normalized_electrical_residual(
    cells: list[int],
    edge_index: dict[tuple[int, int], int],
    incidence: np.ndarray,
    lap_pinv: np.ndarray,
) -> np.ndarray:
    if not cells:
        return np.zeros(len(edge_index), dtype=float)
    vec = path_edge_count_vector(cells, edge_index)
    flow = electrical_flow_vector(int(cells[0]), int(cells[-1]), incidence, lap_pinv)
    residual = vec - flow
    l1 = float(np.sum(np.abs(residual)))
    if l1 <= 1e-12:
        return np.zeros_like(residual)
    return residual / l1


def build_electrical_cycle_basis(
    candidate_bank: list[dict],
    node_cells: np.ndarray,
    grid: int,
    basis_dim: int,
) -> dict:
    """Public graph-Laplacian cycle residual basis.

    For each same-OD candidate path, subtract the public electrical s-t flow
    induced by the candidate cell graph. The residual removes the OD potential
    component and lies in the cycle space up to numerical precision.
    """
    n_nodes = grid * grid
    edge_index: dict[tuple[int, int], int] = {}
    candidate_cells: list[list[list[int]]] = []
    for item in candidate_bank:
        req_cells = []
        for _, nodes in item.get("candidates", []):
            cells = compact_node_cells(nodes, node_cells)
            req_cells.append(cells)
            for edge in compact_cell_edges(cells):
                if edge not in edge_index:
                    edge_index[edge] = len(edge_index)
        candidate_cells.append(req_cells)

    incidence = build_incidence(edge_index, n_nodes)
    lap = incidence @ incidence.T
    lap_pinv = np.linalg.pinv(lap, rcond=1e-8) if lap.size else np.zeros((n_nodes, n_nodes), dtype=float)

    residuals = []
    lookup: dict[tuple[int, int], int] = {}
    basis_rows = []
    divergence_errors = []
    for req_idx, req_cells in enumerate(candidate_cells):
        if not req_cells or not req_cells[0]:
            continue
        public_residual = normalized_electrical_residual(req_cells[0], edge_index, incidence, lap_pinv)
        divergence_errors.append(float(np.linalg.norm(incidence @ public_residual, ord=1)))
        for cand_idx, cells in enumerate(req_cells[1:], start=1):
            if not cells:
                continue
            alt_residual = normalized_electrical_residual(cells, edge_index, incidence, lap_pinv)
            divergence_errors.append(float(np.linalg.norm(incidence @ alt_residual, ord=1)))
            diff = alt_residual - public_residual
            l1 = float(np.sum(np.abs(diff)))
            if l1 <= 1e-12:
                continue
            diff = diff / l1
            lookup[(int(req_idx), int(cand_idx))] = len(residuals)
            residuals.append(diff)
            norm = float(np.linalg.norm(diff))
            if norm > 1e-12:
                basis_rows.append(diff / norm)

    if not basis_rows or basis_dim <= 0:
        return {
            "electrical_basis": np.zeros((0, len(edge_index)), dtype=float),
            "electrical_edge_index": edge_index,
            "electrical_incidence": incidence,
            "electrical_lap_pinv": lap_pinv,
            "electrical_lookup": lookup,
            "electrical_residuals": np.zeros((0, len(edge_index)), dtype=float),
            "electrical_rank": 0,
            "electrical_n_edges": int(len(edge_index)),
            "electrical_n_residuals": 0,
            "electrical_mean_divergence_l1": float(np.mean(divergence_errors)) if divergence_errors else 0.0,
        }

    mat = np.vstack(basis_rows)
    _, singular_values, vt = np.linalg.svd(mat, full_matrices=False)
    rank = int(min(basis_dim, vt.shape[0]))
    return {
        "electrical_basis": np.asarray(vt[:rank], dtype=float),
        "electrical_edge_index": edge_index,
        "electrical_incidence": incidence,
        "electrical_lap_pinv": lap_pinv,
        "electrical_lookup": lookup,
        "electrical_residuals": np.vstack(residuals),
        "electrical_rank": rank,
        "electrical_n_edges": int(len(edge_index)),
        "electrical_n_residuals": int(len(residuals)),
        "electrical_singular_values": np.asarray(singular_values[:rank], dtype=float),
        "electrical_mean_divergence_l1": float(np.mean(divergence_errors)) if divergence_errors else 0.0,
    }


def dp_electrical_cycle_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    electrical_info: dict,
) -> dict:
    """DP release of graph-Laplacian cycle residual coefficients."""
    basis = np.asarray(electrical_info["electrical_basis"], dtype=float)
    edge_index = electrical_info["electrical_edge_index"]
    incidence = np.asarray(electrical_info["electrical_incidence"], dtype=float)
    lap_pinv = np.asarray(electrical_info["electrical_lap_pinv"], dtype=float)
    k = int(basis.shape[0])
    signal = np.zeros(k, dtype=float)
    if k > 0:
        for t in train:
            cells = compact_cell_sequence(np.asarray(t), mn, span, grid)
            residual = normalized_electrical_residual(cells, edge_index, incidence, lap_pinv)
            if float(np.sum(np.abs(residual))) <= 1e-12:
                continue
            coeff = basis @ residual
            l1 = float(np.sum(np.abs(coeff)))
            if l1 > 1.0:
                coeff = coeff / max(l1, 1e-12)
            signal += coeff
    if eps > 0:
        signal = signal + rng.laplace(0.0, 1.0 / eps, size=signal.shape)
    else:
        signal = np.zeros_like(signal)
    return {
        "electrical_basis": basis,
        "electrical_signal": signal,
        "electrical_lookup": electrical_info["electrical_lookup"],
        "electrical_residuals": np.asarray(electrical_info["electrical_residuals"], dtype=float),
        "electrical_eps": float(eps),
    }


def dp_electrical_cycle_energy_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    electrical_info: dict,
) -> dict:
    """DP release of unsigned graph-cycle mode energy.

    This targets reusable cycle/corridor modes regardless of traversal sign.
    Each private trajectory contributes ``abs(U_K r_tau)`` clipped to L1 mass
    one; nonnegative truncation after Laplace noise is post-processing.
    """
    basis = np.asarray(electrical_info["electrical_basis"], dtype=float)
    edge_index = electrical_info["electrical_edge_index"]
    incidence = np.asarray(electrical_info["electrical_incidence"], dtype=float)
    lap_pinv = np.asarray(electrical_info["electrical_lap_pinv"], dtype=float)
    k = int(basis.shape[0])
    signal = np.zeros(k, dtype=float)
    if k > 0:
        for t in train:
            cells = compact_cell_sequence(np.asarray(t), mn, span, grid)
            residual = normalized_electrical_residual(cells, edge_index, incidence, lap_pinv)
            if float(np.sum(np.abs(residual))) <= 1e-12:
                continue
            coeff = np.abs(basis @ residual)
            l1 = float(np.sum(np.abs(coeff)))
            if l1 > 1.0:
                coeff = coeff / max(l1, 1e-12)
            signal += coeff
    if eps > 0:
        signal = np.maximum(signal + rng.laplace(0.0, 1.0 / eps, size=signal.shape), 0.0)
    else:
        signal = np.zeros_like(signal)
    return {
        "electrical_basis": basis,
        "electrical_energy_signal": signal,
        "electrical_lookup": electrical_info["electrical_lookup"],
        "electrical_residuals": np.asarray(electrical_info["electrical_residuals"], dtype=float),
        "electrical_eps": float(eps),
    }


def shuffled_electrical_cycle_reward(reward: dict, rng: np.random.Generator) -> dict:
    key = "electrical_energy_signal" if "electrical_energy_signal" in reward else "electrical_signal"
    signal = np.asarray(reward[key], dtype=float).copy()
    if signal.size:
        signal = signal[rng.permutation(signal.size)]
        if key == "electrical_signal":
            signal = signal * rng.choice(np.asarray([-1.0, 1.0]), size=signal.shape)
    out = {
        "electrical_basis": np.asarray(reward["electrical_basis"], dtype=float),
        "electrical_lookup": reward["electrical_lookup"],
        "electrical_residuals": np.asarray(reward["electrical_residuals"], dtype=float),
        "electrical_eps": float(reward.get("electrical_eps", 0.0)),
    }
    out[key] = signal
    return out


def build_hodge_cycle_basis(
    candidate_bank: list[dict],
    node_cells: np.ndarray,
    grid: int,
    basis_dim: int,
) -> dict:
    """Public cycle-space basis from same-OD candidate path differences.

    A raw directed path-count vector has incidence ``B x = e_dst - e_src``.
    Therefore the raw difference between two candidates with the same endpoints
    lies in ``ker(B)``, the graph cycle/circulation space. We build a public
    low-rank dictionary from these candidate circulations and only use private
    data later through clipped coefficient queries.
    """
    del grid  # The cell ids already encode the grid; kept for a stable API.
    edge_index: dict[tuple[int, int], int] = {}
    candidate_cells: list[list[list[int]]] = []
    for item in candidate_bank:
        req_cells = []
        for _, nodes in item.get("candidates", []):
            cells = compact_node_cells(nodes, node_cells)
            req_cells.append(cells)
            for edge in compact_cell_edges(cells):
                if edge not in edge_index:
                    edge_index[edge] = len(edge_index)
        candidate_cells.append(req_cells)

    residuals = []
    lookup: dict[tuple[int, int], int] = {}
    raw_residuals = []
    for req_idx, req_cells in enumerate(candidate_cells):
        if len(req_cells) < 2:
            continue
        public_vec = path_edge_count_vector(req_cells[0], edge_index)
        for cand_idx, cells in enumerate(req_cells[1:], start=1):
            diff = path_edge_count_vector(cells, edge_index) - public_vec
            l1 = float(np.sum(np.abs(diff)))
            if l1 <= 1e-12:
                continue
            diff = diff / l1
            raw_residuals.append(diff)
            lookup[(int(req_idx), int(cand_idx))] = len(raw_residuals) - 1
            norm = float(np.linalg.norm(diff))
            if norm > 1e-12:
                residuals.append(diff / norm)

    if not residuals or basis_dim <= 0:
        return {
            "hodge_basis": np.zeros((0, len(edge_index)), dtype=float),
            "hodge_edge_index": edge_index,
            "hodge_lookup": lookup,
            "hodge_residuals": np.zeros((0, len(edge_index)), dtype=float),
            "hodge_rank": 0,
            "hodge_n_edges": int(len(edge_index)),
            "hodge_n_residuals": 0,
        }

    mat = np.vstack(residuals)
    _, singular_values, vt = np.linalg.svd(mat, full_matrices=False)
    rank = int(min(basis_dim, vt.shape[0]))
    return {
        "hodge_basis": np.asarray(vt[:rank], dtype=float),
        "hodge_edge_index": edge_index,
        "hodge_lookup": lookup,
        "hodge_residuals": np.vstack(raw_residuals),
        "hodge_rank": rank,
        "hodge_n_edges": int(len(edge_index)),
        "hodge_n_residuals": int(len(raw_residuals)),
        "hodge_singular_values": np.asarray(singular_values[:rank], dtype=float),
    }


def dp_hodge_cycle_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    hodge_info: dict,
) -> dict:
    """DP release of graph-cycle coefficients.

    Each trajectory is converted to a normalized directed cell-edge flow, then
    projected onto a public cycle basis and L1-clipped. The summed coefficient
    vector has L1 sensitivity at most one, so adding iid Laplace(1/eps) noise
    gives epsilon-DP; all path scoring is post-processing.
    """
    basis = np.asarray(hodge_info["hodge_basis"], dtype=float)
    edge_index = hodge_info["hodge_edge_index"]
    k = int(basis.shape[0])
    signal = np.zeros(k, dtype=float)
    if k == 0:
        return {
            "hodge_basis": basis,
            "hodge_signal": signal,
            "hodge_edge_index": edge_index,
            "hodge_lookup": hodge_info["hodge_lookup"],
            "hodge_residuals": np.asarray(hodge_info["hodge_residuals"], dtype=float),
            "hodge_eps": float(eps),
        }
    for t in train:
        cells = compact_cell_sequence(np.asarray(t), mn, span, grid)
        vec = path_edge_count_vector(cells, edge_index)
        mass = float(np.sum(np.abs(vec)))
        if mass <= 1e-12:
            continue
        coeff = basis @ (vec / mass)
        l1 = float(np.sum(np.abs(coeff)))
        if l1 > 1.0:
            coeff = coeff / max(l1, 1e-12)
        signal += coeff
    if eps > 0:
        signal = signal + rng.laplace(0.0, 1.0 / eps, size=signal.shape)
    else:
        signal = np.zeros_like(signal)
    return {
        "hodge_basis": basis,
        "hodge_signal": signal,
        "hodge_edge_index": edge_index,
        "hodge_lookup": hodge_info["hodge_lookup"],
        "hodge_residuals": np.asarray(hodge_info["hodge_residuals"], dtype=float),
        "hodge_eps": float(eps),
    }


def shuffled_hodge_cycle_reward(reward: dict, rng: np.random.Generator) -> dict:
    signal = np.asarray(reward["hodge_signal"], dtype=float).copy()
    if signal.size:
        signal = signal[rng.permutation(signal.size)]
        signal = signal * rng.choice(np.asarray([-1.0, 1.0]), size=signal.shape)
    return {
        "hodge_basis": np.asarray(reward["hodge_basis"], dtype=float),
        "hodge_signal": signal,
        "hodge_edge_index": reward["hodge_edge_index"],
        "hodge_lookup": reward["hodge_lookup"],
        "hodge_residuals": np.asarray(reward["hodge_residuals"], dtype=float),
        "hodge_eps": float(reward.get("hodge_eps", 0.0)),
    }


def build_corridor_residual_basis(
    candidate_bank: list[dict],
    node_cells: np.ndarray,
    grid: int,
    basis_dim: int,
) -> dict[str, np.ndarray | int]:
    """Public low-rank basis for candidate residual route-choice directions.

    Rows are right singular vectors of candidate-minus-public path occupation
    differences. Because the basis is computed only from public candidates, it
    does not consume privacy budget.
    """
    rows = []
    for req_idx, item in enumerate(candidate_bank):
        cands = item.get("candidates", [])
        if len(cands) < 2:
            continue
        public_occ = path_transition_occupation(cands[0][1], node_cells, grid).ravel()
        for _, nodes in cands[1:]:
            diff = path_transition_occupation(nodes, node_cells, grid).ravel() - public_occ
            norm = float(np.linalg.norm(diff))
            if norm > 1e-12:
                rows.append(diff / norm)
    m = grid * grid * grid * grid
    if not rows or basis_dim <= 0:
        return {"basis": np.zeros((0, m), dtype=float), "rank": 0, "n_residuals": 0}
    mat = np.vstack(rows)
    _, singular_values, vt = np.linalg.svd(mat, full_matrices=False)
    rank = int(min(basis_dim, vt.shape[0]))
    basis = np.asarray(vt[:rank], dtype=float)
    return {
        "basis": basis,
        "rank": rank,
        "n_residuals": int(len(rows)),
        "singular_values": np.asarray(singular_values[:rank], dtype=float),
    }


def dp_corridor_residual_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    basis: np.ndarray,
) -> dict[str, np.ndarray]:
    """DP release of low-dimensional route-choice residual coefficients.

    Each trajectory is projected onto a public low-rank residual basis and then
    L1-clipped to one, giving total L1 sensitivity at most one.
    """
    basis = np.asarray(basis, dtype=float)
    k = int(basis.shape[0])
    signal = np.zeros(k, dtype=float)
    if k == 0:
        return {"corridor_basis": basis, "corridor_signal": signal}
    for t in train:
        occ = transition_occupation_from_cells(compact_cell_sequence(np.asarray(t), mn, span, grid), grid).ravel()
        coeff = basis @ occ
        l1 = float(np.sum(np.abs(coeff)))
        if l1 > 1.0:
            coeff = coeff / max(l1, 1e-12)
        signal += coeff
    if eps > 0:
        signal = signal + rng.laplace(0.0, 1.0 / eps, size=signal.shape)
    else:
        signal = np.zeros_like(signal)
    return {"corridor_basis": basis, "corridor_signal": signal}


def shuffled_corridor_residual_reward(reward: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    basis = np.asarray(reward["corridor_basis"], dtype=float)
    signal = np.asarray(reward["corridor_signal"], dtype=float).copy()
    if signal.size:
        signal = signal[rng.permutation(signal.size)]
        signal = signal * rng.choice(np.asarray([-1.0, 1.0]), size=signal.shape)
    return {"corridor_basis": basis, "corridor_signal": signal}


def build_local_corridor_residual_basis(
    candidate_bank: list[dict],
    node_cells: np.ndarray,
    grid: int,
    basis_dim: int,
) -> dict:
    """Public OD-bin-local residual bases for route-choice deviations."""
    m = grid * grid * grid * grid
    rows_by_od: dict[int, list[np.ndarray]] = {}
    raw_residuals = []
    residual_od = []
    lookup: dict[tuple[int, int], int] = {}
    for req_idx, item in enumerate(candidate_bank):
        cands = item.get("candidates", [])
        if len(cands) < 2:
            continue
        od = int(item.get("od_bin", 0))
        public_occ = path_transition_occupation(cands[0][1], node_cells, grid).ravel()
        for cand_idx, (_, nodes) in enumerate(cands[1:], start=1):
            diff = path_transition_occupation(nodes, node_cells, grid).ravel() - public_occ
            norm = float(np.linalg.norm(diff))
            if norm <= 1e-12:
                continue
            lookup[(int(req_idx), int(cand_idx))] = len(raw_residuals)
            raw_residuals.append(diff)
            residual_od.append(od)
            rows_by_od.setdefault(od, []).append(diff / norm)

    basis_by_od: dict[int, np.ndarray] = {}
    singular_by_od: dict[int, np.ndarray] = {}
    rank_by_od: dict[int, int] = {}
    for od, rows in rows_by_od.items():
        if not rows or basis_dim <= 0:
            basis_by_od[int(od)] = np.zeros((0, m), dtype=float)
            singular_by_od[int(od)] = np.zeros(0, dtype=float)
            rank_by_od[int(od)] = 0
            continue
        mat = np.vstack(rows)
        _, singular_values, vt = np.linalg.svd(mat, full_matrices=False)
        rank = int(min(basis_dim, vt.shape[0]))
        basis_by_od[int(od)] = np.asarray(vt[:rank], dtype=float)
        singular_by_od[int(od)] = np.asarray(singular_values[:rank], dtype=float)
        rank_by_od[int(od)] = rank

    residual_matrix = np.vstack(raw_residuals) if raw_residuals else np.zeros((0, m), dtype=float)
    return {
        "local_basis_by_od": basis_by_od,
        "local_singular_by_od": singular_by_od,
        "local_rank_by_od": rank_by_od,
        "local_lookup": lookup,
        "local_residuals": residual_matrix,
        "local_residual_od": np.asarray(residual_od, dtype=int),
        "local_n_residuals": int(len(raw_residuals)),
        "local_n_od_groups": int(len(basis_by_od)),
    }


def dp_local_corridor_residual_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    local_info: dict,
) -> dict:
    """DP release of OD-bin-local residual coefficients.

    Each trajectory contributes to exactly one OD-bin basis, then the
    coefficient vector is L1-clipped to one. Releasing all OD-bin coefficient
    sums with Laplace(1/eps) noise is epsilon-DP because the concatenated query
    has per-record L1 sensitivity at most one.
    """
    basis_by_od = {int(k): np.asarray(v, dtype=float) for k, v in local_info["local_basis_by_od"].items()}
    signal_by_od = {od: np.zeros(int(basis.shape[0]), dtype=float) for od, basis in basis_by_od.items()}
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2:
            continue
        od = od_bin_for_points(arr[0], arr[-1], mn, span)
        basis = basis_by_od.get(int(od))
        if basis is None or basis.size == 0:
            continue
        occ = transition_occupation_from_cells(compact_cell_sequence(arr, mn, span, grid), grid).ravel()
        coeff = basis @ occ
        l1 = float(np.sum(np.abs(coeff)))
        if l1 > 1.0:
            coeff = coeff / max(l1, 1e-12)
        signal_by_od[int(od)] += coeff
    for od, signal in list(signal_by_od.items()):
        if eps > 0 and signal.size:
            signal_by_od[od] = signal + rng.laplace(0.0, 1.0 / eps, size=signal.shape)
        else:
            signal_by_od[od] = np.zeros_like(signal)
    return {
        "local_basis_by_od": basis_by_od,
        "local_signal_by_od": signal_by_od,
        "local_lookup": local_info["local_lookup"],
        "local_residuals": np.asarray(local_info["local_residuals"], dtype=float),
        "local_residual_od": np.asarray(local_info["local_residual_od"], dtype=int),
    }


def shuffled_local_corridor_residual_reward(reward: dict, rng: np.random.Generator) -> dict:
    signal_by_od = {}
    for od, signal in reward["local_signal_by_od"].items():
        arr = np.asarray(signal, dtype=float).copy()
        if arr.size:
            arr = arr[rng.permutation(arr.size)]
            arr = arr * rng.choice(np.asarray([-1.0, 1.0]), size=arr.shape)
        signal_by_od[int(od)] = arr
    return {
        "local_basis_by_od": {int(k): np.asarray(v, dtype=float) for k, v in reward["local_basis_by_od"].items()},
        "local_signal_by_od": signal_by_od,
        "local_lookup": reward["local_lookup"],
        "local_residuals": np.asarray(reward["local_residuals"], dtype=float),
        "local_residual_od": np.asarray(reward["local_residual_od"], dtype=int),
    }


def deterministic_kmeans(features: np.ndarray, k: int, *, iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic k-means for public candidate features."""
    features = np.asarray(features, dtype=float)
    n = int(features.shape[0])
    if n == 0 or k <= 0:
        return np.zeros(0, dtype=int), np.zeros((0, features.shape[1] if features.ndim == 2 else 0), dtype=float)
    k = int(min(k, n))
    centers = [features[0].copy()]
    while len(centers) < k:
        cur = np.vstack(centers)
        d2 = np.min(np.sum((features[:, None, :] - cur[None, :, :]) ** 2, axis=2), axis=1)
        centers.append(features[int(np.argmax(d2))].copy())
    centroids = np.vstack(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist, axis=1).astype(int)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centroids[j] = np.mean(features[mask], axis=0)
    return labels, centroids


def build_cluster_corridor_residual_basis(
    candidate_bank: list[dict],
    node_cells: np.ndarray,
    grid: int,
    basis_dim: int,
    n_clusters: int,
) -> dict:
    """Public corridor-cluster residual bases in directed transition space.

    Candidate-minus-public transition residuals are clustered using only the
    public candidate family. Each cluster then gets its own low-rank residual
    basis. This makes locality a property of the graph path occupation vectors,
    not a travel-semantic bucket.
    """
    m = grid * grid * grid * grid
    raw_residuals = []
    norm_residuals = []
    lookup: dict[tuple[int, int], int] = {}
    for req_idx, item in enumerate(candidate_bank):
        cands = item.get("candidates", [])
        if len(cands) < 2:
            continue
        public_occ = path_transition_occupation(cands[0][1], node_cells, grid).ravel()
        for cand_idx, (_, nodes) in enumerate(cands[1:], start=1):
            diff = path_transition_occupation(nodes, node_cells, grid).ravel() - public_occ
            norm = float(np.linalg.norm(diff))
            if norm <= 1e-12:
                continue
            lookup[(int(req_idx), int(cand_idx))] = len(raw_residuals)
            raw_residuals.append(diff)
            norm_residuals.append(diff / norm)

    if not raw_residuals or basis_dim <= 0 or n_clusters <= 0:
        return {
            "cluster_global_basis": np.zeros((0, m), dtype=float),
            "cluster_centroids": np.zeros((0, 0), dtype=float),
            "cluster_basis_by_id": {},
            "cluster_singular_by_id": {},
            "cluster_rank_by_id": {},
            "cluster_lookup": lookup,
            "cluster_residuals": np.zeros((0, m), dtype=float),
            "cluster_residual_cluster": np.zeros(0, dtype=int),
            "cluster_n_residuals": int(len(raw_residuals)),
            "cluster_n_clusters": 0,
            "cluster_sizes": {},
        }

    mat = np.vstack(norm_residuals)
    _, global_singular, global_vt = np.linalg.svd(mat, full_matrices=False)
    global_rank = int(min(max(basis_dim, 1), global_vt.shape[0]))
    global_basis = np.asarray(global_vt[:global_rank], dtype=float)
    features = mat @ global_basis.T
    feature_norm = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(feature_norm, 1e-12)
    labels, centroids = deterministic_kmeans(features, n_clusters)

    basis_by_id: dict[int, np.ndarray] = {}
    singular_by_id: dict[int, np.ndarray] = {}
    rank_by_id: dict[int, int] = {}
    cluster_sizes: dict[int, int] = {}
    for cluster_id in sorted(set(int(x) for x in labels)):
        rows = mat[labels == cluster_id]
        cluster_sizes[int(cluster_id)] = int(rows.shape[0])
        _, singular_values, vt = np.linalg.svd(rows, full_matrices=False)
        rank = int(min(basis_dim, vt.shape[0]))
        basis_by_id[int(cluster_id)] = np.asarray(vt[:rank], dtype=float)
        singular_by_id[int(cluster_id)] = np.asarray(singular_values[:rank], dtype=float)
        rank_by_id[int(cluster_id)] = rank

    return {
        "cluster_global_basis": global_basis,
        "cluster_global_singular_values": np.asarray(global_singular[:global_rank], dtype=float),
        "cluster_centroids": np.asarray(centroids, dtype=float),
        "cluster_basis_by_id": basis_by_id,
        "cluster_singular_by_id": singular_by_id,
        "cluster_rank_by_id": rank_by_id,
        "cluster_lookup": lookup,
        "cluster_residuals": np.vstack(raw_residuals),
        "cluster_residual_cluster": np.asarray(labels, dtype=int),
        "cluster_n_residuals": int(len(raw_residuals)),
        "cluster_n_clusters": int(len(basis_by_id)),
        "cluster_sizes": cluster_sizes,
    }


def assign_cluster_for_occupation(occ: np.ndarray, cluster_info: dict) -> int | None:
    global_basis = np.asarray(cluster_info["cluster_global_basis"], dtype=float)
    centroids = np.asarray(cluster_info["cluster_centroids"], dtype=float)
    if global_basis.size == 0 or centroids.size == 0:
        return None
    feature = np.asarray(occ, dtype=float).ravel() @ global_basis.T
    norm = float(np.linalg.norm(feature))
    if norm > 1e-12:
        feature = feature / norm
    dist = np.sum((centroids - feature[None, :]) ** 2, axis=1)
    return int(np.argmin(dist))


def dp_cluster_corridor_residual_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    cluster_info: dict,
) -> dict:
    """DP release of public-corridor-cluster residual coefficients.

    A private trajectory is deterministically assigned to one public residual
    cluster from its graph transition occupation, projected into that cluster's
    public basis, L1-clipped to one, and then all cluster sums receive
    Laplace(1/eps) noise. The concatenated query has per-record L1 sensitivity
    at most one, even though the assignment itself depends on the record.
    """
    basis_by_id = {int(k): np.asarray(v, dtype=float) for k, v in cluster_info["cluster_basis_by_id"].items()}
    signal_by_id = {cid: np.zeros(int(basis.shape[0]), dtype=float) for cid, basis in basis_by_id.items()}
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2:
            continue
        occ = transition_occupation_from_cells(compact_cell_sequence(arr, mn, span, grid), grid).ravel()
        cluster_id = assign_cluster_for_occupation(occ, cluster_info)
        if cluster_id is None:
            continue
        basis = basis_by_id.get(int(cluster_id))
        if basis is None or basis.size == 0:
            continue
        coeff = basis @ occ
        l1 = float(np.sum(np.abs(coeff)))
        if l1 > 1.0:
            coeff = coeff / max(l1, 1e-12)
        signal_by_id[int(cluster_id)] += coeff
    for cid, signal in list(signal_by_id.items()):
        if eps > 0 and signal.size:
            signal_by_id[cid] = signal + rng.laplace(0.0, 1.0 / eps, size=signal.shape)
        else:
            signal_by_id[cid] = np.zeros_like(signal)
    return {
        "cluster_basis_by_id": basis_by_id,
        "cluster_signal_by_id": signal_by_id,
        "cluster_lookup": cluster_info["cluster_lookup"],
        "cluster_residuals": np.asarray(cluster_info["cluster_residuals"], dtype=float),
        "cluster_residual_cluster": np.asarray(cluster_info["cluster_residual_cluster"], dtype=int),
    }


def shuffled_cluster_corridor_residual_reward(reward: dict, rng: np.random.Generator) -> dict:
    signal_by_id = {}
    for cid, signal in reward["cluster_signal_by_id"].items():
        arr = np.asarray(signal, dtype=float).copy()
        if arr.size:
            arr = arr[rng.permutation(arr.size)]
            arr = arr * rng.choice(np.asarray([-1.0, 1.0]), size=arr.shape)
        signal_by_id[int(cid)] = arr
    return {
        "cluster_basis_by_id": {int(k): np.asarray(v, dtype=float) for k, v in reward["cluster_basis_by_id"].items()},
        "cluster_signal_by_id": signal_by_id,
        "cluster_lookup": reward["cluster_lookup"],
        "cluster_residuals": np.asarray(reward["cluster_residuals"], dtype=float),
        "cluster_residual_cluster": np.asarray(reward["cluster_residual_cluster"], dtype=int),
    }


def dp_cluster_vote_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    cluster_info: dict,
    *,
    public_prior_weight: float = 5.0,
) -> dict:
    """DP log-lift over public residual clusters.

    Each trajectory contributes one L1-bounded vote to the public residual
    cluster whose normalized occupation direction is closest to it. The released
    histogram is low-dimensional, and the final signal is a public-prior
    stabilized log-likelihood lift.
    """
    n_clusters = int(cluster_info.get("cluster_n_clusters", 0))
    counts = np.zeros(n_clusters, dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2 or n_clusters <= 0:
            continue
        occ = transition_occupation_from_cells(compact_cell_sequence(arr, mn, span, grid), grid).ravel()
        cluster_id = assign_cluster_for_occupation(occ, cluster_info)
        if cluster_id is None or cluster_id < 0 or cluster_id >= n_clusters:
            continue
        counts[int(cluster_id)] += 1.0
    noisy = counts + (rng.laplace(0.0, 1.0 / eps, size=counts.shape) if eps > 0 and counts.size else 0.0)

    public_counts = np.zeros(n_clusters, dtype=float)
    for cid, size in cluster_info.get("cluster_sizes", {}).items():
        cid = int(cid)
        if 0 <= cid < n_clusters:
            public_counts[cid] = float(size)
    eta = 1e-6
    public_prob = (public_counts + eta) / max(float(np.sum(public_counts) + eta * max(n_clusters, 1)), eta)
    private_mass = np.maximum(noisy, 0.0)
    private_prob = (private_mass + float(public_prior_weight) * public_prob + eta) / max(
        float(np.sum(private_mass) + float(public_prior_weight) + eta * max(n_clusters, 1)),
        eta,
    )
    signal = np.log(private_prob) - np.log(public_prob)
    signal = np.clip(signal, -CLUSTER_CORRIDOR_SCORE_CLIP, CLUSTER_CORRIDOR_SCORE_CLIP)
    return {
        "cluster_vote_signal": signal,
        "cluster_vote_noisy_counts": noisy,
        "cluster_vote_public_counts": public_counts,
        "cluster_lookup": cluster_info["cluster_lookup"],
        "cluster_residual_cluster": np.asarray(cluster_info["cluster_residual_cluster"], dtype=int),
    }


def shuffled_cluster_vote_reward(reward: dict, rng: np.random.Generator) -> dict:
    signal = np.asarray(reward["cluster_vote_signal"], dtype=float).copy()
    if signal.size:
        signal = signal[rng.permutation(signal.size)]
    return {
        "cluster_vote_signal": signal,
        "cluster_vote_noisy_counts": np.asarray(reward.get("cluster_vote_noisy_counts", []), dtype=float),
        "cluster_vote_public_counts": np.asarray(reward.get("cluster_vote_public_counts", []), dtype=float),
        "cluster_lookup": reward["cluster_lookup"],
        "cluster_residual_cluster": np.asarray(reward["cluster_residual_cluster"], dtype=int),
    }


def dp_od_cluster_vote_reward(
    train: list[np.ndarray],
    candidate_bank: list[dict],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    cluster_info: dict,
    *,
    public_prior_weight: float = 5.0,
) -> dict:
    """OD-conditioned DP residual-cluster vote histogram."""
    n_od = len(OD_BINS) - 1
    n_clusters = int(cluster_info.get("cluster_n_clusters", 0))
    counts = np.zeros((n_od, n_clusters), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2 or n_clusters <= 0:
            continue
        od = od_bin_for_points(arr[0], arr[-1], mn, span)
        occ = transition_occupation_from_cells(compact_cell_sequence(arr, mn, span, grid), grid).ravel()
        cluster_id = assign_cluster_for_occupation(occ, cluster_info)
        if cluster_id is None or cluster_id < 0 or cluster_id >= n_clusters:
            continue
        counts[int(od), int(cluster_id)] += 1.0
    noisy = counts + (rng.laplace(0.0, 1.0 / eps, size=counts.shape) if eps > 0 and counts.size else 0.0)

    request_od = np.asarray([int(item.get("od_bin", 0)) for item in candidate_bank], dtype=int)
    public_counts = np.zeros((n_od, n_clusters), dtype=float)
    residual_cluster = np.asarray(cluster_info["cluster_residual_cluster"], dtype=int)
    for (req_idx, cand_idx), residual_idx in cluster_info.get("cluster_lookup", {}).items():
        req_idx = int(req_idx)
        if req_idx < 0 or req_idx >= request_od.size:
            continue
        if residual_idx < 0 or residual_idx >= residual_cluster.size:
            continue
        od = int(np.clip(request_od[req_idx], 0, n_od - 1))
        cid = int(residual_cluster[int(residual_idx)])
        if 0 <= cid < n_clusters:
            public_counts[od, cid] += 1.0

    eta = 1e-6
    signal = np.zeros_like(noisy)
    for od in range(n_od):
        public_prob = (public_counts[od] + eta) / max(float(np.sum(public_counts[od]) + eta * max(n_clusters, 1)), eta)
        private_mass = np.maximum(noisy[od], 0.0)
        private_prob = (private_mass + float(public_prior_weight) * public_prob + eta) / max(
            float(np.sum(private_mass) + float(public_prior_weight) + eta * max(n_clusters, 1)),
            eta,
        )
        signal[od] = np.clip(np.log(private_prob) - np.log(public_prob), -CLUSTER_CORRIDOR_SCORE_CLIP, CLUSTER_CORRIDOR_SCORE_CLIP)

    return {
        "od_cluster_vote_signal": signal,
        "od_cluster_vote_noisy_counts": noisy,
        "od_cluster_vote_public_counts": public_counts,
        "od_cluster_vote_request_od": request_od,
        "cluster_lookup": cluster_info["cluster_lookup"],
        "cluster_residual_cluster": residual_cluster,
    }


def shuffled_od_cluster_vote_reward(reward: dict, rng: np.random.Generator) -> dict:
    signal = np.asarray(reward["od_cluster_vote_signal"], dtype=float).copy()
    if signal.size:
        flat = signal.ravel()
        flat = flat[rng.permutation(flat.size)]
        signal = flat.reshape(signal.shape)
    return {
        "od_cluster_vote_signal": signal,
        "od_cluster_vote_noisy_counts": np.asarray(reward.get("od_cluster_vote_noisy_counts", []), dtype=float),
        "od_cluster_vote_public_counts": np.asarray(reward.get("od_cluster_vote_public_counts", []), dtype=float),
        "od_cluster_vote_request_od": np.asarray(reward["od_cluster_vote_request_od"], dtype=int),
        "cluster_lookup": reward["cluster_lookup"],
        "cluster_residual_cluster": np.asarray(reward["cluster_residual_cluster"], dtype=int),
    }


def dp_odcell_cluster_vote_reward(
    train: list[np.ndarray],
    candidate_bank: list[dict],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    cluster_info: dict,
    *,
    od_pair_grid: int = OD_PAIR_GRID,
    public_prior_weight: float = 5.0,
) -> dict:
    """OD-cell-conditioned DP residual-cluster vote histogram."""
    n_odcell = int(od_pair_grid) ** 4
    n_clusters = int(cluster_info.get("cluster_n_clusters", 0))
    counts = np.zeros((n_odcell, n_clusters), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2 or n_clusters <= 0:
            continue
        odcell = od_pair_cell_bin_for_points(arr[0], arr[-1], mn, span, od_pair_grid)
        occ = transition_occupation_from_cells(compact_cell_sequence(arr, mn, span, grid), grid).ravel()
        cluster_id = assign_cluster_for_occupation(occ, cluster_info)
        if cluster_id is None or cluster_id < 0 or cluster_id >= n_clusters:
            continue
        counts[int(odcell), int(cluster_id)] += 1.0
    noisy = counts + (rng.laplace(0.0, 1.0 / eps, size=counts.shape) if eps > 0 and counts.size else 0.0)

    request_odcell = np.asarray(
        [int(np.clip(item.get("od_cell_bin", 0), 0, max(n_odcell - 1, 0))) for item in candidate_bank],
        dtype=int,
    )
    public_counts = np.zeros((n_odcell, n_clusters), dtype=float)
    residual_cluster = np.asarray(cluster_info["cluster_residual_cluster"], dtype=int)
    for (req_idx, cand_idx), residual_idx in cluster_info.get("cluster_lookup", {}).items():
        req_idx = int(req_idx)
        if req_idx < 0 or req_idx >= request_odcell.size:
            continue
        if residual_idx < 0 or residual_idx >= residual_cluster.size:
            continue
        odcell = int(request_odcell[req_idx])
        cid = int(residual_cluster[int(residual_idx)])
        if 0 <= cid < n_clusters:
            public_counts[odcell, cid] += 1.0

    eta = 1e-6
    signal = np.zeros_like(noisy)
    for odcell in range(n_odcell):
        public_prob = (public_counts[odcell] + eta) / max(float(np.sum(public_counts[odcell]) + eta * max(n_clusters, 1)), eta)
        private_mass = np.maximum(noisy[odcell], 0.0)
        private_prob = (private_mass + float(public_prior_weight) * public_prob + eta) / max(
            float(np.sum(private_mass) + float(public_prior_weight) + eta * max(n_clusters, 1)),
            eta,
        )
        signal[odcell] = np.clip(np.log(private_prob) - np.log(public_prob), -CLUSTER_CORRIDOR_SCORE_CLIP, CLUSTER_CORRIDOR_SCORE_CLIP)

    return {
        "odcell_cluster_vote_signal": signal,
        "odcell_cluster_vote_noisy_counts": noisy,
        "odcell_cluster_vote_public_counts": public_counts,
        "odcell_cluster_vote_request_odcell": request_odcell,
        "odcell_cluster_vote_grid": int(od_pair_grid),
        "cluster_lookup": cluster_info["cluster_lookup"],
        "cluster_residual_cluster": residual_cluster,
    }


def shuffled_odcell_cluster_vote_reward(reward: dict, rng: np.random.Generator) -> dict:
    signal = np.asarray(reward["odcell_cluster_vote_signal"], dtype=float).copy()
    if signal.size:
        flat = signal.ravel()
        flat = flat[rng.permutation(flat.size)]
        signal = flat.reshape(signal.shape)
    return {
        "odcell_cluster_vote_signal": signal,
        "odcell_cluster_vote_noisy_counts": np.asarray(reward.get("odcell_cluster_vote_noisy_counts", []), dtype=float),
        "odcell_cluster_vote_public_counts": np.asarray(reward.get("odcell_cluster_vote_public_counts", []), dtype=float),
        "odcell_cluster_vote_request_odcell": np.asarray(reward["odcell_cluster_vote_request_odcell"], dtype=int),
        "odcell_cluster_vote_grid": int(reward.get("odcell_cluster_vote_grid", OD_PAIR_GRID)),
        "cluster_lookup": reward["cluster_lookup"],
        "cluster_residual_cluster": np.asarray(reward["cluster_residual_cluster"], dtype=int),
    }


def _cluster_log_lift(noisy: np.ndarray, public_counts: np.ndarray, public_prior_weight: float) -> np.ndarray:
    eta = 1e-6
    noisy = np.asarray(noisy, dtype=float)
    public_counts = np.asarray(public_counts, dtype=float)
    signal = np.zeros_like(noisy)
    if noisy.ndim == 1:
        rows = [()]
    else:
        rows = list(np.ndindex(noisy.shape[:-1]))
    for idx in rows:
        priv = np.maximum(noisy[idx], 0.0) if idx else np.maximum(noisy, 0.0)
        pub = public_counts[idx] if idx else public_counts
        n_clusters = int(priv.shape[-1])
        public_prob = (pub + eta) / max(float(np.sum(pub) + eta * max(n_clusters, 1)), eta)
        private_prob = (priv + float(public_prior_weight) * public_prob + eta) / max(
            float(np.sum(priv) + float(public_prior_weight) + eta * max(n_clusters, 1)),
            eta,
        )
        lift = np.clip(np.log(private_prob) - np.log(public_prob), -CLUSTER_CORRIDOR_SCORE_CLIP, CLUSTER_CORRIDOR_SCORE_CLIP)
        if idx:
            signal[idx] = lift
        else:
            signal = lift
    return signal


def parse_anchor_via_node(label: str) -> int | None:
    """Extract the public via-node id from an anchor-via candidate label."""
    for part in str(label).split("_"):
        if len(part) > 1 and part[0] == "n" and part[1:].isdigit():
            return int(part[1:])
    return None


def anchor_choice_candidate_diagnostics(candidate_bank: list[dict]) -> dict:
    anchor_ids = []
    candidates_with_anchor = 0
    requests_with_anchor = 0
    for item in candidate_bank:
        request_has_anchor = False
        for label, _ in item.get("candidates", []):
            anchor_id = parse_anchor_via_node(label)
            if anchor_id is None:
                continue
            candidates_with_anchor += 1
            request_has_anchor = True
            anchor_ids.append(int(anchor_id))
        if request_has_anchor:
            requests_with_anchor += 1
    return {
        "unique_anchor_via_nodes": int(len(set(anchor_ids))),
        "candidates_with_anchor_via_node": int(candidates_with_anchor),
        "requests_with_anchor_via_candidate": int(requests_with_anchor),
    }


def dp_anchor_choice_reward(
    train: list[np.ndarray],
    candidate_bank: list[dict],
    coords: np.ndarray,
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    *,
    weights: tuple[float, float] = ANCHOR_CHOICE_WEIGHTS,
    public_prior_weight: float = 5.0,
) -> dict:
    """DP log-lift over public anchor-via channels.

    The public candidate generator defines a finite anchor dictionary. Each
    private trajectory contributes one unit of mass, split between global and
    OD-conditioned histograms, to the closest anchor visited along the route.
    Since the split weights sum to one, the concatenated histogram has L1
    sensitivity at most one and iid Laplace(1/eps) noise gives pure eps-DP.
    """
    wg, wo = (float(x) for x in weights)
    total_w = max(wg + wo, 1e-12)
    wg, wo = wg / total_w, wo / total_w
    n_od = len(OD_BINS) - 1

    raw_anchor_ids = []
    candidate_lookup_raw: dict[tuple[int, int], int] = {}
    for req_idx, item in enumerate(candidate_bank):
        for cand_idx, (label, _) in enumerate(item.get("candidates", [])):
            anchor_id = parse_anchor_via_node(label)
            if anchor_id is None or anchor_id < 0 or anchor_id >= len(coords):
                continue
            candidate_lookup_raw[(int(req_idx), int(cand_idx))] = int(anchor_id)
            raw_anchor_ids.append(int(anchor_id))

    anchor_ids = sorted(set(raw_anchor_ids))
    anchor_to_idx = {int(anchor_id): int(i) for i, anchor_id in enumerate(anchor_ids)}
    candidate_lookup = {
        key: anchor_to_idx[int(anchor_id)]
        for key, anchor_id in candidate_lookup_raw.items()
        if int(anchor_id) in anchor_to_idx
    }
    n_anchors = len(anchor_ids)
    request_od = np.asarray([int(np.clip(item.get("od_bin", 0), 0, n_od - 1)) for item in candidate_bank], dtype=int)

    global_counts = np.zeros(n_anchors, dtype=float)
    od_counts = np.zeros((n_od, n_anchors), dtype=float)
    public_global = np.zeros(n_anchors, dtype=float)
    public_od = np.zeros((n_od, n_anchors), dtype=float)

    for (req_idx, _cand_idx), anchor_idx in candidate_lookup.items():
        if 0 <= req_idx < request_od.size and 0 <= anchor_idx < n_anchors:
            od = int(request_od[int(req_idx)])
            public_global[int(anchor_idx)] += wg
            public_od[od, int(anchor_idx)] += wo

    if n_anchors > 0:
        anchor_coords = np.asarray(coords[np.asarray(anchor_ids, dtype=int)], dtype=float)
        anchor_tree = cKDTree(anchor_coords)
        for t in train:
            arr = np.asarray(t, dtype=float)
            if len(arr) < 2:
                continue
            dist, nearest = anchor_tree.query(arr)
            nearest = np.asarray(nearest, dtype=int)
            dist = np.asarray(dist, dtype=float)
            if nearest.size == 0:
                continue
            anchor_idx = int(nearest[int(np.argmin(dist))])
            if anchor_idx < 0 or anchor_idx >= n_anchors:
                continue
            od = od_bin_for_points(arr[0], arr[-1], mn, span)
            global_counts[anchor_idx] += wg
            od_counts[int(od), anchor_idx] += wo

    if eps > 0:
        global_noisy = global_counts + rng.laplace(0.0, 1.0 / eps, size=global_counts.shape)
        od_noisy = od_counts + rng.laplace(0.0, 1.0 / eps, size=od_counts.shape)
    else:
        global_noisy = np.zeros_like(global_counts)
        od_noisy = np.zeros_like(od_counts)

    return {
        "anchor_choice_signal_global": _cluster_log_lift(global_noisy, public_global, public_prior_weight),
        "anchor_choice_signal_od": _cluster_log_lift(od_noisy, public_od, public_prior_weight),
        "anchor_choice_noisy_global": global_noisy,
        "anchor_choice_noisy_od": od_noisy,
        "anchor_choice_public_global": public_global,
        "anchor_choice_public_od": public_od,
        "anchor_choice_request_od": request_od,
        "anchor_choice_candidate_lookup": candidate_lookup,
        "anchor_choice_anchor_ids": np.asarray(anchor_ids, dtype=int),
        "anchor_choice_weights": np.asarray([wg, wo], dtype=float),
    }


def shuffled_anchor_choice_reward(reward: dict, rng: np.random.Generator) -> dict:
    out = {
        "anchor_choice_noisy_global": np.asarray(reward.get("anchor_choice_noisy_global", []), dtype=float),
        "anchor_choice_noisy_od": np.asarray(reward.get("anchor_choice_noisy_od", []), dtype=float),
        "anchor_choice_public_global": np.asarray(reward.get("anchor_choice_public_global", []), dtype=float),
        "anchor_choice_public_od": np.asarray(reward.get("anchor_choice_public_od", []), dtype=float),
        "anchor_choice_request_od": np.asarray(reward["anchor_choice_request_od"], dtype=int),
        "anchor_choice_candidate_lookup": reward["anchor_choice_candidate_lookup"],
        "anchor_choice_anchor_ids": np.asarray(reward.get("anchor_choice_anchor_ids", []), dtype=int),
        "anchor_choice_weights": np.asarray(reward.get("anchor_choice_weights", ANCHOR_CHOICE_WEIGHTS), dtype=float),
    }
    for key in ("anchor_choice_signal_global", "anchor_choice_signal_od"):
        arr = np.asarray(reward[key], dtype=float).copy()
        if arr.size:
            flat = arr.ravel()
            flat = flat[rng.permutation(flat.size)]
            arr = flat.reshape(arr.shape)
        out[key] = arr
    return out


def od_relative_phase_lateral(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = np.asarray(p, dtype=float)
    delta = b - a
    direct = max(float(np.linalg.norm(delta)), 1e-12)
    axis = delta / direct
    offset = p - a
    phase = float(np.dot(offset, axis) / direct)
    lateral = float(axis[0] * offset[1] - axis[1] * offset[0]) / direct
    return phase, lateral


def anchor_shape_bucket_from_phase_lateral(phase: float, lateral: float) -> int:
    phase_bin = int(np.searchsorted(ANCHOR_SHAPE_PHASE_BINS, float(np.clip(phase, 0.0, 1.0)), side="right") - 1)
    phase_bin = int(np.clip(phase_bin, 0, len(ANCHOR_SHAPE_PHASE_BINS) - 2))
    lateral_bin = int(np.searchsorted(ANCHOR_SHAPE_LATERAL_BINS, float(lateral), side="right") - 1)
    lateral_bin = int(np.clip(lateral_bin, 0, len(ANCHOR_SHAPE_LATERAL_BINS) - 2))
    return int(phase_bin * (len(ANCHOR_SHAPE_LATERAL_BINS) - 1) + lateral_bin)


def anchor_shape_bucket_for_candidate(item: dict, cand_label: str, coords: np.ndarray) -> int | None:
    via = parse_anchor_via_node(cand_label)
    if via is None or via < 0 or via >= len(coords):
        return None
    src = int(item.get("src", -1))
    dst = int(item.get("dst", -1))
    if src < 0 or dst < 0 or src >= len(coords) or dst >= len(coords):
        return None
    phase, lateral = od_relative_phase_lateral(coords[src], coords[dst], coords[int(via)])
    return anchor_shape_bucket_from_phase_lateral(phase, lateral)


def anchor_shape_bucket_for_trajectory(traj: np.ndarray) -> int | None:
    arr = np.asarray(traj, dtype=float)
    if len(arr) < 2:
        return None
    a = arr[0]
    b = arr[-1]
    delta = b - a
    direct = float(np.linalg.norm(delta))
    if direct <= 1e-12:
        return None
    points = arr[1:-1] if len(arr) > 2 else arr
    phases = []
    laterals = []
    for p in points:
        phase, lateral = od_relative_phase_lateral(a, b, p)
        phases.append(float(np.clip(phase, 0.0, 1.0)))
        laterals.append(float(lateral))
    if not laterals:
        return None
    idx = int(np.argmax(np.abs(np.asarray(laterals, dtype=float))))
    return anchor_shape_bucket_from_phase_lateral(phases[idx], laterals[idx])


def dp_anchor_shape_reward(
    train: list[np.ndarray],
    candidate_bank: list[dict],
    coords: np.ndarray,
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    *,
    weights: tuple[float, float] = ANCHOR_SHAPE_WEIGHTS,
    public_prior_weight: float = 5.0,
) -> dict:
    """DP release of OD-relative via-shape preference.

    Candidate via nodes and private trajectories are both mapped to a compact
    graph-geometric bucket: progress phase along the OD chord and signed
    lateral deviation from that chord. The query is a weighted global/OD-bin
    categorical histogram with L1 sensitivity one.
    """
    wg, wo = (float(x) for x in weights)
    total_w = max(wg + wo, 1e-12)
    wg, wo = wg / total_w, wo / total_w
    n_od = len(OD_BINS) - 1
    n_shape = (len(ANCHOR_SHAPE_PHASE_BINS) - 1) * (len(ANCHOR_SHAPE_LATERAL_BINS) - 1)
    request_od = np.asarray([int(np.clip(item.get("od_bin", 0), 0, n_od - 1)) for item in candidate_bank], dtype=int)
    candidate_lookup: dict[tuple[int, int], int] = {}
    public_global = np.zeros(n_shape, dtype=float)
    public_od = np.zeros((n_od, n_shape), dtype=float)
    for req_idx, item in enumerate(candidate_bank):
        for cand_idx, (label, _) in enumerate(item.get("candidates", [])):
            bucket = anchor_shape_bucket_for_candidate(item, label, coords)
            if bucket is None:
                continue
            candidate_lookup[(int(req_idx), int(cand_idx))] = int(bucket)
            od = int(request_od[int(req_idx)])
            public_global[int(bucket)] += wg
            public_od[od, int(bucket)] += wo

    global_counts = np.zeros(n_shape, dtype=float)
    od_counts = np.zeros((n_od, n_shape), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2:
            continue
        bucket = anchor_shape_bucket_for_trajectory(arr)
        if bucket is None:
            continue
        od = od_bin_for_points(arr[0], arr[-1], mn, span)
        global_counts[int(bucket)] += wg
        od_counts[int(od), int(bucket)] += wo

    if eps > 0:
        global_noisy = global_counts + rng.laplace(0.0, 1.0 / eps, size=global_counts.shape)
        od_noisy = od_counts + rng.laplace(0.0, 1.0 / eps, size=od_counts.shape)
    else:
        global_noisy = np.zeros_like(global_counts)
        od_noisy = np.zeros_like(od_counts)

    return {
        "anchor_shape_signal_global": _cluster_log_lift(global_noisy, public_global, public_prior_weight),
        "anchor_shape_signal_od": _cluster_log_lift(od_noisy, public_od, public_prior_weight),
        "anchor_shape_noisy_global": global_noisy,
        "anchor_shape_noisy_od": od_noisy,
        "anchor_shape_public_global": public_global,
        "anchor_shape_public_od": public_od,
        "anchor_shape_request_od": request_od,
        "anchor_shape_candidate_lookup": candidate_lookup,
        "anchor_shape_weights": np.asarray([wg, wo], dtype=float),
        "anchor_shape_phase_bins": np.asarray(ANCHOR_SHAPE_PHASE_BINS, dtype=float),
        "anchor_shape_lateral_bins": np.asarray(ANCHOR_SHAPE_LATERAL_BINS, dtype=float),
    }


def shuffled_anchor_shape_reward(reward: dict, rng: np.random.Generator) -> dict:
    out = {
        "anchor_shape_noisy_global": np.asarray(reward.get("anchor_shape_noisy_global", []), dtype=float),
        "anchor_shape_noisy_od": np.asarray(reward.get("anchor_shape_noisy_od", []), dtype=float),
        "anchor_shape_public_global": np.asarray(reward.get("anchor_shape_public_global", []), dtype=float),
        "anchor_shape_public_od": np.asarray(reward.get("anchor_shape_public_od", []), dtype=float),
        "anchor_shape_request_od": np.asarray(reward["anchor_shape_request_od"], dtype=int),
        "anchor_shape_candidate_lookup": reward["anchor_shape_candidate_lookup"],
        "anchor_shape_weights": np.asarray(reward.get("anchor_shape_weights", ANCHOR_SHAPE_WEIGHTS), dtype=float),
        "anchor_shape_phase_bins": np.asarray(reward.get("anchor_shape_phase_bins", ANCHOR_SHAPE_PHASE_BINS), dtype=float),
        "anchor_shape_lateral_bins": np.asarray(reward.get("anchor_shape_lateral_bins", ANCHOR_SHAPE_LATERAL_BINS), dtype=float),
    }
    for key in ("anchor_shape_signal_global", "anchor_shape_signal_od"):
        arr = np.asarray(reward[key], dtype=float).copy()
        if arr.size:
            flat = arr.ravel()
            flat = flat[rng.permutation(flat.size)]
            arr = flat.reshape(arr.shape)
        out[key] = arr
    return out


def dp_hier_cluster_vote_reward(
    train: list[np.ndarray],
    candidate_bank: list[dict],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    cluster_info: dict,
    *,
    od_pair_grid: int = OD_PAIR_GRID,
    weights: tuple[float, float, float] = HIER_CLUSTER_WEIGHTS,
    public_prior_weight: float = 5.0,
) -> dict:
    """Hierarchical DP residual-cluster vote with global/OD/OD-cell backoff."""
    wg, wo, wc = (float(x) for x in weights)
    total_w = max(wg + wo + wc, 1e-12)
    wg, wo, wc = wg / total_w, wo / total_w, wc / total_w
    n_clusters = int(cluster_info.get("cluster_n_clusters", 0))
    n_od = len(OD_BINS) - 1
    n_odcell = int(od_pair_grid) ** 4
    global_counts = np.zeros(n_clusters, dtype=float)
    od_counts = np.zeros((n_od, n_clusters), dtype=float)
    odcell_counts = np.zeros((n_odcell, n_clusters), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2 or n_clusters <= 0:
            continue
        od = od_bin_for_points(arr[0], arr[-1], mn, span)
        odcell = od_pair_cell_bin_for_points(arr[0], arr[-1], mn, span, od_pair_grid)
        occ = transition_occupation_from_cells(compact_cell_sequence(arr, mn, span, grid), grid).ravel()
        cluster_id = assign_cluster_for_occupation(occ, cluster_info)
        if cluster_id is None or cluster_id < 0 or cluster_id >= n_clusters:
            continue
        cid = int(cluster_id)
        global_counts[cid] += wg
        od_counts[int(od), cid] += wo
        odcell_counts[int(odcell), cid] += wc
    if eps > 0:
        global_noisy = global_counts + rng.laplace(0.0, 1.0 / eps, size=global_counts.shape)
        od_noisy = od_counts + rng.laplace(0.0, 1.0 / eps, size=od_counts.shape)
        odcell_noisy = odcell_counts + rng.laplace(0.0, 1.0 / eps, size=odcell_counts.shape)
    else:
        global_noisy = np.zeros_like(global_counts)
        od_noisy = np.zeros_like(od_counts)
        odcell_noisy = np.zeros_like(odcell_counts)

    request_od = np.asarray([int(item.get("od_bin", 0)) for item in candidate_bank], dtype=int)
    request_odcell = np.asarray(
        [int(np.clip(item.get("od_cell_bin", 0), 0, max(n_odcell - 1, 0))) for item in candidate_bank],
        dtype=int,
    )
    public_global = np.zeros(n_clusters, dtype=float)
    public_od = np.zeros((n_od, n_clusters), dtype=float)
    public_odcell = np.zeros((n_odcell, n_clusters), dtype=float)
    residual_cluster = np.asarray(cluster_info["cluster_residual_cluster"], dtype=int)
    for (req_idx, cand_idx), residual_idx in cluster_info.get("cluster_lookup", {}).items():
        req_idx = int(req_idx)
        if req_idx < 0 or req_idx >= request_od.size or residual_idx < 0 or residual_idx >= residual_cluster.size:
            continue
        cid = int(residual_cluster[int(residual_idx)])
        if cid < 0 or cid >= n_clusters:
            continue
        od = int(np.clip(request_od[req_idx], 0, n_od - 1))
        odcell = int(np.clip(request_odcell[req_idx], 0, n_odcell - 1))
        public_global[cid] += wg
        public_od[od, cid] += wo
        public_odcell[odcell, cid] += wc

    return {
        "hier_cluster_vote_signal_global": _cluster_log_lift(global_noisy, public_global, public_prior_weight),
        "hier_cluster_vote_signal_od": _cluster_log_lift(od_noisy, public_od, public_prior_weight),
        "hier_cluster_vote_signal_odcell": _cluster_log_lift(odcell_noisy, public_odcell, public_prior_weight),
        "hier_cluster_vote_request_od": request_od,
        "hier_cluster_vote_request_odcell": request_odcell,
        "hier_cluster_vote_weights": np.asarray([wg, wo, wc], dtype=float),
        "hier_cluster_vote_grid": int(od_pair_grid),
        "cluster_lookup": cluster_info["cluster_lookup"],
        "cluster_residual_cluster": residual_cluster,
    }


def shuffled_hier_cluster_vote_reward(reward: dict, rng: np.random.Generator) -> dict:
    out = {
        "hier_cluster_vote_request_od": np.asarray(reward["hier_cluster_vote_request_od"], dtype=int),
        "hier_cluster_vote_request_odcell": np.asarray(reward["hier_cluster_vote_request_odcell"], dtype=int),
        "hier_cluster_vote_weights": np.asarray(reward["hier_cluster_vote_weights"], dtype=float),
        "hier_cluster_vote_grid": int(reward.get("hier_cluster_vote_grid", OD_PAIR_GRID)),
        "cluster_lookup": reward["cluster_lookup"],
        "cluster_residual_cluster": np.asarray(reward["cluster_residual_cluster"], dtype=int),
    }
    for key in ("hier_cluster_vote_signal_global", "hier_cluster_vote_signal_od", "hier_cluster_vote_signal_odcell"):
        arr = np.asarray(reward[key], dtype=float).copy()
        if arr.size:
            flat = arr.ravel()
            flat = flat[rng.permutation(flat.size)]
            arr = flat.reshape(arr.shape)
        out[key] = arr
    return out


def dp_signature_vote_reward(
    train: list[np.ndarray],
    candidate_bank: list[dict],
    coords: np.ndarray,
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    *,
    grid: int = 6,
    max_len: int = 5,
    public_prior_weight: float = 5.0,
) -> dict:
    """DP log-lift over compact route-signature buckets."""
    candidate_signature_index: dict[tuple[int, int], int] = {}
    key_to_idx: dict[str, int] = {}
    public_counts_list: list[float] = []
    for req_idx, item in enumerate(candidate_bank):
        for cand_idx, (_, nodes) in enumerate(item.get("candidates", [])):
            traj = resample(coords[np.asarray(nodes, dtype=int)], max(2, int(item["length"])))
            key = route_signature_key(traj, mn, span, grid=grid, max_len=max_len)
            if key is None:
                continue
            if key not in key_to_idx:
                key_to_idx[key] = len(public_counts_list)
                public_counts_list.append(0.0)
            sig_idx = int(key_to_idx[key])
            candidate_signature_index[(int(req_idx), int(cand_idx))] = sig_idx
            public_counts_list[sig_idx] += 1.0
    public_counts = np.asarray(public_counts_list, dtype=float)
    counts = np.zeros_like(public_counts)
    for t in train:
        key = route_signature_key(np.asarray(t, dtype=float), mn, span, grid=grid, max_len=max_len)
        if key is None:
            continue
        sig_idx = key_to_idx.get(key)
        if sig_idx is None:
            continue
        counts[int(sig_idx)] += 1.0
    noisy = counts + (rng.laplace(0.0, 1.0 / eps, size=counts.shape) if eps > 0 and counts.size else 0.0)
    signal = _cluster_log_lift(noisy, public_counts, public_prior_weight) if counts.size else np.zeros(0, dtype=float)
    return {
        "signature_vote_signal": np.clip(signal, -CLUSTER_CORRIDOR_SCORE_CLIP, CLUSTER_CORRIDOR_SCORE_CLIP),
        "signature_vote_noisy_counts": noisy,
        "signature_vote_public_counts": public_counts,
        "signature_vote_key_count": np.asarray([len(key_to_idx)], dtype=int),
        "signature_vote_candidate_lookup": candidate_signature_index,
        "signature_vote_grid": int(grid),
        "signature_vote_max_len": int(max_len),
    }


def shuffled_signature_vote_reward(reward: dict, rng: np.random.Generator) -> dict:
    signal = np.asarray(reward["signature_vote_signal"], dtype=float).copy()
    if signal.size:
        signal = signal[rng.permutation(signal.size)]
    return {
        "signature_vote_signal": signal,
        "signature_vote_noisy_counts": np.asarray(reward.get("signature_vote_noisy_counts", []), dtype=float),
        "signature_vote_public_counts": np.asarray(reward.get("signature_vote_public_counts", []), dtype=float),
        "signature_vote_key_count": np.asarray(reward.get("signature_vote_key_count", []), dtype=int),
        "signature_vote_candidate_lookup": reward["signature_vote_candidate_lookup"],
        "signature_vote_grid": int(reward.get("signature_vote_grid", 6)),
        "signature_vote_max_len": int(reward.get("signature_vote_max_len", 5)),
    }


def cell_rc(cell: int, grid: int) -> tuple[int, int]:
    return int(cell) // int(grid), int(cell) % int(grid)


def public_progress_layers(public_cells: list[int], grid: int, layers: int) -> np.ndarray:
    """Assign every grid cell to the nearest progress layer of a public route."""
    layers = max(int(layers), 1)
    n = grid * grid
    if not public_cells:
        return np.zeros(n, dtype=int)
    first_pos: dict[int, int] = {}
    for i, cell in enumerate(public_cells):
        first_pos.setdefault(int(cell), int(i))
    route_cells = np.asarray(list(first_pos.keys()), dtype=int)
    route_pos = np.asarray([first_pos[int(c)] for c in route_cells], dtype=float)
    denom = max(float(len(public_cells) - 1), 1.0)
    route_layers = np.minimum((route_pos / denom * layers).astype(int), layers - 1)
    route_rc = np.asarray([cell_rc(int(c), grid) for c in route_cells], dtype=float)
    out = np.zeros(n, dtype=int)
    for cell in range(n):
        rc = np.asarray(cell_rc(cell, grid), dtype=float)
        dist = np.sum((route_rc - rc[None, :]) ** 2, axis=1)
        out[cell] = int(route_layers[int(np.argmin(dist))])
    return out


def cut_corridor_feature_from_cells(cells: list[int], layer_of_cell: np.ndarray, grid: int, layers: int) -> np.ndarray:
    """Coarse source-sink cut feature: forward cut layer and entering cell."""
    n = grid * grid
    feat = np.zeros(int(layers) * n, dtype=float)
    for a, b in zip(cells[:-1], cells[1:]):
        if int(a) == int(b):
            continue
        la = int(layer_of_cell[int(a)])
        lb = int(layer_of_cell[int(b)])
        if lb <= la:
            continue
        # Crossing cut la and entering cell b identifies the corridor used.
        feat[int(la) * n + int(b)] += 1.0
    mass = float(np.sum(feat))
    if mass > 1e-12:
        feat = feat / mass
    return feat


def build_cut_corridor_features(
    candidate_bank: list[dict],
    node_cells: np.ndarray,
    road_coords: np.ndarray,
    mn: np.ndarray,
    span: np.ndarray,
    grid: int,
    layers: int,
) -> dict:
    """Public request-local source-sink cut/corridor feature maps."""
    features: dict[tuple[int, int], np.ndarray] = {}
    layer_by_request: list[np.ndarray] = []
    endpoints = []
    nonzero_counts = []
    dim = int(layers) * grid * grid
    for req_idx, item in enumerate(candidate_bank):
        cands = item.get("candidates", [])
        if not cands:
            layer_by_request.append(np.zeros(grid * grid, dtype=int))
            continue
        public_cells = compact_node_cells(cands[0][1], node_cells)
        layer_of_cell = public_progress_layers(public_cells, grid, layers)
        layer_by_request.append(layer_of_cell)
        src = np.asarray(road_coords[int(item["src"])], dtype=float)
        dst = np.asarray(road_coords[int(item["dst"])], dtype=float)
        endpoints.append(np.concatenate([(src - mn) / span, (dst - mn) / span]))
        for cand_idx, (_, nodes) in enumerate(cands):
            cells = compact_node_cells(nodes, node_cells)
            feat = cut_corridor_feature_from_cells(cells, layer_of_cell, grid, layers)
            features[(int(req_idx), int(cand_idx))] = feat
            nonzero_counts.append(int(np.count_nonzero(feat)))
    return {
        "cut_features": features,
        "cut_layer_by_request": layer_by_request,
        "cut_endpoints": np.vstack(endpoints) if endpoints else np.zeros((0, 4), dtype=float),
        "cut_dim": int(dim),
        "cut_layers": int(layers),
        "cut_mean_nonzero": float(np.mean(nonzero_counts)) if nonzero_counts else 0.0,
    }


def assign_request_for_trajectory(arr: np.ndarray, mn: np.ndarray, span: np.ndarray, cut_info: dict) -> int | None:
    endpoints = np.asarray(cut_info["cut_endpoints"], dtype=float)
    if endpoints.size == 0 or len(arr) < 2:
        return None
    sig = np.concatenate([(np.asarray(arr[0], dtype=float) - mn) / span, (np.asarray(arr[-1], dtype=float) - mn) / span])
    dist = np.sum((endpoints - sig[None, :]) ** 2, axis=1)
    return int(np.argmin(dist))


def dp_cut_corridor_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    cut_info: dict,
) -> dict:
    """DP release of request-local source-sink cut/corridor crossing mass."""
    layers = int(cut_info["cut_layers"])
    dim = int(cut_info["cut_dim"])
    signals = [np.zeros(dim, dtype=float) for _ in cut_info["cut_layer_by_request"]]
    for t in train:
        arr = np.asarray(t, dtype=float)
        req_idx = assign_request_for_trajectory(arr, mn, span, cut_info)
        if req_idx is None or req_idx < 0 or req_idx >= len(signals):
            continue
        cells = compact_cell_sequence(arr, mn, span, grid)
        feat = cut_corridor_feature_from_cells(cells, cut_info["cut_layer_by_request"][req_idx], grid, layers)
        l1 = float(np.sum(np.abs(feat)))
        if l1 > 1.0:
            feat = feat / max(l1, 1e-12)
        signals[req_idx] += feat
    for i, signal in enumerate(signals):
        if eps > 0 and signal.size:
            signals[i] = signal + rng.laplace(0.0, 1.0 / eps, size=signal.shape)
        else:
            signals[i] = np.zeros_like(signal)
    return {
        "cut_signal_by_request": signals,
        "cut_features": cut_info["cut_features"],
        "cut_layers": layers,
        "cut_dim": dim,
    }


def shuffled_cut_corridor_reward(reward: dict, rng: np.random.Generator) -> dict:
    signals = []
    for signal in reward["cut_signal_by_request"]:
        arr = np.asarray(signal, dtype=float).copy()
        if arr.size:
            arr = arr[rng.permutation(arr.size)]
        signals.append(arr)
    return {
        "cut_signal_by_request": signals,
        "cut_features": reward["cut_features"],
        "cut_layers": int(reward["cut_layers"]),
        "cut_dim": int(reward["cut_dim"]),
    }


def public_progress_geometry(public_cells: list[int], grid: int, layers: int) -> dict:
    """Public source-sink progress geometry for coarse cut-band features."""
    layer_of_cell = public_progress_layers(public_cells, grid, layers)
    layers = max(int(layers), 1)
    route_rc = np.asarray([cell_rc(int(c), grid) for c in public_cells], dtype=float)
    if route_rc.size == 0:
        route_rc = np.zeros((1, 2), dtype=float)
    denom = max(float(len(public_cells) - 1), 1.0)
    route_layers = np.minimum((np.arange(len(public_cells), dtype=float) / denom * layers).astype(int), layers - 1)
    anchors = np.zeros((layers, 2), dtype=float)
    tangents = np.zeros((layers, 2), dtype=float)
    for layer in range(layers):
        mask = route_layers == layer
        if np.any(mask):
            anchors[layer] = np.mean(route_rc[mask], axis=0)
        else:
            pos = min(max(int(round((layer + 0.5) / layers * max(len(public_cells) - 1, 0))), 0), len(route_rc) - 1)
            anchors[layer] = route_rc[pos]
    for layer in range(layers):
        prev_anchor = anchors[max(layer - 1, 0)]
        next_anchor = anchors[min(layer + 1, layers - 1)]
        tangent = next_anchor - prev_anchor
        if float(np.linalg.norm(tangent)) <= 1e-12 and len(route_rc) >= 2:
            tangent = route_rc[-1] - route_rc[0]
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-12:
            tangent = np.asarray([1.0, 0.0], dtype=float)
            norm = 1.0
        tangents[layer] = tangent / norm
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return {
        "layer_of_cell": layer_of_cell,
        "anchors": anchors,
        "tangents": tangents,
        "normals": normals,
    }


def direction_bucket_for_cells(a: int, b: int, grid: int, buckets: int) -> int:
    ar, ac = cell_rc(int(a), grid)
    br, bc = cell_rc(int(b), grid)
    dr = float(br - ar)
    dc = float(bc - ac)
    if buckets <= 1 or (abs(dr) <= 1e-12 and abs(dc) <= 1e-12):
        return 0
    angle = math.atan2(dr, dc)
    idx = int(math.floor(((angle + math.pi) / (2.0 * math.pi)) * buckets))
    return min(max(idx, 0), buckets - 1)


def lateral_bucket_for_cell(
    cell: int,
    layer: int,
    geometry: dict,
    grid: int,
    buckets: int,
) -> int:
    if buckets <= 1:
        return 0
    rc = np.asarray(cell_rc(int(cell), grid), dtype=float)
    layer = min(max(int(layer), 0), int(geometry["anchors"].shape[0]) - 1)
    signed = float(np.dot(rc - geometry["anchors"][layer], geometry["normals"][layer]))
    width = max(float(grid) / max(float(buckets + 1), 1.0), 1.0)
    center = buckets // 2
    return min(max(int(round(signed / width)) + center, 0), buckets - 1)


def cut_band_corridor_feature_from_cells(
    cells: list[int],
    geometry: dict,
    grid: int,
    layers: int,
    lateral_buckets: int,
    direction_buckets: int,
) -> np.ndarray:
    """Coarse source-sink cut feature: layer, lateral corridor, direction."""
    dim = int(layers) * int(lateral_buckets) * int(direction_buckets)
    feat = np.zeros(dim, dtype=float)
    layer_of_cell = np.asarray(geometry["layer_of_cell"], dtype=int)
    for a, b in zip(cells[:-1], cells[1:]):
        if int(a) == int(b):
            continue
        la = int(layer_of_cell[int(a)])
        lb = int(layer_of_cell[int(b)])
        if lb < la:
            continue
        crossing_layers = range(la, min(lb, int(layers) - 1) + 1) if lb == la else range(la, min(lb, int(layers)))
        for layer in crossing_layers:
            lateral = lateral_bucket_for_cell(int(b), int(layer), geometry, grid, lateral_buckets)
            direction = direction_bucket_for_cells(int(a), int(b), grid, direction_buckets)
            idx = (int(layer) * int(lateral_buckets) + int(lateral)) * int(direction_buckets) + int(direction)
            feat[idx] += 1.0
    mass = float(np.sum(feat))
    if mass > 1e-12:
        feat = feat / mass
    return feat


def build_cut_band_corridor_features(
    candidate_bank: list[dict],
    node_cells: np.ndarray,
    road_coords: np.ndarray,
    mn: np.ndarray,
    span: np.ndarray,
    grid: int,
    layers: int,
    lateral_buckets: int,
    direction_buckets: int,
) -> dict:
    """Public request-local cut-band corridor feature maps."""
    features: dict[tuple[int, int], np.ndarray] = {}
    geometry_by_request = []
    endpoints = []
    nonzero_counts = []
    dim = int(layers) * int(lateral_buckets) * int(direction_buckets)
    for req_idx, item in enumerate(candidate_bank):
        cands = item.get("candidates", [])
        if not cands:
            geometry_by_request.append(public_progress_geometry([], grid, layers))
            continue
        public_cells = compact_node_cells(cands[0][1], node_cells)
        geometry = public_progress_geometry(public_cells, grid, layers)
        geometry_by_request.append(geometry)
        src = np.asarray(road_coords[int(item["src"])], dtype=float)
        dst = np.asarray(road_coords[int(item["dst"])], dtype=float)
        endpoints.append(np.concatenate([(src - mn) / span, (dst - mn) / span]))
        for cand_idx, (_, nodes) in enumerate(cands):
            cells = compact_node_cells(nodes, node_cells)
            feat = cut_band_corridor_feature_from_cells(
                cells,
                geometry,
                grid,
                layers,
                lateral_buckets,
                direction_buckets,
            )
            features[(int(req_idx), int(cand_idx))] = feat
            nonzero_counts.append(int(np.count_nonzero(feat)))
    return {
        "cut_band_features": features,
        "cut_band_geometry_by_request": geometry_by_request,
        "cut_band_endpoints": np.vstack(endpoints) if endpoints else np.zeros((0, 4), dtype=float),
        "cut_band_dim": int(dim),
        "cut_band_layers": int(layers),
        "cut_band_lateral_buckets": int(lateral_buckets),
        "cut_band_direction_buckets": int(direction_buckets),
        "cut_band_mean_nonzero": float(np.mean(nonzero_counts)) if nonzero_counts else 0.0,
    }


def assign_request_for_trajectory_from_endpoints(arr: np.ndarray, mn: np.ndarray, span: np.ndarray, endpoints: np.ndarray) -> int | None:
    endpoints = np.asarray(endpoints, dtype=float)
    if endpoints.size == 0 or len(arr) < 2:
        return None
    sig = np.concatenate([(np.asarray(arr[0], dtype=float) - mn) / span, (np.asarray(arr[-1], dtype=float) - mn) / span])
    dist = np.sum((endpoints - sig[None, :]) ** 2, axis=1)
    return int(np.argmin(dist))


def dp_cut_band_corridor_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    cut_band_info: dict,
) -> dict:
    """DP release of coarse request-local cut-band crossing mass."""
    layers = int(cut_band_info["cut_band_layers"])
    lateral_buckets = int(cut_band_info["cut_band_lateral_buckets"])
    direction_buckets = int(cut_band_info["cut_band_direction_buckets"])
    dim = int(cut_band_info["cut_band_dim"])
    geometries = cut_band_info["cut_band_geometry_by_request"]
    signals = [np.zeros(dim, dtype=float) for _ in geometries]
    for t in train:
        arr = np.asarray(t, dtype=float)
        req_idx = assign_request_for_trajectory_from_endpoints(arr, mn, span, cut_band_info["cut_band_endpoints"])
        if req_idx is None or req_idx < 0 or req_idx >= len(signals):
            continue
        cells = compact_cell_sequence(arr, mn, span, grid)
        feat = cut_band_corridor_feature_from_cells(
            cells,
            geometries[int(req_idx)],
            grid,
            layers,
            lateral_buckets,
            direction_buckets,
        )
        l1 = float(np.sum(np.abs(feat)))
        if l1 > 1.0:
            feat = feat / max(l1, 1e-12)
        signals[int(req_idx)] += feat
    for i, signal in enumerate(signals):
        if eps > 0 and signal.size:
            signals[i] = signal + rng.laplace(0.0, 1.0 / eps, size=signal.shape)
        else:
            signals[i] = np.zeros_like(signal)
    return {
        "cut_band_signal_by_request": signals,
        "cut_band_features": cut_band_info["cut_band_features"],
        "cut_band_layers": layers,
        "cut_band_lateral_buckets": lateral_buckets,
        "cut_band_direction_buckets": direction_buckets,
        "cut_band_dim": dim,
    }


def shuffled_cut_band_corridor_reward(reward: dict, rng: np.random.Generator) -> dict:
    signals = []
    for signal in reward["cut_band_signal_by_request"]:
        arr = np.asarray(signal, dtype=float).copy()
        if arr.size:
            arr = arr[rng.permutation(arr.size)]
        signals.append(arr)
    return {
        "cut_band_signal_by_request": signals,
        "cut_band_features": reward["cut_band_features"],
        "cut_band_layers": int(reward["cut_band_layers"]),
        "cut_band_lateral_buckets": int(reward["cut_band_lateral_buckets"]),
        "cut_band_direction_buckets": int(reward["cut_band_direction_buckets"]),
        "cut_band_dim": int(reward["cut_band_dim"]),
    }


def od_axis_geometry(cells: list[int], grid: int) -> tuple[np.ndarray, np.ndarray, float]:
    if len(cells) >= 2:
        src = np.asarray(cell_rc(int(cells[0]), grid), dtype=float)
        dst = np.asarray(cell_rc(int(cells[-1]), grid), dtype=float)
    elif len(cells) == 1:
        src = np.asarray(cell_rc(int(cells[0]), grid), dtype=float)
        dst = src + np.asarray([1.0, 0.0], dtype=float)
    else:
        src = np.zeros(2, dtype=float)
        dst = np.asarray([1.0, 0.0], dtype=float)
    tangent = dst - src
    length = float(np.linalg.norm(tangent))
    if length <= 1e-12:
        tangent = np.asarray([1.0, 0.0], dtype=float)
        length = 1.0
    else:
        tangent = tangent / length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    return src, np.vstack([tangent, normal]), length


def od_cut_band_feature_from_cells(
    cells: list[int],
    grid: int,
    layers: int,
    lateral_buckets: int,
    direction_buckets: int,
) -> np.ndarray:
    """Global OD-conditioned source-sink cut-band feature."""
    dim = int(layers) * int(lateral_buckets) * int(direction_buckets)
    feat = np.zeros(dim, dtype=float)
    if len(cells) < 2:
        return feat
    src, axes, length = od_axis_geometry(cells, grid)
    tangent = axes[0]
    normal = axes[1]
    width = max(float(grid) / max(float(lateral_buckets + 1), 1.0), 1.0)
    center = int(lateral_buckets) // 2
    for a, b in zip(cells[:-1], cells[1:]):
        if int(a) == int(b):
            continue
        rc_a = np.asarray(cell_rc(int(a), grid), dtype=float)
        rc_b = np.asarray(cell_rc(int(b), grid), dtype=float)
        pa = float(np.dot(rc_a - src, tangent) / max(length, 1e-12))
        pb = float(np.dot(rc_b - src, tangent) / max(length, 1e-12))
        if pb + 1e-9 < pa:
            continue
        layer = min(max(int(math.floor(np.clip(pb, 0.0, 1.0 - 1e-9) * int(layers))), 0), int(layers) - 1)
        signed = float(np.dot(rc_b - src, normal))
        lateral = min(max(int(round(signed / width)) + center, 0), int(lateral_buckets) - 1)
        delta = rc_b - rc_a
        forward = float(np.dot(delta, tangent))
        side = float(np.dot(delta, normal))
        if direction_buckets == 4:
            if forward >= abs(side):
                direction = 0
            elif -forward >= abs(side):
                direction = 3
            elif side >= 0:
                direction = 1
            else:
                direction = 2
        else:
            angle = math.atan2(side, forward)
            direction = min(max(int(math.floor(((angle + math.pi) / (2.0 * math.pi)) * int(direction_buckets))), 0), int(direction_buckets) - 1)
        idx = (int(layer) * int(lateral_buckets) + int(lateral)) * int(direction_buckets) + int(direction)
        feat[idx] += 1.0
    mass = float(np.sum(feat))
    if mass > 1e-12:
        feat = feat / mass
    return feat


def build_od_cut_band_features(
    candidate_bank: list[dict],
    node_cells: np.ndarray,
    grid: int,
    layers: int,
    lateral_buckets: int,
    direction_buckets: int,
) -> dict:
    features: dict[tuple[int, int], np.ndarray] = {}
    request_od = []
    nonzero_counts = []
    dim = int(layers) * int(lateral_buckets) * int(direction_buckets)
    for req_idx, item in enumerate(candidate_bank):
        request_od.append(int(item.get("od_bin", 0)))
        for cand_idx, (_, nodes) in enumerate(item.get("candidates", [])):
            cells = compact_node_cells(nodes, node_cells)
            feat = od_cut_band_feature_from_cells(cells, grid, layers, lateral_buckets, direction_buckets)
            features[(int(req_idx), int(cand_idx))] = feat
            nonzero_counts.append(int(np.count_nonzero(feat)))
    return {
        "od_cut_band_features": features,
        "od_cut_band_request_od": np.asarray(request_od, dtype=int),
        "od_cut_band_dim": int(dim),
        "od_cut_band_layers": int(layers),
        "od_cut_band_lateral_buckets": int(lateral_buckets),
        "od_cut_band_direction_buckets": int(direction_buckets),
        "od_cut_band_mean_nonzero": float(np.mean(nonzero_counts)) if nonzero_counts else 0.0,
        "od_cut_band_n_od_bins": int(len(OD_BINS) - 1),
    }


def dp_od_cut_band_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    od_cut_info: dict,
) -> dict:
    """DP release of OD-bin-global cut-band crossing mass."""
    n_od = int(od_cut_info["od_cut_band_n_od_bins"])
    dim = int(od_cut_info["od_cut_band_dim"])
    layers = int(od_cut_info["od_cut_band_layers"])
    lateral_buckets = int(od_cut_info["od_cut_band_lateral_buckets"])
    direction_buckets = int(od_cut_info["od_cut_band_direction_buckets"])
    signal = np.zeros((n_od, dim), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2:
            continue
        od = od_bin_for_points(arr[0], arr[-1], mn, span)
        cells = compact_cell_sequence(arr, mn, span, grid)
        feat = od_cut_band_feature_from_cells(cells, grid, layers, lateral_buckets, direction_buckets)
        l1 = float(np.sum(np.abs(feat)))
        if l1 > 1.0:
            feat = feat / max(l1, 1e-12)
        signal[int(od)] += feat
    if eps > 0 and signal.size:
        signal = signal + rng.laplace(0.0, 1.0 / eps, size=signal.shape)
    else:
        signal = np.zeros_like(signal)
    return {
        "od_cut_band_signal": signal,
        "od_cut_band_features": od_cut_info["od_cut_band_features"],
        "od_cut_band_request_od": np.asarray(od_cut_info["od_cut_band_request_od"], dtype=int),
        "od_cut_band_layers": layers,
        "od_cut_band_lateral_buckets": lateral_buckets,
        "od_cut_band_direction_buckets": direction_buckets,
        "od_cut_band_dim": dim,
    }


def shuffled_od_cut_band_reward(reward: dict, rng: np.random.Generator) -> dict:
    signal = np.asarray(reward["od_cut_band_signal"], dtype=float).copy()
    for od in range(signal.shape[0]):
        if signal.shape[1] > 0:
            signal[od] = signal[od, rng.permutation(signal.shape[1])]
    return {
        "od_cut_band_signal": signal,
        "od_cut_band_features": reward["od_cut_band_features"],
        "od_cut_band_request_od": np.asarray(reward["od_cut_band_request_od"], dtype=int),
        "od_cut_band_layers": int(reward["od_cut_band_layers"]),
        "od_cut_band_lateral_buckets": int(reward["od_cut_band_lateral_buckets"]),
        "od_cut_band_direction_buckets": int(reward["od_cut_band_direction_buckets"]),
        "od_cut_band_dim": int(reward["od_cut_band_dim"]),
    }


def dp_global_cut_band_corridor_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    cut_band_info: dict,
) -> dict:
    """DP release of one public-path-relative global cut-band signal."""
    layers = int(cut_band_info["cut_band_layers"])
    lateral_buckets = int(cut_band_info["cut_band_lateral_buckets"])
    direction_buckets = int(cut_band_info["cut_band_direction_buckets"])
    dim = int(cut_band_info["cut_band_dim"])
    geometries = cut_band_info["cut_band_geometry_by_request"]
    signal = np.zeros(dim, dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        req_idx = assign_request_for_trajectory_from_endpoints(arr, mn, span, cut_band_info["cut_band_endpoints"])
        if req_idx is None or req_idx < 0 or req_idx >= len(geometries):
            continue
        cells = compact_cell_sequence(arr, mn, span, grid)
        feat = cut_band_corridor_feature_from_cells(
            cells,
            geometries[int(req_idx)],
            grid,
            layers,
            lateral_buckets,
            direction_buckets,
        )
        l1 = float(np.sum(np.abs(feat)))
        if l1 > 1.0:
            feat = feat / max(l1, 1e-12)
        signal += feat
    if eps > 0 and signal.size:
        signal = signal + rng.laplace(0.0, 1.0 / eps, size=signal.shape)
    else:
        signal = np.zeros_like(signal)
    return {
        "global_cut_band_signal": signal,
        "global_cut_band_features": cut_band_info["cut_band_features"],
        "global_cut_band_dim": dim,
        "global_cut_band_layers": layers,
        "global_cut_band_lateral_buckets": lateral_buckets,
        "global_cut_band_direction_buckets": direction_buckets,
    }


def shuffled_global_cut_band_corridor_reward(reward: dict, rng: np.random.Generator) -> dict:
    signal = np.asarray(reward["global_cut_band_signal"], dtype=float).copy()
    if signal.size:
        signal = signal[rng.permutation(signal.size)]
    return {
        "global_cut_band_signal": signal,
        "global_cut_band_features": reward["global_cut_band_features"],
        "global_cut_band_dim": int(reward["global_cut_band_dim"]),
        "global_cut_band_layers": int(reward["global_cut_band_layers"]),
        "global_cut_band_lateral_buckets": int(reward["global_cut_band_lateral_buckets"]),
        "global_cut_band_direction_buckets": int(reward["global_cut_band_direction_buckets"]),
    }


def balanced_pc1_partition(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deterministic public partition with near-equal block sizes.

    The projection direction is computed only from public candidate signatures.
    Equal-frequency chunks trade a little clustering purity for a lower bound on
    per-family sample size, which is exactly the DP signal-to-noise bottleneck.
    """
    features = np.asarray(features, dtype=float)
    n = int(features.shape[0])
    dim = int(features.shape[1]) if features.ndim == 2 else 0
    if n == 0 or k <= 0:
        return np.zeros(0, dtype=int), np.zeros((0, dim), dtype=float), {"projection": "empty"}
    k = int(min(k, n))
    if k == 1:
        labels = np.zeros(n, dtype=int)
        return labels, np.mean(features, axis=0, keepdims=True), {"projection": "single_family"}

    centered = features - np.mean(features, axis=0, keepdims=True)
    if np.linalg.norm(centered) <= 1e-12:
        scores = np.arange(n, dtype=float)
        projection = "index_tiebreak"
    else:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        scores = centered @ direction
        projection = "public_pc1"

    order = np.lexsort((np.arange(n), scores))
    labels = np.zeros(n, dtype=int)
    base = n // k
    extra = n % k
    start = 0
    for family in range(k):
        size = base + (1 if family < extra else 0)
        stop = start + size
        labels[order[start:stop]] = family
        start = stop

    centroids = np.zeros((k, dim), dtype=float)
    for family in range(k):
        mask = labels == family
        if np.any(mask):
            centroids[family] = np.mean(features[mask], axis=0)
    info = {
        "projection": projection,
        "score_min": float(np.min(scores)) if len(scores) else 0.0,
        "score_max": float(np.max(scores)) if len(scores) else 0.0,
    }
    if projection == "public_pc1":
        info["singular_value_1"] = float(singular_values[0]) if len(singular_values) else 0.0
    return labels, centroids, info


def build_cut_band_family_info(cut_band_info: dict, n_families: int, mode: str = "kmeans") -> dict:
    """Public corridor-family clustering for cut-band features."""
    endpoints = np.asarray(cut_band_info["cut_band_endpoints"], dtype=float)
    n_req = int(endpoints.shape[0])
    dim = int(cut_band_info["cut_band_dim"])
    if n_req == 0:
        return {
            "cut_band_family_by_request": np.zeros(0, dtype=int),
            "cut_band_family_centroids": np.zeros((0, 4 + dim), dtype=float),
            "cut_band_family_count": 0,
            "cut_band_family_sizes": {},
            "cut_band_family_mode": str(mode),
            "cut_band_family_balance": {},
        }
    public_feats = []
    for req_idx in range(n_req):
        public_feats.append(np.asarray(cut_band_info["cut_band_features"].get((int(req_idx), 0), np.zeros(dim)), dtype=float))
    public_feats_arr = np.vstack(public_feats)
    rows = np.hstack([2.0 * endpoints, public_feats_arr])
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    rows = rows / np.maximum(norms, 1e-12)
    mode = str(mode)
    if mode == "balanced":
        labels, centroids, balance_info = balanced_pc1_partition(rows, int(n_families))
    elif mode == "kmeans":
        labels, centroids = deterministic_kmeans(rows, int(n_families))
        balance_info = {"projection": "deterministic_kmeans"}
    else:
        raise ValueError(f"unknown cut-band family mode: {mode}")
    sizes = {int(k): int(np.sum(labels == int(k))) for k in sorted(set(int(x) for x in labels))}
    return {
        "cut_band_family_by_request": np.asarray(labels, dtype=int),
        "cut_band_family_centroids": np.asarray(centroids, dtype=float),
        "cut_band_family_count": int(len(sizes)),
        "cut_band_family_sizes": sizes,
        "cut_band_family_mode": mode,
        "cut_band_family_balance": balance_info,
    }


def dp_family_cut_band_corridor_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    cut_band_info: dict,
    family_info: dict,
) -> dict:
    """DP release of public corridor-family cut-band signals."""
    layers = int(cut_band_info["cut_band_layers"])
    lateral_buckets = int(cut_band_info["cut_band_lateral_buckets"])
    direction_buckets = int(cut_band_info["cut_band_direction_buckets"])
    dim = int(cut_band_info["cut_band_dim"])
    geometries = cut_band_info["cut_band_geometry_by_request"]
    family_by_request = np.asarray(family_info["cut_band_family_by_request"], dtype=int)
    n_families = int(family_info["cut_band_family_count"])
    signals = np.zeros((n_families, dim), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        req_idx = assign_request_for_trajectory_from_endpoints(arr, mn, span, cut_band_info["cut_band_endpoints"])
        if req_idx is None or req_idx < 0 or req_idx >= len(geometries) or req_idx >= family_by_request.size:
            continue
        family = int(family_by_request[int(req_idx)])
        if family < 0 or family >= n_families:
            continue
        cells = compact_cell_sequence(arr, mn, span, grid)
        feat = cut_band_corridor_feature_from_cells(
            cells,
            geometries[int(req_idx)],
            grid,
            layers,
            lateral_buckets,
            direction_buckets,
        )
        l1 = float(np.sum(np.abs(feat)))
        if l1 > 1.0:
            feat = feat / max(l1, 1e-12)
        signals[family] += feat
    if eps > 0 and signals.size:
        signals = signals + rng.laplace(0.0, 1.0 / eps, size=signals.shape)
    else:
        signals = np.zeros_like(signals)
    return {
        "family_cut_band_signal": signals,
        "family_cut_band_features": cut_band_info["cut_band_features"],
        "family_cut_band_by_request": family_by_request,
        "family_cut_band_dim": dim,
        "family_cut_band_layers": layers,
        "family_cut_band_lateral_buckets": lateral_buckets,
        "family_cut_band_direction_buckets": direction_buckets,
    }


def shuffled_family_cut_band_corridor_reward(reward: dict, rng: np.random.Generator) -> dict:
    signals = np.asarray(reward["family_cut_band_signal"], dtype=float).copy()
    for family in range(signals.shape[0]):
        if signals.shape[1] > 0:
            signals[family] = signals[family, rng.permutation(signals.shape[1])]
    return {
        "family_cut_band_signal": signals,
        "family_cut_band_features": reward["family_cut_band_features"],
        "family_cut_band_by_request": np.asarray(reward["family_cut_band_by_request"], dtype=int),
        "family_cut_band_dim": int(reward["family_cut_band_dim"]),
        "family_cut_band_layers": int(reward["family_cut_band_layers"]),
        "family_cut_band_lateral_buckets": int(reward["family_cut_band_lateral_buckets"]),
        "family_cut_band_direction_buckets": int(reward["family_cut_band_direction_buckets"]),
    }


def build_cycle_motif_dictionary(candidate_bank: list[dict], node_cells: np.ndarray, grid: int) -> dict:
    """Public OD-local cycle/corridor motif dictionary.

    For each alternative candidate, the difference between its normalized path
    occupation and the public candidate-0 occupation is a finite-candidate
    circulation residual for the same OD pair.
    """
    motifs = []
    request_indices = []
    candidate_indices = []
    lookup: dict[tuple[int, int], int] = {}
    for req_idx, item in enumerate(candidate_bank):
        cands = item.get("candidates", [])
        if len(cands) < 2:
            continue
        public_occ = path_transition_occupation(cands[0][1], node_cells, grid).ravel()
        for cand_idx, (_, nodes) in enumerate(cands[1:], start=1):
            diff = path_transition_occupation(nodes, node_cells, grid).ravel() - public_occ
            norm = float(np.linalg.norm(diff))
            if norm <= 1e-12:
                continue
            lookup[(int(req_idx), int(cand_idx))] = len(motifs)
            motifs.append(diff / norm)
            request_indices.append(int(req_idx))
            candidate_indices.append(int(cand_idx))
    m = grid * grid * grid * grid
    motif_matrix = np.vstack(motifs) if motifs else np.zeros((0, m), dtype=float)
    return {
        "motifs": motif_matrix,
        "request_indices": np.asarray(request_indices, dtype=int),
        "candidate_indices": np.asarray(candidate_indices, dtype=int),
        "lookup": lookup,
    }


def dp_cycle_motif_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    motif_info: dict,
) -> dict:
    motifs = np.asarray(motif_info["motifs"], dtype=float)
    k = int(motifs.shape[0])
    signal = np.zeros(k, dtype=float)
    if k == 0:
        return {"cycle_motifs": motifs, "cycle_signal": signal, "cycle_lookup": motif_info["lookup"], "cycle_eps": float(eps)}
    for t in train:
        occ = transition_occupation_from_cells(compact_cell_sequence(np.asarray(t), mn, span, grid), grid).ravel()
        coeff = motifs @ occ
        l1 = float(np.sum(np.abs(coeff)))
        if l1 > 1.0:
            coeff = coeff / max(l1, 1e-12)
        signal += coeff
    if eps > 0:
        signal = signal + rng.laplace(0.0, 1.0 / eps, size=signal.shape)
    else:
        signal = np.zeros_like(signal)
    return {"cycle_motifs": motifs, "cycle_signal": signal, "cycle_lookup": motif_info["lookup"], "cycle_eps": float(eps)}


def shuffled_cycle_motif_reward(reward: dict, rng: np.random.Generator) -> dict:
    signal = np.asarray(reward["cycle_signal"], dtype=float).copy()
    if signal.size:
        signal = signal[rng.permutation(signal.size)]
        signal = signal * rng.choice(np.asarray([-1.0, 1.0]), size=signal.shape)
    return {
        "cycle_motifs": np.asarray(reward["cycle_motifs"], dtype=float),
        "cycle_signal": signal,
        "cycle_lookup": reward["cycle_lookup"],
        "cycle_eps": float(reward.get("cycle_eps", 0.0)),
    }


def dp_graph_flow_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
    public_flow: np.ndarray,
) -> dict[str, np.ndarray]:
    """DP release of graph path occupation and public-baseline log-lift.

    Each trajectory contributes a normalized cell-transition occupation measure
    with L1 mass at most 1. The released noisy occupation vector is then
    compared with the public candidate-0 occupation baseline.
    """
    n = grid * grid
    hist = np.zeros((n, n), dtype=float)
    for t in train:
        cells = compact_cell_sequence(np.asarray(t), mn, span, grid)
        trans = [(a, b) for a, b in zip(cells[:-1], cells[1:]) if a != b]
        if not trans:
            continue
        total = max(float(len(trans)), 1e-12)
        for a, b in trans:
            hist[int(a), int(b)] += 1.0 / total
    if eps <= 0:
        noisy = np.zeros_like(hist)
    else:
        noisy = np.maximum(hist + rng.laplace(0.0, 1.0 / eps, size=hist.shape), 0.0)
    # Public-flow prior stabilizes the noisy Markov bridge kernel. This is
    # post-processing because public_flow is built from public candidates.
    effective = noisy + GRAPH_FLOW_PUBLIC_PRIOR * np.asarray(public_flow, dtype=float) + SMOOTH_ETA
    row_support = effective.sum(axis=1, keepdims=True)
    uncertainty = 1.0 / np.sqrt(np.maximum(row_support, 1e-12))
    p_dp = effective
    p_dp = p_dp / np.maximum(p_dp.sum(axis=1, keepdims=True), 1e-12)
    p_pub = np.asarray(public_flow, dtype=float) + SMOOTH_ETA
    p_pub = p_pub / np.maximum(p_pub.sum(axis=1, keepdims=True), 1e-12)
    return {
        "graph_flow": np.log(p_dp + 1e-12) - np.log(p_pub + 1e-12),
        "graph_flow_uncertainty": np.repeat(uncertainty, n, axis=1),
    }


def dp_od_transition(train: list[np.ndarray], mn: np.ndarray, span: np.ndarray, rng: np.random.Generator, eps: float, grid: int) -> np.ndarray:
    n = grid * grid
    n_od = len(OD_BINS) - 1
    hist = np.zeros((n_od, n, n), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2:
            continue
        ob = od_bin_for_points(arr[0], arr[-1], mn, span)
        cells = compact_cell_sequence(arr, mn, span, grid)
        trans = [(a, b) for a, b in zip(cells[:-1], cells[1:]) if a != b]
        if not trans:
            continue
        counts: dict[tuple[int, int], float] = {}
        for a, b in trans:
            counts[(a, b)] = counts.get((a, b), 0.0) + 1.0
        total = max(float(sum(counts.values())), 1e-12)
        for (a, b), v in counts.items():
            hist[ob, a, b] += v / total
    if eps <= 0:
        noisy = np.zeros_like(hist)
    else:
        noisy = np.maximum(hist + rng.laplace(0.0, 1.0 / eps, size=hist.shape), 0.0)
    global_noisy = noisy.sum(axis=0)
    global_row = global_noisy + SMOOTH_ETA
    global_row = global_row / np.maximum(global_row.sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros_like(noisy)
    for ob in range(n_od):
        row = noisy[ob] + SMOOTH_ETA
        row = row / np.maximum(row.sum(axis=1, keepdims=True), 1e-12)
        out[ob] = np.log(row) - np.log(global_row + 1e-12)
    return out


def dp_od_length_phase_transition(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
) -> np.ndarray:
    """Release one total-L1-bounded OD/length/phase transition tensor."""
    n = grid * grid
    n_od = len(OD_BINS) - 1
    n_len = len(LENGTH_BINS) - 1
    hist = np.zeros((n_od, n_len, PHASE_BINS, n, n), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2:
            continue
        ob = od_bin_for_points(arr[0], arr[-1], mn, span)
        lb = length_bin_for_len(len(arr))
        cells = compact_cell_sequence(arr, mn, span, grid)
        trans = [(a, b) for a, b in zip(cells[:-1], cells[1:]) if a != b]
        if not trans:
            continue
        counts: dict[tuple[int, int, int, int, int], float] = {}
        denom = max(len(trans) - 1, 1)
        for i, (a, b) in enumerate(trans):
            phase = min(int((i / denom) * PHASE_BINS), PHASE_BINS - 1)
            key = (ob, lb, phase, int(a), int(b))
            counts[key] = counts.get(key, 0.0) + 1.0
        total = max(float(sum(counts.values())), 1e-12)
        for (obx, lbx, phase, a, b), v in counts.items():
            hist[obx, lbx, phase, a, b] += v / total
    if eps <= 0:
        noisy = np.zeros_like(hist)
    else:
        noisy = np.maximum(hist + rng.laplace(0.0, 1.0 / eps, size=hist.shape), 0.0)

    global_noisy = noisy.sum(axis=(0, 1, 2))
    global_row = global_noisy + SMOOTH_ETA
    global_row = global_row / np.maximum(global_row.sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros_like(noisy)
    for ob in range(n_od):
        for lb in range(n_len):
            for phase in range(PHASE_BINS):
                row = noisy[ob, lb, phase] + SMOOTH_ETA
                row = row / np.maximum(row.sum(axis=1, keepdims=True), 1e-12)
                out[ob, lb, phase] = np.log(row) - np.log(global_row + 1e-12)
    return out


def dp_backoff_olp_transition(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
) -> np.ndarray:
    """One-release OLP tensor with post-processed hierarchical backoff."""
    n = grid * grid
    n_od = len(OD_BINS) - 1
    n_len = len(LENGTH_BINS) - 1
    hist = np.zeros((n_od, n_len, PHASE_BINS, n, n), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2:
            continue
        ob = od_bin_for_points(arr[0], arr[-1], mn, span)
        lb = length_bin_for_len(len(arr))
        cells = compact_cell_sequence(arr, mn, span, grid)
        trans = [(a, b) for a, b in zip(cells[:-1], cells[1:]) if a != b]
        if not trans:
            continue
        counts: dict[tuple[int, int, int, int, int], float] = {}
        denom = max(len(trans) - 1, 1)
        for i, (a, b) in enumerate(trans):
            phase = min(int((i / denom) * PHASE_BINS), PHASE_BINS - 1)
            key = (ob, lb, phase, int(a), int(b))
            counts[key] = counts.get(key, 0.0) + 1.0
        total = max(float(sum(counts.values())), 1e-12)
        for (obx, lbx, phase, a, b), v in counts.items():
            hist[obx, lbx, phase, a, b] += v / total
    if eps <= 0:
        noisy = np.zeros_like(hist)
    else:
        noisy = np.maximum(hist + rng.laplace(0.0, 1.0 / eps, size=hist.shape), 0.0)

    def row_norm(x: np.ndarray) -> np.ndarray:
        row = x + SMOOTH_ETA
        return row / np.maximum(row.sum(axis=-1, keepdims=True), 1e-12)

    global_counts = noisy.sum(axis=(0, 1, 2))
    global_row = row_norm(global_counts)
    od_counts = noisy.sum(axis=(1, 2))
    od_len_counts = noisy.sum(axis=2)
    out = np.zeros_like(noisy)
    for ob in range(n_od):
        od_row = row_norm(od_counts[ob])
        for lb in range(n_len):
            od_len_row = row_norm(od_len_counts[ob, lb])
            for phase in range(PHASE_BINS):
                spec = noisy[ob, lb, phase]
                spec_row = row_norm(spec)
                mass = float(spec.sum())
                w_spec = mass / (mass + BACKOFF_TAU)
                parent_mass = float(od_len_counts[ob, lb].sum())
                w_od_len = (1.0 - w_spec) * parent_mass / (parent_mass + BACKOFF_TAU)
                od_mass = float(od_counts[ob].sum())
                w_od = (1.0 - w_spec - w_od_len) * od_mass / (od_mass + BACKOFF_TAU)
                w_global = max(1.0 - w_spec - w_od_len - w_od, 0.0)
                blend = w_spec * spec_row + w_od_len * od_len_row + w_od * od_row + w_global * global_row
                blend = blend / np.maximum(blend.sum(axis=1, keepdims=True), 1e-12)
                out[ob, lb, phase] = np.log(blend + 1e-12) - np.log(global_row + 1e-12)
    return out


def _log_lift_from_counts(noisy: np.ndarray, global_noisy: np.ndarray) -> np.ndarray:
    row = noisy + SMOOTH_ETA
    row = row / np.maximum(row.sum(axis=-1, keepdims=True), 1e-12)
    global_row = global_noisy + SMOOTH_ETA
    global_row = global_row / np.maximum(global_row.sum(axis=-1, keepdims=True), 1e-12)
    return np.log(row + 1e-12) - np.log(global_row + 1e-12)


def dp_factorized_transition_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int,
) -> dict[str, np.ndarray]:
    """Release OD/length/phase transition marginals as one sensitivity-1 workload.

    A trajectory contributes normalized transition mass to three marginal views.
    The view weights sum to <= 1, so the concatenated vector has L1 sensitivity
    at most 1 under add/remove trajectory adjacency.
    """
    n = grid * grid
    n_od = len(OD_BINS) - 1
    n_len = len(LENGTH_BINS) - 1
    w_od, w_len, w_phase = FACTOR_WEIGHTS
    od_hist = np.zeros((n_od, n, n), dtype=float)
    len_hist = np.zeros((n_len, n, n), dtype=float)
    phase_hist = np.zeros((PHASE_BINS, n, n), dtype=float)
    global_hist = np.zeros((n, n), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2:
            continue
        ob = od_bin_for_points(arr[0], arr[-1], mn, span)
        lb = length_bin_for_len(len(arr))
        cells = compact_cell_sequence(arr, mn, span, grid)
        trans = [(a, b) for a, b in zip(cells[:-1], cells[1:]) if a != b]
        if not trans:
            continue
        denom = max(len(trans) - 1, 1)
        total = max(float(len(trans)), 1e-12)
        for i, (a, b) in enumerate(trans):
            phase = min(int((i / denom) * PHASE_BINS), PHASE_BINS - 1)
            mass = 1.0 / total
            od_hist[ob, int(a), int(b)] += w_od * mass
            len_hist[lb, int(a), int(b)] += w_len * mass
            phase_hist[phase, int(a), int(b)] += w_phase * mass
            global_hist[int(a), int(b)] += mass
    if eps <= 0:
        od_noisy = np.zeros_like(od_hist)
        len_noisy = np.zeros_like(len_hist)
        phase_noisy = np.zeros_like(phase_hist)
    else:
        od_noisy = np.maximum(od_hist + rng.laplace(0.0, 1.0 / eps, size=od_hist.shape), 0.0)
        len_noisy = np.maximum(len_hist + rng.laplace(0.0, 1.0 / eps, size=len_hist.shape), 0.0)
        phase_noisy = np.maximum(phase_hist + rng.laplace(0.0, 1.0 / eps, size=phase_hist.shape), 0.0)
    global_noisy = od_noisy.sum(axis=0) + len_noisy.sum(axis=0) + phase_noisy.sum(axis=0)
    return {
        "od": _log_lift_from_counts(od_noisy, global_noisy),
        "length": _log_lift_from_counts(len_noisy, global_noisy),
        "phase": _log_lift_from_counts(phase_noisy, global_noisy),
    }


def _log_lift_vector(noisy: np.ndarray, global_noisy: np.ndarray) -> np.ndarray:
    p = noisy + SMOOTH_ETA
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    g = global_noisy + SMOOTH_ETA
    g = g / max(float(g.sum()), 1e-12)
    return np.log(p + 1e-12) - np.log(g[None, :] + 1e-12)


def dp_route_feature_reward(
    train: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    grid: int = FEATURE_GRID,
) -> dict[str, np.ndarray]:
    """DP route-feature marginals for candidate-level ranking.

    Each trajectory contributes cell-occupancy mass to OD, length, and phase
    views with weights summing to 1. This is a lower-dimensional path-level
    feature workload rather than a transition-bigram workload.
    """
    n = grid * grid
    n_od = len(OD_BINS) - 1
    n_len = len(LENGTH_BINS) - 1
    w_od, w_len, w_phase = FACTOR_WEIGHTS
    od_hist = np.zeros((n_od, n), dtype=float)
    len_hist = np.zeros((n_len, n), dtype=float)
    phase_hist = np.zeros((PHASE_BINS, n), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2:
            continue
        ob = od_bin_for_points(arr[0], arr[-1], mn, span)
        lb = length_bin_for_len(len(arr))
        cells = compact_cell_sequence(arr, mn, span, grid)
        if not cells:
            continue
        total = max(float(len(cells)), 1e-12)
        denom = max(len(cells) - 1, 1)
        for i, cell in enumerate(cells):
            phase = min(int((i / denom) * PHASE_BINS), PHASE_BINS - 1)
            mass = 1.0 / total
            od_hist[ob, int(cell)] += w_od * mass
            len_hist[lb, int(cell)] += w_len * mass
            phase_hist[phase, int(cell)] += w_phase * mass
    if eps <= 0:
        od_noisy = np.zeros_like(od_hist)
        len_noisy = np.zeros_like(len_hist)
        phase_noisy = np.zeros_like(phase_hist)
    else:
        od_noisy = np.maximum(od_hist + rng.laplace(0.0, 1.0 / eps, size=od_hist.shape), 0.0)
        len_noisy = np.maximum(len_hist + rng.laplace(0.0, 1.0 / eps, size=len_hist.shape), 0.0)
        phase_noisy = np.maximum(phase_hist + rng.laplace(0.0, 1.0 / eps, size=phase_hist.shape), 0.0)
    global_noisy = od_noisy.sum(axis=0) + len_noisy.sum(axis=0) + phase_noisy.sum(axis=0)
    return {
        "od_cell": _log_lift_vector(od_noisy, global_noisy),
        "length_cell": _log_lift_vector(len_noisy, global_noisy),
        "phase_cell": _log_lift_vector(phase_noisy, global_noisy),
    }


def shuffled_od_reward(reward: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flat = reward.reshape(reward.shape[0], -1).copy()
    for i in range(flat.shape[0]):
        rng.shuffle(flat[i])
    return flat.reshape(reward.shape)


def shuffled_tensor_reward(reward: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flat = reward.reshape(reward.shape[:-2] + (-1,)).copy()
    for idx in np.ndindex(flat.shape[:-1]):
        rng.shuffle(flat[idx])
    return flat.reshape(reward.shape)


def flatten_reward_values(reward) -> np.ndarray:
    if isinstance(reward, dict):
        return np.concatenate([np.asarray(v, dtype=float).ravel() for v in reward.values()])
    return np.asarray(reward, dtype=float).ravel()


def shuffled_factorized_reward(reward: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {}
    for key, value in reward.items():
        flat = value.reshape(value.shape[0], -1).copy()
        for i in range(flat.shape[0]):
            rng.shuffle(flat[i])
        out[key] = flat.reshape(value.shape)
    return out


def reward_normalizer(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.quantile(finite, [0.05, 0.95])
    return float(lo), max(float(hi - lo), 1e-6)


def path_reward(nodes: list[int], node_cells: np.ndarray, reward: np.ndarray, lo: float, scale: float, *, od_bin: int | None = None) -> float:
    if len(nodes) < 2:
        return 0.0
    vals = []
    table = reward if od_bin is None else reward[int(od_bin)]
    for a, b in zip(nodes[:-1], nodes[1:]):
        ca = int(node_cells[int(a)])
        cb = int(node_cells[int(b)])
        vals.append(float(np.clip((table[ca, cb] - lo) / scale, 0.0, 1.0)))
    return float(np.mean(vals)) if vals else 0.0


def path_olp_reward(
    nodes: list[int],
    node_cells: np.ndarray,
    reward: np.ndarray,
    lo: float,
    scale: float,
    *,
    od_bin: int,
    length_bin: int,
) -> float:
    if len(nodes) < 2:
        return 0.0
    vals = []
    denom = max(len(nodes) - 2, 1)
    for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        phase = min(int((i / denom) * PHASE_BINS), PHASE_BINS - 1)
        ca = int(node_cells[int(a)])
        cb = int(node_cells[int(b)])
        vals.append(float(np.clip((reward[int(od_bin), int(length_bin), phase, ca, cb] - lo) / scale, 0.0, 1.0)))
    return float(np.mean(vals)) if vals else 0.0


def path_factorized_reward(
    nodes: list[int],
    node_cells: np.ndarray,
    reward: dict[str, np.ndarray],
    lo: float,
    scale: float,
    *,
    od_bin: int,
    length_bin: int,
) -> float:
    if len(nodes) < 2:
        return 0.0
    vals = []
    denom = max(len(nodes) - 2, 1)
    for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        phase = min(int((i / denom) * PHASE_BINS), PHASE_BINS - 1)
        ca = int(node_cells[int(a)])
        cb = int(node_cells[int(b)])
        raw = (
            FACTOR_WEIGHTS[0] * reward["od"][int(od_bin), ca, cb]
            + FACTOR_WEIGHTS[1] * reward["length"][int(length_bin), ca, cb]
            + FACTOR_WEIGHTS[2] * reward["phase"][phase, ca, cb]
        )
        vals.append(float(np.clip((raw - lo) / scale, 0.0, 1.0)))
    return float(np.mean(vals)) if vals else 0.0


def path_graph_flow_reward(
    nodes: list[int],
    node_cells: np.ndarray,
    reward: dict[str, np.ndarray],
    lo: float,
    scale: float,
    *,
    lcb_weight: float = 0.0,
) -> float:
    cells = compact_node_cells(nodes, node_cells)
    vals = []
    table = reward["graph_flow"]
    uncertainty = reward.get("graph_flow_uncertainty")
    for a, b in zip(cells[:-1], cells[1:]):
        if a != b:
            raw = float(table[int(a), int(b)])
            if uncertainty is not None and lcb_weight > 0:
                raw -= float(lcb_weight) * float(uncertainty[int(a), int(b)])
            vals.append(float(np.clip((raw - lo) / scale, 0.0, 1.0)))
    return float(np.mean(vals)) if vals else 0.0


def path_graph_flow_llr_reward(
    nodes: list[int],
    node_cells: np.ndarray,
    reward: dict[str, np.ndarray],
    *,
    lcb_weight: float = 0.0,
) -> float:
    """Path-level average log-likelihood ratio under the DP bridge.

    Unlike the exploratory nonnegative edge-score version, this preserves
    negative evidence: a transition preferred by the public bridge over the DP
    bridge lowers the path score.
    """
    cells = compact_node_cells(nodes, node_cells)
    vals = []
    table = reward["graph_flow"]
    uncertainty = reward.get("graph_flow_uncertainty")
    for a, b in zip(cells[:-1], cells[1:]):
        if a == b:
            continue
        raw = float(table[int(a), int(b)])
        if uncertainty is not None and lcb_weight > 0:
            raw -= float(lcb_weight) * float(uncertainty[int(a), int(b)])
        vals.append(float(np.clip(raw, -GRAPH_FLOW_LLR_CLIP, GRAPH_FLOW_LLR_CLIP)))
    return float(np.mean(vals)) if vals else 0.0


def path_corridor_residual_raw(
    nodes: list[int],
    public_nodes: list[int],
    node_cells: np.ndarray,
    reward: dict[str, np.ndarray],
    grid: int,
) -> float:
    basis = np.asarray(reward["corridor_basis"], dtype=float)
    signal = np.asarray(reward["corridor_signal"], dtype=float)
    if basis.size == 0 or signal.size == 0:
        return 0.0
    occ = path_transition_occupation(nodes, node_cells, grid).ravel()
    public_occ = path_transition_occupation(public_nodes, node_cells, grid).ravel()
    coeff = basis @ (occ - public_occ)
    return float(np.dot(signal, coeff))


def path_corridor_residual_reward(
    nodes: list[int],
    public_nodes: list[int],
    node_cells: np.ndarray,
    reward: dict[str, np.ndarray],
    grid: int,
    scale: float,
) -> float:
    raw = path_corridor_residual_raw(nodes, public_nodes, node_cells, reward, grid)
    return float(np.clip(raw / max(scale, 1e-12), -CORRIDOR_SCORE_CLIP, CORRIDOR_SCORE_CLIP))


def path_local_corridor_residual_raw(req_idx: int, cand_idx: int, reward: dict) -> float:
    if cand_idx == 0:
        return 0.0
    residual_idx = reward.get("local_lookup", {}).get((int(req_idx), int(cand_idx)))
    if residual_idx is None:
        return 0.0
    residuals = np.asarray(reward["local_residuals"], dtype=float)
    residual_od = np.asarray(reward["local_residual_od"], dtype=int)
    if residual_idx < 0 or residual_idx >= residuals.shape[0] or residual_idx >= residual_od.size:
        return 0.0
    od = int(residual_od[int(residual_idx)])
    basis = np.asarray(reward["local_basis_by_od"].get(od, np.zeros((0, residuals.shape[1]))), dtype=float)
    signal = np.asarray(reward["local_signal_by_od"].get(od, np.zeros(0)), dtype=float)
    if basis.size == 0 or signal.size == 0:
        return 0.0
    coeff = basis @ residuals[int(residual_idx)]
    return float(np.dot(signal, coeff))


def path_local_corridor_residual_reward(req_idx: int, cand_idx: int, reward: dict, scale: float) -> float:
    raw = path_local_corridor_residual_raw(req_idx, cand_idx, reward)
    return float(np.clip(raw / max(scale, 1e-12), -LOCAL_CORRIDOR_SCORE_CLIP, LOCAL_CORRIDOR_SCORE_CLIP))


def local_corridor_residual_scale(candidate_bank: list[dict], reward: dict) -> tuple[float, float]:
    vals = []
    for req_idx, item in enumerate(candidate_bank):
        for cand_idx, _ in enumerate(item.get("candidates", [])):
            vals.append(abs(path_local_corridor_residual_raw(req_idx, cand_idx, reward)))
    finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    return 0.0, max(float(np.quantile(finite, 0.90)), 1e-6)


def path_cluster_corridor_residual_raw(req_idx: int, cand_idx: int, reward: dict) -> float:
    if cand_idx == 0:
        return 0.0
    residual_idx = reward.get("cluster_lookup", {}).get((int(req_idx), int(cand_idx)))
    if residual_idx is None:
        return 0.0
    residuals = np.asarray(reward["cluster_residuals"], dtype=float)
    residual_cluster = np.asarray(reward["cluster_residual_cluster"], dtype=int)
    if residual_idx < 0 or residual_idx >= residuals.shape[0] or residual_idx >= residual_cluster.size:
        return 0.0
    cluster_id = int(residual_cluster[int(residual_idx)])
    basis = np.asarray(reward["cluster_basis_by_id"].get(cluster_id, np.zeros((0, residuals.shape[1]))), dtype=float)
    signal = np.asarray(reward["cluster_signal_by_id"].get(cluster_id, np.zeros(0)), dtype=float)
    if basis.size == 0 or signal.size == 0:
        return 0.0
    coeff = basis @ residuals[int(residual_idx)]
    return float(np.dot(signal, coeff))


def path_cluster_corridor_residual_reward(req_idx: int, cand_idx: int, reward: dict, scale: float) -> float:
    raw = path_cluster_corridor_residual_raw(req_idx, cand_idx, reward)
    return float(np.clip(raw / max(scale, 1e-12), -CLUSTER_CORRIDOR_SCORE_CLIP, CLUSTER_CORRIDOR_SCORE_CLIP))


def path_cluster_vote_reward(req_idx: int, cand_idx: int, reward: dict) -> float:
    if cand_idx == 0:
        return 0.0
    residual_idx = reward.get("cluster_lookup", {}).get((int(req_idx), int(cand_idx)))
    if residual_idx is None:
        return 0.0
    residual_cluster = np.asarray(reward["cluster_residual_cluster"], dtype=int)
    signal = np.asarray(reward["cluster_vote_signal"], dtype=float)
    if residual_idx < 0 or residual_idx >= residual_cluster.size:
        return 0.0
    cluster_id = int(residual_cluster[int(residual_idx)])
    if cluster_id < 0 or cluster_id >= signal.size:
        return 0.0
    return float(signal[cluster_id])


def path_od_cluster_vote_reward(req_idx: int, cand_idx: int, reward: dict) -> float:
    if cand_idx == 0:
        return 0.0
    residual_idx = reward.get("cluster_lookup", {}).get((int(req_idx), int(cand_idx)))
    if residual_idx is None:
        return 0.0
    residual_cluster = np.asarray(reward["cluster_residual_cluster"], dtype=int)
    request_od = np.asarray(reward["od_cluster_vote_request_od"], dtype=int)
    signal = np.asarray(reward["od_cluster_vote_signal"], dtype=float)
    if residual_idx < 0 or residual_idx >= residual_cluster.size:
        return 0.0
    if req_idx < 0 or req_idx >= request_od.size:
        return 0.0
    od = int(np.clip(request_od[int(req_idx)], 0, signal.shape[0] - 1))
    cluster_id = int(residual_cluster[int(residual_idx)])
    if cluster_id < 0 or cluster_id >= signal.shape[1]:
        return 0.0
    return float(signal[od, cluster_id])


def path_odcell_cluster_vote_reward(req_idx: int, cand_idx: int, reward: dict) -> float:
    if cand_idx == 0:
        return 0.0
    residual_idx = reward.get("cluster_lookup", {}).get((int(req_idx), int(cand_idx)))
    if residual_idx is None:
        return 0.0
    residual_cluster = np.asarray(reward["cluster_residual_cluster"], dtype=int)
    request_odcell = np.asarray(reward["odcell_cluster_vote_request_odcell"], dtype=int)
    signal = np.asarray(reward["odcell_cluster_vote_signal"], dtype=float)
    if residual_idx < 0 or residual_idx >= residual_cluster.size:
        return 0.0
    if req_idx < 0 or req_idx >= request_odcell.size:
        return 0.0
    odcell = int(np.clip(request_odcell[int(req_idx)], 0, signal.shape[0] - 1))
    cluster_id = int(residual_cluster[int(residual_idx)])
    if cluster_id < 0 or cluster_id >= signal.shape[1]:
        return 0.0
    return float(signal[odcell, cluster_id])


def path_hier_cluster_vote_reward(req_idx: int, cand_idx: int, reward: dict) -> float:
    if cand_idx == 0:
        return 0.0
    residual_idx = reward.get("cluster_lookup", {}).get((int(req_idx), int(cand_idx)))
    if residual_idx is None:
        return 0.0
    residual_cluster = np.asarray(reward["cluster_residual_cluster"], dtype=int)
    request_od = np.asarray(reward["hier_cluster_vote_request_od"], dtype=int)
    request_odcell = np.asarray(reward["hier_cluster_vote_request_odcell"], dtype=int)
    global_signal = np.asarray(reward["hier_cluster_vote_signal_global"], dtype=float)
    od_signal = np.asarray(reward["hier_cluster_vote_signal_od"], dtype=float)
    odcell_signal = np.asarray(reward["hier_cluster_vote_signal_odcell"], dtype=float)
    weights = np.asarray(reward.get("hier_cluster_vote_weights", HIER_CLUSTER_WEIGHTS), dtype=float)
    if residual_idx < 0 or residual_idx >= residual_cluster.size:
        return 0.0
    if req_idx < 0 or req_idx >= request_od.size or req_idx >= request_odcell.size:
        return 0.0
    cluster_id = int(residual_cluster[int(residual_idx)])
    if cluster_id < 0 or cluster_id >= global_signal.shape[0]:
        return 0.0
    od = int(np.clip(request_od[int(req_idx)], 0, od_signal.shape[0] - 1))
    odcell = int(np.clip(request_odcell[int(req_idx)], 0, odcell_signal.shape[0] - 1))
    wg, wo, wc = (float(x) for x in weights[:3])
    score = wg * float(global_signal[cluster_id]) + wo * float(od_signal[od, cluster_id]) + wc * float(odcell_signal[odcell, cluster_id])
    return float(np.clip(score, -CLUSTER_CORRIDOR_SCORE_CLIP, CLUSTER_CORRIDOR_SCORE_CLIP))


def path_cluster_hier_consensus_reward(req_idx: int, cand_idx: int, reward: dict, scale: float) -> float:
    if cand_idx == 0:
        return 0.0
    cluster_score = path_cluster_corridor_residual_reward(req_idx, cand_idx, reward, scale)
    hier_score = path_hier_cluster_vote_reward(req_idx, cand_idx, reward)
    return float(min(cluster_score, hier_score))


def path_signature_vote_reward(req_idx: int, cand_idx: int, reward: dict) -> float:
    sig_idx = reward.get("signature_vote_candidate_lookup", {}).get((int(req_idx), int(cand_idx)))
    if sig_idx is None:
        return 0.0
    signal = np.asarray(reward["signature_vote_signal"], dtype=float)
    if sig_idx < 0 or sig_idx >= signal.size:
        return 0.0
    return float(signal[int(sig_idx)])


def path_anchor_choice_reward(req_idx: int, cand_idx: int, reward: dict) -> float:
    anchor_idx = reward.get("anchor_choice_candidate_lookup", {}).get((int(req_idx), int(cand_idx)))
    if anchor_idx is None:
        return 0.0
    global_signal = np.asarray(reward["anchor_choice_signal_global"], dtype=float)
    od_signal = np.asarray(reward["anchor_choice_signal_od"], dtype=float)
    request_od = np.asarray(reward["anchor_choice_request_od"], dtype=int)
    weights = np.asarray(reward.get("anchor_choice_weights", ANCHOR_CHOICE_WEIGHTS), dtype=float)
    if anchor_idx < 0 or anchor_idx >= global_signal.size:
        return 0.0
    if req_idx < 0 or req_idx >= request_od.size or od_signal.ndim != 2 or od_signal.shape[1] <= anchor_idx:
        return 0.0
    od = int(np.clip(request_od[int(req_idx)], 0, od_signal.shape[0] - 1))
    wg, wo = (float(x) for x in weights[:2])
    score = wg * float(global_signal[int(anchor_idx)]) + wo * float(od_signal[od, int(anchor_idx)])
    return float(np.clip(score, -CLUSTER_CORRIDOR_SCORE_CLIP, CLUSTER_CORRIDOR_SCORE_CLIP))


def path_anchor_shape_reward(req_idx: int, cand_idx: int, reward: dict) -> float:
    bucket = reward.get("anchor_shape_candidate_lookup", {}).get((int(req_idx), int(cand_idx)))
    if bucket is None:
        return 0.0
    global_signal = np.asarray(reward["anchor_shape_signal_global"], dtype=float)
    od_signal = np.asarray(reward["anchor_shape_signal_od"], dtype=float)
    request_od = np.asarray(reward["anchor_shape_request_od"], dtype=int)
    weights = np.asarray(reward.get("anchor_shape_weights", ANCHOR_SHAPE_WEIGHTS), dtype=float)
    if bucket < 0 or bucket >= global_signal.size:
        return 0.0
    if req_idx < 0 or req_idx >= request_od.size or od_signal.ndim != 2 or od_signal.shape[1] <= bucket:
        return 0.0
    od = int(np.clip(request_od[int(req_idx)], 0, od_signal.shape[0] - 1))
    wg, wo = (float(x) for x in weights[:2])
    score = wg * float(global_signal[int(bucket)]) + wo * float(od_signal[od, int(bucket)])
    return float(np.clip(score, -CLUSTER_CORRIDOR_SCORE_CLIP, CLUSTER_CORRIDOR_SCORE_CLIP))


def cluster_corridor_residual_scale(candidate_bank: list[dict], reward: dict) -> tuple[float, float]:
    vals = []
    for req_idx, item in enumerate(candidate_bank):
        for cand_idx, _ in enumerate(item.get("candidates", [])):
            vals.append(abs(path_cluster_corridor_residual_raw(req_idx, cand_idx, reward)))
    finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    return 0.0, max(float(np.quantile(finite, 0.90)), 1e-6)


def path_cut_corridor_raw(req_idx: int, cand_idx: int, reward: dict) -> float:
    if cand_idx == 0:
        return 0.0
    signals = reward.get("cut_signal_by_request", [])
    if req_idx < 0 or req_idx >= len(signals):
        return 0.0
    feat = reward.get("cut_features", {}).get((int(req_idx), int(cand_idx)))
    public_feat = reward.get("cut_features", {}).get((int(req_idx), 0))
    if feat is None:
        return 0.0
    signal = np.asarray(signals[int(req_idx)], dtype=float)
    feat = np.asarray(feat, dtype=float)
    public_feat = np.asarray(public_feat, dtype=float) if public_feat is not None else np.zeros_like(feat)
    if signal.size == 0 or feat.size == 0 or signal.size != feat.size:
        return 0.0
    return float(np.dot(signal, feat - public_feat))


def path_cut_corridor_reward(req_idx: int, cand_idx: int, reward: dict, scale: float) -> float:
    raw = path_cut_corridor_raw(req_idx, cand_idx, reward)
    return float(np.clip(raw / max(scale, 1e-12), -CUT_CORRIDOR_SCORE_CLIP, CUT_CORRIDOR_SCORE_CLIP))


def cut_corridor_scale(candidate_bank: list[dict], reward: dict) -> tuple[float, float]:
    vals = []
    for req_idx, item in enumerate(candidate_bank):
        for cand_idx, _ in enumerate(item.get("candidates", [])):
            vals.append(abs(path_cut_corridor_raw(req_idx, cand_idx, reward)))
    finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    return 0.0, max(float(np.quantile(finite, 0.90)), 1e-6)


def path_cut_band_corridor_raw(req_idx: int, cand_idx: int, reward: dict) -> float:
    if cand_idx == 0:
        return 0.0
    signals = reward.get("cut_band_signal_by_request", [])
    if req_idx < 0 or req_idx >= len(signals):
        return 0.0
    feat = reward.get("cut_band_features", {}).get((int(req_idx), int(cand_idx)))
    public_feat = reward.get("cut_band_features", {}).get((int(req_idx), 0))
    if feat is None:
        return 0.0
    signal = np.asarray(signals[int(req_idx)], dtype=float)
    feat = np.asarray(feat, dtype=float)
    public_feat = np.asarray(public_feat, dtype=float) if public_feat is not None else np.zeros_like(feat)
    if signal.size == 0 or feat.size == 0 or signal.size != feat.size:
        return 0.0
    return float(np.dot(signal, feat - public_feat))


def path_cut_band_corridor_reward(req_idx: int, cand_idx: int, reward: dict, scale: float) -> float:
    raw = path_cut_band_corridor_raw(req_idx, cand_idx, reward)
    return float(np.clip(raw / max(scale, 1e-12), -CUT_CORRIDOR_SCORE_CLIP, CUT_CORRIDOR_SCORE_CLIP))


def cut_band_corridor_scale(candidate_bank: list[dict], reward: dict) -> tuple[float, float]:
    vals = []
    for req_idx, item in enumerate(candidate_bank):
        for cand_idx, _ in enumerate(item.get("candidates", [])):
            vals.append(abs(path_cut_band_corridor_raw(req_idx, cand_idx, reward)))
    finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    return 0.0, max(float(np.quantile(finite, 0.90)), 1e-6)


def path_od_cut_band_raw(req_idx: int, cand_idx: int, reward: dict) -> float:
    if cand_idx == 0:
        return 0.0
    request_od = np.asarray(reward.get("od_cut_band_request_od", []), dtype=int)
    if req_idx < 0 or req_idx >= request_od.size:
        return 0.0
    feat = reward.get("od_cut_band_features", {}).get((int(req_idx), int(cand_idx)))
    public_feat = reward.get("od_cut_band_features", {}).get((int(req_idx), 0))
    if feat is None:
        return 0.0
    signal = np.asarray(reward["od_cut_band_signal"], dtype=float)
    od = int(request_od[int(req_idx)])
    if od < 0 or od >= signal.shape[0]:
        return 0.0
    feat = np.asarray(feat, dtype=float)
    public_feat = np.asarray(public_feat, dtype=float) if public_feat is not None else np.zeros_like(feat)
    if signal.shape[1] != feat.size:
        return 0.0
    return float(np.dot(signal[od], feat - public_feat))


def path_od_cut_band_reward(req_idx: int, cand_idx: int, reward: dict, scale: float) -> float:
    raw = path_od_cut_band_raw(req_idx, cand_idx, reward)
    return float(np.clip(raw / max(scale, 1e-12), -CUT_CORRIDOR_SCORE_CLIP, CUT_CORRIDOR_SCORE_CLIP))


def od_cut_band_scale(candidate_bank: list[dict], reward: dict) -> tuple[float, float]:
    vals = []
    for req_idx, item in enumerate(candidate_bank):
        for cand_idx, _ in enumerate(item.get("candidates", [])):
            vals.append(abs(path_od_cut_band_raw(req_idx, cand_idx, reward)))
    finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    return 0.0, max(float(np.quantile(finite, 0.90)), 1e-6)


def path_global_cut_band_raw(req_idx: int, cand_idx: int, reward: dict) -> float:
    if cand_idx == 0:
        return 0.0
    feat = reward.get("global_cut_band_features", {}).get((int(req_idx), int(cand_idx)))
    public_feat = reward.get("global_cut_band_features", {}).get((int(req_idx), 0))
    if feat is None:
        return 0.0
    signal = np.asarray(reward["global_cut_band_signal"], dtype=float)
    feat = np.asarray(feat, dtype=float)
    public_feat = np.asarray(public_feat, dtype=float) if public_feat is not None else np.zeros_like(feat)
    if signal.size != feat.size:
        return 0.0
    return float(np.dot(signal, feat - public_feat))


def path_global_cut_band_reward(req_idx: int, cand_idx: int, reward: dict, scale: float) -> float:
    raw = path_global_cut_band_raw(req_idx, cand_idx, reward)
    return float(np.clip(raw / max(scale, 1e-12), -CUT_CORRIDOR_SCORE_CLIP, CUT_CORRIDOR_SCORE_CLIP))


def global_cut_band_scale(candidate_bank: list[dict], reward: dict) -> tuple[float, float]:
    vals = []
    for req_idx, item in enumerate(candidate_bank):
        for cand_idx, _ in enumerate(item.get("candidates", [])):
            vals.append(abs(path_global_cut_band_raw(req_idx, cand_idx, reward)))
    finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    return 0.0, max(float(np.quantile(finite, 0.90)), 1e-6)


def path_family_cut_band_raw(req_idx: int, cand_idx: int, reward: dict) -> float:
    if cand_idx == 0:
        return 0.0
    family_by_request = np.asarray(reward.get("family_cut_band_by_request", []), dtype=int)
    if req_idx < 0 or req_idx >= family_by_request.size:
        return 0.0
    family = int(family_by_request[int(req_idx)])
    signal = np.asarray(reward["family_cut_band_signal"], dtype=float)
    if family < 0 or family >= signal.shape[0]:
        return 0.0
    feat = reward.get("family_cut_band_features", {}).get((int(req_idx), int(cand_idx)))
    public_feat = reward.get("family_cut_band_features", {}).get((int(req_idx), 0))
    if feat is None:
        return 0.0
    feat = np.asarray(feat, dtype=float)
    public_feat = np.asarray(public_feat, dtype=float) if public_feat is not None else np.zeros_like(feat)
    if signal.shape[1] != feat.size:
        return 0.0
    return float(np.dot(signal[family], feat - public_feat))


def path_family_cut_band_reward(req_idx: int, cand_idx: int, reward: dict, scale: float) -> float:
    raw = path_family_cut_band_raw(req_idx, cand_idx, reward)
    return float(np.clip(raw / max(scale, 1e-12), -CUT_CORRIDOR_SCORE_CLIP, CUT_CORRIDOR_SCORE_CLIP))


def family_cut_band_scale(candidate_bank: list[dict], reward: dict) -> tuple[float, float]:
    vals = []
    for req_idx, item in enumerate(candidate_bank):
        for cand_idx, _ in enumerate(item.get("candidates", [])):
            vals.append(abs(path_family_cut_band_raw(req_idx, cand_idx, reward)))
    finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    return 0.0, max(float(np.quantile(finite, 0.90)), 1e-6)


def make_cut_band_consensus_reward(family_reward: dict, global_reward: dict, family_scale: tuple[float, float], global_scale: tuple[float, float]) -> dict:
    return {
        "cut_band_consensus": True,
        "family_reward": family_reward,
        "global_reward": global_reward,
        "family_scale": float(family_scale[1]),
        "global_scale": float(global_scale[1]),
    }


def path_cut_band_consensus_reward(req_idx: int, cand_idx: int, reward: dict) -> float:
    family_raw = path_family_cut_band_raw(req_idx, cand_idx, reward["family_reward"])
    global_raw = path_global_cut_band_raw(req_idx, cand_idx, reward["global_reward"])
    family_score = float(np.clip(family_raw / max(float(reward["family_scale"]), 1e-12), -CUT_CORRIDOR_SCORE_CLIP, CUT_CORRIDOR_SCORE_CLIP))
    global_score = float(np.clip(global_raw / max(float(reward["global_scale"]), 1e-12), -CUT_CORRIDOR_SCORE_CLIP, CUT_CORRIDOR_SCORE_CLIP))
    return float(0.5 * (family_score + global_score) + 0.5 * min(family_score, global_score))


def path_bridge_corridor_consensus_reward(
    nodes: list[int],
    public_nodes: list[int],
    node_cells: np.ndarray,
    reward: dict[str, np.ndarray],
    grid: int,
    corridor_scale: float,
) -> float:
    bridge = path_graph_flow_llr_reward(nodes, node_cells, reward) / max(GRAPH_FLOW_LLR_CLIP, 1e-12)
    corridor = path_corridor_residual_reward(nodes, public_nodes, node_cells, reward, grid, corridor_scale)
    return float(0.5 * (bridge + corridor) + 0.5 * min(bridge, corridor))


def path_cycle_motif_reward(req_idx: int, cand_idx: int, reward: dict, scale: float, *, lcb: bool = False) -> float:
    if cand_idx == 0:
        return 0.0
    motif_idx = reward.get("cycle_lookup", {}).get((int(req_idx), int(cand_idx)))
    if motif_idx is None:
        return 0.0
    signal = np.asarray(reward["cycle_signal"], dtype=float)
    if motif_idx < 0 or motif_idx >= signal.size:
        return 0.0
    raw = float(signal[int(motif_idx)])
    if lcb:
        eps = max(float(reward.get("cycle_eps", 0.0)), 1e-12)
        raw -= CYCLE_MOTIF_LCB_Z / eps
    return float(np.clip(raw / max(scale, 1e-12), -CYCLE_MOTIF_SCORE_CLIP, CYCLE_MOTIF_SCORE_CLIP))


def cycle_motif_scale(reward: dict) -> tuple[float, float]:
    signal = np.asarray(reward.get("cycle_signal", []), dtype=float)
    finite = np.abs(signal[np.isfinite(signal)])
    if finite.size == 0:
        return 0.0, 1.0
    return 0.0, max(float(np.quantile(finite, 0.90)), 1e-6)


def path_hodge_cycle_reward(req_idx: int, cand_idx: int, reward: dict, scale: float) -> float:
    if cand_idx == 0:
        return 0.0
    residual_idx = reward.get("hodge_lookup", {}).get((int(req_idx), int(cand_idx)))
    if residual_idx is None:
        return 0.0
    residuals = np.asarray(reward["hodge_residuals"], dtype=float)
    basis = np.asarray(reward["hodge_basis"], dtype=float)
    signal = np.asarray(reward["hodge_signal"], dtype=float)
    if residual_idx < 0 or residual_idx >= residuals.shape[0] or basis.size == 0 or signal.size == 0:
        return 0.0
    coeff = basis @ residuals[int(residual_idx)]
    raw = float(np.dot(signal, coeff))
    return float(np.clip(raw / max(scale, 1e-12), -HODGE_CYCLE_SCORE_CLIP, HODGE_CYCLE_SCORE_CLIP))


def hodge_cycle_scale(candidate_bank: list[dict], reward: dict) -> tuple[float, float]:
    vals = []
    for req_idx, item in enumerate(candidate_bank):
        for cand_idx, _ in enumerate(item.get("candidates", [])):
            vals.append(abs(path_hodge_cycle_reward(req_idx, cand_idx, reward, 1.0)))
    finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    return 0.0, max(float(np.quantile(finite, 0.90)), 1e-6)


def path_electrical_cycle_reward(req_idx: int, cand_idx: int, reward: dict, scale: float) -> float:
    if cand_idx == 0:
        return 0.0
    residual_idx = reward.get("electrical_lookup", {}).get((int(req_idx), int(cand_idx)))
    if residual_idx is None:
        return 0.0
    residuals = np.asarray(reward["electrical_residuals"], dtype=float)
    basis = np.asarray(reward["electrical_basis"], dtype=float)
    signal = np.asarray(reward["electrical_signal"], dtype=float)
    if residual_idx < 0 or residual_idx >= residuals.shape[0] or basis.size == 0 or signal.size == 0:
        return 0.0
    coeff = basis @ residuals[int(residual_idx)]
    raw = float(np.dot(signal, coeff))
    return float(np.clip(raw / max(scale, 1e-12), -ELECTRICAL_CYCLE_SCORE_CLIP, ELECTRICAL_CYCLE_SCORE_CLIP))


def path_electrical_cycle_energy_reward(req_idx: int, cand_idx: int, reward: dict, scale: float) -> float:
    if cand_idx == 0:
        return 0.0
    residual_idx = reward.get("electrical_lookup", {}).get((int(req_idx), int(cand_idx)))
    if residual_idx is None:
        return 0.0
    residuals = np.asarray(reward["electrical_residuals"], dtype=float)
    basis = np.asarray(reward["electrical_basis"], dtype=float)
    signal = np.asarray(reward["electrical_energy_signal"], dtype=float)
    if residual_idx < 0 or residual_idx >= residuals.shape[0] or basis.size == 0 or signal.size == 0:
        return 0.0
    coeff = np.abs(basis @ residuals[int(residual_idx)])
    raw = float(np.dot(signal, coeff))
    return float(np.clip(raw / max(scale, 1e-12), 0.0, ELECTRICAL_CYCLE_SCORE_CLIP))


def electrical_cycle_scale(candidate_bank: list[dict], reward: dict) -> tuple[float, float]:
    vals = []
    for req_idx, item in enumerate(candidate_bank):
        for cand_idx, _ in enumerate(item.get("candidates", [])):
            if "electrical_energy_signal" in reward:
                vals.append(abs(path_electrical_cycle_energy_reward(req_idx, cand_idx, reward, 1.0)))
            else:
                vals.append(abs(path_electrical_cycle_reward(req_idx, cand_idx, reward, 1.0)))
    finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    return 0.0, max(float(np.quantile(finite, 0.90)), 1e-6)


def corridor_residual_scale(candidate_bank: list[dict], node_cells: np.ndarray, reward: dict[str, np.ndarray], grid: int) -> tuple[float, float]:
    vals = []
    for item in candidate_bank:
        cands = item.get("candidates", [])
        if not cands:
            continue
        public_nodes = cands[0][1]
        for _, nodes in cands:
            vals.append(abs(path_corridor_residual_raw(nodes, public_nodes, node_cells, reward, grid)))
    finite = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    scale = float(np.quantile(finite, 0.90))
    return 0.0, max(scale, 1e-6)


def path_route_feature_reward(
    nodes: list[int],
    node_cells: np.ndarray,
    reward: dict[str, np.ndarray],
    lo: float,
    scale: float,
    *,
    od_bin: int,
    length_bin: int,
) -> float:
    if len(nodes) < 2:
        return 0.0
    vals = []
    compact = []
    for node in nodes:
        cell = int(node_cells[int(node)])
        if not compact or compact[-1] != cell:
            compact.append(cell)
    denom = max(len(compact) - 1, 1)
    for i, cell in enumerate(compact):
        phase = min(int((i / denom) * PHASE_BINS), PHASE_BINS - 1)
        raw = (
            FACTOR_WEIGHTS[0] * reward["od_cell"][int(od_bin), cell]
            + FACTOR_WEIGHTS[1] * reward["length_cell"][int(length_bin), cell]
            + FACTOR_WEIGHTS[2] * reward["phase_cell"][phase, cell]
        )
        vals.append(float(np.clip((raw - lo) / scale, 0.0, 1.0)))
    return float(np.mean(vals)) if vals else 0.0


def path_route_feature_components(
    nodes: list[int],
    node_cells: np.ndarray,
    reward: dict[str, np.ndarray],
    lo: float,
    scale: float,
    *,
    od_bin: int,
    length_bin: int,
) -> np.ndarray:
    compact = compact_node_cells(nodes, node_cells)
    if not compact:
        return np.zeros(3, dtype=float)
    denom = max(len(compact) - 1, 1)
    vals = [[], [], []]
    for i, cell in enumerate(compact):
        phase = min(int((i / denom) * PHASE_BINS), PHASE_BINS - 1)
        vals[0].append(float(np.clip((reward["od_cell"][int(od_bin), cell] - lo) / scale, 0.0, 1.0)))
        vals[1].append(float(np.clip((reward["length_cell"][int(length_bin), cell] - lo) / scale, 0.0, 1.0)))
        vals[2].append(float(np.clip((reward["phase_cell"][phase, cell] - lo) / scale, 0.0, 1.0)))
    return np.asarray([float(np.mean(v)) if v else 0.0 for v in vals], dtype=float)


def path_hybrid_reward(
    nodes: list[int],
    route_node_cells: np.ndarray,
    factor_node_cells: np.ndarray,
    reward: dict,
    *,
    od_bin: int,
    length_bin: int,
) -> float:
    route_score = path_route_feature_reward(
        nodes,
        route_node_cells,
        reward["hybrid_route"],
        reward["route_scale"][0],
        reward["route_scale"][1],
        od_bin=od_bin,
        length_bin=length_bin,
    )
    factor_score = path_factorized_reward(
        nodes,
        factor_node_cells,
        reward["hybrid_factor"],
        reward["factor_scale"][0],
        reward["factor_scale"][1],
        od_bin=od_bin,
        length_bin=length_bin,
    )
    return float(HYBRID_ROUTE_WEIGHT * route_score + (1.0 - HYBRID_ROUTE_WEIGHT) * factor_score)


def compact_node_cells(nodes: list[int], node_cells: np.ndarray) -> list[int]:
    compact = []
    for node in nodes:
        cell = int(node_cells[int(node)])
        if not compact or compact[-1] != cell:
            compact.append(cell)
    return compact


def append_unique_candidate(
    candidates: list[tuple[str, list[int]]],
    label: str,
    path: list[int] | None,
    coords: np.ndarray,
    *,
    max_stretch: float | None = None,
    base_length: float | None = None,
) -> bool:
    if path is None or len(path) < 2:
        return False
    nodes = [int(x) for x in path]
    key = tuple(nodes)
    if any(tuple(int(x) for x in prev) == key for _, prev in candidates):
        return False
    if max_stretch is not None and base_length is not None and base_length > 1e-12:
        if path_length(coords, nodes) > float(max_stretch) * float(base_length):
            return False
    candidates.append((label, nodes))
    return True


def reverse_adjacency(graph: dict) -> dict[int, list[tuple[int, float]]]:
    rev: dict[int, list[tuple[int, float]]] = {int(node): [] for node in graph}
    for node, adj in graph.items():
        node = int(node)
        for nb, w in adj:
            nb = int(nb)
            rev.setdefault(nb, []).append((node, float(w)))
            rev.setdefault(node, rev.get(node, []))
    return rev


def dijkstra_distances(graph: dict, src: int, *, cutoff: float | None = None) -> dict[int, float]:
    src = int(src)
    limit = float("inf") if cutoff is None else max(float(cutoff), 0.0)
    dist: dict[int, float] = {src: 0.0}
    heap: list[tuple[float, int]] = [(0.0, src)]
    while heap:
        cur, node = heapq.heappop(heap)
        if cur != dist.get(node):
            continue
        if cur > limit:
            continue
        for nb, w in graph.get(node, []):
            nb = int(nb)
            nxt = cur + float(w)
            if nxt > limit:
                continue
            if nxt < dist.get(nb, float("inf")):
                dist[nb] = nxt
                heapq.heappush(heap, (nxt, nb))
    return dist


def shortest_path_with_bans(
    graph: dict,
    coords: np.ndarray,
    src: int,
    dst: int,
    *,
    banned_nodes: set[int] | None = None,
    banned_edges: set[tuple[int, int]] | None = None,
    max_length: float | None = None,
) -> list[int] | None:
    """A* shortest path on the public graph with temporary node/edge bans."""
    src = int(src)
    dst = int(dst)
    blocked_nodes = banned_nodes or set()
    blocked_edges = banned_edges or set()
    if src in blocked_nodes or dst in blocked_nodes:
        return None
    limit = float("inf") if max_length is None else max(float(max_length), 0.0)
    best: dict[int, float] = {src: 0.0}
    parent: dict[int, int] = {}
    heap: list[tuple[float, float, int]] = [(float(np.linalg.norm(coords[src] - coords[dst])), 0.0, src)]
    while heap:
        _, cur, node = heapq.heappop(heap)
        if cur != best.get(node):
            continue
        if node == dst:
            path = [dst]
            while path[-1] != src:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        for nb, w in graph.get(node, []):
            nb = int(nb)
            if nb in blocked_nodes or (int(node), nb) in blocked_edges:
                continue
            nxt = cur + float(w)
            if nxt > limit:
                continue
            if nxt < best.get(nb, float("inf")):
                best[nb] = nxt
                parent[nb] = int(node)
                heuristic = float(np.linalg.norm(coords[nb] - coords[dst]))
                heapq.heappush(heap, (nxt + heuristic, nxt, nb))
    return None


def shortest_path_with_edge_cost(
    graph: dict,
    coords: np.ndarray,
    src: int,
    dst: int,
    edge_cost,
) -> list[int] | None:
    """A* path search with a caller-provided nonnegative public/postprocessed cost."""
    src = int(src)
    dst = int(dst)
    best: dict[int, float] = {src: 0.0}
    parent: dict[int, int] = {}
    heap: list[tuple[float, float, int]] = [(float(np.linalg.norm(coords[src] - coords[dst])), 0.0, src)]
    while heap:
        _, cur, node = heapq.heappop(heap)
        if cur != best.get(node):
            continue
        if node == dst:
            path = [dst]
            while path[-1] != src:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        for nb, w in graph.get(node, []):
            nb = int(nb)
            cost = edge_cost(int(node), nb, float(w))
            if cost is None:
                continue
            cost = float(cost)
            if not np.isfinite(cost) or cost < 0.0:
                continue
            nxt = cur + cost
            if nxt < best.get(nb, float("inf")):
                best[nb] = nxt
                parent[nb] = int(node)
                heuristic = float(np.linalg.norm(coords[nb] - coords[dst]))
                heapq.heappush(heap, (nxt + heuristic, nxt, nb))
    return None


def path_cumulative_lengths(coords: np.ndarray, nodes: list[int]) -> np.ndarray:
    cum = np.zeros(len(nodes), dtype=float)
    for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:]), start=1):
        cum[i] = cum[i - 1] + float(np.linalg.norm(coords[int(a)] - coords[int(b)]))
    return cum


def path_index_at_fraction(cum: np.ndarray, frac: float) -> int:
    if len(cum) <= 1 or float(cum[-1]) <= 1e-12:
        return 0
    target = float(np.clip(frac, 0.0, 1.0)) * float(cum[-1])
    return int(np.searchsorted(cum, target, side="left"))


def segment_replacement_candidates(
    graph: dict,
    coords: np.ndarray,
    src: int,
    dst: int,
    base: list[int],
    *,
    max_candidates: int,
    max_trials: int,
    max_stretch: float,
) -> list[tuple[str, list[int]]]:
    """Public replacement-path candidates around phase-local base segments.

    For a segment base[i:j], temporarily forbid the original segment interior
    and edges, then reconnect base[i] to base[j]. This constructs candidates
    from graph replacement paths rather than one-edge sidetracks.
    """
    if len(base) < 5 or max_candidates <= 0:
        return []
    base = [int(x) for x in base]
    cum = path_cumulative_lengths(coords, base)
    base_length = float(cum[-1])
    if base_length <= 1e-12:
        return []
    cutoff = float(max_stretch) * base_length
    centers = [0.50, 0.35, 0.65, 0.25, 0.75, 0.42, 0.58]
    widths = [0.16, 0.26, 0.38, 0.52]
    windows: list[tuple[float, float, int, int]] = []
    for width in widths:
        for center in centers:
            lo = max(0.02, center - width / 2.0)
            hi = min(0.98, center + width / 2.0)
            i = path_index_at_fraction(cum, lo)
            j = path_index_at_fraction(cum, hi)
            if j - i < 3:
                continue
            i = max(0, min(i, len(base) - 3))
            j = max(i + 3, min(j, len(base) - 1))
            windows.append((abs(center - 0.5), width, i, j))
    windows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    out: list[tuple[str, list[int]]] = []
    seen_windows: set[tuple[int, int]] = set()
    trial_count = 0
    for _, width, i, j in windows:
        if len(out) >= max_candidates or trial_count >= max_trials:
            break
        if (i, j) in seen_windows:
            continue
        seen_windows.add((i, j))
        trial_count += 1
        prefix = base[: i + 1]
        suffix = base[j:]
        segment = base[i : j + 1]
        banned_nodes = set(int(x) for x in segment[1:-1])
        banned_edges = set((int(a), int(b)) for a, b in zip(segment[:-1], segment[1:]))
        allowed_middle = cutoff - float(cum[i]) - (base_length - float(cum[j]))
        if allowed_middle <= 0.0:
            continue
        middle = shortest_path_with_bans(
            graph,
            coords,
            int(base[i]),
            int(base[j]),
            banned_nodes=banned_nodes,
            banned_edges=banned_edges,
            max_length=allowed_middle,
        )
        if middle is None or len(middle) < 3:
            continue
        joined = [int(x) for x in prefix[:-1]] + [int(x) for x in middle] + [int(x) for x in suffix[1:]]
        if len(set(joined)) != len(joined):
            continue
        if path_length(coords, joined) > cutoff:
            continue
        center_phase = 0.5 * (float(cum[i]) + float(cum[j])) / base_length
        width_phase = (float(cum[j]) - float(cum[i])) / base_length
        label = f"public_replace_p{center_phase:.2f}_w{width_phase:.2f}".replace(".", "p")
        out.append((label, joined))
    return out


def geodesic_bridge_candidates(
    graph: dict,
    reverse_graph: dict | None,
    coords: np.ndarray,
    src: int,
    dst: int,
    base: list[int],
    *,
    max_candidates: int,
    max_stretch: float,
) -> list[tuple[str, list[int]]]:
    """Public s-t bridge candidates from graph-distance cut layers.

    Nodes are eligible when their graph-distance slack

        d(src,v) + d(v,dst) - d(src,dst)

    is small enough. Layering by d(src,v)/(d(src,v)+d(v,dst)) gives a
    source-sink cut decomposition, so the candidate family explores graph
    corridors rather than Euclidean waypoints or one-edge perturbations.
    """
    if len(base) < 2 or max_candidates <= 0:
        return []
    base_length = path_length(coords, base)
    if base_length <= 1e-12:
        return []
    cutoff = float(max_stretch) * base_length
    rev = reverse_graph if reverse_graph is not None else reverse_adjacency(graph)
    dist_s = dijkstra_distances(graph, int(src), cutoff=cutoff)
    dist_t = dijkstra_distances(rev, int(dst), cutoff=cutoff)
    direct = max(float(np.linalg.norm(coords[int(src)] - coords[int(dst)])), 1e-12)
    base_set = set(int(x) for x in base)
    rows: list[tuple[float, float, float, int]] = []
    for node, ds in dist_s.items():
        node = int(node)
        if node in base_set or node == int(src) or node == int(dst):
            continue
        dt = dist_t.get(node)
        if dt is None:
            continue
        total = float(ds) + float(dt)
        if total <= 1e-12 or total > cutoff:
            continue
        phase = float(ds) / total
        if phase <= 0.05 or phase >= 0.95:
            continue
        slack = max(0.0, (total - base_length) / base_length)
        rows.append((slack, phase, total, node))
    if not rows:
        return []

    # Conservative layers first, then more lateral exploration. All choices are
    # public and deterministic so privacy is unaffected.
    layer_targets = [0.50, 0.35, 0.65, 0.25, 0.75, 0.42, 0.58]
    layer_width = 0.10
    min_sep = 0.08 * direct
    selected: list[tuple[str, int]] = []
    selected_nodes: list[int] = []
    used: set[int] = set()
    for layer_id, target in enumerate(layer_targets):
        if len(selected) >= max_candidates:
            break
        layer = [r for r in rows if abs(r[1] - target) <= layer_width and r[3] not in used]
        layer.sort(key=lambda r: (r[0], abs(r[1] - target), r[2], r[3]))
        for slack, phase, _, node in layer:
            if len(selected) >= max_candidates:
                break
            if any(float(np.linalg.norm(coords[int(node)] - coords[int(prev)])) < min_sep for prev in selected_nodes):
                continue
            used.add(int(node))
            selected_nodes.append(int(node))
            phase_tag = f"p{phase:.2f}".replace(".", "p")
            slack_tag = f"s{slack:.2f}".replace(".", "p")
            selected.append((f"public_bridge_l{layer_id}_{phase_tag}_{slack_tag}", int(node)))

    if len(selected) < max_candidates:
        rows.sort(key=lambda r: (r[0], abs(r[1] - 0.5), r[2], r[3]))
        for slack, phase, _, node in rows:
            if len(selected) >= max_candidates:
                break
            if int(node) in used:
                continue
            if any(float(np.linalg.norm(coords[int(node)] - coords[int(prev)])) < 0.5 * min_sep for prev in selected_nodes):
                continue
            used.add(int(node))
            selected_nodes.append(int(node))
            phase_tag = f"p{phase:.2f}".replace(".", "p")
            slack_tag = f"s{slack:.2f}".replace(".", "p")
            selected.append((f"public_bridge_fill_{phase_tag}_{slack_tag}", int(node)))

    out: list[tuple[str, list[int]]] = []
    for label, via in selected:
        first = shortest_path(graph, coords, int(src), int(via))
        second = shortest_path(graph, coords, int(via), int(dst))
        if first is None or second is None or len(first) < 2 or len(second) < 2:
            continue
        joined = [int(x) for x in first] + [int(x) for x in second[1:]]
        if path_length(coords, joined) > cutoff:
            continue
        out.append((label, joined))
    return out


def via_corridor_points(src_pt: np.ndarray, dst_pt: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Public OD-corridor landmarks ordered from conservative to exploratory."""
    src_pt = np.asarray(src_pt, dtype=float)
    dst_pt = np.asarray(dst_pt, dtype=float)
    delta = dst_pt - src_pt
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-12:
        return []
    axis = delta / dist
    perp = np.asarray([-axis[1], axis[0]], dtype=float)
    schedule = [
        (0.50, 0.22),
        (0.50, -0.22),
        (0.35, 0.28),
        (0.65, -0.28),
        (0.35, -0.28),
        (0.65, 0.28),
        (0.50, 0.40),
        (0.50, -0.40),
        (0.25, 0.32),
        (0.75, -0.32),
        (0.25, -0.32),
        (0.75, 0.32),
    ]
    out = []
    for frac, offset in schedule:
        pt = src_pt + float(frac) * delta + float(offset) * dist * perp
        label = f"f{frac:.2f}_o{offset:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
        out.append((label, pt))
    return out


def anchor_via_candidates(
    graph: dict,
    coords: np.ndarray,
    src: int,
    dst: int,
    base: list[int],
    anchor_nodes: list[int] | None,
    *,
    max_candidates: int,
    max_trials: int,
    max_stretch: float,
) -> list[tuple[str, list[int]]]:
    """DP-anchor-guided via candidates.

    The anchor nodes are post-processing of the already released DP anchors
    used by the generator. They provide a low-dimensional city-hotspot prior
    for candidate generation without using raw trajectories directly here.
    """
    if not anchor_nodes or max_candidates <= 0:
        return []
    base_length = path_length(coords, base)
    if base_length <= 1e-12:
        return []
    src = int(src)
    dst = int(dst)
    src_pt = coords[src]
    dst_pt = coords[dst]
    direct = max(float(np.linalg.norm(src_pt - dst_pt)), 1e-12)
    rows = []
    for node in anchor_nodes:
        node = int(node)
        if node in {src, dst}:
            continue
        lower = float(np.linalg.norm(coords[node] - src_pt) + np.linalg.norm(coords[node] - dst_pt))
        if lower > float(max_stretch) * base_length:
            continue
        phase = float(np.linalg.norm(coords[node] - src_pt) / max(lower, 1e-12))
        axis = (dst_pt - src_pt) / direct
        offset = coords[node] - src_pt
        lateral = abs(float(axis[0] * offset[1] - axis[1] * offset[0])) / direct
        if phase <= 0.03 or phase >= 0.97:
            continue
        rows.append((lower, abs(phase - 0.5), lateral, node, phase))
    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    out: list[tuple[str, list[int]]] = []
    used: set[int] = set()
    for trial, (_, _, lateral, via, phase) in enumerate(rows[: max(int(max_trials), 0)]):
        if len(out) >= max_candidates:
            break
        if via in used:
            continue
        used.add(via)
        first = shortest_path(graph, coords, src, via)
        second = shortest_path(graph, coords, via, dst)
        if first is None or second is None or len(first) < 2 or len(second) < 2:
            continue
        joined = [int(x) for x in first] + [int(x) for x in second[1:]]
        if path_length(coords, joined) > float(max_stretch) * base_length:
            continue
        label = f"dp_anchor_via_{trial}_n{via}_p{phase:.2f}_l{lateral:.2f}".replace(".", "p")
        out.append((label, joined))
    return out


def sidetrack_deviation_candidates(
    graph: dict,
    coords: np.ndarray,
    src: int,
    dst: int,
    base: list[int],
    *,
    max_candidates: int,
    max_trials: int,
    max_stretch: float,
) -> list[tuple[str, list[int]]]:
    """Public bounded-stretch sidetrack candidates around a shortest path.

    This is a lightweight Eppstein/Yen-style approximation: enumerate one-edge
    deviations from the base shortest path, then reconnect with a public
    shortest tail. It is still public preprocessing, but it explores structured
    replacement-path directions rather than repeatedly penalizing whole routes.
    """
    if len(base) < 2 or max_candidates <= 0:
        return []
    base_length = path_length(coords, base)
    if base_length <= 1e-12:
        return []
    prefix_len = [0.0]
    for a, b in zip(base[:-1], base[1:]):
        prefix_len.append(prefix_len[-1] + float(np.linalg.norm(coords[int(a)] - coords[int(b)])))
    base_next = {int(a): int(b) for a, b in zip(base[:-1], base[1:])}
    base_prev = {int(b): int(a) for a, b in zip(base[:-1], base[1:])}
    base_set = set(int(x) for x in base)
    sidetracks = []
    for i, u in enumerate(base[:-1]):
        u = int(u)
        for nb, w in graph.get(u, []):
            nb = int(nb)
            if nb == base_next.get(u) or nb == base_prev.get(u):
                continue
            if nb in base_set:
                continue
            lower = float(prefix_len[i] + float(w) + np.linalg.norm(coords[nb] - coords[int(dst)]))
            sidetracks.append((lower, i, u, nb))
    sidetracks.sort(key=lambda x: (x[0], x[1], x[3]))

    out: list[tuple[str, list[int]]] = []
    seen_spurs: set[tuple[int, int]] = set()
    for trial, (_, i, u, nb) in enumerate(sidetracks[: max(int(max_trials), 0)]):
        if len(out) >= max_candidates:
            break
        if (u, nb) in seen_spurs:
            continue
        seen_spurs.add((u, nb))
        tail = shortest_path(graph, coords, nb, dst)
        if tail is None or len(tail) < 2:
            continue
        prefix = [int(x) for x in base[: i + 1]]
        prefix_nodes = set(prefix)
        if any(int(x) in prefix_nodes for x in tail[1:]):
            continue
        joined = prefix + [int(x) for x in tail]
        if path_length(coords, joined) > float(max_stretch) * base_length:
            continue
        out.append((f"public_sidetrack_{trial}", joined))
    return out


def generate_public_candidates(
    graph: dict,
    coords: np.ndarray,
    src: int,
    dst: int,
    *,
    max_candidates: int,
    mode: str = "reroute",
    tree: cKDTree | None = None,
    reverse_graph: dict | None = None,
    anchor_nodes: list[int] | None = None,
    max_stretch: float = 2.35,
    spur_trials: int = 24,
) -> list[tuple[str, list[int]]]:
    candidates: list[tuple[str, list[int]]] = []
    if max_candidates <= 0:
        return candidates
    base = shortest_path(graph, coords, src, dst)
    if base is None or len(base) < 2:
        return candidates
    base = [int(x) for x in base]
    base_length = path_length(coords, base)
    append_unique_candidate(candidates, "public_base", base, coords)

    if mode in {"anchor_via", "hybrid_anchor"}:
        for label, path in anchor_via_candidates(
            graph,
            coords,
            src,
            dst,
            base,
            anchor_nodes,
            max_candidates=max_candidates - len(candidates),
            max_trials=spur_trials,
            max_stretch=max_stretch,
        ):
            if len(candidates) >= max_candidates:
                break
            append_unique_candidate(candidates, label, path, coords, max_stretch=max_stretch, base_length=base_length)

    if mode in {"bridge", "hybrid_bridge"}:
        for label, path in geodesic_bridge_candidates(
            graph,
            reverse_graph,
            coords,
            src,
            dst,
            base,
            max_candidates=max_candidates - len(candidates),
            max_stretch=max_stretch,
        ):
            if len(candidates) >= max_candidates:
                break
            append_unique_candidate(candidates, label, path, coords, max_stretch=max_stretch, base_length=base_length)

    if mode in {"replacement", "hybrid_replacement"}:
        for label, path in segment_replacement_candidates(
            graph,
            coords,
            src,
            dst,
            base,
            max_candidates=max_candidates - len(candidates),
            max_trials=spur_trials,
            max_stretch=max_stretch,
        ):
            if len(candidates) >= max_candidates:
                break
            append_unique_candidate(candidates, label, path, coords, max_stretch=max_stretch, base_length=base_length)

    if mode in {"sidetrack", "hybrid_sidetrack"}:
        for label, path in sidetrack_deviation_candidates(
            graph,
            coords,
            src,
            dst,
            base,
            max_candidates=max_candidates - len(candidates),
            max_trials=spur_trials,
            max_stretch=max_stretch,
        ):
            if len(candidates) >= max_candidates:
                break
            append_unique_candidate(candidates, label, path, coords, max_stretch=max_stretch, base_length=base_length)

    if mode in {"via", "hybrid"} and tree is not None:
        used_via = {int(src), int(dst)}
        for tag, pt in via_corridor_points(coords[int(src)], coords[int(dst)]):
            if len(candidates) >= max_candidates:
                break
            via = int(tree.query(pt)[1])
            if via in used_via:
                continue
            used_via.add(via)
            first = shortest_path(graph, coords, src, via)
            second = shortest_path(graph, coords, via, dst)
            if first is None or second is None or len(first) < 2 or len(second) < 2:
                continue
            joined = [int(x) for x in first] + [int(x) for x in second[1:]]
            append_unique_candidate(
                candidates,
                f"public_via_{tag}",
                joined,
                coords,
                max_stretch=max_stretch,
                base_length=base_length,
            )

    if mode in {"reroute", "hybrid", "hybrid_sidetrack", "hybrid_bridge", "hybrid_replacement", "hybrid_anchor"} and len(candidates) < max_candidates:
        work_graph = graph
        for i in range(max_candidates):
            path = shortest_path(work_graph, coords, src, dst)
            if path is None or len(path) < 2:
                break
            append_unique_candidate(candidates, f"public_reroute_{i}", path, coords)
            if len(candidates) >= max_candidates:
                break
            work_graph = penalize_path_edges(work_graph, path, penalty=DIVERSE_REROUTE_PENALTY)
    if mode not in {"reroute", "via", "hybrid", "sidetrack", "hybrid_sidetrack", "bridge", "hybrid_bridge", "replacement", "hybrid_replacement", "anchor_via", "hybrid_anchor"}:
        raise ValueError(f"unknown candidate mode: {mode}")
    return candidates


def softmax_choice(scores: np.ndarray, rng: np.random.Generator) -> int:
    logits = (scores - scores.max()) / max(SOFTMAX_TEMP, 1e-9)
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return int(rng.choice(len(scores), p=probs))


def select_paths(
    candidate_bank: list[dict],
    coords: np.ndarray,
    node_cells: np.ndarray,
    rng: np.random.Generator,
    *,
    mode: str,
    reward: np.ndarray | None = None,
    lo_scale: tuple[float, float] | None = None,
    mn: np.ndarray | None = None,
    span: np.ndarray | None = None,
    real_signature_prob: dict[str, float] | None = None,
    eval_transition_logprob: np.ndarray | None = None,
    factor_node_cells: np.ndarray | None = None,
    switch_margin: float = ROUTE_FEATURE_SWITCH_MARGIN,
    lcb_weight: float = GRAPH_FLOW_LCB_WEIGHT,
    deterministic: bool = False,
) -> tuple[list[np.ndarray], dict]:
    syn = []
    selected_labels: dict[str, int] = {}
    reward_scores = []
    rel_lengths = []
    selected_indices = []
    selected_real_sig_mass = []
    selected_eval_logprob = []
    selected_vs_public_eval_delta = []
    selected_oracle_gap_capture = []
    positive_gap_cases = 0
    positive_gap_better_switches = 0
    positive_gap_bad_switches = 0
    positive_gap_missed_public = 0
    for req_idx, item in enumerate(candidate_bank):
        cands = item["candidates"]
        if not cands:
            continue
        direct = max(float(item["direct"]), 1e-12)
        scores = []
        eval_scores = []
        public_route_feature_bonus = 0.0
        public_route_feature_components = np.zeros(3, dtype=float)
        public_hybrid_bonus = 0.0
        public_factorized_bonus = 0.0
        public_graph_flow_bonus = 0.0
        public_rel_len = path_length(coords, cands[0][1]) / direct
        public_cells = set(compact_node_cells(cands[0][1], node_cells))
        if reward is not None and lo_scale is not None and isinstance(reward, dict) and "graph_flow" in reward and "margin" in mode:
            if "llr" in mode:
                public_graph_flow_bonus = path_graph_flow_llr_reward(
                    cands[0][1],
                    node_cells,
                    reward,
                    lcb_weight=lcb_weight if "lcb" in mode else 0.0,
                )
            else:
                public_graph_flow_bonus = path_graph_flow_reward(
                    cands[0][1],
                    node_cells,
                    reward,
                    lo_scale[0],
                    lo_scale[1],
                    lcb_weight=lcb_weight if "lcb" in mode else 0.0,
                )
        if reward is not None and isinstance(reward, dict) and "hybrid_route" in reward and factor_node_cells is not None:
            public_hybrid_bonus = path_hybrid_reward(
                cands[0][1],
                node_cells,
                factor_node_cells,
                reward,
                od_bin=int(item["od_bin"]),
                length_bin=int(item["length_bin"]),
            )
        if reward is not None and lo_scale is not None and isinstance(reward, dict) and "od" in reward and "factorized_margin" in mode:
            public_factorized_bonus = path_factorized_reward(
                cands[0][1],
                node_cells,
                reward,
                lo_scale[0],
                lo_scale[1],
                od_bin=int(item["od_bin"]),
                length_bin=int(item["length_bin"]),
            )
        if reward is not None and lo_scale is not None and isinstance(reward, dict) and "od_cell" in reward and ("contrast" in mode or "margin" in mode or "adaptive" in mode or "consensus" in mode):
            public_route_feature_bonus = path_route_feature_reward(
                cands[0][1],
                node_cells,
                reward,
                lo_scale[0],
                lo_scale[1],
                od_bin=int(item["od_bin"]),
                length_bin=int(item["length_bin"]),
            )
            if "consensus" in mode:
                public_route_feature_components = path_route_feature_components(
                    cands[0][1],
                    node_cells,
                    reward,
                    lo_scale[0],
                    lo_scale[1],
                    od_bin=int(item["od_bin"]),
                    length_bin=int(item["length_bin"]),
                )
        for cand_idx, (_, nodes) in enumerate(cands):
            rel_len = path_length(coords, nodes) / direct
            traj_for_eval = None
            if mn is not None and span is not None and eval_transition_logprob is not None:
                traj_for_eval = resample(coords[np.asarray(nodes, dtype=int)], max(2, int(item["length"])))
                eval_scores.append(path_eval_logprob(traj_for_eval, mn, span, eval_transition_logprob, grid=6))
            bonus = 0.0
            if mode == "oracle_eval_reward" and mn is not None and span is not None and eval_transition_logprob is not None:
                bonus = float(eval_scores[-1])
            elif reward is not None and lo_scale is not None:
                adaptive_threshold = 0.0
                if isinstance(reward, dict):
                    if "anchor_shape_signal_global" in reward:
                        bonus = path_anchor_shape_reward(req_idx, cand_idx, reward)
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "anchor_choice_signal_global" in reward:
                        bonus = path_anchor_choice_reward(req_idx, cand_idx, reward)
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "signature_vote_signal" in reward:
                        bonus = path_signature_vote_reward(req_idx, cand_idx, reward)
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "cluster_hier_consensus" in reward:
                        bonus = path_cluster_hier_consensus_reward(req_idx, cand_idx, reward, lo_scale[1])
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "electrical_energy_signal" in reward:
                        bonus = path_electrical_cycle_energy_reward(req_idx, cand_idx, reward, lo_scale[1])
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "electrical_signal" in reward:
                        bonus = path_electrical_cycle_reward(req_idx, cand_idx, reward, lo_scale[1])
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "hodge_signal" in reward:
                        bonus = path_hodge_cycle_reward(req_idx, cand_idx, reward, lo_scale[1])
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "cycle_signal" in reward:
                        bonus = path_cycle_motif_reward(req_idx, cand_idx, reward, lo_scale[1], lcb="lcb" in mode)
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "local_signal_by_od" in reward:
                        bonus = path_local_corridor_residual_reward(req_idx, cand_idx, reward, lo_scale[1])
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "hier_cluster_vote_signal_global" in reward:
                        bonus = path_hier_cluster_vote_reward(req_idx, cand_idx, reward)
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "odcell_cluster_vote_signal" in reward:
                        bonus = path_odcell_cluster_vote_reward(req_idx, cand_idx, reward)
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "od_cluster_vote_signal" in reward:
                        bonus = path_od_cluster_vote_reward(req_idx, cand_idx, reward)
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "cluster_vote_signal" in reward:
                        bonus = path_cluster_vote_reward(req_idx, cand_idx, reward)
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "cluster_signal_by_id" in reward:
                        bonus = path_cluster_corridor_residual_reward(req_idx, cand_idx, reward, lo_scale[1])
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "cut_band_consensus" in reward:
                        bonus = path_cut_band_consensus_reward(req_idx, cand_idx, reward)
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "family_cut_band_signal" in reward:
                        bonus = path_family_cut_band_reward(req_idx, cand_idx, reward, lo_scale[1])
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "global_cut_band_signal" in reward:
                        bonus = path_global_cut_band_reward(req_idx, cand_idx, reward, lo_scale[1])
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "od_cut_band_signal" in reward:
                        bonus = path_od_cut_band_reward(req_idx, cand_idx, reward, lo_scale[1])
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "cut_band_signal_by_request" in reward:
                        bonus = path_cut_band_corridor_reward(req_idx, cand_idx, reward, lo_scale[1])
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "cut_signal_by_request" in reward:
                        bonus = path_cut_corridor_reward(req_idx, cand_idx, reward, lo_scale[1])
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "bridge_corridor" in mode and "graph_flow" in reward and "corridor_signal" in reward:
                        bonus = path_bridge_corridor_consensus_reward(
                            nodes,
                            cands[0][1],
                            node_cells,
                            reward,
                            TRANS_GRID,
                            lo_scale[1],
                        )
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "graph_flow" in reward:
                        if "llr" in mode:
                            bonus = path_graph_flow_llr_reward(
                                nodes,
                                node_cells,
                                reward,
                                lcb_weight=lcb_weight if "lcb" in mode else 0.0,
                            )
                        else:
                            bonus = path_graph_flow_reward(
                                nodes,
                                node_cells,
                                reward,
                                lo_scale[0],
                                lo_scale[1],
                                lcb_weight=lcb_weight if "lcb" in mode else 0.0,
                            )
                        if "margin" in mode:
                            bonus -= public_graph_flow_bonus
                            if cand_idx != 0:
                                bonus -= switch_margin
                    elif "corridor_signal" in reward:
                        bonus = path_corridor_residual_reward(
                            nodes,
                            cands[0][1],
                            node_cells,
                            reward,
                            TRANS_GRID,
                            lo_scale[1],
                        )
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                    elif "hybrid_route" in reward and factor_node_cells is not None:
                        bonus = path_hybrid_reward(
                            nodes,
                            node_cells,
                            factor_node_cells,
                            reward,
                            od_bin=int(item["od_bin"]),
                            length_bin=int(item["length_bin"]),
                        )
                        if "margin" in mode:
                            bonus -= public_hybrid_bonus
                            if cand_idx != 0:
                                bonus -= switch_margin
                    elif "od_cell" in reward:
                        bonus = path_route_feature_reward(
                            nodes,
                            node_cells,
                            reward,
                            lo_scale[0],
                            lo_scale[1],
                            od_bin=int(item["od_bin"]),
                            length_bin=int(item["length_bin"]),
                        )
                        if "contrast" in mode or "margin" in mode or "adaptive" in mode:
                            bonus -= public_route_feature_bonus
                        if "margin" in mode and cand_idx != 0:
                            bonus -= switch_margin
                        if "adaptive" in mode and cand_idx != 0:
                            cand_cells = set(compact_node_cells(nodes, node_cells))
                            union = max(len(public_cells | cand_cells), 1)
                            new_fraction = len(cand_cells - public_cells) / union
                            length_extra = max(0.0, rel_len - public_rel_len)
                            adaptive_threshold = (
                                ADAPTIVE_SWITCH_BASE
                                + ADAPTIVE_SWITCH_DIFF * float(new_fraction)
                                + ADAPTIVE_SWITCH_LENGTH * float(length_extra)
                            )
                        if "consensus" in mode:
                            comps = path_route_feature_components(
                                nodes,
                                node_cells,
                                reward,
                                lo_scale[0],
                                lo_scale[1],
                                od_bin=int(item["od_bin"]),
                                length_bin=int(item["length_bin"]),
                            )
                            deltas = comps - public_route_feature_components
                            bonus = float(np.mean(deltas) + CONSENSUS_MIN_WEIGHT * np.min(deltas))
                    else:
                        bonus = path_factorized_reward(
                            nodes,
                            node_cells,
                            reward,
                            lo_scale[0],
                            lo_scale[1],
                            od_bin=int(item["od_bin"]),
                            length_bin=int(item["length_bin"]),
                        )
                        if "factorized_margin" in mode:
                            bonus -= public_factorized_bonus
                            if cand_idx != 0:
                                bonus -= switch_margin
                elif reward.ndim == 5:
                    bonus = path_olp_reward(
                        nodes,
                        node_cells,
                        reward,
                        lo_scale[0],
                        lo_scale[1],
                        od_bin=int(item["od_bin"]),
                        length_bin=int(item["length_bin"]),
                    )
                else:
                    bonus = path_reward(nodes, node_cells, reward, lo_scale[0], lo_scale[1], od_bin=item.get("od_bin") if reward.ndim == 3 else None)
            if mode == "public_only":
                score = -ALPHA_LENGTH * rel_len
            elif mode == "oracle_eval_reward":
                score = bonus
            elif "local_corridor" in mode:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * max(0.0, rel_len - public_rel_len)
            elif "electrical_energy" in mode:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * max(0.0, rel_len - public_rel_len)
            elif "electrical_cycle" in mode:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * max(0.0, rel_len - public_rel_len)
            elif "hodge_cycle" in mode:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * max(0.0, rel_len - public_rel_len)
            elif "cycle_motif" in mode:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * max(0.0, rel_len - public_rel_len)
            elif "graph_flow" in mode and "margin" in mode:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * max(0.0, rel_len - public_rel_len)
            elif "anchor_shape" in mode:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * max(0.0, rel_len - public_rel_len)
            elif "anchor_choice" in mode:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * max(0.0, rel_len - public_rel_len)
            elif "cluster_vote" in mode:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * max(0.0, rel_len - public_rel_len)
            elif "corridor" in mode:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * max(0.0, rel_len - public_rel_len)
            elif "adaptive" in mode:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * max(0.0, rel_len - public_rel_len) - adaptive_threshold
            elif "consensus" in mode:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * max(0.0, rel_len - public_rel_len)
            else:
                score = BETA_REWARD * bonus - ALPHA_LENGTH * rel_len
            scores.append(score)
        scores_arr = np.asarray(scores, dtype=float)
        if mode == "public_uniform_candidate":
            idx = int(rng.integers(0, len(cands)))
        elif mode == "public_length_only_candidate":
            idx = int(np.argmin(np.asarray([path_length(coords, nodes) / direct for _, nodes in cands], dtype=float)))
        else:
            idx = int(np.argmax(scores_arr)) if deterministic else softmax_choice(scores_arr, rng)
        label, nodes = cands[idx]
        selected_indices.append(int(idx))
        selected_labels[label] = selected_labels.get(label, 0) + 1
        rel_lengths.append(path_length(coords, nodes) / direct)
        if eval_scores:
            public_eval = float(eval_scores[0])
            selected_eval = float(eval_scores[idx])
            best_eval = float(max(eval_scores))
            gap = best_eval - public_eval
            selected_vs_public_eval_delta.append(selected_eval - public_eval)
            if gap > 1e-9:
                positive_gap_cases += 1
                selected_oracle_gap_capture.append((selected_eval - public_eval) / gap)
                if idx == 0:
                    positive_gap_missed_public += 1
                elif selected_eval > public_eval + 1e-9:
                    positive_gap_better_switches += 1
                elif selected_eval < public_eval - 1e-9:
                    positive_gap_bad_switches += 1
        if reward is not None and lo_scale is not None:
            if mode == "oracle_eval_reward" and mn is not None and span is not None and eval_transition_logprob is not None:
                reward_scores.append(float(eval_scores[idx]))
            elif isinstance(reward, dict):
                if "anchor_shape_signal_global" in reward:
                    anchor_shape_bonus = path_anchor_shape_reward(req_idx, idx, reward)
                    if "margin" in mode and idx != 0:
                        anchor_shape_bonus -= switch_margin
                    reward_scores.append(anchor_shape_bonus)
                elif "anchor_choice_signal_global" in reward:
                    anchor_choice_bonus = path_anchor_choice_reward(req_idx, idx, reward)
                    if "margin" in mode and idx != 0:
                        anchor_choice_bonus -= switch_margin
                    reward_scores.append(anchor_choice_bonus)
                elif "signature_vote_signal" in reward:
                    signature_bonus = path_signature_vote_reward(req_idx, idx, reward)
                    if "margin" in mode and idx != 0:
                        signature_bonus -= switch_margin
                    reward_scores.append(signature_bonus)
                elif "cluster_hier_consensus" in reward:
                    consensus_bonus = path_cluster_hier_consensus_reward(req_idx, idx, reward, lo_scale[1])
                    if "margin" in mode and idx != 0:
                        consensus_bonus -= switch_margin
                    reward_scores.append(consensus_bonus)
                elif "electrical_energy_signal" in reward:
                    electrical_energy_bonus = path_electrical_cycle_energy_reward(req_idx, idx, reward, lo_scale[1])
                    if "margin" in mode and idx != 0:
                        electrical_energy_bonus -= switch_margin
                    reward_scores.append(electrical_energy_bonus)
                elif "electrical_signal" in reward:
                    electrical_bonus = path_electrical_cycle_reward(req_idx, idx, reward, lo_scale[1])
                    if "margin" in mode and idx != 0:
                        electrical_bonus -= switch_margin
                    reward_scores.append(electrical_bonus)
                elif "hodge_signal" in reward:
                    hodge_bonus = path_hodge_cycle_reward(req_idx, idx, reward, lo_scale[1])
                    if "margin" in mode and idx != 0:
                        hodge_bonus -= switch_margin
                    reward_scores.append(hodge_bonus)
                elif "cycle_signal" in reward:
                    cycle_bonus = path_cycle_motif_reward(req_idx, idx, reward, lo_scale[1], lcb="lcb" in mode)
                    if "margin" in mode and idx != 0:
                        cycle_bonus -= switch_margin
                    reward_scores.append(cycle_bonus)
                elif "local_signal_by_od" in reward:
                    local_bonus = path_local_corridor_residual_reward(req_idx, idx, reward, lo_scale[1])
                    if "margin" in mode and idx != 0:
                        local_bonus -= switch_margin
                    reward_scores.append(local_bonus)
                elif "hier_cluster_vote_signal_global" in reward:
                    hier_cluster_vote_bonus = path_hier_cluster_vote_reward(req_idx, idx, reward)
                    if "margin" in mode and idx != 0:
                        hier_cluster_vote_bonus -= switch_margin
                    reward_scores.append(hier_cluster_vote_bonus)
                elif "odcell_cluster_vote_signal" in reward:
                    odcell_cluster_vote_bonus = path_odcell_cluster_vote_reward(req_idx, idx, reward)
                    if "margin" in mode and idx != 0:
                        odcell_cluster_vote_bonus -= switch_margin
                    reward_scores.append(odcell_cluster_vote_bonus)
                elif "od_cluster_vote_signal" in reward:
                    od_cluster_vote_bonus = path_od_cluster_vote_reward(req_idx, idx, reward)
                    if "margin" in mode and idx != 0:
                        od_cluster_vote_bonus -= switch_margin
                    reward_scores.append(od_cluster_vote_bonus)
                elif "cluster_vote_signal" in reward:
                    cluster_vote_bonus = path_cluster_vote_reward(req_idx, idx, reward)
                    if "margin" in mode and idx != 0:
                        cluster_vote_bonus -= switch_margin
                    reward_scores.append(cluster_vote_bonus)
                elif "cluster_signal_by_id" in reward:
                    cluster_bonus = path_cluster_corridor_residual_reward(req_idx, idx, reward, lo_scale[1])
                    if "margin" in mode and idx != 0:
                        cluster_bonus -= switch_margin
                    reward_scores.append(cluster_bonus)
                elif "cut_band_consensus" in reward:
                    consensus_cut_bonus = path_cut_band_consensus_reward(req_idx, idx, reward)
                    if "margin" in mode and idx != 0:
                        consensus_cut_bonus -= switch_margin
                    reward_scores.append(consensus_cut_bonus)
                elif "family_cut_band_signal" in reward:
                    family_cut_bonus = path_family_cut_band_reward(req_idx, idx, reward, lo_scale[1])
                    if "margin" in mode and idx != 0:
                        family_cut_bonus -= switch_margin
                    reward_scores.append(family_cut_bonus)
                elif "global_cut_band_signal" in reward:
                    global_cut_bonus = path_global_cut_band_reward(req_idx, idx, reward, lo_scale[1])
                    if "margin" in mode and idx != 0:
                        global_cut_bonus -= switch_margin
                    reward_scores.append(global_cut_bonus)
                elif "od_cut_band_signal" in reward:
                    od_cut_bonus = path_od_cut_band_reward(req_idx, idx, reward, lo_scale[1])
                    if "margin" in mode and idx != 0:
                        od_cut_bonus -= switch_margin
                    reward_scores.append(od_cut_bonus)
                elif "cut_band_signal_by_request" in reward:
                    cut_band_bonus = path_cut_band_corridor_reward(req_idx, idx, reward, lo_scale[1])
                    if "margin" in mode and idx != 0:
                        cut_band_bonus -= switch_margin
                    reward_scores.append(cut_band_bonus)
                elif "cut_signal_by_request" in reward:
                    cut_bonus = path_cut_corridor_reward(req_idx, idx, reward, lo_scale[1])
                    if "margin" in mode and idx != 0:
                        cut_bonus -= switch_margin
                    reward_scores.append(cut_bonus)
                elif "bridge_corridor" in mode and "graph_flow" in reward and "corridor_signal" in reward:
                    consensus_bonus = path_bridge_corridor_consensus_reward(
                        nodes,
                        cands[0][1],
                        node_cells,
                        reward,
                        TRANS_GRID,
                        lo_scale[1],
                    )
                    if "margin" in mode and idx != 0:
                        consensus_bonus -= switch_margin
                    reward_scores.append(consensus_bonus)
                elif "graph_flow" in reward:
                    if "llr" in mode:
                        graph_bonus = path_graph_flow_llr_reward(
                            nodes,
                            node_cells,
                            reward,
                            lcb_weight=lcb_weight if "lcb" in mode else 0.0,
                        )
                    else:
                        graph_bonus = path_graph_flow_reward(
                            nodes,
                            node_cells,
                            reward,
                            lo_scale[0],
                            lo_scale[1],
                            lcb_weight=lcb_weight if "lcb" in mode else 0.0,
                        )
                    if "margin" in mode:
                        graph_bonus -= public_graph_flow_bonus
                        if idx != 0:
                            graph_bonus -= switch_margin
                    reward_scores.append(graph_bonus)
                elif "corridor_signal" in reward:
                    corridor_bonus = path_corridor_residual_reward(
                        nodes,
                        cands[0][1],
                        node_cells,
                        reward,
                        TRANS_GRID,
                        lo_scale[1],
                    )
                    if "margin" in mode and idx != 0:
                        corridor_bonus -= switch_margin
                    reward_scores.append(corridor_bonus)
                elif "hybrid_route" in reward and factor_node_cells is not None:
                    hybrid_bonus = path_hybrid_reward(
                        nodes,
                        node_cells,
                        factor_node_cells,
                        reward,
                        od_bin=int(item["od_bin"]),
                        length_bin=int(item["length_bin"]),
                    )
                    if "margin" in mode:
                        hybrid_bonus -= public_hybrid_bonus
                        if idx != 0:
                            hybrid_bonus -= switch_margin
                    reward_scores.append(hybrid_bonus)
                elif "od_cell" in reward:
                    route_bonus = path_route_feature_reward(
                        nodes,
                        node_cells,
                        reward,
                        lo_scale[0],
                        lo_scale[1],
                        od_bin=int(item["od_bin"]),
                        length_bin=int(item["length_bin"]),
                    )
                    if "contrast" in mode:
                        route_bonus -= public_route_feature_bonus
                    if "margin" in mode:
                        route_bonus -= public_route_feature_bonus
                        if idx != 0:
                            route_bonus -= switch_margin
                    if "adaptive" in mode:
                        route_bonus -= public_route_feature_bonus
                        if idx != 0:
                            selected_cells = set(compact_node_cells(nodes, node_cells))
                            union = max(len(public_cells | selected_cells), 1)
                            new_fraction = len(selected_cells - public_cells) / union
                            length_extra = max(0.0, rel_lengths[-1] - public_rel_len)
                            route_bonus -= (
                                ADAPTIVE_SWITCH_BASE
                                + ADAPTIVE_SWITCH_DIFF * float(new_fraction)
                                + ADAPTIVE_SWITCH_LENGTH * float(length_extra)
                            )
                    if "consensus" in mode:
                        selected_components = path_route_feature_components(
                            nodes,
                            node_cells,
                            reward,
                            lo_scale[0],
                            lo_scale[1],
                            od_bin=int(item["od_bin"]),
                            length_bin=int(item["length_bin"]),
                        )
                        deltas = selected_components - public_route_feature_components
                        route_bonus = float(np.mean(deltas) + CONSENSUS_MIN_WEIGHT * np.min(deltas))
                    reward_scores.append(route_bonus)
                else:
                    factor_bonus = path_factorized_reward(
                        nodes,
                        node_cells,
                        reward,
                        lo_scale[0],
                        lo_scale[1],
                        od_bin=int(item["od_bin"]),
                        length_bin=int(item["length_bin"]),
                    )
                    if "factorized_margin" in mode:
                        factor_bonus -= public_factorized_bonus
                        if idx != 0:
                            factor_bonus -= switch_margin
                    reward_scores.append(factor_bonus)
            elif reward.ndim == 5:
                reward_scores.append(
                    path_olp_reward(
                        nodes,
                        node_cells,
                        reward,
                        lo_scale[0],
                        lo_scale[1],
                        od_bin=int(item["od_bin"]),
                        length_bin=int(item["length_bin"]),
                    )
                )
            else:
                reward_scores.append(path_reward(nodes, node_cells, reward, lo_scale[0], lo_scale[1], od_bin=item.get("od_bin") if reward.ndim == 3 else None))
        traj = resample(coords[np.asarray(nodes, dtype=int)], max(2, int(item["length"])))
        if mn is not None and span is not None and real_signature_prob is not None:
            key = route_signature_key(traj, mn, span, grid=6, max_len=5)
            selected_real_sig_mass.append(float(real_signature_prob.get(key, 0.0)) if key is not None else 0.0)
        if mn is not None and span is not None and eval_transition_logprob is not None:
            selected_eval_logprob.append(path_eval_logprob(traj, mn, span, eval_transition_logprob, grid=6))
        syn.append(traj)
    diag = {
        "n_selected": len(syn),
        "selected_labels": selected_labels,
        "selected_indices": selected_indices,
        "mean_relative_candidate_length": float(np.mean(rel_lengths)) if rel_lengths else None,
        "mean_selected_reward": float(np.mean(reward_scores)) if reward_scores else None,
        "mean_selected_real_signature_mass": float(np.mean(selected_real_sig_mass)) if selected_real_sig_mass else None,
        "mean_selected_eval_transition_logprob": float(np.mean(selected_eval_logprob)) if selected_eval_logprob else None,
        "selected_eval_transition_logprobs": selected_eval_logprob,
        "mean_selected_vs_public_eval_delta": float(np.mean(selected_vs_public_eval_delta)) if selected_vs_public_eval_delta else None,
        "positive_oracle_gap_cases": positive_gap_cases,
        "mean_oracle_gap_capture": float(np.mean(selected_oracle_gap_capture)) if selected_oracle_gap_capture else None,
        "positive_gap_win_rate": float(positive_gap_better_switches / positive_gap_cases) if positive_gap_cases else None,
        "positive_gap_bad_switch_rate": float(positive_gap_bad_switches / positive_gap_cases) if positive_gap_cases else None,
        "positive_gap_missed_public_rate": float(positive_gap_missed_public / positive_gap_cases) if positive_gap_cases else None,
    }
    return syn, diag


def transition_tilted_graph(
    graph: dict,
    node_cells: np.ndarray,
    reward: np.ndarray,
    lo_scale: tuple[float, float],
    *,
    gamma: float,
) -> dict:
    """Public graph with DP transition-potential penalties on low-score edges."""
    lo, scale = float(lo_scale[0]), max(float(lo_scale[1]), 1e-12)
    reward = np.asarray(reward, dtype=float)
    tilted = {}
    for node, adj in graph.items():
        node = int(node)
        ca = int(node_cells[node])
        out = []
        for nb, w in adj:
            nb = int(nb)
            cb = int(node_cells[nb])
            bonus = float(np.clip((reward[ca, cb] - lo) / scale, 0.0, 1.0))
            mult = 1.0 + float(gamma) * (1.0 - bonus)
            out.append((nb, float(w) * mult))
        tilted[node] = out
    return tilted


def select_weighted_graph_paths(
    candidate_bank: list[dict],
    coords: np.ndarray,
    weighted_graph: dict,
    *,
    label: str,
    mn: np.ndarray,
    span: np.ndarray,
    real_signature_prob: dict[str, float],
    eval_transition_logprob: np.ndarray,
    max_public_stretch: float | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Decode one path per OD request directly on a weighted public graph."""
    syn = []
    selected_labels: dict[str, int] = {}
    selected_indices = []
    rel_lengths = []
    selected_real_sig_mass = []
    selected_eval_logprob = []
    selected_vs_public_eval_delta = []
    for item in candidate_bank:
        public_nodes = item["candidates"][0][1] if item.get("candidates") else None
        nodes = shortest_path(weighted_graph, coords, int(item["src"]), int(item["dst"]))
        used_label = label
        if nodes is None or len(nodes) < 2:
            nodes = public_nodes
            used_label = f"{label}_fallback_public"
        if nodes is None or len(nodes) < 2:
            continue
        nodes = [int(x) for x in nodes]
        if public_nodes is not None and max_public_stretch is not None:
            public_len = max(path_length(coords, public_nodes), 1e-12)
            if path_length(coords, nodes) > float(max_public_stretch) * public_len:
                nodes = [int(x) for x in public_nodes]
                used_label = f"{label}_stretch_fallback_public"
        direct = max(float(item["direct"]), 1e-12)
        rel_lengths.append(path_length(coords, nodes) / direct)
        selected_indices.append(-1)
        selected_labels[used_label] = selected_labels.get(used_label, 0) + 1
        traj = resample(coords[np.asarray(nodes, dtype=int)], max(2, int(item["length"])))
        if public_nodes is not None:
            public_traj = resample(coords[np.asarray(public_nodes, dtype=int)], max(2, int(item["length"])))
            public_eval = path_eval_logprob(public_traj, mn, span, eval_transition_logprob, grid=6)
            selected_eval = path_eval_logprob(traj, mn, span, eval_transition_logprob, grid=6)
            selected_vs_public_eval_delta.append(float(selected_eval - public_eval))
            selected_eval_logprob.append(float(selected_eval))
        key = route_signature_key(traj, mn, span, grid=6, max_len=5)
        selected_real_sig_mass.append(float(real_signature_prob.get(key, 0.0)) if key is not None else 0.0)
        syn.append(traj)
    diag = {
        "n_selected": len(syn),
        "selected_labels": selected_labels,
        "selected_indices": selected_indices,
        "mean_relative_candidate_length": float(np.mean(rel_lengths)) if rel_lengths else None,
        "mean_selected_reward": None,
        "mean_selected_real_signature_mass": float(np.mean(selected_real_sig_mass)) if selected_real_sig_mass else None,
        "mean_selected_eval_transition_logprob": float(np.mean(selected_eval_logprob)) if selected_eval_logprob else None,
        "selected_eval_transition_logprobs": selected_eval_logprob,
        "mean_selected_vs_public_eval_delta": float(np.mean(selected_vs_public_eval_delta)) if selected_vs_public_eval_delta else None,
        "positive_oracle_gap_cases": None,
        "mean_oracle_gap_capture": None,
        "positive_gap_win_rate": None,
        "positive_gap_bad_switch_rate": None,
        "positive_gap_missed_public_rate": None,
    }
    return syn, diag


def select_olp_tilted_graph_paths(
    candidate_bank: list[dict],
    coords: np.ndarray,
    graph: dict,
    reverse_graph: dict,
    node_cells: np.ndarray,
    reward: np.ndarray,
    lo_scale: tuple[float, float],
    *,
    label: str,
    mn: np.ndarray,
    span: np.ndarray,
    real_signature_prob: dict[str, float],
    eval_transition_logprob: np.ndarray,
    gamma: float,
    max_public_stretch: float,
    accept_bonus_margin: float | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Route each request with an OD/length/phase-conditioned DP transition field."""
    syn = []
    selected_labels: dict[str, int] = {}
    selected_indices = []
    rel_lengths = []
    selected_real_sig_mass = []
    selected_eval_logprob = []
    selected_vs_public_eval_delta = []
    lo, scale = float(lo_scale[0]), max(float(lo_scale[1]), 1e-12)
    reward = np.asarray(reward, dtype=float)
    n_phase = int(reward.shape[2])
    for item in candidate_bank:
        public_nodes = item["candidates"][0][1] if item.get("candidates") else None
        if public_nodes is None or len(public_nodes) < 2:
            continue
        public_len = max(path_length(coords, public_nodes), 1e-12)
        cutoff = max(float(max_public_stretch), 1.0) * public_len
        dist_s = dijkstra_distances(graph, int(item["src"]), cutoff=cutoff)
        dist_t = dijkstra_distances(reverse_graph, int(item["dst"]), cutoff=cutoff)
        od_bin = int(item.get("od_bin", 0))
        length_bin = int(item.get("length_bin", 0))

        def edge_cost(u: int, v: int, w: float) -> float | None:
            ds = dist_s.get(int(u))
            dt = dist_t.get(int(u))
            dsv = dist_s.get(int(v))
            dtv = dist_t.get(int(v))
            if ds is None or dt is None or dsv is None or dtv is None:
                return None
            if float(ds) + float(dt) > cutoff or float(dsv) + float(dtv) > cutoff:
                return None
            denom = max(float(ds) + float(dt), 1e-12)
            phase = min(int((float(ds) / denom) * n_phase), n_phase - 1)
            ca = int(node_cells[int(u)])
            cb = int(node_cells[int(v)])
            val = float(reward[od_bin, length_bin, phase, ca, cb])
            bonus = float(np.clip((val - lo) / scale, 0.0, 1.0))
            return float(w) * (1.0 + float(gamma) * (1.0 - bonus))

        def path_bonus(nodes_for_score: list[int]) -> float:
            vals = []
            for u, v in zip(nodes_for_score[:-1], nodes_for_score[1:]):
                ds = dist_s.get(int(u))
                dt = dist_t.get(int(u))
                if ds is None or dt is None:
                    continue
                denom = max(float(ds) + float(dt), 1e-12)
                phase = min(int((float(ds) / denom) * n_phase), n_phase - 1)
                ca = int(node_cells[int(u)])
                cb = int(node_cells[int(v)])
                val = float(reward[od_bin, length_bin, phase, ca, cb])
                vals.append(float(np.clip((val - lo) / scale, 0.0, 1.0)))
            return float(np.mean(vals)) if vals else 0.0

        nodes = shortest_path_with_edge_cost(graph, coords, int(item["src"]), int(item["dst"]), edge_cost)
        used_label = label
        if nodes is None or len(nodes) < 2:
            nodes = [int(x) for x in public_nodes]
            used_label = f"{label}_fallback_public"
        else:
            nodes = [int(x) for x in nodes]
            if path_length(coords, nodes) > cutoff:
                nodes = [int(x) for x in public_nodes]
                used_label = f"{label}_stretch_fallback_public"
            elif accept_bonus_margin is not None:
                route_bonus = path_bonus(nodes)
                public_bonus = path_bonus([int(x) for x in public_nodes])
                if route_bonus <= public_bonus + float(accept_bonus_margin):
                    nodes = [int(x) for x in public_nodes]
                    used_label = f"{label}_confidence_fallback_public"
        direct = max(float(item["direct"]), 1e-12)
        rel_lengths.append(path_length(coords, nodes) / direct)
        selected_indices.append(-1)
        selected_labels[used_label] = selected_labels.get(used_label, 0) + 1
        traj = resample(coords[np.asarray(nodes, dtype=int)], max(2, int(item["length"])))
        public_traj = resample(coords[np.asarray(public_nodes, dtype=int)], max(2, int(item["length"])))
        public_eval = path_eval_logprob(public_traj, mn, span, eval_transition_logprob, grid=6)
        selected_eval = path_eval_logprob(traj, mn, span, eval_transition_logprob, grid=6)
        selected_vs_public_eval_delta.append(float(selected_eval - public_eval))
        selected_eval_logprob.append(float(selected_eval))
        key = route_signature_key(traj, mn, span, grid=6, max_len=5)
        selected_real_sig_mass.append(float(real_signature_prob.get(key, 0.0)) if key is not None else 0.0)
        syn.append(traj)
    diag = {
        "n_selected": len(syn),
        "selected_labels": selected_labels,
        "selected_indices": selected_indices,
        "mean_relative_candidate_length": float(np.mean(rel_lengths)) if rel_lengths else None,
        "mean_selected_reward": None,
        "mean_selected_real_signature_mass": float(np.mean(selected_real_sig_mass)) if selected_real_sig_mass else None,
        "mean_selected_eval_transition_logprob": float(np.mean(selected_eval_logprob)) if selected_eval_logprob else None,
        "selected_eval_transition_logprobs": selected_eval_logprob,
        "mean_selected_vs_public_eval_delta": float(np.mean(selected_vs_public_eval_delta)) if selected_vs_public_eval_delta else None,
        "positive_oracle_gap_cases": None,
        "mean_oracle_gap_capture": None,
        "positive_gap_win_rate": None,
        "positive_gap_bad_switch_rate": None,
        "positive_gap_missed_public_rate": None,
    }
    return syn, diag


def route_signature_hist(trajs: list[np.ndarray], mn: np.ndarray, span: np.ndarray, *, grid: int, max_len: int = 8) -> dict[str, float]:
    hist: dict[str, float] = {}
    for t in trajs:
        cells = compact_cell_sequence(np.asarray(t), mn, span, grid)
        if not cells:
            continue
        if len(cells) > max_len:
            idx = np.linspace(0, len(cells) - 1, max_len).round().astype(int)
            cells = [cells[int(i)] for i in idx]
        key = "-".join(str(int(x)) for x in cells)
        hist[key] = hist.get(key, 0.0) + 1.0
    return hist


def route_signature_key(traj: np.ndarray, mn: np.ndarray, span: np.ndarray, *, grid: int, max_len: int) -> str | None:
    cells = compact_cell_sequence(np.asarray(traj), mn, span, grid)
    if not cells:
        return None
    if len(cells) > max_len:
        idx = np.linspace(0, len(cells) - 1, max_len).round().astype(int)
        cells = [cells[int(i)] for i in idx]
    return "-".join(str(int(x)) for x in cells)


def normalize_hist(hist: dict[str, float]) -> dict[str, float]:
    total = max(float(sum(hist.values())), 1e-12)
    return {k: float(v) / total for k, v in hist.items()}


def transition_eval_logprob(trajs: list[np.ndarray], mn: np.ndarray, span: np.ndarray, *, grid: int) -> np.ndarray:
    n = grid * grid
    hist = np.zeros((n, n), dtype=float)
    for t in trajs:
        cells = compact_cell_sequence(np.asarray(t), mn, span, grid)
        for a, b in zip(cells[:-1], cells[1:]):
            if a != b:
                hist[int(a), int(b)] += 1.0
    row = hist + 1e-3
    row = row / np.maximum(row.sum(axis=1, keepdims=True), 1e-12)
    return np.log(row)


def path_eval_logprob(traj: np.ndarray, mn: np.ndarray, span: np.ndarray, table: np.ndarray, *, grid: int) -> float:
    cells = compact_cell_sequence(np.asarray(traj), mn, span, grid)
    vals = []
    for a, b in zip(cells[:-1], cells[1:]):
        if a != b:
            vals.append(float(table[int(a), int(b)]))
    return float(np.mean(vals)) if vals else float(np.log(1e-12))


def dict_js(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    av = np.asarray([a.get(k, 0.0) for k in keys], dtype=float)
    bv = np.asarray([b.get(k, 0.0) for k in keys], dtype=float)
    return js_from_counts(av, bv)


def candidate_family_hash(candidate_bank: list[dict]) -> str:
    h = hashlib.sha256()
    for item in candidate_bank:
        h.update(f"{item['origin']}:{item['dest']}:{item['src']}:{item['dst']}:{item['length']}|".encode("utf-8"))
        for label, nodes in item["candidates"]:
            h.update(label.encode("utf-8"))
            h.update(b":")
            h.update(",".join(str(int(x)) for x in nodes).encode("utf-8"))
            h.update(b"|")
    return h.hexdigest()


def candidate_cache_key(config: dict) -> str:
    text = json.dumps(config, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def normalize_candidate_bank_payload(bank: list[dict]) -> list[dict]:
    out = []
    for item in bank:
        copied = dict(item)
        copied["candidates"] = [
            (str(label), [int(x) for x in nodes])
            for label, nodes in item.get("candidates", [])
        ]
        for key in ("origin", "dest", "src", "dst", "od_bin", "od_cell_bin", "length", "length_bin"):
            if key in copied:
                copied[key] = int(copied[key])
        for key in ("direct",):
            if key in copied:
                copied[key] = float(copied[key])
        out.append(copied)
    return out


def candidate_support_diagnostics(
    candidate_bank: list[dict],
    coords: np.ndarray,
    mn: np.ndarray,
    span: np.ndarray,
    eval_logprob: np.ndarray,
) -> dict:
    public_scores = []
    best_scores = []
    best_indices = []
    for item in candidate_bank:
        cands = item["candidates"]
        if not cands:
            continue
        scores = []
        for _, nodes in cands:
            traj = resample(coords[np.asarray(nodes, dtype=int)], max(2, int(item["length"])))
            scores.append(path_eval_logprob(traj, mn, span, eval_logprob, grid=6))
        public_scores.append(float(scores[0]))
        best_idx = int(np.argmax(scores))
        best_indices.append(best_idx)
        best_scores.append(float(scores[best_idx]))
    gaps = [b - p for b, p in zip(best_scores, public_scores)]
    return {
        "mean_public_eval_logprob": float(np.mean(public_scores)) if public_scores else None,
        "mean_best_eval_logprob": float(np.mean(best_scores)) if best_scores else None,
        "mean_best_minus_public": float(np.mean(gaps)) if gaps else None,
        "fraction_with_positive_oracle_gap": float(sum(g > 1e-9 for g in gaps) / len(gaps)) if gaps else None,
        "best_indices": best_indices,
    }


def build_candidate_bank(
    max_requests: int,
    max_candidates: int,
    *,
    load_limit: int,
    train_size: int,
    eval_size: int,
    raw_graph_smoke: bool,
    seed: int,
    candidate_mode: str,
    candidate_max_stretch: float,
    candidate_spur_trials: int,
    candidate_cache_dir: str = "",
    refresh_candidate_cache: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, dict, np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    log_status(f"loading GeoLife limit={load_limit}")
    real = load_geolife_limited(load_limit)
    train = real[:train_size]
    eval_real = real[train_size : train_size + eval_size]
    norm_train, mn, span = normalize_with_public_bbox(train, BEIJING_BBOX)
    centers_norm = dp_anchors(norm_train, 0.20, rng)
    centers_real = centers_norm * span + mn
    draws, meas = sample_joint_draws(norm_train, centers_norm, rng, attempts=max_requests * 8)
    log_status(f"sampled draws={len(draws)} after {time.perf_counter() - t0:.1f}s")
    if raw_graph_smoke:
        log_status("building diagnostic raw graph")
        road_coords, graph = prepare_graph(real, raw_graph=True)
    else:
        log_status("loading Beijing OSM and building public graph")
        osm, _ = load_osm_pickle("beijing")
        road_coords, graph = prepare_graph(real, bbox=BEIJING_BBOX, osm_ways=osm)
    log_status(f"graph ready nodes={len(road_coords)} after {time.perf_counter() - t0:.1f}s")
    rev_graph = reverse_adjacency(graph)
    tree = cKDTree(road_coords)
    anchor_nodes = [int(tree.query(c)[1]) for c in centers_real]
    bank = []
    build_diag = {
        "draws_seen": 0,
        "skipped_same_anchor": 0,
        "skipped_no_candidate": 0,
        "skipped_single_candidate": 0,
        "accepted_multi_candidate": 0,
        "loaded_from_cache": False,
    }
    cache_path: Path | None = None
    if candidate_cache_dir:
        cache_config = {
            "version": 3,
            "max_requests": int(max_requests),
            "max_candidates": int(max_candidates),
            "load_limit": int(load_limit),
            "train_size": int(train_size),
            "eval_size": int(eval_size),
            "raw_graph_smoke": bool(raw_graph_smoke),
            "seed": int(seed),
            "candidate_mode": str(candidate_mode),
            "candidate_max_stretch": float(candidate_max_stretch),
            "candidate_spur_trials": int(candidate_spur_trials),
            "bbox": [float(x) for x in BEIJING_BBOX],
        }
        cache_root = Path(candidate_cache_dir)
        if not cache_root.is_absolute():
            cache_root = OUT_DIR / cache_root
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"candidate_bank_{candidate_cache_key(cache_config)}.json"
        if cache_path.exists() and not refresh_candidate_cache:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("config") == cache_config:
                cached_bank = normalize_candidate_bank_payload(payload.get("bank", []))
                build_diag.update(payload.get("build_diagnostics", {}))
                build_diag["loaded_from_cache"] = True
                build_diag["cache_path"] = str(cache_path)
                setattr(build_candidate_bank, "last_diagnostics", build_diag)
                log_status(f"loaded candidate bank cache requests={len(cached_bank)} from {cache_path}")
                return real, eval_real, road_coords, graph, mn, span, cached_bank
            log_status(f"candidate cache config mismatch, rebuilding {cache_path}")
    for o, d, lb in draws:
        build_diag["draws_seen"] += 1
        if len(bank) >= max_requests:
            break
        src = anchor_nodes[int(o)]
        dst = anchor_nodes[int(d)]
        if src == dst:
            build_diag["skipped_same_anchor"] += 1
            continue
        cands = generate_public_candidates(
            graph,
            road_coords,
            src,
            dst,
            max_candidates=max_candidates,
            mode=candidate_mode,
            tree=tree,
            reverse_graph=rev_graph,
            anchor_nodes=anchor_nodes,
            max_stretch=candidate_max_stretch,
            spur_trials=candidate_spur_trials,
        )
        if not cands:
            build_diag["skipped_no_candidate"] += 1
            continue
        if len(cands) < 2:
            build_diag["skipped_single_candidate"] += 1
            continue
        length = sample_exact_len(meas["len"], int(lb), rng)
        bank.append(
            {
                "origin": int(o),
                "dest": int(d),
                "src": int(src),
                "dst": int(dst),
                "od_bin": od_bin_for_points(centers_real[int(o)], centers_real[int(d)], mn, span),
                "od_cell_bin": od_pair_cell_bin_for_points(centers_real[int(o)], centers_real[int(d)], mn, span),
                "length": int(length),
                "length_bin": length_bin_for_len(int(length)),
                "direct": float(np.linalg.norm(road_coords[src] - road_coords[dst])),
                "candidates": cands,
            }
        )
        build_diag["accepted_multi_candidate"] += 1
        log_status(f"accepted candidate request {len(bank)}/{max_requests} after {time.perf_counter() - t0:.1f}s")
    if cache_path is not None:
        payload = {
            "config": cache_config,
            "build_diagnostics": build_diag,
            "bank": normalize_candidate_bank_payload(bank),
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=True, allow_nan=False), encoding="utf-8")
        build_diag["cache_path"] = str(cache_path)
        log_status(f"wrote candidate bank cache requests={len(bank)} to {cache_path}")
    setattr(build_candidate_bank, "last_diagnostics", build_diag)
    return real, eval_real, road_coords, graph, mn, span, bank


def evaluate_variant(name: str, syn: list[np.ndarray], diag: dict, eval_real: list[np.ndarray], context: list[np.ndarray], road_coords: np.ndarray, graph: dict, mn: np.ndarray, span: np.ndarray) -> dict:
    if SKIP_EDGE_METRICS:
        metrics = {"edge_flow_jsd": None, "edge_coverage_f1": None, "edge_transition_jsd": None, "od_corridor_jsd": None, "edge_unique_ratio": None}
    else:
        metrics = edge_metric_suite(eval_real, syn, context, road_coords, graph)
        metrics["route_signature_jsd"] = dict_js(
            route_signature_hist(eval_real, mn, span, grid=TRANS_GRID),
            route_signature_hist(syn, mn, span, grid=TRANS_GRID),
        )
    metrics["route_signature_jsd_g6"] = dict_js(
        route_signature_hist(eval_real, mn, span, grid=6, max_len=5),
        route_signature_hist(syn, mn, span, grid=6, max_len=5),
    )
    metrics["n_syn"] = len(syn)
    return {"name": name, "metrics": metrics, "diagnostics": diag}


def main() -> None:
    global SKIP_EDGE_METRICS, CYCLE_MOTIF_LCB_Z
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-requests", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--candidate-mode", choices=("reroute", "via", "hybrid", "sidetrack", "hybrid_sidetrack", "bridge", "hybrid_bridge", "replacement", "hybrid_replacement", "anchor_via", "hybrid_anchor"), default="reroute", help="Public candidate family generator.")
    parser.add_argument("--candidate-max-stretch", type=float, default=2.35, help="Maximum via-candidate length divided by the public base path length.")
    parser.add_argument("--candidate-spur-trials", type=int, default=24, help="Maximum public sidetrack spur edges to reconnect per OD request.")
    parser.add_argument("--candidate-cache-dir", default="", help="Optional directory for deterministic public candidate-bank cache.")
    parser.add_argument("--refresh-candidate-cache", action="store_true", help="Rebuild and overwrite the public candidate-bank cache.")
    parser.add_argument("--support-only", action="store_true", help="Stop after candidate support diagnostics; useful for candidate-family exploration.")
    parser.add_argument("--load-limit", type=int, default=999999)
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--eval-size", type=int, default=120)
    parser.add_argument("--raw-graph-smoke", action="store_true", help="Use a small train-derived graph for fast algorithm smoke tests only.")
    parser.add_argument("--deterministic-select", action="store_true", help="Use argmax candidate selection to isolate reward score effects.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-name", default="reward_candidate_probe.json")
    parser.add_argument("--eps-list", default=",".join(str(x) for x in EPS_SWEEP))
    parser.add_argument("--margin-list", default="", help="Optional comma-separated switch margins for extra route/factorized margin sweep.")
    parser.add_argument("--lcb-list", default="", help="Optional comma-separated graph-flow LCB uncertainty weights.")
    parser.add_argument("--corridor-basis-dim", type=int, default=CORRIDOR_BASIS_DIM, help="Public low-rank candidate residual basis dimension for corridor DP reward.")
    parser.add_argument("--local-corridor-basis-dim", type=int, default=LOCAL_CORRIDOR_BASIS_DIM, help="OD-bin-local public residual basis dimension.")
    parser.add_argument("--cluster-corridor-basis-dim", type=int, default=CLUSTER_CORRIDOR_BASIS_DIM, help="Public graph-residual-cluster local basis dimension.")
    parser.add_argument("--cluster-corridor-count", type=int, default=CLUSTER_CORRIDOR_COUNT, help="Number of public graph-residual corridor clusters.")
    parser.add_argument("--cut-corridor-layers", type=int, default=CUT_CORRIDOR_LAYERS, help="Number of source-sink cut progress layers for cut-corridor reward.")
    parser.add_argument("--cut-band-lateral-buckets", type=int, default=CUT_BAND_LATERAL_BUCKETS, help="Number of lateral corridor buckets for cut-band reward.")
    parser.add_argument("--cut-band-direction-buckets", type=int, default=CUT_BAND_DIRECTION_BUCKETS, help="Number of direction buckets for cut-band reward.")
    parser.add_argument("--od-cut-band-layers", type=int, default=OD_CUT_BAND_LAYERS, help="Number of OD-potential source-sink cut layers.")
    parser.add_argument("--od-cut-band-lateral-buckets", type=int, default=OD_CUT_BAND_LATERAL_BUCKETS, help="Number of OD-relative lateral buckets.")
    parser.add_argument("--od-cut-band-direction-buckets", type=int, default=OD_CUT_BAND_DIRECTION_BUCKETS, help="Number of OD-relative direction buckets.")
    parser.add_argument("--cut-band-family-count", type=int, default=CUT_BAND_FAMILY_COUNT, help="Number of public corridor families for family cut-band reward.")
    parser.add_argument(
        "--cut-band-family-mode",
        choices=("kmeans", "balanced"),
        default="kmeans",
        help="Public partition used by family cut-band reward.",
    )
    parser.add_argument("--hodge-cycle-basis-dim", type=int, default=HODGE_CYCLE_BASIS_DIM, help="Public graph-cycle basis dimension for Hodge/circulation DP reward.")
    parser.add_argument("--electrical-cycle-basis-dim", type=int, default=ELECTRICAL_CYCLE_BASIS_DIM, help="Public graph-Laplacian electrical residual basis dimension.")
    parser.add_argument("--cycle-lcb-z", type=float, default=CYCLE_MOTIF_LCB_Z, help="Cycle motif Laplace-scale lower-confidence penalty multiplier.")
    parser.add_argument("--only-graph-flow", action="store_true", help="Run only public/oracle and graph-flow reward variants for faster graph-bridge exploration.")
    parser.add_argument("--focus-core-graph", action="store_true", help="Run only the current core graph-reward candidates and their shuffled controls.")
    parser.add_argument("--focus-anchor-choice", action="store_true", help="Within focus-core-graph, run only anchor-via DP anchor-choice variants and shuffled controls.")
    parser.add_argument("--focus-dp-routing", action="store_true", help="Run DP-tilted graph shortest-path decoding and shuffled controls.")
    parser.add_argument("--dp-routing-gamma", type=float, default=2.0, help="Penalty strength for DP-tilted graph routing.")
    parser.add_argument("--dp-routing-max-public-stretch", type=float, default=1.35, help="Fallback to public path when DP-routed path exceeds this length ratio over public path.")
    parser.add_argument("--skip-edge-metrics", action="store_true", help="Skip expensive edge_metric_suite; keep candidate-level diagnostics for reward exploration.")
    args = parser.parse_args()
    if args.focus_anchor_choice:
        args.focus_core_graph = True
    SKIP_EDGE_METRICS = bool(args.skip_edge_metrics)
    CYCLE_MOTIF_LCB_Z = float(args.cycle_lcb_z)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    real, eval_real, road_coords, graph, mn, span, bank = build_candidate_bank(
        args.max_requests,
        args.max_candidates,
        load_limit=args.load_limit,
        train_size=args.train_size,
        eval_size=args.eval_size,
        raw_graph_smoke=args.raw_graph_smoke,
        seed=args.seed,
        candidate_mode=args.candidate_mode,
        candidate_max_stretch=args.candidate_max_stretch,
        candidate_spur_trials=args.candidate_spur_trials,
        candidate_cache_dir=args.candidate_cache_dir,
        refresh_candidate_cache=bool(args.refresh_candidate_cache),
    )
    context = real[: args.train_size + args.eval_size]
    train = real[: args.train_size]
    eval_logprob = transition_eval_logprob(eval_real, mn, span, grid=6)
    if args.support_only:
        support = candidate_support_diagnostics(bank, road_coords, mn, span, eval_logprob)
        payload = {
            "config": {
                "seed": args.seed,
                "max_requests": args.max_requests,
                "max_candidates": args.max_candidates,
                "actual_requests": len(bank),
                "candidate_mode": str(args.candidate_mode),
                "candidate_max_stretch": float(args.candidate_max_stretch),
                "candidate_spur_trials": int(args.candidate_spur_trials),
                "candidate_cache_dir": str(args.candidate_cache_dir),
                "refresh_candidate_cache": bool(args.refresh_candidate_cache),
                "load_limit": args.load_limit,
                "train_size": args.train_size,
                "eval_size": args.eval_size,
                "support_only": True,
                "script": str(Path(__file__).resolve()),
                "command": " ".join(sys.argv),
                "python": sys.version,
                "platform": platform.platform(),
                "public_osm": not args.raw_graph_smoke,
                "diagnostic_only": bool(args.raw_graph_smoke),
            },
            "candidate_diagnostics": {
                "mean_candidates": float(np.mean([len(x["candidates"]) for x in bank])) if bank else 0.0,
                "min_candidates": int(min([len(x["candidates"]) for x in bank], default=0)),
                "max_candidates": int(max([len(x["candidates"]) for x in bank], default=0)),
                "candidate_labels": [[label for label, _ in item["candidates"]] for item in bank],
                "build": getattr(build_candidate_bank, "last_diagnostics", {}),
                "anchor_choice_candidates": anchor_choice_candidate_diagnostics(bank),
                "support": support,
            },
        }
        out = OUT_DIR / args.out_name
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
        print(out)
        print(json.dumps(payload["candidate_diagnostics"]["support"], indent=2, ensure_ascii=True))
        return
    node_cells = transition_node_cells(road_coords, mn, span, np.zeros((TRANS_GRID * TRANS_GRID, TRANS_GRID * TRANS_GRID)))
    node_cells_g6 = transition_node_cells(road_coords, mn, span, np.zeros((6 * 6, 6 * 6)))
    node_cells_feat = transition_node_cells(road_coords, mn, span, np.zeros((FEATURE_GRID * FEATURE_GRID, FEATURE_GRID * FEATURE_GRID)))
    rev_graph = reverse_adjacency(graph)
    public_flow = public_candidate_transition_flow(bank, node_cells, TRANS_GRID)
    corridor_basis_info = build_corridor_residual_basis(bank, node_cells, TRANS_GRID, args.corridor_basis_dim)
    corridor_basis = np.asarray(corridor_basis_info["basis"], dtype=float)
    local_corridor_info = build_local_corridor_residual_basis(bank, node_cells, TRANS_GRID, args.local_corridor_basis_dim)
    cluster_corridor_info = build_cluster_corridor_residual_basis(
        bank,
        node_cells,
        TRANS_GRID,
        args.cluster_corridor_basis_dim,
        args.cluster_corridor_count,
    )
    cut_corridor_info = build_cut_corridor_features(
        bank,
        node_cells,
        road_coords,
        mn,
        span,
        TRANS_GRID,
        args.cut_corridor_layers,
    )
    cut_band_corridor_info = build_cut_band_corridor_features(
        bank,
        node_cells,
        road_coords,
        mn,
        span,
        TRANS_GRID,
        args.cut_corridor_layers,
        args.cut_band_lateral_buckets,
        args.cut_band_direction_buckets,
    )
    od_cut_band_info = build_od_cut_band_features(
        bank,
        node_cells,
        TRANS_GRID,
        args.od_cut_band_layers,
        args.od_cut_band_lateral_buckets,
        args.od_cut_band_direction_buckets,
    )
    cut_band_family_info = build_cut_band_family_info(
        cut_band_corridor_info,
        args.cut_band_family_count,
        args.cut_band_family_mode,
    )
    cycle_motif_info = build_cycle_motif_dictionary(bank, node_cells, TRANS_GRID)
    hodge_cycle_info = build_hodge_cycle_basis(bank, node_cells, TRANS_GRID, args.hodge_cycle_basis_dim)
    electrical_cycle_info = build_electrical_cycle_basis(bank, node_cells, TRANS_GRID, args.electrical_cycle_basis_dim)
    real_signature_prob = normalize_hist(route_signature_hist(eval_real, mn, span, grid=6, max_len=5))

    payload = {
        "config": {
            "seed": args.seed,
            "max_requests": args.max_requests,
            "max_candidates": args.max_candidates,
            "actual_requests": len(bank),
            "candidate_family": "public graph candidate family; fixed for all reward variants",
            "candidate_mode": str(args.candidate_mode),
            "candidate_max_stretch": float(args.candidate_max_stretch),
            "candidate_spur_trials": int(args.candidate_spur_trials),
            "candidate_cache_dir": str(args.candidate_cache_dir),
            "refresh_candidate_cache": bool(args.refresh_candidate_cache),
            "candidate_family_hash": candidate_family_hash(bank),
            "grid": TRANS_GRID,
            "graph_flow_public_baseline": "candidate-0 normalized cell-transition occupation over the fixed public candidate family",
            "eps_sweep": [float(x) for x in args.eps_list.split(",") if x.strip()],
            "margin_sweep": [float(x) for x in args.margin_list.split(",") if x.strip()],
            "lcb_sweep": [float(x) for x in args.lcb_list.split(",") if x.strip()],
            "corridor_basis_dim": int(args.corridor_basis_dim),
            "local_corridor_basis_dim": int(args.local_corridor_basis_dim),
            "cluster_corridor_basis_dim": int(args.cluster_corridor_basis_dim),
            "cluster_corridor_count": int(args.cluster_corridor_count),
            "cut_corridor_layers": int(args.cut_corridor_layers),
            "cut_band_lateral_buckets": int(args.cut_band_lateral_buckets),
            "cut_band_direction_buckets": int(args.cut_band_direction_buckets),
            "od_cut_band_layers": int(args.od_cut_band_layers),
            "od_cut_band_lateral_buckets": int(args.od_cut_band_lateral_buckets),
            "od_cut_band_direction_buckets": int(args.od_cut_band_direction_buckets),
            "cut_band_family_count": int(args.cut_band_family_count),
            "cut_band_family_mode": str(args.cut_band_family_mode),
            "hodge_cycle_basis_dim": int(args.hodge_cycle_basis_dim),
            "electrical_cycle_basis_dim": int(args.electrical_cycle_basis_dim),
            "cycle_lcb_z": float(args.cycle_lcb_z),
            "alpha_length": ALPHA_LENGTH,
            "beta_reward": BETA_REWARD,
            "softmax_temp": SOFTMAX_TEMP,
            "deterministic_select": bool(args.deterministic_select),
            "load_limit": args.load_limit,
            "train_size": args.train_size,
            "eval_size": args.eval_size,
            "only_graph_flow": bool(args.only_graph_flow),
            "focus_core_graph": bool(args.focus_core_graph),
            "focus_anchor_choice": bool(args.focus_anchor_choice),
            "focus_dp_routing": bool(args.focus_dp_routing),
            "dp_routing_gamma": float(args.dp_routing_gamma),
            "dp_routing_max_public_stretch": float(args.dp_routing_max_public_stretch),
            "script": str(Path(__file__).resolve()),
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "public_osm": not args.raw_graph_smoke,
            "diagnostic_only": bool(args.raw_graph_smoke),
        },
        "candidate_diagnostics": {
            "mean_candidates": float(np.mean([len(x["candidates"]) for x in bank])) if bank else 0.0,
            "min_candidates": int(min([len(x["candidates"]) for x in bank], default=0)),
            "max_candidates": int(max([len(x["candidates"]) for x in bank], default=0)),
            "build": getattr(build_candidate_bank, "last_diagnostics", {}),
            "anchor_choice_candidates": anchor_choice_candidate_diagnostics(bank),
            "invalid_for_reward_claim": len(bank) == 0,
            "support": candidate_support_diagnostics(bank, road_coords, mn, span, eval_logprob),
            "corridor_basis": {
                "rank": int(corridor_basis_info["rank"]),
                "n_residuals": int(corridor_basis_info["n_residuals"]),
                "singular_values": [float(x) for x in np.asarray(corridor_basis_info.get("singular_values", []), dtype=float)],
            },
            "local_corridor_basis": {
                "n_od_groups": int(local_corridor_info["local_n_od_groups"]),
                "n_residuals": int(local_corridor_info["local_n_residuals"]),
                "rank_by_od": {str(int(k)): int(v) for k, v in local_corridor_info["local_rank_by_od"].items()},
                "singular_values_by_od": {
                    str(int(k)): [float(x) for x in np.asarray(v, dtype=float)]
                    for k, v in local_corridor_info["local_singular_by_od"].items()
                },
            },
            "cluster_corridor_basis": {
                "n_clusters": int(cluster_corridor_info["cluster_n_clusters"]),
                "n_residuals": int(cluster_corridor_info["cluster_n_residuals"]),
                "cluster_sizes": {str(int(k)): int(v) for k, v in cluster_corridor_info["cluster_sizes"].items()},
                "rank_by_cluster": {str(int(k)): int(v) for k, v in cluster_corridor_info["cluster_rank_by_id"].items()},
                "global_singular_values": [
                    float(x) for x in np.asarray(cluster_corridor_info.get("cluster_global_singular_values", []), dtype=float)
                ],
                "singular_values_by_cluster": {
                    str(int(k)): [float(x) for x in np.asarray(v, dtype=float)]
                    for k, v in cluster_corridor_info["cluster_singular_by_id"].items()
                },
            },
            "cut_corridor": {
                "layers": int(cut_corridor_info["cut_layers"]),
                "dim_per_request": int(cut_corridor_info["cut_dim"]),
                "n_requests": int(len(cut_corridor_info["cut_layer_by_request"])),
                "mean_candidate_nonzero_features": float(cut_corridor_info["cut_mean_nonzero"]),
            },
            "cut_band_corridor": {
                "layers": int(cut_band_corridor_info["cut_band_layers"]),
                "lateral_buckets": int(cut_band_corridor_info["cut_band_lateral_buckets"]),
                "direction_buckets": int(cut_band_corridor_info["cut_band_direction_buckets"]),
                "dim_per_request": int(cut_band_corridor_info["cut_band_dim"]),
                "n_requests": int(len(cut_band_corridor_info["cut_band_geometry_by_request"])),
                "mean_candidate_nonzero_features": float(cut_band_corridor_info["cut_band_mean_nonzero"]),
            },
            "global_cut_band": {
                "layers": int(cut_band_corridor_info["cut_band_layers"]),
                "lateral_buckets": int(cut_band_corridor_info["cut_band_lateral_buckets"]),
                "direction_buckets": int(cut_band_corridor_info["cut_band_direction_buckets"]),
                "dim": int(cut_band_corridor_info["cut_band_dim"]),
                "aggregation": "one public-path-relative DP vector shared by all requests",
            },
            "family_cut_band": {
                "layers": int(cut_band_corridor_info["cut_band_layers"]),
                "lateral_buckets": int(cut_band_corridor_info["cut_band_lateral_buckets"]),
                "direction_buckets": int(cut_band_corridor_info["cut_band_direction_buckets"]),
                "dim_per_family": int(cut_band_corridor_info["cut_band_dim"]),
                "n_families": int(cut_band_family_info["cut_band_family_count"]),
                "family_sizes": {str(int(k)): int(v) for k, v in cut_band_family_info["cut_band_family_sizes"].items()},
                "family_mode": str(cut_band_family_info["cut_band_family_mode"]),
                "family_balance": cut_band_family_info["cut_band_family_balance"],
                "aggregation": "public corridor-family DP vectors",
            },
            "od_cut_band": {
                "layers": int(od_cut_band_info["od_cut_band_layers"]),
                "lateral_buckets": int(od_cut_band_info["od_cut_band_lateral_buckets"]),
                "direction_buckets": int(od_cut_band_info["od_cut_band_direction_buckets"]),
                "dim_per_od": int(od_cut_band_info["od_cut_band_dim"]),
                "n_od_bins": int(od_cut_band_info["od_cut_band_n_od_bins"]),
                "mean_candidate_nonzero_features": float(od_cut_band_info["od_cut_band_mean_nonzero"]),
            },
            "cycle_motifs": {
                "n_motifs": int(np.asarray(cycle_motif_info["motifs"]).shape[0]),
                "n_requests_with_motif": int(len(set(int(x) for x in np.asarray(cycle_motif_info["request_indices"], dtype=int)))) if np.asarray(cycle_motif_info["request_indices"]).size else 0,
            },
            "hodge_cycle": {
                "rank": int(hodge_cycle_info["hodge_rank"]),
                "n_edges": int(hodge_cycle_info["hodge_n_edges"]),
                "n_residuals": int(hodge_cycle_info["hodge_n_residuals"]),
                "singular_values": [float(x) for x in np.asarray(hodge_cycle_info.get("hodge_singular_values", []), dtype=float)],
            },
            "electrical_cycle": {
                "rank": int(electrical_cycle_info["electrical_rank"]),
                "n_edges": int(electrical_cycle_info["electrical_n_edges"]),
                "n_residuals": int(electrical_cycle_info["electrical_n_residuals"]),
                "mean_divergence_l1": float(electrical_cycle_info["electrical_mean_divergence_l1"]),
                "singular_values": [float(x) for x in np.asarray(electrical_cycle_info.get("electrical_singular_values", []), dtype=float)],
            },
        },
        "variants": {},
    }

    public_rng = np.random.default_rng(args.seed + 17)
    syn, diag = select_paths(
        bank,
        road_coords,
        node_cells,
        public_rng,
        mode="public_only",
        mn=mn,
        span=span,
        real_signature_prob=real_signature_prob,
        eval_transition_logprob=eval_logprob,
        deterministic=args.deterministic_select,
    )
    payload["variants"]["public_only"] = evaluate_variant("public_only", syn, diag, eval_real, context, road_coords, graph, mn, span)

    syn, diag = select_paths(
        bank,
        road_coords,
        node_cells,
        np.random.default_rng(args.seed + 18),
        mode="public_uniform_candidate",
        mn=mn,
        span=span,
        real_signature_prob=real_signature_prob,
        eval_transition_logprob=eval_logprob,
        deterministic=False,
    )
    payload["variants"]["public_uniform_candidate"] = evaluate_variant("public_uniform_candidate", syn, diag, eval_real, context, road_coords, graph, mn, span)

    syn, diag = select_paths(
        bank,
        road_coords,
        node_cells,
        np.random.default_rng(args.seed + 19),
        mode="public_length_only_candidate",
        mn=mn,
        span=span,
        real_signature_prob=real_signature_prob,
        eval_transition_logprob=eval_logprob,
        deterministic=True,
    )
    payload["variants"]["public_length_only_candidate"] = evaluate_variant("public_length_only_candidate", syn, diag, eval_real, context, road_coords, graph, mn, span)

    oracle_scale = reward_normalizer(eval_logprob.ravel())
    syn, diag = select_paths(
        bank,
        road_coords,
        node_cells_g6,
        np.random.default_rng(args.seed + 909),
        mode="oracle_eval_reward",
        reward=eval_logprob,
        lo_scale=oracle_scale,
        mn=mn,
        span=span,
        real_signature_prob=real_signature_prob,
        eval_transition_logprob=eval_logprob,
        deterministic=args.deterministic_select,
    )
    payload["variants"]["oracle_eval_reward_NON_DP"] = evaluate_variant("oracle_eval_reward_NON_DP", syn, diag, eval_real, context, road_coords, graph, mn, span)

    if args.focus_core_graph:
        def run_focused_variant(
            name: str,
            mode: str,
            reward: dict,
            scale: tuple[float, float] | np.ndarray,
            rng_offset: int,
            eps: float,
            *,
            switch_margin_value: float | None = None,
        ) -> None:
            syn_f, diag_f = select_paths(
                bank,
                road_coords,
                node_cells,
                np.random.default_rng(args.seed + int(eps * 1000) + rng_offset),
                mode=mode,
                reward=reward,
                lo_scale=scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                switch_margin=0.05 if switch_margin_value is None else float(switch_margin_value),
                deterministic=args.deterministic_select,
            )
            payload["variants"][name] = evaluate_variant(name, syn_f, diag_f, eval_real, context, road_coords, graph, mn, span)

        for eps in [float(x) for x in args.eps_list.split(",") if x.strip()]:
            if eps <= 0:
                continue

            if args.focus_dp_routing:
                dp_route_reward = dp_global_transition(
                    train,
                    mn,
                    span,
                    np.random.default_rng(args.seed + int(eps * 1000) + 380),
                    eps,
                    TRANS_GRID,
                )
                dp_route_scale = reward_normalizer(dp_route_reward.ravel())
                dp_route_graph = transition_tilted_graph(
                    graph,
                    node_cells,
                    dp_route_reward,
                    dp_route_scale,
                    gamma=float(args.dp_routing_gamma),
                )
                syn_r, diag_r = select_weighted_graph_paths(
                    bank,
                    road_coords,
                    dp_route_graph,
                    label="dp_tilted_route",
                    mn=mn,
                    span=span,
                    real_signature_prob=real_signature_prob,
                    eval_transition_logprob=eval_logprob,
                    max_public_stretch=float(args.dp_routing_max_public_stretch),
                )
                payload["variants"][f"dp_tilted_route_g{float(args.dp_routing_gamma):.2f}_eps_{eps:.2f}"] = evaluate_variant(
                    f"dp_tilted_route_g{float(args.dp_routing_gamma):.2f}_eps_{eps:.2f}",
                    syn_r,
                    diag_r,
                    eval_real,
                    context,
                    road_coords,
                    graph,
                    mn,
                    span,
                )

                shuffled_route_reward = shuffled_od_reward(dp_route_reward, np.random.default_rng(args.seed + int(eps * 1000) + 381))
                shuffled_route_scale = reward_normalizer(shuffled_route_reward.ravel())
                shuffled_route_graph = transition_tilted_graph(
                    graph,
                    node_cells,
                    shuffled_route_reward,
                    shuffled_route_scale,
                    gamma=float(args.dp_routing_gamma),
                )
                syn_r, diag_r = select_weighted_graph_paths(
                    bank,
                    road_coords,
                    shuffled_route_graph,
                    label="shuffled_dp_tilted_route",
                    mn=mn,
                    span=span,
                    real_signature_prob=real_signature_prob,
                    eval_transition_logprob=eval_logprob,
                    max_public_stretch=float(args.dp_routing_max_public_stretch),
                )
                payload["variants"][f"shuffled_dp_tilted_route_g{float(args.dp_routing_gamma):.2f}_eps_{eps:.2f}"] = evaluate_variant(
                    f"shuffled_dp_tilted_route_g{float(args.dp_routing_gamma):.2f}_eps_{eps:.2f}",
                    syn_r,
                    diag_r,
                    eval_real,
                    context,
                    road_coords,
                    graph,
                    mn,
                    span,
                )

                dp_olp_route_reward = dp_od_length_phase_transition(
                    train,
                    mn,
                    span,
                    np.random.default_rng(args.seed + int(eps * 1000) + 382),
                    eps,
                    6,
                )
                dp_olp_route_scale = reward_normalizer(dp_olp_route_reward.ravel())
                syn_r, diag_r = select_olp_tilted_graph_paths(
                    bank,
                    road_coords,
                    graph,
                    rev_graph,
                    node_cells_g6,
                    dp_olp_route_reward,
                    dp_olp_route_scale,
                    label="dp_olp_tilted_route",
                    mn=mn,
                    span=span,
                    real_signature_prob=real_signature_prob,
                    eval_transition_logprob=eval_logprob,
                    gamma=float(args.dp_routing_gamma),
                    max_public_stretch=float(args.dp_routing_max_public_stretch),
                )
                payload["variants"][f"dp_olp_tilted_route_g{float(args.dp_routing_gamma):.2f}_eps_{eps:.2f}"] = evaluate_variant(
                    f"dp_olp_tilted_route_g{float(args.dp_routing_gamma):.2f}_eps_{eps:.2f}",
                    syn_r,
                    diag_r,
                    eval_real,
                    context,
                    road_coords,
                    graph,
                    mn,
                    span,
                )

                syn_r, diag_r = select_olp_tilted_graph_paths(
                    bank,
                    road_coords,
                    graph,
                    rev_graph,
                    node_cells_g6,
                    dp_olp_route_reward,
                    dp_olp_route_scale,
                    label="dp_olp_tilted_route_gated",
                    mn=mn,
                    span=span,
                    real_signature_prob=real_signature_prob,
                    eval_transition_logprob=eval_logprob,
                    gamma=float(args.dp_routing_gamma),
                    max_public_stretch=float(args.dp_routing_max_public_stretch),
                    accept_bonus_margin=0.05,
                )
                payload["variants"][f"dp_olp_tilted_route_gated_m0p050_g{float(args.dp_routing_gamma):.2f}_eps_{eps:.2f}"] = evaluate_variant(
                    f"dp_olp_tilted_route_gated_m0p050_g{float(args.dp_routing_gamma):.2f}_eps_{eps:.2f}",
                    syn_r,
                    diag_r,
                    eval_real,
                    context,
                    road_coords,
                    graph,
                    mn,
                    span,
                )

                shuffled_olp_route_reward = shuffled_tensor_reward(dp_olp_route_reward, np.random.default_rng(args.seed + int(eps * 1000) + 383))
                shuffled_olp_route_scale = reward_normalizer(shuffled_olp_route_reward.ravel())
                syn_r, diag_r = select_olp_tilted_graph_paths(
                    bank,
                    road_coords,
                    graph,
                    rev_graph,
                    node_cells_g6,
                    shuffled_olp_route_reward,
                    shuffled_olp_route_scale,
                    label="shuffled_dp_olp_tilted_route",
                    mn=mn,
                    span=span,
                    real_signature_prob=real_signature_prob,
                    eval_transition_logprob=eval_logprob,
                    gamma=float(args.dp_routing_gamma),
                    max_public_stretch=float(args.dp_routing_max_public_stretch),
                )
                payload["variants"][f"shuffled_dp_olp_tilted_route_g{float(args.dp_routing_gamma):.2f}_eps_{eps:.2f}"] = evaluate_variant(
                    f"shuffled_dp_olp_tilted_route_g{float(args.dp_routing_gamma):.2f}_eps_{eps:.2f}",
                    syn_r,
                    diag_r,
                    eval_real,
                    context,
                    road_coords,
                    graph,
                    mn,
                    span,
                )

                syn_r, diag_r = select_olp_tilted_graph_paths(
                    bank,
                    road_coords,
                    graph,
                    rev_graph,
                    node_cells_g6,
                    shuffled_olp_route_reward,
                    shuffled_olp_route_scale,
                    label="shuffled_dp_olp_tilted_route_gated",
                    mn=mn,
                    span=span,
                    real_signature_prob=real_signature_prob,
                    eval_transition_logprob=eval_logprob,
                    gamma=float(args.dp_routing_gamma),
                    max_public_stretch=float(args.dp_routing_max_public_stretch),
                    accept_bonus_margin=0.05,
                )
                payload["variants"][f"shuffled_dp_olp_tilted_route_gated_m0p050_g{float(args.dp_routing_gamma):.2f}_eps_{eps:.2f}"] = evaluate_variant(
                    f"shuffled_dp_olp_tilted_route_gated_m0p050_g{float(args.dp_routing_gamma):.2f}_eps_{eps:.2f}",
                    syn_r,
                    diag_r,
                    eval_real,
                    context,
                    road_coords,
                    graph,
                    mn,
                    span,
                )

            anchor_shape_reward = dp_anchor_shape_reward(
                train,
                bank,
                road_coords,
                mn,
                span,
                np.random.default_rng(args.seed + int(eps * 1000) + 484),
                eps,
            )
            run_focused_variant(
                f"anchor_shape_eps_{eps:.2f}",
                "anchor_shape_reward",
                anchor_shape_reward,
                (0.0, 1.0),
                485,
                eps,
            )
            for margin_value, margin_tag, margin_offset in [(0.05, "0p050", 486), (0.10, "0p100", 487), (0.15, "0p150", 488), (0.20, "0p200", 489)]:
                run_focused_variant(
                    f"anchor_shape_margin_m{margin_tag}_eps_{eps:.2f}",
                    "anchor_shape_margin_reward",
                    anchor_shape_reward,
                    (0.0, 1.0),
                    margin_offset,
                    eps,
                    switch_margin_value=margin_value,
                )
            shuffled_anchor_shape = shuffled_anchor_shape_reward(anchor_shape_reward, np.random.default_rng(args.seed + int(eps * 1000) + 490))
            run_focused_variant(
                f"shuffled_anchor_shape_eps_{eps:.2f}",
                "shuffled_anchor_shape_reward",
                shuffled_anchor_shape,
                (0.0, 1.0),
                491,
                eps,
            )
            for margin_value, margin_tag, margin_offset in [(0.05, "0p050", 492), (0.10, "0p100", 493), (0.15, "0p150", 494), (0.20, "0p200", 495)]:
                run_focused_variant(
                    f"shuffled_anchor_shape_margin_m{margin_tag}_eps_{eps:.2f}",
                    "shuffled_anchor_shape_margin_reward",
                    shuffled_anchor_shape,
                    (0.0, 1.0),
                    margin_offset,
                    eps,
                    switch_margin_value=margin_value,
                )

            anchor_choice_reward = dp_anchor_choice_reward(
                train,
                bank,
                road_coords,
                mn,
                span,
                np.random.default_rng(args.seed + int(eps * 1000) + 472),
                eps,
            )
            run_focused_variant(
                f"anchor_choice_eps_{eps:.2f}",
                "anchor_choice_reward",
                anchor_choice_reward,
                (0.0, 1.0),
                473,
                eps,
            )
            for margin_value, margin_tag, margin_offset in [(0.05, "0p050", 474), (0.10, "0p100", 475), (0.15, "0p150", 476), (0.20, "0p200", 477)]:
                run_focused_variant(
                    f"anchor_choice_margin_m{margin_tag}_eps_{eps:.2f}",
                    "anchor_choice_margin_reward",
                    anchor_choice_reward,
                    (0.0, 1.0),
                    margin_offset,
                    eps,
                    switch_margin_value=margin_value,
                )
            shuffled_anchor_choice = shuffled_anchor_choice_reward(anchor_choice_reward, np.random.default_rng(args.seed + int(eps * 1000) + 478))
            run_focused_variant(
                f"shuffled_anchor_choice_eps_{eps:.2f}",
                "shuffled_anchor_choice_reward",
                shuffled_anchor_choice,
                (0.0, 1.0),
                479,
                eps,
            )
            for margin_value, margin_tag, margin_offset in [(0.05, "0p050", 480), (0.10, "0p100", 481), (0.15, "0p150", 482), (0.20, "0p200", 483)]:
                run_focused_variant(
                    f"shuffled_anchor_choice_margin_m{margin_tag}_eps_{eps:.2f}",
                    "shuffled_anchor_choice_margin_reward",
                    shuffled_anchor_choice,
                    (0.0, 1.0),
                    margin_offset,
                    eps,
                    switch_margin_value=margin_value,
                )

            if args.focus_anchor_choice:
                continue

            cluster_corridor_reward = dp_cluster_corridor_residual_reward(
                train,
                mn,
                span,
                np.random.default_rng(args.seed + int(eps * 1000) + 298),
                eps,
                TRANS_GRID,
                cluster_corridor_info,
            )
            cluster_corridor_scale_val = cluster_corridor_residual_scale(bank, cluster_corridor_reward)
            run_focused_variant(
                f"cluster_corridor_eps_{eps:.2f}",
                "cluster_corridor_reward",
                cluster_corridor_reward,
                cluster_corridor_scale_val,
                299,
                eps,
            )
            run_focused_variant(
                f"cluster_corridor_margin_m0p050_eps_{eps:.2f}",
                "cluster_corridor_margin_reward",
                cluster_corridor_reward,
                cluster_corridor_scale_val,
                300,
                eps,
                switch_margin_value=0.05,
            )
            shuffled_cluster_corridor = shuffled_cluster_corridor_residual_reward(cluster_corridor_reward, np.random.default_rng(args.seed + int(eps * 1000) + 301))
            shuffled_cluster_corridor_scale = cluster_corridor_residual_scale(bank, shuffled_cluster_corridor)
            run_focused_variant(
                f"shuffled_cluster_corridor_eps_{eps:.2f}",
                "shuffled_cluster_corridor_reward",
                shuffled_cluster_corridor,
                shuffled_cluster_corridor_scale,
                302,
                eps,
            )
            run_focused_variant(
                f"shuffled_cluster_corridor_margin_m0p050_eps_{eps:.2f}",
                "shuffled_cluster_corridor_margin_reward",
                shuffled_cluster_corridor,
                shuffled_cluster_corridor_scale,
                303,
                eps,
                switch_margin_value=0.05,
            )

            cluster_vote_reward = dp_cluster_vote_reward(
                train,
                mn,
                span,
                np.random.default_rng(args.seed + int(eps * 1000) + 360),
                eps,
                TRANS_GRID,
                cluster_corridor_info,
            )
            run_focused_variant(
                f"cluster_vote_eps_{eps:.2f}",
                "cluster_vote_reward",
                cluster_vote_reward,
                (0.0, 1.0),
                361,
                eps,
            )
            run_focused_variant(
                f"cluster_vote_margin_m0p050_eps_{eps:.2f}",
                "cluster_vote_margin_reward",
                cluster_vote_reward,
                (0.0, 1.0),
                362,
                eps,
                switch_margin_value=0.05,
            )
            shuffled_cluster_vote = shuffled_cluster_vote_reward(cluster_vote_reward, np.random.default_rng(args.seed + int(eps * 1000) + 363))
            run_focused_variant(
                f"shuffled_cluster_vote_eps_{eps:.2f}",
                "shuffled_cluster_vote_reward",
                shuffled_cluster_vote,
                (0.0, 1.0),
                364,
                eps,
            )
            run_focused_variant(
                f"shuffled_cluster_vote_margin_m0p050_eps_{eps:.2f}",
                "shuffled_cluster_vote_margin_reward",
                shuffled_cluster_vote,
                (0.0, 1.0),
                365,
                eps,
                switch_margin_value=0.05,
            )

            od_cluster_vote_reward = dp_od_cluster_vote_reward(
                train,
                bank,
                mn,
                span,
                np.random.default_rng(args.seed + int(eps * 1000) + 366),
                eps,
                TRANS_GRID,
                cluster_corridor_info,
            )
            run_focused_variant(
                f"od_cluster_vote_eps_{eps:.2f}",
                "od_cluster_vote_reward",
                od_cluster_vote_reward,
                (0.0, 1.0),
                367,
                eps,
            )
            run_focused_variant(
                f"od_cluster_vote_margin_m0p050_eps_{eps:.2f}",
                "od_cluster_vote_margin_reward",
                od_cluster_vote_reward,
                (0.0, 1.0),
                368,
                eps,
                switch_margin_value=0.05,
            )
            shuffled_od_cluster_vote = shuffled_od_cluster_vote_reward(od_cluster_vote_reward, np.random.default_rng(args.seed + int(eps * 1000) + 369))
            run_focused_variant(
                f"shuffled_od_cluster_vote_eps_{eps:.2f}",
                "shuffled_od_cluster_vote_reward",
                shuffled_od_cluster_vote,
                (0.0, 1.0),
                370,
                eps,
            )
            run_focused_variant(
                f"shuffled_od_cluster_vote_margin_m0p050_eps_{eps:.2f}",
                "shuffled_od_cluster_vote_margin_reward",
                shuffled_od_cluster_vote,
                (0.0, 1.0),
                371,
                eps,
                switch_margin_value=0.05,
            )

            odcell_cluster_vote_reward = dp_odcell_cluster_vote_reward(
                train,
                bank,
                mn,
                span,
                np.random.default_rng(args.seed + int(eps * 1000) + 372),
                eps,
                TRANS_GRID,
                cluster_corridor_info,
            )
            run_focused_variant(
                f"odcell_cluster_vote_eps_{eps:.2f}",
                "odcell_cluster_vote_reward",
                odcell_cluster_vote_reward,
                (0.0, 1.0),
                373,
                eps,
            )
            run_focused_variant(
                f"odcell_cluster_vote_margin_m0p050_eps_{eps:.2f}",
                "odcell_cluster_vote_margin_reward",
                odcell_cluster_vote_reward,
                (0.0, 1.0),
                374,
                eps,
                switch_margin_value=0.05,
            )
            for margin_value, margin_tag, margin_offset in [(0.10, "0p100", 378), (0.15, "0p150", 379), (0.20, "0p200", 380)]:
                run_focused_variant(
                    f"odcell_cluster_vote_margin_m{margin_tag}_eps_{eps:.2f}",
                    "odcell_cluster_vote_margin_reward",
                    odcell_cluster_vote_reward,
                    (0.0, 1.0),
                    margin_offset,
                    eps,
                    switch_margin_value=margin_value,
                )
            shuffled_odcell_cluster_vote = shuffled_odcell_cluster_vote_reward(odcell_cluster_vote_reward, np.random.default_rng(args.seed + int(eps * 1000) + 375))
            run_focused_variant(
                f"shuffled_odcell_cluster_vote_eps_{eps:.2f}",
                "shuffled_odcell_cluster_vote_reward",
                shuffled_odcell_cluster_vote,
                (0.0, 1.0),
                376,
                eps,
            )
            run_focused_variant(
                f"shuffled_odcell_cluster_vote_margin_m0p050_eps_{eps:.2f}",
                "shuffled_odcell_cluster_vote_margin_reward",
                shuffled_odcell_cluster_vote,
                (0.0, 1.0),
                377,
                eps,
                switch_margin_value=0.05,
            )
            for margin_value, margin_tag, margin_offset in [(0.10, "0p100", 381), (0.15, "0p150", 382), (0.20, "0p200", 383)]:
                run_focused_variant(
                    f"shuffled_odcell_cluster_vote_margin_m{margin_tag}_eps_{eps:.2f}",
                    "shuffled_odcell_cluster_vote_margin_reward",
                    shuffled_odcell_cluster_vote,
                    (0.0, 1.0),
                    margin_offset,
                    eps,
                    switch_margin_value=margin_value,
                )

            for prior_weight, prior_tag, prior_offset in [(10.0, "p10", 400), (20.0, "p20", 430)]:
                odcell_prior_reward = dp_odcell_cluster_vote_reward(
                    train,
                    bank,
                    mn,
                    span,
                    np.random.default_rng(args.seed + int(eps * 1000) + prior_offset),
                    eps,
                    TRANS_GRID,
                    cluster_corridor_info,
                    public_prior_weight=prior_weight,
                )
                run_focused_variant(
                    f"odcell_cluster_vote_prior_{prior_tag}_eps_{eps:.2f}",
                    "odcell_cluster_vote_reward",
                    odcell_prior_reward,
                    (0.0, 1.0),
                    prior_offset + 1,
                    eps,
                )
                for margin_value, margin_tag, margin_offset in [(0.05, "0p050", 2), (0.10, "0p100", 3), (0.20, "0p200", 4)]:
                    run_focused_variant(
                        f"odcell_cluster_vote_prior_{prior_tag}_margin_m{margin_tag}_eps_{eps:.2f}",
                        "odcell_cluster_vote_margin_reward",
                        odcell_prior_reward,
                        (0.0, 1.0),
                        prior_offset + margin_offset,
                        eps,
                        switch_margin_value=margin_value,
                    )
                shuffled_odcell_prior = shuffled_odcell_cluster_vote_reward(odcell_prior_reward, np.random.default_rng(args.seed + int(eps * 1000) + prior_offset + 5))
                run_focused_variant(
                    f"shuffled_odcell_cluster_vote_prior_{prior_tag}_eps_{eps:.2f}",
                    "shuffled_odcell_cluster_vote_reward",
                    shuffled_odcell_prior,
                    (0.0, 1.0),
                    prior_offset + 6,
                    eps,
                )
                for margin_value, margin_tag, margin_offset in [(0.05, "0p050", 7), (0.10, "0p100", 8), (0.20, "0p200", 9)]:
                    run_focused_variant(
                        f"shuffled_odcell_cluster_vote_prior_{prior_tag}_margin_m{margin_tag}_eps_{eps:.2f}",
                        "shuffled_odcell_cluster_vote_margin_reward",
                        shuffled_odcell_prior,
                        (0.0, 1.0),
                        prior_offset + margin_offset,
                        eps,
                        switch_margin_value=margin_value,
                    )

            hier_cluster_vote_reward = dp_hier_cluster_vote_reward(
                train,
                bank,
                mn,
                span,
                np.random.default_rng(args.seed + int(eps * 1000) + 384),
                eps,
                TRANS_GRID,
                cluster_corridor_info,
            )
            run_focused_variant(
                f"hier_cluster_vote_eps_{eps:.2f}",
                "hier_cluster_vote_reward",
                hier_cluster_vote_reward,
                (0.0, 1.0),
                385,
                eps,
            )
            for margin_value, margin_tag, margin_offset in [(0.05, "0p050", 386), (0.10, "0p100", 387), (0.15, "0p150", 388), (0.20, "0p200", 389)]:
                run_focused_variant(
                    f"hier_cluster_vote_margin_m{margin_tag}_eps_{eps:.2f}",
                    "hier_cluster_vote_margin_reward",
                    hier_cluster_vote_reward,
                    (0.0, 1.0),
                    margin_offset,
                    eps,
                    switch_margin_value=margin_value,
                )
            shuffled_hier_cluster_vote = shuffled_hier_cluster_vote_reward(hier_cluster_vote_reward, np.random.default_rng(args.seed + int(eps * 1000) + 390))
            run_focused_variant(
                f"shuffled_hier_cluster_vote_eps_{eps:.2f}",
                "shuffled_hier_cluster_vote_reward",
                shuffled_hier_cluster_vote,
                (0.0, 1.0),
                391,
                eps,
            )
            for margin_value, margin_tag, margin_offset in [(0.05, "0p050", 392), (0.10, "0p100", 393), (0.15, "0p150", 394), (0.20, "0p200", 395)]:
                run_focused_variant(
                    f"shuffled_hier_cluster_vote_margin_m{margin_tag}_eps_{eps:.2f}",
                    "shuffled_hier_cluster_vote_margin_reward",
                    shuffled_hier_cluster_vote,
                    (0.0, 1.0),
                    margin_offset,
                    eps,
                    switch_margin_value=margin_value,
                )

            split_eps = max(float(eps) / 2.0, 0.0)
            split_cluster_reward = dp_cluster_corridor_residual_reward(
                train,
                mn,
                span,
                np.random.default_rng(args.seed + int(eps * 1000) + 440),
                split_eps,
                TRANS_GRID,
                cluster_corridor_info,
            )
            split_cluster_scale = cluster_corridor_residual_scale(bank, split_cluster_reward)
            split_hier_reward = dp_hier_cluster_vote_reward(
                train,
                bank,
                mn,
                span,
                np.random.default_rng(args.seed + int(eps * 1000) + 441),
                split_eps,
                TRANS_GRID,
                cluster_corridor_info,
            )
            split_consensus_reward = {
                **split_cluster_reward,
                **split_hier_reward,
                "cluster_hier_consensus": np.asarray([1.0], dtype=float),
                "cluster_hier_consensus_total_eps": np.asarray([float(eps)], dtype=float),
                "cluster_hier_consensus_split_eps": np.asarray([float(split_eps)], dtype=float),
            }
            run_focused_variant(
                f"cluster_hier_consensus_split_eps_{eps:.2f}",
                "cluster_hier_consensus_reward",
                split_consensus_reward,
                split_cluster_scale,
                442,
                eps,
            )
            for margin_value, margin_tag, margin_offset in [(0.05, "0p050", 443), (0.10, "0p100", 444), (0.15, "0p150", 445), (0.20, "0p200", 446)]:
                run_focused_variant(
                    f"cluster_hier_consensus_split_margin_m{margin_tag}_eps_{eps:.2f}",
                    "cluster_hier_consensus_margin_reward",
                    split_consensus_reward,
                    split_cluster_scale,
                    margin_offset,
                    eps,
                    switch_margin_value=margin_value,
                )
            shuffled_split_cluster = shuffled_cluster_corridor_residual_reward(split_cluster_reward, np.random.default_rng(args.seed + int(eps * 1000) + 447))
            shuffled_split_hier = shuffled_hier_cluster_vote_reward(split_hier_reward, np.random.default_rng(args.seed + int(eps * 1000) + 448))
            shuffled_split_consensus = {
                **shuffled_split_cluster,
                **shuffled_split_hier,
                "cluster_hier_consensus": np.asarray([1.0], dtype=float),
                "cluster_hier_consensus_total_eps": np.asarray([float(eps)], dtype=float),
                "cluster_hier_consensus_split_eps": np.asarray([float(split_eps)], dtype=float),
            }
            shuffled_split_scale = cluster_corridor_residual_scale(bank, shuffled_split_cluster)
            run_focused_variant(
                f"shuffled_cluster_hier_consensus_split_eps_{eps:.2f}",
                "shuffled_cluster_hier_consensus_reward",
                shuffled_split_consensus,
                shuffled_split_scale,
                449,
                eps,
            )
            for margin_value, margin_tag, margin_offset in [(0.05, "0p050", 450), (0.10, "0p100", 451), (0.15, "0p150", 452), (0.20, "0p200", 453)]:
                run_focused_variant(
                    f"shuffled_cluster_hier_consensus_split_margin_m{margin_tag}_eps_{eps:.2f}",
                    "shuffled_cluster_hier_consensus_margin_reward",
                    shuffled_split_consensus,
                    shuffled_split_scale,
                    margin_offset,
                    eps,
                    switch_margin_value=margin_value,
                )

            signature_vote_reward = dp_signature_vote_reward(
                train,
                bank,
                road_coords,
                mn,
                span,
                np.random.default_rng(args.seed + int(eps * 1000) + 460),
                eps,
                grid=6,
                max_len=5,
            )
            run_focused_variant(
                f"signature_vote_eps_{eps:.2f}",
                "signature_vote_reward",
                signature_vote_reward,
                (0.0, 1.0),
                461,
                eps,
            )
            for margin_value, margin_tag, margin_offset in [(0.05, "0p050", 462), (0.10, "0p100", 463), (0.15, "0p150", 464), (0.20, "0p200", 465)]:
                run_focused_variant(
                    f"signature_vote_margin_m{margin_tag}_eps_{eps:.2f}",
                    "signature_vote_margin_reward",
                    signature_vote_reward,
                    (0.0, 1.0),
                    margin_offset,
                    eps,
                    switch_margin_value=margin_value,
                )
            shuffled_signature_vote = shuffled_signature_vote_reward(signature_vote_reward, np.random.default_rng(args.seed + int(eps * 1000) + 466))
            run_focused_variant(
                f"shuffled_signature_vote_eps_{eps:.2f}",
                "shuffled_signature_vote_reward",
                shuffled_signature_vote,
                (0.0, 1.0),
                467,
                eps,
            )
            for margin_value, margin_tag, margin_offset in [(0.05, "0p050", 468), (0.10, "0p100", 469), (0.15, "0p150", 470), (0.20, "0p200", 471)]:
                run_focused_variant(
                    f"shuffled_signature_vote_margin_m{margin_tag}_eps_{eps:.2f}",
                    "shuffled_signature_vote_margin_reward",
                    shuffled_signature_vote,
                    (0.0, 1.0),
                    margin_offset,
                    eps,
                    switch_margin_value=margin_value,
                )

            cycle_reward = dp_cycle_motif_reward(
                train,
                mn,
                span,
                np.random.default_rng(args.seed + int(eps * 1000) + 250),
                eps,
                TRANS_GRID,
                cycle_motif_info,
            )
            cycle_scale = cycle_motif_scale(cycle_reward)
            run_focused_variant(
                f"cycle_motif_eps_{eps:.2f}",
                "cycle_motif_reward",
                cycle_reward,
                cycle_scale,
                251,
                eps,
            )
            run_focused_variant(
                f"cycle_motif_margin_m0p050_eps_{eps:.2f}",
                "cycle_motif_margin_reward",
                cycle_reward,
                cycle_scale,
                252,
                eps,
                switch_margin_value=0.05,
            )
            shuffled_cycle = shuffled_cycle_motif_reward(cycle_reward, np.random.default_rng(args.seed + int(eps * 1000) + 253))
            shuffled_cycle_scale = cycle_motif_scale(shuffled_cycle)
            run_focused_variant(
                f"shuffled_cycle_motif_eps_{eps:.2f}",
                "shuffled_cycle_motif_reward",
                shuffled_cycle,
                shuffled_cycle_scale,
                254,
                eps,
            )
            run_focused_variant(
                f"shuffled_cycle_motif_margin_m0p050_eps_{eps:.2f}",
                "shuffled_cycle_motif_margin_reward",
                shuffled_cycle,
                shuffled_cycle_scale,
                255,
                eps,
                switch_margin_value=0.05,
            )

            family_cut_band_reward = dp_family_cut_band_corridor_reward(
                train,
                mn,
                span,
                np.random.default_rng(args.seed + int(eps * 1000) + 328),
                eps,
                TRANS_GRID,
                cut_band_corridor_info,
                cut_band_family_info,
            )
            family_cut_band_scale_val = family_cut_band_scale(bank, family_cut_band_reward)
            run_focused_variant(
                f"family_cut_band_eps_{eps:.2f}",
                "family_cut_band_reward",
                family_cut_band_reward,
                family_cut_band_scale_val,
                329,
                eps,
            )
            run_focused_variant(
                f"family_cut_band_margin_m0p050_eps_{eps:.2f}",
                "family_cut_band_margin_reward",
                family_cut_band_reward,
                family_cut_band_scale_val,
                330,
                eps,
                switch_margin_value=0.05,
            )
            shuffled_family_cut_band = shuffled_family_cut_band_corridor_reward(family_cut_band_reward, np.random.default_rng(args.seed + int(eps * 1000) + 331))
            shuffled_family_cut_band_scale = family_cut_band_scale(bank, shuffled_family_cut_band)
            run_focused_variant(
                f"shuffled_family_cut_band_eps_{eps:.2f}",
                "shuffled_family_cut_band_reward",
                shuffled_family_cut_band,
                shuffled_family_cut_band_scale,
                332,
                eps,
            )
            run_focused_variant(
                f"shuffled_family_cut_band_margin_m0p050_eps_{eps:.2f}",
                "shuffled_family_cut_band_margin_reward",
                shuffled_family_cut_band,
                shuffled_family_cut_band_scale,
                333,
                eps,
                switch_margin_value=0.05,
            )

        args.eps_list = ""

    margin_sweep = [float(x) for x in args.margin_list.split(",") if x.strip()]
    lcb_sweep = [float(x) for x in args.lcb_list.split(",") if x.strip()]
    for eps in [float(x) for x in args.eps_list.split(",") if x.strip()]:
        if eps <= 0:
            continue
        reward_rng = np.random.default_rng(args.seed + int(eps * 1000) + 101)
        select_rng = np.random.default_rng(args.seed + int(eps * 1000) + 202)
        global_reward = dp_global_transition(train, mn, span, reward_rng, eps, TRANS_GRID)
        glo_scale = reward_normalizer(global_reward.ravel())
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            select_rng,
            mode="global_reward",
            reward=global_reward,
            lo_scale=glo_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"global_eps_{eps:.2f}"] = evaluate_variant(f"global_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        reward_rng = np.random.default_rng(args.seed + int(eps * 1000) + 231)
        graph_flow_reward = dp_graph_flow_reward(train, mn, span, reward_rng, eps, TRANS_GRID, public_flow)
        graph_flow_scale = reward_normalizer(graph_flow_reward["graph_flow"].ravel())
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 232),
            mode="graph_flow_reward",
            reward=graph_flow_reward,
            lo_scale=graph_flow_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"graph_flow_eps_{eps:.2f}"] = evaluate_variant(f"graph_flow_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 233),
            mode="graph_flow_margin_reward",
            reward=graph_flow_reward,
            lo_scale=graph_flow_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"graph_flow_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"graph_flow_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 237),
            mode="graph_flow_llr_reward",
            reward=graph_flow_reward,
            lo_scale=graph_flow_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"graph_flow_llr_eps_{eps:.2f}"] = evaluate_variant(f"graph_flow_llr_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 238),
            mode="graph_flow_llr_margin_reward",
            reward=graph_flow_reward,
            lo_scale=graph_flow_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"graph_flow_llr_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"graph_flow_llr_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        local_corridor_reward = dp_local_corridor_residual_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 292),
            eps,
            TRANS_GRID,
            local_corridor_info,
        )
        local_corridor_scale = local_corridor_residual_scale(bank, local_corridor_reward)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 293),
            mode="local_corridor_reward",
            reward=local_corridor_reward,
            lo_scale=local_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"local_corridor_eps_{eps:.2f}"] = evaluate_variant(f"local_corridor_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 294),
            mode="local_corridor_margin_reward",
            reward=local_corridor_reward,
            lo_scale=local_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"local_corridor_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"local_corridor_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_local_corridor = shuffled_local_corridor_residual_reward(local_corridor_reward, np.random.default_rng(args.seed + int(eps * 1000) + 295))
        shuffled_local_corridor_scale = local_corridor_residual_scale(bank, shuffled_local_corridor)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 296),
            mode="shuffled_local_corridor_reward",
            reward=shuffled_local_corridor,
            lo_scale=shuffled_local_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_local_corridor_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_local_corridor_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 297),
            mode="shuffled_local_corridor_margin_reward",
            reward=shuffled_local_corridor,
            lo_scale=shuffled_local_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_local_corridor_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_local_corridor_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        cluster_corridor_reward = dp_cluster_corridor_residual_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 298),
            eps,
            TRANS_GRID,
            cluster_corridor_info,
        )
        cluster_corridor_scale = cluster_corridor_residual_scale(bank, cluster_corridor_reward)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 299),
            mode="cluster_corridor_reward",
            reward=cluster_corridor_reward,
            lo_scale=cluster_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"cluster_corridor_eps_{eps:.2f}"] = evaluate_variant(f"cluster_corridor_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 300),
            mode="cluster_corridor_margin_reward",
            reward=cluster_corridor_reward,
            lo_scale=cluster_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"cluster_corridor_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"cluster_corridor_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_cluster_corridor = shuffled_cluster_corridor_residual_reward(cluster_corridor_reward, np.random.default_rng(args.seed + int(eps * 1000) + 301))
        shuffled_cluster_corridor_scale = cluster_corridor_residual_scale(bank, shuffled_cluster_corridor)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 302),
            mode="shuffled_cluster_corridor_reward",
            reward=shuffled_cluster_corridor,
            lo_scale=shuffled_cluster_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_cluster_corridor_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_cluster_corridor_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 303),
            mode="shuffled_cluster_corridor_margin_reward",
            reward=shuffled_cluster_corridor,
            lo_scale=shuffled_cluster_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_cluster_corridor_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_cluster_corridor_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        cut_corridor_reward = dp_cut_corridor_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 304),
            eps,
            TRANS_GRID,
            cut_corridor_info,
        )
        cut_corridor_scale_val = cut_corridor_scale(bank, cut_corridor_reward)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 305),
            mode="cut_corridor_reward",
            reward=cut_corridor_reward,
            lo_scale=cut_corridor_scale_val,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"cut_corridor_eps_{eps:.2f}"] = evaluate_variant(f"cut_corridor_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 306),
            mode="cut_corridor_margin_reward",
            reward=cut_corridor_reward,
            lo_scale=cut_corridor_scale_val,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"cut_corridor_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"cut_corridor_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_cut_corridor = shuffled_cut_corridor_reward(cut_corridor_reward, np.random.default_rng(args.seed + int(eps * 1000) + 307))
        shuffled_cut_corridor_scale = cut_corridor_scale(bank, shuffled_cut_corridor)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 308),
            mode="shuffled_cut_corridor_reward",
            reward=shuffled_cut_corridor,
            lo_scale=shuffled_cut_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_cut_corridor_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_cut_corridor_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 309),
            mode="shuffled_cut_corridor_margin_reward",
            reward=shuffled_cut_corridor,
            lo_scale=shuffled_cut_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_cut_corridor_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_cut_corridor_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        cut_band_corridor_reward = dp_cut_band_corridor_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 310),
            eps,
            TRANS_GRID,
            cut_band_corridor_info,
        )
        cut_band_corridor_scale_val = cut_band_corridor_scale(bank, cut_band_corridor_reward)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 311),
            mode="cut_band_corridor_reward",
            reward=cut_band_corridor_reward,
            lo_scale=cut_band_corridor_scale_val,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"cut_band_corridor_eps_{eps:.2f}"] = evaluate_variant(f"cut_band_corridor_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 312),
            mode="cut_band_corridor_margin_reward",
            reward=cut_band_corridor_reward,
            lo_scale=cut_band_corridor_scale_val,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"cut_band_corridor_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"cut_band_corridor_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_cut_band_corridor = shuffled_cut_band_corridor_reward(cut_band_corridor_reward, np.random.default_rng(args.seed + int(eps * 1000) + 313))
        shuffled_cut_band_corridor_scale = cut_band_corridor_scale(bank, shuffled_cut_band_corridor)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 314),
            mode="shuffled_cut_band_corridor_reward",
            reward=shuffled_cut_band_corridor,
            lo_scale=shuffled_cut_band_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_cut_band_corridor_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_cut_band_corridor_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 315),
            mode="shuffled_cut_band_corridor_margin_reward",
            reward=shuffled_cut_band_corridor,
            lo_scale=shuffled_cut_band_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_cut_band_corridor_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_cut_band_corridor_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        od_cut_band_reward = dp_od_cut_band_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 316),
            eps,
            TRANS_GRID,
            od_cut_band_info,
        )
        od_cut_band_scale_val = od_cut_band_scale(bank, od_cut_band_reward)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 317),
            mode="od_cut_band_reward",
            reward=od_cut_band_reward,
            lo_scale=od_cut_band_scale_val,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"od_cut_band_eps_{eps:.2f}"] = evaluate_variant(f"od_cut_band_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 318),
            mode="od_cut_band_margin_reward",
            reward=od_cut_band_reward,
            lo_scale=od_cut_band_scale_val,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"od_cut_band_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"od_cut_band_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_od_cut_band = shuffled_od_cut_band_reward(od_cut_band_reward, np.random.default_rng(args.seed + int(eps * 1000) + 319))
        shuffled_od_cut_band_scale = od_cut_band_scale(bank, shuffled_od_cut_band)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 320),
            mode="shuffled_od_cut_band_reward",
            reward=shuffled_od_cut_band,
            lo_scale=shuffled_od_cut_band_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_od_cut_band_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_od_cut_band_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 321),
            mode="shuffled_od_cut_band_margin_reward",
            reward=shuffled_od_cut_band,
            lo_scale=shuffled_od_cut_band_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_od_cut_band_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_od_cut_band_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        global_cut_band_reward = dp_global_cut_band_corridor_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 322),
            eps,
            TRANS_GRID,
            cut_band_corridor_info,
        )
        global_cut_band_scale_val = global_cut_band_scale(bank, global_cut_band_reward)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 323),
            mode="global_cut_band_reward",
            reward=global_cut_band_reward,
            lo_scale=global_cut_band_scale_val,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"global_cut_band_eps_{eps:.2f}"] = evaluate_variant(f"global_cut_band_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 324),
            mode="global_cut_band_margin_reward",
            reward=global_cut_band_reward,
            lo_scale=global_cut_band_scale_val,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"global_cut_band_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"global_cut_band_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_global_cut_band = shuffled_global_cut_band_corridor_reward(global_cut_band_reward, np.random.default_rng(args.seed + int(eps * 1000) + 325))
        shuffled_global_cut_band_scale = global_cut_band_scale(bank, shuffled_global_cut_band)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 326),
            mode="shuffled_global_cut_band_reward",
            reward=shuffled_global_cut_band,
            lo_scale=shuffled_global_cut_band_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_global_cut_band_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_global_cut_band_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 327),
            mode="shuffled_global_cut_band_margin_reward",
            reward=shuffled_global_cut_band,
            lo_scale=shuffled_global_cut_band_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_global_cut_band_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_global_cut_band_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        family_cut_band_reward = dp_family_cut_band_corridor_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 328),
            eps,
            TRANS_GRID,
            cut_band_corridor_info,
            cut_band_family_info,
        )
        family_cut_band_scale_val = family_cut_band_scale(bank, family_cut_band_reward)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 329),
            mode="family_cut_band_reward",
            reward=family_cut_band_reward,
            lo_scale=family_cut_band_scale_val,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"family_cut_band_eps_{eps:.2f}"] = evaluate_variant(f"family_cut_band_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 330),
            mode="family_cut_band_margin_reward",
            reward=family_cut_band_reward,
            lo_scale=family_cut_band_scale_val,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"family_cut_band_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"family_cut_band_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_family_cut_band = shuffled_family_cut_band_corridor_reward(family_cut_band_reward, np.random.default_rng(args.seed + int(eps * 1000) + 331))
        shuffled_family_cut_band_scale = family_cut_band_scale(bank, shuffled_family_cut_band)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 332),
            mode="shuffled_family_cut_band_reward",
            reward=shuffled_family_cut_band,
            lo_scale=shuffled_family_cut_band_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_family_cut_band_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_family_cut_band_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 333),
            mode="shuffled_family_cut_band_margin_reward",
            reward=shuffled_family_cut_band,
            lo_scale=shuffled_family_cut_band_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_family_cut_band_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_family_cut_band_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        half_eps = eps / 2.0
        consensus_family = dp_family_cut_band_corridor_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 334),
            half_eps,
            TRANS_GRID,
            cut_band_corridor_info,
            cut_band_family_info,
        )
        consensus_global = dp_global_cut_band_corridor_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 335),
            half_eps,
            TRANS_GRID,
            cut_band_corridor_info,
        )
        consensus_family_scale = family_cut_band_scale(bank, consensus_family)
        consensus_global_scale = global_cut_band_scale(bank, consensus_global)
        consensus_reward = make_cut_band_consensus_reward(
            consensus_family,
            consensus_global,
            consensus_family_scale,
            consensus_global_scale,
        )
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 336),
            mode="cut_band_consensus_reward",
            reward=consensus_reward,
            lo_scale=(0.0, 1.0),
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"cut_band_consensus_split_eps_{eps:.2f}"] = evaluate_variant(f"cut_band_consensus_split_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 337),
            mode="cut_band_consensus_margin_reward",
            reward=consensus_reward,
            lo_scale=(0.0, 1.0),
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"cut_band_consensus_split_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"cut_band_consensus_split_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_consensus_family = shuffled_family_cut_band_corridor_reward(consensus_family, np.random.default_rng(args.seed + int(eps * 1000) + 338))
        shuffled_consensus_global = shuffled_global_cut_band_corridor_reward(consensus_global, np.random.default_rng(args.seed + int(eps * 1000) + 339))
        shuffled_consensus_reward = make_cut_band_consensus_reward(
            shuffled_consensus_family,
            shuffled_consensus_global,
            family_cut_band_scale(bank, shuffled_consensus_family),
            global_cut_band_scale(bank, shuffled_consensus_global),
        )
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 340),
            mode="shuffled_cut_band_consensus_reward",
            reward=shuffled_consensus_reward,
            lo_scale=(0.0, 1.0),
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_cut_band_consensus_split_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_cut_band_consensus_split_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 341),
            mode="shuffled_cut_band_consensus_margin_reward",
            reward=shuffled_consensus_reward,
            lo_scale=(0.0, 1.0),
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_cut_band_consensus_split_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_cut_band_consensus_split_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        electrical_reward = dp_electrical_cycle_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 280),
            eps,
            TRANS_GRID,
            electrical_cycle_info,
        )
        electrical_scale = electrical_cycle_scale(bank, electrical_reward)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 281),
            mode="electrical_cycle_reward",
            reward=electrical_reward,
            lo_scale=electrical_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"electrical_cycle_eps_{eps:.2f}"] = evaluate_variant(f"electrical_cycle_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 282),
            mode="electrical_cycle_margin_reward",
            reward=electrical_reward,
            lo_scale=electrical_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"electrical_cycle_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"electrical_cycle_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_electrical = shuffled_electrical_cycle_reward(electrical_reward, np.random.default_rng(args.seed + int(eps * 1000) + 283))
        shuffled_electrical_scale = electrical_cycle_scale(bank, shuffled_electrical)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 284),
            mode="shuffled_electrical_cycle_reward",
            reward=shuffled_electrical,
            lo_scale=shuffled_electrical_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_electrical_cycle_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_electrical_cycle_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 285),
            mode="shuffled_electrical_cycle_margin_reward",
            reward=shuffled_electrical,
            lo_scale=shuffled_electrical_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_electrical_cycle_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_electrical_cycle_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        electrical_energy_reward = dp_electrical_cycle_energy_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 286),
            eps,
            TRANS_GRID,
            electrical_cycle_info,
        )
        electrical_energy_scale = electrical_cycle_scale(bank, electrical_energy_reward)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 287),
            mode="electrical_energy_reward",
            reward=electrical_energy_reward,
            lo_scale=electrical_energy_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"electrical_energy_eps_{eps:.2f}"] = evaluate_variant(f"electrical_energy_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 288),
            mode="electrical_energy_margin_reward",
            reward=electrical_energy_reward,
            lo_scale=electrical_energy_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"electrical_energy_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"electrical_energy_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_electrical_energy = shuffled_electrical_cycle_reward(electrical_energy_reward, np.random.default_rng(args.seed + int(eps * 1000) + 289))
        shuffled_electrical_energy_scale = electrical_cycle_scale(bank, shuffled_electrical_energy)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 290),
            mode="shuffled_electrical_energy_reward",
            reward=shuffled_electrical_energy,
            lo_scale=shuffled_electrical_energy_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_electrical_energy_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_electrical_energy_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 291),
            mode="shuffled_electrical_energy_margin_reward",
            reward=shuffled_electrical_energy,
            lo_scale=shuffled_electrical_energy_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_electrical_energy_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_electrical_energy_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        hodge_reward = dp_hodge_cycle_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 270),
            eps,
            TRANS_GRID,
            hodge_cycle_info,
        )
        hodge_scale = hodge_cycle_scale(bank, hodge_reward)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 271),
            mode="hodge_cycle_reward",
            reward=hodge_reward,
            lo_scale=hodge_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"hodge_cycle_eps_{eps:.2f}"] = evaluate_variant(f"hodge_cycle_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 272),
            mode="hodge_cycle_margin_reward",
            reward=hodge_reward,
            lo_scale=hodge_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"hodge_cycle_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"hodge_cycle_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_hodge = shuffled_hodge_cycle_reward(hodge_reward, np.random.default_rng(args.seed + int(eps * 1000) + 273))
        shuffled_hodge_scale = hodge_cycle_scale(bank, shuffled_hodge)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 274),
            mode="shuffled_hodge_cycle_reward",
            reward=shuffled_hodge,
            lo_scale=shuffled_hodge_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_hodge_cycle_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_hodge_cycle_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 275),
            mode="shuffled_hodge_cycle_margin_reward",
            reward=shuffled_hodge,
            lo_scale=shuffled_hodge_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_hodge_cycle_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_hodge_cycle_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        cycle_reward = dp_cycle_motif_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 250),
            eps,
            TRANS_GRID,
            cycle_motif_info,
        )
        cycle_scale = cycle_motif_scale(cycle_reward)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 251),
            mode="cycle_motif_reward",
            reward=cycle_reward,
            lo_scale=cycle_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"cycle_motif_eps_{eps:.2f}"] = evaluate_variant(f"cycle_motif_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 252),
            mode="cycle_motif_margin_reward",
            reward=cycle_reward,
            lo_scale=cycle_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"cycle_motif_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"cycle_motif_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 256),
            mode="cycle_motif_lcb_reward",
            reward=cycle_reward,
            lo_scale=cycle_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"cycle_motif_lcb_z{CYCLE_MOTIF_LCB_Z:.1f}_eps_{eps:.2f}"] = evaluate_variant(f"cycle_motif_lcb_z{CYCLE_MOTIF_LCB_Z:.1f}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 257),
            mode="cycle_motif_lcb_margin_reward",
            reward=cycle_reward,
            lo_scale=cycle_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"cycle_motif_lcb_margin_z{CYCLE_MOTIF_LCB_Z:.1f}_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"cycle_motif_lcb_margin_z{CYCLE_MOTIF_LCB_Z:.1f}_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_cycle = shuffled_cycle_motif_reward(cycle_reward, np.random.default_rng(args.seed + int(eps * 1000) + 253))
        shuffled_cycle_scale = cycle_motif_scale(shuffled_cycle)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 254),
            mode="shuffled_cycle_motif_reward",
            reward=shuffled_cycle,
            lo_scale=shuffled_cycle_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_cycle_motif_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_cycle_motif_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 255),
            mode="shuffled_cycle_motif_margin_reward",
            reward=shuffled_cycle,
            lo_scale=shuffled_cycle_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_cycle_motif_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_cycle_motif_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 258),
            mode="shuffled_cycle_motif_lcb_reward",
            reward=shuffled_cycle,
            lo_scale=shuffled_cycle_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_cycle_motif_lcb_z{CYCLE_MOTIF_LCB_Z:.1f}_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_cycle_motif_lcb_z{CYCLE_MOTIF_LCB_Z:.1f}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 259),
            mode="shuffled_cycle_motif_lcb_margin_reward",
            reward=shuffled_cycle,
            lo_scale=shuffled_cycle_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_cycle_motif_lcb_margin_z{CYCLE_MOTIF_LCB_Z:.1f}_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_cycle_motif_lcb_margin_z{CYCLE_MOTIF_LCB_Z:.1f}_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        corridor_reward = dp_corridor_residual_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 260),
            eps,
            TRANS_GRID,
            corridor_basis,
        )
        corridor_scale = corridor_residual_scale(bank, node_cells, corridor_reward, TRANS_GRID)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 261),
            mode="corridor_residual_reward",
            reward=corridor_reward,
            lo_scale=corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"corridor_residual_eps_{eps:.2f}"] = evaluate_variant(f"corridor_residual_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 262),
            mode="corridor_residual_margin_reward",
            reward=corridor_reward,
            lo_scale=corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"corridor_residual_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"corridor_residual_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_corridor = shuffled_corridor_residual_reward(corridor_reward, np.random.default_rng(args.seed + int(eps * 1000) + 263))
        shuffled_corridor_scale = corridor_residual_scale(bank, node_cells, shuffled_corridor, TRANS_GRID)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 264),
            mode="shuffled_corridor_residual_reward",
            reward=shuffled_corridor,
            lo_scale=shuffled_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_corridor_residual_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_corridor_residual_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 265),
            mode="shuffled_corridor_residual_margin_reward",
            reward=shuffled_corridor,
            lo_scale=shuffled_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_corridor_residual_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_corridor_residual_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        for margin in margin_sweep:
            margin_tag = f"{margin:.3f}".replace(".", "p")
            syn, diag = select_paths(
                bank,
                road_coords,
                node_cells,
                np.random.default_rng(args.seed + int(eps * 1000) + int(margin * 10000) + 276),
                mode="corridor_residual_margin_reward",
                reward=corridor_reward,
                lo_scale=corridor_scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                switch_margin=margin,
                deterministic=args.deterministic_select,
            )
            payload["variants"][f"corridor_residual_margin_m{margin_tag}_eps_{eps:.2f}"] = evaluate_variant(f"corridor_residual_margin_m{margin_tag}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

            syn, diag = select_paths(
                bank,
                road_coords,
                node_cells,
                np.random.default_rng(args.seed + int(eps * 1000) + int(margin * 10000) + 277),
                mode="shuffled_corridor_residual_margin_reward",
                reward=shuffled_corridor,
                lo_scale=shuffled_corridor_scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                switch_margin=margin,
                deterministic=args.deterministic_select,
            )
            payload["variants"][f"shuffled_corridor_residual_margin_m{margin_tag}_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_corridor_residual_margin_m{margin_tag}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        split_eps = eps / 2.0
        bridge_half = dp_graph_flow_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 266),
            split_eps,
            TRANS_GRID,
            public_flow,
        )
        corridor_half = dp_corridor_residual_reward(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 267),
            split_eps,
            TRANS_GRID,
            corridor_basis,
        )
        bridge_corridor_reward = {**bridge_half, **corridor_half}
        bridge_corridor_scale = corridor_residual_scale(bank, node_cells, bridge_corridor_reward, TRANS_GRID)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 268),
            mode="bridge_corridor_consensus_reward",
            reward=bridge_corridor_reward,
            lo_scale=bridge_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"bridge_corridor_consensus_eps_{eps:.2f}"] = evaluate_variant(f"bridge_corridor_consensus_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 269),
            mode="bridge_corridor_consensus_margin_reward",
            reward=bridge_corridor_reward,
            lo_scale=bridge_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"bridge_corridor_consensus_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"bridge_corridor_consensus_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_bridge_half = shuffled_factorized_reward(bridge_half, np.random.default_rng(args.seed + int(eps * 1000) + 270))
        shuffled_corridor_half = shuffled_corridor_residual_reward(corridor_half, np.random.default_rng(args.seed + int(eps * 1000) + 271))
        shuffled_bridge_corridor = {**shuffled_bridge_half, **shuffled_corridor_half}
        shuffled_bridge_corridor_scale = corridor_residual_scale(bank, node_cells, shuffled_bridge_corridor, TRANS_GRID)
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 272),
            mode="shuffled_bridge_corridor_consensus_reward",
            reward=shuffled_bridge_corridor,
            lo_scale=shuffled_bridge_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_bridge_corridor_consensus_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_bridge_corridor_consensus_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 273),
            mode="shuffled_bridge_corridor_consensus_margin_reward",
            reward=shuffled_bridge_corridor,
            lo_scale=shuffled_bridge_corridor_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_bridge_corridor_consensus_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_bridge_corridor_consensus_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 239),
            mode="graph_flow_lcb_reward",
            reward=graph_flow_reward,
            lo_scale=graph_flow_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"graph_flow_lcb_w{GRAPH_FLOW_LCB_WEIGHT:.2f}_eps_{eps:.2f}"] = evaluate_variant(f"graph_flow_lcb_w{GRAPH_FLOW_LCB_WEIGHT:.2f}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 240),
            mode="graph_flow_lcb_margin_reward",
            reward=graph_flow_reward,
            lo_scale=graph_flow_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"graph_flow_lcb_margin_w{GRAPH_FLOW_LCB_WEIGHT:.2f}_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"graph_flow_lcb_margin_w{GRAPH_FLOW_LCB_WEIGHT:.2f}_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_graph_flow = shuffled_factorized_reward(graph_flow_reward, np.random.default_rng(args.seed + int(eps * 1000) + 234))
        shuffled_graph_flow_scale = reward_normalizer(shuffled_graph_flow["graph_flow"].ravel())
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 235),
            mode="shuffled_graph_flow_reward",
            reward=shuffled_graph_flow,
            lo_scale=shuffled_graph_flow_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_graph_flow_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_graph_flow_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 236),
            mode="shuffled_graph_flow_margin_reward",
            reward=shuffled_graph_flow,
            lo_scale=shuffled_graph_flow_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_graph_flow_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_graph_flow_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 2371),
            mode="shuffled_graph_flow_llr_reward",
            reward=shuffled_graph_flow,
            lo_scale=shuffled_graph_flow_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_graph_flow_llr_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_graph_flow_llr_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 2372),
            mode="shuffled_graph_flow_llr_margin_reward",
            reward=shuffled_graph_flow,
            lo_scale=shuffled_graph_flow_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_graph_flow_llr_margin_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_graph_flow_llr_margin_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 241),
            mode="shuffled_graph_flow_lcb_reward",
            reward=shuffled_graph_flow,
            lo_scale=shuffled_graph_flow_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_graph_flow_lcb_w{GRAPH_FLOW_LCB_WEIGHT:.2f}_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_graph_flow_lcb_w{GRAPH_FLOW_LCB_WEIGHT:.2f}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 242),
            mode="shuffled_graph_flow_lcb_margin_reward",
            reward=shuffled_graph_flow,
            lo_scale=shuffled_graph_flow_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            switch_margin=0.05,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_graph_flow_lcb_margin_w{GRAPH_FLOW_LCB_WEIGHT:.2f}_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_graph_flow_lcb_margin_w{GRAPH_FLOW_LCB_WEIGHT:.2f}_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        for lcb in lcb_sweep:
            lcb_tag = f"{lcb:.3f}".replace(".", "p")
            syn, diag = select_paths(
                bank,
                road_coords,
                node_cells,
                np.random.default_rng(args.seed + int(eps * 1000) + int(lcb * 10000) + 243),
                mode="graph_flow_lcb_reward",
                reward=graph_flow_reward,
                lo_scale=graph_flow_scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                lcb_weight=lcb,
                deterministic=args.deterministic_select,
            )
            payload["variants"][f"graph_flow_lcb_w{lcb_tag}_eps_{eps:.2f}"] = evaluate_variant(f"graph_flow_lcb_w{lcb_tag}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

            syn, diag = select_paths(
                bank,
                road_coords,
                node_cells,
                np.random.default_rng(args.seed + int(eps * 1000) + int(lcb * 10000) + 244),
                mode="graph_flow_lcb_margin_reward",
                reward=graph_flow_reward,
                lo_scale=graph_flow_scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                switch_margin=0.05,
                lcb_weight=lcb,
                deterministic=args.deterministic_select,
            )
            payload["variants"][f"graph_flow_lcb_margin_w{lcb_tag}_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"graph_flow_lcb_margin_w{lcb_tag}_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

            syn, diag = select_paths(
                bank,
                road_coords,
                node_cells,
                np.random.default_rng(args.seed + int(eps * 1000) + int(lcb * 10000) + 245),
                mode="shuffled_graph_flow_lcb_reward",
                reward=shuffled_graph_flow,
                lo_scale=shuffled_graph_flow_scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                lcb_weight=lcb,
                deterministic=args.deterministic_select,
            )
            payload["variants"][f"shuffled_graph_flow_lcb_w{lcb_tag}_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_graph_flow_lcb_w{lcb_tag}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

            syn, diag = select_paths(
                bank,
                road_coords,
                node_cells,
                np.random.default_rng(args.seed + int(eps * 1000) + int(lcb * 10000) + 246),
                mode="shuffled_graph_flow_lcb_margin_reward",
                reward=shuffled_graph_flow,
                lo_scale=shuffled_graph_flow_scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                switch_margin=0.05,
                lcb_weight=lcb,
                deterministic=args.deterministic_select,
            )
            payload["variants"][f"shuffled_graph_flow_lcb_margin_w{lcb_tag}_m0p050_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_graph_flow_lcb_margin_w{lcb_tag}_m0p050_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        for margin in margin_sweep:
            margin_tag = f"{margin:.3f}".replace(".", "p")
            syn, diag = select_paths(
                bank,
                road_coords,
                node_cells,
                np.random.default_rng(args.seed + int(eps * 1000) + int(margin * 10000) + 237),
                mode="graph_flow_margin_reward",
                reward=graph_flow_reward,
                lo_scale=graph_flow_scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                switch_margin=margin,
                deterministic=args.deterministic_select,
            )
            payload["variants"][f"graph_flow_margin_m{margin_tag}_eps_{eps:.2f}"] = evaluate_variant(f"graph_flow_margin_m{margin_tag}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

            syn, diag = select_paths(
                bank,
                road_coords,
                node_cells,
                np.random.default_rng(args.seed + int(eps * 1000) + int(margin * 10000) + 238),
                mode="shuffled_graph_flow_margin_reward",
                reward=shuffled_graph_flow,
                lo_scale=shuffled_graph_flow_scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                switch_margin=margin,
                deterministic=args.deterministic_select,
            )
            payload["variants"][f"shuffled_graph_flow_margin_m{margin_tag}_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_graph_flow_margin_m{margin_tag}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        if args.only_graph_flow:
            continue

        reward_rng = np.random.default_rng(args.seed + int(eps * 1000) + 303)
        select_rng = np.random.default_rng(args.seed + int(eps * 1000) + 404)
        od_reward = dp_od_transition(train, mn, span, reward_rng, eps, TRANS_GRID)
        od_scale = reward_normalizer(od_reward.ravel())
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            select_rng,
            mode="od_reward",
            reward=od_reward,
            lo_scale=od_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"od_eps_{eps:.2f}"] = evaluate_variant(f"od_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled = shuffled_od_reward(od_reward, np.random.default_rng(args.seed + int(eps * 1000) + 505))
        shuf_scale = reward_normalizer(shuffled.ravel())
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 606),
            mode="shuffled_od_reward",
            reward=shuffled,
            lo_scale=shuf_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_od_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_od_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        reward_rng = np.random.default_rng(args.seed + int(eps * 1000) + 707)
        select_rng = np.random.default_rng(args.seed + int(eps * 1000) + 808)
        olp_reward = dp_od_length_phase_transition(train, mn, span, reward_rng, eps, TRANS_GRID)
        olp_scale = reward_normalizer(olp_reward.ravel())
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            select_rng,
            mode="olp_reward",
            reward=olp_reward,
            lo_scale=olp_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"olp_eps_{eps:.2f}"] = evaluate_variant(f"olp_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_olp = shuffled_tensor_reward(olp_reward, np.random.default_rng(args.seed + int(eps * 1000) + 909))
        shuffled_olp_scale = reward_normalizer(shuffled_olp.ravel())
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 1001),
            mode="shuffled_olp_reward",
            reward=shuffled_olp,
            lo_scale=shuffled_olp_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_olp_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_olp_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        reward_rng = np.random.default_rng(args.seed + int(eps * 1000) + 1101)
        select_rng = np.random.default_rng(args.seed + int(eps * 1000) + 1202)
        bolp_reward = dp_backoff_olp_transition(train, mn, span, reward_rng, eps, TRANS_GRID)
        bolp_scale = reward_normalizer(bolp_reward.ravel())
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            select_rng,
            mode="backoff_olp_reward",
            reward=bolp_reward,
            lo_scale=bolp_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"bolp_eps_{eps:.2f}"] = evaluate_variant(f"bolp_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_bolp = shuffled_tensor_reward(bolp_reward, np.random.default_rng(args.seed + int(eps * 1000) + 1303))
        shuffled_bolp_scale = reward_normalizer(shuffled_bolp.ravel())
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 1404),
            mode="shuffled_backoff_olp_reward",
            reward=shuffled_bolp,
            lo_scale=shuffled_bolp_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_bolp_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_bolp_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        reward_rng = np.random.default_rng(args.seed + int(eps * 1000) + 1505)
        select_rng = np.random.default_rng(args.seed + int(eps * 1000) + 1606)
        fact_reward = dp_factorized_transition_reward(train, mn, span, reward_rng, eps, TRANS_GRID)
        fact_scale = reward_normalizer(flatten_reward_values(fact_reward))
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            select_rng,
            mode="factorized_reward",
            reward=fact_reward,
            lo_scale=fact_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"factorized_eps_{eps:.2f}"] = evaluate_variant(f"factorized_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 1656),
            mode="factorized_margin_reward",
            reward=fact_reward,
            lo_scale=fact_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"factorized_margin_eps_{eps:.2f}"] = evaluate_variant(f"factorized_margin_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_fact = shuffled_factorized_reward(fact_reward, np.random.default_rng(args.seed + int(eps * 1000) + 1707))
        shuffled_fact_scale = reward_normalizer(flatten_reward_values(shuffled_fact))
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 1808),
            mode="shuffled_factorized_reward",
            reward=shuffled_fact,
            lo_scale=shuffled_fact_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_factorized_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_factorized_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells,
            np.random.default_rng(args.seed + int(eps * 1000) + 1859),
            mode="shuffled_factorized_margin_reward",
            reward=shuffled_fact,
            lo_scale=shuffled_fact_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_factorized_margin_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_factorized_margin_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        for margin in margin_sweep:
            margin_tag = f"{margin:.3f}".replace(".", "p")
            syn, diag = select_paths(
                bank,
                road_coords,
                node_cells,
                np.random.default_rng(args.seed + int(eps * 1000) + int(margin * 10000) + 1861),
                mode="factorized_margin_reward",
                reward=fact_reward,
                lo_scale=fact_scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                switch_margin=margin,
                deterministic=args.deterministic_select,
            )
            payload["variants"][f"factorized_margin_m{margin_tag}_eps_{eps:.2f}"] = evaluate_variant(f"factorized_margin_m{margin_tag}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

            syn, diag = select_paths(
                bank,
                road_coords,
                node_cells,
                np.random.default_rng(args.seed + int(eps * 1000) + int(margin * 10000) + 1867),
                mode="shuffled_factorized_margin_reward",
                reward=shuffled_fact,
                lo_scale=shuffled_fact_scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                switch_margin=margin,
                deterministic=args.deterministic_select,
            )
            payload["variants"][f"shuffled_factorized_margin_m{margin_tag}_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_factorized_margin_m{margin_tag}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        reward_rng = np.random.default_rng(args.seed + int(eps * 1000) + 1909)
        select_rng = np.random.default_rng(args.seed + int(eps * 1000) + 2010)
        routefeat_reward = dp_route_feature_reward(train, mn, span, reward_rng, eps, FEATURE_GRID)
        routefeat_scale = reward_normalizer(flatten_reward_values(routefeat_reward))
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells_feat,
            select_rng,
            mode="route_feature_reward",
            reward=routefeat_reward,
            lo_scale=routefeat_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"routefeat_eps_{eps:.2f}"] = evaluate_variant(f"routefeat_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells_feat,
            np.random.default_rng(args.seed + int(eps * 1000) + 2061),
            mode="route_feature_contrast_reward",
            reward=routefeat_reward,
            lo_scale=routefeat_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"routefeat_contrast_eps_{eps:.2f}"] = evaluate_variant(f"routefeat_contrast_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells_feat,
            np.random.default_rng(args.seed + int(eps * 1000) + 2086),
            mode="route_feature_margin_reward",
            reward=routefeat_reward,
            lo_scale=routefeat_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"routefeat_margin_eps_{eps:.2f}"] = evaluate_variant(f"routefeat_margin_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells_feat,
            np.random.default_rng(args.seed + int(eps * 1000) + 2097),
            mode="route_feature_adaptive_reward",
            reward=routefeat_reward,
            lo_scale=routefeat_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"routefeat_adaptive_eps_{eps:.2f}"] = evaluate_variant(f"routefeat_adaptive_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells_feat,
            np.random.default_rng(args.seed + int(eps * 1000) + 2104),
            mode="route_feature_consensus_reward",
            reward=routefeat_reward,
            lo_scale=routefeat_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"routefeat_consensus_eps_{eps:.2f}"] = evaluate_variant(f"routefeat_consensus_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        hybrid_rng = np.random.default_rng(args.seed + int(eps * 1000) + 2109)
        hybrid_route = dp_route_feature_reward(train, mn, span, hybrid_rng, eps / 2.0, FEATURE_GRID)
        hybrid_factor = dp_factorized_transition_reward(train, mn, span, hybrid_rng, eps / 2.0, TRANS_GRID)
        hybrid_reward = {
            "hybrid_route": hybrid_route,
            "hybrid_factor": hybrid_factor,
            "route_scale": reward_normalizer(flatten_reward_values(hybrid_route)),
            "factor_scale": reward_normalizer(flatten_reward_values(hybrid_factor)),
        }
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells_feat,
            np.random.default_rng(args.seed + int(eps * 1000) + 2110),
            mode="hybrid_margin_reward",
            reward=hybrid_reward,
            lo_scale=(0.0, 1.0),
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            factor_node_cells=node_cells,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"hybrid_margin_eps_{eps:.2f}"] = evaluate_variant(f"hybrid_margin_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_routefeat = shuffled_factorized_reward(routefeat_reward, np.random.default_rng(args.seed + int(eps * 1000) + 2111))
        shuffled_routefeat_scale = reward_normalizer(flatten_reward_values(shuffled_routefeat))
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells_feat,
            np.random.default_rng(args.seed + int(eps * 1000) + 2212),
            mode="shuffled_route_feature_reward",
            reward=shuffled_routefeat,
            lo_scale=shuffled_routefeat_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_routefeat_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_routefeat_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells_feat,
            np.random.default_rng(args.seed + int(eps * 1000) + 2263),
            mode="shuffled_route_feature_contrast_reward",
            reward=shuffled_routefeat,
            lo_scale=shuffled_routefeat_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_routefeat_contrast_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_routefeat_contrast_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells_feat,
            np.random.default_rng(args.seed + int(eps * 1000) + 2288),
            mode="shuffled_route_feature_margin_reward",
            reward=shuffled_routefeat,
            lo_scale=shuffled_routefeat_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_routefeat_margin_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_routefeat_margin_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        for margin in margin_sweep:
            margin_tag = f"{margin:.3f}".replace(".", "p")
            syn, diag = select_paths(
                bank,
                road_coords,
                node_cells_feat,
                np.random.default_rng(args.seed + int(eps * 1000) + int(margin * 10000) + 2269),
                mode="route_feature_margin_reward",
                reward=routefeat_reward,
                lo_scale=routefeat_scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                switch_margin=margin,
                deterministic=args.deterministic_select,
            )
            payload["variants"][f"routefeat_margin_m{margin_tag}_eps_{eps:.2f}"] = evaluate_variant(f"routefeat_margin_m{margin_tag}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

            syn, diag = select_paths(
                bank,
                road_coords,
                node_cells_feat,
                np.random.default_rng(args.seed + int(eps * 1000) + int(margin * 10000) + 2275),
                mode="shuffled_route_feature_margin_reward",
                reward=shuffled_routefeat,
                lo_scale=shuffled_routefeat_scale,
                mn=mn,
                span=span,
                real_signature_prob=real_signature_prob,
                eval_transition_logprob=eval_logprob,
                switch_margin=margin,
                deterministic=args.deterministic_select,
            )
            payload["variants"][f"shuffled_routefeat_margin_m{margin_tag}_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_routefeat_margin_m{margin_tag}_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells_feat,
            np.random.default_rng(args.seed + int(eps * 1000) + 2299),
            mode="shuffled_route_feature_adaptive_reward",
            reward=shuffled_routefeat,
            lo_scale=shuffled_routefeat_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_routefeat_adaptive_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_routefeat_adaptive_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells_feat,
            np.random.default_rng(args.seed + int(eps * 1000) + 2306),
            mode="shuffled_route_feature_consensus_reward",
            reward=shuffled_routefeat,
            lo_scale=shuffled_routefeat_scale,
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_routefeat_consensus_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_routefeat_consensus_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

        shuffled_hybrid_route = shuffled_factorized_reward(hybrid_route, np.random.default_rng(args.seed + int(eps * 1000) + 2311))
        shuffled_hybrid_factor = shuffled_factorized_reward(hybrid_factor, np.random.default_rng(args.seed + int(eps * 1000) + 2312))
        shuffled_hybrid_reward = {
            "hybrid_route": shuffled_hybrid_route,
            "hybrid_factor": shuffled_hybrid_factor,
            "route_scale": reward_normalizer(flatten_reward_values(shuffled_hybrid_route)),
            "factor_scale": reward_normalizer(flatten_reward_values(shuffled_hybrid_factor)),
        }
        syn, diag = select_paths(
            bank,
            road_coords,
            node_cells_feat,
            np.random.default_rng(args.seed + int(eps * 1000) + 2313),
            mode="shuffled_hybrid_margin_reward",
            reward=shuffled_hybrid_reward,
            lo_scale=(0.0, 1.0),
            mn=mn,
            span=span,
            real_signature_prob=real_signature_prob,
            eval_transition_logprob=eval_logprob,
            factor_node_cells=node_cells,
            deterministic=args.deterministic_select,
        )
        payload["variants"][f"shuffled_hybrid_margin_eps_{eps:.2f}"] = evaluate_variant(f"shuffled_hybrid_margin_eps_{eps:.2f}", syn, diag, eval_real, context, road_coords, graph, mn, span)

    public_indices = payload["variants"]["public_only"]["diagnostics"].get("selected_indices", [])
    public_logps = payload["variants"]["public_only"]["diagnostics"].get("selected_eval_transition_logprobs", [])
    for name, item in payload["variants"].items():
        indices = item["diagnostics"].get("selected_indices", [])
        logps = item["diagnostics"].get("selected_eval_transition_logprobs", [])
        n = min(len(public_indices), len(indices))
        n_logp = min(len(public_logps), len(logps))
        item["paired_vs_public"] = {
            "selection_change_rate": float(sum(1 for i in range(n) if indices[i] != public_indices[i]) / n) if n else None,
            "eval_logprob_win_rate": float(sum(1 for i in range(n_logp) if logps[i] > public_logps[i] + 1e-9) / n_logp) if n_logp else None,
            "eval_logprob_loss_rate": float(sum(1 for i in range(n_logp) if logps[i] < public_logps[i] - 1e-9) / n_logp) if n_logp else None,
            "real_signature_mass_delta": (
                None
                if item["diagnostics"].get("mean_selected_real_signature_mass") is None
                or payload["variants"]["public_only"]["diagnostics"].get("mean_selected_real_signature_mass") is None
                else float(
                    item["diagnostics"]["mean_selected_real_signature_mass"]
                    - payload["variants"]["public_only"]["diagnostics"]["mean_selected_real_signature_mass"]
                )
            ),
            "eval_transition_logprob_delta": (
                None
                if item["diagnostics"].get("mean_selected_eval_transition_logprob") is None
                or payload["variants"]["public_only"]["diagnostics"].get("mean_selected_eval_transition_logprob") is None
                else float(
                    item["diagnostics"]["mean_selected_eval_transition_logprob"]
                    - payload["variants"]["public_only"]["diagnostics"]["mean_selected_eval_transition_logprob"]
                )
            ),
        }

    out = OUT_DIR / args.out_name
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    print(out)
    def fmt_metric(value):
        return "NA" if value is None else f"{float(value):.4f}"

    for name, item in payload["variants"].items():
        m = item["metrics"]
        print(
            f"{name:24s} n={m['n_syn']:3d} "
            f"od_corr={fmt_metric(m['od_corridor_jsd'])} "
            f"edge_trans={fmt_metric(m['edge_transition_jsd'])} "
            f"route_sig_g6={fmt_metric(m['route_signature_jsd_g6'])} "
            f"eval_logp={item['diagnostics'].get('mean_selected_eval_transition_logprob')} "
            f"change={item.get('paired_vs_public', {}).get('selection_change_rate')}"
        )


if __name__ == "__main__":
    main()
