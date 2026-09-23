"""DP finite-candidate graphical route-choice experiment.

This is a first controlled prototype of the model described in
``dp_finite_candidate_route_choice_model.md``. It intentionally avoids old
reward variants and tests only:

1. public baselines;
2. DP route-choice moment fidelity;
3. finite-candidate maximum-entropy decoding;
4. shuffled-moment controls.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial import cKDTree

import reward_candidate_probe as probe


STRETCH_BINS = np.asarray([0.0, 1.05, 1.15, 1.30, 1.60, 2.00, np.inf], dtype=float)
FEATURE_WEIGHTS = (0.20, 0.50, 0.30)  # global corridor, OD-corridor, OD-stretch.


def parse_float_list(text: str) -> list[float]:
    if str(text).strip().lower() in {"", "none", "null", "-"}:
        return []
    return [float(x) for x in str(text).split(",") if x.strip()]


def filter_candidate_bank_by_base_stretch(candidate_bank: list[dict], coords: np.ndarray, max_stretch: float) -> list[dict]:
    if max_stretch <= 0:
        return candidate_bank
    out = []
    for item in candidate_bank:
        cands = item.get("candidates", [])
        if not cands:
            out.append(dict(item, candidates=[]))
            continue
        base_len = max(float(probe.path_length(coords, cands[0][1])), 1e-12)
        kept = []
        for cand_idx, cand in enumerate(cands):
            label, nodes = cand
            if cand_idx == 0 or float(probe.path_length(coords, nodes)) / base_len <= float(max_stretch):
                kept.append((label, nodes))
        out.append(dict(item, candidates=kept))
    return out


def polyline_length(points: np.ndarray) -> float:
    arr = np.asarray(points, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))


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


def corridor_bucket_from_phase_lateral(phase: float, lateral: float) -> int:
    phase_bins = probe.ANCHOR_SHAPE_PHASE_BINS
    lateral_bins = probe.ANCHOR_SHAPE_LATERAL_BINS
    pbin = int(np.searchsorted(phase_bins, float(np.clip(phase, 0.0, 1.0)), side="right") - 1)
    pbin = int(np.clip(pbin, 0, len(phase_bins) - 2))
    lbin = int(np.searchsorted(lateral_bins, float(lateral), side="right") - 1)
    lbin = int(np.clip(lbin, 0, len(lateral_bins) - 2))
    return int(pbin * (len(lateral_bins) - 1) + lbin)


def corridor_bucket_for_points(points: np.ndarray) -> int:
    arr = np.asarray(points, dtype=float)
    if len(arr) < 2:
        return corridor_bucket_from_phase_lateral(0.5, 0.0)
    a = arr[0]
    b = arr[-1]
    interior = arr[1:-1] if len(arr) > 2 else arr
    laterals = []
    phases = []
    for p in interior:
        phase, lateral = od_relative_phase_lateral(a, b, p)
        phases.append(float(np.clip(phase, 0.0, 1.0)))
        laterals.append(float(lateral))
    if not laterals:
        return corridor_bucket_from_phase_lateral(0.5, 0.0)
    idx = int(np.argmax(np.abs(np.asarray(laterals, dtype=float))))
    return corridor_bucket_from_phase_lateral(phases[idx], laterals[idx])


def stretch_bucket(value: float) -> int:
    idx = int(np.searchsorted(STRETCH_BINS, float(value), side="right") - 1)
    return int(np.clip(idx, 0, len(STRETCH_BINS) - 2))


def feature_layout() -> dict:
    n_od = len(probe.OD_BINS) - 1
    n_corr = (len(probe.ANCHOR_SHAPE_PHASE_BINS) - 1) * (len(probe.ANCHOR_SHAPE_LATERAL_BINS) - 1)
    n_stretch = len(STRETCH_BINS) - 1
    off_global = 0
    off_od_corr = off_global + n_corr
    off_od_stretch = off_od_corr + n_od * n_corr
    dim = off_od_stretch + n_od * n_stretch
    return {
        "n_od": int(n_od),
        "n_corr": int(n_corr),
        "n_stretch": int(n_stretch),
        "off_global": int(off_global),
        "off_od_corr": int(off_od_corr),
        "off_od_stretch": int(off_od_stretch),
        "dim": int(dim),
    }


def make_feature_vector(od: int, corr: int, stretch: int, layout: dict, weights: tuple[float, float, float]) -> np.ndarray:
    wg, wo, ws = (float(x) for x in weights)
    total = max(wg + wo + ws, 1e-12)
    wg, wo, ws = wg / total, wo / total, ws / total
    n_od = int(layout["n_od"])
    n_corr = int(layout["n_corr"])
    n_stretch = int(layout["n_stretch"])
    od = int(np.clip(od, 0, n_od - 1))
    corr = int(np.clip(corr, 0, n_corr - 1))
    stretch = int(np.clip(stretch, 0, n_stretch - 1))
    vec = np.zeros(int(layout["dim"]), dtype=float)
    vec[int(layout["off_global"]) + corr] += wg
    vec[int(layout["off_od_corr"]) + od * n_corr + corr] += wo
    vec[int(layout["off_od_stretch"]) + od * n_stretch + stretch] += ws
    return vec


def path_size_factors(candidates: list[tuple[str, list[int]]], coords: np.ndarray) -> np.ndarray:
    edge_counts: dict[tuple[int, int], int] = {}
    edge_lengths: list[list[float]] = []
    for _label, nodes in candidates:
        rows = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            edge = (int(a), int(b))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            rows.append(float(np.linalg.norm(coords[int(a)] - coords[int(b)])))
        edge_lengths.append(rows)
    out = []
    for (_label, nodes), lengths in zip(candidates, edge_lengths):
        plen = max(float(sum(lengths)), 1e-12)
        ps = 0.0
        for (a, b), elen in zip(zip(nodes[:-1], nodes[1:]), lengths):
            ps += (float(elen) / plen) / max(float(edge_counts.get((int(a), int(b)), 1)), 1.0)
        out.append(max(ps, 1e-9))
    return np.asarray(out, dtype=float)


def build_candidate_model(
    bank: list[dict],
    coords: np.ndarray,
    *,
    alpha_len: float,
    alpha_ps: float,
    weights: tuple[float, float, float],
) -> dict:
    layout = feature_layout()
    phi_by_request: list[np.ndarray] = []
    log_prior_by_request: list[np.ndarray] = []
    base_lengths = []
    endpoint_features = []
    for item in bank:
        cands = item.get("candidates", [])
        if not cands:
            phi_by_request.append(np.zeros((0, int(layout["dim"])), dtype=float))
            log_prior_by_request.append(np.zeros(0, dtype=float))
            base_lengths.append(1.0)
            endpoint_features.append(np.zeros(4, dtype=float))
            continue
        base_len = max(float(probe.path_length(coords, cands[0][1])), 1e-12)
        base_lengths.append(base_len)
        src = int(item.get("src", cands[0][1][0]))
        dst = int(item.get("dst", cands[0][1][-1]))
        endpoint_features.append(np.concatenate([coords[src], coords[dst]]))
        od = int(np.clip(item.get("od_bin", 0), 0, int(layout["n_od"]) - 1))
        ps = path_size_factors(cands, coords)
        rows = []
        utilities = []
        for _label, nodes in cands:
            path_coords = coords[np.asarray(nodes, dtype=int)]
            corr = corridor_bucket_for_points(path_coords)
            rel_len = float(probe.path_length(coords, nodes)) / base_len
            sb = stretch_bucket(rel_len)
            rows.append(make_feature_vector(od, corr, sb, layout, weights))
            utilities.append(-float(alpha_len) * rel_len)
        utilities = np.asarray(utilities, dtype=float) + float(alpha_ps) * np.log(ps + 1e-9)
        log_prior = utilities - logsumexp(utilities)
        phi_by_request.append(np.vstack(rows))
        log_prior_by_request.append(log_prior)
    endpoint_features = np.asarray(endpoint_features, dtype=float)
    return {
        "layout": layout,
        "phi_by_request": phi_by_request,
        "log_prior_by_request": log_prior_by_request,
        "base_lengths": np.asarray(base_lengths, dtype=float),
        "endpoint_features": endpoint_features,
        "endpoint_tree": cKDTree(endpoint_features) if len(endpoint_features) else None,
        "weights": np.asarray(weights, dtype=float),
        "alpha_len": float(alpha_len),
        "alpha_ps": float(alpha_ps),
    }


def request_index_for_traj(traj: np.ndarray, model: dict) -> int:
    tree = model.get("endpoint_tree")
    if tree is None:
        return 0
    arr = np.asarray(traj, dtype=float)
    if len(arr) < 2:
        point = np.zeros(4, dtype=float)
    else:
        point = np.concatenate([arr[0], arr[-1]])
    return int(tree.query(point)[1])


def trajectory_feature_vector(
    traj: np.ndarray,
    bank: list[dict],
    model: dict,
    mn: np.ndarray,
    span: np.ndarray,
    *,
    weights: tuple[float, float, float],
) -> np.ndarray:
    layout = model["layout"]
    arr = np.asarray(traj, dtype=float)
    if len(arr) < 2:
        return np.zeros(int(layout["dim"]), dtype=float)
    req_idx = request_index_for_traj(arr, model)
    od = int(np.clip(bank[req_idx].get("od_bin", probe.od_bin_for_points(arr[0], arr[-1], mn, span)), 0, int(layout["n_od"]) - 1))
    corr = corridor_bucket_for_points(arr)
    base_len = max(float(model["base_lengths"][req_idx]), 1e-12)
    sb = stretch_bucket(polyline_length(arr) / base_len)
    vec = make_feature_vector(od, corr, sb, layout, weights)
    l1 = float(np.sum(np.abs(vec)))
    if l1 > 1.0:
        vec = vec / l1
    return vec


def average_private_moment(
    trajs: list[np.ndarray],
    bank: list[dict],
    model: dict,
    mn: np.ndarray,
    span: np.ndarray,
    *,
    weights: tuple[float, float, float],
) -> np.ndarray:
    dim = int(model["layout"]["dim"])
    total = np.zeros(dim, dtype=float)
    if not trajs:
        return total
    for traj in trajs:
        total += trajectory_feature_vector(traj, bank, model, mn, span, weights=weights)
    return total / float(len(trajs))


def dp_average_moment(
    train: list[np.ndarray],
    bank: list[dict],
    model: dict,
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    *,
    weights: tuple[float, float, float],
) -> np.ndarray:
    dim = int(model["layout"]["dim"])
    total = np.zeros(dim, dtype=float)
    for traj in train:
        total += trajectory_feature_vector(traj, bank, model, mn, span, weights=weights)
    if eps > 0:
        total += rng.laplace(0.0, 1.0 / eps, size=dim)
    else:
        total = np.zeros(dim, dtype=float)
    return total / max(float(len(train)), 1.0)


def shuffled_moment(moment: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(moment, dtype=float).copy()
    if arr.size:
        arr = arr[rng.permutation(arr.size)]
    return arr


def model_moment(theta: np.ndarray, model: dict) -> tuple[np.ndarray, float, list[np.ndarray]]:
    dim = int(model["layout"]["dim"])
    total = np.zeros(dim, dtype=float)
    logz_sum = 0.0
    probs_by_request = []
    requests = max(len(model["phi_by_request"]), 1)
    omega = 1.0 / float(requests)
    for phi, log_prior in zip(model["phi_by_request"], model["log_prior_by_request"]):
        if phi.shape[0] == 0:
            probs_by_request.append(np.zeros(0, dtype=float))
            continue
        logits = log_prior + phi @ theta
        lz = float(logsumexp(logits))
        probs = np.exp(logits - lz)
        total += omega * (probs @ phi)
        logz_sum += omega * lz
        probs_by_request.append(probs)
    return total, logz_sum, probs_by_request


def fit_theta(moment: np.ndarray, model: dict, ridge: float, *, max_iter: int = 300) -> dict:
    dim = int(model["layout"]["dim"])
    target = np.asarray(moment, dtype=float).reshape(dim)
    ridge = max(float(ridge), 1e-8)

    def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
        mm, a_val, _ = model_moment(theta, model)
        val = float(a_val - np.dot(theta, target) + 0.5 * ridge * float(np.dot(theta, theta)))
        grad = mm - target + ridge * theta
        return val, grad

    opt = minimize(
        lambda x: fun(x)[0],
        np.zeros(dim, dtype=float),
        jac=lambda x: fun(x)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-10, "gtol": 1e-7},
    )
    theta = np.asarray(opt.x, dtype=float)
    mm, _a, probs = model_moment(theta, model)
    return {
        "theta": theta,
        "model_moment": mm,
        "probs_by_request": probs,
        "success": bool(opt.success),
        "message": str(opt.message),
        "nit": int(opt.nit),
        "objective": float(opt.fun),
        "theta_l2": float(np.linalg.norm(theta)),
        "moment_l2_to_target": float(np.linalg.norm(mm - target)),
    }


def prior_probs(model: dict) -> list[np.ndarray]:
    return [np.exp(log_prior) for log_prior in model["log_prior_by_request"]]


def select_from_probs(
    bank: list[dict],
    coords: np.ndarray,
    probs_by_request: list[np.ndarray],
    rng: np.random.Generator,
    mn: np.ndarray,
    span: np.ndarray,
    eval_logprob: np.ndarray,
    *,
    deterministic: bool,
    label: str,
) -> tuple[list[np.ndarray], dict]:
    syn = []
    selected_labels: dict[str, int] = {}
    selected_indices = []
    rel_lengths = []
    selected_eval_logprob = []
    selected_vs_public_eval_delta = []
    selected_gap_capture = []
    positive_gap_cases = 0
    positive_gap_better = 0
    positive_gap_bad = 0
    positive_gap_missed = 0
    for req_idx, item in enumerate(bank):
        cands = item.get("candidates", [])
        if not cands:
            continue
        probs = np.asarray(probs_by_request[req_idx], dtype=float) if req_idx < len(probs_by_request) else np.zeros(0)
        if probs.size != len(cands) or float(np.sum(probs)) <= 0:
            probs = np.zeros(len(cands), dtype=float)
            probs[0] = 1.0
        probs = np.maximum(probs, 0.0)
        probs = probs / max(float(np.sum(probs)), 1e-12)
        idx = int(np.argmax(probs)) if deterministic else int(rng.choice(len(cands), p=probs))
        label_i, nodes = cands[idx]
        selected_indices.append(idx)
        selected_labels[str(label_i)] = selected_labels.get(str(label_i), 0) + 1
        base_len = max(float(probe.path_length(coords, cands[0][1])), 1e-12)
        rel_lengths.append(float(probe.path_length(coords, nodes)) / base_len)
        eval_scores = []
        for _cand_label, cand_nodes in cands:
            traj_eval = probe.resample(coords[np.asarray(cand_nodes, dtype=int)], max(2, int(item["length"])))
            eval_scores.append(probe.path_eval_logprob(traj_eval, mn, span, eval_logprob, grid=6))
        public_eval = float(eval_scores[0])
        selected_eval = float(eval_scores[idx])
        best_eval = float(max(eval_scores))
        selected_eval_logprob.append(selected_eval)
        selected_vs_public_eval_delta.append(selected_eval - public_eval)
        gap = best_eval - public_eval
        if gap > 1e-9:
            positive_gap_cases += 1
            selected_gap_capture.append((selected_eval - public_eval) / gap)
            if idx == 0:
                positive_gap_missed += 1
            elif selected_eval > public_eval + 1e-9:
                positive_gap_better += 1
            elif selected_eval < public_eval - 1e-9:
                positive_gap_bad += 1
        traj = probe.resample(coords[np.asarray(nodes, dtype=int)], max(2, int(item["length"])))
        syn.append(traj)
    diag = {
        "n_selected": len(syn),
        "selector": str(label),
        "selected_labels": selected_labels,
        "selected_indices": selected_indices,
        "mean_relative_candidate_length": float(np.mean(rel_lengths)) if rel_lengths else None,
        "mean_selected_eval_transition_logprob": float(np.mean(selected_eval_logprob)) if selected_eval_logprob else None,
        "selected_eval_transition_logprobs": selected_eval_logprob,
        "mean_selected_vs_public_eval_delta": float(np.mean(selected_vs_public_eval_delta)) if selected_vs_public_eval_delta else None,
        "positive_oracle_gap_cases": int(positive_gap_cases),
        "mean_oracle_gap_capture": float(np.mean(selected_gap_capture)) if selected_gap_capture else None,
        "positive_gap_win_rate": float(positive_gap_better / positive_gap_cases) if positive_gap_cases else None,
        "positive_gap_bad_switch_rate": float(positive_gap_bad / positive_gap_cases) if positive_gap_cases else None,
        "positive_gap_missed_public_rate": float(positive_gap_missed / positive_gap_cases) if positive_gap_cases else None,
    }
    return syn, diag


def distances_to_eval_moment(moment: np.ndarray, eval_moment: np.ndarray) -> dict:
    diff = np.asarray(moment, dtype=float) - np.asarray(eval_moment, dtype=float)
    return {
        "l1": float(np.sum(np.abs(diff))),
        "l2": float(np.linalg.norm(diff)),
        "linf": float(np.max(np.abs(diff))) if diff.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-requests", type=int, default=30)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--candidate-mode", choices=("anchor_via", "hybrid_anchor", "reroute", "hybrid"), default="anchor_via")
    parser.add_argument("--candidate-max-stretch", type=float, default=2.35)
    parser.add_argument("--candidate-spur-trials", type=int, default=32)
    parser.add_argument("--candidate-cache-dir", default="candidate_cache")
    parser.add_argument("--refresh-candidate-cache", action="store_true")
    parser.add_argument("--filter-candidate-stretch", type=float, default=1.60)
    parser.add_argument("--load-limit", type=int, default=2600)
    parser.add_argument("--train-size", type=int, default=700)
    parser.add_argument("--eval-size", type=int, default=260)
    parser.add_argument("--seed", type=int, default=probe.SEED)
    parser.add_argument("--eps-list", default="0.4,0.8,1.6")
    parser.add_argument("--ridge-list", default="0.05,0.10,0.30,1.00")
    parser.add_argument("--alpha-len", type=float, default=1.0)
    parser.add_argument("--alpha-ps", type=float, default=0.7)
    parser.add_argument("--feature-weights", default=",".join(str(x) for x in FEATURE_WEIGHTS))
    parser.add_argument("--deterministic-select", action="store_true")
    parser.add_argument("--skip-edge-metrics", action="store_true")
    parser.add_argument("--out-name", default="dp_fc_grc_experiment.json")
    args = parser.parse_args()

    weights_raw = parse_float_list(args.feature_weights)
    weights = tuple(weights_raw[:3]) if len(weights_raw) >= 3 else FEATURE_WEIGHTS
    probe.SKIP_EDGE_METRICS = bool(args.skip_edge_metrics)
    probe.OUT_DIR.mkdir(parents=True, exist_ok=True)

    real, eval_real, road_coords, graph, mn, span, bank = probe.build_candidate_bank(
        args.max_requests,
        args.max_candidates,
        load_limit=args.load_limit,
        train_size=args.train_size,
        eval_size=args.eval_size,
        raw_graph_smoke=False,
        seed=args.seed,
        candidate_mode=args.candidate_mode,
        candidate_max_stretch=args.candidate_max_stretch,
        candidate_spur_trials=args.candidate_spur_trials,
        candidate_cache_dir=args.candidate_cache_dir,
        refresh_candidate_cache=bool(args.refresh_candidate_cache),
    )
    bank = filter_candidate_bank_by_base_stretch(bank, road_coords, float(args.filter_candidate_stretch))
    train = real[: args.train_size]
    context = real[: args.train_size + args.eval_size]
    eval_logprob = probe.transition_eval_logprob(eval_real, mn, span, grid=6)
    node_cells = probe.transition_node_cells(
        road_coords,
        mn,
        span,
        np.zeros((probe.TRANS_GRID * probe.TRANS_GRID, probe.TRANS_GRID * probe.TRANS_GRID)),
    )
    node_cells_g6 = probe.transition_node_cells(road_coords, mn, span, np.zeros((6 * 6, 6 * 6)))
    model = build_candidate_model(
        bank,
        road_coords,
        alpha_len=float(args.alpha_len),
        alpha_ps=float(args.alpha_ps),
        weights=weights,
    )
    eval_moment = average_private_moment(eval_real, bank, model, mn, span, weights=weights)
    public_mm, _a0, _p0 = model_moment(np.zeros(int(model["layout"]["dim"]), dtype=float), model)

    payload = {
        "config": {
            "seed": int(args.seed),
            "max_requests": int(args.max_requests),
            "max_candidates": int(args.max_candidates),
            "actual_requests": int(len(bank)),
            "candidate_mode": str(args.candidate_mode),
            "candidate_max_stretch": float(args.candidate_max_stretch),
            "filter_candidate_stretch": float(args.filter_candidate_stretch),
            "candidate_cache_dir": str(args.candidate_cache_dir),
            "candidate_family_hash": probe.candidate_family_hash(bank),
            "eps_list": parse_float_list(args.eps_list),
            "ridge_list": parse_float_list(args.ridge_list),
            "alpha_len": float(args.alpha_len),
            "alpha_ps": float(args.alpha_ps),
            "feature_weights": [float(x) for x in weights],
            "feature_dim": int(model["layout"]["dim"]),
            "deterministic_select": bool(args.deterministic_select),
            "skip_edge_metrics": bool(args.skip_edge_metrics),
            "script": str(Path(__file__).resolve()),
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "candidate_diagnostics": {
            "mean_candidates": float(np.mean([len(x["candidates"]) for x in bank])) if bank else 0.0,
            "min_candidates": int(min([len(x["candidates"]) for x in bank], default=0)),
            "max_candidates": int(max([len(x["candidates"]) for x in bank], default=0)),
            "support": probe.candidate_support_diagnostics(bank, road_coords, mn, span, eval_logprob),
            "build": getattr(probe.build_candidate_bank, "last_diagnostics", {}),
        },
        "moment_diagnostics": {
            "public_prior_to_eval": distances_to_eval_moment(public_mm, eval_moment),
        },
        "optimization": {},
        "variants": {},
        }

    def add_probe_variant(name: str, mode: str, rng_offset: int, reward=None, lo_scale=None, cells=None) -> None:
        syn, diag = probe.select_paths(
            bank,
            road_coords,
            node_cells if cells is None else cells,
            np.random.default_rng(args.seed + rng_offset),
            mode=mode,
            reward=reward,
            lo_scale=lo_scale,
            mn=mn,
            span=span,
            eval_transition_logprob=eval_logprob,
            deterministic=bool(args.deterministic_select),
        )
        payload["variants"][name] = probe.evaluate_variant(name, syn, diag, eval_real, context, road_coords, graph, mn, span)
        m = payload["variants"][name]["metrics"]
        d = payload["variants"][name]["diagnostics"]
        print(
            f"{name:48s} n={m['n_syn']:3d} "
            f"od_corr={m.get('od_corridor_jsd')} edge_trans={m.get('edge_transition_jsd')} "
            f"route_sig_g6={m.get('route_signature_jsd_g6')} eval_logp={d.get('mean_selected_eval_transition_logprob')} "
            f"gap_win={d.get('positive_gap_win_rate')}",
            flush=True,
        )

    def add_variant(name: str, probs: list[np.ndarray], rng_offset: int) -> None:
        syn, diag = select_from_probs(
            bank,
            road_coords,
            probs,
            np.random.default_rng(args.seed + rng_offset),
            mn,
            span,
            eval_logprob,
            deterministic=bool(args.deterministic_select),
            label=name,
        )
        payload["variants"][name] = probe.evaluate_variant(name, syn, diag, eval_real, context, road_coords, graph, mn, span)
        m = payload["variants"][name]["metrics"]
        d = payload["variants"][name]["diagnostics"]
        print(
            f"{name:48s} n={m['n_syn']:3d} "
            f"od_corr={m.get('od_corridor_jsd')} edge_trans={m.get('edge_transition_jsd')} "
            f"route_sig_g6={m.get('route_signature_jsd_g6')} eval_logp={d.get('mean_selected_eval_transition_logprob')} "
            f"gap_win={d.get('positive_gap_win_rate')}",
            flush=True,
        )

    add_probe_variant("public_only", "public_only", 17)
    add_probe_variant("public_uniform_candidate", "public_uniform_candidate", 18)
    add_probe_variant("public_length_only_candidate", "public_length_only_candidate", 19)
    oracle_scale = probe.reward_normalizer(eval_logprob.ravel())
    add_probe_variant(
        "oracle_eval_reward_NON_DP",
        "oracle_eval_reward",
        909,
        reward=eval_logprob,
        lo_scale=oracle_scale,
        cells=node_cells_g6,
    )
    add_variant("public_path_size_prior", prior_probs(model), 501)

    for eps in parse_float_list(args.eps_list):
        if eps <= 0:
            continue
        dp_m = dp_average_moment(
            train,
            bank,
            model,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 610),
            eps,
            weights=weights,
        )
        shuf_m = shuffled_moment(dp_m, np.random.default_rng(args.seed + int(eps * 1000) + 611))
        eps_key = f"eps_{eps:.2f}"
        payload["moment_diagnostics"][eps_key] = {
            "dp_to_eval": distances_to_eval_moment(dp_m, eval_moment),
            "shuffled_to_eval": distances_to_eval_moment(shuf_m, eval_moment),
            "dp_to_public": distances_to_eval_moment(dp_m, public_mm),
        }
        for ridge in parse_float_list(args.ridge_list):
            ridge_tag = f"{ridge:.2f}".replace(".", "p")
            eps_tag = f"{eps:.2f}"
            fit = fit_theta(dp_m, model, ridge)
            name = f"dp_fc_grc_eps_{eps_tag}_ridge_{ridge_tag}"
            payload["optimization"][name] = {
                "success": fit["success"],
                "message": fit["message"],
                "nit": fit["nit"],
                "objective": fit["objective"],
                "theta_l2": fit["theta_l2"],
                "moment_l2_to_target": fit["moment_l2_to_target"],
                "model_to_eval": distances_to_eval_moment(fit["model_moment"], eval_moment),
            }
            add_variant(name, fit["probs_by_request"], int(eps * 1000) + int(ridge * 1000) + 700)

            sfit = fit_theta(shuf_m, model, ridge)
            sname = f"shuffled_dp_fc_grc_eps_{eps_tag}_ridge_{ridge_tag}"
            payload["optimization"][sname] = {
                "success": sfit["success"],
                "message": sfit["message"],
                "nit": sfit["nit"],
                "objective": sfit["objective"],
                "theta_l2": sfit["theta_l2"],
                "moment_l2_to_target": sfit["moment_l2_to_target"],
                "model_to_eval": distances_to_eval_moment(sfit["model_moment"], eval_moment),
            }
            add_variant(sname, sfit["probs_by_request"], int(eps * 1000) + int(ridge * 1000) + 900)

    out = probe.OUT_DIR / args.out_name
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
