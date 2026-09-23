"""Candidate-conditioned route-choice metric probe.

This script evaluates the DP finite-candidate route-choice model without
mixing in global OD frequency, candidate support, or road coverage effects.

For each public candidate request, it builds a soft held-out route-choice label:

1. find the nearest held-out trajectories by ordered endpoint and length;
2. compare those real trajectories to every candidate path using grid-flow and
   grid-transition distances;
3. average soft nearest-candidate labels.

Then every public/DP/shuffled probability model is evaluated on the same fixed
candidate set using cross-entropy, KL, top-1 agreement, and paired wins.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree

import reward_candidate_probe as probe
import dp_fc_grc_experiment as fc


EXACT_CHOICE_SUPPORT_ANCHOR_EPS = 0.20


def parse_float_list(text: str) -> list[float]:
    return fc.parse_float_list(text)


def softmax(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=float)
    if arr.size == 0:
        return arr
    z = logsumexp(arr)
    return np.exp(arr - z)


def polyline_len(points: np.ndarray) -> float:
    arr = np.asarray(points, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))


def normalized_hist(indices: list[int], size: int) -> np.ndarray:
    vec = np.zeros(int(size), dtype=float)
    for idx in indices:
        if 0 <= int(idx) < size:
            vec[int(idx)] += 1.0
    s = float(vec.sum())
    if s > 0:
        vec /= s
    return vec


def path_cell_features(points: np.ndarray, mn: np.ndarray, span: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    cells = probe.compact_cell_sequence(np.asarray(points, dtype=float), mn, span, int(grid))
    n = int(grid) * int(grid)
    flow = normalized_hist(cells, n)
    trans = np.zeros(n * n, dtype=float)
    for a, b in zip(cells[:-1], cells[1:]):
        if int(a) != int(b):
            trans[int(a) * n + int(b)] += 1.0
    s = float(trans.sum())
    if s > 0:
        trans /= s
    return flow, trans


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def request_endpoint_vector(item: dict, coords: np.ndarray, mn: np.ndarray, span: np.ndarray) -> np.ndarray:
    src = int(item.get("src", item["candidates"][0][1][0]))
    dst = int(item.get("dst", item["candidates"][0][1][-1]))
    a = (coords[src] - mn) / np.maximum(span, 1e-12)
    b = (coords[dst] - mn) / np.maximum(span, 1e-12)
    return np.concatenate([a, b]).astype(float)


def eval_endpoint_matrix(eval_real: list[np.ndarray], mn: np.ndarray, span: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    lens = []
    for traj in eval_real:
        arr = np.asarray(traj, dtype=float)
        if len(arr) < 2:
            rows.append(np.zeros(4, dtype=float))
            lens.append(0.0)
            continue
        a = (arr[0] - mn) / np.maximum(span, 1e-12)
        b = (arr[-1] - mn) / np.maximum(span, 1e-12)
        rows.append(np.concatenate([a, b]).astype(float))
        lens.append(polyline_len(arr))
    return np.vstack(rows) if rows else np.zeros((0, 4), dtype=float), np.asarray(lens, dtype=float)


def candidate_features_for_request(
    item: dict,
    coords: np.ndarray,
    mn: np.ndarray,
    span: np.ndarray,
    grid: int,
) -> list[dict]:
    feats = []
    for label, nodes in item.get("candidates", []):
        pts = coords[np.asarray(nodes, dtype=int)]
        flow, trans = path_cell_features(pts, mn, span, grid)
        feats.append(
            {
                "label": str(label),
                "nodes": [int(x) for x in nodes],
                "length": polyline_len(pts),
                "flow": flow,
                "transition": trans,
            }
        )
    return feats


def build_route_choice_targets(
    bank: list[dict],
    coords: np.ndarray,
    eval_real: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    *,
    grid: int,
    k_eval_matches: int,
    endpoint_weight: float,
    length_weight: float,
    match_tau: float,
    label_tau: float,
    trans_weight: float,
    flow_weight: float,
    length_label_weight: float,
) -> dict:
    eval_endpoints, eval_lengths = eval_endpoint_matrix(eval_real, mn, span)
    eval_features = [path_cell_features(np.asarray(t, dtype=float), mn, span, grid) for t in eval_real]

    targets: list[np.ndarray] = []
    diagnostics = []
    for req_idx, item in enumerate(bank):
        cands = item.get("candidates", [])
        if not cands or len(eval_real) == 0:
            targets.append(np.zeros(len(cands), dtype=float))
            diagnostics.append({"request": int(req_idx), "usable": False})
            continue

        req_ep = request_endpoint_vector(item, coords, mn, span)
        req_len = max(float(item.get("length", 1)), 1.0)
        ep_dist = np.linalg.norm(eval_endpoints - req_ep[None, :], axis=1)
        len_dist = np.abs(np.log(np.maximum(eval_lengths, 1e-12) / req_len))
        match_score = float(endpoint_weight) * ep_dist + float(length_weight) * len_dist
        top_k = min(int(k_eval_matches), len(eval_real))
        top_idx = np.argsort(match_score)[:top_k]
        match_probs = softmax(-match_score[top_idx] / max(float(match_tau), 1e-9))

        cand_feats = candidate_features_for_request(item, coords, mn, span, grid)
        target = np.zeros(len(cand_feats), dtype=float)
        best_indices = []
        best_distances = []
        for w, eval_idx in zip(match_probs, top_idx):
            real_flow, real_trans = eval_features[int(eval_idx)]
            real_len = max(float(eval_lengths[int(eval_idx)]), 1e-12)
            distances = []
            for feat in cand_feats:
                d_trans = total_variation(real_trans, feat["transition"])
                d_flow = total_variation(real_flow, feat["flow"])
                d_len = abs(math.log(max(float(feat["length"]), 1e-12) / real_len))
                distances.append(
                    float(trans_weight) * d_trans
                    + float(flow_weight) * d_flow
                    + float(length_label_weight) * min(d_len, 3.0)
                )
            dist_arr = np.asarray(distances, dtype=float)
            cand_probs = softmax(-dist_arr / max(float(label_tau), 1e-9))
            target += float(w) * cand_probs
            best_indices.append(int(np.argmin(dist_arr)))
            best_distances.append(float(np.min(dist_arr)))
        target = np.maximum(target, 0.0)
        target = target / max(float(target.sum()), 1e-12)
        entropy = -float(np.sum(target[target > 0] * np.log(target[target > 0])))
        targets.append(target)
        diagnostics.append(
            {
                "request": int(req_idx),
                "usable": True,
                "top_eval_indices": [int(x) for x in top_idx],
                "mean_match_score": float(np.mean(match_score[top_idx])),
                "min_match_score": float(np.min(match_score[top_idx])),
                "target_entropy": entropy,
                "target_effective_candidates": float(np.exp(entropy)),
                "target_argmax": int(np.argmax(target)),
                "target_public_mass": float(target[0]) if target.size else None,
                "hard_best_indices": best_indices,
                "mean_best_candidate_distance": float(np.mean(best_distances)) if best_distances else None,
            }
        )

    usable = [d for d in diagnostics if d.get("usable")]
    return {
        "targets": targets,
        "diagnostics": diagnostics,
        "summary": {
            "n_requests": int(len(bank)),
            "n_usable": int(len(usable)),
            "mean_match_score": float(np.mean([d["mean_match_score"] for d in usable])) if usable else None,
            "mean_target_entropy": float(np.mean([d["target_entropy"] for d in usable])) if usable else None,
            "mean_target_effective_candidates": float(np.mean([d["target_effective_candidates"] for d in usable])) if usable else None,
            "mean_target_public_mass": float(np.mean([d["target_public_mass"] for d in usable])) if usable else None,
            "fraction_target_argmax_public": float(np.mean([d["target_argmax"] == 0 for d in usable])) if usable else None,
            "mean_best_candidate_distance": float(np.mean([d["mean_best_candidate_distance"] for d in usable])) if usable else None,
        },
    }


def build_exact_eval_candidate_bank(
    eval_real: list[np.ndarray],
    train: list[np.ndarray],
    coords: np.ndarray,
    graph: dict,
    mn: np.ndarray,
    span: np.ndarray,
    *,
    max_requests: int,
    max_candidates: int,
    seed: int,
    candidate_mode: str,
    candidate_max_stretch: float,
    candidate_spur_trials: int,
) -> list[dict]:
    """Build evaluation-only candidate sets from held-out trajectory endpoints."""
    rng = np.random.default_rng(int(seed))
    norm_train, anchor_mn, anchor_span = probe.normalize_with_public_bbox(train, probe.BEIJING_BBOX)
    centers_norm = probe.dp_anchors(norm_train, EXACT_CHOICE_SUPPORT_ANCHOR_EPS, rng)
    centers_real = centers_norm * anchor_span + anchor_mn
    tree = cKDTree(coords)
    anchor_nodes = [int(tree.query(c)[1]) for c in centers_real]
    reverse_graph = probe.reverse_adjacency(graph)

    out: list[dict] = []
    for eval_idx, traj in enumerate(eval_real):
        if len(out) >= int(max_requests):
            break
        arr = np.asarray(traj, dtype=float)
        if len(arr) < 2:
            continue
        src = int(tree.query(arr[0])[1])
        dst = int(tree.query(arr[-1])[1])
        if src == dst:
            continue
        cands = probe.generate_public_candidates(
            graph,
            coords,
            src,
            dst,
            max_candidates=max_candidates,
            mode=candidate_mode,
            tree=tree,
            reverse_graph=reverse_graph,
            anchor_nodes=anchor_nodes,
            max_stretch=candidate_max_stretch,
            spur_trials=candidate_spur_trials,
        )
        if len(cands) < 2:
            continue
        out.append(
            {
                "origin": -1,
                "dest": -1,
                "src": int(src),
                "dst": int(dst),
                "eval_index": int(eval_idx),
                "od_bin": probe.od_bin_for_points(coords[src], coords[dst], mn, span),
                "od_cell_bin": probe.od_pair_cell_bin_for_points(coords[src], coords[dst], mn, span),
                "length": int(len(arr)),
                "length_bin": probe.length_bin_for_len(int(len(arr))),
                "direct": float(np.linalg.norm(coords[src] - coords[dst])),
                "candidates": cands,
            }
        )
    return out


def build_exact_route_choice_targets(
    bank: list[dict],
    coords: np.ndarray,
    eval_real: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    *,
    grid: int,
    label_tau: float,
    trans_weight: float,
    flow_weight: float,
    length_label_weight: float,
) -> dict:
    targets: list[np.ndarray] = []
    diagnostics = []
    for req_idx, item in enumerate(bank):
        cands = item.get("candidates", [])
        eval_idx = int(item.get("eval_index", -1))
        if not cands or eval_idx < 0 or eval_idx >= len(eval_real):
            targets.append(np.zeros(len(cands), dtype=float))
            diagnostics.append({"request": int(req_idx), "usable": False})
            continue
        real = np.asarray(eval_real[eval_idx], dtype=float)
        real_flow, real_trans = path_cell_features(real, mn, span, grid)
        real_len = max(polyline_len(real), 1e-12)
        cand_feats = candidate_features_for_request(item, coords, mn, span, grid)
        distances = []
        for feat in cand_feats:
            d_trans = total_variation(real_trans, feat["transition"])
            d_flow = total_variation(real_flow, feat["flow"])
            d_len = abs(math.log(max(float(feat["length"]), 1e-12) / real_len))
            distances.append(
                float(trans_weight) * d_trans
                + float(flow_weight) * d_flow
                + float(length_label_weight) * min(d_len, 3.0)
            )
        dist_arr = np.asarray(distances, dtype=float)
        target = softmax(-dist_arr / max(float(label_tau), 1e-9))
        entropy = -float(np.sum(target[target > 0] * np.log(target[target > 0])))
        targets.append(target)
        diagnostics.append(
            {
                "request": int(req_idx),
                "usable": True,
                "eval_index": int(eval_idx),
                "target_entropy": entropy,
                "target_effective_candidates": float(np.exp(entropy)),
                "target_argmax": int(np.argmax(target)),
                "target_public_mass": float(target[0]) if target.size else None,
                "mean_best_candidate_distance": float(np.min(dist_arr)) if dist_arr.size else None,
                "candidate_distances": [float(x) for x in dist_arr],
            }
        )
    usable = [d for d in diagnostics if d.get("usable")]
    return {
        "targets": targets,
        "diagnostics": diagnostics,
        "summary": {
            "n_requests": int(len(bank)),
            "n_usable": int(len(usable)),
            "mean_match_score": 0.0,
            "mean_target_entropy": float(np.mean([d["target_entropy"] for d in usable])) if usable else None,
            "mean_target_effective_candidates": float(np.mean([d["target_effective_candidates"] for d in usable])) if usable else None,
            "mean_target_public_mass": float(np.mean([d["target_public_mass"] for d in usable])) if usable else None,
            "fraction_target_argmax_public": float(np.mean([d["target_argmax"] == 0 for d in usable])) if usable else None,
            "mean_best_candidate_distance": float(np.mean([d["mean_best_candidate_distance"] for d in usable])) if usable else None,
        },
    }


def compact_tokens(tokens: list[int], max_len: int) -> list[int]:
    if not tokens:
        return []
    compact = []
    for tok in tokens:
        tok = int(tok)
        if not compact or compact[-1] != tok:
            compact.append(tok)
    if len(compact) > int(max_len):
        idx = np.linspace(0, len(compact) - 1, int(max_len)).round().astype(int)
        compact = [compact[int(i)] for i in idx]
    return compact


def grid_signature(points: np.ndarray, mn: np.ndarray, span: np.ndarray, *, grid: int, max_len: int) -> str:
    cells = probe.compact_cell_sequence(np.asarray(points, dtype=float), mn, span, int(grid))
    cells = compact_tokens(cells, int(max_len))
    return "-".join(str(int(x)) for x in cells) if cells else "empty"


def cut_band_signature(points: np.ndarray, *, layers: int, max_len: int) -> str:
    arr = np.asarray(points, dtype=float)
    if len(arr) < 2:
        return "empty"
    src = arr[0]
    dst = arr[-1]
    delta = dst - src
    direct = max(float(np.linalg.norm(delta)), 1e-12)
    axis = delta / direct
    band_edges = np.asarray([-np.inf, -0.60, -0.35, -0.18, -0.06, 0.06, 0.18, 0.35, 0.60, np.inf])
    tokens = []
    for p in arr:
        off = p - src
        phase = float(np.dot(off, axis) / direct)
        lateral = float(axis[0] * off[1] - axis[1] * off[0]) / direct
        layer = int(np.floor(np.clip(phase, 0.0, 0.999999) * int(layers)))
        band = int(np.searchsorted(band_edges, lateral, side="right") - 1)
        band = int(np.clip(band, 0, len(band_edges) - 2))
        tokens.append(layer * (len(band_edges) - 1) + band)
    tokens = compact_tokens(tokens, int(max_len))
    return "-".join(str(int(x)) for x in tokens) if tokens else "empty"


def signature_views_for_points(
    points: np.ndarray,
    mn: np.ndarray,
    span: np.ndarray,
    *,
    od_bin: int,
    grid: int,
    max_len: int,
    layers: int,
) -> list[tuple[str, str, float]]:
    grid_key = grid_signature(points, mn, span, grid=int(grid), max_len=int(max_len))
    cut_key = cut_band_signature(points, layers=int(layers), max_len=int(max_len))
    od = int(od_bin)
    # Total per-trajectory mass is 1, so Laplace scale 1/epsilon is valid.
    return [
        ("global_grid", grid_key, 0.25),
        ("od_grid", f"{od}|{grid_key}", 0.25),
        ("global_cut", cut_key, 0.25),
        ("od_cut", f"{od}|{cut_key}", 0.25),
    ]


def build_signature_counts(
    trajs: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    *,
    grid: int,
    max_len: int,
    layers: int,
) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, float]] = {}
    for traj in trajs:
        arr = np.asarray(traj, dtype=float)
        if len(arr) < 2:
            continue
        od = probe.od_bin_for_points(arr[0], arr[-1], mn, span)
        for view, key, weight in signature_views_for_points(arr, mn, span, od_bin=od, grid=grid, max_len=max_len, layers=layers):
            if view not in counts:
                counts[view] = {}
            counts[view][key] = counts[view].get(key, 0.0) + float(weight)
    return counts


def dp_signature_counts(
    trajs: list[np.ndarray],
    mn: np.ndarray,
    span: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    *,
    grid: int,
    max_len: int,
    layers: int,
) -> dict[str, dict[str, float]]:
    counts = build_signature_counts(trajs, mn, span, grid=grid, max_len=max_len, layers=layers)
    if eps <= 0:
        return {view: {key: 0.0 for key in table} for view, table in counts.items()}
    noisy: dict[str, dict[str, float]] = {}
    scale = 1.0 / float(eps)
    for view, table in counts.items():
        noisy[view] = {}
        for key, val in table.items():
            noisy[view][key] = max(0.0, float(val) + float(rng.laplace(0.0, scale)))
    return noisy


def shuffle_signature_counts(counts: dict[str, dict[str, float]], rng: np.random.Generator) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for view, table in counts.items():
        keys = list(table.keys())
        vals = np.asarray([float(table[k]) for k in keys], dtype=float)
        if vals.size:
            vals = vals[rng.permutation(vals.size)]
        out[view] = {key: float(val) for key, val in zip(keys, vals)}
    return out


def signature_probs_for_bank(
    bank: list[dict],
    coords: np.ndarray,
    mn: np.ndarray,
    span: np.ndarray,
    counts: dict[str, dict[str, float]],
    *,
    grid: int,
    max_len: int,
    layers: int,
    pseudo_count: float,
    temperature: float,
    uniform_mix: float,
) -> list[np.ndarray]:
    probs = []
    for item in bank:
        cands = item.get("candidates", [])
        if not cands:
            probs.append(np.zeros(0, dtype=float))
            continue
        od = int(item.get("od_bin", 0))
        scores = []
        for _label, nodes in cands:
            pts = coords[np.asarray(nodes, dtype=int)]
            score = 0.0
            for view, key, weight in signature_views_for_points(pts, mn, span, od_bin=od, grid=grid, max_len=max_len, layers=layers):
                val = float(counts.get(view, {}).get(key, 0.0))
                score += float(weight) * math.log(max(val, 0.0) + float(pseudo_count))
            scores.append(score)
        p = softmax(np.asarray(scores, dtype=float) / max(float(temperature), 1e-9))
        mix = float(np.clip(uniform_mix, 0.0, 1.0))
        if p.size:
            p = (1.0 - mix) * p + mix * np.full(p.size, 1.0 / p.size, dtype=float)
            p = p / max(float(p.sum()), 1e-12)
        probs.append(p)
    return probs


def signature_count_diagnostics(counts: dict[str, dict[str, float]]) -> dict:
    out = {}
    for view, table in counts.items():
        vals = np.asarray(list(table.values()), dtype=float)
        out[view] = {
            "n_keys": int(len(table)),
            "sum": float(vals.sum()) if vals.size else 0.0,
            "positive_keys": int(np.sum(vals > 0)) if vals.size else 0,
            "max": float(vals.max()) if vals.size else 0.0,
        }
    return out


def align_probs(probs: list[np.ndarray], bank: list[dict]) -> list[np.ndarray]:
    out = []
    for idx, item in enumerate(bank):
        k = len(item.get("candidates", []))
        if k <= 0:
            out.append(np.zeros(0, dtype=float))
            continue
        p = np.asarray(probs[idx], dtype=float) if idx < len(probs) else np.zeros(0, dtype=float)
        if p.size != k or float(np.sum(p)) <= 0 or not np.all(np.isfinite(p)):
            p = np.zeros(k, dtype=float)
            p[0] = 1.0
        p = np.maximum(p, 0.0)
        p = p / max(float(p.sum()), 1e-12)
        out.append(p)
    return out


def public_base_probs(bank: list[dict]) -> list[np.ndarray]:
    out = []
    for item in bank:
        k = len(item.get("candidates", []))
        p = np.zeros(k, dtype=float)
        if k:
            p[0] = 1.0
        out.append(p)
    return out


def uniform_probs(bank: list[dict]) -> list[np.ndarray]:
    out = []
    for item in bank:
        k = len(item.get("candidates", []))
        out.append(np.full(k, 1.0 / k, dtype=float) if k else np.zeros(0, dtype=float))
    return out


def length_soft_probs(bank: list[dict], coords: np.ndarray, alpha_len: float) -> list[np.ndarray]:
    out = []
    for item in bank:
        cands = item.get("candidates", [])
        if not cands:
            out.append(np.zeros(0, dtype=float))
            continue
        base_len = max(float(probe.path_length(coords, cands[0][1])), 1e-12)
        rel = np.asarray([float(probe.path_length(coords, nodes)) / base_len for _label, nodes in cands], dtype=float)
        out.append(softmax(-float(alpha_len) * rel))
    return out


def length_hard_probs(bank: list[dict], coords: np.ndarray) -> list[np.ndarray]:
    out = []
    for item in bank:
        cands = item.get("candidates", [])
        if not cands:
            out.append(np.zeros(0, dtype=float))
            continue
        lens = [float(probe.path_length(coords, nodes)) for _label, nodes in cands]
        idx = int(np.argmin(lens))
        p = np.zeros(len(cands), dtype=float)
        p[idx] = 1.0
        out.append(p)
    return out


def per_request_cross_entropy(targets: list[np.ndarray], probs: list[np.ndarray]) -> np.ndarray:
    vals = []
    for q, p in zip(targets, probs):
        if q.size == 0 or p.size == 0:
            continue
        p = np.maximum(np.asarray(p, dtype=float), 1e-12)
        p = p / max(float(p.sum()), 1e-12)
        q = np.asarray(q, dtype=float)
        q = q / max(float(q.sum()), 1e-12)
        vals.append(-float(np.sum(q * np.log(p))))
    return np.asarray(vals, dtype=float)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> list[float | None]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return [None, None]
    means = []
    for _ in range(int(n_boot)):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        means.append(float(np.mean(sample)))
    lo, hi = np.percentile(np.asarray(means, dtype=float), [2.5, 97.5])
    return [float(lo), float(hi)]


def evaluate_prob_model(
    name: str,
    probs: list[np.ndarray],
    targets: list[np.ndarray],
    *,
    baseline_ce: np.ndarray | None,
    rng: np.random.Generator,
) -> dict:
    probs = align_probs(probs, [{"candidates": [None] * len(q)} for q in targets])
    ces = per_request_cross_entropy(targets, probs)
    entropies = []
    kls = []
    briers = []
    top1 = []
    target_argmax_prob = []
    expected_target_mass = []
    ranks = []
    for q, p in zip(targets, probs):
        if q.size == 0 or p.size == 0:
            continue
        q = q / max(float(q.sum()), 1e-12)
        p = np.maximum(p, 1e-12)
        p = p / max(float(p.sum()), 1e-12)
        ent = -float(np.sum(q[q > 0] * np.log(q[q > 0])))
        entropies.append(ent)
        ce = -float(np.sum(q * np.log(p)))
        kls.append(ce - ent)
        briers.append(float(np.sum((p - q) ** 2)))
        q_arg = int(np.argmax(q))
        p_arg = int(np.argmax(p))
        top1.append(float(q_arg == p_arg))
        target_argmax_prob.append(float(p[q_arg]))
        expected_target_mass.append(float(np.dot(q, p)))
        order = np.argsort(-p)
        rank = int(np.where(order == q_arg)[0][0]) + 1
        ranks.append(float(rank))
    out = {
        "name": str(name),
        "n": int(len(ces)),
        "cross_entropy": float(np.mean(ces)) if ces.size else None,
        "target_entropy": float(np.mean(entropies)) if entropies else None,
        "kl_to_target": float(np.mean(kls)) if kls else None,
        "brier": float(np.mean(briers)) if briers else None,
        "top1_agreement": float(np.mean(top1)) if top1 else None,
        "target_argmax_probability": float(np.mean(target_argmax_prob)) if target_argmax_prob else None,
        "expected_target_probability": float(np.mean(expected_target_mass)) if expected_target_mass else None,
        "mean_target_argmax_rank": float(np.mean(ranks)) if ranks else None,
        "per_request_cross_entropy": [float(x) for x in ces],
    }
    if baseline_ce is not None and baseline_ce.size == ces.size and ces.size:
        gain = baseline_ce - ces
        out.update(
            {
                "mean_ce_gain_vs_baseline": float(np.mean(gain)),
                "paired_win_rate_vs_baseline": float(np.mean(gain > 1e-9)),
                "paired_loss_rate_vs_baseline": float(np.mean(gain < -1e-9)),
                "ce_gain_ci95_vs_baseline": bootstrap_ci(gain, rng),
            }
        )
    return out


def summarize_and_print(name: str, metrics: dict) -> None:
    print(
        f"{name:46s} "
        f"ce={metrics.get('cross_entropy')} "
        f"kl={metrics.get('kl_to_target')} "
        f"top1={metrics.get('top1_agreement')} "
        f"p_arg={metrics.get('target_argmax_probability')} "
        f"gain={metrics.get('mean_ce_gain_vs_baseline')} "
        f"win={metrics.get('paired_win_rate_vs_baseline')}",
        flush=True,
    )


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
    parser.add_argument("--eps-list", default="0.4,1.6")
    parser.add_argument("--ridge-list", default="0.10,0.30")
    parser.add_argument("--alpha-len", type=float, default=1.0)
    parser.add_argument("--alpha-ps", type=float, default=0.7)
    parser.add_argument("--feature-weights", default=",".join(str(x) for x in fc.FEATURE_WEIGHTS))
    parser.add_argument("--label-grid", type=int, default=8)
    parser.add_argument("--k-eval-matches", type=int, default=8)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--length-match-weight", type=float, default=0.08)
    parser.add_argument("--match-tau", type=float, default=0.18)
    parser.add_argument("--label-tau", type=float, default=0.18)
    parser.add_argument("--label-trans-weight", type=float, default=0.65)
    parser.add_argument("--label-flow-weight", type=float, default=0.25)
    parser.add_argument("--label-length-weight", type=float, default=0.10)
    parser.add_argument("--target-source", choices=("matched_eval", "exact_eval"), default="matched_eval")
    parser.add_argument("--signature-grid", type=int, default=8)
    parser.add_argument("--signature-max-len", type=int, default=6)
    parser.add_argument("--signature-layers", type=int, default=6)
    parser.add_argument("--signature-pseudo-count", type=float, default=0.50)
    parser.add_argument("--signature-temp-list", default="1.0,2.0")
    parser.add_argument("--signature-mix-list", default="0.25,0.50")
    parser.add_argument("--out-name", default="route_choice_metric_probe.json")
    args = parser.parse_args()

    weights_raw = parse_float_list(args.feature_weights)
    weights = tuple(weights_raw[:3]) if len(weights_raw) >= 3 else fc.FEATURE_WEIGHTS
    probe.OUT_DIR.mkdir(parents=True, exist_ok=True)

    real, eval_real, road_coords, graph, mn, span, fit_bank = probe.build_candidate_bank(
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
    fit_bank = fc.filter_candidate_bank_by_base_stretch(fit_bank, road_coords, float(args.filter_candidate_stretch))
    train = real[: args.train_size]
    eval_bank = fit_bank
    if args.target_source == "exact_eval":
        eval_bank = build_exact_eval_candidate_bank(
            eval_real,
            train,
            road_coords,
            graph,
            mn,
            span,
            max_requests=args.max_requests,
            max_candidates=args.max_candidates,
            seed=args.seed,
            candidate_mode=args.candidate_mode,
            candidate_max_stretch=args.candidate_max_stretch,
            candidate_spur_trials=args.candidate_spur_trials,
        )
        eval_bank = fc.filter_candidate_bank_by_base_stretch(eval_bank, road_coords, float(args.filter_candidate_stretch))

    fit_model = fc.build_candidate_model(
        fit_bank,
        road_coords,
        alpha_len=float(args.alpha_len),
        alpha_ps=float(args.alpha_ps),
        weights=weights,
    )
    eval_model = fc.build_candidate_model(
        eval_bank,
        road_coords,
        alpha_len=float(args.alpha_len),
        alpha_ps=float(args.alpha_ps),
        weights=weights,
    )
    eval_moment = fc.average_private_moment(eval_real, fit_bank, fit_model, mn, span, weights=weights)
    public_mm, _a0, _p0 = fc.model_moment(np.zeros(int(fit_model["layout"]["dim"]), dtype=float), fit_model)
    if args.target_source == "exact_eval":
        target_payload = build_exact_route_choice_targets(
            eval_bank,
            road_coords,
            eval_real,
            mn,
            span,
            grid=int(args.label_grid),
            label_tau=float(args.label_tau),
            trans_weight=float(args.label_trans_weight),
            flow_weight=float(args.label_flow_weight),
            length_label_weight=float(args.label_length_weight),
        )
    else:
        target_payload = build_route_choice_targets(
            eval_bank,
            road_coords,
            eval_real,
            mn,
            span,
            grid=int(args.label_grid),
            k_eval_matches=int(args.k_eval_matches),
            endpoint_weight=float(args.endpoint_weight),
            length_weight=float(args.length_match_weight),
            match_tau=float(args.match_tau),
            label_tau=float(args.label_tau),
            trans_weight=float(args.label_trans_weight),
            flow_weight=float(args.label_flow_weight),
            length_label_weight=float(args.label_length_weight),
        )
    targets = target_payload["targets"]

    payload = {
        "config": {
            "seed": int(args.seed),
            "max_requests": int(args.max_requests),
            "max_candidates": int(args.max_candidates),
            "actual_fit_requests": int(len(fit_bank)),
            "actual_eval_requests": int(len(eval_bank)),
            "candidate_mode": str(args.candidate_mode),
            "candidate_max_stretch": float(args.candidate_max_stretch),
            "filter_candidate_stretch": float(args.filter_candidate_stretch),
            "candidate_cache_dir": str(args.candidate_cache_dir),
            "fit_candidate_family_hash": probe.candidate_family_hash(fit_bank),
            "eval_candidate_family_hash": probe.candidate_family_hash(eval_bank),
            "eps_list": parse_float_list(args.eps_list),
            "ridge_list": parse_float_list(args.ridge_list),
            "alpha_len": float(args.alpha_len),
            "alpha_ps": float(args.alpha_ps),
            "feature_weights": [float(x) for x in weights],
            "feature_dim": int(fit_model["layout"]["dim"]),
            "target_source": str(args.target_source),
            "label_grid": int(args.label_grid),
            "k_eval_matches": int(args.k_eval_matches),
            "endpoint_weight": float(args.endpoint_weight),
            "length_match_weight": float(args.length_match_weight),
            "match_tau": float(args.match_tau),
            "label_tau": float(args.label_tau),
            "label_trans_weight": float(args.label_trans_weight),
            "label_flow_weight": float(args.label_flow_weight),
            "label_length_weight": float(args.label_length_weight),
            "signature_grid": int(args.signature_grid),
            "signature_max_len": int(args.signature_max_len),
            "signature_layers": int(args.signature_layers),
            "signature_pseudo_count": float(args.signature_pseudo_count),
            "signature_temp_list": parse_float_list(args.signature_temp_list),
            "signature_mix_list": parse_float_list(args.signature_mix_list),
            "script": str(Path(__file__).resolve()),
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "candidate_diagnostics": {
            "fit_mean_candidates": float(np.mean([len(x["candidates"]) for x in fit_bank])) if fit_bank else 0.0,
            "fit_min_candidates": int(min([len(x["candidates"]) for x in fit_bank], default=0)),
            "fit_max_candidates": int(max([len(x["candidates"]) for x in fit_bank], default=0)),
            "eval_mean_candidates": float(np.mean([len(x["candidates"]) for x in eval_bank])) if eval_bank else 0.0,
            "eval_min_candidates": int(min([len(x["candidates"]) for x in eval_bank], default=0)),
            "eval_max_candidates": int(max([len(x["candidates"]) for x in eval_bank], default=0)),
            "build": getattr(probe.build_candidate_bank, "last_diagnostics", {}),
        },
        "target_diagnostics": {
            "summary": target_payload["summary"],
            "by_request": target_payload["diagnostics"],
        },
        "moment_diagnostics": {
            "public_prior_to_eval": fc.distances_to_eval_moment(public_mm, eval_moment),
        },
        "optimization": {},
        "signature_diagnostics": {},
        "variants": {},
    }

    variant_probs = {
        "public_base_onehot": public_base_probs(eval_bank),
        "public_uniform_candidate": uniform_probs(eval_bank),
        "public_length_hard": length_hard_probs(eval_bank, road_coords),
        "public_length_soft": length_soft_probs(eval_bank, road_coords, float(args.alpha_len)),
        "public_path_size_prior": fc.prior_probs(eval_model),
    }

    base_metrics = evaluate_prob_model(
        "public_base_onehot",
        variant_probs["public_base_onehot"],
        targets,
        baseline_ce=None,
        rng=np.random.default_rng(args.seed + 901),
    )
    baseline_ce = np.asarray(base_metrics["per_request_cross_entropy"], dtype=float)
    payload["variants"]["public_base_onehot"] = base_metrics
    summarize_and_print("public_base_onehot", base_metrics)

    for name in ["public_uniform_candidate", "public_length_hard", "public_length_soft", "public_path_size_prior"]:
        metrics = evaluate_prob_model(
            name,
            variant_probs[name],
            targets,
            baseline_ce=baseline_ce,
            rng=np.random.default_rng(args.seed + 902),
        )
        payload["variants"][name] = metrics
        summarize_and_print(name, metrics)

    train_signature_counts = build_signature_counts(
        train,
        mn,
        span,
        grid=int(args.signature_grid),
        max_len=int(args.signature_max_len),
        layers=int(args.signature_layers),
    )
    payload["signature_diagnostics"]["train_non_dp"] = signature_count_diagnostics(train_signature_counts)
    for temp in parse_float_list(args.signature_temp_list):
        for mix in parse_float_list(args.signature_mix_list):
            temp_tag = f"{temp:.2f}".replace(".", "p")
            mix_tag = f"{mix:.2f}".replace(".", "p")
            name = f"train_signature_NON_DP_temp_{temp_tag}_mix_{mix_tag}"
            probs = signature_probs_for_bank(
                eval_bank,
                road_coords,
                mn,
                span,
                train_signature_counts,
                grid=int(args.signature_grid),
                max_len=int(args.signature_max_len),
                layers=int(args.signature_layers),
                pseudo_count=float(args.signature_pseudo_count),
                temperature=float(temp),
                uniform_mix=float(mix),
            )
            metrics = evaluate_prob_model(
                name,
                probs,
                targets,
                baseline_ce=baseline_ce,
                rng=np.random.default_rng(args.seed + 904),
            )
            payload["variants"][name] = metrics
            summarize_and_print(name, metrics)

    for eps in parse_float_list(args.eps_list):
        if eps <= 0:
            continue
        dp_m = fc.dp_average_moment(
            train,
            fit_bank,
            fit_model,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 610),
            eps,
            weights=weights,
        )
        shuf_m = fc.shuffled_moment(dp_m, np.random.default_rng(args.seed + int(eps * 1000) + 611))
        eps_key = f"eps_{eps:.2f}"
        payload["moment_diagnostics"][eps_key] = {
            "dp_to_eval": fc.distances_to_eval_moment(dp_m, eval_moment),
            "shuffled_to_eval": fc.distances_to_eval_moment(shuf_m, eval_moment),
            "dp_to_public": fc.distances_to_eval_moment(dp_m, public_mm),
        }
        sig_counts = dp_signature_counts(
            train,
            mn,
            span,
            np.random.default_rng(args.seed + int(eps * 1000) + 710),
            eps,
            grid=int(args.signature_grid),
            max_len=int(args.signature_max_len),
            layers=int(args.signature_layers),
        )
        sig_shuffled = shuffle_signature_counts(sig_counts, np.random.default_rng(args.seed + int(eps * 1000) + 711))
        payload["signature_diagnostics"][eps_key] = {
            "dp": signature_count_diagnostics(sig_counts),
            "shuffled": signature_count_diagnostics(sig_shuffled),
        }
        for temp in parse_float_list(args.signature_temp_list):
            for mix in parse_float_list(args.signature_mix_list):
                temp_tag = f"{temp:.2f}".replace(".", "p")
                mix_tag = f"{mix:.2f}".replace(".", "p")
                for prefix, table in [("dp_signature", sig_counts), ("shuffled_dp_signature", sig_shuffled)]:
                    name = f"{prefix}_eps_{eps:.2f}_temp_{temp_tag}_mix_{mix_tag}"
                    probs = signature_probs_for_bank(
                        eval_bank,
                        road_coords,
                        mn,
                        span,
                        table,
                        grid=int(args.signature_grid),
                        max_len=int(args.signature_max_len),
                        layers=int(args.signature_layers),
                        pseudo_count=float(args.signature_pseudo_count),
                        temperature=float(temp),
                        uniform_mix=float(mix),
                    )
                    metrics = evaluate_prob_model(
                        name,
                        probs,
                        targets,
                        baseline_ce=baseline_ce,
                        rng=np.random.default_rng(args.seed + int(eps * 1000) + int(temp * 100) + int(mix * 1000) + 905),
                    )
                    payload["variants"][name] = metrics
                    summarize_and_print(name, metrics)
        for ridge in parse_float_list(args.ridge_list):
            ridge_tag = f"{ridge:.2f}".replace(".", "p")
            eps_tag = f"{eps:.2f}"
            fit = fc.fit_theta(dp_m, fit_model, ridge)
            shuf_fit = fc.fit_theta(shuf_m, fit_model, ridge)
            for prefix, fit_obj in [("dp_fc_grc", fit), ("shuffled_dp_fc_grc", shuf_fit)]:
                name = f"{prefix}_eps_{eps_tag}_ridge_{ridge_tag}"
                payload["optimization"][name] = {
                    "success": bool(fit_obj["success"]),
                    "message": str(fit_obj["message"]),
                    "nit": int(fit_obj["nit"]),
                    "objective": float(fit_obj["objective"]),
                    "theta_l2": float(fit_obj["theta_l2"]),
                    "moment_l2_to_target": float(fit_obj["moment_l2_to_target"]),
                }
                metrics = evaluate_prob_model(
                    name,
                    fc.model_moment(fit_obj["theta"], eval_model)[2],
                    targets,
                    baseline_ce=baseline_ce,
                    rng=np.random.default_rng(args.seed + int(eps * 1000) + int(ridge * 1000) + 903),
                )
                payload["variants"][name] = metrics
                summarize_and_print(name, metrics)

    out = probe.OUT_DIR / args.out_name
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
