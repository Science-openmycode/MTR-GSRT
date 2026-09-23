"""DP route-structure potential experiment.

This script tests a trajectory-specific alternative to the generic OD/length/shape
PGM: keep the stable pre-PGM jointBucketExact sampler, but add a DP grid
coverage potential to the road decoder.  The potential is a fixed-support,
clipped one-trajectory contribution histogram with Laplace noise, then used only
as post-processing during public graph routing.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from cross_dataset_ara_mode_benchmark import (
    K,
    LENGTH_EDGES,
    SHAPE_VALUES,
    dp_anchors,
    fit_measurements,
    fit_pgm,
    od_dist_bin,
    sample_exact_len,
)
from compare_joint_vs_pgm_city_osm import compare_metrics
from visualize_city_independent_osm_synthesis import (
    ARA_CODEX,
    BEIJING_BBOX,
    N_EVAL,
    N_GEN,
    N_LOAD,
    N_TRAIN,
    PORTO_BBOX,
    SEED,
    bbox_of,
    draw_trajs,
    largest_component,
    load_geolife,
    load_oldenburg,
    load_osm_pickle,
    load_porto,
    normalize_with_meta,
    osm_graph,
    raw_traj_graph,
    resample,
    shortest_path,
)


FIG = ARA_CODEX / "results" / "figures" / "route_structure_potential"
EPS_CORRIDOR = 0.20
GRID = 32
LAMBDA = 0.85
OD_BINS = np.asarray([0.0, 0.12, 0.25, 0.50, 0.80, np.inf])
DIR_BINS = 8
SHRINK_RHO = 0.55
EXP_N_GEN = 140
TRANS_GRID = 12
TRANS_LAMBDA = 0.75
STOCHASTIC_LAMBDAS = (0.0, 0.35, 0.75, 1.10)
STOCHASTIC_NO_BASE_LAMBDAS = (0.35, 0.75, 1.10)
STOCHASTIC_BETA = 1.35
STOCHASTIC_ALPHA = 0.30
STOCHASTIC_TEMP = 0.65
DIVERSE_REROUTE_PENALTY = 2.5
HTRANSFORM_BETA = 1.10
HTRANSFORM_PROGRESS = 5.50
HTRANSFORM_LENGTH = 0.90
HTRANSFORM_VISIT = 3.00
HTRANSFORM_TEMP = 0.38
HTRANSFORM_MAX_MULT = 1.55
HTRANSFORM_MAX_TOTAL_MULT = 1.45


def cell_id(pt, mn, span, grid=GRID):
    q = (np.asarray(pt, dtype=float) - mn) / span
    r = min(max(int(q[0] * grid), 0), grid - 1)
    c = min(max(int(q[1] * grid), 0), grid - 1)
    return r, c


def dp_cell_potential(train, mn, span, rng, eps=EPS_CORRIDOR, grid=GRID):
    hist = np.zeros((grid, grid), dtype=float)
    for t in train:
        cells = []
        for p in np.asarray(t, dtype=float):
            cell = cell_id(p, mn, span, grid)
            if not cells or cells[-1] != cell:
                cells.append(cell)
        uniq = sorted(set(cells))
        if not uniq:
            continue
        weight = 1.0 / len(uniq)
        for r, c in uniq:
            hist[r, c] += weight
    noisy = np.maximum(hist + rng.laplace(0.0, 1.0 / eps, size=hist.shape), 0.0)
    if noisy.max() <= 1e-12:
        noisy[:] = 1.0
    pot = np.log1p(noisy)
    pot = pot / max(float(pot.max()), 1e-12)
    return pot


def od_bin_for_points(a, b, mn, span):
    na = (np.asarray(a, dtype=float) - mn) / span
    nb = (np.asarray(b, dtype=float) - mn) / span
    d = float(np.linalg.norm(na - nb))
    return min(max(int(np.searchsorted(OD_BINS, d, side="right") - 1), 0), len(OD_BINS) - 2)


def od_dir_bin_for_points(a, b, mn, span):
    ob = od_bin_for_points(a, b, mn, span)
    na = (np.asarray(a, dtype=float) - mn) / span
    nb = (np.asarray(b, dtype=float) - mn) / span
    v = nb - na
    angle = float(np.arctan2(v[0], v[1]))
    db = int(np.floor(((angle + np.pi) / (2 * np.pi)) * DIR_BINS)) % DIR_BINS
    return ob * DIR_BINS + db


def dp_od_cell_potentials(train, mn, span, rng, eps=EPS_CORRIDOR, grid=GRID):
    n_bins = len(OD_BINS) - 1
    hist = np.zeros((n_bins, grid, grid), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2:
            continue
        ob = od_bin_for_points(arr[0], arr[-1], mn, span)
        cells = []
        for p in arr:
            cell = cell_id(p, mn, span, grid)
            if not cells or cells[-1] != cell:
                cells.append(cell)
        uniq = sorted(set(cells))
        if not uniq:
            continue
        weight = 1.0 / len(uniq)
        for r, c in uniq:
            hist[ob, r, c] += weight
    noisy = np.maximum(hist + rng.laplace(0.0, 1.0 / eps, size=hist.shape), 0.0)
    for i in range(n_bins):
        if noisy[i].max() <= 1e-12:
            noisy[i] = 1.0
    pot = np.log1p(noisy)
    mx = np.maximum(pot.reshape(n_bins, -1).max(axis=1), 1e-12)
    pot = pot / mx[:, None, None]
    return pot


def dp_od_dir_cell_potentials(train, mn, span, rng, eps=EPS_CORRIDOR, grid=GRID):
    n_bins = (len(OD_BINS) - 1) * DIR_BINS
    hist = np.zeros((n_bins, grid, grid), dtype=float)
    for t in train:
        arr = np.asarray(t, dtype=float)
        if len(arr) < 2:
            continue
        ob = od_dir_bin_for_points(arr[0], arr[-1], mn, span)
        cells = []
        for p in arr:
            cell = cell_id(p, mn, span, grid)
            if not cells or cells[-1] != cell:
                cells.append(cell)
        uniq = sorted(set(cells))
        if not uniq:
            continue
        weight = 1.0 / len(uniq)
        for r, c in uniq:
            hist[ob, r, c] += weight
    noisy = np.maximum(hist + rng.laplace(0.0, 1.0 / eps, size=hist.shape), 0.0)
    for i in range(n_bins):
        if noisy[i].max() <= 1e-12:
            noisy[i] = 1.0
    pot = np.log1p(noisy)
    mx = np.maximum(pot.reshape(n_bins, -1).max(axis=1), 1e-12)
    pot = pot / mx[:, None, None]
    return pot


def shrink_potential(conditioned, global_potential, rho=SHRINK_RHO):
    return rho * conditioned + (1.0 - rho) * global_potential


def apply_potential_to_graph(graph, coords, potential, mn, span, lam=LAMBDA):
    bonus = np.zeros(len(coords), dtype=float)
    for i, p in enumerate(coords):
        r, c = cell_id(p, mn, span, potential.shape[0])
        bonus[i] = float(potential[r, c])
    weighted = {}
    for node, adj in graph.items():
        out = []
        for nb, w in adj:
            edge_bonus = 0.5 * (bonus[node] + bonus[nb])
            multiplier = 1.0 + lam * (1.0 - edge_bonus)
            out.append((nb, max(float(w) * multiplier, 1e-12)))
        weighted[node] = out
    return weighted


def dp_transition_potential(train, mn, span, rng, eps=EPS_CORRIDOR, grid=TRANS_GRID):
    n = grid * grid
    hist = np.zeros((n, n), dtype=float)
    for t in train:
        cells = []
        for p in np.asarray(t, dtype=float):
            r, c = cell_id(p, mn, span, grid)
            idx = r * grid + c
            if not cells or cells[-1] != idx:
                cells.append(idx)
        trans = [(a, b) for a, b in zip(cells[:-1], cells[1:]) if a != b]
        if not trans:
            continue
        counts = {}
        for a, b in trans:
            counts[(a, b)] = counts.get((a, b), 0.0) + 1.0
        total = sum(counts.values())
        for (a, b), v in counts.items():
            hist[a, b] += v / total
    noisy = np.maximum(hist + rng.laplace(0.0, 1.0 / eps, size=hist.shape), 0.0)
    row = noisy + 1e-6
    row = row / np.maximum(row.sum(axis=1, keepdims=True), 1e-12)
    global_col = row.mean(axis=0)
    global_col = global_col / max(global_col.sum(), 1e-12)
    log_row = np.log(row)
    log_global = np.log(global_col + 1e-12)
    return log_row - log_global[None, :]


def apply_transition_to_graph(graph, coords, trans_potential, mn, span, lam=TRANS_LAMBDA):
    grid = int(round(np.sqrt(trans_potential.shape[0])))
    node_cell = []
    for p in coords:
        r, c = cell_id(p, mn, span, grid)
        node_cell.append(r * grid + c)
    finite = trans_potential[np.isfinite(trans_potential)]
    lo, hi = np.quantile(finite, [0.05, 0.95]) if finite.size else (-1.0, 1.0)
    scale = max(hi - lo, 1e-6)
    weighted = {}
    for node, adj in graph.items():
        a = node_cell[node]
        out = []
        for nb, w in adj:
            b = node_cell[nb]
            z = float(np.clip((trans_potential[a, b] - lo) / scale, 0.0, 1.0))
            multiplier = 1.0 + lam * (1.0 - z)
            out.append((nb, max(float(w) * multiplier, 1e-12)))
        weighted[node] = out
    return weighted


def transition_node_cells(coords, mn, span, trans_potential):
    grid = int(round(np.sqrt(trans_potential.shape[0])))
    cells = []
    for p in coords:
        r, c = cell_id(p, mn, span, grid)
        cells.append(r * grid + c)
    return np.asarray(cells, dtype=int)


def transition_bonus_normalizer(trans_potential):
    finite = trans_potential[np.isfinite(trans_potential)]
    lo, hi = np.quantile(finite, [0.05, 0.95]) if finite.size else (-1.0, 1.0)
    return float(lo), max(float(hi - lo), 1e-6)


def path_length(coords, nodes):
    arr = coords[np.asarray(nodes, dtype=int)]
    if len(arr) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def path_transition_bonus(nodes, node_cells, trans_potential, lo, scale):
    if nodes is None or len(nodes) < 2:
        return 0.0
    vals = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        ca, cb = int(node_cells[a]), int(node_cells[b])
        vals.append(float(np.clip((trans_potential[ca, cb] - lo) / scale, 0.0, 1.0)))
    return float(np.mean(vals)) if vals else 0.0


def unique_candidate_paths(candidates):
    seen = set()
    out = []
    for label, nodes in candidates:
        if nodes is None or len(nodes) < 2:
            continue
        key = tuple(int(x) for x in nodes)
        if key in seen:
            continue
        seen.add(key)
        out.append((label, nodes))
    return out


def penalize_path_edges(graph, nodes, penalty=DIVERSE_REROUTE_PENALTY):
    if nodes is None or len(nodes) < 2:
        return graph
    penalized = {}
    blocked = set()
    for a, b in zip(nodes[:-1], nodes[1:]):
        blocked.add((int(a), int(b)))
        blocked.add((int(b), int(a)))
    for node, adj in graph.items():
        out = []
        for nb, w in adj:
            mult = penalty if (int(node), int(nb)) in blocked else 1.0
            out.append((nb, float(w) * mult))
        penalized[node] = out
    return penalized


def sample_stochastic_bridge_path(graphs, coords, src, dst, rng, node_cells, trans_potential, lo, scale, *, reroute=False):
    candidates = []
    for label, graph in graphs:
        first = shortest_path(graph, coords, src, dst)
        candidates.append((label, first))
        if reroute and first is not None:
            alt_graph = penalize_path_edges(graph, first)
            candidates.append((label + "+penalty", shortest_path(alt_graph, coords, src, dst)))
    candidates = unique_candidate_paths(candidates)
    if not candidates:
        return None, None, 0

    direct = max(float(np.linalg.norm(coords[int(src)] - coords[int(dst)])), 1e-12)
    scores = []
    for _, nodes in candidates:
        rel_len = path_length(coords, nodes) / direct
        bonus = path_transition_bonus(nodes, node_cells, trans_potential, lo, scale)
        scores.append(STOCHASTIC_BETA * bonus - STOCHASTIC_ALPHA * rel_len)
    scores = np.asarray(scores, dtype=float)
    logits = (scores - scores.max()) / max(STOCHASTIC_TEMP, 1e-9)
    probs = np.exp(logits)
    probs = probs / probs.sum()
    idx = int(rng.choice(len(candidates), p=probs))
    return candidates[idx][1], candidates[idx][0], len(candidates)


def sample_h_transform_bridge_path(graph, coords, src, dst, rng, node_cells, trans_potential, lo, scale):
    base = shortest_path(graph, coords, src, dst)
    if base is None or len(base) < 2:
        return None, None, 0
    direct = max(float(np.linalg.norm(coords[int(src)] - coords[int(dst)])), 1e-12)
    max_steps = max(int(len(base) * HTRANSFORM_MAX_MULT), len(base) + 8)
    max_total = max(int(len(base) * HTRANSFORM_MAX_TOTAL_MULT), len(base) + 4)
    cur = int(src)
    dst = int(dst)
    path = [cur]
    visits = {cur: 1}
    for _ in range(max_steps):
        if cur == dst:
            break
        adj = graph.get(cur, [])
        if not adj:
            break
        cur_dist = float(np.linalg.norm(coords[cur] - coords[dst]))
        scores = []
        nbs = []
        for nb, w in adj:
            nb = int(nb)
            nb_dist = float(np.linalg.norm(coords[nb] - coords[dst]))
            ca, cb = int(node_cells[cur]), int(node_cells[nb])
            bonus = float(np.clip((trans_potential[ca, cb] - lo) / scale, 0.0, 1.0))
            progress = (cur_dist - nb_dist) / direct
            revisit = visits.get(nb, 0)
            score = (
                HTRANSFORM_BETA * bonus
                + HTRANSFORM_PROGRESS * progress
                - HTRANSFORM_LENGTH * float(w) / direct
                - HTRANSFORM_VISIT * revisit
            )
            scores.append(score)
            nbs.append(nb)
        scores = np.asarray(scores, dtype=float)
        logits = (scores - scores.max()) / max(HTRANSFORM_TEMP, 1e-9)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        cur = int(rng.choice(np.asarray(nbs, dtype=int), p=probs))
        path.append(cur)
        visits[cur] = visits.get(cur, 0) + 1
        if cur == dst:
            break
    if path[-1] != dst:
        tail = shortest_path(graph, coords, path[-1], dst)
        if tail is None or len(tail) < 2:
            return base, "fallback_base", 1
        if len(path) + len(tail) - 1 > max_total:
            return base, "fallback_base", 1
        path.extend(tail[1:])
    compact = []
    for node in path:
        if not compact or compact[-1] != node:
            compact.append(node)
    return compact, "h_transform", len(compact)


def prepare_graph(real, bbox=None, osm_ways=None, raw_graph=False):
    if osm_ways is not None and bbox is not None:
        coords, graph = osm_graph(osm_ways, bbox)
    elif raw_graph:
        # Diagnostic fallback for Oldenburg only. This graph is derived from the
        # training split, so it avoids eval leakage, but it is not a public OSM
        # graph and should not be described as zero-budget public post-processing.
        coords, graph = raw_traj_graph(real[:N_TRAIN], precision=1)
    else:
        raise ValueError("Need OSM or raw graph")
    return largest_component(coords, graph)


def normalize_with_public_bbox(trajs, bbox):
    if bbox is None:
        return normalize_with_meta(trajs)
    mn = np.asarray([bbox[0], bbox[2]], dtype=float)
    mx = np.asarray([bbox[1], bbox[3]], dtype=float)
    span = np.maximum(mx - mn, 1e-12)
    return [(np.asarray(t, dtype=float) - mn) / span for t in trajs], mn, span


def sample_joint_draws(norm_train, centers_norm, rng, attempts=EXP_N_GEN * 4, *, pgm=False):
    meas = fit_measurements(norm_train, centers_norm, rng, pgm=pgm)
    n_s = len(SHAPE_VALUES)
    draws = []
    if pgm:
        n_l = len(LENGTH_EDGES) - 1
        model = fit_pgm(meas, centers_norm, iters=60)
        probs = model.ravel() / max(model.sum(), 1e-12)
        flat = rng.choice(model.size, size=attempts, p=probs)
        for draw in flat:
            o = int(draw // (K * n_l * n_s))
            rem = int(draw % (K * n_l * n_s))
            d = int(rem // (n_l * n_s))
            rem = int(rem % (n_l * n_s))
            draws.append((o, d, int(rem // n_s)))
    else:
        od_flat = rng.choice(K * K, size=attempts, p=meas["od"].ravel())
        for odx in od_flat:
            o, d = int(odx // K), int(odx % K)
            probs = meas["joint"][od_dist_bin(o, d, centers_norm)].ravel()
            probs = probs / probs.sum()
            lsx = int(rng.choice(probs.size, p=probs))
            draws.append((o, d, int(lsx // n_s)))
    return draws, meas


def synthesize(real, method, *, bbox=None, osm_ways=None, raw_graph=False):
    rng = np.random.default_rng(SEED)
    train = real[:N_TRAIN]
    norm_train, mn, span = normalize_with_public_bbox(train, bbox)
    centers_norm = dp_anchors(norm_train, 0.20, rng)
    centers_real = centers_norm * span + mn
    draws, meas = sample_joint_draws(norm_train, centers_norm, rng, pgm=(method == "dpPGM"))
    route_specs = []
    for o, d, lb in draws:
        route_specs.append((o, d, sample_exact_len(meas["len"], lb, rng)))
    road_coords, graph = prepare_graph(real, bbox=bbox, osm_ways=osm_ways, raw_graph=raw_graph)
    tree = cKDTree(road_coords)
    anchor_nodes = [int(tree.query(c)[1]) for c in centers_real]
    potential = None
    od_potentials = None
    od_dir_potentials = None
    global_for_shrink = None
    route_graph = graph
    trans_potential = None
    transition_route_graphs = None
    transition_cells = None
    transition_lo_scale = None
    if method == "corridorPotential":
        potential = dp_cell_potential(train, mn, span, rng)
    elif method == "odCorridorPotential":
        od_potentials = dp_od_cell_potentials(train, mn, span, rng)
    elif method == "hierOdDirPotential":
        global_for_shrink = dp_cell_potential(train, mn, span, rng)
        od_dir_potentials = dp_od_dir_cell_potentials(train, mn, span, rng)
    elif method == "transitionBridge":
        trans_potential = dp_transition_potential(train, mn, span, rng)
        route_graph = apply_transition_to_graph(graph, road_coords, trans_potential, mn, span)
    elif method in {"stochasticTransitionBridge", "stochasticTransitionNoBase", "diverseStochasticBridge", "hTransformBridge"}:
        trans_potential = dp_transition_potential(train, mn, span, rng)
        transition_route_graphs = []
        lam_set = STOCHASTIC_NO_BASE_LAMBDAS if method == "stochasticTransitionNoBase" else STOCHASTIC_LAMBDAS
        for lam in lam_set:
            if lam <= 1e-12:
                transition_route_graphs.append((f"lambda={lam:.2f}", graph))
            else:
                transition_route_graphs.append((f"lambda={lam:.2f}", apply_transition_to_graph(graph, road_coords, trans_potential, mn, span, lam=lam)))
        transition_cells = transition_node_cells(road_coords, mn, span, trans_potential)
        transition_lo_scale = transition_bonus_normalizer(trans_potential)
    if potential is not None:
        route_graph = apply_potential_to_graph(graph, road_coords, potential, mn, span)
    od_route_graphs = {}

    syn = []
    route_nodes = []
    diag = {"method": method, "attempted": 0, "routed": 0, "skipped_same_anchor": 0, "skipped_unroutable": 0}
    selected_graphs = {}
    candidate_counts = []
    for o, d, length in route_specs:
        if len(syn) >= EXP_N_GEN:
            break
        diag["attempted"] += 1
        if anchor_nodes[o] == anchor_nodes[d]:
            diag["skipped_same_anchor"] += 1
            continue
        if od_potentials is not None:
            ob = od_bin_for_points(centers_real[o], centers_real[d], mn, span)
            if ob not in od_route_graphs:
                od_route_graphs[ob] = apply_potential_to_graph(graph, road_coords, od_potentials[ob], mn, span)
            nodes = shortest_path(od_route_graphs[ob], road_coords, anchor_nodes[o], anchor_nodes[d])
        elif od_dir_potentials is not None:
            ob = od_dir_bin_for_points(centers_real[o], centers_real[d], mn, span)
            if ob not in od_route_graphs:
                shrunk = shrink_potential(od_dir_potentials[ob], global_for_shrink)
                od_route_graphs[ob] = apply_potential_to_graph(graph, road_coords, shrunk, mn, span)
            nodes = shortest_path(od_route_graphs[ob], road_coords, anchor_nodes[o], anchor_nodes[d])
        elif transition_route_graphs is not None:
            if method == "hTransformBridge":
                nodes, selected, n_cand = sample_h_transform_bridge_path(
                    graph,
                    road_coords,
                    anchor_nodes[o],
                    anchor_nodes[d],
                    rng,
                    transition_cells,
                    trans_potential,
                    transition_lo_scale[0],
                    transition_lo_scale[1],
                )
            else:
                nodes, selected, n_cand = sample_stochastic_bridge_path(
                    transition_route_graphs,
                    road_coords,
                    anchor_nodes[o],
                    anchor_nodes[d],
                    rng,
                    transition_cells,
                    trans_potential,
                    transition_lo_scale[0],
                    transition_lo_scale[1],
                    reroute=(method == "diverseStochasticBridge"),
                )
            candidate_counts.append(n_cand)
            selected_graphs[selected] = selected_graphs.get(selected, 0) + 1
        else:
            nodes = shortest_path(route_graph, road_coords, anchor_nodes[o], anchor_nodes[d])
        if nodes is None or len(nodes) < 2:
            diag["skipped_unroutable"] += 1
            continue
        diag["routed"] += 1
        route_nodes.append(len(nodes))
        syn.append(resample(road_coords[np.asarray(nodes, dtype=int)], max(2, length)))
    diag["mean_route_nodes"] = float(np.mean(route_nodes)) if route_nodes else None
    if candidate_counts:
        diag["mean_candidate_paths"] = float(np.mean(candidate_counts))
        diag["selected_candidate_graphs"] = selected_graphs
    diag["epsilon_extra_corridor"] = EPS_CORRIDOR if method in {
        "corridorPotential",
        "odCorridorPotential",
        "hierOdDirPotential",
        "transitionBridge",
        "stochasticTransitionBridge",
        "stochasticTransitionNoBase",
        "diverseStochasticBridge",
        "hTransformBridge",
    } else 0.0
    diag["paired_route_specs"] = True
    diag["public_bbox_normalization"] = bbox is not None
    diag["raw_graph_diagnostic_only"] = bool(raw_graph)
    return syn, diag


def plot_city(name, real, baseline, global_pot, od_pot, hier_pot, trans, bbox, *, osm_ways=None, bg_oldenburg=None):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 6, figsize=(24, 5.2), constrained_layout=True)
    draw_trajs(axes[0], real, bbox, "#1f5aa6", f"{name}: real", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    draw_trajs(axes[1], baseline, bbox, "#2ca25f", f"{name}: jointBucketExact", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    draw_trajs(axes[2], global_pot, bbox, "#d95f02", f"{name}: global potential", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    draw_trajs(axes[3], od_pot, bbox, "#7b3294", f"{name}: OD-conditioned potential", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    draw_trajs(axes[4], hier_pot, bbox, "#b2182b", f"{name}: hierarchical OD-dir", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    draw_trajs(axes[5], trans, bbox, "#542788", f"{name}: transition bridge", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    fig.suptitle(f"{name}: DP route potential variants", fontsize=13, fontweight="bold")
    out = FIG / f"{name.lower()}_route_potential_variants.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def run_city(name, loader, bbox=None, osm_city=None, raw_graph=False):
    real = loader(limit=N_LOAD)
    if bbox is None:
        bbox = bbox_of(real[: N_TRAIN + N_EVAL])
    osm, osm_path = load_osm_pickle(osm_city) if osm_city else (None, None)
    joint, joint_diag = synthesize(real, "jointBucketExact", bbox=bbox, osm_ways=osm, raw_graph=raw_graph)
    corridor, corridor_diag = synthesize(real, "corridorPotential", bbox=bbox, osm_ways=osm, raw_graph=raw_graph)
    od_corridor, od_diag = synthesize(real, "odCorridorPotential", bbox=bbox, osm_ways=osm, raw_graph=raw_graph)
    hier, hier_diag = synthesize(real, "hierOdDirPotential", bbox=bbox, osm_ways=osm, raw_graph=raw_graph)
    trans, trans_diag = synthesize(real, "transitionBridge", bbox=bbox, osm_ways=osm, raw_graph=raw_graph)
    fig = plot_city(name, real, joint, corridor, od_corridor, hier, trans, bbox, osm_ways=osm, bg_oldenburg=real if raw_graph else None)
    eval_real = real[N_TRAIN : N_TRAIN + N_EVAL]
    context = real[: N_TRAIN + N_EVAL]
    return {
        "figure": str(fig),
        "osm": str(osm_path) if osm_path else "train-split raw graph diagnostic background",
        "n_real_loaded": len(real),
        "n_syn": {
            "jointBucketExact": len(joint),
            "corridorPotential": len(corridor),
            "odCorridorPotential": len(od_corridor),
            "hierOdDirPotential": len(hier),
            "transitionBridge": len(trans),
        },
        "diagnostics": {
            "jointBucketExact": joint_diag,
            "corridorPotential": corridor_diag,
            "odCorridorPotential": od_diag,
            "hierOdDirPotential": hier_diag,
            "transitionBridge": trans_diag,
        },
        "metrics": {
            "jointBucketExact": compare_metrics(eval_real, joint, context),
            "corridorPotential": compare_metrics(eval_real, corridor, context),
            "odCorridorPotential": compare_metrics(eval_real, od_corridor, context),
            "hierOdDirPotential": compare_metrics(eval_real, hier, context),
            "transitionBridge": compare_metrics(eval_real, trans, context),
        },
    }


def plot_metrics(results):
    keys = [
        "spatial_jsd_multiscale",
        "trip_jsd",
        "len_jsd",
        "step_jsd",
        "shape_jsd",
        "coverage_f1_24",
        "hot_cell_recall_48",
        "transition_jsd_24",
        "radial_jsd",
        "center_periphery_l1",
        "nn_dtw",
        "nn_hausdorff",
    ]
    fig, axes = plt.subplots(len(results), 1, figsize=(13, 8), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (city, payload) in zip(axes, results.items()):
        x = np.arange(len(keys))
        width = 0.36
        for off, method, color in [
            (-1.5 * width, "jointBucketExact", "#2ca25f"),
            (-0.5 * width, "corridorPotential", "#d95f02"),
            (0.5 * width, "odCorridorPotential", "#7b3294"),
            (1.5 * width, "hierOdDirPotential", "#b2182b"),
            (2.5 * width, "transitionBridge", "#542788"),
        ]:
            vals = [payload["metrics"][method][k] for k in keys]
            ax.bar(x + off, vals, width, label=method, color=color, alpha=0.85)
        ax.set_title(city, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")
    out = FIG / "route_structure_potential_metric_bars.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    results = {
        "Beijing": run_city("Beijing", load_geolife, BEIJING_BBOX, "beijing"),
        "Porto": run_city("Porto", load_porto, PORTO_BBOX, "porto"),
        "Oldenburg": run_city("Oldenburg", load_oldenburg, None, None, raw_graph=True),
    }
    metric_fig = plot_metrics(results)
    out = ARA_CODEX / "results" / "route_structure_potential_experiment.json"
    out.write_text(json.dumps({"results": results, "metric_figure": str(metric_fig)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)
    for payload in results.values():
        print(payload["figure"])
    print(metric_fig)


if __name__ == "__main__":
    main()
