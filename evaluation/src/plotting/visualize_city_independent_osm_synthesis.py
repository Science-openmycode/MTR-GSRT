"""Visualize per-dataset independent synthesis on each dataset's own base map.

Beijing and Porto use their own cached OSM road graphs.  Oldenburg is a
simulated road-network dataset without a separate OSM graph in the workspace, so
its full raw trajectory coverage is used as the road-like background.
"""
from __future__ import annotations

import csv
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
ARA_CODEX = ROOT / "ARA_codex"
DATA = ROOT / "data"
EXP_DATA = ROOT / "semdp_traj_paper" / "experiments" / "datasets"
FINAL = ROOT / "semdp_traj_paper" / "experiments" / "paper_comparison" / "final"
sys.path.insert(0, str(ARA_CODEX / "scripts"))

from cross_dataset_ara_mode_benchmark import (  # noqa: E402
    LENGTH_EDGES,
    SHAPE_VALUES,
    dp_anchors,
    fit_measurements,
    fit_pgm,
    load_geolife,
    load_oldenburg,
    load_porto,
    normalize_dataset,
    sample_exact_len,
)


N_TRAIN = 3000
N_EVAL = 1000
N_GEN = 220
N_LOAD = 10000
SEED = 42
K = 25
BEIJING_BBOX = (39.75, 40.15, 116.10, 116.65)
PORTO_BBOX = (41.10, 41.20, -8.70, -8.55)
FIG_DIR = ARA_CODEX / "results" / "figures" / "city_independent_osm"


def load_osm_pickle(city):
    candidates = [
        FINAL / "cross_city" / f"osm_{city}.pkl",
        FINAL / "osm_dbscan" / f"osm_{city}.pkl",
        ROOT / "ara_final" / "evidence" / "tables" / f"osm_cache_{city}.pkl",
    ]
    for path in candidates:
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f), path
    return None, None


def clipped(traj, bbox):
    arr = np.asarray(traj, dtype=float)
    if arr.ndim != 2 or len(arr) < 2:
        return arr[:0]
    mask = (bbox[0] <= arr[:, 0]) & (arr[:, 0] <= bbox[1]) & (bbox[2] <= arr[:, 1]) & (arr[:, 1] <= bbox[3])
    return arr[mask]


def bbox_of(trajs, pad=0.05):
    pts = np.vstack([np.asarray(t, dtype=float) for t in trajs if len(t) >= 2])
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    sp = np.maximum(mx - mn, 1e-9)
    return (mn[0] - pad * sp[0], mx[0] + pad * sp[0], mn[1] - pad * sp[1], mx[1] + pad * sp[1])


def normalize_with_meta(trajs):
    norm, meta = normalize_dataset(trajs)
    mn = np.asarray(meta["min"], dtype=float)
    mx = np.asarray(meta["max"], dtype=float)
    span = np.maximum(mx - mn, 1e-12)
    return norm, mn, span


def denorm(trajs, mn, span):
    return [np.asarray(t) * span + mn for t in trajs]


def osm_graph(osm_ways, bbox):
    coords = []
    edges = []
    index = {}

    def node_id(lat, lon):
        key = (round(float(lat), 6), round(float(lon), 6))
        if key not in index:
            index[key] = len(coords)
            coords.append([float(lat), float(lon)])
        return index[key]

    for way in osm_ways:
        geom = []
        for p in way.get("geometry", []):
            lat, lon = float(p["lat"]), float(p["lon"])
            if bbox[0] - 0.02 <= lat <= bbox[1] + 0.02 and bbox[2] - 0.02 <= lon <= bbox[3] + 0.02:
                geom.append((lat, lon))
        for (a_lat, a_lon), (b_lat, b_lon) in zip(geom[:-1], geom[1:]):
            a = node_id(a_lat, a_lon)
            b = node_id(b_lat, b_lon)
            if a != b:
                edges.append((a, b))
    arr = np.asarray(coords, dtype=float)
    graph = {i: [] for i in range(len(arr))}
    for a, b in edges:
        w = float(np.linalg.norm(arr[a] - arr[b]))
        graph[a].append((b, w))
        graph[b].append((a, w))
    return arr, graph


def raw_traj_graph(trajs, precision=1):
    coords = []
    index = {}
    graph = {}

    def node_id(pt):
        key = (round(float(pt[0]), precision), round(float(pt[1]), precision))
        if key not in index:
            index[key] = len(coords)
            coords.append([float(pt[0]), float(pt[1])])
            graph[index[key]] = []
        return index[key]

    for traj in trajs:
        arr = np.asarray(traj, dtype=float)
        for a, b in zip(arr[:-1], arr[1:]):
            ia, ib = node_id(a), node_id(b)
            if ia == ib:
                continue
            w = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
            graph[ia].append((ib, w))
            graph[ib].append((ia, w))
    return np.asarray(coords, dtype=float), graph


def largest_component(coords, graph):
    seen = set()
    best = []
    for node in range(len(coords)):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb, _ in graph.get(cur, []):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(comp) > len(best):
            best = comp
    keep = set(best)
    remap = {old: i for i, old in enumerate(best)}
    new_coords = coords[np.asarray(best, dtype=int)]
    new_graph = {remap[old]: [] for old in best}
    for old in best:
        for nb, w in graph.get(old, []):
            if nb in keep:
                new_graph[remap[old]].append((remap[nb], w))
    return new_coords, new_graph


def shortest_path(graph, coords, src, dst, max_visits=1000000):
    import heapq

    src, dst = int(src), int(dst)
    if src == dst:
        return None

    def heuristic(node):
        return float(np.linalg.norm(coords[node] - coords[dst]))

    dist = {src: 0.0}
    prev = {}
    heap = [(heuristic(src), 0.0, src)]
    visits = 0
    while heap and visits < max_visits:
        _, cost, node = heapq.heappop(heap)
        visits += 1
        if node == dst:
            break
        if cost > dist.get(node, float("inf")):
            continue
        for nb, w in graph.get(node, []):
            nc = cost + w
            if nc < dist.get(nb, float("inf")):
                dist[nb] = nc
                prev[nb] = node
                heapq.heappush(heap, (nc + heuristic(nb), nc, nb))
    if dst not in dist:
        return None
    out = [dst]
    cur = dst
    seen = {cur}
    while cur != src:
        cur = prev.get(cur)
        if cur is None or cur in seen:
            return None
        out.append(cur)
        seen.add(cur)
    out.reverse()
    return out


def resample(poly, n):
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 2:
        return poly
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] <= 1e-12:
        return np.repeat(poly[:1], n, axis=0)
    targets = np.linspace(0, cum[-1], n)
    out = []
    j = 0
    for t in targets:
        while j < len(seg) - 1 and cum[j + 1] < t:
            j += 1
        frac = 0.0 if seg[j] <= 1e-12 else (t - cum[j]) / seg[j]
        out.append(poly[j] * (1 - frac) + poly[j + 1] * frac)
    return np.asarray(out)


def synthesize_city(trajs, *, osm_ways=None, bbox=None, raw_graph=False):
    rng = np.random.default_rng(SEED)
    train = trajs[:N_TRAIN]
    norm_train, mn, span = normalize_with_meta(train)
    centers_norm = dp_anchors(norm_train, 0.20, rng)
    meas = fit_measurements(norm_train, centers_norm, rng, pgm=True)
    pgm = fit_pgm(meas, centers_norm, iters=60)
    probs = pgm.ravel() / pgm.sum()
    draws = rng.choice(pgm.size, size=N_GEN * 4, p=probs)

    centers_real = centers_norm * span + mn
    if osm_ways is not None and bbox is not None:
        road_coords, graph = osm_graph(osm_ways, bbox)
        road_coords, graph = largest_component(road_coords, graph)
        tree = cKDTree(road_coords)
        anchor_nodes = [int(tree.query(c)[1]) for c in centers_real]
    elif raw_graph:
        road_coords, graph = raw_traj_graph(trajs[:N_TRAIN + N_EVAL], precision=1)
        road_coords, graph = largest_component(road_coords, graph)
        tree = cKDTree(road_coords)
        anchor_nodes = [int(tree.query(c)[1]) for c in centers_real]
    else:
        road_coords, graph, anchor_nodes = None, None, None

    syn = []
    diagnostics = {
        "attempted": 0,
        "routed": 0,
        "skipped_same_anchor": 0,
        "skipped_unroutable": 0,
        "mean_route_nodes": None,
    }
    route_nodes = []
    n_l = len(LENGTH_EDGES) - 1
    n_s = len(SHAPE_VALUES)
    for draw in draws:
        if len(syn) >= N_GEN:
            break
        diagnostics["attempted"] += 1
        o = int(draw // (K * n_l * n_s))
        rem = int(draw % (K * n_l * n_s))
        d = int(rem // (n_l * n_s))
        rem = int(rem % (n_l * n_s))
        lb = int(rem // n_s)
        length = sample_exact_len(meas["len"], lb, rng)
        if road_coords is not None:
            if anchor_nodes[o] == anchor_nodes[d]:
                diagnostics["skipped_same_anchor"] += 1
                continue
            nodes = shortest_path(graph, road_coords, anchor_nodes[o], anchor_nodes[d])
            if nodes is None or len(nodes) < 2:
                diagnostics["skipped_unroutable"] += 1
                continue
            diagnostics["routed"] += 1
            route_nodes.append(len(nodes))
            poly = road_coords[np.asarray(nodes, dtype=int)]
        else:
            poly = np.linspace(centers_real[o], centers_real[d], max(2, length))
        syn.append(resample(poly, max(2, length)))
    if route_nodes:
        diagnostics["mean_route_nodes"] = float(np.mean(route_nodes))
    return syn, diagnostics


def draw_roads(ax, osm_ways, bbox):
    if osm_ways is None:
        return
    for way in osm_ways:
        seg = []
        for p in way.get("geometry", []):
            lat, lon = float(p["lat"]), float(p["lon"])
            if bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]:
                seg.append((lat, lon))
            else:
                if len(seg) >= 2:
                    arr = np.asarray(seg)
                    ax.plot(arr[:, 1], arr[:, 0], color="#ccd1d9", lw=0.18, alpha=0.45, zorder=0)
                seg = []
        if len(seg) >= 2:
            arr = np.asarray(seg)
            ax.plot(arr[:, 1], arr[:, 0], color="#ccd1d9", lw=0.18, alpha=0.45, zorder=0)


def draw_oldenburg_background(ax, trajs, bbox):
    rng = np.random.default_rng(123)
    idx = rng.choice(len(trajs), size=min(7000, len(trajs)), replace=False)
    for i in idx:
        t = np.asarray(trajs[int(i)], dtype=float)
        if len(t) >= 2:
            ax.plot(t[:, 1], t[:, 0], color="#d0d0d0", lw=0.12, alpha=0.05, zorder=0)


def draw_trajs(ax, trajs, bbox, color, title, *, bg_osm=None, bg_oldenburg=None):
    if bg_osm is not None:
        draw_roads(ax, bg_osm, bbox)
    if bg_oldenburg is not None:
        draw_oldenburg_background(ax, bg_oldenburg, bbox)
    rng = np.random.default_rng(99)
    idx = rng.choice(len(trajs), size=min(450, len(trajs)), replace=False)
    for i in idx:
        t = np.asarray(trajs[int(i)], dtype=float)
        if len(t) < 2:
            continue
        ax.plot(t[:, 1], t[:, 0], color=color, lw=0.75, alpha=0.22, zorder=2)
    ax.set_xlim(bbox[2], bbox[3])
    ax.set_ylim(bbox[0], bbox[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_city(name, real, syn, bbox, osm_ways=None, bg_oldenburg=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    draw_trajs(axes[0], real, bbox, "#1f5aa6", f"{name}: real trajectories", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    draw_trajs(axes[1], syn, bbox, "#d95f02", f"{name}: DP-PGM synthetic", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    fig.suptitle(f"{name}: independently trained/synthesized on its own base map", fontsize=13, fontweight="bold")
    out = FIG_DIR / f"{name.lower()}_real_vs_dppgm_on_basemap.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    outputs = {}
    for name, loader, bbox, osm_city in [
        ("Beijing", load_geolife, BEIJING_BBOX, "beijing"),
        ("Porto", load_porto, PORTO_BBOX, "porto"),
    ]:
        real = loader(limit=N_LOAD)
        osm, osm_path = load_osm_pickle(osm_city)
        syn, diag = synthesize_city(real, osm_ways=osm, bbox=bbox)
        out = plot_city(name, real, syn, bbox, osm_ways=osm)
        outputs[name] = {"figure": str(out), "osm": str(osm_path), "n_real": len(real), "n_syn": len(syn), "diagnostics": diag}
        print(out)

    old = load_oldenburg(limit=N_LOAD)
    bbox = bbox_of(old[:N_TRAIN + N_EVAL])
    syn, diag = synthesize_city(old, osm_ways=None, bbox=None, raw_graph=True)
    out = plot_city("Oldenburg", old, syn, bbox, osm_ways=None, bg_oldenburg=old)
    outputs["Oldenburg"] = {"figure": str(out), "osm": "raw trajectory coverage background", "n_real": len(old), "n_syn": len(syn), "diagnostics": diag}
    print(out)

    summary = ARA_CODEX / "results" / "city_independent_osm_visualizations.json"
    summary.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
