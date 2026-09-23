"""Focused DP transition-bridge experiment.

This is a focused follow-up to route_structure_potential_experiment.py.  It
compares only the stable jointBucketExact sampler against a DP clipped
fixed-support transition artifact used as a bridge/energy decoder.  The second
variant uses the same DP artifact but samples among several public-graph path
candidates to test whether a maximum-entropy-style decoder avoids MAP collapse.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_joint_vs_pgm_city_osm import compare_metrics
from route_structure_potential_experiment import (
    BEIJING_BBOX,
    FIG,
    PORTO_BBOX,
    ARA_CODEX,
    N_EVAL,
    N_TRAIN,
    bbox_of,
    draw_trajs,
    load_geolife,
    load_oldenburg,
    load_osm_pickle,
    load_porto,
    prepare_graph,
    shortest_path,
    synthesize,
)
from scipy.spatial import cKDTree


METHODS = [
    ("jointBucketExact", "#2ca25f"),
    ("dpPGM", "#8c8c8c"),
    ("corridorPotential", "#7570b3"),
    ("odCorridorPotential", "#1b9e77"),
    ("hierOdDirPotential", "#e7298a"),
    ("transitionBridge", "#542788"),
    ("stochasticTransitionBridge", "#d95f02"),
    ("stochasticTransitionNoBase", "#b2182b"),
    ("diverseStochasticBridge", "#1f78b4"),
    ("hTransformBridge", "#33a02c"),
]


def plot_city(name, real, synths, bbox, *, osm_ways=None, bg_oldenburg=None):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 1 + len(METHODS), figsize=(4.2 * (1 + len(METHODS)), 5.2), constrained_layout=True)
    draw_trajs(axes[0], real, bbox, "#1f5aa6", f"{name}: real", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    for ax, (method, color) in zip(axes[1:], METHODS):
        draw_trajs(ax, synths[method], bbox, color, f"{name}: {method}", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    fig.suptitle(f"{name}: full ARA route-structure comparison", fontsize=13, fontweight="bold")
    out = FIG / f"{name.lower()}_transition_bridge_focused.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def js_from_counts(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.sum() <= 0:
        a = np.ones_like(a)
    if b.sum() <= 0:
        b = np.ones_like(b)
    a = a / a.sum()
    b = b / b.sum()
    m = 0.5 * (a + b)
    mask_a = a > 0
    mask_b = b > 0
    kl_a = np.sum(a[mask_a] * np.log2(a[mask_a] / m[mask_a]))
    kl_b = np.sum(b[mask_b] * np.log2(b[mask_b] / m[mask_b]))
    return float(0.5 * (kl_a + kl_b))


def build_edge_index(graph):
    edges = {}
    edge_list = []
    for a, adj in graph.items():
        for b, _ in adj:
            key = (min(int(a), int(b)), max(int(a), int(b)))
            if key not in edges:
                edges[key] = len(edge_list)
                edge_list.append(key)
    return edges, edge_list


def build_edge_midpoint_tree(edge_list, road_coords):
    mids = []
    for a, b in edge_list:
        mids.append(0.5 * (road_coords[int(a)] + road_coords[int(b)]))
    return cKDTree(np.asarray(mids, dtype=float)) if mids else None


def trace_to_edge_ids(traj, edge_tree):
    arr = np.asarray(traj, dtype=float)
    if len(arr) < 2:
        return []
    mids = 0.5 * (arr[:-1] + arr[1:])
    out = []
    for p in mids:
        out.append(int(edge_tree.query(p)[1]))
    return out


def edge_metric_suite(real_eval, syn, context, road_coords, graph):
    edge_index, edge_list = build_edge_index(graph)
    n_edges = len(edge_list)
    if n_edges == 0:
        return {
            "edge_flow_jsd": 1.0,
            "edge_coverage_f1": 0.0,
            "edge_transition_jsd": 1.0,
            "od_corridor_jsd": 1.0,
            "edge_unique_ratio": 0.0,
        }
    edge_tree = build_edge_midpoint_tree(edge_list, road_coords)
    if edge_tree is None:
        return {
            "edge_flow_jsd": 1.0,
            "edge_coverage_f1": 0.0,
            "edge_transition_jsd": 1.0,
            "od_corridor_jsd": 1.0,
            "edge_unique_ratio": 0.0,
        }

    def edge_sequences(trajs):
        return [trace_to_edge_ids(t, edge_tree) for t in trajs if len(t) >= 2]

    real_seq = edge_sequences(real_eval)
    syn_seq = edge_sequences(syn)

    def flow(seqs):
        h = np.zeros(n_edges, dtype=float)
        for seq in seqs:
            for e in seq:
                h[e] += 1.0
        return h

    real_flow = flow(real_seq)
    syn_flow = flow(syn_seq)
    real_active = set(np.flatnonzero(real_flow > 0))
    syn_active = set(np.flatnonzero(syn_flow > 0))
    if real_active and syn_active:
        precision = len(real_active & syn_active) / len(syn_active)
        recall = len(real_active & syn_active) / len(real_active)
        edge_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    else:
        edge_f1 = 0.0

    top_edges = np.argsort(flow(edge_sequences(context)))[-512:]
    top_map = {int(e): i for i, e in enumerate(top_edges)}

    def transition_hist(seqs):
        h = np.zeros((len(top_edges), len(top_edges)), dtype=float)
        for seq in seqs:
            compact = []
            for e in seq:
                if e in top_map and (not compact or compact[-1] != e):
                    compact.append(e)
            for a, b in zip(compact[:-1], compact[1:]):
                h[top_map[a], top_map[b]] += 1.0
        return h

    def od_corridor_hist(trajs, seqs):
        h = np.zeros((5, len(top_edges)), dtype=float)
        pts = np.vstack(context)
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = np.maximum(mx - mn, 1e-12)
        bins = np.asarray([0.0, 0.12, 0.25, 0.50, 0.80, np.inf])
        for t, seq in zip(trajs, seqs):
            arr = np.asarray(t, dtype=float)
            if len(arr) < 2:
                continue
            d = float(np.linalg.norm((arr[-1] - arr[0]) / span))
            ob = min(max(int(np.searchsorted(bins, d, side="right") - 1), 0), 4)
            for e in seq:
                if e in top_map:
                    h[ob, top_map[e]] += 1.0
        return h

    return {
        "edge_flow_jsd": js_from_counts(real_flow, syn_flow),
        "edge_coverage_f1": float(edge_f1),
        "edge_transition_jsd": js_from_counts(transition_hist(real_seq), transition_hist(syn_seq)),
        "od_corridor_jsd": js_from_counts(od_corridor_hist(real_eval, real_seq), od_corridor_hist(syn, syn_seq)),
        "edge_unique_ratio": float(len(syn_active) / max(len(syn_seq), 1)),
    }


def run_city(name, loader, bbox=None, osm_city=None, raw_graph=False):
    real = loader(limit=10000)
    if bbox is None:
        bbox = bbox_of(real[: N_TRAIN + N_EVAL])
    osm, osm_path = load_osm_pickle(osm_city) if osm_city else (None, None)
    synths = {}
    diagnostics = {}
    for method, _ in METHODS:
        synths[method], diagnostics[method] = synthesize(real, method, bbox=bbox, osm_ways=osm, raw_graph=raw_graph)
    fig = plot_city(name, real, synths, bbox, osm_ways=osm, bg_oldenburg=real if raw_graph else None)
    eval_real = real[N_TRAIN : N_TRAIN + N_EVAL]
    context = real[: N_TRAIN + N_EVAL]
    road_coords, graph = prepare_graph(real, bbox=bbox, osm_ways=osm, raw_graph=raw_graph)
    edge_metrics = {
        method: edge_metric_suite(eval_real, synths[method], context, road_coords, graph)
        for method, _ in METHODS
    }
    return {
        "figure": str(fig),
        "osm": str(osm_path) if osm_path else "train-split raw graph diagnostic background",
        "n_real_loaded": len(real),
        "n_syn": {method: len(synths[method]) for method, _ in METHODS},
        "diagnostics": diagnostics,
        "metrics": {
            method: {**compare_metrics(eval_real, synths[method], context), **edge_metrics[method]}
            for method, _ in METHODS
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
        "edge_flow_jsd",
        "edge_coverage_f1",
        "edge_transition_jsd",
        "od_corridor_jsd",
        "edge_unique_ratio",
        "nn_dtw",
        "nn_hausdorff",
    ]
    fig, axes = plt.subplots(len(results), 1, figsize=(18, 9), constrained_layout=True)
    for ax, (city, payload) in zip(axes, results.items()):
        x = np.arange(len(keys))
        width = 0.085
        offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width
        for off, (method, color) in zip(offsets, METHODS):
            vals = [payload["metrics"][method][k] for k in keys]
            ax.bar(x + off, vals, width, label=method, color=color, alpha=0.85)
        ax.set_title(city, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")
    out = FIG / "transition_bridge_metric_bars.png"
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
    out = ARA_CODEX / "results" / "transition_bridge_experiment.json"
    out.write_text(json.dumps({"results": results, "metric_figure": str(metric_fig)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)
    for payload in results.values():
        print(payload["figure"])
    print(metric_fig)


if __name__ == "__main__":
    main()
