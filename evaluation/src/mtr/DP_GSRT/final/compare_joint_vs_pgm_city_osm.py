"""Compare the last pre-PGM joint model against the DP-PGM model per city.

The comparison is deliberately visual-first: every dataset is trained and
synthesized independently, then routed on its own OSM/raw graph, and plotted as
real vs jointBucketExact vs dpPGM on the same city base map.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cross_dataset_ara_mode_benchmark import (
    K,
    LENGTH_EDGES,
    SHAPE_VALUES,
    dp_anchors,
    fit_measurements,
    fit_pgm,
    js_hist,
    metrics as normalized_metrics,
    od_dist_bin,
    sample_exact_len,
)
from visualize_city_independent_osm_synthesis import (
    ARA_CODEX,
    BEIJING_BBOX,
    FIG_DIR,
    N_EVAL,
    N_GEN,
    N_LOAD,
    N_TRAIN,
    PORTO_BBOX,
    SEED,
    bbox_of,
    draw_roads,
    draw_trajs,
    load_geolife,
    load_oldenburg,
    load_osm_pickle,
    load_porto,
    normalize_with_meta,
    osm_graph,
    raw_traj_graph,
    largest_component,
    resample,
    shortest_path,
)
from scipy.spatial import cKDTree


OUT_DIR = FIG_DIR / "joint_vs_pgm"


def synthesize_variant(trajs, method, *, osm_ways=None, bbox=None, raw_graph=False):
    rng = np.random.default_rng(SEED)
    train = trajs[:N_TRAIN]
    norm_train, mn, span = normalize_with_meta(train)
    centers_norm = dp_anchors(norm_train, 0.20, rng)
    meas = fit_measurements(norm_train, centers_norm, rng, pgm=(method == "dpPGM"))

    centers_real = centers_norm * span + mn
    if osm_ways is not None and bbox is not None:
        road_coords, graph = osm_graph(osm_ways, bbox)
        road_coords, graph = largest_component(road_coords, graph)
    elif raw_graph:
        road_coords, graph = raw_traj_graph(trajs[: N_TRAIN + N_EVAL], precision=1)
        road_coords, graph = largest_component(road_coords, graph)
    else:
        road_coords, graph = None, None

    if road_coords is not None:
        tree = cKDTree(road_coords)
        anchor_nodes = [int(tree.query(c)[1]) for c in centers_real]
    else:
        anchor_nodes = None

    n_l = len(LENGTH_EDGES) - 1
    n_s = len(SHAPE_VALUES)
    if method == "jointBucketExact":
        od_flat = rng.choice(K * K, size=N_GEN * 4, p=meas["od"].ravel())
        draws = []
        for odx in od_flat:
            o, d = int(odx // K), int(odx % K)
            probs = meas["joint"][od_dist_bin(o, d, centers_norm)].ravel()
            probs = probs / probs.sum()
            lsx = int(rng.choice(probs.size, p=probs))
            draws.append((o, d, int(lsx // n_s)))
    elif method == "dpPGM":
        pgm = fit_pgm(meas, centers_norm, iters=60)
        probs = pgm.ravel() / pgm.sum()
        flat = rng.choice(pgm.size, size=N_GEN * 4, p=probs)
        draws = []
        for draw in flat:
            o = int(draw // (K * n_l * n_s))
            rem = int(draw % (K * n_l * n_s))
            d = int(rem // (n_l * n_s))
            rem = int(rem % (n_l * n_s))
            draws.append((o, d, int(rem // n_s)))
    else:
        raise ValueError(method)

    syn = []
    route_nodes = []
    diag = {
        "method": method,
        "attempted": 0,
        "routed": 0,
        "skipped_same_anchor": 0,
        "skipped_unroutable": 0,
        "mean_route_nodes": None,
    }
    for o, d, lb in draws:
        if len(syn) >= N_GEN:
            break
        diag["attempted"] += 1
        length = sample_exact_len(meas["len"], lb, rng)
        if road_coords is not None:
            if anchor_nodes[o] == anchor_nodes[d]:
                diag["skipped_same_anchor"] += 1
                continue
            nodes = shortest_path(graph, road_coords, anchor_nodes[o], anchor_nodes[d])
            if nodes is None or len(nodes) < 2:
                diag["skipped_unroutable"] += 1
                continue
            diag["routed"] += 1
            route_nodes.append(len(nodes))
            poly = road_coords[np.asarray(nodes, dtype=int)]
        else:
            poly = np.linspace(centers_real[o], centers_real[d], max(2, length))
        syn.append(resample(poly, max(2, length)))

    if route_nodes:
        diag["mean_route_nodes"] = float(np.mean(route_nodes))
    return syn, diag


def norm_by(raw, mn, span):
    return [(np.asarray(t, dtype=float) - mn) / span for t in raw if len(t) >= 2]


def active_f1(real, syn, g=24):
    def active(trajs):
        cells = set()
        for t in trajs:
            arr = np.asarray(t)
            for p in arr:
                r = min(max(int(p[0] * g), 0), g - 1)
                c = min(max(int(p[1] * g), 0), g - 1)
                cells.add((r, c))
        return cells

    a, b = active(real), active(syn)
    if not a or not b:
        return 0.0
    precision = len(a & b) / len(b)
    recall = len(a & b) / len(a)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def transition_jsd(real, syn, g=24):
    def hist(trajs):
        h = np.zeros((g * g, g * g), dtype=float)
        for t in trajs:
            arr = np.asarray(t)
            if len(arr) < 2:
                continue
            cells = []
            for p in arr:
                r = min(max(int(p[0] * g), 0), g - 1)
                c = min(max(int(p[1] * g), 0), g - 1)
                cell = r * g + c
                if not cells or cells[-1] != cell:
                    cells.append(cell)
            for a, b in zip(cells[:-1], cells[1:]):
                h[a, b] += 1
        return h

    return js_hist(hist(real), hist(syn))


def radial_jsd(real, syn):
    pts = np.vstack(real)
    center = pts.mean(axis=0)

    def vals(trajs):
        return np.asarray([np.linalg.norm(np.asarray(t) - center, axis=1).mean() for t in trajs if len(t) >= 2])

    rv, sv = vals(real), vals(syn)
    hi = max(float(np.quantile(rv, 0.99)) if len(rv) else 1.0, float(np.quantile(sv, 0.99)) if len(sv) else 1.0, 1e-6)
    bins = np.linspace(0, hi, 25)
    return js_hist(np.histogram(rv, bins=bins)[0], np.histogram(sv, bins=bins)[0])


def resample_fixed(t, n=24):
    return resample(np.asarray(t, dtype=float), n)


def dtw_distance(a, b):
    a = resample_fixed(a)
    b = resample_fixed(b)
    prev = np.full(len(b) + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, len(a) + 1):
        cur = np.full(len(b) + 1, np.inf)
        for j in range(1, len(b) + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[-1] / max(len(a), 1))


def hausdorff_distance(a, b):
    a = resample_fixed(a)
    b = resample_fixed(b)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def nearest_shape_distance(real, syn, rng, max_real=70, max_syn=45):
    if not real or not syn:
        return {"nn_dtw": float("inf"), "nn_hausdorff": float("inf")}
    real_idx = rng.choice(len(real), size=min(max_real, len(real)), replace=False)
    syn_idx = rng.choice(len(syn), size=min(max_syn, len(syn)), replace=False)
    real_pool = [real[int(i)] for i in real_idx]
    dtw_vals = []
    haus_vals = []
    for i in syn_idx:
        s = syn[int(i)]
        dtw_vals.append(min(dtw_distance(s, r) for r in real_pool))
        haus_vals.append(min(hausdorff_distance(s, r) for r in real_pool))
    return {"nn_dtw": float(np.mean(dtw_vals)), "nn_hausdorff": float(np.mean(haus_vals))}


def center_periphery_error(real, syn):
    pts = np.vstack(real)
    center = pts.mean(axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    q1, q2 = np.quantile(d, [0.50, 0.85])

    def bucket_probs(trajs):
        vals = []
        for t in trajs:
            arr = np.asarray(t)
            vals.extend(np.linalg.norm(arr - center, axis=1).tolist())
        vals = np.asarray(vals)
        h = np.asarray([(vals <= q1).sum(), ((vals > q1) & (vals <= q2)).sum(), (vals > q2).sum()], dtype=float)
        return h / max(h.sum(), 1.0)

    return float(np.abs(bucket_probs(real) - bucket_probs(syn)).sum())


def multiscale_spatial_jsd(real, syn, grids=(12, 24, 48)):
    def hist(trajs, g):
        h = np.zeros((g, g), dtype=float)
        for t in trajs:
            arr = np.asarray(t)
            for p in arr:
                r = min(max(int(p[0] * g), 0), g - 1)
                c = min(max(int(p[1] * g), 0), g - 1)
                h[r, c] += 1
        return h

    vals = [js_hist(hist(real, g), hist(syn, g)) for g in grids]
    return {f"spatial_jsd_g{g}": float(v) for g, v in zip(grids, vals)} | {"spatial_jsd_multiscale": float(np.mean(vals))}


def hot_cell_recall(real, syn, g=48, top_mass=0.80):
    def hist(trajs):
        h = np.zeros((g, g), dtype=float)
        for t in trajs:
            for p in np.asarray(t):
                r = min(max(int(p[0] * g), 0), g - 1)
                c = min(max(int(p[1] * g), 0), g - 1)
                h[r, c] += 1
        return h

    rh = hist(real)
    sh = hist(syn)
    flat = rh.ravel()
    if flat.sum() <= 0:
        return 0.0
    order = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[order]) / flat.sum()
    hot = order[cum <= top_mass]
    if len(hot) == 0:
        hot = order[:1]
    return float(flat[hot][sh.ravel()[hot] > 0].sum() / max(flat[hot].sum(), 1.0))


def compare_metrics(real_eval, syn, all_real):
    _, mn, span = normalize_with_meta(all_real)
    r = norm_by(real_eval, mn, span)
    s = norm_by(syn, mn, span)
    out = normalized_metrics(r, s)
    out["coverage_f1_24"] = active_f1(r, s)
    out.update(multiscale_spatial_jsd(r, s))
    out["hot_cell_recall_48"] = hot_cell_recall(r, s)
    out["transition_jsd_24"] = transition_jsd(r, s)
    out["radial_jsd"] = radial_jsd(r, s)
    rc = np.vstack(r).mean(axis=0)
    sc = np.vstack(s).mean(axis=0)
    out["centroid_error_norm"] = float(np.linalg.norm(rc - sc))
    out["center_periphery_l1"] = center_periphery_error(r, s)
    out.update(nearest_shape_distance(r, s, np.random.default_rng(2026)))
    return out


def plot_compare(name, real, joint, pgm, bbox, *, osm_ways=None, bg_oldenburg=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    draw_trajs(axes[0], real, bbox, "#1f5aa6", f"{name}: real", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    draw_trajs(axes[1], joint, bbox, "#2ca25f", f"{name}: jointBucketExact", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    draw_trajs(axes[2], pgm, bbox, "#d95f02", f"{name}: dpPGM", bg_osm=osm_ways, bg_oldenburg=bg_oldenburg)
    fig.suptitle(f"{name}: pre-PGM joint model vs DP-PGM on the same base map", fontsize=13, fontweight="bold")
    out = OUT_DIR / f"{name.lower()}_joint_vs_pgm_on_basemap.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_metric_bars(results):
    keys = [
        "js",
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
        "centroid_error_norm",
    ]
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 8), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (city, payload) in zip(axes, results.items()):
        x = np.arange(len(keys))
        width = 0.36
        for offset, method, color in [(-width / 2, "jointBucketExact", "#2ca25f"), (width / 2, "dpPGM", "#d95f02")]:
            vals = [payload["metrics"][method][k] for k in keys]
            ax.bar(x + offset, vals, width, label=method, color=color, alpha=0.85)
        ax.set_title(city, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")
    out = OUT_DIR / "joint_vs_pgm_metric_bars.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def run_city(name, loader, bbox, osm_city=None, raw_graph=False):
    real = loader(limit=N_LOAD)
    osm, osm_path = load_osm_pickle(osm_city) if osm_city else (None, None)
    joint, joint_diag = synthesize_variant(real, "jointBucketExact", osm_ways=osm, bbox=bbox, raw_graph=raw_graph)
    pgm, pgm_diag = synthesize_variant(real, "dpPGM", osm_ways=osm, bbox=bbox, raw_graph=raw_graph)
    fig = plot_compare(name, real, joint, pgm, bbox, osm_ways=osm, bg_oldenburg=real if raw_graph else None)
    real_eval = real[N_TRAIN : N_TRAIN + N_EVAL]
    return {
        "figure": str(fig),
        "osm": str(osm_path) if osm_path else "raw trajectory coverage background",
        "n_real_loaded": len(real),
        "n_eval": len(real_eval),
        "n_syn": {"jointBucketExact": len(joint), "dpPGM": len(pgm)},
        "diagnostics": {"jointBucketExact": joint_diag, "dpPGM": pgm_diag},
        "metrics": {
            "jointBucketExact": compare_metrics(real_eval, joint, real[: N_TRAIN + N_EVAL]),
            "dpPGM": compare_metrics(real_eval, pgm, real[: N_TRAIN + N_EVAL]),
        },
    }


def main():
    results = {
        "Beijing": run_city("Beijing", load_geolife, BEIJING_BBOX, "beijing"),
        "Porto": run_city("Porto", load_porto, PORTO_BBOX, "porto"),
    }
    old = load_oldenburg(limit=N_LOAD)
    old_bbox = bbox_of(old[: N_TRAIN + N_EVAL])
    results["Oldenburg"] = run_city("Oldenburg", load_oldenburg, old_bbox, raw_graph=True)
    metric_fig = plot_metric_bars(results)
    summary = ARA_CODEX / "results" / "joint_vs_pgm_city_osm_comparison.json"
    summary.write_text(json.dumps({"results": results, "metric_figure": str(metric_fig)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary)
    for payload in results.values():
        print(payload["figure"])
    print(metric_fig)


if __name__ == "__main__":
    main()
