"""Cross-dataset trajectory-mode benchmark for ARA variants.

This benchmark intentionally avoids Beijing-specific OSM routing.  It evaluates
whether the trajectory-mode generators themselves generalize across available
datasets: GeoLife Beijing, Porto taxi, and Oldenburg simulated trajectories.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EXP_DATA = ROOT / "semdp_traj_paper" / "experiments" / "datasets"
OUT = ROOT / "ARA_codex" / "results"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1042, 2042]
N_TRAIN = 800
N_EVAL = 300
N_GEN = 200
EPS = 1.0
K = 25
GRID = 20
LENGTH_EDGES = np.asarray([2, 6, 10, 15, 22, 32, 48, 70, 101], dtype=int)
SHAPE_BINS = np.asarray([1.0, 1.15, 1.35, 1.75, 2.5, 4.0, 7.0, 12.0, 20.0, np.inf])
SHAPE_VALUES = np.asarray([1.075, 1.25, 1.55, 2.1, 3.25, 5.5, 9.5, 16.0, 24.0])
OD_DIST_BINS = np.asarray([0.0, 0.05, 0.12, 0.25, 0.50, 0.80, np.inf])


def load_geolife():
    bbox = (39.75, 40.15, 116.10, 116.65)
    root = EXP_DATA / "geolife" / "Geolife Trajectories 1.3" / "Data"
    out = []
    for plt in sorted(root.glob("*/Trajectory/*.plt")):
        if len(out) >= limit:
            break
        pts = []
        with plt.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(6):
                next(f, None)
            for line in f:
                p = line.strip().split(",")
                if len(p) < 2:
                    continue
                try:
                    lat, lon = float(p[0]), float(p[1])
                except ValueError:
                    continue
                if bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]:
                    pts.append([lat, lon])
        if len(pts) >= 2:
            out.append(np.asarray(pts, dtype=float))
    return out


def load_porto(limit=1400):
    bbox = (41.10, 41.20, -8.70, -8.55)
    path = EXP_DATA / "porto" / "train.csv"
    out = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(out) >= limit:
                break
            raw = row.get("POLYLINE")
            if not raw:
                continue
            try:
                pts = json.loads(raw)
            except json.JSONDecodeError:
                continue
            arr = np.asarray([[p[1], p[0]] for p in pts if len(p) >= 2], dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 2:
                continue
            arr = arr[(arr[:, 0] >= bbox[0]) & (arr[:, 0] <= bbox[1]) & (arr[:, 1] >= bbox[2]) & (arr[:, 1] <= bbox[3])]
            if len(arr) >= 2:
                out.append(arr[:100])
    return out


def load_oldenburg(limit=1400):
    out = []
    with (DATA / "oldenburg.dat").open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(out) >= limit:
                break
            if ":" not in line:
                continue
            pts = []
            for item in line.split(":", 1)[1].split(";"):
                item = item.strip()
                if "," not in item:
                    continue
                try:
                    x, y = item.split(",", 1)
                    pts.append([float(x), float(y)])
                except ValueError:
                    pass
            if len(pts) >= 2:
                out.append(np.asarray(pts, dtype=float))
    return out


def normalize_dataset(trajs):
    pts = np.vstack(trajs)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(mx - mn, 1e-12)
    return [(np.asarray(t, dtype=float) - mn) / span for t in trajs], {"min": mn.tolist(), "max": mx.tolist()}


def l1_noisy_hist(hist, eps, rng):
    noisy = np.maximum(hist + rng.laplace(0.0, 1.0 / eps, size=hist.shape), 0.0)
    if noisy.sum() <= 1e-12:
        noisy[:] = 1.0
    return noisy / noisy.sum()


def nearest(pt, centers):
    return int(np.argmin(np.linalg.norm(centers - pt, axis=1)))


def path_len(t):
    return float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1))) if len(t) >= 2 else 0.0


def sinuosity(t):
    if len(t) < 3:
        return 1.0
    direct = float(np.linalg.norm(t[-1] - t[0]))
    return 1.0 if direct <= 1e-12 else max(1.0, path_len(t) / direct)


def len_bucket(n):
    return min(max(int(np.searchsorted(LENGTH_EDGES, n, side="right") - 1), 0), len(LENGTH_EDGES) - 2)


def shape_bucket(t):
    s = min(sinuosity(t), SHAPE_BINS[-2])
    return min(max(int(np.searchsorted(SHAPE_BINS, s, side="right") - 1), 0), len(SHAPE_VALUES) - 1)


def od_dist_bin(a, b, centers):
    d = float(np.linalg.norm(centers[int(a)] - centers[int(b)]))
    return min(max(int(np.searchsorted(OD_DIST_BINS, d, side="right") - 1), 0), len(OD_DIST_BINS) - 2)


def dp_anchors(train, eps, rng):
    hist = np.zeros((GRID, GRID), dtype=float)
    for t in train:
        cells = set()
        for p in t:
            r = min(max(int(p[0] * GRID), 0), GRID - 1)
            c = min(max(int(p[1] * GRID), 0), GRID - 1)
            cells.add((r, c))
        for r, c in cells:
            hist[r, c] += 1.0 / max(len(cells), 1)
    noisy = hist + rng.laplace(0.0, 1.0 / eps, size=hist.shape)
    idx = np.argsort(noisy.ravel())[-K:]
    centers = np.asarray([[(i // GRID + 0.5) / GRID, (i % GRID + 0.5) / GRID] for i in idx], dtype=float)
    return centers


def fit_measurements(train, centers, rng, *, pgm=False, eps_total=EPS, eps_anchor=0.20):
    """Release the fixed-support trip-condition measurements.

    ``eps_total`` is the full budget for the base mode model and
    ``eps_anchor`` is the part already spent on DP anchors.  Older experiments
    used the module defaults EPS=1.0 and eps_anchor=0.20, which gives the same
    per-table budgets as before.  Making the budget explicit prevents later
    scripts from recording one epsilon while silently using another.
    """
    eps_anchorless = max(float(eps_total) - float(eps_anchor), 1e-12)
    budget_scale = eps_anchorless / 0.80
    if pgm:
        budgets = {
            "od": 0.20 * budget_scale,
            "dls": 0.20 * budget_scale,
            "ol": 0.10 * budget_scale,
            "dl": 0.10 * budget_scale,
            "len": 0.20 * budget_scale,
        }
    else:
        budgets = {"od": 0.25 * budget_scale, "joint": 0.30 * budget_scale, "len": 0.25 * budget_scale}
    n_l = len(LENGTH_EDGES) - 1
    n_s = len(SHAPE_VALUES)
    od = np.zeros((K, K), dtype=float)
    exact_len = np.zeros(99, dtype=float)
    joint = np.zeros((len(OD_DIST_BINS) - 1, n_l, n_s), dtype=float)
    ol = np.zeros((K, n_l), dtype=float)
    dl = np.zeros((K, n_l), dtype=float)
    for t in train:
        if len(t) < 2:
            continue
        o, d = nearest(t[0], centers), nearest(t[-1], centers)
        lb, sb = len_bucket(len(t)), shape_bucket(t)
        od[o, d] += 1.0
        exact_len[min(max(len(t), 2), 100) - 2] += 1.0
        joint[od_dist_bin(o, d, centers), lb, sb] += 1.0
        ol[o, lb] += 1.0
        dl[d, lb] += 1.0
    if pgm:
        return {
            "od": l1_noisy_hist(od, budgets["od"], rng),
            "joint": l1_noisy_hist(joint, budgets["dls"], rng),
            "ol": l1_noisy_hist(ol, budgets["ol"], rng),
            "dl": l1_noisy_hist(dl, budgets["dl"], rng),
            "len": l1_noisy_hist(exact_len, budgets["len"], rng),
        }
    return {
        "od": l1_noisy_hist(od, budgets["od"], rng),
        "joint": l1_noisy_hist(joint, budgets["joint"], rng),
        "len": l1_noisy_hist(exact_len, budgets["len"], rng),
    }


def fit_pgm(meas, centers, iters=60):
    n_l = len(LENGTH_EDGES) - 1
    n_s = len(SHAPE_VALUES)
    p = np.ones((K, K, n_l, n_s), dtype=float)
    p /= p.sum()
    masks = []
    for db in range(len(OD_DIST_BINS) - 1):
        mask = np.zeros((K, K), dtype=bool)
        for i in range(K):
            for j in range(K):
                mask[i, j] = od_dist_bin(i, j, centers) == db
        masks.append(mask)
    for _ in range(iters):
        p *= (meas["od"] / np.maximum(p.sum(axis=(2, 3)), 1e-15))[:, :, None, None]
        p *= (meas["ol"] / np.maximum(p.sum(axis=(1, 3)), 1e-15))[:, None, :, None]
        p *= (meas["dl"] / np.maximum(p.sum(axis=(0, 3)), 1e-15))[None, :, :, None]
        for db, mask in enumerate(masks):
            for lb in range(n_l):
                for sb in range(n_s):
                    slab = p[:, :, lb, sb]
                    slab[mask] *= meas["joint"][db, lb, sb] / max(float(slab[mask].sum()), 1e-15)
                    p[:, :, lb, sb] = slab
        p = p / max(float(p.sum()), 1e-15)
    return p


def sample_exact_len(length_dist, lb, rng):
    bins = np.arange(2, 101)
    lo, hi = LENGTH_EDGES[lb], LENGTH_EDGES[lb + 1]
    mask = (bins >= lo) & (bins < hi)
    prob = length_dist * mask
    if prob.sum() <= 1e-12:
        prob = length_dist.copy()
    prob = prob / prob.sum()
    return int(rng.choice(bins, p=prob))


def synthesize(name, meas, centers, rng):
    syn = []
    if name == "independentDP":
        od_flat = rng.choice(K * K, size=N_GEN, p=meas["od"].ravel())
        joint_global = meas["joint"].sum(axis=0).ravel()
        joint_global = joint_global / joint_global.sum()
        ls_flat = rng.choice(joint_global.size, size=N_GEN, p=joint_global)
    elif name == "jointBucketExact":
        od_flat = rng.choice(K * K, size=N_GEN, p=meas["od"].ravel())
        ls_flat = []
        for x in od_flat:
            o, d = int(x // K), int(x % K)
            probs = meas["joint"][od_dist_bin(o, d, centers)].ravel()
            probs = probs / probs.sum()
            ls_flat.append(int(rng.choice(probs.size, p=probs)))
    elif name == "dpPGM":
        p = fit_pgm(meas, centers)
        draws = rng.choice(p.size, size=N_GEN, p=p.ravel() / p.sum())
        od_flat = []
        ls_flat = []
        n_l, n_s = len(LENGTH_EDGES) - 1, len(SHAPE_VALUES)
        for draw in draws:
            o = draw // (K * n_l * n_s)
            rem = draw % (K * n_l * n_s)
            d = rem // (n_l * n_s)
            rem = rem % (n_l * n_s)
            od_flat.append(int(o * K + d))
            ls_flat.append(int(rem))
    else:
        raise ValueError(name)
    for odx, lsx in zip(od_flat, ls_flat):
        o, d = int(odx // K), int(odx % K)
        lb = int(lsx // len(SHAPE_VALUES))
        length = sample_exact_len(meas["len"], lb, rng)
        a, b = centers[o], centers[d]
        line = np.linspace(a, b, max(length, 2))
        syn.append(line)
    return syn


def hist_points(trajs, g=12):
    h = np.zeros((g, g), dtype=float)
    for t in trajs:
        for p in t:
            r = min(max(int(p[0] * g), 0), g - 1)
            c = min(max(int(p[1] * g), 0), g - 1)
            h[r, c] += 1
    return h


def js_hist(a, b):
    af, bf = a.ravel().astype(float), b.ravel().astype(float)
    return float(jensenshannon(af / max(af.sum(), 1), bf / max(bf.sum(), 1)))


def metrics(real, syn):
    out = {"js": js_hist(hist_points(real), hist_points(syn))}
    def odh(trajs, g=8):
        h = np.zeros((g * g, g * g), dtype=float)
        for t in trajs:
            if len(t) < 2:
                continue
            so = min(max(int(t[0, 0] * g), 0), g - 1) * g + min(max(int(t[0, 1] * g), 0), g - 1)
            sd = min(max(int(t[-1, 0] * g), 0), g - 1) * g + min(max(int(t[-1, 1] * g), 0), g - 1)
            h[so, sd] += 1
        return h
    out["trip_jsd"] = js_hist(odh(real), odh(syn))
    rl = np.asarray([len(t) for t in real]); sl = np.asarray([len(t) for t in syn])
    max_len = int(max(rl.max(), sl.max()))
    out["len_jsd"] = js_hist(np.bincount(rl, minlength=max_len + 1)[2:], np.bincount(sl, minlength=max_len + 1)[2:])
    def step_vals(trajs):
        vals = []
        for t in trajs:
            if len(t) >= 2:
                vals.extend(np.linalg.norm(np.diff(t, axis=0), axis=1).tolist())
        return np.asarray(vals)
    rs, ss = step_vals(real), step_vals(syn)
    mx = max(float(np.quantile(rs, 0.99)) if len(rs) else 1, float(np.quantile(ss, 0.99)) if len(ss) else 1, 1e-6)
    bins = np.linspace(0, mx, 30)
    out["step_jsd"] = js_hist(np.histogram(rs, bins=bins)[0], np.histogram(ss, bins=bins)[0])
    rshape = np.asarray([sinuosity(t) for t in real])
    sshape = np.asarray([sinuosity(t) for t in syn])
    bins = np.asarray([1, 1.1, 1.25, 1.5, 2, 3, 5, 10, 25])
    out["shape_jsd"] = js_hist(np.histogram(rshape, bins=bins)[0], np.histogram(sshape, bins=bins)[0])
    out["sinuosity_median"] = float(np.median(sshape))
    return out


def run_dataset(name, loader):
    raw = loader()
    if len(raw) < N_TRAIN + N_EVAL:
        return None
    trajs, norm = normalize_dataset(raw[: N_TRAIN + N_EVAL])
    train = trajs[:N_TRAIN]
    eval_real = trajs[N_TRAIN:N_TRAIN + N_EVAL]
    rows = {}
    for method in ["independentDP", "jointBucketExact", "dpPGM"]:
        vals = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            centers = dp_anchors(train, 0.20, rng)
            meas = fit_measurements(train, centers, rng, pgm=(method == "dpPGM"))
            syn = synthesize(method, meas, centers, rng)
            vals.append(metrics(eval_real, syn))
        rows[method] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        rows[method + "_std"] = {k: float(np.std([v[k] for v in vals])) for k in vals[0]}
    return {"name": name, "n": len(raw), "normalization": norm, "means": rows}


def write_markdown(results):
    lines = [
        "# Cross-Dataset ARA Mode Benchmark",
        "",
        "This benchmark is road-agnostic: it evaluates trajectory-mode synthesis on normalized coordinates.",
        "It should be read alongside Beijing road-local metrics; it does not measure map-matching or road validity.",
        "",
        "| Dataset | Method | JS | TripJSD | LenJSD | StepJSD | ShapeJSD | SynSinuosityMed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in results:
        for m in ["independentDP", "jointBucketExact", "dpPGM"]:
            v = ds["means"][m]
            lines.append(
                f"| {ds['name']} | {m} | {v['js']:.4f} | {v['trip_jsd']:.4f} | {v['len_jsd']:.4f} | "
                f"{v['step_jsd']:.4f} | {v['shape_jsd']:.4f} | {v['sinuosity_median']:.4f} |"
            )
    (OUT / "cross_dataset_ara_mode_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(results):
    metrics_keys = ["js", "trip_jsd", "len_jsd", "step_jsd", "shape_jsd"]
    methods = ["independentDP", "jointBucketExact", "dpPGM"]
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.2), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, ds in zip(axes, results):
        x = np.arange(len(metrics_keys))
        for i, method in enumerate(methods):
            vals = [ds["means"][method][k] for k in metrics_keys]
            ax.bar(x + (i - 1) * 0.25, vals, width=0.25, label=method)
        ax.set_title(ds["name"])
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_keys, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "cross_dataset_ara_mode_benchmark.png", dpi=180)

    fig, axes = plt.subplots(len(results), 4, figsize=(14, 3.5 * len(results)))
    if len(results) == 1:
        axes = np.asarray([axes])
    for r, ds in enumerate(results):
        raw = {"geolife": load_geolife, "porto": load_porto, "oldenburg": load_oldenburg}[ds["name"]]()
        trajs, _ = normalize_dataset(raw[: N_TRAIN + N_EVAL])
        eval_real = trajs[N_TRAIN:N_TRAIN + N_EVAL]
        rng = np.random.default_rng(42)
        centers = dp_anchors(trajs[:N_TRAIN], 0.20, rng)
        axes[r, 0].imshow(hist_points(eval_real), origin="lower", cmap="magma")
        axes[r, 0].set_title(f"{ds['name']} real")
        for c, method in enumerate(methods, start=1):
            meas = fit_measurements(trajs[:N_TRAIN], centers, rng, pgm=(method == "dpPGM"))
            syn = synthesize(method, meas, centers, rng)
            axes[r, c].imshow(hist_points(syn), origin="lower", cmap="magma")
            axes[r, c].set_title(method)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "cross_dataset_ara_mode_heatmaps.png", dpi=180)


def main():
    datasets = [
        ("geolife", load_geolife),
        ("porto", load_porto),
        ("oldenburg", load_oldenburg),
    ]
    results = []
    for name, loader in datasets:
        print(f"Running {name}...")
        res = run_dataset(name, loader)
        if res is not None:
            results.append(res)
    (OUT / "cross_dataset_ara_mode_benchmark.json").write_text(json.dumps({"datasets": results}, indent=2), encoding="utf-8")
    write_markdown(results)
    plot(results)
    print(OUT / "cross_dataset_ara_mode_benchmark.md")
    print(FIG / "cross_dataset_ara_mode_benchmark.png")
    print(FIG / "cross_dataset_ara_mode_heatmaps.png")


if __name__ == "__main__":
    main()
