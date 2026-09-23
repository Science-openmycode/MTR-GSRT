"""Full-size downstream utility tasks for Figure 2 synthetic releases.

This script evaluates downstream mobility-mining tasks on the same full-size
synthetic datasets used by Figure 2.  It does not regenerate trajectories.

Protocol:
- real data: GeoLife full cache, n=17,123
- synthetic data: results/figures/full_size_shared_protocol/generated/*.pkl
- every synthetic method is evaluated with its full 17,123-trajectory release
- train/test predictive tasks use an 80/20 split of the real data only for
  evaluation; synthetic releases are the TSTR training sets
"""
from __future__ import annotations

import csv
import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ARA = Path(__file__).parents[1]
ROOT = ARA.parent
FULL_REAL = ROOT / "ARA_codex_help" / "data" / "geolife_full_17123.pkl"
GEN = ARA / "results" / "figures" / "full_size_shared_protocol" / "generated"
OUT = ARA / "results" / "downstream_full_protocol"
PAPER = ARA / "paper_latex"

BBOX = (39.75, 40.15, 116.10, 116.65)
METHODS = [
    "SPRT-native",
    "SPRT-OSM",
    "PrivTrace-native",
    "PrivTrace-OSM",
    "DPTrajPM-native",
    "DPTrajPM-OSM",
    "DPStd-native",
    "DPStd-OSM",
    "ARA-DP-GSRT",
]

DISPLAY = {
    "ARA-DP-GSRT": "MTR / DP-GSRT",
}


def normalize(obj) -> list[np.ndarray]:
    trajs = obj.get("trajectories", obj.get("synthetic", obj)) if isinstance(obj, dict) else obj
    out = []
    for traj in trajs:
        arr = np.asarray(traj, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 2:
            arr = arr[:, :2]
            arr = arr[np.isfinite(arr).all(axis=1)]
            if len(arr) >= 2:
                out.append(arr.astype(np.float64))
    return out


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_real() -> list[np.ndarray]:
    return normalize(load_pickle(FULL_REAL))


def load_synthetic(name: str) -> list[np.ndarray]:
    path = GEN / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return normalize(load_pickle(path))


def cell_id(points: np.ndarray, grid: int) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = BBOX
    y = np.floor((points[:, 0] - lat_min) / max(lat_max - lat_min, 1e-12) * grid).astype(int)
    x = np.floor((points[:, 1] - lon_min) / max(lon_max - lon_min, 1e-12) * grid).astype(int)
    y = np.clip(y, 0, grid - 1)
    x = np.clip(x, 0, grid - 1)
    return y * grid + x


def traj_cells(traj: np.ndarray, grid: int) -> np.ndarray:
    ids = cell_id(np.asarray(traj, dtype=float), grid)
    if len(ids) == 0:
        return ids
    keep = np.r_[True, ids[1:] != ids[:-1]]
    return ids[keep]


def occupancy_counts(trajs: list[np.ndarray], grid: int) -> np.ndarray:
    counts = np.zeros(grid * grid, dtype=float)
    for traj in trajs:
        ids = np.unique(cell_id(np.asarray(traj, dtype=float), grid))
        counts[ids] += 1.0
    return counts


def js_counts(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float).ravel()
    bv = np.asarray(b, dtype=float).ravel()
    if av.sum() <= 0 or bv.sum() <= 0:
        return 1.0
    return float(jensenshannon(av / av.sum(), bv / bv.sum()))


def make_range_queries(grid: int, n_queries: int = 256, seed: int = 20260704):
    rng = np.random.default_rng(seed + grid)
    queries = []
    for _ in range(n_queries):
        h = int(rng.integers(1, max(2, grid // 2 + 1)))
        w = int(rng.integers(1, max(2, grid // 2 + 1)))
        y0 = int(rng.integers(0, grid - h + 1))
        x0 = int(rng.integers(0, grid - w + 1))
        cells = [y * grid + x for y in range(y0, y0 + h) for x in range(x0, x0 + w)]
        queries.append(np.asarray(cells, dtype=int))
    return queries


def count_query_metrics(real: list[np.ndarray], syn: list[np.ndarray]) -> dict[str, float]:
    jsds = []
    smapes = []
    ares = []
    for grid in [8, 16, 32]:
        rc = occupancy_counts(real, grid)
        sc = occupancy_counts(syn, grid)
        jsds.append(js_counts(rc, sc))
        for cells in make_range_queries(grid):
            r = float(rc[cells].sum())
            s = float(sc[cells].sum())
            smapes.append(abs(s - r) / max(s + r, 1.0))
            if r >= 5:
                ares.append(abs(s - r) / r)
    return {
        "A1_count_jsd_mean": float(np.mean(jsds)),
        "A1_range_smape": float(np.mean(smapes)),
        "A1_range_are_nonzero": float(np.mean(ares)) if ares else math.nan,
    }


def od_counts(trajs: list[np.ndarray], grid: int = 8) -> np.ndarray:
    mat = np.zeros((grid * grid, grid * grid), dtype=float)
    for traj in trajs:
        arr = np.asarray(traj, dtype=float)
        if len(arr) < 2:
            continue
        o = int(cell_id(arr[:1], grid)[0])
        d = int(cell_id(arr[-1:], grid)[0])
        mat[o, d] += 1.0
    return mat


def ndcg_at_k(real_scores: np.ndarray, syn_scores: np.ndarray, k: int = 20) -> float:
    order = np.argsort(-syn_scores)[:k]
    gains = real_scores[order]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(real_scores)[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def od_demand_metrics(real: list[np.ndarray], syn: list[np.ndarray]) -> dict[str, float]:
    ro = od_counts(real, 8).ravel()
    so = od_counts(syn, 8).ravel()
    rho = spearmanr(ro, so).statistic
    rho = 0.0 if not np.isfinite(rho) else float(rho)
    top_real = set(np.argsort(-ro)[:20].tolist())
    top_syn = set(np.argsort(-so)[:20].tolist())
    return {
        "OD_spearman": rho,
        "OD_top20_overlap": len(top_real & top_syn) / 20.0,
        "OD_ndcg20": ndcg_at_k(ro, so, 20),
        "OD_jsd": js_counts(ro, so),
    }


def trajectory_features(trajs: list[np.ndarray], dest_grid: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    lat_min, lat_max, lon_min, lon_max = BBOX
    for traj in trajs:
        arr = np.asarray(traj, dtype=float)
        if len(arr) < 2:
            continue
        idx = max(1, int(round((len(arr) - 1) * 0.20)))
        p0 = arr[0]
        p1 = arr[idx]
        pend = arr[-1]
        delta = p1 - p0
        norm = np.array([
            (p0[0] - lat_min) / (lat_max - lat_min),
            (p0[1] - lon_min) / (lon_max - lon_min),
            (p1[0] - lat_min) / (lat_max - lat_min),
            (p1[1] - lon_min) / (lon_max - lon_min),
            delta[0] / (lat_max - lat_min),
            delta[1] / (lon_max - lon_min),
        ])
        cells = traj_cells(arr[: idx + 1], dest_grid)
        prefix_changes = float(len(cells))
        direct = float(np.linalg.norm(delta))
        angle = math.atan2(float(delta[0]), float(delta[1])) if direct > 0 else 0.0
        o_cell = int(cell_id(arr[:1], dest_grid)[0])
        o_y, o_x = divmod(o_cell, dest_grid)
        feat = np.r_[norm, direct, math.sin(angle), math.cos(angle), prefix_changes / 20.0, o_y / dest_grid, o_x / dest_grid]
        xs.append(feat)
        ys.append(int(cell_id(pend.reshape(1, 2), dest_grid)[0]))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)


def topk_acc_from_proba(proba: np.ndarray, classes: np.ndarray, y_true: np.ndarray, k: int) -> float:
    k = min(k, proba.shape[1])
    top_idx = np.argsort(-proba, axis=1)[:, :k]
    pred_classes = classes[top_idx]
    return float(np.mean([yt in row for yt, row in zip(y_true, pred_classes)]))


def destination_tstr_metrics(train: list[np.ndarray], real_eval: list[np.ndarray], prefix: str = "B1") -> dict[str, float]:
    out = {}
    for grid in [8, 16]:
        x_train, y_train = trajectory_features(train, grid)
        x_test, y_test = trajectory_features(real_eval, grid)
        if len(np.unique(y_train)) < 2:
            out[f"{prefix}_dest{grid}_top1"] = 0.0
            out[f"{prefix}_dest{grid}_top5"] = 0.0
            out[f"{prefix}_dest{grid}_macro_f1"] = 0.0
            continue
        clf = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                max_iter=1000,
                tol=1e-3,
                class_weight="balanced",
                random_state=20260704,
            ),
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        proba = clf.predict_proba(x_test)
        classes = clf.named_steps["sgdclassifier"].classes_
        out[f"{prefix}_dest{grid}_top1"] = float(np.mean(pred == y_test))
        out[f"{prefix}_dest{grid}_top5"] = topk_acc_from_proba(proba, classes, y_test, 5)
        out[f"{prefix}_dest{grid}_macro_f1"] = float(f1_score(y_test, pred, average="macro", zero_division=0))
    return out


def grid_transition_set(traj: np.ndarray, grid: int = 16) -> tuple[tuple[int, ...], frozenset[int]]:
    cells = traj_cells(np.asarray(traj, dtype=float), grid)
    trans = []
    for a, b in zip(cells[:-1], cells[1:]):
        if int(a) != int(b):
            trans.append(int(a) * grid * grid + int(b))
    if not trans:
        return tuple(), frozenset()
    return tuple(trans), frozenset(trans)


def od_key(traj: np.ndarray, grid: int = 8) -> int:
    arr = np.asarray(traj, dtype=float)
    o = int(cell_id(arr[:1], grid)[0])
    d = int(cell_id(arr[-1:], grid)[0])
    return o * grid * grid + d


def f1_set(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    p = inter / len(b)
    r = inter / len(a)
    return 2 * p * r / (p + r)


def build_grid_route_prototypes(train: list[np.ndarray], max_per_od: int = 20):
    counters: dict[int, Counter] = defaultdict(Counter)
    sets: dict[tuple[int, tuple[int, ...]], frozenset[int]] = {}
    for traj in train:
        arr = np.asarray(traj, dtype=float)
        if len(arr) < 2:
            continue
        od = od_key(arr, 8)
        sig, eset = grid_transition_set(arr, 16)
        if not sig:
            continue
        counters[od][sig] += 1
        sets[(od, sig)] = eset
    protos = {}
    for od, counter in counters.items():
        ranked = []
        for sig, freq in counter.most_common(max_per_od):
            ranked.append({"sig": sig, "freq": int(freq), "edges": sets[(od, sig)]})
        protos[od] = ranked
    return protos


def grid_route_choice_metrics(train: list[np.ndarray], real_eval: list[np.ndarray]) -> dict[str, float]:
    protos = build_grid_route_prototypes(train, max_per_od=20)
    top1_hits = []
    top3_hits = []
    top5_hits = []
    mrrs = []
    top1_f1 = []
    best5_f1 = []
    covered = 0
    total = 0
    threshold = 0.35
    for traj in real_eval:
        arr = np.asarray(traj, dtype=float)
        if len(arr) < 2:
            continue
        _, real_edges = grid_transition_set(arr, 16)
        if not real_edges:
            continue
        total += 1
        cand = protos.get(od_key(arr, 8), [])
        if not cand:
            top1_hits.append(0.0)
            top3_hits.append(0.0)
            top5_hits.append(0.0)
            mrrs.append(0.0)
            top1_f1.append(0.0)
            best5_f1.append(0.0)
            continue
        covered += 1
        f1s = [f1_set(real_edges, c["edges"]) for c in cand]
        top1_f1.append(float(f1s[0]))
        best5_f1.append(float(max(f1s[:5])))
        top1_hits.append(float(f1s[0] >= threshold))
        top3_hits.append(float(max(f1s[:3]) >= threshold))
        top5_hits.append(float(max(f1s[:5]) >= threshold))
        rr = 0.0
        for i, val in enumerate(f1s, start=1):
            if val >= threshold:
                rr = 1.0 / i
                break
        mrrs.append(rr)
    return {
        "B2_grid_route_od_coverage": covered / max(total, 1),
        "B2_grid_route_top1_hit": float(np.mean(top1_hits)) if top1_hits else 0.0,
        "B2_grid_route_top3_hit": float(np.mean(top3_hits)) if top3_hits else 0.0,
        "B2_grid_route_top5_hit": float(np.mean(top5_hits)) if top5_hits else 0.0,
        "B2_grid_route_mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "B2_grid_route_top1_transition_f1": float(np.mean(top1_f1)) if top1_f1 else 0.0,
        "B2_grid_route_best5_transition_f1": float(np.mean(best5_f1)) if best5_f1 else 0.0,
    }


def destination_conditioned_next_region_metrics(
    train: list[np.ndarray],
    real_eval: list[np.ndarray],
    grid: int = 16,
    alpha: float = 0.5,
) -> dict[str, float]:
    """Evaluate a synthetic-trained sequential route-choice model.

    The current public region and destination region define a route-decision
    state.  Synthetic transition frequencies provide its next-region
    distribution; a current-region frequency model is the deterministic
    backoff for an unseen destination-conditioned state.  Decisions are
    averaged within each real test trip before corpus aggregation so that a
    long trip is not treated as many independent evaluation units.
    """
    conditional: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    backoff: dict[int, Counter[int]] = defaultdict(Counter)
    for trajectory in train:
        sequence = traj_cells(np.asarray(trajectory, dtype=float), grid).tolist()
        if len(sequence) < 2:
            continue
        destination = int(sequence[-1])
        for current, following in zip(sequence[:-1], sequence[1:]):
            conditional[(destination, int(current))][int(following)] += 1
            backoff[int(current)][int(following)] += 1

    trip_accuracy: list[float] = []
    trip_nll: list[float] = []
    state_count = grid * grid
    for trajectory in real_eval:
        sequence = traj_cells(np.asarray(trajectory, dtype=float), grid).tolist()
        if len(sequence) < 2:
            trip_accuracy.append(0.0)
            trip_nll.append(math.log(state_count))
            continue
        destination = int(sequence[-1])
        correct: list[float] = []
        losses: list[float] = []
        for current, following in zip(sequence[:-1], sequence[1:]):
            counter = conditional.get((destination, int(current)))
            if not counter:
                counter = backoff.get(int(current), Counter())
            if not counter:
                correct.append(0.0)
                losses.append(math.log(state_count))
                continue
            prediction = min(counter, key=lambda item: (-counter[item], item))
            correct.append(float(prediction == int(following)))
            probability = (counter.get(int(following), 0) + alpha) / (
                sum(counter.values()) + alpha * state_count
            )
            losses.append(-math.log(probability))
        trip_accuracy.append(float(np.mean(correct)))
        trip_nll.append(float(np.mean(losses)))
    return {
        "B2_next_region_accuracy": float(np.mean(trip_accuracy)) if trip_accuracy else 0.0,
        "B2_next_region_nll": float(np.mean(trip_nll)) if trip_nll else math.log(state_count),
    }


def tex_float(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.3f}"


def write_outputs(rows: list[dict], summary: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    (OUT / "downstream_full_protocol.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    keys = ["method"] + [k for k in rows[0].keys() if k != "method"]
    with (OUT / "downstream_full_protocol.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    compact_cols = [
        ("A1_range_smape", "Count"),
        ("OD_top20_overlap", "OD@20"),
        ("OD_ndcg20", "OD NDCG"),
        ("B1_dest8_top5", "Dest@5"),
        ("B1_dest16_macro_f1", "Dest-F1"),
        ("B2_grid_route_top5_hit", "GridRoute@5"),
        ("B2_grid_route_best5_transition_f1", "GridTrans-F1"),
    ]
    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Method & " + " & ".join(label for _, label in compact_cols) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        vals = [tex_float(row.get(k, math.nan)) for k, _ in compact_cols]
        lines.append(row["method"] + " & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (PAPER / "tables_kdd_downstream_full.tex").write_text("\n".join(lines), encoding="utf-8")

    detail_cols = [
        ("A1_count_jsd_mean", "CountJSD"),
        ("A1_range_smape", "RangeSMAPE"),
        ("OD_spearman", "OD $\\rho$"),
        ("OD_top20_overlap", "OD@20"),
        ("OD_ndcg20", "NDCG@20"),
        ("B1_dest8_top1", "Dest8@1"),
        ("B1_dest8_top5", "Dest8@5"),
        ("B1_dest16_top5", "Dest16@5"),
        ("B2_grid_route_top1_hit", "GridRoute@1"),
        ("B2_grid_route_top5_hit", "GridRoute@5"),
        ("B2_grid_route_best5_transition_f1", "GridTransF1"),
    ]
    lines = [
        "\\begin{tabular}{lrrrrrrrrrrr}",
        "\\toprule",
        "Method & " + " & ".join(label for _, label in detail_cols) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        vals = [tex_float(row.get(k, math.nan)) for k, _ in detail_cols]
        lines.append(row["method"] + " & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (PAPER / "tables_kdd_downstream_full_detail.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    print("[load] real full cache", flush=True)
    real = load_real()
    split = int(len(real) * 0.8)
    real_train = real[:split]
    real_eval = real[split:]
    print(f"[load] real={len(real)} train={len(real_train)} eval={len(real_eval)}", flush=True)
    rows = []

    upper = {"method": "Real-train upper"}
    upper.update(count_query_metrics(real, real_train))
    upper.update(od_demand_metrics(real, real_train))
    upper.update(destination_tstr_metrics(real_train, real_eval))
    upper.update(grid_route_choice_metrics(real_train, real_eval))
    rows.append(upper)

    for name in METHODS:
        print(f"[method] {name}", flush=True)
        syn = load_synthetic(name)
        row = {"method": DISPLAY.get(name, name)}
        row.update(count_query_metrics(real, syn))
        row.update(od_demand_metrics(real, syn))
        row.update(destination_tstr_metrics(syn, real_eval))
        row.update(grid_route_choice_metrics(syn, real_eval))
        rows.append(row)
        print(
            f"  Count={row['A1_range_smape']:.3f} OD@20={row['OD_top20_overlap']:.3f} "
            f"Dest@5={row['B1_dest8_top5']:.3f} "
            f"GridRoute@5={row['B2_grid_route_top5_hit']:.3f}",
            flush=True,
        )

    summary = {
        "protocol": {
            "real_path": str(FULL_REAL),
            "synthetic_dir": str(GEN),
            "n_real": len(real),
            "n_real_train": len(real_train),
            "n_real_eval": len(real_eval),
            "methods": METHODS,
            "tasks": {
                "A1": "multi-resolution trajectory-occupancy count queries",
                "A3_A4": "OD demand mining: Spearman, top-20 overlap, NDCG@20",
                "B1": "TSTR destination prediction at 8x8 and 16x16 grids",
                "B2": "OD-conditioned route prototype retrieval with 16x16 corridor transitions",
            },
        },
        "rows": rows,
        "elapsed_sec": time.perf_counter() - t0,
    }
    write_outputs(rows, summary)
    print(f"[done] {OUT / 'downstream_full_protocol.json'}", flush=True)
    print(f"[done] {PAPER / 'tables_kdd_downstream_full.tex'}", flush=True)


if __name__ == "__main__":
    main()
