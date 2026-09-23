"""Representation-neutral TSTR mobility workloads for every synthetic release.

These tasks deliberately operate on one public grid, so coordinate, cell, and
road-path releases can all participate without evaluator-side road repair.
Road-native workloads remain in evaluate_road_mining_tasks.py.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PUBLIC_RELEASE = Path(__file__).resolve().parents[1]
for path in [PUBLIC_RELEASE, PUBLIC_RELEASE / "pipeline", PUBLIC_RELEASE / "src" / "plotting", PUBLIC_RELEASE / "src" / "experiments"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from public_utils import PUBLIC_ROOT, jsonable, load_trajectories, parse_bbox, write_json
from evaluate_kdd_revised_statistics import ids, sampled


def range_query_are_fast(train: list[np.ndarray], test: list[np.ndarray], bbox, seed: int, n_queries: int = 100) -> float:
    """Same trajectory-intersects-rectangle workload as the legacy evaluator, vectorized per trajectory."""
    rng = np.random.default_rng(seed); lat0, lat1, lon0, lon1 = bbox
    rects = np.empty((n_queries, 4), dtype=float)
    for i in range(n_queries):
        a, b = sorted(rng.uniform(lat0, lat1, 2)); c, d = sorted(rng.uniform(lon0, lon1, 2))
        if b - a < .03 * (lat1 - lat0): b = min(lat1, a + .03 * (lat1 - lat0))
        if d - c < .03 * (lon1 - lon0): d = min(lon1, c + .03 * (lon1 - lon0))
        rects[i] = a, b, c, d
    def answers(trajs):
        counts = np.zeros(n_queries, dtype=float)
        for traj in trajs:
            pts = sampled(np.asarray(traj, dtype=float), 128)
            if len(pts) == 0: continue
            hit = ((pts[:, None, 0] >= rects[None, :, 0]) & (pts[:, None, 0] <= rects[None, :, 1]) &
                   (pts[:, None, 1] >= rects[None, :, 2]) & (pts[:, None, 1] <= rects[None, :, 3])).any(axis=0)
            counts += hit
        return counts / max(len(trajs), 1)
    a, b = answers(train), answers(test)
    return float(np.mean(np.abs(a - b) / np.maximum(b, 1.0 / max(len(test), 1))))


def grid_sequences(trajs: list[np.ndarray], bbox, grid: int = 32) -> list[list[int]]:
    out = []
    for traj in trajs:
        cells = ids(sampled(np.asarray(traj, dtype=float), 128), bbox, grid).astype(int).tolist()
        cells = [x for i, x in enumerate(cells) if i == 0 or x != cells[i - 1]]
        if len(cells) >= 3:
            out.append(cells)
    return out


def _rank_metrics(test_cases, model2, model1) -> dict[str, float]:
    hit1 = hit5 = rr = 0.0; covered = total = 0
    for a, b, truth in test_cases:
        total += 1
        counts = model2.get((a, b)) or model1.get(b)
        if not counts:
            continue
        covered += 1
        ranked = [x for x, _ in counts.most_common(5)]
        if ranked and ranked[0] == truth: hit1 += 1
        if truth in ranked:
            rank = ranked.index(truth) + 1; hit5 += 1; rr += 1.0 / rank
    d = max(covered, 1)
    return {"coverage": covered / max(total, 1), "hit_at_1": hit1 / d, "hit_at_5": hit5 / d, "mrr": rr / d, "query_count": float(total)}


def next_cell_metrics(train: list[list[int]], test: list[list[int]]) -> dict[str, float]:
    model2: dict[tuple[int, int], Counter] = defaultdict(Counter)
    model1: dict[int, Counter] = defaultdict(Counter)
    for seq in train:
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
            model2[(a, b)][c] += 1; model1[b][c] += 1
    cases = [(a, b, c) for seq in test for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:])]
    return {f"next_cell_{k}": v for k, v in _rank_metrics(cases, model2, model1).items()}


def destination_metrics(train: list[list[int]], test: list[list[int]]) -> dict[str, float]:
    """Predict final public-grid cell from the first 30% of a trajectory."""
    model2: dict[tuple[int, int], Counter] = defaultdict(Counter)
    model1: dict[int, Counter] = defaultdict(Counter)
    for seq in train:
        cut = max(2, int(math.ceil(.30 * len(seq))))
        if cut < len(seq):
            model2[(seq[0], seq[cut - 1])][seq[-1]] += 1
            model1[seq[cut - 1]][seq[-1]] += 1
    cases = []
    for seq in test:
        cut = max(2, int(math.ceil(.30 * len(seq))))
        if cut < len(seq): cases.append((seq[0], seq[cut - 1], seq[-1]))
    return {f"destination_{k}": v for k, v in _rank_metrics(cases, model2, model1).items()}


def evaluate(name: str, train_trajs, test_seqs, bbox, seed: int) -> dict[str, float | str]:
    seqs = grid_sequences(train_trajs, bbox)
    row: dict[str, float | str] = {"method": name, "release_trajectory_count": float(len(train_trajs)), "eligible_grid_sequences": float(len(seqs))}
    row.update(next_cell_metrics(seqs, test_seqs)); row.update(destination_metrics(seqs, test_seqs))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Representation-neutral full-release TSTR mobility tasks.")
    p.add_argument("--train-real", required=True); p.add_argument("--test-real", required=True)
    p.add_argument("--synthetic", action="append", default=[]); p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--bbox", nargs=4, type=float, default=parse_bbox(None)); p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--out-dir", default=str(PUBLIC_ROOT / "outputs" / "generic_mobility_tasks"))
    args = p.parse_args()
    if not args.synthetic: raise ValueError("at least one --synthetic release is required")
    names = args.names or [Path(x).stem for x in args.synthetic]
    if len(names) != len(args.synthetic): raise ValueError("--names must match --synthetic")
    bbox = tuple(args.bbox); out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    real_train, real_test = load_trajectories(args.train_real, None), load_trajectories(args.test_real, None)
    test_seqs = grid_sequences(real_test, bbox)
    print(f"[generic-tstr] held-out grid sequences: {len(test_seqs):,}; train={len(real_train):,}, test={len(real_test):,}", flush=True)
    def make_row(name, paths):
        row = evaluate(name, paths, test_seqs, bbox, args.seed)
        row["range_query_are"] = range_query_are_fast(paths, real_test, bbox, args.seed)
        return row
    rows = [make_row("Real-train", real_train)]
    for name, path in zip(names, args.synthetic):
        print(f"[generic-tstr] evaluating {name}", flush=True)
        rows.append(make_row(name, load_trajectories(path, None)))
    fields = list(rows[0])
    with (out / "generic_mobility_tasks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{k: jsonable(v) for k, v in x.items()} for x in rows])
    write_json(out / "generic_mobility_tasks.json", {"protocol": "strict_full_release_tstr_generic_grid_v1", "grid": 32, "train_count": len(real_train), "test_count": len(real_test), "rows": rows})
    print(f"[generic-tstr] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
