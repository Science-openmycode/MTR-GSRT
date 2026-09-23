"""Strict TSTR evaluation for road-network mining workloads.

Each release trains the same public, topology-aware count model.  Test
queries are extracted only from a held-out real split.  Coordinate releases
are not rerouted or repaired: the evaluator reports their eligible-query
coverage explicitly, which is part of the release object's utility for a
road-network workload.
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
for path in [PUBLIC_RELEASE, PUBLIC_RELEASE / "pipeline", PUBLIC_RELEASE / "src" / "plotting", PUBLIC_RELEASE / "src" / "experiments", PUBLIC_RELEASE / "src" / "mtr" / "shared_graph"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from public_utils import PUBLIC_ROOT, filter_osm_ways_by_bbox, jsonable, load_osm_ways, load_trajectories, parse_bbox, write_json
from evaluate_kdd_revised_statistics import graph_context, sampled


def directed_runs(trajs: list[np.ndarray], tree, edges: set[tuple[int, int]], max_points: int = 256) -> list[list[int]]:
    """Convert points to maximal consecutive runs of public directed OSM edges."""
    runs: list[list[int]] = []
    for traj in trajs:
        pts = sampled(np.asarray(traj, dtype=float), max_points)
        nodes = tree.query(pts, k=1)[1].astype(int).tolist()
        nodes = [u for i, u in enumerate(nodes) if i == 0 or u != nodes[i - 1]]
        current: list[int] = []
        for u, v in zip(nodes[:-1], nodes[1:]):
            if (u, v) in edges:
                if not current:
                    current = [u, v]
                elif current[-1] == u:
                    current.append(v)
                else:
                    if len(current) >= 4:
                        runs.append(current)
                    current = [u, v]
            else:
                if len(current) >= 4:
                    runs.append(current)
                current = []
        if len(current) >= 4:
            runs.append(current)
    return runs


def edge_tokens(seq: list[int]) -> list[tuple[int, int]]:
    return list(zip(seq[:-1], seq[1:]))


def build_backoff_model(runs: list[list[int]]) -> tuple[dict[tuple[int, int], Counter], dict[int, Counter]]:
    tri: dict[tuple[int, int], Counter] = defaultdict(Counter)
    uni: dict[int, Counter] = defaultdict(Counter)
    for seq in runs:
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
            tri[(a, b)][c] += 1
            uni[b][c] += 1
    return tri, uni


def continuation_metrics(train_runs: list[list[int]], test_runs: list[list[int]]) -> dict[str, float]:
    tri, uni = build_backoff_model(train_runs)
    hit1 = hit5 = rr = nll = 0.0
    covered = total = 0
    for seq in test_runs:
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
            total += 1
            counts = tri.get((a, b)) or uni.get(b)
            if not counts:
                continue
            covered += 1
            ranked = [node for node, _ in counts.most_common(5)]
            if ranked and ranked[0] == c:
                hit1 += 1
            if c in ranked:
                rank = ranked.index(c) + 1
                hit5 += 1
                rr += 1.0 / rank
            nll -= math.log((counts[c] + 1e-12) / sum(counts.values()))
    denom = max(covered, 1)
    return {
        "continuation_query_count": float(total),
        "continuation_coverage": covered / max(total, 1),
        "continuation_hit_at_1": hit1 / denom,
        "continuation_hit_at_5": hit5 / denom,
        "continuation_mrr": rr / denom,
        "continuation_nll": nll / denom,
    }


def _qgrams(seq: list[int], q: int = 2) -> set[tuple[int, ...]]:
    return {tuple(seq[i:i + q]) for i in range(max(0, len(seq) - q + 1))}


def retrieval_metrics(train_runs: list[list[int]], test_runs: list[list[int]], max_candidates: int = 512) -> dict[str, float]:
    """Retrieve synthetic routes from observed real prefixes; relevance uses hidden real suffixes."""
    index: dict[tuple[int, ...], set[int]] = defaultdict(set)
    route_grams: list[set[tuple[int, ...]]] = []
    route_edges: list[set[tuple[int, int]]] = []
    for i, seq in enumerate(train_runs):
        grams = _qgrams(seq)
        route_grams.append(grams)
        route_edges.append(set(edge_tokens(seq)))
        for gram in grams:
            index[gram].add(i)
    recall = {1: 0.0, 5: 0.0, 10: 0.0}
    ndcg = {5: 0.0, 10: 0.0}
    mrr = overlap = 0.0
    covered = total = 0
    for seq in test_runs:
        cut = max(3, int(math.ceil(.30 * len(seq))))
        if cut >= len(seq) - 1:
            continue
        prefix, suffix = seq[:cut], set(edge_tokens(seq[cut - 1:]))
        if not suffix:
            continue
        total += 1
        query_grams = _qgrams(prefix)
        candidates: set[int] = set()
        for gram in query_grams:
            candidates.update(index.get(gram, set()))
        if not candidates:
            continue
        covered += 1
        # A fixed cap avoids a high-degree public road node dominating runtime.
        candidate_ids = sorted(candidates)[:max_candidates]
        scored = []
        for i in candidate_ids:
            prefix_score = len(query_grams & route_grams[i])
            inter = len(suffix & route_edges[i])
            union = len(suffix | route_edges[i])
            relevance = inter / union if union else 0.0
            scored.append((prefix_score, relevance, i))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        rel = np.asarray([x[1] for x in scored], dtype=float)
        relevant = rel >= .10
        for k in recall:
            recall[k] += float(np.any(relevant[:k]))
        for k in ndcg:
            gains = rel[:k]
            dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
            ideal = np.sort(rel)[::-1][:k]
            idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
            ndcg[k] += dcg / idcg if idcg else 0.0
        ranks = np.flatnonzero(relevant)
        if len(ranks):
            mrr += 1.0 / (int(ranks[0]) + 1)
        overlap += float(rel[0]) if len(rel) else 0.0
    denom = max(covered, 1)
    return {
        "retrieval_query_count": float(total),
        "retrieval_coverage": covered / max(total, 1),
        "retrieval_recall_at_1": recall[1] / denom,
        "retrieval_recall_at_5": recall[5] / denom,
        "retrieval_recall_at_10": recall[10] / denom,
        "retrieval_ndcg_at_5": ndcg[5] / denom,
        "retrieval_ndcg_at_10": ndcg[10] / denom,
        "retrieval_mrr": mrr / denom,
        "retrieval_top1_suffix_edge_jaccard": overlap / denom,
    }


def corridor_metrics(train_runs: list[list[int]], test_runs: list[list[int]], n: int, k: int) -> dict[str, float]:
    def counts(runs):
        out = Counter()
        for seq in runs:
            out.update(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
        return out
    syn, real = counts(train_runs), counts(test_runs)
    a = {p for p, _ in syn.most_common(k)}
    b = {p for p, _ in real.most_common(k)}
    inter = a & b
    syn_weight = sum(real[p] for p in a)
    real_weight = sum(real[p] for p in b)
    shared = sum(real[p] for p in inter)
    p = shared / syn_weight if syn_weight else 0.0
    r = shared / real_weight if real_weight else 0.0
    return {f"corridor_n{n}_p_at_{k}": p, f"corridor_n{n}_r_at_{k}": r, f"corridor_n{n}_f1_at_{k}": 2 * p * r / max(p + r, 1e-12)}


def evaluate(name: str, train: list[np.ndarray], test_runs: list[list[int]], ctx) -> dict[str, float | str]:
    _coords, edges, tree, _edge_index = ctx
    runs = directed_runs(train, tree, edges)
    row: dict[str, float | str] = {"method": name, "release_trajectory_count": float(len(train)), "eligible_directed_runs": float(len(runs))}
    row.update(continuation_metrics(runs, test_runs))
    row.update(retrieval_metrics(runs, test_runs))
    row.update(corridor_metrics(runs, test_runs, 3, 100))
    row.update(corridor_metrics(runs, test_runs, 4, 100))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Strict full-release TSTR evaluation for road-network mining tasks.")
    p.add_argument("--train-real", required=True)
    p.add_argument("--test-real", required=True)
    p.add_argument("--synthetic", action="append", default=[])
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--osm-cache", default="data/osm/osm_cache_beijing.pkl")
    p.add_argument("--bbox", nargs=4, type=float, default=parse_bbox(None))
    p.add_argument("--out-dir", default=str(PUBLIC_ROOT / "outputs" / "road_network_mining"))
    args = p.parse_args()
    if not args.synthetic:
        raise ValueError("at least one --synthetic release is required")
    names = args.names or [Path(x).stem for x in args.synthetic]
    if len(names) != len(args.synthetic):
        raise ValueError("--names must match --synthetic")
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    bbox = tuple(args.bbox)
    real_train = load_trajectories(args.train_real, None)
    real_test = load_trajectories(args.test_real, None)
    ctx = graph_context(filter_osm_ways_by_bbox(load_osm_ways(args.osm_cache), bbox), bbox)
    print(f"[road-tstr] public graph has {len(ctx[1]):,} directed edges", flush=True)
    test_runs = directed_runs(real_test, ctx[2], ctx[1])
    print(f"[road-tstr] extracted {len(test_runs):,} directed runs from {len(real_test):,} held-out real trajectories", flush=True)
    rows = [evaluate("Real-train", real_train, test_runs, ctx)]
    for name, path in zip(names, args.synthetic):
        print(f"[road-tstr] evaluating {name}: {path}", flush=True)
        rows.append(evaluate(name, load_trajectories(path, None), test_runs, ctx))
    fields = list(rows[0])
    with (out / "road_network_mining_tasks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows([{k: jsonable(v) for k, v in row.items()} for row in rows])
    write_json(out / "road_network_mining_tasks.json", {
        "protocol": "strict_full_release_tstr_road_network_mining_v1",
        "train_count": len(real_train), "test_count": len(real_test), "heldout_directed_runs": len(test_runs),
        "notes": "All releases use one public OSM graph and are evaluated without rerouting. Scores except coverage are conditional on covered held-out queries.",
        "rows": rows,
    })
    print(f"[road-tstr] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
