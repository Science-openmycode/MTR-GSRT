"""Recreate the four public family resamplers from one saved STMatch carrier.

The STMatch carrier is an explicit public adapter of the named synthetic corpus.
No real trajectory, real-derived route pool, or other method's private release is
read by these resamplers. Every output keeps the carrier's fixed slot count.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from route_metric_core import load_pickle, normalize_routes, valid_routes  # noqa: E402


def rooted(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def family(route, labels) -> tuple[int, ...]:
    if not route:
        return ()
    values = [labels.get(route[0][0], -1)] + [labels.get(edge[1], -1) for edge in route]
    output = []
    for value in values:
        value = int(value)
        if not output or value != output[-1]:
            output.append(value)
    return tuple(output)


def family_pool(routes, labels):
    grouped = defaultdict(list)
    for route in valid_routes(routes):
        grouped[family(route, labels)].append(route)
    return grouped


def head_counts(routes, labels, threshold: int) -> Counter:
    counts = Counter(family(route, labels) for route in valid_routes(routes))
    return Counter({key: value for key, value in counts.items() if value >= threshold})


def allocate_mass(weights: dict, total: int) -> Counter:
    raw = {key: max(float(value), 0.0) * total for key, value in weights.items()}
    result = Counter({key: int(math.floor(value)) for key, value in raw.items()})
    remaining = total - sum(result.values())
    if remaining > 0:
        for key in sorted(raw, key=lambda item: raw[item] - result[item], reverse=True)[:remaining]:
            result[key] += 1
    return +result


def sample_routes(pool, count: int, rng):
    if count <= 0:
        return []
    indices = rng.choice(len(pool), count, replace=count > len(pool))
    return [pool[int(index)] for index in indices]


def realize_counts(counts: Counter, primary_pool, fallback, slots: int, rng):
    output, unsupported = [], 0
    for value, count in counts.items():
        options = primary_pool.get(value, ())
        if not options:
            unsupported += count
            continue
        order = rng.permutation(len(options))
        ordered = [options[int(index)] for index in order]
        output.extend(ordered[index % len(ordered)] for index in range(count))
    output.extend(sample_routes(fallback, slots - len(output), rng))
    rng.shuffle(output)
    return output, unsupported


def additive(head, source_total, primary_pool, tail, slots, rng):
    target = {key: value / max(source_total, 1) for key, value in head.items()}
    head_mass = min(sum(target.values()), 1.0)
    counts = allocate_mass(target, slots)
    wanted = int(round(head_mass * slots))
    if sum(counts.values()) != wanted:
        counts = allocate_mass({key: value / max(sum(target.values()), 1e-15)
                                for key, value in target.items()}, wanted)
    return realize_counts(counts, primary_pool, tail, slots, rng)


def length_ot(head, source_total, primary_pool, tail, slots, rng):
    target = {key: value / max(source_total, 1) for key, value in head.items()}
    wanted = int(round(min(sum(target.values()), 1.0) * slots))
    norm = sum(target.values())
    counts = allocate_mass({key: value / norm for key, value in target.items()}, wanted) if norm else Counter()
    output, unsupported = [], 0
    for value, count in counts.items():
        options = primary_pool.get(value, ())
        if not options:
            unsupported += count
            continue
        ordered = sorted(options, key=len)
        quantiles = (np.arange(count, dtype=float) + .5) / max(count, 1)
        indices = np.minimum((quantiles * len(ordered)).astype(int), len(ordered) - 1)
        output.extend(ordered[int(index)] for index in indices)
    output.extend(sample_routes(tail, slots - len(output), rng))
    rng.shuffle(output)
    return output, unsupported


def residual(head, source_total, primary_pool, tail, labels, slots, rng):
    target = {key: value / max(source_total, 1) for key, value in head.items()}
    tail_counts = Counter(family(route, labels) for route in valid_routes(tail))
    total = float(sum(tail_counts.values()))
    q = {key: value / total for key, value in tail_counts.items()} if total else {}
    low, high = 0.0, 1.0
    for _ in range(80):
        beta = (low + high) / 2.0
        value = beta + sum(max(weight - beta * q.get(key, 0.0), 0.0)
                           for key, weight in target.items())
        if value < 1.0:
            low = beta
        else:
            high = beta
    beta = (low + high) / 2.0
    weights = {key: max(value - beta * q.get(key, 0.0), 0.0)
               for key, value in target.items()}
    wanted = int(round((1.0 - beta) * slots))
    norm = sum(weights.values())
    counts = allocate_mass({key: value / norm for key, value in weights.items()}, wanted) if norm else Counter()
    routes, unsupported = realize_counts(counts, primary_pool, tail, slots, rng)
    return routes, unsupported, beta


def carrier_reweight(head, source_total, carrier, tail, labels, slots, rng):
    carrier_pool = family_pool(carrier, labels)
    supported = {key: value / max(source_total, 1) for key, value in head.items()
                 if key in carrier_pool}
    wanted = int(round(min(sum(supported.values()), 1.0) * slots))
    norm = sum(supported.values())
    counts = allocate_mass({key: value / norm for key, value in supported.items()}, wanted) if norm else Counter()
    return realize_counts(counts, carrier_pool, tail, slots, rng)


def save(path: Path, routes) -> None:
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as packed:
            pickle.dump(routes, packed, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stmatch-route", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--method-index", type=int, required=True,
                        help="historical seed index: SPRT=0, PrivTrace=1, DPTraj-PM=2, DPStd=3, MTR-GSRT=5")
    parser.add_argument("--edge-cache", default="public_assets/ordered_portal_route_cache.pkl.gz")
    parser.add_argument("--head-threshold", type=int, default=36)
    parser.add_argument("--seed-schedule", choices=("full-road-v1", "tstr-v1"),
                        default="full-road-v1",
                        help="Select the frozen independent random streams for the full-road or strict TSTR experiment")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    # The historical seed schedule also had a no-Q5 row at index 4. Retain
    # that gap so this five-method comparison reproduces its MTR-GSRT routes.
    methods = ("SPRT", "PrivTrace", "DPTraj-PM", "DPStd", "MTR no-Q5", "MTR-GSRT")
    if args.method_index not in range(len(methods)) or args.method != methods[args.method_index]:
        parser.error("--method and --method-index must match the frozen method order")
    if args.head_threshold < 1:
        parser.error("--head-threshold must be positive")
    source, cache_path, output = rooted(args.stmatch_route), rooted(args.edge_cache), rooted(args.out_dir)
    if output.exists():
        raise FileExistsError(output)
    cache = load_pickle(cache_path)
    base = normalize_routes(load_pickle(source), cache)
    slots = len(base)
    if slots == 0:
        raise ValueError("STMatch carrier is empty")
    labels = {int(k): int(v) for k, v in cache["labels96"].items()}
    head = head_counts(base, labels, args.head_threshold)
    pool = family_pool(base, labels)
    index = args.method_index
    seed_bases = ((20271000, 20272000, 20273000, 20274000)
                  if args.seed_schedule == "full-road-v1"
                  else (20281000, 20282000, 20283000, 20284000))
    generated = {
        "Family_additive": (*additive(head, slots, pool, base, slots,
                                      np.random.default_rng(seed_bases[0] + index)), None),
        "Family_residual": residual(head, slots, pool, base, labels, slots,
                                    np.random.default_rng(seed_bases[1] + index)),
        "Family_length-OT": (*length_ot(head, slots, pool, base, slots,
                                        np.random.default_rng(seed_bases[2] + index)), None),
        "Self-carrier_reweight": (*carrier_reweight(head, slots, base, base, labels, slots,
                                                    np.random.default_rng(seed_bases[3] + index)), None),
    }
    output.mkdir(parents=True)
    for name, (routes, unsupported, beta) in generated.items():
        if len(routes) != slots:
            raise AssertionError(f"{name} changed the public slot count")
        path = output / f"{name}.pkl.gz"
        save(path, routes)
        (output / f"{name}.manifest.json").write_text(json.dumps({
            "schema": "single_m_family_router_v1", "method": args.method, "router": name,
            "stmatch_carrier_sha256": digest(source), "edge_cache_sha256": digest(cache_path),
            "output_sha256": digest(path), "slots": slots,
            "valid_routes": len(valid_routes(routes)), "head_threshold": args.head_threshold,
            "head_families": len(head), "unsupported_head_slots": unsupported,
            "residual_beta": beta,
            "seed_schedule": args.seed_schedule, "seed": seed_bases[("Family_additive", "Family_residual", "Family_length-OT", "Self-carrier_reweight").index(name)] + index,
            "private_source_rule": "only this method's saved STMatch carrier; no real or foreign private route pool",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{args.method}/{name}: {slots} slots, {len(valid_routes(routes))} connected")


if __name__ == "__main__":
    main()
