"""Full-OD ordered physical-road mass recovery from explicit route objects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from route_metric_core import (
    _canonical_decompose, load_pickle, normalize_routes, valid_routes,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SEED = 20260916


def weighted_lcs_mass(left, right, weights=None):
    positions = defaultdict(list)
    for index, token in enumerate(right, start=1):
        positions[token].append(index)
    tree = [0.0] * (len(right) + 1)

    def prefix_max(index):
        result = 0.0
        while index:
            result = max(result, tree[index])
            index -= index & -index
        return result

    for token in left:
        reward = 1.0 if weights is None else float(weights.get(token, 0.0))
        for index in reversed(positions.get(token, ())):
            value = prefix_max(index - 1) + reward
            while index < len(tree):
                if value > tree[index]:
                    tree[index] = value
                index += index & -index
    return prefix_max(len(right))


def exact_transport(left_mass, right_mass, matches):
    rows, columns = matches.shape
    if rows * columns > 250_000:
        raise ValueError("exact transport stratum exceeds 250,000 variables")
    constraints = lil_matrix((rows + columns, rows * columns), dtype=float)
    for row in range(rows):
        constraints[row, row * columns:(row + 1) * columns] = 1.0
    for column in range(columns):
        constraints[rows + column, column::columns] = 1.0
    ceiling = float(np.max(matches))
    result = linprog((ceiling - matches).reshape(-1),
                     A_eq=constraints.tocsr(),
                     b_eq=np.concatenate((left_mass, right_mass)),
                     bounds=(0.0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"transport solver failed: {result.message}")
    return float(np.sum(result.x.reshape(rows, columns) * matches))


def edge_lengths(cache, network):
    frame = gpd.read_file(network)[["id", "length_m"]]
    edge_nodes = {int(key): tuple(value) for key, value in cache["edge_nodes"].items()}
    result = {}
    for row in frame.itertuples(index=False):
        edge = edge_nodes.get(int(row.id))
        if edge is None:
            continue
        length = max(float(row.length_m), 1e-3) / 1000.0
        if edge not in result or length < result[edge]:
            result[edge] = length
    if not result:
        raise ValueError("network and edge cache contain no common road edges")
    return result


def records(routes, labels, prefix):
    result = []
    for index, route in enumerate(routes):
        if not route:
            continue
        backbone, _ = _canonical_decompose(route)
        result.append({
            "id": f"{prefix}:{index}",
            "stratum": (int(labels[route[0][0]]), int(labels[route[-1][1]])),
            "edges": route,
            "turns": tuple(zip(route, route[1:])),
            "backbone": backbone,
        })
    return result


def sampled(records_in, cap, side, stratum):
    salt = f"{SAMPLE_SEED}|{side}|{stratum}"
    return sorted(records_in, key=lambda item: hashlib.sha256(
        f"{salt}|{item['id']}".encode("utf-8")).digest())[:cap]


def aggregate(records_in, view):
    counts = Counter(item[view] for item in records_in)
    sequences = sorted(counts, key=repr)
    masses = np.asarray([counts[sequence] for sequence in sequences], dtype=float)
    masses /= masses.sum()
    return sequences, masses


def sequence_mass(sequence, view, lengths):
    if view == "turns":
        return float(len(sequence))
    return float(sum(lengths.get(edge, 0.0) for edge in sequence))


def score_view(real_groups, synthetic_groups, view, lengths, real_n, synthetic_n, cap):
    matched = real_total = synthetic_total = 0.0
    for stratum in sorted(set(real_groups) | set(synthetic_groups)):
        real_full, synthetic_full = real_groups.get(stratum, []), synthetic_groups.get(stratum, [])
        real_probability = len(real_full) / real_n
        synthetic_probability = len(synthetic_full) / synthetic_n
        if real_full:
            real_seq, real_mass = aggregate(sampled(real_full, cap, "real", stratum), view)
            real_total += real_probability * sum(
                p * sequence_mass(seq, view, lengths) for seq, p in zip(real_seq, real_mass))
        if synthetic_full:
            synthetic_seq, synthetic_mass = aggregate(
                sampled(synthetic_full, cap, "syn", stratum), view)
            synthetic_total += synthetic_probability * sum(
                p * sequence_mass(seq, view, lengths)
                for seq, p in zip(synthetic_seq, synthetic_mass))
        if real_full and synthetic_full:
            weights = None if view == "turns" else lengths
            matrix = np.asarray([
                [weighted_lcs_mass(left, right, weights) for right in synthetic_seq]
                for left in real_seq
            ], dtype=float)
            matched += min(real_probability, synthetic_probability) * exact_transport(
                real_mass, synthetic_mass, matrix)
    return (matched / real_total if real_total else 0.0,
            matched / synthetic_total if synthetic_total else 0.0)


def evaluate(real_routes, synthetic_routes, cache, lengths, method, cap, sampling_label=None):
    labels = {int(key): int(value) for key, value in cache["labels24"].items()}
    real_records = records(real_routes, labels, "real")
    synthetic_records = records(valid_routes(synthetic_routes), labels, sampling_label or method)
    real_groups, synthetic_groups = defaultdict(list), defaultdict(list)
    for record in real_records:
        real_groups[record["stratum"]].append(record)
    for record in synthetic_records:
        synthetic_groups[record["stratum"]].append(record)
    output = {"method": method, "RoadYield": len(synthetic_records) / len(synthetic_routes)}
    output["DemandFid"] = sum(min(len(real_groups[key]) / len(real_records),
                                  len(synthetic_groups[key]) / len(synthetic_routes))
                              for key in set(real_groups) | set(synthetic_groups))
    for view, prefix in (("edges", "Road"), ("turns", "Turn"), ("backbone", "Backbone")):
        recall, support = score_view(real_groups, synthetic_groups, view, lengths,
                                     len(real_records), len(synthetic_routes), cap)
        output[f"{prefix}Recovery"] = recall
        output[f"{prefix}Support"] = support
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-routes", type=Path, required=True)
    parser.add_argument("--synthetic-routes", type=Path, required=True)
    parser.add_argument("--method", required=True, help="stable method label used for SHA-256 stratum sampling")
    parser.add_argument("--sampling-label", help="archived SHA-256 sampling label; defaults to --method")
    parser.add_argument("--edge-cache", type=Path,
                        default=ROOT / "public_assets" / "ordered_portal_route_cache.pkl.gz")
    parser.add_argument("--network", type=Path,
                        default=ROOT / "public_assets" / "beijing_network" / "network.shp")
    parser.add_argument("--cap-per-stratum", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.cap_per_stratum < 2:
        parser.error("--cap-per-stratum must be at least 2")
    inputs = {name: path.resolve() for name, path in (
        ("real_routes", args.real_routes), ("synthetic_routes", args.synthetic_routes),
        ("edge_cache", args.edge_cache), ("network", args.network))}
    cache = load_pickle(inputs["edge_cache"])
    real = valid_routes(normalize_routes(load_pickle(inputs["real_routes"]), cache))
    synthetic = normalize_routes(load_pickle(inputs["synthetic_routes"]), cache)
    if not real or not synthetic:
        raise ValueError("real reference and synthetic route slots must be nonempty")
    result = evaluate(real, synthetic, cache, edge_lengths(cache, inputs["network"]),
                      args.method, args.cap_per_stratum, args.sampling_label)
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    (out / "manifest.json").write_text(json.dumps({
        "protocol": "full_od_ordered_physical_mass_v1",
        "method": args.method,
        "sampling_label": args.sampling_label or args.method,
        "cap_per_stratum": args.cap_per_stratum,
        "inputs_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                          for key, path in inputs.items()},
        "results_sha256": hashlib.sha256((out / "results.csv").read_bytes()).hexdigest(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out / "results.csv")


if __name__ == "__main__":
    main()
