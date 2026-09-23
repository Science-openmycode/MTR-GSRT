"""Revised KDD statistical evaluation: fidelity plus directed-road validity.

This evaluator deliberately does not use the legacy admission-gate metrics.
All methods are compared with a shared real reference, grid, public OSM graph,
and deterministic finite-sample real-vs-real calibration row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon

PUBLIC_RELEASE = Path(__file__).resolve().parents[1]
for path in [PUBLIC_RELEASE, PUBLIC_RELEASE / "src" / "plotting", PUBLIC_RELEASE / "src" / "experiments", PUBLIC_RELEASE / "src" / "mtr" / "shared_graph"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from public_utils import PUBLIC_ROOT, filter_osm_ways_by_bbox, jsonable, load_osm_ways, load_trajectories, parse_bbox, write_json
from visualize_city_independent_osm_synthesis import osm_graph

EARTH_KM = 6371.0088


def km_xy(points: np.ndarray, lat0: float) -> np.ndarray:
    a = np.asarray(points, dtype=float)
    return np.column_stack((np.deg2rad(a[:, 1]) * EARTH_KM * math.cos(math.radians(lat0)), np.deg2rad(a[:, 0]) * EARTH_KM))


def segment_lengths_km(traj: np.ndarray) -> np.ndarray:
    a = np.asarray(traj, dtype=float)
    if len(a) < 2:
        return np.zeros(0)
    p, q = np.deg2rad(a[:-1]), np.deg2rad(a[1:])
    h = np.sin((q[:, 0] - p[:, 0]) / 2) ** 2 + np.cos(p[:, 0]) * np.cos(q[:, 0]) * np.sin((q[:, 1] - p[:, 1]) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.minimum(1.0, np.sqrt(h)))


def sampled(a: np.ndarray, n: int = 48) -> np.ndarray:
    if len(a) <= n:
        return a
    return a[np.linspace(0, len(a) - 1, n).round().astype(int)]


def ids(points: np.ndarray, bbox: tuple[float, float, float, float], grid: int) -> np.ndarray:
    a = np.asarray(points, dtype=float)
    lat0, lat1, lon0, lon1 = bbox
    r = np.clip(((a[:, 0] - lat0) / max(lat1 - lat0, 1e-12) * grid).astype(int), 0, grid - 1)
    c = np.clip(((a[:, 1] - lon0) / max(lon1 - lon0, 1e-12) * grid).astype(int), 0, grid - 1)
    return r * grid + c


def prob(count: np.ndarray) -> np.ndarray:
    x = np.asarray(count, dtype=float) + 1e-12
    return x / x.sum()


def jsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(jensenshannon(prob(a), prob(b), base=2.0) ** 2)


def grid_density(trajs: list[np.ndarray], bbox: tuple[float, float, float, float], grid: int) -> np.ndarray:
    out = np.zeros(grid * grid)
    for t in trajs:
        np.add.at(out, ids(sampled(np.asarray(t, dtype=float), 128), bbox, grid), 1)
    return out


def endpoint_density(trajs: list[np.ndarray], bbox: tuple[float, float, float, float], grid: int, last: bool) -> np.ndarray:
    out = np.zeros(grid * grid)
    for t in trajs:
        a = np.asarray(t, dtype=float)
        if len(a) >= 2:
            out[int(ids(a[[-1 if last else 0]], bbox, grid)[0])] += 1
    return out


def length_hist(trajs: list[np.ndarray], edges: np.ndarray) -> np.ndarray:
    vals = [segment_lengths_km(t).sum() for t in trajs if len(t) >= 2]
    return np.histogram(vals, bins=edges)[0]


def dtw_km(a: np.ndarray, b: np.ndarray, lat0: float) -> float:
    x, y = km_xy(sampled(a, 32), lat0), km_xy(sampled(b, 32), lat0)
    prev = np.full(len(y) + 1, np.inf); prev[0] = 0.0
    for i in range(1, len(x) + 1):
        cur = np.full(len(y) + 1, np.inf)
        for j in range(1, len(y) + 1):
            cur[j] = float(np.linalg.norm(x[i - 1] - y[j - 1])) + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[-1] / max(len(x) + len(y), 1))


def discrete_frechet_km(a: np.ndarray, b: np.ndarray, lat0: float) -> float:
    """Discrete Frechet distance on the same 32-point public resampling."""
    x, y = km_xy(sampled(a, 32), lat0), km_xy(sampled(b, 32), lat0)
    ca = np.full((len(x), len(y)), np.inf)
    for i in range(len(x)):
        for j in range(len(y)):
            d = float(np.linalg.norm(x[i] - y[j]))
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)
    return float(ca[-1, -1])


def conditional_nn_dtw(
    real: list[np.ndarray],
    syn: list[np.ndarray],
    bbox: tuple[float, float, float, float],
    lat0: float,
    seed: int,
    sample_count: int,
    candidate_count: int,
    od_grid: int,
) -> float:
    """Mean nearest-neighbour DTW within public coarse OD/length strata."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for t in syn:
        a = np.asarray(t, dtype=float)
        if len(a) < 2:
            continue
        key = (int(ids(a[[0]], bbox, od_grid)[0]), int(ids(a[[-1]], bbox, od_grid)[0]), min(5, int(np.log1p(segment_lengths_km(a).sum()))))
        groups.setdefault(key, []).append(a)
    # DTW is an individual-level sample metric.  The fixed public seed keeps
    # the same 400 reference records and at most 24 candidates for every row;
    # all marginal and road-segment metrics still use every release record.
    samples = real if len(real) <= sample_count else [real[i] for i in rng.choice(len(real), sample_count, replace=False)]
    vals = []
    fallback = list(syn)
    for a in samples:
        arr = np.asarray(a, dtype=float)
        if len(arr) < 2:
            continue
        key = (int(ids(arr[[0]], bbox, od_grid)[0]), int(ids(arr[[-1]], bbox, od_grid)[0]), min(5, int(np.log1p(segment_lengths_km(arr).sum()))))
        pool = groups.get(key) or groups.get((key[0], key[1], 0)) or fallback
        if len(pool) > candidate_count:
            pool = [pool[i] for i in rng.choice(len(pool), candidate_count, replace=False)]
        vals.append(min(dtw_km(arr, b, lat0) for b in pool))
    return float(np.mean(vals)) if vals else float("nan")


def conditional_nn_frechet(
    real: list[np.ndarray], syn: list[np.ndarray], bbox: tuple[float, float, float, float], lat0: float,
    seed: int, sample_count: int, candidate_count: int, od_grid: int,
) -> float:
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for t in syn:
        a = np.asarray(t, dtype=float)
        if len(a) < 2:
            continue
        key = (int(ids(a[[0]], bbox, od_grid)[0]), int(ids(a[[-1]], bbox, od_grid)[0]), min(5, int(np.log1p(segment_lengths_km(a).sum()))))
        groups.setdefault(key, []).append(a)
    samples = real if len(real) <= sample_count else [real[i] for i in rng.choice(len(real), sample_count, replace=False)]
    vals, fallback = [], list(syn)
    for a in samples:
        arr = np.asarray(a, dtype=float)
        if len(arr) < 2:
            continue
        key = (int(ids(arr[[0]], bbox, od_grid)[0]), int(ids(arr[[-1]], bbox, od_grid)[0]), min(5, int(np.log1p(segment_lengths_km(arr).sum()))))
        pool = groups.get(key) or groups.get((key[0], key[1], 0)) or fallback
        if len(pool) > candidate_count:
            pool = [pool[i] for i in rng.choice(len(pool), candidate_count, replace=False)]
        vals.append(min(discrete_frechet_km(arr, b, lat0) for b in pool))
    return float(np.mean(vals)) if vals else float("nan")


def graph_context(osm_ways: list[dict], bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, set[tuple[int, int]], cKDTree, dict[tuple[int, int], int]]:
    coords, graph = osm_graph(osm_ways, bbox)
    # OSM exports may contain duplicate directed adjacency entries.  Deduplicate
    # before indexing so the histogram has one contiguous coordinate per edge.
    edges = sorted({(int(u), int(v)) for u, adj in graph.items() for v, _ in adj})
    edge_index = {e: i for i, e in enumerate(edges)}
    return np.asarray(coords, dtype=float), set(edges), cKDTree(np.asarray(coords, dtype=float)), edge_index


def production_witness_edges(osm_ways: list[dict]) -> set[tuple[int, int]]:
    """Rebuild the exact indexed public graph used by the production decoder."""
    final_dir = PUBLIC_RELEASE / "src" / "mtr" / "DP_GSRT" / "final"
    if str(final_dir) not in sys.path:
        sys.path.insert(0, str(final_dir))
    import graph_voronoi_doptimal_support_probe as support
    import route_structure_potential_experiment as route

    _, graph = route.prepare_graph([], bbox=support.BBOX, osm_ways=osm_ways, raw_graph=False)
    return {(int(u), int(v)) for u, adjacency in graph.items() for v, _ in adjacency}


def road_counts(trajs: list[np.ndarray], coords: np.ndarray, edges: set[tuple[int, int]], tree: cKDTree, edge_index: dict[tuple[int, int], int]) -> tuple[np.ndarray, float, float]:
    counts = np.zeros(len(edge_index) + 1)
    valid, total, compatible = 0, 0, 0
    for t in trajs:
        nodes = tree.query(sampled(np.asarray(t, dtype=float), 256), k=1)[1].astype(int).tolist()
        nodes = [n for i, n in enumerate(nodes) if i == 0 or n != nodes[i - 1]]
        local_valid = local_total = 0
        for u, v in zip(nodes[:-1], nodes[1:]):
            if u == v:
                continue
            total += 1; local_total += 1
            e = (int(u), int(v))
            if e in edges:
                valid += 1; local_valid += 1; counts[edge_index[e]] += 1
            else:
                counts[-1] += 1
        if local_total and local_valid / local_total >= 0.95:
            compatible += 1
    return counts, (valid / total if total else 0.0), compatible / max(len(trajs), 1)


def witness_validity(
    path: str | Path | None,
    edges: set[tuple[int, int]],
    expected_count: int,
    public_graph_sha256: str | None = None,
    coordinate_sha256: str | None = None,
) -> float | None:
    """Validate the consumer-visible directed-edge witness sidecar, if supplied."""
    if path is None:
        return None
    witness_path = Path(path)
    if witness_path.suffix.lower() == ".json":
        package = json.loads(witness_path.read_text(encoding="utf-8"))
        if package.get("schema_id") != "mtr-road-witness-release-v1":
            raise ValueError("Unknown witness release schema")
        if int(package.get("record_count", -1)) != expected_count:
            raise ValueError("Witness package record_count disagrees with the synthetic release")
        package_graph_sha256 = package.get("public_graph", {}).get("sha256")
        if public_graph_sha256 is not None and package_graph_sha256 != public_graph_sha256:
            raise ValueError("Witness package is bound to a different public graph")
        package_coordinate_sha256 = package.get("coordinate_artifact", {}).get("sha256")
        if coordinate_sha256 is not None and package_coordinate_sha256 != coordinate_sha256:
            raise ValueError("Witness package is bound to a different coordinate artifact")
        witnesses = package.get("records")
    else:
        import pickle
        with witness_path.open("rb") as handle:
            witnesses = pickle.load(handle)
    if not isinstance(witnesses, list) or len(witnesses) != expected_count:
        raise ValueError("Witness sidecar must contain one record for every synthetic slot")
    valid = 0
    for slot, witness in enumerate(witnesses):
        if not isinstance(witness, dict) or int(witness.get("slot", -1)) != slot:
            raise ValueError("Witness sidecar slot order is invalid")
        pairs = witness.get("directed_edges")
        nodes = witness.get("node_sequence")
        if not isinstance(pairs, list) or not isinstance(nodes, list):
            continue
        if witness.get("witness_valid_by_construction") is False:
            continue
        try:
            normalized_pairs = [(int(pair[0]), int(pair[1])) for pair in pairs if len(pair) == 2]
            normalized_nodes = [int(node) for node in nodes]
        except (TypeError, ValueError, IndexError):
            continue
        if (
            not normalized_pairs
            or len(normalized_pairs) != len(pairs)
            or len(normalized_nodes) != len(normalized_pairs) + 1
        ):
            continue
        if any(
            normalized_pairs[index] != (normalized_nodes[index], normalized_nodes[index + 1])
            for index in range(len(normalized_pairs))
        ):
            continue
        if all(pair in edges for pair in normalized_pairs):
            valid += 1
    return valid / max(expected_count, 1)


def evaluate(real: list[np.ndarray], syn: list[np.ndarray], bbox: tuple[float, float, float, float], lat0: float, ctx, seed: int, cdtw_samples: int, cdtw_candidates: int, cdtw_od_grid: int) -> dict:
    coords, edges, tree, edge_index = ctx
    all_lengths = np.asarray([segment_lengths_km(t).sum() for t in [*real, *syn] if len(t) >= 2])
    length_edges = np.linspace(0, max(float(np.quantile(all_lengths, .99)), 1e-6), 33)
    rc, rv, ry = road_counts(real, coords, edges, tree, edge_index)
    sc, sv, sy = road_counts(syn, coords, edges, tree, edge_index)
    return {
        "grid_density_jsd": jsd(grid_density(real, bbox, 64), grid_density(syn, bbox, 64)),
        "trip_error": .5 * (jsd(endpoint_density(real, bbox, 32, False), endpoint_density(syn, bbox, 32, False)) + jsd(endpoint_density(real, bbox, 32, True), endpoint_density(syn, bbox, 32, True))),
        "path_length_jsd": jsd(length_hist(real, length_edges), length_hist(syn, length_edges)),
        "conditional_nn_dtw_km": conditional_nn_dtw(real, syn, bbox, lat0, seed, cdtw_samples, cdtw_candidates, cdtw_od_grid),
        "conditional_nn_frechet_km": conditional_nn_frechet(real, syn, bbox, lat0, seed, cdtw_samples, cdtw_candidates, cdtw_od_grid),
        "road_segment_jsd": jsd(rc, sc),
        "directed_road_validity": sv,
        "route_compatible_yield": sy,
    }


def release_metadata(path: str | Path, output_count: int) -> dict:
    """Read optional method metadata without treating it as an evaluation input."""
    release = Path(path)
    manifest_path = release.with_suffix(".json")
    meta = {
        "release_count": int(output_count),
        "input_count": int(output_count),
        "output_yield": 1.0,
        "privacy_unit": "central population release",
    }
    if not manifest_path.exists():
        return meta
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return meta
    if "input_count" in payload:
        meta["input_count"] = int(payload["input_count"])
    if "output_count" in payload:
        meta["release_count"] = int(payload["output_count"])
    if "output_yield" in payload:
        meta["output_yield"] = float(payload["output_yield"])
    if "privacy_unit" in payload:
        meta["privacy_unit"] = str(payload["privacy_unit"])
    return meta


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the revised KDD main statistical protocol.")
    p.add_argument("--real", default="geolife")
    p.add_argument("--synthetic", action="append", required=True)
    p.add_argument("--witness", action="append", default=None)
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--osm-cache", default="data/osm/osm_cache_beijing.pkl")
    p.add_argument("--bbox", nargs=4, type=float, default=parse_bbox(None))
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--cdtw-samples", type=int, default=400)
    p.add_argument("--cdtw-candidates", type=int, default=24)
    p.add_argument("--cdtw-od-grid", type=int, default=12)
    p.add_argument("--out-dir", default=str(PUBLIC_ROOT / "outputs" / "kdd_revised" / "main_statistics"))
    args = p.parse_args(); bbox = tuple(args.bbox)
    out = Path(args.out_dir).absolute(); out.mkdir(parents=True, exist_ok=True)
    print("[revised-stats] loading full real reference", flush=True)
    real = load_trajectories(args.real, limit=None)
    lat0 = float(np.mean(np.vstack([t[[0, -1]] for t in real])[:, 0]))
    ways = filter_osm_ways_by_bbox(load_osm_ways(args.osm_cache), bbox)
    print("[revised-stats] building public directed-road graph", flush=True)
    ctx = graph_context(ways, bbox)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(real)); half = len(real) // 2
    rows = [{
        "method": "Real-vs-real calibration",
        "release_count": half,
        "input_count": half,
        "output_yield": 1.0,
        "privacy_unit": "finite-sample reference",
        **evaluate([real[i] for i in perm[:half]], [real[i] for i in perm[half:2*half]], bbox, lat0, ctx, args.seed, args.cdtw_samples, args.cdtw_candidates, args.cdtw_od_grid),
    }]
    names = args.names or [Path(x).stem for x in args.synthetic]
    if len(names) != len(args.synthetic):
        raise ValueError("--names must have one label per --synthetic file")
    if args.witness is not None and len(args.witness) != len(args.synthetic):
        raise ValueError("--witness must have one optional sidecar per --synthetic file")
    witness_paths = args.witness or [None] * len(args.synthetic)
    for name, path, witness_path in zip(names, args.synthetic, witness_paths):
        print(f"[revised-stats] evaluating {name}", flush=True)
        syn = load_trajectories(path, limit=None)
        rows.append({
            "method": name,
            "synthetic_file": str(Path(path)),
            "synthetic_sha256": sha256_file(path),
            "witness_file": str(Path(witness_path)) if witness_path else None,
            "witness_sha256": sha256_file(witness_path) if witness_path else None,
            "witness_valid": witness_validity(
                witness_path,
                production_witness_edges(ways),
                len(syn),
                sha256_file(args.osm_cache),
                sha256_file(path),
            ),
            **release_metadata(path, len(syn)),
            **evaluate(real, syn, bbox, lat0, ctx, args.seed, args.cdtw_samples, args.cdtw_candidates, args.cdtw_od_grid),
        })
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "main_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{k: jsonable(v) for k, v in r.items()} for r in rows])
    protocol = {
        "name": "kdd_revised_full_beijing_v3",
        "seed": args.seed,
        "real_source": str(args.real),
        "real_sha256": sha256_file(args.real) if Path(args.real).is_file() else None,
        "osm_cache": str(Path(args.osm_cache)),
        "osm_cache_sha256": sha256_file(args.osm_cache),
        "evaluator_sha256": sha256_file(__file__),
        "bbox": bbox,
        "grid_density_grid": 64,
        "endpoint_grid": 32,
        "length_bins": 32,
        "conditional_dtw": {
            "reference_samples": args.cdtw_samples,
            "candidates_per_reference": args.cdtw_candidates,
            "od_grid": args.cdtw_od_grid,
            "length_strata": "min(5, floor(log1p(path_length_km)))",
            "trajectory_resample_points": 32,
            "distances": ["normalized DTW path cost", "discrete Frechet"],
        },
        "road_snap_max_points": 256,
        "route_compatible_threshold": 0.95,
    }
    write_json(out / "main_statistics.json", {"protocol": protocol, "real_count": len(real), "metrics": list(fields[1:]), "rows": rows})
    print(f"[revised-stats] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
