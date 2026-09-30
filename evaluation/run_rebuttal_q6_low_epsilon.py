"""Complete low-budget audit across every MTR-GSRT query family and release layer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import t


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PUBLIC = ROOT / "generation" / "mtr_gsrt"
EPSILON_DIRS = {
    "1/5": "eps_1_5", "2/5": "eps_2_5", "7/10": "eps_7_10",
    "1": "eps_1_1", "7/5": "eps_7_5", "2": "eps_2_1", "3": "eps_3_1",
}
SEEDS = tuple(range(20260719, 20260724))
BASE_FRACTIONS = {
    "endpoint": Fraction(8, 35), "od": Fraction(33, 140),
    "geometry": Fraction(4, 35), "length": Fraction(19, 140),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dist(values):
    values = np.maximum(np.asarray(values, dtype=float).ravel(), 0.0)
    return values / max(float(values.sum()), 1e-15)


def cpc(first, second):
    return float(np.minimum(dist(first), dist(second)).sum())


def jsd(first, second):
    p, q = dist(first), dist(second); m = 0.5 * (p + q)
    total = 0.0
    for values in (p, q):
        keep = values > 0
        total += 0.5 * float(np.sum(values[keep] * np.log2(values[keep] / m[keep])))
    return total


def expected_abs_noise(epsilon: Fraction, sensitivity: int) -> float:
    x = float(epsilon) / float(sensitivity)
    a = math.exp(-x)
    return float(2.0 * a / (-math.expm1(-2.0 * x)))


def dominated_mass(exact, noise):
    exact = np.asarray(exact, dtype=float).ravel()
    positive = exact > 0
    dominated = positive & (exact <= noise)
    return float(exact[dominated].sum() / max(float(exact.sum()), 1e-15))


def ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def exact_workloads(real_path: Path, osm_path: Path, bbox: tuple[float, ...], capacity: int):
    sys.path.insert(0, str(PUBLIC))
    from generation.common.runtime import add_runtime_paths
    from generation.mtr.generate_sweep import _base_totals, _portal_totals
    from public_utils import filter_osm_ways_by_bbox, load_trajectories
    add_runtime_paths()
    import audit_two_level_semimarkov_dp as audit
    import compact_graph_flow_experiment as compact
    import graph_voronoi_doptimal_support_probe as support
    import route_structure_potential_experiment as route

    real = load_trajectories(str(real_path), limit=None)
    if len(real) != capacity:
        raise ValueError(f"Expected {capacity} complete records, found {len(real)}")
    with osm_path.open("rb") as handle:
        osm = filter_osm_ways_by_bbox(pickle.load(handle), bbox)
    support.BBOX = bbox
    coords, graph = route.prepare_graph([], bbox=bbox, osm_ways=osm, raw_graph=False)
    endpoint_coords = coords[np.asarray(support.farthest_point_landmarks(coords, 256), dtype=int)]
    coarse_coords = coords[np.asarray(support.farthest_point_landmarks(coords, 24), dtype=int)]
    base = _base_totals(
        real, cKDTree(endpoint_coords), cKDTree(coarse_coords), support.standardized_xy(coarse_coords)
    )
    context96, _ = audit.build_context(coords, graph, 24, 96, 4, 3)
    flow = compact.aggregate(real, context96)
    q5 = _portal_totals(real, coords, graph)
    return base, flow, q5


def generation_seconds(protocol_path: Path, protocol: dict,
                       performance_path: Path | None) -> float:
    if performance_path is None:
        if "elapsed_sec" not in protocol:
            raise ValueError(f"{protocol_path}: supply --generation-performance-root for new releases")
        return float(protocol["elapsed_sec"])
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    if performance.get("classification") != "LOCAL_PERFORMANCE_DIAGNOSTIC_NOT_DP_RELEASE":
        raise ValueError(f"{performance_path}: wrong performance-log classification")
    if performance.get("release_protocol_sha256") != sha256_file(protocol_path):
        raise ValueError(f"{performance_path}: release protocol SHA-256 mismatch")
    elapsed = float(performance["generation_elapsed_sec"])
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f"{performance_path}: invalid generation time")
    return elapsed


def utility_rows(route_choice_csv: Path, metrics_root: Path, generation_root: Path,
                 performance_root: Path | None = None):
    rows = []
    route_choice = {}
    with route_choice_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            route_choice[(row["epsilon"], int(row["seed"]), row["metric"])] = float(row["value"])
    for epsilon, directory in EPSILON_DIRS.items():
        for seed in SEEDS:
            metric_path = metrics_root / directory / f"seed_{seed}" / "metrics/metrics.json"
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))["metrics"]
            protocol_path = generation_root / directory / f"seed_{seed}" / "protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            performance_path = (performance_root / directory / f"seed_{seed}.json"
                                if performance_root else None)
            item = {
                "epsilon": epsilon, "seed": seed,
                "Grid": metrics["grid_density_jsd"], "Trip": metrics["trip_error"],
                "Len": metrics["path_length_jsd"], "OD": metrics["OD_jsd"],
                "RoadSeg": metrics["road_segment_jsd"],
                "DirectedRoadValidity": metrics.get(
                    "coordinate_projection_directed_road_validity",
                    metrics["directed_road_validity"]),
                "RouteCompatibleYield": metrics.get(
                    "coordinate_projection_route_compatible_yield",
                    metrics["route_compatible_yield"]),
                "RouteMRR": (metrics["B2_route_mrr"] if "B2_route_mrr" in metrics
                             else metrics["B2_grid_route_mrr"]),
                "DestinationTop5": metrics["B1_dest8_top5"],
                "WitnessValid": metrics["witness_valid"],
                "RC_NDCG": route_choice[(epsilon, seed, "road_choice_ndcg")],
                "NextRoadAcc": route_choice[(epsilon, seed, "next_road_accuracy")],
                "NextRoadNLL": route_choice[(epsilon, seed, "next_road_nll")],
                "FallbackCount": protocol["decoder"]["fallback_count"],
                "GenerationSeconds": generation_seconds(protocol_path, protocol, performance_path),
            }
            rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True)
    parser.add_argument("--osm-cache", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--public-capacity", type=int, required=True)
    parser.add_argument("--generation-root", required=True)
    parser.add_argument("--generation-performance-root",
                        help="external local timing logs under eps_*/seed_*.json for new releases")
    parser.add_argument("--metrics-root", required=True)
    parser.add_argument("--route-choice-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-sha256")
    parser.add_argument("--osm-sha256")
    args = parser.parse_args()
    def rooted(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else ROOT / path).resolve()
    real_path = rooted(args.real)
    osm_path = rooted(args.osm_cache)
    generation_root = rooted(args.generation_root)
    performance_root = (rooted(args.generation_performance_root)
                        if args.generation_performance_root else None)
    metrics_root = rooted(args.metrics_root)
    route_choice_csv = rooted(args.route_choice_csv)
    out = rooted(args.out_dir)
    if out.exists():
        parser.error(f"Output directory already exists: {out}")
    for path, expected in ((real_path, args.input_sha256), (osm_path, args.osm_sha256)):
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected and sha256_file(path).lower() != expected.lower():
            raise ValueError(f"SHA-256 mismatch: {path}")
    if args.public_capacity <= 0:
        parser.error("--public-capacity must be positive")
    for directory in (generation_root, metrics_root):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if not route_choice_csv.is_file():
        raise FileNotFoundError(route_choice_csv)
    input_files = {
        "real": real_path,
        "osm_cache": osm_path,
        "route_choice_csv": route_choice_csv,
    }
    for _, directory in EPSILON_DIRS.items():
        for seed in SEEDS:
            prefix = f"{directory}/seed_{seed}"
            input_files[f"{prefix}/dp_transcript.npz"] = generation_root / prefix / "dp_transcript.npz"
            input_files[f"{prefix}/protocol.json"] = generation_root / prefix / "protocol.json"
            input_files[f"{prefix}/metrics.json"] = metrics_root / prefix / "metrics/metrics.json"
            if performance_root is not None:
                input_files[f"{directory}/seed_{seed}.performance.json"] = (
                    performance_root / directory / f"seed_{seed}.json"
                )
    missing = [name for name, path in input_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Q6 matrix inputs: {missing}")
    global OUT
    OUT = out
    OUT.mkdir(parents=True, exist_ok=True)
    base, flow, q5 = exact_workloads(real_path, osm_path, tuple(args.bbox), args.public_capacity)
    exact = {
        "base_endpoint": base["endpoint"], "base_od": base["od"],
        "base_geometry": base["geometry"], "base_length": base["point_length"],
        "flow_fine_occupancy": flow["fine_occupancy"],
        "flow_fine_flow": flow["fine_flow"], "flow_dwell": flow["dwell"],
        **{f"q5_{name}": values for name, values in q5.items()},
    }
    query_family = {
        **{f"base_{name}": "Demand" for name in ("endpoint", "od", "geometry", "length")},
        **{f"flow_{name}": "Graph-flow" for name in ("fine_occupancy", "fine_flow", "dwell")},
        **{f"q5_{name}": "Portal-Fiber" for name in q5},
    }
    detailed = []
    for epsilon_text, directory in EPSILON_DIRS.items():
        epsilon = Fraction(epsilon_text)
        noise_by_block = {
            "base_endpoint": expected_abs_noise(epsilon * BASE_FRACTIONS["endpoint"], 2),
            "base_od": expected_abs_noise(epsilon * BASE_FRACTIONS["od"], 1),
            "base_geometry": expected_abs_noise(epsilon * BASE_FRACTIONS["geometry"], 1),
            "base_length": expected_abs_noise(epsilon * BASE_FRACTIONS["length"], 1),
        }
        graph_noise = expected_abs_noise(epsilon / 7, 1_000_000)
        for name in exact:
            if name.startswith("flow_") or name.startswith("q5_"):
                noise_by_block[name] = graph_noise
        for seed in SEEDS:
            path = generation_root / directory / f"seed_{seed}" / "dp_transcript.npz"
            with np.load(path, allow_pickle=False) as archive:
                for name, truth in exact.items():
                    released = np.asarray(archive[name], dtype=float)
                    detailed.append({
                        "epsilon": epsilon_text, "epsilon_float": float(epsilon), "seed": seed,
                        "family": query_family[name], "block": name,
                        "cpc": cpc(truth, released), "jsd": jsd(truth, released),
                        "noise_dominated_mass": dominated_mass(truth, noise_by_block[name]),
                        "expected_abs_noise": noise_by_block[name],
                    })
    detail_path = OUT / "q6_all_query_blocks_detailed.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0])); writer.writeheader(); writer.writerows(detailed)

    query_summary = []
    for epsilon in EPSILON_DIRS:
        for family in ("Demand", "Graph-flow", "Portal-Fiber"):
            current = [row for row in detailed if row["epsilon"] == epsilon and row["family"] == family]
            by_seed = defaultdict(list)
            for row in current:
                by_seed[row["seed"]].append(row)
            seed_cpc = [float(np.mean([r["cpc"] for r in values])) for values in by_seed.values()]
            seed_jsd = [float(np.mean([r["jsd"] for r in values])) for values in by_seed.values()]
            seed_dom = [float(np.mean([r["noise_dominated_mass"] for r in values])) for values in by_seed.values()]
            cpc_m, cpc_l, cpc_h = ci(seed_cpc); jsd_m, jsd_l, jsd_h = ci(seed_jsd)
            query_summary.append({
                "epsilon": epsilon, "family": family,
                "mean_block_cpc": cpc_m, "cpc_ci95_low": cpc_l, "cpc_ci95_high": cpc_h,
                "mean_block_jsd": jsd_m, "jsd_ci95_low": jsd_l, "jsd_ci95_high": jsd_h,
                "mean_noise_dominated_mass": float(np.mean(seed_dom)),
            })
    qsum_path = OUT / "q6_query_family_summary.csv"
    with qsum_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(query_summary[0])); writer.writeheader(); writer.writerows(query_summary)

    utility = utility_rows(route_choice_csv, metrics_root, generation_root, performance_root)
    utility_path = OUT / "q6_full_release_utility_detailed.csv"
    with utility_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(utility[0])); writer.writeheader(); writer.writerows(utility)
    utility_summary = []
    for epsilon in EPSILON_DIRS:
        current = [row for row in utility if row["epsilon"] == epsilon]
        item = {"epsilon": epsilon, "seed_count": len(current)}
        for metric in list(current[0])[2:]:
            values = [float(row[metric]) for row in current]
            mean, low, high = ci(values)
            item[f"{metric}_mean"] = mean; item[f"{metric}_ci95_low"] = low; item[f"{metric}_ci95_high"] = high
        utility_summary.append(item)
    usum_path = OUT / "q6_full_release_utility_summary.csv"
    with usum_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(utility_summary[0])); writer.writeheader(); writer.writerows(utility_summary)
    (OUT / "q6_complete_low_epsilon_manifest.json").write_text(json.dumps({
        "classification": "INTERNAL_PRIVATE_UTILITY_DIAGNOSTIC_NOT_A_DP_RELEASE",
        "exact_workloads_serialized": False,
        "epsilon_values": list(EPSILON_DIRS), "seeds": list(SEEDS),
        "query_families": ["Demand", "Graph-flow", "Portal-Fiber"],
        "release_layers": ["statistical", "road", "route-choice", "task", "fallback", "runtime"],
        "input_hashes": {name: sha256_file(path) for name, path in input_files.items()},
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"query_summary": query_summary, "utility_summary": utility_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
