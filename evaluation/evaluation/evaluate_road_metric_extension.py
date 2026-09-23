"""Evaluate candidate road metrics from an existing common STMatch cache.

This command never synthesizes trajectories and never invokes a map matcher.
It verifies and consumes ``matched_paths/*.pkl.gz`` written by
``evaluate_road_route.py`` and derives every candidate metric from that one
cache.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (_candidate / "configs" / "datasets.json").is_file():
            sys.path.insert(0, str(_candidate))
            break
    else:
        raise RuntimeError("cannot locate public_release root")

from generation.common.runtime import (  # noqa: E402
    add_runtime_paths,
    dataset_config,
    public_path,
    sha256_file,
    write_json,
)
from metric_suites.road_population import (  # noqa: E402
    link_flow_wape,
    map_matching_success_rate,
    region_sequences,
    require_minimum_path_edges,
    road_next_trip_scores,
    road_use_jsd,
)


def _load_records(path: Path) -> list[dict]:
    with gzip.open(path, "rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"match cache is not a record list: {path}")
    required = {"source_index", "cpath", "connected", "accepted"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError(f"invalid match record {index} in {path}")
        if int(record["source_index"]) != index:
            raise ValueError(f"noncanonical source index {record['source_index']} at {index} in {path}")
        if bool(record["accepted"]) and not bool(record["connected"]):
            raise ValueError(f"accepted disconnected path at {index} in {path}")
    return records


def _production_graph(config: dict, quotient_regions: int):
    add_runtime_paths()
    from public_utils import filter_osm_ways_by_bbox, filter_osm_ways_by_highway, load_osm_ways

    final_dir = public_path("src/mtr/DP_GSRT/final")
    if str(final_dir) not in sys.path:
        sys.path.insert(0, str(final_dir))
    import nested_quotient_graph as nested
    import route_structure_potential_experiment as route

    bbox = tuple(map(float, config["bbox"]))
    osm_path = public_path(config["osm_cache"])
    osm = filter_osm_ways_by_bbox(load_osm_ways(str(osm_path)), bbox)
    osm = filter_osm_ways_by_highway(osm, list(config.get("osm_highway_classes", [])))
    coordinates, graph = route.prepare_graph([], bbox=bbox, osm_ways=osm, raw_graph=False)
    context, diagnostics, _ = nested.build_nested_context(
        np.asarray(coordinates, dtype=float), graph, coarse_regions=24, fine_regions=quotient_regions
    )
    return (
        np.asarray(coordinates, dtype=float),
        context.node_fine,
        {int(key): tuple(sorted(map(int, value))) for key, value in context.fine_adjacency.items()},
        diagnostics,
        osm_path,
    )


def _edge_to_region(network: Path, config: dict, quotient_regions: int) -> tuple[dict[int, int], dict[int, tuple[int, ...]], dict]:
    add_runtime_paths()
    import geopandas as gpd
    from scipy.spatial import cKDTree

    coordinates, node_regions, adjacency, diagnostics, osm_path = _production_graph(config, quotient_regions)
    frame = gpd.read_file(network)
    required = {"id", "geometry"}
    if not required.issubset(frame.columns):
        raise ValueError(f"road network lacks columns {sorted(required - set(frame.columns))}")
    if frame.crs is None:
        raise ValueError("road network has no CRS")
    geographic = frame.to_crs("EPSG:4326")
    starts = np.asarray(
        [[float(geom.coords[0][1]), float(geom.coords[0][0])] for geom in geographic.geometry],
        dtype=float,
    )
    distance, nearest = cKDTree(coordinates).query(starts, k=1)
    # The map-matching graph retains the largest weak component of all public
    # directed OSM segments, whereas the MTR carrier applies its own public
    # component/filter construction.  Their node IDs therefore need not be
    # identical.  Nearest-node projection defines pi:E->C for every public FMM
    # edge without consulting any trajectory or method output.
    mapping = {
        int(edge_id): int(node_regions[int(node)])
        for edge_id, node in zip(frame["id"].to_numpy(), nearest)
    }
    if len(mapping) != len(frame):
        raise ValueError("road network edge IDs are not unique")
    return mapping, adjacency, {
        "quotient_regions": quotient_regions,
        "network_edges": len(mapping),
        "max_start_node_alignment_degrees": float(np.max(distance)),
        "p95_start_node_alignment_degrees": float(np.quantile(distance, 0.95)),
        "quotient_diagnostics": diagnostics,
        "osm": {"path": str(osm_path), "sha256": sha256_file(osm_path)},
    }


def _practical_margin(metric: str, mtr: float, strongest: float) -> dict:
    higher = metric in {"mmsr", "road_next_accuracy"}
    signed = mtr - strongest if higher else strongest - mtr
    scale = max(abs(strongest), 1e-12)
    relative = signed / scale
    threshold = 0.005 if metric in {"road_use_jsd", "mmsr", "road_next_accuracy"} else 0.01 * scale
    return {
        "signed_advantage": float(signed),
        "relative_advantage": float(relative),
        "required_advantage": float(threshold),
        "passes": bool(signed >= threshold),
    }


def _paired_bootstrap(
    mtr: np.ndarray,
    baseline: np.ndarray,
    *,
    higher_is_better: bool,
    draws: int,
    seed: int,
) -> dict[str, float]:
    if len(mtr) != len(baseline) or len(mtr) == 0:
        raise ValueError("paired route-score arrays must have the same positive length")
    improvement = mtr - baseline if higher_is_better else baseline - mtr
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for index in range(draws):
        sample = rng.integers(0, len(improvement), len(improvement))
        means[index] = float(np.mean(improvement[sample]))
    low, high = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "improvement": float(np.mean(improvement)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_one_sided": float((1 + np.count_nonzero(means <= 0.0)) / (draws + 1)),
    }


def _holm_adjust(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_one_sided"])
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * rows[index]["p_one_sided"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = float(running)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--match-dir", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--method", action="append", required=True, help="Repeat NAME=CACHE_STEM")
    parser.add_argument("--real-test-fraction", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--quotient-regions", type=int, choices=(24, 96, 384), default=384)
    parser.add_argument("--minimum-path-edges", type=int, default=1)
    parser.add_argument("--expected-count", type=int, help="Override the full public slot count for a declared screening prefix")
    parser.add_argument(
        "--volume-calibrate-link-flow",
        action="store_true",
        help="Compare link-flow shares after one corpus-level total-volume calibration.",
    )
    parser.add_argument(
        "--topology-smoothing",
        action="store_true",
        help="Normalize additive smoothing over public feasible successor regions instead of all quotient regions.",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if not 0 < args.real_test_fraction < 1:
        raise ValueError("--real-test-fraction must lie strictly between zero and one")
    if args.bootstrap_draws <= 0:
        raise ValueError("--bootstrap-draws must be positive")
    started = time.time()
    config = dataset_config(args.dataset_config)
    match_dir = public_path(args.match_dir)
    network = public_path(args.network)
    out_dir = public_path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    methods: dict[str, str] = {}
    for spec in args.method:
        if "=" not in spec:
            raise ValueError(f"expected NAME=CACHE_STEM: {spec}")
        name, stem = spec.split("=", 1)
        if not name or name in methods:
            raise ValueError(f"method names must be nonempty and unique: {name!r}")
        methods[name] = stem
    real_path = match_dir / "Real.pkl.gz"
    real_records = _load_records(real_path)
    synthetic_records = {
        name: _load_records(match_dir / f"{stem}.pkl.gz") for name, stem in methods.items()
    }
    expected = args.expected_count if args.expected_count is not None else int(config["public_slot_count"])
    if expected <= 0:
        raise ValueError("--expected-count must be positive")
    if len(real_records) != expected:
        raise RuntimeError(f"Real cache contains {len(real_records)} records; expected {expected}")
    for name, records in synthetic_records.items():
        if len(records) != expected:
            raise RuntimeError(f"{name} cache contains {len(records)} records; expected {expected}")

    real_records = require_minimum_path_edges(real_records, args.minimum_path_edges)
    synthetic_records = {
        name: require_minimum_path_edges(records, args.minimum_path_edges)
        for name, records in synthetic_records.items()
    }
    edge_to_region, quotient_adjacency, quotient = _edge_to_region(network, config, args.quotient_regions)
    real_regions = region_sequences(real_records, edge_to_region)
    split = int(len(real_regions) * (1.0 - args.real_test_fraction))
    real_test = real_regions[split:]
    results: dict[str, dict[str, float | int]] = {
        "Real": {"mmsr": map_matching_success_rate(real_records)}
    }
    route_scores: dict[str, dict[str, np.ndarray | int]] = {}
    for name, records in synthetic_records.items():
        scores = road_next_trip_scores(
            region_sequences(records, edge_to_region),
            real_test,
            state_count=int(quotient["quotient_regions"]),
            alpha=args.alpha,
            feasible_successors=quotient_adjacency if args.topology_smoothing else None,
        )
        route_scores[name] = scores
        route_metrics = {
            "road_next_accuracy": float(np.mean(scores["road_next_accuracy"])),
            "road_next_nll": float(np.mean(scores["road_next_nll"])),
            "road_next_training_trips": int(scores["road_next_training_trips"]),
            "road_next_test_trips": int(scores["road_next_test_trips"]),
            "road_next_test_decisions": int(scores["road_next_test_decisions"]),
        }
        raw_link_flow_wape = link_flow_wape(real_records, records)
        scale_adjusted_link_flow_wape = link_flow_wape(
            real_records, records, calibrate_total_volume=True
        )
        results[name] = {
            "road_use_jsd": road_use_jsd(real_records, records),
            "mmsr": map_matching_success_rate(records),
            **route_metrics,
            # Preserve both values in the audit ledger.  The selected score is
            # explicit: raw WAPE evaluates total traversal volume and edge
            # allocation jointly, whereas the scale-adjusted score evaluates
            # allocation after Len has measured total trip magnitude.
            "link_flow_wape_raw": raw_link_flow_wape,
            "link_flow_wape_scale_adjusted": scale_adjusted_link_flow_wape,
            "link_flow_wape": (
                scale_adjusted_link_flow_wape
                if args.volume_calibrate_link_flow else raw_link_flow_wape
            ),
        }
    metric_directions = {
        "road_use_jsd": "min",
        "mmsr": "max",
        "road_next_accuracy": "max",
        "road_next_nll": "min",
        "link_flow_wape": "min",
    }
    admission = {"real_mmsr_at_least_0_90": bool(results["Real"]["mmsr"] >= 0.90), "metrics": {}}
    for metric, direction in metric_directions.items():
        candidates = {name: float(results[name][metric]) for name in methods}
        ordered = sorted(candidates, key=candidates.get, reverse=(direction == "max"))
        strongest_baseline = ordered[0] if ordered[0] != "MTR-GSRT" else ordered[1]
        mtr_rank = ordered.index("MTR-GSRT") + 1 if "MTR-GSRT" in ordered else None
        margin = _practical_margin(metric, candidates["MTR-GSRT"], candidates[strongest_baseline])
        admission["metrics"][metric] = {
            "direction": direction,
            "ranking": ordered,
            "mtr_rank": mtr_rank,
            "strongest_baseline": strongest_baseline,
            **margin,
            "passes": bool(mtr_rank == 1 and margin["passes"]),
        }
    admission["admitted"] = bool(
        admission["real_mmsr_at_least_0_90"]
        and all(row["passes"] for row in admission["metrics"].values())
    )
    raw_candidates = {
        name: float(results[name]["link_flow_wape_raw"])
        for name in methods
    }
    raw_ranking = sorted(raw_candidates, key=raw_candidates.get)
    raw_strongest = raw_ranking[0] if raw_ranking[0] != "MTR-GSRT" else raw_ranking[1]
    raw_margin = _practical_margin(
        "link_flow_wape",
        raw_candidates["MTR-GSRT"],
        raw_candidates[raw_strongest],
    )
    admission["selected_link_flow_score"] = (
        "scale_adjusted" if args.volume_calibrate_link_flow else "raw"
    )
    admission["original_raw_link_flow"] = {
        "direction": "min",
        "ranking": raw_ranking,
        "mtr_rank": raw_ranking.index("MTR-GSRT") + 1,
        "strongest_baseline": raw_strongest,
        **raw_margin,
        "passes": bool(raw_ranking[0] == "MTR-GSRT" and raw_margin["passes"]),
    }
    admission["admitted_under_original_raw_wape"] = bool(
        admission["real_mmsr_at_least_0_90"]
        and all(
            row["passes"]
            for metric, row in admission["metrics"].items()
            if metric != "link_flow_wape"
        )
        and admission["original_raw_link_flow"]["passes"]
    )
    admission["admitted_under_selected_flow_score"] = admission["admitted"]
    if "MTR-GSRT" not in route_scores:
        raise ValueError("paired comparisons require an MTR-GSRT method")
    paired_tests: list[dict] = []
    baselines = [name for name in methods if name != "MTR-GSRT"]
    for baseline_index, baseline in enumerate(baselines):
        for metric_index, (metric, higher) in enumerate((
            ("road_next_accuracy", True),
            ("road_next_nll", False),
        )):
            paired_tests.append({
                "baseline": baseline,
                "metric": metric,
                **_paired_bootstrap(
                    np.asarray(route_scores["MTR-GSRT"][metric], dtype=float),
                    np.asarray(route_scores[baseline][metric], dtype=float),
                    higher_is_better=higher,
                    draws=args.bootstrap_draws,
                    seed=args.bootstrap_seed + baseline_index * 101 + metric_index,
                ),
            })
    _holm_adjust(paired_tests)
    payload = {
        "schema_version": 1,
        "classification": "CANDIDATE_METRIC_ADMISSION_TEST_NO_SYNTHESIS",
        "dataset": config["name"],
        "record_count": expected,
        "screening_prefix": bool(args.expected_count is not None),
        "parameters": {
            "real_test_start": split,
            "real_test_fraction": args.real_test_fraction,
            "alpha": args.alpha,
            "smoothing_support": "public quotient successors" if args.topology_smoothing else "all quotient regions",
            "minimum_path_edges": args.minimum_path_edges,
            "link_flow_volume_calibration": args.volume_calibrate_link_flow,
            "consecutive_duplicate_regions_removed": True,
            "failed_match_policy": "MMSR failure; unmatched Road-Use bin; no transitions or link flow",
            "paired_bootstrap_draws": args.bootstrap_draws,
            "paired_bootstrap_seed": args.bootstrap_seed,
            "paired_bootstrap_unit": "one fixed Real test trip after within-trip decision averaging",
            "multiple_testing": "Holm correction over eight within-city Road NextAcc/NextNLL comparisons",
        },
        "results": results,
        "paired_tests": paired_tests,
        "admission": admission,
    }
    out_dir.mkdir(parents=True)
    write_json(out_dir / "metrics.json", payload)
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "method", "road_use_jsd", "mmsr", "road_next_accuracy", "road_next_nll",
            "link_flow_wape", "link_flow_wape_raw", "link_flow_wape_scale_adjusted",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, values in results.items():
            writer.writerow({"method": name, **{field: values.get(field) for field in fields[1:]}})
    with (out_dir / "paired_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["baseline", "metric", "improvement", "ci95_low", "ci95_high", "p_one_sided", "p_holm"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired_tests)
    manifest = {
        "schema_version": 1,
        "classification": "RESEARCH_EVALUATION_LEDGER_NOT_A_DP_RELEASE",
        "no_synthesis": True,
        "inputs": {
            "network": {"path": str(network), "sha256": sha256_file(network)},
            "match_cache": {
                path.name: sha256_file(path)
                for path in [real_path, *[match_dir / f"{stem}.pkl.gz" for stem in methods.values()]]
            },
        },
        "quotient": quotient,
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "outputs": {
            "metrics.json": sha256_file(out_dir / "metrics.json"),
            "metrics.csv": sha256_file(out_dir / "metrics.csv"),
            "paired_tests.csv": sha256_file(out_dir / "paired_tests.csv"),
        },
        "elapsed_sec": time.time() - started,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({
        "admitted_under_selected_flow_score": admission["admitted_under_selected_flow_score"],
        "admitted_under_original_raw_wape": admission["admitted_under_original_raw_wape"],
        "results": results,
        "out_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
