"""Evaluate road-choice metrics from existing common STMatch caches.

This command performs no trajectory synthesis and invokes no map matcher.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
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
from metric_suites.road_choice import (  # noqa: E402
    matched_choice_trips,
    next_road_trip_scores,
    public_crossing_options,
    road_choice_fidelity,
)


def _load_records(path: Path) -> tuple[list[dict], int]:
    """Load a cache and validate either local or legacy contiguous indices."""
    with gzip.open(path, "rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"match cache is not a record list: {path}")
    required = {"source_index", "cpath", "connected", "accepted"}
    offset = int(records[0]["source_index"]) if records else 0
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError(f"invalid match record {index} in {path}")
        if int(record["source_index"]) != offset + index:
            raise ValueError(
                f"noncontiguous source index {record['source_index']} at {index} in {path}"
            )
        if bool(record["accepted"]) and not bool(record["connected"]):
            raise ValueError(f"accepted disconnected path at {index} in {path}")
    return records, offset


def _edge_endpoint_regions(
    network: Path,
    config: dict,
    quotient_regions: int,
) -> tuple[
    dict[int, tuple[int, int]],
    dict[int, tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    dict,
]:
    add_runtime_paths()
    import geopandas as gpd
    from scipy.spatial import cKDTree
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
    coordinates = np.asarray(coordinates, dtype=float)
    context, diagnostics, _ = nested.build_nested_context(
        coordinates, graph, coarse_regions=24, fine_regions=quotient_regions
    )
    node_regions = context.node_fine
    carrier_edge_regions = {
        (int(source), int(target)): (
            int(node_regions[int(source)]),
            int(node_regions[int(target)]),
        )
        for source, neighbors in graph.items()
        for target, _weight in neighbors
    }
    frame = gpd.read_file(network)
    required = {"id", "geometry"}
    if not required.issubset(frame.columns):
        raise ValueError(f"road network lacks columns {sorted(required - set(frame.columns))}")
    if frame.crs is None:
        raise ValueError("road network has no CRS")
    geographic = frame.to_crs("EPSG:4326")
    starts = np.asarray([
        [float(geometry.coords[0][1]), float(geometry.coords[0][0])]
        for geometry in geographic.geometry
    ])
    ends = np.asarray([
        [float(geometry.coords[-1][1]), float(geometry.coords[-1][0])]
        for geometry in geographic.geometry
    ])
    tree = cKDTree(coordinates)
    start_distance, start_node = tree.query(starts, k=1)
    end_distance, end_node = tree.query(ends, k=1)
    mapping = {
        int(edge_id): (
            int(node_regions[int(source_node)]),
            int(node_regions[int(target_node)]),
        )
        for edge_id, source_node, target_node in zip(
            frame["id"].to_numpy(), start_node, end_node
        )
    }
    if len(mapping) != len(frame):
        raise ValueError("road network edge IDs are not unique")
    endpoint_nodes = {
        int(edge_id): (int(source_node), int(target_node))
        for edge_id, source_node, target_node in zip(
            frame["id"].to_numpy(), start_node, end_node
        )
    }
    return mapping, endpoint_nodes, carrier_edge_regions, {
        "quotient_regions": int(quotient_regions),
        "network_edges": int(len(mapping)),
        "carrier_directed_edges": int(len(carrier_edge_regions)),
        "network_endpoint_pair_coverage": float(
            len(set(endpoint_nodes.values()) & set(carrier_edge_regions))
            / max(len(set(endpoint_nodes.values())), 1)
        ),
        "endpoint_pair_ambiguities": int(
            len(endpoint_nodes) - len(set(endpoint_nodes.values()))
        ),
        "max_endpoint_alignment_degrees": float(
            max(np.max(start_distance), np.max(end_distance))
        ),
        "p95_endpoint_alignment_degrees": float(
            max(np.quantile(start_distance, 0.95), np.quantile(end_distance, 0.95))
        ),
        "quotient_diagnostics": diagnostics,
        "osm": {"path": str(osm_path), "sha256": sha256_file(osm_path)},
    }


def _load_witness_trips(
    path: Path,
    *,
    expected: int,
    option_context: dict[tuple[int, int], object],
) -> tuple[list[list[tuple[object, tuple[int, int]]]], dict]:
    with path.open("rb") as handle:
        witnesses = pickle.load(handle)
    if not isinstance(witnesses, list) or len(witnesses) != expected:
        raise ValueError(f"witness file must contain {expected} records: {path}")
    trips: list[list[tuple[object, int]]] = []
    total_edges = 0
    mapped_edges = 0
    for index, witness in enumerate(witnesses):
        if not isinstance(witness, dict) or int(witness.get("slot", -1)) != index:
            raise ValueError(f"invalid witness slot {index} in {path}")
        events: list[tuple[object, int]] = []
        for raw_edge in witness.get("directed_edges", ()):
            total_edges += 1
            pair = tuple(map(int, raw_edge))
            context = option_context.get(pair)
            if context is not None:
                events.append((context, pair))
                mapped_edges += 1
        trips.append(events)
    return trips, {
        "path": str(path),
        "sha256": sha256_file(path),
        "records": len(witnesses),
        "directed_edges": total_edges,
        "eligible_choice_edges": mapped_edges,
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
        raise ValueError("paired score arrays must have the same positive length")
    improvement = mtr - baseline if higher_is_better else baseline - mtr
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for index in range(draws):
        sample = rng.integers(0, len(improvement), len(improvement))
        means[index] = float(np.mean(improvement[sample]))
    low, high = np.quantile(means, (0.025, 0.975))
    return {
        "improvement": float(np.mean(improvement)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_one_sided": float((1 + np.count_nonzero(means <= 0.0)) / (draws + 1)),
    }


def _holm_adjust(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_one_sided"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["p_one_sided"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = float(running)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--match-dir", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--method", action="append", default=[], help="Repeat NAME=CACHE_STEM")
    parser.add_argument("--witness", action="append", default=[], help="Repeat NAME=WITNESS_PATH")
    parser.add_argument("--real-test-fraction", type=float, default=0.2)
    parser.add_argument("--quotient-regions", type=int, choices=(24, 96, 384), default=384)
    parser.add_argument("--mixture-weight", type=float, default=0.5)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if not 0.0 < args.real_test_fraction < 1.0:
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
    for specification in args.method:
        if "=" not in specification:
            raise ValueError(f"expected NAME=CACHE_STEM: {specification}")
        name, stem = specification.split("=", 1)
        if not name or name in methods:
            raise ValueError(f"method names must be nonempty and unique: {name!r}")
        methods[name] = stem
    witness_methods: dict[str, Path] = {}
    for specification in args.witness:
        if "=" not in specification:
            raise ValueError(f"expected NAME=WITNESS_PATH: {specification}")
        name, path = specification.split("=", 1)
        if not name or name in methods or name in witness_methods:
            raise ValueError(f"method names must be nonempty and unique: {name!r}")
        witness_methods[name] = public_path(path)
    if not methods and not witness_methods:
        raise ValueError("at least one --method or --witness is required")

    real_path = match_dir / "Real.pkl.gz"
    real_records, real_offset = _load_records(real_path)
    loaded = {
        name: _load_records(match_dir / f"{stem}.pkl.gz")
        for name, stem in methods.items()
    }
    synthetic_records = {name: value[0] for name, value in loaded.items()}
    source_index_offsets = {
        "Real": real_offset,
        **{name: value[1] for name, value in loaded.items()},
    }
    expected = args.expected_count if args.expected_count is not None else int(
        config["public_slot_count"]
    )
    for name, records in {"Real": real_records, **synthetic_records}.items():
        if len(records) != expected:
            raise RuntimeError(f"{name} cache contains {len(records)} records; expected {expected}")

    edge_regions, edge_nodes, carrier_edge_regions, quotient = _edge_endpoint_regions(
        network, config, args.quotient_regions
    )
    carrier_ids = {index: pair for index, pair in enumerate(carrier_edge_regions)}
    carrier_regions = {
        index: carrier_edge_regions[pair] for index, pair in carrier_ids.items()
    }
    public_options, _ = public_crossing_options(carrier_regions, carrier_ids)
    option_context = {
        option: context
        for context, options in public_options.items()
        for option in options
    }
    edge_context = {
        edge_id: option_context[pair]
        for edge_id, pair in edge_nodes.items()
        if pair in option_context
    }
    real_trips = matched_choice_trips(real_records, edge_context, edge_nodes)
    split = int(len(real_trips) * (1.0 - args.real_test_fraction))
    real_test = real_trips[split:]

    results: dict[str, dict[str, float | int]] = {}
    trip_scores: dict[str, dict[str, np.ndarray | int]] = {}
    all_trips = {
        name: matched_choice_trips(records, edge_context, edge_nodes)
        for name, records in synthetic_records.items()
    }
    witness_inputs: dict[str, dict] = {}
    for name, path in witness_methods.items():
        all_trips[name], witness_inputs[name] = _load_witness_trips(
            path,
            expected=expected,
            option_context=option_context,
        )
    for name, trips in all_trips.items():
        fidelity = road_choice_fidelity(trips, real_test, public_options)
        scores = next_road_trip_scores(
            trips, real_test, public_options, mixture_weight=args.mixture_weight
        )
        trip_scores[name] = scores
        results[name] = {
            **fidelity,
            "next_road_accuracy": float(np.mean(scores["next_road_accuracy"])),
            "next_road_nll": float(np.mean(scores["next_road_nll"])),
            "next_road_training_trips": int(scores["next_road_training_trips"]),
            "next_road_training_events": int(scores["next_road_training_events"]),
            "next_road_test_trips": int(scores["next_road_test_trips"]),
            "next_road_test_events": int(scores["next_road_test_events"]),
        }

    paired_tests: list[dict] = []
    if "MTR-GSRT" in trip_scores:
        for baseline_index, baseline in enumerate(
            name for name in trip_scores if name != "MTR-GSRT"
        ):
            for metric_index, (metric, higher) in enumerate((
                ("next_road_accuracy", True),
                ("next_road_nll", False),
            )):
                paired_tests.append({
                    "baseline": baseline,
                    "metric": metric,
                    **_paired_bootstrap(
                        np.asarray(trip_scores["MTR-GSRT"][metric], dtype=float),
                        np.asarray(trip_scores[baseline][metric], dtype=float),
                        higher_is_better=higher,
                        draws=args.bootstrap_draws,
                        seed=args.bootstrap_seed + baseline_index * 101 + metric_index,
                    ),
                })
    _holm_adjust(paired_tests)
    payload = {
        "schema_version": 1,
        "classification": "ROAD_CHOICE_EVALUATION_NO_SYNTHESIS",
        "dataset": config["name"],
        "record_count": expected,
        "screening_prefix": bool(args.expected_count is not None),
        "definition": {
            "context": "ordered pair of distinct public quotient regions",
            "option": "directed public FMM road edge crossing the context",
            "context_eligibility": "at least two public options and positive frozen-real-test mass",
            "context_selection_uses_method_output": False,
            "population_weighting": "unit mass per contributing trajectory",
            "real_test_start": split,
            "real_test_fraction": args.real_test_fraction,
            "next_road_mixture_weight": args.mixture_weight,
            "failed_match_policy": "zero choice events",
            "source_index_offsets": source_index_offsets,
        },
        "public_support": {
            "choice_contexts": len(public_options),
            "choice_edges": len(edge_context),
        },
        "witness_inputs": witness_inputs,
        "results": results,
        "paired_tests": paired_tests,
    }
    out_dir.mkdir(parents=True)
    write_json(out_dir / "metrics.json", payload)
    fields = [
        "method", "road_choice_cpc", "road_choice_ndcg",
        "next_road_accuracy", "next_road_nll",
        "road_choice_contexts", "road_choice_real_test_trips",
        "road_choice_real_test_events", "road_choice_synthetic_trips",
        "road_choice_synthetic_events",
    ]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, values in results.items():
            writer.writerow({"method": name, **{field: values.get(field) for field in fields[1:]}})
    with (out_dir / "paired_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        test_fields = [
            "baseline", "metric", "improvement", "ci95_low", "ci95_high",
            "p_one_sided", "p_holm",
        ]
        writer = csv.DictWriter(handle, fieldnames=test_fields)
        writer.writeheader()
        writer.writerows(paired_tests)
    module_path = Path(sys.modules["metric_suites.road_choice"].__file__).resolve()
    manifest = {
        "schema_version": 1,
        "classification": "RESEARCH_EVALUATION_LEDGER_NOT_A_DP_RELEASE",
        "no_synthesis": True,
        "inputs": {
            "network": {"path": str(network), "sha256": sha256_file(network)},
            "match_cache": {
                path.name: sha256_file(path)
                for path in [
                    real_path,
                    *[match_dir / f"{stem}.pkl.gz" for stem in methods.values()],
                ]
            },
            "witness": witness_inputs,
        },
        "quotient": quotient,
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "metric_core": {"path": str(module_path), "sha256": sha256_file(module_path)},
        },
        "outputs": {
            "metrics.json": sha256_file(out_dir / "metrics.json"),
            "metrics.csv": sha256_file(out_dir / "metrics.csv"),
            "paired_tests.csv": sha256_file(out_dir / "paired_tests.csv"),
        },
        "elapsed_sec": time.time() - started,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"results": results, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
