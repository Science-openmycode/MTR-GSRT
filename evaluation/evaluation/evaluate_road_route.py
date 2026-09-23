"""Evaluate Road Constraints and route-choice preservation on public road IDs.

Example (PowerShell):

    python evaluation/evaluate_road_route.py `
      --dataset-config geolife --real geolife `
      --synthetic "SPRT=outputs/synthetic_releases/current_best/sprt_native.pkl" `
      --synthetic "MTR-GSRT=outputs/.../dp_gsrt_portal_qrsp.pkl" `
      --witness "MTR-GSRT=outputs/.../road_witness_release_v1.json" `
      --fmm-network C:/fmm/network.shp --ubodt C:/fmm/ubodt.txt `
      --fmm-runtime-dir C:/path/to/fmm/runtime `
      --out-dir results/road_route_metrics/beijing
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
        if (
            (_candidate / "configs" / "datasets.json").is_file()
            and (_candidate / "generation" / "common" / "runtime.py").is_file()
        ):
            sys.path.insert(0, str(_candidate))
            break
    else:
        raise RuntimeError("cannot locate public_release root")

from generation.common.runtime import (  # noqa: E402
    PUBLIC_RELEASE,
    add_runtime_paths,
    dataset_config,
    public_path,
    sha256_file,
    write_json,
)
from metric_suites.road_route import (  # noqa: E402
    MatchRecord,
    load_witness_records,
    od_conditioned_transition_jsd,
    road_realizability_metrics,
    route_recommendation_metrics,
    run_fmm,
    witness_coordinate_metrics,
    witness_node_trajectories,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument(
        "--synthetic",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat once per evaluated synthesis method.",
    )
    parser.add_argument(
        "--witness",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Optional witness release; NAME must match one --synthetic entry.",
    )
    parser.add_argument("--fmm-network", type=Path, required=True)
    parser.add_argument("--matcher", choices=("stmatch", "fmm"), default="stmatch")
    parser.add_argument("--ubodt", type=Path)
    parser.add_argument("--fmm-bin", type=Path)
    parser.add_argument("--stmatch-bin", type=Path)
    parser.add_argument("--ubodt-bin", type=Path)
    parser.add_argument("--fmm-runtime-dir", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--radius-m", type=float, default=200.0)
    parser.add_argument("--gps-error-m", type=float, default=50.0)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--min-observation-share", type=float, default=0.95)
    parser.add_argument("--witness-distance-m", type=float, default=200.0)
    parser.add_argument("--witness-monotone-share", type=float, default=0.95)
    parser.add_argument("--od-grid", type=int, default=8)
    parser.add_argument("--max-prototypes", type=int, default=20)
    parser.add_argument("--route-hit-threshold", type=float, default=0.35)
    parser.add_argument("--route-only", action="store_true")
    parser.add_argument("--no-save-matched-paths", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Public deterministic prefix for smoke tests; omit for the registered full corpus.",
    )
    parser.add_argument("--out-dir", required=True)
    return parser


def _named_paths(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, received: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in output:
            raise ValueError(f"method names must be nonempty and unique: {name!r}")
        path = public_path(raw_path.strip())
        if not path.is_file():
            raise FileNotFoundError(f"input for {name} does not exist: {path}")
        output[name] = path
    return output


def _resolve_real(config: dict, spec: str):
    registered_names = {
        config["name"],
        *[str(value).lower() for value in config.get("aliases", [])],
    }
    if spec.lower() in registered_names and config.get("data"):
        registered = config["data"]
        candidate = public_path(registered)
        return candidate if candidate.is_file() else registered
    candidate = public_path(spec)
    return candidate if candidate.is_file() else spec


def _default_binary(name: str) -> Path:
    return (
        PUBLIC_RELEASE
        / "third_party"
        / "fmm-v0.1.1"
        / "cyang-kth-fmm-344fb8c"
        / "build"
        / "Release"
        / name
    )


def _production_graph(osm, bbox):
    final_dir = PUBLIC_RELEASE / "src" / "mtr" / "DP_GSRT" / "final"
    if str(final_dir) not in sys.path:
        sys.path.insert(0, str(final_dir))
    import route_structure_potential_experiment as route

    coordinates, graph = route.prepare_graph(
        [],
        bbox=bbox,
        osm_ways=osm,
        raw_graph=False,
    )
    return np.asarray(coordinates, dtype=float), graph


def _serializable_record(record: MatchRecord) -> dict:
    return {
        "source_index": record.source_index,
        "cpath": record.cpath,
        "opath": record.opath,
        "residual_m": record.residual_m,
        "connected": record.connected,
        "observation_share": record.observation_share,
        "accepted": record.accepted,
    }


def _save_records(path: Path, records: list[MatchRecord]) -> None:
    with gzip.open(path, "wb", compresslevel=6) as handle:
        pickle.dump([_serializable_record(record) for record in records], handle, pickle.HIGHEST_PROTOCOL)


def _finite_metrics(metrics: dict[str, float | None]) -> None:
    for name, value in metrics.items():
        if value is not None and not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite metric: {name}={value}")


def main() -> None:
    args = build_parser().parse_args()
    add_runtime_paths()
    from public_utils import (
        filter_osm_ways_by_bbox,
        filter_osm_ways_by_highway,
        load_osm_ways,
        load_trajectories,
    )

    started = time.time()
    config = dataset_config(args.dataset_config)
    if config.get("bbox") is None or config.get("osm_cache") is None:
        raise ValueError("dataset configuration must declare bbox and osm_cache")
    bbox = tuple(map(float, config["bbox"]))
    osm_path = public_path(config["osm_cache"])
    if not osm_path.is_file():
        raise FileNotFoundError(f"OSM cache does not exist: {osm_path}")
    expected_count = int(config["public_slot_count"])
    count = min(expected_count, args.limit) if args.limit is not None else expected_count
    real_spec = _resolve_real(config, args.real)
    real = load_trajectories(str(real_spec), limit=count)
    if len(real) != count:
        raise RuntimeError(f"real input has {len(real)} records; expected {count}")
    synthetic_paths = _named_paths(args.synthetic)
    witness_paths = _named_paths(args.witness)
    unknown_witnesses = set(witness_paths).difference(synthetic_paths)
    if unknown_witnesses:
        raise ValueError(f"witness names have no matching synthetic input: {sorted(unknown_witnesses)}")
    synthetic = {
        name: load_trajectories(str(path), limit=count if args.limit is not None else None)
        for name, path in synthetic_paths.items()
    }
    for name, trajectories in synthetic.items():
        if len(trajectories) != count:
            raise RuntimeError(f"{name} has {len(trajectories)} records; expected {count}")

    out_dir = public_path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    match_dir = out_dir / "matched_paths"
    match_dir.mkdir()

    matcher_binary = (
        args.stmatch_bin or _default_binary("stmatch.exe")
        if args.matcher == "stmatch"
        else args.fmm_bin or _default_binary("fmm.exe")
    ).resolve()
    ubodt_binary = (
        (args.ubodt_bin or _default_binary("ubodt_gen.exe")).resolve()
        if args.matcher == "fmm"
        else None
    )
    network = args.fmm_network.resolve()
    ubodt = args.ubodt.resolve() if args.ubodt else None
    if args.matcher == "fmm" and ubodt is None:
        raise ValueError("--ubodt is required with --matcher fmm")
    required_paths = [matcher_binary, network]
    if ubodt_binary is not None:
        required_paths.append(ubodt_binary)
    if ubodt is not None:
        required_paths.append(ubodt)
    for required in required_paths:
        if not required.is_file():
            raise FileNotFoundError(required)

    osm = filter_osm_ways_by_bbox(load_osm_ways(str(osm_path)), bbox)
    osm = filter_osm_ways_by_highway(osm, list(config.get("osm_highway_classes", [])))
    if not osm:
        raise RuntimeError("configured public OSM selection is empty")
    public_coordinates, _ = _production_graph(osm, bbox)

    route_input_trajectories: dict[str, list[np.ndarray]] = {}
    witness_metrics: dict[str, dict[str, float]] = {}
    for name, trajectories in synthetic.items():
        if name in witness_paths:
            witnesses = load_witness_records(witness_paths[name])
            if args.limit is not None:
                witnesses = witnesses[:count]
            witness_metrics[name] = witness_coordinate_metrics(
                trajectories,
                witnesses,
                public_coordinates,
                args.max_points,
                args.witness_distance_m,
                args.witness_monotone_share,
            )
            witness_trajectories = witness_node_trajectories(
                witnesses,
                public_coordinates,
                args.max_points,
            )
            route_input_trajectories[name + ":witness-route"] = witness_trajectories

    # FMM spends substantial fixed time loading the public UBODT.  Match every
    # corpus in one invocation, then split by frozen offsets.  This changes no
    # record and avoids loading the ~GB public table once per method.
    corpora: dict[str, list[np.ndarray]] = {"Real": real, **synthetic, **route_input_trajectories}
    offsets: dict[str, tuple[int, int]] = {}
    combined: list[np.ndarray] = []
    for name, trajectories in corpora.items():
        start = len(combined)
        combined.extend(trajectories)
        offsets[name] = (start, len(combined))
    combined_records, invocation = run_fmm(
        combined,
        network,
        ubodt,
        matcher_binary,
        ubodt_binary,
        args.fmm_runtime_dir.resolve() if args.fmm_runtime_dir else None,
        args.max_points,
        args.radius_m,
        args.gps_error_m,
        args.candidates,
        args.min_observation_share,
        args.work_root.resolve() if args.work_root else None,
        args.route_only,
    )
    invocation["corpus_offsets"] = {
        name: {"start": start, "stop": stop}
        for name, (start, stop) in offsets.items()
    }
    invocations: dict[str, dict] = {"combined": invocation}
    split_records = {
        name: combined_records[start:stop]
        for name, (start, stop) in offsets.items()
    }
    real_records = split_records["Real"]
    if not args.no_save_matched_paths:
        _save_records(match_dir / "Real.pkl.gz", real_records)
    coordinate_records: dict[str, list[MatchRecord]] = {}
    route_records: dict[str, list[MatchRecord]] = {}
    for name in synthetic:
        coordinate_records[name] = split_records[name]
        route_key = name + ":witness-route"
        route_records[name] = split_records.get(route_key, split_records[name])
        if not args.no_save_matched_paths:
            _save_records(match_dir / f"{name}.pkl.gz", coordinate_records[name])
            if route_key in split_records:
                _save_records(match_dir / f"{name}.witness.pkl.gz", route_records[name])

    split = int(len(real) * 0.8)
    results: dict[str, dict[str, float | None]] = {
        "Real": ({} if args.route_only else road_realizability_metrics(real_records))
    }
    for name, trajectories in synthetic.items():
        method_metrics: dict[str, float | None] = {}
        if not args.route_only:
            method_metrics.update(road_realizability_metrics(coordinate_records[name]))
        method_metrics.update(witness_metrics.get(name, {
            "witness_coordinate_consistent": None,
            "witness_coordinate_all_within": None,
            "witness_coordinate_p95_median_m": None,
            "witness_coordinate_p95_p95_m": None,
            "witness_coordinate_max_p95_m": None,
            "witness_coordinate_monotone_mean": None,
        }))
        method_metrics["od_conditioned_road_transition_jsd"] = od_conditioned_transition_jsd(
            real,
            real_records,
            trajectories,
            route_records[name],
            bbox,
            args.od_grid,
        )
        method_metrics.update(
            route_recommendation_metrics(
                trajectories,
                route_records[name],
                real[split:],
                real_records[split:],
                bbox,
                args.od_grid,
                args.max_prototypes,
                args.route_hit_threshold,
            )
        )
        _finite_metrics(method_metrics)
        results[name] = method_metrics

    metrics_payload = {
        "schema_version": 1,
        "metric_semantics": {
            "road_realizable": "fraction of all released records accepted as one connected directed FMM path; failures remain zero",
            "witness_coordinate_consistent": "fraction whose p95 and endpoint residuals meet the public distance threshold and whose ordered projection meets the monotonicity threshold",
            "od_conditioned_road_transition_jsd": "real-OD-weighted JSD over actual FMM road-edge transitions, with failed matches assigned to an INVALID outcome",
            "road_route_*": "frequency-ranked route recommendation over actual directed FMM edge IDs",
        },
        "parameters": {
            "max_points": args.max_points,
            "radius_m": args.radius_m,
            "gps_error_m": args.gps_error_m,
            "candidates": args.candidates,
            "min_observation_share": args.min_observation_share,
            "witness_distance_m": args.witness_distance_m,
            "witness_monotone_share": args.witness_monotone_share,
            "od_grid": args.od_grid,
            "max_prototypes": args.max_prototypes,
            "route_hit_threshold": args.route_hit_threshold,
            "real_eval_start": split,
            "route_only": args.route_only,
        },
        "results": results,
    }
    write_json(out_dir / "metrics.json", metrics_payload)
    metric_names = sorted({metric for values in results.values() for metric in values})
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *metric_names])
        writer.writeheader()
        for name, values in results.items():
            writer.writerow({"method": name, **values})
    manifest = {
        "schema_version": 1,
        "classification": "RESEARCH_EVALUATION_LEDGER_NOT_A_DP_RELEASE",
        "dataset": config["name"],
        "record_count": count,
        "inputs": {
            "real": str(real_spec),
            "synthetic": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in synthetic_paths.items()
            },
            "witness": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in witness_paths.items()
            },
            "osm": {"path": str(osm_path), "sha256": sha256_file(osm_path)},
            "fmm_network": {
                "path": str(network),
                "sha256": sha256_file(network),
            },
            "matcher": args.matcher,
            "ubodt": (
                {"path": str(ubodt), "sha256": sha256_file(ubodt)}
                if ubodt is not None else None
            ),
        },
        "evaluator": {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "metric_module_sha256": sha256_file(Path(__file__).resolve().parent / "metric_suites" / "road_route.py"),
        },
        "invocations": invocations,
        "outputs": {
            "metrics.json": sha256_file(out_dir / "metrics.json"),
            "metrics.csv": sha256_file(out_dir / "metrics.csv"),
            "matched_paths": {
                path.name: sha256_file(path)
                for path in sorted(match_dir.glob("*.pkl.gz"))
            },
        },
        "elapsed_sec": time.time() - started,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
