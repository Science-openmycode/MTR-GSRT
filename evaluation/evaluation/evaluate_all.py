from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if ((_candidate / "configs" / "datasets.json").is_file()
                and (_candidate / "generation" / "common" / "runtime.py").is_file()):
            sys.path.insert(0, str(_candidate))
            break
    else:
        raise RuntimeError("cannot locate public_release root")

from generation.common.runtime import (  # noqa: E402
    PUBLIC_RELEASE,
    add_runtime_paths,
    configure_logging,
    dataset_config,
    jsonable,
    public_path,
    require_new_directory,
    sha256_file,
    write_json,
)
from public_utils import thin_osm_ways_public  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate all manuscript utility and WitnessValid metrics for one release."
    )
    parser.add_argument("--real", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--witness")
    parser.add_argument("--dataset-config")
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--osm-cache")
    parser.add_argument("--public-slot-count", type=int)
    parser.add_argument("--task-mode", choices=("retrospective",), default="retrospective")
    parser.add_argument("--evaluation-seed", type=int, default=20260713)
    parser.add_argument("--cdtw-samples", type=int, default=400)
    parser.add_argument("--cdtw-candidates", type=int, default=24)
    parser.add_argument("--cdtw-od-grid", type=int, default=12)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--log-file")
    return parser


def _resolve_config(args: argparse.Namespace) -> dict:
    config = dataset_config(args.dataset_config) if args.dataset_config else {"name": "custom"}
    bbox = args.bbox or config.get("bbox")
    osm = args.osm_cache or config.get("osm_cache")
    if bbox is None or osm is None:
        raise ValueError("evaluation requires a public bbox and OSM cache")
    slot_count = int(args.public_slot_count or config.get("public_slot_count", 0))
    if slot_count <= 0:
        raise ValueError("custom evaluation requires a positive --public-slot-count")
    osm_path = public_path(osm)
    if not osm_path.is_file():
        raise FileNotFoundError(f"OSM cache not found: {osm_path}")
    registered_names = {config["name"], *[str(value).lower() for value in config.get("aliases", [])]}
    if str(args.real).lower() in registered_names and config.get("data"):
        registered_data = config["data"]
        registered_path = public_path(registered_data)
        real_data = registered_path if registered_path.is_file() else registered_data
    else:
        real_data = public_path(args.real) if Path(args.real).suffix else args.real
    expected_real_hash = config.get("data_sha256")
    if expected_real_hash is not None:
        real_path = Path(real_data)
        if not real_path.is_file():
            raise FileNotFoundError(f"frozen real trajectory input not found: {real_path}")
        actual_real_hash = sha256_file(real_path)
        if actual_real_hash != expected_real_hash:
            raise RuntimeError(f"frozen real trajectory input hash mismatch: {actual_real_hash}")
    return {
        "name": config["name"],
        "real_data": real_data,
        "real_data_sha256": expected_real_hash,
        "bbox": tuple(map(float, bbox)),
        "osm": osm_path,
        "osm_max_ways": int(config.get("osm_max_ways", 0)),
        "osm_highway_classes": list(config.get("osm_highway_classes", [])),
        "public_slot_count": slot_count,
    }


def _witness_edges(osm, bbox) -> set[tuple[int, int]]:
    import graph_voronoi_doptimal_support_probe as support
    import route_structure_potential_experiment as route

    support.BBOX = bbox
    _, graph = route.prepare_graph([], bbox=bbox, osm_ways=osm, raw_graph=False)
    return {(int(u), int(v)) for u, adjacency in graph.items() for v, _ in adjacency}


def _generator_protocol(synthetic: Path) -> dict:
    candidate = synthetic.parent / "protocol.json"
    if not candidate.is_file():
        return {}
    return json.loads(candidate.read_text(encoding="utf-8"))


def _real_input_metadata(spec: str, count: int) -> dict:
    candidate = public_path(spec)
    if candidate.is_file():
        return {
            "spec": spec,
            "path": str(candidate),
            "sha256": sha256_file(candidate),
            "count": count,
        }
    return {
        "spec": spec,
        "count": count,
        "sha256": None,
        "hash_note": "Named/multi-file loader input has no single source-file hash.",
    }


def main() -> None:
    args = build_parser().parse_args()
    add_runtime_paths()
    from public_utils import (
        filter_osm_ways_by_bbox,
        filter_osm_ways_by_highway,
        load_osm_ways,
        load_trajectories,
    )
    import downstream_full_protocol as tasks
    import evaluate_kdd_revised_statistics as stats

    config = _resolve_config(args)
    tasks.BBOX = config["bbox"]
    synthetic_path = public_path(args.synthetic)
    witness_path = public_path(args.witness) if args.witness else None
    if not synthetic_path.is_file():
        raise FileNotFoundError(f"synthetic release not found: {synthetic_path}")
    if witness_path is not None and not witness_path.is_file():
        raise FileNotFoundError(f"witness sidecar not found: {witness_path}")
    out_dir = require_new_directory(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    log_file = public_path(args.log_file) if args.log_file else out_dir / "evaluation.log"
    logger = configure_logging(log_file, "evaluate-all")
    started = time.time()

    logger.info("loading real and synthetic trajectories")
    real = load_trajectories(str(config["real_data"]), limit=config["public_slot_count"])
    synthetic = load_trajectories(str(synthetic_path), limit=None)
    if not real or not synthetic:
        raise RuntimeError("real and synthetic datasets must both be non-empty")
    if len(real) != config["public_slot_count"]:
        raise RuntimeError(
            "matched evaluation requires the configured public real-data count: "
            f"expected {config['public_slot_count']}, received {len(real)}"
        )
    if len(synthetic) != config["public_slot_count"]:
        raise RuntimeError(
            "matched evaluation requires one synthetic record per public slot: "
            f"expected {config['public_slot_count']}, received {len(synthetic)}"
        )
    osm = filter_osm_ways_by_bbox(load_osm_ways(str(config["osm"])), config["bbox"])
    osm = filter_osm_ways_by_highway(osm, config["osm_highway_classes"])
    osm_before_thinning = len(osm)
    if config["osm_max_ways"] > 0:
        osm = thin_osm_ways_public(osm, config["bbox"], config["osm_max_ways"])
    if not osm:
        raise RuntimeError("public OSM cache has no ways inside the configured bbox")
    lat0 = float(np.mean(np.vstack([trajectory[[0, -1]] for trajectory in real])[:, 0]))
    logger.info("building public directed graph and statistical metrics")
    context = stats.graph_context(osm, config["bbox"])
    metrics = stats.evaluate(
        real, synthetic, config["bbox"], lat0, context, args.evaluation_seed,
        args.cdtw_samples, args.cdtw_candidates, args.cdtw_od_grid,
    )
    metrics["witness_valid"] = stats.witness_validity(
        witness_path,
        _witness_edges(osm, config["bbox"]),
        len(synthetic),
        sha256_file(config["osm"]),
        sha256_file(synthetic_path),
    )

    logger.info("running retrospective query, OD, destination and grid-route tasks")
    split = int(len(real) * 0.8)
    real_eval = real[split:]
    metrics.update(tasks.count_query_metrics(real, synthetic))
    metrics.update(tasks.od_demand_metrics(real, synthetic))
    metrics.update(tasks.destination_tstr_metrics(synthetic, real_eval))
    metrics.update(tasks.grid_route_choice_metrics(synthetic, real_eval))
    metrics.update(tasks.destination_conditioned_next_region_metrics(synthetic, real_eval))
    for name, value in metrics.items():
        if value is not None and isinstance(value, (int, float, np.number)) and not math.isfinite(float(value)):
            raise RuntimeError(f"metric is not finite: {name}={value}")

    generator = _generator_protocol(synthetic_path)
    protocol = {
        "schema_version": 1,
        "dataset": config["name"],
        "task_mode": args.task_mode,
        "task_warning": (
            "Retrospective TSTR proxy: the synthetic release may have been fitted on the full real corpus; "
            "this is not unseen-contributor generalization."
        ),
        "real_spec": str(config["real_data"]),
        "real_sha256": config["real_data_sha256"],
        "real_count": len(real),
        "public_slot_count": config["public_slot_count"],
        "real_eval_start": split,
        "synthetic_count": len(synthetic),
        "output_yield": len(synthetic) / max(len(real), 1),
        "bbox": config["bbox"],
        "osm_cache": str(config["osm"]),
        "osm_max_ways": config["osm_max_ways"],
        "osm_highway_classes": config["osm_highway_classes"],
        "osm_selection": "bbox_filter_then_highway_class_filter_then_optional_deterministic_spatial_thinning",
        "osm_bbox_way_count": osm_before_thinning,
        "osm_selected_way_count": len(osm),
        "evaluation_seed": args.evaluation_seed,
        "epsilon_total_rational": generator.get("privacy", {}).get("epsilon_total_rational"),
        "noise_seed": generator.get("privacy", {}).get("noise_seed"),
        "algorithm_id": generator.get("algorithm_id"),
        "evaluator": {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "statistics_sha256": sha256_file(PUBLIC_RELEASE / "pipeline" / "evaluate_kdd_revised_statistics.py"),
            "tasks_sha256": sha256_file(PUBLIC_RELEASE / "analysis_scripts" / "downstream_full_protocol.py"),
        },
    }
    payload = {"protocol": protocol, "metrics": jsonable(metrics)}
    write_json(out_dir / "metrics.json", payload)
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for name, value in sorted(metrics.items()):
            writer.writerow({"metric": name, "value": jsonable(value)})
    manifest = {
        "schema_version": 1,
        "classification": "RESEARCH_EVALUATION_LEDGER_NOT_A_DP_RELEASE",
        "inputs": {
            "real": _real_input_metadata(str(config["real_data"]), len(real)),
            "synthetic": {"path": str(synthetic_path), "sha256": sha256_file(synthetic_path)},
            "witness": (
                {"path": str(witness_path), "sha256": sha256_file(witness_path)}
                if witness_path else None
            ),
            "osm": {"path": str(config["osm"]), "sha256": sha256_file(config["osm"])},
        },
        "outputs": {
            "metrics.json": sha256_file(out_dir / "metrics.json"),
            "metrics.csv": sha256_file(out_dir / "metrics.csv"),
        },
        "elapsed_sec": time.time() - started,
    }
    write_json(out_dir / "manifest.json", manifest)
    logger.info("wrote all metrics to %s", out_dir)


if __name__ == "__main__":
    main()
