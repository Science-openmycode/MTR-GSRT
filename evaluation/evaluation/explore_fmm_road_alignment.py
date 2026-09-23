"""Compare continuous public map-matching road alignment across releases.

Unlike WitnessAlignment, this diagnostic is available to every release.  A
shared public FMM matcher infers a directed road path for each coordinate
sequence.  It is an exploratory comparison, not a replacement for the
consumer-visible WitnessValid certificate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "configs" / "datasets.json").is_file():
            sys.path.insert(0, str(candidate))
            break

from generation.common.runtime import dataset_config, public_path, write_json  # noqa: E402
from metric_suites.road_route import _jsd_counters, od_conditioned_transition_jsd, run_fmm  # noqa: E402


def _named_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"expected NAME=PATH, got {specification!r}")
    name, value = specification.split("=", 1)
    if not name:
        raise ValueError("method name must not be empty")
    return name, public_path(value)


def _summarize(records, tau_m: float) -> dict[str, float]:
    scores = []
    residuals = []
    for record in records:
        if not record.connected or not record.residual_m:
            scores.append(0.0)
            continue
        q95 = float(np.quantile(record.residual_m, 0.95))
        residuals.append(q95)
        scores.append(float(record.observation_share) * math.exp(-q95 / tau_m))
    accepted = np.asarray([record.accepted for record in records], dtype=float)
    return {
        "record_count": len(records),
        "fmm_road_alignment_mean": float(np.mean(scores)) if scores else 0.0,
        "fmm_road_alignment_median": float(np.median(scores)) if scores else 0.0,
        "fmm_road_alignment_p05": float(np.quantile(scores, 0.05)) if scores else 0.0,
        "fmm_connected_fraction": float(np.mean([record.connected for record in records])) if records else 0.0,
        "fmm_accepted_fraction": float(np.mean(accepted)) if len(accepted) else 0.0,
        "fmm_q95_residual_median_m": float(np.median(residuals)) if residuals else None,
    }


def _edge_counts(records) -> Counter:
    counts = Counter()
    for record in records:
        if not record.accepted:
            counts["__INVALID__"] += 1.0
        else:
            counts.update(record.cpath)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True, help="Repeat NAME=RELEASE_PATH")
    parser.add_argument("--real", default=None, help="Optional real trajectory release for completed-path fidelity.")
    parser.add_argument("--dataset-config", default=None, help="Required with --real to calculate OD-conditioned JSD.")
    parser.add_argument("--real-calibration-offset", type=int, default=None, help="Optional disjoint real slice offset for finite-sample calibration.")
    parser.add_argument("--network", required=True)
    parser.add_argument("--fmm", required=True)
    parser.add_argument("--fmm-runtime-dir", default=None, help="Directory containing FMM runtime DLLs.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--max-points", type=int, default=32)
    parser.add_argument("--radius-m", type=float, default=200.0)
    parser.add_argument("--gps-error-m", type=float, default=50.0)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--min-observation-share", type=float, default=0.95)
    parser.add_argument("--tau-m", type=float, default=200.0)
    args = parser.parse_args()
    if args.limit < 1 or args.tau_m <= 0:
        raise ValueError("--limit and --tau-m must be positive")
    if bool(args.real) != bool(args.dataset_config):
        raise ValueError("--real and --dataset-config must be supplied together")

    from public_utils import load_trajectories
    methods = dict(_named_path(value) for value in args.method)
    if len(methods) != len(args.method):
        raise ValueError("method names must be unique")
    corpora = {name: load_trajectories(str(path), limit=args.limit) for name, path in methods.items()}
    if any(len(corpus) != args.limit for corpus in corpora.values()):
        raise ValueError("every method must contain at least --limit trajectories")
    names = list(corpora)
    real = load_trajectories(str(public_path(args.real)), limit=args.limit) if args.real else None
    calibration = None
    if args.real_calibration_offset is not None:
        if real is None or args.real_calibration_offset < args.limit:
            raise ValueError("--real-calibration-offset requires --real and must be at least --limit")
        real_source = load_trajectories(
            str(public_path(args.real)), limit=args.real_calibration_offset + args.limit
        )
        if len(real_source) != args.real_calibration_offset + args.limit:
            raise ValueError("real release lacks the requested calibration slice")
        real = real_source[:args.limit]
        calibration = real_source[args.real_calibration_offset:args.real_calibration_offset + args.limit]
    if real is not None and len(real) != args.limit:
        raise ValueError("real release must contain at least --limit trajectories")
    combined = (
        ([*real] if real is not None else [])
        + ([*calibration] if calibration is not None else [])
        + [trajectory for name in names for trajectory in corpora[name]]
    )
    records, invocation = run_fmm(
        combined,
        public_path(args.network),
        None,
        public_path(args.fmm),
        None,
        public_path(args.fmm_runtime_dir) if args.fmm_runtime_dir else None,
        args.max_points,
        args.radius_m,
        args.gps_error_m,
        args.candidates,
        args.min_observation_share,
        route_only=False,
    )
    real_records = records[:args.limit] if real is not None else None
    calibration_records = records[args.limit:2 * args.limit] if calibration is not None else None
    offset = args.limit * (1 + int(calibration is not None)) if real is not None else 0
    results = {}
    for index, name in enumerate(names):
        start = offset + index * args.limit
        method_records = records[start:start + args.limit]
        results[name] = _summarize(method_records, args.tau_m)
        if real_records is not None:
            results[name]["completed_edge_flow_jsd"] = _jsd_counters(_edge_counts(real_records), _edge_counts(method_records))
            results[name]["completed_od_transition_jsd"] = od_conditioned_transition_jsd(
                real, real_records, corpora[name], method_records,
                tuple(map(float, dataset_config(args.dataset_config)["bbox"])), 8,
            )
    payload = {
        "schema_version": 1,
        "classification": "EXPLORATORY_SHARED_PUBLIC_FMM_ROAD_ALIGNMENT_NOT_PAPER_METRIC",
        "definition": "I_connected * observation_share * exp(-q95_FMM_residual_m/tau_m); unmatched or disconnected records receive zero",
        "interpretation": "shared-evaluator continuous road compatibility, not a released-witness certificate",
        "parameters": {
            "limit": args.limit, "max_points": args.max_points, "radius_m": args.radius_m,
            "gps_error_m": args.gps_error_m, "candidates": args.candidates,
            "min_observation_share": args.min_observation_share, "tau_m": args.tau_m,
        },
        "methods": {name: str(path) for name, path in methods.items()},
        "real": str(public_path(args.real)) if args.real else None,
        "results": results,
        "real_vs_real_calibration": (
            {
                "completed_edge_flow_jsd": _jsd_counters(_edge_counts(real_records), _edge_counts(calibration_records)),
                "completed_od_transition_jsd": od_conditioned_transition_jsd(
                    real, real_records, calibration, calibration_records,
                    tuple(map(float, dataset_config(args.dataset_config)["bbox"])), 8,
                ),
            }
            if calibration_records is not None else None
        ),
        "fmm_invocation": invocation,
    }
    out_dir = public_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "fmm_road_alignment.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
