"""Cache one common road-matching pass for existing trajectory corpora only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (_candidate / "configs" / "datasets.json").is_file():
            sys.path.insert(0, str(_candidate))
            break
    else:
        raise RuntimeError("cannot locate public_release root")

from generation.common.runtime import add_runtime_paths, dataset_config, public_path, sha256_file, write_json  # noqa: E402
from metric_suites.road_route import run_fmm  # noqa: E402


def _trajectory_content_sha256(trajectories) -> str:
    import numpy as np

    digest = hashlib.sha256()
    digest.update(len(trajectories).to_bytes(8, "little"))
    for trajectory in trajectories:
        array = np.ascontiguousarray(np.asarray(trajectory, dtype="<f8"))
        digest.update(len(array).to_bytes(8, "little"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _serializable(record, offset: int) -> dict:
    return {
        "source_index": int(record.source_index) + offset,
        "cpath": tuple(int(value) for value in record.cpath),
        "opath": tuple(int(value) for value in record.opath),
        "residual_m": tuple(float(value) for value in record.residual_m),
        "connected": bool(record.connected),
        "observation_share": float(record.observation_share),
        "accepted": bool(record.accepted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--corpus", action="append", default=[], help="Repeat NAME=PATH; omit for Real-only calibration")
    parser.add_argument("--network", required=True)
    parser.add_argument("--stmatch-bin")
    parser.add_argument("--runtime-dir", help="Directory containing the FMM GDAL and Boost runtime DLLs")
    parser.add_argument("--max-points", type=int, default=32)
    parser.add_argument("--radius-m", type=float, default=200.0)
    parser.add_argument("--gps-error-m", type=float, default=50.0)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument(
        "--corpus-workers",
        type=int,
        default=1,
        help="Number of corpora matched concurrently; total STMatch processes are bounded by corpus-workers times parallel-workers.",
    )
    parser.add_argument("--omp-threads-per-worker", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Deterministic public prefix for screening only")
    parser.add_argument(
        "--strict-observation-audit",
        action="store_true",
        help="Request opath/pgeom and require at least 0.95 observation coverage instead of route-only cpath acceptance.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    add_runtime_paths()
    from public_utils import load_trajectories

    config = dataset_config(args.dataset_config)
    full_count = int(config["public_slot_count"])
    expected = min(full_count, args.limit) if args.limit is not None else full_count
    if expected <= 0:
        raise ValueError("--limit must be positive")
    if args.parallel_workers <= 0:
        raise ValueError("--parallel-workers must be positive")
    if args.corpus_workers <= 0:
        raise ValueError("--corpus-workers must be positive")
    if args.omp_threads_per_worker <= 0:
        raise ValueError("--omp-threads-per-worker must be positive")
    # Each chunk launches one OpenMP-enabled STMatch process.  Bounding the
    # inner team prevents parallel chunks from multiplying into hundreds of
    # competing threads on shared machines.
    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads_per_worker)
    out_dir = public_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    match_dir = out_dir / "matched_paths"
    match_dir.mkdir(exist_ok=True)
    network = public_path(args.network)
    binary = public_path(args.stmatch_bin) if args.stmatch_bin else public_path(
        "third_party/fmm-v0.1.1/cyang-kth-fmm-344fb8c/build/Release/stmatch.exe"
    )
    runtime_dir = Path(args.runtime_dir).resolve() if args.runtime_dir else None
    registered = {str(config["name"]).lower(), *[str(value).lower() for value in config.get("aliases", [])]}
    real_spec = config["data"] if args.real.lower() in registered else args.real
    corpora: list[tuple[str, str]] = [("Real", real_spec)]
    seen = {"Real"}
    for spec in args.corpus:
        if "=" not in spec:
            raise ValueError(f"expected NAME=PATH: {spec}")
        name, path = spec.split("=", 1)
        if not name or name in seen:
            raise ValueError(f"duplicate or empty corpus name: {name!r}")
        seen.add(name)
        corpora.append((name, path))
    parameters = {
        "matcher": "stmatch",
        "max_points": args.max_points,
        "radius_m": args.radius_m,
        "gps_error_m": args.gps_error_m,
        "candidates": args.candidates,
        "route_only": not args.strict_observation_audit,
        "runtime_dir": str(runtime_dir) if runtime_dir else None,
        "screening_limit": args.limit,
    }
    completed: dict[str, dict] = {}
    started = time.time()

    def process_corpus(name: str, spec: str) -> tuple[str, dict]:
        source = public_path(spec)
        trajectories = None
        if source.is_file():
            source_for_loader = str(source)
            input_hash = sha256_file(source)
            input_hash_semantics = "file_sha256"
        else:
            source_for_loader = spec
            trajectories = load_trajectories(source_for_loader, limit=expected)
            input_hash = _trajectory_content_sha256(trajectories)
            input_hash_semantics = "canonical_float64_trajectory_content_sha256"
        target = match_dir / f"{name}.pkl.gz"
        sidecar = match_dir / f"{name}.manifest.json"
        if args.resume and target.is_file() and sidecar.is_file():
            prior = json.loads(sidecar.read_text(encoding="utf-8"))
            prior_parameters = dict(prior.get("parameters", {}))
            prior_parameters.pop("batch_size", None)
            prior_parameters.setdefault("screening_limit", None)
            if (
                prior.get("input_sha256") == input_hash
                and prior.get("network_sha256") == sha256_file(network)
                and prior_parameters == parameters
                and prior.get("output_sha256") == sha256_file(target)
                and int(prior.get("record_count", -1)) == expected
            ):
                print(f"[resume] {name}", flush=True)
                return name, prior
            raise RuntimeError(f"incomplete or incompatible resume cache for {name}: {target}")
        if target.exists() or sidecar.exists():
            raise FileExistsError(f"refusing to overwrite partial cache for {name}: {target}")
        if trajectories is None:
            trajectories = load_trajectories(source_for_loader, limit=expected)
        if len(trajectories) != expected:
            raise RuntimeError(f"{name} contains {len(trajectories)} records; expected {expected}")
        records_by_start: dict[int, list[dict]] = {}
        invocations_by_start: dict[int, dict] = {}
        corpus_started = time.time()

        def match_chunk(start: int, stop: int) -> tuple[int, list[dict], dict]:
            matched, invocation = run_fmm(
                trajectories[start:stop], network, None, binary, None, runtime_dir,
                args.max_points, args.radius_m, args.gps_error_m, args.candidates,
                0.95, None, not args.strict_observation_audit,
            )
            serialized = [_serializable(record, start) for record in matched]
            invocation["start"] = start
            invocation["stop"] = stop
            return start, serialized, invocation

        chunks = [
            (start, min(start + args.batch_size, expected))
            for start in range(0, expected, args.batch_size)
        ]
        if args.parallel_workers == 1:
            completed_chunks = (match_chunk(start, stop) for start, stop in chunks)
            for start, serialized, invocation in completed_chunks:
                records_by_start[start] = serialized
                invocations_by_start[start] = invocation
                print(f"[{name}] {invocation['stop']}/{expected}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
                futures = {
                    executor.submit(match_chunk, start, stop): (start, stop)
                    for start, stop in chunks
                }
                for future in as_completed(futures):
                    start, serialized, invocation = future.result()
                    records_by_start[start] = serialized
                    invocations_by_start[start] = invocation
                    print(f"[{name}] completed {invocation['start']}:{invocation['stop']}", flush=True)

        records: list[dict] = []
        invocations: list[dict] = []
        for start, _ in chunks:
            records.extend(records_by_start[start])
            invocations.append(invocations_by_start[start])
        if len(records) != expected:
            raise RuntimeError(f"{name} cache assembled {len(records)} records; expected {expected}")
        with gzip.open(target, "wb", compresslevel=6) as handle:
            pickle.dump(records, handle, pickle.HIGHEST_PROTOCOL)
        side = {
            "schema_version": 1,
            "classification": "COMMON_MAP_MATCH_CACHE_NO_SYNTHESIS",
            "corpus": name,
            "record_count": expected,
            "input": source_for_loader,
            "input_sha256": input_hash,
            "input_hash_semantics": input_hash_semantics,
            "network": str(network),
            "network_sha256": sha256_file(network),
            "parameters": parameters,
            "invocations": invocations,
            "output": str(target),
            "output_sha256": sha256_file(target),
            "elapsed_sec": time.time() - corpus_started,
            "execution_parallel_workers": args.parallel_workers,
            "execution_batch_size": args.batch_size,
            "execution_omp_threads_per_worker": args.omp_threads_per_worker,
        }
        write_json(sidecar, side)
        return name, side

    if args.corpus_workers == 1:
        for name, spec in corpora:
            completed_name, side = process_corpus(name, spec)
            completed[completed_name] = side
    else:
        with ThreadPoolExecutor(max_workers=args.corpus_workers) as executor:
            futures = {
                executor.submit(process_corpus, name, spec): name
                for name, spec in corpora
            }
            unordered: dict[str, dict] = {}
            for future in as_completed(futures):
                completed_name, side = future.result()
                unordered[completed_name] = side
                print(f"[corpus] completed {completed_name}", flush=True)
        completed = {name: unordered[name] for name, _ in corpora}
    manifest = {
        "schema_version": 1,
        "classification": "COMMON_MAP_MATCH_CACHE_NO_SYNTHESIS",
        "no_synthesis": True,
        "dataset": config["name"],
        "record_count": expected,
        "full_public_slot_count": full_count,
        "parameters": parameters,
        "execution_corpus_workers": args.corpus_workers,
        "corpora": completed,
        "elapsed_sec": time.time() - started,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "out_dir": str(out_dir), "corpora": list(completed)}, indent=2))


if __name__ == "__main__":
    main()
