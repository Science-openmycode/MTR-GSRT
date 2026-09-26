from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

from route_metric_core import evaluate_routes, load_pickle, normalize_routes, route_counters, valid_routes


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "package.json").read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def verify_saved_gsrt_arm(arm: Path, evaluated: Path, mode: str, args: argparse.Namespace) -> None:
    protocol = json.loads((arm / "protocol.json").read_text(encoding="utf-8"))
    expected = (
        (protocol["postprocessing"], "component_mode", mode),
        (protocol["postprocessing"], "decoder_seed", args.decoder_seed),
        (protocol["privacy"], "noise_seed", args.noise_seed),
        (protocol["privacy"], "epsilon_total_rational", args.epsilon_total),
    )
    for settings, key, value in expected:
        if settings.get(key) != value:
            raise ValueError(f"saved ablation arm has different {key}: {arm}")
    if protocol["public_slot_count"] != args.public_slot_count:
        raise ValueError(f"saved ablation arm has different public slot count: {arm}")
    if Path(protocol["data_spec"]).resolve() != Path(args.data).resolve():
        raise ValueError(f"saved ablation arm has different real input path: {arm}")
    ledger = json.loads((evaluated / "manifest.json").read_text(encoding="utf-8"))
    checks = [
        (Path(args.data).resolve(), ledger["inputs"]["real"]["sha256"]),
        (arm / "trajectories.pkl", ledger["inputs"]["synthetic"]["sha256"]),
        (arm / "road_witnesses.pkl", ledger["inputs"]["witness"]["sha256"]),
        (evaluated / "metrics.json", ledger["outputs"]["metrics.json"]),
    ]
    for path, expected_hash in checks:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != expected_hash.lower():
            raise ValueError(f"saved ablation hash mismatch: {path}")


def write(rows: list[dict], output: Path, protocol: str, extra: dict | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    result = output / "results.csv"
    with result.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"protocol": protocol, "rows": len(rows), "result": str(result.resolve())}
    payload.update(extra or {})
    (output / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def gsrt(args: argparse.Namespace, output: Path) -> None:
    modes = args.component_modes or ["full", "no-portal-fiber", "no-graph-flow", "demand-only"]
    labels = {
        "full": "MTR-GSRT",
        "no-portal-fiber": "No Portal-Fiber measurement",
        "no-graph-flow": "No compact graph-flow measurement",
        "demand-only": "Demand measurements only",
    }
    rows = []
    keys = {
        "Grid": "grid_density_jsd", "Trip": "trip_error", "Len": "path_length_jsd",
        "WitnessValid": "witness_valid", "NextRoadAcc": "B2_next_region_accuracy",
        "NextRoadNLL": "B2_next_region_nll",
    }
    cache = load_pickle(Path(args.edge_cache).resolve()) if args.real_routes else None
    real = (valid_routes(normalize_routes(load_pickle(Path(args.real_routes).resolve()), cache))
            if args.real_routes else None)
    reference = route_counters(real, cache) if real is not None else None
    for mode in modes:
        arm = output / "arms" / mode
        evaluated = output / "raw" / mode
        generation = [
            sys.executable, str(ROOT / CONFIG["main_generation_entry"]),
            "--data", str(Path(args.data).resolve()),
            "--epsilon-total", args.epsilon_total,
            "--noise-seed", str(args.noise_seed),
            "--decoder-seed", str(args.decoder_seed),
            "--public-slot-count", str(args.public_slot_count),
            "--osm-cache", str(Path(args.osm_cache).resolve()),
            "--component-mode", mode,
            "--out-dir", str(arm),
        ]
        if args.dataset_config:
            generation += ["--dataset-config", args.dataset_config]
        if args.bbox:
            generation += ["--bbox", *(str(value) for value in args.bbox)]
        if args.request_seed is not None:
            generation += ["--request-seed", str(args.request_seed)]
        if args.limit is not None:
            generation += ["--limit", str(args.limit)]
        if not args.reuse_generated:
            run(generation)
        evaluation = [
            sys.executable, str(ROOT / "evaluation" / "evaluation" / "evaluate_all.py"),
            "--real", str(Path(args.data).resolve()),
            "--synthetic", str(arm / "trajectories.pkl"),
            "--witness", str(arm / "road_witnesses.pkl"),
            "--osm-cache", str(Path(args.osm_cache).resolve()),
            "--public-slot-count", str(args.public_slot_count),
            "--out-dir", str(evaluated),
        ]
        if args.dataset_config:
            evaluation += ["--dataset-config", args.dataset_config]
        if args.bbox:
            evaluation += ["--bbox", *(str(value) for value in args.bbox)]
        if args.reuse_generated:
            verify_saved_gsrt_arm(arm, evaluated, mode, args)
        else:
            run(evaluation)
        metrics = json.loads((evaluated / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        row = {"Ablation arm": labels[mode], **{label: metrics[key] for label, key in keys.items()}}
        if real is not None:
            routes = normalize_routes(load_pickle(arm / "road_witnesses.pkl"), cache)
            scored = evaluate_routes(routes, real, cache, reference)
            row.update({key: scored[key] for key in (
                "RoadYield", "DemandFid", "BTF", "RC-CPC",
                "EdgeCPC", "TurnCPC", "FamilyCPC", "BranchCPC", "ExitAcc",
                "ODPF96", "ODPF384")})
        rows.append(row)
    write(rows, output, "generated_mtr_gsrt_algorithm_ablation_v2", {
        "component_modes": modes,
        "generation": "each arm was synthesized from the supplied dataset",
        "reuse_generated": bool(args.reuse_generated),
        "real_routes": str(Path(args.real_routes).resolve()) if args.real_routes else None,
    })


def save_routes(path: Path, routes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=6) as handle:
        pickle.dump(routes, handle, pickle.HIGHEST_PROTOCOL)


def dfr(args: argparse.Namespace, output: Path) -> None:
    if args.reuse_generated:
        cache = load_pickle(Path(args.edge_cache).resolve())
        real = valid_routes(normalize_routes(load_pickle(Path(args.real_routes).resolve()), cache))
        reference = route_counters(real, cache)
        selected = args.variants or [
            "no-endpoint-measurement", "endpoint-measurement-only",
            "family-measured-unmatched-router", "full"]
        rows, hashes = [], {}
        for name in selected:
            path = output / "arms" / f"{name}.pkl.gz"
            raw = load_pickle(path)
            routes = normalize_routes(raw, cache)
            if len(routes) != args.public_slot_count:
                raise ValueError(f"saved DFR arm has wrong public slot count: {path}")
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({"Ablation arm": name,
                         **evaluate_routes(routes, real, cache, reference)})
        write(rows, output, "generated_mtr_dfr_algorithm_ablation_v2", {
            "variants": selected, "reuse_generated": True,
            "arm_sha256": hashes,
            "real_routes_sha256": hashlib.sha256(
                Path(args.real_routes).resolve().read_bytes()).hexdigest(),
        })
        return
    algorithm = ROOT / "generation" / "mtr_dfr" / "algorithm"
    dependencies = ROOT / "generation" / "mtr_dfr" / "dependencies"
    sys.path[:0] = [str(dependencies), str(algorithm)]
    module = importlib.import_module("run_minimal_strong_family")
    module.EDGE_CACHE = Path(args.edge_cache).resolve()
    module.REAL_MATCHED = Path(args.real_routes).resolve()
    module.NETWORK = Path(args.network).resolve()
    module.SLOTS = int(args.public_slot_count)
    cache, real_raw = module.load_private_inputs()
    real = valid_routes(normalize_routes(real_raw, cache))

    router = module.EdgeStateRouter(module.NETWORK, cache["edge_nodes"],
                                    {None: {}}, {None: {}}, 0.0, 0.0)
    endpoints, endpoint_transcript = module.measure_and_sample_endpoints(
        cache, real_raw, args.seed, length_aware=args.length_aware, router=router)
    router, endpoint_only, endpoint_states = module.public_shortest_routes(cache, endpoints, router)
    full, family_transcript = module.route_family_release(
        cache, real_raw, endpoints, endpoint_only, router, args.seed)

    edges, _attrs = module.build_public_endpoint_domain(cache)
    rng = np.random.default_rng(args.seed + 910000)
    public_endpoints = [
        (edges[int(i)], edges[int(j)])
        for i, j in zip(rng.integers(0, len(edges), size=module.SLOTS),
                        rng.integers(0, len(edges), size=module.SLOTS))
    ]
    public_router = module.EdgeStateRouter(module.NETWORK, cache["edge_nodes"],
                                           {None: {}}, {None: {}}, 0.0, 0.0)
    public_router, public_tail, public_states = module.public_shortest_routes(
        cache, public_endpoints, public_router)
    no_endpoint, _ = module.route_family_release(
        cache, real_raw, public_endpoints, public_tail, public_router, args.seed)

    all_arms = {
        "no-endpoint-measurement": no_endpoint,
        "endpoint-measurement-only": endpoint_only,
        "family-measured-unmatched-router": endpoint_only,
        "full": full,
    }
    selected = args.variants or list(all_arms)
    reference = route_counters(real, cache)
    rows = []
    for name in selected:
        routes = all_arms[name]
        path = output / "arms" / f"{name}.pkl.gz"
        save_routes(path, routes)
        metrics = evaluate_routes(normalize_routes(routes, cache), real, cache, reference)
        rows.append({"Ablation arm": name, **metrics})
    write(rows, output, "generated_mtr_dfr_algorithm_ablation_v2", {
        "variants": selected,
        "generation": "all selected arms were synthesized from the supplied matched dataset",
        "endpoint_transcript": endpoint_transcript,
        "family_transcript": family_transcript,
        "endpoint_tail_states": dict(endpoint_states),
        "public_tail_states": dict(public_states),
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate algorithm-level ablation releases from an explicit dataset.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data", help="Real coordinate trajectories for MTR-GSRT.")
    parser.add_argument("--dataset-config")
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--osm-cache")
    parser.add_argument("--epsilon-total", default="7/5")
    parser.add_argument("--noise-seed", type=int, default=20260719)
    parser.add_argument("--decoder-seed", type=int, default=30260719)
    parser.add_argument("--request-seed", type=int)
    parser.add_argument("--public-slot-count", type=int, default=17123)
    parser.add_argument("--limit", type=int,
                        help="Smoke-test only; must equal --public-slot-count.")
    parser.add_argument("--component-modes", nargs="+", choices=(
        "full", "no-portal-fiber", "no-graph-flow", "demand-only"))
    parser.add_argument("--reuse-generated", action="store_true",
                        help="re-evaluate existing saved arms without synthesizing them again")
    parser.add_argument("--real-routes", help="Real matched road routes for MTR-DFR or optional GSRT road-choice ablation.")
    parser.add_argument("--edge-cache", default=str(
        ROOT / "public_assets" / "ordered_portal_route_cache.pkl.gz"))
    parser.add_argument("--network", default=str(
        ROOT / "public_assets" / "beijing_network" / "network.shp"))
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--length-aware", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=(
        "no-endpoint-measurement", "endpoint-measurement-only",
        "family-measured-unmatched-router", "full"))
    args = parser.parse_args()
    output = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if args.public_slot_count <= 0:
        parser.error("--public-slot-count must be positive")
    if CONFIG["concrete_algorithm"] == "mtr_gsrt":
        if not args.data or not args.osm_cache:
            parser.error("MTR-GSRT requires --data and --osm-cache")
        gsrt(args, output)
    else:
        if not args.real_routes:
            parser.error("MTR-DFR requires --real-routes")
        dfr(args, output)


if __name__ == "__main__":
    main()
