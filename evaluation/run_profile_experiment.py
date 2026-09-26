from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve_input(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def named_path(text: str) -> tuple[str, Path]:
    try:
        name, value = text.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from exc
    return name, resolve_input(value)


def similarity(value: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(value)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publication_road_metrics(
    metrics: dict, synthetic: Path, witness: Path | None,
    road_routes: Path | None = None, edge_cache: Path | None = None,
    expected_count: int | None = None,
) -> tuple[float, float, str]:
    """Score the delivered road object; keep coordinate-subsampling diagnostics separate."""
    if witness is not None and road_routes is not None:
        raise ValueError("supply one road-object representation per synthetic release")
    if road_routes is not None:
        if edge_cache is None:
            raise ValueError("directed road routes require a public --edge-cache")
        derivation = synthetic.parent / (synthetic.name + ".manifest.json")
        if not derivation.is_file():
            raise FileNotFoundError(f"road-route coordinates require a derivation manifest: {derivation}")
        record = json.loads(derivation.read_text(encoding="utf-8"))
        if record.get("schema") != "road-route-coordinate-derivation-v1":
            raise ValueError("unknown road-route coordinate derivation schema")
        for label, path in (("coordinates", synthetic), ("route_source", road_routes),
                            ("edge_cache", edge_cache)):
            item = record.get(label)
            if not isinstance(item, dict) or item.get("sha256", "").lower() != _sha256(path).lower():
                raise RuntimeError(f"road-route derivation hash mismatch: {label}")
        from route_metric_core import load_pickle, normalize_routes
        cache = load_pickle(edge_cache)
        routes = normalize_routes(load_pickle(road_routes), cache)
        expected_count = int(record["record_count"]) if expected_count is None else int(expected_count)
        if len(routes) != expected_count or int(record["record_count"]) != expected_count:
            raise ValueError("road-route slot count differs from coordinate release")
        public_edges = {tuple(map(int, edge)) for edge in cache["edge_nodes"].values()}
        valid = sum(bool(route) and all(edge in public_edges for edge in route)
                    and all(a[1] == b[0] for a, b in zip(route, route[1:]))
                    for route in routes)
        score = valid / max(expected_count, 1)
        return score, score, "derived_directed_routes"
    if witness is None:
        return (float(metrics["route_compatible_yield"]),
                float(metrics["directed_road_validity"]), "coordinate_projection")
    if witness.parent != synthetic.parent:
        raise ValueError("witness and coordinates must come from the same release directory")
    manifest_path = synthetic.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"witness release requires a hash manifest: {manifest_path}")
    outputs = json.loads(manifest_path.read_text(encoding="utf-8")).get("outputs", {})
    for path in (synthetic, witness):
        expected = outputs.get(path.name)
        if not isinstance(expected, str) or expected.lower() != _sha256(path).lower():
            raise RuntimeError(f"release manifest hash mismatch: {path}")
    validity = metrics.get("witness_valid")
    if validity is None:
        raise RuntimeError("witness validation did not produce a road validity score")
    return float(validity), float(validity), "directed_witness"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the unified evaluator for every synthetic corpus and aggregate a fresh profile."
    )
    parser.add_argument("--real", required=True)
    parser.add_argument("--synthetic", action="append", type=named_path, required=True)
    parser.add_argument("--witness", action="append", type=named_path, default=[])
    parser.add_argument("--road-routes", action="append", type=named_path, default=[])
    parser.add_argument("--edge-cache")
    parser.add_argument("--dataset-config")
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--public-slot-count", type=int)
    parser.add_argument("--osm-cache", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    witnesses = dict(args.witness)
    road_routes = dict(args.road_routes)
    synthetic_names = [name for name, _ in args.synthetic]
    if len(set(synthetic_names)) != len(synthetic_names):
        parser.error("each synthetic method name must be unique")
    if len(witnesses) != len(args.witness) or len(road_routes) != len(args.road_routes):
        parser.error("duplicate road-object method name")
    if (set(witnesses) | set(road_routes)) - set(synthetic_names):
        parser.error("road-object method name has no corresponding --synthetic")
    if set(witnesses) & set(road_routes):
        parser.error("one method cannot provide both --witness and --road-routes")
    if road_routes and not args.edge_cache:
        parser.error("--road-routes requires --edge-cache")
    edge_cache = resolve_input(args.edge_cache) if args.edge_cache else None
    if edge_cache is not None and not edge_cache.is_file():
        parser.error(f"public edge cache not found: {edge_cache}")
    output = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    road_sources = {}
    for name, synthetic in args.synthetic:
        raw = output / "raw" / name.replace(" ", "_")
        command = [
            sys.executable, str(ROOT / "evaluation" / "evaluation" / "evaluate_all.py"),
            "--real", str(resolve_input(args.real)), "--synthetic", str(synthetic),
            "--osm-cache", str(resolve_input(args.osm_cache)), "--out-dir", str(raw),
        ]
        if args.dataset_config:
            command += ["--dataset-config", args.dataset_config]
        if args.bbox:
            command += ["--bbox", *(str(value) for value in args.bbox)]
        if args.public_slot_count:
            command += ["--public-slot-count", str(args.public_slot_count)]
        if name in witnesses:
            command += ["--witness", str(witnesses[name])]
        print("RUN:", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        evaluation = json.loads((raw / "metrics.json").read_text(encoding="utf-8"))
        metrics = evaluation["metrics"]
        road_yield, dir_valid, road_source = publication_road_metrics(
            metrics, synthetic, witnesses.get(name), road_routes.get(name), edge_cache,
            evaluation["protocol"]["synthetic_count"],
        )
        road_sources[name] = road_source
        values = {
            "RoadYield": road_yield,
            "DirValid": dir_valid,
            "WitnessValid": (road_yield if name in road_routes else metrics.get("witness_valid")),
            "TripSim": similarity(metrics.get("trip_error", 1.0)),
            "GridSim": similarity(metrics.get("grid_density_jsd", 1.0)),
            "LengthSim": similarity(metrics.get("path_length_jsd", 1.0)),
            "NextRoadAcc": metrics.get("B2_next_region_accuracy", 0.0),
            "RouteMRR": metrics.get("B2_grid_route_mrr", 0.0),
            "RouteBest5F1": metrics.get("B2_grid_route_best5_transition_f1", 0.0),
        }
        for metric, value in values.items():
            rows.append({"algorithm": name, "layer": "executed_unified_evaluation", "metric": metric,
                         "value": value, "status": "VALID" if value is not None else "NOT_EMITTED"})

    result = output / "results.csv"
    with result.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({
        "protocol": "executed_full_population_profile_v1",
        "corpora": [name for name, _ in args.synthetic],
        "road_metric_source": road_sources,
        "road_route_inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in road_routes.items()
        },
        "witness_inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in witnesses.items()
        },
        "coordinate_projection_diagnostics": {
            name: str((output / "raw" / name.replace(" ", "_") / "metrics.json").resolve())
            for name, _ in args.synthetic
        },
        "result": str(result.resolve()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} freshly evaluated metric rows to {result}")


if __name__ == "__main__":
    main()
