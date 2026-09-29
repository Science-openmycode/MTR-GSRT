from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from synthesis_lineage import bind_records, bind_split


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "evaluation" / "pipeline"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_release_hash(path: Path, seen: frozenset[Path] = frozenset()) -> str:
    """Bind a routed view to the release it reconstructs, not just its label."""
    path = path.resolve()
    if path in seen or len(seen) >= 32:
        raise ValueError("Cyclic or excessive release derivation chain")
    seen = seen | {path}
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".manifest.json")
    if sidecar.is_file():
        manifest = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        source_record = None
        if manifest.get("schema") == "road-route-coordinate-derivation-v1":
            if manifest.get("coordinates", {}).get("sha256") != digest:
                raise ValueError(f"Road coordinate derivation output hash mismatch: {path}")
            source_record = manifest.get("route_source", {})
        elif manifest.get("router") in {"FMM", "STMatch"}:
            if manifest.get("output", {}).get("sha256") != digest:
                raise ValueError(f"Routed view output hash mismatch: {path}")
            source_record = manifest.get("inputs", {}).get("source", {})
        if source_record is not None:
            source = source_record.get("sha256")
            if not source:
                raise ValueError(f"Routed view missing source release binding: {path}")
            raw_source = source_record.get("path")
            if raw_source:
                original = Path(raw_source)
                original = original if original.is_absolute() else path.parent / original
                if original.is_file():
                    if sha256_file(original) != source:
                        raise ValueError(f"Derivation source file hash mismatch: {original}")
                    return base_release_hash(original, seen)
            return source
    return digest


def measurement_name(name: str) -> str:
    for router in ("Native", "FMM", "STMatch"):
        for separator in (" x ", " × "):
            suffix = separator + router
            if name.endswith(suffix):
                return name[:-len(suffix)]
    return name


def run(script: str, common: list[str], out_dir: Path) -> None:
    command = [sys.executable, str(PIPELINE / script), *common, "--out-dir", str(out_dir)]
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_ratio(value: str, reference: str) -> float:
    a, b = float(value), float(reference)
    if b <= 0:
        return 1.0 if a <= 0 else 0.0
    return max(0.0, min(1.0, a / b))


def retained(row: dict[str, str], reference: dict[str, str], coverage: str, quality: str) -> float:
    """Overall task utility: answered-query share times conditional quality."""
    for source in (row, reference):
        for key in (coverage, quality):
            value = float(source[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid task probability {key}: {value}")
    numerator = float(row[coverage]) * float(row[quality])
    denominator = float(reference[coverage]) * float(reference[quality])
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def aggregate(generic_dir: Path, road_dir: Path, output: Path, input_binding: dict | None = None) -> None:
    generic = read_rows(generic_dir / "generic_mobility_tasks.csv")
    road = read_rows(road_dir / "road_network_mining_tasks.csv")
    generic_by = {row["method"]: row for row in generic}
    road_by = {row["method"]: row for row in road}
    if len(generic_by) != len(generic) or len(road_by) != len(road):
        raise ValueError("Duplicate method names in task evaluator output")
    if set(generic_by) != set(road_by):
        raise ValueError("Generic and road evaluator method sets differ")
    real_g = generic_by["Real-train"]
    real_r = road_by["Real-train"]
    names = [name for name in generic_by if name != "Real-train"]
    fields = [
        ("Next-cell Hit@1", generic_by, real_g, "next_cell_coverage", "next_cell_hit_at_1"),
        ("Next-cell MRR", generic_by, real_g, "next_cell_coverage", "next_cell_mrr"),
        ("Destination Hit@5", generic_by, real_g, "destination_coverage", "destination_hit_at_5"),
        ("Destination MRR", generic_by, real_g, "destination_coverage", "destination_mrr"),
        ("Road continuation Hit@1", road_by, real_r, "continuation_coverage", "continuation_hit_at_1"),
        ("Road continuation MRR", road_by, real_r, "continuation_coverage", "continuation_mrr"),
        ("Route retrieval NDCG@5", road_by, real_r, "retrieval_coverage", "retrieval_ndcg_at_5"),
    ]
    rows = [{"Pipeline": "Real-train", **{label: 1.0 for label, *_ in fields}}]
    for name in names:
        row = {"Pipeline": name}
        for label, source, reference, coverage, quality in fields:
            if name not in source:
                raise RuntimeError(f"{name!r} missing from {quality} evaluator output")
            row[label] = retained(source[name], reference, coverage, quality)
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    result = output / "results.csv"
    with result.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({
        "protocol": "train_test_bound_task_scoring_v2" if input_binding else "legacy_task_scoring_without_split_binding",
        "raw_generic_results": str((generic_dir / "generic_mobility_tasks.json").resolve()),
        "raw_road_results": str((road_dir / "road_network_mining_tasks.json").resolve()),
        "result": str(result.resolve()),
        "normalization": "coverage times conditional quality, divided by the Real-train product; Real-train=1",
        "pipelines": [row["Pipeline"] for row in rows],
        "split_binding": input_binding,
        "synthesis_lineage_verified": bool(input_binding and input_binding.get("synthesis_lineage")
            and set(input_binding["synthesis_lineage"]) == set(input_binding["base_releases"])),
        "disjoint_split_verified": bool(input_binding and input_binding.get("disjoint_split")),
        "lineage_note": "Executed generator input binding is checked separately from task scoring; it is local research provenance, not a DP release or formal noninterference proof.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute strict train-only TSTR models and aggregate their held-out scores."
    )
    parser.add_argument("--train-real", required=True)
    parser.add_argument("--test-real", required=True)
    parser.add_argument("--synthetic", action="append", required=True,
                        help="Synthetic training release; repeat once per method.")
    parser.add_argument("--road-synthetic", action="append",
                        help="Optional full directed-road coordinate view for road tasks; repeat in the same order.")
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--osm-cache", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--synthesis-lineage", action="append", default=[],
                        help="MEASUREMENT=local generator record.json; repeat per base measurement")
    parser.add_argument("--require-synthesis-lineage", action="store_true",
                        help="Reject before scoring unless every release has executed train-only provenance")
    parser.add_argument("--split-audit", help="Local audit_tstr_split.py record binding disjoint indexed inputs")
    args = parser.parse_args()
    if len(args.names) != len(args.synthetic):
        parser.error("--names must contain one name for every repeated --synthetic argument")
    if len(set(args.names)) != len(args.names) or "Real-train" in args.names:
        parser.error("--names must be unique and must not use the reserved Real-train name")
    if args.road_synthetic is not None and len(args.road_synthetic) != len(args.synthetic):
        parser.error("--road-synthetic must be omitted or repeated once per --synthetic")
    for value in [*args.synthetic, *(args.road_synthetic or [])]:
        if Path(value).suffix.lower() not in {".pkl", ".npy", ".npz", ".json"}:
            parser.error("TSTR requires coordinate views (.pkl/.npy/.npz/.json); "
                         "convert directed-road objects with routes-to-coordinates first")
    def resolved(value: str) -> str:
        path = Path(value)
        return str((path if path.is_absolute() else ROOT / path).resolve())
    output = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if output.exists():
        parser.error(f"Choose a new task output directory; existing results are not overwritten: {output}")
    train_path, test_path = Path(resolved(args.train_real)), Path(resolved(args.test_real))
    input_binding = {
        "classification": "LOCAL_RESEARCH_AUDIT_NOT_A_DP_RELEASE",
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "bbox": args.bbox,
        "seed": args.seed,
        "osm_sha256": sha256_file(Path(resolved(args.osm_cache))),
    }
    if input_binding["train_sha256"] == input_binding["test_sha256"]:
        parser.error("train and test inputs are identical")
    input_binding["synthetic"] = {
        name: sha256_file(Path(resolved(path)))
        for name, path in zip(args.names, args.synthetic)
    }
    input_binding["base_releases"] = {
        measurement_name(name): base_release_hash(Path(resolved(path)))
        for name, path in zip(args.names, args.synthetic)
    }
    if len(input_binding["base_releases"]) != len(args.names):
        parser.error("duplicate measurement names after router suffix normalization")
    if args.road_synthetic:
        for name, path in zip(args.names, args.road_synthetic):
            if base_release_hash(Path(resolved(path))) != input_binding["base_releases"][measurement_name(name)]:
                parser.error(f"{name}: coordinate and road views use different base releases")
    input_binding["synthesis_lineage"] = bind_records(
        args.synthesis_lineage, ROOT, input_binding, args.require_synthesis_lineage)
    if args.require_synthesis_lineage and not args.split_audit:
        parser.error("Strict TSTR also requires --split-audit from audit_tstr_split.py")
    input_binding["disjoint_split"] = (bind_split(Path(resolved(args.split_audit)),
        input_binding["train_sha256"], input_binding["test_sha256"]) if args.split_audit else None)
    generic_dir, road_dir = output / "raw_generic", output / "raw_road"
    common = ["--train-real", resolved(args.train_real), "--test-real", resolved(args.test_real)]
    for path in args.synthetic:
        common += ["--synthetic", resolved(path)]
    common += ["--names", *args.names, "--bbox", *map(str, args.bbox)]
    run("evaluate_generic_mobility_tasks.py", [*common, "--seed", str(args.seed)], generic_dir)
    road_common = ["--train-real", resolved(args.train_real), "--test-real", resolved(args.test_real)]
    for path in (args.road_synthetic or args.synthetic):
        road_common += ["--synthetic", resolved(path)]
    road_common += ["--names", *args.names, "--bbox", *map(str, args.bbox)]
    run("evaluate_road_mining_tasks.py", [*road_common, "--osm-cache", resolved(args.osm_cache)], road_dir)
    aggregate(generic_dir, road_dir, output, input_binding)


if __name__ == "__main__":
    main()
