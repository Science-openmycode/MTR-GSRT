from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ALL_METRICS = [
    "Next-cell Hit@1",
    "Next-cell MRR",
    "Destination Hit@5",
    "Destination MRR",
    "Road continuation Hit@1",
    "Road continuation MRR",
    "Route retrieval NDCG@5",
]

MAIN_METRICS = [
    "Next-cell Hit@1",
    "Next-cell MRR",
    "Destination Hit@5",
    "Road continuation Hit@1",
    "Road continuation MRR",
    "Route retrieval NDCG@5",
]


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["Pipeline"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate pipeline names in {path}")
    for row in rows:
        for metric in ALL_METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid retention {metric} in {path}: {value}")
    if "Real-train" not in names:
        raise ValueError(f"Missing Real-train reference in {path}")
    if any(float(row[metric]) != 1 for row in rows if row["Pipeline"] == "Real-train" for metric in ALL_METRICS):
        raise ValueError(f"Real-train reference must equal one in {path}")
    return {row["Pipeline"]: row for row in rows}


def validate_bindings(paths: dict[str, Path]) -> dict:
    bindings = {}
    for router, path in paths.items():
        manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8-sig"))
        if manifest.get("protocol") not in {"strict_train_only_tstr_executed_v1", "train_test_bound_task_scoring_v2"}:
            raise ValueError(f"{router}: retrospective or unknown TSTR protocol")
        binding = manifest.get("split_binding")
        if not binding or not binding.get("train_sha256") or not binding.get("test_sha256"):
            raise ValueError(f"{router}: missing train/test hash binding; recompute with run-tstr")
        if binding["train_sha256"] == binding["test_sha256"]:
            raise ValueError(f"{router}: identical train/test inputs")
        bindings[router] = binding
    if any(not binding.get("base_releases") for binding in bindings.values()):
        raise ValueError("Missing base release hashes; rerun task scoring with routed-view manifests")
    for key in ("train_sha256", "test_sha256", "bbox", "seed", "osm_sha256", "base_releases", "synthesis_lineage", "disjoint_split"):
        values = [json.dumps(binding.get(key), sort_keys=True) for binding in bindings.values()]
        if len(set(values)) != 1:
            raise ValueError(f"Router tables use different {key}")
    return bindings


def base_name(name: str, router: str) -> str:
    suffix = f" x {router}"
    unicode_suffix = f" × {router}"
    if name.endswith(unicode_suffix):
        return name[: -len(unicode_suffix)]
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine three executed TSTR result tables into an M-by-R matrix.")
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--fmm", type=Path, required=True)
    parser.add_argument("--stmatch", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-synthesis-lineage", action="store_true",
                        help="Only aggregate matrices with executed generation evidence for every base release")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for name in ("native", "fmm", "stmatch", "out_dir"):
        path = getattr(args, name)
        setattr(args, name, (path if path.is_absolute() else root / path).resolve())
    paths = {"Native": args.native, "FMM": args.fmm, "STMatch": args.stmatch}
    bindings = validate_bindings(paths)
    lineage_verified = all(bool(b.get("synthesis_lineage")) and
                           set(b["synthesis_lineage"]) == set(b["base_releases"])
                           for b in bindings.values())
    if args.require_synthesis_lineage and (not lineage_verified or not all(b.get("disjoint_split") for b in bindings.values())):
        parser.error("Missing executed generation lineage; cannot certify a strict synthesis matrix")
    tables = {router: read(path) for router, path in paths.items()}
    methods = [name for name in tables["Native"] if name != "Real-train"]
    rows: list[dict[str, object]] = []
    for router, table in tables.items():
        indexed = {base_name(name, router): row for name, row in table.items() if name != "Real-train"}
        if len(indexed) != len(table) - 1:
            raise ValueError(f"{router}: duplicate method after router suffix normalization")
        missing = sorted(set(methods) - set(indexed))
        if missing:
            raise RuntimeError(f"{router} results missing methods: {missing}")
        if set(indexed) != set(methods):
            raise ValueError(f"{router}: unexpected methods {sorted(set(indexed) - set(methods))}")
        for method in methods:
            for metric in ALL_METRICS:
                rows.append({
                    "measurement": method,
                    "router": router,
                    "metric": metric,
                    "retention": float(indexed[method][metric]),
                })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main_rows = [row for row in rows if row["metric"] in MAIN_METRICS]
    with (args.out_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["measurement", "router", "metric", "retention"])
        writer.writeheader(); writer.writerows(main_rows)
    with (args.out_dir / "results_all.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["measurement", "router", "metric", "retention"])
        writer.writeheader(); writer.writerows(rows)
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "protocol": "train_test_bound_measurement_by_reconstruction_matrix_v4",
        "main_metrics": MAIN_METRICS,
        "all_metrics": ALL_METRICS,
        "methods": methods,
        "routers": list(tables),
        "cell_semantics": "coverage-adjusted task utility divided by Real-train utility",
        "main_matrix_cells": len(main_rows),
        "all_matrix_cells": len(rows),
        "split_bindings": bindings,
        "synthesis_lineage_verified": lineage_verified,
        "disjoint_split_verified": all(bool(b.get("disjoint_split")) for b in bindings.values()),
        "inputs": {router: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for router, path in paths.items()},
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
