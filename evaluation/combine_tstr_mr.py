from __future__ import annotations

import argparse
import csv
import json
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
        return {row["Pipeline"]: row for row in csv.DictReader(handle)}


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
    args = parser.parse_args()
    tables = {"Native": read(args.native), "FMM": read(args.fmm), "STMatch": read(args.stmatch)}
    methods = [name for name in tables["Native"] if name != "Real-train"]
    rows: list[dict[str, object]] = []
    for router, table in tables.items():
        indexed = {base_name(name, router): row for name, row in table.items() if name != "Real-train"}
        missing = sorted(set(methods) - set(indexed))
        if missing:
            raise RuntimeError(f"{router} results missing methods: {missing}")
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
        "protocol": "strict_train_only_tstr_measurement_by_public_reconstruction_v3",
        "main_metrics": MAIN_METRICS,
        "all_metrics": ALL_METRICS,
        "methods": methods,
        "routers": list(tables),
        "cell_semantics": "coverage-adjusted task utility divided by Real-train utility",
        "main_matrix_cells": len(main_rows),
        "all_matrix_cells": len(rows),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
