from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "evaluation" / "pipeline"


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


def aggregate(generic_dir: Path, road_dir: Path, output: Path) -> None:
    generic = read_rows(generic_dir / "generic_mobility_tasks.csv")
    road = read_rows(road_dir / "road_network_mining_tasks.csv")
    generic_by = {row["method"]: row for row in generic}
    road_by = {row["method"]: row for row in road}
    real_g = generic_by["Real-train"]
    real_r = road_by["Real-train"]
    names = [name for name in generic_by if name != "Real-train"]
    fields = [
        ("Next-cell Hit@1", generic_by, real_g, "next_cell_hit_at_1"),
        ("Next-cell MRR", generic_by, real_g, "next_cell_mrr"),
        ("Destination Hit@5", generic_by, real_g, "destination_hit_at_5"),
        ("Road continuation Hit@1", road_by, real_r, "continuation_hit_at_1"),
        ("Route retrieval NDCG@5", road_by, real_r, "retrieval_ndcg_at_5"),
    ]
    rows = [{"Pipeline": "Real-train", **{label: 1.0 for label, *_ in fields}}]
    for name in names:
        row = {"Pipeline": name}
        for label, source, reference, key in fields:
            if name not in source:
                raise RuntimeError(f"{name!r} missing from {key} evaluator output")
            row[label] = finite_ratio(source[name][key], reference[key])
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    result = output / "results.csv"
    with result.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({
        "protocol": "strict_train_only_tstr_executed_v1",
        "raw_generic_results": str((generic_dir / "generic_mobility_tasks.json").resolve()),
        "raw_road_results": str((road_dir / "road_network_mining_tasks.json").resolve()),
        "result": str(result.resolve()),
        "normalization": "each task score divided by the Real-train score; Real-train=1",
        "pipelines": [row["Pipeline"] for row in rows],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute strict train-only TSTR models and aggregate their held-out scores."
    )
    parser.add_argument("--train-real", required=True)
    parser.add_argument("--test-real", required=True)
    parser.add_argument("--synthetic", action="append", required=True,
                        help="Synthetic training release; repeat once per method.")
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--osm-cache", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.names) != len(args.synthetic):
        parser.error("--names must contain one name for every repeated --synthetic argument")
    def resolved(value: str) -> str:
        path = Path(value)
        return str((path if path.is_absolute() else ROOT / path).resolve())
    output = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    generic_dir, road_dir = output / "raw_generic", output / "raw_road"
    common = ["--train-real", resolved(args.train_real), "--test-real", resolved(args.test_real)]
    for path in args.synthetic:
        common += ["--synthetic", resolved(path)]
    common += ["--names", *args.names, "--bbox", *map(str, args.bbox)]
    run("evaluate_generic_mobility_tasks.py", [*common, "--seed", str(args.seed)], generic_dir)
    run("evaluate_road_mining_tasks.py", [*common, "--osm-cache", resolved(args.osm_cache)], road_dir)
    aggregate(generic_dir, road_dir, output)


if __name__ == "__main__":
    main()
