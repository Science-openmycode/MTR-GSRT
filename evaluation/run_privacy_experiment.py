from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute release-only membership attacks against one synthetic trajectory corpus."
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--members", required=True)
    parser.add_argument("--nonmembers", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--bbox", nargs=4, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    output = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    population = output / "population_linkage.json"
    gda_root = output / "gda_mia"
    commands = [
        [sys.executable, str(ROOT / "evaluation" / "pipeline" / "evaluate_population_linkage.py"),
         "--members", str(Path(args.members).resolve()), "--nonmembers", str(Path(args.nonmembers).resolve()),
         "--release", str(Path(args.release).resolve()), "--out", str(population),
         "--landmarks", "16", "--seed", str(args.seed)],
        [sys.executable, str(ROOT / "evaluation" / "pipeline" / "evaluate_gda_mia.py"),
         "--members", str(Path(args.members).resolve()), "--nonmembers", str(Path(args.nonmembers).resolve()),
         "--reference", str(Path(args.reference).resolve()), "--release", args.method,
         str(Path(args.release).resolve()), "--bbox", *args.bbox, "--out", str(gda_root),
         "--landmarks", "8", "--seed", str(args.seed)],
    ]
    for command in commands:
        print("RUN:", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    population_result = json.loads(population.read_text(encoding="utf-8"))
    gda_result = json.loads(gda_root.with_suffix(".json").read_text(encoding="utf-8"))
    gda_row = gda_result["rows"][0]
    rows = [{"method": args.method,
             "Population-linkage AUC": population_result["auc"],
             "GDA-MIA AUC": gda_row.get("auc", gda_row.get("aucroc"))}]
    result = output / "results.csv"
    with result.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({
        "protocol": "executed_release_only_privacy_attacks_v1", "commands": commands,
        "result": str(result.resolve())
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote executed attack scores to {result}")


if __name__ == "__main__":
    main()
