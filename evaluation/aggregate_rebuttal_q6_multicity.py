"""Aggregate the five-seed Porto/SF rebuttal releases and generation costs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SEEDS = range(20260719, 20260724)
METRICS = {
    "Grid": "grid_density_jsd",
    "Trip": "trip_error",
    "Len": "path_length_jsd",
    "OD": "OD_jsd",
    "RoadSeg": "road_segment_jsd",
    "NextRegionAcc": "B2_next_region_accuracy",
    "NextRegionNLL": "B2_next_region_nll",
    "RouteMRR": "B2_grid_route_mrr",
    "WitnessValid": "witness_valid",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: list[float]) -> tuple[float, float, float]:
    series = pd.Series(values, dtype=float)
    mean = float(series.mean())
    if len(series) < 2:
        return mean, mean, mean
    # Two-sided 95% Student-t intervals for n=2,...,5.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}[len(series)]
    half = critical * float(series.std(ddof=1)) / math.sqrt(len(series))
    return mean, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for city in ("porto", "sf"):
        parser.add_argument(f"--{city}-generation-root", required=True)
        parser.add_argument(f"--{city}-metrics-root", required=True)
        parser.add_argument(f"--{city}-seed-suffix", default="")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    def rooted(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else ROOT / path).resolve()
    out = rooted(args.out_dir)
    if out.exists():
        parser.error(f"Output directory already exists: {out}")
    sources = {}
    for city in ("porto", "sf"):
        generation = rooted(getattr(args, f"{city}_generation_root"))
        metrics = rooted(getattr(args, f"{city}_metrics_root"))
        suffix = getattr(args, f"{city}_seed_suffix")
        for seed in SEEDS:
            sources[(city, seed)] = (
                generation / f"seed_{seed}{suffix}" / "protocol.json",
                metrics / f"seed_{seed}" / "metrics.json",
            )
    missing = [f"{city}/{seed}/{path.name}" for (city, seed), pair in sources.items()
               for path in pair if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing multicity seed inputs: {missing}")
    rows: list[dict[str, object]] = []
    for city in ("porto", "sf"):
        for seed in SEEDS:
            protocol_path, metric_path = sources[(city, seed)]
            metric_payload = json.loads(metric_path.read_text(encoding="utf-8"))
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            row: dict[str, object] = {"city": city, "seed": seed}
            for label, key in METRICS.items():
                row[label] = metric_payload["metrics"][key]
            row["generation_sec"] = protocol["elapsed_sec"]
            row["fallback_count"] = protocol["decoder"]["fallback_count"]
            row["output_count"] = protocol["output_count"]
            rows.append(row)

    detailed = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=False)
    detailed.to_csv(out / "q6_multicity_five_seed_detailed.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for city, frame in detailed.groupby("city"):
        summary: dict[str, object] = {"city": city, "n_seeds": len(frame)}
        for column in list(METRICS) + ["generation_sec", "fallback_count"]:
            mean, low, high = summarize(frame[column].astype(float).tolist())
            summary[f"{column}_mean"] = mean
            summary[f"{column}_ci_low"] = low
            summary[f"{column}_ci_high"] = high
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(out / "q6_multicity_five_seed_summary.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "classification": "RESEARCH_MULTICITY_FIVE_SEED_AGGREGATE_NO_SYNTHESIS",
        "seeds": list(SEEDS),
        "input_hashes": {
            f"{city}/seed_{seed}/{path.name}": sha256_file(path)
            for (city, seed), pair in sources.items() for path in pair
        },
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "output_hashes": {
            name: sha256_file(out / name)
            for name in ("q6_multicity_five_seed_detailed.csv", "q6_multicity_five_seed_summary.csv")
        },
    }, indent=2) + "\n", encoding="utf-8")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
