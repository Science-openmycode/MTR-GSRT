"""Aggregate the 7x5 MTR road-choice sweep without rerunning synthesis."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (_candidate / "configs" / "datasets.json").is_file():
            sys.path.insert(0, str(_candidate))
            break

from generation.common.runtime import public_path, sha256_file, write_json  # noqa: E402

PATTERN = re.compile(r"^eps_(\d+)_(\d+)_seed_(\d+)$")
METRICS = (
    "road_choice_cpc", "road_choice_ndcg",
    "next_road_accuracy", "next_road_nll",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input")
    sources.add_argument("--input-dir")
    parser.add_argument("--config", default="configs/mtr_epsilon_sweep_beijing.json")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    config_path = public_path(args.config)
    out_dir = public_path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    input_files: list[Path]
    if args.input:
        input_files = [public_path(args.input)]
    else:
        input_dir = public_path(args.input_dir)
        input_files = sorted(input_dir.glob("*/metrics.json"))
        if not input_files:
            raise FileNotFoundError(f"no per-run metrics found under {input_dir}")
    combined_results: dict[str, dict] = {}
    for source in input_files:
        payload = json.loads(source.read_text(encoding="utf-8"))
        for name, values in payload["results"].items():
            if name in combined_results:
                raise ValueError(f"duplicate sweep method across inputs: {name}")
            if len(payload["results"]) != 1 and len(input_files) > 1:
                raise ValueError(f"per-run input must contain one result: {source}")
            if len(input_files) > 1 and source.parent.name != name:
                raise ValueError(
                    f"run directory {source.parent.name!r} does not match method {name!r}"
                )
            combined_results[name] = values
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        (epsilon, int(seed))
        for epsilon in config["epsilons"]
        for seed in config["noise_seeds"]
    }
    rows: list[dict] = []
    found: set[tuple[str, int]] = set()
    for name, values in combined_results.items():
        match = PATTERN.match(name)
        if not match:
            raise ValueError(f"unexpected sweep method name: {name}")
        numerator, denominator, seed_text = match.groups()
        epsilon_text = (
            numerator if denominator == "1" else f"{numerator}/{denominator}"
        )
        key = (epsilon_text, int(seed_text))
        if key in found:
            raise ValueError(f"duplicate sweep cell: {key}")
        found.add(key)
        for metric in METRICS:
            value = float(values[metric])
            if not np.isfinite(value):
                raise ValueError(f"nonfinite {metric} at {key}")
            rows.append({
                "epsilon": epsilon_text,
                "epsilon_float": float(numerator) / float(denominator),
                "seed": int(seed_text),
                "metric": metric,
                "value": value,
            })
    if found != expected:
        raise ValueError(f"7x5 matrix mismatch; missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    draws = int(config["bootstrap_draws"])
    summary: list[dict] = []
    for epsilon in config["epsilons"]:
        epsilon_float = float(Fraction(epsilon))
        for metric in METRICS:
            values = np.asarray([
                row["value"] for row in rows
                if row["epsilon"] == epsilon and row["metric"] == metric
            ], dtype=float)
            means = np.asarray([
                np.mean(values[rng.integers(0, len(values), len(values))])
                for _ in range(draws)
            ])
            low, high = np.quantile(means, (0.025, 0.975))
            summary.append({
                "epsilon": epsilon,
                "epsilon_float": epsilon_float,
                "metric": metric,
                "mean": float(np.mean(values)),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "seed_count": len(values),
            })
    out_dir.mkdir(parents=True)
    for filename, data, fields in (
        ("metrics_long.csv", rows, ("epsilon","epsilon_float","seed","metric","value")),
        ("metrics_summary.csv", summary, ("epsilon","epsilon_float","metric","mean","ci95_low","ci95_high","seed_count")),
    ):
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    manifest = {
        "schema_version": 1,
        "classification": "PRIVACY_BUDGET_ROAD_CHOICE_AGGREGATE_NO_SYNTHESIS",
        "inputs": {
            str(source): sha256_file(source)
            for source in input_files
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "aggregator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "matrix": {"epsilons": 7, "seeds": 5, "cells": 35},
        "outputs": {
            name: sha256_file(out_dir / name)
            for name in ("metrics_long.csv", "metrics_summary.csv")
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
