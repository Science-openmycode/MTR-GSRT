"""Render the two-city native road-choice comparison used as Figure 4.

The inputs are evaluation ledgers from evaluate_native_road_choice.py.  No
trajectory is map matched, routed, or regenerated here.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("MTR-GSRT", "SPRT", "PrivTrace", "DPTraj-PM", "DPStd")
COLORS = {
    "MTR-GSRT": "#C44E52",
    "SPRT": "#4C72B0",
    "PrivTrace": "#55A868",
    "DPTraj-PM": "#8172B3",
    "DPStd": "#CCB974",
}
INPUTS = {
    "Beijing": ROOT / "results/paper_refresh_20260727/native_road_choice_beijing_v1/native_road_choice_metrics.json",
    "Porto": ROOT / "results/paper_refresh_20260727/native_road_choice_porto_v1/native_road_choice_metrics.json",
}


def penalized_nll(result: dict[str, float]) -> float:
    coverage = float(result["native_directed_edge_evidence_yield"])
    if coverage <= 0.0:
        return math.inf
    return float(result["next_road_nll"]) - math.log(coverage)


def main() -> None:
    output = ROOT / "results/paper_refresh_20260727/native_road_choice_multicity_v1"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for city, source in INPUTS.items():
        payload = json.loads(source.read_text(encoding="utf-8"))
        for method in METHODS:
            result = payload["results"][method]
            rows.append({
                "city": city,
                "method": method,
                "evidence_yield": float(result["native_directed_edge_evidence_yield"]),
                "unusable_rate": 1.0 - float(result["native_directed_edge_evidence_yield"]),
                "coverage_adjusted_next_road_accuracy": float(result["coverage_adjusted_next_road_accuracy"]),
                "penalized_next_road_nll": penalized_nll(result),
            })
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({
        "classification": "TWO_CITY_NATIVE_ROAD_CHOICE_NO_SYNTHETIC_FMM_COMPLETION",
        "inputs": {city: str(path) for city, path in INPUTS.items()},
        "nll_definition": "next_road_nll - log(native_directed_edge_evidence_yield); infinity at zero evidence",
        "rows": rows,
    }, indent=2), encoding="utf-8")

    cities = tuple(INPUTS)
    x = np.arange(len(cities))
    width = 0.15
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.25), constrained_layout=True)
    panels = (
        ("evidence_yield", "Native road evidence yield", "higher is better"),
        ("coverage_adjusted_next_road_accuracy", "Next-road accuracy", "higher is better"),
        ("penalized_next_road_nll", "Next-road NLL", "lower is better"),
    )
    finite_nll = [row["penalized_next_road_nll"] for row in rows if math.isfinite(row["penalized_next_road_nll"])]
    infinity_cap = max(finite_nll) + 0.9
    for axis, (key, title, ylabel) in zip(axes, panels):
        for index, method in enumerate(METHODS):
            selected = [next(row for row in rows if row["city"] == city and row["method"] == method) for city in cities]
            values = [row[key] for row in selected]
            capped = [infinity_cap if not math.isfinite(value) else value for value in values]
            bars = axis.bar(x + (index - 2) * width, capped, width, color=COLORS[method], label=method)
            if key == "penalized_next_road_nll":
                for bar, value in zip(bars, values):
                    if not math.isfinite(value):
                        axis.text(bar.get_x() + bar.get_width() / 2, infinity_cap + 0.08, r"$\infty$", ha="center", va="bottom", fontsize=9)
        axis.set_title(title, fontsize=9)
        axis.set_xticks(x, cities)
        axis.set_ylabel(ylabel, fontsize=8)
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
        axis.tick_params(labelsize=8)
    axes[0].set_ylim(0, 1.08)
    axes[1].set_ylim(0, 0.36)
    axes[2].set_ylim(0, infinity_cap + 0.5)
    axes[2].legend(fontsize=7, frameon=False, loc="upper right")
    for extension in ("pdf", "png"):
        figure.savefig(output / f"cross_city_native_road_choice.{extension}", dpi=260, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
