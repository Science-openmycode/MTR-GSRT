"""Recompute Figure-4 native destination-conditioned next-region utility.

Loads only frozen real corpora and already generated synthetic releases.  It
does not call FMM, routing, training of a synthesizer, or trajectory synthesis.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis_scripts.downstream_full_protocol as downstream
from public_utils import load_trajectories


CITIES = {
    "Beijing": {
        "real": "outputs/kdd_revised/split/real_full_frozen.pkl",
        "real_count": 17123,
        "bbox": (39.75, 40.15, 116.10, 116.65),
        "paths": {
            "MTR-GSRT": "results/active_hierarchy_release_v1_20260726/multicity/beijing/release/trajectories.pkl",
            "SPRT": "outputs/synthetic_releases/current_best/sprt_native.pkl",
            "PrivTrace": "outputs/synthetic_releases/current_best/privtrace_native.pkl",
            "DPTraj-PM": "outputs/synthetic_releases/current_best/dptrajpm_native.pkl",
            "DPStd": "outputs/synthetic_releases/current_best/dpstd_native.pkl",
        },
    },
    "Porto": {
        "real": "porto",
        "real_count": 20000,
        "bbox": (41.10, 41.20, -8.70, -8.55),
        "paths": {
            "MTR-GSRT": "results/active_hierarchy_release_v1_20260726/multicity/porto/release/trajectories.pkl",
            "SPRT": "results/synthetic_datasets/baselines/porto/sprt_native_eps_7_5_seed_42/sprt-native.pkl",
            "PrivTrace": "results/synthetic_datasets/baselines/porto/privtrace_native_eps_7_5_seed_42/privtrace-native.pkl",
            "DPTraj-PM": "results/synthetic_datasets/baselines/porto/dptrajpm_native_eps_7_5_seed_42/dptrajpm-native.pkl",
            "DPStd": "results/synthetic_datasets/baselines/porto/dpstd_native_eps_7_5_seed_42/dpstd-native.pkl",
        },
    },
    "San Francisco": {
        "real": "data/trajectories/frozen/sf_cabspotting_trips_20k.pkl",
        "real_count": 20000,
        "bbox": (37.60, 37.85, -122.55, -122.30),
        "paths": {
            "MTR-GSRT": "results/active_hierarchy_release_v1_20260726/multicity/sf/release/trajectories.pkl",
            "SPRT": "results/synthetic_datasets/baselines/sf_trip20k/sprt_native_eps_7_5_seed_42/sprt-native.pkl",
            "PrivTrace": "results/synthetic_datasets/baselines/sf_trip20k/privtrace_native_eps_7_5_seed_42/privtrace-native.pkl",
            "DPTraj-PM": "results/synthetic_datasets/baselines/sf_trip20k/dptrajpm_native_eps_7_5_seed_42/dptrajpm-native.pkl",
            "DPStd": "results/synthetic_datasets/baselines/sf_trip20k/dpstd_native_eps_7_5_seed_42/dpstd-native.pkl",
        },
    },
}
METHODS = ("MTR-GSRT", "SPRT", "PrivTrace", "DPTraj-PM", "DPStd")
COLORS = {"MTR-GSRT": "#C44E52", "SPRT": "#4C72B0", "PrivTrace": "#55A868", "DPTraj-PM": "#8172B3", "DPStd": "#CCB974"}


def public_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    output = public_path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, inputs = [], {}
    for city, spec in CITIES.items():
        real = load_trajectories(spec["real"], limit=int(spec["real_count"]))
        real_test = real[int(0.8 * len(real)):]
        for method, source in spec["paths"].items():
            path = public_path(source)
            synthetic = load_trajectories(str(path), limit=None)
            # The inherited evaluator keeps its public grid bounds as module state.
            downstream.BBOX = tuple(spec["bbox"])
            metrics = downstream.destination_conditioned_next_region_metrics(synthetic, real_test)
            rows.append({"city": city, "method": method, "synthetic_count": len(synthetic), "real_test_count": len(real_test), **metrics})
            inputs[f"{city}:{method}"] = {"path": str(path), "sha256": digest(path)}
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "metrics.json").write_text(json.dumps({
        "classification": "NATIVE_RELEASE_MULTICITY_NEXT_REGION_EVALUATION_NO_ROUTING",
        "protocol": {"grid": 16, "real_test_fraction": 0.2, "routing": "none", "synthetic_fmm": False},
        "inputs": inputs, "rows": rows,
    }, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), constrained_layout=True)
    x = np.arange(len(CITIES)); width = 0.15
    for index, method in enumerate(METHODS):
        selected = [next(row for row in rows if row["city"] == city and row["method"] == method) for city in CITIES]
        axes[0].bar(x + (index - 2) * width, [row["B2_next_region_accuracy"] for row in selected], width, label=method, color=COLORS[method])
        axes[1].bar(x + (index - 2) * width, [row["B2_next_region_nll"] for row in selected], width, label=method, color=COLORS[method])
    for ax, title in zip(axes, ("Destination-conditioned Next-region Accuracy", "Destination-conditioned Next-region NLL")):
        ax.set_title(title, fontsize=9); ax.set_xticks(x, CITIES.keys()); ax.grid(axis="y", alpha=.25, linewidth=.5); ax.tick_params(labelsize=8)
    axes[0].set_ylabel("higher is better", fontsize=8); axes[1].set_ylabel("lower is better", fontsize=8)
    axes[1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.savefig(output / "cross_city_navigation_native.pdf", bbox_inches="tight")
    fig.savefig(output / "cross_city_navigation_native.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"rows": len(rows), "out_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
