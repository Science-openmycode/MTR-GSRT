"""Create publication-style summary figures for ARA-Codex results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ARA_codex" / "results"
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_means(name):
    with (RESULTS / name).open("r", encoding="utf-8") as f:
        return json.load(f)["means"]


def metric_panel(ax, means, methods, labels, metric, title, lower_better=True):
    vals = [means[m][metric] for m in methods]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"][: len(vals)]
    ax.bar(np.arange(len(vals)), vals, color=colors, width=0.72)
    ax.set_xticks(np.arange(len(vals)), labels, rotation=25, ha="right")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ylabel = metric + (" (lower is better)" if lower_better else " (higher is better)")
    ax.set_ylabel(ylabel)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)


def plot_full_k64():
    means = load_means("road_corridor_local_markov_experiment_geolife_full_k64_key.json")
    methods = ["dpOD-template-control", "dpOD-clippedShape-template-control", "roadLocalMarkov-strict"]
    labels = ["Template", "Clipped shape", "ARA strict"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    panels = [
        ("js", "Grid distribution JS", True),
        ("range_are", "Range ARE", True),
        ("step_mean_km", "Mean step distance", True),
        ("turn_jsd", "Turn distribution JS", True),
        ("sinuosity_median", "Median sinuosity", True),
        ("route_hold", "Route hold rate", True),
    ]
    for ax, (metric, title, lower) in zip(axes.flat, panels):
        metric_panel(ax, means, methods, labels, metric, title, lower)
    fig.suptitle("Full GeoLife K=64: Road-Local Markov versus Template Controls", fontsize=13, fontweight="bold")
    out = FIG / "paper_full_k64_core_metrics.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_corridor_budget():
    means = load_means("dp_corridor_potential_experiment_medium_k64_5seed_budget_matched.json")
    methods = [
        "roadLocalMarkov-strict",
        "roadLocalMarkov-strict-budgetMatched",
        "dpRoadCellPotential-gentle",
        "dpRoadCellPotential-soft",
    ]
    labels = ["Strict", "Strict eps=1", "Cell gentle", "Cell soft"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    panels = [
        ("js", "Grid distribution JS", True),
        ("range_are", "Range ARE", True),
        ("step_jsd", "Step distribution JS", True),
        ("turn_jsd", "Turn distribution JS", True),
        ("sinuosity_median", "Median sinuosity", True),
        ("reid", "ReID metric", True),
    ]
    for ax, (metric, title, lower) in zip(axes.flat, panels):
        metric_panel(ax, means, methods, labels, metric, title, lower)
    fig.suptitle("Budget-Matched Corridor Study: Signal and Failure Modes", fontsize=13, fontweight="bold")
    out = FIG / "paper_corridor_budget_matched_metrics.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_mechanism_ladder():
    rows = [
        ("Template", load_means("road_corridor_local_markov_experiment_geolife_full_k64_key.json")["dpOD-template-control"]),
        ("ARA strict", load_means("road_corridor_local_markov_experiment_geolife_full_k64_key.json")["roadLocalMarkov-strict"]),
        ("Transition bridge", load_means("dp_anchor_transition_experiment_geolife_full_k64_key.json")["dpAnchorTransition-bridge"]),
        ("Cell potential", load_means("dp_corridor_potential_experiment_geolife_full_k64_key.json")["dpRoadCellPotential-gentle"]),
    ]
    metrics = ["js", "range_are", "step_jsd", "turn_jsd", "sinuosity_median"]
    raw = np.array([[r[1][m] for m in metrics] for r in rows], dtype=float)
    normalized = raw / np.maximum(raw[0], 1e-9)
    fig, ax = plt.subplots(figsize=(10.5, 4.7), constrained_layout=True)
    x = np.arange(len(metrics))
    width = 0.18
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for i, (label, _) in enumerate(rows):
        ax.bar(x + (i - 1.5) * width, normalized[i], width=width, label=label, color=colors[i])
    ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x, ["JS", "Range ARE", "Step JSD", "Turn JSD", "Sinuosity"])
    ax.set_ylabel("Metric normalized by template control")
    ax.set_title("Mechanism Ladder: What Each Added Layer Changes", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    out = FIG / "paper_mechanism_ladder.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    outs = [plot_full_k64(), plot_corridor_budget(), plot_mechanism_ladder()]
    for out in outs:
        print(out)


if __name__ == "__main__":
    main()
