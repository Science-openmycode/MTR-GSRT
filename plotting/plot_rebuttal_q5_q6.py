"""Create compact rebuttal figures from the audited Q5/Q6 CSV artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT: Path
FIG: Path
ABLATION: Path
COLORS = {
    "Demand": "#4C78A8",
    "Graph-flow": "#59A14F",
    "Portal-Fiber": "#E45756",
}


def save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def q6_figure() -> None:
    query = pd.read_csv(OUT / "q6_query_family_summary.csv")
    utility = pd.read_csv(OUT / "q6_full_release_utility_summary.csv")
    order = ["1/5", "2/5", "7/10", "1", "7/5", "2", "3"]
    position = {value: index for index, value in enumerate(order)}
    query["order"] = query["epsilon"].astype(str).map(position)
    utility["order"] = utility["epsilon"].astype(str).map(position)
    query = query.sort_values("order")
    utility = utility.sort_values("order")
    x = np.arange(len(order))

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), constrained_layout=True)
    for family in COLORS:
        current = query[query["family"] == family]
        axes[0, 0].plot(x, current["mean_block_cpc"], marker="o", lw=2,
                        label=family, color=COLORS[family])
        axes[0, 1].plot(x, current["mean_noise_dominated_mass"], marker="o", lw=2,
                        label=family, color=COLORS[family])
    axes[1, 0].plot(x, utility["Grid_mean"], marker="o", label="Grid JSD")
    axes[1, 0].plot(x, utility["OD_mean"], marker="s", label="OD JSD")
    axes[1, 0].plot(x, utility["RoadSeg_mean"], marker="^", label="Road-segment JSD")
    axes[1, 1].plot(x, utility["RC_NDCG_mean"], marker="o", label="RC-NDCG")
    axes[1, 1].plot(x, utility["NextRoadAcc_mean"], marker="s", label="NextRoadAcc")
    axes[1, 1].plot(x, utility["RouteMRR_mean"], marker="^", label="Route MRR")
    titles = [
        "(a) Query-family CPC", "(b) Noise-dominated mass",
        "(c) Complete-release distribution error", "(d) Route-choice and task utility",
    ]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x, order)
        ax.set_xlabel(r"Total privacy budget $\epsilon$")
        ax.grid(axis="y", alpha=.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel("CPC (higher is better)")
    axes[0, 1].set_ylabel("Mass fraction (lower is better)")
    axes[1, 0].set_ylabel("JSD (lower is better)")
    axes[1, 1].set_ylabel("Score (higher is better)")
    save(fig, "q6_complete_low_epsilon_audit")


def q5_partition_figure() -> None:
    path = OUT / "q5_partition_sensitivity_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path).sort_values(["strategy", "fine_regions"])
    nested = frame[frame["strategy"] == "nested"].copy()
    flat = frame[frame["strategy"] == "flat"].copy()
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 10.0), constrained_layout=True)

    def metric(ax, column, label, color):
        for current, style, suffix in ((nested, "-", "nested"), (flat, "--", "flat")):
            if current.empty:
                continue
            x = current["fine_regions"].to_numpy(dtype=float)
            y = current[f"{column}_mean"].to_numpy(dtype=float)
            ax.plot(x, y, style, marker="o", lw=2, color=color,
                    label=f"{label} ({suffix})")
            low = f"{column}_ci95_low"
            high = f"{column}_ci95_high"
            if low in current and high in current:
                ax.fill_between(x, current[low], current[high], color=color, alpha=.10)

    metric(axes[0, 0], "portal_global_cpc", "Global CPC", "#4C78A8")
    metric(axes[0, 0], "portal_context_ndcg", "Context NDCG", "#F28E2B")
    metric(axes[0, 1], "portal_context_top1", "Context Top-1", "#59A14F")
    metric(axes[0, 1], "portal_global_jsd", "Global JSD", "#B07AA1")

    for current, style, suffix in ((nested, "-", "nested"), (flat, "--", "flat")):
        if current.empty:
            continue
        x = current["fine_regions"].to_numpy(dtype=float)
        axes[1, 0].plot(x, current["positive_contexts"] / 1000.0, style, marker="o", lw=2,
                        color="#4C78A8", label=f"Positive contexts ({suffix})")
        axes[1, 0].plot(x, current["portal_atoms"] / 1000.0, style, marker="s", lw=2,
                        color="#E45756", label=f"Portal atoms ({suffix})")
        axes[1, 1].plot(x, current["noise_dominated_mass"], style, marker="o", lw=2,
                        color="#E45756", label=f"Noise-dominated mass ({suffix})")
        axes[1, 1].plot(x, current["context_coverage"], style, marker="s", lw=2,
                        color="#59A14F", label=f"Context coverage ({suffix})")

        axes[2, 0].plot(x, current["crossing_fine_cell_fraction"], style,
                        marker="o", lw=2, color="#B07AA1",
                        label=f"Cross-parent cells ({suffix})")
        axes[2, 0].plot(x, current["crossing_node_fraction"], style,
                        marker="s", lw=2, color="#F28E2B",
                        label=f"Nodes in cross-parent cells ({suffix})")
        axes[2, 1].plot(x, current["mean_signal_noise_ratio"], style,
                        marker="o", lw=2, color="#4C78A8",
                        label=f"Mean signal/noise ({suffix})")
        axes[2, 1].plot(x, current["atoms_per_positive_context"], style,
                        marker="s", lw=2, color="#59A14F",
                        label=f"Atoms/context ({suffix})")

    titles = [
        "(a) Global and context fidelity",
        "(b) Local ranking and distribution error",
        "(c) Support growth",
        "(d) Coverage-noise trade-off",
        "(e) Coarse-fine hierarchy consistency",
        "(f) Signal dilution within the support",
    ]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(32, 1050)
        ax.set_xticks([48, 128, 256, 384, 512, 768, 1024])
        ax.set_xlabel("Number of fine regions K")
        ax.axvline(384, color="#777777", lw=1, ls=":", alpha=.8)
        ax.grid(axis="y", alpha=.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[1, 0].set_ylabel("Count (thousands)")
    axes[1, 1].set_ylim(0, 1)
    axes[2, 0].set_ylim(-.02, .82)
    axes[2, 0].set_ylabel("Fraction")
    axes[2, 1].axhline(1.0, color="#777777", lw=1, ls=":", alpha=.8)
    axes[2, 1].set_ylabel("Ratio / count")
    save(fig, "q5_partition_granularity_sensitivity")


def q5_component_figure() -> None:
    path = ABLATION / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))["results"]
    order = [
        "Full",
        "no-crossing-flow-measurement",
        "no-local-information-projection",
        "no-crossing-road-sampling",
    ]
    labels = ["Full", "No flow", "No projection", "MAP portal"]
    rows = [{"arm": name, **payload[name]} for name in order]
    frame = pd.DataFrame(rows)
    metric_names = {
        "road_choice_cpc": "RC-CPC", "road_choice_ndcg": "RC-NDCG",
        "next_road_accuracy": "NextRoadAcc",
    }
    usable = [key for key in metric_names if key in frame.columns]
    if not usable:
        return
    x = np.arange(len(frame))
    width = .75 / len(usable)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [3.2, 1]})
    ax = axes[0]
    for index, metric in enumerate(usable):
        ax.bar(x + (index - (len(usable)-1)/2) * width, frame[metric], width,
               label=metric_names[metric])
    ax.set_xticks(x, labels, rotation=16, ha="right")
    ax.set_ylabel("Score (higher is better)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, ncol=len(usable))
    axes[1].bar(x, frame["next_road_nll"], color="#B07AA1")
    axes[1].set_xticks(x, labels, rotation=16, ha="right")
    axes[1].set_ylabel("NextRoadNLL (lower is better)")
    axes[1].set_ylim(1.8, 2.15)
    axes[1].grid(axis="y", alpha=.25)
    axes[0].set_title("(a) Route-choice and next-road scores", fontweight="bold")
    axes[1].set_title("(b) Next-road NLL", fontweight="bold")
    save(fig, "q5_portal_component_ablation")


def main() -> None:
    global OUT, FIG, ABLATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("q5-partition", "q5-ablation", "q6", "all"), required=True)
    parser.add_argument("--input-dir", required=True, help="Directory containing partition and/or Q6 CSVs")
    parser.add_argument("--ablation-dir", help="Directory containing the Q5 ablation manifest")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    def rooted(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else ROOT / path).resolve()
    OUT, FIG = rooted(args.input_dir), rooted(args.out_dir)
    ABLATION = rooted(args.ablation_dir) if args.ablation_dir else OUT / "q5_portal_component_ablation"
    if not OUT.is_dir():
        raise FileNotFoundError(OUT)
    if FIG.exists():
        parser.error(f"Output figure directory must be new: {FIG}")
    selected = ("q5-partition", "q5-ablation", "q6") if args.stage == "all" else (args.stage,)
    inputs = {
        "q5-partition": OUT / "q5_partition_sensitivity_summary.csv",
        "q5-ablation": ABLATION / "manifest.json",
        "q6": OUT / "q6_query_family_summary.csv",
    }
    for stage in selected:
        if not inputs[stage].is_file():
            raise FileNotFoundError(inputs[stage])
    for stage in selected:
        {"q5-partition": q5_partition_figure,
         "q5-ablation": q5_component_figure,
         "q6": q6_figure}[stage]()
    (FIG / "figure_manifest.json").write_text(json.dumps({
        "stages": selected,
        "inputs": {stage: {"path": str(inputs[stage]),
                           "sha256": hashlib.sha256(inputs[stage].read_bytes()).hexdigest()}
                   for stage in selected},
        "outputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in FIG.iterdir() if path.is_file()},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
