from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "experiment_results" / "frozen"
OUT = ROOT / "experiment_results" / "regenerated_figures"
OUT.mkdir(parents=True, exist_ok=True)
PUBLISHED = ROOT / "experiment_results" / "published_figures"
CONFIG = json.loads((ROOT / "config" / "package.json").read_text(encoding="utf-8"))
PALETTE = ["#244A73", "#4C78A8", "#3A8D8F", "#59A14F", "#F28E2B", "#E15759"]
DATA_ROOT: Path | None = None


def read_table(experiment: str, fallback: str, **kwargs) -> pd.DataFrame:
    prepared = DATA_ROOT / experiment / "results.csv" if DATA_ROOT is not None else None
    if DATA_ROOT is not None and (prepared is None or not prepared.is_file()):
        raise FileNotFoundError(
            f"executed result is missing for {experiment}: {prepared}; run the experiment before plotting"
        )
    source = prepared if prepared is not None else FROZEN / fallback
    table = pd.read_csv(source, **kwargs)
    return table.drop(columns=["experiment"], errors="ignore")


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def annotated_heatmap(frame: pd.DataFrame, title: str, name: str) -> None:
    values = frame.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.2, 0.82 * len(frame.columns)), max(3.5, 0.48 * len(frame))))
    image = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(frame.columns)), frame.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(frame.index)), frame.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(j, i, "—" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(value) and value > .62 else "#1F2933")
    ax.set_title(title, weight="bold")
    fig.colorbar(image, ax=ax, label="Metric value (higher is better)")
    save(fig, name)


def plot_profile() -> None:
    table = read_table("profile", "metrics.csv")
    table = table[table["status"].eq("VALID")].copy()
    frame = table.pivot_table(index="algorithm", columns="metric", values="value", aggfunc="first")
    preferred = ["RoadYield", "DirValid", "WitnessValid", "DemandFid", "GridSim",
                 "LengthSim", "NextRoadAcc", "RouteMRR", "RouteBest5F1", "EdgeF1",
                 "BTF", "RC-CPC", "RC_CPC", "EdgeCPC", "TurnCPC", "FamilyCPC"]
    columns = [column for column in preferred if column in frame.columns]
    annotated_heatmap(frame[columns], "Frozen full-metric profile", "01_overall_profile")


def plot_framework() -> None:
    table = read_table("framework", "shared_mtr_framework_lift.csv")
    metrics = ["RoadYield", "BTF", "FamilyCPC"]
    labels = [f"{row.Measurement}\n{row.Stage}" for row in table.itertuples()]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.4, 4.8))
    width = .24
    for index, metric in enumerate(metrics):
        ax.bar(x + (index - 1) * width, table[metric], width, label=metric, color=PALETTE[index + 1])
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Measurement–routing lift", weight="bold")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=.2)
    save(fig, "02_mtr_framework_lift")


def plot_privacy() -> None:
    table = read_table("privacy", "shared_mtr_privacy_attacks.csv").set_index("method")
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    x = np.arange(len(table.index)); width = .18
    for index, metric in enumerate(table.columns):
        ax.bar(x + (index - 1.5) * width, table[metric], width, label=metric, color=PALETTE[index + 1])
    ax.axhline(.5, color="#6B7280", linestyle="--", linewidth=1, label="random AUC reference")
    ax.set_xticks(x, table.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Attack score")
    ax.set_title("Published-interface privacy audits", weight="bold")
    ax.legend(ncol=3, fontsize=8)
    save(fig, "03_privacy_attacks")


def plot_mr() -> None:
    table = read_table("mr", "mr_matrix.csv")
    frame = table.pivot_table(index="M", columns="R", values="BTF", aggfunc="first")
    annotated_heatmap(frame, "Measurement × routing matrix (BTF)", "04_mr_matrix")


def plot_ablation() -> None:
    table = read_table("ablation", "ablation.csv")
    first = table.columns[0]
    if str(first).startswith("Unnamed") or first in {"metric", "component", "ablation"}:
        table = table.set_index(first)
    if "metric" in table.index.names or table.index.name == "metric":
        table = table.T
    numeric = table.apply(pd.to_numeric, errors="coerce")
    annotated_heatmap(numeric, "Algorithm-level ablation", "05_ablation")


def copy_published(stem: str, output_stem: str | None = None) -> None:
    output_stem = output_stem or stem
    for suffix in (".png", ".pdf"):
        source = PUBLISHED / f"{stem}{suffix}"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, OUT / f"{output_stem}{suffix}")


def plot_evidence_chain() -> None:
    copy_published("00_ordered_evidence_chain")


def plot_tstr() -> None:
    table = read_table("tstr", "strict_tstr.csv")
    if {"measurement", "router", "metric", "retention"}.issubset(table.columns):
        metrics = [
            "Next-cell Hit@1", "Next-cell MRR", "Destination Hit@5",
            "Road continuation Hit@1", "Road continuation MRR", "Route retrieval NDCG@5",
        ]
        methods = list(dict.fromkeys(table["measurement"].astype(str)))
        routers = [name for name in ("Native", "FMM", "STMatch")
                   if name in set(table["router"].astype(str))]
        fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True)
        cmap = plt.get_cmap("YlGnBu")
        for ax, metric in zip(axes.flat, metrics):
            part = table[table["metric"].eq(metric)]
            frame = (part.pivot(index="measurement", columns="router", values="retention")
                     .reindex(index=methods, columns=routers))
            values = frame.to_numpy(dtype=float)
            image = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax.set_xticks(np.arange(len(routers)), routers)
            ax.set_yticks(np.arange(len(methods)), methods)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    value = values[i, j]
                    ax.text(j, i, "—" if not np.isfinite(value) else f"{value:.2f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if np.isfinite(value) and value > .62 else "#1F2933")
            ax.set_title(metric, weight="bold")
            ax.set_xlabel("Public reconstruction R")
            ax.set_ylabel("Private measurement / release M")
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.78, pad=.02)
        colorbar.set_label("Task utility retained relative to Real-train")
        fig.suptitle("Strict train-only TSTR: measurement × public reconstruction", weight="bold")
        save(fig, "06_strict_tstr")
        return
    if {"metric", "value"}.issubset(table.columns):
        labels = table["metric"].astype(str).tolist()
        values = table["value"].astype(float).to_numpy()
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
        bars = ax.bar(np.arange(len(labels)), values, color=PALETTE[1:1 + len(labels)])
        ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Task score")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + .02, f"{value:.3f}", ha="center")
        ax.set_title("Strict train-only TSTR", weight="bold")
        ax.grid(axis="y", alpha=.2)
        save(fig, "06_strict_tstr")
        return
    frame = table.set_index(table.columns[0]).apply(pd.to_numeric, errors="coerce")
    annotated_heatmap(frame, "Strict train-only TSTR utility retention", "06_strict_tstr")


def plot_structure() -> None:
    table = read_table("structure", "structure.csv")
    if {"Structure", "Retention"}.issubset(table.columns):
        fig, ax = plt.subplots(figsize=(9.0, 4.4))
        ax.bar(np.arange(len(table)), table["Retention"].astype(float), color=PALETTE[1])
        ax.set_xticks(np.arange(len(table)), table["Structure"], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Retention relative to Real")
        ax.set_title("Trajectory-structure retention", weight="bold")
        ax.grid(axis="y", alpha=.2)
        save(fig, "07_trajectory_structure_diagnostics")
        return
    if {"dataset", "UsableRatio", "RevisitRatio", "ClosedTripRatio", "MeanDirectness"}.issubset(table.columns):
        frame = table.set_index("dataset")[["UsableRatio", "RevisitRatio", "ClosedTripRatio", "MeanDirectness"]]
        annotated_heatmap(frame, "Executed trajectory-structure diagnostics", "07_trajectory_structure_diagnostics")
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(table))
    ax.bar(x - .18, table["loaded_count"], .36, label="Loaded", color=PALETTE[1])
    ax.bar(x + .18, table["plotted_count"], .36, label="Plotted", color=PALETTE[2])
    ax.set_xticks(x, table["dataset"], rotation=25, ha="right")
    ax.set_ylabel("Trajectory count")
    ax.set_title("Beijing full-corpus trajectory visualization inputs", weight="bold")
    ax.legend()
    save(fig, "07_beijing_visual_comparison")


def main() -> None:
    global DATA_ROOT
    parser = argparse.ArgumentParser(description="Regenerate one experiment figure or the complete sequence.")
    parser.add_argument(
        "--figure",
        choices=("evidence", "framework", "privacy", "profile", "mr", "ablation", "tstr", "structure", "all"),
        default="all",
    )
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    if args.data_root is not None:
        DATA_ROOT = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    jobs = {
        "evidence": plot_evidence_chain,
        "framework": plot_framework,
        "privacy": plot_privacy,
        "profile": plot_profile,
        "mr": plot_mr,
        "ablation": plot_ablation,
        "tstr": plot_tstr,
        "structure": plot_structure,
    }
    selected = list(jobs) if args.figure == "all" else [args.figure]
    for name in selected:
        jobs[name]()
    print(f"Wrote {', '.join(selected)} figure(s) to {OUT}")


if __name__ == "__main__":
    main()
