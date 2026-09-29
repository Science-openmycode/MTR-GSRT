"""One shared 5x8 MTR measurement-by-routing comparison for both rebuttals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.metric_suites.road_choice import (  # noqa: E402
    matched_choice_trips, next_road_trip_scores, public_crossing_options,
    road_choice_fidelity,
)
from route_metric_core import load_pickle, normalize_routes, valid_routes  # noqa: E402


METHODS = ("SPRT", "PrivTrace", "DPTraj-PM", "DPStd", "MTR-GSRT")
ROUTERS = ("Original", "Nearest", "Full FMM", "Full STMatch",
           "Family additive", "Family residual", "Family length-OT",
           "Self-carrier reweight")
FAMILY_FILES = ("Family_additive", "Family_residual", "Family_length-OT",
                "Self-carrier_reweight")


def rooted(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_path(method: str, router: str, native: Path, direct: Path, family: Path,
                fmm: Path | None = None, stmatch: Path | None = None) -> Path:
    if router == "Original":
        return (native / method / "Original.pkl" if method == "MTR-GSRT"
                else direct / method / "Original.pkl.gz")
    if router == "Nearest":
        return direct / method / "Nearest.pkl.gz"
    if router == "Full FMM":
        return fmm / f"{method}.pkl.gz" if fmm else native / method / "FMM.pkl.gz"
    if router == "Full STMatch":
        return stmatch / f"{method}.pkl.gz" if stmatch else native / method / "STMatch.pkl.gz"
    return family / method / f"{FAMILY_FILES[ROUTERS.index(router) - 4]}.pkl.gz"


def bound_input(path: Path, network_hash: str | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    h = digest(path)
    side = path.with_name(path.name.removesuffix(".pkl.gz").removesuffix(".pkl") + ".manifest.json")
    record = {"path": str(path), "sha256": h}
    if side.is_file():
        metadata = json.loads(side.read_text(encoding="utf-8"))
        expected = metadata.get("output_sha256")
        if expected and expected.lower() != h.lower():
            raise ValueError(f"saved route hash mismatch: {path}")
        if network_hash and metadata.get("network_sha256") not in (None, network_hash):
            raise ValueError(f"public network hash mismatch: {path}")
        record["manifest"] = str(side)
    return record


def prepare_real(real_path: Path, cache: dict, fraction: float, region_key: str):
    edge_nodes = {int(key): tuple(map(int, value)) for key, value in cache["edge_nodes"].items()}
    labels = {int(key): int(value) for key, value in cache[region_key].items()}
    edge_regions = {
        edge_id: (labels[pair[0]], labels[pair[1]])
        for edge_id, pair in edge_nodes.items()
        if pair[0] in labels and pair[1] in labels
    }
    public_options, edge_context = public_crossing_options(edge_regions, edge_nodes)
    option_context = {option: context for context, options in public_options.items()
                      for option in options}
    real_records = load_pickle(real_path)
    real_trips = matched_choice_trips(real_records, edge_context, edge_nodes)
    split = int(len(real_trips) * (1.0 - fraction))
    return real_trips[split:], public_options, option_context, {
        "path": str(real_path), "sha256": digest(real_path),
        "records": len(real_records), "test_start": split,
        "test_records": len(real_trips) - split,
        "public_choice_contexts": len(public_options),
    }


def route_choice_trips(routes, option_context):
    trips = []
    for route in routes:
        if not route or any(left[1] != right[0] for left, right in zip(route, route[1:])):
            trips.append([])
        else:
            trips.append([(option_context[edge], edge) for edge in route
                          if edge in option_context])
    return trips


def evaluate(raw, cache, real_test, public_options, option_context, slots, mixture_weight):
    routes = normalize_routes(raw, cache)
    if len(routes) != slots:
        raise ValueError(f"expected {slots} release slots, found {len(routes)}")
    valid_count = len(valid_routes(routes))
    yield_value = valid_count / slots
    trips = route_choice_trips(routes, option_context)
    fidelity = road_choice_fidelity(trips, real_test, public_options)
    scores = next_road_trip_scores(trips, real_test, public_options,
                                   mixture_weight=mixture_weight)
    accuracy = float(np.mean(scores["next_road_accuracy"]))
    nll = float(np.mean(scores["next_road_nll"]))
    return {
        "RoadYield": yield_value,
        "RC-CPC": float(fidelity["road_choice_cpc"] * yield_value),
        "RC-NDCG": float(fidelity["road_choice_ndcg"] * yield_value),
        "NextRoadAcc": accuracy * yield_value,
        "NextRoadNLL": nll - math.log(yield_value) if yield_value else math.inf,
        "conditional_RC-CPC": float(fidelity["road_choice_cpc"]),
        "conditional_RC-NDCG": float(fidelity["road_choice_ndcg"]),
        "conditional_NextRoadAcc": accuracy,
        "conditional_NextRoadNLL": nll,
        "valid_route_count": valid_count,
        "choice_trip_count": int(fidelity["road_choice_synthetic_trips"]),
        "choice_event_count": int(fidelity["road_choice_synthetic_events"]),
    }


def plot(rows, path: Path) -> None:
    metrics = (("RoadYield", "Road evidence yield", True),
               ("RC-CPC", "Yield-adjusted RC-CPC", True),
               ("RC-NDCG", "Yield-adjusted RC-NDCG", True),
               ("NextRoadAcc", "Yield-adjusted next-road accuracy", True),
               ("NextRoadNLL", "Penalized next-road NLL", False))
    lookup = {(row["M"], row["R"]): row for row in rows}
    fig, axes = plt.subplots(1, 5, figsize=(27, 5.9), constrained_layout=True)
    for panel, (ax, (metric, title, higher)) in enumerate(zip(axes, metrics)):
        matrix = np.asarray([[float(lookup[(method, router)][metric]) for router in ROUTERS]
                             for method in METHODS])
        finite = matrix[np.isfinite(matrix)]
        if higher:
            image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
            low, high = 0.0, 1.0
        else:
            low = float(finite.min()) if finite.size else 0.0
            high = float(np.quantile(finite, .95)) if finite.size else 1.0
            if high <= low:
                high = low + 1.0
            image = ax.imshow(np.where(np.isfinite(matrix), matrix, high), cmap="YlGnBu_r",
                              vmin=low, vmax=high, aspect="auto")
        ax.set_title(f"{chr(65 + panel)}  {title}", fontsize=12, weight="bold", pad=12)
        ax.set_xticks(range(len(ROUTERS)), ROUTERS, rotation=48, ha="right", fontsize=8)
        ax.set_yticks(range(len(METHODS)), METHODS, fontsize=9)
        for i in range(len(METHODS)):
            for j in range(len(ROUTERS)):
                value = matrix[i, j]
                shown = high if not np.isfinite(value) else value
                foreground = "white" if (shown > (low + high) / 2 if higher
                                         else shown < (low + high) / 2) else "black"
                ax.text(j, i, "$\\infty$" if not np.isfinite(value) else f"{value:.3f}",
                        ha="center", va="center", fontsize=7.1, color=foreground)
        ax.add_patch(Rectangle((-.5, len(METHODS) - 1.5), len(ROUTERS), 1,
                               fill=False, edgecolor="#D62728", linewidth=2.2))
        fig.colorbar(image, ax=ax, fraction=.036, pad=.02).ax.tick_params(labelsize=7)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("MTR measurement × routing under common road-choice metrics",
                 fontsize=15, weight="bold")
    fig.savefig(path.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-routes", required=True,
                        help="real-only matched road reference; never published as synthetic output")
    parser.add_argument("--edge-cache", default="public_assets/ordered_portal_route_cache.pkl.gz")
    parser.add_argument("--network", default="public_assets/beijing_network/network.shp")
    parser.add_argument("--native-dir", default="datasets/synthetic/route_experiments")
    parser.add_argument("--fmm-dir", help="optional freshly generated matched_paths directory")
    parser.add_argument("--stmatch-dir", help="optional freshly generated matched_paths directory")
    parser.add_argument("--direct-dir", default="experiment_results/rebuttal_shared_mr/direct")
    parser.add_argument("--family-dir", default="experiment_results/rebuttal_shared_mr/family")
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--real-test-fraction", type=float, default=.2)
    parser.add_argument("--mixture-weight", type=float, default=.5)
    parser.add_argument("--region-key", default="labels384")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.slots <= 0 or not 0 < args.real_test_fraction < 1 or not 0 <= args.mixture_weight <= 1:
        parser.error("invalid slot count, test fraction, or mixture weight")
    real_path, cache_path, network_path = map(rooted, (args.real_routes, args.edge_cache, args.network))
    native, direct, family_dir = map(rooted, (args.native_dir, args.direct_dir, args.family_dir))
    fmm_dir = rooted(args.fmm_dir) if args.fmm_dir else None
    stmatch_dir = rooted(args.stmatch_dir) if args.stmatch_dir else None
    output = rooted(args.out_dir)
    if output.exists():
        raise FileExistsError(output)
    cache = load_pickle(cache_path)
    if args.region_key not in cache:
        parser.error(f"public cache has no {args.region_key}")
    real_test, public_options, option_context, real_meta = prepare_real(
        real_path, cache, args.real_test_fraction, args.region_key)
    network_hash = digest(network_path)
    rows, provenance = [], {}
    for method in METHODS:
        for router in ROUTERS:
            path = source_path(method, router, native, direct, family_dir,
                               fmm_dir, stmatch_dir)
            info = bound_input(path, network_hash)
            values = evaluate(load_pickle(path), cache, real_test, public_options,
                              option_context, args.slots, args.mixture_weight)
            rows.append({"M": method, "R": router, **values,
                         "source": str(path), "sha256": info["sha256"]})
            provenance[f"{method}/{router}"] = info
            print(f"{method}/{router}: Y={values['RoadYield']:.4f} "
                  f"CPC={values['RC-CPC']:.4f} NDCG={values['RC-NDCG']:.4f}", flush=True)
    if len(rows) != len(METHODS) * len(ROUTERS):
        raise AssertionError("incomplete 5x8 matrix")
    output.mkdir(parents=True)
    table = output / "mr_paper_metrics.csv"
    with table.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, output / "mr_paper_metrics_heatmap")
    (output / "manifest.json").write_text(json.dumps({
        "schema": "shared_rebuttal_1_q2_and_rebuttal_2_q4_v1",
        "scope": ["rebuttal1-Q2", "rebuttal2-Q4"],
        "method_order": METHODS, "router_order": ROUTERS,
        "slots": args.slots, "real_reference": real_meta,
        "public_choice_contexts": len(public_options),
        "region_key": args.region_key, "real_test_fraction": args.real_test_fraction,
        "mixture_weight": args.mixture_weight,
        "edge_cache_sha256": digest(cache_path), "network_sha256": network_hash,
        "route_sources": provenance, "results_sha256": digest(table),
        "metric_contract": {
            "RoadYield": "connected nonempty road routes / all public slots",
            "RC-CPC": "conditional road-choice CPC times RoadYield",
            "RC-NDCG": "conditional road-choice NDCG times RoadYield",
            "NextRoadAcc": "conditional next-road accuracy times RoadYield",
            "NextRoadNLL": "conditional NLL minus log(RoadYield); infinite at zero yield",
        },
        "native_interface_note": "Original coordinate rows require direct observed graph adjacency; the separate native evidence audit counts any directly observed edge.",
        "family_carrier_note": "Family routers resample each method's STMatch carrier, not its raw coordinates.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} shared cells to {output}")


if __name__ == "__main__":
    main()
