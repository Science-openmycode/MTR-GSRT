"""Execute the four-baseline, seven-router strict-TSTR comparison.

Router selection uses a fixed subset of the real *training* partition. This
is an evaluation diagnostic, not a claim that selecting a release mechanism
with private validation data is free DP post-processing.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

from run_tstr_experiment import aggregate, base_release_hash, sha256_file
from synthesis_lineage import bind_split


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "evaluation" / "pipeline"
METHODS = ("SPRT", "PrivTrace", "DPTraj-PM", "DPStd")
ROUTERS = ("Native", "FMM", "STMatch", "Family additive", "Family residual",
           "Family length-OT", "Self-carrier reweight")
SLUGS = {"SPRT": "sprt", "PrivTrace": "privtrace", "DPTraj-PM": "dptraj_pm", "DPStd": "dpstd"}
FAMILY_FILES = {"Family additive": "family_additive", "Family residual": "family_residual",
                "Family length-OT": "family_length_ot", "Self-carrier reweight": "self_carrier_reweight"}
METRICS = ("Next-cell Hit@1", "Next-cell MRR", "Destination Hit@5",
           "Road continuation Hit@1")


def rooted(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def parse_sources(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--source requires METHOD=PATH")
        name, path = value.split("=", 1)
        if name in result:
            raise ValueError(f"Duplicate source method: {name}")
        result[name] = rooted(path)
    if set(result) != set(METHODS):
        raise ValueError(f"--source must contain exactly {METHODS}")
    return result


def candidate_paths(args, sources: dict[str, Path]) -> dict[tuple[str, str], tuple[Path, Path]]:
    views, families = rooted(args.views_root), rooted(args.family_root)
    result = {}
    for method in METHODS:
        slug = SLUGS[method]
        result[(method, "Native")] = (sources[method], sources[method])
        for router in ("FMM", "STMatch"):
            key = router.lower()
            result[(method, router)] = (
                views / "generic" / key / f"{slug}.pkl",
                views / "road" / key / f"{slug}.pkl",
            )
        for router, filename in FAMILY_FILES.items():
            result[(method, router)] = (
                families / method / f"{filename}_generic.pkl",
                families / method / f"{filename}_road.pkl",
            )
    if set(result) != {(method, router) for method in METHODS for router in ROUTERS}:
        raise AssertionError("Seven-router candidate matrix is incomplete")
    for (method, router), pair in result.items():
        expected = sha256_file(sources[method])
        for path in pair:
            if not path.is_file():
                raise FileNotFoundError(path)
            observed = base_release_hash(path)
            if observed != expected:
                raise ValueError(f"{method}/{router} view derives from {observed}, expected {expected}: {path}")
    return result


def write_pickle(path: Path, value) -> None:
    with path.open("xb") as handle:
        pickle.dump(value, handle, pickle.HIGHEST_PROTOCOL)


def selection_split(train: Path, fit_path: Path, validation_path: Path,
                    fraction: float, seed: int) -> dict:
    with train.open("rb") as handle:
        rows = pickle.load(handle)
    if len(rows) < 2 or not 0 < fraction < 1:
        raise ValueError("Selection split needs at least two training records and 0<fraction<1")
    order = np.random.default_rng(seed + 91).permutation(len(rows))
    cut = int(round((1.0 - fraction) * len(rows)))
    if not 0 < cut < len(rows):
        raise ValueError("Selection fraction creates an empty partition")
    fit = [rows[int(index)] for index in order[:cut]]
    validation = [rows[int(index)] for index in order[cut:]]
    write_pickle(fit_path, fit)
    write_pickle(validation_path, validation)
    return {"train_records": len(rows), "selection_fit_records": len(fit),
            "selection_validation_records": len(validation), "fit_sha256": sha256_file(fit_path),
            "validation_sha256": sha256_file(validation_path), "seed": seed + 91,
            "validation_fraction": fraction}


def run_tasks(name: str, train: Path, test: Path,
              candidates: list[tuple[str, Path, Path]], args, binding: dict) -> Path:
    target = rooted(args.out_dir) / name
    if target.exists():
        raise FileExistsError(target)
    generic_dir, road_dir = target / "raw_generic", target / "raw_road"
    names = [label for label, _, _ in candidates]
    generic = [sys.executable, str(PIPELINE / "evaluate_generic_mobility_tasks.py"),
               "--train-real", str(train), "--test-real", str(test)]
    road = [sys.executable, str(PIPELINE / "evaluate_road_mining_tasks.py"),
            "--train-real", str(train), "--test-real", str(test)]
    for _, generic_path, road_path in candidates:
        generic += ["--synthetic", str(generic_path)]
        road += ["--synthetic", str(road_path)]
    common = ["--names", *names, "--bbox", *map(str, args.bbox)]
    generic += [*common, "--seed", str(args.seed), "--out-dir", str(generic_dir)]
    road += [*common, "--osm-cache", str(rooted(args.osm_cache)), "--out-dir", str(road_dir)]
    for command in (generic, road):
        print("RUN:", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    aggregate(generic_dir, road_dir, target, binding)
    return target / "results.csv"


def read_results(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["Pipeline"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate task result labels")
    return {row["Pipeline"]: row for row in rows}


def select_routers(path: Path) -> tuple[dict[str, str], list[dict]]:
    results = read_results(path)
    expected = {"Real-train"} | {f"{method} :: {router}" for method in METHODS for router in ROUTERS}
    if set(results) != expected:
        raise ValueError(f"Validation rows differ from complete 4x7 set: {sorted(expected ^ set(results))}")
    scored = []
    selected = {}
    for method in METHODS:
        choices = []
        for router in ROUTERS:
            row = results[f"{method} :: {router}"]
            values = [float(row[metric]) for metric in METRICS]
            if not all(np.isfinite(value) and 0 <= value <= 1 for value in values):
                raise ValueError(f"Invalid validation score: {method}/{router}")
            score = sum(values) / len(values)
            scored.append({"method": method, "router": router, "selection_score": score,
                           **{metric: value for metric, value in zip(METRICS, values)}})
            choices.append((router, score))
        selected[method] = sorted(choices, key=lambda item: (-item[1], item[0]))[0][0]
    return selected, scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "final", "all"), default="all")
    parser.add_argument("--train-real", required=True)
    parser.add_argument("--test-real", required=True)
    parser.add_argument("--split-audit", required=True)
    parser.add_argument("--source", action="append", required=True, help="METHOD=train-only release; repeat four times")
    parser.add_argument("--mtr-source", required=True, help="Train-only MTR-GSRT coordinates")
    parser.add_argument("--views-root", required=True, help="FMM/STMatch generic and road view root")
    parser.add_argument("--family-root", required=True, help="Four method-specific family view directories")
    parser.add_argument("--osm-cache", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--diagnostic-private-validation", action="store_true",
                        help="Acknowledge that real-training-set router selection is not free DP post-processing")
    args = parser.parse_args()
    if not args.diagnostic_private_validation:
        parser.error("Supply --diagnostic-private-validation; this router selection uses raw real-train labels")
    train, test, mtr = rooted(args.train_real), rooted(args.test_real), rooted(args.mtr_source)
    if sha256_file(train) == sha256_file(test):
        parser.error("train/test files are identical")
    split = bind_split(rooted(args.split_audit), sha256_file(train), sha256_file(test))
    sources = parse_sources(args.source)
    paths = candidate_paths(args, sources)
    out = rooted(args.out_dir)
    hashes = {method: sha256_file(source) for method, source in sources.items()}
    hashes["MTR-GSRT"] = sha256_file(mtr)
    binding = {"classification": "RESEARCH_ONLY_PRIVATE_VALIDATION_ROUTER_SELECTION_NOT_A_DP_RELEASE",
               "train_sha256": sha256_file(train), "test_sha256": sha256_file(test),
               "bbox": args.bbox, "seed": args.seed,
               "osm_sha256": sha256_file(rooted(args.osm_cache)),
               "base_releases": hashes, "disjoint_split": split, "synthesis_lineage": {}}
    if args.stage in ("validate", "all"):
        if out.exists():
            parser.error(f"Choose a fresh output root for validation: {out}")
        out.mkdir(parents=True)
        fit, validation = out / "selection_real_fit.pkl", out / "selection_real_validation.pkl"
        split_info = selection_split(train, fit, validation, args.validation_fraction, args.seed)
        candidates = [(f"{method} :: {router}", *paths[(method, router)])
                      for method in METHODS for router in ROUTERS]
        validation_csv = run_tasks("validation", fit, validation, candidates, args, binding)
        selected, scores = select_routers(validation_csv)
        with (out / "validation_all_router_scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
            writer.writeheader(); writer.writerows(scores)
        (out / "selected_routers.json").write_text(json.dumps({
            "classification": binding["classification"], "selected": selected,
            "criterion": "unweighted mean of four Real-train-normalized task scores",
            "validation_results_sha256": sha256_file(validation_csv),
            "base_releases": hashes, "train_sha256": binding["train_sha256"],
            "test_sha256": binding["test_sha256"], "selection_split": split_info,
            "test_records_used_for_selection": False,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("VALIDATION SELECTED:", selected, flush=True)
    if args.stage in ("final", "all"):
        selection_file = out / "selected_routers.json"
        selected_data = json.loads(selection_file.read_text(encoding="utf-8-sig"))
        if (selected_data.get("base_releases") != hashes
                or selected_data.get("train_sha256") != binding["train_sha256"]
                or selected_data.get("test_sha256") != binding["test_sha256"]
                or selected_data.get("validation_results_sha256") != sha256_file(out / "validation/results.csv")):
            raise ValueError("Router selection does not bind this split/release batch")
        selected = selected_data["selected"]
        if set(selected) != set(METHODS) or any(selected[method] not in ROUTERS for method in METHODS):
            raise ValueError("Incomplete or invalid selected router mapping")
        recomputed, _ = select_routers(out / "validation/results.csv")
        if selected != recomputed:
            raise ValueError("Selected router mapping differs from validation scores")
        candidates = [(f"{method} :: Native", sources[method], sources[method]) for method in METHODS]
        candidates += [(f"{method} :: {selected[method]}", *paths[(method, selected[method])])
                       for method in METHODS if selected[method] != "Native"]
        candidates += [("MTR-GSRT :: Native", mtr, mtr)]
        if len({item[0] for item in candidates}) != len(candidates):
            raise ValueError("Duplicate final task label")
        final_csv = run_tasks("final_test", train, test, candidates, args, binding)
        final_rows = read_results(final_csv)
        table = []
        for method in METHODS:
            for variant, router in (("Native", "Native"), ("Best public routing", selected[method])):
                label = f"{method} :: {router}"
                table.append({"method": method, "variant": variant, "router": router,
                              **{metric: float(final_rows[label][metric]) for metric in METRICS}})
        table.append({"method": "MTR-GSRT", "variant": "Native", "router": "GSRT",
                      **{metric: float(final_rows["MTR-GSRT :: Native"][metric]) for metric in METRICS}})
        with (out / "final_native_best_mtr.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
        (out / "manifest.json").write_text(json.dumps({
            "classification": binding["classification"], "candidate_routers": ROUTERS,
            "selected_routers": selected, "metrics": METRICS, "split_binding": binding,
            "validation_results_sha256": sha256_file(out / "validation/results.csv"),
            "final_results_sha256": sha256_file(final_csv),
            "table_sha256": sha256_file(out / "final_native_best_mtr.csv"),
            "native_rows_recomputed_in_same_final_batch": True,
            "scope_note": "This diagnostic selects a router with raw real-training validation labels; it is not a no-extra-budget DP release.",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"FINAL TABLE: {out / 'final_native_best_mtr.csv'}", flush=True)


if __name__ == "__main__":
    main()
