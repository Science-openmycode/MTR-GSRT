from __future__ import annotations

import argparse
import hashlib
import json
import gzip
import os
import pickle
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "package.json").read_text(encoding="utf-8"))
TEXT_SUFFIXES = {
    ".bat", ".bib", ".cfg", ".csv", ".ini", ".json", ".md", ".ps1",
    ".py", ".sh", ".tex", ".toml", ".txt", ".yaml", ".yml",
}


def canonical_bytes(path: Path) -> bytes:
    """Return platform-independent bytes for text and exact bytes for artifacts."""
    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitattributes", ".gitignore"}:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def stage_inventory() -> int:
    print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
    return 0


def stage_verify() -> int:
    manifest_path = ROOT / "manifests" / "files.sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"missing: {entry['path']}")
            continue
        payload = canonical_bytes(path)
        if len(payload) != entry["size"]:
            failures.append(f"size: {entry['path']}")
            continue
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            failures.append(f"sha256: {entry['path']}")
    actual_data = [p for p in (ROOT / "datasets" / "synthetic").rglob("*") if p.is_file()]
    if not actual_data:
        failures.append("no packaged synthetic datasets")
    forbidden_data_names = {"real.pkl", "real.pkl.gz", "real_full_frozen.pkl", "raw.pkl"}
    for path in actual_data:
        if path.name.lower() in forbidden_data_names:
            failures.append(f"private/raw dataset name in package: {path.relative_to(ROOT)}")
    baseline_source = ROOT / "generation" / "baselines"
    if CONFIG["includes_baseline_code"] != baseline_source.is_dir():
        failures.append("baseline-code inclusion does not match package contract")
    forbidden_algorithm = CONFIG["forbidden_algorithm_token"].lower()
    for path in ROOT.rglob("*"):
        if forbidden_algorithm in path.name.lower():
            failures.append(f"cross-algorithm contamination: {path.relative_to(ROOT)}")
    if failures:
        print("VERIFY FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 2
    print(f"VERIFY OK: {CONFIG['package_id']} ({len(manifest['files'])} files)")
    return 0


def run(command: list[str]) -> int:
    print("RUN:", subprocess.list2cmdline(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def forwarded(values: list[str]) -> list[str]:
    return values[1:] if values and values[0] == "--" else values


def stage_plot(args: list[str]) -> int:
    return run([
        sys.executable,
        str(ROOT / "plotting" / "regenerate_figures.py"),
        *forwarded(args),
    ])


def stage_smoke() -> int:
    code = stage_verify()
    if code:
        return code
    inspected = []
    for path in sorted((ROOT / "datasets" / "synthetic").rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(".pkl.gz"):
            with gzip.open(path, "rb") as handle:
                value = pickle.load(handle)
        elif path.suffix.lower() == ".pkl":
            with path.open("rb") as handle:
                value = pickle.load(handle)
        elif path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            continue
        length = len(value) if hasattr(value, "__len__") else None
        inspected.append({"path": path.relative_to(ROOT).as_posix(),
                          "type": type(value).__name__, "length": length})
    if not inspected:
        print("SMOKE FAILED: no serializable synthetic artifacts", file=sys.stderr)
        return 4
    print(json.dumps({"smoke": "OK", "artifacts": inspected}, ensure_ascii=False, indent=2))
    return 0


def stage_generate_main(args: list[str]) -> int:
    entry = ROOT / CONFIG["main_generation_entry"]
    return run([sys.executable, str(entry), *forwarded(args)])


def stage_generate_baselines(args: list[str]) -> int:
    if not CONFIG["includes_baseline_code"]:
        print(
            "This package intentionally contains only precomputed baseline synthetic datasets. "
            "Use package 03 or 04 to regenerate baselines from source.",
            file=sys.stderr,
        )
        return 3
    entry = ROOT / "generation" / "baselines" / "algorithms" / "literature_baselines" / "entrypoints" / "generate.py"
    return run([sys.executable, str(entry), *forwarded(args)])


def stage_evaluate(args: list[str]) -> int:
    entry = ROOT / "evaluation" / "evaluation" / "evaluate_all.py"
    return run([sys.executable, str(entry), *forwarded(args)])


def stage_helper(name: str, args: list[str]) -> int:
    entries = {
        "prepare-geolife": ROOT / "commands" / "prepare_geolife_beijing.py",
        "prepare-split": ROOT / "evaluation" / "pipeline" / "prepare_kdd_revised_split.py",
        "prepare-attack-split": ROOT / "evaluation" / "pipeline" / "prepare_privacy_attack_splits.py",
        "prepare-road-reference": ROOT / "evaluation" / "evaluation" / "cache_common_road_matches.py",
        "subset-routes": ROOT / "evaluation" / "subset_matched_routes.py",
        "routes-to-coordinates": ROOT / "evaluation" / "road_routes_to_coordinates.py",
    }
    return run([sys.executable, str(entries[name]), *forwarded(args)])


def stage_tstr(args: list[str]) -> int:
    entry = ROOT / "evaluation" / "run_tstr_experiment.py"
    return run([sys.executable, str(entry), *forwarded(args)])


def stage_combine_tstr_mr(args: list[str]) -> int:
    entry = ROOT / "evaluation" / "combine_tstr_mr.py"
    return run([sys.executable, str(entry), *forwarded(args)])


def stage_materialize_tstr_mr(args: list[str]) -> int:
    entry = ROOT / "evaluation" / "materialize_tstr_mr.py"
    return run([sys.executable, str(entry), *forwarded(args)])


def stage_verify_matcher(args: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check the Windows FMM/STMatch runtime before matching trajectories.")
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--fmm-bin", default="public_assets/matcher/fmm.exe")
    parser.add_argument("--stmatch-bin", default="public_assets/matcher/stmatch.exe")
    parsed = parser.parse_args(forwarded(args))

    def resolved(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else ROOT / path).resolve()

    runtime_dir = resolved(parsed.runtime_dir)
    binaries = [resolved(parsed.fmm_bin), resolved(parsed.stmatch_bin)]
    required = [runtime_dir / "gdal204.dll", runtime_dir / "boost_serialization.dll", *binaries]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("MATCHER RUNTIME FAILED: missing files:", file=sys.stderr)
        for path in missing:
            print(f"- {path}", file=sys.stderr)
        return 5
    if os.name == "nt":
        for binary in binaries:
            if not (binary.parent / "FMMLIB.dll").is_file():
                print(f"MATCHER RUNTIME FAILED: missing {binary.parent / 'FMMLIB.dll'}", file=sys.stderr)
                return 5
    env = os.environ.copy()
    env["PATH"] = str(runtime_dir) + os.pathsep + env.get("PATH", "")
    for binary in binaries:
        try:
            result = subprocess.run([str(binary), "--help"], cwd=binary.parent, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, errors="replace", timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"MATCHER RUNTIME FAILED: {binary.name}: {exc}", file=sys.stderr)
            return 5
        if result.returncode not in (0, 1) or not (result.stdout.strip() or result.stderr.strip()):
            print(f"MATCHER RUNTIME FAILED: {binary.name} exit={result.returncode}", file=sys.stderr)
            print((result.stderr or result.stdout).strip(), file=sys.stderr)
            return 5
    print("MATCHER RUNTIME OK: fmm.exe and stmatch.exe started")
    return 0


def stage_ablation(args: list[str]) -> int:
    entry = ROOT / "evaluation" / "run_ablation_experiment.py"
    return run([sys.executable, str(entry), *forwarded(args)])


def stage_executed(name: str, args: list[str]) -> int:
    entry = ROOT / "evaluation" / f"run_{name}_experiment.py"
    return run([sys.executable, str(entry), *forwarded(args)])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Root command for generation, evaluation, plotting, and integrity verification."
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("inventory", "verify", "smoke", "all-precomputed"):
        sub.add_parser(name)
    plot = sub.add_parser("plot")
    plot.add_argument("args", nargs=argparse.REMAINDER)
    for name in ("generate-main", "generate-baselines", "evaluate",
                 "verify-matcher", "prepare-geolife", "prepare-split", "prepare-attack-split", "prepare-road-reference", "subset-routes",
                 "routes-to-coordinates", "run-tstr", "materialize-tstr-mr", "combine-tstr-mr",
                 "run-ablation", "run-framework", "run-privacy", "run-profile", "run-mr", "run-structure"):
        child = sub.add_parser(name)
        child.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    if ns.stage == "inventory":
        return stage_inventory()
    if ns.stage == "verify":
        return stage_verify()
    if ns.stage == "plot":
        return stage_plot(ns.args)
    if ns.stage == "smoke":
        return stage_smoke()
    if ns.stage == "generate-main":
        return stage_generate_main(ns.args)
    if ns.stage == "generate-baselines":
        return stage_generate_baselines(ns.args)
    if ns.stage == "evaluate":
        return stage_evaluate(ns.args)
    if ns.stage == "verify-matcher":
        return stage_verify_matcher(ns.args)
    if ns.stage in {"prepare-geolife", "prepare-split", "prepare-attack-split", "prepare-road-reference", "subset-routes", "routes-to-coordinates"}:
        return stage_helper(ns.stage, ns.args)
    if ns.stage == "run-tstr":
        return stage_tstr(ns.args)
    if ns.stage == "combine-tstr-mr":
        return stage_combine_tstr_mr(ns.args)
    if ns.stage == "materialize-tstr-mr":
        return stage_materialize_tstr_mr(ns.args)
    if ns.stage == "run-ablation":
        return stage_ablation(ns.args)
    if ns.stage.startswith("run-"):
        return stage_executed(ns.stage.removeprefix("run-"), ns.args)
    if ns.stage == "all-precomputed":
        code = stage_verify()
        return code if code else stage_plot(["--figure", "all"])
    raise AssertionError(ns.stage)


if __name__ == "__main__":
    raise SystemExit(main())
