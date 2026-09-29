"""Execute an existing generator with train-only input and a separate local ledger."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from synthesis_lineage import SCHEMA, sha, source_digest


def rooted(raw: str) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else ROOT / path).resolve()


def preflight(train: Path, test: Path, record: Path, out: Path, extra: list[str]) -> None:
    if not train.is_file() or not test.is_file():
        raise ValueError("train/test files must exist")
    if sha(train) == sha(test):
        raise ValueError("train and test inputs are identical")
    if record.exists() or record.with_suffix(".log").exists() or out.exists():
        raise ValueError("Choose new record and output paths; existing runs are not overwritten")
    if record.is_relative_to(out):
        raise ValueError("Local lineage record must be outside the synthetic release directory")
    forbidden = {"--data", "--out-dir", "--real-matched", "--dataset-config", "--extra-arg", "--check-only"}
    for token in extra:
        if token.split("=", 1)[0] in forbidden:
            raise ValueError(f"Cannot override train-only binding: {token}")
        if str(test) in token:
            raise ValueError("Test input must not appear in generator arguments")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-real", required=True)
    parser.add_argument("--test-real", required=True, help="Hash binding only; never forwarded to generator")
    parser.add_argument("--record", required=True, help="New local provenance JSON outside release directory")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--generator", choices=("main", "baseline"), default="main")
    parser.add_argument("--train-routes", help="DFR only: matched training paths with input/output-bound sidecar")
    parser.add_argument("generator_args", nargs=argparse.REMAINDER,
                        help="After --: existing generator options excluding private input/output overrides")
    args = parser.parse_args()
    train, test, record, out = map(rooted, (args.train_real, args.test_real, args.record, args.out_dir))
    extra = args.generator_args[1:] if args.generator_args[:1] == ["--"] else args.generator_args
    preflight(train, test, record, out, extra)
    config = json.loads((ROOT / "config/package.json").read_text(encoding="utf-8"))
    if args.generator == "baseline":
        if not config["includes_baseline_code"]:
            parser.error("This folder has saved baselines only; use folder 03 or 04 for baseline generation")
        entry = ROOT / "generation/baselines/algorithms/literature_baselines/entrypoints/generate.py"
        bound = ["--data", str(train)]
    else:
        entry = ROOT / config["main_generation_entry"]
        if config["concrete_algorithm"] == "mtr_dfr":
            if not args.train_routes:
                parser.error("DFR requires --train-routes derived only from --train-real")
            routes = rooted(args.train_routes)
            side = routes.with_name(routes.name.replace(".pkl.gz", ".manifest.json"))
            meta = json.loads(side.read_text(encoding="utf-8-sig"))
            if meta.get("input_sha256") != sha(train) or meta.get("output_sha256") != sha(routes):
                raise ValueError("DFR training routes are not bound to the training input/output")
            bound = ["--real-matched", str(routes)]
        else:
            bound = ["--data", str(train)]
    command = [sys.executable, str(entry), *bound, "--out-dir", str(out), *extra]
    ledger = {"schema": SCHEMA, "classification": "LOCAL_RESEARCH_LINEAGE_NOT_A_DP_RELEASE",
              "generator": args.generator, "algorithm": config["concrete_algorithm"],
              "train_sha256": sha(train), "test_sha256": sha(test),
              "source_sha256": source_digest(ROOT), "command": command,
              "input_policy": "train_only_explicit_generator_input", "status": "running",
              "started_unix": time.time()}
    record.parent.mkdir(parents=True, exist_ok=True)
    def save():
        record.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save()
    log_path = record.with_suffix(".log")
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    with log_path.open("xb") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    ledger.update(returncode=result.returncode, ended_unix=time.time(), log_sha256=sha(log_path))
    ledger["outputs"] = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    stable = (ledger["train_sha256"] == sha(train) and ledger["test_sha256"] == sha(test)
              and ledger["source_sha256"] == source_digest(ROOT))
    ledger["status"] = "passed" if result.returncode == 0 and stable and ledger["outputs"] else "failed"
    save()
    if ledger["status"] != "passed":
        raise SystemExit(result.returncode or 2)
    print(f"EXECUTED TRAIN-ONLY RECORD: {record}", flush=True)


if __name__ == "__main__":
    main()
