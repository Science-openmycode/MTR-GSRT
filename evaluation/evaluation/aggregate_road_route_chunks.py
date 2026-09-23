"""Validate and aggregate per-method actual-road route evaluations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path


def _public_release() -> Path:
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (
            (candidate / "configs" / "datasets.json").is_file()
            and (candidate / "generation" / "common" / "runtime.py").is_file()
        ):
            return candidate
    raise RuntimeError("cannot locate public_release root")


PUBLIC_RELEASE = _public_release()
if str(PUBLIC_RELEASE) not in sys.path:
    sys.path.insert(0, str(PUBLIC_RELEASE))

from generation.common.runtime import public_path, sha256_file, write_json  # noqa: E402


DEFAULT_METHODS = ("SPRT", "PrivTrace", "DPTraj-PM", "DPStd", "MTR-GSRT")
LOWER_IS_BETTER = {"od_conditioned_road_transition_jsd"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-dir", required=True)
    result.add_argument("--out-dir", required=True)
    result.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    result.add_argument("--verify-only", action="store_true")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(method: str) -> str:
    return method.replace("-", "_")


def _verify_bound_file(binding: dict, label: str) -> None:
    path = Path(binding["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label}: bound file is missing: {path}")
    if binding.get("sha256") != _sha256(path):
        raise RuntimeError(f"{label}: bound file hash mismatch: {path}")


def _load_chunk(root: Path, method: str) -> tuple[dict, dict, Path]:
    directory = root / _safe_name(method)
    metrics_path = directory / "metrics.json"
    manifest_path = directory / "manifest.json"
    for required in (metrics_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing {required}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_outputs = manifest.get("outputs", {})
    for name in ("metrics.json", "metrics.csv"):
        path = directory / name
        expected = expected_outputs.get(name)
        if not path.is_file() or expected != _sha256(path):
            raise RuntimeError(f"{method}: output hash mismatch for {name}")
    inputs = manifest["inputs"]
    for name, binding in inputs["synthetic"].items():
        _verify_bound_file(binding, f"{method}: synthetic {name}")
    for name, binding in inputs["witness"].items():
        _verify_bound_file(binding, f"{method}: witness {name}")
    _verify_bound_file(inputs["osm"], f"{method}: OSM")
    _verify_bound_file(inputs["fmm_network"], f"{method}: road network")
    if inputs.get("ubodt") is not None:
        _verify_bound_file(inputs["ubodt"], f"{method}: UBODT")
    evaluator_path = Path(manifest["evaluator"]["entrypoint"])
    if manifest["evaluator"]["entrypoint_sha256"] != _sha256(evaluator_path):
        raise RuntimeError(f"{method}: evaluator source hash mismatch")
    metric_module = evaluator_path.parent / "metric_suites" / "road_route.py"
    if manifest["evaluator"]["metric_module_sha256"] != _sha256(metric_module):
        raise RuntimeError(f"{method}: metric module source hash mismatch")
    result_names = set(metrics.get("results", {})).difference({"Real"})
    if result_names != {method}:
        raise RuntimeError(
            f"{method}: expected one method result, received {sorted(result_names)}"
        )
    values = metrics["results"][method]
    for name, value in values.items():
        if value is not None and not math.isfinite(float(value)):
            raise RuntimeError(f"{method}: non-finite {name}={value}")
    return metrics, manifest, directory


def _rankings(rows: dict[str, dict]) -> dict[str, list[dict]]:
    metric_names = sorted({key for row in rows.values() for key in row})
    output: dict[str, list[dict]] = {}
    for metric in metric_names:
        available = [
            (method, float(values[metric]))
            for method, values in rows.items()
            if values.get(metric) is not None
        ]
        reverse = metric not in LOWER_IS_BETTER
        available.sort(key=lambda item: item[1], reverse=reverse)
        ranked = []
        prior_value = None
        prior_rank = 0
        for index, (method, value) in enumerate(available, start=1):
            if prior_value is None or value != prior_value:
                prior_rank = index
                prior_value = value
            ranked.append({"rank": prior_rank, "method": method, "value": value})
        output[metric] = ranked
    return output


def main() -> None:
    args = parser().parse_args()
    started = time.time()
    input_dir = public_path(args.input_dir)
    out_dir = public_path(args.out_dir)
    if out_dir.exists() and not args.verify_only:
        raise FileExistsError(f"output directory already exists: {out_dir}")
    if args.verify_only and not out_dir.is_dir():
        raise FileNotFoundError(f"aggregate output directory does not exist: {out_dir}")

    chunks: dict[str, tuple[dict, dict, Path]] = {
        method: _load_chunk(input_dir, method) for method in args.methods
    }
    first_metrics, first_manifest, _ = chunks[args.methods[0]]
    invariants = {
        "dataset": first_manifest["dataset"],
        "record_count": first_manifest["record_count"],
        "real": first_manifest["inputs"]["real"],
        "network_sha256": first_manifest["inputs"]["fmm_network"]["sha256"],
        "matcher": first_manifest["inputs"]["matcher"],
        "evaluator_sha256": first_manifest["evaluator"]["entrypoint_sha256"],
        "metric_module_sha256": first_manifest["evaluator"]["metric_module_sha256"],
        "parameters": first_metrics["parameters"],
        "metric_semantics": first_metrics["metric_semantics"],
    }
    for method, (metrics, manifest, _) in chunks.items():
        observed = {
            "dataset": manifest["dataset"],
            "record_count": manifest["record_count"],
            "real": manifest["inputs"]["real"],
            "network_sha256": manifest["inputs"]["fmm_network"]["sha256"],
            "matcher": manifest["inputs"]["matcher"],
            "evaluator_sha256": manifest["evaluator"]["entrypoint_sha256"],
            "metric_module_sha256": manifest["evaluator"]["metric_module_sha256"],
            "parameters": metrics["parameters"],
            "metric_semantics": metrics["metric_semantics"],
        }
        if observed != invariants:
            differing = sorted(key for key in invariants if invariants[key] != observed[key])
            raise RuntimeError(f"{method}: incompatible chunk fields: {differing}")

    rows = {
        method: chunks[method][0]["results"][method]
        for method in args.methods
    }
    rankings = _rankings(rows)
    payload = {
        "schema_version": 1,
        "classification": "RESEARCH_EVALUATION_LEDGER_NOT_A_DP_RELEASE",
        "dataset": invariants["dataset"],
        "record_count": invariants["record_count"],
        "methods": list(args.methods),
        "metric_semantics": invariants["metric_semantics"],
        "parameters": invariants["parameters"],
        "results": rows,
        "rankings": rankings,
    }

    if args.verify_only:
        expected_metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        if expected_metrics != payload:
            raise RuntimeError("aggregate metrics do not match the current validated chunks")
        existing_manifest = json.loads(
            (out_dir / "manifest.json").read_text(encoding="utf-8")
        )
        for name in ("metrics.json", "metrics.csv", "rankings.csv"):
            if existing_manifest["outputs"].get(name) != sha256_file(out_dir / name):
                raise RuntimeError(f"aggregate output hash mismatch for {name}")
        if existing_manifest["aggregator"].get("sha256") != sha256_file(Path(__file__).resolve()):
            raise RuntimeError("aggregate evaluator source hash mismatch")
        for method, (_, manifest, directory) in chunks.items():
            recorded = existing_manifest["chunks"].get(method, {})
            if recorded.get("metrics_sha256") != sha256_file(directory / "metrics.json"):
                raise RuntimeError(f"{method}: aggregate chunk metrics hash mismatch")
            if recorded.get("manifest_sha256") != sha256_file(directory / "manifest.json"):
                raise RuntimeError(f"{method}: aggregate chunk manifest hash mismatch")
        print(f"verified {out_dir}")
        return

    out_dir.mkdir(parents=True)
    write_json(out_dir / "metrics.json", payload)
    metric_names = sorted({metric for values in rows.values() for metric in values})
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *metric_names])
        writer.writeheader()
        for method in args.methods:
            writer.writerow({"method": method, **rows[method]})
    with (out_dir / "rankings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["metric", "direction", "rank", "method", "value"]
        )
        writer.writeheader()
        for metric, values in rankings.items():
            direction = "lower" if metric in LOWER_IS_BETTER else "higher"
            for item in values:
                writer.writerow({"metric": metric, "direction": direction, **item})

    manifest = {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "invariants": invariants,
        "chunks": {
            method: {
                "directory": str(directory),
                "metrics_sha256": sha256_file(directory / "metrics.json"),
                "manifest_sha256": sha256_file(directory / "manifest.json"),
                "synthetic": manifest["inputs"]["synthetic"][method],
            }
            for method, (_, manifest, directory) in chunks.items()
        },
        "aggregator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "outputs": {
            name: sha256_file(out_dir / name)
            for name in ("metrics.json", "metrics.csv", "rankings.csv")
        },
        "elapsed_sec": time.time() - started,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
