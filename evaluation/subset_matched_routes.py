from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_indices(manifest: dict, count: int) -> list[int]:
    indices = [int(v) for v in manifest["train_indices"]]
    tests = [int(v) for v in manifest.get("test_indices", [])]
    if len(set(indices)) != len(indices) or len(set(tests)) != len(tests):
        raise ValueError("split contains duplicate indices")
    if set(indices) & set(tests):
        raise ValueError("train/test split indices overlap")
    if any(index < 0 or index >= count for index in indices + tests):
        raise ValueError("split indices outside the full matched-route corpus")
    if manifest.get("total_count", count) != count:
        raise ValueError("split total_count differs from matched-route count")
    if manifest.get("train_count", len(indices)) != len(indices):
        raise ValueError("split train_count differs from train_indices")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Select train-only matched routes using a frozen split manifest.")
    parser.add_argument("--matched-full", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for key in ("matched_full", "split_manifest", "out"):
        path = getattr(args, key)
        setattr(args, key, (path if path.is_absolute() else ROOT / path).resolve())
    opener = gzip.open if args.matched_full.suffix == ".gz" else open
    with opener(args.matched_full, "rb") as handle:
        rows = pickle.load(handle)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    indices = validate_indices(manifest, len(rows))
    selected = [rows[index] for index in indices]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_open = gzip.open if args.out.suffix == ".gz" else open
    with out_open(args.out, "wb") as handle:
        pickle.dump(selected, handle, protocol=pickle.HIGHEST_PROTOCOL)
    audit = {
        "classification": "LOCAL_RESEARCH_SPLIT_AUDIT_NOT_A_DP_RELEASE",
        "selected_count": len(selected),
        "split_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
        "full_routes_sha256": hashlib.sha256(args.matched_full.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "train_file": manifest.get("train_file"),
    }
    args.out.with_name(args.out.name + ".split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(selected)} train-only matched routes to {args.out.resolve()}")


if __name__ == "__main__":
    main()
