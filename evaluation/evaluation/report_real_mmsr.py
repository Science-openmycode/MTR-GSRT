"""Report full-corpus Real MMSR from a saved common map-match cache."""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--minimum-path-edges", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.cache.resolve(), "rb") as handle:
        records = pickle.load(handle)
    if len(records) != args.expected_count:
        raise RuntimeError(f"cache has {len(records)} records; expected {args.expected_count}")
    if args.minimum_path_edges < 1:
        raise ValueError("--minimum-path-edges must be positive")
    accepted = sum(
        bool(row.get("accepted")) and len(row.get("cpath", ())) >= args.minimum_path_edges
        for row in records
    )
    payload = {
        "schema_version": 1,
        "selection_data": "Real only",
        "record_count": len(records),
        "accepted": accepted,
        "real_mmsr": accepted / len(records),
        "minimum_path_edges": args.minimum_path_edges,
    }
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
