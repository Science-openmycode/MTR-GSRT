from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select train-only matched routes using a frozen split manifest.")
    parser.add_argument("--matched-full", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.matched_full.suffix == ".gz" else open
    with opener(args.matched_full, "rb") as handle:
        rows = pickle.load(handle)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    indices = [int(v) for v in manifest["train_indices"]]
    if max(indices, default=-1) >= len(rows):
        raise RuntimeError("split indices exceed the full matched-route corpus")
    selected = [rows[index] for index in indices]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_open = gzip.open if args.out.suffix == ".gz" else open
    with out_open(args.out, "wb") as handle:
        pickle.dump(selected, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {len(selected)} train-only matched routes to {args.out.resolve()}")


if __name__ == "__main__":
    main()
