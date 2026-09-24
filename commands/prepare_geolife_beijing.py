"""Rebuild the Beijing paper input from the public GeoLife 1.3 PLT files.

The frozen full corpus is the seeded train permutation followed by the test
permutation. This script never reads a private repository path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


DEFAULT_BBOX = (39.75, 40.15, 116.10, 116.65)
PAPER_SHA256 = "6a160ca557fbd7bab97af489b56c931e73532c498c390e31965c0d61e53361fd"


class _DigestSink:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, chunk: bytes) -> int:
        view = memoryview(chunk)
        self.digest.update(view)
        self.size += view.nbytes
        return view.nbytes


def read_geolife(data_dir: Path, bbox: tuple[float, ...], limit: int | None) -> list[np.ndarray]:
    if data_dir.name != "Data" or not data_dir.is_dir():
        raise ValueError("--source-dir must be the GeoLife 1.3 Data directory")
    files = sorted(data_dir.glob("*/Trajectory/*.plt"))
    if not files:
        raise ValueError(f"No GeoLife PLT files found under {data_dir}")
    trajectories: list[np.ndarray] = []
    for path in files:
        points: list[list[float]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as source:
            for _ in range(6):
                next(source, None)
            for line in source:
                fields = line.strip().split(",")
                if len(fields) < 2:
                    continue
                try:
                    lat, lon = float(fields[0]), float(fields[1])
                except ValueError:
                    continue
                if bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]:
                    points.append([lat, lon])
        if len(points) >= 2:
            trajectories.append(np.asarray(points, dtype=float))
            if limit is not None and len(trajectories) >= limit:
                break
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Unzipped GeoLife 1.3/Data directory")
    parser.add_argument("--out", type=Path, help="Output real_full_frozen.pkl; omitted with --check-only")
    parser.add_argument("--seed", type=int, default=20260713, help="Public train/test permutation seed")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    parser.add_argument("--limit", type=int, help="First N eligible trajectories for a quick smoke test")
    parser.add_argument("--expected-sha256", help="Fail before writing if the serialized bytes differ")
    parser.add_argument("--check-only", action="store_true", help="Compute hash without saving the corpus")
    args = parser.parse_args()
    if not args.check_only and args.out is None:
        parser.error("--out is required unless --check-only is set")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must lie between 0 and 1")
    if not args.check_only and args.out is not None and args.out.exists():
        parser.error(f"Output already exists: {args.out}; choose a new path")

    trajectories = read_geolife(args.source_dir.resolve(), tuple(args.bbox), args.limit)
    if len(trajectories) < 2:
        parser.error("The selected GeoLife files contain fewer than two eligible trajectories")
    n_train = int(round(len(trajectories) * args.train_fraction))
    permutation = np.random.default_rng(args.seed).permutation(len(trajectories))
    ordered = [trajectories[int(index)] for index in permutation]
    sink = _DigestSink()
    pickle.dump(ordered, sink, protocol=pickle.HIGHEST_PROTOCOL)
    sha256 = sink.digest.hexdigest()
    if args.expected_sha256 and sha256.lower() != args.expected_sha256.lower():
        raise SystemExit(f"SHA-256 mismatch: got {sha256}, expected {args.expected_sha256}")
    if args.out is not None and not args.check_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as destination:
            pickle.dump(ordered, destination, protocol=pickle.HIGHEST_PROTOCOL)
        with args.out.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != sha256:
            raise RuntimeError(f"Output hash differs from preflight digest: {actual} != {sha256}")
    print(json.dumps({"trajectory_count": len(ordered), "train_count": n_train,
                      "test_count": len(ordered) - n_train, "serialized_bytes": sink.size,
                      "sha256": sha256, "paper_sha256_match": sha256 == PAPER_SHA256,
                      "output": str(args.out.resolve()) if args.out and not args.check_only else None},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
