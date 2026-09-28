"""Freeze a deterministic 20k Cabspotting trip-level benchmark.

The raw Cabspotting files are taxi histories, not individual trips.  This
preprocessor converts occupied, temporally contiguous runs into the trajectory
records used by the trip-level DP experiment.  Selection is deterministic and
does not inspect downstream utility metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BBOX = (37.60, 37.85, -122.55, -122.30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_distance_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    latitude = math.radians(0.5 * (first[0] + second[0]))
    dy = (second[0] - first[0]) * 111.32
    dx = (second[1] - first[1]) * 111.32 * math.cos(latitude)
    return math.hypot(dx, dy)


def path_length_km(rows: list[tuple[float, float, float, int]]) -> float:
    return sum(
        metric_distance_km((a[1], a[2]), (b[1], b[2]))
        for a, b in zip(rows[:-1], rows[1:])
    )


def parse_file(path: Path) -> list[tuple[float, float, float, int]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                occupied = int(float(parts[2]))
                timestamp = float(parts[3])
            except ValueError:
                continue
            rows.append((timestamp, lat, lon, occupied))
    rows.sort(key=lambda row: row[0])
    return rows


def trip_candidates(
    path: Path,
    bbox: tuple[float, float, float, float],
    *,
    max_gap_seconds: float,
    max_speed_kmh: float,
    min_points: int,
    min_duration_seconds: float,
    min_length_km: float,
) -> list[tuple[bytes, str, np.ndarray, float, float]]:
    candidates = []
    current: list[tuple[float, float, float, int]] = []

    def flush() -> None:
        nonlocal current
        if len(current) >= min_points:
            duration = current[-1][0] - current[0][0]
            length = path_length_km(current)
            if duration >= min_duration_seconds and length >= min_length_km:
                identifier = f"{path.name}:{int(current[0][0])}:{int(current[-1][0])}"
                rank = hashlib.sha256(identifier.encode("utf-8")).digest()
                coordinates = np.asarray([[row[1], row[2]] for row in current], dtype=float)
                candidates.append((rank, identifier, coordinates, duration, length))
        current = []

    previous = None
    for row in parse_file(path):
        timestamp, lat, lon, occupied = row
        inside = bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]
        discontinuity = False
        if previous is not None:
            delta = timestamp - previous[0]
            if delta <= 0 or delta > max_gap_seconds:
                discontinuity = True
            else:
                distance = metric_distance_km((previous[1], previous[2]), (lat, lon))
                discontinuity = distance / (delta / 3600.0) > max_speed_kmh
        if occupied != 1 or not inside or discontinuity:
            flush()
        if occupied == 1 and inside:
            current.append(row)
        previous = row
    flush()
    return candidates


def quantiles(values: list[float]) -> dict[str, float]:
    levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    result = np.quantile(np.asarray(values, dtype=float), levels)
    return {str(level): float(value) for level, value in zip(levels, result)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX)
    parser.add_argument("--max-gap-seconds", type=float, default=300.0)
    parser.add_argument("--max-speed-kmh", type=float, default=160.0)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--min-duration-seconds", type=float, default=120.0)
    parser.add_argument("--min-length-km", type=float, default=0.5)
    args = parser.parse_args()
    args.out = args.out if args.out.is_absolute() else PACKAGE_ROOT / args.out
    if args.count < 1:
        parser.error("--count must be positive")
    if args.out.exists() or args.out.with_suffix(".manifest.json").exists():
        raise FileExistsError("refusing to overwrite a frozen SF trip artifact")

    source = args.source_dir if args.source_dir.is_absolute() else PACKAGE_ROOT / args.source_dir
    args.out = args.out if args.out.is_absolute() else PACKAGE_ROOT / args.out
    files = sorted(source.glob("*.txt"))
    candidates = []
    raw_digest = hashlib.sha256()
    for index, path in enumerate(files, 1):
        file_hash = sha256(path)
        raw_digest.update(path.name.encode("utf-8"))
        raw_digest.update(bytes.fromhex(file_hash))
        candidates.extend(
            trip_candidates(
                path,
                tuple(args.bbox),
                max_gap_seconds=args.max_gap_seconds,
                max_speed_kmh=args.max_speed_kmh,
                min_points=args.min_points,
                min_duration_seconds=args.min_duration_seconds,
                min_length_km=args.min_length_km,
            )
        )
        if index % 50 == 0:
            print(f"parsed {index}/{len(files)} raw files", flush=True)
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < args.count:
        raise RuntimeError(f"only {len(candidates)} eligible trips for {args.count} fixed slots")
    selected = candidates[: args.count]
    trajectories = [item[2] for item in selected]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)

    selected_digest = hashlib.sha256()
    for _, identifier, _, _, _ in selected:
        selected_digest.update(identifier.encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "classification": "FROZEN_PUBLIC_BENCHMARK_PREPROCESSING",
        "privacy_unit": "one preprocessed occupied taxi trip; not one taxi/user",
        "raw_file_count": len(files),
        "raw_tree_sha256": raw_digest.hexdigest(),
        "eligible_trip_count": len(candidates),
        "selected_trip_count": len(selected),
        "selection": "smallest SHA-256(file_name:start_timestamp:end_timestamp)",
        "selected_identifier_sha256": selected_digest.hexdigest(),
        "parameters": {
            "bbox": list(args.bbox),
            "occupied_value": 1,
            "max_gap_seconds": args.max_gap_seconds,
            "max_speed_kmh": args.max_speed_kmh,
            "min_points": args.min_points,
            "min_duration_seconds": args.min_duration_seconds,
            "min_length_km": args.min_length_km,
        },
        "statistics": {
            "point_count_quantiles": quantiles([len(item[2]) for item in selected]),
            "duration_seconds_quantiles": quantiles([item[3] for item in selected]),
            "path_length_km_quantiles": quantiles([item[4] for item in selected]),
        },
        "output": {"path": str(args.out), "sha256": sha256(args.out)},
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
