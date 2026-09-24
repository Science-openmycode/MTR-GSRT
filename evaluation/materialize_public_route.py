from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np


def load_pickle(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 2)
    if len(points) == 1 or count <= 1:
        return np.repeat(points[:1], max(1, count), axis=0)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 0:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, cumulative[-1], count)
    result = np.empty((count, 2), dtype=float)
    for axis in range(2):
        result[:, axis] = np.interp(targets, cumulative, points[:, axis])
    return result


def route_coordinates(record: dict, edge_geometry: dict[int, object]) -> np.ndarray:
    coordinates: list[tuple[float, float]] = []
    for edge_id in record.get("cpath", ()):
        geometry = edge_geometry.get(int(edge_id))
        if geometry is None:
            return np.empty((0, 2), dtype=float)
        points = list(geometry.coords)
        if coordinates and points:
            previous = np.asarray(coordinates[-1])
            direct = np.linalg.norm(previous - np.asarray(points[0]))
            reverse = np.linalg.norm(previous - np.asarray(points[-1]))
            if reverse < direct:
                points.reverse()
            if np.linalg.norm(previous - np.asarray(points[0])) < 1e-10:
                points = points[1:]
        coordinates.extend(points)
    if not coordinates:
        return np.empty((0, 2), dtype=float)
    # Shapefile geometry is (lon, lat); trajectory convention is (lat, lon).
    return np.asarray([(lat, lon) for lon, lat in coordinates], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a public road reconstruction from saved FMM/STMatch "
            "cpaths, preserving each source trajectory's coordinate count."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--sampling", choices=("source-count", "edge-vertices"), default="source-count")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = load_pickle(args.source)
    matches = load_pickle(args.matches)
    if len(source) < len(matches):
        raise RuntimeError(f"count mismatch: source={len(source)}, matches={len(matches)}")
    source = source[: len(matches)]

    frame = gpd.read_file(args.network)
    if frame.crs is None:
        raise RuntimeError("network CRS is missing")
    frame = frame.to_crs(4326)
    edge_geometry = {int(row.id): row.geometry for row in frame.itertuples()}

    output = []
    routed = 0
    accepted = 0
    fallbacks = 0
    for trajectory, record in zip(source, matches):
        original = np.asarray(trajectory, dtype=float)
        path = route_coordinates(record, edge_geometry)
        if len(path) >= 2:
            output.append(
                sample_polyline(path, max(2, len(original)))
                if args.sampling == "source-count"
                else path
            )
            routed += 1
            accepted += int(bool(record.get("accepted", False)))
        else:
            output.append(original)
            fallbacks += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        pickle.dump(output, handle, pickle.HIGHEST_PROTOCOL)
    manifest = {
        "schema_version": 1,
        "classification": "PUBLIC_POSTPROCESSING_OF_TRAIN_ONLY_SYNTHETIC_RELEASE",
        "router": args.router,
        "policy": (
            "connected nonempty cpath, resampled to source coordinate count; otherwise source trajectory fallback"
            if args.sampling == "source-count"
            else "connected nonempty cpath with every public edge vertex; otherwise source trajectory fallback"
        ),
        "sampling": args.sampling,
        "record_count": len(output),
        "routed_count": routed,
        "accepted_count": accepted,
        "fallback_count": fallbacks,
        "routed_share": routed / len(output),
        "accepted_share": accepted / len(output),
        "inputs": {
            "source": {"path": str(args.source.resolve()), "sha256": sha256(args.source)},
            "matches": {"path": str(args.matches.resolve()), "sha256": sha256(args.matches)},
            "network": {"path": str(args.network.resolve()), "sha256": sha256(args.network)},
        },
        "output": {"path": str(args.out.resolve()), "sha256": sha256(args.out)},
    }
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
