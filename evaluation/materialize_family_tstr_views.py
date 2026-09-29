"""Convert four public family-route lists into strict-TSTR coordinate views.

The routes must already have been derived from the same named synthetic
training release.  Invalid route slots retain that synthetic record; no real
trajectory, validation record, or test record enters this post-processing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np

from route_metric_core import load_pickle


ROOT = Path(__file__).resolve().parents[1]
ROUTERS = {
    "Family_additive": "family_additive",
    "Family_residual": "family_residual",
    "Family_length-OT": "family_length_ot",
    "Self-carrier_reweight": "self_carrier_reweight",
}


def rooted(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def sample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) < 2:
        return np.repeat(points[:1], max(2, count), axis=0) if len(points) else points.reshape(0, 2)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 0:
        return np.repeat(points[:1], max(2, count), axis=0)
    targets = np.linspace(0.0, cumulative[-1], max(2, count))
    result = np.empty((len(targets), 2), dtype=float)
    for axis in range(2):
        result[:, axis] = np.interp(targets, cumulative, points[:, axis])
    return result


def node_coordinates(network: Path, edge_nodes: dict) -> dict[int, tuple[float, float]]:
    frame = gpd.read_file(network).to_crs(4326)
    coordinates: dict[int, tuple[float, float]] = {}
    for row in frame.itertuples(index=False):
        edge = edge_nodes.get(int(row.id))
        if edge is None or row.geometry is None or row.geometry.is_empty:
            continue
        points = list(row.geometry.coords)
        if not points:
            continue
        u, v = map(int, edge)
        coordinates.setdefault(u, (float(points[0][1]), float(points[0][0])))
        coordinates.setdefault(v, (float(points[-1][1]), float(points[-1][0])))
    return coordinates


def route_points(route, coordinates: dict[int, tuple[float, float]]) -> np.ndarray:
    if not route or any(left[1] != right[0] for left, right in zip(route, route[1:])):
        return np.empty((0, 2), dtype=float)
    nodes = [int(route[0][0]), *(int(edge[1]) for edge in route)]
    if any(node not in coordinates for node in nodes):
        return np.empty((0, 2), dtype=float)
    return np.asarray([coordinates[node] for node in nodes], dtype=float)


def write_view(path: Path, values: list[np.ndarray]) -> None:
    with path.open("xb") as handle:
        pickle.dump(values, handle, pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Named train-only synthetic coordinate release")
    parser.add_argument("--stmatch-route", required=True,
                        help="The same method's STMatch carrier, with its input-binding manifest")
    parser.add_argument("--family-dir", required=True, help="Four route lists from materialize_family_routers.py")
    parser.add_argument("--network", required=True)
    parser.add_argument("--edge-cache", default="public_assets/ordered_portal_route_cache.pkl.gz")
    parser.add_argument("--method", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    source = rooted(args.source)
    stmatch = rooted(args.stmatch_route)
    family_dir = rooted(args.family_dir)
    network = rooted(args.network)
    edge_cache = rooted(args.edge_cache)
    output = rooted(args.out_dir)
    if output.exists():
        raise FileExistsError(output)
    source_rows = load_pickle(source)
    if not source_rows:
        raise ValueError("Synthetic source contains no records")
    cache = load_pickle(edge_cache)
    edge_nodes = {int(key): tuple(map(int, value)) for key, value in cache["edge_nodes"].items()}
    coordinates = node_coordinates(network, edge_nodes)
    source_sha = digest(source)
    network_sha = digest(network)
    cache_sha = digest(edge_cache)
    stmatch_sha = digest(stmatch)
    stmatch_manifest = stmatch.with_name(stmatch.name.removesuffix(".pkl.gz") + ".manifest.json")
    match_binding = json.loads(stmatch_manifest.read_text(encoding="utf-8-sig"))
    for key, expected in (("input_sha256", source_sha), ("network_sha256", network_sha),
                          ("output_sha256", stmatch_sha)):
        if match_binding.get(key) != expected:
            raise ValueError(f"STMatch carrier does not bind this release/network: {key}")
    prepared = {}
    for router, slug in ROUTERS.items():
        route_file = family_dir / f"{router}.pkl.gz"
        route_manifest = family_dir / f"{router}.manifest.json"
        metadata = json.loads(route_manifest.read_text(encoding="utf-8-sig"))
        if metadata.get("method") != args.method or metadata.get("output_sha256") != digest(route_file):
            raise ValueError(f"Family-route manifest mismatch: {router}")
        if metadata.get("edge_cache_sha256") != cache_sha:
            raise ValueError(f"Family-route edge cache differs: {router}")
        if metadata.get("stmatch_carrier_sha256") != stmatch_sha:
            raise ValueError(f"Family route uses a different STMatch carrier: {router}")
        routes = load_pickle(route_file)
        if len(routes) != len(source_rows):
            raise ValueError(f"{router} has {len(routes)} routes for {len(source_rows)} records")
        generic, road, connected = [], [], 0
        for trajectory, route in zip(source_rows, routes):
            original = np.asarray(trajectory, dtype=float)
            points = route_points(route, coordinates)
            if len(points) >= 2:
                generic.append(sample_polyline(points, len(original)))
                road.append(points)
                connected += 1
            else:
                generic.append(original)
                road.append(original)
        prepared[slug] = (route_file, generic, road, connected)
    output.mkdir(parents=True)
    for slug, (route_file, generic, road, connected) in prepared.items():
        generic_path = output / f"{slug}_generic.pkl"
        road_path = output / f"{slug}_road.pkl"
        write_view(generic_path, generic)
        write_view(road_path, road)
        base = {
            "schema": "tstr-family-coordinate-view-v1",
            "method": args.method,
            "router": slug,
            "record_count": len(source_rows),
            "connected_route_count": connected,
            "fallback_count": len(source_rows) - connected,
            "source": {"path": str(source), "sha256": source_sha},
            "stmatch_source": {"path": str(stmatch), "sha256": stmatch_sha},
            "route_source": {"path": str(route_file), "sha256": digest(route_file)},
            "network": {"path": str(network), "sha256": network_sha},
            "edge_cache": {"path": str(edge_cache), "sha256": cache_sha},
        }
        for view, path in (("generic", generic_path), ("road", road_path)):
            manifest = {**base, "view": view,
                        "coordinates": {"path": str(path), "sha256": digest(path)}}
            (output / f"{path.name}.manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{args.method}/{slug}: {len(source_rows)} slots, {connected} connected routes")


if __name__ == "__main__":
    main()
