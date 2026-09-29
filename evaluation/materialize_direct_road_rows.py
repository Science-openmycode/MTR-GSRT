"""Materialize unadapted and nearest-edge routes from saved synthetic coordinates.

Neither mode inserts a connecting road path. A disconnected edge sequence stays
disconnected, so the common road evaluator counts it as a failed full route.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]


def rooted(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_points(raw, limit: int) -> np.ndarray:
    points = np.asarray(raw, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points[:, :2]).all():
        return np.empty((0, 2), dtype=float)
    if len(points) > limit:
        points = points[np.linspace(0, len(points) - 1, limit).round().astype(int)]
    return points[:, :2]


def original_route(points: np.ndarray, node_tree: cKDTree, node_ids: np.ndarray,
                   edge_id: dict[tuple[int, int], int]) -> tuple[int, ...]:
    if len(points) < 2:
        return ()
    indices = node_tree.query(points, k=1)[1]
    nodes = node_ids[np.asarray(indices, dtype=int)]
    nodes = nodes[np.r_[True, np.diff(nodes) != 0]]
    if len(nodes) < 2:
        return ()
    # An unadapted coordinate sequence supplies only directly observed adjacent
    # graph nodes; it is not completed by a shortest path or map matcher.
    ids = [edge_id.get((int(u), int(v))) for u, v in zip(nodes[:-1], nodes[1:])]
    return tuple(int(value) for value in ids) if all(value is not None for value in ids) else ()


def nearest_route(points: np.ndarray, lines: shapely.STRtree, edge_ids: np.ndarray,
                  radius_m: float) -> tuple[int, ...]:
    if len(points) < 2:
        return ()
    query = shapely.points(points[:, 0], points[:, 1])
    indices = np.asarray(lines.nearest(query), dtype=int)
    distances = shapely.distance(query, lines.geometries.take(indices))
    if not np.isfinite(distances).all() or np.any(distances > radius_m):
        return ()
    selected = edge_ids[indices]
    selected = selected[np.r_[True, np.diff(selected) != 0]]
    return tuple(int(value) for value in selected)


def save_route(path: Path, routes: list[tuple[int, ...]]) -> None:
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as packed:
            pickle.dump(routes, packed, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="saved synthetic lat/lon trajectories")
    parser.add_argument("--method", required=True, help="measurement method label")
    parser.add_argument("--mode", choices=("original", "nearest", "both"), default="both")
    parser.add_argument("--network", default="public_assets/beijing_network/network.shp")
    parser.add_argument("--edge-cache", default="public_assets/ordered_portal_route_cache.pkl.gz")
    parser.add_argument("--max-points", type=int, default=32)
    parser.add_argument("--radius-m", type=float, default=200.0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.max_points < 2 or args.radius_m <= 0:
        parser.error("--max-points must be >=2 and --radius-m must be positive")
    source, network_path, cache_path = map(rooted, (args.input, args.network, args.edge_cache))
    output = rooted(args.out_dir)
    if output.exists():
        raise FileExistsError(output)
    with source.open("rb") as stream:
        trajectories = pickle.load(stream)
    if not isinstance(trajectories, list):
        raise ValueError("synthetic input must be a list of trajectories")
    with gzip.open(cache_path, "rb") as stream:
        cache = pickle.load(stream)
    frame = gpd.read_file(network_path)[["id", "source", "target", "geometry"]]
    if frame.crs is None:
        raise ValueError("public road network requires a projected CRS")
    edge_ids = frame["id"].to_numpy(dtype=np.int64)
    edge_nodes = {int(k): tuple(map(int, v)) for k, v in cache["edge_nodes"].items()}
    if set(map(int, edge_ids)) != set(edge_nodes):
        raise ValueError("network edge IDs and public route cache disagree")
    # FMM's source/target IDs and the route cache's graph-node IDs use
    # different enumerations. Edge IDs are shared and bind the two systems.
    edge_id = {pair: key for key, pair in edge_nodes.items()}
    node_xy: dict[int, tuple[float, float]] = {}
    for row in frame.itertuples(index=False):
        source_node, target_node = edge_nodes[int(row.id)]
        node_xy.setdefault(source_node, tuple(row.geometry.coords[0]))
        node_xy.setdefault(target_node, tuple(row.geometry.coords[-1]))
    node_ids = np.asarray(sorted(node_xy), dtype=np.int64)
    node_tree = cKDTree(np.asarray([node_xy[int(key)] for key in node_ids], dtype=float))
    line_tree = shapely.STRtree(frame.geometry.to_numpy())
    transformer = Transformer.from_crs("EPSG:4326", frame.crs, always_xy=True)
    modes = ("original", "nearest") if args.mode == "both" else (args.mode,)
    routes = {mode: [] for mode in modes}
    for record in trajectories:
        latlon = sample_points(record, args.max_points)
        if len(latlon):
            x, y = transformer.transform(latlon[:, 1], latlon[:, 0])
            points = np.column_stack((x, y))
        else:
            points = np.empty((0, 2), dtype=float)
        if "original" in routes:
            routes["original"].append(original_route(points, node_tree, node_ids, edge_id))
        if "nearest" in routes:
            routes["nearest"].append(nearest_route(points, line_tree, edge_ids, args.radius_m))
    output.mkdir(parents=True)
    for mode, values in routes.items():
        name = "Original" if mode == "original" else "Nearest"
        path = output / f"{name}.pkl.gz"
        save_route(path, values)
        (output / f"{name}.manifest.json").write_text(json.dumps({
            "schema": "direct_road_rows_v1", "method": args.method, "mode": mode,
            "input_sha256": digest(source), "network_sha256": digest(network_path),
            "edge_cache_sha256": digest(cache_path), "output_sha256": digest(path),
            "record_count": len(values), "nonempty_count": sum(bool(x) for x in values),
            "max_points": args.max_points, "radius_m": args.radius_m,
            "routing": "no path insertion; consecutive nearest duplicates collapsed",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{args.method}/{name}: {len(values)} slots, {sum(bool(x) for x in values)} nonempty")


if __name__ == "__main__":
    main()
