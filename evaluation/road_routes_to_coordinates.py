from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np


def load(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert directed-road routes to WGS84 coordinate trajectories.")
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--edge-cache", type=Path,
                        help="Public edge cache; maps canonical route endpoints to network edge IDs.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = gpd.read_file(args.network)
    if frame.crs is None:
        raise RuntimeError("network CRS is missing")
    frame = frame.to_crs(4326)
    if args.edge_cache:
        edge_cache = load(args.edge_cache)
        edge_nodes = edge_cache.get("edge_nodes")
        if not isinstance(edge_nodes, dict):
            raise ValueError("edge cache must contain an edge_nodes mapping")
        geometry = {
            tuple(map(int, edge_nodes[int(row.id)])): row.geometry
            for row in frame.itertuples()
            if int(row.id) in edge_nodes
        }
    else:
        geometry = {
            (int(row.source), int(row.target)): row.geometry
            for row in frame.itertuples()
        }
    trajectories = []
    for slot, route in enumerate(load(args.routes)):
        coordinates = []
        for edge in route:
            line = geometry.get(tuple(edge))
            if line is None:
                raise ValueError(f"route slot {slot} contains an edge missing from the public network: {edge}")
            points = list(line.coords)
            if coordinates and points and coordinates[-1] == points[0]:
                points = points[1:]
            coordinates.extend(points)
        if len(coordinates) < 2:
            raise ValueError(f"route slot {slot} does not form a coordinate trajectory")
        trajectories.append(np.asarray([(lat, lon) for lon, lat in coordinates], dtype=float))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        pickle.dump(trajectories, handle, pickle.HIGHEST_PROTOCOL)
    (args.out.parent / (args.out.name + ".manifest.json")).write_text(
        json.dumps({
            "schema": "road-route-coordinate-derivation-v1",
            "route_source": {"path": str(args.routes.resolve()), "sha256": sha256(args.routes)},
            "network": {"path": str(args.network.resolve()), "sha256": sha256(args.network)},
            "edge_cache": ({"path": str(args.edge_cache.resolve()), "sha256": sha256(args.edge_cache)}
                           if args.edge_cache else None),
            "coordinates": {"path": str(args.out.resolve()), "sha256": sha256(args.out)},
            "record_count": len(trajectories),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(trajectories):,} coordinate trajectories to {args.out.resolve()}")


if __name__ == "__main__":
    main()
