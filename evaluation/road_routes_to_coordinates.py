from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np


def load(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert directed-road routes to WGS84 coordinate trajectories.")
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = gpd.read_file(args.network)
    if frame.crs is None:
        raise RuntimeError("network CRS is missing")
    frame = frame.to_crs(4326)
    geometry = {
        (int(row.source), int(row.target)): row.geometry
        for row in frame.itertuples()
    }
    trajectories = []
    for route in load(args.routes):
        coordinates = []
        for edge in route:
            line = geometry.get(tuple(edge))
            if line is None:
                continue
            points = list(line.coords)
            if coordinates and points and coordinates[-1] == points[0]:
                points = points[1:]
            coordinates.extend(points)
        trajectories.append(np.asarray([(lat, lon) for lon, lat in coordinates], dtype=float))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        pickle.dump(trajectories, handle, pickle.HIGHEST_PROTOCOL)
    print(f"wrote {len(trajectories):,} coordinate trajectories to {args.out.resolve()}")


if __name__ == "__main__":
    main()
