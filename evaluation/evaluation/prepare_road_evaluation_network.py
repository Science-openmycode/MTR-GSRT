"""Build the exact segment-level directed FMM graph used by road metrics.

Unlike a way-level export, every pair of consecutive OSM nodes becomes one
directed edge.  Intersections that occur inside a long OSM way therefore remain
connected in the evaluation graph.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
from pyproj import CRS, Transformer
from shapely.geometry import LineString

if __package__ in {None, ""}:
    for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (
            (_candidate / "configs" / "datasets.json").is_file()
            and (_candidate / "generation" / "common" / "runtime.py").is_file()
        ):
            sys.path.insert(0, str(_candidate))
            break
    else:
        raise RuntimeError("cannot locate public_release root")

from generation.common.runtime import (  # noqa: E402
    PUBLIC_RELEASE,
    dataset_config,
    public_path,
    sha256_file,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-config", required=True)
    result.add_argument("--osm-cache")
    result.add_argument("--out-dir", required=True)
    result.add_argument("--ubodt-delta-m", type=float, default=3000.0)
    result.add_argument("--ubodt-format", choices=("csv", "binary"), default="binary")
    result.add_argument("--skip-ubodt", action="store_true")
    result.add_argument("--ubodt-bin", type=Path)
    result.add_argument("--fmm-runtime-dir", type=Path)
    return result


def utm_crs(bbox: tuple[float, float, float, float]) -> CRS:
    latitude = (bbox[0] + bbox[1]) / 2.0
    longitude = (bbox[2] + bbox[3]) / 2.0
    zone = int(math.floor((longitude + 180.0) / 6.0) + 1)
    epsg = (32600 if latitude >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def _oneway(tags: dict) -> tuple[bool, bool]:
    value = str(tags.get("oneway", "")).lower()
    if value in {"-1", "reverse"}:
        return False, True
    if value in {"yes", "true", "1"}:
        return True, False
    if str(tags.get("junction", "")).lower() == "roundabout" and value not in {"no", "false", "0"}:
        return True, False
    return True, True


def segment_rows(ways: list[dict], projected: CRS) -> list[dict]:
    transformer = Transformer.from_crs("EPSG:4326", projected, always_xy=True)
    node_ids: dict[str, int] = {}
    next_node = 1
    rows: list[dict] = []

    def node_id(value) -> int:
        nonlocal next_node
        key = str(value)
        if key not in node_ids:
            node_ids[key] = next_node
            next_node += 1
        return node_ids[key]

    def append_edge(
        source_key,
        target_key,
        first: tuple[float, float],
        second: tuple[float, float],
        highway: str,
        osm_way,
    ) -> None:
        x0, y0 = transformer.transform(first[1], first[0])
        x1, y1 = transformer.transform(second[1], second[0])
        geometry = LineString([(x0, y0), (x1, y1)])
        if geometry.length <= 0:
            return
        rows.append(
            {
                "id": len(rows) + 1,
                "source": node_id(source_key),
                "target": node_id(target_key),
                "length_m": float(geometry.length),
                "highway": str(highway)[:30],
                "osm_way": str(osm_way)[:30],
                "geometry": geometry,
            }
        )

    for way in ways:
        geometry = [
            (float(point["lat"]), float(point["lon"]))
            for point in way.get("geometry", [])
            if "lat" in point and "lon" in point
        ]
        nodes = list(way.get("nodes", []))
        if len(geometry) < 2 or len(nodes) != len(geometry):
            continue
        tags = way.get("tags", {})
        forward, reverse = _oneway(tags)
        for index in range(len(nodes) - 1):
            if forward:
                append_edge(
                    nodes[index],
                    nodes[index + 1],
                    geometry[index],
                    geometry[index + 1],
                    tags.get("highway", "road"),
                    way.get("id", ""),
                )
            if reverse:
                append_edge(
                    nodes[index + 1],
                    nodes[index],
                    geometry[index + 1],
                    geometry[index],
                    tags.get("highway", "road"),
                    way.get("id", ""),
                )
    graph = nx.DiGraph()
    graph.add_edges_from((row["source"], row["target"]) for row in rows)
    if not graph:
        raise RuntimeError("OSM cache produced an empty segment graph")
    largest = max(nx.weakly_connected_components(graph), key=len)
    retained = [
        row
        for row in rows
        if row["source"] in largest and row["target"] in largest
    ]
    for edge_id, row in enumerate(retained, start=1):
        row["id"] = edge_id
    return retained


def _run_ubodt(
    network: Path,
    output: Path,
    binary: Path,
    runtime_dir: Path | None,
    delta_m: float,
) -> dict:
    stage = Path(tempfile.mkdtemp(prefix="mtr_ubodt_"))
    try:
        for companion in network.parent.glob(network.stem + ".*"):
            shutil.copy2(companion, stage / ("network" + companion.suffix))
        for dependency in (binary, binary.parent / "FMMLIB.dll"):
            if not dependency.is_file():
                raise FileNotFoundError(dependency)
            shutil.copy2(dependency, stage / dependency.name)
        if runtime_dir is not None:
            for name in ("gdal204.dll", "boost_serialization.dll"):
                source = runtime_dir / name
                if not source.is_file():
                    raise FileNotFoundError(source)
                shutil.copy2(source, stage / name)
        staged_output = stage / ("ubodt" + output.suffix.lower())
        command = [
            str(stage / binary.name),
            "--network", str(stage / "network.shp"),
            "--network_id", "id",
            "--source", "source",
            "--target", "target",
            "--delta", str(delta_m),
            "--output", str(staged_output),
            "--use_omp",
        ]
        environment = dict(os.environ)
        runtime_paths = [str(stage)]
        if runtime_dir is not None:
            runtime_paths.append(str(runtime_dir))
        environment["PATH"] = os.pathsep.join(runtime_paths) + os.pathsep + environment.get("PATH", "")
        completed = subprocess.run(
            command,
            cwd=stage,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged_output), output)
        return {
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    args = parser().parse_args()
    from public_utils import (
        filter_osm_ways_by_bbox,
        filter_osm_ways_by_highway,
        load_osm_ways,
    )

    started = time.time()
    config = dataset_config(args.dataset_config)
    bbox = tuple(map(float, config["bbox"]))
    osm_path = public_path(args.osm_cache or config["osm_cache"])
    out_dir = public_path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    ways = filter_osm_ways_by_bbox(load_osm_ways(str(osm_path)), bbox)
    ways = filter_osm_ways_by_highway(ways, list(config.get("osm_highway_classes", [])))
    projected = utm_crs(bbox)
    rows = segment_rows(ways, projected)
    network = out_dir / "network.shp"
    gpd.GeoDataFrame(rows, geometry="geometry", crs=projected).to_file(network, index=False)
    ubodt = out_dir / ("ubodt.bin" if args.ubodt_format == "binary" else "ubodt.txt")
    invocation = None
    if not args.skip_ubodt:
        binary = args.ubodt_bin or (
            PUBLIC_RELEASE
            / "third_party"
            / "fmm-v0.1.1"
            / "cyang-kth-fmm-344fb8c"
            / "build"
            / "Release"
            / "ubodt_gen.exe"
        )
        invocation = _run_ubodt(
            network,
            ubodt,
            binary.resolve(),
            args.fmm_runtime_dir.resolve() if args.fmm_runtime_dir else None,
            args.ubodt_delta_m,
        )
    manifest = {
        "schema_version": 1,
        "dataset": config["name"],
        "construction": "one directed FMM edge per consecutive OSM-node pair",
        "bbox": bbox,
        "projected_crs": projected.to_string(),
        "osm": {"path": str(osm_path), "sha256": sha256_file(osm_path)},
        "selected_osm_ways": len(ways),
        "directed_segment_edges": len(rows),
        "ubodt_delta_m": None if args.skip_ubodt else args.ubodt_delta_m,
        "ubodt_format": None if args.skip_ubodt else args.ubodt_format,
        "ubodt_invocation": invocation,
        "outputs": {
            path.name: sha256_file(path)
            for path in sorted(out_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
        "elapsed_sec": time.time() - started,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
