"""Convert consumer-visible road witnesses to public node polylines.

This is deterministic evaluation post-processing.  It reads no private data
and performs no trajectory synthesis.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (_candidate / "configs" / "datasets.json").is_file():
            sys.path.insert(0, str(_candidate))
            break
    else:
        raise RuntimeError("cannot locate public_release root")

from generation.common.runtime import (  # noqa: E402
    dataset_config,
    public_path,
    sha256_file,
    write_json,
)


def _public_graph(config: dict) -> tuple[np.ndarray, dict[int, set[int]], Path]:
    from generation.common.runtime import add_runtime_paths

    add_runtime_paths()
    from public_utils import filter_osm_ways_by_bbox, filter_osm_ways_by_highway, load_osm_ways

    final_dir = public_path("src/mtr/DP_GSRT/final")
    if str(final_dir) not in sys.path:
        sys.path.insert(0, str(final_dir))
    import route_structure_potential_experiment as route

    bbox = tuple(map(float, config["bbox"]))
    osm_path = public_path(config["osm_cache"])
    osm = filter_osm_ways_by_bbox(load_osm_ways(str(osm_path)), bbox)
    osm = filter_osm_ways_by_highway(osm, list(config.get("osm_highway_classes", [])))
    coordinates, graph = route.prepare_graph([], bbox=bbox, osm_ways=osm, raw_graph=False)
    adjacency = {
        int(source): {int(target) for target, _weight in neighbors}
        for source, neighbors in graph.items()
    }
    return np.asarray(coordinates, dtype=float), adjacency, osm_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--witness", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = dataset_config(args.dataset_config)
    witness_path = public_path(args.witness)
    out_path = public_path(args.out)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")
    coordinates, adjacency, osm_path = _public_graph(config)
    with witness_path.open("rb") as handle:
        witnesses = pickle.load(handle)
    if not isinstance(witnesses, list):
        raise ValueError("witness file must contain a list")

    trajectories: list[np.ndarray] = []
    edge_count = 0
    for slot, witness in enumerate(witnesses):
        if not isinstance(witness, dict) or int(witness.get("slot", -1)) != slot:
            raise ValueError(f"invalid witness slot {slot}")
        nodes = tuple(map(int, witness.get("node_sequence", ())))
        edges = tuple(tuple(map(int, edge)) for edge in witness.get("directed_edges", ()))
        if len(nodes) < 2 or edges != tuple(zip(nodes[:-1], nodes[1:])):
            raise ValueError(f"witness sequence mismatch at slot {slot}")
        if any(target not in adjacency.get(source, set()) for source, target in edges):
            raise ValueError(f"non-edge in witness slot {slot}")
        trajectories.append(np.asarray(coordinates[list(nodes)], dtype=float))
        edge_count += len(edges)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "schema_version": 1,
        "classification": "PUBLIC_WITNESS_EVALUATION_ADAPTER_NO_SYNTHESIS",
        "dataset": str(config["name"]),
        "records": len(trajectories),
        "directed_edges": edge_count,
        "inputs": {
            "witness": {"path": str(witness_path), "sha256": sha256_file(witness_path)},
            "osm": {"path": str(osm_path), "sha256": sha256_file(osm_path)},
        },
        "adapter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "output": {"path": str(out_path), "sha256": sha256_file(out_path)},
    }
    write_json(out_path.with_suffix(out_path.suffix + ".manifest.json"), manifest)
    print(json.dumps({"status": "complete", **manifest["output"]}, indent=2))


if __name__ == "__main__":
    main()
