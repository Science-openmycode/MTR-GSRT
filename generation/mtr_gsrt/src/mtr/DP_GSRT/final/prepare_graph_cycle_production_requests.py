"""Prepare full DP requests using only a committed sanitizer transcript."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ARA = Path(__file__).resolve().parents[1]
for path in [ARA / "dp_reward_exploration", ARA / "public_release", ARA / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import geometric_base_requests as geometric
import graph_cycle_dp_production_sanitizer as sanitizer
import graph_voronoi_doptimal_support_probe as support
import route_structure_potential_experiment as route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitized-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--production-postprocess", action="store_true", required=True)
    return parser.parse_args()


def verify_sanitizer(directory: Path) -> dict:
    raise RuntimeError(
        "Directory-based verification is disabled; use the ledger-sealed in-memory gate"
    )


def sample_request_arrays(
    sanitized_dir: Path,
    manifest: dict,
    osm_path: Path,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    raise RuntimeError(
        "Path-based sampling is disabled; consume authenticated in-memory snapshots only"
    )


def sample_request_arrays_from_objects(
    measurements: dict[str, np.ndarray],
    osm,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Sample requests from already authenticated in-memory DP and public snapshots."""
    coords, _ = route.prepare_graph([], bbox=support.BBOX, osm_ways=osm, raw_graph=False)
    fine_nodes = support.farthest_point_landmarks(coords, 256)
    fine_coords = coords[np.asarray(fine_nodes, dtype=int)]
    fine_tree = cKDTree(fine_coords)
    coarse_nodes = support.farthest_point_landmarks(coords, 24)
    coarse_coords = coords[np.asarray(coarse_nodes, dtype=int)]
    coarse_tree = cKDTree(coarse_coords)
    fine_to_coarse = np.asarray(coarse_tree.query(fine_coords)[1], dtype=int)
    graph_to_fine = np.asarray(fine_tree.query(coords)[1], dtype=int)
    graph_to_coarse = np.asarray(coarse_tree.query(coords)[1], dtype=int)
    metric_scale = np.asarray([111.32, 111.32 * np.cos(np.deg2rad(40.0))])
    metric_coords = coords * metric_scale
    requests = geometric.sample_geometric_requests(
        measurements,
        fine_to_coarse,
        support.standardized_xy(coarse_coords),
        fine_nodes,
        fine_coords,
        np.arange(len(coords), dtype=int),
        coords,
        graph_to_fine,
        graph_to_coarse,
        cKDTree(metric_coords),
        metric_coords,
        count=sanitizer.PUBLIC_CAPACITY,
        rng=rng,
    )
    return {
        "source": np.asarray([item[0] for item in requests], dtype=np.int32),
        "destination": np.asarray([item[1] for item in requests], dtype=np.int32),
        "point_count": np.asarray([item[2] for item in requests], dtype=np.int32),
        "target_length_km": np.asarray([item[3] for item in requests], dtype=float),
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise RuntimeError("Production request postprocessor requires a new output directory")
    osm_path = (
        Path(__file__).resolve().parents[2]
        / "ara_final"
        / "evidence"
        / "tables"
        / "osm_cache_beijing.pkl"
    )
    import portal_fiber_qrsp_production_gate as production_gate

    lineage, arrays, request_digest, _ = production_gate.sample_committed_base_requests(
        osm_path
    )
    if Path(lineage["sanitizer_directory"]).resolve() != args.sanitized_dir.resolve():
        raise RuntimeError("Requested sanitizer directory is not the ledger-sealed transcript")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    request_path = args.out_dir / "dp_requests.npz"
    np.savez_compressed(
        request_path,
        **arrays,
    )
    request_manifest = {
        "schema_version": 1,
        "algorithm": "graph-cycle production DP request postprocessor",
        "privacy": "post-processing of the committed (6/5,0)-DP sanitizer transcript",
        "sanitizer_release_id": lineage["sanitizer_release_id"],
        "sanitizer_manifest_sha256": lineage["sanitizer_manifest_sha256"],
        "request_count": sanitizer.PUBLIC_CAPACITY,
        "request_array_digest": request_digest,
        "request_sha256": sanitizer.sha256_file(request_path),
        "public_osm_sha256": lineage["public_osm_sha256"],
        "randomness": "fresh data-independent SystemRandom-derived post-processing randomness",
    }
    (args.out_dir / "request_manifest.json").write_bytes(
        sanitizer.strict_json_bytes(request_manifest)
    )
    print(str((args.out_dir / "request_manifest.json").resolve()), flush=True)


if __name__ == "__main__":
    main()
