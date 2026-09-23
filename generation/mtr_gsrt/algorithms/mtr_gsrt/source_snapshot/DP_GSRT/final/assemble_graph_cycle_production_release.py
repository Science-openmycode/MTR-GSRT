"""Assemble the full graph-cycle DP-GSRT release by sanitized post-processing."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ARA = Path(__file__).resolve().parents[1]
for path in [ARA / "dp_reward_exploration", ARA / "public_release", ARA / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import compact_graph_flow_experiment as compact
import graph_cycle_dp_production_sanitizer as sanitizer
import graph_voronoi_doptimal_support_probe as support
import route_structure_potential_experiment as route
from audit_two_level_semimarkov_dp import build_context
from public_utils import load_osm_ways
from two_level_maxent_projection import fit_projection, kl_trust_region


class CandidateUnpickler(pickle.Unpickler):
    """Allow only the NumPy primitives needed by integer candidate arrays."""

    def find_class(self, module, name):
        safe = {
            ("numpy", "dtype"): np.dtype,
            ("numpy._core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
            ("numpy.core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
        }
        if (module, name) in safe:
            return safe[(module, name)]
        raise pickle.UnpicklingError(
            f"Candidate payload attempted to load forbidden global {module}.{name}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitized-dir", type=Path, required=True)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--chunk-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--production-postprocess", action="store_true", required=True)
    return parser.parse_args()


def load_verified_candidates(
    chunk_root, request_manifest, expected_count, *, chunk_directories=None
):
    records = []
    manifests = []
    expected_start = 0
    root = Path(chunk_root).resolve()
    if chunk_directories is None:
        directories = [
            path for path in root.iterdir() if (path / "chunk_manifest.json").exists()
        ]
    else:
        directories = [Path(path).resolve() for path in chunk_directories]
        if len(set(directories)) != len(directories):
            raise RuntimeError("Candidate chunk snapshot contains duplicate directories")
        if any(path.parent != root for path in directories):
            raise RuntimeError("Candidate chunk snapshot escapes the frozen chunk root")
    chunk_dirs = sorted(
        directories,
        key=lambda path: json.loads((path / "chunk_manifest.json").read_text())["start"],
    )
    for directory in chunk_dirs:
        manifest = json.loads((directory / "chunk_manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["start"] != expected_start
            or manifest["request_sha256"] != request_manifest["request_sha256"]
            or manifest["sanitizer_release_id"] != request_manifest["sanitizer_release_id"]
            or manifest["candidate_count"] != 8
        ):
            raise RuntimeError("Candidate chunk lineage or range mismatch")
        chunk_path = directory / "candidate_nodes.pkl"
        if sanitizer.sha256_file(chunk_path) != manifest["candidate_nodes_sha256"]:
            raise RuntimeError("Candidate chunk hash mismatch")
        with chunk_path.open("rb") as handle:
            chunk = CandidateUnpickler(handle).load()
        if len(chunk) != manifest["end"] - manifest["start"]:
            raise RuntimeError("Candidate chunk length mismatch")
        for record in chunk:
            if not isinstance(record, (list, tuple)) or len(record) != 8:
                raise RuntimeError("Candidate chunk contains a malformed candidate record")
            for candidate in record:
                values = np.asarray(candidate)
                if (
                    values.ndim != 1
                    or values.size == 0
                    or not np.issubdtype(values.dtype, np.integer)
                    or np.any(values < 0)
                ):
                    raise RuntimeError("Candidate path must be a nonempty integer node vector")
        records.extend(chunk)
        manifests.append(manifest)
        expected_start = int(manifest["end"])
    if expected_start != expected_count or len(records) != expected_count:
        raise RuntimeError("Candidate chunks do not cover the full public request range")
    return records, manifests


def raw_candidate_matrix(candidate_nodes, coords, context):
    template = compact.compact_blocks(np.empty((0, 2)), context)
    offsets, dimension = {}, 0
    for name in compact.BLOCK_BUDGETS:
        offsets[name] = dimension
        dimension += template[name].size
    groups, width = len(candidate_nodes), 8
    indptr = np.zeros(groups * width + 1, dtype=np.int64)
    index_parts, value_parts = [], []
    row = 0
    for request, candidates in enumerate(candidate_nodes, start=1):
        if len(candidates) != width:
            raise RuntimeError("Candidate record width mismatch")
        for path in candidates:
            trajectory = coords[np.asarray(path, dtype=int)]
            blocks = compact.compact_blocks(trajectory, context)
            indices, values = [], []
            for name, budget in compact.BLOCK_BUDGETS.items():
                flat = np.asarray(blocks[name], dtype=float).ravel()
                nonzero = np.flatnonzero(flat)
                indices.append(offsets[name] + nonzero)
                values.append(float(budget) / 1_000_000.0 * flat[nonzero])
            current_indices = np.concatenate(indices).astype(np.int32, copy=False)
            current_values = np.concatenate(values)
            index_parts.append(current_indices)
            value_parts.append(current_values)
            row += 1
            indptr[row] = indptr[row - 1] + len(current_indices)
        if request % 500 == 0:
            print(f"[public-postprocess-features] {request}/{groups}", flush=True)
    return csr_matrix(
        (np.concatenate(value_parts), np.concatenate(index_parts), indptr),
        shape=(groups * width, dimension),
    )


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise RuntimeError("Production assembler requires a new output directory")
    sanitizer_manifest = json.loads(
        (args.sanitized_dir / "sanitizer_manifest.json").read_text(encoding="utf-8")
    )
    request_manifest = json.loads(
        (args.request_dir / "request_manifest.json").read_text(encoding="utf-8")
    )
    if request_manifest["sanitizer_release_id"] != sanitizer_manifest["release_id"]:
        raise RuntimeError("Request and sanitizer release IDs differ")
    request_path = args.request_dir / "dp_requests.npz"
    if sanitizer.sha256_file(request_path) != request_manifest["request_sha256"]:
        raise RuntimeError("DP request hash mismatch")
    with np.load(request_path, allow_pickle=False) as archive:
        point_count = np.asarray(archive["point_count"], dtype=int)
    groups = len(point_count)
    if groups != sanitizer.PUBLIC_CAPACITY:
        raise RuntimeError("Production request count mismatch")
    candidate_nodes, chunk_manifests = load_verified_candidates(
        args.chunk_root, request_manifest, groups
    )
    osm_path = (
        Path(__file__).resolve().parents[2]
        / "ara_final"
        / "evidence"
        / "tables"
        / "osm_cache_beijing.pkl"
    )
    if sanitizer.sha256_file(osm_path) != sanitizer_manifest["public_osm"]["sha256"]:
        raise RuntimeError("Public OSM hash mismatch")
    coords, graph = route.prepare_graph(
        [], bbox=support.BBOX, osm_ways=load_osm_ways(osm_path), raw_graph=False
    )
    context, graph_diag = build_context(coords, graph, 24, 96, 4, 3)
    matrix = raw_candidate_matrix(candidate_nodes, coords, context)
    flow_path = args.sanitized_dir / "sanitized_compact_graph_flow.npz"
    if sanitizer.sha256_file(flow_path) != sanitizer_manifest["sanitized_outputs"][flow_path.name]:
        raise RuntimeError("Sanitized graph-flow hash mismatch")
    with np.load(flow_path, allow_pickle=False) as archive:
        released = {name: np.asarray(archive[name], dtype=float) for name in archive.files}
    target = np.concatenate([released[name].ravel() for name in compact.BLOCK_BUDGETS])
    probability, optimizer = fit_projection(
        matrix,
        target,
        groups=groups,
        candidates_per_group=8,
        noise_variance=0.25 + 2.0 * groups / (0.2**2 * sanitizer.PUBLIC_CAPACITY**2),
        max_iterations=160,
    )
    probability, trust = kl_trust_region(probability, 0.20)
    rng = np.random.default_rng(random.SystemRandom().getrandbits(128))
    selected_paths, control_paths, selected, controls = [], [], [], []
    for request in range(groups):
        choice = int(rng.choice(8, p=probability[request]))
        control = int(rng.integers(0, 8))
        selected_paths.append(
            route.resample(
                coords[np.asarray(candidate_nodes[request][choice], dtype=int)],
                max(2, int(point_count[request])),
            )
        )
        control_paths.append(
            route.resample(
                coords[np.asarray(candidate_nodes[request][control], dtype=int)],
                max(2, int(point_count[request])),
            )
        )
        selected.append(choice)
        controls.append(control)
        if (request + 1) % 500 == 0:
            print(f"[public-postprocess-materialize] {request+1}/{groups}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    dp_path = args.out_dir / "graph_cycle_dp_gsrt_full.pkl"
    control_path = args.out_dir / "graph_cycle_public_only_full.pkl"
    with dp_path.open("wb") as handle:
        pickle.dump(selected_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with control_path.open("wb") as handle:
        pickle.dump(control_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
    probability_path = args.out_dir / "selection_probability.npy"
    np.save(probability_path, probability, allow_pickle=False)
    release_manifest = {
        "schema_version": 1,
        "algorithm": "length-conditioned reflection-cycle compact-flow DP-GSRT",
        "privacy": (
            "Both outputs are post-processing of one committed trajectory-record-level "
            "(6/5,0)-DP sanitizer transcript."
        ),
        "sanitizer_release_id": sanitizer_manifest["release_id"],
        "request_sha256": request_manifest["request_sha256"],
        "candidate_chunk_sha256": [item["candidate_nodes_sha256"] for item in chunk_manifests],
        "candidate_count": 8,
        "output_count": groups,
        "optimizer": optimizer,
        "kl_trust_region": trust,
        "graph": graph_diag,
        "selected_candidate_counts": np.bincount(selected, minlength=8).tolist(),
        "control_candidate_counts": np.bincount(controls, minlength=8).tolist(),
        "outputs": {
            dp_path.name: sanitizer.sha256_file(dp_path),
            control_path.name: sanitizer.sha256_file(control_path),
            probability_path.name: sanitizer.sha256_file(probability_path),
        },
    }
    (args.out_dir / "release_manifest.json").write_bytes(
        sanitizer.strict_json_bytes(release_manifest)
    )
    print(str((args.out_dir / "release_manifest.json").resolve()), flush=True)


if __name__ == "__main__":
    main()
