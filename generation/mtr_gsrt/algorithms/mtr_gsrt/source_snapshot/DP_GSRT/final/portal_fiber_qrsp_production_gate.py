"""Fail-closed lineage checks for production portal-fiber QRSP post-processing."""
from __future__ import annotations

import json
import hashlib
import io
import pickle
import random
from fractions import Fraction
from pathlib import Path

import numpy as np

import graph_cycle_dp_production_sanitizer as base_sanitizer
import portal_fiber_route_production_sanitizer as portal_sanitizer
import prepare_graph_cycle_production_requests as request_builder


def _request_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("source", "destination", "point_count", "target_length_km"):
        values = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def sample_committed_base_requests(
    osm_path: Path, *, request_seed: int | None = None,
) -> tuple[dict, dict[str, np.ndarray], str, object]:
    """Sample requests directly from the ledger-anchored base DP transcript."""
    ledger = base_sanitizer.read_privacy_ledger()
    if ledger is None or ledger.get("status") != "committed":
        raise RuntimeError("The base DP privacy event is not committed")
    sanitized_dir = Path(ledger["output_directory"]).resolve()
    seal = base_sanitizer.read_committed_artifact_seal()
    if (
        seal is None
        or seal.get("release_id") != ledger.get("release_id")
        or Path(seal.get("output_directory", "")).resolve() != sanitized_dir
    ):
        raise RuntimeError("The base DP transcript lacks a matching ledger artifact seal")
    manifest_path = sanitized_dir / "sanitizer_manifest.json"
    base_path = sanitized_dir / "sanitized_base_geometry.npz"
    with manifest_path.open("rb") as handle:
        manifest_bytes = handle.read()
    with base_path.open("rb") as handle:
        base_bytes = handle.read()
    if (
        hashlib.sha256(manifest_bytes).hexdigest() != seal["manifest_sha256"]
        or hashlib.sha256(base_bytes).hexdigest() != seal["base_output_sha256"]
    ):
        raise RuntimeError("The committed base DP transcript bytes disagree with the ledger seal")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("production_release") is not True
        or manifest.get("release_id") != ledger.get("release_id")
        or manifest.get("sanitized_outputs", {}).get(base_path.name)
        != seal["base_output_sha256"]
    ):
        raise RuntimeError("The sealed base sanitizer manifest is invalid")
    with np.load(io.BytesIO(base_bytes), allow_pickle=False) as archive:
        measurements = {
            name: np.asarray(archive[name], dtype=float) for name in archive.files
        }
    with osm_path.open("rb") as handle:
        osm_bytes = handle.read()
    if hashlib.sha256(osm_bytes).hexdigest() != manifest["public_osm"]["sha256"]:
        raise RuntimeError("Public OSM bytes disagree with the sealed base transcript")
    try:
        osm = pickle.loads(osm_bytes)
    except Exception as error:
        raise RuntimeError("Sealed public OSM cache is invalid") from error
    rng = np.random.default_rng(
        int(request_seed) if request_seed is not None
        else random.SystemRandom().getrandbits(128)
    )
    arrays = request_builder.sample_request_arrays_from_objects(
        measurements, osm, rng
    )
    required = {"source", "destination", "point_count", "target_length_km"}
    if set(arrays) != required:
        raise RuntimeError("Internally generated DP request schema mismatch")
    sizes = {int(np.asarray(arrays[name]).size) for name in required}
    if sizes != {int(base_sanitizer.PUBLIC_CAPACITY)}:
        raise RuntimeError("Production requests do not equal the frozen full capacity")
    if (
        np.asarray(arrays["source"]).dtype != np.dtype(np.int32)
        or np.asarray(arrays["destination"]).dtype != np.dtype(np.int32)
        or np.asarray(arrays["point_count"]).dtype != np.dtype(np.int32)
        or np.asarray(arrays["target_length_km"]).dtype != np.dtype(float)
        or np.any(np.asarray(arrays["source"]) < 0)
        or np.any(np.asarray(arrays["destination"]) < 0)
        or np.any(np.asarray(arrays["point_count"]) < 2)
        or not np.all(np.isfinite(arrays["target_length_km"]))
        or np.any(np.asarray(arrays["target_length_km"]) < 0.0)
    ):
        raise RuntimeError("Internally generated DP requests are outside the frozen schema")
    lineage = {
        "schema_version": 1,
        "request_count": int(base_sanitizer.PUBLIC_CAPACITY),
        "sanitizer_release_id": manifest["release_id"],
        "sanitizer_directory": str(sanitized_dir),
        "sanitizer_manifest_sha256": seal["manifest_sha256"],
        "public_osm_sha256": hashlib.sha256(osm_bytes).hexdigest(),
        "randomness": (
            f"fixed public post-processing seed {int(request_seed)}"
            if request_seed is not None
            else "fresh SystemRandom-derived post-processing randomness"
        ),
    }
    return lineage, arrays, _request_digest(arrays), osm


def verify_portal_release(
    release_dir: Path,
    public_osm_sha256: str,
    *,
    capacity: int,
    coarse_occupancy_regions: int,
    coarse_transition_regions: int,
    fine_regions: int,
    macro_regions: int,
    phase_count: int,
    portals_per_fine_edge: int,
    expected_block_shapes: dict[str, tuple[int, ...]],
) -> tuple[dict, dict[str, np.ndarray], str, str, Fraction]:
    if int(capacity) != int(base_sanitizer.PUBLIC_CAPACITY):
        raise RuntimeError("Portal-fiber production capacity is not the frozen full capacity")
    manifest_path = release_dir / "portal_fiber_route_manifest.json"
    release_path = release_dir / "portal_fiber_route_dp_release.npz"
    event = portal_sanitizer.read_event()
    seal = portal_sanitizer.read_event_artifact_seal()
    if (
        event is None
        or event.get("status") != "committed"
        or Path(event.get("output_directory", "")).resolve() != release_dir.resolve()
        or seal is None
        or seal.get("release_id") != event.get("release_id")
        or Path(seal.get("output_directory", "")).resolve() != release_dir.resolve()
    ):
        raise RuntimeError("Portal-fiber release lacks a matching committed artifact seal")
    with manifest_path.open("rb") as handle:
        manifest_bytes = handle.read()
    with release_path.open("rb") as handle:
        release_bytes = handle.read()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    release_sha256 = hashlib.sha256(release_bytes).hexdigest()
    if (
        manifest_sha256 != seal.get("manifest_sha256")
        or release_sha256 != seal.get("release_sha256")
    ):
        raise RuntimeError("Portal-fiber transcript bytes disagree with the ledger seal")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("production_release") is not True
        or manifest.get("query_version") != portal_sanitizer.QUERY_VERSION
        or manifest.get("privacy", {}).get("delta") != 0.0
        or manifest.get("privacy", {}).get("adjacency")
        != "fixed-public-capacity add/remove with null slot"
        or manifest.get("privacy", {}).get("random_source") != "OS SystemRandom"
        or manifest.get("privacy", {}).get("integer_l1_sensitivity")
        != portal_sanitizer.ROUTE_LATTICE
        or manifest.get("privacy", {}).get("normalized_l1_sensitivity") != 1
        or manifest.get("privacy", {}).get("exact_query_exported") is not False
    ):
        raise RuntimeError("Portal-fiber manifest is not a certified pure-DP query transcript")
    expected_hash = manifest.get("outputs", {}).get(release_path.name)
    if expected_hash is None or release_sha256 != expected_hash:
        raise RuntimeError("Portal-fiber release hash mismatch")
    if (
        event.get("release_id") != manifest.get("release_id")
    ):
        raise RuntimeError("Portal-fiber release is not bound to the committed privacy event")
    expected_parameters = {
        "capacity": int(capacity),
        "coarse_occupancy_regions": int(coarse_occupancy_regions),
        "coarse_transition_regions": int(coarse_transition_regions),
        "fine_regions": int(fine_regions),
        "macro_regions": int(macro_regions),
        "phase_count": int(phase_count),
        "portals_per_fine_edge": int(portals_per_fine_edge),
        "block_masses": portal_sanitizer.query.BLOCK_BUDGETS,
    }
    public_parameters = manifest.get("public_parameters", {})
    for name, expected in expected_parameters.items():
        if public_parameters.get(name) != expected:
            raise RuntimeError(f"Portal-fiber public parameter mismatch: {name}")
    if manifest.get("public_osm_sha256") != public_osm_sha256:
        raise RuntimeError("Portal-fiber OSM hash mismatch")
    base_commitment = base_sanitizer.read_base_private_input()
    base_manifest = manifest.get("base_private_input_lineage", {})
    if (
        base_commitment is None
        or base_manifest.get("same_private_source_verified") is not True
        or base_manifest.get("commitment_domain") != base_commitment.get("domain")
        or base_manifest.get("base_release_id") != base_commitment.get("release_id")
        or base_manifest.get("commitment_sha256")
        != portal_sanitizer.base_commitment_digest(base_commitment)
    ):
        raise RuntimeError("Portal-fiber query is not bound to the sealed base private input")
    for name, path in portal_sanitizer.frozen_source_paths().items():
        if manifest.get("source_sha256", {}).get(name) != portal_sanitizer.sha256_file(path):
            raise RuntimeError(f"Portal-fiber source lineage mismatch: {name}")
    released = {}
    with np.load(io.BytesIO(release_bytes), allow_pickle=False) as archive:
        if tuple(archive.files) != tuple(portal_sanitizer.BLOCK_NAMES):
            raise RuntimeError("Portal-fiber block schema mismatch")
        for name in archive.files:
            values = np.asarray(archive[name], dtype=float)
            released[name] = values.copy()
            if tuple(values.shape) != tuple(expected_block_shapes.get(name, ())):
                raise RuntimeError(f"Portal-fiber block dimension mismatch: {name}")
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise RuntimeError("Portal-fiber release contains an invalid projected value")
            expected_mass = (
                int(capacity)
                * int(portal_sanitizer.query.BLOCK_BUDGETS[name])
                / float(portal_sanitizer.ROUTE_LATTICE)
            )
            if not np.isclose(values.sum(), expected_mass, rtol=1e-10, atol=1e-8):
                raise RuntimeError(f"Portal-fiber projected block mass mismatch: {name}")
        if (
            int(public_parameters.get("portal_atoms", -1))
            != int(np.asarray(archive["portal_fiber_flow"]).size - 1)
        ):
            raise RuntimeError("Portal-fiber atom count mismatch")
    epsilon = Fraction(manifest["privacy"]["epsilon_route_rational"])
    if (
        epsilon != portal_sanitizer.PRODUCTION_ROUTE_EPSILON
        or event.get("epsilon_rational")
        != f"{epsilon.numerator}/{epsilon.denominator}"
    ):
        raise RuntimeError("Portal-fiber epsilon disagrees with the privacy ledger")
    return manifest, released, release_sha256, manifest_sha256, epsilon
