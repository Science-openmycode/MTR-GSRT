"""Single-event production sanitizer for the portal-fiber route query."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sqlite3
import uuid
from fractions import Fraction
from pathlib import Path

import numpy as np

import audit_two_level_semimarkov_dp as audit
import certified_discrete_dp as certified
import dp_graph_voronoi_release as projection
import graph_voronoi_doptimal_support_probe as support
import graph_cycle_dp_production_sanitizer as base_lineage
import nested_quotient_graph as nested
import portal_fiber_route_release_development as query
import route_structure_potential_experiment as route
import dp_two_level_semimarkov_release as quantization_core
import public_utils as public_module
from dp_two_level_semimarkov_release import ROUTE_LATTICE
from public_utils import load_osm_ways


PRIVACY_DOMAIN = "geolife-full:portal-fiber-route-query-v1"
QUERY_VERSION = "portal-fiber-route-query-v1"
BLOCK_NAMES = tuple(query.BLOCK_BUDGETS)
LEDGER_PATH = Path(r"C:\ProgramData\ARA_DP_GSRT\privacy_ledger.sqlite3")
ARTIFACT_SEAL_TABLE = "portal_fiber_artifact_seals"
PRODUCTION_ROUTE_EPSILON = Fraction(1, 5)


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_fraction(value: str) -> Fraction:
    parsed = Fraction(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("epsilon must be positive")
    return parsed


def validate_production_epsilon(epsilon: Fraction) -> None:
    if epsilon != PRODUCTION_ROUTE_EPSILON:
        raise RuntimeError("Production portal q5 epsilon is frozen to 1/5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--osm", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=parse_positive_int, required=True)
    parser.add_argument("--epsilon-route", type=parse_fraction, required=True)
    parser.add_argument("--coarse-occupancy-regions", type=parse_positive_int, required=True)
    parser.add_argument("--coarse-transition-regions", type=parse_positive_int, required=True)
    parser.add_argument("--fine-regions", type=parse_positive_int, required=True)
    parser.add_argument("--macro-regions", type=parse_positive_int, required=True)
    parser.add_argument("--phase-count", type=parse_positive_int, required=True)
    parser.add_argument("--portals-per-fine-edge", type=parse_positive_int, required=True)
    parser.add_argument("--production-release", action="store_true", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_source_paths() -> dict[str, Path]:
    return {
        "sanitizer": Path(__file__).resolve(),
        "query_core": Path(query.__file__).resolve(),
        "noise_core": Path(certified.__file__).resolve(),
        "quantization_core": Path(quantization_core.__file__).resolve(),
        "projection_core": Path(projection.__file__).resolve(),
        "coarse_context_core": Path(audit.__file__).resolve(),
        "nested_context_core": Path(nested.__file__).resolve(),
        "transition_core": Path(query.endpoint_release.__file__).resolve(),
        "semimarkov_query_core": Path(query.query.__file__).resolve(),
        "graph_core": Path(route.__file__).resolve(),
        "support_core": Path(support.__file__).resolve(),
        "public_utils": Path(public_module.__file__).resolve(),
        "base_lineage_core": Path(base_lineage.__file__).resolve(),
    }


def ledger_connection() -> sqlite3.Connection:
    """Open the one process-independent privacy ledger at its frozen system path."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(LEDGER_PATH, timeout=30.0, isolation_level=None)
    journal_mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
    connection.execute("PRAGMA synchronous=FULL")
    synchronous = int(connection.execute("PRAGMA synchronous").fetchone()[0])
    if journal_mode != "wal" or synchronous != 2:
        connection.close()
        raise RuntimeError("Global privacy ledger requires SQLite WAL and synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS privacy_events (
            domain TEXT PRIMARY KEY,
            release_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            epsilon_rational TEXT NOT NULL,
            output_directory TEXT NOT NULL
        )
        """
    )
    return connection


def reserve_event(release_id: str, epsilon: Fraction, out_dir: Path) -> None:
    connection = ledger_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO privacy_events(
                domain, release_id, status, epsilon_rational, output_directory
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                PRIVACY_DOMAIN,
                release_id,
                "reserved_fail_closed_budget_consumed",
                f"{epsilon.numerator}/{epsilon.denominator}",
                str(out_dir.resolve()),
            ),
        )
        connection.execute("COMMIT")
    except sqlite3.IntegrityError as error:
        connection.execute("ROLLBACK")
        raise RuntimeError("Portal-fiber privacy event has already been consumed") from error
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def finalize_event(release_id: str, out_dir: Path) -> None:
    connection = ledger_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(
            """
            UPDATE privacy_events SET status = ?
            WHERE domain = ? AND release_id = ? AND status = ? AND output_directory = ?
            """,
            (
                "committed",
                PRIVACY_DOMAIN,
                release_id,
                "reserved_fail_closed_budget_consumed",
                str(out_dir.resolve()),
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Portal-fiber privacy reservation is missing or mismatched")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def read_event() -> dict | None:
    if not LEDGER_PATH.exists():
        return None
    connection = ledger_connection()
    try:
        row = connection.execute(
            """
            SELECT domain, release_id, status, epsilon_rational, output_directory
            FROM privacy_events WHERE domain = ?
            """,
            (PRIVACY_DOMAIN,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return dict(
        zip(
            ("domain", "release_id", "status", "epsilon_rational", "output_directory"),
            row,
        )
    )


def ensure_artifact_seal_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ARTIFACT_SEAL_TABLE} (
            domain TEXT PRIMARY KEY,
            release_id TEXT NOT NULL UNIQUE,
            output_directory TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            release_sha256 TEXT NOT NULL
        )
        """
    )


def seal_event_artifacts(release_id: str, out_dir: Path) -> None:
    manifest = out_dir / "portal_fiber_route_manifest.json"
    release = out_dir / "portal_fiber_route_dp_release.npz"
    connection = ledger_connection()
    try:
        ensure_artifact_seal_table(connection)
        connection.execute("BEGIN IMMEDIATE")
        event = connection.execute(
            "SELECT release_id, status, output_directory FROM privacy_events WHERE domain = ?",
            (PRIVACY_DOMAIN,),
        ).fetchone()
        if event != (
            release_id,
            "reserved_fail_closed_budget_consumed",
            str(out_dir.resolve()),
        ):
            raise RuntimeError("Portal artifact seal has no matching reserved event")
        connection.execute(
            f"""
            INSERT INTO {ARTIFACT_SEAL_TABLE}(
                domain, release_id, output_directory, manifest_sha256, release_sha256
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                PRIVACY_DOMAIN,
                release_id,
                str(out_dir.resolve()),
                sha256_file(manifest),
                sha256_file(release),
            ),
        )
        connection.execute("COMMIT")
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
    finally:
        connection.close()


def read_event_artifact_seal() -> dict | None:
    if not LEDGER_PATH.exists():
        return None
    connection = ledger_connection()
    try:
        ensure_artifact_seal_table(connection)
        row = connection.execute(
            f"""
            SELECT domain, release_id, output_directory, manifest_sha256, release_sha256
            FROM {ARTIFACT_SEAL_TABLE} WHERE domain = ?
            """,
            (PRIVACY_DOMAIN,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return dict(
        zip(
            ("domain", "release_id", "output_directory", "manifest_sha256", "release_sha256"),
            row,
        )
    )


def write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def base_commitment_digest(commitment: dict) -> str:
    names = (
        "domain",
        "release_id",
        "source_path",
        "source_sha256",
        "source_bytes",
        "source_mtime_ns",
        "raw_record_count",
        "valid_record_count",
        "loader_source_sha256",
    )
    payload = json.dumps(
        {name: commitment[name] for name in names},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_base_private_input_snapshot(
    real_path: Path, capacity: int
) -> tuple[dict, list[np.ndarray]]:
    """Validate and normalize one immutable byte snapshot of the sealed private file."""
    commitment = base_lineage.read_base_private_input()
    if commitment is None:
        raise RuntimeError("The base DP mechanism has not sealed a canonical private input")
    resolved = real_path.resolve()
    if int(capacity) != int(base_lineage.PUBLIC_CAPACITY):
        raise RuntimeError("Portal q5 capacity must equal the frozen base public capacity")
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError("Portal q5 private input must be a regular non-symlink file")
    with resolved.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        payload = handle.read()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if (
        resolved != Path(commitment["source_path"]).resolve()
        or payload_sha256 != commitment["source_sha256"]
        or int(stat.st_size) != int(commitment["source_bytes"])
        or int(stat.st_mtime_ns) != int(commitment["source_mtime_ns"])
    ):
        raise RuntimeError("Portal q5 private input disagrees with the sealed base database")
    if len(str(commitment["loader_source_sha256"])) != 64:
        raise RuntimeError("The base preprocessing identity is malformed")
    try:
        raw = pickle.loads(payload)
    except Exception as error:
        raise RuntimeError("Sealed private input is not a valid trajectory pickle") from error
    support.validate_raw_record_container(raw, capacity)
    if len(raw) != int(commitment["raw_record_count"]):
        raise RuntimeError("Portal q5 raw record count disagrees with the sealed base database")
    # The base and q5 preprocessors are independently frozen per-record maps.
    # Composition is taken on raw fixed slots, so their source hashes need not
    # be equal; q5's own preprocessor is bound by frozen_source_sha256 below.
    records = support.normalize_trajectory_records(raw)
    if len(records) > len(raw):
        raise RuntimeError("Portal q5 preprocessing expanded the fixed raw slot database")
    return commitment, records


def main() -> None:
    args = parse_args()
    validate_production_epsilon(args.epsilon_route)
    if args.out_dir.exists():
        raise RuntimeError("Production sanitizer requires a new output directory")
    if LEDGER_PATH.resolve().is_relative_to(args.out_dir.resolve()):
        raise RuntimeError("Privacy ledger cannot be stored inside the release directory")
    if sum(query.BLOCK_BUDGETS.values()) != ROUTE_LATTICE:
        raise RuntimeError("Frozen block masses do not sum to Q")
    if args.coarse_transition_regions < args.coarse_occupancy_regions:
        raise RuntimeError("Coarse transition partition cannot be coarser than occupancy support")
    if args.fine_regions < args.coarse_transition_regions:
        raise RuntimeError("Fine partition cannot be coarser than the coarse transition partition")
    max_block_mass = max(int(value) for value in query.BLOCK_BUDGETS.values())
    if int(args.capacity) > np.iinfo(np.int64).max // max_block_mass:
        raise RuntimeError("Fixed capacity and block mass exceed the int64 aggregation domain")
    frozen_osm_sha256 = sha256_file(args.osm)
    frozen_source_sha256 = {
        name: sha256_file(path) for name, path in frozen_source_paths().items()
    }

    coords, graph = route.prepare_graph(
        [], bbox=support.BBOX, osm_ways=load_osm_ways(args.osm), raw_graph=False
    )
    coarse_context, _ = audit.build_context(
        coords,
        graph,
        args.coarse_occupancy_regions,
        args.coarse_transition_regions,
        args.macro_regions,
        args.phase_count,
    )
    fine_context, _, _ = nested.build_nested_context(
        coords,
        graph,
        args.coarse_occupancy_regions,
        args.fine_regions,
        args.macro_regions,
        args.phase_count,
    )
    portals = query.diverse_portals(
        graph,
        coords,
        fine_context.node_fine,
        count=args.portals_per_fine_edge,
    )
    offsets, portal_count = query.portal_layout(portals)
    templates = {
        "coarse24_occupancy": np.zeros(args.coarse_occupancy_regions),
        "fine384_occupancy": np.zeros(args.fine_regions),
        "fine96_flow": np.zeros(len(coarse_context.fine_edge_index) + 1),
        "fine384_flow": np.zeros(len(fine_context.fine_edge_index) + 1),
        "portal_fiber_flow": np.zeros(portal_count + 1),
    }
    if set(templates) != set(BLOCK_NAMES):
        raise RuntimeError("Production query schema disagrees with the frozen block order")

    release_id = str(uuid.uuid4())
    base_private_input, records = load_base_private_input_snapshot(
        args.real, args.capacity
    )
    if not isinstance(records, (list, tuple)) or len(records) > args.capacity:
        raise RuntimeError("Private input is outside the fixed public slot domain")

    # Reserve the one-shot event only after every non-query preflight succeeds.
    reserve_event(release_id, args.epsilon_route, args.out_dir)
    totals = {name: np.zeros_like(value, dtype=np.int64) for name, value in templates.items()}
    for record in records:
        blocks = query.trajectory_blocks(
            record, coarse_context, fine_context, portals, offsets, coords
        )
        quantized = query.quantize_trajectory_blocks(blocks)
        if quantized is None:
            continue
        for name in BLOCK_NAMES:
            totals[name] += quantized[name]

    exact = np.concatenate([totals[name].ravel() for name in BLOCK_NAMES])
    noisy, sampler = certified.add_exact_discrete_laplace(
        exact,
        epsilon_numerator=args.epsilon_route.numerator,
        epsilon_denominator=args.epsilon_route.denominator,
        sensitivity=ROUTE_LATTICE,
        rng=random.SystemRandom(),
    )
    if sampler["random_source"] != "OS SystemRandom":
        raise RuntimeError("Production sanitizer did not obtain system randomness")

    released, cursor = {}, 0
    for name in BLOCK_NAMES:
        size = int(totals[name].size)
        block = noisy[cursor : cursor + size].reshape(totals[name].shape)
        released[name] = (
            projection.project_simplex(
                block.astype(float), float(args.capacity) * query.BLOCK_BUDGETS[name]
            )
            / float(ROUTE_LATTICE)
        )
        cursor += size

    if sha256_file(args.osm) != frozen_osm_sha256 or any(
        sha256_file(path) != frozen_source_sha256[name]
        for name, path in frozen_source_paths().items()
    ):
        raise RuntimeError("Frozen source or public OSM changed during the q5 production query")

    staging = args.out_dir.parent / f".{args.out_dir.name}.staging-{release_id}"
    staging.mkdir(parents=True, exist_ok=False)
    release_path = staging / "portal_fiber_route_dp_release.npz"
    with release_path.open("xb") as handle:
        np.savez_compressed(handle, **released)
        handle.flush()
        os.fsync(handle.fileno())
    manifest = {
        "schema_version": 1,
        "query_version": QUERY_VERSION,
        "release_id": release_id,
        "production_release": True,
        "privacy": {
            "adjacency": "fixed-public-capacity add/remove with null slot",
            "epsilon_route_rational": (
                f"{args.epsilon_route.numerator}/{args.epsilon_route.denominator}"
            ),
            "delta": 0.0,
            "integer_l1_sensitivity": ROUTE_LATTICE,
            "normalized_l1_sensitivity": 1,
            "random_source": sampler["random_source"],
            "exact_query_exported": False,
        },
        "public_parameters": {
            "capacity": args.capacity,
            "coarse_occupancy_regions": args.coarse_occupancy_regions,
            "coarse_transition_regions": args.coarse_transition_regions,
            "fine_regions": args.fine_regions,
            "macro_regions": args.macro_regions,
            "phase_count": args.phase_count,
            "portals_per_fine_edge": args.portals_per_fine_edge,
            "block_masses": query.BLOCK_BUDGETS,
            "portal_atoms": portal_count,
        },
        "public_osm_sha256": frozen_osm_sha256,
        "base_private_input_lineage": {
            "commitment_domain": base_private_input["domain"],
            "base_release_id": base_private_input["release_id"],
            "commitment_sha256": base_commitment_digest(base_private_input),
            "same_private_source_verified": True,
        },
        "source_sha256": frozen_source_sha256,
        "global_privacy_ledger": {
            "domain": PRIVACY_DOMAIN,
            "path": str(LEDGER_PATH),
            "status_after_commit": "committed",
        },
        "outputs": {release_path.name: sha256_file(release_path)},
        "forbidden_outputs": ["exact_query.npz", "private_record_count", "random_seed"],
    }
    manifest_path = staging / "portal_fiber_route_manifest.json"
    write_fsynced(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )
    os.replace(staging, args.out_dir)
    seal_event_artifacts(release_id, args.out_dir)
    finalize_event(release_id, args.out_dir)
    print(str((args.out_dir / manifest_path.name).resolve()), flush=True)


if __name__ == "__main__":
    main()
