"""Fail-closed production sanitizer for graph-cycle DP-GSRT."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import pickle
import random
import sqlite3
import sys
import uuid
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import cKDTree

import audit_two_level_semimarkov_dp as audit_module
import certified_discrete_dp as discrete_dp
import compact_graph_flow_experiment as compact
import dp_graph_voronoi_release as base
import dp_graph_flow_routing_probe as flow_probe
import dp_two_level_semimarkov_release as dp_release
import geometric_base_requests as geometric
import graph_voronoi_doptimal_support_probe as support
import fixed_slot_domain as fixed_slot
import public_utils
import route_structure_potential_experiment as route
import two_level_semimarkov_query as semimarkov_query
from public_utils import load_osm_ways


PUBLIC_CAPACITY = 17_123
LEDGER_PATH = Path(r"C:\ProgramData\ARA_DP_GSRT\privacy_ledger.sqlite3")
LEDGER_DOMAIN = "geolife-full-17123:graph-cycle-compact-flow-v1"
BASE_PRIVATE_INPUT_TABLE = "base_private_input_commitments"
ARTIFACT_SEAL_TABLE = "committed_artifact_seals"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--private-source", type=Path, default=support.DEFAULT_REAL)
    parser.add_argument("--production-release", action="store_true", required=True)
    return parser.parse_args()


def strict_json_bytes(value) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False).encode("utf-8")


def write_fsynced(path: Path, payload: bytes, *, exclusive: bool = False) -> None:
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def public_edges(values) -> list[float | str]:
    return [float(value) if np.isfinite(value) else "inf" for value in values]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def ledger_connection() -> sqlite3.Connection:
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
        CREATE TABLE IF NOT EXISTS releases (
            domain TEXT PRIMARY KEY,
            release_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            epsilon_rational TEXT NOT NULL,
            delta TEXT NOT NULL,
            output_directory TEXT NOT NULL,
            staging_directory TEXT NOT NULL
        )
        """
    )
    return connection


def ensure_base_private_input_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {BASE_PRIVATE_INPUT_TABLE} (
            domain TEXT PRIMARY KEY,
            release_id TEXT NOT NULL UNIQUE,
            source_path TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            source_bytes INTEGER NOT NULL,
            source_mtime_ns INTEGER NOT NULL,
            raw_record_count INTEGER NOT NULL,
            valid_record_count INTEGER NOT NULL,
            loader_source_sha256 TEXT NOT NULL
        )
        """
    )


def ensure_artifact_seal_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ARTIFACT_SEAL_TABLE} (
            domain TEXT PRIMARY KEY,
            release_id TEXT NOT NULL UNIQUE,
            output_directory TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            base_output_sha256 TEXT NOT NULL,
            flow_output_sha256 TEXT NOT NULL
        )
        """
    )


def seal_committed_artifacts(release_id: str, output_directory: Path) -> None:
    """Persist immutable hashes for the exact bytes later post-processors may consume."""
    manifest = output_directory / "sanitizer_manifest.json"
    base_output = output_directory / "sanitized_base_geometry.npz"
    flow_output = output_directory / "sanitized_compact_graph_flow.npz"
    connection = ledger_connection()
    try:
        ensure_artifact_seal_table(connection)
        connection.execute("BEGIN IMMEDIATE")
        release = connection.execute(
            "SELECT release_id, status, output_directory FROM releases WHERE domain = ?",
            (LEDGER_DOMAIN,),
        ).fetchone()
        if release != (
            release_id,
            "reserved_fail_closed_budget_consumed",
            str(output_directory.resolve()),
        ):
            raise RuntimeError("Artifact seal has no matching reserved base release")
        connection.execute(
            f"""
            INSERT INTO {ARTIFACT_SEAL_TABLE}(
                domain, release_id, output_directory, manifest_sha256,
                base_output_sha256, flow_output_sha256
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                LEDGER_DOMAIN,
                release_id,
                str(output_directory.resolve()),
                sha256_file(manifest),
                sha256_file(base_output),
                sha256_file(flow_output),
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


def read_committed_artifact_seal() -> dict | None:
    if not LEDGER_PATH.exists():
        return None
    connection = ledger_connection()
    try:
        ensure_artifact_seal_table(connection)
        row = connection.execute(
            f"""
            SELECT domain, release_id, output_directory, manifest_sha256,
                   base_output_sha256, flow_output_sha256
            FROM {ARTIFACT_SEAL_TABLE} WHERE domain = ?
            """,
            (LEDGER_DOMAIN,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    keys = (
        "domain",
        "release_id",
        "output_directory",
        "manifest_sha256",
        "base_output_sha256",
        "flow_output_sha256",
    )
    return dict(zip(keys, row))


def seal_base_private_input(
    release_id: str,
    source_path: Path,
    *,
    source_sha256: str,
    source_bytes: int,
    source_mtime_ns: int,
    raw_record_count: int,
    valid_record_count: int,
    loader_source_sha256: str,
) -> None:
    """Bind the base query to one canonical private database inside the ACL ledger."""
    fixed_slot.validate_record_count(raw_record_count, PUBLIC_CAPACITY)
    fixed_slot.validate_record_count(valid_record_count, raw_record_count)
    connection = ledger_connection()
    try:
        ensure_base_private_input_table(connection)
        connection.execute("BEGIN IMMEDIATE")
        release = connection.execute(
            "SELECT release_id, status FROM releases WHERE domain = ?",
            (LEDGER_DOMAIN,),
        ).fetchone()
        if release != (release_id, "reserved_fail_closed_budget_consumed"):
            raise RuntimeError("Base private-input commitment has no matching reservation")
        connection.execute(
            f"""
            INSERT INTO {BASE_PRIVATE_INPUT_TABLE}(
                domain, release_id, source_path, source_sha256, source_bytes,
                source_mtime_ns, raw_record_count, valid_record_count,
                loader_source_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                LEDGER_DOMAIN,
                release_id,
                str(source_path.resolve()),
                source_sha256,
                int(source_bytes),
                int(source_mtime_ns),
                int(raw_record_count),
                int(valid_record_count),
                loader_source_sha256,
            ),
        )
        connection.execute("COMMIT")
    except Exception as error:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        if isinstance(error, sqlite3.IntegrityError):
            raise RuntimeError("Base private input is already ledger-sealed") from error
        raise
    finally:
        connection.close()


def read_base_private_input() -> dict | None:
    if not LEDGER_PATH.exists():
        return None
    connection = ledger_connection()
    try:
        ensure_base_private_input_table(connection)
        row = connection.execute(
            f"""
            SELECT domain, release_id, source_path, source_sha256, source_bytes,
                   source_mtime_ns, raw_record_count, valid_record_count,
                   loader_source_sha256
            FROM {BASE_PRIVATE_INPUT_TABLE} WHERE domain = ?
            """,
            (LEDGER_DOMAIN,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    keys = [
        "domain",
        "release_id",
        "source_path",
        "source_sha256",
        "source_bytes",
        "source_mtime_ns",
        "raw_record_count",
        "valid_record_count",
        "loader_source_sha256",
    ]
    return dict(zip(keys, row))


def reserve_privacy_ledger(release_id: str, out_dir: Path, staging: Path) -> None:
    connection = ledger_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO releases(
                domain, release_id, status, epsilon_rational, delta,
                output_directory, staging_directory
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                LEDGER_DOMAIN,
                release_id,
                "reserved_fail_closed_budget_consumed",
                "6/5",
                "0",
                str(out_dir.resolve()),
                str(staging.resolve()),
            ),
        )
        connection.execute("COMMIT")
    except sqlite3.IntegrityError as error:
        connection.execute("ROLLBACK")
        raise RuntimeError(
            "Global privacy ledger already reserves this release domain; repeat release is forbidden"
        ) from error
    finally:
        connection.close()


def finalize_privacy_ledger(release_id: str, out_dir: Path) -> None:
    connection = ledger_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(
            """
            UPDATE releases SET status = ?, output_directory = ?
            WHERE domain = ? AND release_id = ?
            """,
            ("committed", str(out_dir.resolve()), LEDGER_DOMAIN, release_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Privacy ledger reservation is missing or mismatched")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def read_privacy_ledger() -> dict | None:
    if not LEDGER_PATH.exists():
        return None
    connection = ledger_connection()
    try:
        row = connection.execute(
            """
            SELECT domain, release_id, status, epsilon_rational, delta,
                   output_directory, staging_directory
            FROM releases WHERE domain = ?
            """,
            (LEDGER_DOMAIN,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    keys = [
        "domain",
        "release_id",
        "status",
        "epsilon_rational",
        "delta",
        "output_directory",
        "staging_directory",
    ]
    return dict(zip(keys, row))


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise RuntimeError("Production sanitizer requires a new, nonexistent output directory")
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    release_id = str(uuid.uuid4())
    staging = args.out_dir.parent / f".{args.out_dir.name}.staging-{release_id}"
    source_files = [
        Path(__file__).resolve(),
        Path(geometric.__file__).resolve(),
        Path(compact.__file__).resolve(),
        Path(discrete_dp.__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(dp_release.__file__).resolve(),
        Path(semimarkov_query.__file__).resolve(),
        Path(audit_module.__file__).resolve(),
        Path(flow_probe.__file__).resolve(),
        Path(support.__file__).resolve(),
        Path(fixed_slot.__file__).resolve(),
        Path(route.__file__).resolve(),
        Path(public_utils.__file__).resolve(),
    ]
    source_sha256_before = {display_path(path): sha256_file(path) for path in source_files}
    osm_path = (
        Path(__file__).resolve().parents[2]
        / "ara_final"
        / "evidence"
        / "tables"
        / "osm_cache_beijing.pkl"
    )
    osm_sha256_before = sha256_file(osm_path)
    reserve_privacy_ledger(release_id, args.out_dir, staging)
    private_source = args.private_source.resolve()
    if not private_source.is_file() or private_source.is_symlink():
        raise RuntimeError("Private source must be a regular local file")
    with private_source.open("rb") as handle:
        private_stat_before = os.fstat(handle.fileno())
        private_payload = handle.read()
    private_hash_before = hashlib.sha256(private_payload).hexdigest()
    try:
        raw_private_records = pickle.loads(private_payload)
    except Exception as error:
        raise RuntimeError("Private source is not a valid trajectory pickle") from error
    support.validate_raw_record_container(raw_private_records, PUBLIC_CAPACITY)
    real = support.normalize_trajectory_records(raw_private_records)
    valid_record_count, _ = fixed_slot.validate_trajectory_records(real, PUBLIC_CAPACITY)
    seal_base_private_input(
        release_id,
        private_source,
        source_sha256=private_hash_before,
        source_bytes=private_stat_before.st_size,
        source_mtime_ns=private_stat_before.st_mtime_ns,
        raw_record_count=len(raw_private_records),
        valid_record_count=valid_record_count,
        loader_source_sha256=sha256_file(Path(support.__file__).resolve()),
    )

    osm = load_osm_ways(osm_path)
    coords, graph = route.prepare_graph([], bbox=support.BBOX, osm_ways=osm, raw_graph=False)
    fine_nodes = support.farthest_point_landmarks(coords, 256)
    fine_coords = coords[np.asarray(fine_nodes, dtype=int)]
    fine_tree = cKDTree(fine_coords)
    coarse_nodes = support.farthest_point_landmarks(coords, 24)
    coarse_coords = coords[np.asarray(coarse_nodes, dtype=int)]
    coarse_tree = cKDTree(coarse_coords)
    fine_to_coarse = np.asarray(coarse_tree.query(fine_coords)[1], dtype=int)

    base_release = geometric.fit_geometric_measurements(
        real,
        fine_tree,
        coarse_tree,
        support.standardized_xy(coarse_coords),
        fine_count=256,
        coarse_count=24,
        capacity=PUBLIC_CAPACITY,
        exact_rng=random.SystemRandom(),
    )
    base_release["od"], sinkhorn = base.sinkhorn_od_projection(
        base_release["od"], base_release["endpoint"], fine_to_coarse, 24
    )
    base_samplers = base_release.pop("samplers")

    context, graph_diag = audit_module.build_context(coords, graph, 24, 96, 4, 3)
    flow_totals = compact.aggregate(real, context)
    flow_release, flow_sampler = compact.release(
        flow_totals, PUBLIC_CAPACITY, random.SystemRandom(), Fraction(1, 5)
    )

    staging.mkdir(parents=False, exist_ok=False)
    base_path = staging / "sanitized_base_geometry.npz"
    flow_path = staging / "sanitized_compact_graph_flow.npz"
    np.savez_compressed(base_path, **base_release)
    np.savez_compressed(flow_path, **flow_release)
    fsync_file(base_path)
    fsync_file(flow_path)

    source_sha256_after = {display_path(path): sha256_file(path) for path in source_files}
    private_stat_after = private_source.stat()
    if (
        source_sha256_after != source_sha256_before
        or sha256_file(osm_path) != osm_sha256_before
        or sha256_file(private_source) != private_hash_before
        or private_stat_after.st_size != private_stat_before.st_size
        or private_stat_after.st_mtime_ns != private_stat_before.st_mtime_ns
    ):
        raise RuntimeError("Frozen source or public OSM changed during the production query")
    manifest = {
        "schema_version": 1,
        "release_id": release_id,
        "production_release": True,
        "certified_release": False,
        "independent_audit_status": "pending",
        "privacy": {
            "adjacency": (
                "fixed 17,123-slot add/remove adjacency: one slot changes between one "
                "complete trajectory record and a null record"
            ),
            "protected_unit": "one complete trajectory record",
            "multi_record_user_note": (
                "If one person contributes multiple records, user-level protection follows "
                "only by group privacy or prior one-record-per-user clipping."
            ),
            "epsilon_base_rational": "1/1",
            "epsilon_graph_flow_rational": "1/5",
            "epsilon_total_rational": "6/5",
            "delta": 0.0,
            "composition": "sequential composition",
            "postprocessing_condition": (
                "Any routing algorithm that accesses only these sanitized releases, fixed public "
                "objects, and data-independent randomness is privacy-free post-processing."
            ),
        },
        "fixed_domain": {
            "public_capacity": PUBLIC_CAPACITY,
            "input_record_count": "any integer from 0 through public_capacity",
            "null_slot_representation": "omitted records are canonical implicit null slots",
            "fine_endpoint_landmarks": 256,
            "coarse_regions": 24,
            "fine_graph_regions": 96,
        },
        "planned_postprocessor_parameters": {"candidate_count": 8},
        "base_query": {
            "budgets_rational": {
                name: f"{value.numerator}/{value.denominator}"
                for name, value in geometric.BASE_BUDGETS.items()
            },
            "samplers": base_samplers,
            "sinkhorn": sinkhorn,
            "point_count_edges": geometric.POINT_COUNT_EDGES.tolist(),
            "chord_edges_km": public_edges(geometric.CHORD_EDGES_KM),
            "path_edges_km": public_edges(geometric.PATH_EDGES_KM),
        },
        "graph_flow_query": {
            "integer_l1_sensitivity": 1_000_000,
            "block_budgets": compact.BLOCK_BUDGETS,
            "block_dimensions": {
                name: int(values.size) for name, values in flow_release.items()
            },
            "sampler": flow_sampler,
            "graph": graph_diag,
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sqlite_runtime": sqlite3.sqlite_version,
            "sqlite_wrapper": sqlite3.version,
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        },
        "query_specification_version": "graph-cycle-compact-flow-v1",
        "global_privacy_ledger": {
            "domain": LEDGER_DOMAIN,
            "backend": "SQLite transaction with synchronous=FULL",
            "path": str(LEDGER_PATH),
            "reservation_status": "reserved_fail_closed_budget_consumed",
            "staging_directory": str(staging.resolve()),
        },
        "public_osm": {"path": display_path(osm_path), "sha256": osm_sha256_before},
        "sanitized_outputs": {
            base_path.name: sha256_file(base_path),
            flow_path.name: sha256_file(flow_path),
        },
        "source_sha256": source_sha256_before,
        "forbidden_outputs": [
            "raw trajectories",
            "exact query vectors",
            "observed private norms",
            "private record-count progress",
            "reproducible noise seeds",
        ],
    }
    manifest_path = staging / "sanitizer_manifest.json"
    write_fsynced(manifest_path, strict_json_bytes(manifest))
    os.replace(staging, args.out_dir)
    seal_committed_artifacts(release_id, args.out_dir)
    finalize_privacy_ledger(release_id, args.out_dir)
    print(str((args.out_dir / manifest_path.name).resolve()), flush=True)


if __name__ == "__main__":
    main()
