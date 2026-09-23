"""Road-object and route-choice metrics on one frozen directed FMM network.

The module deliberately separates three questions:

* ``road_realizable`` asks whether a coordinate record can be matched to one
  connected directed road path without dropping most observations.
* ``witness_coordinate_*`` asks whether an MTR coordinate record agrees with
  its consumer-visible directed-path witness.
* the route metrics compare *matched public-road edge sequences*.  They never
  rename grid-cell transitions as road edges.

All missing/failed matches remain in the denominator.
"""
from __future__ import annotations

import csv
import json
import math
import os
import pickle
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from shapely import wkt


@dataclass(frozen=True)
class MatchRecord:
    source_index: int
    cpath: tuple[int, ...]
    opath: tuple[int, ...]
    residual_m: tuple[float, ...]
    connected: bool
    observation_share: float
    accepted: bool


def sample_trajectory(trajectory: np.ndarray, max_points: int) -> np.ndarray:
    arr = np.asarray(trajectory, dtype=float)
    if len(arr) <= max_points:
        return arr
    indices = np.linspace(0, len(arr) - 1, max_points).round().astype(int)
    return arr[np.unique(indices)]


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in (value or "").split(",") if item.strip().lstrip("-").isdigit())


def _copy_shapefile(source: Path, target_stem: Path) -> None:
    companions = list(source.parent.glob(source.stem + ".*"))
    if not companions:
        raise FileNotFoundError(f"FMM network is missing: {source}")
    for companion in companions:
        shutil.copy2(companion, target_stem.with_suffix(companion.suffix))


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _copy_runtime(
    fmm: Path,
    ubodt: Path | None,
    runtime_dir: Path | None,
    stage: Path,
) -> tuple[Path, Path | None]:
    binaries = [fmm]
    if os.name == "nt":
        binaries.append(fmm.parent / "FMMLIB.dll")
    else:
        linux_libraries = [
            fmm.parent / "libFMMLIB.so",
            fmm.parent.parent / "build_linux_release" / "libFMMLIB.so",
        ]
        library = next((path for path in linux_libraries if path.is_file()), None)
        if library is None:
            raise FileNotFoundError(
                "FMM runtime file is missing: expected libFMMLIB.so beside the "
                f"binary or in build_linux_release (binary={fmm})"
            )
        binaries.append(library)
    if ubodt is not None:
        binaries.append(ubodt)
    for binary in binaries:
        if not binary.is_file():
            raise FileNotFoundError(f"FMM runtime file is missing: {binary}")
        shutil.copy2(binary, stage / binary.name)
    if runtime_dir is not None:
        for name in ("gdal204.dll", "boost_serialization.dll"):
            source = runtime_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"FMM runtime dependency is missing: {source}")
            shutil.copy2(source, stage / name)
    return stage / fmm.name, (stage / ubodt.name if ubodt is not None else None)


def network_metadata(network: Path) -> tuple[dict[int, tuple[int, int]], CRS]:
    frame = gpd.read_file(network)
    required = {"id", "source", "target", "geometry"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"FMM network lacks required columns: {sorted(missing)}")
    if frame.crs is None:
        raise ValueError("FMM network must declare its projected CRS")
    endpoints = {
        int(row.id): (int(row.source), int(row.target))
        for row in frame.itertuples()
    }
    return endpoints, CRS(frame.crs)


def _is_connected(cpath: tuple[int, ...], endpoints: dict[int, tuple[int, int]]) -> bool:
    if not cpath or any(edge not in endpoints for edge in cpath):
        return False
    return all(
        endpoints[left][1] == endpoints[right][0]
        for left, right in zip(cpath[:-1], cpath[1:])
    )


def _projected_residuals(
    sampled_latlon: np.ndarray,
    projected_wkt: str,
    to_projected: Transformer,
) -> tuple[float, ...]:
    try:
        matched = np.asarray(wkt.loads(projected_wkt).coords, dtype=float)
    except Exception:
        return tuple()
    if len(matched) != len(sampled_latlon):
        return tuple()
    source = np.asarray(
        [to_projected.transform(float(lon), float(lat)) for lat, lon in sampled_latlon],
        dtype=float,
    )
    return tuple(np.linalg.norm(source - matched, axis=1).astype(float).tolist())


def run_fmm(
    trajectories: list[np.ndarray],
    network: Path,
    ubodt_table: Path | None,
    fmm: Path,
    ubodt_binary: Path | None,
    runtime_dir: Path | None,
    max_points: int,
    radius_m: float,
    gps_error_m: float,
    candidates: int,
    min_observation_share: float,
    work_root: Path | None = None,
    route_only: bool = False,
) -> tuple[list[MatchRecord], dict]:
    """Map-match every trajectory; failed rows are returned as rejected records."""
    endpoints, projected_crs = network_metadata(network)
    to_projected = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    sampled = [sample_trajectory(np.asarray(t, dtype=float), max_points) for t in trajectories]
    parent = str(work_root) if work_root is not None else None
    stage = Path(tempfile.mkdtemp(prefix="mtr_road_metric_", dir=parent))
    try:
        stage_network = stage / "network.shp"
        _copy_shapefile(network, stage_network)
        stage_fmm, _ = _copy_runtime(fmm, ubodt_binary, runtime_dir, stage)
        stage_ubodt = None
        if ubodt_table is not None:
            stage_ubodt = stage / ("ubodt" + ubodt_table.suffix.lower())
            _link_or_copy(ubodt_table, stage_ubodt)
        gps_path = stage / "gps.csv"
        with gps_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=";", lineterminator="\n")
            writer.writerow(["id", "geom"])
            for index, trajectory in enumerate(sampled):
                coords = [
                    to_projected.transform(float(lon), float(lat))
                    for lat, lon in trajectory
                ]
                geometry = "LINESTRING(" + ",".join(f"{x:.3f} {y:.3f}" for x, y in coords) + ")"
                writer.writerow([index, geometry])
        output_path = stage / "matches.csv"
        command = [
            str(stage_fmm),
            "--network", str(stage_network),
            "--network_id", "id",
            "--source", "source",
            "--target", "target",
            "--gps", str(gps_path),
            "--gps_id", "id",
            "--gps_geom", "geom",
            "--candidates", str(candidates),
            "--radius", str(radius_m),
            "--error", str(gps_error_m),
            "--output", str(output_path),
            "--output_fields", ("cpath" if route_only else "opath,cpath,pgeom"),
            "--use_omp",
        ]
        if stage_ubodt is not None:
            command[1:1] = ["--ubodt", str(stage_ubodt)]
        environment = dict(os.environ)
        runtime_path = [str(stage)]
        if runtime_dir is not None:
            runtime_path.append(str(runtime_dir))
        environment["PATH"] = os.pathsep.join(runtime_path) + os.pathsep + environment.get("PATH", "")
        if os.name != "nt":
            environment["LD_LIBRARY_PATH"] = (
                str(stage)
                + os.pathsep
                + environment.get("LD_LIBRARY_PATH", "")
            )
        try:
            completed = subprocess.run(
                command,
                cwd=stage,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "FMM failed with exit code "
                f"{error.returncode}; stdout={error.stdout!r}; stderr={error.stderr!r}"
            ) from error
        if not output_path.is_file():
            raise RuntimeError(
                "FMM returned success without creating its output; "
                f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
            )
        parsed: dict[int, MatchRecord] = {}
        with output_path.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                source_index = int(row["id"])
                cpath = _parse_ints(row.get("cpath", ""))
                opath = _parse_ints(row.get("opath", ""))
                residuals = (
                    tuple()
                    if route_only else
                    _projected_residuals(sampled[source_index], row.get("pgeom", ""), to_projected)
                )
                observation_share = (
                    float(bool(cpath))
                    if route_only else
                    len(opath) / max(len(sampled[source_index]), 1)
                )
                connected = _is_connected(cpath, endpoints)
                residual_ok = (
                    True if route_only else
                    bool(residuals) and float(np.quantile(residuals, 0.95)) <= radius_m
                )
                accepted = (
                    connected
                    and observation_share >= min_observation_share
                    and residual_ok
                )
                parsed[source_index] = MatchRecord(
                    source_index=source_index,
                    cpath=cpath,
                    opath=opath,
                    residual_m=residuals,
                    connected=connected,
                    observation_share=float(observation_share),
                    accepted=bool(accepted),
                )
        records = [
            parsed.get(
                index,
                MatchRecord(index, tuple(), tuple(), tuple(), False, 0.0, False),
            )
            for index in range(len(trajectories))
        ]
        invocation = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "network_crs": projected_crs.to_string(),
            "max_points": max_points,
            "radius_m": radius_m,
            "gps_error_m": gps_error_m,
            "candidates": candidates,
            "min_observation_share": min_observation_share,
            "route_only": route_only,
        }
        return records, invocation
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def road_realizability_metrics(records: list[MatchRecord]) -> dict[str, float]:
    accepted = [record for record in records if record.accepted]
    residuals = [
        distance
        for record in accepted
        for distance in record.residual_m
    ]
    return {
        "road_realizable": float(np.mean([record.accepted for record in records])) if records else 0.0,
        "road_connected_path_rate": float(np.mean([record.connected for record in records])) if records else 0.0,
        "road_observation_coverage_mean": float(np.mean([record.observation_share for record in records])) if records else 0.0,
        "road_projection_residual_median_m": float(np.median(residuals)) if residuals else math.nan,
        "road_projection_residual_p95_m": float(np.quantile(residuals, 0.95)) if residuals else math.nan,
    }


def _xy_m(points: np.ndarray, latitude0: float) -> np.ndarray:
    earth_m = 6_371_008.8
    arr = np.asarray(points, dtype=float)
    return np.column_stack(
        (
            np.deg2rad(arr[:, 1]) * earth_m * math.cos(math.radians(latitude0)),
            np.deg2rad(arr[:, 0]) * earth_m,
        )
    )


def _point_segment_projection(
    points: np.ndarray,
    polyline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return point-to-polyline distances and monotone arc positions."""
    if len(polyline) < 2:
        return np.full(len(points), np.inf), np.full(len(points), np.nan)
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    lengths2 = np.einsum("ij,ij->i", vectors, vectors)
    lengths = np.sqrt(lengths2)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    midpoints = (starts + polyline[1:]) / 2.0
    tree = cKDTree(midpoints)
    k = min(12, len(starts))
    _, candidates = tree.query(points, k=k)
    candidates = np.asarray(candidates, dtype=int)
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    distances = np.empty(len(points), dtype=float)
    positions = np.empty(len(points), dtype=float)
    for index, (point, segment_ids) in enumerate(zip(points, candidates)):
        segment_starts = starts[segment_ids]
        segment_vectors = vectors[segment_ids]
        denominator = np.maximum(lengths2[segment_ids], 1e-12)
        fractions = np.clip(
            np.einsum("ij,ij->i", point - segment_starts, segment_vectors) / denominator,
            0.0,
            1.0,
        )
        projected = segment_starts + fractions[:, None] * segment_vectors
        candidate_distances = np.linalg.norm(projected - point, axis=1)
        best = int(np.argmin(candidate_distances))
        segment = int(segment_ids[best])
        distances[index] = float(candidate_distances[best])
        positions[index] = float(cumulative[segment] + fractions[best] * lengths[segment])
    return distances, positions


def load_witness_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        package = json.loads(path.read_text(encoding="utf-8"))
        records = package.get("records")
    else:
        with path.open("rb") as handle:
            records = pickle.load(handle)
    if not isinstance(records, list):
        raise ValueError("witness release must contain a record list")
    return records


def witness_coordinate_metrics(
    trajectories: list[np.ndarray],
    witnesses: list[dict],
    public_node_coordinates: np.ndarray,
    max_points: int,
    distance_threshold_m: float,
    monotone_pair_threshold: float,
) -> dict[str, float]:
    if len(trajectories) != len(witnesses):
        raise ValueError("witness count must equal coordinate release count")
    consistent = []
    all_within = []
    p95_distances = []
    maximum_distances = []
    monotone_shares = []
    endpoint_distances = []
    for slot, (trajectory, witness) in enumerate(zip(trajectories, witnesses)):
        if int(witness.get("slot", -1)) != slot:
            raise ValueError(f"witness slot mismatch at {slot}")
        nodes = np.asarray(witness.get("node_sequence", []), dtype=int)
        if len(nodes) < 2 or np.any(nodes < 0) or np.any(nodes >= len(public_node_coordinates)):
            consistent.append(False)
            all_within.append(False)
            p95_distances.append(math.inf)
            maximum_distances.append(math.inf)
            monotone_shares.append(0.0)
            endpoint_distances.append(math.inf)
            continue
        sampled = sample_trajectory(np.asarray(trajectory, dtype=float), max_points)
        latitude0 = float(np.mean(sampled[:, 0]))
        point_xy = _xy_m(sampled, latitude0)
        path_xy = _xy_m(public_node_coordinates[nodes], latitude0)
        distances, positions = _point_segment_projection(point_xy, path_xy)
        monotone_share = (
            float(np.mean(np.diff(positions) >= -1e-6))
            if len(positions) > 1
            else 1.0
        )
        p95 = float(np.quantile(distances, 0.95))
        maximum = float(np.max(distances))
        endpoint = float(max(distances[0], distances[-1]))
        p95_distances.append(p95)
        maximum_distances.append(maximum)
        monotone_shares.append(monotone_share)
        endpoint_distances.append(endpoint)
        all_within.append(maximum <= distance_threshold_m)
        consistent.append(
            p95 <= distance_threshold_m
            and endpoint <= distance_threshold_m
            and monotone_share >= monotone_pair_threshold
        )
    finite_p95 = np.asarray([value for value in p95_distances if math.isfinite(value)], dtype=float)
    finite_max = np.asarray([value for value in maximum_distances if math.isfinite(value)], dtype=float)
    return {
        "witness_coordinate_consistent": float(np.mean(consistent)) if consistent else 0.0,
        "witness_coordinate_all_within": float(np.mean(all_within)) if all_within else 0.0,
        "witness_coordinate_p95_median_m": float(np.median(finite_p95)) if len(finite_p95) else math.nan,
        "witness_coordinate_p95_p95_m": float(np.quantile(finite_p95, 0.95)) if len(finite_p95) else math.nan,
        "witness_coordinate_max_p95_m": float(np.quantile(finite_max, 0.95)) if len(finite_max) else math.nan,
        "witness_coordinate_monotone_mean": float(np.mean(monotone_shares)) if monotone_shares else 0.0,
    }


def witness_node_trajectories(
    witnesses: list[dict],
    public_node_coordinates: np.ndarray,
    max_points: int,
) -> list[np.ndarray]:
    output = []
    for witness in witnesses:
        nodes = np.asarray(witness.get("node_sequence", []), dtype=int)
        if len(nodes) < 2 or np.any(nodes < 0) or np.any(nodes >= len(public_node_coordinates)):
            output.append(np.empty((0, 2), dtype=float))
        else:
            output.append(sample_trajectory(public_node_coordinates[nodes], max_points))
    return output


def _cell_id(point: np.ndarray, bbox: tuple[float, float, float, float], grid: int) -> int:
    lat0, lat1, lon0, lon1 = bbox
    row = int(np.clip((point[0] - lat0) / max(lat1 - lat0, 1e-12) * grid, 0, grid - 1))
    col = int(np.clip((point[1] - lon0) / max(lon1 - lon0, 1e-12) * grid, 0, grid - 1))
    return row * grid + col


def _od_key(trajectory: np.ndarray, bbox: tuple[float, float, float, float], grid: int) -> int:
    arr = np.asarray(trajectory, dtype=float)
    cells = grid * grid
    return _cell_id(arr[0], bbox, grid) * cells + _cell_id(arr[-1], bbox, grid)


def _edge_f1(reference: Iterable[int], candidate: Iterable[int]) -> float:
    reference_set = set(reference)
    candidate_set = set(candidate)
    if not reference_set or not candidate_set:
        return 0.0
    overlap = len(reference_set.intersection(candidate_set))
    return 2.0 * overlap / (len(reference_set) + len(candidate_set))


def _jsd_counters(left: Counter, right: Counter) -> float:
    keys = sorted(set(left).union(right), key=str)
    if not keys:
        return 0.0
    a = np.asarray([float(left[key]) for key in keys], dtype=float)
    b = np.asarray([float(right[key]) for key in keys], dtype=float)
    if a.sum() <= 0 or b.sum() <= 0:
        return 1.0
    return float(jensenshannon(a / a.sum(), b / b.sum(), base=2.0) ** 2)


def _conditional_transition_counts(
    trajectories: list[np.ndarray],
    records: list[MatchRecord],
    bbox: tuple[float, float, float, float],
    od_grid: int,
) -> dict[int, Counter]:
    output: dict[int, Counter] = defaultdict(Counter)
    for trajectory, record in zip(trajectories, records):
        od = _od_key(trajectory, bbox, od_grid)
        if not record.accepted or len(record.cpath) < 2:
            output[od][("__INVALID__",)] += 1.0
            continue
        transitions = list(zip(record.cpath[:-1], record.cpath[1:]))
        weight = 1.0 / len(transitions)
        for transition in transitions:
            output[od][transition] += weight
    return output


def od_conditioned_transition_jsd(
    real_trajectories: list[np.ndarray],
    real_records: list[MatchRecord],
    synthetic_trajectories: list[np.ndarray],
    synthetic_records: list[MatchRecord],
    bbox: tuple[float, float, float, float],
    od_grid: int,
) -> float:
    real_counts = _conditional_transition_counts(real_trajectories, real_records, bbox, od_grid)
    synthetic_counts = _conditional_transition_counts(
        synthetic_trajectories, synthetic_records, bbox, od_grid
    )
    weights = Counter(
        _od_key(trajectory, bbox, od_grid)
        for trajectory in real_trajectories
    )
    total = max(sum(weights.values()), 1)
    return float(
        sum(
            (count / total) * _jsd_counters(real_counts[od], synthetic_counts.get(od, Counter()))
            for od, count in weights.items()
        )
    )


def route_recommendation_metrics(
    synthetic_trajectories: list[np.ndarray],
    synthetic_records: list[MatchRecord],
    real_trajectories: list[np.ndarray],
    real_records: list[MatchRecord],
    bbox: tuple[float, float, float, float],
    od_grid: int,
    max_prototypes: int,
    hit_threshold: float,
) -> dict[str, float]:
    """Frequency-ranked route recommendation on actual directed FMM edge IDs."""
    counters: dict[int, Counter] = defaultdict(Counter)
    for trajectory, record in zip(synthetic_trajectories, synthetic_records):
        if record.accepted and record.cpath:
            counters[_od_key(trajectory, bbox, od_grid)][record.cpath] += 1
    prototypes = {
        od: [path for path, _ in counter.most_common(max_prototypes)]
        for od, counter in counters.items()
    }
    top1_hits: list[float] = []
    top5_hits: list[float] = []
    reciprocal_ranks: list[float] = []
    top1_f1: list[float] = []
    best5_f1: list[float] = []
    covered = 0
    for trajectory, record in zip(real_trajectories, real_records):
        if not record.accepted or not record.cpath:
            top1_hits.append(0.0)
            top5_hits.append(0.0)
            reciprocal_ranks.append(0.0)
            top1_f1.append(0.0)
            best5_f1.append(0.0)
            continue
        candidates = prototypes.get(_od_key(trajectory, bbox, od_grid), [])
        if not candidates:
            top1_hits.append(0.0)
            top5_hits.append(0.0)
            reciprocal_ranks.append(0.0)
            top1_f1.append(0.0)
            best5_f1.append(0.0)
            continue
        covered += 1
        scores = [_edge_f1(record.cpath, candidate) for candidate in candidates]
        top1_f1.append(float(scores[0]))
        best5_f1.append(float(max(scores[:5])))
        top1_hits.append(float(scores[0] >= hit_threshold))
        top5_hits.append(float(max(scores[:5]) >= hit_threshold))
        reciprocal_rank = 0.0
        for rank, score in enumerate(scores, start=1):
            if score >= hit_threshold:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
    denominator = max(len(real_trajectories), 1)
    return {
        "road_route_od_coverage": covered / denominator,
        "road_route_top1": float(np.mean(top1_hits)) if top1_hits else 0.0,
        "road_route_top5": float(np.mean(top5_hits)) if top5_hits else 0.0,
        "road_route_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "road_route_top1_edge_f1": float(np.mean(top1_f1)) if top1_f1 else 0.0,
        "road_route_best5_edge_f1": float(np.mean(best5_f1)) if best5_f1 else 0.0,
    }
