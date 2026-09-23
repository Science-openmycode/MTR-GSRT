from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


_RUNTIME_ROOT = Path(__file__).resolve().parent
PUBLIC_ROOT = _RUNTIME_ROOT
PACKAGE_ROOT = next(
    (parent for parent in (_RUNTIME_ROOT, *_RUNTIME_ROOT.parents)
     if (parent / "config" / "package.json").is_file()),
    _RUNTIME_ROOT,
)
ARA_CODEX = PACKAGE_ROOT
REPO_ROOT = PACKAGE_ROOT
ARA_FINAL = REPO_ROOT / "ara_final"
SCRIPT_ROOT = ARA_CODEX / "scripts"
PUBLIC_DATA = PUBLIC_ROOT / "data"
PUBLIC_TRAJ = PUBLIC_DATA / "trajectories"
PUBLIC_OSM = PUBLIC_DATA / "osm"


def add_project_paths() -> None:
    """Expose the original research modules without changing their locations."""
    for path in [
        PUBLIC_ROOT,
        PUBLIC_ROOT / "src" / "reproduced_papers" / "common",
        PUBLIC_ROOT / "src" / "reproduced_papers" / "SPRT",
        PUBLIC_ROOT / "src" / "reproduced_papers" / "PrivTrace",
        PUBLIC_ROOT / "src" / "reproduced_papers" / "DPTrajPM",
        PUBLIC_ROOT / "src" / "reproduced_papers" / "DPStd",
        PUBLIC_ROOT / "src" / "plotting",
        PUBLIC_ROOT / "src" / "experiments",
        ARA_FINAL,
        ARA_FINAL / "src" / "execution",
        SCRIPT_ROOT,
    ]:
        s = str(path)
        if path.exists() and s not in sys.path:
            sys.path.insert(0, s)


def parse_bbox(values: list[float] | tuple[float, ...] | None) -> tuple[float, float, float, float]:
    if not values:
        return (39.75, 40.15, 116.10, 116.65)
    if len(values) != 4:
        raise ValueError("--bbox needs four values: lat_min lat_max lon_min lon_max")
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def bbox_dict(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    return {
        "lat_min": bbox[0],
        "lat_max": bbox[1],
        "lon_min": bbox[2],
        "lon_max": bbox[3],
    }


def _coerce_traj_list(obj: Any) -> list[np.ndarray]:
    if isinstance(obj, dict):
        for key in ["trajs", "trajectories", "data", "real", "synthetic"]:
            if key in obj:
                obj = obj[key]
                break
    if isinstance(obj, np.ndarray) and obj.dtype != object and obj.ndim == 3:
        seq = [obj[i] for i in range(obj.shape[0])]
    else:
        seq = list(obj)

    out: list[np.ndarray] = []
    for tr in seq:
        arr = np.asarray(tr, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 2:
            out.append(arr[:, :2])
    return out


def _load_public_geolife(limit: int | None = None) -> list[np.ndarray]:
    data_dir = PUBLIC_TRAJ / "geolife" / "Geolife Trajectories 1.3" / "Data"
    bbox = parse_bbox(None)
    trajs: list[np.ndarray] = []
    for plt in sorted(data_dir.glob("*/Trajectory/*.plt")):
        pts = []
        with plt.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(6):
                next(f, None)
            for line in f:
                p = line.strip().split(",")
                if len(p) < 2:
                    continue
                try:
                    lat, lon = float(p[0]), float(p[1])
                except ValueError:
                    continue
                if bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]:
                    pts.append([lat, lon])
        if len(pts) >= 2:
            trajs.append(np.asarray(pts, dtype=float))
            if limit is not None and len(trajs) >= limit:
                return trajs
    return trajs


def _load_public_porto(limit: int | None = None) -> list[np.ndarray]:
    path = PUBLIC_TRAJ / "porto" / "train.csv"
    bbox = (41.10, 41.20, -8.70, -8.55)
    import csv

    trajs: list[np.ndarray] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("POLYLINE")
            if not raw:
                continue
            try:
                pts = json.loads(raw)
            except json.JSONDecodeError:
                continue
            arr = np.asarray([[p[1], p[0]] for p in pts if len(p) >= 2], dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 2:
                continue
            mask = (bbox[0] <= arr[:, 0]) & (arr[:, 0] <= bbox[1]) & (bbox[2] <= arr[:, 1]) & (arr[:, 1] <= bbox[3])
            arr = arr[mask]
            if len(arr) >= 2:
                trajs.append(arr[:100])
                if limit is not None and len(trajs) >= limit:
                    return trajs
    return trajs


def _load_public_oldenburg(limit: int | None = None) -> list[np.ndarray]:
    path = PUBLIC_TRAJ / "oldenburg.dat"
    trajs: list[np.ndarray] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ":" not in line:
                continue
            pts = []
            for item in line.split(":", 1)[1].split(";"):
                item = item.strip()
                if "," not in item:
                    continue
                try:
                    x, y = item.split(",", 1)
                    pts.append([float(x), float(y)])
                except ValueError:
                    continue
            if len(pts) >= 2:
                trajs.append(np.asarray(pts, dtype=float))
                if limit is not None and len(trajs) >= limit:
                    return trajs
    return trajs


def _load_sf_cabspotting(limit: int | None = None) -> list[np.ndarray]:
    candidates = [
        PUBLIC_TRAJ / "cabspottingdata" / "cabspottingdata",
        REPO_ROOT / "data" / "cabspottingdata" / "cabspottingdata",
        REPO_ROOT.parent / "LDPTrace-main" / "LDPTrace-main" / "data" / "san_francisco" / "extracted" / "CabSpotting-master" / "CabSpotting-master",
    ]
    data_dir = next((p for p in candidates if p.exists()), None)
    if data_dir is None:
        raise FileNotFoundError(
            "SF cabspotting files not found. Expected public_release/data/trajectories/cabspottingdata/ "
            "or parent data/cabspottingdata/cabspottingdata."
        )
    bbox = (37.60, 37.85, -122.55, -122.30)
    trajs: list[np.ndarray] = []
    for txt in sorted(data_dir.glob("*.txt")):
        rows = []
        with txt.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                try:
                    lat, lon, ts = float(parts[0]), float(parts[1]), float(parts[3])
                except ValueError:
                    continue
                if bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]:
                    rows.append((ts, lat, lon))
        if len(rows) >= 2:
            rows.sort(key=lambda x: x[0])
            arr = np.asarray([[lat, lon] for _, lat, lon in rows], dtype=float)
            # Long taxi traces can dominate runtime; keep a deterministic,
            # shape-preserving subsample while retaining the full cab route.
            if len(arr) > 400:
                idx = np.linspace(0, len(arr) - 1, 400).round().astype(int)
                arr = arr[idx]
            trajs.append(arr)
            if limit is not None and len(trajs) >= limit:
                return trajs
    return trajs


def load_trajectories(spec: str, limit: int | None = None) -> list[np.ndarray]:
    """Load trajectories from a named built-in source or a simple serialized file."""
    add_project_paths()
    key = spec.lower()
    if key in {"geolife", "beijing", "beijing_geolife"}:
        if (PUBLIC_TRAJ / "geolife").exists():
            return _load_public_geolife(limit=limit)
        import eval_layered_framework as layered
        return layered.load_geolife(limit=limit if limit is not None else 999999)
    if key == "oldenburg":
        if (PUBLIC_TRAJ / "oldenburg.dat").exists():
            return _load_public_oldenburg(limit=limit)
        from visualize_city_independent_osm_synthesis import load_oldenburg
        trajs = load_oldenburg()
        return trajs[:limit] if limit else trajs
    if key == "porto":
        if (PUBLIC_TRAJ / "porto" / "train.csv").exists():
            return _load_public_porto(limit=limit)
        from visualize_city_independent_osm_synthesis import load_porto
        return load_porto(limit=limit if limit is not None else 999999)
    if key in {"sf", "sf_bay", "cabspotting", "san_francisco"}:
        return _load_sf_cabspotting(limit=limit)

    path = Path(spec)
    if not path.is_absolute():
        public_path = PUBLIC_ROOT / path
        path = public_path if public_path.exists() else REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"trajectory data not found: {path}")

    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            obj = pickle.load(f)
    elif path.suffix.lower() == ".npy":
        obj = np.load(path, allow_pickle=True)
    elif path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=True)
        if "trajectories" in z.files:
            obj = z["trajectories"]
        elif "trajs" in z.files:
            obj = z["trajs"]
        else:
            obj = z[z.files[0]]
    elif path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("supported data formats: geolife, oldenburg, porto, .pkl, .npy, .npz, .json")

    trajs = _coerce_traj_list(obj)
    return trajs[:limit] if limit else trajs


def load_osm_ways(osm_cache: str | None = None):
    add_project_paths()
    if osm_cache:
        path = Path(osm_cache)
        if not path.is_absolute():
            public_path = PUBLIC_ROOT / path
            bundled_path = PUBLIC_OSM / Path(osm_cache).name
            if public_path.exists():
                path = public_path
            elif bundled_path.exists():
                path = bundled_path
            else:
                path = REPO_ROOT / path
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
    bundled = PUBLIC_OSM / "osm_cache_beijing.pkl"
    if bundled.exists():
        with bundled.open("rb") as f:
            return pickle.load(f)

    import eval_layered_framework as layered

    return layered.load_osm_ways()


def filter_osm_ways_by_bbox(
    osm_ways: list[dict],
    bbox: tuple[float, float, float, float],
    pad_ratio: float = 0.08,
) -> list[dict]:
    """Keep OSM ways with at least one geometry point near the city bbox."""
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_pad = max(lat_max - lat_min, 1e-9) * pad_ratio
    lon_pad = max(lon_max - lon_min, 1e-9) * pad_ratio
    lo_lat, hi_lat = lat_min - lat_pad, lat_max + lat_pad
    lo_lon, hi_lon = lon_min - lon_pad, lon_max + lon_pad
    out = []
    for way in osm_ways:
        geom = way.get("geometry", []) if isinstance(way, dict) else []
        for p in geom:
            try:
                lat, lon = float(p["lat"]), float(p["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            if lo_lat <= lat <= hi_lat and lo_lon <= lon <= hi_lon:
                out.append(way)
                break
    return out


def filter_osm_ways_by_highway(
    osm_ways: list[dict],
    highway_classes: list[str] | tuple[str, ...] | None,
) -> list[dict]:
    """Select a public, dataset-configured set of OSM highway classes.

    This is useful for vehicle-trajectory datasets whose public OSM extracts
    also contain large numbers of footways, steps, and indoor paths.  The
    selection depends only on public OSM tags and a predeclared class list.
    """
    if not highway_classes:
        return osm_ways
    allowed = {str(value) for value in highway_classes}
    return [
        way for way in osm_ways
        if isinstance(way, dict)
        and str(way.get("tags", {}).get("highway", "")) in allowed
    ]


def thin_osm_ways_public(
    osm_ways: list[dict],
    bbox: tuple[float, float, float, float],
    max_ways: int | None,
    grid: int = 80,
) -> list[dict]:
    """Deterministically reduce a public OSM graph while preserving city coverage."""
    if max_ways is None or max_ways <= 0 or len(osm_ways) <= max_ways:
        return osm_ways
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_span = max(lat_max - lat_min, 1e-9)
    lon_span = max(lon_max - lon_min, 1e-9)
    buckets: dict[tuple[int, int], list[tuple[float, int, dict]]] = {}
    for idx, way in enumerate(osm_ways):
        geom = way.get("geometry", []) if isinstance(way, dict) else []
        pts = []
        for p in geom:
            try:
                pts.append((float(p["lat"]), float(p["lon"])))
            except (KeyError, TypeError, ValueError):
                continue
        if len(pts) < 2:
            continue
        lat_c = sum(p[0] for p in pts) / len(pts)
        lon_c = sum(p[1] for p in pts) / len(pts)
        gy = min(grid - 1, max(0, int((lat_c - lat_min) / lat_span * grid)))
        gx = min(grid - 1, max(0, int((lon_c - lon_min) / lon_span * grid)))
        length = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            length += abs(a[0] - b[0]) + abs(a[1] - b[1])
        buckets.setdefault((gy, gx), []).append((-length, idx, way))
    ordered_cells = sorted(buckets)
    for cell in ordered_cells:
        buckets[cell].sort()
    selected: list[dict] = []
    cursor = 0
    while len(selected) < max_ways and ordered_cells:
        cell = ordered_cells[cursor % len(ordered_cells)]
        bucket = buckets[cell]
        if bucket:
            selected.append(bucket.pop(0)[2])
        if not bucket:
            ordered_cells.remove(cell)
            if not ordered_cells:
                break
            cursor %= len(ordered_cells)
        else:
            cursor += 1
    return selected


def jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return jsonable(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return out


def save_trajectories_npz(path: str | Path, trajectories: list[np.ndarray]) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray([np.asarray(t, dtype=float) for t in trajectories], dtype=object)
    np.savez_compressed(out, trajectories=arr)
    return out


def save_trajectories_pkl(path: str | Path, trajectories: list[np.ndarray]) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump([np.asarray(t, dtype=float) for t in trajectories], f, protocol=pickle.HIGHEST_PROTOCOL)
    return out
