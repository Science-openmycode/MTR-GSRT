"""Freeze Porto CSV trajectories using the published coordinate/filter order."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import pickle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def prepare(source, bbox, count):
    trajectories = []
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "POLYLINE" not in (reader.fieldnames or []):
            raise ValueError("Porto CSV must contain POLYLINE")
        for row in reader:
            try:
                points = json.loads(row.get("POLYLINE") or "[]")
                arr = np.asarray([[p[1], p[0]] for p in points if len(p) >= 2], dtype=float)
            except (ValueError, TypeError, IndexError):
                continue
            if arr.ndim != 2 or len(arr) < 2:
                continue
            mask = (np.isfinite(arr).all(axis=1) &
                    (arr[:, 0] >= bbox[0]) & (arr[:, 0] <= bbox[1]) &
                    (arr[:, 1] >= bbox[2]) & (arr[:, 1] <= bbox[3]))
            arr = arr[mask]
            if len(arr) >= 2:
                trajectories.append(arr[:100])
                if len(trajectories) == count:
                    return trajectories
    raise ValueError(f"Only {len(trajectories)} eligible trajectories; requested {count}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--bbox", type=float, nargs=4, default=(41.10,41.20,-8.70,-8.55))
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.count <= 0 or not (-90 <= args.bbox[0] < args.bbox[1] <= 90 and
                               -180 <= args.bbox[2] < args.bbox[3] <= 180):
        parser.error("positive count and ordered geographic bbox required")
    source, out = resolve(args.source_csv), resolve(args.out)
    side = out.with_suffix(".manifest.json")
    if out.exists() or side.exists():
        raise FileExistsError(f"Output already exists: {out}")
    trajectories = prepare(source, args.bbox, args.count)
    payload = pickle.dumps(trajectories, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(payload).hexdigest()
    if args.expected_sha256 and digest != args.expected_sha256.lower():
        raise ValueError(f"Output hash mismatch: {digest}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    record = dict(privacy_unit="one retained Porto trip", coordinate_order="latitude,longitude",
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  output_sha256=digest, count=len(trajectories), bbox=args.bbox,
                  selection="CSV order; bbox-filter coordinates; first 100 points; first count eligible trips",
                  classification="LOCAL_BENCHMARK_PREPROCESSING_NOT_A_DP_RELEASE")
    side.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record))
if __name__ == "__main__":
    main()
