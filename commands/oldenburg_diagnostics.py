"""Reproduce the public Oldenburg input audit.

This command deliberately evaluates the simulated Oldenburg road-network
corpus as a diagnostic benchmark.  It does not claim a public OSM/Portal-
Fiber MTR-GSRT release for Oldenburg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path, limit: int | None = None) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if ":" not in line:
                continue
            pts: list[tuple[float, float]] = []
            for item in line.split(":", 1)[1].split(";"):
                if "," not in item:
                    continue
                try:
                    x, y = item.strip().split(",", 1)
                    pts.append((float(x), float(y)))
                except ValueError:
                    continue
            if len(pts) >= 2:
                out.append(pts)
                if limit is not None and len(out) >= limit:
                    break
    return out


def stats(trajs: list[list[tuple[float, float]]]) -> dict[str, float | int]:
    lengths = [len(t) for t in trajs]
    steps = [((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
             for t in trajs for (x1, y1), (x2, y2) in zip(t, t[1:])]
    # The published Oldenburg assessment calls ``trip`` the endpoint distance,
    # not the cumulative polyline length.  Keep that definition here so the
    # audit is numerically comparable to the stored assessment.
    trips = [((t[-1][0] - t[0][0]) ** 2 + (t[-1][1] - t[0][1]) ** 2) ** 0.5
             for t in trajs]
    return {
        "n": len(trajs),
        "mean_len": sum(lengths) / len(lengths),
        "median_len": sorted(lengths)[len(lengths) // 2],
        "min_len": min(lengths),
        "max_len": max(lengths),
        "step_mean": sum(steps) / len(steps),
        "trip_mean": sum(trips) / len(trips),
        "min_x": min(x for t in trajs for x, _ in t),
        "max_x": max(x for t in trajs for x, _ in t),
        "min_y": min(y for t in trajs for _, y in t),
        "max_y": max(y for t in trajs for _, y in t),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("datasets/raw/oldenburg.dat"))
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--out-dir", type=Path, default=Path("experiment_results/oldenburg/recomputed"))
    args = ap.parse_args()
    trajs = load(args.data, args.limit)
    if not trajs:
        raise SystemExit(f"No valid trajectories found in {args.data}")
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    payload = {
        "dataset": "Oldenburg simulated road-network trajectories",
        "input": str(args.data),
        "sha256": digest,
        "coordinate_system": "local projected coordinates (not WGS84)",
        "diagnostic_only": True,
        "stats": stats(trajs),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "input_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
