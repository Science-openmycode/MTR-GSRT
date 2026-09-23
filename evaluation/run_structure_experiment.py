from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def named_path(text: str) -> tuple[str, Path]:
    name, value = text.split("=", 1)
    path = Path(value)
    return name, (path if path.is_absolute() else ROOT / path).resolve()


def load(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def point(value):
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    return float(value[0]), float(value[1])


def summarize(trajectories) -> dict[str, float]:
    usable = []
    for trajectory in trajectories:
        try:
            pts = [point(v) for v in trajectory]
        except (TypeError, ValueError, IndexError):
            continue
        if len(pts) >= 2:
            usable.append(pts)
    if not usable:
        return {"UsableRatio": 0.0, "MeanPoints": 0.0, "RevisitRatio": 0.0,
                "ClosedTripRatio": 0.0, "MeanDirectness": 0.0}
    revisit, closed, directness, point_counts = [], [], [], []
    for pts in usable:
        point_counts.append(len(pts))
        cells = [(round(x, 3), round(y, 3)) for x, y in pts]
        revisit.append(1.0 - len(set(cells)) / len(cells))
        segment = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        chord = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        directness.append(chord / segment if segment else 1.0)
        closed.append(float(chord <= 0.01 * max(segment, 1e-12)))
    return {
        "UsableRatio": len(usable) / max(len(trajectories), 1),
        "MeanPoints": sum(point_counts) / len(point_counts),
        "RevisitRatio": sum(revisit) / len(revisit),
        "ClosedTripRatio": sum(closed) / len(closed),
        "MeanDirectness": sum(directness) / len(directness),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute trajectory-structure diagnostics from trajectory files.")
    parser.add_argument("--dataset", action="append", type=named_path, required=True, help="NAME=PATH; repeat")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [{"dataset": name, **summarize(load(path))} for name, path in args.dataset]
    output = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    result = output / "results.csv"
    with result.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({
        "protocol": "executed_coordinate_trajectory_structure_v1", "rows": len(rows),
        "result": str(result.resolve())
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote freshly computed structure diagnostics to {result}")


if __name__ == "__main__":
    main()
