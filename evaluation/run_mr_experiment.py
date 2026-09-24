from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from route_metric_core import evaluate_routes, load_pickle, normalize_routes, route_counters, valid_routes


ROOT = Path(__file__).resolve().parents[1]


def parse_item(text: str) -> tuple[str, str, Path]:
    try:
        label, value = text.split("=", 1)
        measurement, router = label.split("::", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected M::R=PATH") from exc
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return measurement, router, path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute an M-by-R matrix from saved synthetic route objects."
    )
    parser.add_argument("--real-routes", required=True)
    parser.add_argument("--edge-cache", default=str(ROOT / "public_assets" / "ordered_portal_route_cache.pkl.gz"))
    parser.add_argument("--route", action="append", type=parse_item, default=[], help="M::R=PATH; repeat")
    parser.add_argument("--route-dir", default="datasets/synthetic/route_experiments")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cache = load_pickle(Path(args.edge_cache).resolve())
    real = valid_routes(normalize_routes(load_pickle(Path(args.real_routes).resolve()), cache))
    reference = route_counters(real, cache)
    items = list(args.route)
    route_dir = Path(args.route_dir)
    if not route_dir.is_absolute():
        route_dir = ROOT / route_dir
    for path in sorted(route_dir.glob("*/*")):
        if path.is_file() and (path.suffix == ".pkl" or path.name.endswith(".pkl.gz")):
            items.append((path.parent.name, path.name.replace(".pkl.gz", "").replace(".pkl", ""), path))
    if not items:
        parser.error("no --route entries and no packaged route datasets")

    rows = []
    seen = set()
    for measurement, router, path in items:
        key = (measurement, router)
        if key in seen:
            continue
        seen.add(key)
        routes = normalize_routes(load_pickle(path), cache)
        rows.append({"M": measurement, "R": router, **evaluate_routes(routes, real, cache, reference)})

    output = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    result = output / "results.csv"
    with result.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({
        "protocol": "executed_route_object_m_by_r_v1",
        "real_routes": str(Path(args.real_routes).resolve()),
        "rows": len(rows),
        "result": str(result.resolve()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} executed M×R cells to {result}")


if __name__ == "__main__":
    main()
