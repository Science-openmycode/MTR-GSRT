from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from route_metric_core import evaluate_routes, load_pickle, normalize_routes, route_counters


ROOT = Path(__file__).resolve().parents[1]


def named_path(text: str) -> tuple[str, Path]:
    name, value = text.split("=", 1)
    path = Path(value)
    return name, (path if path.is_absolute() else ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute the road-object lift from native releases and their public routed outputs."
    )
    parser.add_argument("--real-routes", required=True)
    parser.add_argument("--edge-cache", default=str(ROOT / "public_assets" / "ordered_portal_route_cache.pkl.gz"))
    parser.add_argument("--routed", action="append", type=named_path, required=True, help="NAME=ROUTE_PICKLE; repeat")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cache = load_pickle(Path(args.edge_cache).resolve())
    real = normalize_routes(load_pickle(Path(args.real_routes).resolve()), cache)
    reference = route_counters(real, cache)
    rows = []
    for name, path in args.routed:
        # A coordinate-only statistical release contains no consumer-visible road object.
        rows.append({"Measurement": name, "Stage": "native coordinate release", "StageKey": "native",
                     "RoadYield": 0.0, "BTF": 0.0, "FamilyCPC": 0.0})
        routes = normalize_routes(load_pickle(path), cache)
        metrics = evaluate_routes(routes, real, cache, reference)
        rows.append({"Measurement": name, "Stage": "MTR public routing", "StageKey": "routed",
                     "RoadYield": metrics["RoadYield"], "BTF": metrics["BTF"],
                     "FamilyCPC": metrics["FamilyCPC"]})
    output = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    result = output / "results.csv"
    with result.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({
        "protocol": "executed_native_vs_public_routing_v1", "rows": len(rows),
        "result": str(result.resolve())
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote executed framework comparison to {result}")


if __name__ == "__main__":
    main()
