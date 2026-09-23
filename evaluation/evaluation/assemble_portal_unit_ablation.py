"""Assemble the eight paper metrics for the saved Portal-Fiber unit ablation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (_candidate / "configs" / "datasets.json").is_file():
            sys.path.insert(0, str(_candidate))
            break

from generation.common.runtime import public_path, sha256_file, write_json  # noqa: E402


METHODS = {
    "MTR-GSRT": "full",
    "No-Portal-Unit": "no_portal_unit",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-unified", required=True)
    parser.add_argument("--ablated-unified", required=True)
    parser.add_argument("--road-choice", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    sources = {
        "full": public_path(args.full_unified),
        "no_portal_unit": public_path(args.ablated_unified),
        "road_choice": public_path(args.road_choice),
    }
    out_dir = public_path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    unified = {
        arm: json.loads(sources[arm].read_text(encoding="utf-8"))["metrics"]
        for arm in ("full", "no_portal_unit")
    }
    road = json.loads(sources["road_choice"].read_text(encoding="utf-8"))["results"]
    if set(road) != set(METHODS):
        raise ValueError(f"unexpected road-choice methods: {sorted(road)}")
    rows: list[dict] = []
    for method, arm in METHODS.items():
        statistical = unified[arm]
        route = road[method]
        values = {
            "Grid": float(statistical["grid_density_jsd"]),
            "Trip": float(statistical["trip_error"]),
            "Len": float(statistical["path_length_jsd"]),
            "WitnessValid": float(statistical["witness_valid"]),
            "RC-CPC": float(route["road_choice_cpc"]),
            "RC-NDCG": float(route["road_choice_ndcg"]),
            "NextRoadAcc": float(route["next_road_accuracy"]),
            "NextRoadNLL": float(route["next_road_nll"]),
        }
        rows.append({"method": method, **values})
    payload = {
        "schema_version": 1,
        "classification": "SAVED_RELEASE_COMPONENT_ABLATION_NO_SYNTHESIS",
        "seed": 20260719,
        "arms": {
            "MTR-GSRT": "Portal-Fiber crossing flow, local information projection, and public crossing-edge sampling active",
            "No-Portal-Unit": "the three-part portal-choice unit removed while the remaining MTR-GSRT construction is fixed",
        },
        "metrics": rows,
    }
    out_dir.mkdir(parents=True)
    write_json(out_dir / "metrics.json", payload)
    fields = tuple(rows[0])
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "classification": payload["classification"],
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "assembler": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "outputs": {
            name: sha256_file(out_dir / name)
            for name in ("metrics.json", "metrics.csv")
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
