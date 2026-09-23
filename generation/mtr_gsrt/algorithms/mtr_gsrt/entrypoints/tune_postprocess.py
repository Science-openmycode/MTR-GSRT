"""Tune MTR routing using only a released DP transcript and public graph."""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "configs" / "datasets.json").is_file():
            sys.path.insert(0, str(candidate))
            break

from generation.common.runtime import dataset_config, public_path, sha256_file, write_json
from generation.mtr.generate import Q5_BLOCK_NAMES, _decode
from public_utils import (
    filter_osm_ways_by_bbox,
    filter_osm_ways_by_highway,
    load_osm_ways,
    thin_osm_ways_public,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--config", default="configs/mtr_sf_trip20k_postprocess_tuning.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate", action="append")
    args = parser.parse_args()

    release_dir = public_path(args.release_dir)
    transcript_path = release_dir / "dp_transcript.npz"
    if not transcript_path.is_file():
        raise FileNotFoundError("released DP transcript is missing")
    tuning = json.loads(public_path(args.config).read_text(encoding="utf-8"))
    dataset = dataset_config(tuning["dataset_config"])
    bbox = tuple(float(value) for value in dataset["bbox"])
    osm_path = public_path(dataset["osm_cache"])
    osm = filter_osm_ways_by_bbox(load_osm_ways(str(osm_path)), bbox)
    osm = filter_osm_ways_by_highway(osm, dataset.get("osm_highway_classes", []))
    osm = thin_osm_ways_public(osm, bbox, int(dataset.get("osm_max_ways", 0)))

    with np.load(transcript_path, allow_pickle=False) as transcript:
        measurements = {
            "endpoint": transcript["base_endpoint"],
            "od": transcript["base_od"],
            "geometry": transcript["base_geometry"],
            "length": transcript["base_length"],
        }
        q5 = {name: transcript[f"q5_{name}"] for name in Q5_BLOCK_NAMES}

    selected_names = set(args.candidate or [])
    candidates = [
        candidate for candidate in tuning["candidates"]
        if not selected_names or candidate["name"] in selected_names
    ]
    if not candidates:
        raise ValueError("no tuning candidates selected")
    out_root = public_path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate in candidates:
        destination = out_root / candidate["name"]
        if destination.exists():
            raise FileExistsError(f"candidate output already exists: {destination}")
        destination.mkdir()
        decoder_out = _decode(
            destination,
            measurements,
            q5,
            osm,
            sha256_file(osm_path),
            int(dataset["public_slot_count"]),
            Fraction(tuning["epsilon_total"]),
            Fraction(tuning["q5_epsilon"]),
            int(tuning["decoder_seed"]),
            int(tuning["request_seed"]),
            bbox,
            betas=candidate["betas"],
            length_proposals=int(candidate["length_proposals"]),
            length_log_penalty=float(candidate["length_log_penalty"]),
            od_likelihood_ratio_cap=float(candidate["od_likelihood_ratio_cap"]),
        )
        report = json.loads((decoder_out / "protocol.json").read_text(encoding="utf-8"))
        for source, name in (
            (decoder_out / "dp_gsrt_portal_qrsp.pkl", "trajectories.pkl"),
            (decoder_out / "road_witnesses.pkl", "road_witnesses.pkl"),
            (decoder_out / "protocol.json", "decoder_protocol.json"),
        ):
            shutil.move(str(source), destination / name)
        shutil.rmtree(decoder_out)
        length = report["length_conditioning"]
        fallback_rate = float(report["fallback_rate"])
        score = float(length["selected_log_error_mean"]) + 2.0 * fallback_rate
        row = {
            **candidate,
            "score": score,
            "fallback_count": int(report["fallback_count"]),
            "fallback_rate": fallback_rate,
            "selected_log_error_mean": float(length["selected_log_error_mean"]),
            "selected_log_error_p90": float(length["selected_log_error_p90"]),
            "outputs": {
                name: sha256_file(destination / name)
                for name in ("trajectories.pkl", "road_witnesses.pkl", "decoder_protocol.json")
            },
        }
        write_json(destination / "tuning_candidate.json", row)
        results.append(row)

    results.sort(key=lambda row: (row["score"], row["selected_log_error_p90"], row["name"]))
    summary = {
        "schema_version": 1,
        "classification": tuning["classification"],
        "selection_access": tuning["selection_access"],
        "forbidden_selection_access": tuning["forbidden_selection_access"],
        "source_dp_transcript": {
            "path": str(transcript_path),
            "sha256": sha256_file(transcript_path),
        },
        "public_osm": {"path": str(osm_path), "sha256": sha256_file(osm_path)},
        "score": tuning["score"],
        "selected": results[0]["name"],
        "results": results,
    }
    write_json(out_root / "tuning_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
