"""Publish a fixed KL trust-radius sweep from one DP probability transcript."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


ARA = Path(__file__).resolve().parents[1]
for path in [ARA / "dp_reward_exploration", ARA / "public_release", ARA / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import graph_cycle_dp_production_sanitizer as sanitizer
import graph_voronoi_doptimal_support_probe as support
import route_structure_potential_experiment as route
from assemble_graph_cycle_production_release import load_verified_candidates
from public_utils import load_osm_ways


TRUST_RADII = (0.10, 0.20, 0.40, 0.60)
PUBLIC_POSTPROCESS_SEED = 20260713


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitized-dir", type=Path, required=True)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--chunk-root", type=Path, required=True)
    parser.add_argument("--base-release-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def mean_kl_uniform(probability):
    width = probability.shape[1]
    return float(
        np.mean(
            np.sum(
                probability
                * (np.log(np.maximum(probability, 1e-300)) + np.log(width)),
                axis=1,
            )
        )
    )


def tempered(probability, beta):
    logits = float(beta) * np.log(np.maximum(probability, 1e-300))
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def radius_projection(probability, radius):
    current = mean_kl_uniform(probability)
    if abs(current - float(radius)) < 1e-10:
        return probability, 1.0, current
    if radius < current:
        low, high = 0.0, 1.0
    else:
        low, high = 1.0, 2.0
        while mean_kl_uniform(tempered(probability, high)) < radius and high < 128.0:
            low, high = high, 2.0 * high
    for _ in range(60):
        middle = 0.5 * (low + high)
        if mean_kl_uniform(tempered(probability, middle)) < radius:
            low = middle
        else:
            high = middle
    output = tempered(probability, 0.5 * (low + high))
    return output, 0.5 * (low + high), mean_kl_uniform(output)


def choices_from_uniforms(probability, uniforms):
    return np.asarray(
        [
            min(int(np.searchsorted(np.cumsum(row), draw, side="right")), len(row) - 1)
            for row, draw in zip(probability, uniforms)
        ],
        dtype=np.int8,
    )


def materialize(candidate_nodes, choices, point_count, coords):
    output = []
    for request, choice in enumerate(choices):
        output.append(
            route.resample(
                coords[np.asarray(candidate_nodes[request][int(choice)], dtype=int)],
                max(2, int(point_count[request])),
            )
        )
    return output


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise RuntimeError("Trust sweep requires a new output directory")
    request_manifest = json.loads(
        (args.request_dir / "request_manifest.json").read_text(encoding="utf-8")
    )
    request_path = args.request_dir / "dp_requests.npz"
    sanitizer_manifest = json.loads(
        (args.sanitized_dir / "sanitizer_manifest.json").read_text(encoding="utf-8")
    )
    release_manifest = json.loads(
        (args.base_release_dir / "release_manifest.json").read_text(encoding="utf-8")
    )
    release_id = sanitizer_manifest["release_id"]
    if (
        request_manifest["sanitizer_release_id"] != release_id
        or release_manifest["sanitizer_release_id"] != release_id
        or sanitizer.sha256_file(request_path) != request_manifest["request_sha256"]
    ):
        raise RuntimeError("Sanitizer, request, or base release lineage mismatch")
    with np.load(request_path, allow_pickle=False) as archive:
        point_count = np.asarray(archive["point_count"], dtype=int)
    candidate_nodes, chunk_manifests = load_verified_candidates(
        args.chunk_root, request_manifest, len(point_count)
    )
    probability_path = args.base_release_dir / "selection_probability.npy"
    expected_probability_hash = release_manifest.get("outputs", {}).get(
        probability_path.name
    )
    if (
        expected_probability_hash is None
        or sanitizer.sha256_file(probability_path) != expected_probability_hash
    ):
        raise RuntimeError("Selection probability is unbound or its hash mismatches")
    probability = np.load(probability_path, allow_pickle=False)
    if probability.shape != (len(point_count), 8):
        raise RuntimeError("Base DP probability shape mismatch")
    osm_path = (
        Path(__file__).resolve().parents[2]
        / "ara_final"
        / "evidence"
        / "tables"
        / "osm_cache_beijing.pkl"
    )
    if sanitizer.sha256_file(osm_path) != sanitizer_manifest["public_osm"]["sha256"]:
        raise RuntimeError("Public OSM hash mismatch")
    coords, _ = route.prepare_graph(
        [], bbox=support.BBOX, osm_ways=load_osm_ways(osm_path), raw_graph=False
    )
    rng = np.random.default_rng(PUBLIC_POSTPROCESS_SEED)
    uniforms = rng.random(len(point_count))
    controls = rng.integers(0, 8, size=len(point_count), dtype=np.int8)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    control_path = args.out_dir / "public_control_common.pkl"
    with control_path.open("wb") as handle:
        pickle.dump(
            materialize(candidate_nodes, controls, point_count, coords),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    results = {}
    for radius in TRUST_RADII:
        current, multiplier, achieved = radius_projection(probability, radius)
        choices = choices_from_uniforms(current, uniforms)
        filename = f"dp_gsrt_tau_{int(round(radius * 100)):03d}.pkl"
        path = args.out_dir / filename
        with path.open("wb") as handle:
            pickle.dump(
                materialize(candidate_nodes, choices, point_count, coords),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        results[f"{radius:.2f}"] = {
            "temperature_multiplier_from_tau020": multiplier,
            "achieved_mean_kl": achieved,
            "candidate_counts": np.bincount(choices, minlength=8).tolist(),
            "filename": filename,
            "sha256": sanitizer.sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "privacy": "fixed post-processing sweep of one committed (6/5,0)-DP transcript",
        "all_variants_published_before_private_evaluation": True,
        "public_postprocess_seed": PUBLIC_POSTPROCESS_SEED,
        "sanitizer_release_id": sanitizer_manifest["release_id"],
        "selection_probability_sha256": expected_probability_hash,
        "request_sha256": request_manifest["request_sha256"],
        "candidate_chunk_sha256": [item["candidate_nodes_sha256"] for item in chunk_manifests],
        "control": {
            "filename": control_path.name,
            "sha256": sanitizer.sha256_file(control_path),
            "candidate_counts": np.bincount(controls, minlength=8).tolist(),
        },
        "variants": results,
    }
    (args.out_dir / "trust_sweep_manifest.json").write_bytes(
        sanitizer.strict_json_bytes(manifest)
    )
    print(str((args.out_dir / "trust_sweep_manifest.json").resolve()), flush=True)


if __name__ == "__main__":
    main()
