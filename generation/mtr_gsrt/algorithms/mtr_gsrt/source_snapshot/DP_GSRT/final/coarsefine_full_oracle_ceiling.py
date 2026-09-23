"""NON-DP exact-moment oracle ceiling for the frozen public K=8 candidate family."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

import graph_cycle_dp_production_sanitizer as lineage
import graph_voronoi_doptimal_support_probe as support
import mass_calibrated_full_development as mass_runner
import mass_calibrated_graph_flow as mass
import multires_graph_flow_experiment as legacy
import route_structure_potential_experiment as route
from assemble_graph_cycle_production_release import load_verified_candidates
from audit_two_level_semimarkov_dp import build_context
from public_utils import load_osm_ways
from resample_production_trust_sweep import materialize
from two_level_maxent_projection import fit_projection, kl_trust_region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--public-matrix-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--kl-radius", type=float, default=0.60)
    parser.add_argument("--max-iterations", type=int, default=240)
    parser.add_argument("--acknowledge-non-private-oracle", action="store_true", required=True)
    return parser.parse_args()


def legacy_matrix_from_mass_cache(matrix, slices):
    mapping = [
        ("coarse24_occupancy", 1.0),
        ("fine384_occupancy", 1.0),
        (
            "fine96_flow",
            legacy.BLOCK_BUDGETS["fine96_flow"] / mass.FINE96_FLOW_BUDGET,
        ),
        (
            "fine384_flow",
            legacy.BLOCK_BUDGETS["fine384_flow"] / mass.FINE384_FLOW_BUDGET,
        ),
        (
            "od_conditioned_fine96_flow",
            legacy.BLOCK_BUDGETS["od_conditioned_fine96_flow"]
            / mass.OD_FLOW_BUDGET,
        ),
    ]
    columns, scales = [], []
    for name, scale in mapping:
        start, stop = slices[name]
        columns.extend(range(start, stop))
        scales.extend([float(scale)] * (stop - start))
    selected = matrix[:, np.asarray(columns, dtype=int)].tocsr()
    return selected.multiply(np.asarray(scales, dtype=float)).tocsr()


def choices(probability: np.ndarray, seed: int) -> np.ndarray:
    draws = np.random.default_rng(int(seed) + 50_001).random(len(probability))
    return np.asarray(
        [
            min(int(np.searchsorted(np.cumsum(row), draw, side="right")), 7)
            for row, draw in zip(probability, draws)
        ],
        dtype=np.int8,
    )


def main() -> None:
    args = parse_args()
    resolved = str(args.out_dir.resolve()).lower()
    if (
        not args.out_dir.name.lower().startswith("non_dp")
        or "production" in resolved
        or "certified" in resolved
    ):
        raise RuntimeError("Oracle output must be isolated in a NON_DP root")
    if args.out_dir.exists():
        raise RuntimeError("Oracle output directory must be new")
    cache_manifest = args.public_matrix_cache.with_suffix(".json")
    if not args.public_matrix_cache.is_file() or not cache_manifest.is_file():
        raise RuntimeError("Verified public mass-calibrated matrix cache is required")

    request_manifest = json.loads(
        (args.production_root / "requests" / "request_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    with np.load(
        args.production_root / "requests" / "dp_requests.npz", allow_pickle=False
    ) as archive:
        point_count = np.asarray(archive["point_count"], dtype=int)
    candidate_nodes, chunk_manifests = load_verified_candidates(
        args.production_root / "candidate_chunks",
        request_manifest,
        len(point_count),
    )
    osm_path = (
        Path(__file__).resolve().parents[2]
        / "ara_final"
        / "evidence"
        / "tables"
        / "osm_cache_beijing.pkl"
    )
    coords, graph = route.prepare_graph(
        [], bbox=support.BBOX, osm_ways=load_osm_ways(osm_path), raw_graph=False
    )
    context96, _ = build_context(coords, graph, 24, 96, 4, 3)
    context384, _ = build_context(coords, graph, 24, 384, 4, 3)
    observed_cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    expected_cache = {
        "public_only": True,
        "candidate_chunk_sha256": [
            item["candidate_nodes_sha256"] for item in chunk_manifests
        ],
        "public_osm_sha256": lineage.sha256_file(osm_path),
        "component_budgets": mass.COMPONENT_BUDGETS,
    }
    if observed_cache != expected_cache:
        raise RuntimeError("Public matrix cache lineage mismatch")
    mass_matrix = load_npz(args.public_matrix_cache).tocsr()
    slices = mass_runner.component_slices(context96, context384)
    matrix = legacy_matrix_from_mass_cache(mass_matrix, slices)

    real = support.load_trajectories(support.DEFAULT_REAL)
    exact = legacy.aggregate(real, context96, context384)
    target = np.concatenate(
        [np.asarray(exact[name], dtype=float).ravel() for name in legacy.BLOCK_BUDGETS]
    ) / legacy.ROUTE_LATTICE
    probability, optimizer = fit_projection(
        matrix,
        target,
        groups=len(point_count),
        candidates_per_group=8,
        noise_variance=1e-6,
        max_iterations=int(args.max_iterations),
    )
    constrained, trust = kl_trust_region(probability, float(args.kl_radius))
    selected = choices(constrained, int(args.seed))
    unconstrained_selected = choices(probability, int(args.seed))

    staging = args.out_dir.parent / f".{args.out_dir.name}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "DO_NOT_RELEASE.txt").write_text(
        "NON-PRIVATE exact-moment oracle. Contains private-data products.\n",
        encoding="ascii",
    )
    outputs = {
        "maxent_projected.pkl": materialize(
            candidate_nodes, selected, point_count, coords
        ),
        "oracle_unconstrained.pkl": materialize(
            candidate_nodes, unconstrained_selected, point_count, coords
        ),
    }
    for name, values in outputs.items():
        with (staging / name).open("wb") as handle:
            pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    control = (
        args.out_dir.parent
        / "coarsefine_odflow_full_seed17"
        / "uniform_candidate_control.pkl"
    )
    shutil.copyfile(control, staging / "uniform_candidate_control.pkl")
    np.save(staging / "selection_probability_tau060.npy", constrained, allow_pickle=False)
    np.save(staging / "selection_probability_unconstrained.npy", probability, allow_pickle=False)
    protocol = {
        "algorithm": "NON-PRIVATE exact-moment oracle ceiling on frozen K=8 family",
        "classification": "NON_PRIVATE_ORACLE_DO_NOT_RELEASE",
        "contains_private_data_products": True,
        "certified_release": False,
        "seed": int(args.seed),
        "exact_query": "legacy five Coarse-Fine blocks without DP noise",
        "optimizer": optimizer,
        "kl_trust_region": trust,
        "candidate_chunk_sha256": [
            item["candidate_nodes_sha256"] for item in chunk_manifests
        ],
        "public_candidate_matrix_sha256": lineage.sha256_file(
            args.public_matrix_cache
        ),
        "outputs": {
            path.name: lineage.sha256_file(path)
            for path in staging.iterdir()
            if path.is_file() and path.name != "protocol.json"
        },
    }
    (staging / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    os.replace(staging, args.out_dir)
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
