"""NON-DP probe of noise-energy residual shrinkage toward public moments."""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

import coarsefine_full_oracle_ceiling as oracle
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
    parser.add_argument("--dp-development-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilon-route", type=float, default=0.20)
    parser.add_argument("--kl-radius", type=float, default=0.60)
    parser.add_argument("--acknowledge-non-dp-development", action="store_true", required=True)
    return parser.parse_args()


def discrete_laplace_variance_normalized(
    *, epsilon: float, sensitivity: int
) -> float:
    exponent = float(epsilon) / int(sensitivity)
    one_minus_q = -math.expm1(-exponent)
    q = math.exp(-exponent)
    return float(2.0 * q / (one_minus_q * one_minus_q) / (int(sensitivity) ** 2))


def positive_part_block_shrinkage(
    target: np.ndarray,
    public_mean: np.ndarray,
    block_sizes: dict[str, int],
    *,
    noise_variance: float,
) -> tuple[np.ndarray, dict]:
    result = np.empty_like(target, dtype=float)
    diagnostics, offset = {}, 0
    for name, size in block_sizes.items():
        stop = offset + int(size)
        residual = np.asarray(target[offset:stop] - public_mean[offset:stop])
        energy = float(residual @ residual)
        # For additive zero-mean noise with coordinate variance sigma^2,
        # E||target-public||^2 = ||signal||^2 + d*sigma^2 exactly.
        noise_energy = int(size) * float(noise_variance)
        alpha = max(
            0.0,
            1.0 - noise_energy / max(energy, 1e-15),
        )
        result[offset:stop] = public_mean[offset:stop] + alpha * residual
        diagnostics[name] = {
            "dimension": int(size),
            "residual_l2_squared": energy,
            "raw_additive_noise_energy": noise_energy,
            "estimated_nonnegative_signal_energy": max(energy - noise_energy, 0.0),
            "positive_part_alpha": alpha,
        }
        offset = stop
    if offset != len(target):
        raise AssertionError("Block shrinkage did not consume the complete target")
    return result, diagnostics


def main() -> None:
    args = parse_args()
    resolved = str(args.out_dir.resolve()).lower()
    if (
        not args.out_dir.name.lower().startswith("non_dp")
        or "production" in resolved
        or "certified" in resolved
    ):
        raise RuntimeError("Shrinkage probe must be isolated in a NON_DP root")
    if args.out_dir.exists():
        raise RuntimeError("Shrinkage output directory must be new")

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
    mass_matrix = load_npz(args.public_matrix_cache).tocsr()
    matrix = oracle.legacy_matrix_from_mass_cache(
        mass_matrix, mass_runner.component_slices(context96, context384)
    )
    public_mean = np.asarray(
        matrix.T @ np.full(matrix.shape[0], 1.0 / 8.0)
    ).ravel()
    release_path = args.dp_development_dir / "multires_graph_flow_release.npz"
    with np.load(release_path, allow_pickle=False) as archive:
        released = {
            name: np.asarray(archive[name], dtype=float)
            for name in legacy.BLOCK_BUDGETS
        }
    target = np.concatenate(
        [released[name].ravel() for name in legacy.BLOCK_BUDGETS]
    )
    variance = discrete_laplace_variance_normalized(
        epsilon=float(args.epsilon_route), sensitivity=legacy.ROUTE_LATTICE
    )
    shrunk, shrinkage = positive_part_block_shrinkage(
        target,
        public_mean,
        {name: int(released[name].size) for name in legacy.BLOCK_BUDGETS},
        noise_variance=variance,
    )
    probability, optimizer = fit_projection(
        matrix,
        shrunk,
        groups=len(point_count),
        candidates_per_group=8,
        noise_variance=0.25
        + 2.0 * len(point_count)
        / (float(args.epsilon_route) ** 2 * len(point_count) ** 2),
        max_iterations=180,
    )
    probability, trust = kl_trust_region(probability, float(args.kl_radius))
    draws = np.random.default_rng(int(args.seed) + 50_001).random(len(point_count))
    selected = np.asarray(
        [
            min(int(np.searchsorted(np.cumsum(row), draw, side="right")), 7)
            for row, draw in zip(probability, draws)
        ],
        dtype=np.int8,
    )
    staging = args.out_dir.parent / f".{args.out_dir.name}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "DO_NOT_RELEASE.txt").write_text(
        "NON-DP fixed-seed shrinkage development artifact.\n", encoding="ascii"
    )
    with (staging / "maxent_projected.pkl").open("wb") as handle:
        pickle.dump(
            materialize(candidate_nodes, selected, point_count, coords),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    control = args.dp_development_dir / "uniform_candidate_control.pkl"
    shutil.copyfile(control, staging / "uniform_candidate_control.pkl")
    np.save(staging / "selection_probability.npy", probability, allow_pickle=False)
    protocol = {
        "algorithm": "Noise-Energy Residual Shrinkage toward public candidate moments",
        "classification": "NON_DP_FIXED_SEED_DO_NOT_RELEASE",
        "certified_release": False,
        "dp_safe_if_applied_to_a_certified_transcript": True,
        "statistical_basis": (
            "For the raw additive mechanism, E||Y-mu||^2 equals "
            "||theta-mu||^2+d*sigma^2 for any independent zero-mean noise with "
            "finite variance. This development probe applies the raw-noise energy "
            "as a conservative calibration after simplex post-processing; it does "
            "not claim a James-Stein risk-dominance theorem."
        ),
        "normalized_discrete_laplace_variance": variance,
        "shrinkage": shrinkage,
        "optimizer": optimizer,
        "kl_trust_region": trust,
        "candidate_chunk_sha256": [
            item["candidate_nodes_sha256"] for item in chunk_manifests
        ],
        "dp_development_release_sha256": lineage.sha256_file(release_path),
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
