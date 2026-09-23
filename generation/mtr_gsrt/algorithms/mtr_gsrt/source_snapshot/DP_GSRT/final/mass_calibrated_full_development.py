"""Five-seed full-size development runner for mass-calibrated OD graph flow."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

import graph_cycle_dp_production_sanitizer as lineage
import graph_voronoi_doptimal_support_probe as support
import mass_calibrated_graph_flow as calibrated
import route_structure_potential_experiment as route
from assemble_graph_cycle_production_release import load_verified_candidates
from audit_two_level_semimarkov_dp import build_context
from public_utils import load_osm_ways
from resample_production_trust_sweep import materialize
from two_level_maxent_projection import fit_projection, kl_trust_region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--public-matrix-cache", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    parser.add_argument("--kl-radius", type=float, default=0.60)
    parser.add_argument("--max-iterations", type=int, default=180)
    parser.add_argument(
        "--conditional-flow-preconditioning", action="store_true", required=True
    )
    parser.add_argument("--acknowledge-non-dp-development", action="store_true", required=True)
    return parser.parse_args()


def legacy_folder(root: Path, seed: int) -> Path:
    if int(seed) == 17:
        return root / "coarsefine_odflow_full_seed17"
    return root / f"NON_DP_coarsefine_odflow_full_seed{int(seed)}"


def candidate_matrix(
    candidate_nodes, coords: np.ndarray, context96, context384
) -> tuple[csr_matrix, dict[str, tuple[int, int]]]:
    template = calibrated.empty_components(context96, context384)
    offsets, dimension = {}, 0
    for name in calibrated.COMPONENT_BUDGETS:
        offsets[name] = dimension
        dimension += int(template[name].size)
    groups, width = len(candidate_nodes), 8
    indptr = np.zeros(groups * width + 1, dtype=np.int64)
    index_parts, value_parts = [], []
    row = 0
    for request, candidates in enumerate(candidate_nodes, start=1):
        if len(candidates) != width:
            raise RuntimeError("Candidate width mismatch")
        for nodes in candidates:
            trajectory = coords[np.asarray(nodes, dtype=int)]
            components = calibrated.graph_components(trajectory, context96, context384)
            indices, values = [], []
            for name, budget in calibrated.COMPONENT_BUDGETS.items():
                flat = np.asarray(components[name], dtype=float).ravel()
                nonzero = np.flatnonzero(flat)
                indices.append(offsets[name] + nonzero)
                values.append(float(budget) / calibrated.ROUTE_LATTICE * flat[nonzero])
            current_indices = np.concatenate(indices).astype(np.int32, copy=False)
            current_values = np.concatenate(values)
            index_parts.append(current_indices)
            value_parts.append(current_values)
            row += 1
            indptr[row] = indptr[row - 1] + len(current_indices)
        if request % 500 == 0:
            print(f"[mass-calibrated-features] {request}/{groups}", flush=True)
    matrix = csr_matrix(
        (np.concatenate(value_parts), np.concatenate(index_parts), indptr),
        shape=(groups * width, dimension),
    )
    slices = {
        name: (offsets[name], offsets[name] + int(template[name].size))
        for name in calibrated.COMPONENT_BUDGETS
    }
    return matrix, slices


def component_slices(context96, context384) -> dict[str, tuple[int, int]]:
    template = calibrated.empty_components(context96, context384)
    result, offset = {}, 0
    for name in calibrated.COMPONENT_BUDGETS:
        result[name] = (offset, offset + int(template[name].size))
        offset = result[name][1]
    return result


def conditional_precondition(
    matrix: csr_matrix,
    target: np.ndarray,
    slices: dict[str, tuple[int, int]],
    mass: dict,
) -> tuple[csr_matrix, np.ndarray, dict]:
    """Separate transition incidence from conditional direction moments."""
    activity = {
        "fine96_flow": float(mass["fine96_active_fraction"]),
        "fine384_flow": float(mass["fine384_active_fraction"]),
        "od_conditioned_fine96_flow": float(mass["fine96_active_fraction"]),
    }
    if any(not np.isfinite(value) or value <= 0.0 or value > 1.0 for value in activity.values()):
        raise RuntimeError("Conditional moments require positive DP-released activity")
    column_scale = np.ones(matrix.shape[1], dtype=float)
    block_scale = {}
    for name, fraction in activity.items():
        start, stop = slices[name]
        block_scale[name] = 1.0 / fraction
        column_scale[start:stop] = block_scale[name]
    return (
        matrix.multiply(column_scale).tocsr(),
        np.asarray(target, dtype=float) * column_scale,
        {
            "definition": (
                "invertible block preconditioner: activity is the extensive margin; "
                "flow/activity is the intensive conditional-direction margin"
            ),
            "dp_postprocessing_only": True,
            "block_scale": block_scale,
        },
    )


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.kl_radius) or float(args.kl_radius) < 0.0:
        raise ValueError("--kl-radius must be finite and nonnegative")
    if not args.out_root.name.lower().startswith("non_dp"):
        raise RuntimeError("Development out-root name must start with NON_DP")
    resolved_output = str(args.out_root.resolve()).lower()
    if "production" in resolved_output or "certified" in resolved_output:
        raise RuntimeError("NON_DP development output must be outside production/certified roots")
    if args.out_root.exists():
        raise RuntimeError("Development out-root must not exist")

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
    if len(point_count) != calibrated.PUBLIC_CAPACITY:
        raise RuntimeError("Expected the full fixed public capacity")

    real = support.load_trajectories(support.DEFAULT_REAL)
    if len(real) != calibrated.PUBLIC_CAPACITY:
        raise RuntimeError("Expected exactly 17,123 private development trajectories")
    osm_path = (
        Path(__file__).resolve().parents[2]
        / "ara_final"
        / "evidence"
        / "tables"
        / "osm_cache_beijing.pkl"
    )
    if not osm_path.is_file():
        raise RuntimeError(f"Public OSM cache is missing: {osm_path}")
    coords, graph = route.prepare_graph(
        [], bbox=support.BBOX, osm_ways=load_osm_ways(osm_path), raw_graph=False
    )
    context96, graph96 = build_context(coords, graph, 24, 96, 4, 3)
    context384, graph384 = build_context(coords, graph, 24, 384, 4, 3)
    if (
        len(context96.fine_edge_index) != 438
        or len(context384.fine_edge_index) != 1746
    ):
        raise RuntimeError(
            "Public quotient-graph dimensions differ from the frozen activity split"
        )
    totals = calibrated.aggregate(real, context96, context384)
    cache_manifest = args.public_matrix_cache.with_suffix(".json")
    cache_lineage = {
        "public_only": True,
        "candidate_chunk_sha256": [
            item["candidate_nodes_sha256"] for item in chunk_manifests
        ],
        "public_osm_sha256": lineage.sha256_file(osm_path),
        "component_budgets": calibrated.COMPONENT_BUDGETS,
    }
    slices = component_slices(context96, context384)
    if args.public_matrix_cache.is_file() and cache_manifest.is_file():
        observed = json.loads(cache_manifest.read_text(encoding="utf-8"))
        if observed != cache_lineage:
            raise RuntimeError("Public candidate-matrix cache lineage mismatch")
        matrix = load_npz(args.public_matrix_cache).tocsr()
    elif args.public_matrix_cache.exists() or cache_manifest.exists():
        raise RuntimeError("Public matrix cache is incomplete")
    else:
        matrix, computed_slices = candidate_matrix(
            candidate_nodes, coords, context96, context384
        )
        if computed_slices != slices:
            raise AssertionError("Candidate feature slices changed during construction")
        args.public_matrix_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.public_matrix_cache.with_suffix(".staging.npz")
        save_npz(temporary, matrix, compressed=True)
        os.replace(temporary, args.public_matrix_cache)
        cache_manifest.write_text(
            json.dumps(cache_lineage, indent=2), encoding="utf-8"
        )
    expected_shape = (len(point_count) * 8, slices[next(reversed(slices))][1])
    if matrix.shape != expected_shape:
        raise RuntimeError("Public candidate-matrix cache shape mismatch")

    args.out_root.mkdir(parents=True, exist_ok=False)
    (args.out_root / "DO_NOT_RELEASE.txt").write_text(
        "NON-DP fixed-seed private development root. Never certify or distribute.\n",
        encoding="ascii",
    )
    epsilon = Fraction(1, 5)
    for seed in args.seeds:
        final_output = args.out_root / f"seed{int(seed)}"
        output = args.out_root / f".seed{int(seed)}.staging"
        if final_output.exists() or output.exists():
            raise RuntimeError(f"Seed output or staging path already exists: {final_output}")
        output.mkdir(parents=False, exist_ok=False)
        (output / "DO_NOT_RELEASE.txt").write_text(
            "NON-DP fixed-seed development artifact.\n", encoding="ascii"
        )
        released, sampler, mass = calibrated.release(
            totals,
            capacity=calibrated.PUBLIC_CAPACITY,
            exact_rng=random.Random(int(seed) + 79_919),
            epsilon=epsilon,
            production_randomness=False,
        )
        target = np.concatenate(
            [released[name].ravel() for name in calibrated.COMPONENT_BUDGETS]
        )
        groups = len(point_count)
        fit_matrix, fit_target, preconditioner = conditional_precondition(
            matrix, target, slices, mass
        )
        probability, optimizer = fit_projection(
            fit_matrix,
            fit_target,
            groups=groups,
            candidates_per_group=8,
            noise_variance=0.25
            + 2.0 * groups
            / (float(epsilon) ** 2 * calibrated.PUBLIC_CAPACITY**2),
            max_iterations=int(args.max_iterations),
        )
        probability, trust = kl_trust_region(probability, float(args.kl_radius))
        draws = np.random.default_rng(int(seed) + 50_001).random(groups)
        selected = np.asarray(
            [
                min(int(np.searchsorted(np.cumsum(row), draw, side="right")), 7)
                for row, draw in zip(probability, draws)
            ],
            dtype=np.int8,
        )
        with (output / "maxent_projected.pkl").open("wb") as handle:
            pickle.dump(
                materialize(candidate_nodes, selected, point_count, coords),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        legacy_control = legacy_folder(args.out_root.parent, int(seed)) / (
            "uniform_candidate_control.pkl"
        )
        if not legacy_control.is_file():
            raise RuntimeError(f"Missing frozen same-candidate control: {legacy_control}")
        shutil.copyfile(legacy_control, output / "uniform_candidate_control.pkl")
        np.save(output / "selection_probability.npy", probability, allow_pickle=False)
        np.savez_compressed(
            output / "mass_calibrated_release.npz", **released
        )
        protocol = {
            "algorithm": "mass-calibrated hierarchical coarse-fine OD graph-flow maximum entropy",
            "development_artifact": True,
            "certified_release": False,
            "privacy_status": "NON_DP_FIXED_SEED_DO_NOT_RELEASE",
            "seed": int(seed),
            "counterfactual_single_fresh_run_privacy_only": {
                "epsilon_route_rational": "1/5",
                "integer_l1_sensitivity": calibrated.ROUTE_LATTICE,
                "delta": 0.0,
            },
            "actual_development_privacy": (
                "NONE: deterministic fixed-seed transcript; never release. "
                "Multiple seeds are not one epsilon=1/5 mechanism."
            ),
            "activity_split_rule": "gamma=1/(1+d^(1/3)); nearest-integer public budget",
            "component_budgets": calibrated.COMPONENT_BUDGETS,
            "component_dimensions": {
                name: int(value.size)
                for name, value in calibrated.empty_components(
                    context96, context384
                ).items()
            },
            "mass_calibration": mass,
            "conditional_flow_preconditioner": preconditioner,
            "sampler": sampler,
            "optimizer": optimizer,
            "kl_trust_region": trust,
            "graph96": graph96,
            "graph384": graph384,
            "public_osm_sha256": lineage.sha256_file(osm_path),
            "public_candidate_matrix_sha256": lineage.sha256_file(
                args.public_matrix_cache
            ),
            "candidate_chunk_sha256": [
                item["candidate_nodes_sha256"] for item in chunk_manifests
            ],
            "uniform_control_source": str(legacy_control.resolve()),
            "uniform_control_sha256": lineage.sha256_file(legacy_control),
            "selected_candidate_counts": np.bincount(
                selected, minlength=8
            ).tolist(),
            "outputs": {
                name: lineage.sha256_file(output / name)
                for name in (
                    "maxent_projected.pkl",
                    "uniform_candidate_control.pkl",
                    "selection_probability.npy",
                    "mass_calibrated_release.npz",
                )
            },
        }
        (output / "protocol.json").write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )
        os.replace(output, final_output)
        print(
            json.dumps(
                {
                    "seed": int(seed),
                    "output": str(final_output.resolve()),
                    "mass_calibration": mass,
                    "optimizer": optimizer,
                    "trust": trust,
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
