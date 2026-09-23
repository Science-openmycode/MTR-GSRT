"""Maximum-entropy I-projection over a finite DP-postprocessed route bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import logsumexp


ARA = Path(__file__).resolve().parents[1]
for path in [ARA / "dp_reward_exploration", ARA / "public_release", ARA / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import graph_voronoi_doptimal_support_probe as support  # noqa: E402
import route_structure_potential_experiment as route  # noqa: E402
from audit_two_level_semimarkov_dp import build_context  # noqa: E402
from public_utils import load_osm_ways  # noqa: E402
from two_level_semimarkov_query import WEIGHTS, trajectory_blocks  # noqa: E402


CANDIDATE_FILES = [
    "two_level_public.pkl",
    "two_level_dp_coarse.pkl",
    "two_level_dp_fine.pkl",
    "two_level_dp.pkl",
]
BLOCK_ORDER = list(WEIGHTS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sparse_feature_row(blocks: dict[str, np.ndarray], offsets: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    values = []
    for name in BLOCK_ORDER:
        flat = np.asarray(blocks[name], dtype=float).ravel()
        nonzero = np.flatnonzero(flat)
        indices.extend((offsets[name] + nonzero).tolist())
        values.extend((float(WEIGHTS[name]) * flat[nonzero]).tolist())
    return np.asarray(indices, dtype=np.int32), np.asarray(values, dtype=float)


def candidate_matrix(candidates: list[list[np.ndarray]], context) -> tuple[csr_matrix, dict[str, int]]:
    count = len(candidates[0])
    width = len(candidates)
    template = trajectory_blocks(np.empty((0, 2)), context)
    offsets = {}
    dimension = 0
    for name in BLOCK_ORDER:
        offsets[name] = dimension
        dimension += int(np.prod(template[name].shape))
    total_rows = count * width
    indptr = np.empty(total_rows + 1, dtype=np.int64)
    indptr[0] = 0
    index_parts = []
    value_parts = []
    row = 0
    for request in range(count):
        for candidate in range(width):
            indices, values = sparse_feature_row(trajectory_blocks(candidates[candidate][request], context), offsets)
            index_parts.append(indices)
            value_parts.append(values)
            row += 1
            indptr[row] = indptr[row - 1] + len(indices)
        if (request + 1) % 200 == 0:
            print(f"[maxent-features] {request + 1}/{count}", flush=True)
    all_indices = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int32)
    all_values = np.concatenate(value_parts) if value_parts else np.empty(0, dtype=float)
    matrix = csr_matrix((all_values, all_indices, indptr), shape=(total_rows, dimension), dtype=float)
    return matrix, offsets


def flatten_release(released: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(released[name], dtype=float).ravel() for name in BLOCK_ORDER])


def fit_projection(
    matrix: csr_matrix,
    target: np.ndarray,
    *,
    groups: int,
    candidates_per_group: int,
    noise_variance: float,
    max_iterations: int,
) -> tuple[np.ndarray, dict]:
    dimension = matrix.shape[1]

    def objective(theta: np.ndarray):
        scores = np.asarray(matrix @ theta).reshape(groups, candidates_per_group)
        normalizer = logsumexp(scores, axis=1, keepdims=True)
        probability = np.exp(scores - normalizer)
        value = float(np.sum(normalizer) - groups * np.log(candidates_per_group) - theta @ target)
        value += 0.5 * float(noise_variance) * float(theta @ theta)
        gradient = np.asarray(matrix.T @ probability.ravel()).ravel() - target + float(noise_variance) * theta
        return value, gradient

    result = minimize(
        objective,
        np.zeros(dimension, dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iterations), "ftol": 1e-10, "gtol": 1e-6, "maxls": 40},
    )
    scores = np.asarray(matrix @ result.x).reshape(groups, candidates_per_group)
    probability = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
    expected = np.asarray(matrix.T @ probability.ravel()).ravel()
    diagnostics = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "moment_l1": float(np.sum(np.abs(expected - target))),
        "moment_l2": float(np.linalg.norm(expected - target)),
        "mean_candidate_entropy": float(np.mean(-np.sum(probability * np.log(np.maximum(probability, 1e-300)), axis=1))),
        "uniform_entropy": float(np.log(candidates_per_group)),
        "mean_max_probability": float(np.mean(np.max(probability, axis=1))),
    }
    return probability, diagnostics


def kl_trust_region(probability: np.ndarray, radius: float) -> tuple[np.ndarray, dict]:
    """Temper an exponential-family solution to an average KL-to-uniform ball."""
    if not np.isfinite(radius) or float(radius) < 0.0:
        raise ValueError("KL radius must be finite and nonnegative")
    width = probability.shape[1]
    log_uniform = -float(np.log(width))

    def tempered(beta: float) -> np.ndarray:
        logits = float(beta) * np.log(np.maximum(probability, 1e-300))
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

    def mean_kl(values: np.ndarray) -> float:
        return float(
            np.mean(np.sum(values * (np.log(np.maximum(values, 1e-300)) - log_uniform), axis=1))
        )

    original_kl = mean_kl(probability)
    if original_kl <= float(radius) + 1e-12:
        return probability, {
            "active": False,
            "radius": float(radius),
            "temperature_multiplier": 1.0,
            "mean_kl_before": original_kl,
            "mean_kl_after": original_kl,
            "pinsker_mean_tv_upper_bound": float(np.sqrt(0.5 * float(radius))),
        }
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if mean_kl(tempered(middle)) <= float(radius):
            low = middle
        else:
            high = middle
    constrained = tempered(low)
    return constrained, {
        "active": True,
        "radius": float(radius),
        "temperature_multiplier": float(low),
        "mean_kl_before": original_kl,
        "mean_kl_after": mean_kl(constrained),
        "pinsker_mean_tv_upper_bound": float(np.sqrt(0.5 * float(radius))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--public-capacity", type=int, default=17123)
    parser.add_argument("--epsilon-route", type=float, default=0.20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--route-release", type=Path)
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--production-release", action="store_true")
    parser.add_argument("--kl-radius", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.epsilon_route) or float(args.epsilon_route) <= 0.0:
        raise ValueError("--epsilon-route must be finite and strictly positive")
    manifest_path = args.candidate_dir / "mechanism_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Missing mechanism_manifest.json for candidate provenance")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported candidate provenance manifest")
    if args.production_release:
        if not manifest.get("certified_release") or manifest.get("development_artifact"):
            raise RuntimeError("Production projection requires certified non-development lineage")
        if manifest.get("privacy_lineage", {}).get("noise_implementation") != "bit-exact two-sided geometric":
            raise RuntimeError("Production projection requires exact discrete-noise lineage")
        if args.route_release is not None:
            raise RuntimeError("Production projection must use the route release bound inside candidate-dir")
        if int(args.public_capacity) != 17123 or Fraction(str(float(args.epsilon_route))) != Fraction(1, 5):
            raise RuntimeError("Production projection requires frozen capacity and epsilon=1/5")
        if int(args.max_iterations) != 120 or int(args.max_candidates or 4) != 4:
            raise RuntimeError("Production projection requires frozen K=4 and 120 optimizer iterations")
        if args.kl_radius is None or Fraction(str(float(args.kl_radius))) != Fraction(1, 5):
            raise RuntimeError("Production projection requires the frozen KL radius tau=1/5")
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise RuntimeError("Production projection output directory must be new and empty")
    candidate_files = sorted(path.name for path in args.candidate_dir.glob("public_candidate_*.pkl"))
    declared_candidates = int(manifest.get("configuration", {}).get("candidate_count", -1))
    declared_candidate_files = set(manifest.get("candidate_sha256", {}))
    if len(candidate_files) != declared_candidates or set(candidate_files) != declared_candidate_files:
        raise RuntimeError("Candidate-file count does not match the mechanism manifest")
    if len(candidate_files) < 2:
        candidate_files = CANDIDATE_FILES
    if args.max_candidates is not None:
        candidate_files = candidate_files[: int(args.max_candidates)]
    if len(candidate_files) < 2:
        raise RuntimeError("Maximum-entropy projection requires at least two candidates per request")
    candidates = []
    for filename in candidate_files:
        expected_hash = manifest.get("candidate_sha256", {}).get(filename)
        candidate_path = args.candidate_dir / filename
        if expected_hash is None or sha256_file(candidate_path) != expected_hash:
            raise RuntimeError(f"Candidate provenance hash mismatch: {filename}")
        with candidate_path.open("rb") as handle:
            candidates.append([np.asarray(item, dtype=float) for item in pickle.load(handle)])
    counts = {len(items) for items in candidates}
    if len(counts) != 1:
        raise RuntimeError("Candidate files are not request-aligned")
    groups = counts.pop()
    width = len(candidates)
    declared_groups = int(manifest.get("configuration", {}).get("output_slots", -1))
    if groups != declared_groups:
        raise RuntimeError("Candidate request count does not match the mechanism manifest")
    if args.production_release:
        configuration = manifest.get("configuration", {})
        required_configuration = {
            "public_capacity": 17123,
            "output_slots": 17123,
            "candidate_count": 4,
            "route_integer_lattice": 1_000_000,
            "epsilon_route_rational": "1/5",
        }
        if any(configuration.get(key) != value for key, value in required_configuration.items()):
            raise RuntimeError("Production manifest configuration does not match the frozen specification")

    osm_path = ARA.parent / "ara_final" / "evidence" / "tables" / "osm_cache_beijing.pkl"
    if sha256_file(osm_path) != manifest.get("public_osm", {}).get("sha256"):
        raise RuntimeError("Public OSM graph does not match the candidate lineage")
    osm = load_osm_ways(osm_path)
    coords, graph = route.prepare_graph([], bbox=support.BBOX, osm_ways=osm, raw_graph=False)
    context, _ = build_context(coords, graph, 24, 96, 4, 3)
    release_path = args.route_release or (args.candidate_dir / "route_release.npz")
    if not release_path.exists():
        raise RuntimeError(
            "Missing sanitized route release; maximum-entropy projection must not query raw trajectories"
        )
    expected_release_hash = manifest.get("privacy_lineage", {}).get("route_release_sha256")
    if expected_release_hash is None or sha256_file(release_path) != expected_release_hash:
        raise RuntimeError("Sanitized route release does not match the candidate privacy lineage")
    expected_integer_hash = manifest.get("privacy_lineage", {}).get("integer_route_release_sha256")
    if expected_integer_hash is not None:
        integer_path = args.candidate_dir / "route_release_integer_noisy.npz"
        if not integer_path.exists() or sha256_file(integer_path) != expected_integer_hash:
            raise RuntimeError("Integer route release does not match the candidate privacy lineage")
    lineage = manifest["privacy_lineage"]
    if abs(float(lineage["route_epsilon"]) - float(args.epsilon_route)) > 1e-12:
        raise RuntimeError("Route epsilon does not match the candidate privacy lineage")
    with np.load(release_path, allow_pickle=False) as archive:
        missing = set(BLOCK_ORDER) - set(archive.files)
        if missing:
            raise RuntimeError(f"Sanitized route release is missing blocks: {sorted(missing)}")
        released = {name: np.asarray(archive[name], dtype=float) for name in BLOCK_ORDER}
    expected_shapes = {
        name: trajectory_blocks(np.empty((0, 2)), context)[name].shape for name in BLOCK_ORDER
    }
    for name in BLOCK_ORDER:
        if released[name].shape != expected_shapes[name] or not np.all(np.isfinite(released[name])):
            raise RuntimeError(f"Invalid sanitized route-release block: {name}")
    target = flatten_release(released) * (float(groups) / float(args.public_capacity))
    matrix, _ = candidate_matrix(candidates, context)
    # In the aggregate dual, 1/4 is the worst-case variance of a bounded
    # [0,1] candidate feature. The second term is the normalized Laplace
    # moment variance after multiplying the average objective by N.
    sampling_variance_bound = 0.25
    dp_noise_term = 2.0 * float(groups) / (
        float(args.epsilon_route) ** 2 * float(args.public_capacity) ** 2
    )
    noise_variance = sampling_variance_bound + dp_noise_term
    probability, diagnostics = fit_projection(
        matrix,
        target,
        groups=groups,
        candidates_per_group=width,
        noise_variance=noise_variance,
        max_iterations=args.max_iterations,
    )
    trust_region = None
    if args.kl_radius is not None:
        probability, trust_region = kl_trust_region(probability, args.kl_radius)

    rng = np.random.default_rng(int(args.seed) + 5001)
    projected = []
    uniform = []
    selected = []
    for index in range(groups):
        choice = int(rng.choice(width, p=probability[index]))
        uniform_choice = int(rng.integers(0, width))
        projected.append(candidates[choice][index])
        uniform.append(candidates[uniform_choice][index])
        selected.append(choice)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    projected_path = args.out_dir / "maxent_projected.pkl"
    uniform_path = args.out_dir / "uniform_candidate_control.pkl"
    with projected_path.open("wb") as handle:
        pickle.dump(projected, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with uniform_path.open("wb") as handle:
        pickle.dump(uniform, handle, protocol=pickle.HIGHEST_PROTOCOL)
    selection_hist = np.bincount(np.asarray(selected, dtype=int), minlength=width)
    report = {
        "algorithm": "KL-trust-region finite-support maximum-entropy I-projection",
        "development_artifact": not bool(args.production_release),
        "certified_release": bool(args.production_release),
        "privacy": (
            "certified post-processing of one sensitivity-one epsilon_route-DP moment release"
            if args.production_release
            else "development-only post-processing; no production privacy certification"
        ),
        "route_release": str(release_path.resolve()),
        "groups": int(groups),
        "candidates_per_group": int(width),
        "candidate_labels": candidate_files,
        "noise_variance_rule": "1/4 bounded-feature variance + 2N/(N0^2 epsilon_route^2)",
        "noise_variance": float(noise_variance),
        "optimizer": diagnostics,
        "kl_trust_region": trust_region,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "selected_candidate_counts": selection_hist.tolist(),
        "outputs": {"projected": projected_path.name, "uniform": uniform_path.name},
    }
    (args.out_dir / "protocol.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
