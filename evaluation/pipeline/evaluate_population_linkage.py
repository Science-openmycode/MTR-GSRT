"""Black-box population-linkage test for a synthetic trajectory release."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def load(path: Path) -> list[np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        for key in ("trajectories", "synthetic", "routes", "data", "dp_gsrt_portal_qrsp"):
            if key in payload:
                payload = payload[key]
                break
    result = []
    for trajectory in payload:
        array = np.asarray(trajectory, dtype=float)
        if array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 2:
            result.append(array[:, :2])
    if not result:
        raise ValueError(f"No coordinate trajectories found in {path}")
    return result


def landmarks(trajectories: list[np.ndarray], count: int) -> np.ndarray:
    vectors = []
    for trajectory in trajectories:
        locations = np.linspace(0.0, len(trajectory) - 1, count)
        low = np.floor(locations).astype(int)
        high = np.ceil(locations).astype(int)
        fraction = (locations - low)[:, None]
        vectors.append(trajectory[low] * (1.0 - fraction) + trajectory[high] * fraction)
    return np.stack(vectors)


def nearest_distance(query: np.ndarray, release: np.ndarray, batch_size: int = 8) -> np.ndarray:
    radius_m = 6_371_008.8
    results = []
    release_lat = np.radians(release[None, :, :, 0])
    release_lon = np.radians(release[None, :, :, 1])
    for start in range(0, len(query), batch_size):
        block = query[start:start + batch_size]
        qlat = np.radians(block[:, None, :, 0])
        qlon = np.radians(block[:, None, :, 1])
        h = np.sin((release_lat - qlat) / 2.0) ** 2 + np.cos(qlat) * np.cos(release_lat) * np.sin((release_lon - qlon) / 2.0) ** 2
        distances = 2.0 * radius_m * np.arcsin(np.minimum(1.0, np.sqrt(h)))
        results.append(distances.mean(axis=2).min(axis=1))
    return np.concatenate(results)


def summary(scores: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict[str, float | list[float]]:
    auc = float(roc_auc_score(labels, scores))
    threshold = float(np.median(scores))
    accuracy = float(balanced_accuracy_score(labels, scores >= threshold))
    bootstrap = []
    for _ in range(2_000):
        indices = rng.integers(0, len(labels), size=len(labels))
        if len(np.unique(labels[indices])) == 2:
            bootstrap.append(roc_auc_score(labels[indices], scores[indices]))
    return {
        "auc": auc,
        "balanced_accuracy_median_threshold": accuracy,
        "auc_bootstrap95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", required=True, type=Path)
    parser.add_argument("--nonmembers", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--landmarks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    members, nonmembers, release = load(args.members), load(args.nonmembers), load(args.release)
    n = min(len(members), len(nonmembers))
    rng = np.random.default_rng(args.seed)
    members = [members[i] for i in rng.choice(len(members), n, replace=False)]
    nonmembers = [nonmembers[i] for i in rng.choice(len(nonmembers), n, replace=False)]
    representation = landmarks(release, args.landmarks)
    member_distances = nearest_distance(landmarks(members, args.landmarks), representation)
    nonmember_distances = nearest_distance(landmarks(nonmembers, args.landmarks), representation)
    labels = np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)]
    scores = -np.r_[member_distances, nonmember_distances]
    report = {
        "attack": "population_nearest_landmark_linkage",
        "interpretation": "black-box member/non-member separation, not a one-to-one correspondence claim",
        "landmarks": args.landmarks,
        "members": n,
        "release_size": len(release),
        "member_nearest_distance_m_mean": float(member_distances.mean()),
        "nonmember_nearest_distance_m_mean": float(nonmember_distances.mean()),
        **summary(scores, labels, rng),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
