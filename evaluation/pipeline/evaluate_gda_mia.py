"""Run the black-box Generated-Distribution MIA (GDA-MIA) on trajectory releases.

This is a trajectory-domain implementation of Zhang et al. (WACV 2024).  It
uses released synthetic trajectories as positive training examples and a
disjoint, same-city reference collection as negative examples.  A fixed
classifier then scores unseen member and non-member candidate trajectories.
The interface is deliberately release-only: it neither queries a model nor
uses record/output pairings.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def load(path: Path) -> list[np.ndarray]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, dict):
        for key in ("trajectories", "synthetic", "routes", "data", "dp_gsrt_portal_qrsp"):
            if key in value:
                value = value[key]
                break
    result = []
    for item in value:
        array = np.asarray(item, dtype=float)
        if array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 2:
            result.append(array[:, :2])
    if not result:
        raise ValueError(f"No coordinate trajectories found in {path}")
    return result


def features(trajectories: list[np.ndarray], bbox: tuple[float, float, float, float], landmarks: int) -> np.ndarray:
    """Public fixed-length trajectory representation for the attack classifier."""
    lat0, lat1, lon0, lon1 = bbox
    vectors = []
    for trajectory in trajectories:
        positions = np.linspace(0.0, len(trajectory) - 1, landmarks)
        low = np.floor(positions).astype(int)
        high = np.ceil(positions).astype(int)
        fraction = (positions - low)[:, None]
        points = trajectory[low] * (1.0 - fraction) + trajectory[high] * fraction
        points[:, 0] = (points[:, 0] - lat0) / (lat1 - lat0)
        points[:, 1] = (points[:, 1] - lon0) / (lon1 - lon0)
        vectors.append(points.ravel())
    return np.asarray(vectors, dtype=float)


def bootstrap_auc(scores: np.ndarray, labels: np.ndarray, seed: int, rounds: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    member = np.flatnonzero(labels == 1)
    nonmember = np.flatnonzero(labels == 0)
    values = []
    for _ in range(rounds):
        indices = np.r_[rng.choice(member, len(member), replace=True), rng.choice(nonmember, len(nonmember), replace=True)]
        values.append(float(roc_auc_score(labels[indices], scores[indices])))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def low_fpr_tpr(scores: np.ndarray, labels: np.ndarray, limit: float) -> tuple[float, float]:
    negatives = np.sort(scores[labels == 0])[::-1]
    allowed = max(1, int(np.floor(limit * len(negatives))))
    threshold = float(negatives[allowed - 1])
    predicted = scores >= threshold
    return float(predicted[labels == 0].mean()), float(predicted[labels == 1].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", required=True, type=Path)
    parser.add_argument("--nonmembers", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--release", nargs=2, action="append", required=True, metavar=("NAME", "PATH"))
    parser.add_argument("--bbox", required=True, nargs=4, type=float)
    parser.add_argument("--landmarks", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--max-train-per-class", type=int, default=12000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    members, nonmembers, reference = load(args.members), load(args.nonmembers), load(args.reference)
    n = min(len(members), len(nonmembers), args.max_candidates)
    members = [members[i] for i in rng.choice(len(members), n, replace=False)]
    nonmembers = [nonmembers[i] for i in rng.choice(len(nonmembers), n, replace=False)]
    query = np.vstack([features(members, tuple(args.bbox), args.landmarks), features(nonmembers, tuple(args.bbox), args.landmarks)])
    labels = np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)]
    reference_features = features(reference, tuple(args.bbox), args.landmarks)
    rows = []
    for row_index, (name, release_path) in enumerate(args.release):
        generated = features(load(Path(release_path)), tuple(args.bbox), args.landmarks)
        train_n = min(len(generated), len(reference_features), args.max_train_per_class)
        pos = generated[rng.choice(len(generated), train_n, replace=False)]
        neg = reference_features[rng.choice(len(reference_features), train_n, replace=False)]
        train_x = np.vstack([pos, neg])
        train_y = np.r_[np.ones(train_n, dtype=int), np.zeros(train_n, dtype=int)]
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=args.seed + row_index,
        )
        classifier.fit(train_x, train_y)
        scores = classifier.predict_proba(query)[:, 1]
        auc = float(roc_auc_score(labels, scores))
        ci_low, ci_high = bootstrap_auc(scores, labels, args.seed + 997 * row_index, args.bootstrap)
        fpr1, tpr1 = low_fpr_tpr(scores, labels, 0.01)
        fpr5, tpr5 = low_fpr_tpr(scores, labels, 0.05)
        row = {
            "method": name,
            "attack": "GDA-MIA",
            "attack_classifier": "HistGradientBoostingClassifier-fixed",
            "auc": auc,
            "auc_bootstrap95_low": ci_low,
            "auc_bootstrap95_high": ci_high,
            "accuracy": float(((scores >= 0.5) == labels).mean()),
            "tpr_at_1pct_fpr": tpr1,
            "realized_fpr_at_1pct": fpr1,
            "tpr_at_5pct_fpr": tpr5,
            "realized_fpr_at_5pct": fpr5,
            "candidate_members_per_class": n,
            "generated_positive_count": train_n,
            "reference_negative_count": train_n,
            "landmarks": args.landmarks,
        }
        rows.append(row)
        print(f"[GDA-MIA] {name}: AUC={auc:.4f}, TPR@1%={tpr1:.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix(".json").write_text(json.dumps({
        "attack": "GDA-MIA trajectory adaptation",
        "citation": "Zhang et al., Generated Distributions Are All You Need for Membership Inference Attacks Against Generative Models, WACV 2024",
        "threat_model": "black-box release-only: generated trajectories are positive attack examples; disjoint same-domain reference trajectories are negative examples",
        "warning": "A cross-method comparison requires all releases to have been generated from the same frozen member-training manifest.",
        "bbox": args.bbox,
        "rows": rows,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
