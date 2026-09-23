"""Materialize a disjoint member/non-member/reference attack protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / "outputs" / "kdd_revised" / "split"
DEFAULT_OUT = ROOT / "outputs" / "privacy_attack_audit" / "manifests" / "beijing_kdd_split_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> list[np.ndarray]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, dict):
        for key in ("trajectories", "routes", "synthetic", "data"):
            if key in value:
                value = value[key]
                break
    return [np.asarray(item, dtype=float)[:, :2] for item in value if len(item) >= 2]


def save(path: Path, trajectories: list[np.ndarray]) -> None:
    with path.open("wb") as handle:
        pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--attack-candidates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    split_dir, out_dir = args.split_dir.resolve(), args.out_dir.resolve()
    train_path, test_path = split_dir / "train.pkl", split_dir / "test.pkl"
    source_manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    train, test = load(train_path), load(test_path)
    train_ids = np.asarray(source_manifest["train_indices"], dtype=int)
    test_ids = np.asarray(source_manifest["test_indices"], dtype=int)
    if len(train) != len(train_ids) or len(test) != len(test_ids):
        raise RuntimeError("source split trajectory counts disagree with its frozen index manifest")
    count = min(int(args.attack_candidates), len(train), len(test) // 2)
    if count <= 0:
        raise RuntimeError("attack-candidates must leave a nonempty reference split")

    rng = np.random.default_rng(args.seed)
    member_rows = rng.choice(len(train), size=count, replace=False)
    shuffled_test = rng.permutation(len(test))
    nonmember_rows, reference_rows = shuffled_test[:count], shuffled_test[count:]
    if set(train_ids[member_rows]).intersection(test_ids[nonmember_rows]).intersection(test_ids[reference_rows]):
        raise RuntimeError("unexpected source-ID overlap")

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "member_training": ("member_training.pkl", train, train_ids),
        "member_candidates": ("member_candidates.pkl", [train[i] for i in member_rows], train_ids[member_rows]),
        "nonmember_candidates": ("nonmember_candidates.pkl", [test[i] for i in nonmember_rows], test_ids[nonmember_rows]),
        "reference": ("reference.pkl", [test[i] for i in reference_rows], test_ids[reference_rows]),
    }
    manifest = {
        "protocol": "beijing_kdd_disjoint_privacy_attack_split_v1",
        "seed": int(args.seed),
        "source_split_manifest": str((split_dir / "split_manifest.json").resolve()),
        "source_split_sha256": sha256(split_dir / "split_manifest.json"),
        "source_train_sha256": sha256(train_path),
        "source_test_sha256": sha256(test_path),
        "member_training_count": len(train),
        "attack_candidate_count_per_class": count,
        "reference_count": len(reference_rows),
        "source_indices": {},
        "artifacts": {},
    }
    for role, (name, trajectories, source_ids) in artifacts.items():
        path = out_dir / name
        save(path, trajectories)
        manifest["source_indices"][role] = [int(value) for value in source_ids]
        manifest["artifacts"][role] = {"path": str(path), "count": len(trajectories), "sha256": sha256(path)}
        print(f"[attack-split] wrote {role}: {len(trajectories):,}", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[attack-split] manifest: {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
