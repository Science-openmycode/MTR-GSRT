"""Create the fixed, disjoint train/test split for KDD task-utility runs.

The split is intentionally independent of every synthesizer.  Persisting the
index manifest prevents accidental evaluation on the private fitting corpus.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PUBLIC_RELEASE = Path(__file__).resolve().parents[1]
if str(PUBLIC_RELEASE) not in sys.path:
    sys.path.insert(0, str(PUBLIC_RELEASE))

from public_utils import PUBLIC_ROOT, load_trajectories, save_trajectories_pkl, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Create a deterministic disjoint train/test trajectory split.")
    p.add_argument("--data", default="geolife")
    p.add_argument("--train-fraction", type=float, default=0.80)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--input-order", action="store_true", help="Use an input already arranged as train followed by test")
    p.add_argument("--out-dir", default=str(PUBLIC_ROOT / "outputs" / "kdd_revised" / "split"))
    args = p.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must lie strictly between zero and one")

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("[split] loading the full corpus", flush=True)
    trajs = load_trajectories(args.data, limit=None)
    perm = (np.arange(len(trajs)) if args.input_order
            else np.random.default_rng(args.seed).permutation(len(trajs)))
    n_train = int(round(len(trajs) * args.train_fraction))
    if not 0 < n_train < len(trajs):
        raise ValueError("The selected fraction must leave at least one train and one test trajectory")
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    train = [trajs[int(i)] for i in train_idx]
    test = [trajs[int(i)] for i in test_idx]
    train_path = save_trajectories_pkl(out / "train.pkl", train)
    test_path = save_trajectories_pkl(out / "test.pkl", test)
    manifest = {
        "protocol": "kdd_revised_disjoint_tstr_split_v1",
        "data": args.data,
        "seed": int(args.seed),
        "split_mode": "input_order" if args.input_order else "seeded_permutation",
        "seed_applied_to_input": not args.input_order,
        "train_fraction": float(args.train_fraction),
        "total_count": len(trajs),
        "train_count": len(train),
        "test_count": len(test),
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_file": str(Path(train_path).resolve()),
        "test_file": str(Path(test_path).resolve()),
        "privacy_note": "Indices are a benchmark protocol artifact. A deployment must fix this protocol externally or account for any private selection rule.",
    }
    write_json(out / "split_manifest.json", manifest)
    print(f"[split] wrote {len(train):,} train and {len(test):,} held-out test trajectories -> {out}", flush=True)


if __name__ == "__main__":
    main()
