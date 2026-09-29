"""Check saved train/test records against disjoint indices in their full source."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from synthesis_lineage import sha
spec = importlib.util.spec_from_file_location("split_public_utils", ROOT / "evaluation/public_utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def check_indices(manifest, total):
    train, test = manifest["train_indices"], manifest["test_indices"]
    if manifest.get("total_count") != total or not train or not test:
        raise ValueError("Split total differs from source or one partition is empty")
    if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < total for i in train + test):
        raise ValueError("Invalid split index")
    if len(set(train + test)) != total or len(train) + len(test) != total:
        raise ValueError("Split indices overlap, repeat, or omit source records")
    if len(train) != manifest["train_count"] or len(test) != manifest["test_count"]:
        raise ValueError("Split counts differ from indices")
    return train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full-real", "split-manifest", "train-real", "test-real", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("full_real", "split_manifest", "train_real", "test_real", "out"):
        path = getattr(args, name)
        setattr(args, name, (path if path.is_absolute() else ROOT / path).resolve())
    if args.out.exists():
        parser.error("Choose a new split audit output; existing evidence is not overwritten")
    full = utils.load_trajectories(str(args.full_real))
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8-sig"))
    train_indices, test_indices = check_indices(manifest, len(full))
    for path, indices in ((args.train_real, train_indices), (args.test_real, test_indices)):
        saved = utils.load_trajectories(str(path))
        if len(saved) != len(indices) or any(not np.array_equal(row, full[index])
                                           for row, index in zip(saved, indices)):
            raise ValueError(f"Saved split differs from indexed source records: {path}")
    record = {"schema": "verified_disjoint_split_v1", "status": "passed",
              "classification": "LOCAL_RESEARCH_SPLIT_AUDIT_NOT_A_DP_RELEASE",
              "full_sha256": sha(args.full_real), "train_sha256": sha(args.train_real),
              "test_sha256": sha(args.test_real), "manifest_sha256": sha(args.split_manifest),
              "train_count": len(train_indices), "test_count": len(test_indices),
              "all_source_indices_used_once": True,
              "note": "Record-index disjointness; duplicate-valued trajectories are not treated as the same record."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("SPLIT AUDIT PASSED", len(train_indices), len(test_indices), flush=True)


if __name__ == "__main__":
    main()
