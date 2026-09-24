from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "evaluation" / "pipeline" / "prepare_kdd_revised_split.py"


class SplitOrderTest(unittest.TestCase):
    def test_already_grouped_corpus_is_not_permuted_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "full.pkl"
            trajectories = [np.array([[float(i), 1.0], [float(i), 2.0]]) for i in range(5)]
            with source.open("wb") as handle:
                pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)
            out = base / "split"
            subprocess.run([sys.executable, str(SCRIPT), "--data", str(source),
                            "--train-fraction", "0.6", "--input-order", "--out-dir", str(out)],
                           check=True, capture_output=True, text=True)
            with (out / "train.pkl").open("rb") as handle:
                train = pickle.load(handle)
            with (out / "test.pkl").open("rb") as handle:
                test = pickle.load(handle)
            self.assertEqual([int(item[0, 0]) for item in train], [0, 1, 2])
            self.assertEqual([int(item[0, 0]) for item in test], [3, 4])
            manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["split_mode"], "input_order")
            self.assertFalse(manifest["seed_applied_to_input"])


if __name__ == "__main__":
    unittest.main()
