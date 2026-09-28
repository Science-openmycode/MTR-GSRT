from __future__ import annotations
import importlib.util
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("subset", ROOT / "evaluation/subset_matched_routes.py")
subset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subset)


class SplitIndicesTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual([0, 2], subset.validate_indices(
            {"train_indices": [0, 2], "test_indices": [1], "total_count": 3}, 3))

    def test_invalid(self):
        for manifest in ({"train_indices": [-1]}, {"train_indices": [0, 0]},
                         {"train_indices": [0], "test_indices": [0]},
                         {"train_indices": [0], "total_count": 4},
                         {"train_indices": [0], "train_count": 2}):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                subset.validate_indices(manifest, 3)
