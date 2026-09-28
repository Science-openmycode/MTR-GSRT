from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("combine", ROOT / "evaluation/combine_tstr_mr.py")
combine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(combine)


class ProtocolBindingTests(unittest.TestCase):
    def make_manifest(self, root, protocol="strict_train_only_tstr_executed_v1", train="a"):
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(json.dumps({
            "protocol": protocol,
            "split_binding": {"train_sha256": train, "test_sha256": "b",
                              "bbox": [1, 2, 3, 4], "seed": 42, "osm_sha256": "map",
                              "base_releases": {"SPRT": "release-hash"}}
        }), encoding="utf-8")
        return root / "results.csv"

    def test_different_split_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {"Native": self.make_manifest(root / "n"),
                     "FMM": self.make_manifest(root / "f", train="c")}
            with self.assertRaisesRegex(ValueError, "different train_sha256"):
                combine.validate_bindings(paths)

    def test_retrospective_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_manifest(Path(tmp), protocol="retrospective")
            with self.assertRaisesRegex(ValueError, "retrospective"):
                combine.validate_bindings({"Native": path})

    def test_missing_binding_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text('{"protocol":"strict_train_only_tstr_executed_v1"}')
            with self.assertRaisesRegex(ValueError, "missing train/test"):
                combine.validate_bindings({"Native": root / "results.csv"})

    def test_same_split_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {name: self.make_manifest(root / name) for name in ("Native", "FMM", "STMatch")}
            self.assertEqual(3, len(combine.validate_bindings(paths)))

    def test_new_scoring_protocol_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_manifest(Path(tmp), protocol="train_test_bound_task_scoring_v2")
            self.assertEqual(1, len(combine.validate_bindings({"Native": path})))

    def test_mixed_release_batches_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {name: self.make_manifest(root / name) for name in ("Native", "FMM")}
            manifest = root / "FMM/manifest.json"
            data = json.loads(manifest.read_text())
            data["split_binding"]["base_releases"]["SPRT"] = "another-release"
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "different base_releases"):
                combine.validate_bindings(paths)

    def test_duplicate_and_nan_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            row = {"Pipeline": "Real-train", **dict.fromkeys(combine.ALL_METRICS, 1)}
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows([row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                combine.read(path)
            row[combine.ALL_METRICS[0]] = float("nan")
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "Invalid retention"):
                combine.read(path)


if __name__ == "__main__":
    unittest.main()
