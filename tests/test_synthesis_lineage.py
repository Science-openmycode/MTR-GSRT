from __future__ import annotations
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from synthesis_lineage import SCHEMA, bind_records, bind_split, validate_record

spec = importlib.util.spec_from_file_location("train_only", ROOT / "commands/generate_train_only.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
spec_tasks = importlib.util.spec_from_file_location("bound_tasks", ROOT / "evaluation/run_tstr_experiment.py")
tasks = importlib.util.module_from_spec(spec_tasks)
spec_tasks.loader.exec_module(tasks)
spec_split = importlib.util.spec_from_file_location("checked_split", ROOT / "evaluation/audit_tstr_split.py")
split = importlib.util.module_from_spec(spec_split)
spec_split.loader.exec_module(split)


class SynthesisLineageTests(unittest.TestCase):
    def test_raw_road_objects_rejected_before_training(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scores"
            argv = ["tasks", "--train-real", "missing_train.pkl", "--test-real", "missing_test.pkl",
                    "--synthetic", "coordinates.pkl", "--road-synthetic", "road_routes.pkl.gz",
                    "--names", "M", "--osm-cache", "missing_osm.pkl",
                    "--bbox", "0", "1", "0", "1", "--out-dir", str(output)]
            errors = io.StringIO()
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(errors):
                with self.assertRaises(SystemExit) as result:
                    tasks.main()
            self.assertEqual(result.exception.code, 2)
            self.assertIn("routes-to-coordinates", errors.getvalue())
            self.assertFalse(output.exists())

    def fixture(self, root):
        record = root / "record.json"
        data = {"schema": SCHEMA, "status": "passed", "train_sha256": "train",
                "test_sha256": "test", "outputs": {"trajectories.pkl": "release"},
                "source_sha256": "source", "returncode": 0, "generator": "main",
                "input_policy": "train_only_explicit_generator_input"}
        record.write_text(json.dumps(data), encoding="utf-8")
        return record, data

    def test_success_and_wrong_split_or_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, _ = self.fixture(Path(tmp))
            self.assertEqual("release", validate_record(record, "train", "test", "release")["release_sha256"])
            for train, test, release in [("test", "train", "release"), ("train", "test", "other")]:
                with self.assertRaises(ValueError):
                    validate_record(record, train, test, release)

    def test_failed_run_not_certified(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, data = self.fixture(Path(tmp))
            data["status"] = "failed"
            record.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "successfully executed"):
                validate_record(record, "train", "test", "release")

    def test_all_methods_required_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record, _ = self.fixture(root)
            binding = {"train_sha256": "train", "test_sha256": "test",
                       "base_releases": {"M": "release", "N": "release"}}
            with self.assertRaisesRegex(ValueError, "every measurement"):
                bind_records([f"M={record}"], root, binding, True)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                bind_records([f"M={record}", f"M={record}"], root, binding)

    def test_preflight_does_not_allow_overrides_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = root / "train.pkl", root / "test.pkl"
            train.write_bytes(b"train"); test.write_bytes(b"test")
            record, out = root / "record.json", root / "out"
            module.preflight(train, test, record, out, ["--seed", "42"])
            for override in ("--data=test", "--out-dir", "--real-matched", "--extra-arg"):
                with self.assertRaisesRegex(ValueError, "override"):
                    module.preflight(train, test, record, out, [override])
            with self.assertRaisesRegex(ValueError, "outside"):
                module.preflight(train, test, out / "record.json", out, [])
            record.write_text("existing")
            with self.assertRaisesRegex(ValueError, "overwritten"):
                module.preflight(train, test, record, out, [])

    def test_split_binding_rejects_wrong_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps({"schema": "verified_disjoint_split_v1", "status": "passed",
                "all_source_indices_used_once": True, "train_sha256": "train", "test_sha256": "test",
                "full_sha256": "full", "train_count": 80, "test_count": 20}))
            self.assertEqual(80, bind_split(path, "train", "test")["train_count"])
            with self.assertRaisesRegex(ValueError, "mismatched"):
                bind_split(path, "other", "test")

    def test_dfr_coordinate_and_routed_view_trace_to_generated_roads(self):
        from synthesis_lineage import sha
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roads, coords, view = root / "roads.pkl.gz", root / "coords.pkl", root / "view.pkl"
            for path in (roads, coords, view):
                path.write_bytes(path.name.encode())
            coords.with_name(coords.name + ".manifest.json").write_text(json.dumps({
                "schema": "road-route-coordinate-derivation-v1", "coordinates": {"sha256": sha(coords)},
                "route_source": {"sha256": sha(roads), "path": str(roads)}}))
            view.with_name(view.name + ".manifest.json").write_text(json.dumps({
                "router": "FMM", "output": {"sha256": sha(view)},
                "inputs": {"source": {"sha256": sha(coords), "path": str(coords)}}}))
            self.assertEqual(sha(roads), tasks.base_release_hash(coords))
            self.assertEqual(sha(roads), tasks.base_release_hash(view))
            roads.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source file hash mismatch"):
                tasks.base_release_hash(view)

    def test_split_indices_overlap_or_wrong_total_rejected(self):
        good = {"train_indices": [0, 1], "test_indices": [2], "train_count": 2, "test_count": 1, "total_count": 3}
        self.assertEqual(([0, 1], [2]), split.check_indices(good, 3))
        for bad in ({**good, "test_indices": [1]}, {**good, "total_count": 4}):
            with self.assertRaises(ValueError):
                split.check_indices(bad, 3)


if __name__ == "__main__":
    unittest.main()
