from pathlib import Path
import importlib.util
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

class CityInputTest(unittest.TestCase):
    def test_porto_coordinate_order_and_hash_rejection(self):
        with tempfile.TemporaryDirectory(prefix="city 路径 ") as directory:
            folder = Path(directory)
            source = folder / "train.csv"
            import csv
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["POLYLINE"])
                writer.writeheader()
                writer.writerow({"POLYLINE": json.dumps([[-8.6,41.15],[-8.61,41.16]])})
            out = folder / "porto.pkl"
            command = [sys.executable, str(ROOT/"commands/prepare_porto.py"),
                       "--source-csv", str(source), "--out", str(out), "--count", "1"]
            bad = subprocess.run(command+["--expected-sha256","0"*64], capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertFalse(out.exists())
            result = subprocess.run(command, cwd=folder, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            with out.open("rb") as handle:
                np.testing.assert_allclose(pickle.load(handle)[0], [[41.15,-8.6],[41.16,-8.61]])
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_sf_occupied_trip_and_empty_count(self):
        with tempfile.TemporaryDirectory(prefix="sf 路径 ") as directory:
            folder = Path(directory)
            source = folder / "raw"
            source.mkdir()
            (source / "new_fixture.txt").write_text(
                "37.7 -122.4 0 100\n37.701 -122.4 1 200\n"
                "37.702 -122.4 1 260\n37.703 -122.4 1 320\n37.704 -122.4 0 400\n",
                encoding="utf-8")
            command = [sys.executable, str(ROOT/"commands/prepare_sf_trips.py"),
                       "--source-dir", str(source), "--out", str(folder/"sf.pkl"),
                       "--count","1","--min-points","2","--min-length-km","0.01"]
            result = subprocess.run(command, cwd=folder, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            with (folder/"sf.pkl").open("rb") as handle:
                rows = pickle.load(handle)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]), 3)
            np.testing.assert_allclose(rows[0][0], [37.701,-122.4])

if __name__ == "__main__":
    unittest.main()
