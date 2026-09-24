from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "commands" / "prepare_geolife_beijing.py"


class PrepareGeoLifeTest(unittest.TestCase):
    def test_rebuild_and_hash_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "Data"
            trajectory_dir = data / "000" / "Trajectory"
            trajectory_dir.mkdir(parents=True)
            for name, lat in (("a.plt", 39.8), ("b.plt", 39.9)):
                (trajectory_dir / name).write_text(
                    "header\n" * 6 + f"{lat},116.2,0,0\n{lat + 0.001},116.201,0,0\n",
                    encoding="utf-8",
                )
            out = base / "real.pkl"
            cmd = [sys.executable, str(SCRIPT), "--source-dir", str(data), "--out", str(out)]
            rejected = subprocess.run([*cmd, "--expected-sha256", "0" * 64], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(out.exists())
            accepted = subprocess.run(cmd, check=True, capture_output=True, text=True)
            record = json.loads(accepted.stdout)
            self.assertEqual(record["trajectory_count"], 2)
            self.assertEqual(record["train_count"], 2)
            self.assertEqual(record["sha256"], hashlib.sha256(out.read_bytes()).hexdigest())
            with out.open("rb") as source:
                self.assertEqual(len(pickle.load(source)), 2)


if __name__ == "__main__":
    unittest.main()
