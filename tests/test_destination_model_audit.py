import importlib.util
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("destination_tasks", ROOT / "evaluation/analysis_scripts/downstream_full_protocol.py")
tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks)


class DestinationAuditTests(unittest.TestCase):
    def test_optional_audit_preserves_scores_and_hashes(self):
        trajectories = []
        for i in range(30):
            start = [39.80 + .003 * i, 116.15 + .004 * i]
            end = [39.85, 116.25] if i % 2 else [40.05, 116.50]
            trajectories.append(np.linspace(start, end, 6))
        direct = tasks.destination_tstr_metrics(trajectories, trajectories)
        first, second = {}, {}
        self.assertEqual(direct, tasks.destination_tstr_metrics(trajectories, trajectories, audit=first))
        tasks.destination_tstr_metrics(trajectories, trajectories, audit=second)
        self.assertEqual(first["models"], second["models"])
        self.assertEqual({"8", "16"}, set(first["models"]))
        self.assertEqual(64, len(first["models"]["8"]["hashes"]["probabilities"]))


if __name__ == "__main__":
    unittest.main()
