from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class FrozenMetricConsistencyTest(unittest.TestCase):
    def test_framework_rows_match_published_contract(self) -> None:
        actual = rows(ROOT / "experiment_results/executed_reference/framework/results.csv")
        frozen = rows(ROOT / "experiment_results/frozen/shared_mtr_framework_lift.csv")
        lookup = {(row["Measurement"], row["StageKey"]): row for row in frozen}
        self.assertEqual(len(actual), 8)
        for row in actual:
            key = (row["Measurement"], "waypoint" if row["StageKey"] == "routed" else "native")
            for metric in ("RoadYield", "BTF", "FamilyCPC"):
                self.assertAlmostEqual(float(row[metric]), float(lookup[key][metric]), places=12)

    def test_mr_rows_match_published_contract(self) -> None:
        actual = rows(ROOT / "experiment_results/executed_reference/mr/results.csv")
        frozen = rows(ROOT / "experiment_results/frozen/mr_matrix.csv")
        lookup = {(row["M"], row["R"]): row for row in frozen}
        names = {"FMM": "Full FMM", "STMatch": "Full STMatch", "Original": "Original"}
        self.assertEqual(len(actual), 9)
        for row in actual:
            expected = lookup[(row["M"], names[row["R"]])]
            for metric in ("RoadYield", "BTF", "FamilyCPC"):
                self.assertAlmostEqual(float(row[metric]), float(expected[metric]), places=12)


if __name__ == "__main__":
    unittest.main()
