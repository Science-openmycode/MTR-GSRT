from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from route_metric_core import evaluate_routes, route_counters  # noqa: E402


class RouteRegionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = {
            "outdegree": {0: 1, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1},
            "labels96": {0: 10, 1: 10, 2: 20, 3: 30, 4: 40, 5: 30},
        }
        self.upper = ((0, 1), (1, 2), (2, 3))
        self.lower = ((0, 1), (1, 4), (4, 5))

    def test_region_family_uses_node_labels(self) -> None:
        counts = route_counters([self.upper, self.lower], self.cache)
        self.assertEqual(counts["family"][(10, 20, 30)], 1)
        self.assertEqual(counts["family"][(10, 40, 30)], 1)
        self.assertNotIn((-1,), counts["family"])

    def test_family_cpc_distinguishes_two_valid_corridors(self) -> None:
        same = evaluate_routes([self.upper], [self.upper], self.cache)
        other = evaluate_routes([self.lower], [self.upper], self.cache)
        self.assertEqual(same["FamilyCPC"], 1.0)
        self.assertEqual(other["FamilyCPC"], 0.0)
        self.assertEqual(other["RoadYield"], 1.0)

    def test_btf_does_not_renormalize_empty_output_slots(self) -> None:
        scores = evaluate_routes([self.upper, ()], [self.upper, self.upper], self.cache)
        self.assertAlmostEqual(scores["BTF"], 0.5)
        self.assertEqual(scores["RC-CPC"], 1.0)
        self.assertEqual(scores["FamilyCPC"], 0.5)


if __name__ == "__main__":
    unittest.main()
