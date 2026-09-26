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
            "labels24": {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 3},
            "labels384": {0: 10, 1: 10, 2: 20, 3: 30, 4: 40, 5: 30},
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

    def test_support_and_equal_mass_iou_keep_their_original_definitions(self) -> None:
        same = evaluate_routes([self.upper], [self.upper], self.cache)
        other = evaluate_routes([self.lower], [self.upper], self.cache)
        for name in ("EdgeRecall", "EdgePrecision", "EdgeF1", "TurnRecall",
                     "TurnPrecision", "TurnF1", "EdgeIoU", "TurnIoU"):
            self.assertAlmostEqual(same[name], 1.0)
        for name in ("EdgeRecall", "EdgePrecision", "EdgeF1"):
            self.assertAlmostEqual(other[name], 1 / 3)
        for name in ("TurnRecall", "TurnPrecision", "TurnF1", "TurnIoU"):
            self.assertEqual(other[name], 0.0)
        self.assertAlmostEqual(other["EdgeIoU"], 1 / 5)

    def test_branch_scores_use_conditional_real_branch_mass(self) -> None:
        same = evaluate_routes([self.upper, self.lower], [self.upper, self.lower], self.cache)
        only_upper = evaluate_routes([self.upper], [self.upper, self.lower], self.cache)
        self.assertAlmostEqual(same["BranchCPC"], 1.0)
        self.assertAlmostEqual(same["ExitAcc"], 1.0)
        self.assertAlmostEqual(same["ODPF96"], 1.0)
        self.assertAlmostEqual(same["ODPF384"], 1.0)
        self.assertAlmostEqual(same["DemandFid"], 1.0)
        self.assertAlmostEqual(only_upper["BranchCPC"], 0.5)
        self.assertAlmostEqual(only_upper["ExitAcc"], 1.0)

    def test_rc_cpc_normalizes_decisions_within_each_route(self) -> None:
        loop = ((0, 1), (1, 2), (2, 1), (1, 4))
        value = evaluate_routes([loop, loop], [self.upper, loop], self.cache)
        self.assertAlmostEqual(value["RC-CPC"], 0.75)


if __name__ == "__main__":
    unittest.main()
