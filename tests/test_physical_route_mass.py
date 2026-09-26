from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from run_physical_experiment import evaluate  # noqa: E402


class PhysicalMassTest(unittest.TestCase):
    def test_identity_is_unit_recovery_and_support(self) -> None:
        upper = ((0, 1), (1, 2), (2, 3))
        loop = ((0, 1), (1, 2), (2, 1), (1, 4))
        routes = [upper, loop]
        cache = {"labels24": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}}
        lengths = {edge: 0.1 for route in routes for edge in route}
        values = evaluate(routes, routes, cache, lengths, "test", 20)
        for name in ("RoadYield", "DemandFid", "RoadRecovery", "RoadSupport",
                     "TurnRecovery", "TurnSupport", "BackboneRecovery", "BackboneSupport"):
            self.assertAlmostEqual(values[name], 1.0)


if __name__ == "__main__":
    unittest.main()
