from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackagedEvaluationPathsTest(unittest.TestCase):
    def test_beijing_osm_cache_is_packaged(self) -> None:
        datasets = json.loads((ROOT / "evaluation/configs/datasets.json").read_text(encoding="utf-8"))
        relative = datasets["datasets"]["geolife"]["osm_cache"]
        self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
