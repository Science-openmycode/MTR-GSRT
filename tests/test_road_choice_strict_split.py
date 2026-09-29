"""Train-only road-choice scoring splits source slots before map-match filtering."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation" / "evaluation" / "evaluate_road_choice.py"
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("road_choice_strict_split", SOURCE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_explicit_boundary_precedes_match_filter(monkeypatch):
    records = [
        {"source_index": index, "accepted": index in {0, 2, 4, 5}}
        for index in range(6)
    ]

    def accepted_only(subset, _context, _nodes):
        return [row["source_index"] for row in subset if row["accepted"]]

    monkeypatch.setattr(module, "matched_choice_trips", accepted_only)
    strict, boundary = module._select_real_test_trips(records, {}, {}, 0.2, 3)
    assert boundary == 3
    assert strict == [4, 5]

    legacy, accepted_boundary = module._select_real_test_trips(records, {}, {}, 0.2, None)
    assert accepted_boundary == 3
    assert legacy == [5]
