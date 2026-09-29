"""The 28-candidate router selection must be complete and deterministic."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
spec = importlib.util.spec_from_file_location(
    "rebuttal_tstr", ROOT / "evaluation/run_rebuttal_tstr_selection.py")
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_table(path: Path, omit: str | None = None):
    fields = ["Pipeline", *module.METRICS]
    rows = [{"Pipeline": "Real-train", **dict.fromkeys(module.METRICS, 1.0)}]
    for method in module.METHODS:
        for router in module.ROUTERS:
            label = f"{method} :: {router}"
            if label == omit:
                continue
            value = 0.9 if router == "Family residual" else 0.2
            rows.append({"Pipeline": label, **dict.fromkeys(module.METRICS, value)})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def test_selection_uses_all_28_candidates_and_four_metrics(tmp_path):
    path = tmp_path / "results.csv"
    make_table(path)
    selected, scored = module.select_routers(path)
    assert len(scored) == 28
    assert selected == {method: "Family residual" for method in module.METHODS}
    assert all(row["selection_score"] == 0.9 for row in scored if row["router"] == "Family residual")


def test_missing_candidate_fails_closed(tmp_path):
    path = tmp_path / "results.csv"
    make_table(path, omit="SPRT :: FMM")
    with pytest.raises(ValueError, match="complete 4x7"):
        module.select_routers(path)
