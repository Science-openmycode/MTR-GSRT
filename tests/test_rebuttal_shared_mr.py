"""Both rebuttals share one measured M-by-R result and explicit route sources."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation/evaluate_rebuttal_mr.py"
SPEC = importlib.util.spec_from_file_location("rebuttal_mr_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_source_selection_supports_saved_and_fresh_matchers(tmp_path):
    native, direct, family = (tmp_path / name for name in ("native", "direct", "family"))
    fmm, stmatch = tmp_path / "fmm", tmp_path / "stmatch"
    assert module.source_path("SPRT", "Nearest", native, direct, family) == direct / "SPRT/Nearest.pkl.gz"
    assert module.source_path("SPRT", "Full FMM", native, direct, family) == native / "SPRT/FMM.pkl.gz"
    assert module.source_path("SPRT", "Full FMM", native, direct, family, fmm, stmatch) == fmm / "SPRT.pkl.gz"
    assert module.source_path("SPRT", "Full STMatch", native, direct, family, fmm, stmatch) == stmatch / "SPRT.pkl.gz"
    assert module.source_path("MTR-GSRT", "Original", native, direct, family) == native / "MTR-GSRT/Original.pkl"


def test_saved_shared_matrix_has_no_placeholder_cells():
    directory = ROOT / "experiment_results/rebuttal_shared_mr/metrics"
    with (directory / "mr_paper_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 40
    assert len({(row["M"], row["R"]) for row in rows}) == 40
    assert set(manifest["scope"]) == {"rebuttal1-Q2", "rebuttal2-Q4"}
    assert all(row["source"] and row["sha256"] for row in rows)
    assert "EXPLICIT_EMPTY_ROAD_EVIDENCE" not in (directory / "mr_paper_metrics.csv").read_text(encoding="utf-8-sig")
    assert float(next(row["RoadYield"] for row in rows if row["M"] == "PrivTrace" and row["R"] == "Nearest")) > 0
