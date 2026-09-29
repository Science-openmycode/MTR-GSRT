"""The bundled Q5 research ablation preserves its public budget and interface."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "generation/mtr_gsrt/generation/mtr/generate.py"
SPEC = importlib.util.spec_from_file_location("mtr_portal_unit_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_portal_unit_is_explicit_research_mode():
    parser = module.build_parser()
    required = ["--data", "fixture.pkl", "--epsilon-total", "7/5",
                "--noise-seed", "1", "--decoder-seed", "2", "--out-dir", "out"]
    assert parser.parse_args(required + ["--portal-unit-mode", "ablated"]).portal_unit_mode == "ablated"
    assert parser.parse_args(required).portal_unit_mode == "full"
    assert sum(module.PORTAL_UNIT_ABLATED_MASSES.values()) == 1_000_000
    assert set(module.PORTAL_UNIT_ABLATED_MASSES) == set(module.Q5_BLOCK_NAMES) - {
        "portal_fiber_flow"
    }


def test_research_decoder_is_separate_from_audited_default():
    research = ROOT / "generation/mtr_gsrt/src/mtr/DP_GSRT/final/portal_unit_research_decoder.py"
    audited = ROOT / "generation/mtr_gsrt/src/mtr/DP_GSRT/final/quotient_rsp_bridge_development.py"
    assert research.is_file() and audited.is_file()
    assert "--portal-unit-ablation" in research.read_text(encoding="utf-8")
    assert "--portal-unit-ablation" not in audited.read_text(encoding="utf-8")
    assert "portal_unit_mode == \"ablated\"" in SOURCE.read_text(encoding="utf-8")
