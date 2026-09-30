"""The efficient epsilon sweep follows the current public release interface."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "generation/mtr_gsrt/generation/mtr/generate_sweep.py"
SPEC = importlib.util.spec_from_file_location("shared_query_sweep_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_registered_sweep_configuration_uses_public_input_bound(tmp_path):
    private = tmp_path / "frozen.pkl"
    private.write_bytes(b"fixture")
    config = module.resolve_sweep_config("geolife", str(private))
    assert config["input_capacity"] == 17123
    assert config["capacity"] == 17123
    assert config["name"] == "geolife"
    assert Path(config["data"]) == private


def test_sweep_does_not_publish_actual_input_count_or_timing():
    source = SOURCE.read_text(encoding="utf-8")
    assert '"input_record_count": "not_released"' in source
    assert '"public_input_capacity": config["input_capacity"]' in source
    assert '"elapsed_sec":' not in source
    assert "_load_private_input(config[\"data\"], config[\"input_capacity\"]" in source
