import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AGGREGATOR = ROOT / "evaluation" / "aggregate_rebuttal_q6_multicity.py"
SPEC = importlib.util.spec_from_file_location("q6_local_performance", AGGREGATOR)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_new_protocol_requires_separate_bound_performance_log(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"output_count": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="separate"):
        module.generation_cost(protocol, {"output_count": 3}, None)
    performance = tmp_path / "performance.json"
    payload = {
        "classification": "LOCAL_PERFORMANCE_DIAGNOSTIC_NOT_DP_RELEASE",
        "release_protocol_sha256": module.sha256_file(protocol),
        "generation_elapsed_sec": 9.0,
        "peak_rss_bytes": 123456,
        "stages_sec": {
            "input_and_public_graph_preparation": 1.0,
            "private_measurement": 3.0,
            "public_routing_and_serialization": 4.0,
        },
    }
    performance.write_text(json.dumps(payload), encoding="utf-8")
    row = module.generation_cost(protocol, {"output_count": 3}, performance)
    assert row["generation_sec"] == 9.0
    assert row["private_measurement_sec"] == 3.0
    assert row["peak_rss_bytes"] == 123456
    payload["release_protocol_sha256"] = "0" * 64
    performance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.generation_cost(protocol, {}, performance)


def test_historical_protocol_cost_remains_compatible(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    assert module.generation_cost(protocol, {"elapsed_sec": 7.5}, None) == {
        "generation_sec": 7.5,
    }


def test_peak_rss_uses_process_resident_memory():
    source = ROOT / "generation" / "mtr_gsrt" / "generation" / "mtr" / "generate.py"
    spec = importlib.util.spec_from_file_location("mtr_local_performance", source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert helper._peak_rss_bytes() > 0
