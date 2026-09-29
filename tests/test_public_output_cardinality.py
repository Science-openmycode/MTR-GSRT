"""The output count is public and independent of occupied input records."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "generation/mtr_gsrt/generation/mtr/generate.py"
SPEC = importlib.util.spec_from_file_location("mtr_cardinality_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.parametrize("output_count", [3, 5, 7])
def test_public_input_capacity_independent_of_output_count(tmp_path, output_count):
    source = tmp_path / "private.pkl"
    source.write_bytes(b"test fixture")
    osm = ROOT / "generation/mtr_gsrt/data/osm/osm_cache_beijing.pkl"
    args = argparse.Namespace(
        data=str(source), dataset_config=None, bbox=[39.75, 40.15, 116.10, 116.65],
        osm_cache=str(osm), public_slot_count=output_count, public_input_capacity=5,
        verify_frozen_input=False,
    )
    config = module._resolve_configuration(args)
    assert config["input_capacity"] == 5
    assert config["capacity"] == output_count
    calls = []

    def load(path, *, limit):
        calls.append((path, limit))
        return [object()] * 5

    private, legacy_name = module._load_private_input(config["data"], config["input_capacity"], load)
    assert len(private) == 5
    assert legacy_name is False
    assert calls == [(str(source), None)]


def test_neighbor_occupancy_does_not_change_public_count():
    for occupied in (0, 4, 5):
        real, named = module._load_private_input(
            "private-file.pkl", 5, lambda path, *, limit: [object()] * occupied
        )
        assert len(real) == occupied
        assert named is False
    with pytest.raises(RuntimeError, match="predeclared public input capacity"):
        module._load_private_input(
            "private-file.pkl", 5, lambda path, *, limit: [object()] * 6
        )


def test_legacy_named_input_uses_public_input_bound_not_output_count():
    calls = []

    def load(path, *, limit):
        calls.append((path, limit))
        return [object()] * 5

    private, named = module._load_private_input("porto", 5, load)
    assert len(private) == 5 and named
    assert calls == [("porto", 5)]


def test_no_exact_input_cardinality_in_public_protocol_source():
    source = SOURCE.read_text(encoding="utf-8")
    assert '"input_record_count": "not_released"' in source
    assert '"public_input_capacity": config["input_capacity"]' in source
    assert '"public_slot_count": config["capacity"]' in source
    assert '"output_count": config["capacity"]' in source
    assert '"elapsed_sec"' not in source


def test_frozen_hash_preflight_is_not_default_mechanism(tmp_path, monkeypatch):
    private = tmp_path / "private.pkl"
    private.write_bytes(b"neighboring preprocessed input")
    osm = ROOT / "generation/mtr_gsrt/data/osm/osm_cache_beijing.pkl"
    monkeypatch.setattr(module, "dataset_config", lambda _: {
        "name": "fixture", "data": str(private), "data_sha256": "0" * 64,
        "bbox": [39.75, 40.15, 116.10, 116.65], "osm_cache": str(osm),
        "public_slot_count": 5,
    })
    args = argparse.Namespace(
        data="fixture", dataset_config=None, bbox=None, osm_cache=None,
        public_slot_count=3, public_input_capacity=5, verify_frozen_input=False,
    )
    assert module._resolve_configuration(args)["input_capacity"] == 5
    args.verify_frozen_input = True
    with pytest.raises(RuntimeError, match="local preflight"):
        module._resolve_configuration(args)
