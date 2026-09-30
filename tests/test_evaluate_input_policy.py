from __future__ import annotations

import argparse
import pickle
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "evaluation/evaluation/evaluate_all.py"), run_name="evaluate_all_test")
_resolve_config = MODULE["_resolve_config"]


def _args(real: str, allow: bool) -> argparse.Namespace:
    return argparse.Namespace(
        real=real,
        dataset_config="geolife",
        bbox=(39.75, 40.15, 116.10, 116.65),
        osm_cache="generation/mtr_gsrt/data/osm/osm_cache_beijing.pkl",
        public_slot_count=3,
        allow_unregistered_input=allow,
    )


def test_unregistered_real_input_requires_explicit_opt_in(tmp_path):
    path = tmp_path / "external.pkl"
    with path.open("wb") as handle:
        pickle.dump([], handle, pickle.HIGHEST_PROTOCOL)

    with pytest.raises(RuntimeError, match="frozen real trajectory input hash mismatch"):
        _resolve_config(_args(str(path), allow=False))

    config = _resolve_config(_args(str(path), allow=True))
    assert config["real_data_sha256"]
    assert config["real_data_sha256"] != "6a160ca557fbd7bab97af489b56c931e73532c498c390e31965c0d61e53361fd"
