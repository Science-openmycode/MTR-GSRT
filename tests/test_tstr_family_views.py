"""TSTR family views must preserve source slots and provenance."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))


def import_source(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "evaluation" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


views = import_source("materialize_family_tstr_views")
tstr = import_source("run_tstr_experiment")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_valid_route_and_invalid_fallback_preserve_source_count():
    nodes = {1: (40.0, 116.0), 2: (40.1, 116.1), 3: (40.2, 116.2)}
    route = ((1, 2), (2, 3))
    assert views.route_points(route, nodes).shape == (3, 2)
    assert views.sample_polyline(views.route_points(route, nodes), 7).shape == (7, 2)
    assert views.route_points(((1, 2), (3, 1)), nodes).size == 0


def test_family_manifest_binds_base_release_and_route_file(tmp_path):
    source, route, match = tmp_path / "train.pkl", tmp_path / "family.pkl", tmp_path / "STMatch.pkl.gz"
    output = tmp_path / "generic.pkl"
    for path, value in ((source, [np.zeros((2, 2))]), (route, [[(1, 2)]]),
                        (match, [{"cpath": [1]}]), (output, [np.ones((2, 2))])):
        path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    match.with_name("STMatch.manifest.json").write_text(json.dumps({
        "input_sha256": file_sha(source), "output_sha256": file_sha(match),
    }), encoding="utf-8")
    output.with_name(output.name + ".manifest.json").write_text(json.dumps({
        "schema": "tstr-family-coordinate-view-v1",
        "coordinates": {"sha256": file_sha(output)},
        "route_source": {"path": str(route), "sha256": file_sha(route)},
        "stmatch_source": {"path": str(match), "sha256": file_sha(match)},
        "source": {"path": str(source), "sha256": file_sha(source)},
    }), encoding="utf-8")
    assert tstr.base_release_hash(output) == file_sha(source)
    route.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Family route source"):
        tstr.base_release_hash(output)


def test_stmatch_wrapper_case_does_not_erase_source_binding(tmp_path):
    source, output = tmp_path / "source.pkl", tmp_path / "road.pkl"
    source.write_bytes(b"synthetic training release")
    output.write_bytes(b"public reconstructed view")
    output.with_name(output.name + ".manifest.json").write_text(json.dumps({
        "router": "STMATCH", "output": {"sha256": file_sha(output)},
        "inputs": {"source": {"path": str(source), "sha256": file_sha(source)}},
    }), encoding="utf-8")
    assert tstr.base_release_hash(output) == file_sha(source)
