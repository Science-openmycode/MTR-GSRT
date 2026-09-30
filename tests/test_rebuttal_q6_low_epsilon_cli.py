import subprocess
import sys
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "evaluation" / "run_rebuttal_q6_low_epsilon.py"
SPEC = importlib.util.spec_from_file_location("q6_low_epsilon_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def command(tmp_path):
    return [
        sys.executable, str(SCRIPT),
        "--real", str(tmp_path / "real.pkl"),
        "--osm-cache", str(tmp_path / "osm.pkl"),
        "--bbox", "0", "1", "0", "1",
        "--public-capacity", "2",
        "--generation-root", str(tmp_path / "generation"),
        "--metrics-root", str(tmp_path / "metrics"),
        "--route-choice-csv", str(tmp_path / "choice.csv"),
        "--out-dir", str(tmp_path / "out"),
    ]


def test_q6_rejects_wrong_private_input_hash_before_output(tmp_path):
    (tmp_path / "real.pkl").write_bytes(b"real")
    (tmp_path / "osm.pkl").write_bytes(b"osm")
    result = subprocess.run(command(tmp_path) + ["--input-sha256", "0" * 64],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr
    assert not (tmp_path / "out").exists()


def test_q6_rejects_incomplete_budget_matrix_before_output(tmp_path):
    (tmp_path / "real.pkl").write_bytes(b"real")
    (tmp_path / "osm.pkl").write_bytes(b"osm")
    (tmp_path / "generation").mkdir()
    (tmp_path / "metrics").mkdir()
    (tmp_path / "choice.csv").write_text("epsilon,seed,metric,value\n", encoding="utf-8")
    result = subprocess.run(command(tmp_path), capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing Q6 matrix inputs" in result.stderr
    assert not (tmp_path / "out").exists()


def test_current_release_timing_and_metric_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "EPSILON_DIRS", {"7/5": "eps_7_5"})
    monkeypatch.setattr(module, "SEEDS", (20260719,))
    leaf = Path("eps_7_5") / "seed_20260719"
    gen = tmp_path / "gen"
    met = tmp_path / "met"
    perf = tmp_path / "perf"
    (gen / leaf).mkdir(parents=True)
    (met / leaf / "metrics").mkdir(parents=True)
    (perf / "eps_7_5").mkdir(parents=True)
    protocol_path = gen / leaf / "protocol.json"
    protocol_path.write_text(json.dumps({"decoder": {"fallback_count": 1}}), encoding="utf-8")
    (perf / "eps_7_5" / "seed_20260719.json").write_text(json.dumps({
        "classification": "LOCAL_PERFORMANCE_DIAGNOSTIC_NOT_DP_RELEASE",
        "release_protocol_sha256": module.sha256_file(protocol_path),
        "generation_elapsed_sec": 9.5,
    }), encoding="utf-8")
    metrics = {
        "grid_density_jsd": 0.1, "trip_error": 0.2, "path_length_jsd": 0.3,
        "OD_jsd": 0.4, "road_segment_jsd": 0.5,
        "directed_road_validity": 1.0,
        "coordinate_projection_directed_road_validity": 0.33,
        "route_compatible_yield": 1.0,
        "coordinate_projection_route_compatible_yield": 0.04,
        "B2_grid_route_mrr": 0.6, "B1_dest8_top5": 0.7, "witness_valid": 1.0,
    }
    (met / leaf / "metrics" / "metrics.json").write_text(
        json.dumps({"metrics": metrics}), encoding="utf-8")
    choice = tmp_path / "choice.csv"
    choice.write_text(
        "epsilon,seed,metric,value\n"
        "7/5,20260719,road_choice_ndcg,0.8\n"
        "7/5,20260719,next_road_accuracy,0.9\n"
        "7/5,20260719,next_road_nll,1.1\n", encoding="utf-8")
    row = module.utility_rows(choice, met, gen, perf)[0]
    assert row["DirectedRoadValidity"] == 0.33
    assert row["RouteCompatibleYield"] == 0.04
    assert row["RouteMRR"] == 0.6
    assert row["GenerationSeconds"] == 9.5
    with pytest.raises(ValueError, match="generation-performance-root"):
        module.utility_rows(choice, met, gen)
