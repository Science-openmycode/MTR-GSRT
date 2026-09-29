import subprocess
import sys
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multicity_rejects_missing_seed_instead_of_partial_aggregate(tmp_path):
    for name in ("porto_generation", "porto_metrics", "sf_generation", "sf_metrics"):
        (tmp_path / name).mkdir()
    command = [
        sys.executable, str(ROOT / "evaluation" / "aggregate_rebuttal_q6_multicity.py"),
        "--porto-generation-root", str(tmp_path / "porto_generation"),
        "--porto-metrics-root", str(tmp_path / "porto_metrics"),
        "--sf-generation-root", str(tmp_path / "sf_generation"),
        "--sf-metrics-root", str(tmp_path / "sf_metrics"),
        "--out-dir", str(tmp_path / "out"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing multicity seed inputs" in result.stderr
    assert not (tmp_path / "out").exists()


def test_multicity_accepts_bound_local_costs_from_new_protocols(tmp_path):
    metric_keys = (
        "grid_density_jsd", "trip_error", "path_length_jsd", "OD_jsd",
        "road_segment_jsd", "B2_next_region_accuracy", "B2_next_region_nll",
        "B2_grid_route_mrr", "witness_valid",
    )
    command = [sys.executable, str(ROOT / "evaluation" / "aggregate_rebuttal_q6_multicity.py")]
    for city in ("porto", "sf"):
        generation = tmp_path / city / "generation"
        metrics = tmp_path / city / "metrics"
        timing = tmp_path / city / "timing"
        for seed in range(20260719, 20260724):
            release = generation / f"seed_{seed}"
            score = metrics / f"seed_{seed}"
            release.mkdir(parents=True)
            score.mkdir(parents=True)
            timing.mkdir(exist_ok=True)
            protocol = release / "protocol.json"
            protocol.write_text(json.dumps({
                "output_count": 2, "decoder": {"fallback_count": 0},
            }), encoding="utf-8")
            (score / "metrics.json").write_text(json.dumps({
                "metrics": {key: 0.5 for key in metric_keys},
            }), encoding="utf-8")
            digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
            (timing / f"seed_{seed}.json").write_text(json.dumps({
                "classification": "LOCAL_PERFORMANCE_DIAGNOSTIC_NOT_DP_RELEASE",
                "release_protocol_sha256": digest,
                "generation_elapsed_sec": 10.0,
                "peak_rss_bytes": 123456,
                "stages_sec": {
                    "input_and_public_graph_preparation": 1.0,
                    "private_measurement": 3.0,
                    "public_routing_and_serialization": 5.0,
                },
            }), encoding="utf-8")
        command += [
            f"--{city}-generation-root", str(generation),
            f"--{city}-metrics-root", str(metrics),
            f"--{city}-performance-root", str(timing),
        ]
    out = tmp_path / "out"
    result = subprocess.run(command + ["--out-dir", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    detailed = (out / "q6_multicity_five_seed_detailed.csv").read_text(encoding="utf-8")
    assert "peak_rss_bytes" in detailed
    assert "private_measurement_sec" in detailed
    assert "123456" in detailed
