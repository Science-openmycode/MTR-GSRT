import subprocess
import sys
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
