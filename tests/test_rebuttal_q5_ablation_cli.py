import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ablation_requires_source_transcript_before_creating_output(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "evaluation" / "run_rebuttal_q5_portal_ablation.py"),
        "--source", str(tmp_path / "missing_release"),
        "--osm-cache", str(tmp_path / "osm.pkl"),
        "--network", str(tmp_path / "network.shp"),
        "--real-match-cache", str(tmp_path / "real.pkl.gz"),
        "--out-dir", str(tmp_path / "output"),
    ], text=True, capture_output=True)
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert not (tmp_path / "output").exists()
