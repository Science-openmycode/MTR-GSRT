import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "evaluation" / "run_rebuttal_q6_low_epsilon.py"


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
