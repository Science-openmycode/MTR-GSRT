import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "evaluation" / "run_rebuttal_q5_partition.py"


def test_partition_cli_rejects_unsupported_configuration(tmp_path):
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--real", str(tmp_path / "real.pkl"),
        "--osm-cache", str(tmp_path / "osm.pkl"), "--bbox", "0", "1", "0", "1",
        "--public-capacity", "2", "--spec", "flat:17",
        "--out-dir", str(tmp_path / "out"),
    ], text=True, capture_output=True)
    assert result.returncode != 0
    assert "Unsupported or duplicate partition specification" in result.stderr
    assert not (tmp_path / "out").exists()


def test_partition_cli_rejects_wrong_input_hash(tmp_path):
    real, osm = tmp_path / "real.pkl", tmp_path / "osm.pkl"
    real.write_bytes(b"real")
    osm.write_bytes(b"osm")
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--real", str(real), "--osm-cache", str(osm),
        "--bbox", "0", "1", "0", "1", "--public-capacity", "2",
        "--input-sha256", "0" * 64, "--out-dir", str(tmp_path / "out"),
    ], text=True, capture_output=True)
    assert result.returncode != 0
    assert "Input SHA-256 mismatch" in result.stderr
    assert not (tmp_path / "out").exists()
