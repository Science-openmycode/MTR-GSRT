import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_attack_assembler_rejects_missing_raw_attacks(tmp_path):
    split = tmp_path / "split.json"
    split.write_text("{}", encoding="utf-8")
    result = subprocess.run([
        sys.executable,
        str(ROOT / "evaluation" / "assemble_rebuttal_q6_threat_model.py"),
        "--attack-root", str(tmp_path / "attacks"),
        "--split-manifest", str(split),
        "--out-dir", str(tmp_path / "out"),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing attack inputs" in result.stderr
    assert not (tmp_path / "out").exists()
