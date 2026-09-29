import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_partition_plot_needs_real_summary(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "plotting" / "plot_rebuttal_q5_q6.py"),
        "--stage", "q5-partition", "--input-dir", str(tmp_path),
        "--out-dir", str(tmp_path / "figures"),
    ], text=True, capture_output=True)
    assert result.returncode != 0
    assert "q5_partition_sensitivity_summary.csv" in result.stderr
    assert not (tmp_path / "figures").exists()
