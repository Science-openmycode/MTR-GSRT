import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_routing_equations_match_decoder(tmp_path):
    output = tmp_path / "q3.json"
    subprocess.run([sys.executable, str(ROOT / "evaluation" / "audit_rebuttal_q3_routing.py"),
                    "--out", str(output)], cwd=tmp_path, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["classification"] == "PUBLIC_TOY_ROUTING_EXAMPLE_NO_PRIVATE_DATA"
    assert result["chosen_beta"] == 0.5
    assert len(result["two_path_routing"]) == 3
    assert result["two_path_routing"][0]["expected_cost"] > result["two_path_routing"][-1]["expected_cost"]
    assert result["portal_projected_probability"][0] > 0.8
