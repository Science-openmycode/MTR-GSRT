"""Reproduce the cross-city destination-conditioned route-decision task.

This public entry point evaluates a model trained only on each synthetic
corpus.  Given the current public region and trip destination, the model
predicts the next public region on frozen real test trips.  It reports
trip-balanced accuracy and negative log likelihood, plus paired trip-bootstrap
comparisons between MTR-GSRT and every statistical baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from check.route_metric_replacement_20260724.screen_replacement import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="results/route_decision_metrics/current",
        help="Output directory, relative to public_release unless absolute.",
    )
    args = parser.parse_args()
    output = Path(args.out_dir)
    if not output.is_absolute():
        output = ROOT / output
    payload = run(output.resolve())
    print(json.dumps({
        "status": payload["status"],
        "output_dir": str(output.resolve()),
        "rows": len(payload["summary"]),
        "comparisons": len(payload["comparisons"]),
    }, indent=2))


if __name__ == "__main__":
    main()
