"""Post-process native road-choice results with explicit evidence penalties."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for method, values in payload["results"].items():
        coverage = float(values["native_directed_edge_evidence_yield"])
        raw_nll = float(values["next_road_nll"])
        rows.append({
            "method": method,
            "road_evidence_yield": coverage,
            "invalid_rate": 1.0 - coverage,
            "rc_cpc_native": float(values["coverage_adjusted_rc_cpc"]),
            "rc_ndcg_native": float(values["coverage_adjusted_rc_ndcg"]),
            "next_road_acc_native": float(values["coverage_adjusted_next_road_accuracy"]),
            "next_road_nll_native": None if coverage == 0.0 else raw_nll - math.log(coverage),
            "raw_conditional_next_road_nll": raw_nll,
        })
    output = {
        "schema_version": 1,
        "classification": "POSTPROCESS_OF_NATIVE_NO_FMM_ROAD_CHOICE_RESULTS",
        "definition": {
            "coverage_adjusted_utility": "conditional utility multiplied by native directed-road evidence yield",
            "next_road_nll_native": "conditional NextRoadNLL - log(native directed-road evidence yield); null denotes +infinity for zero evidence",
            "invalid_rate": "one minus native directed-road evidence yield",
        },
        "source": str(source),
        "rows": rows,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
