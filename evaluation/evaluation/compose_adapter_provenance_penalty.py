"""Apply native-release evidence penalties to a public-adapter diagnostic.

An adapter may insert a road path between sparse published coordinates.  Its
conditional route-choice score must therefore be multiplied by evidence that
was already present in the native release, rather than by the adapter's own
success rate.  Zero native evidence gives infinite provenance-penalized NLL.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


NAME_MAP = {
    "SPRT": "SPRT", "PrivTrace": "PrivTrace", "DPTraj_PM": "DPTraj-PM",
    "DPStd": "DPStd", "MTR_DP_GSRT": "MTR-GSRT",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    adapter = json.loads(Path(args.adapter).read_text(encoding="utf-8"))
    native = json.loads(Path(args.native).read_text(encoding="utf-8"))["results"]
    rows = []
    for adapter_name, native_name in NAME_MAP.items():
        values = adapter[adapter_name]["road_choice_penalized"]
        evidence = float(native[native_name]["native_directed_edge_evidence_yield"])
        rows.append({
            "method": adapter_name,
            "native_road_evidence_yield": evidence,
            "adapter_route_yield": float(values["route_evidence_yield"]),
            "adapter_rc_cpc_conditional": float(values["rc_cpc_conditional"]),
            "adapter_rc_ndcg_conditional": float(values["rc_ndcg_conditional"]),
            "adapter_next_road_accuracy_conditional": float(values["next_road_accuracy_conditional"]),
            "provenance_penalized_rc_cpc": evidence * float(values["rc_cpc_conditional"]),
            "provenance_penalized_rc_ndcg": evidence * float(values["rc_ndcg_conditional"]),
            "provenance_penalized_next_road_accuracy": evidence * float(values["next_road_accuracy_conditional"]),
            "provenance_penalized_next_road_nll": None if evidence == 0.0 else float(values["next_road_nll_conditional"]) - math.log(evidence),
        })
    payload = {
        "classification": "EXPLORATORY_PUBLIC_ADAPTER_WITH_NATIVE_PROVENANCE_PENALTY",
        "definition": {
            "utility": "conditional utility of fixed public Snap-and-Connect completion multiplied by the native directed-road evidence yield before completion",
            "nll": "conditional adapter NLL minus log(native evidence yield); null represents infinity",
            "interpretation": "generator plus public adapter, with inserted public roads not credited as native generator evidence",
        },
        "rows": rows,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
