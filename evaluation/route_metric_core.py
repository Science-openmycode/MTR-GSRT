from __future__ import annotations

import gzip
import pickle
from collections import Counter
from pathlib import Path


def load_pickle(path: Path):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def normalize_routes(raw, cache) -> list[tuple[tuple[int, int], ...]]:
    """Convert every supported saved route representation to directed edges."""
    edge_nodes = {int(k): tuple(v) for k, v in cache.get("edge_nodes", {}).items()}
    routes: list[tuple[tuple[int, int], ...]] = []
    for row in raw:
        if isinstance(row, dict):
            if row.get("node_sequence"):
                nodes = [int(v) for v in row["node_sequence"]]
                route = tuple(zip(nodes, nodes[1:]))
            elif row.get("accepted") and row.get("cpath"):
                route = tuple(edge_nodes[int(e)] for e in row["cpath"] if int(e) in edge_nodes)
            else:
                route = ()
        else:
            seq = tuple(row or ())
            if seq and isinstance(seq[0], (int, float)):
                route = tuple(edge_nodes[int(e)] for e in seq if int(e) in edge_nodes)
            else:
                route = tuple(tuple(map(int, edge)) for edge in seq)
        routes.append(route)
    return routes


def _cpc(left: Counter, right: Counter) -> float:
    left_total, right_total = sum(left.values()), sum(right.values())
    if not left_total or not right_total:
        return 0.0
    keys = left.keys() | right.keys()
    return float(sum(min(left[k] / left_total, right[k] / right_total) for k in keys))


def _compress(values) -> tuple:
    out = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return tuple(out)


def route_counters(routes, cache) -> dict:
    outdegree = {int(k): int(v) for k, v in cache["outdegree"].items()}
    labels = cache["labels96"]
    result = {name: Counter() for name in ("edge", "turn", "decision", "family")}
    valid = 0
    decision_slots = 0
    for route in routes:
        route = tuple(route)
        if not route or not all(a[1] == b[0] for a, b in zip(route, route[1:])):
            continue
        valid += 1
        result["edge"].update(route)
        result["turn"].update(zip(route, route[1:]))
        decisions = [(a, b) for a, b in zip(route, route[1:]) if outdegree.get(a[1], 0) > 1]
        result["decision"].update(decisions)
        decision_slots += bool(decisions)
        result["family"][_compress(labels.get(edge, -1) for edge in route)] += 1
    result["valid"] = valid
    result["decision_slots"] = decision_slots
    return result


def evaluate_routes(routes, real_routes, cache, reference=None) -> dict[str, float]:
    reference = reference or route_counters(real_routes, cache)
    synthetic = route_counters(routes, cache)
    slots = max(len(routes), 1)
    real_decision_yield = reference["decision_slots"] / max(len(real_routes), 1)
    decision_overlap = _cpc(reference["decision"], synthetic["decision"])
    # The published BTF contract includes the public real-evidence yield.
    btf = min(1.0, decision_overlap / real_decision_yield) if real_decision_yield else 0.0
    return {
        "RoadYield": synthetic["valid"] / slots,
        "BTF": btf,
        "RC-CPC": decision_overlap,
        "EdgeCPC": _cpc(reference["edge"], synthetic["edge"]),
        "TurnCPC": _cpc(reference["turn"], synthetic["turn"]),
        "FamilyCPC": _cpc(reference["family"], synthetic["family"]),
    }
