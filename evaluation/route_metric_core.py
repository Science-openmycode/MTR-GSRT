from __future__ import annotations

import gzip
import pickle
from collections import Counter, defaultdict
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


def _support_scores(real: Counter, synthetic: Counter) -> tuple[float, float, float]:
    real_support, synthetic_support = set(real), set(synthetic)
    common = len(real_support & synthetic_support)
    recall = common / len(real_support) if real_support else 1.0
    precision = common / len(synthetic_support) if synthetic_support else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return recall, precision, f1


def _branch_scores(real_routes, synthetic_routes) -> tuple[float, float]:
    """Conditional branch overlap and dominant-exit accuracy on real branch mass."""
    counters = []
    for routes in (real_routes, synthetic_routes):
        choices = defaultdict(Counter)
        for route in routes:
            weight = 1.0 / max(len(route) - 1, 1)
            for left, right in zip(route, route[1:]):
                choices[left][right] += weight
        counters.append(choices)
    real, synthetic = counters
    branches = [entry for entry in real if len(real[entry]) > 1]
    evidence = sum(sum(real[entry].values()) for entry in branches)
    overlap = dominant = 0.0
    for entry in branches:
        mass = sum(real[entry].values())
        synthetic_mass = sum(synthetic[entry].values())
        if synthetic_mass:
            real_probability = {key: value / mass for key, value in real[entry].items()}
            synthetic_probability = {key: value / synthetic_mass
                                     for key, value in synthetic[entry].items()}
            overlap += mass * sum(min(real_probability.get(key, 0.0),
                                      synthetic_probability.get(key, 0.0))
                                  for key in real_probability.keys() | synthetic_probability.keys())
            dominant += mass * float(max(real_probability, key=real_probability.get)
                                     == max(synthetic_probability, key=synthetic_probability.get))
    return overlap / max(evidence, 1e-12), dominant / max(evidence, 1e-12)


def _od_prefix_workload(routes, od_labels, route_labels):
    groups = defaultdict(Counter)
    od_counts = Counter()
    for route in routes:
        if not route:
            continue
        od = (int(od_labels.get(route[0][0], -1)),
              int(od_labels.get(route[-1][1], -1)))
        path = _compress([int(route_labels.get(route[0][0], -1))] +
                         [int(route_labels.get(edge[1], -1)) for edge in route])
        if len(path) < 2:
            continue
        weight = 1.0 / (len(path) - 1)
        for length in range(2, len(path) + 1):
            groups[od][path[:length]] += weight
        od_counts[od] += 1
    return groups, od_counts


def _od_prefix_fidelity(real_routes, synthetic_routes, cache, resolution, route_yield):
    od_labels = {int(k): int(v) for k, v in cache["labels24"].items()}
    route_labels = {int(k): int(v) for k, v in cache[f"labels{resolution}"].items()}
    real_groups, real_counts = _od_prefix_workload(real_routes, od_labels, route_labels)
    synthetic_groups, _ = _od_prefix_workload(synthetic_routes, od_labels, route_labels)
    total = sum(real_counts.values())
    if not total:
        return 0.0
    fidelity = sum((count / total) * _cpc(real_groups[od], synthetic_groups.get(od, Counter()))
                   for od, count in real_counts.items())
    return fidelity * route_yield


def _demand_fidelity(real_routes, synthetic_routes, cache, synthetic_slots):
    labels = {int(k): int(v) for k, v in cache["labels24"].items()}
    def counts(routes):
        return Counter((labels.get(route[0][0], -1), labels.get(route[-1][1], -1))
                       for route in routes if route)
    real, synthetic = counts(real_routes), counts(synthetic_routes)
    real_n = max(len(real_routes), 1)
    synthetic_n = max(synthetic_slots, 1)
    return float(sum(min(real[key] / real_n, synthetic[key] / synthetic_n)
                     for key in real.keys() | synthetic.keys()))


def valid_routes(routes) -> list[tuple[tuple[int, int], ...]]:
    """Condition the real reference on observed connected road evidence."""
    return [tuple(route) for route in routes
            if route and all(left[1] == right[0] for left, right in zip(route, route[1:]))]


def route_counters(routes, cache) -> dict:
    outdegree = {int(k): int(v) for k, v in cache["outdegree"].items()}
    labels = {int(node): int(region) for node, region in cache["labels96"].items()}
    result = {name: Counter() for name in ("edge", "turn", "decision", "decision_unit", "family")}
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
        if decisions:
            # Full-road BTF gives each public slot at most unit turn mass.
            # Occurrence-weighted CPC is retained as a separate metric.
            mass = 1.0 / (max(len(routes), 1) * len(decisions))
            for decision in decisions:
                result["decision_unit"][decision] += mass
        decision_slots += bool(decisions)
        # labels96 maps road *nodes* to regions. A route is a sequence of
        # directed (source, target) edges, so include both endpoints.
        regions = [labels.get(int(route[0][0]), -1)]
        regions.extend(labels.get(int(target), -1) for _, target in route)
        result["family"][_compress(regions)] += 1
    result["valid"] = valid
    result["decision_slots"] = decision_slots
    return result


def evaluate_routes(routes, real_routes, cache, reference=None) -> dict[str, float]:
    reference = reference or route_counters(real_routes, cache)
    synthetic = route_counters(routes, cache)
    slots = max(len(routes), 1)
    real_decision_yield = reference["decision_slots"] / max(len(real_routes), 1)
    # The frozen RC-CPC normalizes each contributing route to unit decision
    # mass before comparing the conditional distributions. Raw occurrence CPC
    # weights long routes more heavily and is a different estimand.
    decision_overlap = _cpc(reference["decision_unit"], synthetic["decision_unit"])
    bounded_overlap = sum(min(reference["decision_unit"][key], synthetic["decision_unit"][key])
                          for key in reference["decision_unit"].keys() | synthetic["decision_unit"].keys())
    # The published BTF contract divides subprobability overlap by public
    # real-evidence yield; missing synthetic slots cannot be renormalized away.
    btf = min(1.0, bounded_overlap / real_decision_yield) if real_decision_yield else 0.0
    edge_cpc = _cpc(reference["edge"], synthetic["edge"])
    turn_cpc = _cpc(reference["turn"], synthetic["turn"])
    edge_recall, edge_precision, edge_f1 = _support_scores(reference["edge"], synthetic["edge"])
    turn_recall, turn_precision, turn_f1 = _support_scores(reference["turn"], synthetic["turn"])
    branch_cpc, exit_acc = _branch_scores(valid_routes(real_routes), valid_routes(routes))
    valid_real, valid_synthetic = valid_routes(real_routes), valid_routes(routes)
    road_yield = synthetic["valid"] / slots
    return {
        "RoadYield": road_yield,
        "DemandFid": _demand_fidelity(valid_real, valid_synthetic, cache, slots),
        "BTF": btf,
        "RC-CPC": decision_overlap,
        "EdgeCPC": edge_cpc,
        "TurnCPC": turn_cpc,
        "FamilyCPC": _cpc(reference["family"], synthetic["family"]) * synthetic["valid"] / slots,
        "EdgeRecall": edge_recall,
        "EdgePrecision": edge_precision,
        "EdgeF1": edge_f1,
        "TurnRecall": turn_recall,
        "TurnPrecision": turn_precision,
        "TurnF1": turn_f1,
        "EdgeIoU": edge_cpc / (2.0 - edge_cpc),
        "TurnIoU": turn_cpc / (2.0 - turn_cpc),
        "BranchCPC": branch_cpc,
        "ExitAcc": exit_acc,
        "ODPF96": _od_prefix_fidelity(valid_real, valid_synthetic, cache, 96, road_yield),
        "ODPF384": _od_prefix_fidelity(valid_real, valid_synthetic, cache, 384, road_yield),
    }
