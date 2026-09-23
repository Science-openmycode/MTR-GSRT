"""Population road-use and road-dependent replacement metrics.

All metrics consume the same map-matching records.  A failed record remains a
failure: it contributes only to the unmatched Road-Use bin and never supplies
an edge traversal or a route-choice transition.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def _accepted_path(record: Mapping | object) -> tuple[int, ...]:
    accepted = bool(record["accepted"] if isinstance(record, Mapping) else record.accepted)
    if not accepted:
        return tuple()
    raw = record["cpath"] if isinstance(record, Mapping) else record.cpath
    return tuple(int(edge) for edge in raw)


def edge_histogram(records: Sequence[Mapping | object]) -> Counter[int | str]:
    """Count directed-edge traversals and one unmatched token per failed trip."""
    counts: Counter[int | str] = Counter()
    for record in records:
        path = _accepted_path(record)
        if path:
            counts.update(path)
        else:
            counts["__UNMATCHED__"] += 1
    return counts


def _jsd_counts(first: Counter, second: Counter) -> float:
    support = sorted(set(first) | set(second), key=str)
    p = np.asarray([first[key] for key in support], dtype=float)
    q = np.asarray([second[key] for key in support], dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("Road-Use JSD requires nonempty distributions")
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def road_use_jsd(real_records: Sequence[Mapping | object], synthetic_records: Sequence[Mapping | object]) -> float:
    return _jsd_counts(edge_histogram(real_records), edge_histogram(synthetic_records))


def map_matching_success_rate(records: Sequence[Mapping | object]) -> float:
    if not records:
        return 0.0
    return float(np.mean([bool(_accepted_path(record)) for record in records]))


def link_flow_wape(
    real_records: Sequence[Mapping | object],
    synthetic_records: Sequence[Mapping | object],
    *,
    calibrate_total_volume: bool = False,
) -> float:
    real = edge_histogram(real_records)
    synthetic = edge_histogram(synthetic_records)
    real.pop("__UNMATCHED__", None)
    synthetic.pop("__UNMATCHED__", None)
    denominator = float(sum(real.values()))
    if denominator <= 0:
        raise ValueError("Link-Flow WAPE requires at least one real edge traversal")
    support = set(real) | set(synthetic)
    if calibrate_total_volume:
        synthetic_total = float(sum(synthetic.values()))
        if synthetic_total <= 0:
            raise ValueError("volume-calibrated Link-Flow WAPE requires positive synthetic flow")
        scale = denominator / synthetic_total
        return float(sum(abs(scale * synthetic[key] - real[key]) for key in support) / denominator)
    return float(sum(abs(synthetic[key] - real[key]) for key in support) / denominator)


def require_minimum_path_edges(
    records: Sequence[Mapping | object], minimum: int,
) -> list[Mapping | object]:
    """Apply a method-independent route-object qualification to cached matches."""
    if minimum < 1:
        raise ValueError("minimum path length must be positive")
    output: list[Mapping | object] = []
    for record in records:
        path = _accepted_path(record)
        if isinstance(record, Mapping):
            qualified = dict(record)
            qualified["accepted"] = bool(path) and len(path) >= minimum
            output.append(qualified)
        else:
            if len(path) < minimum:
                raise TypeError("minimum-path filtering requires serialized mapping records")
            output.append(record)
    return output


def region_sequences(
    records: Sequence[Mapping | object],
    edge_to_region: Mapping[int, int],
) -> list[tuple[int, ...] | None]:
    """Project accepted edge paths and collapse consecutive repeated regions."""
    output: list[tuple[int, ...] | None] = []
    for record in records:
        path = _accepted_path(record)
        if not path:
            output.append(None)
            continue
        regions: list[int] = []
        for edge in path:
            if edge not in edge_to_region:
                raise KeyError(f"matched edge {edge} is absent from the frozen public quotient mapping")
            region = int(edge_to_region[edge])
            if not regions or regions[-1] != region:
                regions.append(region)
        output.append(tuple(regions) if len(regions) >= 2 else None)
    return output


def road_next_trip_scores(
    synthetic_sequences: Iterable[tuple[int, ...] | None],
    real_test_sequences: Iterable[tuple[int, ...] | None],
    *,
    state_count: int,
    alpha: float = 0.5,
    feasible_successors: Mapping[int, Sequence[int]] | None = None,
) -> dict[str, np.ndarray | int]:
    if state_count <= 1 or alpha <= 0:
        raise ValueError("state_count must exceed one and alpha must be positive")
    conditional: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    backoff: dict[int, Counter[int]] = defaultdict(Counter)
    training_trips = 0
    for sequence in synthetic_sequences:
        if sequence is None or len(sequence) < 2:
            continue
        training_trips += 1
        destination = int(sequence[-1])
        for current, following in zip(sequence[:-1], sequence[1:]):
            conditional[(int(current), destination)][int(following)] += 1
            backoff[int(current)][int(following)] += 1

    trip_accuracy: list[float] = []
    trip_nll: list[float] = []
    evaluated_decisions = 0
    for sequence in real_test_sequences:
        if sequence is None or len(sequence) < 2:
            continue
        destination = int(sequence[-1])
        correct: list[float] = []
        losses: list[float] = []
        for current, following in zip(sequence[:-1], sequence[1:]):
            counter = conditional.get((int(current), destination))
            if not counter:
                counter = backoff.get(int(current), Counter())
            support_size = state_count
            if feasible_successors is not None:
                support = tuple(int(value) for value in feasible_successors.get(int(current), ()))
                if int(following) not in support:
                    raise ValueError(
                        f"observed road transition {current}->{following} is absent from public quotient support"
                    )
                support_size = len(support)
                if support_size <= 0:
                    raise ValueError(f"road region {current} has no feasible public successor")
            if not counter:
                correct.append(0.0)
                losses.append(math.log(support_size))
            else:
                if feasible_successors is not None and any(key not in support for key in counter):
                    raise ValueError(f"training transitions leave public support at road region {current}")
                prediction = min(counter, key=lambda item: (-counter[item], item))
                correct.append(float(prediction == int(following)))
                probability = (counter.get(int(following), 0) + alpha) / (
                    sum(counter.values()) + alpha * support_size
                )
                losses.append(-math.log(probability))
            evaluated_decisions += 1
        trip_accuracy.append(float(np.mean(correct)))
        trip_nll.append(float(np.mean(losses)))
    if not trip_accuracy:
        raise ValueError("fixed real road-test set contains no evaluable decisions")
    return {
        "road_next_accuracy": np.asarray(trip_accuracy, dtype=float),
        "road_next_nll": np.asarray(trip_nll, dtype=float),
        "road_next_training_trips": int(training_trips),
        "road_next_test_trips": int(len(trip_accuracy)),
        "road_next_test_decisions": int(evaluated_decisions),
    }


def road_next_metrics(
    synthetic_sequences: Iterable[tuple[int, ...] | None],
    real_test_sequences: Iterable[tuple[int, ...] | None],
    *,
    state_count: int,
    alpha: float = 0.5,
    feasible_successors: Mapping[int, Sequence[int]] | None = None,
) -> dict[str, float | int]:
    scores = road_next_trip_scores(
        synthetic_sequences,
        real_test_sequences,
        state_count=state_count,
        alpha=alpha,
        feasible_successors=feasible_successors,
    )
    return {
        "road_next_accuracy": float(np.mean(scores["road_next_accuracy"])),
        "road_next_nll": float(np.mean(scores["road_next_nll"])),
        "road_next_training_trips": int(scores["road_next_training_trips"]),
        "road_next_test_trips": int(scores["road_next_test_trips"]),
        "road_next_test_decisions": int(scores["road_next_test_decisions"]),
    }
