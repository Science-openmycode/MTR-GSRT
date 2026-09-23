"""Road-choice fidelity and next-road prediction on fixed public supports.

The metric core is deliberately independent of a particular road adapter.
Each adapter supplies:

* a public choice context ``c``;
* the complete public option set ``E(c)``; and
* one sequence of observed ``(c, e)`` events per trajectory.

All eligible contexts are determined by public support and the frozen real
test set.  No context is selected from a method's utility.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np


Context = Hashable
Option = Hashable
ChoiceEvent = tuple[Context, Option]
ChoiceTrip = Sequence[ChoiceEvent]


def _canonical(values: Iterable[Option]) -> tuple[Option, ...]:
    return tuple(sorted(values, key=lambda value: (type(value).__name__, repr(value))))


def validate_public_options(
    public_options: Mapping[Context, Sequence[Option]],
) -> dict[Context, tuple[Option, ...]]:
    normalized: dict[Context, tuple[Option, ...]] = {}
    for context, raw_options in public_options.items():
        options = _canonical(raw_options)
        if len(options) != len(set(options)):
            raise ValueError(f"duplicate public road option at choice context {context!r}")
        if len(options) >= 2:
            normalized[context] = options
    if not normalized:
        raise ValueError("road-choice evaluation requires a public context with at least two options")
    return normalized


def public_crossing_options(
    edge_to_regions: Mapping[int, tuple[int, int]],
    edge_to_options: Mapping[int, Option] | None = None,
) -> tuple[dict[Context, tuple[Option, ...]], dict[int, Context]]:
    """Derive choices from ordered crossings in the public quotient graph.

    ``edge_to_options`` may collapse parallel network features onto the
    consumer-visible directed-node pair represented by a road witness.
    """
    grouped: dict[Context, set[Option]] = defaultdict(set)
    for edge_id, pair in edge_to_regions.items():
        if len(pair) != 2:
            raise ValueError(f"edge {edge_id!r} must map to two endpoint regions")
        source_region, target_region = pair
        if source_region != target_region:
            option = edge_id if edge_to_options is None else edge_to_options[edge_id]
            grouped[(source_region, target_region)].add(option)
    support = validate_public_options(grouped)
    edge_context = {
        int(edge_id): tuple(pair)
        for edge_id, pair in edge_to_regions.items()
        if tuple(pair) in support
    }
    return support, edge_context


def matched_choice_trips(
    records: Iterable[Mapping[str, object]],
    edge_context: Mapping[int, Context],
    edge_options: Mapping[int, Option] | None = None,
) -> list[list[ChoiceEvent]]:
    """Convert accepted matched paths into public road-choice events."""
    trips: list[list[ChoiceEvent]] = []
    for record in records:
        if not bool(record.get("accepted", False)):
            trips.append([])
            continue
        events: list[ChoiceEvent] = []
        for raw_edge in record.get("cpath", ()):
            edge_id = int(raw_edge)
            context = edge_context.get(edge_id)
            if context is not None:
                option = edge_id if edge_options is None else edge_options[edge_id]
                events.append((context, option))
        trips.append(events)
    return trips


def normalized_choice_counts(
    trips: Iterable[ChoiceTrip],
    public_options: Mapping[Context, Sequence[Option]],
) -> tuple[dict[Context, Counter[Option]], int, int]:
    """Count choices with unit mass per contributing trajectory.

    A trajectory containing ``k`` eligible decisions contributes ``1/k`` to
    each decision.  This prevents long paths from dominating a
    trajectory-record-level population comparison.
    """
    support = validate_public_options(public_options)
    counts: dict[Context, Counter[Option]] = defaultdict(Counter)
    contributing_trips = 0
    event_count = 0
    for trip in trips:
        selected: list[ChoiceEvent] = []
        for context, option in trip:
            if context not in support:
                continue
            if option not in support[context]:
                raise ValueError(
                    f"observed option {option!r} is outside public support at {context!r}"
                )
            selected.append((context, option))
        if not selected:
            continue
        contributing_trips += 1
        event_count += len(selected)
        mass = 1.0 / len(selected)
        for context, option in selected:
            counts[context][option] += mass
    return dict(counts), contributing_trips, event_count


def _distribution(
    counter: Mapping[Option, float] | None,
    options: Sequence[Option],
) -> np.ndarray:
    values = np.asarray(
        [0.0 if counter is None else float(counter.get(option, 0.0)) for option in options],
        dtype=float,
    )
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("road-choice counts must be finite and nonnegative")
    total = float(values.sum())
    return values / total if total > 0 else np.full(len(options), 1.0 / len(options))


def _conditional_or_zero(
    counter: Mapping[Option, float] | None,
    options: Sequence[Option],
) -> np.ndarray:
    values = np.asarray(
        [0.0 if counter is None else float(counter.get(option, 0.0)) for option in options],
        dtype=float,
    )
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("road-choice counts must be finite and nonnegative")
    total = float(values.sum())
    return values / total if total > 0 else np.zeros(len(options), dtype=float)


def _tie_aware_dcg(relevance: np.ndarray, score: np.ndarray) -> float:
    """Expected linear-gain DCG under uniform ordering inside score ties."""
    if len(relevance) != len(score) or len(relevance) == 0:
        raise ValueError("DCG vectors must have the same positive length")
    order = np.argsort(-score, kind="stable")
    discounts = 1.0 / np.log2(np.arange(2, len(score) + 2, dtype=float))
    dcg = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        mean_relevance = float(np.mean(relevance[order[start:end]]))
        dcg += mean_relevance * float(np.sum(discounts[start:end]))
        start = end
    return dcg


def road_choice_fidelity(
    synthetic_trips: Iterable[ChoiceTrip],
    real_test_trips: Iterable[ChoiceTrip],
    public_options: Mapping[Context, Sequence[Option]],
) -> dict[str, float | int]:
    """Compute joint context-road CPC and demand-weighted complete-list NDCG."""
    support = validate_public_options(public_options)
    real_counts, real_trips, real_events = normalized_choice_counts(real_test_trips, support)
    synthetic_counts, synthetic_trips_count, synthetic_events = normalized_choice_counts(
        synthetic_trips, support
    )
    eligible = [context for context in support if sum(real_counts.get(context, {}).values()) > 0]
    if not eligible:
        raise ValueError("frozen real test set contains no public road-choice event")
    total_real_mass = float(sum(sum(counter.values()) for counter in real_counts.values()))
    total_synthetic_mass = float(
        sum(sum(counter.values()) for counter in synthetic_counts.values())
    )
    cpc = 0.0
    ndcg = 0.0
    for context in eligible:
        options = support[context]
        p = _distribution(real_counts[context], options)
        q = _conditional_or_zero(synthetic_counts.get(context), options)
        weight = float(sum(real_counts[context].values())) / total_real_mass
        real_joint = np.asarray(
            [float(real_counts[context].get(option, 0.0)) / total_real_mass for option in options]
        )
        synthetic_joint = (
            np.asarray(
                [
                    float(synthetic_counts.get(context, {}).get(option, 0.0))
                    / total_synthetic_mass
                    for option in options
                ]
            )
            if total_synthetic_mass > 0
            else np.zeros(len(options), dtype=float)
        )
        cpc += float(np.minimum(real_joint, synthetic_joint).sum())
        dcg = _tie_aware_dcg(p, q)
        ideal = _tie_aware_dcg(p, p)
        if ideal <= 0:
            raise ValueError(f"road-choice context has zero ideal DCG: {context!r}")
        ndcg += weight * dcg / ideal
    if not (-1e-12 <= cpc <= 1.0 + 1e-12 and -1e-12 <= ndcg <= 1.0 + 1e-12):
        raise AssertionError(f"road-choice scores left [0,1]: CPC={cpc}, NDCG={ndcg}")
    return {
        "road_choice_cpc": float(np.clip(cpc, 0.0, 1.0)),
        "road_choice_ndcg": float(np.clip(ndcg, 0.0, 1.0)),
        "road_choice_contexts": int(len(eligible)),
        "road_choice_real_test_trips": int(real_trips),
        "road_choice_real_test_events": int(real_events),
        "road_choice_synthetic_trips": int(synthetic_trips_count),
        "road_choice_synthetic_events": int(synthetic_events),
    }


def next_road_trip_scores(
    synthetic_trips: Iterable[ChoiceTrip],
    real_test_trips: Iterable[ChoiceTrip],
    public_options: Mapping[Context, Sequence[Option]],
    *,
    mixture_weight: float = 0.5,
) -> dict[str, np.ndarray | int]:
    """Evaluate a synthetic next-road law on frozen real-test trajectories."""
    if not 0.0 < mixture_weight < 1.0:
        raise ValueError("mixture_weight must lie strictly between zero and one")
    support = validate_public_options(public_options)
    synthetic_counts, training_trips, training_events = normalized_choice_counts(
        synthetic_trips, support
    )
    trip_accuracy: list[float] = []
    trip_nll: list[float] = []
    evaluated_events = 0
    for trip in real_test_trips:
        selected = [(context, option) for context, option in trip if context in support]
        if not selected:
            continue
        correct: list[float] = []
        losses: list[float] = []
        for context, observed in selected:
            options = support[context]
            if observed not in options:
                raise ValueError(
                    f"real-test option {observed!r} is outside public support at {context!r}"
                )
            empirical = _distribution(synthetic_counts.get(context), options)
            best = float(np.max(empirical))
            prediction = next(
                option
                for option, probability in zip(options, empirical)
                if math.isclose(float(probability), best, rel_tol=0.0, abs_tol=1e-15)
            )
            correct.append(float(prediction == observed))
            index = options.index(observed)
            probability = (
                (1.0 - mixture_weight) / len(options)
                + mixture_weight * float(empirical[index])
            )
            losses.append(-math.log(probability))
            evaluated_events += 1
        trip_accuracy.append(float(np.mean(correct)))
        trip_nll.append(float(np.mean(losses)))
    if not trip_accuracy:
        raise ValueError("frozen real test set contains no public road-choice event")
    return {
        "next_road_accuracy": np.asarray(trip_accuracy, dtype=float),
        "next_road_nll": np.asarray(trip_nll, dtype=float),
        "next_road_training_trips": int(training_trips),
        "next_road_training_events": int(training_events),
        "next_road_test_trips": int(len(trip_accuracy)),
        "next_road_test_events": int(evaluated_events),
    }


def road_choice_metrics(
    synthetic_trips: Iterable[ChoiceTrip],
    real_test_trips: Iterable[ChoiceTrip],
    public_options: Mapping[Context, Sequence[Option]],
    *,
    mixture_weight: float = 0.5,
) -> dict[str, float | int]:
    synthetic = list(synthetic_trips)
    real_test = list(real_test_trips)
    fidelity = road_choice_fidelity(synthetic, real_test, public_options)
    scores = next_road_trip_scores(
        synthetic,
        real_test,
        public_options,
        mixture_weight=mixture_weight,
    )
    return {
        **fidelity,
        "next_road_accuracy": float(np.mean(scores["next_road_accuracy"])),
        "next_road_nll": float(np.mean(scores["next_road_nll"])),
        "next_road_training_trips": int(scores["next_road_training_trips"]),
        "next_road_training_events": int(scores["next_road_training_events"]),
        "next_road_test_trips": int(scores["next_road_test_trips"]),
        "next_road_test_events": int(scores["next_road_test_events"]),
        "next_road_mixture_weight": float(mixture_weight),
    }
