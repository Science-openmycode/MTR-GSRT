from __future__ import annotations

from fractions import Fraction


BUDGET_FRACTIONS = {
    "endpoint": Fraction(8, 35),
    "od": Fraction(33, 140),
    "geometry": Fraction(4, 35),
    "point_length": Fraction(19, 140),
    "compact_graph_flow": Fraction(1, 7),
    "portal_fiber_q5": Fraction(1, 7),
}


def allocate(total: Fraction) -> dict[str, Fraction]:
    if total <= 0:
        raise ValueError("total epsilon must be positive")
    allocation = {name: total * fraction for name, fraction in BUDGET_FRACTIONS.items()}
    if sum(allocation.values(), Fraction(0, 1)) != total:
        raise RuntimeError("MTR budget allocation does not compose to total epsilon")
    return allocation


def demand_allocation(total: Fraction) -> dict[str, Fraction]:
    values = allocate(total)
    return {name: values[name] for name in ("endpoint", "od", "geometry", "point_length")}
