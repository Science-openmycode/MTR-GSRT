"""Executable fixed-capacity trajectory domain with implicit null slots."""
from __future__ import annotations

import numpy as np


def validate_record_count(count: int, capacity: int) -> tuple[int, int]:
    count = int(count)
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("Public capacity must be positive")
    if count < 0 or count > capacity:
        raise RuntimeError("Input is outside the fixed-capacity trajectory domain")
    return count, capacity - count


def validate_trajectory_records(records, capacity: int) -> tuple[int, int]:
    """Validate non-null records; omitted slots are canonical implicit nulls."""
    count, null_slots = validate_record_count(len(records), capacity)
    for record in records:
        values = np.asarray(record, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != 2
            or len(values) < 2
            or not np.all(np.isfinite(values))
            or np.any(np.abs(values[:, 0]) > 90.0)
            or np.any(np.abs(values[:, 1]) > 180.0)
        ):
            raise RuntimeError("Private loader returned a malformed trajectory")
    return count, null_slots
