"""Shared filter-threshold validation and policy constants."""

from __future__ import annotations

import math

FILTER_THRESHOLD_MIN = 0.0
FILTER_THRESHOLD_MAX = 1.0
FILTER_THRESHOLD_EXPECTATION = "a finite value in [0.0, 1.0]"
ZERO_FILTER_THRESHOLD_POLICY = "zero_bypass_v1"
ZERO_FILTER_THRESHOLD_BACKEND = "filter_threshold_bypass"


def validate_filter_threshold(value: object, *, field_name: str) -> float:
    """Validate one filter-threshold value against the shared public contract."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be {FILTER_THRESHOLD_EXPECTATION}.")
    as_float = float(value)
    if not (FILTER_THRESHOLD_MIN <= as_float <= FILTER_THRESHOLD_MAX):
        raise ValueError(f"{field_name} must be {FILTER_THRESHOLD_EXPECTATION}.")
    return as_float


def is_filter_threshold_bypass(value: float) -> bool:
    """Return whether one validated threshold requests explicit bypass mode."""

    return float(value) == FILTER_THRESHOLD_MIN
