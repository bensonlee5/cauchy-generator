"""Scalar validation helpers for config normalization."""

from __future__ import annotations

import math
from typing import Any


def _validate_finite_float_field(
    *,
    field_name: str,
    value: Any,
    lo: float,
    hi: float | None,
    lo_inclusive: bool,
    hi_inclusive: bool,
    expectation: str,
) -> float:
    """Validate a float field against finite bounds and normalize it."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be {expectation}, got {value!r}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be {expectation}, got {value!r}.") from exc
    lo_ok = parsed >= lo if lo_inclusive else parsed > lo
    hi_ok = True
    if hi is not None:
        hi_ok = parsed <= hi if hi_inclusive else parsed < hi
    if not math.isfinite(parsed) or not (lo_ok and hi_ok):
        raise ValueError(f"{field_name} must be {expectation}, got {parsed!r}.")
    return parsed


def _validate_optional_finite_float_field(
    *,
    field_name: str,
    value: Any,
    lo: float,
    hi: float | None,
    lo_inclusive: bool,
    hi_inclusive: bool,
    expectation: str,
) -> float | None:
    """Validate an optional float field against finite bounds and normalize it."""

    if value is None:
        return None
    return _validate_finite_float_field(
        field_name=field_name,
        value=value,
        lo=lo,
        hi=hi,
        lo_inclusive=lo_inclusive,
        hi_inclusive=hi_inclusive,
        expectation=expectation,
    )


def _validate_int_field(
    *,
    field_name: str,
    value: Any,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Validate and normalize integer fields with optional inclusive bounds."""

    if isinstance(value, bool):
        expectation = (
            f"an integer in [{minimum}, {maximum}]"
            if maximum is not None
            else f"an integer >= {minimum}"
        )
        raise ValueError(f"{field_name} must be {expectation}, got {value!r}.")

    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        normalized = value.strip()
        signless = normalized[1:] if normalized.startswith(("+", "-")) else normalized
        if not signless.isdigit():
            expectation = (
                f"an integer in [{minimum}, {maximum}]"
                if maximum is not None
                else f"an integer >= {minimum}"
            )
            raise ValueError(f"{field_name} must be {expectation}, got {value!r}.")
        parsed = int(normalized)
    else:
        expectation = (
            f"an integer in [{minimum}, {maximum}]"
            if maximum is not None
            else f"an integer >= {minimum}"
        )
        raise ValueError(f"{field_name} must be {expectation}, got {value!r}.")

    if parsed < minimum or (maximum is not None and parsed > maximum):
        expectation = (
            f"an integer in [{minimum}, {maximum}]"
            if maximum is not None
            else f"an integer >= {minimum}"
        )
        raise ValueError(f"{field_name} must be {expectation}, got {parsed!r}.")
    return parsed


def _validate_optional_int_field(
    *,
    field_name: str,
    value: Any,
    minimum: int,
    maximum: int | None = None,
) -> int | None:
    """Validate and normalize optional integer fields with inclusive bounds."""

    if value is None:
        return None
    return _validate_int_field(
        field_name=field_name,
        value=value,
        minimum=minimum,
        maximum=maximum,
    )


def _validate_min_max_pair(
    *,
    name: str,
    min_value: int | None,
    max_value: int | None,
    max_label: str,
) -> None:
    """Validate that optional min/max values are ordered when both are provided."""

    if min_value is None or max_value is None:
        return
    if min_value > max_value:
        raise ValueError(f"{name} must be <= {max_label}, got {min_value} > {max_value}.")
