"""Shared Click parsing helpers and CLI choices."""

from __future__ import annotations

import math
from typing import Any

import click
import yaml

from dagzoo.hardware_policy import list_hardware_policies
from dagzoo.rng import SEED32_MAX, SEED32_MIN

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
HARDWARE_POLICY_CHOICES = tuple(list_hardware_policies())

POSITIVE_INT = click.IntRange(min=1)
NON_NEGATIVE_INT = click.IntRange(min=0)
SEED32_INT = click.IntRange(min=SEED32_MIN, max=SEED32_MAX)


def device_choice() -> click.Choice[str]:
    """Return the shared device choice type."""

    return click.Choice(DEVICE_CHOICES)


def hardware_policy_choice() -> click.Choice[str]:
    """Return the shared hardware-policy choice type."""

    return click.Choice(HARDWARE_POLICY_CHOICES)


def parse_finite_float(value: str | float, *, flag: str) -> float:
    """Parse one finite float for a Click option."""

    raw = str(value)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise click.BadParameter(f"Invalid {flag} value '{raw}'. Expected a number.") from exc
    if not math.isfinite(parsed):
        raise click.BadParameter(f"Invalid {flag} value '{raw}'. Expected a finite number.")
    return parsed


def parse_bounded_float(
    value: str | float,
    *,
    flag: str,
    lo: float,
    hi: float | None,
    lo_inclusive: bool,
    hi_inclusive: bool,
    expectation: str,
) -> float:
    """Parse a finite float and enforce explicit numeric bounds."""

    parsed = parse_finite_float(value, flag=flag)
    raw = str(value)
    lo_ok = parsed >= lo if lo_inclusive else parsed > lo
    hi_ok = True
    if hi is not None:
        hi_ok = parsed <= hi if hi_inclusive else parsed < hi
    if lo_ok and hi_ok:
        return parsed
    raise click.BadParameter(f"Invalid {flag} value '{raw}'. Expected {expectation}.")


def parse_warn_threshold_pct(value: str | float) -> float:
    """Parse non-negative finite warn threshold percentages."""

    return parse_bounded_float(
        value,
        flag="--warn-threshold-pct",
        lo=0.0,
        hi=None,
        lo_inclusive=True,
        hi_inclusive=False,
        expectation="a finite value >= 0",
    )


def parse_fail_threshold_pct(value: str | float) -> float:
    """Parse non-negative finite fail threshold percentages."""

    return parse_bounded_float(
        value,
        flag="--fail-threshold-pct",
        lo=0.0,
        hi=None,
        lo_inclusive=True,
        hi_inclusive=False,
        expectation="a finite value >= 0",
    )


def parse_set_override(raw: str) -> tuple[str, Any]:
    """Parse a dotted-path override in ``path=value`` form."""

    path_raw, separator, value_raw = raw.partition("=")
    if not separator:
        raise click.BadParameter(f"Invalid --set value '{raw}'. Expected dotted.path=value.")
    path = path_raw.strip()
    if not path or path.startswith(".") or path.endswith(".") or ".." in path:
        raise click.BadParameter(f"Invalid --set path '{path_raw}'. Expected dotted.path segments.")
    for segment in path.split("."):
        if not segment:
            raise click.BadParameter(
                f"Invalid --set path '{path_raw}'. Expected dotted.path segments."
            )
    try:
        value = yaml.safe_load(value_raw)
    except yaml.YAMLError as exc:
        raise click.BadParameter(
            f"Invalid --set value '{value_raw}'. Expected YAML-scalar compatible syntax."
        ) from exc
    return path, value


def warn_threshold_pct_callback(
    _ctx: click.Context,
    _param: click.Parameter,
    value: float | None,
) -> float | None:
    """Validate and normalize ``--warn-threshold-pct`` values."""

    if value is None:
        return None
    return parse_warn_threshold_pct(value)


def fail_threshold_pct_callback(
    _ctx: click.Context,
    _param: click.Parameter,
    value: float | None,
) -> float | None:
    """Validate and normalize ``--fail-threshold-pct`` values."""

    if value is None:
        return None
    return parse_fail_threshold_pct(value)


def set_overrides_callback(
    _ctx: click.Context,
    _param: click.Parameter,
    value: tuple[str, ...],
) -> tuple[tuple[str, Any], ...]:
    """Parse repeated ``--set`` arguments into typed override tuples."""

    return tuple(parse_set_override(raw) for raw in value)
