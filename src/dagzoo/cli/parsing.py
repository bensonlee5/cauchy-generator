"""Shared Click parameter types and CLI choices."""

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


def parse_finite_float(raw: str, *, flag: str) -> float:
    """Parse one finite float for a Click option."""

    try:
        value = float(raw)
    except ValueError as exc:
        raise click.BadParameter(f"Invalid {flag} value '{raw}'. Expected a number.") from exc
    if not math.isfinite(value):
        raise click.BadParameter(f"Invalid {flag} value '{raw}'. Expected a finite number.")
    return value


def parse_bounded_float(
    raw: str,
    *,
    flag: str,
    lo: float,
    hi: float | None,
    lo_inclusive: bool,
    hi_inclusive: bool,
    expectation: str,
) -> float:
    """Parse a finite float and enforce explicit numeric bounds."""

    value = parse_finite_float(raw, flag=flag)
    lo_ok = value >= lo if lo_inclusive else value > lo
    hi_ok = True
    if hi is not None:
        hi_ok = value <= hi if hi_inclusive else value < hi
    if lo_ok and hi_ok:
        return value
    raise click.BadParameter(f"Invalid {flag} value '{raw}'. Expected {expectation}.")


def parse_warn_threshold_pct(raw: str) -> float:
    """Parse non-negative finite warn threshold percentages."""

    return parse_bounded_float(
        raw,
        flag="--warn-threshold-pct",
        lo=0.0,
        hi=None,
        lo_inclusive=True,
        hi_inclusive=False,
        expectation="a finite value >= 0",
    )


def parse_fail_threshold_pct(raw: str) -> float:
    """Parse non-negative finite fail threshold percentages."""

    return parse_bounded_float(
        raw,
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


class _NonNegativeFiniteFloatType(click.ParamType):
    """Click parameter type for non-negative finite floats."""

    name = "float"

    def __init__(self, *, flag: str) -> None:
        self._flag = flag

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> float:
        raw = str(value)
        try:
            parser = (
                parse_warn_threshold_pct
                if self._flag == "--warn-threshold-pct"
                else parse_fail_threshold_pct
            )
            return parser(raw)
        except click.BadParameter as exc:
            self.fail(exc.message, param, ctx)


class _SetOverrideType(click.ParamType):
    """Click parameter type for ``--set dotted.path=value`` overrides."""

    name = "dotted.path=value"

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> tuple[str, Any]:
        raw = str(value)
        try:
            return parse_set_override(raw)
        except click.BadParameter as exc:
            self.fail(exc.message, param, ctx)


WARN_THRESHOLD_PCT = _NonNegativeFiniteFloatType(flag="--warn-threshold-pct")
FAIL_THRESHOLD_PCT = _NonNegativeFiniteFloatType(flag="--fail-threshold-pct")
SET_OVERRIDE = _SetOverrideType()
