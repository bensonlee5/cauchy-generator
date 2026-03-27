"""Argument parser types and shared CLI choices."""

from __future__ import annotations

import argparse
import math
from typing import Any

import yaml

from dagzoo.hardware_policy import list_hardware_policies
from dagzoo.rng import SEED32_MAX, SEED32_MIN

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
HARDWARE_POLICY_CHOICES = list_hardware_policies()


def positive_int(value: str) -> int:
    """argparse type: parse an integer > 0."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}.")
    return parsed


def non_negative_int(value: str) -> int:
    """argparse type: parse an integer >= 0."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got {value}.")
    return parsed


def seed_32bit_int(value: str) -> int:
    """argparse type: parse an integer seed in the unsigned 32-bit range."""

    parsed = int(value)
    if parsed < SEED32_MIN or parsed > SEED32_MAX:
        raise argparse.ArgumentTypeError(
            f"Expected a seed in [{SEED32_MIN}, {SEED32_MAX}], got {value}."
        )
    return parsed


def parse_finite_float(raw: str, *, flag: str) -> float:
    """argparse helper: parse a finite float."""

    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid {flag} value '{raw}'. Expected a number."
        ) from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"Invalid {flag} value '{raw}'. Expected a finite number.")
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
    """argparse helper: parse a finite float and enforce explicit numeric bounds."""

    value = parse_finite_float(raw, flag=flag)
    lo_ok = value >= lo if lo_inclusive else value > lo
    hi_ok = True
    if hi is not None:
        hi_ok = value <= hi if hi_inclusive else value < hi
    if lo_ok and hi_ok:
        return value
    raise argparse.ArgumentTypeError(f"Invalid {flag} value '{raw}'. Expected {expectation}.")


def parse_warn_threshold_pct_arg(raw: str) -> float:
    """argparse type: parse non-negative finite warn threshold percentages."""

    return parse_bounded_float(
        raw,
        flag="--warn-threshold-pct",
        lo=0.0,
        hi=None,
        lo_inclusive=True,
        hi_inclusive=False,
        expectation="a finite value >= 0",
    )


def parse_fail_threshold_pct_arg(raw: str) -> float:
    """argparse type: parse non-negative finite fail threshold percentages."""

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
    """argparse type: parse a dotted-path override in ``path=value`` form."""

    path_raw, separator, value_raw = raw.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(
            f"Invalid --set value '{raw}'. Expected dotted.path=value."
        )
    path = path_raw.strip()
    if not path or path.startswith(".") or path.endswith(".") or ".." in path:
        raise argparse.ArgumentTypeError(
            f"Invalid --set path '{path_raw}'. Expected dotted.path segments."
        )
    for segment in path.split("."):
        if not segment:
            raise argparse.ArgumentTypeError(
                f"Invalid --set path '{path_raw}'. Expected dotted.path segments."
            )
    try:
        value = yaml.safe_load(value_raw)
    except yaml.YAMLError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --set value '{value_raw}'. Expected YAML-scalar compatible syntax."
        ) from exc
    return path, value
