"""Argument parser types and shared CLI choices."""

from __future__ import annotations

import argparse
import math

from dagzoo.config import (
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MNAR,
    MISSINGNESS_MECHANISM_NONE,
    normalize_missing_mechanism,
)
from dagzoo.filter_thresholds import (
    FILTER_THRESHOLD_EXPECTATION,
    FILTER_THRESHOLD_MAX,
    FILTER_THRESHOLD_MIN,
)
from dagzoo.hardware_policy import list_hardware_policies
from dagzoo.rng import SEED32_MAX, SEED32_MIN

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
HARDWARE_POLICY_CHOICES = list_hardware_policies()
MISSINGNESS_MECHANISM_CLI_CHOICES = (
    MISSINGNESS_MECHANISM_NONE,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MNAR,
)


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


def filter_n_jobs(value: str) -> int:
    """argparse type: parse filter worker count (-1 or >= 1)."""

    parsed = int(value)
    if parsed == -1 or parsed >= 1:
        return parsed
    raise argparse.ArgumentTypeError(f"Expected -1 or an integer >= 1 for --n-jobs, got {value}.")


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


def parse_missing_rate_arg(raw: str) -> float:
    """argparse type: parse missing rate in [0, 1]."""

    return parse_bounded_float(
        raw,
        flag="--missing-rate",
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )


def parse_missing_mar_observed_fraction_arg(raw: str) -> float:
    """argparse type: parse MAR observed-feature fraction in (0, 1]."""

    return parse_bounded_float(
        raw,
        flag="--missing-mar-observed-fraction",
        lo=0.0,
        hi=1.0,
        lo_inclusive=False,
        hi_inclusive=True,
        expectation="a finite value in (0, 1]",
    )


def parse_missing_mar_logit_scale_arg(raw: str) -> float:
    """argparse type: parse MAR logit scale > 0."""

    return parse_bounded_float(
        raw,
        flag="--missing-mar-logit-scale",
        lo=0.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 0",
    )


def parse_missing_mnar_logit_scale_arg(raw: str) -> float:
    """argparse type: parse MNAR logit scale > 0."""

    return parse_bounded_float(
        raw,
        flag="--missing-mnar-logit-scale",
        lo=0.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 0",
    )


def parse_easy_skill_threshold_arg(raw: str) -> float:
    """argparse type: parse small-shot ease threshold in [0, 1]."""

    return parse_bounded_float(
        raw,
        flag="--easy-skill-threshold",
        lo=FILTER_THRESHOLD_MIN,
        hi=FILTER_THRESHOLD_MAX,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation=FILTER_THRESHOLD_EXPECTATION,
    )


def parse_easy_gain_threshold_arg(raw: str) -> float:
    """argparse type: parse full-vs-small gain threshold in [0, 1]."""

    return parse_bounded_float(
        raw,
        flag="--easy-gain-threshold",
        lo=FILTER_THRESHOLD_MIN,
        hi=FILTER_THRESHOLD_MAX,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation=FILTER_THRESHOLD_EXPECTATION,
    )


def parse_hard_skill_threshold_arg(raw: str) -> float:
    """argparse type: parse hard-side garbage threshold in [0, 1]."""

    return parse_bounded_float(
        raw,
        flag="--hard-skill-threshold",
        lo=FILTER_THRESHOLD_MIN,
        hi=FILTER_THRESHOLD_MAX,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation=FILTER_THRESHOLD_EXPECTATION,
    )


def parse_stump_skill_threshold_arg(raw: str) -> float:
    """argparse type: parse stump-veto threshold in [0, 1]."""

    return parse_bounded_float(
        raw,
        flag="--stump-skill-threshold",
        lo=FILTER_THRESHOLD_MIN,
        hi=FILTER_THRESHOLD_MAX,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation=FILTER_THRESHOLD_EXPECTATION,
    )


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


def parse_missing_mechanism_arg(raw: str) -> str:
    """argparse type: normalize missingness mechanism values."""

    try:
        return normalize_missing_mechanism(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
