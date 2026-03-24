"""Filter-calibration helpers built on the diversity-audit engine."""

from __future__ import annotations

import math
from typing import Any, Sequence

from dagzoo.config import GeneratorConfig
from dagzoo.filter_thresholds import (
    FILTER_THRESHOLD_MAX,
    FILTER_THRESHOLD_MIN,
    validate_filter_threshold,
)

DEFAULT_FILTER_CALIBRATION_DELTAS: tuple[float, ...] = (-0.15, -0.10, -0.05, 0.0, 0.05)
_CALIBRATION_THRESHOLD_MIN = FILTER_THRESHOLD_MIN
_CALIBRATION_THRESHOLD_MAX = FILTER_THRESHOLD_MAX


def validate_filter_calibration_threshold(
    value: object,
    *,
    field_name: str,
) -> float:
    """Validate one calibration threshold value."""

    return validate_filter_threshold(value, field_name=field_name)


def _normalize_threshold_candidate(value: float) -> float:
    """Normalize one requested threshold to a stable persisted float."""

    return round(float(value), 6)


def _clamp_threshold_candidate(value: float) -> float:
    """Clamp one requested threshold into the default calibration sweep window."""

    return min(_CALIBRATION_THRESHOLD_MAX, max(_CALIBRATION_THRESHOLD_MIN, float(value)))


def format_filter_calibration_threshold(value: object) -> str:
    """Render a threshold value with stable precision for user-facing surfaces."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "-"
    normalized = _normalize_threshold_candidate(float(value))
    rendered = f"{normalized:.6f}".rstrip("0").rstrip(".")
    if "." not in rendered:
        rendered = f"{rendered}.0"
    return rendered


def resolve_filter_calibration_thresholds(
    *,
    baseline_threshold: float,
    thresholds: Sequence[float] | None,
) -> list[float]:
    """Resolve the requested threshold sweep including the baseline threshold."""

    baseline_value = _normalize_threshold_candidate(
        validate_filter_calibration_threshold(
            baseline_threshold,
            field_name="baseline_threshold",
        )
    )
    if thresholds is None:
        raw_candidates = [
            _clamp_threshold_candidate(float(baseline_value) + float(delta))
            for delta in DEFAULT_FILTER_CALIBRATION_DELTAS
        ]
    else:
        raw_candidates = [
            _normalize_threshold_candidate(
                validate_filter_calibration_threshold(
                    value,
                    field_name="thresholds",
                )
            )
            for value in thresholds
        ]

    normalized = {_normalize_threshold_candidate(float(value)) for value in raw_candidates}
    normalized.add(baseline_value)
    return sorted(normalized)


def run_filter_calibration(
    *,
    config: GeneratorConfig,
    config_path: str,
    thresholds: Sequence[float] | None,
    suite: str,
    num_datasets: int | None,
    warmup: int | None,
    device: str | None,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
) -> dict[str, Any]:
    """Fail fast until a calibration workflow exists for the small-shot ease filter."""

    _ = (
        config,
        config_path,
        thresholds,
        suite,
        num_datasets,
        warmup,
        device,
        warn_threshold_pct,
        fail_threshold_pct,
    )
    raise NotImplementedError(
        "filter-calibration is not supported for the small-shot ease filter yet."
    )
