"""Threshold-calibration helpers built on the diversity-audit engine."""

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


def _threshold_label(value: float) -> str:
    """Return a stable variant label for one requested threshold."""

    return f"thr_{format_filter_calibration_threshold(value)}"


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


def _build_threshold_variant(config: GeneratorConfig, *, threshold: float) -> GeneratorConfig:
    """Clone one config and override only the requested filter threshold."""

    _ = (config, threshold)
    raise NotImplementedError(
        "filter-calibration is not supported for the small-shot ease filter yet."
    )


def _candidate_entry(
    entry: dict[str, Any],
    *,
    threshold_requested: float,
    diversity_status: str,
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one flattened calibration candidate record."""

    filter_summary = entry.get("filter_summary")
    if not isinstance(filter_summary, dict):
        filter_summary = {}
    comparison_payload = comparison if isinstance(comparison, dict) else {}
    return {
        "label": entry.get("label"),
        "config_path": entry.get("config_path"),
        "suite": entry.get("suite"),
        "num_datasets": entry.get("num_datasets"),
        "warmup_datasets": entry.get("warmup_datasets"),
        "requested_device": entry.get("requested_device"),
        "resolved_device": entry.get("resolved_device"),
        "resolved_config": entry.get("resolved_config"),
        "threshold_requested": float(threshold_requested),
        "threshold_effective_mean": filter_summary.get("threshold_effective_mean"),
        "threshold_delta_mean": filter_summary.get("threshold_delta_mean"),
        "wins_ratio_mean": filter_summary.get("wins_ratio_mean"),
        "n_valid_oob_mean": filter_summary.get("n_valid_oob_mean"),
        "accepted_true_fraction": filter_summary.get("accepted_true_fraction"),
        "reason_counts": dict(filter_summary.get("reason_counts", {})),
        "datasets_per_minute": entry.get("datasets_per_minute"),
        "filter_datasets_per_minute": entry.get("filter_datasets_per_minute"),
        "filter_accepted_datasets_per_minute": entry.get("filter_accepted_datasets_per_minute"),
        "filter_accepted_datasets_measured": entry.get("filter_accepted_datasets_measured"),
        "filter_rejected_datasets_measured": entry.get("filter_rejected_datasets_measured"),
        "filter_acceptance_rate_dataset_level": entry.get("filter_acceptance_rate_dataset_level"),
        "filter_rejection_rate_dataset_level": entry.get("filter_rejection_rate_dataset_level"),
        "mechanism_family_summary": entry.get("mechanism_family_summary"),
        "diversity_status": str(diversity_status),
        "diversity_composite_shift_pct": comparison_payload.get("diversity_composite_shift_pct"),
        "diversity_metric_shift_pct": comparison_payload.get("diversity_metric_shift_pct"),
        "datasets_per_minute_delta_pct": comparison_payload.get("datasets_per_minute_delta_pct"),
        "filter_accepted_datasets_per_minute_delta_pct": comparison_payload.get(
            "filter_accepted_datasets_per_minute_delta_pct"
        ),
    }


def _ranking_value(candidate: dict[str, Any]) -> float:
    """Return ranking score for candidate selection."""

    value = candidate.get("filter_accepted_datasets_per_minute")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return float("-inf")
    return float(value)


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
    """Run threshold-only filter calibration against the rewritten audit engine."""

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
