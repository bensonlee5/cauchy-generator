"""Threshold-calibration helpers built on the diversity-audit engine."""

from __future__ import annotations

import math
from typing import Any, Sequence

from dagzoo.config import GeneratorConfig, clone_generator_config
from dagzoo.filter_thresholds import (
    FILTER_THRESHOLD_MAX,
    FILTER_THRESHOLD_MIN,
    validate_filter_threshold,
)

from .runner import run_effective_diversity_audit

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

    variant = clone_generator_config(config, revalidate=False)
    variant.filter.threshold = float(threshold)
    return variant


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

    if not bool(config.filter.enabled):
        raise ValueError("filter-calibration requires filter.enabled: true in the resolved config.")

    baseline_threshold = _normalize_threshold_candidate(
        validate_filter_calibration_threshold(
            config.filter.threshold,
            field_name="filter.threshold",
        )
    )
    threshold_candidates = resolve_filter_calibration_thresholds(
        baseline_threshold=baseline_threshold,
        thresholds=thresholds,
    )
    variant_thresholds = [value for value in threshold_candidates if value != baseline_threshold]
    variant_labels = [_threshold_label(value) for value in variant_thresholds]
    audit_report = run_effective_diversity_audit(
        baseline_config=config,
        baseline_config_path=config_path,
        variant_configs=[
            _build_threshold_variant(config, threshold=value) for value in variant_thresholds
        ],
        variant_config_paths=[config_path] * len(variant_thresholds),
        variant_labels=variant_labels,
        suite=suite,
        num_datasets=num_datasets,
        warmup=warmup,
        device=device,
        warn_threshold_pct=warn_threshold_pct,
        fail_threshold_pct=fail_threshold_pct,
    )

    baseline_entry = audit_report["baseline"]
    variant_entries = audit_report.get("variants", [])
    comparisons = audit_report.get("comparisons", [])
    if len(variant_entries) != len(variant_thresholds):
        raise ValueError("audit_report variants length did not match calibration thresholds.")
    if len(comparisons) != len(variant_thresholds):
        raise ValueError("audit_report comparisons length did not match calibration thresholds.")

    baseline_candidate = _candidate_entry(
        baseline_entry,
        threshold_requested=baseline_threshold,
        diversity_status="reference",
        comparison=None,
    )
    candidates = [baseline_candidate]
    for threshold, variant_entry, comparison in zip(
        variant_thresholds,
        variant_entries,
        comparisons,
        strict=True,
    ):
        candidates.append(
            _candidate_entry(
                variant_entry,
                threshold_requested=threshold,
                diversity_status=str(
                    comparison.get("diversity_status", "insufficient_metrics")
                    if isinstance(comparison, dict)
                    else "insufficient_metrics"
                ),
                comparison=comparison,
            )
        )
    candidates.sort(
        key=lambda item: (
            float(item.get("threshold_requested", 0.0)),
            str(item.get("label", "")),
        )
    )

    best_overall = max(candidates, key=_ranking_value)
    passing_candidates = [
        candidate for candidate in candidates if str(candidate.get("diversity_status")) == "pass"
    ]
    best_passing = max(passing_candidates, key=_ranking_value) if passing_candidates else None
    overall_status = (
        str(audit_report.get("summary", {}).get("overall_status", "insufficient_metrics"))
        if comparisons
        else "reference"
    )

    return {
        "schema_name": "dagzoo_filter_calibration_report",
        "schema_version": 1,
        "generated_at": audit_report.get("generated_at"),
        "baseline": baseline_candidate,
        "candidates": candidates,
        "comparisons": comparisons,
        "summary": {
            "overall_status": overall_status,
            "warn_threshold_pct": float(warn_threshold_pct),
            "fail_threshold_pct": float(fail_threshold_pct),
            "num_candidates": len(candidates),
            "baseline_threshold_requested": float(baseline_threshold),
            "best_overall_threshold_requested": float(best_overall["threshold_requested"]),
            "best_overall_diversity_status": str(best_overall["diversity_status"]),
            "best_passing_threshold_requested": (
                float(best_passing["threshold_requested"]) if best_passing is not None else None
            ),
            "probe_num_datasets": audit_report.get("summary", {}).get("probe_num_datasets"),
            "probe_warmup_datasets": audit_report.get("summary", {}).get("probe_warmup_datasets"),
            "threshold_candidates_requested": [
                float(candidate["threshold_requested"]) for candidate in candidates
            ],
        },
    }
