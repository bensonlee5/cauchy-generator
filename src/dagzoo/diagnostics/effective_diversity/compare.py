"""Coverage-summary comparison helpers for diversity audits."""

from __future__ import annotations

from statistics import median
from typing import Any

from dagzoo.math import coerce_optional_finite_float as _coerce_optional_finite_float

CORE_DIVERSITY_METRICS: tuple[str, ...] = (
    "linearity_proxy",
    "nonlinearity_proxy",
    "wins_ratio_proxy",
    "pearson_abs_mean",
    "pearson_abs_max",
    "snr_proxy_db",
    "class_entropy",
    "majority_minority_ratio",
    "categorical_ratio",
    "cat_cardinality_mean",
    "graph_edge_density",
    "graph_indegree_std",
    "graph_outdegree_std",
    "graph_depth_ratio",
    "graph_target_depth_ratio",
    "graph_reachability_ratio",
    "graph_ancestor_overlap_mean",
    "graph_target_ancestor_fraction",
    "mechanism_family_cooccurrence_ratio",
)
STRUCTURAL_DIVERSITY_METRICS: tuple[str, ...] = (
    "categorical_ratio",
    "cat_cardinality_mean",
    "graph_edge_density",
    "graph_indegree_std",
    "graph_outdegree_std",
    "graph_depth_ratio",
    "graph_target_depth_ratio",
    "graph_reachability_ratio",
    "graph_ancestor_overlap_mean",
    "graph_target_ancestor_fraction",
    "mechanism_family_cooccurrence_ratio",
)


def validate_diversity_thresholds(
    *,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
) -> tuple[float, float]:
    """Validate diversity threshold values and ordering."""

    warn_value = _coerce_optional_finite_float(warn_threshold_pct)
    fail_value = _coerce_optional_finite_float(fail_threshold_pct)
    if warn_value is None or fail_value is None:
        raise ValueError("warn_threshold_pct and fail_threshold_pct must be finite values.")
    if warn_value < 0.0 or fail_value < 0.0:
        raise ValueError("warn_threshold_pct and fail_threshold_pct must be >= 0.")
    if warn_value > fail_value:
        raise ValueError("warn_threshold_pct must be <= fail_threshold_pct.")
    return float(warn_value), float(fail_value)


def _metric_payload(summary: dict[str, Any], metric: str) -> dict[str, Any] | None:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    payload = metrics.get(metric)
    if not isinstance(payload, dict):
        return None
    return payload


def _metric_mean(summary: dict[str, Any], metric: str) -> float | None:
    payload = _metric_payload(summary, metric)
    if payload is None:
        return None
    return _coerce_optional_finite_float(payload.get("mean"))


def _metric_quantile(summary: dict[str, Any], metric: str, key: str) -> float | None:
    payload = _metric_payload(summary, metric)
    if payload is None:
        return None
    quantiles = payload.get("quantiles")
    if not isinstance(quantiles, dict):
        return None
    return _coerce_optional_finite_float(quantiles.get(key))


def _delta_pct(current: float | None, baseline: float | None) -> float | None:
    """Return percentage delta vs baseline when both values are comparable."""

    if current is None or baseline is None:
        return None
    if baseline == 0.0:
        return 0.0 if current == 0.0 else None
    return float(((current - baseline) / abs(baseline)) * 100.0)


def metric_shift_pct(
    *,
    baseline_summary: dict[str, Any],
    variant_summary: dict[str, Any],
    metric: str,
) -> float | None:
    """Compute the normalized percent shift for one diversity metric."""

    baseline_mean = _metric_mean(baseline_summary, metric)
    variant_mean = _metric_mean(variant_summary, metric)
    baseline_p50 = _metric_quantile(baseline_summary, metric, "p50")
    variant_p50 = _metric_quantile(variant_summary, metric, "p50")
    baseline_p25 = _metric_quantile(baseline_summary, metric, "p25")
    baseline_p75 = _metric_quantile(baseline_summary, metric, "p75")
    if (
        baseline_mean is None
        or variant_mean is None
        or baseline_p50 is None
        or variant_p50 is None
        or baseline_p25 is None
        or baseline_p75 is None
    ):
        return None
    iqr = abs(float(baseline_p75) - float(baseline_p25))
    scale = max(abs(float(baseline_mean)), iqr, 1e-6)
    weighted_shift = (0.6 * abs(float(variant_mean) - float(baseline_mean)) / scale) + (
        0.4 * abs(float(variant_p50) - float(baseline_p50)) / scale
    )
    return float(weighted_shift * 100.0)


def _metric_shift_map(
    *,
    baseline_summary: dict[str, Any],
    variant_summary: dict[str, Any],
    metrics: tuple[str, ...],
) -> dict[str, float]:
    shifts: dict[str, float] = {}
    for metric in metrics:
        shift = metric_shift_pct(
            baseline_summary=baseline_summary,
            variant_summary=variant_summary,
            metric=metric,
        )
        if shift is not None:
            shifts[metric] = float(shift)
    return shifts


def _composite_shift_pct(metric_shift_map: dict[str, float]) -> float | None:
    if not metric_shift_map:
        return None
    return float(median(metric_shift_map.values()))


def classify_diversity_status(
    composite_shift_pct: float | None,
    *,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
) -> str:
    """Classify one comparison severity from the composite shift."""

    warn_threshold_pct, fail_threshold_pct = validate_diversity_thresholds(
        warn_threshold_pct=warn_threshold_pct,
        fail_threshold_pct=fail_threshold_pct,
    )
    if composite_shift_pct is None:
        return "insufficient_metrics"
    if float(composite_shift_pct) >= float(fail_threshold_pct):
        return "fail"
    if float(composite_shift_pct) >= float(warn_threshold_pct):
        return "warn"
    return "pass"


def compare_coverage_summaries(
    *,
    baseline_summary: dict[str, Any],
    variant_summary: dict[str, Any],
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    include_structural_summary: bool = False,
) -> dict[str, Any]:
    """Compare one variant coverage summary against a baseline summary."""

    metric_shift_map = _metric_shift_map(
        baseline_summary=baseline_summary,
        variant_summary=variant_summary,
        metrics=CORE_DIVERSITY_METRICS,
    )
    composite_shift_pct = _composite_shift_pct(metric_shift_map)
    comparison = {
        "diversity_metric_shift_pct": metric_shift_map,
        "diversity_composite_shift_pct": composite_shift_pct,
        "diversity_status": classify_diversity_status(
            composite_shift_pct,
            warn_threshold_pct=warn_threshold_pct,
            fail_threshold_pct=fail_threshold_pct,
        ),
    }
    if include_structural_summary:
        structural_metric_shift_map = _metric_shift_map(
            baseline_summary=baseline_summary,
            variant_summary=variant_summary,
            metrics=STRUCTURAL_DIVERSITY_METRICS,
        )
        comparison["structural_diversity_metric_shift_pct"] = structural_metric_shift_map
        comparison["structural_diversity_composite_shift_pct"] = _composite_shift_pct(
            structural_metric_shift_map
        )
    return comparison


def build_comparison_record(
    *,
    baseline_entry: dict[str, Any],
    variant_entry: dict[str, Any],
    warn_threshold_pct: float,
    fail_threshold_pct: float,
) -> dict[str, Any]:
    """Build one report comparison record for a variant entry."""

    comparison = compare_coverage_summaries(
        baseline_summary=baseline_entry["coverage_summary"],
        variant_summary=variant_entry["coverage_summary"],
        warn_threshold_pct=warn_threshold_pct,
        fail_threshold_pct=fail_threshold_pct,
    )
    comparison["variant_label"] = variant_entry["label"]
    comparison["datasets_per_minute_delta_pct"] = _delta_pct(
        variant_entry.get("datasets_per_minute"),
        baseline_entry.get("datasets_per_minute"),
    )
    comparison["filter_accepted_datasets_per_minute_delta_pct"] = _delta_pct(
        variant_entry.get("filter_accepted_datasets_per_minute"),
        baseline_entry.get("filter_accepted_datasets_per_minute"),
    )
    return comparison


def summarize_comparison_status(comparisons: list[dict[str, Any]]) -> str:
    """Collapse per-variant severities into one audit-level status."""

    statuses = {str(item.get("diversity_status", "insufficient_metrics")) for item in comparisons}
    if "fail" in statuses:
        return "fail"
    if "insufficient_metrics" in statuses:
        return "insufficient_metrics"
    if "warn" in statuses:
        return "warn"
    if "pass" in statuses:
        return "pass"
    return "insufficient_metrics"
