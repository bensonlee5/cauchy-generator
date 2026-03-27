"""Run-level coverage aggregation and artifact writers."""

from __future__ import annotations

import datetime as dt
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dagzoo.config import GeneratorConfig
from dagzoo.math import sanitize_json as _sanitize_json
from dagzoo.types import DatasetBundle

from .metrics import extract_dataset_metrics
from .types import DatasetMetrics

_DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_DEFAULT_MAX_VALUES_PER_METRIC = 50_000
_STEERING_MIXTURE_COMPONENTS = ("gaussian", "laplace", "student_t")
_METRIC_FIELD_NAMES = (
    "n_rows",
    "n_features",
    "n_classes",
    "categorical_ratio",
    "pearson_abs_mean",
    "pearson_abs_max",
    "class_entropy",
    "majority_minority_ratio",
    "snr_proxy_db",
    "cat_cardinality_mean",
)


@dataclass(slots=True)
class CoverageAggregationConfig:
    """Configuration for run-level coverage aggregation."""

    include_spearman: bool = False
    histogram_bins: int = 10
    quantiles: tuple[float, ...] = _DEFAULT_QUANTILES
    underrepresented_threshold: float = 0.5
    max_values_per_metric: int | None = _DEFAULT_MAX_VALUES_PER_METRIC
    target_bands: dict[str, tuple[float, float]] = field(default_factory=dict)
    steering_config: GeneratorConfig | None = None


@dataclass(slots=True)
class _ValueAccumulator:
    count: int = 0
    total: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf

    def update(self, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        as_float = float(value)
        if not math.isfinite(as_float):
            return
        self.count += 1
        self.total += as_float
        self.min_value = min(self.min_value, as_float)
        self.max_value = max(self.max_value, as_float)

    def finalize(self) -> dict[str, Any]:
        if self.count <= 0:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {
            "count": int(self.count),
            "min": float(self.min_value),
            "max": float(self.max_value),
            "mean": float(self.total / self.count),
        }


@dataclass(slots=True)
class _MetricAccumulator:
    count: int = 0
    missing_count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf
    values: list[float] = field(default_factory=list)
    sample_limit: int | None = None
    rng_seed: int = 0
    _seen_values: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)

    def update(self, value: float | int | None) -> None:
        if value is None:
            self.missing_count += 1
            return
        as_float = float(value)
        if not math.isfinite(as_float):
            self.missing_count += 1
            return
        self.count += 1
        self.total += as_float
        self.total_sq += as_float * as_float
        self.min_value = min(self.min_value, as_float)
        self.max_value = max(self.max_value, as_float)
        self._seen_values += 1
        if self.sample_limit is None:
            self.values.append(as_float)
            return
        if len(self.values) < self.sample_limit:
            self.values.append(as_float)
            return
        # Deterministic reservoir sampling bounds memory while preserving an unbiased sample.
        replace_idx = self._rng.randint(0, self._seen_values - 1)
        if replace_idx < self.sample_limit:
            self.values[replace_idx] = as_float

    def finalize(
        self,
        *,
        quantiles: tuple[float, ...],
        histogram_bins: int,
        underrepresented_threshold: float,
        target_band: tuple[float, float] | None,
    ) -> dict[str, Any]:
        if self.count <= 0:
            return {
                "count": 0,
                "missing_count": int(self.missing_count),
                "observed_min": None,
                "observed_max": None,
                "mean": None,
                "std": None,
                "sampled_count": 0,
                "sampled_fraction": 0.0,
                "quantiles": {f"p{int(round(q * 100)):02d}": None for q in quantiles},
                "histogram": {
                    "num_bins": int(histogram_bins),
                    "covered_bins": 0,
                    "coverage_ratio": 0.0,
                    "bins": [],
                },
                "underrepresented_bins": [],
                "target_band": _target_band_payload(target_band),
            }

        values = np.asarray(self.values, dtype=np.float64)
        mean = float(self.total / self.count)
        variance = max(0.0, float((self.total_sq / self.count) - (mean * mean)))
        std = math.sqrt(variance)
        sampled_count = int(values.size)
        quantile_map = {
            f"p{int(round(q * 100)):02d}": float(np.quantile(values, q)) for q in quantiles
        }

        histogram = _build_histogram(
            values,
            bins=histogram_bins,
            value_range=target_band,
        )
        underrepresented_bins: list[dict[str, Any]] = []
        target_payload = _target_band_payload(target_band)
        if target_band is not None:
            underrepresented_bins, in_target_count, in_target_fraction = _underrepresented_bins(
                values=values,
                bins=histogram["bins"],
                target_band=target_band,
                underrepresented_threshold=underrepresented_threshold,
            )
            if target_payload is not None:
                target_payload["in_target_count"] = in_target_count
                target_payload["in_target_fraction"] = in_target_fraction

        return {
            "count": int(self.count),
            "missing_count": int(self.missing_count),
            "observed_min": float(self.min_value),
            "observed_max": float(self.max_value),
            "mean": mean,
            "std": float(std),
            "sampled_count": sampled_count,
            "sampled_fraction": float(sampled_count / self.count) if self.count > 0 else 0.0,
            "quantiles": quantile_map,
            "histogram": histogram,
            "underrepresented_bins": underrepresented_bins,
            "target_band": target_payload,
        }


class CoverageAggregator:
    """Streaming aggregator for run-level dataset diagnostics coverage."""

    def __init__(self, config: CoverageAggregationConfig | None = None) -> None:
        cfg = config or CoverageAggregationConfig()
        self._config = CoverageAggregationConfig(
            include_spearman=bool(cfg.include_spearman),
            histogram_bins=max(1, int(cfg.histogram_bins)),
            quantiles=_normalize_quantiles(cfg.quantiles),
            underrepresented_threshold=max(0.0, float(cfg.underrepresented_threshold)),
            max_values_per_metric=_normalize_max_values_per_metric(cfg.max_values_per_metric),
            target_bands=_normalize_target_bands(cfg.target_bands),
            steering_config=cfg.steering_config,
        )
        self._num_datasets = 0
        self._task_counts: dict[str, int] = {}
        self._mechanism_family_bundles_with_metadata = 0
        self._mechanism_family_total_function_plans = 0
        self._mechanism_family_sampled_counts: dict[str, int] = {}
        self._mechanism_family_dataset_presence: dict[str, int] = {}
        self._mechanism_variant_sampled_counts: dict[str, int] = {}
        self._mechanism_variant_dataset_presence: dict[str, int] = {}
        self._metrics = {
            name: _MetricAccumulator(
                sample_limit=self._config.max_values_per_metric,
                rng_seed=idx + 1,
            )
            for idx, name in enumerate(_METRIC_FIELD_NAMES)
        }

    @property
    def num_datasets(self) -> int:
        """Return number of ingested datasets."""

        return int(self._num_datasets)

    def update_bundle(self, bundle: DatasetBundle) -> DatasetMetrics:
        """Extract metrics from one bundle and update aggregator state."""

        metrics = extract_dataset_metrics(bundle, include_spearman=self._config.include_spearman)
        self.update_metrics(metrics)
        self._update_mechanism_families(bundle)
        return metrics

    def update_metrics(self, metrics: DatasetMetrics) -> None:
        """Update aggregator state from one metrics payload."""

        self._num_datasets += 1
        self._task_counts[metrics.task] = self._task_counts.get(metrics.task, 0) + 1
        for metric_name in _METRIC_FIELD_NAMES:
            value = getattr(metrics, metric_name)
            self._metrics[metric_name].update(value)

    def build_summary(self) -> dict[str, Any]:
        """Build finalized run-level coverage summary."""

        summary_metrics: dict[str, Any] = {}
        for metric_name, accumulator in self._metrics.items():
            summary_metrics[metric_name] = accumulator.finalize(
                quantiles=self._config.quantiles,
                histogram_bins=self._config.histogram_bins,
                underrepresented_threshold=self._config.underrepresented_threshold,
                target_band=self._config.target_bands.get(metric_name),
            )

        return {
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "num_datasets": int(self._num_datasets),
            "task_counts": dict(sorted(self._task_counts.items())),
            "histogram_bins": int(self._config.histogram_bins),
            "quantiles": list(self._config.quantiles),
            "max_values_per_metric": self._config.max_values_per_metric,
            "mechanism_family_summary": self._build_mechanism_family_summary(),
            "metrics": summary_metrics,
        }

    def _update_mechanism_families(self, bundle: DatasetBundle) -> None:
        metadata = bundle.metadata.get("mechanism_families")
        if not isinstance(metadata, dict):
            return
        sampled_family_counts = metadata.get("sampled_family_counts")
        if not isinstance(sampled_family_counts, dict):
            return

        normalized_counts: dict[str, int] = {}
        for raw_family, raw_count in sampled_family_counts.items():
            if isinstance(raw_family, str) and not isinstance(raw_count, bool):
                if isinstance(raw_count, (int, float)) and math.isfinite(float(raw_count)):
                    count = max(0, int(raw_count))
                    if count > 0:
                        normalized_counts[str(raw_family)] = count
        sampled_variant_counts = metadata.get("sampled_variant_counts")
        normalized_variant_counts: dict[str, int] = {}
        if isinstance(sampled_variant_counts, dict):
            for raw_label, raw_count in sampled_variant_counts.items():
                if isinstance(raw_label, str) and not isinstance(raw_count, bool):
                    if isinstance(raw_count, (int, float)) and math.isfinite(float(raw_count)):
                        count = max(0, int(raw_count))
                        if count > 0:
                            normalized_variant_counts[str(raw_label)] = count
        self._mechanism_family_bundles_with_metadata += 1
        for family, count in normalized_counts.items():
            self._mechanism_family_sampled_counts[family] = (
                self._mechanism_family_sampled_counts.get(family, 0) + int(count)
            )
            self._mechanism_family_dataset_presence[family] = (
                self._mechanism_family_dataset_presence.get(family, 0) + 1
            )
        for label, count in normalized_variant_counts.items():
            self._mechanism_variant_sampled_counts[label] = (
                self._mechanism_variant_sampled_counts.get(label, 0) + int(count)
            )
            self._mechanism_variant_dataset_presence[label] = (
                self._mechanism_variant_dataset_presence.get(label, 0) + 1
            )

        total_function_plans = metadata.get("total_function_plans")
        if (
            not isinstance(total_function_plans, bool)
            and isinstance(total_function_plans, (int, float))
            and math.isfinite(float(total_function_plans))
        ):
            self._mechanism_family_total_function_plans += max(0, int(total_function_plans))

    def _build_mechanism_family_summary(self) -> dict[str, Any]:
        bundles_with_metadata = int(self._mechanism_family_bundles_with_metadata)
        num_datasets = int(self._num_datasets)
        metadata_denominator = num_datasets if num_datasets > 0 else 0
        mean_total_function_plans = (
            float(self._mechanism_family_total_function_plans / bundles_with_metadata)
            if bundles_with_metadata > 0
            else 0.0
        )
        return {
            "metadata_coverage_rate": (
                float(bundles_with_metadata / metadata_denominator)
                if metadata_denominator > 0
                else 0.0
            ),
            "bundles_with_metadata": bundles_with_metadata,
            "sampled_family_counts": dict(sorted(self._mechanism_family_sampled_counts.items())),
            "dataset_presence_rate_by_family": {
                family: (float(count / metadata_denominator) if metadata_denominator > 0 else 0.0)
                for family, count in sorted(self._mechanism_family_dataset_presence.items())
            },
            "sampled_variant_counts": dict(sorted(self._mechanism_variant_sampled_counts.items())),
            "dataset_presence_rate_by_variant": {
                label: (float(count / metadata_denominator) if metadata_denominator > 0 else 0.0)
                for label, count in sorted(self._mechanism_variant_dataset_presence.items())
            },
            "mean_total_function_plans": float(mean_total_function_plans),
        }


def write_coverage_summary_json(summary: dict[str, Any], out_path: str | Path) -> Path:
    """Write run-level coverage summary JSON artifact."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_json(summary), f, indent=2, sort_keys=True, allow_nan=False)
    return path


def write_coverage_summary_markdown(summary: dict[str, Any], out_path: str | Path) -> Path:
    """Write a concise markdown artifact for run-level coverage."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Coverage Summary",
        "",
        f"- Generated at: `{summary.get('generated_at', '-')}`",
        f"- Num datasets: `{summary.get('num_datasets', 0)}`",
        f"- Histogram bins: `{summary.get('histogram_bins', 0)}`",
        f"- Quantiles: `{summary.get('quantiles', [])}`",
        f"- Max sampled values/metric: `{summary.get('max_values_per_metric', '-')}`",
    ]
    task_counts = summary.get("task_counts", {})
    if isinstance(task_counts, dict) and task_counts:
        task_parts = [f"{name}={count}" for name, count in sorted(task_counts.items())]
        lines.append(f"- Task counts: `{', '.join(task_parts)}`")
    mechanism_family_summary = summary.get("mechanism_family_summary", {})
    if isinstance(mechanism_family_summary, dict):
        lines.extend(
            [
                f"- Mechanism metadata coverage: `{_fmt(mechanism_family_summary.get('metadata_coverage_rate'))}`",
                f"- Bundles with mechanism metadata: `{_fmt(mechanism_family_summary.get('bundles_with_metadata'), digits=0)}`",
                f"- Mean total function plans: `{_fmt(mechanism_family_summary.get('mean_total_function_plans'))}`",
            ]
        )
    lines.extend(["", "## Metrics", ""])
    lines.append("| Metric | Min | Max | p50 | Covered Bins | Underrepresented Bins |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    metrics_payload = summary.get("metrics", {})
    if isinstance(metrics_payload, dict):
        for metric_name in sorted(metrics_payload):
            metric = metrics_payload[metric_name]
            if not isinstance(metric, dict):
                continue
            histogram = metric.get("histogram", {})
            under_bins = metric.get("underrepresented_bins", [])
            lines.append(
                "| "
                f"{metric_name} | "
                f"{_fmt(metric.get('observed_min'))} | "
                f"{_fmt(metric.get('observed_max'))} | "
                f"{_fmt((metric.get('quantiles') or {}).get('p50'))} | "
                f"{_fmt((histogram or {}).get('covered_bins'), digits=0)} | "
                f"{_fmt(len(under_bins), digits=0)} |"
            )

    if isinstance(mechanism_family_summary, dict):
        lines.extend(["", "## Mechanism Families", ""])
        sampled_family_counts = mechanism_family_summary.get("sampled_family_counts", {})
        dataset_presence = mechanism_family_summary.get("dataset_presence_rate_by_family", {})
        if isinstance(sampled_family_counts, dict) and sampled_family_counts:
            lines.append("| Family | Sampled Count | Dataset Presence Rate |")
            lines.append("|---|---:|---:|")
            for family in sorted(sampled_family_counts):
                lines.append(
                    "| "
                    f"{family} | "
                    f"{_fmt(sampled_family_counts.get(family), digits=0)} | "
                    f"{_fmt((dataset_presence or {}).get(family))} |"
                )
        else:
            lines.append("- No realized mechanism-family metadata was observed.")
        sampled_variant_counts = mechanism_family_summary.get("sampled_variant_counts", {})
        variant_presence = mechanism_family_summary.get("dataset_presence_rate_by_variant", {})
        if isinstance(sampled_variant_counts, dict) and sampled_variant_counts:
            lines.extend(["", "### Mechanism Variants", ""])
            lines.append("| Variant | Sampled Count | Dataset Presence Rate |")
            lines.append("|---|---:|---:|")
            for label in sorted(sampled_variant_counts):
                lines.append(
                    "| "
                    f"{label} | "
                    f"{_fmt(sampled_variant_counts.get(label), digits=0)} | "
                    f"{_fmt((variant_presence or {}).get(label))} |"
                )

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path


def _normalize_quantiles(quantiles: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    raw = list(quantiles) if quantiles else list(_DEFAULT_QUANTILES)
    normalized: list[float] = []
    for q in raw:
        value = float(q)
        if 0.0 <= value <= 1.0:
            normalized.append(value)
    if not normalized:
        return _DEFAULT_QUANTILES
    return tuple(sorted(set(normalized)))


def _normalize_max_values_per_metric(raw: object) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return _DEFAULT_MAX_VALUES_PER_METRIC
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float):
        if not math.isfinite(raw):
            return _DEFAULT_MAX_VALUES_PER_METRIC
        value = int(raw)
    elif isinstance(raw, str):
        try:
            value = int(raw)
        except ValueError:
            return _DEFAULT_MAX_VALUES_PER_METRIC
    else:
        return _DEFAULT_MAX_VALUES_PER_METRIC
    if value <= 0:
        return None
    return value


def _normalize_target_bands(
    target_bands: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    normalized: dict[str, tuple[float, float]] = {}
    for metric_name, band in target_bands.items():
        if not isinstance(metric_name, str):
            continue
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            continue
        lo = float(band[0])
        hi = float(band[1])
        if not math.isfinite(lo) or not math.isfinite(hi):
            continue
        if lo <= hi:
            normalized[metric_name] = (lo, hi)
        else:
            normalized[metric_name] = (hi, lo)
    return normalized


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[str(key)] = counts.get(str(key), 0) + 1


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    if not math.isfinite(as_float):
        return None
    return int(as_float)


def _min_optional_int(current: int | None, candidate: int) -> int:
    if current is None:
        return int(candidate)
    return min(int(current), int(candidate))


def _max_optional_int(current: int | None, candidate: int) -> int:
    if current is None:
        return int(candidate)
    return max(int(current), int(candidate))


def _bundle_config_missing_rate(bundle: DatasetBundle) -> float | None:
    config_payload = bundle.metadata.get("config")
    if not isinstance(config_payload, dict):
        return None
    dataset_payload = config_payload.get("dataset")
    if not isinstance(dataset_payload, dict):
        return None
    return _coerce_optional_float(dataset_payload.get("missing_rate"))


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    if not math.isfinite(as_float):
        return None
    return as_float


def _normalized_noise_weights(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, float] = {}
    for component in _STEERING_MIXTURE_COMPONENTS:
        value = raw.get(component)
        as_float = _coerce_optional_float(value)
        if as_float is not None:
            normalized[component] = as_float
    return normalized


def _finalize_value_accumulator_map(
    accumulators: dict[str, _ValueAccumulator],
) -> dict[str, dict[str, Any]]:
    return {
        component: accumulator.finalize() for component, accumulator in sorted(accumulators.items())
    }


def _build_resolution_checks_payload(
    *,
    datasets_checked: int,
    datasets_matching: int,
    mismatch_counts: dict[str, int],
    mismatched_dataset_indices: list[int],
) -> dict[str, Any]:
    mismatched = max(0, int(datasets_checked) - int(datasets_matching))
    return {
        "datasets_checked": int(datasets_checked),
        "datasets_matching": int(datasets_matching),
        "datasets_mismatched": int(mismatched),
        "match_rate": (
            float(datasets_matching / datasets_checked) if datasets_checked > 0 else None
        ),
        "mismatch_counts": dict(sorted(mismatch_counts.items())),
        "mismatched_dataset_indices": [int(index) for index in mismatched_dataset_indices],
    }


def _steering_resolution_mismatches(
    bundle: DatasetBundle,
    expected_config: GeneratorConfig,
    expected_shift,
) -> list[str]:
    mismatches: list[str] = []
    config_payload = bundle.metadata.get("config")
    config_payload = config_payload if isinstance(config_payload, dict) else {}
    dataset_payload = config_payload.get("dataset")
    dataset_payload = dataset_payload if isinstance(dataset_payload, dict) else {}

    observed_missing_rate = _coerce_optional_float(dataset_payload.get("missing_rate"))
    if not _float_matches(observed_missing_rate, float(expected_config.dataset.missing_rate)):
        mismatches.append("config.dataset.missing_rate")
    observed_mechanism = dataset_payload.get("missing_mechanism")
    if str(observed_mechanism) != str(expected_config.dataset.missing_mechanism):
        mismatches.append("config.dataset.missing_mechanism")

    shift_payload = bundle.metadata.get("shift")
    shift_payload = shift_payload if isinstance(shift_payload, dict) else {}
    if str(shift_payload.get("mode")) != str(expected_config.shift.mode):
        mismatches.append("metadata.shift.mode")
    if not _float_matches(shift_payload.get("graph_scale"), float(expected_shift.graph_scale)):
        mismatches.append("metadata.shift.graph_scale")
    if not _float_matches(
        shift_payload.get("variance_scale"),
        float(expected_shift.variance_scale),
    ):
        mismatches.append("metadata.shift.variance_scale")
    if not _float_matches(
        shift_payload.get("mechanism_logit_tilt"),
        float(expected_shift.mechanism_logit_tilt),
    ):
        mismatches.append("metadata.shift.mechanism_logit_tilt")

    noise_payload = bundle.metadata.get("noise_distribution")
    noise_payload = noise_payload if isinstance(noise_payload, dict) else {}
    if str(noise_payload.get("family_requested")) != str(expected_config.noise.family):
        mismatches.append("metadata.noise_distribution.family_requested")
    if _normalized_noise_weights(noise_payload.get("mixture_weights")) != _normalized_noise_weights(
        expected_config.noise.mixture_weights
    ):
        mismatches.append("metadata.noise_distribution.mixture_weights")
    return mismatches


def _float_matches(lhs: object, rhs: object, *, atol: float = 1e-9, rtol: float = 1e-9) -> bool:
    lhs_float = _coerce_optional_float(lhs)
    rhs_float = _coerce_optional_float(rhs)
    if lhs_float is None or rhs_float is None:
        return lhs_float is None and rhs_float is None
    return math.isclose(lhs_float, rhs_float, abs_tol=atol, rel_tol=rtol)


def _fmt_counts(payload: object) -> str:
    if not isinstance(payload, dict) or not payload:
        return "-"
    parts = []
    for key in sorted(payload):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(f"{key}={int(value)}")
    return ", ".join(parts) if parts else "-"


def _fmt_int_range(payload: object) -> str:
    if not isinstance(payload, dict):
        return "-"
    lo = _coerce_optional_int(payload.get("min"))
    hi = _coerce_optional_int(payload.get("max"))
    if lo is None or hi is None:
        return "-"
    return f"{lo}-{hi}"


def _fmt_value_range(payload: object) -> str:
    if not isinstance(payload, dict):
        return "-"
    lo = _coerce_optional_float(payload.get("min"))
    hi = _coerce_optional_float(payload.get("max"))
    if lo is None or hi is None:
        return "-"
    return f"{lo:.3f}-{hi:.3f}"


def _fmt_scalar_summary(payload: object) -> str:
    if not isinstance(payload, dict):
        return "-"
    mean = _coerce_optional_float(payload.get("mean"))
    lo = _coerce_optional_float(payload.get("min"))
    hi = _coerce_optional_float(payload.get("max"))
    if mean is None or lo is None or hi is None:
        return "-"
    return f"{mean:.3f} ({lo:.3f}-{hi:.3f})"


def _build_histogram(
    values: np.ndarray,
    *,
    bins: int,
    value_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    if values.size <= 0:
        return {
            "num_bins": int(bins),
            "covered_bins": 0,
            "coverage_ratio": 0.0,
            "bins": [],
        }
    if value_range is None:
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        if v_min == v_max:
            lo = v_min - 0.5
            hi = v_max + 0.5
        else:
            lo = v_min
            hi = v_max
    else:
        lo = float(value_range[0])
        hi = float(value_range[1])
        if lo > hi:
            lo, hi = hi, lo
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = -0.5
        hi = 0.5
    elif not hi > lo or np.isclose(lo, hi, rtol=1e-12, atol=1e-12):
        center = float((lo + hi) / 2.0)
        span = max(0.5, abs(center) * 1e-6)
        lo = center - span
        hi = center + span
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    total = float(np.sum(counts))
    bins_payload: list[dict[str, Any]] = []
    for i in range(bins):
        count = int(counts[i])
        bins_payload.append(
            {
                "index": int(i),
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "count": count,
                "fraction": float(count / total) if total > 0 else 0.0,
            }
        )
    covered_bins = int(sum(1 for b in bins_payload if b["count"] > 0))
    return {
        "num_bins": int(bins),
        "covered_bins": covered_bins,
        "coverage_ratio": float(covered_bins / bins) if bins > 0 else 0.0,
        "bins": bins_payload,
    }


def _underrepresented_bins(
    values: np.ndarray,
    bins: list[dict[str, Any]],
    *,
    target_band: tuple[float, float],
    underrepresented_threshold: float,
) -> tuple[list[dict[str, Any]], int, float]:
    lo, hi = target_band
    in_target_mask = (values >= lo) & (values <= hi)
    in_target_count = int(np.sum(in_target_mask))
    total_count = int(values.size)
    in_target_fraction = float(in_target_count / total_count) if total_count > 0 else 0.0

    overlapping = [
        b
        for b in bins
        if isinstance(b.get("lower"), (int, float))
        and isinstance(b.get("upper"), (int, float))
        and not (float(b["upper"]) <= lo or float(b["lower"]) >= hi)
    ]
    if not overlapping:
        return [], in_target_count, in_target_fraction
    if in_target_count == 0:
        return overlapping, in_target_count, in_target_fraction

    edges = [float(bins[0]["lower"])] + [float(b["upper"]) for b in bins]
    in_target_values = values[in_target_mask]
    in_target_bin_counts, _ = np.histogram(
        in_target_values, bins=np.asarray(edges, dtype=np.float64)
    )
    expected_count_per_bin = float(in_target_count / len(overlapping))
    threshold_count = expected_count_per_bin * underrepresented_threshold

    underrepresented: list[dict[str, Any]] = []
    for b in overlapping:
        idx = int(b.get("index", -1))
        if idx < 0 or idx >= int(in_target_bin_counts.size):
            continue
        in_target_bin_count = int(in_target_bin_counts[idx])
        if float(in_target_bin_count) < threshold_count:
            annotated = dict(b)
            annotated["in_target_count"] = in_target_bin_count
            annotated["in_target_fraction"] = (
                float(in_target_bin_count / in_target_count) if in_target_count > 0 else 0.0
            )
            underrepresented.append(annotated)
    return underrepresented, in_target_count, in_target_fraction


def _target_band_payload(target_band: tuple[float, float] | None) -> dict[str, float] | None:
    if target_band is None:
        return None
    return {"min": float(target_band[0]), "max": float(target_band[1])}


def _fmt(value: Any, digits: int = 3) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    as_float = float(value)
    if not math.isfinite(as_float):
        return "-"
    return f"{as_float:.{digits}f}"
