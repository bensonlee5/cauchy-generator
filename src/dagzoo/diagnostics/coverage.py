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
    "graph_edge_density",
    "graph_indegree_std",
    "graph_outdegree_std",
    "graph_depth_ratio",
    "graph_target_depth_ratio",
    "graph_reachability_ratio",
    "graph_ancestor_overlap_mean",
    "graph_target_ancestor_fraction",
    "mechanism_family_cooccurrence_ratio",
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

    def merge_stats(
        self,
        *,
        count: object,
        total: object,
        min_value: object,
        max_value: object,
    ) -> None:
        count_value = _coerce_optional_int(count)
        total_value = _coerce_optional_float(total)
        min_stat = _coerce_optional_float(min_value)
        max_stat = _coerce_optional_float(max_value)
        if (
            count_value is None
            or count_value <= 0
            or total_value is None
            or min_stat is None
            or max_stat is None
        ):
            return
        self.count += int(count_value)
        self.total += float(total_value)
        self.min_value = min(self.min_value, float(min_stat))
        self.max_value = max(self.max_value, float(max_stat))

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
        self._parity_surface_bundles_with_metadata = 0
        self._parity_converter_method_counts: dict[str, int] = {}
        self._parity_converter_variant_counts: dict[str, int] = {}
        self._parity_converter_method_variant_counts: dict[str, int] = {}
        self._parity_gp_variant_counts: dict[str, int] = {}
        self._parity_kernel_signed_counts: dict[str, int] = {}
        self._parity_matrix_kind_counts: dict[str, int] = {}
        self._parity_activation_base_kind_counts: dict[str, int] = {}
        self._parity_root_base_kind_counts: dict[str, int] = {}
        self._parity_source_kind_counts: dict[str, int] = {}
        self._parity_combine_kind_counts: dict[str, int] = {}
        self._parity_aggregation_kind_counts: dict[str, int] = {}
        self._parity_parent_arity_counts: dict[str, int] = {}
        self._parity_source_shape_policy_counts: dict[str, int] = {}
        self._parity_kernel_gamma = _ValueAccumulator()
        self._parity_categorical_cardinality = _ValueAccumulator()
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
        self._update_parity_surface(bundle)
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
            "parity_surface_summary": self._build_parity_surface_summary(),
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

    def _update_count_summary(
        self,
        payload: object,
        *,
        destination: dict[str, int],
    ) -> None:
        if not isinstance(payload, dict):
            return
        for raw_label, raw_count in payload.items():
            if not isinstance(raw_label, str) or isinstance(raw_count, bool):
                continue
            if not isinstance(raw_count, (int, float)) or not math.isfinite(float(raw_count)):
                continue
            count = max(0, int(raw_count))
            if count > 0:
                destination[str(raw_label)] = int(destination.get(str(raw_label), 0)) + int(count)

    def _update_parity_surface(self, bundle: DatasetBundle) -> None:
        metadata = bundle.metadata.get("parity_surface")
        if not isinstance(metadata, dict):
            return
        self._parity_surface_bundles_with_metadata += 1
        self._update_count_summary(
            metadata.get("converter_method_counts"),
            destination=self._parity_converter_method_counts,
        )
        self._update_count_summary(
            metadata.get("converter_variant_counts"),
            destination=self._parity_converter_variant_counts,
        )
        self._update_count_summary(
            metadata.get("converter_method_variant_counts"),
            destination=self._parity_converter_method_variant_counts,
        )
        self._update_count_summary(
            metadata.get("gp_variant_counts"),
            destination=self._parity_gp_variant_counts,
        )
        self._update_count_summary(
            metadata.get("kernel_signed_counts"),
            destination=self._parity_kernel_signed_counts,
        )
        self._update_count_summary(
            metadata.get("matrix_kind_counts"),
            destination=self._parity_matrix_kind_counts,
        )
        self._update_count_summary(
            metadata.get("activation_base_kind_counts"),
            destination=self._parity_activation_base_kind_counts,
        )
        self._update_count_summary(
            metadata.get("root_base_kind_counts"),
            destination=self._parity_root_base_kind_counts,
        )
        self._update_count_summary(
            metadata.get("source_kind_counts"),
            destination=self._parity_source_kind_counts,
        )
        self._update_count_summary(
            metadata.get("combine_kind_counts"),
            destination=self._parity_combine_kind_counts,
        )
        self._update_count_summary(
            metadata.get("aggregation_kind_counts"),
            destination=self._parity_aggregation_kind_counts,
        )
        self._update_count_summary(
            metadata.get("parent_arity_counts"),
            destination=self._parity_parent_arity_counts,
        )
        self._update_count_summary(
            metadata.get("source_shape_policy_counts"),
            destination=self._parity_source_shape_policy_counts,
        )
        kernel_gamma = metadata.get("kernel_gamma")
        if isinstance(kernel_gamma, dict):
            self._parity_kernel_gamma.merge_stats(
                count=kernel_gamma.get("count"),
                total=kernel_gamma.get("total"),
                min_value=kernel_gamma.get("min"),
                max_value=kernel_gamma.get("max"),
            )
        categorical_cardinality = metadata.get("categorical_cardinality")
        if isinstance(categorical_cardinality, dict):
            self._parity_categorical_cardinality.merge_stats(
                count=categorical_cardinality.get("count"),
                total=categorical_cardinality.get("total"),
                min_value=categorical_cardinality.get("min"),
                max_value=categorical_cardinality.get("max"),
            )

    def _build_parity_surface_summary(self) -> dict[str, Any]:
        bundles_with_metadata = int(self._parity_surface_bundles_with_metadata)
        num_datasets = int(self._num_datasets)
        metadata_denominator = num_datasets if num_datasets > 0 else 0
        return {
            "schema_name": "dagzoo_parity_surface_summary",
            "schema_version": 1,
            "metadata_coverage_rate": (
                float(bundles_with_metadata / metadata_denominator)
                if metadata_denominator > 0
                else 0.0
            ),
            "bundles_with_metadata": bundles_with_metadata,
            "converter_method_counts": dict(sorted(self._parity_converter_method_counts.items())),
            "converter_variant_counts": dict(sorted(self._parity_converter_variant_counts.items())),
            "converter_method_variant_counts": dict(
                sorted(self._parity_converter_method_variant_counts.items())
            ),
            "gp_variant_counts": dict(sorted(self._parity_gp_variant_counts.items())),
            "kernel_gamma": self._parity_kernel_gamma.finalize(),
            "kernel_signed_counts": dict(sorted(self._parity_kernel_signed_counts.items())),
            "matrix_kind_counts": dict(sorted(self._parity_matrix_kind_counts.items())),
            "activation_base_kind_counts": dict(
                sorted(self._parity_activation_base_kind_counts.items())
            ),
            "root_base_kind_counts": dict(sorted(self._parity_root_base_kind_counts.items())),
            "source_kind_counts": dict(sorted(self._parity_source_kind_counts.items())),
            "combine_kind_counts": dict(sorted(self._parity_combine_kind_counts.items())),
            "aggregation_kind_counts": dict(sorted(self._parity_aggregation_kind_counts.items())),
            "parent_arity_counts": dict(sorted(self._parity_parent_arity_counts.items())),
            "source_shape_policy_counts": dict(
                sorted(self._parity_source_shape_policy_counts.items())
            ),
            "categorical_cardinality": self._parity_categorical_cardinality.finalize(),
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
    parity_surface_summary = summary.get("parity_surface_summary", {})
    if isinstance(mechanism_family_summary, dict):
        lines.extend(
            [
                f"- Mechanism metadata coverage: `{_fmt(mechanism_family_summary.get('metadata_coverage_rate'))}`",
                f"- Bundles with mechanism metadata: `{_fmt(mechanism_family_summary.get('bundles_with_metadata'), digits=0)}`",
                f"- Mean total function plans: `{_fmt(mechanism_family_summary.get('mean_total_function_plans'))}`",
            ]
        )
    if isinstance(parity_surface_summary, dict):
        lines.extend(
            [
                f"- Parity-surface metadata coverage: `{_fmt(parity_surface_summary.get('metadata_coverage_rate'))}`",
                f"- Bundles with parity-surface metadata: `{_fmt(parity_surface_summary.get('bundles_with_metadata'), digits=0)}`",
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

    if isinstance(parity_surface_summary, dict):
        lines.extend(["", "## Parity Surface", ""])
        lines.extend(
            _parity_surface_markdown_lines(
                "Converter methods",
                parity_surface_summary.get("converter_method_counts"),
            )
        )
        lines.extend(
            _parity_surface_markdown_lines(
                "Converter variants",
                parity_surface_summary.get("converter_variant_counts"),
            )
        )
        lines.extend(
            _parity_surface_markdown_lines(
                "Converter method+variant",
                parity_surface_summary.get("converter_method_variant_counts"),
            )
        )
        lines.extend(
            _parity_surface_markdown_lines(
                "GP variants",
                parity_surface_summary.get("gp_variant_counts"),
            )
        )
        lines.extend(
            _parity_surface_markdown_lines(
                "Kernel signed counts",
                parity_surface_summary.get("kernel_signed_counts"),
            )
        )
        lines.extend(
            _parity_surface_markdown_lines(
                "Matrix kinds",
                parity_surface_summary.get("matrix_kind_counts"),
            )
        )
        lines.extend(
            _parity_surface_markdown_lines(
                "Activation-matrix base kinds",
                parity_surface_summary.get("activation_base_kind_counts"),
            )
        )
        lines.extend(
            _parity_surface_markdown_lines(
                "Root base kinds",
                parity_surface_summary.get("root_base_kind_counts"),
            )
        )
        lines.extend(
            _parity_surface_markdown_lines(
                "Source kinds",
                parity_surface_summary.get("source_kind_counts"),
            )
        )
        lines.extend(
            _parity_surface_markdown_lines(
                "Combine kinds",
                parity_surface_summary.get("combine_kind_counts"),
            )
        )
        lines.extend(
            _parity_surface_markdown_lines(
                "Aggregation kinds",
                parity_surface_summary.get("aggregation_kind_counts"),
            )
        )
        lines.extend(
            _parity_surface_markdown_lines(
                "Parent arities",
                parity_surface_summary.get("parent_arity_counts"),
            )
        )
        lines.extend(
            _parity_surface_markdown_lines(
                "Source-shape policy",
                parity_surface_summary.get("source_shape_policy_counts"),
            )
        )
        lines.extend(
            _parity_surface_scalar_markdown_lines(
                "Kernel gamma",
                parity_surface_summary.get("kernel_gamma"),
            )
        )
        lines.extend(
            _parity_surface_scalar_markdown_lines(
                "Categorical cardinality",
                parity_surface_summary.get("categorical_cardinality"),
            )
        )

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path


def _parity_surface_markdown_lines(title: str, payload: object) -> list[str]:
    lines = [f"### {title}", ""]
    if not isinstance(payload, dict) or not payload:
        lines.append("- No observed counts.")
        lines.append("")
        return lines
    lines.append("| Label | Sampled Count |")
    lines.append("|---|---:|")
    for label in sorted(payload):
        lines.append(f"| {label} | {_fmt(payload.get(label), digits=0)} |")
    lines.append("")
    return lines


def _parity_surface_scalar_markdown_lines(title: str, payload: object) -> list[str]:
    lines = [f"### {title}", ""]
    if not isinstance(payload, dict) or int(_coerce_optional_int(payload.get("count")) or 0) <= 0:
        lines.append("- No observed values.")
        lines.append("")
        return lines
    lines.append(
        "- Count / mean / range: "
        f"`{_fmt(payload.get('count'), digits=0)}` / "
        f"`{_fmt(payload.get('mean'))}` / "
        f"`{_fmt(payload.get('min'))}-{_fmt(payload.get('max'))}`"
    )
    lines.append("")
    return lines


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


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    if not math.isfinite(as_float):
        return None
    return int(as_float)


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
