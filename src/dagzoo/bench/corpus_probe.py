"""Shared benchmark-style corpus probe helpers for audit workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dagzoo.bench.runtime_support import _resolve_benchmark_run_counts
from dagzoo.bench.stage_metrics import FilterStageMeasurement, replay_filter_stage_metrics
from dagzoo.bench.throughput import iter_throughput_measure_bundles, run_throughput_benchmark
from dagzoo.config import GeneratorConfig, clone_generator_config
from dagzoo.core.fixed_layout.runtime import realize_generation_config_for_run
from dagzoo.diagnostics.coverage import CoverageAggregationConfig, CoverageAggregator
from dagzoo.diagnostics_targets import build_diagnostics_aggregation_config

_REQUIRED_AUDIT_QUANTILES = (0.25, 0.5, 0.75)


@dataclass(slots=True)
class CorpusProbeResult:
    """Materialized one-config probe result for diversity audit workflows."""

    label: str
    config_path: str | None
    suite: str
    num_datasets: int
    warmup_datasets: int
    requested_device: str
    resolved_device: str
    resolved_config: dict[str, Any]
    datasets_per_minute: float
    filter_datasets_per_minute: float | None
    filter_accepted_datasets_per_minute: float | None
    filter_accepted_datasets_measured: int
    filter_rejected_datasets_measured: int
    filter_acceptance_rate_dataset_level: float | None
    filter_rejection_rate_dataset_level: float | None
    coverage_summary: dict[str, Any]
    filter_summary: dict[str, Any] | None


def resolve_corpus_probe_counts(
    config: GeneratorConfig,
    *,
    suite: str,
    num_datasets_override: int | None,
    warmup_override: int | None,
) -> tuple[int, int]:
    """Resolve one shared audit probe size using benchmark-style smoke caps."""

    if suite not in {"smoke", "standard"}:
        raise ValueError(f"Unsupported diversity-audit suite: {suite!r}")

    num_datasets = int(config.benchmark.num_datasets)
    warmup = int(config.benchmark.warmup_datasets)
    if num_datasets_override is not None:
        num_datasets = int(num_datasets_override)
    if warmup_override is not None:
        warmup = int(warmup_override)
    return _resolve_benchmark_run_counts(
        num_datasets=num_datasets,
        warmup_datasets=warmup,
        suite=suite,
    )


def build_corpus_probe_coverage_config(config: GeneratorConfig) -> CoverageAggregationConfig:
    """Build coverage aggregation config with quantiles needed for delta scoring."""

    base = build_diagnostics_aggregation_config(config)
    quantiles = tuple(
        sorted(set(float(q) for q in base.quantiles) | set(_REQUIRED_AUDIT_QUANTILES))
    )
    return CoverageAggregationConfig(
        include_spearman=bool(base.include_spearman),
        histogram_bins=max(1, int(base.histogram_bins)),
        quantiles=quantiles,
        underrepresented_threshold=float(base.underrepresented_threshold),
        max_values_per_metric=base.max_values_per_metric,
        target_bands=dict(base.target_bands),
    )


def _effective_probe_coverage_config(
    *,
    realized_config: GeneratorConfig,
    coverage_config: CoverageAggregationConfig | None,
) -> CoverageAggregationConfig:
    """Build one per-probe coverage config from shared settings plus realized steering."""

    template = (
        coverage_config
        if coverage_config is not None
        else build_corpus_probe_coverage_config(realized_config)
    )
    return CoverageAggregationConfig(
        include_spearman=bool(template.include_spearman),
        histogram_bins=max(1, int(template.histogram_bins)),
        quantiles=tuple(float(value) for value in template.quantiles),
        underrepresented_threshold=float(template.underrepresented_threshold),
        max_values_per_metric=template.max_values_per_metric,
        target_bands=dict(template.target_bands),
        steering_config=clone_generator_config(realized_config, revalidate=False),
    )


def _build_filter_summary(
    measurement: FilterStageMeasurement,
    *,
    config: GeneratorConfig,
) -> dict[str, Any]:
    """Normalize filter replay telemetry into the audit report shape."""

    return {
        "filter_mode": "small_shot_ease_v1",
        "ease_k_small": int(config.filter.ease_k_small),
        "easy_skill_threshold": float(config.filter.easy_skill_threshold),
        "easy_gain_threshold": float(config.filter.easy_gain_threshold),
        "hard_skill_threshold": float(config.filter.hard_skill_threshold),
        "stump_skill_threshold": (
            None
            if config.filter.stump_skill_threshold is None
            else float(config.filter.stump_skill_threshold)
        ),
        "use_lineage_veto": bool(config.filter.use_lineage_veto),
        "min_target_indegree": int(config.filter.min_target_indegree),
        "min_target_relevant_feature_count": int(config.filter.min_target_relevant_feature_count),
        "min_target_relevant_feature_fraction": float(
            config.filter.min_target_relevant_feature_fraction
        ),
        "classification_kappa_threshold": float(config.filter.classification_kappa_threshold),
        "classification_require_prediction_diversity": bool(
            config.filter.classification_require_prediction_diversity
        ),
        "accepted_true_fraction": measurement.accepted_true_fraction,
        "skill_small_mean": measurement.skill_small_mean,
        "skill_full_mean": measurement.skill_full_mean,
        "skill_gain_mean": measurement.skill_gain_mean,
        "skill_small_lb95_mean": measurement.skill_small_lb95_mean,
        "skill_gain_ub95_mean": measurement.skill_gain_ub95_mean,
        "skill_full_ub95_mean": measurement.skill_full_ub95_mean,
        "stump_skill_mean": measurement.stump_skill_mean,
        "reason_counts": dict(sorted(measurement.reason_counts.items())),
    }


def run_corpus_probe(
    config: GeneratorConfig,
    *,
    label: str,
    config_path: str | None,
    suite: str,
    num_datasets: int,
    warmup: int,
    device: str | None,
    probe_seed: int | None = None,
    coverage_config: CoverageAggregationConfig | None = None,
) -> CorpusProbeResult:
    """Run one benchmark-style corpus probe for diversity/comparison workflows."""

    realized_config, run_seed, requested_device, resolved_device = (
        realize_generation_config_for_run(
            config,
            seed=int(config.seed if probe_seed is None else probe_seed),
            device=device,
        )
    )
    realized_config.seed = int(run_seed)
    generation_config = clone_generator_config(realized_config, revalidate=False)
    generation_config.seed = int(run_seed)
    generation_config.filter.enabled = False

    throughput = run_throughput_benchmark(
        generation_config,
        num_datasets=num_datasets,
        warmup_datasets=warmup,
        device=requested_device,
        benchmark_fast_prepare=False,
    )

    coverage_aggregator = CoverageAggregator(
        _effective_probe_coverage_config(
            realized_config=realized_config,
            coverage_config=coverage_config,
        )
    )
    filter_datasets_per_minute: float | None = None
    filter_accepted_datasets_per_minute: float | None = None
    filter_accepted_datasets_measured = 0
    filter_rejected_datasets_measured = 0
    filter_acceptance_rate_dataset_level: float | None = None
    filter_rejection_rate_dataset_level: float | None = None
    filter_summary: dict[str, Any] | None = None
    analysis_stream = iter_throughput_measure_bundles(
        generation_config,
        num_datasets=num_datasets,
        device=requested_device,
        benchmark_fast_prepare=False,
    )

    if bool(realized_config.filter.enabled):

        def _record_accepted_bundle(bundle) -> None:
            coverage_aggregator.update_bundle(bundle)

        filter_measurement = replay_filter_stage_metrics(
            analysis_stream,
            config=realized_config,
            on_accepted_bundle=_record_accepted_bundle,
        )
        filter_datasets_per_minute = float(filter_measurement.datasets_per_minute)
        filter_accepted_datasets_measured = int(filter_measurement.filter_accepted_datasets)
        filter_rejected_datasets_measured = int(filter_measurement.filter_rejected_datasets)
        total_measured = filter_accepted_datasets_measured + filter_rejected_datasets_measured
        filter_acceptance_rate_dataset_level = (
            float(filter_accepted_datasets_measured) / float(total_measured)
            if total_measured > 0
            else None
        )
        filter_rejection_rate_dataset_level = (
            float(filter_rejected_datasets_measured) / float(total_measured)
            if total_measured > 0
            else None
        )
        filter_accepted_datasets_per_minute = (
            float(filter_datasets_per_minute) * float(filter_acceptance_rate_dataset_level)
            if filter_acceptance_rate_dataset_level is not None
            else None
        )
        filter_summary = _build_filter_summary(filter_measurement, config=realized_config)
    else:
        for bundle in analysis_stream:
            coverage_aggregator.update_bundle(bundle)

    coverage_summary = coverage_aggregator.build_summary()
    return CorpusProbeResult(
        label=label,
        config_path=config_path,
        suite=str(suite),
        num_datasets=int(num_datasets),
        warmup_datasets=int(warmup),
        requested_device=str(requested_device),
        resolved_device=str(resolved_device),
        resolved_config=realized_config.to_dict(),
        datasets_per_minute=float(throughput.get("datasets_per_minute", 0.0)),
        filter_datasets_per_minute=filter_datasets_per_minute,
        filter_accepted_datasets_per_minute=filter_accepted_datasets_per_minute,
        filter_accepted_datasets_measured=int(filter_accepted_datasets_measured),
        filter_rejected_datasets_measured=int(filter_rejected_datasets_measured),
        filter_acceptance_rate_dataset_level=filter_acceptance_rate_dataset_level,
        filter_rejection_rate_dataset_level=filter_rejection_rate_dataset_level,
        coverage_summary=coverage_summary,
        filter_summary=filter_summary,
    )
