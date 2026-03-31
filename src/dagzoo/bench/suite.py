"""Benchmark suite orchestration across runtime presets."""

from __future__ import annotations

import contextlib
import datetime as dt
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from dagzoo.bench.baseline import compare_summary_to_baseline
from dagzoo.bench.collectors import (
    _BundleMetricsCollector,
    _compose_bundle_callback,
    build_filtering_metrics,
    build_missingness_metrics,
    build_noise_metrics,
    build_pressure_metrics,
    build_shift_metrics,
)
from dagzoo.bench.constants import (
    DIAGNOSTICS_DUPLICATE_PRESET_SUFFIX_BASE,
    MIB,
    MICROBENCH_REPEATS,
    MISSINGNESS_RATE_FAIL_ABS_ERROR,
    MISSINGNESS_RATE_WARN_ABS_ERROR,
    NOISE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE,
    SHIFT_GUARDRAIL_DIRECTIONAL_GATING_MIN_SAMPLE,
    SHIFT_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE,
)
from dagzoo.bench.guardrails import (
    _build_guardrail_issue,
    _collect_scenario_regression_issues,
    _issue_sort_key,
    _severity_from_thresholds,
    _status_from_issues,
)
from dagzoo.bench.metrics import degradation_percent, reproducibility_signatures
from dagzoo.bench.micro import run_microbenchmarks
from dagzoo.bench.preset_specs import (
    PresetRunSpec,
    _cap_smoke_rows_spec,
    _copy_runtime_config,
    _smoke_caps_for_spec,
)
from dagzoo.bench.preset_specs import (
    resolve_preset_run_specs as _resolve_preset_run_specs,
)
from dagzoo.bench.runtime_support import (
    _artifact_pointer,
    _build_diagnostics_aggregator,
    _build_shift_directional_check,
    _collect_latency,
    _latency_sample_count,
    _peak_rss_mb,
    _preset_counts,
    _sanitize_preset_key,
    _synchronize_accelerator,
)
from dagzoo.bench.stage_metrics import (
    FilterStageMeasurement,
    StageSampleCollector,
    measure_filter_stage_metrics,
    measure_write_stage_metrics,
)
from dagzoo.bench.throughput import (
    _benchmark_precompute_classification_attempt_plan,
    _throughput_measure_seed,
    run_throughput_benchmark,
)
from dagzoo.config import (
    MISSINGNESS_MECHANISM_NONE,
    NOISE_FAMILY_GAUSSIAN,
    SHIFT_MODE_OFF,
    GeneratorConfig,
    effective_config_payload,
)
from dagzoo.core.config_predicates import (
    missingness_enabled as _is_missingness_enabled,
)
from dagzoo.core.config_predicates import (
    non_gaussian_noise_enabled as _is_noise_enabled,
)
from dagzoo.core.config_predicates import (
    shift_enabled as _is_shift_enabled,
)
from dagzoo.core.config_resolution import (
    append_config_diff_events,
    resolve_benchmark_preset_config,
    serialize_resolution_events,
)
from dagzoo.core.dataset import generate_batch_iter
from dagzoo.core.fixed_layout.prepare import normalize_fixed_layout_target_cells
from dagzoo.core.fixed_layout.runtime import (
    prepare_canonical_fixed_layout_run,
    realize_generation_config_for_run,
)
from dagzoo.core.shift import resolve_shift_runtime_params
from dagzoo.diagnostics import (
    CoverageAggregator,
    write_coverage_summary_json,
    write_coverage_summary_markdown,
)
from dagzoo.rng import KeyedRng
from dagzoo.types import DatasetBundle


def _collect_reproducibility(
    config: GeneratorConfig,
    *,
    device: str | None,
    num_datasets: int,
) -> dict[str, Any]:
    """Generate two deterministic runs and compare content digests."""

    n = max(1, num_datasets)
    run_seed = KeyedRng(int(config.seed)).child_seed("bench", "suite", "reproducibility")
    sig_a, workload_a = reproducibility_signatures(
        generate_batch_iter(config, num_datasets=n, seed=run_seed, device=device)
    )
    sig_b, workload_b = reproducibility_signatures(
        generate_batch_iter(config, num_datasets=n, seed=run_seed, device=device)
    )
    return {
        "reproducibility_datasets": n,
        "reproducibility_signature": sig_a,
        "reproducibility_match": bool(sig_a == sig_b),
        "reproducibility_workload_signature": workload_a,
        "reproducibility_workload_match": bool(workload_a == workload_b),
    }


def _cpu_busy_pct_of_wall(*, cpu_time_seconds: float, elapsed_seconds: float) -> float:
    """Return process CPU time as a percentage of wall time for one stage."""

    if elapsed_seconds <= 0.0:
        return 0.0
    return float(cpu_time_seconds) / float(elapsed_seconds) * 100.0


def _build_fixed_layout_evidence(
    config: GeneratorConfig,
    *,
    device: str | None,
    num_datasets: int,
) -> dict[str, int]:
    """Resolve fixed-layout batch and chunking evidence for one benchmark run."""

    sample_n = max(1, int(num_datasets))
    prepared = prepare_canonical_fixed_layout_run(
        config,
        num_datasets=sample_n,
        seed=_throughput_measure_seed(config),
        device=device,
        precompute_classification_attempt_plan=_benchmark_precompute_classification_attempt_plan(
            config,
            benchmark_fast_prepare=True,
        ),
    )
    per_dataset_cells = int(prepared.plan.n_train + prepared.plan.n_test) * max(
        1, int(prepared.plan.layout.n_features)
    )
    batch_size = max(1, int(prepared.batch_size))
    chunk_count = ((sample_n - 1) // batch_size) + 1
    tail_chunk_size = sample_n - ((chunk_count - 1) * batch_size)
    return {
        "fixed_layout_target_cells_effective": int(
            normalize_fixed_layout_target_cells(prepared.config.runtime.fixed_layout_target_cells)
        ),
        "fixed_layout_per_dataset_cells": int(per_dataset_cells),
        "fixed_layout_realized_batch_size": int(batch_size),
        "fixed_layout_chunk_count": int(chunk_count),
        "fixed_layout_tail_chunk_size": int(max(1, tail_chunk_size)),
    }


def _build_scenario(
    enabled: bool,
    *,
    metrics: dict[str, Any] | None = None,
    control_metrics: dict[str, Any] | None = None,
    issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scenario_issues = list(issues or [])
    payload: dict[str, Any] = {
        "enabled": bool(enabled),
        "status": "off" if not enabled else _status_from_issues(scenario_issues),
        "metrics": dict(metrics or {}),
        "issues": scenario_issues,
    }
    if control_metrics is not None:
        payload["control_metrics"] = dict(control_metrics)
    return payload


def _build_on_bundle_callback(
    *,
    diagnostics_aggregator: CoverageAggregator | None,
    collector: _BundleMetricsCollector,
    stage_sample_cap: int,
) -> tuple[Callable[[DatasetBundle], None], StageSampleCollector]:
    stage_sample_collector = StageSampleCollector(max_samples=stage_sample_cap)
    composed_on_bundle_callback = _compose_bundle_callback(
        diagnostics_aggregator=diagnostics_aggregator,
        collector=collector,
    )

    def on_bundle_callback(bundle: DatasetBundle) -> None:
        stage_sample_collector.update(bundle)
        if composed_on_bundle_callback is not None:
            composed_on_bundle_callback(bundle)

    return on_bundle_callback, stage_sample_collector


def _run_control_benchmark(
    config: GeneratorConfig,
    *,
    requested_device: str | None,
    num_datasets: int,
    warmup: int,
    stage_sample_cap: int,
    diagnostics_enabled: bool,
) -> tuple[float, _BundleMetricsCollector]:
    diagnostics_aggregator = _build_diagnostics_aggregator(config) if diagnostics_enabled else None
    collector = _BundleMetricsCollector(expected_noise_family_requested=str(config.noise.family))
    on_bundle_callback, stage_sample_collector = _build_on_bundle_callback(
        diagnostics_aggregator=diagnostics_aggregator,
        collector=collector,
        stage_sample_cap=stage_sample_cap,
    )
    throughput = run_throughput_benchmark(
        config,
        num_datasets=num_datasets,
        warmup_datasets=warmup,
        device=requested_device,
        on_bundle=on_bundle_callback,
    )
    stage_sample_collector.bundles.clear()
    return float(throughput.get("datasets_per_minute", 0.0)), collector


def _missingness_scenario(
    *,
    config: GeneratorConfig,
    generation_config: GeneratorConfig,
    collector: _BundleMetricsCollector,
    requested_device: str | None,
    num_datasets: int,
    warmup: int,
    stage_sample_cap: int,
    current_dpm: float,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    diagnostics_enabled: bool,
) -> dict[str, Any]:
    if not _is_missingness_enabled(config):
        return _build_scenario(False)

    candidate_metrics = build_missingness_metrics(
        collector,
        target_rate=float(config.dataset.missing_rate),
    )
    issues: list[dict[str, Any]] = []
    metadata_coverage_rate = float(candidate_metrics["metadata_coverage_rate"])
    if metadata_coverage_rate < 1.0:
        issues.append(
            _build_guardrail_issue(
                metric="missingness_metadata_coverage",
                severity="fail",
                current=metadata_coverage_rate,
                baseline=1.0,
                degradation_pct=float(max(0.0, (1.0 - metadata_coverage_rate) * 100.0)),
                detail="Missingness metadata must be present for all generated bundles.",
            )
        )

    rate_abs_error = float(candidate_metrics["rate_abs_error"])
    rate_error_pp = rate_abs_error * 100.0
    rate_severity = _severity_from_thresholds(
        rate_abs_error,
        warn=MISSINGNESS_RATE_WARN_ABS_ERROR,
        fail=MISSINGNESS_RATE_FAIL_ABS_ERROR,
    )
    if rate_severity != "pass":
        threshold_pp = (
            MISSINGNESS_RATE_FAIL_ABS_ERROR * 100.0
            if rate_severity == "fail"
            else MISSINGNESS_RATE_WARN_ABS_ERROR * 100.0
        )
        issues.append(
            _build_guardrail_issue(
                metric="missingness_realized_rate_error_pp",
                severity=rate_severity,
                current=float(rate_error_pp),
                baseline=float(threshold_pp),
                degradation_pct=float(rate_error_pp),
                detail="Realized missing rate drifted from configured target.",
            )
        )

    baseline_config = _copy_runtime_config(generation_config)
    baseline_config.dataset.missing_rate = 0.0
    baseline_config.dataset.missing_mechanism = MISSINGNESS_MECHANISM_NONE
    baseline_dpm, _ = _run_control_benchmark(
        baseline_config,
        requested_device=requested_device,
        num_datasets=num_datasets,
        warmup=warmup,
        stage_sample_cap=stage_sample_cap,
        diagnostics_enabled=diagnostics_enabled,
    )
    runtime_degradation = degradation_percent("datasets_per_minute", current_dpm, baseline_dpm)
    runtime_degradation_value = (
        float(runtime_degradation) if runtime_degradation is not None else 0.0
    )
    runtime_severity = _severity_from_thresholds(
        runtime_degradation_value,
        warn=float(warn_threshold_pct),
        fail=float(fail_threshold_pct),
    )
    if runtime_severity != "pass":
        issues.append(
            _build_guardrail_issue(
                metric="missingness_runtime_degradation_pct",
                severity=runtime_severity,
                current=current_dpm,
                baseline=baseline_dpm,
                degradation_pct=runtime_degradation_value,
                detail=(
                    "Missingness-enabled throughput regressed versus an equivalent "
                    "missingness-disabled control run."
                ),
            )
        )

    candidate_metrics["datasets_per_minute"] = current_dpm
    candidate_metrics["rate_warn_abs_error"] = float(MISSINGNESS_RATE_WARN_ABS_ERROR)
    candidate_metrics["rate_fail_abs_error"] = float(MISSINGNESS_RATE_FAIL_ABS_ERROR)
    return _build_scenario(
        True,
        metrics=candidate_metrics,
        control_metrics={"datasets_per_minute": baseline_dpm},
        issues=issues,
    )


def _shift_scenario(
    *,
    config: GeneratorConfig,
    generation_config: GeneratorConfig,
    collector: _BundleMetricsCollector,
    requested_device: str | None,
    num_datasets: int,
    warmup: int,
    stage_sample_cap: int,
    current_dpm: float,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    diagnostics_enabled: bool,
) -> dict[str, Any]:
    if not _is_shift_enabled(config):
        return _build_scenario(False)

    shift_params = resolve_shift_runtime_params(config)
    candidate_metrics = build_shift_metrics(collector)
    baseline_config = _copy_runtime_config(generation_config)
    baseline_config.shift.enabled = False
    baseline_config.shift.mode = SHIFT_MODE_OFF
    baseline_config.shift.graph_scale = None
    baseline_config.shift.mechanism_scale = None
    baseline_config.shift.variance_scale = None
    baseline_dpm, baseline_collector = _run_control_benchmark(
        baseline_config,
        requested_device=requested_device,
        num_datasets=num_datasets,
        warmup=warmup,
        stage_sample_cap=stage_sample_cap,
        diagnostics_enabled=diagnostics_enabled,
    )
    control_metrics = build_shift_metrics(baseline_collector)
    control_metrics["datasets_per_minute"] = baseline_dpm

    runtime_degradation = degradation_percent("datasets_per_minute", current_dpm, baseline_dpm)
    runtime_degradation_value = (
        float(runtime_degradation) if runtime_degradation is not None else 0.0
    )
    runtime_severity = _severity_from_thresholds(
        runtime_degradation_value,
        warn=float(warn_threshold_pct),
        fail=float(fail_threshold_pct),
    )
    runtime_gating_enabled = num_datasets >= SHIFT_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE
    directional_gating_enabled = num_datasets >= SHIFT_GUARDRAIL_DIRECTIONAL_GATING_MIN_SAMPLE

    issues: list[dict[str, Any]] = []
    metadata_coverage_rate = float(candidate_metrics["metadata_coverage_rate"])
    if metadata_coverage_rate < 1.0:
        issues.append(
            _build_guardrail_issue(
                metric="shift_metadata_coverage",
                severity="fail",
                current=metadata_coverage_rate,
                baseline=1.0,
                degradation_pct=float(max(0.0, (1.0 - metadata_coverage_rate) * 100.0)),
                detail="Shift metadata must be present for all shift-enabled bundles.",
            )
        )
    shift_enabled_coverage_rate = float(candidate_metrics["shift_enabled_coverage_rate"])
    if shift_enabled_coverage_rate < 1.0:
        issues.append(
            _build_guardrail_issue(
                metric="shift_enabled_metadata_coverage",
                severity="fail",
                current=shift_enabled_coverage_rate,
                baseline=1.0,
                degradation_pct=float(max(0.0, (1.0 - shift_enabled_coverage_rate) * 100.0)),
                detail="Shift-enabled benchmark runs must emit shift.enabled=true metadata.",
            )
        )
    if runtime_gating_enabled and runtime_severity != "pass":
        issues.append(
            _build_guardrail_issue(
                metric="shift_runtime_degradation_pct",
                severity=runtime_severity,
                current=current_dpm,
                baseline=baseline_dpm,
                degradation_pct=runtime_degradation_value,
                detail=(
                    "Shift-enabled throughput regressed versus an equivalent "
                    "shift-disabled control run."
                ),
            )
        )

    graph_check, graph_issue = _build_shift_directional_check(
        metric="graph_edge_density",
        enabled=float(shift_params.graph_scale) > 0.0,
        gating_enabled=directional_gating_enabled,
        current=candidate_metrics.get("mean_graph_edge_density"),
        baseline=control_metrics.get("mean_graph_edge_density"),
        detail=(
            "Graph shift should increase mean graph edge density "
            "relative to a shift-disabled control run."
        ),
    )
    mechanism_check, mechanism_issue = _build_shift_directional_check(
        metric="mechanism_nonlinear_mass",
        enabled=float(shift_params.mechanism_scale) > 0.0,
        gating_enabled=directional_gating_enabled,
        current=candidate_metrics.get("mean_mechanism_nonlinear_mass"),
        baseline=control_metrics.get("mean_mechanism_nonlinear_mass"),
        detail=(
            "Mechanism shift should increase nonlinear family mass "
            "relative to a shift-disabled control run."
        ),
    )
    noise_check, noise_issue = _build_shift_directional_check(
        metric="noise_variance_multiplier",
        enabled=float(shift_params.variance_scale) > 0.0,
        gating_enabled=directional_gating_enabled,
        current=candidate_metrics.get("mean_noise_variance_multiplier"),
        baseline=control_metrics.get("mean_noise_variance_multiplier"),
        detail=(
            "Noise shift should increase noise variance multiplier "
            "relative to a shift-disabled control run."
        ),
    )
    for maybe_issue in (graph_issue, mechanism_issue, noise_issue):
        if maybe_issue is not None:
            issues.append(maybe_issue)

    candidate_metrics.update(
        {
            "datasets_per_minute": current_dpm,
            "mode": str(shift_params.mode),
            "graph_scale": float(shift_params.graph_scale),
            "mechanism_scale": float(shift_params.mechanism_scale),
            "variance_scale": float(shift_params.variance_scale),
            "runtime_gating_enabled": bool(runtime_gating_enabled),
            "directional_gating_enabled": bool(directional_gating_enabled),
            "directional_checks": {
                "graph_edge_density": graph_check,
                "mechanism_nonlinear_mass": mechanism_check,
                "noise_variance_multiplier": noise_check,
            },
        }
    )
    return _build_scenario(
        True,
        metrics=candidate_metrics,
        control_metrics=control_metrics,
        issues=issues,
    )


def _noise_scenario(
    *,
    config: GeneratorConfig,
    generation_config: GeneratorConfig,
    collector: _BundleMetricsCollector,
    requested_device: str | None,
    num_datasets: int,
    warmup: int,
    stage_sample_cap: int,
    current_dpm: float,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    diagnostics_enabled: bool,
) -> dict[str, Any]:
    if not _is_noise_enabled(config):
        return _build_scenario(False)

    candidate_metrics = build_noise_metrics(collector)
    baseline_config = _copy_runtime_config(generation_config)
    baseline_config.noise.family = NOISE_FAMILY_GAUSSIAN
    baseline_config.noise.base_scale = 1.0
    baseline_config.noise.student_t_df = 5.0
    baseline_config.noise.mixture_weights = None
    baseline_dpm, baseline_collector = _run_control_benchmark(
        baseline_config,
        requested_device=requested_device,
        num_datasets=num_datasets,
        warmup=warmup,
        stage_sample_cap=stage_sample_cap,
        diagnostics_enabled=diagnostics_enabled,
    )
    control_metrics = build_noise_metrics(baseline_collector)
    control_metrics["datasets_per_minute"] = baseline_dpm
    control_metrics["family_requested"] = str(baseline_config.noise.family)

    runtime_degradation = degradation_percent("datasets_per_minute", current_dpm, baseline_dpm)
    runtime_degradation_value = (
        float(runtime_degradation) if runtime_degradation is not None else 0.0
    )
    runtime_severity = _severity_from_thresholds(
        runtime_degradation_value,
        warn=float(warn_threshold_pct),
        fail=float(fail_threshold_pct),
    )
    runtime_gating_enabled = num_datasets >= NOISE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE

    issues: list[dict[str, Any]] = []
    metadata_coverage_rate = float(candidate_metrics["metadata_coverage_rate"])
    if metadata_coverage_rate < 1.0:
        issues.append(
            _build_guardrail_issue(
                metric="noise_metadata_coverage",
                severity="fail",
                current=metadata_coverage_rate,
                baseline=1.0,
                degradation_pct=float(max(0.0, (1.0 - metadata_coverage_rate) * 100.0)),
                detail="Noise metadata must be present for all generated bundles.",
            )
        )
    metadata_valid_rate = float(candidate_metrics["metadata_valid_rate"])
    if metadata_valid_rate < 1.0:
        issues.append(
            _build_guardrail_issue(
                metric="noise_metadata_validity",
                severity="fail",
                current=metadata_valid_rate,
                baseline=1.0,
                degradation_pct=float(max(0.0, (1.0 - metadata_valid_rate) * 100.0)),
                detail="Noise metadata must be valid and consistent with configured family.",
            )
        )
    if runtime_gating_enabled and runtime_severity != "pass":
        issues.append(
            _build_guardrail_issue(
                metric="noise_runtime_degradation_pct",
                severity=runtime_severity,
                current=current_dpm,
                baseline=baseline_dpm,
                degradation_pct=runtime_degradation_value,
                detail=(
                    "Non-gaussian noise throughput regressed versus an equivalent "
                    "gaussian-noise control run."
                ),
            )
        )

    candidate_metrics["datasets_per_minute"] = current_dpm
    candidate_metrics["family_requested"] = str(config.noise.family)
    candidate_metrics["runtime_gating_enabled"] = bool(runtime_gating_enabled)
    return _build_scenario(
        True,
        metrics=candidate_metrics,
        control_metrics=control_metrics,
        issues=issues,
    )


def run_preset_benchmark(
    spec: PresetRunSpec,
    *,
    suite: str,
    num_datasets_override: int | None,
    warmup_override: int | None,
    collect_memory: bool,
    collect_reproducibility: bool,
    include_micro: bool,
    hardware_policy: str,
    collect_diagnostics: bool,
    diagnostics_root_dir: Path | None,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    diagnostics_occurrence_index: int,
    diagnostics_occurrence_total: int,
) -> dict[str, Any]:
    """Run one benchmark preset and collect throughput, latency, and scenario metrics."""

    def _resolve_preset_for_requested_device(requested_device: str):
        return resolve_benchmark_preset_config(
            preset_key=spec.key,
            config=spec.config,
            preset_device=requested_device,
            suite=suite,
            hardware_policy=hardware_policy,
            smoke_caps=_smoke_caps_for_spec(spec),
        )

    normalized_preset_device = (spec.device or spec.config.runtime.device or "auto").lower()
    resolved_preset = _resolve_preset_for_requested_device(normalized_preset_device)
    pre_realization_config = _copy_runtime_config(resolved_preset["config"])
    trace_events = list(resolved_preset["trace_events"])
    if suite == "smoke":
        _cap_smoke_rows_spec(pre_realization_config)
        append_config_diff_events(
            resolved_preset["config"],
            pre_realization_config,
            source="benchmark.smoke_rows_cap",
            events=trace_events,
        )
    config, _run_seed, requested_device, _resolved_device = realize_generation_config_for_run(
        pre_realization_config,
        seed=resolved_preset["config"].seed,
        device=str(resolved_preset["requested_device"]),
    )
    append_config_diff_events(
        pre_realization_config,
        config,
        source="benchmark.run_realization",
        events=trace_events,
    )
    hw = resolved_preset["hardware"]
    num_datasets, warmup = _preset_counts(
        config,
        preset_key=spec.key,
        suite=suite,
        num_datasets_override=num_datasets_override,
        warmup_override=warmup_override,
    )

    rss_before = _peak_rss_mb() if collect_memory else 0.0
    if collect_memory and hw.backend == "cuda" and torch.cuda.is_available():
        _synchronize_accelerator(requested_device)
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats()

    diagnostics_enabled = bool(collect_diagnostics and diagnostics_root_dir is not None)
    diagnostics_aggregator: CoverageAggregator | None = None
    if diagnostics_enabled:
        diagnostics_aggregator = _build_diagnostics_aggregator(config)

    stage_sample_cap = _latency_sample_count(config, suite, num_datasets)
    collector = _BundleMetricsCollector(expected_noise_family_requested=str(config.noise.family))

    generation_config = _copy_runtime_config(config)
    generation_config.filter.enabled = False
    fixed_layout_evidence = _build_fixed_layout_evidence(
        generation_config,
        device=requested_device,
        num_datasets=num_datasets,
    )
    on_bundle_callback, stage_sample_collector = _build_on_bundle_callback(
        diagnostics_aggregator=diagnostics_aggregator,
        collector=collector,
        stage_sample_cap=stage_sample_cap,
    )

    result = run_throughput_benchmark(
        generation_config,
        num_datasets=num_datasets,
        warmup_datasets=warmup,
        device=requested_device,
        on_bundle=on_bundle_callback,
    )
    generation_elapsed_seconds = float(result.get("elapsed_seconds", 0.0) or 0.0)
    generation_cpu_time_seconds = float(result.get("cpu_time_seconds", 0.0) or 0.0)
    prepare_elapsed_seconds = float(result.get("prepare_elapsed_seconds", 0.0) or 0.0)
    prepare_cpu_time_seconds = float(result.get("prepare_cpu_time_seconds", 0.0) or 0.0)
    sampled_bundles = stage_sample_collector.bundles
    stage_sample_datasets = len(sampled_bundles)

    write_stage_measurement = (
        measure_write_stage_metrics(sampled_bundles, config=config)
        if sampled_bundles
        else measure_write_stage_metrics([], config=config)
    )
    filter_stage_enabled = bool(config.filter.enabled)
    filter_stage_measurement: FilterStageMeasurement | None = None
    if filter_stage_enabled and sampled_bundles:
        filter_stage_measurement = measure_filter_stage_metrics(sampled_bundles, config=config)

    pressure_metrics = build_pressure_metrics(collector)
    generation_dpm = float(result.get("datasets_per_minute", 0.0))
    write_dpm = float(getattr(write_stage_measurement, "datasets_per_minute", 0.0))
    filtering_metrics: dict[str, Any] | None = None
    filter_dpm: float | None = None
    filter_accepted_datasets_per_minute: float | None = None
    filter_acceptance_rate_dataset_level: float | None = None
    filter_rejection_rate_dataset_level: float | None = None
    filter_attempts_total = int(pressure_metrics["filter_attempts_total"])
    filter_rejections_total = int(pressure_metrics["filter_rejections_total"])
    filter_retry_dataset_count = int(pressure_metrics["filter_retry_dataset_count"])
    filter_retry_dataset_rate = pressure_metrics["filter_retry_dataset_rate"]
    filter_accepted_datasets_measured = 0
    filter_rejected_datasets_measured = 0
    if filter_stage_measurement is not None:
        filtering_metrics = build_filtering_metrics(
            collector,
            filter_stage_measurement=filter_stage_measurement,
        )
        filter_dpm = float(filtering_metrics["datasets_per_minute"])
        filter_accepted_datasets_per_minute = filtering_metrics["accepted_datasets_per_minute"]
        filter_acceptance_rate_dataset_level = filtering_metrics["acceptance_rate_dataset_level"]
        filter_rejection_rate_dataset_level = filtering_metrics["rejection_rate_dataset_level"]
        filter_attempts_total = int(filtering_metrics["filter_attempts_total"])
        filter_rejections_total = int(filtering_metrics["filter_rejections_total"])
        filter_retry_dataset_count = int(filtering_metrics["filter_retry_dataset_count"])
        filter_retry_dataset_rate = filtering_metrics["filter_retry_dataset_rate"]
        filter_accepted_datasets_measured = int(filtering_metrics["accepted_datasets"])
        filter_rejected_datasets_measured = int(filtering_metrics["rejected_datasets"])

    result["preset_key"] = spec.key
    result["suite"] = suite
    result["device"] = requested_device
    result["generation_mode"] = str(result.get("generation_mode", "dynamic"))
    result["dataset_rows_total"] = int(config.dataset.n_train + config.dataset.n_test)
    result["dataset_n_train"] = int(config.dataset.n_train)
    result["dataset_n_test"] = int(config.dataset.n_test)
    result["hardware_backend"] = hw.backend
    result["hardware_device_name"] = hw.device_name
    result["hardware_memory_gb"] = hw.total_memory_gb
    result["hardware_peak_flops"] = hw.peak_flops
    result["hardware_tier"] = hw.tier
    result["hardware_policy"] = str(hardware_policy)
    result["effective_config"] = effective_config_payload(config)
    result["effective_config_trace"] = serialize_resolution_events(trace_events)
    result["diagnostics_enabled"] = diagnostics_enabled
    result["diagnostics_artifacts"] = None
    result["generation_datasets_per_minute"] = generation_dpm
    result["prepare_elapsed_seconds"] = prepare_elapsed_seconds
    result["prepare_cpu_time_seconds"] = prepare_cpu_time_seconds
    result["prepare_cpu_busy_pct_of_wall"] = _cpu_busy_pct_of_wall(
        cpu_time_seconds=prepare_cpu_time_seconds,
        elapsed_seconds=prepare_elapsed_seconds,
    )
    result["generation_elapsed_seconds"] = generation_elapsed_seconds
    result["generation_cpu_time_seconds"] = generation_cpu_time_seconds
    result["generation_cpu_busy_pct_of_wall"] = _cpu_busy_pct_of_wall(
        cpu_time_seconds=generation_cpu_time_seconds,
        elapsed_seconds=generation_elapsed_seconds,
    )
    result["raw_batch_elapsed_seconds"] = float(result.get("raw_batch_elapsed_seconds", 0.0) or 0.0)
    result["raw_batch_cpu_time_seconds"] = float(
        result.get("raw_batch_cpu_time_seconds", 0.0) or 0.0
    )
    result["node_apply_elapsed_seconds"] = float(
        result.get("node_apply_elapsed_seconds", 0.0) or 0.0
    )
    result["node_apply_cpu_time_seconds"] = float(
        result.get("node_apply_cpu_time_seconds", 0.0) or 0.0
    )
    result["converter_elapsed_seconds"] = float(result.get("converter_elapsed_seconds", 0.0) or 0.0)
    result["converter_cpu_time_seconds"] = float(
        result.get("converter_cpu_time_seconds", 0.0) or 0.0
    )
    result["feature_materialization_elapsed_seconds"] = float(
        result.get("feature_materialization_elapsed_seconds", 0.0) or 0.0
    )
    result["feature_materialization_cpu_time_seconds"] = float(
        result.get("feature_materialization_cpu_time_seconds", 0.0) or 0.0
    )
    result["write_datasets_per_minute"] = float(write_dpm)
    result["write_stage_elapsed_seconds"] = float(
        getattr(write_stage_measurement, "elapsed_seconds", 0.0)
    )
    result["write_stage_cpu_time_seconds"] = float(
        getattr(write_stage_measurement, "cpu_time_seconds", 0.0)
    )
    result["write_stage_bytes_written"] = int(getattr(write_stage_measurement, "bytes_written", 0))
    result["write_stage_mib_per_second"] = float(
        getattr(write_stage_measurement, "mib_per_second", 0.0)
    )
    result["filter_datasets_per_minute"] = float(filter_dpm) if filter_dpm is not None else None
    result["filter_accepted_datasets_per_minute"] = filter_accepted_datasets_per_minute
    result["filter_stage_elapsed_seconds"] = (
        float(getattr(filter_stage_measurement, "elapsed_seconds", 0.0))
        if filter_stage_measurement is not None
        else None
    )
    result["filter_stage_cpu_time_seconds"] = (
        float(getattr(filter_stage_measurement, "cpu_time_seconds", 0.0))
        if filter_stage_measurement is not None
        else None
    )
    result["stage_sample_datasets"] = int(stage_sample_datasets)
    result["filter_stage_enabled"] = filter_stage_enabled
    result["total_attempts"] = int(pressure_metrics["attempts_total"])
    result["mean_attempts_per_dataset"] = float(pressure_metrics["attempts_per_dataset_mean"])
    result["retry_dataset_count"] = int(pressure_metrics["retry_dataset_count"])
    result["retry_dataset_rate"] = pressure_metrics["retry_dataset_rate"]
    result["filter_attempts_total"] = int(filter_attempts_total)
    result["filter_rejections_total"] = int(filter_rejections_total)
    result["filter_accepted_datasets_measured"] = int(filter_accepted_datasets_measured)
    result["filter_rejected_datasets_measured"] = int(filter_rejected_datasets_measured)
    result["filter_acceptance_rate_dataset_level"] = filter_acceptance_rate_dataset_level
    result["filter_rejection_rate_dataset_level"] = filter_rejection_rate_dataset_level
    result["filter_rejection_rate_attempt_level"] = pressure_metrics[
        "filter_rejection_rate_attempt_level"
    ]
    result["filter_retry_dataset_count"] = int(filter_retry_dataset_count)
    result["filter_retry_dataset_rate"] = filter_retry_dataset_rate
    result["estimated_attempts_per_minute"] = generation_dpm * float(
        pressure_metrics["attempts_per_dataset_mean"]
    )
    result.update(fixed_layout_evidence)

    sampled_bundles.clear()

    latency_stats: Mapping[str, float | None] = _collect_latency(
        generation_config,
        device=requested_device,
        num_samples=_latency_sample_count(config, suite, num_datasets),
    )
    result.update(latency_stats)

    if collect_memory:
        result["peak_rss_mb"] = max(0.0, _peak_rss_mb() - rss_before)
        if hw.backend == "cuda" and torch.cuda.is_available():
            _synchronize_accelerator(requested_device)
            try:
                allocated_mb = torch.cuda.max_memory_allocated() / MIB
                reserved_mb = torch.cuda.max_memory_reserved() / MIB
                total_memory_mb = (
                    float(hw.total_memory_gb) * 1024.0
                    if isinstance(hw.total_memory_gb, (int, float)) and hw.total_memory_gb > 0.0
                    else None
                )
                result["peak_cuda_allocated_mb"] = allocated_mb
                result["peak_cuda_reserved_mb"] = reserved_mb
                result["peak_cuda_total_memory_mb"] = total_memory_mb
                result["peak_cuda_allocated_pct_of_total_memory"] = (
                    (float(allocated_mb) / float(total_memory_mb) * 100.0)
                    if total_memory_mb is not None and total_memory_mb > 0.0
                    else None
                )
                result["peak_cuda_reserved_pct_of_total_memory"] = (
                    (float(reserved_mb) / float(total_memory_mb) * 100.0)
                    if total_memory_mb is not None and total_memory_mb > 0.0
                    else None
                )
                result["peak_cuda_headroom_mb"] = (
                    max(0.0, float(total_memory_mb) - float(reserved_mb))
                    if total_memory_mb is not None
                    else None
                )
            except Exception:
                result["peak_cuda_allocated_mb"] = None
                result["peak_cuda_reserved_mb"] = None
                result["peak_cuda_total_memory_mb"] = None
                result["peak_cuda_allocated_pct_of_total_memory"] = None
                result["peak_cuda_reserved_pct_of_total_memory"] = None
                result["peak_cuda_headroom_mb"] = None

    if collect_reproducibility:
        repro_n = min(num_datasets, max(1, int(config.benchmark.reproducibility_num_datasets)))
        result.update(
            _collect_reproducibility(
                generation_config,
                device=requested_device,
                num_datasets=repro_n,
            )
        )

    if include_micro:
        result.update(
            run_microbenchmarks(
                generation_config,
                device=requested_device,
                repeats=MICROBENCH_REPEATS,
                include_generate_one=True,
            )
        )

    scenarios = {
        "baseline": _build_scenario(
            True,
            metrics={
                "datasets_per_minute": generation_dpm,
                "generation_datasets_per_minute": generation_dpm,
                "write_datasets_per_minute": write_dpm,
                "filter_datasets_per_minute": filter_dpm,
                "filter_accepted_datasets_per_minute": filter_accepted_datasets_per_minute,
                "latency_p95_ms": result.get("latency_p95_ms"),
                "peak_rss_mb": result.get("peak_rss_mb"),
            },
        ),
        "throughput": _build_scenario(
            True,
            metrics={
                **build_pressure_metrics(collector),
                "datasets_per_minute": generation_dpm,
                "generation_datasets_per_minute": generation_dpm,
                "write_datasets_per_minute": write_dpm,
                "filter_datasets_per_minute": filter_dpm,
                "filter_accepted_datasets_per_minute": filter_accepted_datasets_per_minute,
                "stage_sample_datasets": int(stage_sample_datasets),
            },
        ),
        "filtering": _build_scenario(
            filter_stage_measurement is not None and filter_stage_enabled,
            metrics=filtering_metrics or {},
        ),
    }
    scenarios["missingness"] = _missingness_scenario(
        config=config,
        generation_config=generation_config,
        collector=collector,
        requested_device=requested_device,
        num_datasets=num_datasets,
        warmup=warmup,
        stage_sample_cap=stage_sample_cap,
        current_dpm=generation_dpm,
        warn_threshold_pct=warn_threshold_pct,
        fail_threshold_pct=fail_threshold_pct,
        diagnostics_enabled=diagnostics_enabled,
    )
    scenarios["shift"] = _shift_scenario(
        config=config,
        generation_config=generation_config,
        collector=collector,
        requested_device=requested_device,
        num_datasets=num_datasets,
        warmup=warmup,
        stage_sample_cap=stage_sample_cap,
        current_dpm=generation_dpm,
        warn_threshold_pct=warn_threshold_pct,
        fail_threshold_pct=fail_threshold_pct,
        diagnostics_enabled=diagnostics_enabled,
    )
    scenarios["noise"] = _noise_scenario(
        config=config,
        generation_config=generation_config,
        collector=collector,
        requested_device=requested_device,
        num_datasets=num_datasets,
        warmup=warmup,
        stage_sample_cap=stage_sample_cap,
        current_dpm=generation_dpm,
        warn_threshold_pct=warn_threshold_pct,
        fail_threshold_pct=fail_threshold_pct,
        diagnostics_enabled=diagnostics_enabled,
    )
    result["scenarios"] = scenarios

    if (
        diagnostics_enabled
        and diagnostics_aggregator is not None
        and diagnostics_root_dir is not None
    ):
        preset_segment = _sanitize_preset_key(spec.key)
        if diagnostics_occurrence_total > 1:
            run_number = diagnostics_occurrence_index + DIAGNOSTICS_DUPLICATE_PRESET_SUFFIX_BASE
            preset_segment = f"{preset_segment}_run{run_number}"
        preset_diagnostics_dir = diagnostics_root_dir / "diagnostics" / preset_segment
        summary = diagnostics_aggregator.build_summary()
        json_path = write_coverage_summary_json(
            summary, preset_diagnostics_dir / "coverage_summary.json"
        )
        md_path = write_coverage_summary_markdown(
            summary, preset_diagnostics_dir / "coverage_summary.md"
        )
        result["diagnostics_artifacts"] = {
            "json": _artifact_pointer(json_path),
            "markdown": _artifact_pointer(md_path),
        }

    return result


def resolve_preset_run_specs(
    *,
    preset_keys: list[str] | None,
    config_path: str | None,
) -> list[PresetRunSpec]:
    """Resolve requested preset keys into concrete benchmark run specs."""

    return _resolve_preset_run_specs(
        preset_keys=preset_keys,
        config_path=config_path,
    )


def run_benchmark_suite(
    preset_specs: list[PresetRunSpec],
    *,
    suite: str,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    baseline_payload: dict[str, Any] | None,
    num_datasets_override: int | None,
    warmup_override: int | None,
    collect_memory: bool,
    collect_reproducibility: bool,
    collect_diagnostics: bool,
    diagnostics_root_dir: Path | None,
    fail_on_regression: bool,
    hardware_policy: str,
) -> dict[str, Any]:
    """Run a benchmark suite over one or more presets and attach regression diagnostics."""

    normalized_suite = suite.lower().strip()
    if normalized_suite not in {"smoke", "standard", "full"}:
        raise ValueError(f"Unsupported suite: {suite}")
    if collect_diagnostics and diagnostics_root_dir is None:
        raise ValueError("Benchmark diagnostics collection requires a diagnostics_root_dir.")

    include_micro = normalized_suite == "full"
    enable_repro = collect_reproducibility or normalized_suite == "full"
    key_totals: Counter[str] = Counter(spec.key for spec in preset_specs)
    key_seen: dict[str, int] = {}

    preset_results: list[dict[str, Any]] = []
    for spec in preset_specs:
        occurrence_index = key_seen.get(spec.key, 0)
        key_seen[spec.key] = occurrence_index + 1
        preset_results.append(
            run_preset_benchmark(
                spec,
                suite=normalized_suite,
                num_datasets_override=num_datasets_override,
                warmup_override=warmup_override,
                collect_memory=collect_memory,
                collect_reproducibility=enable_repro,
                collect_diagnostics=collect_diagnostics,
                diagnostics_root_dir=diagnostics_root_dir,
                warn_threshold_pct=float(warn_threshold_pct),
                fail_threshold_pct=float(fail_threshold_pct),
                include_micro=include_micro,
                hardware_policy=hardware_policy,
                diagnostics_occurrence_index=occurrence_index,
                diagnostics_occurrence_total=key_totals[spec.key],
            )
        )

    summary: dict[str, Any] = {
        "suite": normalized_suite,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "preset_results": preset_results,
    }

    if baseline_payload is not None:
        regression = compare_summary_to_baseline(
            summary,
            baseline_payload,
            warn_threshold_pct=warn_threshold_pct,
            fail_threshold_pct=fail_threshold_pct,
        )
    else:
        regression = {
            "status": "pass",
            "warn_threshold_pct": float(warn_threshold_pct),
            "fail_threshold_pct": float(fail_threshold_pct),
            "issues": [],
        }

    additional_issues = _collect_scenario_regression_issues(preset_results)
    if additional_issues:
        existing_issues = regression.get("issues", [])
        if not isinstance(existing_issues, list):
            existing_issues = []
        all_issues = [*existing_issues, *additional_issues]
        regression["issues"] = sorted(all_issues, key=_issue_sort_key)
        regression["status"] = _status_from_issues(regression["issues"])

    regression["fail_on_regression"] = bool(fail_on_regression)
    regression["hard_fail"] = bool(fail_on_regression and regression.get("status") == "fail")
    summary["regression"] = regression
    return summary
