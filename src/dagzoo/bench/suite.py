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
    _compose_bundle_callback,
    _MissingnessAcceptanceCollector,
    _NoiseGuardrailCollector,
    _ShiftGuardrailCollector,
    _ThroughputPressureCollector,
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
    _collect_guardrail_regression_issues,
    _collect_lineage_guardrails,
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
    _is_missingness_enabled,
    _is_noise_enabled,
    _is_shift_enabled,
    _latency_sample_count,
    _peak_rss_mb,
    _preset_counts,
    _sanitize_preset_key,
    _synchronize_accelerator,
)
from dagzoo.bench.stage_metrics import (
    StageSampleCollector,
    measure_filter_stage_metrics,
    measure_write_stage_metrics,
)
from dagzoo.bench.throughput import _throughput_measure_seed, run_throughput_benchmark
from dagzoo.config import (
    MISSINGNESS_MECHANISM_NONE,
    NOISE_FAMILY_GAUSSIAN,
    SHIFT_MODE_OFF,
    GeneratorConfig,
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
    """Run one benchmark preset and collect throughput, latency, and optional diagnostics."""

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
    pre_realization_config = _copy_runtime_config(resolved_preset.config)
    trace_events = list(resolved_preset.trace_events)
    if suite == "smoke":
        _cap_smoke_rows_spec(pre_realization_config)
        append_config_diff_events(
            resolved_preset.config,
            pre_realization_config,
            source="benchmark.smoke_rows_cap",
            events=trace_events,
        )
    config, _run_seed, requested_device, _resolved_device = realize_generation_config_for_run(
        pre_realization_config,
        seed=resolved_preset.config.seed,
        device=resolved_preset.requested_device,
    )
    append_config_diff_events(
        pre_realization_config,
        config,
        source="benchmark.run_realization",
        events=trace_events,
    )
    hw = resolved_preset.hardware
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

    missingness_enabled = _is_missingness_enabled(config)
    missingness_acceptance = (
        _MissingnessAcceptanceCollector(target_rate=float(config.dataset.missing_rate))
        if missingness_enabled
        else None
    )
    shift_enabled = _is_shift_enabled(config)
    shift_guardrails = _ShiftGuardrailCollector() if shift_enabled else None
    noise_enabled = _is_noise_enabled(config)
    noise_guardrails = (
        _NoiseGuardrailCollector(expected_family_requested=str(config.noise.family))
        if noise_enabled
        else None
    )
    stage_sample_cap = _latency_sample_count(config, suite, num_datasets)

    generation_config = _copy_runtime_config(config)
    generation_config.filter.enabled = False
    fixed_layout_evidence = _build_fixed_layout_evidence(
        generation_config,
        device=requested_device,
        num_datasets=num_datasets,
    )

    def _build_throughput_on_bundle_callback(
        *,
        diagnostics_aggregator: CoverageAggregator | None,
        missingness_acceptance: _MissingnessAcceptanceCollector | None,
        shift_guardrails: _ShiftGuardrailCollector | None,
        noise_guardrails: _NoiseGuardrailCollector | None,
    ) -> tuple[
        Callable[[DatasetBundle], None],
        StageSampleCollector,
        _ThroughputPressureCollector,
    ]:
        throughput_pressure = _ThroughputPressureCollector()
        stage_sample_collector = StageSampleCollector(max_samples=stage_sample_cap)
        composed_on_bundle_callback = _compose_bundle_callback(
            diagnostics_aggregator=diagnostics_aggregator,
            missingness_acceptance=missingness_acceptance,
            shift_guardrails=shift_guardrails,
            noise_guardrails=noise_guardrails,
            throughput_pressure=throughput_pressure,
        )

        def on_bundle_callback(bundle: DatasetBundle) -> None:
            stage_sample_collector.update(bundle)
            if composed_on_bundle_callback is not None:
                composed_on_bundle_callback(bundle)

        return on_bundle_callback, stage_sample_collector, throughput_pressure

    (
        on_bundle_callback,
        stage_sample_collector,
        throughput_pressure,
    ) = _build_throughput_on_bundle_callback(
        diagnostics_aggregator=diagnostics_aggregator,
        missingness_acceptance=missingness_acceptance,
        shift_guardrails=shift_guardrails,
        noise_guardrails=noise_guardrails,
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
    sampled_bundles = stage_sample_collector.bundles
    stage_sample_datasets = len(sampled_bundles)
    write_stage_measurement = (
        measure_write_stage_metrics(sampled_bundles, config=config)
        if sampled_bundles
        else measure_write_stage_metrics([], config=config)
    )
    write_dpm = float(getattr(write_stage_measurement, "datasets_per_minute", 0.0))
    filter_stage_enabled = bool(config.filter.enabled)
    filter_dpm: float | None
    if filter_stage_enabled:
        filter_stage_measurement = (
            measure_filter_stage_metrics(sampled_bundles, config=config)
            if sampled_bundles
            else None
        )
        filter_dpm = (
            float(filter_stage_measurement.datasets_per_minute)
            if filter_stage_measurement is not None
            else 0.0
        )
    else:
        filter_stage_measurement = None
        filter_dpm = None
    throughput_pressure_summary = throughput_pressure.build_summary()
    filter_attempts_total = int(throughput_pressure_summary["filter_attempts_total"])
    filter_rejections_total = int(throughput_pressure_summary["filter_rejections_total"])
    filter_retry_dataset_count = int(throughput_pressure_summary["filter_retry_dataset_count"])
    filter_retry_dataset_denominator = int(throughput_pressure_summary["datasets_seen"])
    filter_accepted_datasets_measured = 0
    filter_rejected_datasets_measured = 0
    if filter_stage_measurement is not None:
        filter_attempts_total = int(filter_stage_measurement.filter_attempts_total)
        filter_rejections_total = int(filter_stage_measurement.filter_rejections_total)
        filter_retry_dataset_count = int(filter_stage_measurement.filter_rejections_total)
        filter_retry_dataset_denominator = int(stage_sample_datasets)
        filter_accepted_datasets_measured = int(filter_stage_measurement.filter_accepted_datasets)
        filter_rejected_datasets_measured = int(filter_stage_measurement.filter_rejected_datasets)
    filter_rejection_rate_attempt_level = (
        float(filter_rejections_total) / float(filter_attempts_total)
        if filter_attempts_total > 0
        else None
    )
    filter_dataset_yield_total = (
        filter_accepted_datasets_measured + filter_rejected_datasets_measured
    )
    filter_acceptance_rate_dataset_level = (
        float(filter_accepted_datasets_measured) / float(filter_dataset_yield_total)
        if filter_dataset_yield_total > 0
        else None
    )
    filter_accepted_datasets_per_minute = (
        float(filter_dpm) * float(filter_acceptance_rate_dataset_level)
        if filter_dpm is not None and filter_acceptance_rate_dataset_level is not None
        else None
    )
    filter_rejection_rate_dataset_level = (
        float(filter_rejected_datasets_measured) / float(filter_dataset_yield_total)
        if filter_dataset_yield_total > 0
        else None
    )
    filter_retry_dataset_rate = (
        float(filter_retry_dataset_count) / float(filter_retry_dataset_denominator)
        if filter_retry_dataset_denominator > 0 and filter_attempts_total > 0
        else None
    )

    generation_dpm = float(result.get("datasets_per_minute", 0.0))
    mean_attempts_per_dataset = float(throughput_pressure_summary["attempts_per_dataset_mean"])

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
    result["effective_config"] = config.to_dict()
    result["effective_config_trace"] = serialize_resolution_events(trace_events)
    result["diagnostics_enabled"] = diagnostics_enabled
    result["diagnostics_artifacts"] = None
    result["generation_datasets_per_minute"] = generation_dpm
    result["generation_elapsed_seconds"] = generation_elapsed_seconds
    result["generation_cpu_time_seconds"] = generation_cpu_time_seconds
    result["generation_cpu_busy_pct_of_wall"] = _cpu_busy_pct_of_wall(
        cpu_time_seconds=generation_cpu_time_seconds,
        elapsed_seconds=generation_elapsed_seconds,
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
    result["total_attempts"] = int(throughput_pressure_summary["attempts_total"])
    result["mean_attempts_per_dataset"] = mean_attempts_per_dataset
    result["retry_dataset_count"] = int(throughput_pressure_summary["retry_dataset_count"])
    result["retry_dataset_rate"] = throughput_pressure_summary["retry_dataset_rate"]
    result["filter_attempts_total"] = int(filter_attempts_total)
    result["filter_rejections_total"] = int(filter_rejections_total)
    result["filter_accepted_datasets_measured"] = int(filter_accepted_datasets_measured)
    result["filter_rejected_datasets_measured"] = int(filter_rejected_datasets_measured)
    result["filter_acceptance_rate_dataset_level"] = filter_acceptance_rate_dataset_level
    result["filter_rejection_rate_dataset_level"] = filter_rejection_rate_dataset_level
    result["filter_rejection_rate_attempt_level"] = filter_rejection_rate_attempt_level
    result["filter_retry_dataset_count"] = filter_retry_dataset_count
    result["filter_retry_dataset_rate"] = filter_retry_dataset_rate
    result["estimated_attempts_per_minute"] = generation_dpm * mean_attempts_per_dataset
    result.update(fixed_layout_evidence)
    result["missingness_guardrails"] = {"enabled": False}
    result["lineage_guardrails"] = {"enabled": False}
    result["shift_guardrails"] = {"enabled": False}
    result["noise_guardrails"] = {"enabled": False}

    # Stage probes are complete; release retained bundles before latency and
    # control-run guardrail benchmarks to avoid unnecessary memory retention.
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

    if missingness_enabled and missingness_acceptance is not None:
        baseline_config = _copy_runtime_config(generation_config)
        baseline_config.dataset.missing_rate = 0.0
        baseline_config.dataset.missing_mechanism = MISSINGNESS_MECHANISM_NONE
        missingness_baseline_diagnostics_aggregator: CoverageAggregator | None = None
        if diagnostics_aggregator is not None:
            missingness_baseline_diagnostics_aggregator = _build_diagnostics_aggregator(
                baseline_config
            )
        baseline_missingness_acceptance = _MissingnessAcceptanceCollector(
            target_rate=float(config.dataset.missing_rate)
        )
        # Keep control-run callback instrumentation equivalent so runtime delta
        # reflects missingness overhead instead of callback overhead skew.
        (
            baseline_on_bundle_callback,
            baseline_stage_sample_collector,
            _,
        ) = _build_throughput_on_bundle_callback(
            diagnostics_aggregator=missingness_baseline_diagnostics_aggregator,
            missingness_acceptance=baseline_missingness_acceptance,
            shift_guardrails=_ShiftGuardrailCollector() if shift_enabled else None,
            noise_guardrails=(
                _NoiseGuardrailCollector(expected_family_requested=str(config.noise.family))
                if noise_enabled
                else None
            ),
        )

        baseline_throughput = run_throughput_benchmark(
            baseline_config,
            num_datasets=num_datasets,
            warmup_datasets=warmup,
            device=requested_device,
            on_bundle=baseline_on_bundle_callback,
        )
        baseline_stage_sample_collector.bundles.clear()
        baseline_dpm = float(baseline_throughput.get("datasets_per_minute", 0.0))
        current_dpm = float(result.get("datasets_per_minute", 0.0))
        runtime_degradation = degradation_percent("datasets_per_minute", current_dpm, baseline_dpm)
        runtime_degradation_value = (
            float(runtime_degradation) if runtime_degradation is not None else 0.0
        )
        runtime_severity = _severity_from_thresholds(
            runtime_degradation_value,
            warn=float(warn_threshold_pct),
            fail=float(fail_threshold_pct),
        )

        acceptance_summary = missingness_acceptance.build_summary()
        issues = list(acceptance_summary["issues"])
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

        result["missingness_guardrails"] = {
            "enabled": True,
            "mechanism": str(config.dataset.missing_mechanism),
            "target_rate": float(config.dataset.missing_rate),
            "metadata_coverage_rate": float(acceptance_summary["metadata_coverage_rate"]),
            "realized_rate_overall": float(acceptance_summary["realized_rate_overall"]),
            "rate_abs_error": float(acceptance_summary["rate_abs_error"]),
            "rate_warn_abs_error": float(MISSINGNESS_RATE_WARN_ABS_ERROR),
            "rate_fail_abs_error": float(MISSINGNESS_RATE_FAIL_ABS_ERROR),
            "runtime_baseline_datasets_per_minute": baseline_dpm,
            "runtime_degradation_pct": (
                float(runtime_degradation) if runtime_degradation is not None else None
            ),
            "runtime_warn_threshold_pct": float(warn_threshold_pct),
            "runtime_fail_threshold_pct": float(fail_threshold_pct),
            "issues": issues,
            "status": _status_from_issues(issues),
        }

    if shift_enabled and shift_guardrails is not None:
        shift_params = resolve_shift_runtime_params(config)
        baseline_config = _copy_runtime_config(generation_config)
        baseline_config.shift.enabled = False
        baseline_config.shift.mode = SHIFT_MODE_OFF
        baseline_config.shift.graph_scale = None
        baseline_config.shift.mechanism_scale = None
        baseline_config.shift.variance_scale = None
        shift_baseline_diagnostics_aggregator: CoverageAggregator | None = None
        if diagnostics_aggregator is not None:
            shift_baseline_diagnostics_aggregator = _build_diagnostics_aggregator(baseline_config)
        shift_baseline_missingness_acceptance = (
            _MissingnessAcceptanceCollector(target_rate=float(config.dataset.missing_rate))
            if missingness_acceptance is not None
            else None
        )
        baseline_shift_guardrails = _ShiftGuardrailCollector()
        (
            baseline_on_bundle_callback,
            baseline_stage_sample_collector,
            _,
        ) = _build_throughput_on_bundle_callback(
            diagnostics_aggregator=shift_baseline_diagnostics_aggregator,
            missingness_acceptance=shift_baseline_missingness_acceptance,
            shift_guardrails=baseline_shift_guardrails,
            noise_guardrails=(
                _NoiseGuardrailCollector(expected_family_requested=str(config.noise.family))
                if noise_enabled
                else None
            ),
        )

        baseline_throughput = run_throughput_benchmark(
            baseline_config,
            num_datasets=num_datasets,
            warmup_datasets=warmup,
            device=requested_device,
            on_bundle=baseline_on_bundle_callback,
        )
        baseline_stage_sample_collector.bundles.clear()
        baseline_dpm = float(baseline_throughput.get("datasets_per_minute", 0.0))
        current_dpm = float(result.get("datasets_per_minute", 0.0))
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

        current_summary = shift_guardrails.build_summary()
        baseline_summary = baseline_shift_guardrails.build_summary()

        shift_issues: list[dict[str, Any]] = []
        metadata_coverage_rate = float(current_summary["metadata_coverage_rate"])
        if metadata_coverage_rate < 1.0:
            shift_issues.append(
                _build_guardrail_issue(
                    metric="shift_metadata_coverage",
                    severity="fail",
                    current=metadata_coverage_rate,
                    baseline=1.0,
                    degradation_pct=float(max(0.0, (1.0 - metadata_coverage_rate) * 100.0)),
                    detail="Shift metadata must be present for all shift-enabled bundles.",
                )
            )
        shift_enabled_coverage_rate = float(current_summary["shift_enabled_coverage_rate"])
        if shift_enabled_coverage_rate < 1.0:
            shift_issues.append(
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
            shift_issues.append(
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
            current=current_summary.get("mean_graph_edge_density"),
            baseline=baseline_summary.get("mean_graph_edge_density"),
            detail=(
                "Graph shift should increase mean graph edge density "
                "relative to a shift-disabled control run."
            ),
        )
        mechanism_check, mechanism_issue = _build_shift_directional_check(
            metric="mechanism_nonlinear_mass",
            enabled=float(shift_params.mechanism_scale) > 0.0,
            gating_enabled=directional_gating_enabled,
            current=current_summary.get("mean_mechanism_nonlinear_mass"),
            baseline=baseline_summary.get("mean_mechanism_nonlinear_mass"),
            detail=(
                "Mechanism shift should increase nonlinear family mass "
                "relative to a shift-disabled control run."
            ),
        )
        noise_check, noise_issue = _build_shift_directional_check(
            metric="noise_variance_multiplier",
            enabled=float(shift_params.variance_scale) > 0.0,
            gating_enabled=directional_gating_enabled,
            current=current_summary.get("mean_noise_variance_multiplier"),
            baseline=baseline_summary.get("mean_noise_variance_multiplier"),
            detail=(
                "Noise shift should increase noise variance multiplier "
                "relative to a shift-disabled control run."
            ),
        )
        for maybe_issue in (graph_issue, mechanism_issue, noise_issue):
            if maybe_issue is not None:
                shift_issues.append(maybe_issue)

        result["shift_guardrails"] = {
            "enabled": True,
            "mode": str(shift_params.mode),
            "graph_scale": float(shift_params.graph_scale),
            "mechanism_scale": float(shift_params.mechanism_scale),
            "variance_scale": float(shift_params.variance_scale),
            "sample_datasets": int(num_datasets),
            "metadata_coverage_rate": metadata_coverage_rate,
            "shift_enabled_coverage_rate": shift_enabled_coverage_rate,
            "runtime_gating_enabled": bool(runtime_gating_enabled),
            "runtime_gating_min_sample_datasets": int(SHIFT_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE),
            "runtime_gating_suppressed_reason": (
                None if runtime_gating_enabled else "insufficient_sample_size"
            ),
            "directional_gating_enabled": bool(directional_gating_enabled),
            "directional_gating_min_sample_datasets": int(
                SHIFT_GUARDRAIL_DIRECTIONAL_GATING_MIN_SAMPLE
            ),
            "directional_gating_suppressed_reason": (
                None if directional_gating_enabled else "insufficient_sample_size"
            ),
            "current_means": {
                "graph_edge_density": current_summary.get("mean_graph_edge_density"),
                "edge_odds_multiplier": current_summary.get("mean_edge_odds_multiplier"),
                "mechanism_nonlinear_mass": current_summary.get("mean_mechanism_nonlinear_mass"),
                "noise_variance_multiplier": current_summary.get("mean_noise_variance_multiplier"),
            },
            "baseline_means": {
                "graph_edge_density": baseline_summary.get("mean_graph_edge_density"),
                "edge_odds_multiplier": baseline_summary.get("mean_edge_odds_multiplier"),
                "mechanism_nonlinear_mass": baseline_summary.get("mean_mechanism_nonlinear_mass"),
                "noise_variance_multiplier": baseline_summary.get("mean_noise_variance_multiplier"),
            },
            "directional_checks": {
                "graph_edge_density": graph_check,
                "mechanism_nonlinear_mass": mechanism_check,
                "noise_variance_multiplier": noise_check,
            },
            "runtime_baseline_datasets_per_minute": baseline_dpm,
            "runtime_with_shift_datasets_per_minute": current_dpm,
            "runtime_degradation_pct": (
                float(runtime_degradation) if runtime_degradation is not None else None
            ),
            "runtime_warn_threshold_pct": float(warn_threshold_pct),
            "runtime_fail_threshold_pct": float(fail_threshold_pct),
            "issues": shift_issues,
            "status": _status_from_issues(shift_issues),
        }

    if noise_enabled and noise_guardrails is not None:
        baseline_config = _copy_runtime_config(generation_config)
        baseline_config.noise.family = NOISE_FAMILY_GAUSSIAN
        baseline_config.noise.base_scale = 1.0
        baseline_config.noise.student_t_df = 5.0
        baseline_config.noise.mixture_weights = None
        noise_baseline_diagnostics_aggregator: CoverageAggregator | None = None
        if diagnostics_aggregator is not None:
            noise_baseline_diagnostics_aggregator = _build_diagnostics_aggregator(baseline_config)
        noise_baseline_missingness_acceptance = (
            _MissingnessAcceptanceCollector(target_rate=float(config.dataset.missing_rate))
            if missingness_acceptance is not None
            else None
        )
        noise_baseline_shift_guardrails = _ShiftGuardrailCollector() if shift_enabled else None
        baseline_noise_guardrails = _NoiseGuardrailCollector(
            expected_family_requested=str(baseline_config.noise.family)
        )
        (
            baseline_on_bundle_callback,
            baseline_stage_sample_collector,
            _,
        ) = _build_throughput_on_bundle_callback(
            diagnostics_aggregator=noise_baseline_diagnostics_aggregator,
            missingness_acceptance=noise_baseline_missingness_acceptance,
            shift_guardrails=noise_baseline_shift_guardrails,
            noise_guardrails=baseline_noise_guardrails,
        )
        baseline_throughput = run_throughput_benchmark(
            baseline_config,
            num_datasets=num_datasets,
            warmup_datasets=warmup,
            device=requested_device,
            on_bundle=baseline_on_bundle_callback,
        )
        baseline_stage_sample_collector.bundles.clear()

        baseline_dpm = float(baseline_throughput.get("datasets_per_minute", 0.0))
        current_dpm = float(result.get("datasets_per_minute", 0.0))
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

        current_summary = noise_guardrails.build_summary()
        baseline_summary = baseline_noise_guardrails.build_summary()
        noise_issues = list(current_summary["issues"])
        if runtime_gating_enabled and runtime_severity != "pass":
            noise_issues.append(
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

        result["noise_guardrails"] = {
            "enabled": True,
            "family_requested": str(config.noise.family),
            "sample_datasets": int(num_datasets),
            "metadata_coverage_rate": float(current_summary["metadata_coverage_rate"]),
            "metadata_valid_rate": float(current_summary["metadata_valid_rate"]),
            "valid_metadata_count": int(current_summary["valid_metadata_count"]),
            "sampled_family_counts": dict(current_summary["sampled_family_counts"]),
            "invalid_reason_counts": dict(current_summary["invalid_reason_counts"]),
            "runtime_gating_enabled": bool(runtime_gating_enabled),
            "runtime_gating_min_sample_datasets": int(NOISE_GUARDRAIL_RUNTIME_GATING_MIN_SAMPLE),
            "runtime_gating_suppressed_reason": (
                None if runtime_gating_enabled else "insufficient_sample_size"
            ),
            "runtime_baseline_family_requested": str(baseline_config.noise.family),
            "runtime_baseline_sampled_family_counts": dict(
                baseline_summary["sampled_family_counts"]
            ),
            "runtime_baseline_datasets_per_minute": baseline_dpm,
            "runtime_with_noise_datasets_per_minute": current_dpm,
            "runtime_degradation_pct": (
                float(runtime_degradation) if runtime_degradation is not None else None
            ),
            "runtime_warn_threshold_pct": float(warn_threshold_pct),
            "runtime_fail_threshold_pct": float(fail_threshold_pct),
            "issues": noise_issues,
            "status": _status_from_issues(noise_issues),
        }

    result["lineage_guardrails"] = _collect_lineage_guardrails(
        generation_config,
        suite=suite,
        num_datasets=num_datasets,
        device=requested_device,
        warn_threshold_pct=float(warn_threshold_pct),
        fail_threshold_pct=float(fail_threshold_pct),
    )

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

    regression: dict[str, Any]
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

    missingness_issues = _collect_guardrail_regression_issues(
        preset_results, guardrail_key="missingness_guardrails"
    )
    lineage_issues = _collect_guardrail_regression_issues(
        preset_results, guardrail_key="lineage_guardrails"
    )
    shift_issues = _collect_guardrail_regression_issues(
        preset_results, guardrail_key="shift_guardrails"
    )
    noise_issues = _collect_guardrail_regression_issues(
        preset_results, guardrail_key="noise_guardrails"
    )
    additional_issues = [*missingness_issues, *lineage_issues, *shift_issues, *noise_issues]
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
