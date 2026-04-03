"""Benchmark command handler."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dagzoo.bench.baseline import (
    build_baseline_payload,
    load_baseline,
    write_baseline,
)
from dagzoo.bench.report import write_suite_json, write_suite_markdown
from dagzoo.bench.suite import resolve_preset_run_specs, run_benchmark_suite
from dagzoo.config import GeneratorConfig

from ..common import load_config_or_usage_error, raise_usage_error
from ..effective_config import (
    print_effective_config_payload as emit_effective_config_payload,
)
from ..effective_config import (
    print_resolution_trace as emit_resolution_trace,
)
from ..effective_config import (
    write_effective_config_payload,
    write_effective_config_trace,
)


def _default_benchmark_config(config: str | None) -> GeneratorConfig:
    """Load benchmark defaults from custom config, falling back to dataclass defaults."""

    if config:
        return load_config_or_usage_error(config)
    return GeneratorConfig()


def _default_benchmark_artifact_dir() -> Path:
    """Return a timestamped benchmark artifact directory path."""

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    return Path("benchmarks") / "results" / timestamp


def _benchmark_artifact_dir(
    *,
    out_dir: str | Path | None,
    json_out: str | Path | None,
) -> Path | None:
    """Resolve optional output directory for benchmark summary artifacts."""

    if out_dir:
        return Path(out_dir)
    if json_out:
        return None
    return _default_benchmark_artifact_dir()


def _benchmark_diagnostics_root_dir(
    *,
    diagnostics: bool,
    diagnostics_out_dir: str | Path | None,
    artifact_dir: Path | None,
) -> Path | None:
    """Resolve diagnostics artifact root for benchmark preset coverage summaries."""

    if not bool(diagnostics):
        return None
    if diagnostics_out_dir:
        return Path(diagnostics_out_dir)
    if artifact_dir is not None:
        return artifact_dir
    return _default_benchmark_artifact_dir()


def _sanitize_preset_segment(preset_key: str) -> str:
    """Normalize preset key into a filesystem-safe segment."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(preset_key)).strip("._-")
    return normalized or "preset"


def _write_benchmark_effective_configs(
    summary: dict[str, Any], artifact_dir: Path
) -> tuple[list[Path], list[Path]]:
    """Persist per-preset effective config payloads and resolution traces."""

    preset_results = summary.get("preset_results", [])
    if not isinstance(preset_results, list):
        return [], []

    config_paths: list[Path] = []
    trace_paths: list[Path] = []
    key_counts: dict[str, int] = {}
    out_root = artifact_dir / "effective_configs"
    out_root.mkdir(parents=True, exist_ok=True)

    for idx, result in enumerate(preset_results):
        if not isinstance(result, dict):
            continue
        payload = result.get("effective_config")
        if not isinstance(payload, dict):
            continue
        key = _sanitize_preset_segment(str(result.get("preset_key", f"preset_{idx}")))
        key_counts[key] = key_counts.get(key, 0) + 1
        count = key_counts[key]
        suffix = f"_run{count}" if count > 1 else ""
        path = out_root / f"{key}{suffix}.yaml"
        write_effective_config_payload(payload, path)
        config_paths.append(path)

        trace_payload = result.get("effective_config_trace")
        if isinstance(trace_payload, list):
            trace_path = out_root / f"{key}{suffix}_trace.yaml"
            write_effective_config_trace(trace_payload, trace_path)
            trace_paths.append(trace_path)
    return config_paths, trace_paths


def _print_preset_result_line(result: dict[str, Any]) -> None:
    """Print one compact preset benchmark summary line."""

    scenarios = result.get("scenarios")
    if not isinstance(scenarios, dict):
        scenarios = {}

    def _scenario_hint(name: str, label: str) -> str:
        scenario = scenarios.get(name)
        if not isinstance(scenario, dict) or not bool(scenario.get("enabled")):
            return ""
        return f" {label}={scenario.get('status', 'pass')}"

    diagnostics_hint = ""
    artifacts = result.get("diagnostics_artifacts")
    if isinstance(artifacts, dict):
        json_pointer = artifacts.get("json")
        if isinstance(json_pointer, str) and json_pointer:
            diagnostics_hint = f" diagnostics={json_pointer}"

    filtering_hint = _scenario_hint("filtering", "filtering")
    missingness_hint = _scenario_hint("missingness", "missingness")
    shift_hint = _scenario_hint("shift", "shift")
    noise_hint = _scenario_hint("noise", "noise")
    throughput_hint = _scenario_hint("throughput", "throughput")

    stage_hint = (
        " "
        f"gen/min={float(result.get('generation_datasets_per_minute', 0.0)):.2f} "
        f"write/min={float(result.get('write_datasets_per_minute', 0.0)):.2f}"
    )
    filter_stage_dpm = result.get("filter_datasets_per_minute")
    filter_stage_hint = (
        f" filter/min={float(filter_stage_dpm):.2f}"
        if isinstance(filter_stage_dpm, (int, float))
        else " filter/min=-"
    )
    filter_accept_stage_dpm = result.get("filter_accepted_datasets_per_minute")
    filter_accept_stage_hint = (
        f" filter_accepted/min={float(filter_accept_stage_dpm):.2f}"
        if isinstance(filter_accept_stage_dpm, (int, float))
        else " filter_accepted/min=-"
    )
    filter_reject_ratio = result.get("filter_rejection_rate_attempt_level")
    filter_accept_dataset_ratio = result.get("filter_acceptance_rate_dataset_level")
    filter_reject_dataset_ratio = result.get("filter_rejection_rate_dataset_level")
    filter_retry_ratio = result.get("filter_retry_dataset_rate")
    filter_reject_hint = (
        f" filter_reject_attempt_pct={float(filter_reject_ratio) * 100.0:.2f}"
        if isinstance(filter_reject_ratio, (int, float))
        else " filter_reject_attempt_pct=-"
    )
    filter_accept_dataset_hint = (
        f" filter_accept_dataset_pct={float(filter_accept_dataset_ratio) * 100.0:.2f}"
        if isinstance(filter_accept_dataset_ratio, (int, float))
        else " filter_accept_dataset_pct=-"
    )
    filter_reject_dataset_hint = (
        f" filter_reject_dataset_pct={float(filter_reject_dataset_ratio) * 100.0:.2f}"
        if isinstance(filter_reject_dataset_ratio, (int, float))
        else " filter_reject_dataset_pct=-"
    )
    filter_retry_hint = (
        f" filter_retry_dataset_pct={float(filter_retry_ratio) * 100.0:.2f}"
        if isinstance(filter_retry_ratio, (int, float))
        else " filter_retry_dataset_pct=-"
    )
    latency_p95 = result.get("latency_p95_ms")
    latency_hint = (
        f"latency_p95_ms={float(latency_p95):.2f}"
        if isinstance(latency_p95, (int, float))
        else "latency_p95_ms=-"
    )

    print(
        f"[{result.get('preset_key')}] device={result.get('device')} "
        f"rows={result.get('dataset_rows_total', '-')} "
        f"mode={result.get('generation_mode', 'dynamic')} "
        f"backend={result.get('hardware_backend')} "
        f"datasets/min={float(result.get('datasets_per_minute', 0.0)):.2f} "
        f"{latency_hint}"
        f"{stage_hint}{filter_stage_hint}{filter_accept_stage_hint}{filter_reject_hint}"
        f"{filter_accept_dataset_hint}{filter_reject_dataset_hint}{filter_retry_hint}"
        f"{diagnostics_hint}{filtering_hint}{missingness_hint}{shift_hint}{noise_hint}{throughput_hint}"
    )


def run_benchmark_command(
    *,
    config: str | None = None,
    device: str | None = None,
    num_datasets: int | None = None,
    warmup: int | None = None,
    hardware_policy: str = "none",
    json_out: str | Path | None = None,
    suite: str | None = None,
    preset: Sequence[str] | None = None,
    baseline: str | Path | None = None,
    out_dir: str | Path | None = None,
    fail_on_regression: bool = False,
    warn_threshold_pct: float | None = None,
    fail_threshold_pct: float | None = None,
    no_memory: bool = False,
    collect_reproducibility: bool = False,
    save_baseline: str | Path | None = None,
    diagnostics: bool = False,
    diagnostics_out_dir: str | Path | None = None,
    print_effective_config: bool = False,
    print_resolution_trace: bool = False,
) -> int:
    """Execute the ``benchmark`` command."""

    artifact_dir = _benchmark_artifact_dir(out_dir=out_dir, json_out=json_out)
    diagnostics_root_dir = _benchmark_diagnostics_root_dir(
        diagnostics=diagnostics,
        diagnostics_out_dir=diagnostics_out_dir,
        artifact_dir=artifact_dir,
    )

    default_cfg = _default_benchmark_config(config)
    suite_name = (suite or default_cfg.benchmark.suite).strip().lower()
    warn_pct = (
        float(warn_threshold_pct)
        if warn_threshold_pct is not None
        else float(default_cfg.benchmark.warn_threshold_pct)
    )
    fail_pct = (
        float(fail_threshold_pct)
        if fail_threshold_pct is not None
        else float(default_cfg.benchmark.fail_threshold_pct)
    )

    preset_specs = resolve_preset_run_specs(
        preset_keys=list(preset) if preset else None,
        config_path=config,
    )
    if device and len(preset_specs) > 1:
        raise_usage_error(
            "benchmark --device cannot be combined with multiple --preset values; "
            "the override would be ambiguous."
        )
    effective_device_override = device if device and len(preset_specs) == 1 else None
    for spec in preset_specs:
        normalized_requested_device = (
            effective_device_override or spec.device or spec.config.runtime.device or "auto"
        ).lower()
        if effective_device_override is not None:
            spec.device = normalized_requested_device

    baseline_payload = load_baseline(baseline) if baseline else None

    try:
        summary = run_benchmark_suite(
            preset_specs,
            suite=suite_name,
            warn_threshold_pct=warn_pct,
            fail_threshold_pct=fail_pct,
            baseline_payload=baseline_payload,
            num_datasets_override=num_datasets,
            warmup_override=warmup,
            collect_memory=not bool(no_memory),
            collect_reproducibility=(
                bool(collect_reproducibility) or bool(default_cfg.benchmark.collect_reproducibility)
            ),
            collect_diagnostics=bool(diagnostics),
            diagnostics_root_dir=diagnostics_root_dir,
            fail_on_regression=bool(fail_on_regression),
            hardware_policy=str(hardware_policy),
        )
    except NotImplementedError as exc:
        raise_usage_error(str(exc))
    if print_effective_config:
        for result in summary.get("preset_results", []):
            if not isinstance(result, dict):
                continue
            payload = result.get("effective_config")
            if not isinstance(payload, dict):
                continue
            preset_key = str(result.get("preset_key", "unknown"))
            emit_effective_config_payload(payload, header=f"Effective config [{preset_key}]:")
    if print_resolution_trace:
        for result in summary.get("preset_results", []):
            if not isinstance(result, dict):
                continue
            trace_payload = result.get("effective_config_trace")
            if not isinstance(trace_payload, list):
                continue
            preset_key = str(result.get("preset_key", "unknown"))
            emit_resolution_trace(trace_payload, header=f"Resolution trace [{preset_key}]:")

    for result in summary.get("preset_results", []):
        _print_preset_result_line(result)

    regression = summary.get("regression", {})
    print(
        f"Regression status={regression.get('status', 'pass')} issues={len(regression.get('issues', []))}"
    )

    if artifact_dir is not None:
        json_path = write_suite_json(summary, artifact_dir / "summary.json")
        md_path = write_suite_markdown(summary, artifact_dir / "summary.md")
        effective_paths, trace_paths = _write_benchmark_effective_configs(summary, artifact_dir)
        if effective_paths or trace_paths:
            print(
                "Wrote benchmark effective configs under: "
                f"{(artifact_dir / 'effective_configs').resolve()}"
            )
        print(f"Wrote benchmark artifacts: {json_path} and {md_path}")

    if json_out:
        path = write_suite_json(summary, json_out)
        print(f"Wrote benchmark JSON: {path}")

    if save_baseline:
        payload = build_baseline_payload(summary)
        baseline_path = write_baseline(payload, save_baseline)
        print(f"Wrote benchmark baseline: {baseline_path}")

    hard_fail = bool(regression.get("hard_fail"))
    return 1 if hard_fail else 0
