"""Benchmark suite artifact writers (JSON and Markdown)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from dagzoo.math import sanitize_json as _sanitize_json


def write_suite_json(summary: dict[str, Any], out_path: str | Path) -> Path:
    """Write a suite summary as JSON."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_json(summary), f, indent=2, sort_keys=True)
    return path


def _format_float(value: Any, digits: int = 3) -> str:
    """Render floats consistently for markdown tables."""

    if not isinstance(value, (int, float)):
        return "-"
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _format_percent(value: Any, digits: int = 2) -> str:
    """Render fractional values as percentages for markdown tables."""

    if not isinstance(value, (int, float)):
        return "-"
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value) * 100.0:.{digits}f}"


def _format_match(value: Any) -> str:
    """Render benchmark reproducibility booleans for markdown tables."""

    if isinstance(value, bool):
        return "match" if value else "mismatch"
    return "-"


def _build_preset_table(preset_results: list[dict[str, Any]]) -> list[str]:
    """Create a markdown table summarizing per-preset performance metrics."""

    lines = [
        "| Preset | Rows | Mode | Device | Backend | Datasets/min | Gen/min | Write/min | Filter/min | Filter Accepted/min | Repro | Workload | Filter Reject % (attempt) | Filter Accept % (dataset) | Filter Reject % (dataset) | Filter Retry % (dataset) | Elapsed (s) | Latency p95 (ms) | Peak RSS (MB) | Diagnostics | Missingness | Lineage | Shift | Noise |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for result in preset_results:
        diagnostics_state = "on" if bool(result.get("diagnostics_enabled")) else "off"
        missingness_state = "off"
        guardrails = result.get("missingness_guardrails")
        if isinstance(guardrails, dict) and bool(guardrails.get("enabled")):
            missingness_state = str(guardrails.get("status", "pass"))
        lineage_state = "off"
        lineage_guardrails = result.get("lineage_guardrails")
        if isinstance(lineage_guardrails, dict) and bool(lineage_guardrails.get("enabled")):
            lineage_state = str(lineage_guardrails.get("status", "pass"))
        shift_state = "off"
        shift_guardrails = result.get("shift_guardrails")
        if isinstance(shift_guardrails, dict) and bool(shift_guardrails.get("enabled")):
            shift_state = str(shift_guardrails.get("status", "pass"))
        noise_state = "off"
        noise_guardrails = result.get("noise_guardrails")
        if isinstance(noise_guardrails, dict) and bool(noise_guardrails.get("enabled")):
            noise_state = str(noise_guardrails.get("status", "pass"))
        lines.append(
            "| "
            f"{result.get('preset_key', '-')} | "
            f"{result.get('dataset_rows_total', '-')} | "
            f"{result.get('generation_mode', 'dynamic')} | "
            f"{result.get('device', '-')} | "
            f"{result.get('hardware_backend', '-')} | "
            f"{_format_float(result.get('datasets_per_minute'), 2)} | "
            f"{_format_float(result.get('generation_datasets_per_minute'), 2)} | "
            f"{_format_float(result.get('write_datasets_per_minute'), 2)} | "
            f"{_format_float(result.get('filter_datasets_per_minute'), 2)} | "
            f"{_format_float(result.get('filter_accepted_datasets_per_minute'), 2)} | "
            f"{_format_match(result.get('reproducibility_match'))} | "
            f"{_format_match(result.get('reproducibility_workload_match'))} | "
            f"{_format_percent(result.get('filter_rejection_rate_attempt_level'), 2)} | "
            f"{_format_percent(result.get('filter_acceptance_rate_dataset_level'), 2)} | "
            f"{_format_percent(result.get('filter_rejection_rate_dataset_level'), 2)} | "
            f"{_format_percent(result.get('filter_retry_dataset_rate'), 2)} | "
            f"{_format_float(result.get('elapsed_seconds'), 3)} | "
            f"{_format_float(result.get('latency_p95_ms'), 2)} | "
            f"{_format_float(result.get('peak_rss_mb'), 2)} | "
            f"{diagnostics_state} |"
            f" {missingness_state} |"
            f" {lineage_state} |"
            f" {shift_state} |"
            f" {noise_state} |"
        )
    return lines


def _build_diagnostics_table(preset_results: list[dict[str, Any]]) -> list[str]:
    """Create a markdown table with per-preset diagnostics artifact pointers."""

    lines = [
        "| Preset | Coverage JSON | Coverage Markdown |",
        "|---|---|---|",
    ]
    for result in preset_results:
        artifacts = result.get("diagnostics_artifacts")
        json_path = "-"
        md_path = "-"
        if isinstance(artifacts, dict):
            json_value = artifacts.get("json")
            md_value = artifacts.get("markdown")
            if isinstance(json_value, str) and json_value:
                json_path = f"`{json_value}`"
            if isinstance(md_value, str) and md_value:
                md_path = f"`{md_value}`"
        lines.append(f"| {result.get('preset_key', '-')} | {json_path} | {md_path} |")
    return lines


def _build_bottleneck_evidence_lines(preset_results: list[dict[str, Any]]) -> list[str]:
    """Create a compact markdown summary of bottleneck evidence per preset."""

    lines = ["## Bottleneck Evidence", ""]
    for result in preset_results:
        lines.append(f"### {result.get('preset_key', '-')}")
        lines.append(
            "- Generation: "
            f"`wall={_format_float(result.get('generation_elapsed_seconds'), 3)}s`, "
            f"`cpu={_format_float(result.get('generation_cpu_time_seconds'), 3)}s`, "
            f"`cpu_busy_pct={_format_float(result.get('generation_cpu_busy_pct_of_wall'), 2)}`"
        )
        lines.append(
            "- Fixed layout: "
            f"`target_cells={result.get('fixed_layout_target_cells_effective', '-')}`, "
            f"`per_dataset_cells={result.get('fixed_layout_per_dataset_cells', '-')}`, "
            f"`batch={result.get('fixed_layout_realized_batch_size', '-')}`, "
            f"`chunks={result.get('fixed_layout_chunk_count', '-')}`, "
            f"`tail={result.get('fixed_layout_tail_chunk_size', '-')}`"
        )
        lines.append(
            "- Write replay: "
            f"`sample_datasets={result.get('stage_sample_datasets', '-')}`, "
            f"`wall={_format_float(result.get('write_stage_elapsed_seconds'), 3)}s`, "
            f"`cpu={_format_float(result.get('write_stage_cpu_time_seconds'), 3)}s`, "
            f"`bytes={result.get('write_stage_bytes_written', '-')}`, "
            f"`mib_per_s={_format_float(result.get('write_stage_mib_per_second'), 2)}`"
        )
        filter_elapsed = result.get("filter_stage_elapsed_seconds")
        filter_cpu = result.get("filter_stage_cpu_time_seconds")
        if filter_elapsed is None and filter_cpu is None:
            lines.append("- Filter replay: `disabled`")
        else:
            lines.append(
                "- Filter replay: "
                f"`wall={_format_float(filter_elapsed, 3)}s`, "
                f"`cpu={_format_float(filter_cpu, 3)}s`, "
                f"`datasets_per_min={_format_float(result.get('filter_datasets_per_minute'), 2)}`"
            )
        peak_reserved = result.get("peak_cuda_reserved_mb")
        peak_reserved_pct = result.get("peak_cuda_reserved_pct_of_total_memory")
        peak_headroom = result.get("peak_cuda_headroom_mb")
        if peak_reserved is not None or peak_headroom is not None:
            lines.append(
                "- CUDA memory: "
                f"`reserved_mb={_format_float(peak_reserved, 2)}`, "
                f"`reserved_pct={_format_float(peak_reserved_pct, 2)}`, "
                f"`headroom_mb={_format_float(peak_headroom, 2)}`"
            )
        lines.append("")
    return lines


def write_suite_markdown(summary: dict[str, Any], out_path: str | Path) -> Path:
    """Write a concise markdown report for one benchmark suite run."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Benchmark Suite Report",
        "",
        f"- Suite: `{summary.get('suite', '-')}`",
        f"- Generated at: `{summary.get('generated_at', '-')}`",
    ]

    regression = summary.get("regression", {})
    if isinstance(regression, dict):
        lines.append(f"- Regression status: `{regression.get('status', 'pass')}`")
    lines.append("")

    preset_results = summary.get("preset_results", [])
    if isinstance(preset_results, list) and preset_results:
        lines.append("## Presets")
        lines.extend(_build_preset_table(preset_results))
        lines.append("")
        lines.extend(_build_bottleneck_evidence_lines(preset_results))
        if any(bool(result.get("diagnostics_enabled")) for result in preset_results):
            lines.append("## Diagnostics Artifacts")
            lines.extend(_build_diagnostics_table(preset_results))
            lines.append("")

    if isinstance(regression, dict) and regression.get("issues"):
        lines.append("## Regression Issues")
        lines.append("| Severity | Preset | Metric | Current | Baseline | Degradation % |")
        lines.append("|---|---|---|---:|---:|---:|")
        for issue in regression["issues"]:
            lines.append(
                "| "
                f"{issue.get('severity')} | "
                f"{issue.get('preset')} | "
                f"{issue.get('metric')} | "
                f"{_format_float(issue.get('current'), 3)} | "
                f"{_format_float(issue.get('baseline'), 3)} | "
                f"{_format_float(issue.get('degradation_pct'), 2)} |"
            )
        lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path
