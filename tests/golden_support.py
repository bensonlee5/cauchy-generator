from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mdformat

GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"

_ABSOLUTE_PATH_PLACEHOLDER = "<ABSOLUTE_PATH>"
_TIMESTAMP_PLACEHOLDER = "<TIMESTAMP>"
_RUN_ID_PLACEHOLDER = "<RUN_ID>"
_SHA256_PLACEHOLDER = "<SHA256>"
_SECONDS_PLACEHOLDER = "<SECONDS>"
_RATE_PLACEHOLDER = "<RATE>"
_MS_PLACEHOLDER = "<MS>"
_MB_PLACEHOLDER = "<MB>"
_PERCENT_PLACEHOLDER = "<PERCENT>"

_HEX_ID_RE = re.compile(r"\b[0-9a-f]{32}\b")
_ISO_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}T[0-9:.\-+Z]+\b")


def load_golden_text(name: str) -> str:
    return (GOLDENS_DIR / name).read_text(encoding="utf-8")


def load_golden_json(name: str) -> Any:
    return json.loads(load_golden_text(name))


def assert_normalized_json_equal(
    actual: Any,
    golden_name: str,
    *,
    normalizer,
) -> None:
    assert normalizer(actual) == load_golden_json(golden_name)


def assert_normalized_text_equal(
    actual: str,
    golden_name: str,
    *,
    normalizer,
) -> None:
    assert normalizer(actual) == load_golden_text(golden_name)


def normalize_benchmark_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return _normalize_json(summary, key_path=())


def normalize_benchmark_summary_markdown(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.strip().splitlines():
        line = raw_line.rstrip()
        if line.startswith("- Generated at:"):
            lines.append("- Generated at: `<TIMESTAMP>`")
            continue
        if line.startswith("| ") and line.endswith(" |"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == 24 and cells[0] != "Preset":
                cells[5] = _RATE_PLACEHOLDER
                cells[6] = _RATE_PLACEHOLDER
                cells[7] = _RATE_PLACEHOLDER
                cells[16] = _SECONDS_PLACEHOLDER
                cells[17] = _MS_PLACEHOLDER
                cells[18] = _MB_PLACEHOLDER
                lines.append("| " + " | ".join(cells) + " |")
                continue
            if len(cells) == 3 and cells[0] != "Preset":
                cells[1] = f"`{_ABSOLUTE_PATH_PLACEHOLDER}`"
                cells[2] = f"`{_ABSOLUTE_PATH_PLACEHOLDER}`"
                lines.append("| " + " | ".join(cells) + " |")
                continue
        if line.startswith("- Preparation:"):
            lines.append(
                "- Preparation: `wall=<SECONDS>s`, `cpu=<SECONDS>s`, `cpu_busy_pct=<PERCENT>`"
            )
            continue
        if line.startswith("- Generation:"):
            lines.append(
                "- Generation: `wall=<SECONDS>s`, `cpu=<SECONDS>s`, `cpu_busy_pct=<PERCENT>`"
            )
            continue
        if line.startswith("- Raw batch:"):
            lines.append(
                "- Raw batch: `wall=<SECONDS>s`, `cpu=<SECONDS>s`, `node_apply_wall=<SECONDS>s`, "
                "`converter_wall=<SECONDS>s`, `feature_wall=<SECONDS>s`"
            )
            continue
        if line.startswith("- Fixed layout:"):
            lines.append(
                "- Fixed layout: `target_cells=1024`, `per_dataset_cells=32`, `batch=2`, "
                "`chunks=1`, `tail=0`"
            )
            continue
        if line.startswith("- Write replay:"):
            lines.append(
                "- Write replay: `sample_datasets=2`, `wall=<SECONDS>s`, `cpu=<SECONDS>s`, "
                "`bytes=4096`, `mib_per_s=<RATE>`"
            )
            continue
        if line.startswith("- Filter replay:"):
            lines.append("- Filter replay: `disabled`")
            continue
        if line.startswith("- CUDA memory:"):
            lines.append(
                "- CUDA memory: `reserved_mb=<MB>`, `reserved_pct=<PERCENT>`, `headroom_mb=<MB>`"
            )
            continue
        lines.append(_normalize_inline_paths(line))
    return mdformat.text("\n".join(lines).rstrip() + "\n", extensions=("gfm",))


def normalize_handoff_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_json(payload, key_path=())
    assert isinstance(normalized, dict)
    return normalized


def normalize_coverage_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    linearity = metrics.get("linearity_proxy", {}) if isinstance(metrics, dict) else {}
    wins = metrics.get("wins_ratio_proxy", {}) if isinstance(metrics, dict) else {}
    if not isinstance(linearity, dict):
        linearity = {}
    if not isinstance(wins, dict):
        wins = {}

    return {
        "generated_at": _TIMESTAMP_PLACEHOLDER,
        "num_datasets": summary.get("num_datasets", 0),
        "task_counts": summary.get("task_counts", {}),
        "histogram_bins": summary.get("histogram_bins", 0),
        "quantiles": summary.get("quantiles", []),
        "max_values_per_metric": summary.get("max_values_per_metric"),
        "mechanism_family_summary": _project_mechanism_family_summary(summary),
        "metrics": {
            "linearity_proxy": _project_metric(linearity),
            "wins_ratio_proxy": _project_metric(wins),
        },
    }


def _project_mechanism_family_summary(summary: dict[str, Any]) -> dict[str, Any]:
    mechanism = summary.get("mechanism_family_summary", {})
    if not isinstance(mechanism, dict):
        return {
            "metadata_coverage_rate": 0.0,
            "bundles_with_metadata": 0,
            "mean_total_function_plans": 0.0,
        }
    return {
        "metadata_coverage_rate": mechanism.get("metadata_coverage_rate", 0.0),
        "bundles_with_metadata": mechanism.get("bundles_with_metadata", 0),
        "mean_total_function_plans": mechanism.get("mean_total_function_plans", 0.0),
    }


def _project_metric(metric: dict[str, Any]) -> dict[str, Any]:
    histogram = metric.get("histogram", {})
    if not isinstance(histogram, dict):
        histogram = {}
    quantiles = metric.get("quantiles", {})
    if not isinstance(quantiles, dict):
        quantiles = {}
    return {
        "count": metric.get("count", 0),
        "missing_count": metric.get("missing_count", 0),
        "observed_min": metric.get("observed_min"),
        "observed_max": metric.get("observed_max"),
        "mean": metric.get("mean"),
        "std": metric.get("std"),
        "sampled_count": metric.get("sampled_count", 0),
        "sampled_fraction": metric.get("sampled_fraction", 0.0),
        "quantiles": quantiles,
        "histogram": {
            "num_bins": histogram.get("num_bins", 0),
            "covered_bins": histogram.get("covered_bins", 0),
            "coverage_ratio": histogram.get("coverage_ratio", 0.0),
        },
        "underrepresented_bins": metric.get("underrepresented_bins", []),
        "target_band": metric.get("target_band"),
    }


def _normalize_json(value: Any, *, key_path: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_json(child, key_path=key_path + (str(key),))
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_json(child, key_path=key_path + (str(index),))
            for index, child in enumerate(value)
        ]
    if isinstance(value, str):
        return _normalize_string(value, key_path=key_path)
    if isinstance(value, (int, float)):
        return _normalize_number(value, key_path=key_path)
    return value


def _normalize_string(value: str, *, key_path: tuple[str, ...]) -> str:
    key = key_path[-1] if key_path else ""
    if Path(value).is_absolute():
        return _ABSOLUTE_PATH_PLACEHOLDER
    if key in {"generated_at", "timestamp", "started_at", "completed_at"}:
        return _TIMESTAMP_PLACEHOLDER
    if key in {"generate_run_id", "generated_corpus_id", "run_id", "request_run"}:
        return _RUN_ID_PLACEHOLDER
    if key in {"effective_config_sha256", "effective_config_trace_sha256"}:
        return _SHA256_PLACEHOLDER
    if _HEX_ID_RE.fullmatch(value) and key.endswith("_id"):
        return _RUN_ID_PLACEHOLDER
    if _HEX_ID_RE.fullmatch(value) and key.endswith("_sha256"):
        return _SHA256_PLACEHOLDER
    if _ISO_TIMESTAMP_RE.fullmatch(value):
        return _TIMESTAMP_PLACEHOLDER
    return value


def _normalize_number(value: int | float, *, key_path: tuple[str, ...]) -> Any:
    key = key_path[-1] if key_path else ""
    if key in {
        "datasets_per_minute",
        "generation_datasets_per_minute",
        "write_datasets_per_minute",
        "filter_datasets_per_minute",
        "filter_accepted_datasets_per_minute",
        "filter_rejection_rate_attempt_level",
        "filter_acceptance_rate_dataset_level",
        "filter_rejection_rate_dataset_level",
        "filter_retry_dataset_rate",
        "elapsed_seconds",
        "prepare_elapsed_seconds",
        "prepare_cpu_time_seconds",
        "prepare_cpu_busy_pct_of_wall",
        "generation_elapsed_seconds",
        "generation_cpu_time_seconds",
        "generation_cpu_busy_pct_of_wall",
        "raw_batch_elapsed_seconds",
        "raw_batch_cpu_time_seconds",
        "node_apply_elapsed_seconds",
        "converter_elapsed_seconds",
        "feature_materialization_elapsed_seconds",
        "write_stage_elapsed_seconds",
        "write_stage_cpu_time_seconds",
        "write_stage_mib_per_second",
        "datasets_per_minute",
        "latency_p95_ms",
        "peak_rss_mb",
        "peak_cuda_reserved_mb",
        "peak_cuda_reserved_pct_of_total_memory",
        "peak_cuda_headroom_mb",
    }:
        if key.endswith("_pct") or "pct" in key:
            return _PERCENT_PLACEHOLDER
        if key.endswith("_ms") or "latency" in key:
            return _MS_PLACEHOLDER
        if key.endswith("_mb") or "rss" in key or "headroom" in key or "reserved" in key:
            return _MB_PLACEHOLDER
        if key.endswith("_per_minute") or key.endswith("_per_second") or "throughput" in key:
            return _RATE_PLACEHOLDER
        return _SECONDS_PLACEHOLDER
    return value


def _normalize_inline_paths(line: str) -> str:
    if "`" not in line:
        return line
    return re.sub(r"`/[^`]+`", f"`{_ABSOLUTE_PATH_PLACEHOLDER}`", line)
