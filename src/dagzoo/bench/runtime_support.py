"""Reusable benchmark runtime-support helpers."""

from __future__ import annotations

import contextlib
import hashlib
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch

from dagzoo.config import (
    MISSINGNESS_MECHANISM_NONE,
    NOISE_FAMILY_GAUSSIAN,
    GeneratorConfig,
)
from dagzoo.core.dataset import generate_batch_iter, generate_one
from dagzoo.diagnostics import CoverageAggregator
from dagzoo.diagnostics_targets import build_diagnostics_aggregation_config
from dagzoo.rng import KeyedRng

from .constants import (
    KIB,
    MIB,
    PRESET_KEY_HASH_SUFFIX_LEN,
    SMOKE_LATENCY_SAMPLES_CAP,
    SMOKE_NUM_DATASETS_CAP,
    SMOKE_WARMUP_DATASETS_CAP,
)
from .guardrails import _build_guardrail_issue
from .metrics import percent_change, reproducibility_signatures, summarize_latencies


def _peak_rss_mb() -> float:
    """Return process max resident set size in MiB."""

    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / MIB
    return rss / KIB


def _resolved_device_type(device: str | None) -> str | None:
    """Resolve one benchmark device hint to a concrete torch device type when possible."""

    if device is None:
        return None

    normalized = str(device).strip().lower()
    if not normalized:
        return None
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    with contextlib.suppress(RuntimeError, TypeError, ValueError):
        return torch.device(normalized).type
    return None


def _synchronize_accelerator(device: str | None) -> None:
    """Wait for queued accelerator work on the resolved benchmark device."""

    device_type = _resolved_device_type(device)
    if device_type == "cuda" and torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
    if device_type == "mps":
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None and hasattr(mps_module, "synchronize"):
            with contextlib.suppress(Exception):
                mps_module.synchronize()


def _preset_counts(
    config: GeneratorConfig,
    *,
    preset_key: str,
    suite: str,
    num_datasets_override: int | None,
    warmup_override: int | None,
) -> tuple[int, int]:
    """Resolve benchmark dataset and warmup counts for a preset and suite mode."""

    preset_map = config.benchmark.presets.get(preset_key, {})
    num = int(preset_map.get("num_datasets", config.benchmark.num_datasets))
    warmup = int(preset_map.get("warmup_datasets", config.benchmark.warmup_datasets))

    if num_datasets_override is not None:
        num = int(num_datasets_override)
    if warmup_override is not None:
        warmup = int(warmup_override)

    if suite == "smoke":
        num = min(num, SMOKE_NUM_DATASETS_CAP)
        warmup = min(warmup, SMOKE_WARMUP_DATASETS_CAP)

    return max(1, num), max(0, warmup)


def _latency_sample_count(config: GeneratorConfig, suite: str, num_datasets: int) -> int:
    """Choose per-preset latency sample count for the requested suite level."""

    n = max(1, min(int(config.benchmark.latency_num_samples), num_datasets))
    if suite == "smoke":
        return min(n, SMOKE_LATENCY_SAMPLES_CAP)
    return n


def _collect_latency(
    config: GeneratorConfig,
    *,
    device: str | None,
    num_samples: int,
) -> dict[str, float]:
    """Collect per-dataset latency samples by repeatedly calling ``generate_one``."""

    latency_root = KeyedRng(int(config.seed)).keyed("bench", "suite", "latency")
    samples: list[float] = []
    timing_device = device or config.runtime.device
    for i in range(max(1, num_samples)):
        seed = latency_root.child_seed("sample", i)
        _synchronize_accelerator(timing_device)
        start = time.perf_counter()
        _ = generate_one(config, seed=seed, device=device)
        _synchronize_accelerator(timing_device)
        samples.append(time.perf_counter() - start)
    return summarize_latencies(samples)


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


def _sanitize_preset_key(preset_key: str) -> str:
    """Normalize preset key into a filesystem-safe unique path segment."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", preset_key).strip("._-")
    if not normalized:
        normalized = "preset"
    suffix = hashlib.sha1(preset_key.encode("utf-8")).hexdigest()[:PRESET_KEY_HASH_SUFFIX_LEN]
    return f"{normalized}_{suffix}"


def _artifact_pointer(path: Path) -> str:
    """Return a summary-safe pointer for diagnostics artifacts."""

    return str(path.resolve())


def _build_diagnostics_aggregator(config: GeneratorConfig) -> CoverageAggregator:
    """Create a diagnostics coverage aggregator from config."""

    return CoverageAggregator(build_diagnostics_aggregation_config(config.diagnostics))


def _is_missingness_enabled(config: GeneratorConfig) -> bool:
    """Return whether missingness is enabled in config."""

    return bool(
        float(config.dataset.missing_rate) > 0.0
        and str(config.dataset.missing_mechanism).strip().lower() != MISSINGNESS_MECHANISM_NONE
    )


def _is_shift_enabled(config: GeneratorConfig) -> bool:
    """Return whether shift controls are enabled in config."""

    return bool(config.shift.enabled)


def _is_noise_enabled(config: GeneratorConfig) -> bool:
    """Return whether non-gaussian noise controls are enabled in config."""

    return str(config.noise.family).strip().lower() != NOISE_FAMILY_GAUSSIAN


def _build_shift_directional_check(
    *,
    metric: str,
    enabled: bool,
    gating_enabled: bool,
    current: float | None,
    baseline: float | None,
    detail: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build directional check payload and optional issue for one shift metric."""

    payload: dict[str, Any] = {
        "enabled": bool(enabled),
        "gating_enabled": bool(gating_enabled),
        "current": current,
        "baseline": baseline,
        "status": "not_applicable",
        "detail": detail,
    }
    if not enabled:
        payload["reason"] = "axis_inactive"
        return payload, None
    if not gating_enabled:
        payload["status"] = "suppressed"
        payload["reason"] = "insufficient_sample_size"
        return payload, None
    if current is None or baseline is None:
        payload["status"] = "fail"
        issue = _build_guardrail_issue(
            metric=f"shift_{metric}_directionality_unavailable",
            severity="fail",
            current=current,
            baseline=baseline,
            degradation_pct=None,
            detail=f"{detail} Directional check could not be computed from benchmark samples.",
        )
        return payload, issue
    if float(current) > float(baseline):
        payload["status"] = "pass"
        return payload, None

    payload["status"] = "fail"
    current_value = float(current)
    baseline_value = float(baseline)
    raw_change = percent_change(current_value, baseline_value)
    raw_degradation = -raw_change if raw_change is not None else None
    issue = _build_guardrail_issue(
        metric=f"shift_{metric}_directionality",
        severity="fail",
        current=current_value,
        baseline=baseline_value,
        degradation_pct=(float(raw_degradation) if raw_degradation is not None else None),
        detail=detail,
    )
    return payload, issue
