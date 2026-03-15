"""Throughput benchmark harness."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from typing import Any

from dagzoo.bench.constants import (
    SECONDS_PER_MINUTE,
    THROUGHPUT_SLO_DATASETS_PER_MINUTE,
)
from dagzoo.config import GeneratorConfig, clone_generator_config
from dagzoo.core.dataset import _iter_prepared_canonical_batch_iter, generate_batch_iter
from dagzoo.core.fixed_layout.runtime import prepare_canonical_fixed_layout_run
from dagzoo.hardware import detect_hardware
from dagzoo.hardware_policy import (
    resolve_cuda_fixed_layout_target_cells_limits,
    round_fixed_layout_target_cells,
)
from dagzoo.rng import KeyedRng
from dagzoo.types import DatasetBundle

from .runtime_support import _synchronize_accelerator

_CPU_FIXED_LAYOUT_TARGET_CELLS_SWEEP: tuple[int, ...] = (
    4_000_000,
    8_000_000,
    12_000_000,
    16_000_000,
)
_CUDA_FIXED_LAYOUT_TARGET_CELLS_SWEEP_MULTIPLIERS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
_RAW_BATCH_METRIC_KEYS: tuple[str, ...] = (
    "raw_batch_elapsed_seconds",
    "raw_batch_cpu_time_seconds",
    "node_apply_elapsed_seconds",
    "node_apply_cpu_time_seconds",
    "converter_elapsed_seconds",
    "converter_cpu_time_seconds",
    "feature_materialization_elapsed_seconds",
    "feature_materialization_cpu_time_seconds",
)


def _throughput_root(config: GeneratorConfig) -> KeyedRng:
    """Return the shared keyed RNG root for throughput benchmark runs."""

    return KeyedRng(int(config.seed)).keyed("bench", "throughput")


def _throughput_warmup_seed(config: GeneratorConfig) -> int:
    """Return the deterministic warmup seed for throughput benchmarks."""

    return _throughput_root(config).child_seed("warmup")


def _throughput_measure_seed(config: GeneratorConfig) -> int:
    """Return the deterministic measured-run seed for throughput benchmarks."""

    return _throughput_root(config).child_seed("measure")


def _default_fixed_layout_target_cells_sweep_values(
    config: GeneratorConfig,
    *,
    device: str | None,
) -> tuple[int, ...]:
    """Return a small target-cell sweep grid matched to the detected hardware tier."""

    requested_device = device or config.runtime.device
    hardware = detect_hardware(requested_device)
    if hardware.backend != "cuda":
        return _CPU_FIXED_LAYOUT_TARGET_CELLS_SWEEP

    target_floor, target_cap = resolve_cuda_fixed_layout_target_cells_limits(hardware)
    current_target = int(config.runtime.fixed_layout_target_cells or 0)
    baseline = max(current_target, int(target_floor or 0))
    if baseline <= 0:
        return _CPU_FIXED_LAYOUT_TARGET_CELLS_SWEEP

    effective_cap = max(int(target_cap or baseline), baseline)
    candidates: list[int] = []
    for multiplier in _CUDA_FIXED_LAYOUT_TARGET_CELLS_SWEEP_MULTIPLIERS:
        scaled = int(math.ceil(float(baseline) * float(multiplier)))
        rounded = round_fixed_layout_target_cells(scaled)
        candidates.append(max(baseline, min(rounded, effective_cap)))
    return tuple(sorted(set(candidates)))


def run_fixed_layout_target_cells_sweep(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    warmup_datasets: int = 1,
    device: str | None = None,
    target_cells_values: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    """Measure throughput across a small fixed-layout auto-batch target sweep."""

    raw_values = (
        tuple(target_cells_values)
        if target_cells_values is not None
        else _default_fixed_layout_target_cells_sweep_values(config, device=device)
    )
    normalized_values = tuple(
        sorted({_normalize_target_cells_value(value) for value in raw_values})
    )
    if not normalized_values:
        raise ValueError("target_cells_values must contain at least one positive integer.")

    results: list[dict[str, Any]] = []
    for target_cells in normalized_values:
        tuned = clone_generator_config(config, revalidate=False)
        tuned.runtime.fixed_layout_target_cells = int(target_cells)
        result = run_throughput_benchmark(
            tuned,
            num_datasets=num_datasets,
            warmup_datasets=warmup_datasets,
            device=device,
        )
        results.append(
            {
                **result,
                "fixed_layout_target_cells": int(target_cells),
            }
        )

    best = max(
        results,
        key=lambda item: (
            float(item["datasets_per_minute"]),
            -int(item["fixed_layout_target_cells"]),
        ),
    )
    hardware = detect_hardware(device or config.runtime.device)
    return {
        "hardware_backend": str(hardware.backend),
        "hardware_tier": str(hardware.tier),
        "device": str(device or config.runtime.device or "auto"),
        "target_cells_values": [int(value) for value in normalized_values],
        "recommended_fixed_layout_target_cells": int(best["fixed_layout_target_cells"]),
        "results": results,
    }


def _normalize_target_cells_value(value: int) -> int:
    """Validate one target-cell sweep candidate."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"fixed-layout target cells must be positive integers, got {value!r}.")
    if value <= 0:
        raise ValueError(f"fixed-layout target cells must be positive integers, got {value!r}.")
    return int(value)


def _consume_generation(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int,
    device: str | None,
    on_bundle: Callable[[DatasetBundle], object] | None = None,
) -> None:
    """Run generation for ``num_datasets`` items while discarding outputs."""

    for bundle in generate_batch_iter(
        config,
        num_datasets=num_datasets,
        seed=seed,
        device=device,
    ):
        if on_bundle is not None:
            on_bundle(bundle)


def iter_throughput_measure_bundles(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    device: str | None = None,
) -> Iterator[DatasetBundle]:
    """Yield the deterministic measured corpus used by throughput benchmarks."""

    prepared = prepare_canonical_fixed_layout_run(
        config,
        num_datasets=num_datasets,
        seed=_throughput_measure_seed(config),
        device=device,
    )
    yield from _iter_prepared_canonical_batch_iter(prepared, num_datasets=num_datasets)


def run_throughput_benchmark(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    warmup_datasets: int = 10,
    device: str | None = None,
    on_bundle: Callable[[DatasetBundle], object] | None = None,
) -> dict[str, Any]:
    """Measure end-to-end generation throughput for a benchmark preset."""

    timing_device = device or config.runtime.device
    if warmup_datasets > 0:
        _consume_generation(
            config,
            num_datasets=warmup_datasets,
            seed=_throughput_warmup_seed(config),
            device=device,
        )

    _synchronize_accelerator(timing_device)
    start = time.perf_counter()
    start_cpu = time.process_time()
    raw_batch_metric_totals = {key: 0.0 for key in _RAW_BATCH_METRIC_KEYS}
    prepared = prepare_canonical_fixed_layout_run(
        config,
        num_datasets=num_datasets,
        seed=_throughput_measure_seed(config),
        device=device,
    )
    _synchronize_accelerator(timing_device)
    prepare_elapsed_seconds = time.perf_counter() - start
    prepare_cpu_time_seconds = time.process_time() - start_cpu

    def _on_raw_batch_metrics(metrics: dict[str, float]) -> None:
        for key in _RAW_BATCH_METRIC_KEYS:
            raw_batch_metric_totals[key] += float(metrics.get(key, 0.0) or 0.0)

    for bundle in _iter_prepared_canonical_batch_iter(
        prepared,
        num_datasets=num_datasets,
        on_raw_batch_metrics=_on_raw_batch_metrics,
    ):
        if on_bundle is not None:
            on_bundle(bundle)
    _synchronize_accelerator(timing_device)
    elapsed = time.perf_counter() - start
    cpu_time_seconds = time.process_time() - start_cpu
    dps = num_datasets / elapsed if elapsed > 0 else 0.0
    dpm = dps * SECONDS_PER_MINUTE
    return {
        "preset": config.benchmark.preset_name,
        "num_datasets": num_datasets,
        "warmup_datasets": warmup_datasets,
        "elapsed_seconds": elapsed,
        "cpu_time_seconds": cpu_time_seconds,
        "prepare_elapsed_seconds": prepare_elapsed_seconds,
        "prepare_cpu_time_seconds": prepare_cpu_time_seconds,
        "datasets_per_second": dps,
        "datasets_per_minute": dpm,
        "slo_pass_100_datasets_per_min": dpm >= THROUGHPUT_SLO_DATASETS_PER_MINUTE,
        "generation_mode": "fixed_batched",
        **raw_batch_metric_totals,
    }
