"""Helpers for benchmark stage-level throughput measurement."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dagzoo.bench.constants import SECONDS_PER_MINUTE
from dagzoo.config import GeneratorConfig
from dagzoo.filtering.extra_trees_filter import _apply_extra_trees_filter_numpy
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream
from dagzoo.math import coerce_optional_finite_float as _coerce_optional_finite_float
from dagzoo.math import to_numpy as _to_numpy
from dagzoo.rng import SEED32_MAX
from dagzoo.types import DatasetBundle


@dataclass(slots=True)
class StageSampleCollector:
    """Collect a bounded sample of emitted bundles for stage timing probes."""

    max_samples: int
    bundles: list[DatasetBundle] = field(default_factory=list)

    def update(self, bundle: DatasetBundle) -> None:
        """Store one emitted bundle if the sample has remaining capacity."""

        if len(self.bundles) >= max(0, int(self.max_samples)):
            return
        self.bundles.append(bundle)


@dataclass(slots=True)
class FilterStageMeasurement:
    """Deferred filter stage replay metrics over sampled bundles."""

    datasets_per_minute: float
    filter_attempts_total: int
    filter_accepted_datasets: int
    filter_rejections_total: int
    filter_rejected_datasets: int
    accepted_true_fraction: float | None = None
    wins_ratio_mean: float | None = None
    threshold_effective_mean: float | None = None
    threshold_delta_mean: float | None = None
    n_valid_oob_mean: float | None = None
    elapsed_seconds: float = 0.0
    cpu_time_seconds: float = 0.0
    reason_counts: dict[str, int] = field(default_factory=dict)
    accepted_bundles: list[DatasetBundle] = field(default_factory=list)


@dataclass(slots=True)
class WriteStageMeasurement:
    """Replay write-stage throughput and resource usage over sampled bundles."""

    datasets_per_minute: float
    elapsed_seconds: float
    cpu_time_seconds: float
    bytes_written: int
    mib_per_second: float


def _directory_size_bytes(root: str | Path) -> int:
    """Return the total size of files under one directory tree."""

    total = 0
    for path in Path(root).rglob("*"):
        if path.is_file():
            total += int(path.stat().st_size)
    return int(total)


def measure_write_stage_metrics(
    bundles: Sequence[DatasetBundle],
    *,
    config: GeneratorConfig,
) -> WriteStageMeasurement:
    """Measure parquet write throughput and resource usage on sampled bundles."""

    num_bundles = len(bundles)
    if num_bundles <= 0:
        return WriteStageMeasurement(
            datasets_per_minute=0.0,
            elapsed_seconds=0.0,
            cpu_time_seconds=0.0,
            bytes_written=0,
            mib_per_second=0.0,
        )

    start = time.perf_counter()
    start_cpu = time.process_time()
    with tempfile.TemporaryDirectory(prefix="dagzoo_stage_write_") as tmp_dir:
        _ = write_packed_parquet_shards_stream(
            bundles,
            out_dir=tmp_dir,
            shard_size=max(1, int(config.output.shard_size)),
            compression=str(config.output.compression),
        )
        bytes_written = _directory_size_bytes(tmp_dir)
    elapsed = time.perf_counter() - start
    cpu_time_seconds = time.process_time() - start_cpu
    if elapsed <= 0.0:
        return WriteStageMeasurement(
            datasets_per_minute=0.0,
            elapsed_seconds=0.0,
            cpu_time_seconds=max(0.0, float(cpu_time_seconds)),
            bytes_written=int(bytes_written),
            mib_per_second=0.0,
        )
    return WriteStageMeasurement(
        datasets_per_minute=(float(num_bundles) / elapsed) * SECONDS_PER_MINUTE,
        elapsed_seconds=float(elapsed),
        cpu_time_seconds=max(0.0, float(cpu_time_seconds)),
        bytes_written=int(bytes_written),
        mib_per_second=float(bytes_written) / (1024.0 * 1024.0) / elapsed,
    )


def measure_write_datasets_per_minute(
    bundles: Sequence[DatasetBundle],
    *,
    config: GeneratorConfig,
) -> float:
    """Measure parquet write throughput on sampled bundles."""

    return float(measure_write_stage_metrics(bundles, config=config).datasets_per_minute)


def _coerce_bundle_seed(bundle: DatasetBundle, *, fallback_seed: int) -> int:
    """Resolve per-bundle seed for deterministic deferred filter replay."""

    metadata = bundle.metadata
    for key in ("dataset_seed", "seed"):
        raw_seed: Any = metadata.get(key)
        if isinstance(raw_seed, bool):
            raw_seed = None
        if isinstance(raw_seed, float):
            if raw_seed.is_integer():
                raw_seed = int(raw_seed)
            else:
                raw_seed = None
        if isinstance(raw_seed, int):
            return int(raw_seed % (SEED32_MAX + 1))
    return int(fallback_seed % (SEED32_MAX + 1))


def _mean_or_none(values: list[float]) -> float | None:
    """Return the mean of collected values or ``None`` when empty."""

    if not values:
        return None
    return float(sum(values) / len(values))


def replay_filter_stage_metrics(
    bundles: Iterable[DatasetBundle],
    *,
    config: GeneratorConfig,
    on_accepted_bundle: Callable[[DatasetBundle], None] | None = None,
) -> FilterStageMeasurement:
    """Replay deferred filter stage over a bundle stream and return throughput + outcomes."""

    if not bool(config.filter.enabled):
        return FilterStageMeasurement(
            datasets_per_minute=0.0,
            filter_attempts_total=0,
            filter_accepted_datasets=0,
            filter_rejections_total=0,
            filter_rejected_datasets=0,
        )

    attempts_total = 0
    accepted_total = 0
    rejections_total = 0
    wins_ratio_values: list[float] = []
    threshold_effective_values: list[float] = []
    threshold_delta_values: list[float] = []
    n_valid_oob_values: list[float] = []
    reason_counts: dict[str, int] = {}
    start = time.perf_counter()
    start_cpu = time.process_time()
    callback_elapsed = 0.0
    callback_cpu_elapsed = 0.0
    for idx, bundle in enumerate(bundles):
        x_train = _to_numpy(bundle.X_train).astype("float32", copy=False)
        x_test = _to_numpy(bundle.X_test).astype("float32", copy=False)
        y_train = _to_numpy(bundle.y_train)
        y_test = _to_numpy(bundle.y_test)
        x_all = x_train
        if x_test.size > 0:
            x_all = np.concatenate([x_train, x_test], axis=0)
        y_all = y_train
        if y_test.size > 0:
            y_all = np.concatenate([y_train, y_test], axis=0)
        if str(config.dataset.task) == "classification":
            y_all = y_all.astype("int64", copy=False)
        else:
            y_all = y_all.astype("float32", copy=False)

        attempts_total += 1
        accepted, _details = _apply_extra_trees_filter_numpy(
            x_all,
            y_all,
            task=str(config.dataset.task),
            seed=_coerce_bundle_seed(bundle, fallback_seed=config.seed + idx),
            n_estimators=int(config.filter.n_estimators),
            max_depth=int(config.filter.max_depth),
            min_samples_leaf=int(config.filter.min_samples_leaf),
            max_leaf_nodes=config.filter.max_leaf_nodes,
            max_features=config.filter.max_features,
            n_bootstrap=int(config.filter.n_bootstrap),
            threshold=float(config.filter.threshold),
            n_jobs=int(config.filter.n_jobs),
        )
        if bool(accepted):
            accepted_total += 1
            if on_accepted_bundle is not None:
                callback_start = time.perf_counter()
                callback_start_cpu = time.process_time()
                on_accepted_bundle(bundle)
                callback_elapsed += max(0.0, time.perf_counter() - callback_start)
                callback_cpu_elapsed += max(0.0, time.process_time() - callback_start_cpu)
        else:
            rejections_total += 1
        wins_ratio = _coerce_optional_finite_float(_details.get("wins_ratio"))
        if wins_ratio is not None:
            wins_ratio_values.append(float(wins_ratio))
        threshold_effective = _coerce_optional_finite_float(_details.get("threshold_effective"))
        if threshold_effective is not None:
            threshold_effective_values.append(float(threshold_effective))
        threshold_delta = _coerce_optional_finite_float(_details.get("threshold_delta"))
        if threshold_delta is not None:
            threshold_delta_values.append(float(threshold_delta))
        n_valid_oob = _coerce_optional_finite_float(_details.get("n_valid_oob"))
        if n_valid_oob is not None:
            n_valid_oob_values.append(float(n_valid_oob))
        reason = _details.get("reason")
        if isinstance(reason, str) and reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    elapsed = max(0.0, time.perf_counter() - start - callback_elapsed)
    cpu_time_seconds = max(0.0, time.process_time() - start_cpu - callback_cpu_elapsed)
    dpm = ((float(attempts_total) / elapsed) * SECONDS_PER_MINUTE) if elapsed > 0.0 else 0.0
    return FilterStageMeasurement(
        datasets_per_minute=float(dpm),
        filter_attempts_total=int(attempts_total),
        filter_accepted_datasets=int(accepted_total),
        filter_rejections_total=int(rejections_total),
        filter_rejected_datasets=int(rejections_total),
        accepted_true_fraction=(
            float(accepted_total) / float(attempts_total) if attempts_total > 0 else None
        ),
        wins_ratio_mean=_mean_or_none(wins_ratio_values),
        threshold_effective_mean=_mean_or_none(threshold_effective_values),
        threshold_delta_mean=_mean_or_none(threshold_delta_values),
        n_valid_oob_mean=_mean_or_none(n_valid_oob_values),
        elapsed_seconds=float(elapsed),
        cpu_time_seconds=float(cpu_time_seconds),
        reason_counts=dict(sorted(reason_counts.items())),
    )


def measure_filter_stage_metrics(
    bundles: Sequence[DatasetBundle],
    *,
    config: GeneratorConfig,
    collect_accepted_bundles: bool = False,
) -> FilterStageMeasurement:
    """Replay deferred filter stage over sampled bundles and return throughput + outcomes."""

    accepted_bundles: list[DatasetBundle] = []
    measurement = replay_filter_stage_metrics(
        bundles,
        config=config,
        on_accepted_bundle=accepted_bundles.append if collect_accepted_bundles else None,
    )
    if collect_accepted_bundles:
        measurement.accepted_bundles = accepted_bundles
    return measurement
