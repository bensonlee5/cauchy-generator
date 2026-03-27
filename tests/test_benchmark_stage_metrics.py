from pathlib import Path

import numpy as np
import pytest

from dagzoo.bench.collectors import _BundleMetricsCollector, build_pressure_metrics
from dagzoo.bench.stage_metrics import (
    StageSampleCollector,
    measure_filter_stage_metrics,
    measure_write_datasets_per_minute,
    measure_write_stage_metrics,
    replay_filter_stage_metrics,
)
from dagzoo.config import GeneratorConfig
from dagzoo.types import DatasetBundle


def _bundle(
    *,
    metadata: dict[str, object],
    runtime_metrics: dict[str, object] | None = None,
) -> DatasetBundle:
    return DatasetBundle(
        X_train=np.zeros((8, 4), dtype=np.float32),
        y_train=np.zeros(8, dtype=np.int64),
        X_test=np.zeros((4, 4), dtype=np.float32),
        y_test=np.zeros(4, dtype=np.int64),
        feature_types=["num", "num", "num", "num"],
        metadata=metadata,
        runtime_metrics={} if runtime_metrics is None else runtime_metrics,
    )


def test_pressure_metrics_track_attempt_and_filter_rejections() -> None:
    collector = _BundleMetricsCollector(expected_noise_family_requested="gaussian")
    collector.update(
        _bundle(
            metadata={
                "generation_attempts": {
                    "total_attempts": 1,
                    "filter_attempts": 1,
                    "filter_rejections": 0,
                }
            }
        )
    )
    collector.update(
        _bundle(
            metadata={
                "generation_attempts": {
                    "total_attempts": 3,
                    "filter_attempts": 3,
                    "filter_rejections": 2,
                }
            }
        )
    )

    summary = build_pressure_metrics(collector)
    assert summary["datasets_seen"] == 2
    assert summary["attempts_total"] == 4
    assert summary["attempts_per_dataset_mean"] == 2.0
    assert summary["retry_dataset_count"] == 1
    assert summary["retry_dataset_rate"] == 0.5
    assert summary["filter_attempts_total"] == 4
    assert summary["filter_rejections_total"] == 2
    assert summary["filter_rejection_rate_attempt_level"] == 0.5
    assert summary["filter_retry_dataset_count"] == 1
    assert summary["filter_retry_dataset_rate"] == 0.5


def test_pressure_metrics_fall_back_to_legacy_metadata() -> None:
    collector = _BundleMetricsCollector(expected_noise_family_requested="gaussian")
    collector.update(
        _bundle(
            metadata={
                "attempt_used": 2,
                "filter": {"enabled": True, "accepted": True},
            }
        )
    )
    collector.update(_bundle(metadata={"attempt_used": 0, "filter": {"enabled": False}}))

    summary = build_pressure_metrics(collector)
    assert summary["datasets_seen"] == 2
    assert summary["attempts_total"] == 4
    assert summary["retry_dataset_count"] == 1
    assert summary["filter_attempts_total"] == 1
    assert summary["filter_rejections_total"] == 0
    assert summary["filter_rejection_rate_attempt_level"] == 0.0
    assert summary["filter_retry_dataset_count"] == 0
    assert summary["filter_retry_dataset_rate"] == 0.0


def test_pressure_metrics_filter_rates_are_none_when_filter_not_attempted() -> None:
    collector = _BundleMetricsCollector(expected_noise_family_requested="gaussian")
    collector.update(_bundle(metadata={"attempt_used": 0, "filter": {"enabled": False}}))
    summary = build_pressure_metrics(collector)
    assert summary["filter_attempts_total"] == 0
    assert summary["filter_rejection_rate_attempt_level"] is None
    assert summary["filter_retry_dataset_rate"] is None


def test_stage_sample_collector_caps_samples() -> None:
    collector = StageSampleCollector(max_samples=2)
    collector.update(_bundle(metadata={"seed": 1}))
    collector.update(_bundle(metadata={"seed": 2}))
    collector.update(_bundle(metadata={"seed": 3}))
    assert len(collector.bundles) == 2


def test_stage_metric_helpers_return_zero_for_empty_samples() -> None:
    cfg = GeneratorConfig()
    assert measure_filter_stage_metrics([], config=cfg).datasets_per_minute == 0.0
    assert measure_write_datasets_per_minute([], config=cfg) == 0.0
    write_measurement = measure_write_stage_metrics([], config=cfg)
    assert write_measurement.elapsed_seconds == 0.0
    assert write_measurement.cpu_time_seconds == 0.0
    assert write_measurement.bytes_written == 0
    assert write_measurement.mib_per_second == 0.0


def test_filter_stage_metric_replays_filter_and_reports_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.filter.enabled = True
    replay_seeds: list[int] = []

    def _stub_filter(*_args, **_kwargs):
        replay_seeds.append(int(_kwargs["seed"]))
        return bool(int(_kwargs["seed"]) % 2), {"n_valid_oob": 128}

    monkeypatch.setattr("dagzoo.bench.stage_metrics._apply_extra_trees_filter_numpy", _stub_filter)
    bundles = [
        _bundle(metadata={"seed": 11, "dataset_seed": 21}),
        _bundle(metadata={"seed": 12, "dataset_seed": 22}),
    ]
    measurement = measure_filter_stage_metrics(bundles, config=cfg)
    assert measurement.filter_attempts_total == 2
    assert measurement.filter_accepted_datasets == 1
    assert measurement.filter_rejections_total == 1
    assert measurement.filter_rejected_datasets == 1
    assert measurement.datasets_per_minute > 0.0
    assert measurement.elapsed_seconds >= 0.0
    assert measurement.cpu_time_seconds >= 0.0
    assert replay_seeds == [21, 22]
    assert measure_filter_stage_metrics(bundles, config=cfg).datasets_per_minute > 0.0


def test_filter_stage_metric_returns_zero_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.filter.enabled = False
    calls: dict[str, int] = {"count": 0}

    def _stub_filter(*_args, **_kwargs):
        calls["count"] += 1
        return True, {}

    monkeypatch.setattr("dagzoo.bench.stage_metrics._apply_extra_trees_filter_numpy", _stub_filter)
    measurement = measure_filter_stage_metrics([_bundle(metadata={})], config=cfg)
    assert measurement.datasets_per_minute == 0.0
    assert measurement.filter_attempts_total == 0
    assert measurement.filter_accepted_datasets == 0
    assert measurement.filter_rejections_total == 0
    assert measurement.filter_rejected_datasets == 0
    assert calls["count"] == 0


def test_filter_stage_metric_uses_fallback_seed_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.filter.enabled = True
    cfg.seed = 77
    replay_seeds: list[int] = []

    def _stub_filter(*_args, **_kwargs):
        replay_seeds.append(int(_kwargs["seed"]))
        return True, {}

    monkeypatch.setattr("dagzoo.bench.stage_metrics._apply_extra_trees_filter_numpy", _stub_filter)
    _ = measure_filter_stage_metrics(
        [
            _bundle(metadata={}),
            _bundle(metadata={"seed": 100}),
        ],
        config=cfg,
    )
    assert replay_seeds == [77, 100]


def test_filter_stage_metric_falls_back_to_legacy_seed_when_dataset_seed_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.filter.enabled = True
    replay_seeds: list[int] = []

    def _stub_filter(*_args, **_kwargs):
        replay_seeds.append(int(_kwargs["seed"]))
        return True, {}

    monkeypatch.setattr("dagzoo.bench.stage_metrics._apply_extra_trees_filter_numpy", _stub_filter)
    _ = measure_filter_stage_metrics(
        [
            _bundle(metadata={"seed": 41}),
            _bundle(metadata={"seed": 42}),
        ],
        config=cfg,
    )
    assert replay_seeds == [41, 42]


def test_replay_filter_stage_metrics_streams_and_invokes_accept_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.filter.enabled = True
    accepted_markers: list[int] = []

    def _stub_filter(*_args, **kwargs):
        accepted = int(kwargs["seed"]) % 2 == 0
        details = {"reason": "too_hard_garbage"} if not accepted else {}
        return accepted, details

    monkeypatch.setattr("dagzoo.bench.stage_metrics._apply_extra_trees_filter_numpy", _stub_filter)
    measurement = replay_filter_stage_metrics(
        (_bundle(metadata={"dataset_seed": seed}) for seed in (10, 11, 12)),
        config=cfg,
        on_accepted_bundle=lambda bundle: accepted_markers.append(
            int(bundle.metadata["dataset_seed"])
        ),
    )

    assert measurement.filter_attempts_total == 3
    assert measurement.filter_accepted_datasets == 2
    assert measurement.filter_rejected_datasets == 1
    assert measurement.reason_counts == {"too_hard_garbage": 1}
    assert accepted_markers == [10, 12]


def test_replay_filter_stage_metrics_excludes_accept_callback_time_from_throughput(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.filter.enabled = True

    def _stub_filter(*_args, **_kwargs):
        return True, {}

    perf_counter_values = iter((0.0, 1.0, 6.0, 8.0))

    monkeypatch.setattr("dagzoo.bench.stage_metrics._apply_extra_trees_filter_numpy", _stub_filter)
    monkeypatch.setattr(
        "dagzoo.bench.stage_metrics.time.perf_counter",
        lambda: next(perf_counter_values),
    )

    measurement = replay_filter_stage_metrics(
        [_bundle(metadata={"dataset_seed": 10})],
        config=cfg,
        on_accepted_bundle=lambda _bundle: None,
    )

    assert measurement.datasets_per_minute == pytest.approx(20.0)
    assert measurement.elapsed_seconds == pytest.approx(3.0)


def test_replay_filter_stage_metrics_tracks_cpu_time_excluding_accept_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig()
    cfg.filter.enabled = True

    def _stub_filter(*_args, **_kwargs):
        return True, {}

    perf_counter_values = iter((0.0, 1.0, 6.0, 8.0))
    process_time_values = iter((10.0, 10.5, 13.0, 14.0))

    monkeypatch.setattr("dagzoo.bench.stage_metrics._apply_extra_trees_filter_numpy", _stub_filter)
    monkeypatch.setattr(
        "dagzoo.bench.stage_metrics.time.perf_counter",
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr(
        "dagzoo.bench.stage_metrics.time.process_time",
        lambda: next(process_time_values),
    )

    measurement = replay_filter_stage_metrics(
        [_bundle(metadata={"dataset_seed": 10})],
        config=cfg,
        on_accepted_bundle=lambda _bundle: None,
    )

    assert measurement.cpu_time_seconds == pytest.approx(1.5)


def test_measure_write_stage_metrics_reports_elapsed_cpu_and_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig()
    perf_counter_values = iter((0.0, 4.0))
    process_time_values = iter((1.0, 2.5))

    def _stub_write(_bundles, *, out_dir, shard_size: int, compression: str) -> int:
        assert shard_size > 0
        assert compression
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "sample.bin").write_bytes(b"x" * 2048)
        return 1

    monkeypatch.setattr(
        "dagzoo.bench.stage_metrics.write_packed_parquet_shards_stream",
        _stub_write,
    )
    monkeypatch.setattr(
        "dagzoo.bench.stage_metrics.time.perf_counter",
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr(
        "dagzoo.bench.stage_metrics.time.process_time",
        lambda: next(process_time_values),
    )

    measurement = measure_write_stage_metrics([_bundle(metadata={"seed": 1})], config=cfg)

    assert measurement.datasets_per_minute == pytest.approx(15.0)
    assert measurement.elapsed_seconds == pytest.approx(4.0)
    assert measurement.cpu_time_seconds == pytest.approx(1.5)
    assert measurement.bytes_written == 2048
    assert measurement.mib_per_second == pytest.approx(2048.0 / (1024.0 * 1024.0 * 4.0))
