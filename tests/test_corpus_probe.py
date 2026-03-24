import numpy as np
import pytest

from dagzoo.bench.corpus_probe import resolve_corpus_probe_counts, run_corpus_probe
from dagzoo.bench.stage_metrics import FilterStageMeasurement
from dagzoo.config import GeneratorConfig
from dagzoo.diagnostics.coverage import CoverageAggregationConfig
from dagzoo.types import DatasetBundle


def _bundle(*, seed: int) -> DatasetBundle:
    return DatasetBundle(
        X_train=np.zeros((8, 4), dtype=np.float32),
        y_train=np.zeros(8, dtype=np.int64),
        X_test=np.zeros((4, 4), dtype=np.float32),
        y_test=np.zeros(4, dtype=np.int64),
        feature_types=["num", "num", "num", "num"],
        metadata={"dataset_seed": seed},
    )


def test_resolve_corpus_probe_counts_uses_baseline_defaults_and_smoke_caps() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")

    assert resolve_corpus_probe_counts(
        cfg,
        suite="smoke",
        num_datasets_override=None,
        warmup_override=None,
    ) == (25, 5)
    assert resolve_corpus_probe_counts(
        cfg,
        suite="standard",
        num_datasets_override=None,
        warmup_override=None,
    ) == (200, 10)
    assert resolve_corpus_probe_counts(
        cfg,
        suite="smoke",
        num_datasets_override=7,
        warmup_override=2,
    ) == (7, 2)


def test_run_corpus_probe_uses_second_pass_analysis_without_throughput_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.filter.enabled = False
    cfg.runtime.device = "cpu"

    captured: dict[str, object] = {}

    def _stub_run_throughput_benchmark(*_args, **kwargs):
        captured["on_bundle"] = kwargs.get("on_bundle")
        return {"datasets_per_minute": 123.0}

    monkeypatch.setattr(
        "dagzoo.bench.corpus_probe.run_throughput_benchmark",
        _stub_run_throughput_benchmark,
    )
    monkeypatch.setattr(
        "dagzoo.bench.corpus_probe.iter_throughput_measure_bundles",
        lambda *_args, **_kwargs: (_bundle(seed=idx + 1) for idx in range(3)),
    )

    result = run_corpus_probe(
        cfg,
        label="baseline",
        config_path="configs/default.yaml",
        suite="smoke",
        num_datasets=3,
        warmup=1,
        device="cpu",
    )

    assert captured["on_bundle"] is None
    assert result.datasets_per_minute == 123.0
    assert result.coverage_summary["num_datasets"] == 3


def test_run_corpus_probe_streams_filter_enabled_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.filter.enabled = True
    cfg.runtime.device = "cpu"

    captured: dict[str, object] = {}

    def _stub_run_throughput_benchmark(*_args, **kwargs):
        captured["on_bundle"] = kwargs.get("on_bundle")
        return {"datasets_per_minute": 99.0}

    def _stub_replay_filter_stage_metrics(bundles, *, config, on_accepted_bundle=None):
        assert bool(config.filter.enabled)
        assert not isinstance(bundles, list)
        observed = list(bundles)
        assert len(observed) == 3
        assert on_accepted_bundle is not None
        on_accepted_bundle(observed[0])
        on_accepted_bundle(observed[2])
        return FilterStageMeasurement(
            datasets_per_minute=30.0,
            filter_attempts_total=3,
            filter_accepted_datasets=2,
            filter_rejections_total=1,
            filter_rejected_datasets=1,
            accepted_true_fraction=2.0 / 3.0,
            skill_small_mean=0.82,
            skill_full_mean=0.9,
            skill_gain_mean=0.08,
            skill_small_lb95_mean=0.78,
            skill_gain_ub95_mean=0.09,
            skill_full_ub95_mean=0.93,
            stump_skill_mean=0.45,
            reason_counts={"too_hard_garbage": 1},
        )

    monkeypatch.setattr(
        "dagzoo.bench.corpus_probe.run_throughput_benchmark",
        _stub_run_throughput_benchmark,
    )
    monkeypatch.setattr(
        "dagzoo.bench.corpus_probe.iter_throughput_measure_bundles",
        lambda *_args, **_kwargs: (_bundle(seed=idx + 1) for idx in range(3)),
    )
    monkeypatch.setattr(
        "dagzoo.bench.corpus_probe.replay_filter_stage_metrics",
        _stub_replay_filter_stage_metrics,
    )

    result = run_corpus_probe(
        cfg,
        label="baseline",
        config_path="configs/default.yaml",
        suite="smoke",
        num_datasets=3,
        warmup=1,
        device="cpu",
    )

    assert captured["on_bundle"] is None
    assert result.filter_datasets_per_minute == 30.0
    assert result.filter_accepted_datasets_per_minute == pytest.approx(20.0)
    assert result.coverage_summary["num_datasets"] == 2
    assert result.filter_summary["reason_counts"] == {"too_hard_garbage": 1}


def test_run_corpus_probe_uses_shared_probe_seed_for_generation_and_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.filter.enabled = False
    cfg.runtime.device = "cpu"
    cfg.seed = 11
    captured: dict[str, int] = {}

    def _stub_run_throughput_benchmark(config, **_kwargs):
        captured["throughput_seed"] = int(config.seed)
        return {"datasets_per_minute": 123.0}

    def _stub_iter_throughput_measure_bundles(config, **_kwargs):
        captured["analysis_seed"] = int(config.seed)
        return (_bundle(seed=idx + 1) for idx in range(2))

    monkeypatch.setattr(
        "dagzoo.bench.corpus_probe.run_throughput_benchmark",
        _stub_run_throughput_benchmark,
    )
    monkeypatch.setattr(
        "dagzoo.bench.corpus_probe.iter_throughput_measure_bundles",
        _stub_iter_throughput_measure_bundles,
    )

    result = run_corpus_probe(
        cfg,
        label="baseline",
        config_path="configs/default.yaml",
        suite="smoke",
        num_datasets=2,
        warmup=0,
        device="cpu",
        probe_seed=777,
    )

    assert captured["throughput_seed"] == 777
    assert captured["analysis_seed"] == 777
    assert result.resolved_config["seed"] == 777


def test_run_corpus_probe_uses_injected_coverage_config_over_local_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.filter.enabled = False
    cfg.runtime.device = "cpu"
    cfg.diagnostics.histogram_bins = 99
    cfg.diagnostics.quantiles = [0.1, 0.9]
    cfg.diagnostics.max_values_per_metric = 1

    monkeypatch.setattr(
        "dagzoo.bench.corpus_probe.run_throughput_benchmark",
        lambda *_args, **_kwargs: {"datasets_per_minute": 123.0},
    )
    monkeypatch.setattr(
        "dagzoo.bench.corpus_probe.iter_throughput_measure_bundles",
        lambda *_args, **_kwargs: (_bundle(seed=idx + 1) for idx in range(3)),
    )

    result = run_corpus_probe(
        cfg,
        label="baseline",
        config_path="configs/default.yaml",
        suite="smoke",
        num_datasets=3,
        warmup=0,
        device="cpu",
        coverage_config=CoverageAggregationConfig(
            histogram_bins=7,
            quantiles=(0.25, 0.5, 0.75),
            max_values_per_metric=2,
        ),
    )

    assert result.coverage_summary["histogram_bins"] == 7
    assert result.coverage_summary["quantiles"] == [0.25, 0.5, 0.75]
    assert result.coverage_summary["max_values_per_metric"] == 2
