import queue
import threading
import typing
from types import SimpleNamespace

import pytest

from dagzoo.bench.micro import run_microbenchmarks
from dagzoo.bench.public_throughput_smoke import (
    build_public_throughput_smoke_baseline_payload,
    build_public_throughput_smoke_summary,
)
from dagzoo.bench.throughput import (
    run_fixed_layout_target_cells_sweep,
    run_heterogeneous_throughput_benchmark,
    run_stratified_throughput_benchmark,
    run_throughput_benchmark,
)
from dagzoo.config import GeneratorConfig
from dagzoo.hardware import HardwareInfo
from dagzoo.rng import KeyedRng
from dagzoo.runtime_profiling import record_runtime_profile_metric


def _tiny_parallel_config() -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 8
    cfg.dataset.n_features_min = 4
    cfg.dataset.n_features_max = 6
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 5
    cfg.runtime.device = "cpu"
    cfg.filter.enabled = False
    cfg.benchmark.preset_name = "parallel_test"
    return cfg


def test_run_throughput_benchmark_uses_streaming_generation(
    monkeypatch,
) -> None:
    warmup_calls: list[tuple[int, int, str | None]] = []
    prepare_calls: list[tuple[int, int, str | None, bool]] = []
    measure_calls: list[tuple[int, int]] = []

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        warmup_calls.append((num_datasets, int(seed or 0), device))
        for _ in range(num_datasets):
            yield None

    def _stub_prepare_canonical_fixed_layout_run(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        precompute_classification_attempt_plan: bool = True,
    ):
        _ = batch_size
        prepare_calls.append(
            (num_datasets, int(seed or 0), device, bool(precompute_classification_attempt_plan))
        )
        return SimpleNamespace(
            config=_config,
            plan=object(),
            run_seed=int(seed or 0),
            batch_size=num_datasets,
        )

    def _stub_iter_prepared_canonical_batch_iter(
        prepared,
        *,
        num_datasets: int,
        on_raw_batch_metrics=None,
    ):
        _ = on_raw_batch_metrics
        measure_calls.append((int(prepared.run_seed), int(num_datasets)))
        for _ in range(num_datasets):
            yield None

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput.prepare_canonical_fixed_layout_run",
        _stub_prepare_canonical_fixed_layout_run,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._iter_prepared_canonical_batch_iter",
        _stub_iter_prepared_canonical_batch_iter,
    )

    cfg = GeneratorConfig()
    result = run_throughput_benchmark(
        cfg,
        num_datasets=3,
        warmup_datasets=2,
        device="cpu",
    )

    assert warmup_calls == [
        (2, KeyedRng(cfg.seed).child_seed("bench", "throughput", "warmup"), "cpu"),
    ]
    assert prepare_calls == [
        (3, KeyedRng(cfg.seed).child_seed("bench", "throughput", "measure"), "cpu", True),
    ]
    assert measure_calls == [
        (KeyedRng(cfg.seed).child_seed("bench", "throughput", "measure"), 3),
    ]
    assert result["num_datasets"] == 3
    assert result["warmup_datasets"] == 2

    assert float(typing.cast(float, result["datasets_per_minute"])) >= 0.0


def test_run_throughput_benchmark_updates_callback_on_measured_generation(
    monkeypatch,
) -> None:
    observed: list[int] = []

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        yield from range(num_datasets)

    def _stub_prepare_canonical_fixed_layout_run(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        precompute_classification_attempt_plan: bool = True,
    ):
        _ = (batch_size, precompute_classification_attempt_plan)
        return SimpleNamespace(
            config=_config,
            plan=object(),
            run_seed=int(seed or 0),
            batch_size=num_datasets,
        )

    def _stub_iter_prepared_canonical_batch_iter(
        _prepared,
        *,
        num_datasets: int,
        on_raw_batch_metrics=None,
    ):
        _ = on_raw_batch_metrics
        yield from range(num_datasets)

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput.prepare_canonical_fixed_layout_run",
        _stub_prepare_canonical_fixed_layout_run,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._iter_prepared_canonical_batch_iter",
        _stub_iter_prepared_canonical_batch_iter,
    )

    cfg = GeneratorConfig()
    run_throughput_benchmark(
        cfg,
        num_datasets=4,
        warmup_datasets=2,
        device="cpu",
        on_bundle=lambda bundle: observed.append(typing.cast(int, bundle)),
    )
    assert observed == [0, 1, 2, 3]


def test_run_throughput_benchmark_uses_sequential_generation(
    monkeypatch,
) -> None:
    warmup_calls: list[tuple[int, int, str | None]] = []
    prepare_calls: list[tuple[int, int, str | None, bool]] = []

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        warmup_calls.append((num_datasets, int(seed or 0), device))
        yield from range(num_datasets)

    def _stub_prepare_canonical_fixed_layout_run(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        precompute_classification_attempt_plan: bool = True,
    ):
        _ = batch_size
        prepare_calls.append(
            (num_datasets, int(seed or 0), device, bool(precompute_classification_attempt_plan))
        )
        return SimpleNamespace(
            config=_config,
            plan=object(),
            run_seed=int(seed or 0),
            batch_size=num_datasets,
        )

    def _stub_iter_prepared_canonical_batch_iter(
        _prepared,
        *,
        num_datasets: int,
        on_raw_batch_metrics=None,
    ):
        _ = on_raw_batch_metrics
        yield from range(num_datasets)

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput.prepare_canonical_fixed_layout_run",
        _stub_prepare_canonical_fixed_layout_run,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._iter_prepared_canonical_batch_iter",
        _stub_iter_prepared_canonical_batch_iter,
    )

    cfg = GeneratorConfig()
    cfg.runtime.device = "cpu"

    result = run_throughput_benchmark(
        cfg,
        num_datasets=4,
        warmup_datasets=2,
        device="cpu",
    )

    assert warmup_calls == [
        (2, KeyedRng(cfg.seed).child_seed("bench", "throughput", "warmup"), "cpu"),
    ]
    assert prepare_calls == [
        (4, KeyedRng(cfg.seed).child_seed("bench", "throughput", "measure"), "cpu", True),
    ]
    assert result["num_datasets"] == 4


def test_run_throughput_benchmark_fast_prepare_still_skips_retry_plan_for_regression(
    monkeypatch,
) -> None:
    prepare_calls: list[tuple[int, int, str | None, bool]] = []

    def _stub_prepare_canonical_fixed_layout_run(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        precompute_classification_attempt_plan: bool = True,
    ):
        _ = batch_size
        prepare_calls.append(
            (num_datasets, int(seed or 0), device, bool(precompute_classification_attempt_plan))
        )
        return SimpleNamespace(
            config=_config,
            plan=object(),
            run_seed=int(seed or 0),
            batch_size=num_datasets,
        )

    monkeypatch.setattr(
        "dagzoo.bench.throughput.prepare_canonical_fixed_layout_run",
        _stub_prepare_canonical_fixed_layout_run,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._iter_prepared_canonical_batch_iter",
        lambda _prepared, *, num_datasets, on_raw_batch_metrics=None: iter(range(num_datasets)),
    )

    cfg = _tiny_parallel_config()
    run_throughput_benchmark(
        cfg,
        num_datasets=4,
        warmup_datasets=0,
        device="cpu",
    )

    assert prepare_calls == [
        (4, KeyedRng(cfg.seed).child_seed("bench", "throughput", "measure"), "cpu", False),
    ]


def test_run_throughput_benchmark_synchronizes_accelerator_for_timed_cuda_path(
    monkeypatch,
) -> None:
    sync_calls: list[str | None] = []

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        yield from range(num_datasets)

    def _stub_prepare_canonical_fixed_layout_run(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        precompute_classification_attempt_plan: bool = True,
    ):
        _ = (batch_size, precompute_classification_attempt_plan)
        return SimpleNamespace(
            config=_config,
            plan=object(),
            run_seed=int(seed or 0),
            batch_size=num_datasets,
        )

    def _stub_iter_prepared_canonical_batch_iter(
        _prepared,
        *,
        num_datasets: int,
        on_raw_batch_metrics=None,
    ):
        _ = on_raw_batch_metrics
        yield from range(num_datasets)

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput.prepare_canonical_fixed_layout_run",
        _stub_prepare_canonical_fixed_layout_run,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._iter_prepared_canonical_batch_iter",
        _stub_iter_prepared_canonical_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._synchronize_accelerator",
        lambda device: sync_calls.append(device),
    )

    cfg = GeneratorConfig()
    cfg.runtime.device = "cuda"
    run_throughput_benchmark(
        cfg,
        num_datasets=4,
        warmup_datasets=2,
        device="cuda",
    )

    assert sync_calls == ["cuda", "cuda", "cuda"]


def test_run_throughput_benchmark_reports_generation_cpu_time(
    monkeypatch,
) -> None:
    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        yield from range(num_datasets)

    def _stub_prepare_canonical_fixed_layout_run(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        precompute_classification_attempt_plan: bool = True,
    ):
        _ = (batch_size, precompute_classification_attempt_plan)
        return SimpleNamespace(
            config=_config,
            plan=object(),
            run_seed=int(seed or 0),
            batch_size=num_datasets,
        )

    def _stub_iter_prepared_canonical_batch_iter(
        _prepared,
        *,
        num_datasets: int,
        on_raw_batch_metrics=None,
    ):
        _ = on_raw_batch_metrics
        yield from range(num_datasets)

    perf_counter_values = iter((0.0, 2.0, 5.0))
    process_time_values = iter((10.0, 10.25, 11.0))

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput.prepare_canonical_fixed_layout_run",
        _stub_prepare_canonical_fixed_layout_run,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._iter_prepared_canonical_batch_iter",
        _stub_iter_prepared_canonical_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput.time.perf_counter", lambda: next(perf_counter_values)
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput.time.process_time", lambda: next(process_time_values)
    )

    cfg = GeneratorConfig()
    result = run_throughput_benchmark(
        cfg,
        num_datasets=4,
        warmup_datasets=0,
        device="cpu",
    )

    assert result["prepare_elapsed_seconds"] == pytest.approx(2.0)
    assert result["prepare_cpu_time_seconds"] == pytest.approx(0.25)
    assert result["elapsed_seconds"] == pytest.approx(5.0)
    assert result["cpu_time_seconds"] == pytest.approx(1.0)
    assert result["raw_batch_elapsed_seconds"] == pytest.approx(0.0)
    assert result["node_apply_elapsed_seconds"] == pytest.approx(0.0)


def test_run_throughput_benchmark_aggregates_raw_batch_metrics(monkeypatch) -> None:
    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = (num_datasets, seed, device)
        return iter(())

    def _stub_prepare_canonical_fixed_layout_run(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        precompute_classification_attempt_plan: bool = True,
    ):
        _ = (batch_size, precompute_classification_attempt_plan)
        return SimpleNamespace(
            config=_config,
            plan=object(),
            run_seed=int(seed or 0),
            batch_size=num_datasets,
        )

    def _stub_iter_prepared_canonical_batch_iter(
        _prepared,
        *,
        num_datasets: int,
        on_raw_batch_metrics=None,
    ):
        assert on_raw_batch_metrics is not None
        on_raw_batch_metrics(
            {
                "raw_batch_elapsed_seconds": 1.5,
                "raw_batch_cpu_time_seconds": 0.75,
                "node_apply_elapsed_seconds": 0.8,
                "converter_elapsed_seconds": 0.2,
            }
        )
        on_raw_batch_metrics(
            {
                "raw_batch_elapsed_seconds": 0.5,
                "node_apply_elapsed_seconds": 0.3,
                "feature_materialization_elapsed_seconds": 0.1,
            }
        )
        yield from range(num_datasets)

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput.prepare_canonical_fixed_layout_run",
        _stub_prepare_canonical_fixed_layout_run,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._iter_prepared_canonical_batch_iter",
        _stub_iter_prepared_canonical_batch_iter,
    )

    result = run_throughput_benchmark(
        GeneratorConfig(),
        num_datasets=3,
        warmup_datasets=0,
        device="cpu",
    )

    assert result["raw_batch_elapsed_seconds"] == pytest.approx(2.0)
    assert result["raw_batch_cpu_time_seconds"] == pytest.approx(0.75)
    assert result["node_apply_elapsed_seconds"] == pytest.approx(1.1)
    assert result["converter_elapsed_seconds"] == pytest.approx(0.2)
    assert result["feature_materialization_elapsed_seconds"] == pytest.approx(0.1)


def test_run_throughput_benchmark_profile_runtime_adds_internal_metrics(monkeypatch) -> None:
    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = (num_datasets, seed, device)
        return iter(())

    def _stub_prepare_canonical_fixed_layout_run(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        precompute_classification_attempt_plan: bool = True,
    ):
        _ = (batch_size, precompute_classification_attempt_plan)
        return SimpleNamespace(
            config=_config,
            plan=object(),
            run_seed=int(seed or 0),
            batch_size=num_datasets,
        )

    def _stub_iter_prepared_canonical_batch_iter(
        _prepared,
        *,
        num_datasets: int,
        on_raw_batch_metrics=None,
    ):
        _ = on_raw_batch_metrics
        record_runtime_profile_metric("profile_fixed_layout_rng_keyed_count", 3.0)
        record_runtime_profile_metric("profile_rng_torch_generator_elapsed_seconds", 0.125)
        record_runtime_profile_metric("profile_node_apply_tree_elapsed_seconds", 0.25)
        record_runtime_profile_metric(
            "profile_node_apply_piecewise_exclusive_elapsed_seconds", 0.05
        )
        record_runtime_profile_metric("profile_node_apply_product_call_count", 2.0)
        yield from range(num_datasets)

    monkeypatch.setattr(
        "dagzoo.bench.throughput.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput.prepare_canonical_fixed_layout_run",
        _stub_prepare_canonical_fixed_layout_run,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._iter_prepared_canonical_batch_iter",
        _stub_iter_prepared_canonical_batch_iter,
    )

    result = run_throughput_benchmark(
        GeneratorConfig(),
        num_datasets=3,
        warmup_datasets=0,
        device="cpu",
        profile_runtime=True,
    )

    assert result["profile_fixed_layout_rng_keyed_count"] == pytest.approx(3.0)
    assert result["profile_rng_torch_generator_elapsed_seconds"] == pytest.approx(0.125)
    assert result["profile_node_apply_tree_elapsed_seconds"] == pytest.approx(0.25)
    assert result["profile_node_apply_piecewise_exclusive_elapsed_seconds"] == pytest.approx(0.05)
    assert result["profile_node_apply_product_call_count"] == pytest.approx(2.0)


def test_run_heterogeneous_throughput_benchmark_aggregates_stage_metrics(monkeypatch) -> None:
    iter_calls: list[tuple[int, int, str | None, bool]] = []
    sync_calls: list[str | None] = []

    def _stub_realize_generation_config_for_run(
        config: GeneratorConfig,
        *,
        seed: int | None = None,
        device: str | None = None,
        prefer_cpu_for_mps_auto: bool = False,
    ):
        assert prefer_cpu_for_mps_auto is True
        return config, int(seed or 0), str(device or config.runtime.device), "cpu", None

    def _stub_generate_batch_with_heterogeneous_layout_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        on_raw_batch_metrics=None,
    ):
        _ = batch_size
        iter_calls.append((num_datasets, int(seed or 0), device, on_raw_batch_metrics is not None))
        if on_raw_batch_metrics is not None:
            on_raw_batch_metrics(
                {
                    "heterogeneous_descriptor_resolution_elapsed_seconds": 0.4,
                    "heterogeneous_split_resolution_elapsed_seconds": 0.3,
                    "heterogeneous_postprocess_elapsed_seconds": 0.2,
                    "heterogeneous_metadata_finalization_elapsed_seconds": 0.1,
                }
            )
        yield from range(num_datasets)

    monkeypatch.setattr(
        "dagzoo.bench.throughput.realize_generation_config_for_run",
        _stub_realize_generation_config_for_run,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._generate_batch_with_heterogeneous_layout_iter",
        _stub_generate_batch_with_heterogeneous_layout_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._synchronize_accelerator",
        lambda device: sync_calls.append(device),
    )

    cfg = GeneratorConfig()
    result = run_heterogeneous_throughput_benchmark(
        cfg,
        num_datasets=3,
        warmup_datasets=2,
        device="auto",
    )

    assert iter_calls == [
        (2, KeyedRng(cfg.seed).child_seed("bench", "throughput", "warmup"), "auto", False),
        (3, KeyedRng(cfg.seed).child_seed("bench", "throughput", "measure"), "auto", True),
    ]
    assert sync_calls == ["cpu", "cpu"]
    assert result["generation_mode"] == "heterogeneous_grouped"
    assert result["heterogeneous_descriptor_resolution_elapsed_seconds"] == pytest.approx(0.4)
    assert result["heterogeneous_split_resolution_elapsed_seconds"] == pytest.approx(0.3)
    assert result["heterogeneous_postprocess_elapsed_seconds"] == pytest.approx(0.2)
    assert result["heterogeneous_metadata_finalization_elapsed_seconds"] == pytest.approx(0.1)


def test_run_stratified_throughput_benchmark_aggregates_scheduler_metrics(monkeypatch) -> None:
    iter_calls: list[tuple[int, int, str | None, bool]] = []
    sync_calls: list[str | None] = []

    def _stub_realize_generation_config_for_run(
        config: GeneratorConfig,
        *,
        seed: int | None = None,
        device: str | None = None,
        prefer_cpu_for_mps_auto: bool = False,
    ):
        assert prefer_cpu_for_mps_auto is True
        return config, int(seed or 0), str(device or config.runtime.device), "cpu", None

    def _stub_generate_batch_with_stratified_layout_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        on_raw_batch_metrics=None,
    ):
        _ = batch_size
        iter_calls.append((num_datasets, int(seed or 0), device, on_raw_batch_metrics is not None))
        if on_raw_batch_metrics is not None:
            on_raw_batch_metrics(
                {
                    "stratified_descriptor_window_fill_ratio_sum": 1.0,
                    "stratified_descriptor_window_count": 1.0,
                    "stratified_stratum_size_sum": 6.0,
                    "stratified_stratum_count": 2.0,
                    "stratified_executed_microbatch_size_sum": 6.0,
                    "stratified_executed_microbatch_count": 3.0,
                    "stratified_scalar_fallback_dataset_count": 1.0,
                    "stratified_scheduler_elapsed_seconds": 0.25,
                    "stratified_scheduler_cpu_time_seconds": 0.1,
                }
            )
        yield from range(num_datasets)

    monkeypatch.setattr(
        "dagzoo.bench.throughput.realize_generation_config_for_run",
        _stub_realize_generation_config_for_run,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._generate_batch_with_stratified_layout_iter",
        _stub_generate_batch_with_stratified_layout_iter,
    )
    monkeypatch.setattr(
        "dagzoo.bench.throughput._synchronize_accelerator",
        lambda device: sync_calls.append(device),
    )

    cfg = GeneratorConfig()
    result = run_stratified_throughput_benchmark(
        cfg,
        num_datasets=4,
        warmup_datasets=2,
        device="auto",
    )

    assert iter_calls == [
        (2, KeyedRng(cfg.seed).child_seed("bench", "throughput", "warmup"), "auto", False),
        (4, KeyedRng(cfg.seed).child_seed("bench", "throughput", "measure"), "auto", True),
    ]
    assert sync_calls == ["cpu", "cpu"]
    assert result["generation_mode"] == "heterogeneous_stratified"
    assert result["stratified_descriptor_window_fill_ratio_sum"] == pytest.approx(1.0)
    assert result["stratified_descriptor_window_count"] == pytest.approx(1.0)
    assert result["stratified_stratum_size_sum"] == pytest.approx(6.0)
    assert result["stratified_executed_microbatch_size_sum"] == pytest.approx(6.0)
    assert result["stratified_scalar_fallback_dataset_count"] == pytest.approx(1.0)
    assert result["stratified_scheduler_elapsed_seconds"] == pytest.approx(0.25)


def test_build_public_throughput_smoke_summary_reports_ratios_and_stage_shares(
    monkeypatch,
) -> None:
    fixed_result = {
        "datasets_per_minute": 2400.0,
        "elapsed_seconds": 5.0,
    }
    heterogeneous_result = {
        "datasets_per_minute": 900.0,
        "elapsed_seconds": 10.0,
        "heterogeneous_descriptor_resolution_elapsed_seconds": 3.0,
        "raw_batch_elapsed_seconds": 4.0,
        "heterogeneous_logical_cohort_count": 24.0,
        "heterogeneous_physical_microbatch_count": 5.0,
        "heterogeneous_physical_microbatch_size_sum": 11.0,
        "heterogeneous_physical_microbatch_predicted_utilization_sum": 4.2,
        "heterogeneous_executor_fallback_dataset_count": 7.0,
        "heterogeneous_supported_singleton_dataset_count": 6.0,
        "mixed_source_node_slot_count": 10.0,
        "mixed_source_bucket_count": 14.0,
        "mixed_source_bucket_dataset_sum": 21.0,
        "mixed_converter_bucket_count": 30.0,
        "mixed_converter_bucket_dataset_sum": 54.0,
    }
    stratified_result = {
        "datasets_per_minute": 1200.0,
        "elapsed_seconds": 8.0,
        "heterogeneous_descriptor_resolution_elapsed_seconds": 2.0,
        "stratified_scheduler_elapsed_seconds": 1.5,
        "raw_batch_elapsed_seconds": 3.0,
        "heterogeneous_logical_cohort_count": 18.0,
        "heterogeneous_physical_microbatch_count": 4.0,
        "heterogeneous_physical_microbatch_size_sum": 9.0,
        "heterogeneous_physical_microbatch_predicted_utilization_sum": 3.1,
        "heterogeneous_executor_fallback_dataset_count": 5.0,
        "heterogeneous_supported_singleton_dataset_count": 4.0,
        "mixed_source_node_slot_count": 8.0,
        "mixed_source_bucket_count": 12.0,
        "mixed_source_bucket_dataset_sum": 18.0,
        "mixed_converter_bucket_count": 24.0,
        "mixed_converter_bucket_dataset_sum": 42.0,
    }
    monkeypatch.setattr(
        "dagzoo.bench.public_throughput_smoke.run_throughput_benchmark",
        lambda *_args, **_kwargs: fixed_result,
    )
    monkeypatch.setattr(
        "dagzoo.bench.public_throughput_smoke.run_heterogeneous_throughput_benchmark",
        lambda *_args, **_kwargs: heterogeneous_result,
    )
    monkeypatch.setattr(
        "dagzoo.bench.public_throughput_smoke.run_stratified_throughput_benchmark",
        lambda *_args, **_kwargs: stratified_result,
    )

    cfg = _tiny_parallel_config()
    summary = build_public_throughput_smoke_summary(
        cfg,
        num_datasets=24,
        warmup_datasets=4,
        device="cpu",
    )

    result = summary["preset_results"][0]
    assert result["preset_key"] == "parallel_test"
    assert result["fixed_datasets_per_minute"] == pytest.approx(2400.0)
    assert result["heterogeneous_datasets_per_minute"] == pytest.approx(900.0)
    assert result["stratified_datasets_per_minute"] == pytest.approx(1200.0)
    assert result["heterogeneous_vs_fixed_ratio"] == pytest.approx(0.375)
    assert result["stratified_vs_fixed_ratio"] == pytest.approx(0.5)
    assert result["heterogeneous_descriptor_share"] == pytest.approx(0.3)
    assert result["heterogeneous_raw_batch_share"] == pytest.approx(0.4)
    assert result["heterogeneous_logical_cohort_count"] == pytest.approx(24.0)
    assert result["heterogeneous_mixed_physical_dataset_count"] == pytest.approx(11.0)
    assert result["heterogeneous_executor_fallback_dataset_count"] == pytest.approx(7.0)
    assert result["heterogeneous_supported_singleton_dataset_count"] == pytest.approx(6.0)
    assert result["heterogeneous_avg_predicted_utilization"] == pytest.approx(0.84)
    assert result["heterogeneous_avg_supported_buckets_per_node_slot"] == pytest.approx(1.4)
    assert result["heterogeneous_avg_datasets_per_supported_bucket"] == pytest.approx(1.5)
    assert result["heterogeneous_converter_bucket_count"] == pytest.approx(30.0)
    assert result["heterogeneous_avg_converter_bucket_size"] == pytest.approx(1.8)
    assert result["stratified_descriptor_share"] == pytest.approx(0.25)
    assert result["stratified_scheduler_share"] == pytest.approx(0.1875)
    assert result["stratified_raw_batch_share"] == pytest.approx(0.375)
    assert result["stratified_logical_cohort_count"] == pytest.approx(18.0)
    assert result["stratified_mixed_physical_dataset_count"] == pytest.approx(9.0)
    assert result["stratified_executor_fallback_dataset_count"] == pytest.approx(5.0)
    assert result["stratified_supported_singleton_dataset_count"] == pytest.approx(4.0)
    assert result["stratified_avg_predicted_utilization"] == pytest.approx(0.775)
    assert result["stratified_avg_supported_buckets_per_node_slot"] == pytest.approx(1.5)
    assert result["stratified_avg_datasets_per_supported_bucket"] == pytest.approx(1.5)
    assert result["stratified_converter_bucket_count"] == pytest.approx(24.0)
    assert result["stratified_avg_converter_bucket_size"] == pytest.approx(1.75)

    baseline = build_public_throughput_smoke_baseline_payload(summary)
    assert baseline["suite"] == "public_throughput_smoke"
    assert baseline["presets"]["parallel_test"]["heterogeneous_vs_fixed_ratio"] == pytest.approx(
        0.375
    )
    assert baseline["presets"]["parallel_test"]["stratified_scheduler_share"] == pytest.approx(
        0.1875
    )


def test_run_microbenchmarks_emits_heterogeneous_generate_one_metric(
    monkeypatch,
) -> None:
    observed_layout_modes: list[str] = []

    def _stub_generate_one(
        config: GeneratorConfig,
        *,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        observed_layout_modes.append(str(config.runtime.layout_mode))
        return None

    def _stub_time_ms(func, repeats: int, *, device: str | None = None) -> float:
        _ = (repeats, device)
        func()
        return 1.0

    monkeypatch.setattr("dagzoo.bench.micro.generate_one", _stub_generate_one)
    monkeypatch.setattr("dagzoo.bench.micro._time_ms", _stub_time_ms)

    result = run_microbenchmarks(
        _tiny_parallel_config(),
        device="cpu",
        repeats=1,
        include_generate_one=True,
    )

    assert result["micro_generate_one_ms"] == pytest.approx(1.0)
    assert result["micro_generate_one_heterogeneous_ms"] == pytest.approx(1.0)
    assert observed_layout_modes.count("stratified") == 1
    assert observed_layout_modes.count("heterogeneous") == 1


def test_run_throughput_benchmark_callback_exception_does_not_hang_parallel_path() -> None:
    cfg = _tiny_parallel_config()

    result_queue: queue.Queue[BaseException | None] = queue.Queue()

    def _run_benchmark() -> None:
        try:
            run_throughput_benchmark(
                cfg,
                num_datasets=6,
                warmup_datasets=0,
                device="cpu",
                on_bundle=lambda _bundle: (_ for _ in ()).throw(RuntimeError("callback boom")),
            )
        except BaseException as exc:  # pragma: no cover - surfaced via queue assertion
            result_queue.put(exc)
            return
        result_queue.put(None)

    benchmark_thread = threading.Thread(target=_run_benchmark, daemon=True)
    benchmark_thread.start()
    benchmark_thread.join(timeout=5.0)

    assert not benchmark_thread.is_alive()
    error = result_queue.get_nowait()
    assert isinstance(error, RuntimeError)
    assert str(error) == "callback boom"


def test_run_throughput_benchmark_fast_prepare_handles_classification_retries() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.seed = 1
    cfg.dataset.task = "classification"
    cfg.dataset.n_train = 24
    cfg.dataset.n_test = 8
    cfg.dataset.n_features_min = 4
    cfg.dataset.n_features_max = 6
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 4
    cfg.filter.enabled = False
    cfg.filter.max_attempts = 2
    cfg.runtime.device = "cpu"
    cfg.runtime.fixed_layout_target_cells = 2000

    result = run_throughput_benchmark(
        cfg,
        num_datasets=6,
        warmup_datasets=0,
        device="cpu",
    )

    assert result["num_datasets"] == 6
    assert float(typing.cast(float, result["datasets_per_minute"])) > 0.0


def test_run_fixed_layout_target_cells_sweep_uses_tier_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        "dagzoo.bench.throughput.detect_hardware",
        lambda _requested_device: HardwareInfo(
            backend="cpu",
            requested_device="cpu",
            device_name="cpu",
            total_memory_gb=None,
            peak_flops=float("inf"),
            tier="cpu",
        ),
    )

    observed_target_cells: list[int] = []

    def _stub_run_throughput_benchmark(
        cfg: GeneratorConfig,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        on_bundle=None,
    ) -> dict[str, typing.Any]:
        _ = warmup_datasets
        _ = device
        _ = on_bundle
        observed_target_cells.append(int(cfg.runtime.fixed_layout_target_cells or 0))
        return {
            "preset": cfg.benchmark.preset_name,
            "num_datasets": num_datasets,
            "warmup_datasets": 1,
            "elapsed_seconds": 1.0,
            "datasets_per_second": float(cfg.runtime.fixed_layout_target_cells or 0) / 1_000_000.0,
            "datasets_per_minute": float(cfg.runtime.fixed_layout_target_cells or 0) / 100_000.0,
            "generation_mode": "fixed_batched",
        }

    monkeypatch.setattr(
        "dagzoo.bench.throughput.run_throughput_benchmark",
        _stub_run_throughput_benchmark,
    )

    cfg = _tiny_parallel_config()
    sweep = run_fixed_layout_target_cells_sweep(
        cfg,
        num_datasets=3,
        warmup_datasets=1,
        device="cpu",
    )

    assert observed_target_cells == [4_000_000, 8_000_000, 12_000_000, 16_000_000]
    assert sweep["recommended_fixed_layout_target_cells"] == 16_000_000
    assert sweep["target_cells_values"] == [4_000_000, 8_000_000, 12_000_000, 16_000_000]


def test_run_fixed_layout_target_cells_sweep_anchors_cuda_candidates_at_floor(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.bench.throughput.detect_hardware",
        lambda _requested_device: HardwareInfo(
            backend="cuda",
            requested_device="cuda",
            device_name="NVIDIA H100 SXM",
            total_memory_gb=80.0,
            peak_flops=989e12,
            tier="cuda_h100",
        ),
    )

    observed_target_cells: list[int] = []

    def _stub_run_throughput_benchmark(
        cfg: GeneratorConfig,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        on_bundle=None,
    ) -> dict[str, typing.Any]:
        _ = num_datasets
        _ = warmup_datasets
        _ = device
        _ = on_bundle
        target_cells = int(cfg.runtime.fixed_layout_target_cells or 0)
        observed_target_cells.append(target_cells)
        return {
            "preset": cfg.benchmark.preset_name,
            "num_datasets": 3,
            "warmup_datasets": 1,
            "elapsed_seconds": 1.0,
            "datasets_per_second": float(target_cells) / 1_000_000.0,
            "datasets_per_minute": float(target_cells) / 100_000.0,
            "generation_mode": "fixed_batched",
        }

    monkeypatch.setattr(
        "dagzoo.bench.throughput.run_throughput_benchmark",
        _stub_run_throughput_benchmark,
    )

    cfg = _tiny_parallel_config()
    cfg.runtime.device = "cuda"
    sweep = run_fixed_layout_target_cells_sweep(
        cfg,
        num_datasets=3,
        warmup_datasets=1,
        device="cuda",
    )

    assert observed_target_cells == [240_000_000, 256_000_000]
    assert sweep["recommended_fixed_layout_target_cells"] == 256_000_000
    assert sweep["target_cells_values"] == [240_000_000, 256_000_000]


def test_run_fixed_layout_target_cells_sweep_preserves_explicit_candidates_on_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.bench.throughput.detect_hardware",
        lambda _requested_device: HardwareInfo(
            backend="cuda",
            requested_device="cuda",
            device_name="NVIDIA H100 SXM",
            total_memory_gb=80.0,
            peak_flops=989e12,
            tier="cuda_h100",
        ),
    )

    observed_target_cells: list[int] = []

    def _stub_run_throughput_benchmark(
        cfg: GeneratorConfig,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        on_bundle=None,
    ) -> dict[str, typing.Any]:
        _ = num_datasets
        _ = warmup_datasets
        _ = device
        _ = on_bundle
        target_cells = int(cfg.runtime.fixed_layout_target_cells or 0)
        observed_target_cells.append(target_cells)
        return {
            "preset": cfg.benchmark.preset_name,
            "num_datasets": 3,
            "warmup_datasets": 1,
            "elapsed_seconds": 1.0,
            "datasets_per_second": 1.0,
            "datasets_per_minute": 1.0,
            "generation_mode": "fixed_batched",
        }

    monkeypatch.setattr(
        "dagzoo.bench.throughput.run_throughput_benchmark",
        _stub_run_throughput_benchmark,
    )

    cfg = _tiny_parallel_config()
    cfg.runtime.device = "cuda"
    sweep = run_fixed_layout_target_cells_sweep(
        cfg,
        num_datasets=3,
        warmup_datasets=1,
        device="cuda",
        target_cells_values=[32_000_000, 64_000_000],
    )

    assert observed_target_cells == [32_000_000, 64_000_000]
    assert sweep["target_cells_values"] == [32_000_000, 64_000_000]


def test_run_fixed_layout_target_cells_sweep_rejects_invalid_candidates() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        run_fixed_layout_target_cells_sweep(
            _tiny_parallel_config(),
            num_datasets=2,
            warmup_datasets=0,
            device="cpu",
            target_cells_values=[4_000_000, 0],
        )
