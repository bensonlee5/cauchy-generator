import queue
import threading
import typing
from types import SimpleNamespace

import pytest

from dagzoo.bench.throughput import (
    run_fixed_layout_target_cells_sweep,
    run_throughput_benchmark,
)
from dagzoo.config import GeneratorConfig
from dagzoo.hardware import HardwareInfo
from dagzoo.rng import KeyedRng


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
    prepare_calls: list[tuple[int, int, str | None]] = []
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
    ):
        _ = batch_size
        prepare_calls.append((num_datasets, int(seed or 0), device))
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
        (3, KeyedRng(cfg.seed).child_seed("bench", "throughput", "measure"), "cpu"),
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
    ):
        _ = batch_size
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
    prepare_calls: list[tuple[int, int, str | None]] = []

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
    ):
        _ = batch_size
        prepare_calls.append((num_datasets, int(seed or 0), device))
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
        (4, KeyedRng(cfg.seed).child_seed("bench", "throughput", "measure"), "cpu"),
    ]
    assert result["num_datasets"] == 4


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
    ):
        _ = batch_size
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
    ):
        _ = batch_size
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
    ):
        _ = batch_size
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
