from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import load_repo_config

import dagzoo.bench.guardrails as guardrails_mod
import dagzoo.bench.micro as micro_mod
import dagzoo.bench.runtime_support as runtime_support_mod
import dagzoo.bench.suite as suite_mod
from dagzoo.bench.metrics import reproducibility_signatures
from dagzoo.bench.micro import run_microbenchmarks
from dagzoo.bench.report import write_suite_markdown
from dagzoo.bench.suite import PresetRunSpec, resolve_preset_run_specs, run_benchmark_suite
from dagzoo.config import GeneratorConfig
from dagzoo.hardware import HardwareInfo
from dagzoo.rng import KeyedRng
from dagzoo.types import DatasetBundle


def _set_attrs(target: object, **attrs: object) -> None:
    for name, value in attrs.items():
        setattr(target, name, value)


def _tiny_cpu_config() -> GeneratorConfig:
    cfg = load_repo_config()
    _set_attrs(
        cfg.dataset,
        task="regression",
        n_train=32,
        n_test=8,
        n_features_min=8,
        n_features_max=8,
    )
    _set_attrs(cfg.runtime, device="cpu")
    _set_attrs(cfg.graph, n_nodes_min=2, n_nodes_max=6)
    _set_attrs(
        cfg.benchmark,
        num_datasets=2,
        warmup_datasets=0,
        latency_num_samples=2,
        reproducibility_num_datasets=1,
        preset_name="cpu_test",
    )
    cfg.benchmark.presets["cpu_test"] = {
        "device": "cpu",
        "num_datasets": 2,
        "warmup_datasets": 0,
    }
    return cfg


def _tiny_missingness_cpu_config() -> GeneratorConfig:
    cfg = _tiny_cpu_config()
    _set_attrs(
        cfg.dataset,
        missing_rate=0.25,
        missing_mechanism="mcar",
    )
    return cfg


def _tiny_shift_cpu_config() -> GeneratorConfig:
    cfg = _tiny_cpu_config()
    _set_attrs(cfg.shift, enabled=True, mode="mixed")
    _set_attrs(cfg.graph, n_nodes_min=8, n_nodes_max=12)
    return cfg


def _tiny_noise_cpu_config() -> GeneratorConfig:
    cfg = _tiny_cpu_config()
    _set_attrs(
        cfg.noise,
        family="laplace",
        base_scale=1.0,
        student_t_df=6.0,
        mixture_weights=None,
    )
    return cfg


def _stub_fixed_layout_evidence() -> dict[str, int]:
    return {
        "fixed_layout_target_cells_effective": 4_000_000,
        "fixed_layout_per_dataset_cells": 160,
        "fixed_layout_realized_batch_size": 2,
        "fixed_layout_chunk_count": 1,
        "fixed_layout_tail_chunk_size": 2,
    }


def test_reproducibility_workload_signature_ignores_values_but_tracks_layout_metadata() -> None:
    bundle_a = DatasetBundle(
        X_train=np.zeros((2, 2), dtype=np.float32),
        y_train=np.zeros(2, dtype=np.int64),
        X_test=np.ones((1, 2), dtype=np.float32),
        y_test=np.ones(1, dtype=np.int64),
        feature_types=["num", "num"],
        metadata={
            "seed": 11,
            "dataset_seed": 21,
            "dataset_index": 0,
            "attempt_used": 0,
            "layout_signature": "layout-a",
            "layout_plan_signature": "plan-a",
            "n_features": 2,
            "n_categorical_features": 0,
            "n_classes": None,
            "graph_nodes": 3,
            "graph_edges": 2,
            "graph_depth_nodes": 2,
            "noise_distribution": {
                "family_requested": "mixture",
                "family_sampled": "gaussian",
            },
        },
    )
    bundle_b = DatasetBundle(
        X_train=np.full((2, 2), 7.0, dtype=np.float32),
        y_train=np.full(2, 4, dtype=np.int64),
        X_test=np.full((1, 2), 9.0, dtype=np.float32),
        y_test=np.full(1, 5, dtype=np.int64),
        feature_types=["num", "num"],
        metadata=dict(bundle_a.metadata),
    )
    bundle_c = DatasetBundle(
        X_train=np.zeros((2, 2), dtype=np.float32),
        y_train=np.zeros(2, dtype=np.int64),
        X_test=np.ones((1, 2), dtype=np.float32),
        y_test=np.ones(1, dtype=np.int64),
        feature_types=["num", "num"],
        metadata={**bundle_a.metadata, "layout_signature": "layout-b"},
    )
    bundle_d = DatasetBundle(
        X_train=np.zeros((2, 2), dtype=np.float32),
        y_train=np.zeros(2, dtype=np.int64),
        X_test=np.ones((1, 2), dtype=np.float32),
        y_test=np.ones(1, dtype=np.int64),
        feature_types=["num", "num"],
        metadata={
            **bundle_a.metadata,
            "noise_distribution": {
                "family_requested": "mixture",
                "family_sampled": "laplace",
            },
        },
    )

    content_a, workload_a = reproducibility_signatures([bundle_a])
    content_b, workload_b = reproducibility_signatures([bundle_b])
    _, workload_c = reproducibility_signatures([bundle_c])
    _, workload_d = reproducibility_signatures([bundle_d])

    assert content_a != content_b
    assert workload_a == workload_b
    assert workload_a != workload_c
    assert workload_a != workload_d


def test_run_benchmark_suite_smoke_single_profile() -> None:
    cfg = _tiny_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    assert summary["suite"] == "smoke"
    assert summary["regression"]["status"] in {"pass", "warn", "fail"}
    if summary["regression"]["status"] != "pass":
        assert any(
            issue["metric"] == "lineage_export_runtime_degradation_pct"
            for issue in summary["regression"]["issues"]
        )
    assert len(summary["preset_results"]) == 1

    result = summary["preset_results"][0]
    assert result["preset_key"] == "cpu_test"
    assert result["generation_mode"] == "fixed_batched"
    assert result["datasets_per_minute"] > 0
    assert result["generation_datasets_per_minute"] == pytest.approx(result["datasets_per_minute"])
    assert result["write_datasets_per_minute"] >= 0.0
    assert result["filter_datasets_per_minute"] is None
    assert result["filter_accepted_datasets_per_minute"] is None
    assert result["filter_stage_enabled"] is False
    assert "accepted_datasets_measured" not in result
    assert result["filter_rejection_rate_attempt_level"] is None
    assert result["filter_acceptance_rate_dataset_level"] is None
    assert result["filter_rejection_rate_dataset_level"] is None
    assert result["filter_retry_dataset_rate"] is None
    assert result["filter_attempts_total"] == 0
    assert result["filter_rejections_total"] == 0
    assert result["filter_accepted_datasets_measured"] == 0
    assert result["filter_rejected_datasets_measured"] == 0
    assert result["stage_sample_datasets"] >= 1
    assert result["total_attempts"] >= result["stage_sample_datasets"]
    assert result["mean_attempts_per_dataset"] >= 1.0
    assert result["estimated_attempts_per_minute"] >= result["generation_datasets_per_minute"]
    assert result["latency_p95_ms"] >= 0
    lineage_guardrails = result["lineage_guardrails"]
    assert lineage_guardrails["enabled"] is True
    assert lineage_guardrails["status"] in {"pass", "warn", "fail"}


def test_run_preset_benchmark_synchronizes_before_cuda_memory_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    cfg.runtime.device = "cuda"
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cuda")
    hardware = HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="NVIDIA H100 NVL",
        total_memory_gb=94.0,
        peak_flops=835e12,
        tier="cuda_h100",
    )
    sync_calls: list[str | None] = []
    rss_values = iter([100.0, 120.0])

    monkeypatch.setattr(
        suite_mod,
        "resolve_benchmark_preset_config",
        lambda **_kwargs: SimpleNamespace(
            config=cfg,
            trace_events=[],
            requested_device="cuda",
            hardware=hardware,
        ),
    )
    monkeypatch.setattr(
        suite_mod,
        "realize_generation_config_for_run",
        lambda *_args, **_kwargs: (cfg, int(cfg.seed), "cuda", "cuda"),
    )
    monkeypatch.setattr(
        suite_mod,
        "run_throughput_benchmark",
        lambda *_args, **_kwargs: {
            "preset": cfg.benchmark.preset_name,
            "num_datasets": 2,
            "warmup_datasets": 0,
            "elapsed_seconds": 1.0,
            "datasets_per_second": 1.0,
            "datasets_per_minute": 60.0,
            "slo_pass_100_datasets_per_min": False,
            "generation_mode": "fixed_batched",
        },
    )
    monkeypatch.setattr(
        suite_mod,
        "_collect_latency",
        lambda *_args, **_kwargs: {
            "latency_samples": 1.0,
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        },
    )
    monkeypatch.setattr(
        suite_mod,
        "_collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )
    monkeypatch.setattr(
        suite_mod,
        "_build_fixed_layout_evidence",
        lambda *_args, **_kwargs: _stub_fixed_layout_evidence(),
    )
    monkeypatch.setattr(suite_mod, "_peak_rss_mb", lambda: next(rss_values))
    monkeypatch.setattr(
        suite_mod,
        "_synchronize_accelerator",
        lambda device: sync_calls.append(device),
    )
    monkeypatch.setattr(suite_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(suite_mod.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(suite_mod.torch.cuda, "max_memory_allocated", lambda: 2.0 * suite_mod.MIB)
    monkeypatch.setattr(suite_mod.torch.cuda, "max_memory_reserved", lambda: 3.0 * suite_mod.MIB)

    result = suite_mod.run_preset_benchmark(
        spec,
        suite="smoke",
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=True,
        collect_reproducibility=False,
        include_micro=False,
        hardware_policy="none",
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        diagnostics_occurrence_index=0,
        diagnostics_occurrence_total=1,
    )

    assert result["peak_rss_mb"] == pytest.approx(20.0)
    assert result["peak_cuda_allocated_mb"] == pytest.approx(2.0)
    assert result["peak_cuda_reserved_mb"] == pytest.approx(3.0)
    assert result["peak_cuda_allocated_pct_of_total_memory"] == pytest.approx(
        2.0 / (94.0 * 1024.0) * 100.0
    )
    assert result["peak_cuda_reserved_pct_of_total_memory"] == pytest.approx(
        3.0 / (94.0 * 1024.0) * 100.0
    )
    assert result["peak_cuda_headroom_mb"] == pytest.approx((94.0 * 1024.0) - 3.0)
    assert sync_calls == ["cuda", "cuda"]


def test_run_benchmark_suite_builtin_cpu_uses_canonical_generation_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    _set_attrs(cfg.benchmark, preset_name="cpu")
    cfg.benchmark.presets["cpu"] = {
        "device": "cpu",
        "num_datasets": 2,
        "warmup_datasets": 0,
    }
    spec = PresetRunSpec(key="cpu", config=cfg, device="cpu")

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "dagzoo.bench.suite.run_throughput_benchmark",
        lambda config, *, num_datasets, warmup_datasets=10, device=None, on_bundle=None: (
            captured.update(
                {
                    "rows_total": int(config.dataset.n_train + config.dataset.n_test),
                    "device": device,
                    "num_datasets": num_datasets,
                }
            )
            or {
                "preset": config.benchmark.preset_name,
                "num_datasets": num_datasets,
                "warmup_datasets": warmup_datasets,
                "elapsed_seconds": 1.0,
                "datasets_per_second": float(num_datasets),
                "datasets_per_minute": float(num_datasets) * 60.0,
                "slo_pass_100_datasets_per_min": True,
                "generation_mode": "fixed_batched",
            }
        ),
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_latency",
        lambda *_args, **_kwargs: {
            "latency_samples": 1.0,
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        },
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    assert result["generation_mode"] == "fixed_batched"
    assert captured["rows_total"] == int(result["dataset_rows_total"])
    assert captured["device"] == "cpu"


def test_resolve_preset_run_specs_expands_builtin_cpu_rows() -> None:
    specs = resolve_preset_run_specs(preset_keys=["cpu"], config_path=None)

    assert [spec.key for spec in specs] == ["cpu_rows1024", "cpu_rows4096", "cpu_rows8192"]
    assert [int(spec.config.dataset.n_train + spec.config.dataset.n_test) for spec in specs] == [
        1024,
        4096,
        8192,
    ]


def test_resolve_preset_run_specs_uses_recipe_name_for_custom_recipe_config() -> None:
    specs = resolve_preset_run_specs(
        preset_keys=["custom"],
        config_path="recipe:default-baseline",
    )

    assert len(specs) == 1
    assert specs[0].key == "default-baseline"
    assert specs[0].config.benchmark.preset_name == "medium_cuda"


def test_resolve_preset_run_specs_preserves_yaml_custom_preset_name() -> None:
    specs = resolve_preset_run_specs(
        preset_keys=["custom"],
        config_path="configs/preset_shift_benchmark_smoke.yaml",
    )

    assert len(specs) == 1
    assert specs[0].key == "shift_smoke"


def test_run_benchmark_suite_emits_stage_and_filter_pressure_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    _set_attrs(cfg.filter, enabled=True)
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    def _stub_throughput(
        _config,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        _fixed_layout_plan=None,
        _fixed_layout_batch_size: int | None = None,
        on_bundle=None,
    ):
        _ = warmup_datasets
        _ = device
        if on_bundle is not None:
            for i in range(num_datasets):
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata={
                            "seed": i,
                            "attempt_used": i,
                            "filter": {"enabled": True, "accepted": True},
                            "generation_attempts": {
                                "total_attempts": i + 1,
                                "retry_count": i,
                                "filter_attempts": i + 1,
                                "filter_rejections": i,
                            },
                        },
                    )
                )
        dpm = 120.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        return {
            "preset": "cpu_test",
            "num_datasets": num_datasets,
            "warmup_datasets": warmup_datasets,
            "prepare_elapsed_seconds": 0.2,
            "prepare_cpu_time_seconds": 0.1,
            "elapsed_seconds": elapsed,
            "raw_batch_elapsed_seconds": 0.6,
            "raw_batch_cpu_time_seconds": 0.3,
            "node_apply_elapsed_seconds": 0.4,
            "node_apply_cpu_time_seconds": 0.2,
            "converter_elapsed_seconds": 0.1,
            "converter_cpu_time_seconds": 0.05,
            "feature_materialization_elapsed_seconds": 0.08,
            "feature_materialization_cpu_time_seconds": 0.04,
            "datasets_per_second": dps,
            "datasets_per_minute": dpm,
            "slo_pass_100_datasets_per_min": True,
        }

    monkeypatch.setattr("dagzoo.bench.suite.run_throughput_benchmark", _stub_throughput)
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_latency",
        lambda _cfg, *, device, num_samples: {
            "latency_samples": float(num_samples) + (0.0 if device is None else 0.0),
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        },
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite.measure_write_stage_metrics",
        lambda bundles, *, config: SimpleNamespace(
            datasets_per_minute=float(len(bundles)) * 10.0 + float(config.output.shard_size) * 0.0,
            elapsed_seconds=0.5,
            cpu_time_seconds=0.25,
            bytes_written=2048,
            mib_per_second=1.0,
        ),
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite.measure_filter_stage_metrics",
        lambda bundles, *, config: SimpleNamespace(
            datasets_per_minute=float(len(bundles)) * 20.0
            + float(config.filter.n_estimators) * 0.0,
            filter_attempts_total=3,
            filter_accepted_datasets=2,
            filter_rejections_total=1,
            filter_rejected_datasets=1,
            elapsed_seconds=0.75,
            cpu_time_seconds=0.5,
        ),
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite._build_fixed_layout_evidence",
        lambda *_args, **_kwargs: _stub_fixed_layout_evidence(),
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    assert result["generation_datasets_per_minute"] == pytest.approx(120.0)
    assert result["prepare_elapsed_seconds"] == pytest.approx(0.2)
    assert result["prepare_cpu_time_seconds"] == pytest.approx(0.1)
    assert result["prepare_cpu_busy_pct_of_wall"] == pytest.approx(50.0)
    assert result["generation_elapsed_seconds"] == pytest.approx(1.0)
    assert result["generation_cpu_time_seconds"] == pytest.approx(0.0)
    assert result["generation_cpu_busy_pct_of_wall"] == pytest.approx(0.0)
    assert result["raw_batch_elapsed_seconds"] == pytest.approx(0.6)
    assert result["raw_batch_cpu_time_seconds"] == pytest.approx(0.3)
    assert result["node_apply_elapsed_seconds"] == pytest.approx(0.4)
    assert result["converter_elapsed_seconds"] == pytest.approx(0.1)
    assert result["feature_materialization_elapsed_seconds"] == pytest.approx(0.08)
    assert result["write_datasets_per_minute"] == pytest.approx(20.0)
    assert result["write_stage_elapsed_seconds"] == pytest.approx(0.5)
    assert result["write_stage_cpu_time_seconds"] == pytest.approx(0.25)
    assert result["write_stage_bytes_written"] == 2048
    assert result["write_stage_mib_per_second"] == pytest.approx(1.0)
    assert result["filter_datasets_per_minute"] == pytest.approx(40.0)
    assert result["filter_accepted_datasets_per_minute"] == pytest.approx(40.0 * (2.0 / 3.0))
    assert result["filter_stage_elapsed_seconds"] == pytest.approx(0.75)
    assert result["filter_stage_cpu_time_seconds"] == pytest.approx(0.5)
    assert result["filter_stage_enabled"] is True
    assert "accepted_datasets_measured" not in result
    assert result["total_attempts"] == 3
    assert result["mean_attempts_per_dataset"] == pytest.approx(1.5)
    assert result["estimated_attempts_per_minute"] == pytest.approx(180.0)
    assert result["retry_dataset_count"] == 1
    assert result["retry_dataset_rate"] == pytest.approx(0.5)
    assert result["filter_attempts_total"] == 3
    assert result["filter_rejections_total"] == 1
    assert result["filter_accepted_datasets_measured"] == 2
    assert result["filter_rejected_datasets_measured"] == 1
    assert result["filter_acceptance_rate_dataset_level"] == pytest.approx(2.0 / 3.0)
    assert result["filter_rejection_rate_dataset_level"] == pytest.approx(1.0 / 3.0)
    assert result["filter_rejection_rate_attempt_level"] == pytest.approx(1.0 / 3.0)
    assert result["filter_retry_dataset_count"] == 1
    assert result["filter_retry_dataset_rate"] == pytest.approx(0.5)
    assert result["fixed_layout_target_cells_effective"] == 4_000_000
    assert result["fixed_layout_per_dataset_cells"] == 160
    assert result["fixed_layout_realized_batch_size"] == 2
    assert result["fixed_layout_chunk_count"] == 1
    assert result["fixed_layout_tail_chunk_size"] == 2


def test_run_benchmark_suite_filter_enabled_uses_filter_disabled_generation_config_everywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    _set_attrs(cfg.filter, enabled=True)
    _set_attrs(
        cfg.dataset,
        missing_rate=0.25,
        missing_mechanism="mcar",
    )
    _set_attrs(cfg.shift, enabled=True, mode="mixed")
    _set_attrs(cfg.noise, family="laplace")
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    throughput_filter_flags: list[bool] = []
    latency_filter_flags: list[bool] = []
    repro_filter_flags: list[bool] = []
    micro_filter_flags: list[bool] = []
    lineage_filter_flags: list[bool] = []

    def _stub_throughput(
        config,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        _fixed_layout_plan=None,
        _fixed_layout_batch_size: int | None = None,
        on_bundle=None,
    ):
        _ = warmup_datasets
        _ = device
        throughput_filter_flags.append(bool(config.filter.enabled))
        dpm = 100.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None:
            for i in range(num_datasets):
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata={
                            "seed": i,
                            "attempt_used": 0,
                            "generation_attempts": {
                                "total_attempts": 1,
                                "filter_attempts": 0,
                                "filter_rejections": 0,
                            },
                            "missingness": {"missing_count_overall": 4},
                            "graph_edge_density": 0.3,
                            "shift": {
                                "enabled": bool(config.shift.enabled),
                                "edge_odds_multiplier": 1.0,
                                "mechanism_nonlinear_mass": 0.6,
                                "noise_variance_multiplier": 1.0,
                            },
                            "noise_distribution": {
                                "family_requested": str(config.noise.family),
                                "family_sampled": str(config.noise.family),
                                "sampling_strategy": "dataset_level",
                                "base_scale": float(config.noise.base_scale),
                                "student_t_df": float(config.noise.student_t_df),
                                "mixture_weights": None,
                            },
                        },
                    )
                )
        return {
            "preset": config.benchmark.preset_name,
            "num_datasets": num_datasets,
            "warmup_datasets": warmup_datasets,
            "elapsed_seconds": elapsed,
            "datasets_per_second": dps,
            "datasets_per_minute": dpm,
            "slo_pass_100_datasets_per_min": True,
        }

    def _stub_latency(config, *, device: str | None, num_samples: int) -> dict[str, float]:
        _ = device
        latency_filter_flags.append(bool(config.filter.enabled))
        return {
            "latency_samples": float(num_samples),
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        }

    def _stub_repro(config, *, device: str | None, num_datasets: int) -> dict[str, object]:
        _ = device
        repro_filter_flags.append(bool(config.filter.enabled))
        return {
            "reproducibility_datasets": int(num_datasets),
            "reproducibility_signature": "stub",
            "reproducibility_match": True,
            "reproducibility_workload_signature": "workload",
            "reproducibility_workload_match": True,
        }

    def _stub_micro(
        config,
        *,
        device: str | None,
        repeats: int,
        include_generate_one: bool = True,
    ) -> dict[str, float | int | None]:
        _ = device
        _ = include_generate_one
        micro_filter_flags.append(bool(config.filter.enabled))
        return {
            "micro_repeats": int(repeats),
            "micro_random_function_linear_ms": 1.0,
            "micro_node_pipeline_ms": 1.0,
            "micro_generate_one_ms": 1.0,
        }

    def _stub_lineage(
        config,
        *,
        suite: str,
        num_datasets: int,
        device: str | None,
        warn_threshold_pct: float,
        fail_threshold_pct: float,
    ) -> dict[str, object]:
        _ = suite
        _ = num_datasets
        _ = device
        _ = warn_threshold_pct
        _ = fail_threshold_pct
        lineage_filter_flags.append(bool(config.filter.enabled))
        return {"enabled": False}

    monkeypatch.setattr("dagzoo.bench.suite.run_throughput_benchmark", _stub_throughput)
    monkeypatch.setattr("dagzoo.bench.suite._collect_latency", _stub_latency)
    monkeypatch.setattr("dagzoo.bench.suite._collect_reproducibility", _stub_repro)
    monkeypatch.setattr("dagzoo.bench.suite.run_microbenchmarks", _stub_micro)
    monkeypatch.setattr("dagzoo.bench.suite._collect_lineage_guardrails", _stub_lineage)
    monkeypatch.setattr(
        "dagzoo.bench.suite.measure_filter_stage_metrics",
        lambda bundles, *, config: SimpleNamespace(
            datasets_per_minute=float(len(bundles)) * 10.0 + float(config.filter.n_jobs) * 0.0,
            filter_attempts_total=int(len(bundles)),
            filter_accepted_datasets=int(len(bundles)),
            filter_rejections_total=0,
            filter_rejected_datasets=0,
        ),
    )

    summary = run_benchmark_suite(
        [spec],
        suite="full",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=1,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    assert result["filter_stage_enabled"] is True
    assert len(throughput_filter_flags) == 4
    assert throughput_filter_flags == [False, False, False, False]
    assert latency_filter_flags == [False]
    assert repro_filter_flags == [False]
    assert micro_filter_flags == [False]
    assert lineage_filter_flags == [False]


def test_run_benchmark_suite_filter_retry_rate_uses_stage_sample_denominator_when_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    _set_attrs(cfg.filter, enabled=True)
    _set_attrs(cfg.benchmark, latency_num_samples=2)
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    def _stub_throughput(
        config,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        _fixed_layout_plan=None,
        _fixed_layout_batch_size: int | None = None,
        on_bundle=None,
    ):
        _ = warmup_datasets
        _ = device
        assert bool(config.filter.enabled) is False
        dpm = 120.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None:
            for i in range(num_datasets):
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata={
                            "seed": i,
                            "attempt_used": 0,
                            "generation_attempts": {
                                "total_attempts": 1,
                                "retry_count": 0,
                                "filter_attempts": 1,
                                "filter_rejections": 0,
                            },
                        },
                    )
                )
        return {
            "preset": "cpu_test",
            "num_datasets": num_datasets,
            "warmup_datasets": warmup_datasets,
            "elapsed_seconds": elapsed,
            "datasets_per_second": dps,
            "datasets_per_minute": dpm,
            "slo_pass_100_datasets_per_min": True,
        }

    monkeypatch.setattr("dagzoo.bench.suite.run_throughput_benchmark", _stub_throughput)
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_latency",
        lambda _cfg, *, device, num_samples: {
            "latency_samples": float(num_samples) + (0.0 if device is None else 0.0),
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        },
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite.measure_filter_stage_metrics",
        lambda bundles, *, config: SimpleNamespace(
            datasets_per_minute=float(len(bundles)) * 20.0
            + float(config.filter.n_estimators) * 0.0,
            filter_attempts_total=int(len(bundles)),
            filter_accepted_datasets=max(0, int(len(bundles)) - 1),
            filter_rejections_total=1,
            filter_rejected_datasets=1,
        ),
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=4,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    assert "accepted_datasets_measured" not in result
    assert result["stage_sample_datasets"] == 2
    assert result["filter_attempts_total"] == 2
    assert result["filter_rejections_total"] == 1
    assert result["filter_accepted_datasets_per_minute"] == pytest.approx(20.0)
    assert result["filter_accepted_datasets_measured"] == 1
    assert result["filter_rejected_datasets_measured"] == 1
    assert result["filter_acceptance_rate_dataset_level"] == pytest.approx(0.5)
    assert result["filter_rejection_rate_dataset_level"] == pytest.approx(0.5)
    assert result["filter_retry_dataset_count"] == 1
    assert result["filter_retry_dataset_rate"] == pytest.approx(0.5)


def test_missingness_control_run_uses_equivalent_callback_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_missingness_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    original_stage_collector = suite_mod.StageSampleCollector
    stage_collectors: list[object] = []

    class _SpyStageSampleCollector:
        def __init__(self, *, max_samples: int) -> None:
            self._delegate = original_stage_collector(max_samples=max_samples)
            self.bundles = self._delegate.bundles
            stage_collectors.append(self)

        def update(self, bundle: DatasetBundle) -> None:
            self._delegate.update(bundle)

    original_throughput_collector = suite_mod._ThroughputPressureCollector

    class _SpyThroughputPressureCollector:
        instances = 0
        updates = 0

        def __init__(self) -> None:
            type(self).instances += 1
            self._delegate = original_throughput_collector()

        def update(self, bundle: DatasetBundle) -> None:
            type(self).updates += 1
            self._delegate.update(bundle)

        def build_summary(self) -> dict[str, object]:
            return self._delegate.build_summary()

    calls: list[bool] = []

    def _stub_throughput(
        config,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        _fixed_layout_plan=None,
        _fixed_layout_batch_size: int | None = None,
        on_bundle=None,
    ):
        _ = warmup_datasets
        _ = device
        missing_enabled = float(config.dataset.missing_rate) > 0.0
        calls.append(missing_enabled)
        dpm = 80.0 if missing_enabled else 100.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None:
            for i in range(num_datasets):
                metadata: dict[str, object] = {"seed": i, "attempt_used": 0}
                if missing_enabled:
                    metadata["missingness"] = {"missing_count_overall": 4}
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata=metadata,
                    )
                )
        return {
            "preset": config.benchmark.preset_name,
            "num_datasets": num_datasets,
            "warmup_datasets": warmup_datasets,
            "elapsed_seconds": elapsed,
            "datasets_per_second": dps,
            "datasets_per_minute": dpm,
            "slo_pass_100_datasets_per_min": dpm >= 100.0,
        }

    monkeypatch.setattr("dagzoo.bench.suite.StageSampleCollector", _SpyStageSampleCollector)
    monkeypatch.setattr(
        "dagzoo.bench.suite._ThroughputPressureCollector",
        _SpyThroughputPressureCollector,
    )
    monkeypatch.setattr("dagzoo.bench.suite.run_throughput_benchmark", _stub_throughput)
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_latency",
        lambda _cfg, *, device, num_samples: {
            "latency_samples": float(num_samples) + (0.0 if device is None else 0.0),
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        },
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    assert result["missingness_guardrails"]["enabled"] is True
    assert calls == [True, False]
    assert len(stage_collectors) == 2
    assert all(len(getattr(c, "bundles")) == 0 for c in stage_collectors)
    assert _SpyThroughputPressureCollector.instances == 2
    assert _SpyThroughputPressureCollector.updates == 4


def test_run_benchmark_suite_releases_stage_samples_before_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    original_stage_collector = suite_mod.StageSampleCollector
    stage_collectors: list[object] = []

    class _SpyStageSampleCollector:
        def __init__(self, *, max_samples: int) -> None:
            self._delegate = original_stage_collector(max_samples=max_samples)
            self.bundles = self._delegate.bundles
            stage_collectors.append(self)

        def update(self, bundle: DatasetBundle) -> None:
            self._delegate.update(bundle)

    def _stub_throughput(
        _config,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        _fixed_layout_plan=None,
        _fixed_layout_batch_size: int | None = None,
        on_bundle=None,
    ):
        _ = warmup_datasets
        _ = device
        if on_bundle is not None:
            for i in range(num_datasets):
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata={"seed": i, "attempt_used": 0},
                    )
                )
        dpm = 120.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        return {
            "preset": "cpu_test",
            "num_datasets": num_datasets,
            "warmup_datasets": warmup_datasets,
            "elapsed_seconds": elapsed,
            "datasets_per_second": dps,
            "datasets_per_minute": dpm,
            "slo_pass_100_datasets_per_min": True,
        }

    def _stub_collect_latency(_cfg, *, device: str | None, num_samples: int) -> dict[str, float]:
        _ = device
        _ = num_samples
        assert stage_collectors
        assert all(len(getattr(c, "bundles")) == 0 for c in stage_collectors)
        return {
            "latency_samples": 1.0,
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        }

    monkeypatch.setattr("dagzoo.bench.suite.StageSampleCollector", _SpyStageSampleCollector)
    monkeypatch.setattr("dagzoo.bench.suite.run_throughput_benchmark", _stub_throughput)
    monkeypatch.setattr("dagzoo.bench.suite._collect_latency", _stub_collect_latency)
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    assert result["stage_sample_datasets"] == 2


def test_run_benchmark_suite_missingness_guardrails_emit_metrics() -> None:
    cfg = _tiny_missingness_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=100.0,
        fail_threshold_pct=200.0,
        baseline_payload=None,
        num_datasets_override=4,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    guardrails = result["missingness_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] == "pass"
    assert guardrails["metadata_coverage_rate"] == pytest.approx(1.0)
    assert 0.0 <= guardrails["realized_rate_overall"] <= 1.0
    assert guardrails["runtime_baseline_datasets_per_minute"] > 0.0
    lineage_guardrails = result["lineage_guardrails"]
    assert lineage_guardrails["enabled"] is True
    assert lineage_guardrails["status"] in {"pass", "warn", "fail"}


def test_run_benchmark_suite_missingness_runtime_guardrail_updates_regression_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_missingness_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")
    calls: list[dict[str, bool]] = []

    def _stub_throughput(
        config,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        _fixed_layout_plan=None,
        _fixed_layout_batch_size: int | None = None,
        on_bundle=None,
    ):
        _ = warmup_datasets
        _ = device
        missing_enabled = float(config.dataset.missing_rate) > 0.0
        calls.append(
            {
                "missing_enabled": missing_enabled,
                "has_callback": on_bundle is not None,
            }
        )
        dpm = 80.0 if missing_enabled else 100.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None and missing_enabled:
            for i in range(num_datasets):
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata={
                            "seed": i,
                            "attempt_used": 0,
                            "missingness": {"missing_count_overall": 4},
                        },
                    )
                )
        return {
            "preset": config.benchmark.preset_name,
            "num_datasets": num_datasets,
            "warmup_datasets": warmup_datasets,
            "elapsed_seconds": elapsed,
            "datasets_per_second": dps,
            "datasets_per_minute": dpm,
            "slo_pass_100_datasets_per_min": dpm >= 100.0,
        }

    monkeypatch.setattr("dagzoo.bench.suite.run_throughput_benchmark", _stub_throughput)
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_latency",
        lambda _cfg, *, device, num_samples: {
            "latency_samples": float(num_samples) + (0.0 if device is None else 0.0),
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        },
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=1,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    guardrails = result["missingness_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] == "fail"
    assert any(
        issue["metric"] == "missingness_runtime_degradation_pct" and issue["severity"] == "fail"
        for issue in guardrails["issues"]
    )
    assert summary["regression"]["status"] == "fail"
    assert any(
        issue["metric"] == "missingness_runtime_degradation_pct"
        for issue in summary["regression"]["issues"]
    )
    baseline_calls = [call for call in calls if not call["missing_enabled"]]
    assert baseline_calls
    assert all(call["has_callback"] for call in baseline_calls)


def test_run_benchmark_suite_shift_guardrails_emit_metrics() -> None:
    cfg = _tiny_shift_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    guardrails = result["shift_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["mode"] == "mixed"
    assert guardrails["status"] in {"pass", "warn", "fail"}
    assert guardrails["runtime_gating_enabled"] is False
    assert guardrails["directional_gating_enabled"] is False
    assert "directional_checks" in guardrails


def test_run_benchmark_suite_shift_runtime_guardrail_updates_regression_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_shift_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")
    calls: list[dict[str, bool]] = []

    def _stub_throughput(
        config,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        _fixed_layout_plan=None,
        _fixed_layout_batch_size: int | None = None,
        on_bundle=None,
    ):
        _ = warmup_datasets
        _ = device
        shift_enabled = bool(config.shift.enabled)
        calls.append(
            {
                "shift_enabled": shift_enabled,
                "has_callback": on_bundle is not None,
            }
        )
        dpm = 70.0 if shift_enabled else 100.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None:
            for i in range(num_datasets):
                metadata = {
                    "seed": i,
                    "attempt_used": 0,
                    "graph_edge_density": 0.35 if shift_enabled else 0.20,
                    "shift": {
                        "enabled": shift_enabled,
                        "edge_odds_multiplier": 1.3 if shift_enabled else 1.0,
                        "mechanism_nonlinear_mass": 0.72 if shift_enabled else 0.62,
                        "noise_variance_multiplier": 1.25 if shift_enabled else 1.0,
                    },
                }
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata=metadata,
                    )
                )
        return {
            "preset": config.benchmark.preset_name,
            "num_datasets": num_datasets,
            "warmup_datasets": warmup_datasets,
            "elapsed_seconds": elapsed,
            "datasets_per_second": dps,
            "datasets_per_minute": dpm,
            "slo_pass_100_datasets_per_min": dpm >= 100.0,
        }

    monkeypatch.setattr("dagzoo.bench.suite.run_throughput_benchmark", _stub_throughput)
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_latency",
        lambda _cfg, *, device, num_samples: {
            "latency_samples": float(num_samples) + (0.0 if device is None else 0.0),
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        },
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=6,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    guardrails = result["shift_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["runtime_gating_enabled"] is True
    assert guardrails["directional_gating_enabled"] is True
    assert guardrails["status"] == "fail"
    assert any(
        issue["metric"] == "shift_runtime_degradation_pct" and issue["severity"] == "fail"
        for issue in guardrails["issues"]
    )
    assert summary["regression"]["status"] == "fail"
    assert any(
        issue["metric"] == "shift_runtime_degradation_pct"
        for issue in summary["regression"]["issues"]
    )
    baseline_calls = [call for call in calls if not call["shift_enabled"]]
    assert baseline_calls
    assert all(call["has_callback"] for call in baseline_calls)


def test_run_benchmark_suite_shift_directional_guardrail_failure_updates_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_shift_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    def _stub_throughput(
        config,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        _fixed_layout_plan=None,
        _fixed_layout_batch_size: int | None = None,
        on_bundle=None,
    ):
        _ = warmup_datasets
        _ = device
        shift_enabled = bool(config.shift.enabled)
        dpm = 100.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None:
            for i in range(num_datasets):
                metadata = {
                    "seed": i,
                    "attempt_used": 0,
                    "graph_edge_density": 0.20,
                    "shift": {
                        "enabled": shift_enabled,
                        "edge_odds_multiplier": 1.0,
                        "mechanism_nonlinear_mass": 0.60 if shift_enabled else 0.62,
                        "noise_variance_multiplier": 1.0,
                    },
                }
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata=metadata,
                    )
                )
        return {
            "preset": config.benchmark.preset_name,
            "num_datasets": num_datasets,
            "warmup_datasets": warmup_datasets,
            "elapsed_seconds": elapsed,
            "datasets_per_second": dps,
            "datasets_per_minute": dpm,
            "slo_pass_100_datasets_per_min": True,
        }

    monkeypatch.setattr("dagzoo.bench.suite.run_throughput_benchmark", _stub_throughput)
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_latency",
        lambda _cfg, *, device, num_samples: {
            "latency_samples": float(num_samples) + (0.0 if device is None else 0.0),
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        },
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=6,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    guardrails = summary["preset_results"][0]["shift_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] == "fail"
    mechanism_issue = next(
        issue
        for issue in guardrails["issues"]
        if issue["metric"] == "shift_mechanism_nonlinear_mass_directionality"
    )
    expected_degradation = ((0.62 - 0.60) / 0.62) * 100.0
    assert mechanism_issue["degradation_pct"] == pytest.approx(expected_degradation)
    assert summary["regression"]["status"] == "fail"
    regression_issue = next(
        issue
        for issue in summary["regression"]["issues"]
        if issue["metric"] == "shift_mechanism_nonlinear_mass_directionality"
    )
    assert regression_issue["degradation_pct"] == pytest.approx(expected_degradation)


def test_run_benchmark_suite_noise_guardrails_emit_metrics() -> None:
    cfg = _tiny_noise_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    guardrails = result["noise_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] in {"pass", "warn", "fail"}
    assert guardrails["family_requested"] == "laplace"
    assert guardrails["runtime_gating_enabled"] is False
    assert guardrails["metadata_coverage_rate"] == pytest.approx(1.0)


def test_run_benchmark_suite_noise_runtime_guardrail_updates_regression_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_noise_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    def _stub_throughput(
        config,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        _fixed_layout_plan=None,
        _fixed_layout_batch_size: int | None = None,
        on_bundle=None,
    ):
        _ = warmup_datasets
        _ = device
        nongaussian_noise = str(config.noise.family) != "gaussian"
        dpm = 70.0 if nongaussian_noise else 100.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None:
            for i in range(num_datasets):
                metadata = {
                    "seed": i,
                    "attempt_used": 0,
                    "noise_distribution": {
                        "family_requested": str(config.noise.family),
                        "family_sampled": str(config.noise.family),
                        "sampling_strategy": "dataset_level",
                        "base_scale": float(config.noise.base_scale),
                        "student_t_df": float(config.noise.student_t_df),
                        "mixture_weights": None,
                    },
                }
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata=metadata,
                    )
                )
        return {
            "preset": config.benchmark.preset_name,
            "num_datasets": num_datasets,
            "warmup_datasets": warmup_datasets,
            "elapsed_seconds": elapsed,
            "datasets_per_second": dps,
            "datasets_per_minute": dpm,
            "slo_pass_100_datasets_per_min": dpm >= 100.0,
        }

    monkeypatch.setattr("dagzoo.bench.suite.run_throughput_benchmark", _stub_throughput)
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_latency",
        lambda _cfg, *, device, num_samples: {
            "latency_samples": float(num_samples) + (0.0 if device is None else 0.0),
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        },
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=6,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    guardrails = result["noise_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["runtime_gating_enabled"] is True
    assert guardrails["status"] == "fail"
    assert any(
        issue["metric"] == "noise_runtime_degradation_pct" and issue["severity"] == "fail"
        for issue in guardrails["issues"]
    )
    assert summary["regression"]["status"] == "fail"
    assert any(
        issue["metric"] == "noise_runtime_degradation_pct"
        for issue in summary["regression"]["issues"]
    )


def test_run_benchmark_suite_noise_metadata_coverage_failure_updates_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_noise_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    def _stub_throughput(
        config,
        *,
        num_datasets: int,
        warmup_datasets: int = 10,
        device: str | None = None,
        _fixed_layout_plan=None,
        _fixed_layout_batch_size: int | None = None,
        on_bundle=None,
    ):
        _ = warmup_datasets
        _ = device
        dpm = 100.0
        dps = dpm / 60.0
        elapsed = (float(num_datasets) / dps) if dps > 0 else 0.0
        if on_bundle is not None:
            for i in range(num_datasets):
                metadata = {"seed": i, "attempt_used": 0}
                if str(config.noise.family) == "gaussian":
                    metadata["noise_distribution"] = {
                        "family_requested": "gaussian",
                        "family_sampled": "gaussian",
                        "sampling_strategy": "dataset_level",
                        "base_scale": 1.0,
                        "student_t_df": 5.0,
                        "mixture_weights": None,
                    }
                on_bundle(
                    DatasetBundle(
                        X_train=np.zeros((3, 4), dtype=np.float32),
                        y_train=np.zeros(3, dtype=np.int64),
                        X_test=np.zeros((1, 4), dtype=np.float32),
                        y_test=np.zeros(1, dtype=np.int64),
                        feature_types=["num", "num", "num", "num"],
                        metadata=metadata,
                    )
                )
        return {
            "preset": config.benchmark.preset_name,
            "num_datasets": num_datasets,
            "warmup_datasets": warmup_datasets,
            "elapsed_seconds": elapsed,
            "datasets_per_second": dps,
            "datasets_per_minute": dpm,
            "slo_pass_100_datasets_per_min": dpm >= 100.0,
        }

    monkeypatch.setattr("dagzoo.bench.suite.run_throughput_benchmark", _stub_throughput)
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_latency",
        lambda _cfg, *, device, num_samples: {
            "latency_samples": float(num_samples) + (0.0 if device is None else 0.0),
            "latency_mean_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_min_ms": 1.0,
            "latency_max_ms": 1.0,
        },
    )
    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {"enabled": False},
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=6,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    guardrails = result["noise_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] == "fail"
    assert any(
        issue["metric"] == "noise_metadata_coverage" and issue["severity"] == "fail"
        for issue in guardrails["issues"]
    )
    assert summary["regression"]["status"] == "fail"
    assert any(
        issue["metric"] == "noise_metadata_coverage" for issue in summary["regression"]["issues"]
    )


def test_build_shift_directional_check_reports_none_degradation_when_baseline_zero() -> None:
    payload, issue = suite_mod._build_shift_directional_check(
        metric="graph_edge_density",
        enabled=True,
        gating_enabled=True,
        current=0.0,
        baseline=0.0,
        detail="graph axis should increase",
    )

    assert payload["status"] == "fail"
    assert issue is not None
    assert issue["metric"] == "shift_graph_edge_density_directionality"
    assert issue["degradation_pct"] is None


def test_run_benchmark_suite_lineage_runtime_guardrail_updates_regression_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    monkeypatch.setattr(
        "dagzoo.bench.suite._collect_lineage_guardrails",
        lambda *_args, **_kwargs: {
            "enabled": True,
            "sample_datasets": 2,
            "lineage_metadata_coverage_rate": 1.0,
            "runtime_baseline_datasets_per_minute": 100.0,
            "runtime_with_lineage_datasets_per_minute": 70.0,
            "runtime_degradation_pct": 30.0,
            "runtime_warn_threshold_pct": 10.0,
            "runtime_fail_threshold_pct": 20.0,
            "issues": [
                {
                    "metric": "lineage_export_runtime_degradation_pct",
                    "severity": "fail",
                    "current": 70.0,
                    "baseline": 100.0,
                    "degradation_pct": 30.0,
                    "detail": "lineage overhead exceeded threshold",
                }
            ],
            "status": "fail",
        },
    )

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=False,
        diagnostics_root_dir=None,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    guardrails = result["lineage_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] == "fail"
    assert summary["regression"]["status"] == "fail"
    assert any(
        issue["metric"] == "lineage_export_runtime_degradation_pct"
        for issue in summary["regression"]["issues"]
    )


def test_write_suite_markdown_profile_table_includes_shift_and_noise_columns(
    tmp_path: Path,
) -> None:
    summary = {
        "suite": "smoke",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "preset_results": [
            {
                "preset_key": "shift_smoke",
                "device": "cpu",
                "hardware_backend": "cpu",
                "datasets_per_minute": 120.0,
                "filter_accepted_datasets_per_minute": 90.0,
                "elapsed_seconds": 1.0,
                "latency_p95_ms": 4.0,
                "peak_rss_mb": 10.0,
                "reproducibility_match": True,
                "reproducibility_workload_match": False,
                "prepare_elapsed_seconds": 0.4,
                "prepare_cpu_time_seconds": 0.2,
                "prepare_cpu_busy_pct_of_wall": 50.0,
                "generation_elapsed_seconds": 1.0,
                "generation_cpu_time_seconds": 0.3,
                "generation_cpu_busy_pct_of_wall": 30.0,
                "raw_batch_elapsed_seconds": 0.7,
                "raw_batch_cpu_time_seconds": 0.35,
                "node_apply_elapsed_seconds": 0.5,
                "node_apply_cpu_time_seconds": 0.25,
                "converter_elapsed_seconds": 0.2,
                "converter_cpu_time_seconds": 0.1,
                "feature_materialization_elapsed_seconds": 0.12,
                "feature_materialization_cpu_time_seconds": 0.06,
                "fixed_layout_target_cells_effective": 4_000_000,
                "fixed_layout_per_dataset_cells": 1024,
                "fixed_layout_realized_batch_size": 4,
                "fixed_layout_chunk_count": 1,
                "fixed_layout_tail_chunk_size": 4,
                "stage_sample_datasets": 2,
                "write_stage_elapsed_seconds": 0.2,
                "write_stage_cpu_time_seconds": 0.1,
                "write_stage_bytes_written": 4096,
                "write_stage_mib_per_second": 2.0,
                "filter_stage_elapsed_seconds": 0.4,
                "filter_stage_cpu_time_seconds": 0.2,
                "peak_cuda_reserved_mb": 512.0,
                "peak_cuda_reserved_pct_of_total_memory": 25.0,
                "peak_cuda_headroom_mb": 1536.0,
                "filter_acceptance_rate_dataset_level": 0.75,
                "filter_rejection_rate_attempt_level": 0.25,
                "filter_rejection_rate_dataset_level": 0.25,
                "filter_retry_dataset_rate": 0.25,
                "diagnostics_enabled": False,
                "missingness_guardrails": {"enabled": False},
                "lineage_guardrails": {"enabled": False},
                "shift_guardrails": {"enabled": True, "status": "pass"},
                "noise_guardrails": {"enabled": True, "status": "warn"},
            }
        ],
        "regression": {"status": "pass", "issues": []},
    }
    path = write_suite_markdown(summary, tmp_path / "summary.md")
    text = path.read_text(encoding="utf-8")
    assert "| Shift |" in text
    assert "| Noise |" in text
    assert "| Repro |" in text
    assert "- Preparation:" in text
    assert "| Workload |" in text
    assert "Filter Accepted/min" in text
    assert "Filter Accept % (dataset)" in text
    assert "Filter Reject % (attempt)" in text
    assert "Filter Reject % (dataset)" in text
    assert "Filter Retry % (dataset)" in text
    assert "match" in text
    assert "mismatch" in text
    assert "## Bottleneck Evidence" in text
    assert "- Raw batch:" in text
    assert "node_apply_wall=0.500s" in text
    assert "target_cells=4000000" in text
    assert "reserved_mb=512.00" in text
    assert "| shift_smoke |" in text


def test_collect_lineage_guardrails_uses_median_of_three_trials_with_runtime_gating_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for i in range(num_datasets):
            yield DatasetBundle(
                X_train=np.zeros((3, 4), dtype=np.float32),
                y_train=np.zeros(3, dtype=np.int64),
                X_test=np.zeros((1, 4), dtype=np.float32),
                y_test=np.zeros(1, dtype=np.int64),
                feature_types=["num", "num", "num", "num"],
                metadata={
                    "seed": i,
                    "attempt_used": 0,
                    "lineage": {"schema_name": "dagzoo.dag_lineage"},
                },
            )

    trial_values = iter([100.0, 90.0, 1000.0, 100.0, 100.0, 90.0])

    def _stub_measure(
        _bundles,
        *,
        config: GeneratorConfig,
        num_bundles: int,
    ) -> float:
        _ = config
        assert not isinstance(_bundles, list)
        assert num_bundles > 0
        return float(next(trial_values))

    monkeypatch.setattr("dagzoo.bench.guardrails.generate_batch_iter", _stub_generate_batch_iter)
    monkeypatch.setattr(
        "dagzoo.bench.guardrails._measure_persistence_datasets_per_minute",
        _stub_measure,
    )

    guardrails = guardrails_mod._collect_lineage_guardrails(
        cfg,
        suite="smoke",
        num_datasets=2,
        device="cpu",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
    )

    assert guardrails["enabled"] is True
    assert guardrails["runtime_gating_enabled"] is False
    assert guardrails["runtime_gating_min_sample_datasets"] == 6
    assert guardrails["runtime_gating_suppressed_reason"] == "insufficient_sample_size"
    assert guardrails["runtime_trials"] == 3
    assert guardrails["runtime_baseline_trials_dpm"] == [100.0, 1000.0, 100.0]
    assert guardrails["runtime_with_lineage_trials_dpm"] == [90.0, 100.0, 90.0]
    assert guardrails["runtime_baseline_datasets_per_minute"] == pytest.approx(100.0)
    assert guardrails["runtime_with_lineage_datasets_per_minute"] == pytest.approx(90.0)
    assert guardrails["runtime_degradation_pct"] == pytest.approx(10.0)
    assert guardrails["status"] == "pass"
    assert not any(
        issue["metric"] == "lineage_export_runtime_degradation_pct"
        for issue in guardrails["issues"]
    )


def test_collect_lineage_guardrails_smoke_suppresses_runtime_issue_at_sample_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    _set_attrs(cfg.benchmark, latency_num_samples=6)

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for i in range(num_datasets):
            yield DatasetBundle(
                X_train=np.zeros((3, 4), dtype=np.float32),
                y_train=np.zeros(3, dtype=np.int64),
                X_test=np.zeros((1, 4), dtype=np.float32),
                y_test=np.zeros(1, dtype=np.int64),
                feature_types=["num", "num", "num", "num"],
                metadata={
                    "seed": i,
                    "attempt_used": 0,
                    "lineage": {"schema_name": "dagzoo.dag_lineage"},
                },
            )

    trial_values = iter([100.0, 70.0, 100.0, 70.0, 100.0, 70.0])

    def _stub_measure(
        _bundles,
        *,
        config: GeneratorConfig,
        num_bundles: int,
    ) -> float:
        _ = config
        assert not isinstance(_bundles, list)
        assert num_bundles > 0
        return float(next(trial_values))

    monkeypatch.setattr("dagzoo.bench.guardrails.generate_batch_iter", _stub_generate_batch_iter)
    monkeypatch.setattr(
        "dagzoo.bench.guardrails._measure_persistence_datasets_per_minute",
        _stub_measure,
    )

    guardrails = guardrails_mod._collect_lineage_guardrails(
        cfg,
        suite="smoke",
        num_datasets=6,
        device="cpu",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
    )

    assert guardrails["enabled"] is True
    assert guardrails["sample_datasets"] == 5
    assert guardrails["runtime_gating_enabled"] is False
    assert guardrails["runtime_gating_min_sample_datasets"] == 6
    assert guardrails["runtime_gating_suppressed_reason"] == "insufficient_sample_size"
    assert guardrails["runtime_degradation_pct"] == pytest.approx(30.0)
    assert guardrails["status"] == "pass"
    assert not any(
        issue["metric"] == "lineage_export_runtime_degradation_pct" and issue["severity"] == "fail"
        for issue in guardrails["issues"]
    )


def test_collect_lineage_guardrails_emits_runtime_issue_for_standard_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    _set_attrs(cfg.benchmark, latency_num_samples=6)

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for i in range(num_datasets):
            yield DatasetBundle(
                X_train=np.zeros((3, 4), dtype=np.float32),
                y_train=np.zeros(3, dtype=np.int64),
                X_test=np.zeros((1, 4), dtype=np.float32),
                y_test=np.zeros(1, dtype=np.int64),
                feature_types=["num", "num", "num", "num"],
                metadata={
                    "seed": i,
                    "attempt_used": 0,
                    "lineage": {"schema_name": "dagzoo.dag_lineage"},
                },
            )

    trial_values = iter([100.0, 70.0, 100.0, 70.0, 100.0, 70.0])

    def _stub_measure(
        _bundles,
        *,
        config: GeneratorConfig,
        num_bundles: int,
    ) -> float:
        _ = config
        assert not isinstance(_bundles, list)
        assert num_bundles > 0
        return float(next(trial_values))

    monkeypatch.setattr("dagzoo.bench.guardrails.generate_batch_iter", _stub_generate_batch_iter)
    monkeypatch.setattr(
        "dagzoo.bench.guardrails._measure_persistence_datasets_per_minute",
        _stub_measure,
    )

    guardrails = guardrails_mod._collect_lineage_guardrails(
        cfg,
        suite="standard",
        num_datasets=6,
        device="cpu",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
    )

    assert guardrails["enabled"] is True
    assert guardrails["sample_datasets"] == 6
    assert guardrails["runtime_gating_enabled"] is True
    assert guardrails["runtime_gating_min_sample_datasets"] == 5
    assert guardrails["runtime_gating_suppressed_reason"] is None
    assert guardrails["runtime_degradation_pct"] == pytest.approx(30.0)
    assert guardrails["status"] == "fail"
    assert any(
        issue["metric"] == "lineage_export_runtime_degradation_pct" and issue["severity"] == "fail"
        for issue in guardrails["issues"]
    )


def test_collect_lineage_guardrails_reports_unavailable_for_non_runtime_persistence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for i in range(num_datasets):
            yield DatasetBundle(
                X_train=np.zeros((3, 4), dtype=np.float32),
                y_train=np.zeros(3, dtype=np.int64),
                X_test=np.zeros((1, 4), dtype=np.float32),
                y_test=np.zeros(1, dtype=np.int64),
                feature_types=["num", "num", "num", "num"],
                metadata={
                    "seed": i,
                    "attempt_used": 0,
                    "lineage": {"schema_name": "dagzoo.dag_lineage"},
                },
            )

    def _stub_trials(
        *,
        baseline_stage_dir: Path,
        current_stage_dir: Path,
        num_bundles: int,
        config: GeneratorConfig,
        trials: int,
    ) -> tuple[list[float], list[float]]:
        _ = baseline_stage_dir
        _ = current_stage_dir
        _ = num_bundles
        _ = config
        _ = trials
        raise ValueError("codec unavailable")

    monkeypatch.setattr("dagzoo.bench.guardrails.generate_batch_iter", _stub_generate_batch_iter)
    monkeypatch.setattr(
        "dagzoo.bench.guardrails._measure_lineage_persistence_trials",
        _stub_trials,
    )

    guardrails = guardrails_mod._collect_lineage_guardrails(
        cfg,
        suite="smoke",
        num_datasets=2,
        device="cpu",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
    )

    assert guardrails["enabled"] is False
    assert guardrails["reason"] == "unavailable"
    assert "codec unavailable" in guardrails["detail"]


def test_measure_lineage_persistence_trials_replays_staged_bundles_without_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    baseline_stage_dir = tmp_path / "baseline"
    current_stage_dir = tmp_path / "current"
    baseline_stage_dir.mkdir(parents=True, exist_ok=True)
    current_stage_dir.mkdir(parents=True, exist_ok=True)

    bundle = DatasetBundle(
        X_train=np.zeros((3, 4), dtype=np.float32),
        y_train=np.zeros(3, dtype=np.int64),
        X_test=np.zeros((1, 4), dtype=np.float32),
        y_test=np.zeros(1, dtype=np.int64),
        feature_types=["num", "num", "num", "num"],
        metadata={"seed": 0, "attempt_used": 0, "lineage": {"schema_name": "x"}},
    )
    for idx in range(2):
        file_name = f"bundle_{idx:06d}.pkl"
        guardrails_mod._stage_bundle(current_stage_dir / file_name, bundle, strip_lineage=False)
        guardrails_mod._stage_bundle(baseline_stage_dir / file_name, bundle, strip_lineage=True)

    def _unexpected_generate(*_args, **_kwargs):
        raise AssertionError("generate_batch_iter must not run during timed persistence trials")

    call_counts: list[int] = []

    def _stub_measure(
        bundles,
        *,
        config: GeneratorConfig,
        num_bundles: int,
    ) -> float:
        _ = config
        seen = sum(1 for _ in bundles)
        call_counts.append(seen)
        assert num_bundles == 2
        assert seen == 2
        return 100.0 if len(call_counts) % 2 == 1 else 90.0

    monkeypatch.setattr("dagzoo.bench.guardrails.generate_batch_iter", _unexpected_generate)
    monkeypatch.setattr(
        "dagzoo.bench.guardrails._measure_persistence_datasets_per_minute",
        _stub_measure,
    )

    baseline_trials, current_trials = guardrails_mod._measure_lineage_persistence_trials(
        baseline_stage_dir=baseline_stage_dir,
        current_stage_dir=current_stage_dir,
        num_bundles=2,
        config=cfg,
        trials=3,
    )

    assert baseline_trials == [100.0, 100.0, 100.0]
    assert current_trials == [90.0, 90.0, 90.0]
    assert call_counts == [2, 2, 2, 2, 2, 2]


def test_run_microbenchmarks_returns_expected_keys() -> None:
    cfg = _tiny_cpu_config()
    res = run_microbenchmarks(cfg, device="cpu", repeats=1)
    assert res["micro_repeats"] == 1
    assert "micro_random_function_linear_ms" in res
    assert "micro_node_pipeline_ms" in res
    assert "micro_generate_one_ms" in res


def test_run_microbenchmarks_can_skip_generate_one() -> None:
    cfg = _tiny_cpu_config()
    res = run_microbenchmarks(cfg, device="cpu", repeats=1, include_generate_one=False)
    assert res["micro_repeats"] == 1
    assert "micro_random_function_linear_ms" in res
    assert "micro_node_pipeline_ms" in res
    assert res["micro_generate_one_ms"] is None


def test_run_microbenchmarks_synchronizes_generate_one_timing_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    sync_calls: list[str | None] = []

    monkeypatch.setattr(
        micro_mod,
        "_synchronize_accelerator",
        lambda device: sync_calls.append(device),
    )
    monkeypatch.setattr(micro_mod, "generate_one", lambda *_args, **_kwargs: object())

    res = micro_mod.run_microbenchmarks(cfg, device="cuda", repeats=2)

    assert res["micro_generate_one_ms"] is not None
    assert sync_calls == ["cuda", "cuda"]


def test_collect_latency_synchronizes_accelerator_per_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    cfg.runtime.device = "cuda"
    sync_calls: list[str | None] = []

    monkeypatch.setattr(
        runtime_support_mod,
        "_synchronize_accelerator",
        lambda device: sync_calls.append(device),
    )
    monkeypatch.setattr(runtime_support_mod, "generate_one", lambda *_args, **_kwargs: object())

    result = runtime_support_mod._collect_latency(cfg, device="cuda", num_samples=2)

    assert result["latency_p95_ms"] >= 0.0
    assert sync_calls == ["cuda", "cuda", "cuda", "cuda"]


def test_synchronize_accelerator_handles_cuda_and_mps_but_skips_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(runtime_support_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        runtime_support_mod.torch.cuda,
        "synchronize",
        lambda: calls.append("cuda"),
    )
    monkeypatch.setattr(
        runtime_support_mod.torch.backends.mps,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        runtime_support_mod.torch,
        "mps",
        SimpleNamespace(synchronize=lambda: calls.append("mps")),
        raising=False,
    )

    runtime_support_mod._synchronize_accelerator("cpu")
    runtime_support_mod._synchronize_accelerator("cuda")
    runtime_support_mod._synchronize_accelerator("mps")

    assert calls == ["cuda", "mps"]


def test_collect_reproducibility_uses_streaming_generation(
    monkeypatch,
) -> None:
    calls: list[tuple[int, int, str | None]] = []

    def _bundle(value: int) -> DatasetBundle:
        x_train = np.full((2, 2), float(value), dtype=np.float32)
        y_train = np.array([0, 1], dtype=np.int64)
        x_test = np.full((1, 2), float(value), dtype=np.float32)
        y_test = np.array([1], dtype=np.int64)
        return DatasetBundle(
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            feature_types=["num", "num"],
            metadata={"seed": value, "attempt_used": 0},
        )

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        calls.append((num_datasets, int(seed or 0), device))
        for i in range(num_datasets):
            yield _bundle(int(seed or 0) + i)

    monkeypatch.setattr(
        "dagzoo.bench.suite.generate_batch_iter",
        _stub_generate_batch_iter,
    )

    cfg = _tiny_cpu_config()
    out = suite_mod._collect_reproducibility(cfg, device="cpu", num_datasets=3)
    assert out["reproducibility_datasets"] == 3
    assert out["reproducibility_match"] is True
    assert out["reproducibility_workload_match"] is True
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0][1] == KeyedRng(cfg.seed).child_seed("bench", "suite", "reproducibility")


def test_run_benchmark_suite_sanitizes_preset_key_for_diagnostics_paths(tmp_path) -> None:
    cfg = _tiny_cpu_config()
    spec = PresetRunSpec(key="../../escape", config=cfg, device="cpu")
    diagnostics_root = tmp_path / "diag_root"

    summary = run_benchmark_suite(
        [spec],
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=True,
        diagnostics_root_dir=diagnostics_root,
        fail_on_regression=False,
        hardware_policy="none",
    )

    result = summary["preset_results"][0]
    artifacts = result["diagnostics_artifacts"]
    assert isinstance(artifacts, dict)

    json_path = Path(artifacts["json"]).resolve()
    md_path = Path(artifacts["markdown"]).resolve()
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.is_relative_to(diagnostics_root.resolve())
    assert md_path.is_relative_to(diagnostics_root.resolve())
    assert not (tmp_path / "escape" / "coverage_summary.json").exists()


def test_run_benchmark_suite_uses_unique_diagnostics_dirs_for_duplicate_preset_keys(
    tmp_path,
) -> None:
    cfg_a = _tiny_cpu_config()
    cfg_b = _tiny_cpu_config()
    specs = [
        PresetRunSpec(key="cpu_test", config=cfg_a, device="cpu"),
        PresetRunSpec(key="cpu_test", config=cfg_b, device="cpu"),
    ]
    diagnostics_root = tmp_path / "diag_root"

    summary = run_benchmark_suite(
        specs,
        suite="smoke",
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
        baseline_payload=None,
        num_datasets_override=2,
        warmup_override=0,
        collect_memory=False,
        collect_reproducibility=False,
        collect_diagnostics=True,
        diagnostics_root_dir=diagnostics_root,
        fail_on_regression=False,
        hardware_policy="none",
    )

    results = summary["preset_results"]
    assert len(results) == 2
    art_0 = results[0]["diagnostics_artifacts"]
    art_1 = results[1]["diagnostics_artifacts"]
    assert isinstance(art_0, dict)
    assert isinstance(art_1, dict)
    assert art_0["json"] != art_1["json"]
    assert art_0["markdown"] != art_1["markdown"]
    assert Path(art_0["json"]).exists()
    assert Path(art_1["json"]).exists()
    assert Path(art_0["markdown"]).exists()
    assert Path(art_1["markdown"]).exists()
