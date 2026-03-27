from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import load_repo_config

import dagzoo.bench.suite as suite_mod
from dagzoo.bench.suite import PresetRunSpec, run_benchmark_suite
from dagzoo.hardware import HardwareInfo
from dagzoo.types import DatasetBundle


def _set_attrs(target: object, **attrs: object) -> None:
    for name, value in attrs.items():
        setattr(target, name, value)


def _tiny_cpu_config() -> suite_mod.GeneratorConfig:
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


def _hardware() -> HardwareInfo:
    return HardwareInfo(
        backend="cpu",
        requested_device="cpu",
        device_name="cpu",
        total_memory_gb=None,
        peak_flops=float("inf"),
        tier="cpu",
    )


def _bundle_for_config(config: suite_mod.GeneratorConfig) -> DatasetBundle:
    metadata: dict[str, object] = {
        "dataset_index": 0,
        "run_num_datasets": 1,
        "generation_attempts": {
            "total_attempts": 2,
            "filter_attempts": 0,
            "filter_rejections": 0,
        },
        "graph_edge_density": 0.7,
    }
    if float(config.dataset.missing_rate) > 0.0:
        metadata["missingness"] = {
            "missing_count_overall": 20,
            "target_rate": float(config.dataset.missing_rate),
            "realized_rate_overall": 0.5,
            "mechanism": str(config.dataset.missing_mechanism),
        }
    if bool(config.shift.enabled):
        metadata["shift"] = {
            "enabled": True,
            "mode": str(config.shift.mode),
            "edge_odds_multiplier": 1.5,
            "mechanism_nonlinear_mass": 0.8,
            "noise_variance_multiplier": 1.4,
        }
    noise_family = str(config.noise.family)
    if noise_family:
        metadata["noise_distribution"] = {
            "family_requested": noise_family,
            "family_sampled": noise_family,
            "sampling_strategy": "dataset_level",
            "base_scale": 1.0,
            "student_t_df": 6.0,
            "mixture_weights": None,
        }
    return DatasetBundle(
        X_train=np.zeros((4, 5), dtype=np.float32),
        y_train=np.zeros(4, dtype=np.float32),
        X_test=np.zeros((2, 5), dtype=np.float32),
        y_test=np.zeros(2, dtype=np.float32),
        feature_types=["num"] * 5,
        metadata=metadata,
    )


def _install_common_benchmark_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    filter_stage_measurement: object | None = None,
    mutate_bundle=None,
) -> None:
    monkeypatch.setattr(
        suite_mod,
        "resolve_benchmark_preset_config",
        lambda **kwargs: {
            "config": kwargs["config"],
            "trace_events": [],
            "requested_device": kwargs["preset_device"] or "cpu",
            "hardware": _hardware(),
        },
    )
    monkeypatch.setattr(
        suite_mod,
        "realize_generation_config_for_run",
        lambda config, **_kwargs: (config, int(config.seed), str(config.runtime.device), "cpu"),
    )
    monkeypatch.setattr(
        suite_mod,
        "_build_fixed_layout_evidence",
        lambda *_args, **_kwargs: {
            "fixed_layout_target_cells_effective": 4_000_000,
            "fixed_layout_per_dataset_cells": 160,
            "fixed_layout_realized_batch_size": 2,
            "fixed_layout_chunk_count": 1,
            "fixed_layout_tail_chunk_size": 2,
        },
    )
    monkeypatch.setattr(
        suite_mod,
        "_collect_latency",
        lambda *_args, **_kwargs: {"latency_p95_ms": 4.0},
    )
    monkeypatch.setattr(
        suite_mod,
        "measure_write_stage_metrics",
        lambda *_args, **_kwargs: SimpleNamespace(
            datasets_per_minute=150.0,
            elapsed_seconds=0.2,
            cpu_time_seconds=0.1,
            bytes_written=4096,
            mib_per_second=2.0,
        ),
    )
    monkeypatch.setattr(
        suite_mod,
        "measure_filter_stage_metrics",
        lambda *_args, **_kwargs: filter_stage_measurement,
    )

    def _stub_run_throughput(config, *, on_bundle=None, **_kwargs):
        bundle = _bundle_for_config(config)
        if mutate_bundle is not None:
            bundle = mutate_bundle(config, bundle)
        if on_bundle is not None:
            on_bundle(bundle)
        datasets_per_minute = 100.0
        if float(config.dataset.missing_rate) > 0.0:
            datasets_per_minute = 60.0
        elif bool(config.shift.enabled):
            datasets_per_minute = 70.0
        elif str(config.noise.family) != "gaussian":
            datasets_per_minute = 65.0
        return {
            "preset": config.benchmark.preset_name,
            "num_datasets": 1,
            "warmup_datasets": 0,
            "elapsed_seconds": 1.0,
            "cpu_time_seconds": 0.5,
            "datasets_per_second": datasets_per_minute / 60.0,
            "datasets_per_minute": datasets_per_minute,
            "generation_mode": "fixed_batched",
            "prepare_elapsed_seconds": 0.1,
            "prepare_cpu_time_seconds": 0.05,
            "raw_batch_elapsed_seconds": 0.2,
            "raw_batch_cpu_time_seconds": 0.1,
            "node_apply_elapsed_seconds": 0.05,
            "node_apply_cpu_time_seconds": 0.02,
            "converter_elapsed_seconds": 0.04,
            "converter_cpu_time_seconds": 0.01,
            "feature_materialization_elapsed_seconds": 0.03,
            "feature_materialization_cpu_time_seconds": 0.01,
        }

    monkeypatch.setattr(suite_mod, "run_throughput_benchmark", _stub_run_throughput)


def test_run_benchmark_suite_smoke_emits_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _tiny_cpu_config()
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")
    _install_common_benchmark_stubs(monkeypatch)

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
    assert summary["suite"] == "smoke"
    assert result["generation_mode"] == "fixed_batched"
    assert "lineage_guardrails" not in result
    assert set(result["scenarios"]) == {
        "baseline",
        "throughput",
        "filtering",
        "missingness",
        "shift",
        "noise",
    }
    assert result["scenarios"]["baseline"]["status"] == "pass"
    assert result["scenarios"]["throughput"]["status"] == "pass"
    assert result["scenarios"]["filtering"]["status"] == "off"


def test_run_benchmark_suite_missingness_scenario_emits_control_and_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    _set_attrs(cfg.dataset, missing_rate=0.25, missing_mechanism="mcar")
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")
    _install_common_benchmark_stubs(monkeypatch)

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

    scenario = summary["preset_results"][0]["scenarios"]["missingness"]
    assert scenario["enabled"] is True
    assert scenario["control_metrics"]["datasets_per_minute"] == 100.0
    assert scenario["status"] == "fail"
    metrics = {issue["metric"] for issue in scenario["issues"]}
    assert "missingness_realized_rate_error_pp" in metrics
    assert "missingness_runtime_degradation_pct" in metrics
    assert summary["regression"]["status"] == "fail"


def test_run_benchmark_suite_shift_scenario_emits_directional_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    _set_attrs(cfg.shift, enabled=True, mode="mixed")
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")
    _install_common_benchmark_stubs(monkeypatch)

    summary = run_benchmark_suite(
        [spec],
        suite="standard",
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

    scenario = summary["preset_results"][0]["scenarios"]["shift"]
    assert scenario["enabled"] is True
    assert scenario["control_metrics"]["datasets_per_minute"] == 100.0
    assert set(scenario["metrics"]["directional_checks"]) == {
        "graph_edge_density",
        "mechanism_nonlinear_mass",
        "noise_variance_multiplier",
    }


def test_run_benchmark_suite_noise_scenario_reports_invalid_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    _set_attrs(cfg.noise, family="laplace", base_scale=1.0, student_t_df=6.0, mixture_weights=None)
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")

    def _mutate_bundle(config, bundle):
        _ = config
        bundle.metadata["noise_distribution"] = {
            "family_requested": "laplace",
            "family_sampled": "gaussian",
            "sampling_strategy": "dataset_level",
            "base_scale": 1.0,
            "student_t_df": 6.0,
            "mixture_weights": None,
        }
        return bundle

    _install_common_benchmark_stubs(monkeypatch, mutate_bundle=_mutate_bundle)

    summary = run_benchmark_suite(
        [spec],
        suite="standard",
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

    scenario = summary["preset_results"][0]["scenarios"]["noise"]
    assert scenario["enabled"] is True
    assert scenario["status"] == "fail"
    assert scenario["control_metrics"]["family_requested"] == "gaussian"
    assert {issue["metric"] for issue in scenario["issues"]} >= {"noise_metadata_validity"}


def test_run_benchmark_suite_filtering_scenario_uses_candidate_metrics_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_cpu_config()
    cfg.filter.enabled = True
    spec = PresetRunSpec(key="cpu_test", config=cfg, device="cpu")
    filter_stage_measurement = SimpleNamespace(
        datasets_per_minute=45.0,
        filter_attempts_total=3,
        filter_accepted_datasets=2,
        filter_rejections_total=1,
        filter_rejected_datasets=1,
        elapsed_seconds=0.3,
        cpu_time_seconds=0.1,
    )
    _install_common_benchmark_stubs(
        monkeypatch,
        filter_stage_measurement=filter_stage_measurement,
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

    scenario = summary["preset_results"][0]["scenarios"]["filtering"]
    assert scenario["enabled"] is True
    assert "control_metrics" not in scenario
    assert scenario["metrics"]["datasets_per_minute"] == 45.0
    assert scenario["metrics"]["accepted_datasets_per_minute"] == pytest.approx(30.0)
