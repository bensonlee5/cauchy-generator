import json
from pathlib import Path

import yaml
from conftest import load_repo_config, write_config

from dagzoo.bench.constants import SMOKE_N_TEST_CAP, SMOKE_N_TRAIN_CAP
from dagzoo.cli.commands.benchmark import _print_preset_result_line
from dagzoo.cli.entrypoint import main
from dagzoo.cli.parsing import (
    DEVICE_CHOICES,
    HARDWARE_POLICY_CHOICES,
    MISSINGNESS_MECHANISM_CLI_CHOICES,
)


def test_cli_parser_choice_constants_are_stable() -> None:
    assert DEVICE_CHOICES == ("auto", "cpu", "cuda", "mps")
    assert "none" in HARDWARE_POLICY_CHOICES
    assert tuple(MISSINGNESS_MECHANISM_CLI_CHOICES) == ("none", "mcar", "mar", "mnar")


def test_benchmark_cli_writes_json(tmp_path) -> None:
    out = tmp_path / "summary.json"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/default.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["suite"] == "smoke"
    assert len(payload["preset_results"]) == 1
    profile = payload["preset_results"][0]
    assert profile["generation_mode"] == "fixed_batched"
    assert float(profile["generation_datasets_per_minute"]) >= 0.0
    assert float(profile["write_datasets_per_minute"]) >= 0.0
    assert profile["filter_datasets_per_minute"] is None
    assert profile["filter_accepted_datasets_per_minute"] is None
    assert "accepted_datasets_measured" not in profile
    assert profile["filter_accepted_datasets_measured"] == 0
    assert profile["filter_rejected_datasets_measured"] == 0
    assert profile["filter_acceptance_rate_dataset_level"] is None
    assert profile["filter_rejection_rate_dataset_level"] is None
    assert profile["filter_rejection_rate_attempt_level"] is None
    assert profile["filter_retry_dataset_rate"] is None
    lineage_guardrails = profile["lineage_guardrails"]
    assert isinstance(lineage_guardrails, dict)
    assert isinstance(lineage_guardrails["enabled"], bool)
    if lineage_guardrails["enabled"]:
        assert lineage_guardrails["status"] in {"pass", "warn", "fail"}


def test_benchmark_cli_realizes_dataset_rows_once_per_preset(tmp_path) -> None:
    cfg = load_repo_config()
    cfg.dataset.rows = "400..60000"  # type: ignore[assignment]
    config_path = write_config(tmp_path, cfg, "rows_config.yaml")
    out_dir = tmp_path / "rows_benchmark"

    code = main(
        [
            "benchmark",
            "--config",
            str(config_path),
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "1",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    result = payload["preset_results"][0]
    effective_config = result["effective_config"]
    assert int(result["dataset_rows_total"]) <= int(SMOKE_N_TRAIN_CAP + SMOKE_N_TEST_CAP)
    assert effective_config["dataset"]["rows"] is None
    trace_files = sorted((out_dir / "effective_configs").glob("*_trace.yaml"))
    assert trace_files
    trace_payload = yaml.safe_load(trace_files[0].read_text(encoding="utf-8"))
    assert any(
        isinstance(item, dict) and item.get("source") == "benchmark.smoke_rows_cap"
        for item in trace_payload
    )


def test_benchmark_cli_writes_effective_config_trace_artifacts(tmp_path) -> None:
    out_dir = tmp_path / "bench_results"
    code = main(
        [
            "benchmark",
            "--preset",
            "cpu",
            "--suite",
            "smoke",
            "--num-datasets",
            "1",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    trace_files = sorted((out_dir / "effective_configs").glob("*_trace.yaml"))
    assert trace_files
    trace_payload = yaml.safe_load(trace_files[0].read_text(encoding="utf-8"))
    assert isinstance(trace_payload, list)
    assert any(
        isinstance(item, dict) and item.get("source") == "benchmark.suite_smoke_caps"
        for item in trace_payload
    )


def test_benchmark_cli_builtin_cpu_reports_fixed_batched_generation_mode(tmp_path) -> None:
    out = tmp_path / "summary.json"
    code = main(
        [
            "benchmark",
            "--preset",
            "cpu",
            "--suite",
            "smoke",
            "--num-datasets",
            "1",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    preset_results = payload["preset_results"]
    assert [result["preset_key"] for result in preset_results] == [
        "cpu_rows1024",
        "cpu_rows4096",
        "cpu_rows8192",
    ]
    assert [result["dataset_rows_total"] for result in preset_results] == [1024, 4096, 8192]
    assert all(result["generation_mode"] == "fixed_batched" for result in preset_results)


def test_benchmark_cli_filter_smoke_config_reports_accepted_corpus_throughput(tmp_path) -> None:
    out_dir = tmp_path / "filter_smoke_results"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/preset_filter_benchmark_smoke.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    filter_cfg = profile["effective_config"]["filter"]
    assert profile["filter_datasets_per_minute"] is not None
    assert float(profile["filter_accepted_datasets_per_minute"]) > 0.0
    assert int(profile["filter_accepted_datasets_measured"]) > 0
    assert int(profile["filter_rejected_datasets_measured"]) > 0
    assert 0.0 < float(profile["filter_acceptance_rate_dataset_level"]) < 1.0
    assert 0.0 < float(profile["filter_rejection_rate_dataset_level"]) < 1.0
    assert filter_cfg["use_lineage_veto"] is False
    assert filter_cfg["stump_skill_threshold"] == 0.04


def test_benchmark_cli_fail_on_regression(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_payload = {
        "version": 1,
        "suite": "standard",
        "metrics": ["datasets_per_minute"],
        "presets": {
            "medium_cuda": {"datasets_per_minute": 1.0e9},
        },
    }
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    code = main(
        [
            "benchmark",
            "--config",
            "configs/default.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--baseline",
            str(baseline_path),
            "--warn-threshold-pct",
            "1",
            "--fail-threshold-pct",
            "2",
            "--fail-on-regression",
        ]
    )
    assert code == 1


def test_benchmark_cli_fail_on_regression_for_filter_accepted_throughput(
    tmp_path, monkeypatch, capsys
) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_payload = {
        "version": 1,
        "suite": "smoke",
        "metrics": ["filter_accepted_datasets_per_minute"],
        "presets": {
            "custom": {"filter_accepted_datasets_per_minute": 100.0},
        },
    }
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    summary = {
        "suite": "smoke",
        "preset_results": [
            {
                "preset_key": "custom",
                "filter_accepted_datasets_per_minute": 70.0,
                "datasets_per_minute": 1000.0,
                "latency_p95_ms": 1.0,
            }
        ],
        "regression": {
            "status": "fail",
            "issues": [
                {
                    "preset": "custom",
                    "metric": "filter_accepted_datasets_per_minute",
                    "current": 70.0,
                    "baseline": 100.0,
                    "degradation_pct": 30.0,
                    "severity": "fail",
                }
            ],
            "hard_fail": True,
        },
    }

    calls: list[dict[str, object]] = []

    def _fake_run_benchmark_suite(*args, **kwargs):
        calls.append(kwargs)
        return summary

    monkeypatch.setattr(
        "dagzoo.cli.commands.benchmark.run_benchmark_suite",
        _fake_run_benchmark_suite,
    )

    code = main(
        [
            "benchmark",
            "--config",
            "configs/default.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--baseline",
            str(baseline_path),
            "--warn-threshold-pct",
            "1",
            "--fail-threshold-pct",
            "2",
            "--fail-on-regression",
        ]
    )

    assert code == 1
    assert calls == [
        {
            "suite": "smoke",
            "warn_threshold_pct": 1.0,
            "fail_threshold_pct": 2.0,
            "baseline_payload": baseline_payload,
            "num_datasets_override": 2,
            "warmup_override": 0,
            "collect_memory": False,
            "collect_reproducibility": False,
            "collect_diagnostics": False,
            "diagnostics_root_dir": None,
            "fail_on_regression": True,
            "hardware_policy": "none",
        }
    ]
    captured = capsys.readouterr()
    assert "Regression status=fail issues=1" in captured.out
    assert "filter_accepted/min=70.00" in captured.out


def test_benchmark_cli_diagnostics_emits_artifacts(tmp_path) -> None:
    out_dir = tmp_path / "bench_results"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/default.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--diagnostics",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    assert profile["diagnostics_enabled"] is True

    artifacts = profile["diagnostics_artifacts"]
    assert isinstance(artifacts, dict)
    assert isinstance(artifacts["json"], str)
    assert isinstance(artifacts["markdown"], str)
    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["markdown"]).exists()


def test_benchmark_cli_json_out_diagnostics_does_not_emit_default_summary_artifacts(
    tmp_path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "default.yaml"
    json_out = tmp_path / "summary.json"
    diagnostics_out = tmp_path / "diag_out"
    monkeypatch.chdir(tmp_path)

    code = main(
        [
            "benchmark",
            "--config",
            str(config_path),
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--diagnostics",
            "--json-out",
            str(json_out),
            "--diagnostics-out-dir",
            str(diagnostics_out),
        ]
    )
    assert code == 0
    assert json_out.exists()
    assert not (tmp_path / "benchmarks").exists()


def test_benchmark_cli_diagnostics_pointers_resolve_when_roots_differ(tmp_path) -> None:
    out_dir = tmp_path / "bench_results"
    diagnostics_out = tmp_path / "diag_out"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/default.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--diagnostics",
            "--out-dir",
            str(out_dir),
            "--diagnostics-out-dir",
            str(diagnostics_out),
        ]
    )
    assert code == 0

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    artifacts = profile["diagnostics_artifacts"]
    assert isinstance(artifacts, dict)
    json_path = Path(artifacts["json"])
    md_path = Path(artifacts["markdown"])
    assert json_path.is_absolute()
    assert md_path.is_absolute()
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.resolve().is_relative_to(diagnostics_out.resolve())
    assert md_path.resolve().is_relative_to(diagnostics_out.resolve())


def test_benchmark_cli_missingness_guardrails_are_emitted(tmp_path) -> None:
    out = tmp_path / "summary_missingness.json"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/preset_missingness_mcar.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    guardrails = profile["missingness_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] in {"pass", "warn", "fail"}
    lineage_guardrails = profile["lineage_guardrails"]
    assert isinstance(lineage_guardrails, dict)
    assert isinstance(lineage_guardrails["enabled"], bool)
    if lineage_guardrails["enabled"]:
        assert lineage_guardrails["status"] in {"pass", "warn", "fail"}


def test_benchmark_cli_shift_guardrails_are_emitted(tmp_path) -> None:
    out = tmp_path / "summary_shift.json"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/preset_shift_benchmark_smoke.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    guardrails = profile["shift_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] in {"pass", "warn", "fail"}
    assert "directional_checks" in guardrails
    assert "runtime_gating_enabled" in guardrails


def test_benchmark_cli_noise_guardrails_are_emitted(tmp_path) -> None:
    out = tmp_path / "summary_noise.json"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/preset_noise_benchmark_smoke.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    guardrails = profile["noise_guardrails"]
    assert guardrails["enabled"] is True
    assert guardrails["status"] in {"pass", "warn", "fail"}
    assert "metadata_coverage_rate" in guardrails
    assert "runtime_gating_enabled" in guardrails


def test_benchmark_cli_many_class_smoke_preset_emits_runtime_metrics(tmp_path) -> None:
    out = tmp_path / "summary_many_class_smoke.json"
    code = main(
        [
            "benchmark",
            "--config",
            "configs/preset_many_class_benchmark_smoke.yaml",
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    assert float(profile["datasets_per_minute"]) > 0.0
    assert float(profile["latency_p95_ms"]) >= 0.0
    regression = payload["regression"]
    assert regression["status"] in {"pass", "warn", "fail"}
    guardrails = profile["lineage_guardrails"]
    assert isinstance(guardrails, dict)
    assert isinstance(guardrails["enabled"], bool)


def test_print_preset_result_line_includes_shift_status(capsys) -> None:
    _print_preset_result_line(
        {
            "preset_key": "shift_smoke",
            "device": "cpu",
            "hardware_backend": "cpu",
            "datasets_per_minute": 123.0,
            "latency_p95_ms": 4.2,
            "shift_guardrails": {"enabled": True, "status": "pass"},
        }
    )
    output = capsys.readouterr().out
    assert "shift=pass" in output


def test_print_preset_result_line_includes_noise_status(capsys) -> None:
    _print_preset_result_line(
        {
            "preset_key": "noise_smoke",
            "device": "cpu",
            "hardware_backend": "cpu",
            "datasets_per_minute": 123.0,
            "latency_p95_ms": 4.2,
            "noise_guardrails": {"enabled": True, "status": "warn"},
        }
    )
    output = capsys.readouterr().out
    assert "noise=warn" in output


def test_print_preset_result_line_includes_stage_and_filter_rejection_metrics(capsys) -> None:
    _print_preset_result_line(
        {
            "preset_key": "throughput_smoke",
            "device": "cpu",
            "hardware_backend": "cpu",
            "datasets_per_minute": 123.0,
            "generation_datasets_per_minute": 124.0,
            "write_datasets_per_minute": 80.0,
            "filter_datasets_per_minute": 60.0,
            "filter_accepted_datasets_per_minute": 45.0,
            "filter_acceptance_rate_dataset_level": 0.75,
            "filter_rejection_rate_dataset_level": 0.25,
            "filter_rejection_rate_attempt_level": 0.125,
            "filter_retry_dataset_rate": 0.25,
            "latency_p95_ms": 4.2,
        }
    )
    output = capsys.readouterr().out
    assert "gen/min=124.00" in output
    assert "write/min=80.00" in output
    assert "filter/min=60.00" in output
    assert "filter_accepted/min=45.00" in output
    assert "filter_reject_attempt_pct=12.50" in output
    assert "filter_accept_dataset_pct=75.00" in output
    assert "filter_reject_dataset_pct=25.00" in output
    assert "filter_retry_dataset_pct=25.00" in output


def test_print_preset_result_line_handles_missing_latency(capsys) -> None:
    _print_preset_result_line(
        {
            "preset_key": "parallel_cpu",
            "device": "cpu",
            "hardware_backend": "cpu",
            "datasets_per_minute": 123.0,
            "generation_datasets_per_minute": 124.0,
            "write_datasets_per_minute": 80.0,
            "latency_p95_ms": None,
        }
    )
    output = capsys.readouterr().out
    assert "latency_p95_ms=-" in output
