from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import load_repo_config, write_config

from dagzoo.bench.constants import SMOKE_N_TEST_CAP, SMOKE_N_TRAIN_CAP
from dagzoo.cli.commands.benchmark import _print_preset_result_line
from dagzoo.cli.effective_config import effective_config_payload_yaml, effective_config_yaml
from dagzoo.cli.entrypoint import main
from dagzoo.cli.parsing import DEVICE_CHOICES, HARDWARE_POLICY_CHOICES
from dagzoo.config import effective_config_payload


def test_cli_parser_choice_constants_are_stable() -> None:
    assert DEVICE_CHOICES == ("auto", "cpu", "cuda", "mps")
    assert "none" in HARDWARE_POLICY_CHOICES


def test_benchmark_cli_writes_json_with_scenarios(tmp_path) -> None:
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
    payload = json.loads(out.read_text(encoding="utf-8"))
    profile = payload["preset_results"][0]
    assert payload["suite"] == "smoke"
    assert profile["generation_mode"] == "fixed_batched"
    assert "lineage_guardrails" not in profile
    assert set(profile["scenarios"]) == {
        "baseline",
        "throughput",
        "filtering",
        "missingness",
        "shift",
        "noise",
    }


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
    config_files = sorted(
        path
        for path in (out_dir / "effective_configs").glob("*.yaml")
        if not path.name.endswith("_trace.yaml")
    )
    assert config_files
    trace_files = sorted((out_dir / "effective_configs").glob("*_trace.yaml"))
    assert trace_files
    trace_payload = yaml.safe_load(trace_files[0].read_text(encoding="utf-8"))
    assert any(
        isinstance(item, dict) and item.get("source") == "benchmark.smoke_rows_cap"
        for item in trace_payload
    )


def test_effective_config_payload_yaml_matches_config_yaml() -> None:
    cfg = load_repo_config()

    assert effective_config_payload_yaml(effective_config_payload(cfg)) == effective_config_yaml(
        cfg
    )


def test_benchmark_cli_prints_effective_config_with_shared_yaml_shape(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "printed_effective_config"

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
            "1",
            "--warmup",
            "0",
            "--hardware-policy",
            "none",
            "--no-memory",
            "--out-dir",
            str(out_dir),
            "--print-effective-config",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "Effective config [" in output
    assert "dataset:" in output
    assert "benchmark:" in output


@pytest.mark.parametrize(
    ("config_path", "scenario_name"),
    [
        ("configs/preset_filter_benchmark_smoke.yaml", "filtering"),
        ("configs/preset_missingness_mar.yaml", "missingness"),
        ("configs/preset_shift_benchmark_smoke.yaml", "shift"),
        ("configs/preset_noise_benchmark_smoke.yaml", "noise"),
    ],
)
def test_benchmark_cli_emits_expected_scenario(
    tmp_path, config_path: str, scenario_name: str
) -> None:
    out_dir = tmp_path / scenario_name
    code = main(
        [
            "benchmark",
            "--config",
            config_path,
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
    scenario = payload["preset_results"][0]["scenarios"][scenario_name]
    assert scenario["enabled"] is True
    assert scenario["status"] in {"pass", "warn", "fail"}


def test_benchmark_cli_diagnostics_summary_omits_steering_payload(tmp_path) -> None:
    cfg = load_repo_config()
    cfg.diagnostics.enabled = True
    cfg.steering.enabled = True
    cfg.steering.preset = "anti_memorization_piecewise_v1"
    config_path = write_config(tmp_path, cfg, "diagnostics_config.yaml")
    out_dir = tmp_path / "bench_results"

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
            "--diagnostics",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    coverage_path = next((out_dir / "diagnostics").glob("*/coverage_summary.json"))
    coverage_summary = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert "steering" not in coverage_summary
    assert "pearson_abs_mean" in coverage_summary["metrics"]


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/preset_stress_classification_slice_benchmark_smoke.yaml",
        "configs/preset_stress_graph_breadth_benchmark_smoke.yaml",
        "configs/preset_stress_compositional_benchmark_smoke.yaml",
    ],
)
def test_stress_benchmark_presets_emit_expected_scenario_shape(
    tmp_path,
    config_path: str,
) -> None:
    out_dir = tmp_path / Path(config_path).stem

    code = main(
        [
            "benchmark",
            "--config",
            config_path,
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
    profile = payload["preset_results"][0]
    assert profile["diagnostics_enabled"] is False
    assert set(profile["scenarios"]) == {
        "baseline",
        "throughput",
        "filtering",
        "missingness",
        "shift",
        "noise",
    }
    assert profile["scenarios"]["throughput"]["enabled"] is True
    assert profile["scenarios"]["filtering"]["status"] == "off"
    assert profile["scenarios"]["missingness"]["status"] == "off"
    assert profile["scenarios"]["shift"]["status"] == "off"
    assert profile["scenarios"]["noise"]["status"] == "off"


def test_print_preset_result_line_uses_scenario_statuses(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_preset_result_line(
        {
            "preset_key": "cpu",
            "device": "cpu",
            "dataset_rows_total": 1024,
            "generation_mode": "fixed_batched",
            "hardware_backend": "cpu",
            "datasets_per_minute": 100.0,
            "generation_datasets_per_minute": 100.0,
            "write_datasets_per_minute": 80.0,
            "filter_datasets_per_minute": None,
            "filter_accepted_datasets_per_minute": None,
            "filter_rejection_rate_attempt_level": None,
            "filter_acceptance_rate_dataset_level": None,
            "filter_rejection_rate_dataset_level": None,
            "filter_retry_dataset_rate": None,
            "latency_p95_ms": 4.0,
            "scenarios": {
                "filtering": {"enabled": True, "status": "pass"},
                "missingness": {"enabled": True, "status": "fail"},
                "shift": {"enabled": False, "status": "off"},
                "noise": {"enabled": True, "status": "warn"},
                "throughput": {"enabled": True, "status": "pass"},
            },
        }
    )

    captured = capsys.readouterr()
    assert "filtering=pass" in captured.out
    assert "missingness=fail" in captured.out
    assert "noise=warn" in captured.out
    assert "throughput=pass" in captured.out
