import json
from pathlib import Path

import yaml

from dagzoo.bench.gpu_telemetry import GpuTelemetrySample
from dagzoo.bench.h100_validation import (
    ValidationPhase,
    _build_saturation_summary,
    _build_validation_phases,
    _run_validation_phase,
    _write_saturation_config,
    run_h100_validation,
)


def test_build_validation_phases_orders_primary_and_feature_runs(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr("dagzoo.bench.h100_validation._repo_root", lambda: repo_root)

    phases = _build_validation_phases(tmp_path, python_executable="/tmp/python")

    assert [phase.name for phase in phases[:6]] == [
        "cuda_h100_smoke",
        "cuda_h100_standard",
        "cuda_h100_large_shape",
        "cuda_h100_saturation_160000000",
        "cuda_h100_saturation_240000000",
        "cuda_h100_saturation_256000000",
    ]
    assert phases[1].require_telemetry is True
    assert phases[3].target_cells == 160_000_000
    assert "--hardware-policy" in phases[3].command
    assert phases[3].command[phases[3].command.index("--hardware-policy") + 1] == "none"
    assert (
        tmp_path / "generated_configs" / "benchmark_cuda_h100_saturation_160000000.yaml"
    ).exists()


def test_write_saturation_config_sets_target_cells_and_preserves_base_fields(tmp_path) -> None:
    base_config_path = tmp_path / "benchmark_cuda_h100_saturation.yaml"
    base_config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  n_train: 4096",
                "runtime:",
                "  device: cuda",
                "benchmark:",
                "  num_datasets: 1500",
                "",
            ]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "generated" / "benchmark_cuda_h100_saturation_240000000.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written_path = _write_saturation_config(
        base_config_path=base_config_path,
        out_path=out_path,
        target_cells=240_000_000,
    )

    assert written_path == out_path
    assert out_path.exists()
    payload = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert payload["runtime"]["fixed_layout_target_cells"] == 240_000_000
    assert payload["runtime"]["device"] == "cuda"
    assert payload["dataset"]["n_train"] == 4096
    assert payload["benchmark"]["num_datasets"] == 1500


def test_build_saturation_summary_prefers_smaller_target_on_tie() -> None:
    summary = _build_saturation_summary(
        [
            {
                "name": "cuda_h100_saturation_160000000",
                "group": "saturation",
                "target_cells": 160_000_000,
                "datasets_per_minute": 1000.0,
            },
            {
                "name": "cuda_h100_saturation_240000000",
                "group": "saturation",
                "target_cells": 240_000_000,
                "datasets_per_minute": 1000.0,
            },
        ]
    )

    assert summary["recommended_fixed_layout_target_cells"] == 160_000_000


def test_run_validation_phase_writes_required_artifacts_and_telemetry(
    tmp_path,
    monkeypatch,
) -> None:
    class _FakeSampler:
        def __init__(self, *, interval_seconds: float) -> None:
            _ = interval_seconds
            self.samples = [
                GpuTelemetrySample(
                    timestamp_utc="2026-03-15T00:00:00+00:00",
                    gpu_index=0,
                    name="NVIDIA H100 NVL",
                    utilization_gpu_pct=80.0,
                    utilization_memory_pct=25.0,
                    memory_used_mb=2048.0,
                    memory_total_mb=95830.0,
                )
            ]
            self.errors: list[str] = []

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    def _stub_run(command, *, cwd, check: bool) -> object:
        _ = command
        _ = cwd
        _ = check
        summary_path = tmp_path / "phase" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "preset_results": [
                        {
                            "datasets_per_minute": 123.0,
                            "latency_p95_ms": 4.0,
                        }
                    ],
                    "regression": {"status": "pass", "issues": []},
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "phase" / "summary.md").write_text("# summary\n", encoding="utf-8")
        return type("CompletedProcess", (), {"returncode": 0})()

    monkeypatch.setattr("dagzoo.bench.h100_validation.NvidiaSmiSampler", _FakeSampler)
    monkeypatch.setattr("dagzoo.bench.h100_validation.subprocess.run", _stub_run)

    result = _run_validation_phase(
        ValidationPhase(
            name="cuda_h100_standard",
            command=["python", "-m", "dagzoo", "benchmark"],
            out_dir=tmp_path / "phase",
            require_telemetry=True,
        ),
        repo_root=tmp_path,
        telemetry_interval_seconds=1.0,
    )

    assert result["validation_status"] == "pass"
    assert result["datasets_per_minute"] == 123.0
    assert result["artifacts"]["summary_json"] is not None
    assert result["artifacts"]["gpu_telemetry_csv"] is not None
    assert result["artifacts"]["gpu_telemetry_summary_json"] is not None
    assert (tmp_path / "phase" / "gpu_telemetry.csv").exists()
    assert (tmp_path / "phase" / "gpu_telemetry_summary.json").exists()


def test_run_h100_validation_marks_missing_primary_telemetry_incomplete(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.bench.h100_validation._torch_cuda_report",
        lambda: {
            "torch_version": "2.0",
            "cuda_available": True,
            "cuda_count": 1,
            "device_names": ["NVIDIA H100 NVL"],
        },
    )
    monkeypatch.setattr(
        "dagzoo.bench.h100_validation._hardware_report",
        lambda: {
            "backend": "cuda",
            "requested_device": "cuda",
            "device_name": "NVIDIA H100 NVL",
            "total_memory_gb": 94.0,
            "peak_flops": 835e12,
            "tier": "cuda_h100",
        },
    )
    monkeypatch.setattr(
        "dagzoo.bench.h100_validation._build_validation_phases",
        lambda _out_root, *, python_executable: [
            ValidationPhase(
                name="cuda_h100_standard",
                command=[python_executable, "-m", "dagzoo", "benchmark"],
                out_dir=tmp_path / "cuda_h100_standard",
                require_telemetry=True,
            )
        ],
    )
    monkeypatch.setattr(
        "dagzoo.bench.h100_validation._run_validation_phase",
        lambda phase, *, repo_root, telemetry_interval_seconds: {
            "name": phase.name,
            "group": phase.group,
            "target_cells": phase.target_cells,
            "exit_code": 0,
            "validation_status": "incomplete",
            "datasets_per_minute": 100.0,
            "artifacts": {"out_dir": str(phase.out_dir)},
        },
    )

    manifest = run_h100_validation(out_root=tmp_path, telemetry_interval_seconds=1.0)

    assert manifest["overall_status"] == "incomplete"
    assert (tmp_path / "validation_manifest.json").exists()
