"""H100 benchmark validation runner with host-level GPU telemetry."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from dagzoo.hardware import detect_hardware

from .gpu_telemetry import NvidiaSmiSampler, summarize_gpu_telemetry, write_gpu_telemetry_csv

SATURATION_TARGET_CELLS = (160_000_000, 240_000_000, 256_000_000)
FEATURE_RUNS: tuple[tuple[str, str], ...] = (
    ("filter_smoke", "configs/preset_filter_benchmark_smoke.yaml"),
    ("missingness_mar_smoke", "configs/preset_missingness_mar.yaml"),
    ("shift_smoke", "configs/preset_shift_benchmark_smoke.yaml"),
    ("noise_smoke", "configs/preset_noise_benchmark_smoke.yaml"),
    ("many_class_smoke", "configs/preset_many_class_benchmark_smoke.yaml"),
    ("mechanism_baseline_smoke", "configs/preset_mechanism_baseline_benchmark_smoke.yaml"),
    ("mechanism_gp_smoke", "configs/preset_mechanism_gp_benchmark_smoke.yaml"),
    ("mechanism_piecewise_smoke", "configs/preset_mechanism_piecewise_benchmark_smoke.yaml"),
)


@dataclass(slots=True)
class ValidationPhase:
    """One benchmark validation phase."""

    name: str
    command: list[str]
    out_dir: Path
    require_telemetry: bool = False
    group: str = "benchmark"
    target_cells: int | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(payload: dict[str, Any], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out_path


def _load_json(path: str | Path) -> dict[str, Any] | None:
    json_path = Path(path)
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def _torch_cuda_report() -> dict[str, Any]:
    return {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_count": int(torch.cuda.device_count()),
        "device_names": [
            str(torch.cuda.get_device_name(i)) for i in range(int(torch.cuda.device_count()))
        ]
        if torch.cuda.is_available()
        else [],
    }


def _hardware_report() -> dict[str, Any]:
    hardware = detect_hardware("cuda")
    return {
        "backend": str(hardware.backend),
        "requested_device": str(hardware.requested_device),
        "device_name": str(hardware.device_name),
        "total_memory_gb": hardware.total_memory_gb,
        "peak_flops": hardware.peak_flops,
        "tier": str(hardware.tier),
    }


def _build_benchmark_command(
    python_executable: str,
    *,
    extra_args: list[str],
) -> list[str]:
    return [python_executable, "-m", "dagzoo", "benchmark", *extra_args]


def _write_saturation_config(
    *,
    base_config_path: Path,
    out_path: Path,
    target_cells: int,
) -> Path:
    with base_config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected config payload: {base_config_path}")
    runtime_payload = payload.setdefault("runtime", {})
    if not isinstance(runtime_payload, dict):
        raise ValueError(f"runtime must be a mapping in {base_config_path}")
    runtime_payload["fixed_layout_target_cells"] = int(target_cells)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return out_path


def _build_validation_phases(out_root: Path, *, python_executable: str) -> list[ValidationPhase]:
    config_root = out_root / "generated_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    repo_root = _repo_root()
    baseline_path = out_root / "baselines" / "cuda_h100_standard.json"
    saturation_base = repo_root / "configs" / "benchmark_cuda_h100_saturation.yaml"

    phases: list[ValidationPhase] = [
        ValidationPhase(
            name="cuda_h100_smoke",
            out_dir=out_root / "cuda_h100_smoke",
            command=_build_benchmark_command(
                python_executable,
                extra_args=[
                    "--preset",
                    "cuda_h100",
                    "--suite",
                    "smoke",
                    "--hardware-policy",
                    "cuda_tiered_v1",
                    "--out-dir",
                    str(out_root / "cuda_h100_smoke"),
                ],
            ),
        ),
        ValidationPhase(
            name="cuda_h100_standard",
            out_dir=out_root / "cuda_h100_standard",
            require_telemetry=True,
            command=_build_benchmark_command(
                python_executable,
                extra_args=[
                    "--preset",
                    "cuda_h100",
                    "--suite",
                    "standard",
                    "--hardware-policy",
                    "cuda_tiered_v1",
                    "--out-dir",
                    str(out_root / "cuda_h100_standard"),
                    "--save-baseline",
                    str(baseline_path),
                ],
            ),
        ),
        ValidationPhase(
            name="cuda_h100_large_shape",
            out_dir=out_root / "cuda_h100_large_shape",
            require_telemetry=True,
            command=_build_benchmark_command(
                python_executable,
                extra_args=[
                    "--config",
                    str(repo_root / "configs" / "benchmark_cuda_h100_large_shape.yaml"),
                    "--preset",
                    "custom",
                    "--suite",
                    "standard",
                    "--hardware-policy",
                    "cuda_tiered_v1",
                    "--out-dir",
                    str(out_root / "cuda_h100_large_shape"),
                ],
            ),
        ),
    ]

    for target_cells in SATURATION_TARGET_CELLS:
        config_path = _write_saturation_config(
            base_config_path=saturation_base,
            out_path=config_root / f"benchmark_cuda_h100_saturation_{target_cells}.yaml",
            target_cells=target_cells,
        )
        phase_name = f"cuda_h100_saturation_{target_cells}"
        out_dir = out_root / "cuda_h100_saturation" / str(target_cells)
        phases.append(
            ValidationPhase(
                name=phase_name,
                out_dir=out_dir,
                require_telemetry=True,
                group="saturation",
                target_cells=int(target_cells),
                command=_build_benchmark_command(
                    python_executable,
                    extra_args=[
                        "--config",
                        str(config_path),
                        "--preset",
                        "custom",
                        "--suite",
                        "standard",
                        "--hardware-policy",
                        "cuda_tiered_v1",
                        "--out-dir",
                        str(out_dir),
                    ],
                ),
            )
        )

    for run_name, feature_config_path in FEATURE_RUNS:
        phases.append(
            ValidationPhase(
                name=run_name,
                out_dir=out_root / run_name,
                command=_build_benchmark_command(
                    python_executable,
                    extra_args=[
                        "--config",
                        str(repo_root / feature_config_path),
                        "--preset",
                        "custom",
                        "--device",
                        "cuda",
                        "--suite",
                        "smoke",
                        "--hardware-policy",
                        "none",
                        "--no-memory",
                        "--out-dir",
                        str(out_root / run_name),
                    ],
                ),
            )
        )
    return phases


def _phase_metrics_from_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {
            "regression_status": None,
            "datasets_per_minute": None,
            "latency_p95_ms": None,
        }
    preset_results = summary.get("preset_results")
    result = preset_results[0] if isinstance(preset_results, list) and preset_results else {}
    regression = summary.get("regression", {})
    datasets_per_minute = None
    latency_p95_ms = None
    if isinstance(result, dict):
        raw_datasets_per_minute = result.get("datasets_per_minute")
        raw_latency_p95_ms = result.get("latency_p95_ms")
        if isinstance(raw_datasets_per_minute, (int, float)):
            datasets_per_minute = float(raw_datasets_per_minute)
        if isinstance(raw_latency_p95_ms, (int, float)):
            latency_p95_ms = float(raw_latency_p95_ms)
    return {
        "regression_status": (
            str(regression.get("status")) if isinstance(regression, dict) else None
        ),
        "datasets_per_minute": datasets_per_minute,
        "latency_p95_ms": latency_p95_ms,
    }


def _phase_validation_status(
    *,
    exit_code: int,
    summary: dict[str, Any] | None,
    require_telemetry: bool,
    telemetry_summary: dict[str, Any] | None,
) -> str:
    if int(exit_code) != 0:
        return "fail"
    if summary is None:
        return "incomplete"
    if require_telemetry:
        telemetry_available = bool(
            isinstance(telemetry_summary, dict) and telemetry_summary.get("telemetry_available")
        )
        if not telemetry_available:
            return "incomplete"
    return "pass"


def _run_validation_phase(
    phase: ValidationPhase,
    *,
    repo_root: Path,
    telemetry_interval_seconds: float,
) -> dict[str, Any]:
    print(f"Running validation phase: {phase.name}")
    phase.out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_csv_path = phase.out_dir / "gpu_telemetry.csv"
    telemetry_summary_path = phase.out_dir / "gpu_telemetry_summary.json"
    summary_path = phase.out_dir / "summary.json"

    sampler: NvidiaSmiSampler | None = None
    if phase.require_telemetry:
        sampler = NvidiaSmiSampler(interval_seconds=telemetry_interval_seconds)
        sampler.start()

    try:
        completed = subprocess.run(phase.command, cwd=repo_root, check=False)
        exit_code = int(completed.returncode)
    finally:
        if sampler is not None:
            sampler.stop()

    summary = _load_json(summary_path)
    telemetry_summary: dict[str, Any] | None = None
    if phase.require_telemetry:
        samples = list(sampler.samples if sampler is not None else [])
        write_gpu_telemetry_csv(samples, telemetry_csv_path)
        telemetry_summary = summarize_gpu_telemetry(samples)
        telemetry_summary["errors"] = list(sampler.errors if sampler is not None else [])
        _write_json(telemetry_summary, telemetry_summary_path)

    metrics = _phase_metrics_from_summary(summary)
    validation_status = _phase_validation_status(
        exit_code=exit_code,
        summary=summary,
        require_telemetry=phase.require_telemetry,
        telemetry_summary=telemetry_summary,
    )
    return {
        "name": phase.name,
        "group": phase.group,
        "target_cells": phase.target_cells,
        "command": [str(part) for part in phase.command],
        "exit_code": int(exit_code),
        "validation_status": validation_status,
        "require_telemetry": bool(phase.require_telemetry),
        "artifacts": {
            "out_dir": str(phase.out_dir.resolve()),
            "summary_json": str(summary_path.resolve()) if summary_path.exists() else None,
            "summary_md": str((phase.out_dir / "summary.md").resolve())
            if (phase.out_dir / "summary.md").exists()
            else None,
            "gpu_telemetry_csv": str(telemetry_csv_path.resolve())
            if phase.require_telemetry
            else None,
            "gpu_telemetry_summary_json": str(telemetry_summary_path.resolve())
            if phase.require_telemetry
            else None,
        },
        **metrics,
    }


def _build_saturation_summary(phase_results: list[dict[str, Any]]) -> dict[str, Any]:
    saturation_results = [result for result in phase_results if result.get("group") == "saturation"]
    ranked_candidates = [
        result
        for result in saturation_results
        if isinstance(result.get("datasets_per_minute"), (int, float))
    ]
    if not ranked_candidates:
        return {
            "candidate_count": len(saturation_results),
            "recommended_fixed_layout_target_cells": None,
            "recommendation_source": "datasets_per_minute desc, target_cells asc",
            "candidates": saturation_results,
        }

    best = max(
        ranked_candidates,
        key=lambda result: (
            float(result.get("datasets_per_minute", 0.0)),
            -int(result.get("target_cells", 0) or 0),
        ),
    )
    return {
        "candidate_count": len(saturation_results),
        "recommended_fixed_layout_target_cells": int(best["target_cells"]),
        "recommendation_source": "datasets_per_minute desc, target_cells asc",
        "candidates": saturation_results,
    }


def run_h100_validation(
    *,
    out_root: str | Path,
    telemetry_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    """Run the bounded H100 validation suite and return the root manifest."""

    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = _repo_root()
    torch_report = _torch_cuda_report()
    hardware_report = _hardware_report()

    manifest: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "out_root": str(out_dir.resolve()),
        "torch_cuda": torch_report,
        "hardware": hardware_report,
    }

    if not torch_report["cuda_available"]:
        manifest["overall_status"] = "fail"
        manifest["phases"] = []
        _write_json(manifest, out_dir / "validation_manifest.json")
        return manifest
    if hardware_report.get("backend") != "cuda":
        manifest["overall_status"] = "fail"
        manifest["phases"] = []
        _write_json(manifest, out_dir / "validation_manifest.json")
        return manifest

    phases = _build_validation_phases(out_dir, python_executable=sys.executable)
    phase_results = [
        _run_validation_phase(
            phase,
            repo_root=repo_root,
            telemetry_interval_seconds=telemetry_interval_seconds,
        )
        for phase in phases
    ]
    manifest["phases"] = phase_results
    manifest["saturation"] = _build_saturation_summary(phase_results)

    overall_status = "pass"
    if any(result.get("validation_status") == "fail" for result in phase_results):
        overall_status = "fail"
    elif any(result.get("validation_status") == "incomplete" for result in phase_results):
        overall_status = "incomplete"
    manifest["overall_status"] = overall_status
    _write_json(manifest, out_dir / "validation_manifest.json")
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dagzoo.bench.h100_validation",
        description="Run the bounded H100 benchmark validation workflow.",
    )
    parser.add_argument(
        "--out-root",
        default=str(Path("benchmarks") / "results" / "gpu_h100_validation"),
        help="Artifact root for benchmark summaries, telemetry, and the validation manifest.",
    )
    parser.add_argument(
        "--telemetry-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval for host-level nvidia-smi sampling.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the H100 validation runner."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    manifest = run_h100_validation(
        out_root=args.out_root,
        telemetry_interval_seconds=float(args.telemetry_interval_seconds),
    )
    print(f"Validation manifest: {Path(args.out_root).resolve() / 'validation_manifest.json'}")
    return 0 if manifest.get("overall_status") == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
