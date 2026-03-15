"""Host-level GPU telemetry sampling helpers for benchmark validation runs."""

from __future__ import annotations

import csv
import datetime as dt
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GpuTelemetrySample:
    """One parsed ``nvidia-smi`` utilization sample."""

    timestamp_utc: str
    gpu_index: int
    name: str
    utilization_gpu_pct: float
    utilization_memory_pct: float
    memory_used_mb: float
    memory_total_mb: float


def parse_nvidia_smi_csv(text: str, *, timestamp_utc: str) -> list[GpuTelemetrySample]:
    """Parse one ``nvidia-smi`` CSV snapshot into structured samples."""

    samples: list[GpuTelemetrySample] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise ValueError(f"Unexpected nvidia-smi CSV row: {raw_line!r}")
        gpu_index_raw, name, gpu_util_raw, mem_util_raw, mem_used_raw, mem_total_raw = parts
        samples.append(
            GpuTelemetrySample(
                timestamp_utc=str(timestamp_utc),
                gpu_index=int(gpu_index_raw),
                name=str(name),
                utilization_gpu_pct=float(gpu_util_raw),
                utilization_memory_pct=float(mem_util_raw),
                memory_used_mb=float(mem_used_raw),
                memory_total_mb=float(mem_total_raw),
            )
        )
    return samples


def sample_nvidia_smi_once(*, timeout_seconds: float = 10.0) -> list[GpuTelemetrySample]:
    """Collect one instantaneous ``nvidia-smi`` sample across visible GPUs."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=float(timeout_seconds),
    )
    timestamp_utc = dt.datetime.now(dt.UTC).isoformat()
    return parse_nvidia_smi_csv(result.stdout, timestamp_utc=timestamp_utc)


class NvidiaSmiSampler:
    """Background ``nvidia-smi`` sampler for one benchmark phase."""

    def __init__(
        self,
        *,
        interval_seconds: float = 1.0,
        sample_fn: Any = sample_nvidia_smi_once,
    ) -> None:
        self.interval_seconds = max(0.1, float(interval_seconds))
        self._sample_fn = sample_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[GpuTelemetrySample] = []
        self.errors: list[str] = []

    def start(self) -> None:
        """Start sampling until ``stop`` is called."""

        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for the worker to exit."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4.0))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.samples.extend(list(self._sample_fn()))
            except Exception as exc:  # pragma: no cover - exercised via public snapshot
                self.errors.append(str(exc))
            if self._stop_event.wait(self.interval_seconds):
                break


def write_gpu_telemetry_csv(samples: list[GpuTelemetrySample], path: str | Path) -> Path:
    """Write raw GPU telemetry samples to CSV."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp_utc",
                "gpu_index",
                "name",
                "utilization_gpu_pct",
                "utilization_memory_pct",
                "memory_used_mb",
                "memory_total_mb",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.timestamp_utc,
                    sample.gpu_index,
                    sample.name,
                    sample.utilization_gpu_pct,
                    sample.utilization_memory_pct,
                    sample.memory_used_mb,
                    sample.memory_total_mb,
                ]
            )
    return out_path


def summarize_gpu_telemetry(samples: list[GpuTelemetrySample]) -> dict[str, Any]:
    """Summarize raw GPU telemetry samples into aggregate bottleneck evidence."""

    if not samples:
        return {
            "telemetry_available": False,
            "sample_rows": 0,
            "sample_ticks": 0,
            "gpu_count": 0,
            "mean_gpu_utilization_pct": None,
            "max_gpu_utilization_pct": None,
            "mean_memory_utilization_pct": None,
            "max_memory_utilization_pct": None,
            "mean_memory_used_mb": None,
            "max_memory_used_mb": None,
            "per_gpu": {},
        }

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    sample_ticks = len({sample.timestamp_utc for sample in samples})
    by_gpu: dict[int, list[GpuTelemetrySample]] = {}
    for sample in samples:
        by_gpu.setdefault(int(sample.gpu_index), []).append(sample)

    gpu_utils = [float(sample.utilization_gpu_pct) for sample in samples]
    mem_utils = [float(sample.utilization_memory_pct) for sample in samples]
    mem_used = [float(sample.memory_used_mb) for sample in samples]
    per_gpu_summary: dict[str, Any] = {}
    for gpu_index in sorted(by_gpu):
        gpu_samples = by_gpu[gpu_index]
        per_gpu_summary[str(gpu_index)] = {
            "name": str(gpu_samples[0].name),
            "sample_rows": len(gpu_samples),
            "sample_ticks": len({sample.timestamp_utc for sample in gpu_samples}),
            "mean_gpu_utilization_pct": _mean(
                [float(sample.utilization_gpu_pct) for sample in gpu_samples]
            ),
            "max_gpu_utilization_pct": max(
                float(sample.utilization_gpu_pct) for sample in gpu_samples
            ),
            "mean_memory_utilization_pct": _mean(
                [float(sample.utilization_memory_pct) for sample in gpu_samples]
            ),
            "max_memory_utilization_pct": max(
                float(sample.utilization_memory_pct) for sample in gpu_samples
            ),
            "mean_memory_used_mb": _mean([float(sample.memory_used_mb) for sample in gpu_samples]),
            "max_memory_used_mb": max(float(sample.memory_used_mb) for sample in gpu_samples),
            "memory_total_mb": float(gpu_samples[0].memory_total_mb),
        }

    return {
        "telemetry_available": True,
        "sample_rows": len(samples),
        "sample_ticks": int(sample_ticks),
        "gpu_count": len(by_gpu),
        "mean_gpu_utilization_pct": _mean(gpu_utils),
        "max_gpu_utilization_pct": max(gpu_utils),
        "mean_memory_utilization_pct": _mean(mem_utils),
        "max_memory_utilization_pct": max(mem_utils),
        "mean_memory_used_mb": _mean(mem_used),
        "max_memory_used_mb": max(mem_used),
        "per_gpu": per_gpu_summary,
    }


def telemetry_samples_to_json(samples: list[GpuTelemetrySample]) -> list[dict[str, Any]]:
    """Serialize structured GPU telemetry samples for JSON artifacts."""

    return [asdict(sample) for sample in samples]
