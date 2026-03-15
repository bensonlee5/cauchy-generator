from dagzoo.bench.gpu_telemetry import (
    GpuTelemetrySample,
    parse_nvidia_smi_csv,
    summarize_gpu_telemetry,
)


def test_parse_nvidia_smi_csv_parses_multiple_gpu_rows() -> None:
    samples = parse_nvidia_smi_csv(
        "0, NVIDIA H100 NVL, 80, 25, 1024, 95830\n1, NVIDIA H100 NVL, 60, 20, 2048, 95830\n",
        timestamp_utc="2026-03-15T00:00:00+00:00",
    )

    assert [(sample.gpu_index, sample.name) for sample in samples] == [
        (0, "NVIDIA H100 NVL"),
        (1, "NVIDIA H100 NVL"),
    ]
    assert samples[0].utilization_gpu_pct == 80.0
    assert samples[1].memory_used_mb == 2048.0


def test_summarize_gpu_telemetry_reports_mean_and_max_by_gpu() -> None:
    samples = [
        GpuTelemetrySample(
            timestamp_utc="2026-03-15T00:00:00+00:00",
            gpu_index=0,
            name="NVIDIA H100 NVL",
            utilization_gpu_pct=50.0,
            utilization_memory_pct=10.0,
            memory_used_mb=1000.0,
            memory_total_mb=95830.0,
        ),
        GpuTelemetrySample(
            timestamp_utc="2026-03-15T00:00:01+00:00",
            gpu_index=0,
            name="NVIDIA H100 NVL",
            utilization_gpu_pct=70.0,
            utilization_memory_pct=20.0,
            memory_used_mb=1500.0,
            memory_total_mb=95830.0,
        ),
        GpuTelemetrySample(
            timestamp_utc="2026-03-15T00:00:00+00:00",
            gpu_index=1,
            name="NVIDIA H100 NVL",
            utilization_gpu_pct=20.0,
            utilization_memory_pct=5.0,
            memory_used_mb=500.0,
            memory_total_mb=95830.0,
        ),
    ]

    summary = summarize_gpu_telemetry(samples)

    assert summary["telemetry_available"] is True
    assert summary["sample_rows"] == 3
    assert summary["sample_ticks"] == 2
    assert summary["gpu_count"] == 2
    assert summary["mean_gpu_utilization_pct"] == 140.0 / 3.0
    assert summary["max_gpu_utilization_pct"] == 70.0
    assert summary["per_gpu"]["0"]["mean_gpu_utilization_pct"] == 60.0
    assert summary["per_gpu"]["0"]["max_memory_used_mb"] == 1500.0
    assert summary["per_gpu"]["1"]["sample_ticks"] == 1


def test_summarize_gpu_telemetry_handles_empty_sample_set() -> None:
    summary = summarize_gpu_telemetry([])

    assert summary["telemetry_available"] is False
    assert summary["sample_rows"] == 0
    assert summary["per_gpu"] == {}
