"""Internal public-throughput smoke benchmark workflow."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from dagzoo.bench.baseline import (
    build_baseline_payload,
    compare_summary_to_baseline,
    load_baseline,
    write_baseline,
)
from dagzoo.bench.report import write_suite_json
from dagzoo.bench.throughput import (
    run_heterogeneous_throughput_benchmark,
    run_stratified_throughput_benchmark,
    run_throughput_benchmark,
)
from dagzoo.config import GeneratorConfig

_DEFAULT_PUBLIC_THROUGHPUT_SMOKE_METRICS = (
    "fixed_datasets_per_minute",
    "heterogeneous_datasets_per_minute",
    "stratified_datasets_per_minute",
    "heterogeneous_vs_fixed_ratio",
    "stratified_vs_fixed_ratio",
    "heterogeneous_descriptor_share",
    "heterogeneous_raw_batch_share",
    "stratified_descriptor_share",
    "stratified_scheduler_share",
    "stratified_raw_batch_share",
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_share(part: float, total: float) -> float:
    return _safe_ratio(part, total)


def build_public_throughput_smoke_summary(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    warmup_datasets: int,
    device: str | None = None,
) -> dict[str, Any]:
    """Run fixed, heterogeneous, and stratified throughput smoke measurements."""

    fixed_result = run_throughput_benchmark(
        config,
        num_datasets=num_datasets,
        warmup_datasets=warmup_datasets,
        device=device,
    )
    heterogeneous_result = run_heterogeneous_throughput_benchmark(
        config,
        num_datasets=num_datasets,
        warmup_datasets=warmup_datasets,
        device=device,
    )
    stratified_result = run_stratified_throughput_benchmark(
        config,
        num_datasets=num_datasets,
        warmup_datasets=warmup_datasets,
        device=device,
    )

    fixed_dpm = float(fixed_result.get("datasets_per_minute", 0.0) or 0.0)
    heterogeneous_dpm = float(heterogeneous_result.get("datasets_per_minute", 0.0) or 0.0)
    stratified_dpm = float(stratified_result.get("datasets_per_minute", 0.0) or 0.0)
    heterogeneous_elapsed = float(heterogeneous_result.get("elapsed_seconds", 0.0) or 0.0)
    stratified_elapsed = float(stratified_result.get("elapsed_seconds", 0.0) or 0.0)

    preset_key = str(config.benchmark.preset_name or "public_smoke")
    result = {
        "preset_key": preset_key,
        "device": str(device or config.runtime.device or "auto"),
        "num_datasets": int(num_datasets),
        "warmup_datasets": int(warmup_datasets),
        "fixed_datasets_per_minute": fixed_dpm,
        "heterogeneous_datasets_per_minute": heterogeneous_dpm,
        "stratified_datasets_per_minute": stratified_dpm,
        "heterogeneous_vs_fixed_ratio": _safe_ratio(heterogeneous_dpm, fixed_dpm),
        "stratified_vs_fixed_ratio": _safe_ratio(stratified_dpm, fixed_dpm),
        "heterogeneous_descriptor_share": _safe_share(
            float(
                heterogeneous_result.get("heterogeneous_descriptor_resolution_elapsed_seconds", 0.0)
                or 0.0
            ),
            heterogeneous_elapsed,
        ),
        "heterogeneous_raw_batch_share": _safe_share(
            float(heterogeneous_result.get("raw_batch_elapsed_seconds", 0.0) or 0.0),
            heterogeneous_elapsed,
        ),
        "heterogeneous_logical_cohort_count": float(
            heterogeneous_result.get("heterogeneous_logical_cohort_count", 0.0) or 0.0
        ),
        "heterogeneous_mixed_physical_dataset_count": float(
            heterogeneous_result.get("heterogeneous_physical_microbatch_size_sum", 0.0) or 0.0
        ),
        "heterogeneous_executor_fallback_dataset_count": float(
            heterogeneous_result.get("heterogeneous_executor_fallback_dataset_count", 0.0) or 0.0
        ),
        "heterogeneous_supported_singleton_dataset_count": float(
            heterogeneous_result.get("heterogeneous_supported_singleton_dataset_count", 0.0) or 0.0
        ),
        "heterogeneous_avg_predicted_utilization": _safe_ratio(
            float(
                heterogeneous_result.get(
                    "heterogeneous_physical_microbatch_predicted_utilization_sum",
                    0.0,
                )
                or 0.0
            ),
            float(heterogeneous_result.get("heterogeneous_physical_microbatch_count", 0.0) or 0.0),
        ),
        "heterogeneous_avg_supported_buckets_per_node_slot": _safe_ratio(
            float(heterogeneous_result.get("mixed_source_bucket_count", 0.0) or 0.0),
            float(heterogeneous_result.get("mixed_source_node_slot_count", 0.0) or 0.0),
        ),
        "heterogeneous_avg_datasets_per_supported_bucket": _safe_ratio(
            float(heterogeneous_result.get("mixed_source_bucket_dataset_sum", 0.0) or 0.0),
            float(heterogeneous_result.get("mixed_source_bucket_count", 0.0) or 0.0),
        ),
        "heterogeneous_converter_bucket_count": float(
            heterogeneous_result.get("mixed_converter_bucket_count", 0.0) or 0.0
        ),
        "heterogeneous_avg_converter_bucket_size": _safe_ratio(
            float(heterogeneous_result.get("mixed_converter_bucket_dataset_sum", 0.0) or 0.0),
            float(heterogeneous_result.get("mixed_converter_bucket_count", 0.0) or 0.0),
        ),
        "stratified_descriptor_share": _safe_share(
            float(
                stratified_result.get("heterogeneous_descriptor_resolution_elapsed_seconds", 0.0)
                or 0.0
            ),
            stratified_elapsed,
        ),
        "stratified_scheduler_share": _safe_share(
            float(stratified_result.get("stratified_scheduler_elapsed_seconds", 0.0) or 0.0),
            stratified_elapsed,
        ),
        "stratified_raw_batch_share": _safe_share(
            float(stratified_result.get("raw_batch_elapsed_seconds", 0.0) or 0.0),
            stratified_elapsed,
        ),
        "stratified_logical_cohort_count": float(
            stratified_result.get("heterogeneous_logical_cohort_count", 0.0) or 0.0
        ),
        "stratified_mixed_physical_dataset_count": float(
            stratified_result.get("heterogeneous_physical_microbatch_size_sum", 0.0) or 0.0
        ),
        "stratified_executor_fallback_dataset_count": float(
            stratified_result.get("heterogeneous_executor_fallback_dataset_count", 0.0) or 0.0
        ),
        "stratified_supported_singleton_dataset_count": float(
            stratified_result.get("heterogeneous_supported_singleton_dataset_count", 0.0) or 0.0
        ),
        "stratified_avg_predicted_utilization": _safe_ratio(
            float(
                stratified_result.get(
                    "heterogeneous_physical_microbatch_predicted_utilization_sum",
                    0.0,
                )
                or 0.0
            ),
            float(stratified_result.get("heterogeneous_physical_microbatch_count", 0.0) or 0.0),
        ),
        "stratified_avg_supported_buckets_per_node_slot": _safe_ratio(
            float(stratified_result.get("mixed_source_bucket_count", 0.0) or 0.0),
            float(stratified_result.get("mixed_source_node_slot_count", 0.0) or 0.0),
        ),
        "stratified_avg_datasets_per_supported_bucket": _safe_ratio(
            float(stratified_result.get("mixed_source_bucket_dataset_sum", 0.0) or 0.0),
            float(stratified_result.get("mixed_source_bucket_count", 0.0) or 0.0),
        ),
        "stratified_converter_bucket_count": float(
            stratified_result.get("mixed_converter_bucket_count", 0.0) or 0.0
        ),
        "stratified_avg_converter_bucket_size": _safe_ratio(
            float(stratified_result.get("mixed_converter_bucket_dataset_sum", 0.0) or 0.0),
            float(stratified_result.get("mixed_converter_bucket_count", 0.0) or 0.0),
        ),
        "fixed_result": fixed_result,
        "heterogeneous_result": heterogeneous_result,
        "stratified_result": stratified_result,
    }
    return {
        "suite": "public_throughput_smoke",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "preset_results": [result],
    }


def build_public_throughput_smoke_baseline_payload(
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Extract the compact baseline payload used for the public smoke workflow."""

    return build_baseline_payload(
        summary,
        metrics=_DEFAULT_PUBLIC_THROUGHPUT_SMOKE_METRICS,
    )


def _write_public_throughput_smoke_markdown(
    summary: dict[str, Any],
    out_path: str | Path,
) -> Path:
    """Write a concise markdown artifact for the internal public smoke workflow."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preset_results = summary.get("preset_results", [])
    result = preset_results[0] if isinstance(preset_results, list) and preset_results else {}
    regression = summary.get("regression", {})
    lines = [
        "# Public Throughput Smoke",
        "",
        f"- Generated at: `{summary.get('generated_at', '-')}`",
        f"- Regression status: `{regression.get('status', 'pass') if isinstance(regression, dict) else 'pass'}`",
        "",
        "## Ratios",
        "",
        f"- Fixed layout datasets/min: `{float(result.get('fixed_datasets_per_minute', 0.0)):.2f}`",
        f"- Heterogeneous datasets/min: `{float(result.get('heterogeneous_datasets_per_minute', 0.0)):.2f}`",
        f"- Heterogeneous vs fixed ratio: `{float(result.get('heterogeneous_vs_fixed_ratio', 0.0)):.3f}`",
        f"- Stratified datasets/min: `{float(result.get('stratified_datasets_per_minute', 0.0)):.2f}`",
        f"- Stratified vs fixed ratio: `{float(result.get('stratified_vs_fixed_ratio', 0.0)):.3f}`",
        "",
        "## Stage Shares",
        "",
        f"- Heterogeneous descriptor share: `{float(result.get('heterogeneous_descriptor_share', 0.0)):.3f}`",
        f"- Heterogeneous raw-batch share: `{float(result.get('heterogeneous_raw_batch_share', 0.0)):.3f}`",
        f"- Stratified descriptor share: `{float(result.get('stratified_descriptor_share', 0.0)):.3f}`",
        f"- Stratified scheduler share: `{float(result.get('stratified_scheduler_share', 0.0)):.3f}`",
        f"- Stratified raw-batch share: `{float(result.get('stratified_raw_batch_share', 0.0)):.3f}`",
        "",
        "## Coverage",
        "",
        f"- Heterogeneous logical cohorts: `{float(result.get('heterogeneous_logical_cohort_count', 0.0)):.0f}`",
        f"- Heterogeneous mixed physical datasets: `{float(result.get('heterogeneous_mixed_physical_dataset_count', 0.0)):.0f}`",
        f"- Heterogeneous supported singletons: `{float(result.get('heterogeneous_supported_singleton_dataset_count', 0.0)):.0f}`",
        f"- Heterogeneous fallback datasets: `{float(result.get('heterogeneous_executor_fallback_dataset_count', 0.0)):.0f}`",
        f"- Heterogeneous avg predicted utilization: `{float(result.get('heterogeneous_avg_predicted_utilization', 0.0)):.3f}`",
        f"- Heterogeneous avg supported buckets/node slot: `{float(result.get('heterogeneous_avg_supported_buckets_per_node_slot', 0.0)):.3f}`",
        f"- Heterogeneous avg datasets/supported bucket: `{float(result.get('heterogeneous_avg_datasets_per_supported_bucket', 0.0)):.3f}`",
        f"- Heterogeneous converter bucket count: `{float(result.get('heterogeneous_converter_bucket_count', 0.0)):.0f}`",
        f"- Heterogeneous avg converter bucket size: `{float(result.get('heterogeneous_avg_converter_bucket_size', 0.0)):.3f}`",
        f"- Stratified logical cohorts: `{float(result.get('stratified_logical_cohort_count', 0.0)):.0f}`",
        f"- Stratified mixed physical datasets: `{float(result.get('stratified_mixed_physical_dataset_count', 0.0)):.0f}`",
        f"- Stratified supported singletons: `{float(result.get('stratified_supported_singleton_dataset_count', 0.0)):.0f}`",
        f"- Stratified fallback datasets: `{float(result.get('stratified_executor_fallback_dataset_count', 0.0)):.0f}`",
        f"- Stratified avg predicted utilization: `{float(result.get('stratified_avg_predicted_utilization', 0.0)):.3f}`",
        f"- Stratified avg supported buckets/node slot: `{float(result.get('stratified_avg_supported_buckets_per_node_slot', 0.0)):.3f}`",
        f"- Stratified avg datasets/supported bucket: `{float(result.get('stratified_avg_datasets_per_supported_bucket', 0.0)):.3f}`",
        f"- Stratified converter bucket count: `{float(result.get('stratified_converter_bucket_count', 0.0)):.0f}`",
        f"- Stratified avg converter bucket size: `{float(result.get('stratified_avg_converter_bucket_size', 0.0)):.3f}`",
        "",
    ]
    if isinstance(regression, dict) and regression.get("issues"):
        lines.extend(
            [
                "## Regression Issues",
                "",
                "| Severity | Metric | Current | Baseline | Degradation % |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for issue in regression["issues"]:
            lines.append(
                "| "
                f"{issue.get('severity', '-')} | "
                f"{issue.get('metric', '-')} | "
                f"{float(issue.get('current', 0.0)):.3f} | "
                f"{float(issue.get('baseline', 0.0)):.3f} | "
                f"{float(issue.get('degradation_pct', 0.0)):.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dagzoo.bench.public_throughput_smoke",
        description="Internal public heterogeneous throughput smoke workflow.",
    )
    parser.add_argument("--config", default="configs/benchmark_cpu.yaml")
    parser.add_argument("--device")
    parser.add_argument("--num-datasets", type=int)
    parser.add_argument("--warmup-datasets", type=int)
    parser.add_argument("--baseline")
    parser.add_argument("--save-baseline")
    parser.add_argument("--out-dir", default="benchmarks/results/dev_public_smoke")
    parser.add_argument("--warn-threshold-pct", type=float)
    parser.add_argument("--fail-threshold-pct", type=float)
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the internal public-throughput smoke benchmark workflow."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    config = GeneratorConfig.from_yaml(str(args.config))
    num_datasets = (
        int(args.num_datasets)
        if args.num_datasets is not None
        else int(config.benchmark.num_datasets)
    )
    warmup_datasets = (
        int(args.warmup_datasets)
        if args.warmup_datasets is not None
        else int(config.benchmark.warmup_datasets)
    )
    warn_threshold_pct = (
        float(args.warn_threshold_pct)
        if args.warn_threshold_pct is not None
        else float(config.benchmark.warn_threshold_pct)
    )
    fail_threshold_pct = (
        float(args.fail_threshold_pct)
        if args.fail_threshold_pct is not None
        else float(config.benchmark.fail_threshold_pct)
    )

    summary = build_public_throughput_smoke_summary(
        config,
        num_datasets=num_datasets,
        warmup_datasets=warmup_datasets,
        device=args.device,
    )
    if args.baseline:
        summary["regression"] = compare_summary_to_baseline(
            summary,
            load_baseline(str(args.baseline)),
            warn_threshold_pct=warn_threshold_pct,
            fail_threshold_pct=fail_threshold_pct,
            metrics=_DEFAULT_PUBLIC_THROUGHPUT_SMOKE_METRICS,
        )
    else:
        summary["regression"] = {
            "status": "pass",
            "warn_threshold_pct": warn_threshold_pct,
            "fail_threshold_pct": fail_threshold_pct,
            "issues": [],
        }

    out_dir = Path(args.out_dir)
    json_path = write_suite_json(summary, out_dir / "summary.json")
    md_path = _write_public_throughput_smoke_markdown(summary, out_dir / "summary.md")
    result = summary["preset_results"][0]
    print(
        f"[{result['preset_key']}] fixed/min={float(result['fixed_datasets_per_minute']):.2f} "
        f"heterogeneous/min={float(result['heterogeneous_datasets_per_minute']):.2f} "
        f"ratio={float(result['heterogeneous_vs_fixed_ratio']):.3f}"
    )
    print(
        f"[{result['preset_key']}] stratified/min={float(result['stratified_datasets_per_minute']):.2f} "
        f"ratio={float(result['stratified_vs_fixed_ratio']):.3f} "
        f"descriptor_share={float(result['stratified_descriptor_share']):.3f} "
        f"scheduler_share={float(result['stratified_scheduler_share']):.3f}"
    )
    print(f"Wrote public throughput smoke artifacts: {json_path} and {md_path}")

    if args.save_baseline:
        baseline_path = write_baseline(
            build_public_throughput_smoke_baseline_payload(summary),
            str(args.save_baseline),
        )
        print(f"Wrote public throughput smoke baseline: {baseline_path}")

    regression = summary.get("regression", {})
    if (
        args.fail_on_regression
        and isinstance(regression, dict)
        and str(regression.get("status", "pass")) == "fail"
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
