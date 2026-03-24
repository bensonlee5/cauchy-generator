"""Diagnostics-oriented CLI command handlers."""

from __future__ import annotations

import argparse
from pathlib import Path

from dagzoo.diagnostics.effective_diversity import (
    format_filter_calibration_threshold,
    validate_diversity_thresholds,
)

from ..common import get_cli_public_api, load_config_or_usage_error, raise_usage_error


def run_diversity_audit_command(args: argparse.Namespace) -> int:
    """Execute the ``diversity-audit`` command."""

    cli_api = get_cli_public_api()
    try:
        warn_threshold_pct, fail_threshold_pct = validate_diversity_thresholds(
            warn_threshold_pct=float(args.warn_threshold_pct),
            fail_threshold_pct=float(args.fail_threshold_pct),
        )
    except ValueError as exc:
        raise_usage_error(str(exc))
    baseline_config = load_config_or_usage_error(str(args.baseline_config))
    variant_config_paths = [str(path) for path in (args.variant_config or [])]
    variant_configs = [load_config_or_usage_error(path) for path in variant_config_paths]
    if args.device is not None:
        baseline_config.runtime.device = str(args.device)
        for variant_config in variant_configs:
            variant_config.runtime.device = str(args.device)

    try:
        report = cli_api.run_effective_diversity_audit(
            baseline_config=baseline_config,
            baseline_config_path=str(args.baseline_config),
            variant_configs=variant_configs,
            variant_config_paths=variant_config_paths,
            suite=str(args.suite),
            num_datasets=args.num_datasets,
            warmup=args.warmup,
            device=args.device,
            warn_threshold_pct=warn_threshold_pct,
            fail_threshold_pct=fail_threshold_pct,
        )
    except NotImplementedError as exc:
        raise_usage_error(str(exc))

    out_dir = Path(args.out_dir)
    artifact_paths = cli_api.write_effective_diversity_artifacts(report, out_dir=out_dir)
    for key in sorted(artifact_paths):
        print(f"Wrote diversity artifact [{key}]: {artifact_paths[key]}")

    summary = report.get("summary")
    overall_status = "insufficient_metrics"
    if isinstance(summary, dict):
        overall_status = str(summary.get("overall_status", overall_status))
        print(
            "Diversity audit status="
            f"{overall_status} variants={int(summary.get('num_variants', 0))}"
        )

    hard_fail = bool(args.fail_on_regression and overall_status in {"fail", "insufficient_metrics"})
    return 1 if hard_fail else 0


def run_filter_calibration_command(args: argparse.Namespace) -> int:
    """Execute the ``filter-calibration`` command."""

    cli_api = get_cli_public_api()
    try:
        warn_threshold_pct, fail_threshold_pct = validate_diversity_thresholds(
            warn_threshold_pct=float(args.warn_threshold_pct),
            fail_threshold_pct=float(args.fail_threshold_pct),
        )
    except ValueError as exc:
        raise_usage_error(str(exc))
    config = load_config_or_usage_error(str(args.config))
    if args.device is not None:
        config.runtime.device = str(args.device)
    try:
        report = cli_api.run_filter_calibration(
            config=config,
            config_path=str(args.config),
            thresholds=args.thresholds,
            suite=str(args.suite),
            num_datasets=args.num_datasets,
            warmup=args.warmup,
            device=args.device,
            warn_threshold_pct=warn_threshold_pct,
            fail_threshold_pct=fail_threshold_pct,
        )
    except NotImplementedError as exc:
        raise_usage_error(str(exc))
    except ValueError as exc:
        raise_usage_error(str(exc))

    out_dir = Path(args.out_dir)
    artifact_paths = cli_api.write_filter_calibration_artifacts(report, out_dir=out_dir)
    for key in sorted(artifact_paths):
        print(f"Wrote filter calibration artifact [{key}]: {artifact_paths[key]}")

    summary = report.get("summary")
    overall_status = "insufficient_metrics"
    best_overall_status = "insufficient_metrics"
    best_overall_threshold = "-"
    best_passing_threshold = "-"
    num_candidates = 0
    if isinstance(summary, dict):
        overall_status = str(summary.get("overall_status", overall_status))
        best_overall_status = str(summary.get("best_overall_diversity_status", best_overall_status))
        best_overall_raw = summary.get("best_overall_threshold_requested")
        best_overall_threshold = format_filter_calibration_threshold(best_overall_raw)
        best_passing_raw = summary.get("best_passing_threshold_requested")
        best_passing_threshold = format_filter_calibration_threshold(best_passing_raw)
        num_candidates = int(summary.get("num_candidates", 0))
        print(
            "Filter calibration status="
            f"{overall_status} best_overall={best_overall_threshold} "
            f"best_passing={best_passing_threshold} candidates={num_candidates}"
        )

    hard_fail = bool(
        args.fail_on_regression and best_overall_status in {"fail", "insufficient_metrics"}
    )
    return 1 if hard_fail else 0
