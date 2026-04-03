"""Diagnostics-oriented CLI command handlers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from dagzoo.diagnostics.effective_diversity import (
    run_effective_diversity_audit,
    validate_diversity_thresholds,
    write_effective_diversity_artifacts,
)

from ..common import load_config_or_usage_error, raise_usage_error


def run_diversity_audit_command(
    *,
    baseline_config: str,
    variant_config: Sequence[str],
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    fail_on_regression: bool = False,
    suite: str = "standard",
    num_datasets: int | None = None,
    warmup: int | None = None,
    out_dir: str | Path = Path("effective_config_artifacts") / "diversity_audit",
    device: str | None = None,
) -> int:
    """Execute the ``diversity-audit`` command."""

    try:
        warn_threshold_pct, fail_threshold_pct = validate_diversity_thresholds(
            warn_threshold_pct=float(warn_threshold_pct),
            fail_threshold_pct=float(fail_threshold_pct),
        )
    except ValueError as exc:
        raise_usage_error(str(exc))
    baseline = load_config_or_usage_error(str(baseline_config))
    variant_config_paths = [str(path) for path in (variant_config or ())]
    variant_configs = [load_config_or_usage_error(path) for path in variant_config_paths]
    if device is not None:
        baseline.runtime.device = str(device)
        for variant_cfg in variant_configs:
            variant_cfg.runtime.device = str(device)

    try:
        report = run_effective_diversity_audit(
            baseline_config=baseline,
            baseline_config_path=str(baseline_config),
            variant_configs=variant_configs,
            variant_config_paths=variant_config_paths,
            suite=str(suite),
            num_datasets=num_datasets,
            warmup=warmup,
            device=device,
            warn_threshold_pct=warn_threshold_pct,
            fail_threshold_pct=fail_threshold_pct,
        )
    except NotImplementedError as exc:
        raise_usage_error(str(exc))

    output_dir = Path(out_dir)
    artifact_paths = write_effective_diversity_artifacts(report, out_dir=output_dir)
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

    hard_fail = bool(fail_on_regression and overall_status in {"fail", "insufficient_metrics"})
    return 1 if hard_fail else 0
