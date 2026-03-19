#!/usr/bin/env python3
# ruff: noqa: I001
"""Render benchmark PR artifacts from a suite summary and checked-in baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dagzoo.bench.baseline import DEFAULT_GATING_METRICS, load_baseline  # noqa: E402
from dagzoo.bench.metrics import degradation_percent  # noqa: E402
from dagzoo.math import sanitize_json  # noqa: E402


DEFAULT_WARN_THRESHOLD_PCT = 10.0
DEFAULT_FAIL_THRESHOLD_PCT = 20.0


def _load_summary(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path)
    if not summary_path.exists():
        return {"preset_results": []}
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Benchmark summary payload must be a JSON object.")
    return payload


def _coerce_metric_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    if not math.isfinite(as_float):
        return None
    return as_float


def _format_value(value: object, *, digits: int = 3) -> str:
    coerced = _coerce_metric_value(value)
    if coerced is None:
        return "-"
    return f"{coerced:.{digits}f}"


def _format_percent(value: object, *, digits: int = 2) -> str:
    coerced = _coerce_metric_value(value)
    if coerced is None:
        return "-"
    return f"{coerced:.{digits}f}"


def _render_threshold_line(report: dict[str, Any]) -> str:
    return (
        f"- Thresholds: warn `{report['warn_threshold_pct']:.1f}%`, "
        f"fail `{report['fail_threshold_pct']:.1f}%`"
    )


def _severity_from_degradation(
    degradation_pct: float | None,
    *,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
) -> str:
    if degradation_pct is None:
        return "n/a"
    if degradation_pct >= fail_threshold_pct:
        return "fail"
    if degradation_pct >= warn_threshold_pct:
        return "warn"
    return "pass"


def build_benchmark_pr_report(
    summary: dict[str, Any],
    baseline: dict[str, Any],
    *,
    title: str,
    summary_path: str | Path,
    baseline_path: str | Path,
    warn_threshold_pct: float = DEFAULT_WARN_THRESHOLD_PCT,
    fail_threshold_pct: float = DEFAULT_FAIL_THRESHOLD_PCT,
    summary_available: bool = True,
) -> dict[str, Any]:
    """Build a machine-readable benchmark delta report."""

    metric_names = list(baseline.get("metrics", []))
    if not metric_names:
        metric_names = list(DEFAULT_GATING_METRICS)

    baseline_presets = baseline.get("presets", {})
    preset_reports: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    preset_results = summary.get("preset_results", [])
    if not isinstance(preset_results, list):
        preset_results = []

    for result in preset_results:
        if not isinstance(result, dict):
            continue

        preset_key = str(result.get("preset_key", "-"))
        preset_baseline = baseline_presets.get(preset_key)
        preset_rows: list[dict[str, Any]] = []

        for metric in metric_names:
            current = _coerce_metric_value(result.get(metric))
            baseline_value = None
            if isinstance(preset_baseline, dict):
                baseline_value = _coerce_metric_value(preset_baseline.get(metric))

            degradation_pct = (
                degradation_percent(metric, float(current), float(baseline_value))
                if current is not None and baseline_value is not None
                else None
            )
            severity = _severity_from_degradation(
                degradation_pct,
                warn_threshold_pct=warn_threshold_pct,
                fail_threshold_pct=fail_threshold_pct,
            )

            row = {
                "metric": metric,
                "current": current,
                "baseline": baseline_value,
                "degradation_pct": degradation_pct,
                "severity": severity,
            }
            preset_rows.append(row)
            if severity in {"warn", "fail"}:
                issues.append({"preset": preset_key, **row})

        preset_status = "pass"
        if any(row["severity"] == "fail" for row in preset_rows):
            preset_status = "fail"
        elif any(row["severity"] == "warn" for row in preset_rows):
            preset_status = "warn"
        elif any(row["severity"] == "n/a" for row in preset_rows):
            preset_status = "n/a"

        preset_reports.append(
            {
                "preset_key": preset_key,
                "status": preset_status,
                "rows": preset_rows,
            }
        )

    status = "pass"
    if not summary_available:
        status = "missing-summary"
    elif any(issue["severity"] == "fail" for issue in issues):
        status = "fail"
    elif issues:
        status = "warn"

    return {
        "title": title,
        "summary_available": bool(summary_available),
        "summary_path": str(summary_path),
        "baseline_path": str(baseline_path),
        "suite": summary.get("suite"),
        "baseline_suite": baseline.get("suite"),
        "warn_threshold_pct": float(warn_threshold_pct),
        "fail_threshold_pct": float(fail_threshold_pct),
        "status": status,
        "metrics": metric_names,
        "preset_reports": preset_reports,
        "issues": issues,
        "issue_count": len(issues),
    }


def render_comment_markdown(report: dict[str, Any]) -> str:
    """Render the compact PR comment body."""

    lines = [
        f"# {report['title']}",
        "",
        f"- Suite: `{report.get('suite', '-')}`",
        f"- Status: `{report['status']}`",
        _render_threshold_line(report),
    ]

    if not report.get("summary_available", True):
        lines.extend(
            [
                "",
                "_Benchmark summary file was not generated._",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"

    baseline_suite = report.get("baseline_suite")
    if baseline_suite is not None and report.get("suite") != baseline_suite:
        lines.append(f"- Baseline suite: `{baseline_suite}`")

    if report.get("issue_count", 0):
        lines.append(f"- Regression issues: `{report['issue_count']}`")

    preset_reports = report.get("preset_reports", [])
    if not isinstance(preset_reports, list) or not preset_reports:
        lines.extend(["", "_No benchmark preset results were found._"])
        return "\n".join(lines).rstrip() + "\n"

    for preset in preset_reports:
        if not isinstance(preset, dict):
            continue
        lines.extend(
            [
                "",
                f"## {preset.get('preset_key', '-')}",
                "",
                "| Metric | Current | Baseline | Delta % | Severity |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in preset.get("rows", []):
            if not isinstance(row, dict):
                continue
            severity = str(row.get("severity", "n/a"))
            lines.append(
                "| "
                f"{row.get('metric', '-')} | "
                f"{_format_value(row.get('current'))} | "
                f"{_format_value(row.get('baseline'))} | "
                f"{_format_percent(row.get('degradation_pct'))} | "
                f"{severity} |"
            )

    if report.get("issues"):
        lines.extend(["", "## Regression Issues"])
        for issue in report["issues"]:
            lines.append(
                f"- `{issue.get('preset', '-')}` / `{issue.get('metric', '-')}`: "
                f"{issue.get('severity')} at {_format_percent(issue.get('degradation_pct'))}% "
                f"({_format_value(issue.get('current'))} vs {_format_value(issue.get('baseline'))})"
            )

    return "\n".join(lines).rstrip() + "\n"


def render_step_summary_markdown(report: dict[str, Any]) -> str:
    """Render the GitHub step-summary body."""

    lines = [
        f"# {report['title']}",
        "",
        f"- Suite: `{report.get('suite', '-')}`",
        f"- Status: `{report['status']}`",
        f"- Summary: `{report['summary_path']}`",
        f"- Baseline: `{report['baseline_path']}`",
        _render_threshold_line(report),
    ]

    preset_reports = report.get("preset_reports", [])
    if not report.get("summary_available", True):
        lines.extend(
            [
                "",
                "_Benchmark summary file was not generated._",
            ]
        )
    elif not isinstance(preset_reports, list) or not preset_reports:
        lines.extend(
            [
                "",
                "_No benchmark preset results were found._",
            ]
        )
    else:
        for preset in preset_reports:
            if not isinstance(preset, dict):
                continue
            lines.extend(
                [
                    "",
                    f"## {preset.get('preset_key', '-')}",
                    "",
                    "| Metric | Current | Baseline | Delta % | Severity |",
                    "|---|---:|---:|---:|---|",
                ]
            )
            for row in preset.get("rows", []):
                if not isinstance(row, dict):
                    continue
                severity = str(row.get("severity", "n/a"))
                lines.append(
                    "| "
                    f"{row.get('metric', '-')} | "
                    f"{_format_value(row.get('current'))} | "
                    f"{_format_value(row.get('baseline'))} | "
                    f"{_format_percent(row.get('degradation_pct'))} | "
                    f"{severity} |"
                )

    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Comment markdown: `{report.get('comment_path', '-')}`",
            f"- Delta JSON: `{report.get('delta_path', '-')}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_report_artifacts(
    report: dict[str, Any],
    *,
    comment_out: str | Path,
    step_summary_out: str | Path,
    delta_out: str | Path,
) -> tuple[Path, Path, Path]:
    """Write the rendered benchmark artifacts to disk."""

    comment_path = Path(comment_out)
    step_summary_path = Path(step_summary_out)
    delta_path = Path(delta_out)

    report = {
        **report,
        "comment_path": str(comment_path),
        "step_summary_path": str(step_summary_path),
        "delta_path": str(delta_path),
    }

    comment_path.parent.mkdir(parents=True, exist_ok=True)
    step_summary_path.parent.mkdir(parents=True, exist_ok=True)
    delta_path.parent.mkdir(parents=True, exist_ok=True)

    comment_path.write_text(render_comment_markdown(report), encoding="utf-8")
    step_summary_path.write_text(render_step_summary_markdown(report), encoding="utf-8")
    with delta_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_json(report), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

    return comment_path, step_summary_path, delta_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark_pr_report")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--comment-out", required=True)
    parser.add_argument("--step-summary-out", required=True)
    parser.add_argument("--delta-out", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--warn-threshold-pct",
        type=float,
        default=DEFAULT_WARN_THRESHOLD_PCT,
    )
    parser.add_argument(
        "--fail-threshold-pct",
        type=float,
        default=DEFAULT_FAIL_THRESHOLD_PCT,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    parsed = parser.parse_args(args)

    summary_path = Path(parsed.summary_json)
    baseline_path = Path(parsed.baseline_json)
    summary = _load_summary(summary_path)
    baseline = load_baseline(baseline_path)
    report = build_benchmark_pr_report(
        summary,
        baseline,
        title=str(parsed.title),
        summary_path=summary_path,
        baseline_path=baseline_path,
        warn_threshold_pct=float(parsed.warn_threshold_pct),
        fail_threshold_pct=float(parsed.fail_threshold_pct),
        summary_available=summary_path.exists(),
    )
    write_report_artifacts(
        report,
        comment_out=parsed.comment_out,
        step_summary_out=parsed.step_summary_out,
        delta_out=parsed.delta_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
