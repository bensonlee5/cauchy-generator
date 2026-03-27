"""Benchmark scenario issue helpers."""

from __future__ import annotations

import math
from typing import Any


def _build_guardrail_issue(
    *,
    metric: str,
    severity: str,
    current: float | None,
    baseline: float | None,
    degradation_pct: float | None,
    detail: str,
) -> dict[str, Any]:
    """Create a normalized issue payload."""

    return {
        "metric": metric,
        "severity": severity,
        "current": current,
        "baseline": baseline,
        "degradation_pct": degradation_pct,
        "detail": detail,
    }


def _severity_from_thresholds(value: float, *, warn: float, fail: float) -> str:
    """Map a numeric value to pass/warn/fail severity."""

    if value >= fail:
        return "fail"
    if value >= warn:
        return "warn"
    return "pass"


def _status_from_issues(issues: list[dict[str, Any]]) -> str:
    """Collapse per-metric issues into an overall status."""

    severities = {str(issue.get("severity", "pass")) for issue in issues}
    if "fail" in severities:
        return "fail"
    if "warn" in severities:
        return "warn"
    return "pass"


def _issue_sort_key(issue: dict[str, Any]) -> tuple[int, float]:
    """Sort issues by severity then descending degradation percentage."""

    severity = str(issue.get("severity", "warn"))
    rank = 0 if severity == "fail" else 1 if severity == "warn" else 2
    raw_degradation = issue.get("degradation_pct")
    if isinstance(raw_degradation, bool) or not isinstance(raw_degradation, (int, float)):
        return (rank, 0.0)
    degradation = float(raw_degradation)
    if not math.isfinite(degradation):
        return (rank, 0.0)
    return (rank, -degradation)


def _collect_scenario_regression_issues(
    preset_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten scenario issues from preset results into regression issues."""

    issues: list[dict[str, Any]] = []
    for result in preset_results:
        scenarios = result.get("scenarios")
        if not isinstance(scenarios, dict):
            continue
        for scenario_name, scenario in sorted(scenarios.items()):
            if not isinstance(scenario, dict) or not bool(scenario.get("enabled")):
                continue
            raw_issues = scenario.get("issues")
            if not isinstance(raw_issues, list):
                continue
            for issue in raw_issues:
                if not isinstance(issue, dict):
                    continue
                merged = dict(issue)
                merged["preset"] = str(result.get("preset_key"))
                merged["scenario"] = str(scenario_name)
                issues.append(merged)
    return issues
