"""Maintainer helpers for RD-005 follow-on promotion decisions."""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMOTION_STATUS_PROMOTE = "promote"
PROMOTION_STATUS_HOLD_INTERNAL = "hold_internal"
PROMOTION_STATUS_STRUCTURAL_CONTROL_ONLY = "structural_control_only"

LANE_ROLE_INCUMBENT = "incumbent"
LANE_ROLE_CHALLENGER = "challenger"
LANE_ROLE_STRUCTURAL_CONTROL = "structural_control"

DEFAULT_THROUGHPUT_FLOOR_RATIO = 0.85


@dataclass(frozen=True, slots=True)
class RD005FollowOnLane:
    """One carried internal lane in the RD-005 follow-on suite."""

    label: str
    stress_profile: str
    role: str
    description: str


RD005_FOLLOW_ON_LANES: tuple[RD005FollowOnLane, ...] = (
    RD005FollowOnLane(
        label="compositional",
        stress_profile="anti_memorization_piecewise_classification_compositional_slice_v1",
        role=LANE_ROLE_INCUMBENT,
        description="Current promotion incumbent.",
    ),
    RD005FollowOnLane(
        label="graph-breadth",
        stress_profile="anti_memorization_piecewise_classification_graph_breadth_slice_v1",
        role=LANE_ROLE_STRUCTURAL_CONTROL,
        description="Structural extreme control lane.",
    ),
    RD005FollowOnLane(
        label="categorical-cardinality",
        stress_profile="anti_memorization_piecewise_classification_categorical_cardinality_slice_v1",
        role=LANE_ROLE_CHALLENGER,
        description="Categorical/cardinality challenger lane.",
    ),
    RD005FollowOnLane(
        label="hybrid",
        stress_profile="anti_memorization_piecewise_classification_hybrid_slice_v1",
        role=LANE_ROLE_CHALLENGER,
        description="Hybrid structural plus compositional challenger lane.",
    ),
    RD005FollowOnLane(
        label="robustness-composition",
        stress_profile="anti_memorization_piecewise_classification_robustness_composition_slice_v1",
        role=LANE_ROLE_CHALLENGER,
        description="Missingness/shift/noise robustness challenger lane.",
    ),
)

_LANE_INDEX = {lane.label: lane for lane in RD005_FOLLOW_ON_LANES}


def _numeric_value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        candidate = float(value)
        if math.isfinite(candidate):
            return candidate
    return None


def _downstream_mean(entry: dict[str, Any]) -> float | None:
    payload = entry.get("downstream")
    if isinstance(payload, dict):
        return _numeric_value(payload, "mean")
    return None


def _datasets_per_minute_value(entry: dict[str, Any]) -> float | None:
    return _numeric_value(entry, "datasets_per_minute")


def _structural_diversity_value(entry: dict[str, Any]) -> float | None:
    return _numeric_value(entry, "structural_diversity_composite_shift_pct")


def rd005_priority_sort_key(entry: dict[str, Any]) -> tuple[float, float, float]:
    """Sort by structural diversity first, then throughput, then lower downstream mean."""

    structural = _structural_diversity_value(entry) or 0.0
    throughput = _datasets_per_minute_value(entry) or 0.0
    downstream = _downstream_mean(entry)
    downstream_value = float("inf") if downstream is None else float(downstream)
    return (-float(structural), -float(throughput), downstream_value)


def _variant_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants = payload.get("variants")
    if not isinstance(variants, list):
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for entry in variants:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        if isinstance(label, str) and label:
            mapped[label] = entry
    return mapped


def _required_variant_entry(
    payload: dict[str, Any], label: str, *, source_name: str
) -> dict[str, Any]:
    entry = _variant_map(payload).get(label)
    if entry is None:
        raise ValueError(f"{source_name} is missing required RD-005 lane {label!r}.")
    return entry


def _priority_rank(labels: object, label: str) -> int | None:
    if not isinstance(labels, list):
        return None
    try:
        return [str(item) for item in labels].index(label) + 1
    except ValueError:
        return None


def _render_count_snapshot(payload: object) -> str:
    if not isinstance(payload, dict) or not payload:
        return "-"
    parts: list[str] = []
    for label in sorted(payload):
        value = payload.get(label)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(f"{label}={int(value)}")
    return ", ".join(parts) if parts else "-"


def _render_scalar_snapshot(payload: object) -> str:
    if not isinstance(payload, dict):
        return "-"
    count = payload.get("count")
    if not isinstance(count, (int, float)) or int(count) <= 0:
        return "-"
    mean_value = _numeric_value(payload, "mean")
    min_value = _numeric_value(payload, "min")
    max_value = _numeric_value(payload, "max")
    if mean_value is None or min_value is None or max_value is None:
        return "-"
    return f"count={int(count)}, mean={mean_value:.3f}, range={min_value:.3f}-{max_value:.3f}"


def build_rd005_follow_on_report(
    *,
    baseline_config_path: str,
    diversity_report: dict[str, Any],
    parity_report: dict[str, Any],
    pareto_report: dict[str, Any],
    throughput_floor_ratio: float = DEFAULT_THROUGHPUT_FLOOR_RATIO,
) -> dict[str, Any]:
    """Join diversity, parity, and Pareto evidence into one promotion decision."""

    baseline = pareto_report.get("baseline")
    pareto_summary = pareto_report.get("summary")
    if not isinstance(baseline, dict) or not isinstance(pareto_summary, dict):
        raise ValueError("pareto_report does not look like an RD-005 handoff Pareto report.")

    incumbent_lane = next(
        lane for lane in RD005_FOLLOW_ON_LANES if lane.role == LANE_ROLE_INCUMBENT
    )
    structural_control_lane = next(
        lane for lane in RD005_FOLLOW_ON_LANES if lane.role == LANE_ROLE_STRUCTURAL_CONTROL
    )

    baseline_datasets_per_minute = _datasets_per_minute_value(baseline)
    easy_task_ceiling = _numeric_value(pareto_summary, "easy_task_ceiling_downstream_mean")
    if baseline_datasets_per_minute is None or easy_task_ceiling is None:
        raise ValueError("pareto_report baseline is missing throughput or easy-task ceiling data.")

    diversity_baseline = diversity_report.get("baseline")
    parity_baseline = parity_report.get("baseline")
    if not isinstance(diversity_baseline, dict) or not isinstance(parity_baseline, dict):
        raise ValueError("diversity_report or parity_report is missing baseline data.")

    diversity_variant_map = _variant_map(diversity_report)
    parity_variant_map = _variant_map(parity_report)
    pareto_variant_map = _variant_map(pareto_report)

    incumbent_pareto_entry = _required_variant_entry(
        pareto_report,
        incumbent_lane.label,
        source_name="pareto_report",
    )
    incumbent_sort_key = rd005_priority_sort_key(incumbent_pareto_entry)

    lane_entries: list[dict[str, Any]] = []
    for lane in RD005_FOLLOW_ON_LANES:
        pareto_entry = pareto_variant_map.get(lane.label)
        diversity_entry = diversity_variant_map.get(lane.label)
        parity_entry = parity_variant_map.get(lane.label)
        if pareto_entry is None or diversity_entry is None or parity_entry is None:
            missing = []
            if pareto_entry is None:
                missing.append("pareto_report")
            if diversity_entry is None:
                missing.append("diversity_report")
            if parity_entry is None:
                missing.append("parity_report")
            raise ValueError(f"RD-005 lane {lane.label!r} is missing from: {', '.join(missing)}.")

        datasets_per_minute = _datasets_per_minute_value(pareto_entry)
        downstream_mean = _downstream_mean(pareto_entry)
        structural_shift = _structural_diversity_value(pareto_entry)
        diversity_shift = _numeric_value(pareto_entry, "diversity_composite_shift_pct")
        throughput_ratio = (
            None
            if datasets_per_minute is None or baseline_datasets_per_minute <= 0.0
            else float(datasets_per_minute / baseline_datasets_per_minute)
        )
        easy_task_ceiling_pass = bool(pareto_entry.get("easy_task_ceiling_pass"))
        beats_incumbent = None
        if lane.role != LANE_ROLE_INCUMBENT:
            beats_incumbent = bool(rd005_priority_sort_key(pareto_entry) < incumbent_sort_key)

        failure_reasons: list[str] = []
        if structural_shift is None or structural_shift <= 0.0:
            failure_reasons.append("non_positive_structural_diversity_shift")
        if throughput_ratio is None or throughput_ratio < float(throughput_floor_ratio):
            failure_reasons.append("below_throughput_floor")
        if not easy_task_ceiling_pass:
            failure_reasons.append("above_easy_task_ceiling")
        if lane.role != LANE_ROLE_INCUMBENT and not bool(beats_incumbent):
            failure_reasons.append("does_not_beat_incumbent")

        lane_entries.append(
            {
                "label": lane.label,
                "role": lane.role,
                "description": lane.description,
                "stress_profile": lane.stress_profile,
                "config_path": pareto_entry.get("config_path"),
                "regime_id": pareto_entry.get("regime_id"),
                "datasets_per_minute": datasets_per_minute,
                "throughput_ratio_vs_baseline": throughput_ratio,
                "downstream_mean": downstream_mean,
                "easy_task_ceiling_pass": easy_task_ceiling_pass,
                "diversity_status": pareto_entry.get("diversity_status"),
                "diversity_composite_shift_pct": diversity_shift,
                "structural_diversity_composite_shift_pct": structural_shift,
                "structural_diversity_metric_shift_pct": pareto_entry.get(
                    "structural_diversity_metric_shift_pct"
                ),
                "supporting_metrics": pareto_entry.get("supporting_metrics"),
                "beats_incumbent": beats_incumbent,
                "promotion_gate_pass": not failure_reasons,
                "promotion_failure_reasons": failure_reasons,
                "pareto_frontier_member": lane.label
                in {
                    str(item)
                    for item in pareto_summary.get("pareto_frontier_labels", [])
                    if isinstance(item, str)
                },
                "rd005_priority_rank": _priority_rank(
                    pareto_summary.get("priority_variant_labels"), lane.label
                ),
                "parity_priority_rank": _priority_rank(
                    parity_report.get("summary", {}).get("priority_variant_labels"),
                    lane.label,
                ),
                "parity_surface_summary": parity_entry.get("parity_surface_summary")
                or diversity_entry.get("parity_surface_summary")
                or pareto_entry.get("parity_surface_summary")
                or {},
            }
        )

    eligible_lanes = [entry for entry in lane_entries if bool(entry["promotion_gate_pass"])]
    winner = min(eligible_lanes, key=rd005_priority_sort_key) if eligible_lanes else None
    winner_label = None if winner is None else str(winner["label"])

    for entry in lane_entries:
        if winner_label == entry["label"]:
            entry["promotion_status"] = PROMOTION_STATUS_PROMOTE
            entry["promotion_failure_reasons"] = []
            continue
        if entry["promotion_failure_reasons"]:
            entry["promotion_status"] = (
                PROMOTION_STATUS_STRUCTURAL_CONTROL_ONLY
                if entry["role"] == LANE_ROLE_STRUCTURAL_CONTROL
                else PROMOTION_STATUS_HOLD_INTERNAL
            )
            continue
        entry["promotion_failure_reasons"] = ["outranked_by_winner"]
        entry["promotion_status"] = (
            PROMOTION_STATUS_STRUCTURAL_CONTROL_ONLY
            if entry["role"] == LANE_ROLE_STRUCTURAL_CONTROL
            else PROMOTION_STATUS_HOLD_INTERNAL
        )

    baseline_parity_surface = (
        parity_baseline.get("parity_surface_summary")
        or diversity_baseline.get("parity_surface_summary")
        or baseline.get("parity_surface_summary")
        or {}
    )

    return {
        "schema_name": "dagzoo_rd005_follow_on_promotion_report",
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "baseline": {
            "label": str(baseline.get("label", "baseline")),
            "config_path": str(baseline.get("config_path", baseline_config_path)),
            "datasets_per_minute": baseline_datasets_per_minute,
            "downstream_mean": _downstream_mean(baseline),
            "easy_task_ceiling_downstream_mean": easy_task_ceiling,
            "parity_surface_summary": baseline_parity_surface,
        },
        "lanes": lane_entries,
        "artifacts": {
            "baseline_config_path": str(Path(baseline_config_path).resolve()),
            "diversity_summary_json": diversity_report.get("summary", {}).get(
                "source_summary_json"
            ),
            "parity_report_json": parity_report.get("summary", {}).get("source_parity_report_json"),
            "pareto_summary_json": pareto_summary.get("source_pareto_summary_json"),
        },
        "summary": {
            "promotion_decision": "promote" if winner is not None else "no_promotion",
            "winner_label": winner_label,
            "winner_role": None if winner is None else winner["role"],
            "no_promotion_reason": None if winner is not None else "no_lane_cleared_gate",
            "incumbent_label": incumbent_lane.label,
            "structural_control_label": structural_control_lane.label,
            "candidate_labels": [lane.label for lane in RD005_FOLLOW_ON_LANES],
            "challenger_labels": [
                lane.label for lane in RD005_FOLLOW_ON_LANES if lane.role == LANE_ROLE_CHALLENGER
            ],
            "priority_lane_labels": [
                entry["label"] for entry in sorted(lane_entries, key=rd005_priority_sort_key)
            ],
            "parity_priority_labels": [
                str(item)
                for item in parity_report.get("summary", {}).get("priority_variant_labels", [])
                if isinstance(item, str)
            ],
            "pareto_frontier_labels": [
                str(item)
                for item in pareto_summary.get("pareto_frontier_labels", [])
                if isinstance(item, str)
            ],
            "throughput_floor_ratio": float(throughput_floor_ratio),
            "baseline_datasets_per_minute": baseline_datasets_per_minute,
            "easy_task_ceiling_downstream_mean": easy_task_ceiling,
            "eligible_lane_labels": [entry["label"] for entry in eligible_lanes],
        },
    }


def format_rd005_follow_on_markdown(report: dict[str, Any]) -> str:
    """Render a concise maintainer markdown report for the promotion decision."""

    summary = report.get("summary", {})
    baseline = report.get("baseline", {})
    lanes = report.get("lanes", [])
    artifacts = report.get("artifacts", {})
    lines = [
        "# RD-005 Follow-On Promotion Summary",
        "",
        f"- Promotion decision: `{summary.get('promotion_decision', 'no_promotion')}`",
        f"- Winner: `{summary.get('winner_label') or '-'}`",
        f"- Incumbent: `{summary.get('incumbent_label', '-')}`",
        f"- Structural control: `{summary.get('structural_control_label', '-')}`",
        f"- Throughput floor ratio: `{float(summary.get('throughput_floor_ratio') or 0.0):.2f}`",
        "- Easy-task ceiling downstream mean: "
        f"`{float(summary.get('easy_task_ceiling_downstream_mean') or 0.0):.4f}`",
        f"- Baseline config: `{artifacts.get('baseline_config_path', '-')}`",
        f"- Diversity summary: `{artifacts.get('diversity_summary_json', '-')}`",
        f"- Parity report: `{artifacts.get('parity_report_json', '-')}`",
        f"- Pareto summary: `{artifacts.get('pareto_summary_json', '-')}`",
        "",
        "## Baseline",
        "",
        f"- Label: `{baseline.get('label', '-')}`",
        f"- Datasets/minute: `{float(baseline.get('datasets_per_minute') or 0.0):.2f}`",
        f"- Downstream mean: `{float(baseline.get('downstream_mean') or 0.0):.4f}`",
        "",
        "## Promotion Table",
        "",
        "| Lane | Role | Status | Structural Shift % | Throughput Ratio | Downstream Mean | Ceiling | Beats Incumbent | Reasons |",
        "|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for entry in lanes if isinstance(lanes, list) else []:
        reasons = ", ".join(entry.get("promotion_failure_reasons", [])) or "-"
        beats_incumbent = entry.get("beats_incumbent")
        if beats_incumbent is None:
            beats_text = "-"
        else:
            beats_text = "true" if bool(beats_incumbent) else "false"
        lines.append(
            "| "
            f"{entry.get('label', '-')} | "
            f"{entry.get('role', '-')} | "
            f"{entry.get('promotion_status', '-')} | "
            f"{float(entry.get('structural_diversity_composite_shift_pct') or 0.0):.2f} | "
            f"{float(entry.get('throughput_ratio_vs_baseline') or 0.0):.2f} | "
            f"{float(entry.get('downstream_mean') or 0.0):.4f} | "
            f"{'pass' if bool(entry.get('easy_task_ceiling_pass')) else 'fail'} | "
            f"{beats_text} | "
            f"{reasons} |"
        )

    lines.extend(["", "## Priority Order", ""])
    for label in summary.get("priority_lane_labels", []):
        lines.append(f"- `{label}`")
    lines.extend(["", "## Structural Frontier", ""])
    for label in summary.get("pareto_frontier_labels", []):
        lines.append(f"- `{label}`")
    lines.extend(["", "## Parity Snapshots", ""])
    for entry in lanes if isinstance(lanes, list) else []:
        parity_surface_summary = entry.get("parity_surface_summary")
        if not isinstance(parity_surface_summary, dict):
            parity_surface_summary = {}
        lines.extend(
            [
                f"### {entry.get('label', '-')}",
                "",
                f"- Converter methods: `{_render_count_snapshot(parity_surface_summary.get('converter_method_counts'))}`",
                f"- GP variants: `{_render_count_snapshot(parity_surface_summary.get('gp_variant_counts'))}`",
                f"- Matrix kinds: `{_render_count_snapshot(parity_surface_summary.get('matrix_kind_counts'))}`",
                f"- Root base kinds: `{_render_count_snapshot(parity_surface_summary.get('root_base_kind_counts'))}`",
                f"- Source-shape policy: `{_render_count_snapshot(parity_surface_summary.get('source_shape_policy_counts'))}`",
                f"- Kernel gamma: `{_render_scalar_snapshot(parity_surface_summary.get('kernel_gamma'))}`",
                f"- Categorical cardinality: `{_render_scalar_snapshot(parity_surface_summary.get('categorical_cardinality'))}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_rd005_follow_on_artifacts(
    report: dict[str, Any],
    *,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Persist the combined RD-005 follow-on promotion summary."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_json = out_path / "follow_on_promotion_summary.json"
    summary_md = out_path / "follow_on_promotion_summary.md"
    summary_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md.write_text(format_rd005_follow_on_markdown(report), encoding="utf-8")
    return {
        "summary_json": summary_json,
        "summary_md": summary_md,
    }
