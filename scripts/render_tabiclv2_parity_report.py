#!/usr/bin/env python
"""Render a maintainer-facing TabICLv2 parity report from a diversity audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from dagzoo.diagnostics.effective_diversity.compare import compare_coverage_summaries

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _datasets_per_minute_value(entry: dict[str, Any]) -> float:
    value = entry.get("datasets_per_minute")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _structural_diversity_value(entry: dict[str, Any]) -> float:
    value = entry.get("structural_diversity_composite_shift_pct")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _diversity_value(entry: dict[str, Any]) -> float:
    value = entry.get("diversity_composite_shift_pct")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _priority_sort_key(entry: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -_structural_diversity_value(entry),
        -_datasets_per_minute_value(entry),
        -_diversity_value(entry),
    )


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
    if not isinstance(payload, dict) or int(payload.get("count") or 0) <= 0:
        return "-"
    return (
        f"count={int(payload['count'])}, "
        f"mean={float(payload.get('mean') or 0.0):.3f}, "
        f"range={float(payload.get('min') or 0.0):.3f}-{float(payload.get('max') or 0.0):.3f}"
    )


def _comparison_with_structural(
    *,
    baseline: dict[str, Any],
    variant: dict[str, Any],
    comparison: dict[str, Any] | None,
    summary: dict[str, Any],
) -> dict[str, Any]:
    if comparison is None:
        comparison = {}
    baseline_summary = baseline.get("coverage_summary")
    variant_summary = variant.get("coverage_summary")
    if not isinstance(baseline_summary, dict) or not isinstance(variant_summary, dict):
        return dict(comparison)
    structural = compare_coverage_summaries(
        baseline_summary=baseline_summary,
        variant_summary=variant_summary,
        warn_threshold_pct=float(summary.get("warn_threshold_pct", 2.5)),
        fail_threshold_pct=float(summary.get("fail_threshold_pct", 5.0)),
        include_structural_summary=True,
    )
    return {
        **comparison,
        "diversity_status": structural.get(
            "diversity_status",
            comparison.get("diversity_status", "insufficient_metrics"),
        ),
        "diversity_composite_shift_pct": structural.get(
            "diversity_composite_shift_pct",
            comparison.get("diversity_composite_shift_pct"),
        ),
        "structural_diversity_composite_shift_pct": structural.get(
            "structural_diversity_composite_shift_pct"
        ),
        "structural_diversity_metric_shift_pct": structural.get(
            "structural_diversity_metric_shift_pct"
        ),
    }


def _write_markdown_report(report: dict[str, Any], *, out_path: Path) -> None:
    baseline = report["baseline"]
    lines = [
        "# TabICLv2 Parity Report",
        "",
        f"- Source diversity summary: `{report['summary']['source_summary_json']}`",
        f"- Variants: `{report['summary']['num_variants']}`",
        "- Priority order: structural diversity, throughput, then broader diversity shift",
        "",
        "## Priority Order",
        "",
    ]
    for label in report["summary"]["priority_variant_labels"]:
        lines.append(f"- `{label}`")

    lines.extend(
        [
            "",
            "## Baseline",
            "",
            f"- Label: `{baseline['label']}`",
            f"- Config path: `{baseline['config_path']}`",
            f"- Datasets/minute: `{baseline.get('datasets_per_minute', '-')}`",
            "- Parity surface:",
            f"  - Converter methods: `{_render_count_snapshot(baseline['parity_surface_summary'].get('converter_method_counts'))}`",
            f"  - GP variants: `{_render_count_snapshot(baseline['parity_surface_summary'].get('gp_variant_counts'))}`",
            f"  - Matrix kinds: `{_render_count_snapshot(baseline['parity_surface_summary'].get('matrix_kind_counts'))}`",
            f"  - Root base kinds: `{_render_count_snapshot(baseline['parity_surface_summary'].get('root_base_kind_counts'))}`",
            f"  - Source-shape policy: `{_render_count_snapshot(baseline['parity_surface_summary'].get('source_shape_policy_counts'))}`",
            f"  - Kernel gamma: `{_render_scalar_snapshot(baseline['parity_surface_summary'].get('kernel_gamma'))}`",
            f"  - Categorical cardinality: `{_render_scalar_snapshot(baseline['parity_surface_summary'].get('categorical_cardinality'))}`",
            "",
            "## Variants",
            "",
        ]
    )
    for entry in report["variants"]:
        lines.extend(
            [
                f"### {entry['label']}",
                "",
                f"- Config path: `{entry['config_path']}`",
                f"- Diversity status: `{entry['diversity_status']}`",
                f"- Diversity composite shift pct: `{float(entry.get('diversity_composite_shift_pct') or 0.0):.2f}`",
                "- Structural diversity composite shift pct: "
                f"`{float(entry.get('structural_diversity_composite_shift_pct') or 0.0):.2f}`",
                f"- Datasets/minute: `{float(entry.get('datasets_per_minute') or 0.0):.2f}`",
                f"- Datasets/minute delta pct: `{float(entry.get('datasets_per_minute_delta_pct') or 0.0):.2f}`",
                "- Parity surface:",
                f"  - Converter methods: `{_render_count_snapshot(entry['parity_surface_summary'].get('converter_method_counts'))}`",
                f"  - Converter method+variant: `{_render_count_snapshot(entry['parity_surface_summary'].get('converter_method_variant_counts'))}`",
                f"  - GP variants: `{_render_count_snapshot(entry['parity_surface_summary'].get('gp_variant_counts'))}`",
                f"  - Matrix kinds: `{_render_count_snapshot(entry['parity_surface_summary'].get('matrix_kind_counts'))}`",
                f"  - Root base kinds: `{_render_count_snapshot(entry['parity_surface_summary'].get('root_base_kind_counts'))}`",
                f"  - Source-shape policy: `{_render_count_snapshot(entry['parity_surface_summary'].get('source_shape_policy_counts'))}`",
                f"  - Kernel gamma: `{_render_scalar_snapshot(entry['parity_surface_summary'].get('kernel_gamma'))}`",
                f"  - Categorical cardinality: `{_render_scalar_snapshot(entry['parity_surface_summary'].get('categorical_cardinality'))}`",
                "",
            ]
        )
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


@click.command(
    context_settings=CONTEXT_SETTINGS,
    help="Render a maintainer parity report from one diversity-audit summary.json artifact.",
)
@click.option(
    "--summary-json",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to dagzoo diversity-audit summary.json.",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write parity_report.json and parity_report.md.",
)
def cli(*, summary_json: Path, out_dir: Path) -> int:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    baseline = payload.get("baseline")
    variants = payload.get("variants")
    comparisons = payload.get("comparisons")
    summary = payload.get("summary")
    if (
        not isinstance(baseline, dict)
        or not isinstance(variants, list)
        or not isinstance(summary, dict)
    ):
        raise click.ClickException(
            "summary.json does not look like a dagzoo diversity-audit report."
        )

    comparison_map = {
        str(item.get("variant_label")): item
        for item in comparisons
        if isinstance(comparisons, list) and isinstance(item, dict) and item.get("variant_label")
    }
    rendered_variants: list[dict[str, Any]] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        label = str(variant.get("label", "-"))
        rendered_variants.append(
            {
                "label": label,
                "config_path": variant.get("config_path"),
                "datasets_per_minute": variant.get("datasets_per_minute"),
                "datasets_per_minute_delta_pct": (
                    comparison_map.get(label, {}).get("datasets_per_minute_delta_pct")
                ),
                "parity_surface_summary": (
                    variant.get("parity_surface_summary")
                    if isinstance(variant.get("parity_surface_summary"), dict)
                    else {}
                ),
                **_comparison_with_structural(
                    baseline=baseline,
                    variant=variant,
                    comparison=comparison_map.get(label),
                    summary=summary,
                ),
            }
        )
    rendered_variants.sort(key=_priority_sort_key)

    report = {
        "schema_name": "dagzoo_tabiclv2_parity_report",
        "schema_version": 1,
        "baseline": {
            "label": baseline.get("label"),
            "config_path": baseline.get("config_path"),
            "datasets_per_minute": baseline.get("datasets_per_minute"),
            "parity_surface_summary": (
                baseline.get("parity_surface_summary")
                if isinstance(baseline.get("parity_surface_summary"), dict)
                else {}
            ),
        },
        "variants": rendered_variants,
        "summary": {
            "source_summary_json": str(summary_json.resolve()),
            "num_variants": len(rendered_variants),
            "priority_variant_labels": [entry["label"] for entry in rendered_variants],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "parity_report.json"
    md_path = out_dir / "parity_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown_report(report, out_path=md_path)
    print(f"Wrote parity report: {json_path}")
    print(f"Wrote parity markdown: {md_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        result = cli.main(
            args=argv,
            prog_name="render_tabiclv2_parity_report.py",
            standalone_mode=False,
        )
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return int(exc.exit_code)
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.Abort:
        return 1
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
