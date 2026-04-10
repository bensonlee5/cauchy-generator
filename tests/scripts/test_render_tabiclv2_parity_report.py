from __future__ import annotations

import json
from pathlib import Path

from conftest import load_script_module

from dagzoo.diagnostics.effective_diversity import CORE_DIVERSITY_METRICS


def _load_module():
    return load_script_module(
        "render_tabiclv2_parity_report_script",
        "scripts/render_tabiclv2_parity_report.py",
    )


def _coverage_summary(*, mean: float, p25: float, p50: float, p75: float) -> dict[str, object]:
    return {
        "num_datasets": 4,
        "mechanism_family_summary": {
            "metadata_coverage_rate": 1.0,
            "bundles_with_metadata": 4,
            "sampled_family_counts": {"linear": 4},
            "dataset_presence_rate_by_family": {"linear": 1.0},
            "sampled_variant_counts": {},
            "dataset_presence_rate_by_variant": {},
            "mean_total_function_plans": 4.0,
        },
        "parity_surface_summary": {
            "metadata_coverage_rate": 1.0,
            "bundles_with_metadata": 4,
            "converter_method_counts": {"numeric": 12},
            "converter_method_variant_counts": {"numeric.standard": 12},
            "gp_variant_counts": {"gp.standard": 4},
            "matrix_kind_counts": {"dense": 8},
            "root_base_kind_counts": {"piecewise": 4},
            "source_shape_policy_counts": {"parent_arity_reuse": 4},
            "kernel_gamma": {"count": 4, "min": 0.2, "max": 0.8, "mean": 0.5},
            "categorical_cardinality": {"count": 4, "min": 4, "max": 16, "mean": 9.0},
        },
        "metrics": {
            metric: {
                "mean": float(mean),
                "quantiles": {
                    "p25": float(p25),
                    "p50": float(p50),
                    "p75": float(p75),
                },
            }
            for metric in CORE_DIVERSITY_METRICS
        },
    }


def test_render_tabiclv2_parity_report_writes_ranked_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    baseline_summary = _coverage_summary(mean=1.0, p25=0.8, p50=1.0, p75=1.2)
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "schema_name": "dagzoo_diversity_audit_report",
                "baseline": {
                    "label": "baseline",
                    "config_path": "configs/default.yaml",
                    "datasets_per_minute": 100.0,
                    "coverage_summary": baseline_summary,
                    "parity_surface_summary": baseline_summary["parity_surface_summary"],
                },
                "variants": [
                    {
                        "label": "graph-breadth",
                        "config_path": "configs/preset_stress_graph_breadth_benchmark_smoke.yaml",
                        "datasets_per_minute": 82.0,
                        "coverage_summary": _coverage_summary(mean=1.4, p25=1.2, p50=1.4, p75=1.6),
                        "parity_surface_summary": {
                            "metadata_coverage_rate": 1.0,
                            "bundles_with_metadata": 4,
                            "converter_method_counts": {"numeric": 8, "categorical": 4},
                            "converter_method_variant_counts": {
                                "categorical.quantile": 4,
                                "numeric.standard": 8,
                            },
                            "gp_variant_counts": {"gp.multiscale": 4},
                            "matrix_kind_counts": {"dense": 4, "triangular": 4},
                            "root_base_kind_counts": {"piecewise": 2, "tree": 2},
                            "source_shape_policy_counts": {"parent_arity_reuse": 4},
                            "kernel_gamma": {"count": 4, "min": 0.1, "max": 1.2, "mean": 0.6},
                            "categorical_cardinality": {
                                "count": 4,
                                "min": 8,
                                "max": 48,
                                "mean": 20.0,
                            },
                        },
                    },
                    {
                        "label": "compositional",
                        "config_path": "configs/preset_stress_compositional_benchmark_smoke.yaml",
                        "datasets_per_minute": 92.0,
                        "coverage_summary": _coverage_summary(mean=1.6, p25=1.3, p50=1.55, p75=1.8),
                        "parity_surface_summary": {
                            "metadata_coverage_rate": 1.0,
                            "bundles_with_metadata": 4,
                            "converter_method_counts": {"numeric": 10, "categorical": 6},
                            "converter_method_variant_counts": {
                                "categorical.quantile": 6,
                                "numeric.standard": 10,
                            },
                            "gp_variant_counts": {"gp.periodic": 4, "gp.standard": 4},
                            "matrix_kind_counts": {"dense": 6, "triangular": 2},
                            "root_base_kind_counts": {"piecewise": 2, "product": 2, "tree": 2},
                            "source_shape_policy_counts": {"default": 4},
                            "kernel_gamma": {"count": 4, "min": 0.2, "max": 0.9, "mean": 0.55},
                            "categorical_cardinality": {
                                "count": 4,
                                "min": 4,
                                "max": 64,
                                "mean": 24.0,
                            },
                        },
                    },
                ],
                "comparisons": [
                    {
                        "variant_label": "graph-breadth",
                        "diversity_status": "fail",
                        "diversity_composite_shift_pct": 12.0,
                        "datasets_per_minute_delta_pct": -18.0,
                    },
                    {
                        "variant_label": "compositional",
                        "diversity_status": "fail",
                        "diversity_composite_shift_pct": 16.0,
                        "datasets_per_minute_delta_pct": -8.0,
                    },
                ],
                "summary": {
                    "warn_threshold_pct": 2.5,
                    "fail_threshold_pct": 5.0,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "parity_report"
    exit_code = module.main(["--summary-json", str(summary_json), "--out-dir", str(out_dir)])

    assert exit_code == 0
    payload = json.loads((out_dir / "parity_report.json").read_text(encoding="utf-8"))
    markdown = (out_dir / "parity_report.md").read_text(encoding="utf-8")

    assert payload["schema_name"] == "dagzoo_tabiclv2_parity_report"
    assert payload["summary"]["priority_variant_labels"] == ["compositional", "graph-breadth"]
    assert payload["variants"][0]["label"] == "compositional"
    assert "## Priority Order" in markdown
    assert "Converter method+variant" in markdown
    assert "Source-shape policy" in markdown
    assert "compositional" in markdown
