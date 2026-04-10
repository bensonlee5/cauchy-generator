from __future__ import annotations

import json
from pathlib import Path

from conftest import load_script_module


def _load_module():
    return load_script_module(
        "evaluate_rd005_follow_on_suite_script",
        "scripts/evaluate_rd005_follow_on_suite.py",
    )


def _parity_surface(label: str) -> dict[str, object]:
    converter_value = {
        "compositional": 10,
        "graph-breadth": 9,
        "categorical-cardinality": 8,
        "hybrid": 11,
        "robustness-composition": 7,
    }.get(label, 12)
    return {
        "metadata_coverage_rate": 1.0,
        "bundles_with_metadata": 4,
        "converter_method_counts": {"numeric": converter_value},
        "gp_variant_counts": {f"gp.{label}": 4},
        "matrix_kind_counts": {"dense": 4},
        "root_base_kind_counts": {"piecewise": 4},
        "source_shape_policy_counts": {"default": 4},
        "kernel_gamma": {"count": 4, "min": 0.2, "max": 0.8, "mean": 0.5},
        "categorical_cardinality": {"count": 4, "min": 4, "max": 32, "mean": 12.0},
    }


def _fake_diversity_report() -> dict[str, object]:
    return {
        "schema_name": "dagzoo_diversity_audit_report",
        "baseline": {
            "label": "baseline",
            "config_path": "configs/default.yaml",
            "datasets_per_minute": 100.0,
            "parity_surface_summary": _parity_surface("baseline"),
        },
        "variants": [
            {
                "label": "compositional",
                "config_path": "variant_inputs/compositional.yaml",
                "parity_surface_summary": _parity_surface("compositional"),
            },
            {
                "label": "graph-breadth",
                "config_path": "variant_inputs/graph-breadth.yaml",
                "parity_surface_summary": _parity_surface("graph-breadth"),
            },
            {
                "label": "categorical-cardinality",
                "config_path": "variant_inputs/categorical-cardinality.yaml",
                "parity_surface_summary": _parity_surface("categorical-cardinality"),
            },
            {
                "label": "hybrid",
                "config_path": "variant_inputs/hybrid.yaml",
                "parity_surface_summary": _parity_surface("hybrid"),
            },
            {
                "label": "robustness-composition",
                "config_path": "variant_inputs/robustness-composition.yaml",
                "parity_surface_summary": _parity_surface("robustness-composition"),
            },
        ],
        "comparisons": [],
        "summary": {
            "overall_status": "fail",
            "warn_threshold_pct": 2.5,
            "fail_threshold_pct": 5.0,
            "num_variants": 5,
            "probe_num_datasets": 8,
            "probe_warmup_datasets": 0,
            "status_counts": {"fail": 5},
            "core_metrics": [],
        },
    }


def _pareto_variant(
    *,
    label: str,
    structural_shift: float,
    datasets_per_minute: float,
    downstream_mean: float,
) -> dict[str, object]:
    return {
        "label": label,
        "regime_id": f"stress:{label}",
        "config_path": f"variant_inputs/{label}.yaml",
        "generated_corpus_id": f"{label}-corpus",
        "downstream": {"mean": downstream_mean, "median": downstream_mean},
        "datasets_per_minute": datasets_per_minute,
        "diversity_status": "fail",
        "diversity_composite_shift_pct": structural_shift - 2.0,
        "structural_diversity_composite_shift_pct": structural_shift,
        "structural_diversity_metric_shift_pct": {"graph_depth_ratio": structural_shift},
        "easy_task_ceiling_pass": True,
        "supporting_metrics": {"graph_edge_density": 0.7},
        "parity_surface_summary": _parity_surface(label),
    }


def test_evaluate_rd005_follow_on_suite_writes_combined_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    diversity_report = _fake_diversity_report()

    def _fake_run_effective_diversity_audit(**kwargs):
        assert kwargs["variant_labels"] == [
            "compositional",
            "graph-breadth",
            "categorical-cardinality",
            "hybrid",
            "robustness-composition",
        ]
        return diversity_report

    class _ParityScript:
        @staticmethod
        def main(argv):
            args = list(argv)
            out_dir = Path(args[args.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_name": "dagzoo_tabiclv2_parity_report",
                "baseline": diversity_report["baseline"],
                "variants": diversity_report["variants"],
                "summary": {
                    "priority_variant_labels": [
                        "hybrid",
                        "graph-breadth",
                        "categorical-cardinality",
                        "compositional",
                        "robustness-composition",
                    ]
                },
            }
            (out_dir / "parity_report.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (out_dir / "parity_report.md").write_text("# parity\n", encoding="utf-8")
            return 0

    class _ParetoScript:
        @staticmethod
        def main(argv):
            args = list(argv)
            variant_labels = [
                args[index + 1] for index, value in enumerate(args) if value == "--variant-label"
            ]
            assert variant_labels == [
                "compositional",
                "graph-breadth",
                "categorical-cardinality",
                "hybrid",
                "robustness-composition",
            ]
            out_dir = Path(args[args.index("--out-root") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_name": "dagzoo_rd005_handoff_pareto_report",
                "baseline": {
                    "label": "baseline",
                    "config_path": "configs/default.yaml",
                    "datasets_per_minute": 100.0,
                    "downstream": {"mean": 0.50, "median": 0.50},
                    "parity_surface_summary": _parity_surface("baseline"),
                },
                "variants": [
                    _pareto_variant(
                        label="compositional",
                        structural_shift=12.0,
                        datasets_per_minute=90.0,
                        downstream_mean=0.55,
                    ),
                    _pareto_variant(
                        label="graph-breadth",
                        structural_shift=14.0,
                        datasets_per_minute=88.0,
                        downstream_mean=0.56,
                    ),
                    _pareto_variant(
                        label="categorical-cardinality",
                        structural_shift=13.0,
                        datasets_per_minute=86.0,
                        downstream_mean=0.57,
                    ),
                    _pareto_variant(
                        label="hybrid",
                        structural_shift=16.0,
                        datasets_per_minute=89.0,
                        downstream_mean=0.54,
                    ),
                    _pareto_variant(
                        label="robustness-composition",
                        structural_shift=11.0,
                        datasets_per_minute=87.0,
                        downstream_mean=0.53,
                    ),
                ],
                "summary": {
                    "easy_task_ceiling_downstream_mean": 0.60,
                    "priority_variant_labels": [
                        "hybrid",
                        "graph-breadth",
                        "categorical-cardinality",
                        "compositional",
                        "robustness-composition",
                    ],
                    "pareto_frontier_labels": [
                        "baseline",
                        "hybrid",
                        "graph-breadth",
                    ],
                },
            }
            (out_dir / "pareto_summary.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (out_dir / "pareto_summary.md").write_text("# pareto\n", encoding="utf-8")
            return 0

    def _fake_load_repo_script_module(_module_name: str, rel_path: str):
        if rel_path.endswith("render_tabiclv2_parity_report.py"):
            return _ParityScript
        if rel_path.endswith("evaluate_handoff_pareto.py"):
            return _ParetoScript
        raise AssertionError(f"Unexpected script load: {rel_path}")

    monkeypatch.setattr(
        module, "run_effective_diversity_audit", _fake_run_effective_diversity_audit
    )
    monkeypatch.setattr(module, "_load_repo_script_module", _fake_load_repo_script_module)

    out_root = tmp_path / "rd005_follow_on"
    exit_code = module.main(
        [
            "--baseline-config",
            "configs/default.yaml",
            "--out-root",
            str(out_root),
            "--num-datasets",
            "8",
            "--seed",
            "123",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    payload = json.loads(
        (out_root / "follow_on_promotion_summary.json").read_text(encoding="utf-8")
    )
    markdown = (out_root / "follow_on_promotion_summary.md").read_text(encoding="utf-8")

    assert payload["summary"]["candidate_labels"] == [
        "compositional",
        "graph-breadth",
        "categorical-cardinality",
        "hybrid",
        "robustness-composition",
    ]
    assert payload["summary"]["winner_label"] == "hybrid"
    lane_map = {entry["label"]: entry for entry in payload["lanes"]}
    assert lane_map["hybrid"]["promotion_status"] == "promote"
    assert lane_map["graph-breadth"]["promotion_status"] == "structural_control_only"
    assert lane_map["compositional"]["promotion_failure_reasons"] == ["outranked_by_winner"]
    assert lane_map["hybrid"]["parity_surface_summary"]["converter_method_counts"] == {
        "numeric": 11
    }
    assert "## Promotion Table" in markdown
    assert "## Parity Snapshots" in markdown
