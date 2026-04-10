from __future__ import annotations

import json
from pathlib import Path

from dagzoo.diagnostics.rd005_follow_on import (
    PROMOTION_STATUS_HOLD_INTERNAL,
    PROMOTION_STATUS_PROMOTE,
    PROMOTION_STATUS_STRUCTURAL_CONTROL_ONLY,
    build_rd005_follow_on_report,
    write_rd005_follow_on_artifacts,
)


def _parity_surface(
    *,
    converter_numeric: int,
    gp_variant: str = "gp.standard",
    matrix_kind: str = "dense",
    root_kind: str = "piecewise",
    source_policy: str = "default",
) -> dict[str, object]:
    return {
        "metadata_coverage_rate": 1.0,
        "bundles_with_metadata": 4,
        "converter_method_counts": {"numeric": converter_numeric},
        "gp_variant_counts": {gp_variant: 4},
        "matrix_kind_counts": {matrix_kind: 4},
        "root_base_kind_counts": {root_kind: 4},
        "source_shape_policy_counts": {source_policy: 4},
        "kernel_gamma": {"count": 4, "min": 0.2, "max": 0.8, "mean": 0.5},
        "categorical_cardinality": {"count": 4, "min": 4, "max": 32, "mean": 12.0},
    }


def _diversity_report() -> dict[str, object]:
    return {
        "baseline": {
            "label": "baseline",
            "config_path": "configs/default.yaml",
            "datasets_per_minute": 100.0,
            "parity_surface_summary": _parity_surface(converter_numeric=12),
        },
        "variants": [
            {
                "label": "compositional",
                "config_path": "variant_inputs/compositional.yaml",
                "parity_surface_summary": _parity_surface(
                    converter_numeric=10,
                    gp_variant="gp.periodic",
                    matrix_kind="dense",
                    root_kind="product",
                ),
            },
            {
                "label": "graph-breadth",
                "config_path": "variant_inputs/graph-breadth.yaml",
                "parity_surface_summary": _parity_surface(
                    converter_numeric=9,
                    gp_variant="gp.multiscale",
                    matrix_kind="triangular",
                    source_policy="parent_arity_reuse",
                ),
            },
            {
                "label": "categorical-cardinality",
                "config_path": "variant_inputs/categorical-cardinality.yaml",
                "parity_surface_summary": _parity_surface(
                    converter_numeric=8,
                    matrix_kind="dense",
                    source_policy="categorical_reuse",
                ),
            },
            {
                "label": "hybrid",
                "config_path": "variant_inputs/hybrid.yaml",
                "parity_surface_summary": _parity_surface(
                    converter_numeric=11,
                    gp_variant="gp.periodic",
                    matrix_kind="triangular",
                    root_kind="tree",
                    source_policy="parent_arity_reuse",
                ),
            },
            {
                "label": "robustness-composition",
                "config_path": "variant_inputs/robustness-composition.yaml",
                "parity_surface_summary": _parity_surface(
                    converter_numeric=7,
                    gp_variant="gp.standard",
                    matrix_kind="dense",
                    root_kind="piecewise",
                ),
            },
        ],
        "summary": {},
    }


def _parity_report() -> dict[str, object]:
    payload = _diversity_report()
    return {
        "baseline": payload["baseline"],
        "variants": payload["variants"],
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


def _pareto_variant(
    *,
    label: str,
    structural_shift: float,
    datasets_per_minute: float,
    downstream_mean: float,
    easy_task_ceiling_pass: bool = True,
) -> dict[str, object]:
    return {
        "label": label,
        "regime_id": f"stress:{label}",
        "config_path": f"variant_inputs/{label}.yaml",
        "generated_corpus_id": f"{label}-corpus",
        "downstream": {
            "mean": float(downstream_mean),
            "median": float(downstream_mean),
        },
        "datasets_per_minute": float(datasets_per_minute),
        "diversity_status": "fail",
        "diversity_composite_shift_pct": float(structural_shift - 2.0),
        "structural_diversity_composite_shift_pct": float(structural_shift),
        "structural_diversity_metric_shift_pct": {
            "graph_depth_ratio": float(structural_shift),
        },
        "easy_task_ceiling_pass": bool(easy_task_ceiling_pass),
        "supporting_metrics": {"graph_edge_density": 0.7},
        "parity_surface_summary": _parity_surface(converter_numeric=10),
    }


def _pareto_report(
    *,
    compositional: dict[str, object],
    graph_breadth: dict[str, object],
    categorical_cardinality: dict[str, object],
    hybrid: dict[str, object],
    robustness_composition: dict[str, object],
) -> dict[str, object]:
    return {
        "baseline": {
            "label": "baseline",
            "config_path": "configs/default.yaml",
            "datasets_per_minute": 100.0,
            "downstream": {"mean": 0.50, "median": 0.50},
            "parity_surface_summary": _parity_surface(converter_numeric=12),
        },
        "variants": [
            compositional,
            graph_breadth,
            categorical_cardinality,
            hybrid,
            robustness_composition,
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


def test_build_rd005_follow_on_report_promotes_best_eligible_lane() -> None:
    report = build_rd005_follow_on_report(
        baseline_config_path="configs/default.yaml",
        diversity_report=_diversity_report(),
        parity_report=_parity_report(),
        pareto_report=_pareto_report(
            compositional=_pareto_variant(
                label="compositional",
                structural_shift=12.0,
                datasets_per_minute=90.0,
                downstream_mean=0.55,
            ),
            graph_breadth=_pareto_variant(
                label="graph-breadth",
                structural_shift=14.0,
                datasets_per_minute=88.0,
                downstream_mean=0.56,
            ),
            categorical_cardinality=_pareto_variant(
                label="categorical-cardinality",
                structural_shift=13.0,
                datasets_per_minute=86.0,
                downstream_mean=0.57,
            ),
            hybrid=_pareto_variant(
                label="hybrid",
                structural_shift=16.0,
                datasets_per_minute=89.0,
                downstream_mean=0.54,
            ),
            robustness_composition=_pareto_variant(
                label="robustness-composition",
                structural_shift=11.0,
                datasets_per_minute=87.0,
                downstream_mean=0.53,
            ),
        ),
    )

    lane_status = {entry["label"]: entry for entry in report["lanes"]}
    assert report["summary"]["winner_label"] == "hybrid"
    assert lane_status["hybrid"]["promotion_status"] == PROMOTION_STATUS_PROMOTE
    assert lane_status["compositional"]["promotion_status"] == PROMOTION_STATUS_HOLD_INTERNAL
    assert lane_status["compositional"]["promotion_failure_reasons"] == ["outranked_by_winner"]
    assert (
        lane_status["graph-breadth"]["promotion_status"] == PROMOTION_STATUS_STRUCTURAL_CONTROL_ONLY
    )
    assert lane_status["graph-breadth"]["promotion_failure_reasons"] == ["outranked_by_winner"]
    assert lane_status["categorical-cardinality"]["beats_incumbent"] is True


def test_build_rd005_follow_on_report_returns_no_promotion_when_all_lanes_fail_gate() -> None:
    report = build_rd005_follow_on_report(
        baseline_config_path="configs/default.yaml",
        diversity_report=_diversity_report(),
        parity_report=_parity_report(),
        pareto_report=_pareto_report(
            compositional=_pareto_variant(
                label="compositional",
                structural_shift=12.0,
                datasets_per_minute=80.0,
                downstream_mean=0.55,
            ),
            graph_breadth=_pareto_variant(
                label="graph-breadth",
                structural_shift=20.0,
                datasets_per_minute=70.0,
                downstream_mean=0.57,
            ),
            categorical_cardinality=_pareto_variant(
                label="categorical-cardinality",
                structural_shift=9.0,
                datasets_per_minute=82.0,
                downstream_mean=0.62,
                easy_task_ceiling_pass=False,
            ),
            hybrid=_pareto_variant(
                label="hybrid",
                structural_shift=10.0,
                datasets_per_minute=83.0,
                downstream_mean=0.61,
                easy_task_ceiling_pass=False,
            ),
            robustness_composition=_pareto_variant(
                label="robustness-composition",
                structural_shift=8.0,
                datasets_per_minute=84.0,
                downstream_mean=0.63,
                easy_task_ceiling_pass=False,
            ),
        ),
    )

    lane_status = {entry["label"]: entry for entry in report["lanes"]}
    assert report["summary"]["promotion_decision"] == "no_promotion"
    assert report["summary"]["winner_label"] is None
    assert lane_status["compositional"]["promotion_status"] == PROMOTION_STATUS_HOLD_INTERNAL
    assert "below_throughput_floor" in lane_status["compositional"]["promotion_failure_reasons"]
    assert (
        lane_status["graph-breadth"]["promotion_status"] == PROMOTION_STATUS_STRUCTURAL_CONTROL_ONLY
    )
    assert "below_throughput_floor" in lane_status["graph-breadth"]["promotion_failure_reasons"]


def test_build_rd005_follow_on_report_requires_challenger_to_beat_incumbent() -> None:
    report = build_rd005_follow_on_report(
        baseline_config_path="configs/default.yaml",
        diversity_report=_diversity_report(),
        parity_report=_parity_report(),
        pareto_report=_pareto_report(
            compositional=_pareto_variant(
                label="compositional",
                structural_shift=12.0,
                datasets_per_minute=90.0,
                downstream_mean=0.55,
            ),
            graph_breadth=_pareto_variant(
                label="graph-breadth",
                structural_shift=11.0,
                datasets_per_minute=92.0,
                downstream_mean=0.56,
            ),
            categorical_cardinality=_pareto_variant(
                label="categorical-cardinality",
                structural_shift=12.0,
                datasets_per_minute=89.0,
                downstream_mean=0.50,
            ),
            hybrid=_pareto_variant(
                label="hybrid",
                structural_shift=12.0,
                datasets_per_minute=88.0,
                downstream_mean=0.48,
            ),
            robustness_composition=_pareto_variant(
                label="robustness-composition",
                structural_shift=10.0,
                datasets_per_minute=91.0,
                downstream_mean=0.49,
            ),
        ),
    )

    lane_status = {entry["label"]: entry for entry in report["lanes"]}
    assert report["summary"]["winner_label"] == "compositional"
    assert lane_status["compositional"]["promotion_status"] == PROMOTION_STATUS_PROMOTE
    assert lane_status["categorical-cardinality"]["beats_incumbent"] is False
    assert lane_status["categorical-cardinality"]["promotion_status"] == (
        PROMOTION_STATUS_HOLD_INTERNAL
    )
    assert lane_status["categorical-cardinality"]["promotion_failure_reasons"] == [
        "does_not_beat_incumbent"
    ]


def test_write_rd005_follow_on_artifacts_persists_summary_files(tmp_path: Path) -> None:
    report = build_rd005_follow_on_report(
        baseline_config_path="configs/default.yaml",
        diversity_report=_diversity_report(),
        parity_report=_parity_report(),
        pareto_report=_pareto_report(
            compositional=_pareto_variant(
                label="compositional",
                structural_shift=12.0,
                datasets_per_minute=90.0,
                downstream_mean=0.55,
            ),
            graph_breadth=_pareto_variant(
                label="graph-breadth",
                structural_shift=14.0,
                datasets_per_minute=88.0,
                downstream_mean=0.56,
            ),
            categorical_cardinality=_pareto_variant(
                label="categorical-cardinality",
                structural_shift=13.0,
                datasets_per_minute=86.0,
                downstream_mean=0.57,
            ),
            hybrid=_pareto_variant(
                label="hybrid",
                structural_shift=16.0,
                datasets_per_minute=89.0,
                downstream_mean=0.54,
            ),
            robustness_composition=_pareto_variant(
                label="robustness-composition",
                structural_shift=11.0,
                datasets_per_minute=87.0,
                downstream_mean=0.53,
            ),
        ),
    )

    artifact_paths = write_rd005_follow_on_artifacts(report, out_dir=tmp_path)

    payload = json.loads(artifact_paths["summary_json"].read_text(encoding="utf-8"))
    markdown = artifact_paths["summary_md"].read_text(encoding="utf-8")

    assert artifact_paths["summary_json"] == tmp_path / "follow_on_promotion_summary.json"
    assert artifact_paths["summary_md"] == tmp_path / "follow_on_promotion_summary.md"
    assert payload["summary"]["winner_label"] == "hybrid"
    assert "## Promotion Table" in markdown
