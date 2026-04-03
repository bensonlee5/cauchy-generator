from __future__ import annotations

from pathlib import Path

from conftest import load_script_module


def _load_module():
    return load_script_module(
        "evaluate_handoff_pareto_script", "scripts/evaluate_handoff_pareto.py"
    )


def _entry(
    *,
    label: str,
    downstream_mean: float,
    structural_shift: float,
    datasets_per_minute: float,
) -> dict[str, object]:
    return {
        "label": label,
        "downstream": {
            "mean": float(downstream_mean),
            "median": float(downstream_mean),
        },
        "structural_diversity_composite_shift_pct": float(structural_shift),
        "datasets_per_minute": float(datasets_per_minute),
    }


def test_easy_task_ceiling_is_a_maximum_guardrail() -> None:
    module = _load_module()
    baseline = _entry(
        label="baseline",
        downstream_mean=0.50,
        structural_shift=0.0,
        datasets_per_minute=100.0,
    )

    passing = _entry(
        label="passing",
        downstream_mean=0.60,
        structural_shift=5.0,
        datasets_per_minute=90.0,
    )
    failing = _entry(
        label="failing",
        downstream_mean=0.6001,
        structural_shift=6.0,
        datasets_per_minute=95.0,
    )

    assert module._easy_task_ceiling_threshold(baseline) == 0.60
    assert module._passes_easy_task_ceiling(passing, baseline_entry=baseline)
    assert not module._passes_easy_task_ceiling(failing, baseline_entry=baseline)


def test_rank_variants_for_rd005_prefers_structural_shift_then_throughput_then_lower_downstream() -> (
    None
):
    module = _load_module()
    ranked = module._rank_variants_for_rd005(
        [
            _entry(
                label="a", downstream_mean=0.40, structural_shift=12.0, datasets_per_minute=80.0
            ),
            _entry(
                label="b", downstream_mean=0.30, structural_shift=12.0, datasets_per_minute=80.0
            ),
            _entry(
                label="c", downstream_mean=0.20, structural_shift=14.0, datasets_per_minute=70.0
            ),
            _entry(
                label="d", downstream_mean=0.50, structural_shift=12.0, datasets_per_minute=90.0
            ),
        ]
    )

    assert ranked == ["c", "d", "b", "a"]


def test_pareto_frontier_uses_structural_diversity_and_throughput() -> None:
    module = _load_module()
    frontier = module._pareto_frontier(
        [
            _entry(
                label="baseline",
                downstream_mean=0.50,
                structural_shift=0.0,
                datasets_per_minute=100.0,
            ),
            _entry(
                label="v1", downstream_mean=0.30, structural_shift=10.0, datasets_per_minute=80.0
            ),
            _entry(
                label="v2", downstream_mean=0.20, structural_shift=8.0, datasets_per_minute=90.0
            ),
            _entry(
                label="v3", downstream_mean=0.25, structural_shift=8.0, datasets_per_minute=70.0
            ),
        ]
    )

    assert frontier == ["baseline", "v1", "v2"]


def test_write_markdown_report_includes_structural_priority_fields(tmp_path: Path) -> None:
    module = _load_module()
    report = {
        "baseline": {
            "label": "baseline",
            "regime_id": "baseline",
            "generated_corpus_id": "base-corpus",
            "downstream": {"mean": 0.50, "median": 0.50},
            "datasets_per_minute": 100.0,
            "supporting_metrics": {"graph_edge_density": 0.6},
        },
        "variants": [
            {
                "label": "variant",
                "regime_id": "stress:compositional",
                "generated_corpus_id": "variant-corpus",
                "downstream": {"mean": 0.30, "median": 0.25},
                "datasets_per_minute": 95.0,
                "diversity_status": "fail",
                "diversity_composite_shift_pct": 12.0,
                "structural_diversity_composite_shift_pct": 18.0,
                "easy_task_ceiling_pass": True,
                "supporting_metrics": {"graph_edge_density": 0.7},
            }
        ],
        "summary": {
            "num_datasets": 4,
            "seed": 123,
            "device": "cpu",
            "easy_task_ceiling_downstream_mean": 0.60,
            "priority_variant_labels": ["variant"],
            "pareto_frontier_labels": ["baseline", "variant"],
        },
    }

    out_path = tmp_path / "pareto_summary.md"
    module._write_markdown_report(report, out_path=out_path)

    markdown = out_path.read_text(encoding="utf-8")
    assert "## RD-005 Priority Order" in markdown
    assert "Structural diversity composite shift pct: 18.00" in markdown
    assert "Easy-task ceiling pass: `True`" in markdown
    assert "## Structural Frontier" in markdown
