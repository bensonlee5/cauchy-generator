import json

import pytest

from dagzoo.config import GeneratorConfig
from dagzoo.diagnostics.effective_diversity import (
    resolve_filter_calibration_thresholds,
    run_filter_calibration,
    validate_filter_calibration_threshold,
    write_filter_calibration_artifacts,
)


def test_resolve_filter_calibration_thresholds_uses_default_offsets_and_baseline() -> None:
    assert resolve_filter_calibration_thresholds(baseline_threshold=0.95, thresholds=None) == [
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.1, 1.1])
def test_validate_filter_calibration_threshold_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match=r"must be a finite value in \[0.0, 1.0\]"):
        validate_filter_calibration_threshold(value, field_name="thresholds")


def test_run_filter_calibration_is_explicitly_unsupported() -> None:
    cfg = GeneratorConfig.from_yaml("configs/preset_filter_benchmark_smoke.yaml")

    with pytest.raises(
        NotImplementedError,
        match="filter-calibration is not supported for the small-shot ease filter yet",
    ):
        run_filter_calibration(
            config=cfg,
            config_path="configs/preset_filter_benchmark_smoke.yaml",
            thresholds=None,
            suite="smoke",
            num_datasets=10,
            warmup=0,
            device="cpu",
            warn_threshold_pct=2.5,
            fail_threshold_pct=5.0,
        )


def test_write_filter_calibration_artifacts(tmp_path) -> None:
    report = {
        "schema_name": "dagzoo_filter_calibration_report",
        "schema_version": 1,
        "baseline": {
            "label": "baseline",
            "threshold_requested": 0.95,
            "filter_accepted_datasets_per_minute": 45.0,
            "filter_acceptance_rate_dataset_level": 0.5,
            "mechanism_family_summary": {
                "metadata_coverage_rate": 1.0,
                "bundles_with_metadata": 10,
                "sampled_family_counts": {"gp": 10, "linear": 10},
                "dataset_presence_rate_by_family": {"gp": 1.0, "linear": 1.0},
                "sampled_variant_counts": {"gp.multiscale": 10},
                "dataset_presence_rate_by_variant": {"gp.multiscale": 1.0},
                "mean_total_function_plans": 6.0,
            },
        },
        "candidates": [
            {
                "label": "baseline",
                "threshold_requested": 0.95,
                "diversity_status": "reference",
                "filter_accepted_datasets_per_minute": 45.0,
                "filter_acceptance_rate_dataset_level": 0.5,
                "diversity_composite_shift_pct": None,
                "mechanism_family_summary": {
                    "metadata_coverage_rate": 1.0,
                    "bundles_with_metadata": 10,
                    "sampled_family_counts": {"gp": 10, "linear": 10},
                    "dataset_presence_rate_by_family": {"gp": 1.0, "linear": 1.0},
                    "sampled_variant_counts": {"gp.multiscale": 10},
                    "dataset_presence_rate_by_variant": {"gp.multiscale": 1.0},
                    "mean_total_function_plans": 6.0,
                },
            }
        ],
        "comparisons": [],
        "summary": {
            "overall_status": "reference",
            "baseline_threshold_requested": 0.95,
            "best_overall_threshold_requested": 0.95,
            "best_overall_diversity_status": "reference",
            "best_passing_threshold_requested": None,
            "num_candidates": 1,
            "probe_num_datasets": 10,
            "probe_warmup_datasets": 0,
        },
    }

    artifact_paths = write_filter_calibration_artifacts(report, out_dir=tmp_path)
    payload = json.loads(artifact_paths["summary_json"].read_text(encoding="utf-8"))
    markdown = artifact_paths["summary_md"].read_text(encoding="utf-8")

    assert payload["schema_name"] == "dagzoo_filter_calibration_report"
    assert "Best overall threshold" in markdown
    assert "canonical persisted artifacts for filter calibration" in markdown
    assert "## Mechanism Families" in markdown
    assert "gp.multiscale" in markdown
