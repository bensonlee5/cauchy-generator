import json

import pytest

from dagzoo.config import GeneratorConfig
from dagzoo.diagnostics.effective_diversity import (
    resolve_filter_calibration_thresholds,
    run_filter_calibration,
    validate_filter_calibration_threshold,
    write_filter_calibration_artifacts,
)


def _audit_entry(
    *,
    label: str,
    threshold_effective_mean: float,
    filter_accepted_datasets_per_minute: float,
    diversity_acceptance_rate: float,
) -> dict[str, object]:
    return {
        "label": label,
        "config_path": "configs/preset_filter_benchmark_smoke.yaml",
        "suite": "smoke",
        "num_datasets": 10,
        "warmup_datasets": 0,
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "resolved_config": {"filter": {"enabled": True}},
        "datasets_per_minute": 100.0,
        "filter_datasets_per_minute": 50.0,
        "filter_accepted_datasets_per_minute": float(filter_accepted_datasets_per_minute),
        "filter_accepted_datasets_measured": 4,
        "filter_rejected_datasets_measured": 6,
        "filter_acceptance_rate_dataset_level": float(diversity_acceptance_rate),
        "filter_rejection_rate_dataset_level": float(1.0 - diversity_acceptance_rate),
        "mechanism_family_summary": {
            "metadata_coverage_rate": 1.0,
            "bundles_with_metadata": 10,
            "sampled_family_counts": {"linear": 10},
            "dataset_presence_rate_by_family": {"linear": 1.0},
            "sampled_variant_counts": {},
            "dataset_presence_rate_by_variant": {},
            "mean_total_function_plans": 4.0,
        },
        "filter_summary": {
            "accepted_true_fraction": float(diversity_acceptance_rate),
            "wins_ratio_mean": 0.91,
            "threshold_effective_mean": float(threshold_effective_mean),
            "threshold_delta_mean": 0.0,
            "n_valid_oob_mean": 96.0,
            "reason_counts": {"insufficient_oob_predictions": 1},
        },
    }


def test_resolve_filter_calibration_thresholds_uses_default_offsets_and_baseline() -> None:
    assert resolve_filter_calibration_thresholds(baseline_threshold=0.95, thresholds=None) == [
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    ]


def test_resolve_filter_calibration_thresholds_clamps_defaults_into_zero_to_one() -> None:
    assert resolve_filter_calibration_thresholds(baseline_threshold=0.05, thresholds=None) == [
        0.0,
        0.05,
        0.1,
    ]


def test_resolve_filter_calibration_thresholds_dedupes_explicit_override() -> None:
    assert resolve_filter_calibration_thresholds(
        baseline_threshold=0.95,
        thresholds=[1.0, 0.8, 0.95, 1.0],
    ) == [0.8, 0.95, 1.0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.1, 1.1])
def test_validate_filter_calibration_threshold_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match=r"must be a finite value in \[0.0, 1.0\]"):
        validate_filter_calibration_threshold(value, field_name="filter.threshold")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.1, 1.1])
def test_run_filter_calibration_rejects_invalid_baseline_config_threshold(
    value: float,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/preset_filter_benchmark_smoke.yaml")
    cfg.filter.threshold = value

    with pytest.raises(
        ValueError, match=r"filter.threshold must be a finite value in \[0.0, 1.0\]"
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


def test_run_filter_calibration_ranks_best_overall_and_best_passing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/preset_filter_benchmark_smoke.yaml")
    captured: dict[str, object] = {}

    def _stub_run_effective_diversity_audit(**kwargs):
        captured["variant_labels"] = kwargs["variant_labels"]
        return {
            "generated_at": "2026-03-10T08:00:00+00:00",
            "baseline": _audit_entry(
                label="baseline",
                threshold_effective_mean=0.95,
                filter_accepted_datasets_per_minute=40.0,
                diversity_acceptance_rate=0.4,
            ),
            "variants": [
                _audit_entry(
                    label="thr_0.8",
                    threshold_effective_mean=0.8,
                    filter_accepted_datasets_per_minute=60.0,
                    diversity_acceptance_rate=0.6,
                ),
                _audit_entry(
                    label="thr_1.0",
                    threshold_effective_mean=1.0,
                    filter_accepted_datasets_per_minute=80.0,
                    diversity_acceptance_rate=0.3,
                ),
            ],
            "comparisons": [
                {
                    "variant_label": "thr_0.8",
                    "diversity_status": "pass",
                    "diversity_composite_shift_pct": 1.5,
                    "diversity_metric_shift_pct": {"linearity_proxy": 1.5},
                    "datasets_per_minute_delta_pct": 5.0,
                    "filter_accepted_datasets_per_minute_delta_pct": 50.0,
                },
                {
                    "variant_label": "thr_1.0",
                    "diversity_status": "fail",
                    "diversity_composite_shift_pct": 8.0,
                    "diversity_metric_shift_pct": {"linearity_proxy": 8.0},
                    "datasets_per_minute_delta_pct": 0.0,
                    "filter_accepted_datasets_per_minute_delta_pct": 100.0,
                },
            ],
            "summary": {
                "overall_status": "fail",
                "probe_num_datasets": 10,
                "probe_warmup_datasets": 0,
            },
        }

    monkeypatch.setattr(
        "dagzoo.diagnostics.effective_diversity.calibration.run_effective_diversity_audit",
        _stub_run_effective_diversity_audit,
    )

    report = run_filter_calibration(
        config=cfg,
        config_path="configs/preset_filter_benchmark_smoke.yaml",
        thresholds=[0.8, 1.0],
        suite="smoke",
        num_datasets=10,
        warmup=0,
        device="cpu",
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )

    assert captured["variant_labels"] == ["thr_0.8", "thr_1.0"]
    assert report["schema_name"] == "dagzoo_filter_calibration_report"
    assert report["summary"]["baseline_threshold_requested"] == pytest.approx(0.95)
    assert report["summary"]["best_overall_threshold_requested"] == pytest.approx(1.0)
    assert report["summary"]["best_overall_diversity_status"] == "fail"
    assert report["summary"]["best_passing_threshold_requested"] == pytest.approx(0.8)
    assert report["candidates"][0]["diversity_status"] == "pass"
    assert report["baseline"]["diversity_status"] == "reference"
    assert report["candidates"][1]["threshold_effective_mean"] == pytest.approx(0.95)


def test_run_filter_calibration_keeps_fine_grained_threshold_candidates_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/preset_filter_benchmark_smoke.yaml")
    captured: dict[str, object] = {}

    def _stub_run_effective_diversity_audit(**kwargs):
        captured["variant_labels"] = kwargs["variant_labels"]
        return {
            "generated_at": "2026-03-10T08:00:00+00:00",
            "baseline": _audit_entry(
                label="baseline",
                threshold_effective_mean=0.95,
                filter_accepted_datasets_per_minute=40.0,
                diversity_acceptance_rate=0.4,
            ),
            "variants": [
                _audit_entry(
                    label="thr_0.801",
                    threshold_effective_mean=0.801,
                    filter_accepted_datasets_per_minute=61.0,
                    diversity_acceptance_rate=0.61,
                ),
                _audit_entry(
                    label="thr_0.804",
                    threshold_effective_mean=0.804,
                    filter_accepted_datasets_per_minute=54.0,
                    diversity_acceptance_rate=0.54,
                ),
            ],
            "comparisons": [
                {
                    "variant_label": "thr_0.801",
                    "diversity_status": "pass",
                    "diversity_composite_shift_pct": 1.0,
                    "diversity_metric_shift_pct": {"linearity_proxy": 1.0},
                    "datasets_per_minute_delta_pct": 2.0,
                    "filter_accepted_datasets_per_minute_delta_pct": 52.5,
                },
                {
                    "variant_label": "thr_0.804",
                    "diversity_status": "warn",
                    "diversity_composite_shift_pct": 3.5,
                    "diversity_metric_shift_pct": {"linearity_proxy": 3.5},
                    "datasets_per_minute_delta_pct": 1.0,
                    "filter_accepted_datasets_per_minute_delta_pct": 35.0,
                },
            ],
            "summary": {
                "overall_status": "warn",
                "probe_num_datasets": 10,
                "probe_warmup_datasets": 0,
            },
        }

    monkeypatch.setattr(
        "dagzoo.diagnostics.effective_diversity.calibration.run_effective_diversity_audit",
        _stub_run_effective_diversity_audit,
    )

    report = run_filter_calibration(
        config=cfg,
        config_path="configs/preset_filter_benchmark_smoke.yaml",
        thresholds=[0.801, 0.804],
        suite="smoke",
        num_datasets=10,
        warmup=0,
        device="cpu",
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )

    assert captured["variant_labels"] == ["thr_0.801", "thr_0.804"]
    assert [candidate["label"] for candidate in report["candidates"][:2]] == [
        "thr_0.801",
        "thr_0.804",
    ]
    assert report["candidates"][0]["filter_accepted_datasets_per_minute"] == pytest.approx(61.0)
    assert report["candidates"][0]["diversity_status"] == "pass"
    assert report["candidates"][1]["filter_accepted_datasets_per_minute"] == pytest.approx(54.0)
    assert report["candidates"][1]["diversity_status"] == "warn"
    assert report["summary"]["best_overall_threshold_requested"] == pytest.approx(0.801)
    assert report["summary"]["best_passing_threshold_requested"] == pytest.approx(0.801)


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
