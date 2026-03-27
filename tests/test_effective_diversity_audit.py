import copy
import json

import pytest
import yaml

from dagzoo import generate_batch
from dagzoo.bench.corpus_probe import CorpusProbeResult
from dagzoo.config import GeneratorConfig
from dagzoo.diagnostics.effective_diversity import (
    CORE_DIVERSITY_METRICS,
    compare_coverage_summaries,
    run_effective_diversity_audit,
    write_effective_diversity_artifacts,
)
from dagzoo.diagnostics.effective_diversity.compare import summarize_comparison_status


def _coverage_summary(*, mean: float, p25: float, p50: float, p75: float) -> dict[str, object]:
    metrics = {
        metric: {
            "mean": float(mean),
            "quantiles": {
                "p25": float(p25),
                "p50": float(p50),
                "p75": float(p75),
            },
        }
        for metric in CORE_DIVERSITY_METRICS
    }
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
        "metrics": metrics,
    }


def _probe_result(
    *,
    label: str,
    config_path: str,
    coverage_summary: dict[str, object],
    datasets_per_minute: float = 100.0,
    filter_accepted_datasets_per_minute: float | None = None,
) -> CorpusProbeResult:
    return CorpusProbeResult(
        label=label,
        config_path=config_path,
        suite="smoke",
        num_datasets=4,
        warmup_datasets=0,
        requested_device="cpu",
        resolved_device="cpu",
        resolved_config={"runtime": {"device": "cpu"}},
        datasets_per_minute=float(datasets_per_minute),
        filter_datasets_per_minute=None,
        filter_accepted_datasets_per_minute=filter_accepted_datasets_per_minute,
        filter_accepted_datasets_measured=0,
        filter_rejected_datasets_measured=0,
        filter_acceptance_rate_dataset_level=None,
        filter_rejection_rate_dataset_level=None,
        coverage_summary=coverage_summary,
        filter_summary=None,
    )


def _small_probe_config(*, steering_enabled: bool) -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.filter.enabled = False
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 16
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 4
    cfg.graph.n_nodes_max = 4
    cfg.diagnostics.enabled = True
    if steering_enabled:
        cfg.steering.enabled = True
        cfg.steering.preset = "anti_memorization_piecewise_v1"
    cfg.validate_generation_constraints()
    return cfg


def test_compare_coverage_summaries_classifies_shift_severity() -> None:
    baseline = _coverage_summary(mean=1.0, p25=0.8, p50=1.0, p75=1.2)
    same = compare_coverage_summaries(
        baseline_summary=baseline,
        variant_summary=baseline,
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )
    assert same["diversity_status"] == "pass"
    assert same["diversity_composite_shift_pct"] == pytest.approx(0.0)

    shifted = compare_coverage_summaries(
        baseline_summary=baseline,
        variant_summary=_coverage_summary(mean=1.6, p25=1.4, p50=1.5, p75=1.8),
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )
    assert shifted["diversity_status"] == "fail"
    assert shifted["diversity_composite_shift_pct"] is not None
    assert shifted["diversity_metric_shift_pct"]


def test_compare_coverage_summaries_rejects_swapped_thresholds() -> None:
    baseline = _coverage_summary(mean=1.0, p25=0.8, p50=1.0, p75=1.2)
    with pytest.raises(ValueError, match="warn_threshold_pct must be <= fail_threshold_pct"):
        compare_coverage_summaries(
            baseline_summary=baseline,
            variant_summary=baseline,
            warn_threshold_pct=10.0,
            fail_threshold_pct=5.0,
        )


def test_run_effective_diversity_audit_aggregates_variant_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_result = _probe_result(
        label="baseline",
        config_path="configs/base.yaml",
        coverage_summary=_coverage_summary(mean=1.0, p25=0.8, p50=1.0, p75=1.2),
        datasets_per_minute=120.0,
        filter_accepted_datasets_per_minute=60.0,
    )
    first_variant = _probe_result(
        label="variant",
        config_path="configs/variant.yaml",
        coverage_summary=_coverage_summary(mean=1.01, p25=0.81, p50=1.0, p75=1.19),
        datasets_per_minute=118.0,
        filter_accepted_datasets_per_minute=58.0,
    )
    second_variant = _probe_result(
        label="variant_2",
        config_path="configs/variant.yaml",
        coverage_summary=_coverage_summary(mean=1.6, p25=1.4, p50=1.5, p75=1.8),
        datasets_per_minute=90.0,
        filter_accepted_datasets_per_minute=30.0,
    )
    stub_results = [baseline_result, first_variant, second_variant]

    def _stub_run_corpus_probe(*_args, **_kwargs):
        return stub_results.pop(0)

    monkeypatch.setattr(
        "dagzoo.diagnostics.effective_diversity.runner.run_corpus_probe",
        _stub_run_corpus_probe,
    )
    base_config = GeneratorConfig.from_yaml("configs/default.yaml")
    report = run_effective_diversity_audit(
        baseline_config=base_config,
        baseline_config_path="configs/base.yaml",
        variant_configs=[copy.deepcopy(base_config), copy.deepcopy(base_config)],
        variant_config_paths=["configs/variant.yaml", "configs/variant.yaml"],
        suite="smoke",
        num_datasets=4,
        warmup=0,
        device="cpu",
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )

    assert report["schema_name"] == "dagzoo_diversity_audit_report"
    assert report["schema_version"] == 1
    assert report["summary"]["overall_status"] == "fail"
    assert report["summary"]["probe_num_datasets"] == 4
    assert report["summary"]["probe_warmup_datasets"] == 0
    assert report["summary"]["status_counts"]["pass"] == 1
    assert report["summary"]["status_counts"]["fail"] == 1
    assert [item["label"] for item in report["variants"]] == ["variant", "variant_2"]
    assert report["baseline"]["mechanism_family_summary"]["sampled_family_counts"] == {"linear": 4}
    assert report["comparisons"][1][
        "filter_accepted_datasets_per_minute_delta_pct"
    ] == pytest.approx(-50.0)


def test_effective_diversity_artifact_writer(tmp_path) -> None:
    report = {
        "schema_name": "dagzoo_diversity_audit_report",
        "schema_version": 1,
        "baseline": {
            "label": "baseline",
            "config_path": "configs/base.yaml",
            "datasets_per_minute": 123.0,
            "filter_accepted_datasets_per_minute": 45.0,
            "filter_acceptance_rate_dataset_level": 0.5,
            "mechanism_family_summary": {
                "metadata_coverage_rate": 1.0,
                "bundles_with_metadata": 25,
                "sampled_family_counts": {"linear": 25},
                "dataset_presence_rate_by_family": {"linear": 1.0},
                "sampled_variant_counts": {},
                "dataset_presence_rate_by_variant": {},
                "mean_total_function_plans": 3.0,
            },
        },
        "variants": [
            {
                "label": "variant",
                "mechanism_family_summary": {
                    "metadata_coverage_rate": 1.0,
                    "bundles_with_metadata": 25,
                    "sampled_family_counts": {"gp": 25, "linear": 25},
                    "dataset_presence_rate_by_family": {"gp": 1.0, "linear": 1.0},
                    "sampled_variant_counts": {"gp.periodic": 25},
                    "dataset_presence_rate_by_variant": {"gp.periodic": 1.0},
                    "mean_total_function_plans": 6.0,
                },
            }
        ],
        "comparisons": [
            {
                "variant_label": "variant",
                "diversity_status": "warn",
                "diversity_composite_shift_pct": 3.2,
                "datasets_per_minute_delta_pct": -1.5,
                "filter_accepted_datasets_per_minute_delta_pct": -4.0,
            }
        ],
        "summary": {
            "overall_status": "warn",
            "warn_threshold_pct": 2.5,
            "fail_threshold_pct": 5.0,
            "num_variants": 1,
            "probe_num_datasets": 25,
            "probe_warmup_datasets": 5,
        },
    }

    artifact_paths = write_effective_diversity_artifacts(report, out_dir=tmp_path)
    payload = json.loads(artifact_paths["summary_json"].read_text(encoding="utf-8"))
    markdown = artifact_paths["summary_md"].read_text(encoding="utf-8")

    assert payload["schema_name"] == "dagzoo_diversity_audit_report"
    assert "summary.json` / `summary.md` are the canonical persisted artifacts" in markdown
    assert "Probe num datasets" in markdown
    assert "## Mechanism Families" in markdown
    assert "gp.periodic" in markdown


def test_run_effective_diversity_audit_smoke_filter_disabled(tmp_path) -> None:
    baseline = GeneratorConfig.from_yaml("configs/default.yaml")
    baseline.runtime.device = "cpu"
    baseline.filter.enabled = False
    baseline.dataset.n_train = 24
    baseline.dataset.n_test = 12
    baseline.dataset.n_features_min = 8
    baseline.dataset.n_features_max = 8
    baseline.graph.n_nodes_min = 4
    baseline.graph.n_nodes_max = 5

    variant = copy.deepcopy(baseline)
    variant.graph.n_nodes_min = 6
    variant.graph.n_nodes_max = 7

    baseline_path = tmp_path / "baseline.yaml"
    variant_path = tmp_path / "variant.yaml"
    baseline_path.write_text(yaml.safe_dump(baseline.to_dict()), encoding="utf-8")
    variant_path.write_text(yaml.safe_dump(variant.to_dict()), encoding="utf-8")

    report = run_effective_diversity_audit(
        baseline_config=baseline,
        baseline_config_path=str(baseline_path),
        variant_configs=[variant],
        variant_config_paths=[str(variant_path)],
        suite="smoke",
        num_datasets=2,
        warmup=0,
        device="cpu",
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )

    assert report["schema_name"] == "dagzoo_diversity_audit_report"
    assert report["baseline"]["filter_summary"] is None
    assert report["baseline"]["filter_datasets_per_minute"] is None
    assert report["baseline"]["coverage_summary"]["num_datasets"] == 2
    assert report["comparisons"][0]["diversity_status"] in {"pass", "warn", "fail"}


def test_run_effective_diversity_audit_uses_shared_probe_counts_from_baseline_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_config = GeneratorConfig.from_yaml("configs/default.yaml")
    variant_config = GeneratorConfig.from_yaml("configs/preset_shift_benchmark_smoke.yaml")
    observed_counts: list[tuple[int, int]] = []

    def _stub_run_corpus_probe(*_args, **kwargs):
        observed_counts.append((int(kwargs["num_datasets"]), int(kwargs["warmup"])))
        return _probe_result(
            label=str(kwargs["label"]),
            config_path=str(kwargs["config_path"]),
            coverage_summary=_coverage_summary(mean=1.0, p25=0.8, p50=1.0, p75=1.2),
        )

    monkeypatch.setattr(
        "dagzoo.diagnostics.effective_diversity.runner.run_corpus_probe",
        _stub_run_corpus_probe,
    )
    report = run_effective_diversity_audit(
        baseline_config=baseline_config,
        baseline_config_path="configs/default.yaml",
        variant_configs=[variant_config],
        variant_config_paths=["configs/preset_shift_benchmark_smoke.yaml"],
        suite="smoke",
        num_datasets=None,
        warmup=None,
        device="cpu",
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )

    assert observed_counts == [(25, 5), (25, 5)]
    assert report["summary"]["probe_num_datasets"] == 25
    assert report["summary"]["probe_warmup_datasets"] == 5


def test_run_effective_diversity_audit_uses_shared_probe_seed_from_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_config = GeneratorConfig.from_yaml("configs/default.yaml")
    baseline_config.seed = 123
    variant_config = GeneratorConfig.from_yaml("configs/preset_shift_benchmark_smoke.yaml")
    variant_config.seed = 987
    observed_probe_seeds: list[int] = []

    def _stub_run_corpus_probe(*_args, **kwargs):
        observed_probe_seeds.append(int(kwargs["probe_seed"]))
        return _probe_result(
            label=str(kwargs["label"]),
            config_path=str(kwargs["config_path"]),
            coverage_summary=_coverage_summary(mean=1.0, p25=0.8, p50=1.0, p75=1.2),
        )

    monkeypatch.setattr(
        "dagzoo.diagnostics.effective_diversity.runner.run_corpus_probe",
        _stub_run_corpus_probe,
    )
    _ = run_effective_diversity_audit(
        baseline_config=baseline_config,
        baseline_config_path="configs/default.yaml",
        variant_configs=[variant_config],
        variant_config_paths=["configs/preset_shift_benchmark_smoke.yaml"],
        suite="smoke",
        num_datasets=None,
        warmup=None,
        device="cpu",
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )

    assert observed_probe_seeds == [123, 123]


def test_run_effective_diversity_audit_uses_shared_probe_coverage_config_from_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_config = GeneratorConfig.from_yaml("configs/default.yaml")
    baseline_config.diagnostics.histogram_bins = 7
    baseline_config.diagnostics.quantiles = [0.1, 0.9]
    baseline_config.diagnostics.max_values_per_metric = 3
    variant_config = GeneratorConfig.from_yaml("configs/preset_shift_benchmark_smoke.yaml")
    variant_config.diagnostics.histogram_bins = 99
    variant_config.diagnostics.quantiles = [0.2, 0.8]
    variant_config.diagnostics.max_values_per_metric = 50
    observed_coverage_configs: list[tuple[int, tuple[float, ...], int | None]] = []

    def _stub_run_corpus_probe(*_args, **kwargs):
        coverage_config = kwargs["coverage_config"]
        observed_coverage_configs.append(
            (
                int(coverage_config.histogram_bins),
                tuple(float(value) for value in coverage_config.quantiles),
                coverage_config.max_values_per_metric,
            )
        )
        return _probe_result(
            label=str(kwargs["label"]),
            config_path=str(kwargs["config_path"]),
            coverage_summary=_coverage_summary(mean=1.0, p25=0.8, p50=1.0, p75=1.2),
        )

    monkeypatch.setattr(
        "dagzoo.diagnostics.effective_diversity.runner.run_corpus_probe",
        _stub_run_corpus_probe,
    )
    _ = run_effective_diversity_audit(
        baseline_config=baseline_config,
        baseline_config_path="configs/default.yaml",
        variant_configs=[variant_config],
        variant_config_paths=["configs/preset_shift_benchmark_smoke.yaml"],
        suite="smoke",
        num_datasets=None,
        warmup=None,
        device="cpu",
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )

    assert observed_coverage_configs == [
        (7, (0.1, 0.25, 0.5, 0.75, 0.9), 3),
        (7, (0.1, 0.25, 0.5, 0.75, 0.9), 3),
    ]


def test_run_effective_diversity_audit_uses_run_specific_steering_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._compute_wins_ratio_proxy",
        lambda **_kwargs: 0.6,
    )
    monkeypatch.setattr(
        "dagzoo.bench.corpus_probe.run_throughput_benchmark",
        lambda *_args, **_kwargs: {"datasets_per_minute": 123.0},
    )

    def _stub_iter_throughput_measure_bundles(config, *, num_datasets, device, **_kwargs):
        return iter(
            generate_batch(
                config,
                num_datasets=int(num_datasets),
                seed=int(config.seed),
                device=device,
            )
        )

    monkeypatch.setattr(
        "dagzoo.bench.corpus_probe.iter_throughput_measure_bundles",
        _stub_iter_throughput_measure_bundles,
    )

    baseline_config = _small_probe_config(steering_enabled=True)
    variant_config = _small_probe_config(steering_enabled=False)
    report = run_effective_diversity_audit(
        baseline_config=baseline_config,
        baseline_config_path="configs/base.yaml",
        variant_configs=[variant_config],
        variant_config_paths=["configs/variant.yaml"],
        suite="smoke",
        num_datasets=5,
        warmup=0,
        device="cpu",
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )

    assert (
        report["baseline"]["coverage_summary"]["histogram_bins"]
        == report["variants"][0]["coverage_summary"]["histogram_bins"]
    )
    assert "steering" not in report["baseline"]["coverage_summary"]
    assert "steering" not in report["variants"][0]["coverage_summary"]


def test_run_effective_diversity_audit_ignores_diagnostics_only_drift_in_comparisons(
    tmp_path,
) -> None:
    baseline = GeneratorConfig.from_yaml("configs/default.yaml")
    baseline.runtime.device = "cpu"
    baseline.filter.enabled = False
    baseline.dataset.n_train = 24
    baseline.dataset.n_test = 12
    baseline.dataset.n_features_min = 8
    baseline.dataset.n_features_max = 8
    baseline.graph.n_nodes_min = 4
    baseline.graph.n_nodes_max = 5
    baseline.diagnostics.max_values_per_metric = 1
    baseline.diagnostics.histogram_bins = 5
    baseline.diagnostics.quantiles = [0.1, 0.9]

    variant = copy.deepcopy(baseline)
    variant.diagnostics.max_values_per_metric = 50_000
    variant.diagnostics.histogram_bins = 19
    variant.diagnostics.quantiles = [0.2, 0.8]

    baseline_path = tmp_path / "baseline.yaml"
    variant_path = tmp_path / "variant.yaml"
    baseline_path.write_text(yaml.safe_dump(baseline.to_dict()), encoding="utf-8")
    variant_path.write_text(yaml.safe_dump(variant.to_dict()), encoding="utf-8")

    report = run_effective_diversity_audit(
        baseline_config=baseline,
        baseline_config_path=str(baseline_path),
        variant_configs=[variant],
        variant_config_paths=[str(variant_path)],
        suite="smoke",
        num_datasets=4,
        warmup=0,
        device="cpu",
        warn_threshold_pct=0.001,
        fail_threshold_pct=0.01,
    )

    assert report["comparisons"][0]["diversity_status"] == "pass"
    assert report["comparisons"][0]["diversity_composite_shift_pct"] == pytest.approx(0.0)
    assert report["baseline"]["coverage_summary"]["max_values_per_metric"] == 1
    assert report["variants"][0]["coverage_summary"]["max_values_per_metric"] == 1
    assert report["baseline"]["coverage_summary"]["histogram_bins"] == 5
    assert report["variants"][0]["coverage_summary"]["histogram_bins"] == 5
    assert report["baseline"]["coverage_summary"]["quantiles"] == [0.1, 0.25, 0.5, 0.75, 0.9]
    assert report["variants"][0]["coverage_summary"]["quantiles"] == [0.1, 0.25, 0.5, 0.75, 0.9]


def test_run_effective_diversity_audit_marks_insufficient_metrics_as_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_result = _probe_result(
        label="baseline",
        config_path="configs/base.yaml",
        coverage_summary=_coverage_summary(mean=1.0, p25=0.8, p50=1.0, p75=1.2),
    )
    pass_variant = _probe_result(
        label="variant",
        config_path="configs/variant_a.yaml",
        coverage_summary=_coverage_summary(mean=1.0, p25=0.8, p50=1.0, p75=1.2),
    )
    insufficient_variant = _probe_result(
        label="variant_2",
        config_path="configs/variant_b.yaml",
        coverage_summary={"num_datasets": 0, "metrics": {}},
    )
    stub_results = [baseline_result, pass_variant, insufficient_variant]

    monkeypatch.setattr(
        "dagzoo.diagnostics.effective_diversity.runner.run_corpus_probe",
        lambda *_args, **_kwargs: stub_results.pop(0),
    )
    base_config = GeneratorConfig.from_yaml("configs/default.yaml")
    report = run_effective_diversity_audit(
        baseline_config=base_config,
        baseline_config_path="configs/base.yaml",
        variant_configs=[copy.deepcopy(base_config), copy.deepcopy(base_config)],
        variant_config_paths=["configs/variant_a.yaml", "configs/variant_b.yaml"],
        suite="smoke",
        num_datasets=4,
        warmup=0,
        device="cpu",
        warn_threshold_pct=2.5,
        fail_threshold_pct=5.0,
    )

    assert report["summary"]["overall_status"] == "insufficient_metrics"
    assert report["summary"]["status_counts"]["pass"] == 1
    assert report["summary"]["status_counts"]["insufficient_metrics"] == 1


def test_summarize_comparison_status_prioritizes_incomplete_runs() -> None:
    assert summarize_comparison_status([{"diversity_status": "pass"}]) == "pass"
    assert (
        summarize_comparison_status(
            [{"diversity_status": "pass"}, {"diversity_status": "insufficient_metrics"}]
        )
        == "insufficient_metrics"
    )
    assert (
        summarize_comparison_status(
            [{"diversity_status": "warn"}, {"diversity_status": "insufficient_metrics"}]
        )
        == "insufficient_metrics"
    )
