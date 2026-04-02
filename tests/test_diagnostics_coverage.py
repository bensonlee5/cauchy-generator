from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dagzoo.config import DiagnosticsConfig, GeneratorConfig
from dagzoo.diagnostics.coverage import (
    CoverageAggregationConfig,
    CoverageAggregator,
    _bundle_config_missing_rate,
    _coerce_optional_float,
    _coerce_optional_int,
    _float_matches,
    _fmt_counts,
    _fmt_int_range,
    _fmt_scalar_summary,
    _fmt_value_range,
    _ValueAccumulator,
    write_coverage_summary_json,
    write_coverage_summary_markdown,
)
from dagzoo.diagnostics.types import DatasetMetrics
from dagzoo.diagnostics_targets import build_diagnostics_aggregation_config
from dagzoo.types import DatasetBundle


def _metric_fixture(**overrides: float | int | str | None) -> DatasetMetrics:
    base = DatasetMetrics(
        task="classification",
        n_rows=128,
        n_features=16,
        n_classes=3,
        n_categorical_features=4,
        categorical_ratio=0.25,
        graph_edge_density=0.4,
        graph_indegree_std=0.6,
        graph_outdegree_std=0.5,
        graph_depth_ratio=0.75,
        graph_reachability_ratio=0.5,
        graph_ancestor_overlap_mean=0.3,
        graph_target_ancestor_fraction=0.625,
        mechanism_family_cooccurrence_ratio=0.4,
        shift_enabled=0.0,
        shift_graph_scale=0.0,
        shift_mechanism_scale=0.0,
        shift_variance_scale=0.0,
        shift_edge_odds_multiplier=1.0,
        shift_mechanism_nonlinear_mass=0.625,
        shift_noise_variance_multiplier=1.0,
        linearity_proxy=0.5,
        nonlinearity_proxy=0.1,
        wins_ratio_proxy=0.6,
        pearson_abs_mean=0.2,
        pearson_abs_max=0.8,
        spearman_abs_mean=None,
        spearman_abs_max=None,
        class_entropy=1.0,
        majority_minority_ratio=1.2,
        snr_proxy_db=3.0,
        cat_cardinality_min=2,
        cat_cardinality_mean=3.0,
        cat_cardinality_max=5,
    )
    return replace(base, **overrides)


def test_coverage_aggregation_correctness_on_fixtures() -> None:
    agg = CoverageAggregator(
        CoverageAggregationConfig(
            histogram_bins=4,
            quantiles=(0.25, 0.5, 0.75),
            target_bands={"pearson_abs_mean": (0.0, 1.0)},
        )
    )
    fixtures = [
        _metric_fixture(pearson_abs_mean=0.1),
        _metric_fixture(pearson_abs_mean=0.3),
        _metric_fixture(pearson_abs_mean=0.7),
        _metric_fixture(pearson_abs_mean=0.9),
    ]
    for payload in fixtures:
        agg.update_metrics(payload)

    summary = agg.build_summary()
    assert summary["num_datasets"] == 4
    metric = summary["metrics"]["pearson_abs_mean"]
    assert metric["count"] == 4
    assert metric["missing_count"] == 0
    assert metric["observed_min"] == pytest.approx(0.1)
    assert metric["observed_max"] == pytest.approx(0.9)
    assert metric["quantiles"]["p50"] == pytest.approx(0.5)
    assert metric["histogram"]["num_bins"] == 4
    assert "underrepresented_bins" in metric


def test_coverage_artifact_schema_required_keys(tmp_path) -> None:
    agg = CoverageAggregator(CoverageAggregationConfig(histogram_bins=6))
    agg.update_metrics(_metric_fixture())
    summary = agg.build_summary()
    json_path = write_coverage_summary_json(summary, tmp_path / "coverage_summary.json")
    md_path = write_coverage_summary_markdown(summary, tmp_path / "coverage_summary.md")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_path.exists()
    assert md_path.exists()
    assert "generated_at" in payload
    assert payload["num_datasets"] == 1
    assert "metrics" in payload
    assert "steering" not in payload

    required_metric_keys = {
        "count",
        "missing_count",
        "observed_min",
        "observed_max",
        "mean",
        "std",
        "sampled_count",
        "sampled_fraction",
        "quantiles",
        "histogram",
        "underrepresented_bins",
        "target_band",
    }
    metric = payload["metrics"]["pearson_abs_mean"]
    assert required_metric_keys.issubset(set(metric))
    assert {"num_bins", "covered_bins", "coverage_ratio", "bins"}.issubset(set(metric["histogram"]))


def test_coverage_aggregation_bounds_sample_memory() -> None:
    agg = CoverageAggregator(
        CoverageAggregationConfig(
            histogram_bins=4,
            max_values_per_metric=3,
        )
    )
    for idx in range(40):
        agg.update_metrics(_metric_fixture(pearson_abs_mean=float(idx) / 40.0))

    summary = agg.build_summary()
    metric = summary["metrics"]["pearson_abs_mean"]
    assert metric["count"] == 40
    assert metric["sampled_count"] == 3
    assert metric["sampled_fraction"] == pytest.approx(3 / 40)
    assert summary["max_values_per_metric"] == 3


def test_underrepresented_regime_detection_against_target_band() -> None:
    agg = CoverageAggregator(
        CoverageAggregationConfig(
            histogram_bins=4,
            underrepresented_threshold=0.5,
            target_bands={"pearson_abs_mean": (0.0, 1.0)},
        )
    )
    for _ in range(8):
        agg.update_metrics(_metric_fixture(pearson_abs_mean=0.0))
    for _ in range(2):
        agg.update_metrics(_metric_fixture(pearson_abs_mean=1.0))

    summary = agg.build_summary()
    under_bins = summary["metrics"]["pearson_abs_mean"]["underrepresented_bins"]
    assert len(under_bins) >= 2
    assert summary["metrics"]["pearson_abs_mean"]["target_band"]["in_target_count"] == 10


def test_coverage_helper_fallbacks_handle_invalid_inputs() -> None:
    accumulator = _ValueAccumulator()
    for value in (True, "bad", float("inf"), float("nan")):
        accumulator.update(value)

    assert accumulator.finalize() == {"count": 0, "min": None, "max": None, "mean": None}
    assert _coerce_optional_int(True) is None
    assert _coerce_optional_int("7") is None
    assert _coerce_optional_int(float("inf")) is None
    assert _coerce_optional_float(False) is None
    assert _coerce_optional_float("0.5") is None
    assert _coerce_optional_float(float("nan")) is None
    assert (
        _bundle_config_missing_rate(DatasetBundle(None, None, None, None, [], metadata={})) is None
    )
    assert (
        _bundle_config_missing_rate(
            DatasetBundle(None, None, None, None, [], metadata={"config": {"dataset": "bad"}})
        )
        is None
    )
    assert _float_matches(None, None) is True
    assert _float_matches(None, 1.0) is False
    assert _fmt_counts([]) == "-"
    assert _fmt_int_range([]) == "-"
    assert _fmt_int_range({"min": "bad", "max": 3}) == "-"
    assert _fmt_value_range([]) == "-"
    assert _fmt_value_range({"min": 0.1, "max": "bad"}) == "-"
    assert _fmt_scalar_summary([]) == "-"
    assert _fmt_scalar_summary({"min": 0.1, "max": 0.2}) == "-"


def test_build_diagnostics_aggregation_config_accepts_diagnostics_config_directly() -> None:
    aggregation_config = build_diagnostics_aggregation_config(
        DiagnosticsConfig(
            include_spearman=True,
            histogram_bins=7,
            quantiles=[0.1, 0.9],
            max_values_per_metric=25,
            meta_feature_targets={"linearity_proxy": [0.2, 0.8]},
        )
    )

    assert aggregation_config.include_spearman is True
    assert aggregation_config.histogram_bins == 7
    assert aggregation_config.quantiles == (0.1, 0.9)
    assert aggregation_config.max_values_per_metric == 25
    assert aggregation_config.target_bands == {"linearity_proxy": (0.2, 0.8)}


def test_coverage_summary_drops_steering_payload_even_when_present_in_config(tmp_path) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.diagnostics.enabled = True
    cfg.steering.enabled = True
    cfg.steering.preset = "anti_memorization_piecewise_v1"

    agg = CoverageAggregator(build_diagnostics_aggregation_config(cfg))
    agg.update_metrics(_metric_fixture())
    summary = agg.build_summary()
    path = write_coverage_summary_json(summary, tmp_path / "coverage_summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "steering" not in summary
    assert "steering" not in payload
    assert "pearson_abs_mean" in payload["metrics"]
