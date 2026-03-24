from __future__ import annotations

import json
from dataclasses import replace

import pytest
import yaml

from dagzoo.cli import main
from dagzoo.config import GeneratorConfig
from dagzoo.diagnostics.coverage import (
    CoverageAggregationConfig,
    CoverageAggregator,
    write_coverage_summary_json,
    write_coverage_summary_markdown,
)
from dagzoo.diagnostics.types import DatasetMetrics


def _metric_fixture(**overrides: float | int | str | None) -> DatasetMetrics:
    base = DatasetMetrics(
        task="classification",
        n_rows=128,
        n_features=16,
        n_classes=3,
        n_categorical_features=4,
        categorical_ratio=0.25,
        graph_edge_density=0.4,
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
            target_bands={"linearity_proxy": (0.0, 1.0)},
        )
    )
    fixtures = [
        _metric_fixture(linearity_proxy=0.1, wins_ratio_proxy=0.4),
        _metric_fixture(linearity_proxy=0.3, wins_ratio_proxy=0.5),
        _metric_fixture(linearity_proxy=0.7, wins_ratio_proxy=0.8),
        _metric_fixture(linearity_proxy=0.9, wins_ratio_proxy=0.95),
    ]
    for payload in fixtures:
        agg.update_metrics(payload)

    summary = agg.build_summary()
    assert summary["num_datasets"] == 4
    lin = summary["metrics"]["linearity_proxy"]
    assert lin["count"] == 4
    assert lin["missing_count"] == 0
    assert lin["observed_min"] == pytest.approx(0.1)
    assert lin["observed_max"] == pytest.approx(0.9)
    assert lin["quantiles"]["p50"] == pytest.approx(0.5)
    assert lin["histogram"]["num_bins"] == 4
    assert "underrepresented_bins" in lin


def test_coverage_artifact_schema_required_keys(tmp_path) -> None:
    agg = CoverageAggregator(CoverageAggregationConfig(histogram_bins=6))
    agg.update_metrics(_metric_fixture())
    summary = agg.build_summary()
    json_path = write_coverage_summary_json(summary, tmp_path / "coverage_summary.json")
    md_path = write_coverage_summary_markdown(summary, tmp_path / "coverage_summary.md")

    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "generated_at" in payload
    assert payload["num_datasets"] == 1
    assert "metrics" in payload

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
    line_metric = payload["metrics"]["linearity_proxy"]
    assert required_metric_keys.issubset(set(line_metric))
    assert {"num_bins", "covered_bins", "coverage_ratio", "bins"}.issubset(
        set(line_metric["histogram"])
    )


def test_coverage_aggregation_bounds_sample_memory() -> None:
    agg = CoverageAggregator(
        CoverageAggregationConfig(
            histogram_bins=4,
            max_values_per_metric=3,
        )
    )
    for idx in range(40):
        agg.update_metrics(_metric_fixture(linearity_proxy=float(idx) / 40.0))

    summary = agg.build_summary()
    line = summary["metrics"]["linearity_proxy"]
    assert line["count"] == 40
    assert line["sampled_count"] == 3
    assert line["sampled_fraction"] == pytest.approx(3 / 40)
    assert summary["max_values_per_metric"] == 3


def test_underrepresented_regime_detection_against_target_band() -> None:
    agg = CoverageAggregator(
        CoverageAggregationConfig(
            histogram_bins=4,
            underrepresented_threshold=0.5,
            target_bands={"linearity_proxy": (0.0, 1.0)},
        )
    )
    for _ in range(8):
        agg.update_metrics(_metric_fixture(linearity_proxy=0.0))
    for _ in range(2):
        agg.update_metrics(_metric_fixture(linearity_proxy=1.0))

    summary = agg.build_summary()
    under_bins = summary["metrics"]["linearity_proxy"]["underrepresented_bins"]
    assert len(under_bins) >= 2
    assert summary["metrics"]["linearity_proxy"]["target_band"]["in_target_count"] == 10


def test_target_band_counts_do_not_inflate_for_partial_bin_overlap() -> None:
    agg = CoverageAggregator(
        CoverageAggregationConfig(
            histogram_bins=2,
            target_bands={"linearity_proxy": (0.95, 1.05)},
        )
    )
    agg.update_metrics(_metric_fixture(linearity_proxy=0.9))
    agg.update_metrics(_metric_fixture(linearity_proxy=1.1))

    summary = agg.build_summary()
    line = summary["metrics"]["linearity_proxy"]
    assert line["target_band"]["in_target_count"] == 0
    assert line["target_band"]["in_target_fraction"] == pytest.approx(0.0)


def test_target_band_histogram_uses_target_range_for_coverage() -> None:
    agg = CoverageAggregator(
        CoverageAggregationConfig(
            histogram_bins=10,
            underrepresented_threshold=0.5,
            target_bands={"linearity_proxy": (0.0, 1.0)},
        )
    )
    for value in (0.95, 0.96, 0.98, 0.99, 1.0):
        agg.update_metrics(_metric_fixture(linearity_proxy=value))

    summary = agg.build_summary()
    line = summary["metrics"]["linearity_proxy"]
    # Coverage should reflect sparse occupancy inside the full target range.
    assert line["histogram"]["coverage_ratio"] < 1.0
    assert len(line["underrepresented_bins"]) > 0


def test_generate_no_write_with_coverage_enabled_emits_artifacts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._compute_wins_ratio_proxy",
        lambda **_kwargs: 0.6,
    )
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 16
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    cfg.output.out_dir = str(tmp_path / "run")
    cfg.diagnostics.enabled = True
    cfg.diagnostics.histogram_bins = 8
    cfg.diagnostics.meta_feature_targets = {"linearity_proxy": [0.2, 0.8]}
    config_path = tmp_path / "coverage_enabled.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )
    assert code == 0

    json_path = tmp_path / "run" / "coverage_summary.json"
    md_path = tmp_path / "run" / "coverage_summary.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["num_datasets"] == 1
    assert "mechanism_family_summary" in payload
    assert "linearity_proxy" in payload["metrics"]
    assert "shift_edge_odds_multiplier" in payload["metrics"]
    mechanism_summary = payload["mechanism_family_summary"]
    assert "metadata_coverage_rate" in mechanism_summary
    assert "sampled_family_counts" in mechanism_summary
    assert "sampled_variant_counts" in mechanism_summary
    assert "dataset_presence_rate_by_variant" in mechanism_summary
    markdown = md_path.read_text(encoding="utf-8")
    assert "## Mechanism Families" in markdown
