from __future__ import annotations

import json
from dataclasses import replace

import pytest
import yaml

from dagzoo import generate_batch
from dagzoo.cli.entrypoint import main
from dagzoo.config import DiagnosticsConfig, GeneratorConfig
from dagzoo.core.steering import SteeringResolution
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


def _steering_fixture_config(out_dir: str | None = None) -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 16
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 4
    cfg.graph.n_nodes_max = 4
    if out_dir is not None:
        cfg.output.out_dir = out_dir
    cfg.diagnostics.enabled = True
    cfg.diagnostics.histogram_bins = 8
    cfg.steering.enabled = True
    cfg.steering.preset = "anti_memorization_piecewise_v1"
    cfg.validate_generation_constraints()
    return cfg


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
    assert "steering" in payload
    assert payload["steering"]["enabled"] is False

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


def test_coverage_steering_helper_fallbacks_handle_invalid_inputs() -> None:
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
            underrepresented_threshold=0.25,
            max_values_per_metric=11,
            meta_feature_targets={"linearity_proxy": [0.2, 0.8]},
        )
    )

    assert aggregation_config.include_spearman is True
    assert aggregation_config.histogram_bins == 7
    assert aggregation_config.quantiles == (0.1, 0.9)
    assert aggregation_config.underrepresented_threshold == pytest.approx(0.25)
    assert aggregation_config.max_values_per_metric == 11
    assert aggregation_config.target_bands == {"linearity_proxy": (0.2, 0.8)}
    assert aggregation_config.steering_config is None


def test_coverage_summary_includes_dynamic_steering_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._compute_wins_ratio_proxy",
        lambda **_kwargs: 0.6,
    )
    cfg = _steering_fixture_config()
    agg = CoverageAggregator(build_diagnostics_aggregation_config(cfg))

    for bundle in generate_batch(cfg, num_datasets=5, seed=1234, device="cpu"):
        agg.update_bundle(bundle)

    steering = agg.build_summary()["steering"]
    assert steering["enabled"] is True
    assert steering["authoring_form"] == "preset"
    assert steering["preset"] == "anti_memorization_piecewise_v1"
    assert steering["stage_count"] == 4
    assert steering["resolution_checks"]["datasets_checked"] == 5
    assert steering["resolution_checks"]["datasets_mismatched"] == 0
    assert steering["resolution_checks"]["match_rate"] == pytest.approx(1.0)
    assert [stage["name"] for stage in steering["stages"]] == [
        "missingness_ramp",
        "graph_excursion_out",
        "graph_to_noise_handoff",
        "mixture_noise_ramp",
    ]
    assert [stage["dataset_count"] for stage in steering["stages"]] == [1, 1, 1, 2]
    assert steering["stages"][0]["requested"]["dataset"]["missing_mechanism"] == "mcar"
    assert steering["stages"][2]["realized"]["shift_mode_counts"]["mixed"] == 1
    assert steering["stages"][3]["realized"]["noise_family_requested_counts"]["mixture"] == 2
    assert "linearity_proxy" in steering["stages"][3]["metrics"]


def test_coverage_summary_steering_analysis_is_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._compute_wins_ratio_proxy",
        lambda **_kwargs: 0.6,
    )
    cfg = _steering_fixture_config()
    agg_a = CoverageAggregator(build_diagnostics_aggregation_config(cfg))
    agg_b = CoverageAggregator(build_diagnostics_aggregation_config(cfg))

    for bundle in generate_batch(cfg, num_datasets=5, seed=4321, device="cpu"):
        agg_a.update_bundle(bundle)
    for bundle in generate_batch(cfg, num_datasets=5, seed=4321, device="cpu"):
        agg_b.update_bundle(bundle)

    assert agg_a.build_summary()["steering"] == agg_b.build_summary()["steering"]


def test_coverage_summary_steering_guards_skip_invalid_resolution_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _steering_fixture_config()
    agg = CoverageAggregator(build_diagnostics_aggregation_config(cfg))
    metrics = _metric_fixture()

    agg._update_steering(
        DatasetBundle(None, None, None, None, [], metadata={"dataset_index": 0}),
        metrics,
    )

    def _out_of_range_resolution(*_args, **_kwargs) -> SteeringResolution:
        return SteeringResolution(
            config=cfg,
            dataset_index=0,
            run_num_datasets=5,
            progress=0.0,
            stage_index=99,
            stage_name="invalid",
            stage_progress=0.0,
        )

    monkeypatch.setattr("dagzoo.diagnostics.coverage.resolve_steering", _out_of_range_resolution)
    agg._update_steering(
        DatasetBundle(
            None,
            None,
            None,
            None,
            [],
            metadata={"dataset_index": 0, "run_num_datasets": 5},
        ),
        metrics,
    )

    resolution_checks = agg.build_summary()["steering"]["resolution_checks"]
    assert resolution_checks["datasets_checked"] == 0
    assert resolution_checks["datasets_mismatched"] == 0


def test_coverage_summary_steering_records_resolution_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._compute_wins_ratio_proxy",
        lambda **_kwargs: 0.6,
    )
    cfg = _steering_fixture_config()
    agg = CoverageAggregator(build_diagnostics_aggregation_config(cfg))

    bundle = next(iter(generate_batch(cfg, num_datasets=5, seed=1234, device="cpu")))
    bundle.metadata["config"]["dataset"]["missing_rate"] = 0.99
    bundle.metadata["config"]["dataset"]["missing_mechanism"] = "mnar"
    bundle.metadata["shift"] = {
        "mode": "noise_drift",
        "graph_scale": 9.0,
        "variance_scale": 8.0,
        "mechanism_logit_tilt": 7.0,
    }
    bundle.metadata["noise_distribution"]["family_requested"] = "laplace"
    bundle.metadata["noise_distribution"]["mixture_weights"] = {"gaussian": 1.0}

    agg.update_bundle(bundle)

    resolution_checks = agg.build_summary()["steering"]["resolution_checks"]
    assert resolution_checks["datasets_checked"] == 1
    assert resolution_checks["datasets_matching"] == 0
    assert resolution_checks["datasets_mismatched"] == 1
    assert resolution_checks["mismatched_dataset_indices"] == [bundle.metadata["dataset_index"]]
    assert resolution_checks["mismatch_counts"] == {
        "config.dataset.missing_mechanism": 1,
        "config.dataset.missing_rate": 1,
        "metadata.noise_distribution.family_requested": 1,
        "metadata.noise_distribution.mixture_weights": 1,
        "metadata.shift.graph_scale": 1,
        "metadata.shift.mechanism_logit_tilt": 1,
        "metadata.shift.mode": 1,
        "metadata.shift.variance_scale": 1,
    }


def test_coverage_summary_steering_empty_stages_omit_metric_payloads() -> None:
    cfg = _steering_fixture_config()
    steering = CoverageAggregator(build_diagnostics_aggregation_config(cfg)).build_summary()[
        "steering"
    ]

    assert steering["enabled"] is True
    assert steering["stage_count"] == 4
    assert len(steering["stages"]) == 4
    assert all(stage["metrics"] == {} for stage in steering["stages"])


def test_write_coverage_summary_markdown_handles_invalid_steering_payloads(tmp_path) -> None:
    summary = {
        "generated_at": "2026-03-25T00:00:00Z",
        "num_datasets": 1,
        "histogram_bins": 8,
        "quantiles": [0.5],
        "max_values_per_metric": 10,
        "mechanism_family_summary": {},
        "steering": {
            "enabled": True,
            "authoring_form": "preset",
            "preset": None,
            "stage_count": 1,
            "resolution_checks": "bad",
            "stages": [
                "skip-me",
                {
                    "name": "stage-a",
                    "fraction": 0.5,
                    "dataset_count": 1,
                    "dataset_index_range": "bad",
                    "progress_range": "bad",
                    "realized": "bad",
                },
            ],
        },
        "metrics": {},
    }

    markdown = write_coverage_summary_markdown(summary, tmp_path / "coverage_summary.md").read_text(
        encoding="utf-8"
    )

    assert "## Steering" in markdown
    assert "- Datasets checked: `-`" in markdown
    assert "skip-me" not in markdown
    assert "| stage-a | 0.500 | 1 | - | - | - | - | - |" in markdown


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
    assert payload["steering"]["enabled"] is False
    assert "linearity_proxy" in payload["metrics"]
    assert "shift_edge_odds_multiplier" in payload["metrics"]
    mechanism_summary = payload["mechanism_family_summary"]
    assert "metadata_coverage_rate" in mechanism_summary
    assert "sampled_family_counts" in mechanism_summary
    assert "sampled_variant_counts" in mechanism_summary
    assert "dataset_presence_rate_by_variant" in mechanism_summary
    markdown = md_path.read_text(encoding="utf-8")
    assert "## Steering" in markdown
    assert "## Mechanism Families" in markdown


def test_generate_no_write_with_dynamic_steering_emits_steering_artifacts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._compute_wins_ratio_proxy",
        lambda **_kwargs: 0.6,
    )
    cfg = _steering_fixture_config(out_dir=str(tmp_path / "run"))
    config_path = tmp_path / "coverage_steering_enabled.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "5",
            "--seed",
            "1234",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )
    assert code == 0

    payload = json.loads((tmp_path / "run" / "coverage_summary.json").read_text(encoding="utf-8"))
    steering = payload["steering"]
    assert steering["enabled"] is True
    assert steering["resolution_checks"]["datasets_mismatched"] == 0
    assert [stage["name"] for stage in steering["stages"]] == [
        "missingness_ramp",
        "graph_excursion_out",
        "graph_to_noise_handoff",
        "mixture_noise_ramp",
    ]
    markdown = (tmp_path / "run" / "coverage_summary.md").read_text(encoding="utf-8")
    assert "## Steering" in markdown
    assert "missingness_ramp" in markdown
