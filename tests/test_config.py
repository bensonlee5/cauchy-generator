import pytest

from dagzoo.config import (
    MAX_SUPPORTED_CLASS_COUNT,
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MNAR,
    MISSINGNESS_MECHANISM_NONE,
    NOISE_MIXTURE_COMPONENT_GAUSSIAN,
    NOISE_MIXTURE_COMPONENT_LAPLACE,
    NOISE_MIXTURE_COMPONENT_STUDENT_T,
    GeneratorConfig,
)
from dagzoo.io.lineage_schema import validate_metadata_lineage


def test_load_default_config() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    assert cfg.dataset.n_train > 0
    assert cfg.dataset.rows is None
    assert cfg.dataset.n_features_min <= cfg.dataset.n_features_max
    assert cfg.output.shard_size > 0
    assert cfg.diagnostics.enabled is False
    assert cfg.diagnostics.histogram_bins > 0
    assert cfg.diagnostics.quantiles
    assert cfg.diagnostics.underrepresented_threshold >= 0
    assert (
        cfg.diagnostics.max_values_per_metric is None or cfg.diagnostics.max_values_per_metric > 0
    )
    assert cfg.diagnostics.meta_feature_targets == {}
    assert cfg.diagnostics.out_dir is None
    assert cfg.filter.n_estimators > 0
    assert cfg.filter.max_depth >= 0
    assert cfg.filter.max_features == "auto"
    assert cfg.filter.n_jobs == -1
    assert cfg.dataset.missing_rate == 0.0
    assert cfg.dataset.missing_mechanism == MISSINGNESS_MECHANISM_NONE
    assert cfg.dataset.missing_mar_observed_fraction == 0.5
    assert cfg.dataset.missing_mar_logit_scale == 1.0
    assert cfg.dataset.missing_mnar_logit_scale == 1.0


def test_config_package_reexports_noise_mixture_component_constants() -> None:
    assert NOISE_MIXTURE_COMPONENT_GAUSSIAN == "gaussian"
    assert NOISE_MIXTURE_COMPONENT_LAPLACE == "laplace"
    assert NOISE_MIXTURE_COMPONENT_STUDENT_T == "student_t"


def test_default_config_metadata_is_compatible_with_optional_lineage() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    metadata = {
        "seed": int(cfg.seed),
        "config": cfg.to_dict(),
    }
    validate_metadata_lineage(metadata, required=False)


def test_seed_range_accepts_32bit_boundaries() -> None:
    cfg_min = GeneratorConfig.from_dict({"seed": 0})
    cfg_max = GeneratorConfig.from_dict({"seed": 4294967295})
    assert cfg_min.seed == 0
    assert cfg_max.seed == 4294967295


@pytest.mark.parametrize("bad_seed", [-1, 4294967296, True])
def test_seed_range_rejects_out_of_bounds_values(bad_seed: int | bool) -> None:
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
        GeneratorConfig.from_dict({"seed": bad_seed})


def test_class_range_accepts_many_class_envelope_limit() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "dataset": {
                "task": "classification",
                "n_classes_min": 12,
                "n_classes_max": MAX_SUPPORTED_CLASS_COUNT,
            }
        }
    )
    assert cfg.dataset.n_classes_min == 12
    assert cfg.dataset.n_classes_max == MAX_SUPPORTED_CLASS_COUNT


def test_class_range_rejects_values_above_many_class_envelope_limit() -> None:
    with pytest.raises(ValueError, match=r"dataset\.n_classes_max must be an integer in \[2, 32\]"):
        GeneratorConfig.from_dict({"dataset": {"task": "classification", "n_classes_max": 33}})


def test_class_range_rejects_min_greater_than_max() -> None:
    with pytest.raises(ValueError, match=r"dataset\.n_classes_min must be <= n_classes_max"):
        GeneratorConfig.from_dict(
            {"dataset": {"task": "classification", "n_classes_min": 16, "n_classes_max": 12}}
        )


def test_classification_rejects_impossible_class_split_bounds() -> None:
    with pytest.raises(ValueError, match=r"infeasible class/split combination"):
        GeneratorConfig.from_dict(
            {
                "dataset": {
                    "task": "classification",
                    "n_train": 24,
                    "n_test": 24,
                    "n_classes_min": 32,
                    "n_classes_max": 32,
                }
            }
        )


def test_classification_rejects_partial_class_range_when_upper_bound_is_infeasible() -> None:
    with pytest.raises(
        ValueError, match=r"dataset classification split constraints for n_classes_max"
    ):
        GeneratorConfig.from_dict(
            {
                "dataset": {
                    "task": "classification",
                    "n_train": 32,
                    "n_test": 8,
                    "n_classes_min": 2,
                    "n_classes_max": 10,
                }
            }
        )


def test_dataset_rows_accepts_fixed_and_range_and_choices() -> None:
    cfg_fixed = GeneratorConfig.from_dict({"dataset": {"rows": 1024}})
    assert cfg_fixed.dataset.rows is not None
    assert cfg_fixed.dataset.rows.mode == "fixed"
    assert cfg_fixed.dataset.rows.value == 1024

    cfg_range = GeneratorConfig.from_dict({"dataset": {"rows": "400..60000"}})
    assert cfg_range.dataset.rows is not None
    assert cfg_range.dataset.rows.mode == "range"
    assert cfg_range.dataset.rows.start == 400
    assert cfg_range.dataset.rows.stop == 60000

    cfg_choices = GeneratorConfig.from_dict({"dataset": {"rows": [400, 1024, 60000]}})
    assert cfg_choices.dataset.rows is not None
    assert cfg_choices.dataset.rows.mode == "choices"
    assert cfg_choices.dataset.rows.choices == [400, 1024, 60000]


@pytest.mark.parametrize("rows_value", [399, 60001, "399", "60001", "300..600"])
def test_dataset_rows_rejects_out_of_bounds_values(rows_value: object) -> None:
    with pytest.raises(ValueError, match=r"dataset\.rows"):
        GeneratorConfig.from_dict({"dataset": {"rows": rows_value}})


def test_dataset_rows_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match=r"dataset\.rows range start must be <= stop"):
        GeneratorConfig.from_dict({"dataset": {"rows": "2000..400"}})


def test_dataset_rows_rejects_duplicate_choices() -> None:
    with pytest.raises(ValueError, match=r"dataset\.rows must not include duplicate row values"):
        GeneratorConfig.from_dict({"dataset": {"rows": "1024,1024"}})


def test_dataset_rows_classification_checks_min_realizable_train_rows() -> None:
    with pytest.raises(
        ValueError, match=r"dataset classification split constraints for n_classes_max"
    ):
        GeneratorConfig.from_dict(
            {
                "dataset": {
                    "task": "classification",
                    "rows": "400..600",
                    "n_test": 380,
                    "n_classes_min": 2,
                    "n_classes_max": 32,
                }
            }
        )


def test_regression_allows_class_fields_without_split_feasibility_check() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "dataset": {
                "task": "regression",
                "n_train": 24,
                "n_test": 24,
                "n_classes_min": 2,
                "n_classes_max": 32,
            }
        }
    )
    assert cfg.dataset.n_classes_max == 32


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), float("nan"), True])
def test_categorical_ratio_min_bounds_are_validated(value: float | bool) -> None:
    with pytest.raises(
        ValueError, match="dataset.categorical_ratio_min must be a finite value in \\[0, 1\\]"
    ):
        GeneratorConfig.from_dict({"dataset": {"categorical_ratio_min": value}})


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), float("nan"), True])
def test_categorical_ratio_max_bounds_are_validated(value: float | bool) -> None:
    with pytest.raises(
        ValueError, match="dataset.categorical_ratio_max must be a finite value in \\[0, 1\\]"
    ):
        GeneratorConfig.from_dict({"dataset": {"categorical_ratio_max": value}})


def test_categorical_ratio_bounds_accept_endpoints() -> None:
    cfg = GeneratorConfig.from_dict(
        {"dataset": {"categorical_ratio_min": 0.0, "categorical_ratio_max": 1.0}}
    )
    assert cfg.dataset.categorical_ratio_min == pytest.approx(0.0)
    assert cfg.dataset.categorical_ratio_max == pytest.approx(1.0)


def test_load_cuda_presets() -> None:
    cfg_h100 = GeneratorConfig.from_yaml("configs/preset_cuda_h100.yaml")
    assert cfg_h100.runtime.device == "cuda"
    assert cfg_h100.dataset.n_features_max >= 128
    assert cfg_h100.runtime.fixed_layout_target_cells == 32_000_000


def test_load_diagnostics_preset() -> None:
    cfg_diag = GeneratorConfig.from_yaml("configs/preset_diagnostics_on.yaml")
    assert cfg_diag.diagnostics.enabled is True
    assert cfg_diag.diagnostics.histogram_bins >= 8
    assert cfg_diag.diagnostics.quantiles


def test_load_missingness_presets() -> None:
    cfg_mcar = GeneratorConfig.from_yaml("configs/preset_missingness_mcar.yaml")
    assert cfg_mcar.dataset.missing_mechanism == MISSINGNESS_MECHANISM_MCAR
    assert cfg_mcar.dataset.missing_rate > 0.0

    cfg_mar = GeneratorConfig.from_yaml("configs/preset_missingness_mar.yaml")
    assert cfg_mar.dataset.missing_mechanism == MISSINGNESS_MECHANISM_MAR
    assert cfg_mar.dataset.missing_rate > 0.0
    assert cfg_mar.dataset.missing_mar_observed_fraction > 0.0

    cfg_mnar = GeneratorConfig.from_yaml("configs/preset_missingness_mnar.yaml")
    assert cfg_mnar.dataset.missing_mechanism == MISSINGNESS_MECHANISM_MNAR
    assert cfg_mnar.dataset.missing_rate > 0.0
    assert cfg_mnar.dataset.missing_mnar_logit_scale > 0.0


def test_load_benchmark_profiles() -> None:
    cfg_cpu = GeneratorConfig.from_yaml("configs/benchmark_cpu.yaml")
    cfg_desktop = GeneratorConfig.from_yaml("configs/benchmark_cuda_desktop.yaml")
    cfg_h100 = GeneratorConfig.from_yaml("configs/benchmark_cuda_h100.yaml")
    cfg_h100_large = GeneratorConfig.from_yaml("configs/benchmark_cuda_h100_large_shape.yaml")
    cfg_h100_saturation = GeneratorConfig.from_yaml("configs/benchmark_cuda_h100_saturation.yaml")

    assert cfg_cpu.runtime.device == "cpu"
    assert cfg_desktop.runtime.device == "cuda"
    assert cfg_h100.runtime.device == "cuda"
    assert cfg_h100_large.runtime.device == "cuda"
    assert cfg_h100_saturation.runtime.device == "cuda"
    assert cfg_cpu.runtime.fixed_layout_target_cells == 12_000_000
    assert cfg_desktop.runtime.fixed_layout_target_cells == 16_000_000
    assert cfg_h100.runtime.fixed_layout_target_cells == 32_000_000
    assert cfg_h100_large.runtime.fixed_layout_target_cells == 32_000_000
    assert cfg_h100_saturation.runtime.fixed_layout_target_cells == 32_000_000
    assert cfg_h100_large.dataset.n_train == 8192
    assert cfg_h100_large.dataset.n_test == 2048
    assert cfg_h100_large.dataset.n_features_max == 256
    assert cfg_h100_saturation.benchmark.num_datasets == 1500
    assert cfg_h100_saturation.benchmark.warmup_datasets == 25
    assert "cpu" in cfg_h100.benchmark.presets


def test_load_lineage_benchmark_smoke_preset() -> None:
    cfg = GeneratorConfig.from_yaml("configs/preset_lineage_benchmark_smoke.yaml")
    assert cfg.benchmark.suite == "smoke"
    assert cfg.benchmark.preset_name == "lineage_smoke"
    assert "lineage_smoke" in cfg.benchmark.presets
    assert cfg.benchmark.latency_num_samples >= 5


def test_runtime_config_from_dict() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "runtime": {
                "device": "cpu",
                "torch_dtype": "float64",
                "fixed_layout_target_cells": "8000000",
            },
            "diagnostics": {
                "enabled": True,
                "histogram_bins": 12,
                "max_values_per_metric": 1234,
                "meta_feature_targets": {
                    "linearity_proxy": [0.2, 0.8],
                },
            },
        }
    )
    assert cfg.runtime.device == "cpu"
    assert cfg.runtime.torch_dtype == "float64"
    assert cfg.runtime.fixed_layout_target_cells == 8_000_000
    assert cfg.diagnostics.enabled is True
    assert cfg.diagnostics.histogram_bins == 12
    assert cfg.diagnostics.max_values_per_metric == 1234
    assert "linearity_proxy" in cfg.diagnostics.meta_feature_targets


def test_runtime_config_rejects_removed_parallel_generation_keys() -> None:
    with pytest.raises(ValueError, match=r"runtime\.worker_count, runtime\.worker_index"):
        GeneratorConfig.from_dict({"runtime": {"worker_count": 4, "worker_index": 2}})


def test_runtime_config_rejects_generation_engine_key() -> None:
    with pytest.raises(TypeError, match="generation_engine"):
        GeneratorConfig.from_dict({"runtime": {"generation_engine": "appendix_light"}})


def test_runtime_config_rejects_hardware_aware_key() -> None:
    with pytest.raises(TypeError, match="hardware_aware"):
        GeneratorConfig.from_dict({"runtime": {"hardware_aware": True}})


@pytest.mark.parametrize("value", [0, -1, True, "abc"])
def test_runtime_config_rejects_invalid_fixed_layout_target_cells(value: object) -> None:
    with pytest.raises(ValueError, match=r"runtime\.fixed_layout_target_cells must"):
        GeneratorConfig.from_dict({"runtime": {"fixed_layout_target_cells": value}})


def test_legacy_filter_keys_are_rejected() -> None:
    with pytest.raises(TypeError, match="n_trees"):
        GeneratorConfig.from_dict(
            {
                "filter": {
                    "enabled": True,
                    "n_trees": 25,
                    "depth": 6,
                    "n_split_candidates": 8,
                }
            }
        )


def test_legacy_graph_log2_keys_are_rejected() -> None:
    with pytest.raises(TypeError, match="n_nodes_log2_min"):
        GeneratorConfig.from_dict({"graph": {"n_nodes_log2_min": 2, "n_nodes_log2_max": 8}})


def test_filter_n_jobs_accepts_minus_one_and_positive_values() -> None:
    cfg_auto = GeneratorConfig.from_dict({"filter": {"n_jobs": -1}})
    cfg_single = GeneratorConfig.from_dict({"filter": {"n_jobs": 1}})
    cfg_multi = GeneratorConfig.from_dict({"filter": {"n_jobs": 8}})

    assert cfg_auto.filter.n_jobs == -1
    assert cfg_single.filter.n_jobs == 1
    assert cfg_multi.filter.n_jobs == 8


@pytest.mark.parametrize("value", [0, -2, True])
def test_filter_n_jobs_rejects_invalid_values(value: int | bool) -> None:
    with pytest.raises(ValueError, match=r"filter\.n_jobs must"):
        GeneratorConfig.from_dict({"filter": {"n_jobs": value}})


def test_validate_generation_constraints_revalidates_mutated_missingness_state() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.missing_rate = 0.2
    cfg.dataset.missing_mechanism = "none"

    with pytest.raises(
        ValueError,
        match="dataset.missing_mechanism must be mcar, mar, or mnar when dataset.missing_rate > 0",
    ):
        cfg.validate_generation_constraints()


def test_validate_generation_constraints_normalizes_mutated_runtime_device_null_to_auto() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = None  # type: ignore[assignment]

    cfg.validate_generation_constraints()

    assert cfg.runtime.device == "auto"


def test_missingness_mechanism_normalization_is_case_insensitive() -> None:
    cfg = GeneratorConfig.from_dict(
        {"dataset": {"missing_rate": 0.25, "missing_mechanism": "MCAR"}}
    )
    assert cfg.dataset.missing_mechanism == MISSINGNESS_MECHANISM_MCAR


def test_missingness_rejects_none_mechanism_when_rate_is_positive() -> None:
    with pytest.raises(
        ValueError,
        match="dataset.missing_mechanism must be mcar, mar, or mnar when dataset.missing_rate > 0",
    ):
        GeneratorConfig.from_dict({"dataset": {"missing_rate": 0.1, "missing_mechanism": "none"}})


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), float("nan"), True])
def test_missing_rate_bounds_are_validated(value: float | bool) -> None:
    with pytest.raises(
        ValueError, match="dataset.missing_rate must be a finite value in \\[0, 1\\]"
    ):
        GeneratorConfig.from_dict({"dataset": {"missing_rate": value}})


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, float("inf"), float("nan"), True])
def test_missing_mar_observed_fraction_bounds_are_validated(value: float | bool) -> None:
    with pytest.raises(
        ValueError, match="dataset.missing_mar_observed_fraction must be in \\(0, 1\\]"
    ):
        GeneratorConfig.from_dict({"dataset": {"missing_mar_observed_fraction": value}})


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("missing_mar_logit_scale", 0.0),
        ("missing_mar_logit_scale", -1.0),
        ("missing_mar_logit_scale", float("inf")),
        ("missing_mar_logit_scale", float("nan")),
        ("missing_mar_logit_scale", True),
        ("missing_mnar_logit_scale", 0.0),
        ("missing_mnar_logit_scale", -1.0),
        ("missing_mnar_logit_scale", float("inf")),
        ("missing_mnar_logit_scale", float("nan")),
        ("missing_mnar_logit_scale", True),
    ],
)
def test_missing_logit_scale_bounds_are_validated(field_name: str, bad_value: float | bool) -> None:
    with pytest.raises(ValueError, match=rf"dataset.{field_name} must be a finite value > 0"):
        GeneratorConfig.from_dict({"dataset": {field_name: bad_value}})


def test_invalid_missing_mechanism_string_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported missing_mechanism"):
        GeneratorConfig.from_dict({"dataset": {"missing_mechanism": "garbage"}})


def test_unused_missingness_parameters_are_allowed_with_disabled_mechanism() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "dataset": {
                "missing_rate": 0.0,
                "missing_mechanism": "none",
                "missing_mar_observed_fraction": 0.8,
                "missing_mar_logit_scale": 2.5,
                "missing_mnar_logit_scale": 3.5,
            }
        }
    )
    assert cfg.dataset.missing_mechanism == MISSINGNESS_MECHANISM_NONE
    assert cfg.dataset.missing_mar_observed_fraction == 0.8
    assert cfg.dataset.missing_mar_logit_scale == 2.5
    assert cfg.dataset.missing_mnar_logit_scale == 3.5


def test_shift_defaults_are_backward_compatible_when_shift_block_is_absent() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    assert cfg.shift.enabled is False
    assert cfg.shift.mode == "off"
    assert cfg.shift.graph_scale is None
    assert cfg.shift.mechanism_scale is None
    assert cfg.shift.variance_scale is None


@pytest.mark.parametrize("config_path", ("configs/default.yaml", "configs/benchmark_cpu.yaml"))
def test_existing_config_files_parse_with_shift_schema(config_path: str) -> None:
    cfg = GeneratorConfig.from_yaml(config_path)
    assert cfg.shift.enabled is False
    assert cfg.shift.mode == "off"


def test_shift_schema_accepts_profile_with_optional_override() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "shift": {
                "enabled": True,
                "mode": "Graph_Drift",
                "graph_scale": "0.75",
            }
        }
    )
    assert cfg.shift.enabled is True
    assert cfg.shift.mode == "graph_drift"
    assert cfg.shift.graph_scale == 0.75
    assert cfg.shift.mechanism_scale is None
    assert cfg.shift.variance_scale is None


def test_shift_schema_accepts_mixed_profile_with_multiple_overrides() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "shift": {
                "enabled": True,
                "mode": "mixed",
                "graph_scale": 0.2,
                "mechanism_scale": 0.4,
                "variance_scale": 0.6,
            }
        }
    )
    assert cfg.shift.mode == "mixed"
    assert cfg.shift.graph_scale == 0.2
    assert cfg.shift.mechanism_scale == 0.4
    assert cfg.shift.variance_scale == 0.6


def test_shift_schema_accepts_custom_profile_with_at_least_one_override() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "shift": {
                "enabled": True,
                "mode": "custom",
                "mechanism_scale": 0.35,
            }
        }
    )
    assert cfg.shift.mode == "custom"
    assert cfg.shift.mechanism_scale == 0.35


def test_shift_schema_rejects_invalid_profile_value() -> None:
    with pytest.raises(ValueError, match="Unsupported shift.mode"):
        GeneratorConfig.from_dict({"shift": {"enabled": True, "mode": "unknown_mode"}})


def test_shift_schema_rejects_boolean_false_profile_alias() -> None:
    with pytest.raises(ValueError, match="Unsupported shift.mode"):
        GeneratorConfig.from_dict({"shift": {"enabled": False, "mode": False}})


def test_shift_schema_rejects_boolean_true_profile_alias() -> None:
    with pytest.raises(ValueError, match="Unsupported shift.mode"):
        GeneratorConfig.from_dict({"shift": {"enabled": False, "mode": True}})


def test_shift_schema_from_yaml_rejects_unquoted_off_profile(tmp_path) -> None:
    cfg_path = tmp_path / "shift_off.yaml"
    cfg_path.write_text(
        "shift:\n  enabled: false\n  mode: off\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported shift.mode"):
        GeneratorConfig.from_yaml(cfg_path)


def test_shift_schema_from_yaml_accepts_quoted_off_profile(tmp_path) -> None:
    cfg_path = tmp_path / "shift_off_quoted.yaml"
    cfg_path.write_text(
        "shift:\n  enabled: false\n  mode: 'off'\n",
        encoding="utf-8",
    )
    cfg = GeneratorConfig.from_yaml(cfg_path)
    assert cfg.shift.enabled is False
    assert cfg.shift.mode == "off"


def test_shift_schema_rejects_non_boolean_enabled_value() -> None:
    with pytest.raises(ValueError, match="shift.enabled must be a boolean"):
        GeneratorConfig.from_dict({"shift": {"enabled": "true", "mode": "off"}})


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("graph_scale", -0.1),
        ("graph_scale", 1.1),
        ("graph_scale", float("inf")),
        ("graph_scale", float("nan")),
        ("graph_scale", True),
        ("mechanism_scale", -0.1),
        ("mechanism_scale", 1.1),
        ("mechanism_scale", float("inf")),
        ("mechanism_scale", float("nan")),
        ("mechanism_scale", True),
        ("variance_scale", -0.1),
        ("variance_scale", 1.1),
        ("variance_scale", float("inf")),
        ("variance_scale", float("nan")),
        ("variance_scale", True),
    ],
)
def test_shift_override_scales_enforce_finite_unit_interval_bounds(
    field_name: str, bad_value: float | bool
) -> None:
    with pytest.raises(
        ValueError, match=rf"shift\.{field_name} must be a finite value in \[0, 1\]"
    ):
        GeneratorConfig.from_dict(
            {
                "shift": {
                    "enabled": True,
                    "mode": "mixed",
                    field_name: bad_value,
                }
            }
        )


def test_shift_schema_rejects_non_off_profile_when_disabled() -> None:
    with pytest.raises(ValueError, match=r"shift\.mode must be 'off' when shift\.enabled is false"):
        GeneratorConfig.from_dict({"shift": {"enabled": False, "mode": "mixed"}})


def test_shift_schema_rejects_overrides_when_disabled() -> None:
    with pytest.raises(
        ValueError, match=r"shift override scales must be unset when shift\.enabled is false"
    ):
        GeneratorConfig.from_dict(
            {
                "shift": {
                    "enabled": False,
                    "mode": "off",
                    "graph_scale": 0.2,
                }
            }
        )


def test_shift_schema_rejects_off_profile_when_enabled() -> None:
    with pytest.raises(
        ValueError, match=r"shift\.mode must not be 'off' when shift\.enabled is true"
    ):
        GeneratorConfig.from_dict({"shift": {"enabled": True, "mode": "off"}})


def test_shift_schema_requires_custom_profile_to_set_at_least_one_override() -> None:
    with pytest.raises(
        ValueError, match=r"shift\.mode 'custom' requires at least one override scale"
    ):
        GeneratorConfig.from_dict({"shift": {"enabled": True, "mode": "custom"}})


@pytest.mark.parametrize(
    ("profile", "payload", "error_pattern"),
    [
        (
            "graph_drift",
            {"variance_scale": 0.25},
            r"shift\.mode 'graph_drift' only allows shift\.graph_scale override",
        ),
        (
            "mechanism_drift",
            {"graph_scale": 0.25},
            r"shift\.mode 'mechanism_drift' only allows shift\.mechanism_scale override",
        ),
        (
            "noise_drift",
            {"mechanism_scale": 0.25},
            r"shift\.mode 'noise_drift' only allows shift\.variance_scale override",
        ),
    ],
)
def test_shift_schema_rejects_incompatible_profile_override_combinations(
    profile: str,
    payload: dict[str, float],
    error_pattern: str,
) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        GeneratorConfig.from_dict(
            {
                "shift": {
                    "enabled": True,
                    "mode": profile,
                    **payload,
                }
            }
        )
