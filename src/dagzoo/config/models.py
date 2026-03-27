"""Typed configuration models and file loading."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from dagzoo.rng import SEED32_MAX, SEED32_MIN

from .constants import (
    _NOISE_MIXTURE_COMPONENT_VALUE_MAP,
    _PRODUCT_COMPONENT_FAMILIES,
    MAX_SUPPORTED_CLASS_COUNT,
    MISSINGNESS_MECHANISM_NONE,
    NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_MIXTURE,
    SHIFT_MODE_CUSTOM,
    SHIFT_MODE_GRAPH_DRIFT,
    SHIFT_MODE_MECHANISM_DRIFT,
    SHIFT_MODE_MIXED,
    SHIFT_MODE_NOISE_DRIFT,
    SHIFT_MODE_OFF,
    MechanismFamily,
    MissingnessMechanism,
    NoiseFamily,
    NoiseMixtureComponent,
    ShiftMode,
    _SectionT,
)
from .normalization import (
    _normalize_function_family_mix,
    _normalize_noise_mixture_weights,
    normalize_missing_mechanism,
    normalize_noise_family,
    normalize_shift_mode,
)
from .rows import (
    DatasetRowsSpec,
    dataset_rows_bounds,
    normalize_dataset_rows,
    validate_class_split_feasibility,
)
from .scalars import (
    _validate_finite_float_field,
    _validate_int_field,
    _validate_min_max_pair,
    _validate_optional_finite_float_field,
)

_STEERING_PRESET_ANTI_MEMORIZATION_PIECEWISE_V1 = "anti_memorization_piecewise_v1"
_STEERING_FRACTION_TOLERANCE = 1e-6
_STRESS_PROFILE_ANTI_MEMORIZATION_PIECEWISE_CLASSIFICATION_SLICE_V1 = (
    "anti_memorization_piecewise_classification_slice_v1"
)
_STEERING_PRESET_DEFINITIONS: dict[str, dict[str, Any]] = {
    _STEERING_PRESET_ANTI_MEMORIZATION_PIECEWISE_V1: {
        "stages": [
            {
                "name": "missingness_ramp",
                "fraction": 0.25,
                "dataset": {
                    "missing_rate": [0.0, 0.25],
                    "missing_mechanism": "mcar",
                },
            },
            {
                "name": "graph_excursion_out",
                "fraction": 0.25,
                "shift": {
                    "mode": "graph_drift",
                    "graph_scale": [0.0, 0.5],
                },
            },
            {
                "name": "graph_to_noise_handoff",
                "fraction": 0.25,
                "shift": {
                    "mode": "mixed",
                    "graph_scale": [0.5, 0.0],
                    "variance_scale": [0.0, 0.5],
                },
            },
            {
                "name": "mixture_noise_ramp",
                "fraction": 0.25,
                "noise": {
                    "family": "mixture",
                    "student_t_df": 6.0,
                    "mixture_weights": {
                        "gaussian": [1.0, 0.5],
                        "laplace": [0.0, 0.3],
                        "student_t": [0.0, 0.2],
                    },
                },
            },
        ],
    }
}
_STRESS_PROFILE_DEFINITIONS: dict[str, dict[str, Any]] = {
    _STRESS_PROFILE_ANTI_MEMORIZATION_PIECEWISE_CLASSIFICATION_SLICE_V1: {
        "dataset": {
            "task": "classification",
            "n_train": 768,
            "n_test": 256,
            "rows": None,
            "n_features_min": 16,
            "n_features_max": 64,
            "n_classes_min": 2,
            "n_classes_max": 10,
            "categorical_ratio_min": 0.0,
            "categorical_ratio_max": 1.0,
            "max_categorical_cardinality": 9,
        },
        "graph": {
            "n_nodes_min": 2,
            "n_nodes_max": 32,
        },
        "mechanism": {
            "function_family_mix": None,
        },
        "steering": {
            "enabled": True,
            "preset": _STEERING_PRESET_ANTI_MEMORIZATION_PIECEWISE_V1,
            "stages": [],
        },
    }
}
_STRESS_LOCKED_PATHS: tuple[str, ...] = (
    "dataset.task",
    "dataset.n_train",
    "dataset.n_test",
    "dataset.rows",
    "dataset.n_features_min",
    "dataset.n_features_max",
    "dataset.n_classes_min",
    "dataset.n_classes_max",
    "dataset.categorical_ratio_min",
    "dataset.categorical_ratio_max",
    "dataset.max_categorical_cardinality",
    "dataset.missing_rate",
    "dataset.missing_mechanism",
    "dataset.missing_mar_observed_fraction",
    "dataset.missing_mar_logit_scale",
    "dataset.missing_mnar_logit_scale",
    "graph.n_nodes_min",
    "graph.n_nodes_max",
    "mechanism.function_family_mix",
    "shift.enabled",
    "shift.mode",
    "shift.graph_scale",
    "shift.mechanism_scale",
    "shift.variance_scale",
    "noise.family",
    "noise.base_scale",
    "noise.student_t_df",
    "noise.mixture_weights",
    "steering.enabled",
    "steering.preset",
    "steering.stages",
)


def steering_preset_definition(name: str) -> dict[str, Any]:
    """Return a deep copy of one built-in steering preset definition."""

    try:
        return copy.deepcopy(_STEERING_PRESET_DEFINITIONS[str(name)])
    except KeyError as exc:
        supported = ", ".join(sorted(_STEERING_PRESET_DEFINITIONS))
        raise ValueError(
            f"Unsupported steering.preset {name!r}. Expected one of: {supported}."
        ) from exc


def stress_profile_definition(name: str) -> dict[str, Any]:
    """Return a deep copy of one built-in stress-profile definition."""

    try:
        return copy.deepcopy(_STRESS_PROFILE_DEFINITIONS[str(name)])
    except KeyError as exc:
        supported = ", ".join(sorted(_STRESS_PROFILE_DEFINITIONS))
        raise ValueError(
            f"Unsupported stress.profile {name!r}. Expected one of: {supported}."
        ) from exc


def _normalize_stress_profile(value: object) -> str | None:
    """Normalize an optional stress profile name."""

    if value is None:
        return None
    if not isinstance(value, str):
        supported = ", ".join(sorted(_STRESS_PROFILE_DEFINITIONS))
        raise ValueError(f"Unsupported stress.profile {value!r}. Expected one of: {supported}.")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("stress.profile must be a non-empty string when provided.")
    if normalized not in _STRESS_PROFILE_DEFINITIONS:
        supported = ", ".join(sorted(_STRESS_PROFILE_DEFINITIONS))
        raise ValueError(f"Unsupported stress.profile {value!r}. Expected one of: {supported}.")
    return normalized


def _normalize_stress_fields(stress: "StressConfig") -> None:
    """Stage 1: normalize the stress section."""

    stress.profile = _normalize_stress_profile(stress.profile)


def _get_config_path_value(config: object, path: str) -> object:
    current = config
    for component in path.split("."):
        current = getattr(current, component)
    return current


def _set_config_path_value(config: object, path: str, value: object) -> None:
    components = path.split(".")
    current = config
    for component in components[:-1]:
        current = getattr(current, component)
    setattr(current, components[-1], copy.deepcopy(value))


def _apply_stress_profile_patch(config: "GeneratorConfig", *, profile: str) -> None:
    """Apply one built-in stress profile onto an existing config."""

    patch = stress_profile_definition(profile)
    for section_name, section_patch in patch.items():
        if not isinstance(section_patch, dict):
            raise TypeError(
                f"stress profile {profile!r} section {section_name!r} must be a mapping."
            )
        section = getattr(config, section_name)
        for field_name, field_value in section_patch.items():
            setattr(section, field_name, copy.deepcopy(field_value))


def _expected_stress_profile_config(profile: str) -> "GeneratorConfig":
    """Return the canonical resolved config envelope for one stress profile."""

    expected = GeneratorConfig()
    _apply_stress_profile_patch(expected, profile=profile)
    expected.validate_generation_constraints()
    return expected


def _format_stress_locked_value(value: object) -> str:
    if value is None:
        return "unset"
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def _stage2_validate_stress_constraints(config: "GeneratorConfig") -> None:
    """Stage 2: validate stress-profile identity locking before materialization."""

    profile = config.stress.profile
    if profile is None:
        return

    baseline = GeneratorConfig()
    expected = _expected_stress_profile_config(profile)
    for path in _STRESS_LOCKED_PATHS:
        actual_value = _get_config_path_value(config, path)
        baseline_value = _get_config_path_value(baseline, path)
        expected_value = _get_config_path_value(expected, path)
        if actual_value != expected_value and actual_value != baseline_value:
            raise ValueError(
                f"stress.profile {profile!r} locks {path} to "
                f"{_format_stress_locked_value(expected_value)}, got "
                f"{_format_stress_locked_value(actual_value)}."
            )


def materialize_stress_profile(
    config: "GeneratorConfig",
    *,
    revalidate: bool,
) -> "GeneratorConfig":
    """Return a config clone with the selected stress profile applied."""

    effective = copy.deepcopy(config)
    profile = effective.stress.profile
    if profile is None:
        if revalidate:
            effective.validate_generation_constraints()
        return effective
    _apply_stress_profile_patch(effective, profile=profile)
    if revalidate:
        effective.validate_generation_constraints()
    return effective


def _normalize_dataset_fields(dataset: DatasetConfig) -> None:
    """Stage 1: normalize individual dataset fields and scalar bounds."""

    normalized_task = str(dataset.task).strip().lower()
    if normalized_task not in {"classification", "regression"}:
        raise ValueError(
            f"dataset.task must be 'classification' or 'regression', got {dataset.task!r}."
        )
    dataset.task = normalized_task

    dataset.n_train = _validate_int_field(
        field_name="dataset.n_train",
        value=dataset.n_train,
        minimum=1,
    )
    dataset.n_test = _validate_int_field(
        field_name="dataset.n_test",
        value=dataset.n_test,
        minimum=1,
    )
    dataset.rows = normalize_dataset_rows(dataset.rows)
    dataset.n_features_min = _validate_int_field(
        field_name="dataset.n_features_min",
        value=dataset.n_features_min,
        minimum=1,
    )
    dataset.n_features_max = _validate_int_field(
        field_name="dataset.n_features_max",
        value=dataset.n_features_max,
        minimum=1,
    )
    dataset.max_categorical_cardinality = _validate_int_field(
        field_name="dataset.max_categorical_cardinality",
        value=dataset.max_categorical_cardinality,
        minimum=2,
    )

    dataset.categorical_ratio_min = _validate_finite_float_field(
        field_name="dataset.categorical_ratio_min",
        value=dataset.categorical_ratio_min,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    dataset.categorical_ratio_max = _validate_finite_float_field(
        field_name="dataset.categorical_ratio_max",
        value=dataset.categorical_ratio_max,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )

    dataset.n_classes_min = _validate_int_field(
        field_name="dataset.n_classes_min",
        value=dataset.n_classes_min,
        minimum=2,
        maximum=MAX_SUPPORTED_CLASS_COUNT,
    )
    dataset.n_classes_max = _validate_int_field(
        field_name="dataset.n_classes_max",
        value=dataset.n_classes_max,
        minimum=2,
        maximum=MAX_SUPPORTED_CLASS_COUNT,
    )

    dataset.missing_rate = _validate_finite_float_field(
        field_name="dataset.missing_rate",
        value=dataset.missing_rate,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    dataset.missing_mechanism = normalize_missing_mechanism(dataset.missing_mechanism)
    dataset.missing_mar_observed_fraction = _validate_finite_float_field(
        field_name="dataset.missing_mar_observed_fraction",
        value=dataset.missing_mar_observed_fraction,
        lo=0.0,
        hi=1.0,
        lo_inclusive=False,
        hi_inclusive=True,
        expectation="in (0, 1]",
    )
    dataset.missing_mar_logit_scale = _validate_finite_float_field(
        field_name="dataset.missing_mar_logit_scale",
        value=dataset.missing_mar_logit_scale,
        lo=0.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 0",
    )
    dataset.missing_mnar_logit_scale = _validate_finite_float_field(
        field_name="dataset.missing_mnar_logit_scale",
        value=dataset.missing_mnar_logit_scale,
        lo=0.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 0",
    )


def _normalize_graph_fields(graph: GraphConfig) -> None:
    """Stage 1: normalize graph scalar fields."""

    graph.n_nodes_min = _validate_int_field(
        field_name="graph.n_nodes_min",
        value=graph.n_nodes_min,
        minimum=2,
    )
    graph.n_nodes_max = _validate_int_field(
        field_name="graph.n_nodes_max",
        value=graph.n_nodes_max,
        minimum=2,
    )


def _normalize_mechanism_fields(mechanism: MechanismConfig) -> None:
    """Stage 1: normalize mechanism section fields."""

    mechanism.function_family_mix = _normalize_function_family_mix(mechanism.function_family_mix)


def _normalize_shift_fields(shift: ShiftConfig) -> None:
    """Stage 1: normalize shift fields."""

    if not isinstance(shift.enabled, bool):
        raise ValueError(f"shift.enabled must be a boolean, got {shift.enabled!r}.")
    shift.mode = normalize_shift_mode(shift.mode)
    shift.graph_scale = _validate_optional_finite_float_field(
        field_name="shift.graph_scale",
        value=shift.graph_scale,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    shift.mechanism_scale = _validate_optional_finite_float_field(
        field_name="shift.mechanism_scale",
        value=shift.mechanism_scale,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    shift.variance_scale = _validate_optional_finite_float_field(
        field_name="shift.variance_scale",
        value=shift.variance_scale,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )


def _normalize_noise_fields(noise: NoiseConfig) -> None:
    """Stage 1: normalize noise family and scalar fields."""

    noise.family = normalize_noise_family(noise.family)
    noise.base_scale = _validate_finite_float_field(
        field_name="noise.base_scale",
        value=noise.base_scale,
        lo=0.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 0",
    )
    noise.student_t_df = _validate_finite_float_field(
        field_name="noise.student_t_df",
        value=noise.student_t_df,
        lo=2.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 2",
    )
    noise.mixture_weights = _normalize_noise_mixture_weights(noise.mixture_weights)


def _normalize_runtime_fields(runtime: RuntimeConfig) -> None:
    """Stage 1: normalize runtime selector fields."""

    if runtime.device is None:
        runtime.device = "auto"
    elif isinstance(runtime.device, bool):
        raise ValueError(f"runtime.device must be a string or null, got {runtime.device!r}.")
    else:
        runtime.device = str(runtime.device).strip().lower() or "auto"

    if isinstance(runtime.torch_dtype, bool):
        raise ValueError(f"runtime.torch_dtype must be a string, got {runtime.torch_dtype!r}.")
    runtime.torch_dtype = str(runtime.torch_dtype).strip().lower()
    if not runtime.torch_dtype:
        raise ValueError("runtime.torch_dtype must be a non-empty string.")

    if runtime.fixed_layout_target_cells is not None:
        runtime.fixed_layout_target_cells = _validate_int_field(
            field_name="runtime.fixed_layout_target_cells",
            value=runtime.fixed_layout_target_cells,
            minimum=1,
        )


def _normalize_output_fields(_output: OutputConfig) -> None:
    """Stage 1: output section has no additional field normalization."""


def _normalize_diagnostics_fields(_diagnostics: DiagnosticsConfig) -> None:
    """Stage 1: diagnostics section has no additional field normalization."""


def _normalize_benchmark_fields(_benchmark: BenchmarkConfig) -> None:
    """Stage 1: benchmark section has no additional field normalization."""


def _normalize_filter_fields(filter_cfg: FilterConfig) -> None:
    """Stage 1: normalize filter scalar fields."""

    filter_cfg.ease_k_small = _validate_int_field(
        field_name="filter.ease_k_small",
        value=filter_cfg.ease_k_small,
        minimum=1,
    )
    filter_cfg.easy_skill_threshold = _validate_finite_float_field(
        field_name="filter.easy_skill_threshold",
        value=filter_cfg.easy_skill_threshold,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    filter_cfg.easy_gain_threshold = _validate_finite_float_field(
        field_name="filter.easy_gain_threshold",
        value=filter_cfg.easy_gain_threshold,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    filter_cfg.hard_skill_threshold = _validate_finite_float_field(
        field_name="filter.hard_skill_threshold",
        value=filter_cfg.hard_skill_threshold,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    filter_cfg.stump_skill_threshold = _validate_optional_finite_float_field(
        field_name="filter.stump_skill_threshold",
        value=filter_cfg.stump_skill_threshold,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    filter_cfg.max_attempts = _validate_int_field(
        field_name="filter.max_attempts",
        value=filter_cfg.max_attempts,
        minimum=1,
    )
    if not isinstance(filter_cfg.use_lineage_veto, bool):
        raise ValueError(
            f"filter.use_lineage_veto must be a boolean, got {filter_cfg.use_lineage_veto!r}."
        )
    filter_cfg.n_jobs = _validate_int_field(
        field_name="filter.n_jobs",
        value=filter_cfg.n_jobs,
        minimum=-1,
    )
    if filter_cfg.n_jobs == 0:
        raise ValueError("filter.n_jobs must be -1 or an integer >= 1, got 0.")


def _normalize_steering_preset(value: object) -> str | None:
    """Normalize an optional steering preset name."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str):
        supported = ", ".join(sorted(_STEERING_PRESET_DEFINITIONS))
        raise ValueError(f"Unsupported steering.preset {value!r}. Expected one of: {supported}.")
    normalized = value.strip()
    if not normalized:
        raise ValueError("steering.preset must be a non-empty string when provided.")
    if normalized not in _STEERING_PRESET_DEFINITIONS:
        supported = ", ".join(sorted(_STEERING_PRESET_DEFINITIONS))
        raise ValueError(f"Unsupported steering.preset {value!r}. Expected one of: {supported}.")
    return normalized


def _normalize_steering_band(
    *,
    field_name: str,
    value: object | None,
    lo: float,
    hi: float | None,
    lo_inclusive: bool,
    hi_inclusive: bool,
    expectation: str,
) -> list[float] | None:
    """Normalize one directional [start, end] steering band without sorting endpoints."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-value [start, end] band.")
    start = _validate_finite_float_field(
        field_name=f"{field_name}[0]",
        value=value[0],
        lo=lo,
        hi=hi,
        lo_inclusive=lo_inclusive,
        hi_inclusive=hi_inclusive,
        expectation=expectation,
    )
    end = _validate_finite_float_field(
        field_name=f"{field_name}[1]",
        value=value[1],
        lo=lo,
        hi=hi,
        lo_inclusive=lo_inclusive,
        hi_inclusive=hi_inclusive,
        expectation=expectation,
    )
    return [float(start), float(end)]


def _normalize_steering_mixture_weight_bands(
    value: object | None,
) -> dict[NoiseMixtureComponent, list[float]] | None:
    """Normalize optional directional mixture-weight bands."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("steering.stages[*].noise.mixture_weights must be a mapping.")
    if not value:
        raise ValueError(
            "steering.stages[*].noise.mixture_weights must include at least one supported component."
        )

    weights: dict[NoiseMixtureComponent, list[float]] = {}
    for raw_key, raw_value in value.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, str):
            raise ValueError(
                "steering.stages[*].noise.mixture_weights keys must be gaussian, laplace, or student_t."
            )
        normalized_key = raw_key.strip().lower()
        component = _NOISE_MIXTURE_COMPONENT_VALUE_MAP.get(normalized_key)
        if component is None:
            raise ValueError(
                "Unsupported steering.stages[*].noise.mixture_weights key "
                f"{raw_key!r}. Expected gaussian, laplace, or student_t."
            )
        if component in weights:
            raise ValueError(
                "Duplicate steering.stages[*].noise.mixture_weights key "
                f"{raw_key!r} after normalization."
            )
        weights[component] = _normalize_steering_band(
            field_name=f"steering.stages[*].noise.mixture_weights.{component}",
            value=raw_value,
            lo=0.0,
            hi=None,
            lo_inclusive=True,
            hi_inclusive=False,
            expectation="a finite value >= 0",
        ) or [0.0, 0.0]
    return weights


def _normalize_steering_stage_dataset_fields(dataset: SteeringStageDatasetConfig) -> None:
    """Stage 1: normalize one steering dataset-stage block."""

    dataset.missing_rate = _normalize_steering_band(
        field_name="steering.stages[*].dataset.missing_rate",
        value=dataset.missing_rate,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    if dataset.missing_mechanism is not None:
        dataset.missing_mechanism = normalize_missing_mechanism(dataset.missing_mechanism)
    dataset.missing_mar_observed_fraction = _validate_optional_finite_float_field(
        field_name="steering.stages[*].dataset.missing_mar_observed_fraction",
        value=dataset.missing_mar_observed_fraction,
        lo=0.0,
        hi=1.0,
        lo_inclusive=False,
        hi_inclusive=True,
        expectation="in (0, 1]",
    )
    dataset.missing_mar_logit_scale = _validate_optional_finite_float_field(
        field_name="steering.stages[*].dataset.missing_mar_logit_scale",
        value=dataset.missing_mar_logit_scale,
        lo=0.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 0",
    )
    dataset.missing_mnar_logit_scale = _validate_optional_finite_float_field(
        field_name="steering.stages[*].dataset.missing_mnar_logit_scale",
        value=dataset.missing_mnar_logit_scale,
        lo=0.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 0",
    )


def _normalize_steering_stage_shift_fields(shift: SteeringStageShiftConfig) -> None:
    """Stage 1: normalize one steering shift-stage block."""

    if shift.mode is not None:
        shift.mode = normalize_shift_mode(shift.mode)
    shift.graph_scale = _normalize_steering_band(
        field_name="steering.stages[*].shift.graph_scale",
        value=shift.graph_scale,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )
    shift.variance_scale = _normalize_steering_band(
        field_name="steering.stages[*].shift.variance_scale",
        value=shift.variance_scale,
        lo=0.0,
        hi=1.0,
        lo_inclusive=True,
        hi_inclusive=True,
        expectation="a finite value in [0, 1]",
    )


def _normalize_steering_stage_noise_fields(noise: SteeringStageNoiseConfig) -> None:
    """Stage 1: normalize one steering noise-stage block."""

    if noise.family is not None:
        noise.family = normalize_noise_family(noise.family)
    noise.student_t_df = _validate_optional_finite_float_field(
        field_name="steering.stages[*].noise.student_t_df",
        value=noise.student_t_df,
        lo=2.0,
        hi=None,
        lo_inclusive=False,
        hi_inclusive=False,
        expectation="a finite value > 2",
    )
    noise.mixture_weights = _normalize_steering_mixture_weight_bands(noise.mixture_weights)


def _normalize_steering_stage_fields(stage: SteeringStageConfig) -> None:
    """Stage 1: normalize one steering stage block."""

    if isinstance(stage.name, bool) or not isinstance(stage.name, str):
        raise ValueError("steering.stages[*].name must be a non-empty string.")
    stage.name = stage.name.strip()
    if not stage.name:
        raise ValueError("steering.stages[*].name must be a non-empty string.")
    stage.fraction = _validate_finite_float_field(
        field_name="steering.stages[*].fraction",
        value=stage.fraction,
        lo=0.0,
        hi=1.0,
        lo_inclusive=False,
        hi_inclusive=True,
        expectation="a finite value in (0, 1]",
    )
    if stage.dataset is not None:
        stage.dataset = _coerce_section(
            section_name="steering.stages[*].dataset",
            value=stage.dataset,
            section_type=SteeringStageDatasetConfig,
        )
        _normalize_steering_stage_dataset_fields(stage.dataset)
    if stage.shift is not None:
        stage.shift = _coerce_section(
            section_name="steering.stages[*].shift",
            value=stage.shift,
            section_type=SteeringStageShiftConfig,
        )
        _normalize_steering_stage_shift_fields(stage.shift)
    if stage.noise is not None:
        stage.noise = _coerce_section(
            section_name="steering.stages[*].noise",
            value=stage.noise,
            section_type=SteeringStageNoiseConfig,
        )
        _normalize_steering_stage_noise_fields(stage.noise)


def _normalize_steering_fields(steering: SteeringConfig) -> None:
    """Stage 1: normalize the steering section."""

    if not isinstance(steering.enabled, bool):
        raise ValueError(f"steering.enabled must be a boolean, got {steering.enabled!r}.")
    steering.preset = _normalize_steering_preset(steering.preset)

    raw_stages = steering.stages
    if raw_stages is None:
        steering.stages = []
    elif not isinstance(raw_stages, list):
        raise ValueError("steering.stages must be a list.")
    else:
        normalized_stages: list[SteeringStageConfig] = []
        for idx, raw_stage in enumerate(raw_stages):
            stage = _coerce_section(
                section_name=f"steering.stages[{idx}]",
                value=raw_stage,
                section_type=SteeringStageConfig,
            )
            _normalize_steering_stage_fields(stage)
            normalized_stages.append(stage)
        steering.stages = normalized_stages


def _coerce_section(
    *,
    section_name: str,
    value: object,
    section_type: type[_SectionT],
) -> _SectionT:
    """Coerce one top-level section into its canonical dataclass type."""

    if isinstance(value, section_type):
        return value
    if isinstance(value, dict):
        return section_type(**value)
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
        if isinstance(payload, dict):
            return section_type(**payload)
    raise TypeError(
        f"{section_name} must be a {section_type.__name__} or mapping, got {type(value).__name__}."
    )


def _stage1_normalize_generation_sections(config: GeneratorConfig) -> None:
    """Stage 1: field-level normalization/typing for all config sections."""

    config.dataset = _coerce_section(
        section_name="dataset",
        value=config.dataset,
        section_type=DatasetConfig,
    )
    config.graph = _coerce_section(
        section_name="graph",
        value=config.graph,
        section_type=GraphConfig,
    )
    config.mechanism = _coerce_section(
        section_name="mechanism",
        value=config.mechanism,
        section_type=MechanismConfig,
    )
    config.shift = _coerce_section(
        section_name="shift",
        value=config.shift,
        section_type=ShiftConfig,
    )
    config.noise = _coerce_section(
        section_name="noise",
        value=config.noise,
        section_type=NoiseConfig,
    )
    config.steering = _coerce_section(
        section_name="steering",
        value=config.steering,
        section_type=SteeringConfig,
    )
    config.stress = _coerce_section(
        section_name="stress",
        value=config.stress,
        section_type=StressConfig,
    )
    config.runtime = _coerce_section(
        section_name="runtime",
        value=config.runtime,
        section_type=RuntimeConfig,
    )
    config.output = _coerce_section(
        section_name="output",
        value=config.output,
        section_type=OutputConfig,
    )
    config.diagnostics = _coerce_section(
        section_name="diagnostics",
        value=config.diagnostics,
        section_type=DiagnosticsConfig,
    )
    config.benchmark = _coerce_section(
        section_name="benchmark",
        value=config.benchmark,
        section_type=BenchmarkConfig,
    )
    config.filter = _coerce_section(
        section_name="filter",
        value=config.filter,
        section_type=FilterConfig,
    )

    _normalize_dataset_fields(config.dataset)
    _normalize_graph_fields(config.graph)
    _normalize_mechanism_fields(config.mechanism)
    _normalize_shift_fields(config.shift)
    _normalize_noise_fields(config.noise)
    _normalize_steering_fields(config.steering)
    _normalize_stress_fields(config.stress)
    _normalize_runtime_fields(config.runtime)
    _normalize_output_fields(config.output)
    _normalize_diagnostics_fields(config.diagnostics)
    _normalize_benchmark_fields(config.benchmark)
    _normalize_filter_fields(config.filter)


def _stage2_validate_dataset_constraints(dataset: DatasetConfig) -> None:
    """Stage 2: validate dataset cross-field constraints."""

    _validate_min_max_pair(
        name="dataset.n_features_min",
        min_value=dataset.n_features_min,
        max_value=dataset.n_features_max,
        max_label="n_features_max",
    )
    if dataset.categorical_ratio_min > dataset.categorical_ratio_max:
        raise ValueError(
            "dataset.categorical_ratio_min must be <= categorical_ratio_max, "
            f"got {dataset.categorical_ratio_min} > {dataset.categorical_ratio_max}."
        )
    _validate_min_max_pair(
        name="dataset.n_classes_min",
        min_value=dataset.n_classes_min,
        max_value=dataset.n_classes_max,
        max_label="n_classes_max",
    )

    effective_n_train = int(dataset.n_train)
    if dataset.rows is not None:
        bounds = dataset_rows_bounds(dataset.rows)
        assert bounds is not None
        min_total_rows, _ = bounds
        n_test = int(dataset.n_test)
        if min_total_rows <= n_test:
            raise ValueError(
                "dataset.rows minimum total rows must be > dataset.n_test when rows mode is active "
                f"(min_total_rows={min_total_rows}, n_test={n_test})."
            )
        effective_n_train = int(min_total_rows - n_test)

    if dataset.task == "classification":
        validate_class_split_feasibility(
            n_classes=int(dataset.n_classes_min),
            n_train=effective_n_train,
            n_test=int(dataset.n_test),
            context="dataset classification split constraints for n_classes_min",
        )
        validate_class_split_feasibility(
            n_classes=int(dataset.n_classes_max),
            n_train=effective_n_train,
            n_test=int(dataset.n_test),
            context="dataset classification split constraints for n_classes_max",
        )

    if dataset.missing_rate > 0.0 and dataset.missing_mechanism == MISSINGNESS_MECHANISM_NONE:
        raise ValueError(
            "dataset.missing_mechanism must be mcar, mar, or mnar when dataset.missing_rate > 0."
        )


def _stage2_validate_graph_constraints(graph: GraphConfig) -> None:
    """Stage 2: validate graph cross-field constraints."""

    _validate_min_max_pair(
        name="graph.n_nodes_min",
        min_value=graph.n_nodes_min,
        max_value=graph.n_nodes_max,
        max_label="n_nodes_max",
    )


def _stage2_validate_mechanism_constraints(mechanism: MechanismConfig) -> None:
    """Stage 2: validate mechanism-family dependent relationships."""

    family_mix = mechanism.function_family_mix
    if family_mix is None:
        return
    supported = ", ".join(sorted(_PRODUCT_COMPONENT_FAMILIES))
    for family in ("product", "piecewise"):
        if family not in family_mix:
            continue
        has_component_family = any(
            component_family in family_mix for component_family in _PRODUCT_COMPONENT_FAMILIES
        )
        if has_component_family:
            continue
        raise ValueError(
            f"mechanism.function_family_mix assigns positive weight to '{family}' but none of its "
            f"component families are enabled. Add one of: {supported}."
        )


def _stage2_validate_shift_constraints(shift: ShiftConfig) -> None:
    """Stage 2: validate shift mode and override compatibility."""

    has_overrides = any(
        scale is not None
        for scale in (shift.graph_scale, shift.mechanism_scale, shift.variance_scale)
    )
    if not shift.enabled:
        if shift.mode != SHIFT_MODE_OFF:
            raise ValueError("shift.mode must be 'off' when shift.enabled is false.")
        if has_overrides:
            raise ValueError("shift override scales must be unset when shift.enabled is false.")
        return

    if shift.mode == SHIFT_MODE_OFF:
        raise ValueError("shift.mode must not be 'off' when shift.enabled is true.")

    if shift.mode == SHIFT_MODE_CUSTOM and not has_overrides:
        raise ValueError("shift.mode 'custom' requires at least one override scale.")

    if shift.mode == SHIFT_MODE_GRAPH_DRIFT and (
        shift.mechanism_scale is not None or shift.variance_scale is not None
    ):
        raise ValueError("shift.mode 'graph_drift' only allows shift.graph_scale override.")
    if shift.mode == SHIFT_MODE_MECHANISM_DRIFT and (
        shift.graph_scale is not None or shift.variance_scale is not None
    ):
        raise ValueError("shift.mode 'mechanism_drift' only allows shift.mechanism_scale override.")
    if shift.mode == SHIFT_MODE_NOISE_DRIFT and (
        shift.graph_scale is not None or shift.mechanism_scale is not None
    ):
        raise ValueError("shift.mode 'noise_drift' only allows shift.variance_scale override.")


def _stage2_validate_noise_constraints(noise: NoiseConfig) -> None:
    """Stage 2: validate noise-family dependent relationships."""

    if noise.family != NOISE_FAMILY_MIXTURE and noise.mixture_weights is not None:
        raise ValueError("noise.mixture_weights is only allowed when noise.family is 'mixture'.")


def _stage2_validate_steering_stage_constraints(stage: SteeringStageConfig) -> None:
    """Stage 2: validate one steering stage payload."""

    if stage.dataset is None and stage.shift is None and stage.noise is None:
        raise ValueError("steering.stages[*] must set at least one of dataset, shift, or noise.")

    if stage.dataset is not None:
        dataset = stage.dataset
        if dataset.missing_rate is None or dataset.missing_mechanism is None:
            raise ValueError(
                "steering.stages[*].dataset must set both missing_rate and missing_mechanism."
            )
        if (
            max(dataset.missing_rate) > 0.0
            and dataset.missing_mechanism == MISSINGNESS_MECHANISM_NONE
        ):
            raise ValueError(
                "steering.stages[*].dataset.missing_mechanism must be mcar, mar, or mnar "
                "when any steering missing_rate endpoint is > 0."
            )

    if stage.shift is not None:
        shift = stage.shift
        if shift.mode is None:
            raise ValueError(
                "steering.stages[*].shift.mode must be set when a shift block is present."
            )
        shift_mode = str(shift.mode)
        has_graph = shift.graph_scale is not None
        has_variance = shift.variance_scale is not None
        if not has_graph and not has_variance:
            raise ValueError(
                "steering.stages[*].shift must set at least one of graph_scale or variance_scale."
            )
        if shift_mode == SHIFT_MODE_GRAPH_DRIFT and has_variance:
            raise ValueError("steering.stages[*].shift.mode 'graph_drift' only allows graph_scale.")
        if shift_mode == SHIFT_MODE_NOISE_DRIFT and has_graph:
            raise ValueError(
                "steering.stages[*].shift.mode 'noise_drift' only allows variance_scale."
            )
        if shift_mode == SHIFT_MODE_MECHANISM_DRIFT:
            raise ValueError("steering.stages[*].shift.mode 'mechanism_drift' is out of scope.")
        if shift_mode not in {
            SHIFT_MODE_GRAPH_DRIFT,
            SHIFT_MODE_NOISE_DRIFT,
            SHIFT_MODE_MIXED,
            SHIFT_MODE_CUSTOM,
        }:
            raise ValueError(
                "steering.stages[*].shift.mode must be graph_drift, noise_drift, mixed, or custom."
            )
        if shift_mode in {SHIFT_MODE_MIXED, SHIFT_MODE_CUSTOM}:
            shift_config = ShiftConfig(
                enabled=True,
                mode=shift.mode,
                graph_scale=None if shift.graph_scale is None else float(shift.graph_scale[-1]),
                mechanism_scale=None,
                variance_scale=(
                    None if shift.variance_scale is None else float(shift.variance_scale[-1])
                ),
            )
            _stage2_validate_shift_constraints(shift_config)

    if stage.noise is not None:
        noise = stage.noise
        if noise.family is None:
            raise ValueError(
                "steering.stages[*].noise.family must be set when a noise block is present."
            )
        if noise.family != NOISE_FAMILY_MIXTURE:
            raise ValueError("steering.stages[*].noise currently only supports family='mixture'.")
        if noise.mixture_weights is None:
            raise ValueError(
                "steering.stages[*].noise.mixture_weights is required when a noise block is present."
            )
        start_total = 0.0
        end_total = 0.0
        for component_band in noise.mixture_weights.values():
            start_total += float(component_band[0])
            end_total += float(component_band[1])
        if not math.isclose(start_total, 1.0, rel_tol=0.0, abs_tol=_STEERING_FRACTION_TOLERANCE):
            raise ValueError(
                "steering.stages[*].noise.mixture_weights start weights must sum to 1.0."
            )
        if not math.isclose(end_total, 1.0, rel_tol=0.0, abs_tol=_STEERING_FRACTION_TOLERANCE):
            raise ValueError(
                "steering.stages[*].noise.mixture_weights end weights must sum to 1.0."
            )


def steering_stage_definitions(steering: SteeringConfig) -> list[SteeringStageConfig]:
    """Return normalized stage definitions for explicit or preset steering config."""

    if steering.preset is None:
        return [copy.deepcopy(stage) for stage in steering.stages]

    preset_payload = steering_preset_definition(steering.preset)
    raw_stages = preset_payload.get("stages")
    if not isinstance(raw_stages, list):
        raise ValueError(f"steering preset {steering.preset!r} must define a list of stages.")
    normalized_stages: list[SteeringStageConfig] = []
    for idx, raw_stage in enumerate(raw_stages):
        stage = _coerce_section(
            section_name=f"steering.preset[{steering.preset}].stages[{idx}]",
            value=raw_stage,
            section_type=SteeringStageConfig,
        )
        _normalize_steering_stage_fields(stage)
        normalized_stages.append(stage)
    return normalized_stages


def _stage2_validate_steering_constraints(steering: SteeringConfig) -> None:
    """Stage 2: validate steering authoring form and stage semantics."""

    has_preset = steering.preset is not None
    has_explicit_stages = bool(steering.stages)

    if not steering.enabled:
        if has_preset or has_explicit_stages:
            raise ValueError(
                "steering.preset and steering.stages must be unset when steering.enabled is false."
            )
        return

    if has_preset and has_explicit_stages:
        raise ValueError(
            "steering must use exactly one authoring form: steering.preset or steering.stages."
        )
    if not has_preset and not has_explicit_stages:
        raise ValueError(
            "steering.enabled=true requires either steering.preset or steering.stages."
        )

    stages = steering_stage_definitions(steering)
    fraction_total = 0.0
    stage_names: set[str] = set()
    for stage in stages:
        if stage.name in stage_names:
            raise ValueError(f"Duplicate steering.stages name {stage.name!r} after normalization.")
        stage_names.add(stage.name)
        _stage2_validate_steering_stage_constraints(stage)
        fraction_total += float(stage.fraction)

    if not math.isclose(fraction_total, 1.0, rel_tol=0.0, abs_tol=_STEERING_FRACTION_TOLERANCE):
        raise ValueError(f"steering.stages fractions must sum to 1.0, got {fraction_total:.6f}.")


def _stage2_validate_generation_constraints(config: GeneratorConfig) -> None:
    """Stage 2: validate cross-field constraints after section normalization."""

    _stage2_validate_dataset_constraints(config.dataset)
    _stage2_validate_graph_constraints(config.graph)
    _stage2_validate_mechanism_constraints(config.mechanism)
    _stage2_validate_shift_constraints(config.shift)
    _stage2_validate_noise_constraints(config.noise)
    _stage2_validate_steering_constraints(config.steering)
    _stage2_validate_stress_constraints(config)


def _run_generation_validation_stages(config: GeneratorConfig) -> None:
    """Run staged generation validation (stage1 normalize -> stage2 cross-field)."""

    _stage1_normalize_generation_sections(config)
    _stage2_validate_generation_constraints(config)


@dataclass(slots=True)
class DatasetConfig:
    task: str = "classification"
    n_train: int = 768
    n_test: int = 256
    rows: DatasetRowsSpec | None = None
    n_features_min: int = 16
    n_features_max: int = 64
    n_classes_min: int = 2
    n_classes_max: int = 10
    categorical_ratio_min: float = 0.0
    categorical_ratio_max: float = 1.0
    max_categorical_cardinality: int = 9
    missing_rate: float = 0.0
    missing_mechanism: MissingnessMechanism = MISSINGNESS_MECHANISM_NONE
    missing_mar_observed_fraction: float = 0.5
    missing_mar_logit_scale: float = 1.0
    missing_mnar_logit_scale: float = 1.0

    def __post_init__(self) -> None:
        _normalize_dataset_fields(self)


@dataclass(slots=True)
class GraphConfig:
    n_nodes_min: int = 2
    n_nodes_max: int = 32

    def __post_init__(self) -> None:
        _normalize_graph_fields(self)


@dataclass(slots=True)
class MechanismConfig:
    function_family_mix: dict[MechanismFamily, float] | None = None

    def __post_init__(self) -> None:
        _normalize_mechanism_fields(self)


@dataclass(slots=True)
class ShiftConfig:
    enabled: bool = False
    mode: ShiftMode = SHIFT_MODE_OFF
    graph_scale: float | None = None
    mechanism_scale: float | None = None
    variance_scale: float | None = None

    def __post_init__(self) -> None:
        _normalize_shift_fields(self)


@dataclass(slots=True)
class NoiseConfig:
    family: NoiseFamily = NOISE_FAMILY_GAUSSIAN
    base_scale: float = 1.0
    student_t_df: float = 5.0
    mixture_weights: dict[NoiseMixtureComponent, float] | None = None

    def __post_init__(self) -> None:
        _normalize_noise_fields(self)


@dataclass(slots=True)
class SteeringStageDatasetConfig:
    missing_rate: list[float] | None = None
    missing_mechanism: MissingnessMechanism | None = None
    missing_mar_observed_fraction: float | None = None
    missing_mar_logit_scale: float | None = None
    missing_mnar_logit_scale: float | None = None

    def __post_init__(self) -> None:
        _normalize_steering_stage_dataset_fields(self)


@dataclass(slots=True)
class SteeringStageShiftConfig:
    mode: ShiftMode | None = None
    graph_scale: list[float] | None = None
    variance_scale: list[float] | None = None

    def __post_init__(self) -> None:
        _normalize_steering_stage_shift_fields(self)


@dataclass(slots=True)
class SteeringStageNoiseConfig:
    family: NoiseFamily | None = None
    student_t_df: float | None = None
    mixture_weights: dict[NoiseMixtureComponent, list[float]] | None = None

    def __post_init__(self) -> None:
        _normalize_steering_stage_noise_fields(self)


@dataclass(slots=True)
class SteeringStageConfig:
    name: str = ""
    fraction: float = 1.0
    dataset: SteeringStageDatasetConfig | None = None
    shift: SteeringStageShiftConfig | None = None
    noise: SteeringStageNoiseConfig | None = None

    def __post_init__(self) -> None:
        _normalize_steering_stage_fields(self)


@dataclass(slots=True)
class SteeringConfig:
    enabled: bool = False
    preset: str | None = None
    stages: list[SteeringStageConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        _normalize_steering_fields(self)


@dataclass(slots=True)
class StressConfig:
    profile: str | None = None

    def __post_init__(self) -> None:
        _normalize_stress_fields(self)


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    torch_dtype: str = "float32"
    fixed_layout_target_cells: int | None = None


@dataclass(slots=True)
class OutputConfig:
    out_dir: str = "data/run_default"
    shard_size: int = 128
    compression: str = "zstd"


@dataclass(slots=True)
class DiagnosticsConfig:
    enabled: bool = False
    include_spearman: bool = False
    histogram_bins: int = 10
    quantiles: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])
    underrepresented_threshold: float = 0.5
    max_values_per_metric: int | None = 50_000
    meta_feature_targets: dict[str, list[float] | tuple[float, float]] = field(default_factory=dict)
    out_dir: str | None = None


@dataclass(slots=True)
class BenchmarkConfig:
    preset_name: str = "medium_cuda"
    num_datasets: int = 2000
    warmup_datasets: int = 25
    suite: str = "standard"
    warn_threshold_pct: float = 10.0
    fail_threshold_pct: float = 20.0
    collect_memory: bool = True
    collect_reproducibility: bool = False
    reproducibility_num_datasets: int = 2
    latency_num_samples: int = 20
    presets: dict[str, dict[str, int | str]] = field(
        default_factory=lambda: {
            "cpu": {"num_datasets": 200, "warmup_datasets": 10, "device": "cpu"},
            "cuda_desktop": {
                "num_datasets": 2000,
                "warmup_datasets": 25,
                "device": "cuda",
            },
            "cuda_h100": {
                "num_datasets": 5000,
                "warmup_datasets": 50,
                "device": "cuda",
            },
        }
    )


@dataclass(slots=True)
class FilterConfig:
    enabled: bool = False
    n_estimators: int = 25
    max_depth: int = 6
    min_samples_leaf: int = 1
    max_leaf_nodes: int | None = None
    max_features: str | int | float = "auto"
    n_bootstrap: int = 200
    ease_k_small: int = 16
    easy_skill_threshold: float = 0.8
    easy_gain_threshold: float = 0.1
    hard_skill_threshold: float = 0.0
    stump_skill_threshold: float | None = None
    use_lineage_veto: bool = True
    max_attempts: int = 3
    n_jobs: int = -1

    def __post_init__(self) -> None:
        _normalize_filter_fields(self)


@dataclass(slots=True)
class GeneratorConfig:
    seed: int = 1
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    mechanism: MechanismConfig = field(default_factory=MechanismConfig)
    shift: ShiftConfig = field(default_factory=ShiftConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)

    def __post_init__(self) -> None:
        self.seed = _validate_int_field(
            field_name="seed",
            value=self.seed,
            minimum=SEED32_MIN,
            maximum=SEED32_MAX,
        )
        _run_generation_validation_stages(self)

    def validate_generation_constraints(self) -> None:
        """Stage 3: re-run staged validation after runtime/CLI overrides."""

        _run_generation_validation_stages(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GeneratorConfig:
        """Construct `GeneratorConfig` from a nested dictionary payload."""

        data = data or {}
        runtime_payload = dict(data.get("runtime") or {})
        removed_runtime_keys = [
            key for key in ("worker_count", "worker_index") if key in runtime_payload
        ]
        if removed_runtime_keys:
            joined = ", ".join(f"runtime.{key}" for key in removed_runtime_keys)
            raise ValueError(
                f"{joined} is no longer supported. Parallel generation has been removed; "
                "remove these runtime keys from the config."
            )
        steering_payload = dict(data.get("steering") or {})
        if "target_metrics" in steering_payload:
            raise ValueError(
                "steering.target_metrics is not supported yet. Remove it from the config."
            )
        dataset = DatasetConfig(**(data.get("dataset") or {}))
        graph = GraphConfig(**(data.get("graph") or {}))
        mechanism = MechanismConfig(**(data.get("mechanism") or {}))
        shift = ShiftConfig(**(data.get("shift") or {}))
        noise = NoiseConfig(**(data.get("noise") or {}))
        steering = SteeringConfig(**steering_payload)
        stress = StressConfig(**(data.get("stress") or {}))
        runtime = RuntimeConfig(**runtime_payload)
        output = OutputConfig(**(data.get("output") or {}))
        diagnostics = DiagnosticsConfig(**(data.get("diagnostics") or {}))
        benchmark = BenchmarkConfig(**(data.get("benchmark") or {}))
        filter_payload = dict(data.get("filter") or {})
        if "threshold" in filter_payload:
            raise ValueError(
                "filter.threshold has been removed. Use filter.easy_skill_threshold, "
                "filter.easy_gain_threshold, and filter.hard_skill_threshold instead."
            )
        filter_cfg = FilterConfig(**filter_payload)
        seed = data.get("seed", 1)
        return cls(
            seed=seed,
            dataset=dataset,
            graph=graph,
            mechanism=mechanism,
            shift=shift,
            noise=noise,
            steering=steering,
            stress=stress,
            runtime=runtime,
            output=output,
            diagnostics=diagnostics,
            benchmark=benchmark,
            filter=filter_cfg,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> GeneratorConfig:
        """Load config from a YAML file path."""

        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config at {p} must be a mapping at the top level.")
        return cls.from_dict(loaded)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config dataclasses into a plain nested dictionary."""

        return asdict(self)
