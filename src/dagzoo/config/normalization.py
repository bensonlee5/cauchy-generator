"""Shared config normalization helpers."""

from __future__ import annotations

from dagzoo.math import normalize_positive_weights

from .constants import (
    _INTERVENTION_MODE_VALUE_MAP,
    _INTERVENTION_TARGET_KIND_VALUE_MAP,
    _LAYOUT_MODE_VALUE_MAP,
    _MECHANISM_FAMILY_VALUE_MAP,
    _MISSINGNESS_MECHANISM_VALUE_MAP,
    _NOISE_FAMILY_VALUE_MAP,
    _NOISE_MIXTURE_COMPONENT_VALUE_MAP,
    _SHIFT_MODE_VALUE_MAP,
    InterventionMode,
    InterventionTargetKind,
    LayoutMode,
    MechanismFamily,
    MissingnessMechanism,
    NoiseFamily,
    NoiseMixtureComponent,
    ShiftMode,
)
from .scalars import _validate_finite_float_field


def normalize_layout_mode(value: object) -> LayoutMode:
    """Normalize runtime layout mode into a validated internal value."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            f"Unsupported runtime.layout_mode '{value}'. Expected heterogeneous, stratified, or fixed."
        )
    normalized = value.strip().lower()
    result = _LAYOUT_MODE_VALUE_MAP.get(normalized)
    if result is None:
        raise ValueError(
            f"Unsupported runtime.layout_mode '{value}'. Expected heterogeneous, stratified, or fixed."
        )
    return result


def normalize_missing_mechanism(value: str) -> MissingnessMechanism:
    """Normalize missingness mechanism into a validated internal value."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            f"Unsupported missing_mechanism '{value}'. Expected none, mcar, mar, or mnar."
        )
    normalized = value.strip().lower()
    result = _MISSINGNESS_MECHANISM_VALUE_MAP.get(normalized)
    if result is None:
        raise ValueError(
            f"Unsupported missing_mechanism '{value}'. Expected none, mcar, mar, or mnar."
        )
    return result


def normalize_intervention_mode(value: object) -> InterventionMode:
    """Normalize intervention mode into a validated internal value."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            "Unsupported intervention.mode "
            f"'{value}'. Expected observational or hard_interventional."
        )
    normalized = value.strip().lower()
    result = _INTERVENTION_MODE_VALUE_MAP.get(normalized)
    if result is None:
        raise ValueError(
            "Unsupported intervention.mode "
            f"'{value}'. Expected observational or hard_interventional."
        )
    return result


def normalize_intervention_target_kind(value: object) -> InterventionTargetKind:
    """Normalize one intervention target kind into a validated internal value."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            "Unsupported intervention.targets[].target_kind "
            f"'{value}'. Expected target, feature_node, or latent_node."
        )
    normalized = value.strip().lower()
    result = _INTERVENTION_TARGET_KIND_VALUE_MAP.get(normalized)
    if result is None:
        raise ValueError(
            "Unsupported intervention.targets[].target_kind "
            f"'{value}'. Expected target, feature_node, or latent_node."
        )
    return result


def normalize_shift_mode(value: object) -> ShiftMode:
    """Normalize shift mode into a validated internal value."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            "Unsupported shift.mode "
            f"'{value}'. Expected off, graph_drift, mechanism_drift, noise_drift, mixed, or custom."
        )
    normalized = value.strip().lower()
    result = _SHIFT_MODE_VALUE_MAP.get(normalized)
    if result is None:
        raise ValueError(
            "Unsupported shift.mode "
            f"'{value}'. Expected off, graph_drift, mechanism_drift, noise_drift, mixed, or custom."
        )
    return result


def normalize_noise_family(value: object) -> NoiseFamily:
    """Normalize noise family into a validated internal value."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            "Unsupported noise.family "
            f"'{value}'. Expected gaussian, laplace, student_t, or mixture."
        )
    normalized = value.strip().lower()
    result = _NOISE_FAMILY_VALUE_MAP.get(normalized)
    if result is None:
        raise ValueError(
            "Unsupported noise.family "
            f"'{value}'. Expected gaussian, laplace, student_t, or mixture."
        )
    return result


def _normalize_noise_mixture_weights(
    value: object | None,
) -> dict[NoiseMixtureComponent, float] | None:
    """Normalize optional noise-mixture weights mapping."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("noise.mixture_weights must be a mapping.")
    if not value:
        raise ValueError(
            "noise.mixture_weights must include at least one of gaussian, laplace, or student_t."
        )

    weights: dict[NoiseMixtureComponent, float] = {}
    for raw_key, raw_weight in value.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, str):
            raise ValueError(
                "noise.mixture_weights keys must be gaussian, laplace, or student_t strings."
            )
        normalized_key = raw_key.strip().lower()
        component = _NOISE_MIXTURE_COMPONENT_VALUE_MAP.get(normalized_key)
        if component is None:
            raise ValueError(
                "Unsupported noise.mixture_weights key "
                f"'{raw_key}'. Expected gaussian, laplace, or student_t."
            )
        if component in weights:
            raise ValueError(
                f"Duplicate noise.mixture_weights key '{raw_key}' after normalization."
            )
        weight = _validate_finite_float_field(
            field_name=f"noise.mixture_weights.{component}",
            value=raw_weight,
            lo=0.0,
            hi=None,
            lo_inclusive=True,
            hi_inclusive=False,
            expectation="a finite value >= 0",
        )
        weights[component] = float(weight)

    return normalize_positive_weights(weights, field_name="noise.mixture_weights")


def _normalize_function_family_mix(
    value: object | None,
) -> dict[MechanismFamily, float] | None:
    """Normalize optional mechanism family-mix weights mapping."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("mechanism.function_family_mix must be a mapping.")
    if not value:
        raise ValueError(
            "mechanism.function_family_mix must include at least one supported family."
        )

    weights: dict[MechanismFamily, float] = {}
    for raw_key, raw_weight in value.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, str):
            raise ValueError("mechanism.function_family_mix keys must be mechanism family names.")
        normalized_key = raw_key.strip().lower()
        family = _MECHANISM_FAMILY_VALUE_MAP.get(normalized_key)
        if family is None:
            supported = ", ".join(sorted(_MECHANISM_FAMILY_VALUE_MAP))
            raise ValueError(
                "Unsupported mechanism.function_family_mix key "
                f"'{raw_key}'. Expected one of: {supported}."
            )
        if family in weights:
            raise ValueError(
                f"Duplicate mechanism.function_family_mix key '{raw_key}' after normalization."
            )
        weight = _validate_finite_float_field(
            field_name=f"mechanism.function_family_mix.{family}",
            value=raw_weight,
            lo=0.0,
            hi=None,
            lo_inclusive=True,
            hi_inclusive=False,
            expectation="a finite value >= 0",
        )
        weights[family] = float(weight)
    return normalize_positive_weights(
        weights,
        field_name="mechanism.function_family_mix",
    )
