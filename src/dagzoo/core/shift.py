"""Shift runtime parameter resolution for sampling-path integration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dagzoo.config import (
    SHIFT_MODE_CUSTOM,
    SHIFT_MODE_GRAPH_DRIFT,
    SHIFT_MODE_MECHANISM_DRIFT,
    SHIFT_MODE_MIXED,
    SHIFT_MODE_NOISE_DRIFT,
    SHIFT_MODE_OFF,
    GeneratorConfig,
)
from dagzoo.core.layout_types import MechanismFamily

_LOG_TWO = math.log(2.0)
_NOISE_VARIANCE_DB_SPAN = _LOG_TWO / 2.0

MECHANISM_FAMILY_ORDER: tuple[MechanismFamily, ...] = (
    "nn",
    "tree",
    "discretization",
    "gp",
    "linear",
    "quadratic",
    "em",
    "product",
)
MECHANISM_FAMILY_SUPPORTED_ORDER: tuple[MechanismFamily, ...] = (
    *MECHANISM_FAMILY_ORDER,
    "piecewise",
)

MECHANISM_FAMILY_BASE_LOGITS: dict[MechanismFamily, float] = {
    "nn": 0.7,
    "tree": 0.7,
    "discretization": 0.5,
    "gp": 0.5,
    "linear": -0.8,
    "quadratic": -0.6,
    "em": -0.3,
    "product": 0.9,
    "piecewise": 0.8,
}
NONLINEAR_MECHANISM_FAMILIES: tuple[MechanismFamily, ...] = (
    "nn",
    "tree",
    "discretization",
    "gp",
    "product",
    "piecewise",
)

_MODE_DEFAULT_SCALES: dict[str, tuple[float, float, float]] = {
    SHIFT_MODE_OFF: (0.0, 0.0, 0.0),
    SHIFT_MODE_GRAPH_DRIFT: (0.5, 0.0, 0.0),
    SHIFT_MODE_MECHANISM_DRIFT: (0.0, 0.5, 0.0),
    SHIFT_MODE_NOISE_DRIFT: (0.0, 0.0, 0.5),
    SHIFT_MODE_MIXED: (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    SHIFT_MODE_CUSTOM: (0.0, 0.0, 0.0),
}


@dataclass(slots=True, frozen=True)
class ShiftRuntimeParams:
    """Resolved shift runtime parameters for one generation run."""

    enabled: bool
    mode: str
    graph_scale: float
    mechanism_scale: float
    variance_scale: float
    edge_logit_bias_shift: float
    mechanism_logit_tilt: float
    variance_sigma_multiplier: float


def centered_mechanism_family_logits(
    families: tuple[MechanismFamily, ...],
) -> tuple[float, ...]:
    """Return centered family logits used for mechanism drift sampling."""

    if not families:
        return ()
    raw = tuple(float(MECHANISM_FAMILY_BASE_LOGITS.get(name, 0.0)) for name in families)
    mean = sum(raw) / float(len(raw))
    return tuple(value - mean for value in raw)


def mechanism_family_probabilities(
    *,
    mechanism_logit_tilt: float,
    families: tuple[MechanismFamily, ...] | None = None,
    family_weights: dict[MechanismFamily, float] | None = None,
) -> dict[MechanismFamily, float]:
    """Resolve mechanism family probabilities for a given tilt value."""

    if families is None:
        families = (
            MECHANISM_FAMILY_ORDER if family_weights is None else MECHANISM_FAMILY_SUPPORTED_ORDER
        )
    if not families:
        return {}

    if family_weights is None:
        base_weights = {family: 1.0 for family in families}
    else:
        base_weights = {}
        for family in families:
            raw = family_weights.get(family, 0.0)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(
                    f"mechanism family weight for '{family}' must be a finite number, got {raw!r}."
                )
            parsed = float(raw)
            if not math.isfinite(parsed) or parsed < 0.0:
                raise ValueError(
                    f"mechanism family weight for '{family}' must be finite and >= 0, got {raw!r}."
                )
            base_weights[family] = parsed

    positive_weights = {family: weight for family, weight in base_weights.items() if weight > 0.0}
    if not positive_weights:
        raise ValueError(
            "mechanism family weights must include at least one positive family weight."
        )

    total_base = float(sum(positive_weights.values()))
    normalized_base = {
        family: (positive_weights.get(family, 0.0) / total_base) for family in families
    }
    if mechanism_logit_tilt <= 0.0:
        return normalized_base

    centered_logits = centered_mechanism_family_logits(families)
    scaled = []
    for family, centered in zip(families, centered_logits, strict=True):
        weight = float(normalized_base[family])
        if weight <= 0.0:
            scaled.append(float("-inf"))
        else:
            scaled.append(float(math.log(weight) + (mechanism_logit_tilt * centered)))

    max_logit = max(scaled)
    exp_vals = [math.exp(logit - max_logit) if math.isfinite(logit) else 0.0 for logit in scaled]
    denom = sum(exp_vals)
    if denom <= 0.0:
        raise ValueError("mechanism family probabilities are ill-defined for the provided weights.")
    return {family: (exp_val / denom) for family, exp_val in zip(families, exp_vals, strict=True)}


def mechanism_nonlinear_mass(
    *,
    mechanism_logit_tilt: float,
    families: tuple[MechanismFamily, ...] | None = None,
    nonlinear_families: tuple[MechanismFamily, ...] = NONLINEAR_MECHANISM_FAMILIES,
    family_weights: dict[MechanismFamily, float] | None = None,
) -> float:
    """Return probability mass over nonlinear mechanism families."""

    if families is None:
        families = (
            MECHANISM_FAMILY_ORDER if family_weights is None else MECHANISM_FAMILY_SUPPORTED_ORDER
        )
    if not families:
        return 0.0
    probs = mechanism_family_probabilities(
        mechanism_logit_tilt=mechanism_logit_tilt,
        families=families,
        family_weights=family_weights,
    )
    nonlinear_set = set(nonlinear_families)
    mass = float(math.fsum(prob for family, prob in probs.items() if family in nonlinear_set))
    return min(1.0, max(0.0, mass))


def resolve_shift_runtime_params(config: GeneratorConfig) -> ShiftRuntimeParams:
    """Resolve shift mode/defaults/overrides into runtime coefficients."""

    shift = config.shift
    if not shift.enabled:
        return ShiftRuntimeParams(
            enabled=False,
            mode=SHIFT_MODE_OFF,
            graph_scale=0.0,
            mechanism_scale=0.0,
            variance_scale=0.0,
            edge_logit_bias_shift=0.0,
            mechanism_logit_tilt=0.0,
            variance_sigma_multiplier=1.0,
        )

    mode = str(shift.mode)
    default_graph_scale, default_mechanism_scale, default_noise_scale = _MODE_DEFAULT_SCALES[mode]

    graph_scale = (
        float(shift.graph_scale) if shift.graph_scale is not None else float(default_graph_scale)
    )
    mechanism_scale = (
        float(shift.mechanism_scale)
        if shift.mechanism_scale is not None
        else float(default_mechanism_scale)
    )
    variance_scale = (
        float(shift.variance_scale)
        if shift.variance_scale is not None
        else float(default_noise_scale)
    )

    return ShiftRuntimeParams(
        enabled=True,
        mode=mode,
        graph_scale=graph_scale,
        mechanism_scale=mechanism_scale,
        variance_scale=variance_scale,
        edge_logit_bias_shift=float(_LOG_TWO * graph_scale),
        mechanism_logit_tilt=mechanism_scale,
        variance_sigma_multiplier=float(math.exp(_NOISE_VARIANCE_DB_SPAN * variance_scale)),
    )


__all__ = [
    "MECHANISM_FAMILY_BASE_LOGITS",
    "MECHANISM_FAMILY_ORDER",
    "MECHANISM_FAMILY_SUPPORTED_ORDER",
    "NONLINEAR_MECHANISM_FAMILIES",
    "ShiftRuntimeParams",
    "centered_mechanism_family_logits",
    "mechanism_family_probabilities",
    "mechanism_nonlinear_mass",
    "resolve_shift_runtime_params",
]
