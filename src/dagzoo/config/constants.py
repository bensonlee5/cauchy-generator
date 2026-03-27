"""Config literals, enums, and shared constants."""

from __future__ import annotations

from typing import Literal, TypeVar

MechanismFamily = Literal[
    "nn",
    "tree",
    "discretization",
    "gp",
    "linear",
    "quadratic",
    "em",
    "product",
    "piecewise",
]

MissingnessMechanism = Literal["none", "mcar", "mar", "mnar"]
MISSINGNESS_MECHANISM_NONE: Literal["none"] = "none"
MISSINGNESS_MECHANISM_MCAR: Literal["mcar"] = "mcar"
MISSINGNESS_MECHANISM_MAR: Literal["mar"] = "mar"
MISSINGNESS_MECHANISM_MNAR: Literal["mnar"] = "mnar"

_MISSINGNESS_MECHANISM_VALUE_MAP: dict[str, MissingnessMechanism] = {
    MISSINGNESS_MECHANISM_NONE: MISSINGNESS_MECHANISM_NONE,
    MISSINGNESS_MECHANISM_MCAR: MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MAR: MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MNAR: MISSINGNESS_MECHANISM_MNAR,
}

ShiftMode = Literal[
    "off",
    "graph_drift",
    "mechanism_drift",
    "noise_drift",
    "mixed",
    "custom",
]
SHIFT_MODE_OFF: Literal["off"] = "off"
SHIFT_MODE_GRAPH_DRIFT: Literal["graph_drift"] = "graph_drift"
SHIFT_MODE_MECHANISM_DRIFT: Literal["mechanism_drift"] = "mechanism_drift"
SHIFT_MODE_NOISE_DRIFT: Literal["noise_drift"] = "noise_drift"
SHIFT_MODE_MIXED: Literal["mixed"] = "mixed"
SHIFT_MODE_CUSTOM: Literal["custom"] = "custom"

_SHIFT_MODE_VALUE_MAP: dict[str, ShiftMode] = {
    SHIFT_MODE_OFF: SHIFT_MODE_OFF,
    SHIFT_MODE_GRAPH_DRIFT: SHIFT_MODE_GRAPH_DRIFT,
    SHIFT_MODE_MECHANISM_DRIFT: SHIFT_MODE_MECHANISM_DRIFT,
    SHIFT_MODE_NOISE_DRIFT: SHIFT_MODE_NOISE_DRIFT,
    SHIFT_MODE_MIXED: SHIFT_MODE_MIXED,
    SHIFT_MODE_CUSTOM: SHIFT_MODE_CUSTOM,
}

NoiseFamily = Literal["gaussian", "laplace", "student_t", "mixture"]
NOISE_FAMILY_GAUSSIAN: Literal["gaussian"] = "gaussian"
NOISE_FAMILY_LAPLACE: Literal["laplace"] = "laplace"
NOISE_FAMILY_STUDENT_T: Literal["student_t"] = "student_t"
NOISE_FAMILY_MIXTURE: Literal["mixture"] = "mixture"

_NOISE_FAMILY_VALUE_MAP: dict[str, NoiseFamily] = {
    NOISE_FAMILY_GAUSSIAN: NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_LAPLACE: NOISE_FAMILY_LAPLACE,
    NOISE_FAMILY_STUDENT_T: NOISE_FAMILY_STUDENT_T,
    NOISE_FAMILY_MIXTURE: NOISE_FAMILY_MIXTURE,
}

NoiseMixtureComponent = Literal["gaussian", "laplace", "student_t"]
NOISE_MIXTURE_COMPONENT_GAUSSIAN: Literal["gaussian"] = "gaussian"
NOISE_MIXTURE_COMPONENT_LAPLACE: Literal["laplace"] = "laplace"
NOISE_MIXTURE_COMPONENT_STUDENT_T: Literal["student_t"] = "student_t"

_NOISE_MIXTURE_COMPONENT_VALUE_MAP: dict[str, NoiseMixtureComponent] = {
    NOISE_MIXTURE_COMPONENT_GAUSSIAN: NOISE_MIXTURE_COMPONENT_GAUSSIAN,
    NOISE_MIXTURE_COMPONENT_LAPLACE: NOISE_MIXTURE_COMPONENT_LAPLACE,
    NOISE_MIXTURE_COMPONENT_STUDENT_T: NOISE_MIXTURE_COMPONENT_STUDENT_T,
}

_MECHANISM_FAMILY_VALUE_MAP: dict[str, MechanismFamily] = {
    "nn": "nn",
    "tree": "tree",
    "discretization": "discretization",
    "gp": "gp",
    "linear": "linear",
    "quadratic": "quadratic",
    "em": "em",
    "product": "product",
    "piecewise": "piecewise",
}
_PRODUCT_COMPONENT_FAMILIES: frozenset[MechanismFamily] = frozenset(
    {"tree", "discretization", "gp", "linear", "quadratic"}
)

MAX_SUPPORTED_CLASS_COUNT = 32
DATASET_ROWS_MIN_TOTAL = 400
DATASET_ROWS_MAX_TOTAL = 60_000
_SectionT = TypeVar("_SectionT")
RowsMode = Literal["fixed", "range"]
