"""Internal typed fixed-layout execution-plan contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from dagzoo.core.layout_types import AggregationKind, ConverterKind, MechanismFamily

FixedLayoutExecutionContract = Literal["chunk_batched_v1"]
FixedLayoutRootBaseKind = Literal["normal", "uniform", "unit_ball", "normal_cov"]
FixedLayoutMatrixBaseKind = Literal["gaussian", "weights", "singular_values", "kernel"]
FixedLayoutActivationKind = Literal[
    "relu_pow",
    "signed_pow",
    "inv_pow",
    "poly",
    "gumbel_softmax",
]
FixedLayoutGpBranchKind = Literal["ha", "projected"]
FixedLayoutGpVariant = Literal["standard", "periodic", "multiscale"]
FixedLayoutConverterMethod = Literal["neighbor", "softmax"]
FixedLayoutConverterVariant = Literal[
    "input",
    "index_repeat",
    "center",
    "center_random_fn",
    "softmax_points",
]

DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT: FixedLayoutExecutionContract = "chunk_batched_v1"


@dataclass(frozen=True, slots=True)
class FixedActivationPlan:
    name: str
    mode: Literal["fixed"] = "fixed"


@dataclass(frozen=True, slots=True)
class ParametricActivationPlan:
    kind: FixedLayoutActivationKind
    poly_power: int | None = None
    temperature: float | None = None
    mode: Literal["parametric"] = "parametric"


FixedLayoutActivationPlan: TypeAlias = FixedActivationPlan | ParametricActivationPlan


@dataclass(frozen=True, slots=True)
class GaussianMatrixPlan:
    kind: Literal["gaussian"] = "gaussian"


@dataclass(frozen=True, slots=True)
class WeightsMatrixPlan:
    kind: Literal["weights"] = "weights"


@dataclass(frozen=True, slots=True)
class SingularValuesMatrixPlan:
    kind: Literal["singular_values"] = "singular_values"


@dataclass(frozen=True, slots=True)
class KernelMatrixPlan:
    kind: Literal["kernel"] = "kernel"


@dataclass(frozen=True, slots=True)
class ActivationMatrixPlan:
    base_kind: FixedLayoutMatrixBaseKind
    activation: FixedLayoutActivationPlan
    kind: Literal["activation"] = "activation"


FixedLayoutMatrixPlan: TypeAlias = (
    GaussianMatrixPlan
    | WeightsMatrixPlan
    | SingularValuesMatrixPlan
    | KernelMatrixPlan
    | ActivationMatrixPlan
)


@dataclass(frozen=True, slots=True)
class LinearFunctionPlan:
    matrix: FixedLayoutMatrixPlan
    family: Literal["linear"] = "linear"


@dataclass(frozen=True, slots=True)
class QuadraticFunctionPlan:
    matrix: FixedLayoutMatrixPlan
    family: Literal["quadratic"] = "quadratic"


@dataclass(frozen=True, slots=True)
class NeuralNetFunctionPlan:
    n_layers: int
    hidden_width: int
    input_activation: FixedLayoutActivationPlan | None
    output_activation: FixedLayoutActivationPlan | None
    layer_matrices: tuple[FixedLayoutMatrixPlan, ...]
    hidden_activations: tuple[FixedLayoutActivationPlan, ...]
    family: Literal["nn"] = "nn"


@dataclass(frozen=True, slots=True)
class TreeFunctionPlan:
    n_trees: int
    depths: tuple[int, ...]
    family: Literal["tree"] = "tree"


@dataclass(frozen=True, slots=True)
class DiscretizationFunctionPlan:
    n_centers: int
    linear_matrix: FixedLayoutMatrixPlan
    family: Literal["discretization"] = "discretization"


@dataclass(frozen=True, slots=True)
class GpFunctionPlan:
    branch_kind: FixedLayoutGpBranchKind
    variant: FixedLayoutGpVariant = "standard"
    family: Literal["gp"] = "gp"


@dataclass(frozen=True, slots=True)
class EmFunctionPlan:
    m_val: int
    linear_matrix: FixedLayoutMatrixPlan
    family: Literal["em"] = "em"


@dataclass(frozen=True, slots=True)
class ProductFunctionPlan:
    lhs: FixedLayoutFunctionPlan
    rhs: FixedLayoutFunctionPlan
    family: Literal["product"] = "product"


@dataclass(frozen=True, slots=True)
class PiecewiseFunctionPlan:
    gate_matrix: FixedLayoutMatrixPlan
    gate_bias: float
    gate_temperature: float
    lhs: FixedLayoutFunctionPlan
    rhs: FixedLayoutFunctionPlan
    family: Literal["piecewise"] = "piecewise"


FixedLayoutFunctionPlan: TypeAlias = (
    LinearFunctionPlan
    | QuadraticFunctionPlan
    | NeuralNetFunctionPlan
    | TreeFunctionPlan
    | DiscretizationFunctionPlan
    | GpFunctionPlan
    | EmFunctionPlan
    | ProductFunctionPlan
    | PiecewiseFunctionPlan
)


@dataclass(frozen=True, slots=True)
class FixedLayoutLatentPlan:
    required_dim: int
    extra_dim: int
    total_dim: int


@dataclass(frozen=True, slots=True)
class FixedLayoutConverterSpec:
    key: str
    kind: ConverterKind
    dim: int
    cardinality: int | None
    column_start: int
    column_end: int


@dataclass(frozen=True, slots=True)
class NumericConverterPlan:
    kind: Literal["num", "target_reg"]
    warp_enabled: bool


@dataclass(frozen=True, slots=True)
class CategoricalConverterPlan:
    kind: Literal["cat", "target_cls"]
    method: FixedLayoutConverterMethod
    variant: FixedLayoutConverterVariant
    function: FixedLayoutFunctionPlan | None = None


FixedLayoutConverterPlan: TypeAlias = NumericConverterPlan | CategoricalConverterPlan


@dataclass(frozen=True, slots=True)
class NumericConverterGroup:
    spec_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CategoricalConverterGroup:
    spec_indices: tuple[int, ...]


FixedLayoutConverterGroup: TypeAlias = NumericConverterGroup | CategoricalConverterGroup


@dataclass(frozen=True, slots=True)
class FixedLayoutCompiledConverterSlice:
    spec_index: int
    key: str
    column_start: int
    column_end: int
    width: int
    cardinality: int | None


@dataclass(frozen=True, slots=True)
class CompiledNumericConverterGroup:
    spec_indices: tuple[int, ...]
    slices: tuple[FixedLayoutCompiledConverterSlice, ...]
    plans: tuple[NumericConverterPlan, ...]
    warp_enabled: tuple[bool, ...]
    all_unit_width: bool


@dataclass(frozen=True, slots=True)
class CompiledCategoricalConverterGroup:
    spec_indices: tuple[int, ...]
    slices: tuple[FixedLayoutCompiledConverterSlice, ...]
    plan: CategoricalConverterPlan
    category_count: int
    uses_center_random_fn: bool


FixedLayoutCompiledConverterGroup: TypeAlias = (
    CompiledNumericConverterGroup | CompiledCategoricalConverterGroup
)


@dataclass(frozen=True, slots=True)
class RandomPointsNodeSource:
    base_kind: FixedLayoutRootBaseKind
    function: FixedLayoutFunctionPlan


@dataclass(frozen=True, slots=True)
class ConcatNodeSource:
    function: FixedLayoutFunctionPlan


@dataclass(frozen=True, slots=True)
class StackedNodeSource:
    aggregation_kind: AggregationKind
    parent_functions: tuple[FixedLayoutFunctionPlan, ...]


FixedLayoutNodeSource: TypeAlias = RandomPointsNodeSource | ConcatNodeSource | StackedNodeSource


@dataclass(frozen=True, slots=True)
class FixedLayoutNodePlan:
    node_index: int
    parent_indices: tuple[int, ...]
    converter_specs: tuple[FixedLayoutConverterSpec, ...]
    converter_plans: tuple[FixedLayoutConverterPlan, ...]
    converter_groups: tuple[FixedLayoutConverterGroup, ...]
    latent: FixedLayoutLatentPlan
    source: FixedLayoutNodeSource
    compiled_converter_groups: tuple[FixedLayoutCompiledConverterGroup, ...] = ()


@dataclass(frozen=True, slots=True)
class FixedLayoutExecutionPlan:
    node_plans: tuple[FixedLayoutNodePlan, ...] = ()
    execution_contract: FixedLayoutExecutionContract = DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT


def fixed_layout_converter_groups(
    converter_specs: Sequence[FixedLayoutConverterSpec],
    converter_plans: Sequence[FixedLayoutConverterPlan],
) -> tuple[FixedLayoutConverterGroup, ...]:
    groups: dict[tuple[Any, ...], FixedLayoutConverterGroup] = {}
    ordered_keys: list[tuple[Any, ...]] = []
    for spec_index, (spec, plan) in enumerate(zip(converter_specs, converter_plans, strict=True)):
        key: tuple[Any, ...]
        if isinstance(plan, NumericConverterPlan):
            key = ("numeric",)
            group = groups.get(key)
            if group is None:
                ordered_keys.append(key)
                groups[key] = NumericConverterGroup(spec_indices=(int(spec_index),))
            else:
                groups[key] = NumericConverterGroup(
                    spec_indices=(*group.spec_indices, int(spec_index))
                )
            continue

        key = (
            "categorical",
            str(spec.kind),
            str(plan.method),
            str(plan.variant),
            int(spec.dim),
            None if spec.cardinality is None else int(spec.cardinality),
            _converter_function_signature(plan.function),
        )
        group = groups.get(key)
        if group is None:
            ordered_keys.append(key)
            groups[key] = CategoricalConverterGroup(spec_indices=(int(spec_index),))
        else:
            groups[key] = CategoricalConverterGroup(
                spec_indices=(*group.spec_indices, int(spec_index))
            )
    return tuple(groups[key] for key in ordered_keys)


def fixed_layout_signature_payloads(
    execution_plan: FixedLayoutExecutionPlan,
) -> list[dict[str, Any]]:
    return [_signature_node_plan_payload(node_plan) for node_plan in execution_plan.node_plans]


def _converter_function_signature(plan: FixedLayoutFunctionPlan | None) -> str | None:
    if plan is None:
        return None
    encoded = json.dumps(
        _function_plan_payload(plan),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=8).hexdigest()


def _node_plan_payload(
    node_plan: FixedLayoutNodePlan,
    *,
    execution_contract: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "node_index": int(node_plan.node_index),
        "parent_indices": [int(parent_index) for parent_index in node_plan.parent_indices],
        "converter_specs": [
            {
                "key": str(spec.key),
                "kind": str(spec.kind),
                "dim": int(spec.dim),
                "cardinality": None if spec.cardinality is None else int(spec.cardinality),
                "column_start": int(spec.column_start),
                "column_end": int(spec.column_end),
            }
            for spec in node_plan.converter_specs
        ],
        "converter_plans": [_converter_plan_payload(plan) for plan in node_plan.converter_plans],
        "latent": {
            "required_dim": int(node_plan.latent.required_dim),
            "extra_dim": int(node_plan.latent.extra_dim),
            "total_dim": int(node_plan.latent.total_dim),
        },
        "execution_contract": str(execution_contract),
        "converter_groups": _converter_group_payloads(node_plan),
    }
    source = node_plan.source
    if isinstance(source, RandomPointsNodeSource):
        payload["source_kind"] = "random_points"
        payload["base_kind"] = str(source.base_kind)
        payload["function"] = _function_plan_payload(source.function)
        return payload
    payload["source_kind"] = "multi"
    if isinstance(source, ConcatNodeSource):
        payload["combine_kind"] = "concat"
        payload["function"] = _function_plan_payload(source.function)
        return payload
    payload["combine_kind"] = "stack"
    payload["aggregation_kind"] = str(source.aggregation_kind)
    payload["parent_functions"] = [_function_plan_payload(plan) for plan in source.parent_functions]
    return payload


def _signature_node_plan_payload(node_plan: FixedLayoutNodePlan) -> dict[str, Any]:
    payload = _node_plan_payload(
        node_plan,
        execution_contract=DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT,
    )
    payload.pop("converter_groups", None)
    payload.pop("execution_contract", None)
    stripped_specs: list[dict[str, Any]] = []
    for spec in list(payload["converter_specs"]):
        spec_payload = dict(spec)
        spec_payload.pop("column_start", None)
        spec_payload.pop("column_end", None)
        stripped_specs.append(spec_payload)
    payload["converter_specs"] = stripped_specs
    return payload


def _converter_plan_payload(plan: FixedLayoutConverterPlan) -> dict[str, Any]:
    if isinstance(plan, NumericConverterPlan):
        return {
            "kind": str(plan.kind),
            "warp_enabled": bool(plan.warp_enabled),
        }
    payload: dict[str, Any] = {
        "kind": str(plan.kind),
        "method": str(plan.method),
        "variant": str(plan.variant),
    }
    if plan.function is not None:
        payload["function"] = _function_plan_payload(plan.function)
    return payload


def _converter_group_payloads(node_plan: FixedLayoutNodePlan) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for group in node_plan.converter_groups:
        if isinstance(group, NumericConverterGroup):
            payloads.append(
                {
                    "group_kind": "numeric",
                    "group_key": "numeric",
                    "spec_indices": [int(value) for value in group.spec_indices],
                }
            )
            continue

        first_index = int(group.spec_indices[0])
        spec = node_plan.converter_specs[first_index]
        plan = node_plan.converter_plans[first_index]
        if not isinstance(plan, CategoricalConverterPlan):
            raise ValueError("Categorical converter group must reference categorical plans.")
        payloads.append(
            {
                "group_kind": "categorical",
                "group_key": json.dumps(
                    {
                        "kind": str(spec.kind),
                        "method": str(plan.method),
                        "variant": str(plan.variant),
                        "dim": int(spec.dim),
                        "cardinality": (
                            None if spec.cardinality is None else int(spec.cardinality)
                        ),
                        "function_signature": _converter_function_signature(plan.function),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "method": str(plan.method),
                "variant": str(plan.variant),
                "dim": int(spec.dim),
                "cardinality": None if spec.cardinality is None else int(spec.cardinality),
                "function_signature": _converter_function_signature(plan.function),
                "spec_indices": [int(value) for value in group.spec_indices],
            }
        )
    return payloads


def _function_plan_payload(plan: FixedLayoutFunctionPlan) -> dict[str, Any]:
    if isinstance(plan, LinearFunctionPlan):
        return {"family": "linear", "matrix": _matrix_plan_payload(plan.matrix)}
    if isinstance(plan, QuadraticFunctionPlan):
        return {"family": "quadratic", "matrix": _matrix_plan_payload(plan.matrix)}
    if isinstance(plan, NeuralNetFunctionPlan):
        payload: dict[str, Any] = {
            "family": "nn",
            "n_layers": int(plan.n_layers),
            "hidden_width": int(plan.hidden_width),
            "apply_input_activation": bool(plan.input_activation is not None),
            "apply_output_activation": bool(plan.output_activation is not None),
            "layer_matrices": [_matrix_plan_payload(matrix) for matrix in plan.layer_matrices],
            "hidden_activations": [
                _activation_plan_payload(activation) for activation in plan.hidden_activations
            ],
        }
        if plan.input_activation is not None:
            payload["input_activation"] = _activation_plan_payload(plan.input_activation)
        if plan.output_activation is not None:
            payload["output_activation"] = _activation_plan_payload(plan.output_activation)
        return payload
    if isinstance(plan, TreeFunctionPlan):
        return {
            "family": "tree",
            "n_trees": int(plan.n_trees),
            "depths": [int(depth) for depth in plan.depths],
        }
    if isinstance(plan, DiscretizationFunctionPlan):
        return {
            "family": "discretization",
            "n_centers": int(plan.n_centers),
            "linear_matrix": _matrix_plan_payload(plan.linear_matrix),
        }
    if isinstance(plan, GpFunctionPlan):
        return {
            "family": "gp",
            "branch_kind": str(plan.branch_kind),
            "variant": str(plan.variant),
        }
    if isinstance(plan, EmFunctionPlan):
        return {
            "family": "em",
            "m_val": int(plan.m_val),
            "linear_matrix": _matrix_plan_payload(plan.linear_matrix),
        }
    if isinstance(plan, PiecewiseFunctionPlan):
        return {
            "family": "piecewise",
            "gate_matrix": _matrix_plan_payload(plan.gate_matrix),
            "gate_bias": float(plan.gate_bias),
            "gate_temperature": float(plan.gate_temperature),
            "lhs": _function_plan_payload(plan.lhs),
            "rhs": _function_plan_payload(plan.rhs),
        }
    return {
        "family": "product",
        "lhs": _function_plan_payload(plan.lhs),
        "rhs": _function_plan_payload(plan.rhs),
    }


def function_plan_family_counts(plan: FixedLayoutFunctionPlan) -> dict[MechanismFamily, int]:
    """Count realized mechanism families within one function plan tree."""

    counts: dict[MechanismFamily, int] = {}

    def _increment(family: MechanismFamily) -> None:
        counts[family] = int(counts.get(family, 0)) + 1

    if isinstance(plan, ProductFunctionPlan):
        _increment("product")
        for nested_plan in (plan.lhs, plan.rhs):
            for family, count in function_plan_family_counts(nested_plan).items():
                counts[family] = int(counts.get(family, 0)) + int(count)
        return counts
    if isinstance(plan, PiecewiseFunctionPlan):
        _increment("piecewise")
        for nested_plan in (plan.lhs, plan.rhs):
            for family, count in function_plan_family_counts(nested_plan).items():
                counts[family] = int(counts.get(family, 0)) + int(count)
        return counts

    _increment(plan.family)
    return counts


def execution_plan_family_counts(
    execution_plan: FixedLayoutExecutionPlan,
) -> dict[MechanismFamily, int]:
    """Count realized mechanism families across one fixed-layout execution plan."""

    counts: dict[MechanismFamily, int] = {}

    def _merge(plan: FixedLayoutFunctionPlan | None) -> None:
        if plan is None:
            return
        for family, count in function_plan_family_counts(plan).items():
            counts[family] = int(counts.get(family, 0)) + int(count)

    for node_plan in execution_plan.node_plans:
        source = node_plan.source
        if isinstance(source, (RandomPointsNodeSource, ConcatNodeSource)):
            _merge(source.function)
        else:
            for plan in source.parent_functions:
                _merge(plan)
        for converter_plan in node_plan.converter_plans:
            if isinstance(converter_plan, CategoricalConverterPlan):
                _merge(converter_plan.function)

    return {family: int(counts[family]) for family in sorted(counts)}


def function_plan_variant_counts(plan: FixedLayoutFunctionPlan) -> dict[str, int]:
    """Count realized mechanism variants within one function plan tree."""

    counts: dict[str, int] = {}

    def _increment(label: str) -> None:
        counts[label] = int(counts.get(label, 0)) + 1

    if isinstance(plan, ProductFunctionPlan):
        for nested_plan in (plan.lhs, plan.rhs):
            for label, count in function_plan_variant_counts(nested_plan).items():
                counts[label] = int(counts.get(label, 0)) + int(count)
        return counts
    if isinstance(plan, PiecewiseFunctionPlan):
        for nested_plan in (plan.lhs, plan.rhs):
            for label, count in function_plan_variant_counts(nested_plan).items():
                counts[label] = int(counts.get(label, 0)) + int(count)
        return counts
    if isinstance(plan, GpFunctionPlan):
        _increment(f"gp.{plan.variant}")
    return counts


def execution_plan_variant_counts(
    execution_plan: FixedLayoutExecutionPlan,
) -> dict[str, int]:
    """Count realized mechanism variants across one fixed-layout execution plan."""

    counts: dict[str, int] = {}

    def _merge(plan: FixedLayoutFunctionPlan | None) -> None:
        if plan is None:
            return
        for label, count in function_plan_variant_counts(plan).items():
            counts[label] = int(counts.get(label, 0)) + int(count)

    for node_plan in execution_plan.node_plans:
        source = node_plan.source
        if isinstance(source, (RandomPointsNodeSource, ConcatNodeSource)):
            _merge(source.function)
        else:
            for plan in source.parent_functions:
                _merge(plan)
        for converter_plan in node_plan.converter_plans:
            if isinstance(converter_plan, CategoricalConverterPlan):
                _merge(converter_plan.function)

    return {label: int(counts[label]) for label in sorted(counts)}


def _matrix_plan_payload(plan: FixedLayoutMatrixPlan) -> dict[str, Any]:
    if isinstance(plan, GaussianMatrixPlan):
        return {"kind": "gaussian"}
    if isinstance(plan, WeightsMatrixPlan):
        return {"kind": "weights"}
    if isinstance(plan, SingularValuesMatrixPlan):
        return {"kind": "singular_values"}
    if isinstance(plan, KernelMatrixPlan):
        return {"kind": "kernel"}
    return {
        "kind": "activation",
        "base_kind": str(plan.base_kind),
        "activation": _activation_plan_payload(plan.activation),
    }


def _activation_plan_payload(plan: FixedLayoutActivationPlan) -> dict[str, Any]:
    if isinstance(plan, FixedActivationPlan):
        return {"mode": "fixed", "name": str(plan.name)}
    payload: dict[str, Any] = {
        "mode": "parametric",
        "kind": str(plan.kind),
    }
    if plan.poly_power is not None:
        payload["poly_power"] = int(plan.poly_power)
    if plan.temperature is not None:
        payload["temperature"] = float(plan.temperature)
    return payload


__all__ = [
    "DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT",
    "ActivationMatrixPlan",
    "CategoricalConverterGroup",
    "CategoricalConverterPlan",
    "ConcatNodeSource",
    "DiscretizationFunctionPlan",
    "EmFunctionPlan",
    "FixedActivationPlan",
    "FixedLayoutActivationPlan",
    "FixedLayoutConverterGroup",
    "FixedLayoutConverterPlan",
    "FixedLayoutConverterSpec",
    "FixedLayoutExecutionPlan",
    "FixedLayoutFunctionPlan",
    "FixedLayoutGpVariant",
    "FixedLayoutLatentPlan",
    "FixedLayoutMatrixPlan",
    "FixedLayoutNodePlan",
    "FixedLayoutNodeSource",
    "GaussianMatrixPlan",
    "GpFunctionPlan",
    "KernelMatrixPlan",
    "LinearFunctionPlan",
    "NeuralNetFunctionPlan",
    "NumericConverterGroup",
    "NumericConverterPlan",
    "ParametricActivationPlan",
    "PiecewiseFunctionPlan",
    "ProductFunctionPlan",
    "QuadraticFunctionPlan",
    "RandomPointsNodeSource",
    "SingularValuesMatrixPlan",
    "StackedNodeSource",
    "TreeFunctionPlan",
    "WeightsMatrixPlan",
    "execution_plan_family_counts",
    "execution_plan_variant_counts",
    "fixed_layout_converter_groups",
    "fixed_layout_signature_payloads",
    "function_plan_family_counts",
    "function_plan_variant_counts",
]
