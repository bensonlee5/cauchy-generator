"""Shared typed execution-plan sampling helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, cast

import torch

from dagzoo.core import execution_sampling_common as _sampling_common
from dagzoo.core.fixed_layout.plan_types import (
    ActivationMatrixPlan,
    CategoricalConverterPlan,
    ConcatNodeSource,
    DiscretizationFunctionPlan,
    EmFunctionPlan,
    FixedActivationPlan,
    FixedLayoutActivationKind,
    FixedLayoutActivationPlan,
    FixedLayoutConverterMethod,
    FixedLayoutConverterPlan,
    FixedLayoutConverterSpec,
    FixedLayoutConverterVariant,
    FixedLayoutFunctionPlan,
    FixedLayoutGpVariant,
    FixedLayoutLatentPlan,
    FixedLayoutMatrixBaseKind,
    FixedLayoutMatrixPlan,
    FixedLayoutNodePlan,
    FixedLayoutRootBaseKind,
    GaussianMatrixPlan,
    GpFunctionPlan,
    KernelMatrixPlan,
    LinearFunctionPlan,
    NeuralNetFunctionPlan,
    NumericConverterPlan,
    ParametricActivationPlan,
    PiecewiseFunctionPlan,
    ProductFunctionPlan,
    QuadraticFunctionPlan,
    RandomPointsNodeSource,
    SingularValuesMatrixPlan,
    StackedNodeSource,
    TreeFunctionPlan,
    WeightsMatrixPlan,
    fixed_layout_converter_groups,
)
from dagzoo.core.layout_types import AggregationKind, ConverterKind, MechanismFamily
from dagzoo.core.shift import (
    MECHANISM_FAMILY_ORDER,
    MECHANISM_FAMILY_SUPPORTED_ORDER,
    mechanism_family_probabilities,
)
from dagzoo.core.validation import (
    RetryableDegeneracyError,
    validate_converter_plan_nondegeneracy,
    validate_function_plan_nondegeneracy,
    validate_node_plan_nondegeneracy,
)
from dagzoo.functions.activations import fixed_activation_names
from dagzoo.math import log_uniform as _log_uniform
from dagzoo.rng import KeyedRng
from dagzoo.sampling.correlated import sample_correlated_choice, sample_correlated_num

_generator_device = _sampling_common._generator_device
_rand_scalar = _sampling_common._rand_scalar
_randint_scalar = _sampling_common._randint_scalar
_resolve_sampling_generator = _sampling_common._resolve_sampling_generator
_resolve_sampling_root = _sampling_common._resolve_sampling_root
_sample_bool = _sampling_common._sample_bool

_MATRIX_KIND_CHOICES: tuple[str, ...] = (
    "gaussian",
    "weights",
    "singular_values",
    "kernel",
    "activation",
)
_MATRIX_BASE_KIND_CHOICES: tuple[FixedLayoutMatrixBaseKind, ...] = (
    "gaussian",
    "weights",
    "singular_values",
    "kernel",
)
_ROOT_BASE_KIND_CHOICES: tuple[FixedLayoutRootBaseKind, ...] = (
    "normal",
    "uniform",
    "unit_ball",
    "normal_cov",
)
_PARAM_ACTIVATION_CHOICES: tuple[FixedLayoutActivationKind, ...] = (
    "relu_pow",
    "signed_pow",
    "inv_pow",
    "poly",
    "gumbel_softmax",
)
_GP_VARIANT_CHOICES: tuple[FixedLayoutGpVariant, ...] = (
    "standard",
    "periodic",
    "multiscale",
)
_WIDTH_ONE_INVALID_FIXED_ACTIVATIONS: frozenset[str] = frozenset(
    {"softmax", "onehot_argmax", "argsort", "rank"}
)
_WIDTH_ONE_INVALID_PARAMETRIC_ACTIVATIONS: frozenset[str] = frozenset({"gumbel_softmax"})
_AGGREGATION_KIND_ORDER: tuple[AggregationKind, ...] = ("sum", "product", "max", "logsumexp")
_PRODUCT_COMPONENT_FAMILIES: tuple[MechanismFamily, ...] = (
    "tree",
    "discretization",
    "gp",
    "linear",
    "quadratic",
)
_JOINT_VARIANTS: tuple[tuple[FixedLayoutConverterMethod, FixedLayoutConverterVariant], ...] = (
    ("neighbor", "input"),
    ("neighbor", "index_repeat"),
    ("neighbor", "center"),
    ("neighbor", "center_random_fn"),
    ("softmax", "input"),
    ("softmax", "index_repeat"),
    ("softmax", "softmax_points"),
)
_COMPOSITIONAL_STRESS_PROFILE = "anti_memorization_piecewise_classification_compositional_slice_v1"


def _matrix_kernel_correlation_enabled(stress_profile_name: str | None) -> bool:
    return str(stress_profile_name) == _COMPOSITIONAL_STRESS_PROFILE


class ConverterSpecLike(Protocol):
    """Minimal protocol shared by scalar and fixed-layout converter specs."""

    @property
    def key(self) -> str: ...

    @property
    def kind(self) -> ConverterKind: ...

    @property
    def dim(self) -> int: ...

    @property
    def cardinality(self) -> int | None: ...


ConverterSpecsInput = Sequence[ConverterSpecLike] | Sequence[FixedLayoutConverterSpec]


def _activation_plan_label(plan: FixedLayoutActivationPlan) -> str:
    if isinstance(plan, FixedActivationPlan):
        return str(plan.name)
    return str(plan.kind)


def _activation_plan_is_width_compatible(
    plan: FixedLayoutActivationPlan,
    *,
    width: int | None,
) -> bool:
    if width is None or int(width) != 1:
        return True
    if isinstance(plan, FixedActivationPlan):
        return str(plan.name) not in _WIDTH_ONE_INVALID_FIXED_ACTIVATIONS
    return str(plan.kind) not in _WIDTH_ONE_INVALID_PARAMETRIC_ACTIVATIONS


def _validate_activation_plan_width(
    plan: FixedLayoutActivationPlan,
    *,
    width: int | None,
    context: str,
) -> None:
    if _activation_plan_is_width_compatible(plan, width=width):
        return
    raise RetryableDegeneracyError(
        "width_incompatible_activation",
        message=(
            f"{context} activation {_activation_plan_label(plan)!r} is invalid for width=1 "
            "because it deterministically collapses the channel dimension."
        ),
    )


def _validate_neural_net_function_plan_widths(
    plan: NeuralNetFunctionPlan,
    *,
    input_dim: int | None,
    out_dim: int,
) -> None:
    hidden_width = max(1, int(plan.hidden_width))
    if plan.input_activation is not None:
        _validate_activation_plan_width(
            plan.input_activation,
            width=input_dim,
            context="NeuralNetFunctionPlan.input_activation",
        )
    for hidden_index, hidden_activation in enumerate(plan.hidden_activations):
        _validate_activation_plan_width(
            hidden_activation,
            width=hidden_width,
            context=f"NeuralNetFunctionPlan.hidden_activations[{hidden_index}]",
        )
    if plan.output_activation is not None:
        _validate_activation_plan_width(
            plan.output_activation,
            width=int(out_dim),
            context="NeuralNetFunctionPlan.output_activation",
        )


def sample_function_family(
    generator: torch.Generator | None = None,
    *,
    keyed_rng: KeyedRng | None = None,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None = None,
    families: Sequence[MechanismFamily] | None = None,
    device: str | None = None,
) -> MechanismFamily:
    """Sample one mechanism family with optional logit tilt."""

    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="sample_function_family",
    )
    family_order = tuple(
        families
        if families is not None
        else (
            MECHANISM_FAMILY_SUPPORTED_ORDER
            if function_family_mix is not None
            else MECHANISM_FAMILY_ORDER
        )
    )
    if not family_order:
        raise ValueError("At least one mechanism family must be available for sampling.")
    if mechanism_logit_tilt <= 0.0 and function_family_mix is None:
        return sample_correlated_choice(
            keyed_rng,
            name="mechanism_family",
            values=family_order,
            device=resolved_device,
        )

    probs_by_family = mechanism_family_probabilities(
        mechanism_logit_tilt=mechanism_logit_tilt,
        families=family_order,
        family_weights=function_family_mix,
    )
    positive_families = tuple(
        family for family in family_order if float(probs_by_family.get(family, 0.0)) > 0.0
    )
    if not positive_families:
        raise ValueError("No eligible mechanism families are available for sampling.")
    return sample_correlated_choice(
        keyed_rng,
        name="mechanism_family",
        values=positive_families,
        device=resolved_device,
        base_probs=tuple(float(probs_by_family[family]) for family in positive_families),
    )


def _higher_order_component_mix(
    family: Literal["product", "piecewise"],
    function_family_mix: dict[MechanismFamily, float] | None,
) -> dict[MechanismFamily, float]:
    if function_family_mix is None:
        return {family: 1.0 for family in _PRODUCT_COMPONENT_FAMILIES}
    filtered = {
        family: float(weight)
        for family, weight in function_family_mix.items()
        if family in _PRODUCT_COMPONENT_FAMILIES and float(weight) > 0.0
    }
    if not filtered:
        raise ValueError(
            f"mechanism.function_family_mix enables '{family}' but disables all {family} "
            "component families for fixed-layout plan sampling."
        )
    return filtered


def _sample_product_component_family(
    generator: torch.Generator | None = None,
    *,
    keyed_rng: KeyedRng | None = None,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
    device: str | None = None,
) -> MechanismFamily:
    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="sample_product_component_family",
    )
    component_mix = _higher_order_component_mix("product", function_family_mix)
    family = sample_function_family(
        keyed_rng=keyed_rng,
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=component_mix,
        families=_PRODUCT_COMPONENT_FAMILIES,
        device=resolved_device,
    )
    if family == "product":
        raise ValueError("Product subplans must resolve to non-product mechanism families.")
    return family


def _sample_piecewise_component_family(
    generator: torch.Generator | None = None,
    *,
    keyed_rng: KeyedRng | None = None,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
    device: str | None = None,
) -> MechanismFamily:
    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="sample_piecewise_component_family",
    )
    component_mix = _higher_order_component_mix("piecewise", function_family_mix)
    return sample_function_family(
        keyed_rng=keyed_rng,
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=component_mix,
        families=_PRODUCT_COMPONENT_FAMILIES,
        device=resolved_device,
    )


def sample_activation_plan(
    generator: torch.Generator | None = None,
    *,
    keyed_rng: KeyedRng | None = None,
    device: str | None = None,
    width: int | None = None,
) -> FixedLayoutActivationPlan:
    """Sample one activation plan using the shared fixed-layout schema."""

    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="sample_activation_plan",
    )
    generator = keyed_rng.torch_rng(device=resolved_device)
    max_attempts = max(8, len(fixed_activation_names()) + len(_PARAM_ACTIVATION_CHOICES))
    for _attempt in range(max_attempts):
        if _rand_scalar(generator) < (1.0 / 3.0):
            choice = _PARAM_ACTIVATION_CHOICES[
                int(_randint_scalar(0, len(_PARAM_ACTIVATION_CHOICES), generator))
            ]
            if choice == "poly":
                plan: FixedLayoutActivationPlan = ParametricActivationPlan(
                    kind=choice,
                    poly_power=int(_randint_scalar(2, 6, generator)),
                )
            elif choice == "gumbel_softmax":
                plan = ParametricActivationPlan(
                    kind=choice,
                    temperature=float(
                        _log_uniform(
                            keyed_rng.keyed("gumbel_softmax_temperature").torch_rng(
                                device=resolved_device
                            ),
                            0.25,
                            4.0,
                            resolved_device,
                        )
                    ),
                )
            else:
                plan = ParametricActivationPlan(kind=choice)
        else:
            fixed = fixed_activation_names()
            name = fixed[int(_randint_scalar(0, len(fixed), generator))]
            plan = FixedActivationPlan(name=name)
        if _activation_plan_is_width_compatible(plan, width=width):
            return plan
    raise ValueError(f"Failed to sample a width-compatible activation plan for width={width}.")


def sample_matrix_plan(
    generator: torch.Generator | None = None,
    *,
    keyed_rng: KeyedRng | None = None,
    device: str | None = None,
    stress_profile_name: str | None = None,
) -> FixedLayoutMatrixPlan:
    """Sample one matrix-family plan."""

    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="sample_matrix_plan",
    )
    correlated = _matrix_kernel_correlation_enabled(stress_profile_name)
    if correlated:
        kind = cast(
            str,
            sample_correlated_choice(
                keyed_rng.keyed("kind"),
                name="matrix_family",
                values=_MATRIX_KIND_CHOICES,
                device=resolved_device,
            ),
        )
    else:
        generator = keyed_rng.keyed("kind").torch_rng(device=resolved_device)
        kind = _MATRIX_KIND_CHOICES[int(_randint_scalar(0, len(_MATRIX_KIND_CHOICES), generator))]
    if kind == "gaussian":
        return GaussianMatrixPlan()
    if kind == "weights":
        return WeightsMatrixPlan()
    if kind == "singular_values":
        return SingularValuesMatrixPlan()
    if kind == "kernel":
        gamma = (
            float(
                sample_correlated_num(
                    keyed_rng.keyed("gamma"),
                    name="kernel_gamma",
                    low=0.1,
                    high=10.0,
                    device=resolved_device,
                    log_scale=True,
                )
            )
            if correlated
            else float(
                _log_uniform(
                    keyed_rng.keyed("gamma").torch_rng(device=resolved_device),
                    0.1,
                    10.0,
                    resolved_device,
                )
            )
        )
        signed = (
            bool(
                sample_correlated_choice(
                    keyed_rng.keyed("signed"),
                    name="kernel_signed",
                    values=(False, True),
                    device=resolved_device,
                )
            )
            if correlated
            else bool(_sample_bool(keyed_rng.keyed("signed").torch_rng(device=resolved_device)))
        )
        return KernelMatrixPlan(gamma=gamma, signed=signed)
    if correlated:
        base_kind = cast(
            FixedLayoutMatrixBaseKind,
            sample_correlated_choice(
                keyed_rng.keyed("base_kind"),
                name="activation_matrix_base_kind",
                values=_MATRIX_BASE_KIND_CHOICES,
                device=resolved_device,
            ),
        )
    else:
        base_generator = keyed_rng.keyed("base_kind").torch_rng(device=resolved_device)
        base_kind = _MATRIX_BASE_KIND_CHOICES[
            int(_randint_scalar(0, len(_MATRIX_BASE_KIND_CHOICES), base_generator))
        ]
    return ActivationMatrixPlan(
        base_kind=base_kind,
        activation=sample_activation_plan(
            keyed_rng=keyed_rng.keyed("activation"),
            device=resolved_device,
        ),
    )


def sample_function_plan_for_family(
    generator: torch.Generator | None = None,
    *,
    keyed_rng: KeyedRng | None = None,
    family: MechanismFamily,
    input_dim: int | None = None,
    out_dim: int,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
    device: str | None = None,
    stress_profile_name: str | None = None,
) -> FixedLayoutFunctionPlan:
    """Sample one typed function plan for an explicit family."""

    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="sample_function_plan_for_family",
    )
    last_error: RetryableDegeneracyError | None = None
    for attempt in range(8):
        attempt_root = keyed_rng if attempt == 0 else keyed_rng.keyed("retry", attempt)
        try:
            plan = _sample_function_plan_for_family_once(
                keyed_rng=attempt_root,
                family=family,
                input_dim=input_dim,
                out_dim=out_dim,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            )
            validate_function_plan_nondegeneracy(plan)
            return plan
        except RetryableDegeneracyError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Function-plan sampling exhausted without yielding or erroring.")


def _sample_function_plan_for_family_once(
    generator: torch.Generator | None = None,
    *,
    keyed_rng: KeyedRng | None = None,
    family: MechanismFamily,
    input_dim: int | None = None,
    out_dim: int,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
    device: str | None = None,
    stress_profile_name: str | None = None,
) -> FixedLayoutFunctionPlan:
    """Sample one typed function plan for an explicit family without retries."""

    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="_sample_function_plan_for_family_once",
    )
    if family == "linear":
        return LinearFunctionPlan(
            matrix=sample_matrix_plan(
                keyed_rng=keyed_rng.keyed("matrix"),
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            )
        )
    if family == "quadratic":
        return QuadraticFunctionPlan(
            matrix=sample_matrix_plan(
                keyed_rng=keyed_rng.keyed("matrix"),
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            )
        )
    if family == "nn":
        n_layers = int(
            _randint_scalar(
                1,
                4,
                keyed_rng.keyed("n_layers").torch_rng(device=resolved_device),
            )
        )
        hidden_width = int(
            _log_uniform(
                keyed_rng.keyed("hidden_width").torch_rng(device=resolved_device),
                1.0,
                127.0,
                resolved_device,
            )
        )
        input_activation = (
            sample_activation_plan(
                keyed_rng=keyed_rng.keyed("input_activation"),
                device=resolved_device,
                width=input_dim,
            )
            if _sample_bool(
                keyed_rng.keyed("input_activation_enabled").torch_rng(device=resolved_device)
            )
            else None
        )
        output_activation = (
            sample_activation_plan(
                keyed_rng=keyed_rng.keyed("output_activation"),
                device=resolved_device,
                width=int(out_dim),
            )
            if _sample_bool(
                keyed_rng.keyed("output_activation_enabled").torch_rng(device=resolved_device)
            )
            else None
        )
        layer_count = max(1, n_layers)
        plan = NeuralNetFunctionPlan(
            n_layers=n_layers,
            hidden_width=max(1, hidden_width),
            input_activation=input_activation,
            output_activation=output_activation,
            layer_matrices=tuple(
                sample_matrix_plan(
                    keyed_rng=keyed_rng.keyed("layer_matrix", layer_index),
                    device=resolved_device,
                    stress_profile_name=stress_profile_name,
                )
                for layer_index in range(layer_count)
            ),
            hidden_activations=tuple(
                sample_activation_plan(
                    keyed_rng=keyed_rng.keyed("hidden_activation", layer_index),
                    device=resolved_device,
                    width=max(1, hidden_width),
                )
                for layer_index in range(max(0, layer_count - 1))
            ),
        )
        _validate_neural_net_function_plan_widths(
            plan,
            input_dim=input_dim,
            out_dim=int(out_dim),
        )
        return plan
    if family == "tree":
        n_trees = int(
            _log_uniform(
                keyed_rng.keyed("n_trees").torch_rng(device=resolved_device),
                1.0,
                32.0,
                resolved_device,
            )
        )
        n_trees = max(1, n_trees)
        return TreeFunctionPlan(
            n_trees=n_trees,
            depths=tuple(
                int(
                    _randint_scalar(
                        1,
                        8,
                        keyed_rng.keyed("depth", tree_index).torch_rng(device=resolved_device),
                    )
                )
                for tree_index in range(n_trees)
            ),
        )
    if family == "discretization":
        n_centers = int(
            _log_uniform(
                keyed_rng.keyed("n_centers").torch_rng(device=resolved_device),
                2.0,
                128.0,
                resolved_device,
            )
        )
        return DiscretizationFunctionPlan(
            n_centers=max(2, n_centers),
            linear_matrix=sample_matrix_plan(
                keyed_rng=keyed_rng.keyed("linear_matrix"),
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            ),
        )
    if family == "gp":
        branch_kind = sample_correlated_choice(
            keyed_rng.keyed("branch_kind"),
            name="gp_branch_kind",
            values=("ha", "projected"),
            device=resolved_device,
        )
        variant = sample_correlated_choice(
            keyed_rng.keyed("variant"),
            name="gp_variant",
            values=_GP_VARIANT_CHOICES,
            device=resolved_device,
        )
        return GpFunctionPlan(
            branch_kind=cast(Literal["ha", "projected"], branch_kind),
            variant=cast(FixedLayoutGpVariant, variant),
        )
    if family == "em":
        m_val = int(
            _log_uniform(
                keyed_rng.keyed("m_val").torch_rng(device=resolved_device),
                2.0,
                float(max(16, 2 * out_dim)),
                resolved_device,
            )
        )
        return EmFunctionPlan(
            m_val=max(2, m_val),
            linear_matrix=sample_matrix_plan(
                keyed_rng=keyed_rng.keyed("linear_matrix"),
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            ),
        )
    if family == "product":
        lhs_root = keyed_rng.keyed("lhs")
        rhs_root = keyed_rng.keyed("rhs")
        lhs_family = _sample_product_component_family(
            keyed_rng=lhs_root.keyed("family"),
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            device=resolved_device,
        )
        rhs_family = _sample_product_component_family(
            keyed_rng=rhs_root.keyed("family"),
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            device=resolved_device,
        )
        return ProductFunctionPlan(
            lhs=sample_function_plan_for_family(
                keyed_rng=lhs_root.keyed("plan"),
                family=lhs_family,
                input_dim=input_dim,
                out_dim=out_dim,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            ),
            rhs=sample_function_plan_for_family(
                keyed_rng=rhs_root.keyed("plan"),
                family=rhs_family,
                input_dim=input_dim,
                out_dim=out_dim,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            ),
        )
    if family == "piecewise":
        lhs_root = keyed_rng.keyed("lhs")
        rhs_root = keyed_rng.keyed("rhs")
        lhs_family = _sample_piecewise_component_family(
            keyed_rng=lhs_root.keyed("family"),
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            device=resolved_device,
        )
        rhs_family = _sample_piecewise_component_family(
            keyed_rng=rhs_root.keyed("family"),
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            device=resolved_device,
        )
        gate_temperature = float(
            _log_uniform(
                keyed_rng.keyed("gate_temperature").torch_rng(device=resolved_device),
                2.0,
                16.0,
                resolved_device,
            )
        )
        gate_bias_generator = keyed_rng.keyed("gate_bias").torch_rng(device=resolved_device)
        gate_bias = (2.0 * _rand_scalar(gate_bias_generator)) - 1.0
        gate_bias *= 1.5
        return PiecewiseFunctionPlan(
            gate_matrix=sample_matrix_plan(
                keyed_rng=keyed_rng.keyed("gate_matrix"),
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            ),
            gate_bias=gate_bias,
            gate_temperature=gate_temperature,
            lhs=sample_function_plan_for_family(
                keyed_rng=lhs_root.keyed("plan"),
                family=lhs_family,
                input_dim=input_dim,
                out_dim=out_dim,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            ),
            rhs=sample_function_plan_for_family(
                keyed_rng=rhs_root.keyed("plan"),
                family=rhs_family,
                input_dim=input_dim,
                out_dim=out_dim,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            ),
        )
    raise ValueError(f"Unsupported mechanism family in fixed-layout plan sampling: {family!r}")


def sample_function_plan(
    generator: torch.Generator | None = None,
    *,
    keyed_rng: KeyedRng | None = None,
    input_dim: int | None = None,
    out_dim: int,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
    device: str | None = None,
    stress_profile_name: str | None = None,
) -> FixedLayoutFunctionPlan:
    """Sample one typed function plan using the shared family sampler."""

    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="sample_function_plan",
    )
    family = sample_function_family(
        keyed_rng=keyed_rng.keyed("family"),
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=function_family_mix,
        device=resolved_device,
    )
    return sample_function_plan_for_family(
        keyed_rng=keyed_rng.keyed("plan"),
        family=family,
        input_dim=input_dim,
        out_dim=out_dim,
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=function_family_mix,
        device=resolved_device,
        stress_profile_name=stress_profile_name,
    )


def sample_converter_plan(
    spec: ConverterSpecLike,
    generator: torch.Generator | None = None,
    *,
    keyed_rng: KeyedRng | None = None,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
    method_override: str | None = None,
    device: str | None = None,
    stress_profile_name: str | None = None,
) -> FixedLayoutConverterPlan:
    """Sample one typed converter plan for a converter spec."""

    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="sample_converter_plan",
    )
    if spec.kind in {"num", "target_reg"}:
        warp_generator = keyed_rng.keyed("warp_enabled").torch_rng(device=resolved_device)
        return NumericConverterPlan(
            kind=cast(Literal["num", "target_reg"], spec.kind),
            warp_enabled=not _sample_bool(warp_generator),
        )

    selected_method_raw, variant_raw = sample_correlated_choice(
        keyed_rng.keyed("joint_variant"),
        name="converter_joint_variant",
        values=_JOINT_VARIANTS,
        device=resolved_device,
    )
    if method_override is None:
        selected_method = selected_method_raw
    else:
        normalized_method = method_override.strip().lower()
        if normalized_method not in {"neighbor", "softmax"}:
            raise ValueError(f"Unsupported categorical converter method: {normalized_method!r}.")
        selected_method = cast(FixedLayoutConverterMethod, normalized_method)
    variant = cast(FixedLayoutConverterVariant, variant_raw)
    if variant == "center_random_fn":
        plan = CategoricalConverterPlan(
            kind=cast(Literal["cat", "target_cls"], spec.kind),
            method=selected_method,
            variant=variant,
            function=sample_function_plan(
                keyed_rng=keyed_rng.keyed("function"),
                input_dim=max(1, int(spec.dim)),
                out_dim=max(1, int(spec.dim)),
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            ),
        )
        validate_converter_plan_nondegeneracy(plan)
        return plan
    plan = CategoricalConverterPlan(
        kind=cast(Literal["cat", "target_cls"], spec.kind),
        method=selected_method,
        variant=variant,
    )
    validate_converter_plan_nondegeneracy(plan)
    return plan


def typed_converter_specs(
    converter_specs: ConverterSpecsInput,
) -> tuple[FixedLayoutConverterSpec, ...]:
    """Lift scalar converter specs into typed fixed-layout converter specs."""

    typed_specs: list[FixedLayoutConverterSpec] = []
    column_cursor = 0
    for spec in converter_specs:
        spec_dim = max(1, int(spec.dim))
        typed_specs.append(
            FixedLayoutConverterSpec(
                key=str(spec.key),
                kind=spec.kind,
                dim=int(spec.dim),
                cardinality=None if spec.cardinality is None else int(spec.cardinality),
                column_start=int(column_cursor),
                column_end=int(column_cursor + spec_dim),
            )
        )
        column_cursor += spec_dim
    return tuple(typed_specs)


def sample_latent_plan(
    converter_specs: ConverterSpecsInput,
    *,
    generator: torch.Generator | None = None,
    keyed_rng: KeyedRng | None = None,
    device: str,
) -> FixedLayoutLatentPlan:
    """Sample the shared latent-width plan for one node."""

    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="sample_latent_plan",
    )
    required_dim = int(sum(max(1, int(spec.dim)) for spec in converter_specs))
    extra_dim = max(
        1,
        int(
            _log_uniform(
                keyed_rng.keyed("extra_dim").torch_rng(device=resolved_device),
                1.0,
                32.0,
                resolved_device,
            )
        ),
    )
    return FixedLayoutLatentPlan(
        required_dim=required_dim,
        extra_dim=extra_dim,
        total_dim=int(required_dim + extra_dim),
    )


def sample_root_source_plan(
    generator: torch.Generator | None = None,
    *,
    keyed_rng: KeyedRng | None = None,
    out_dim: int,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
    device: str | None = None,
    stress_profile_name: str | None = None,
) -> RandomPointsNodeSource:
    """Sample one root-source plan."""

    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="sample_root_source_plan",
    )
    if _matrix_kernel_correlation_enabled(stress_profile_name):
        base_kind = cast(
            FixedLayoutRootBaseKind,
            sample_correlated_choice(
                keyed_rng.keyed("base_kind"),
                name="root_base_kind",
                values=_ROOT_BASE_KIND_CHOICES,
                device=resolved_device,
            ),
        )
    else:
        base_generator = keyed_rng.keyed("base_kind").torch_rng(device=resolved_device)
        base_kind = _ROOT_BASE_KIND_CHOICES[
            int(_randint_scalar(0, len(_ROOT_BASE_KIND_CHOICES), base_generator))
        ]
    return RandomPointsNodeSource(
        base_kind=base_kind,
        function=sample_function_plan(
            keyed_rng=keyed_rng.keyed("function"),
            input_dim=int(out_dim),
            out_dim=out_dim,
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            device=resolved_device,
            stress_profile_name=stress_profile_name,
        ),
    )


def sample_multi_source_plan(
    generator: torch.Generator | None = None,
    *,
    keyed_rng: KeyedRng | None = None,
    parent_count: int,
    parent_input_dims: Sequence[int] | None = None,
    out_dim: int,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
    aggregation_kind: AggregationKind | None = None,
    device: str | None = None,
    stress_profile_name: str | None = None,
) -> ConcatNodeSource | StackedNodeSource:
    """Sample one shared multi-parent source plan."""

    if parent_count <= 0:
        raise ValueError(f"parent_count must be > 0, got {parent_count}")
    resolved_parent_input_dims = (
        [int(dimension) for dimension in parent_input_dims]
        if parent_input_dims is not None
        else [int(out_dim)] * int(parent_count)
    )
    if len(resolved_parent_input_dims) != int(parent_count):
        raise ValueError(
            "parent_input_dims must align with parent_count for multi-source plan sampling."
        )
    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="sample_multi_source_plan",
    )
    combine_kind = sample_correlated_choice(
        keyed_rng.keyed("combine_kind"),
        name="multi_source_combine_kind",
        values=("concat", "stack"),
        device=resolved_device,
    )
    if combine_kind == "concat":
        return ConcatNodeSource(
            function=sample_function_plan(
                keyed_rng=keyed_rng.keyed("function"),
                input_dim=int(sum(int(dimension) for dimension in resolved_parent_input_dims)),
                out_dim=out_dim,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            )
        )
    resolved_aggregation_kind = aggregation_kind
    if resolved_aggregation_kind is None:
        resolved_aggregation_kind = sample_correlated_choice(
            keyed_rng.keyed("aggregation"),
            name="multi_source_aggregation_kind",
            values=_AGGREGATION_KIND_ORDER,
            device=resolved_device,
        )
    return StackedNodeSource(
        aggregation_kind=cast(AggregationKind, resolved_aggregation_kind),
        parent_functions=tuple(
            sample_function_plan(
                keyed_rng=keyed_rng.keyed("parent", parent_index),
                input_dim=int(resolved_parent_input_dims[parent_index]),
                out_dim=out_dim,
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
                device=resolved_device,
                stress_profile_name=stress_profile_name,
            )
            for parent_index in range(parent_count)
        ),
    )


def sample_node_plan(
    *,
    node_index: int,
    parent_indices: Sequence[int],
    parent_output_dims: Sequence[int] | None = None,
    converter_specs: ConverterSpecsInput,
    generator: torch.Generator | None = None,
    keyed_rng: KeyedRng | None = None,
    device: str,
    mechanism_logit_tilt: float,
    function_family_mix: dict[MechanismFamily, float] | None,
    stress_profile_name: str | None = None,
) -> FixedLayoutNodePlan:
    """Sample one typed node execution plan."""

    keyed_rng, resolved_device = _resolve_sampling_root(
        generator=generator,
        keyed_rng=keyed_rng,
        device=device,
        namespace="sample_node_plan",
    )
    latent = sample_latent_plan(
        converter_specs,
        keyed_rng=keyed_rng.keyed("latent"),
        device=resolved_device,
    )
    typed_specs = typed_converter_specs(converter_specs)
    converter_plans = tuple(
        sample_converter_plan(
            spec,
            keyed_rng=keyed_rng.keyed("converter", spec_index),
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            device=resolved_device,
            stress_profile_name=stress_profile_name,
        )
        for spec_index, spec in enumerate(typed_specs)
    )
    source: ConcatNodeSource | StackedNodeSource | RandomPointsNodeSource
    if parent_indices:
        resolved_parent_output_dims = (
            [int(dimension) for dimension in parent_output_dims]
            if parent_output_dims is not None
            else [int(latent.total_dim)] * len(parent_indices)
        )
        if len(resolved_parent_output_dims) != len(parent_indices):
            raise ValueError(
                "parent_output_dims must align with parent_indices for node plan sampling."
            )
        source = sample_multi_source_plan(
            keyed_rng=keyed_rng.keyed("source"),
            parent_count=len(parent_indices),
            parent_input_dims=resolved_parent_output_dims,
            out_dim=int(latent.total_dim),
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            device=resolved_device,
            stress_profile_name=stress_profile_name,
        )
    else:
        source = sample_root_source_plan(
            keyed_rng=keyed_rng.keyed("source"),
            out_dim=int(latent.total_dim),
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            device=resolved_device,
            stress_profile_name=stress_profile_name,
        )
    node_plan = FixedLayoutNodePlan(
        node_index=int(node_index),
        parent_indices=tuple(int(parent_index) for parent_index in parent_indices),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=latent,
        source=source,
    )
    validate_node_plan_nondegeneracy(node_plan)
    return node_plan


__all__ = [
    "_AGGREGATION_KIND_ORDER",
    "_JOINT_VARIANTS",
    "_ROOT_BASE_KIND_CHOICES",
    "sample_activation_plan",
    "sample_converter_plan",
    "sample_function_family",
    "sample_function_plan",
    "sample_function_plan_for_family",
    "sample_latent_plan",
    "sample_matrix_plan",
    "sample_multi_source_plan",
    "sample_node_plan",
    "sample_root_source_plan",
    "typed_converter_specs",
]
