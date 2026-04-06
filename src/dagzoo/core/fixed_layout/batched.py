"""Batched fixed-layout execution-plan sampling and generation helpers."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from typing import Any, cast

import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core.execution_semantics import (
    sample_node_plan,
)
from dagzoo.core.layout import (
    _ancestor_nodes_for_target,
    _build_node_specs,
    _build_target_specs,
)
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.postprocess.postprocess import postprocess_feature_matrix
from dagzoo.rng import KeyedRng
from dagzoo.runtime_profiling import record_runtime_profile_metric
from dagzoo.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec

from .batch_common import (
    FixedLayoutBatchRng,
    _aggregate_batch_incrementally,
    _aggregate_parent_outputs_batch,
    _batch_standardize,
    _lp_distances_to_centers,
    _nearest_lp_center_indices,
    _sample_random_weights_batch,
    _sanitize_and_batch_standardize,
)
from .batch_functions import (
    _apply_activation_plan,
    _apply_discretization_batch,
    _apply_em_batch,
    _apply_gp_batch,
    _apply_leaf_pair_batch,
    _apply_linear_batch,
    _apply_nn_batch,
    _apply_piecewise_gate_with_params,
    _apply_quadratic_batch,
    _apply_sampled_leaf_function_batch,
    _apply_tree_batch,
    _concat_piecewise_gate_params,
    _concat_sampled_leaf_function_params,
    _sample_leaf_function_params_batch,
    _sample_piecewise_gate_params,
    _sample_random_matrix_from_plan_batch,
    _sample_random_points_batch,
    _SampledPiecewiseGateParams,
)
from .plan_types import (
    DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT,
    CategoricalConverterPlan,
    CompiledCategoricalConverterGroup,
    CompiledNumericConverterGroup,
    ConcatNodeSource,
    DiscretizationFunctionPlan,
    EmFunctionPlan,
    FixedActivationPlan,
    FixedLayoutActivationPlan,
    FixedLayoutCompiledConverterGroup,
    FixedLayoutCompiledConverterSlice,
    FixedLayoutConverterSpec,
    FixedLayoutExecutionPlan,
    FixedLayoutFunctionPlan,
    FixedLayoutNodePlan,
    GpFunctionPlan,
    LinearFunctionPlan,
    NeuralNetFunctionPlan,
    NumericConverterGroup,
    NumericConverterPlan,
    ParametricActivationPlan,
    PiecewiseFunctionPlan,
    ProductFunctionPlan,
    QuadraticFunctionPlan,
    RandomPointsNodeSource,
    StackedNodeSource,
    TreeFunctionPlan,
    fixed_layout_signature_payloads,
)

_FIXED_LAYOUT_EXECUTION_CONTRACT = DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT


@dataclass(frozen=True, slots=True)
class _CompiledFunctionProgramStep:
    """One compiled execution-program step for a fixed-layout function subtree."""

    plan: FixedLayoutFunctionPlan
    family_name: str
    root_rng_path: tuple[str | int, ...]
    lhs_index: int | None = None
    rhs_index: int | None = None
    gate_matrix_rng_path: tuple[str | int, ...] | None = None
    standardize_input_boundary: bool = False


@dataclass(frozen=True, slots=True)
class _CompiledFunctionExecutionContext:
    """Prepared execution metadata for one fixed-layout function subtree."""

    plan: FixedLayoutFunctionPlan
    family_name: str
    root_rng_path: tuple[str | int, ...]
    program_steps: tuple[_CompiledFunctionProgramStep, ...]
    root_program_index: int
    cached_rng_paths: tuple[tuple[str | int, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class _PreparedNodeExecutionContext:
    """Prepared execution metadata for one fixed-layout node."""

    node_plan: FixedLayoutNodePlan
    node_rng_path: tuple[str | int, ...]
    root_source_base_rng_path: tuple[str | int, ...] | None = None
    root_source_function: _CompiledFunctionExecutionContext | None = None
    concat_function: _CompiledFunctionExecutionContext | None = None
    parent_function_contexts: tuple[_CompiledFunctionExecutionContext, ...] = ()
    converter_nested_function_contexts: tuple[_CompiledFunctionExecutionContext | None, ...] = ()
    cached_rng_paths: tuple[tuple[str | int, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class _PreparedFixedLayoutExecutionContext:
    """Prepared fixed-layout execution metadata reused across runtime calls."""

    node_contexts: tuple[_PreparedNodeExecutionContext, ...]
    target_ancestor_node_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PreparedBatchNodeExecutionContext:
    """Per-batch cached node and subtree RNG handles for prepared execution."""

    node_rng: FixedLayoutBatchRng
    child_rngs: dict[tuple[str | int, ...], FixedLayoutBatchRng]


@dataclass(frozen=True, slots=True)
class _FixedLayoutBatchExecutionContext:
    """Per-batch cached RNG handles for prepared fixed-layout execution."""

    node_contexts: tuple[_PreparedBatchNodeExecutionContext, ...]


@dataclass(frozen=True, slots=True)
class _MixedPreparedLogicalCohort:
    """One exact logical cohort executed within a heterogeneous physical microbatch."""

    layout: LayoutPlan
    execution_plan: FixedLayoutExecutionPlan
    prepared_execution_context: _PreparedFixedLayoutExecutionContext
    dataset_seeds: tuple[int, ...]
    noise_spec: NoiseSamplingSpec | None
    task: str
    n_rows: int


@dataclass(frozen=True, slots=True)
class _MixedNodeRun:
    """One logical cohort segment for mixed node execution."""

    node_context: _PreparedNodeExecutionContext
    batch_node_context: _PreparedBatchNodeExecutionContext
    start: int
    stop: int
    noise_spec: NoiseSamplingSpec | None


@dataclass(frozen=True, slots=True)
class _MixedFunctionRun:
    """One logical cohort segment for mixed compiled-function execution."""

    function_context: _CompiledFunctionExecutionContext
    batch_node_context: _PreparedBatchNodeExecutionContext
    start: int
    stop: int
    noise_spec: NoiseSamplingSpec | None


def _append_compiled_function_program_step(
    plan: FixedLayoutFunctionPlan,
    *,
    root_rng_path: tuple[str | int, ...],
    program_steps: list[_CompiledFunctionProgramStep],
) -> int:
    """Append one function subtree to a flat compiled execution program."""

    family_name = _function_family_name(plan)
    if isinstance(plan, ProductFunctionPlan):
        product_root = (*root_rng_path, "product")
        lhs_index = _append_compiled_function_program_step(
            plan.lhs,
            root_rng_path=(*product_root, "lhs"),
            program_steps=program_steps,
        )
        rhs_index = _append_compiled_function_program_step(
            plan.rhs,
            root_rng_path=(*product_root, "rhs"),
            program_steps=program_steps,
        )
        program_steps.append(
            _CompiledFunctionProgramStep(
                plan=plan,
                family_name=family_name,
                root_rng_path=product_root,
                lhs_index=lhs_index,
                rhs_index=rhs_index,
                standardize_input_boundary=True,
            )
        )
        return len(program_steps) - 1
    if isinstance(plan, PiecewiseFunctionPlan):
        piecewise_root = (*root_rng_path, "piecewise")
        lhs_index = _append_compiled_function_program_step(
            plan.lhs,
            root_rng_path=(*piecewise_root, "lhs"),
            program_steps=program_steps,
        )
        rhs_index = _append_compiled_function_program_step(
            plan.rhs,
            root_rng_path=(*piecewise_root, "rhs"),
            program_steps=program_steps,
        )
        program_steps.append(
            _CompiledFunctionProgramStep(
                plan=plan,
                family_name=family_name,
                root_rng_path=piecewise_root,
                lhs_index=lhs_index,
                rhs_index=rhs_index,
                gate_matrix_rng_path=(*piecewise_root, "gate_matrix"),
                standardize_input_boundary=True,
            )
        )
        return len(program_steps) - 1
    program_steps.append(
        _CompiledFunctionProgramStep(
            plan=plan,
            family_name=family_name,
            root_rng_path=tuple(root_rng_path),
        )
    )
    return len(program_steps) - 1


def _compiled_program_cached_rng_paths(
    program_steps: tuple[_CompiledFunctionProgramStep, ...],
) -> tuple[tuple[str | int, ...], ...]:
    """Return the subtree RNG paths reused during one compiled program execution."""

    cached: list[tuple[str | int, ...]] = []
    seen: set[tuple[str | int, ...]] = set()
    for step in program_steps:
        if step.lhs_index is None and step.rhs_index is None and step.root_rng_path:
            if step.root_rng_path not in seen:
                cached.append(step.root_rng_path)
                seen.add(step.root_rng_path)
        if step.gate_matrix_rng_path is not None and step.gate_matrix_rng_path not in seen:
            cached.append(step.gate_matrix_rng_path)
            seen.add(step.gate_matrix_rng_path)
    return tuple(cached)


def _compile_function_execution_context(
    plan: FixedLayoutFunctionPlan,
    *,
    root_rng_path: tuple[str | int, ...] = (),
) -> _CompiledFunctionExecutionContext:
    """Compile reusable execution metadata for one function subtree."""

    program_steps: list[_CompiledFunctionProgramStep] = []
    root_program_index = _append_compiled_function_program_step(
        plan,
        root_rng_path=tuple(root_rng_path),
        program_steps=program_steps,
    )
    compiled_steps = tuple(program_steps)
    return _CompiledFunctionExecutionContext(
        plan=plan,
        family_name=_function_family_name(plan),
        root_rng_path=tuple(root_rng_path),
        program_steps=compiled_steps,
        root_program_index=int(root_program_index),
        cached_rng_paths=_compiled_program_cached_rng_paths(compiled_steps),
    )


def _target_ancestor_node_indices(layout: LayoutPlan) -> tuple[int, ...]:
    """Return topologically ordered target-ancestor node indices for one layout."""

    adjacency = layout.adjacency
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.as_tensor(adjacency, dtype=torch.bool, device="cpu")
    ancestors = _ancestor_nodes_for_target(
        adjacency.to(device="cpu", dtype=torch.bool),
        target_to_node=int(layout.target_to_node),
    )
    return tuple(sorted(int(node_index) for node_index in ancestors))


def _prepare_node_execution_context(
    node_plan: FixedLayoutNodePlan,
) -> _PreparedNodeExecutionContext:
    """Compile reusable execution metadata for one fixed-layout node."""

    source = node_plan.source
    root_source_base_rng_path: tuple[str | int, ...] | None = None
    root_source_function: _CompiledFunctionExecutionContext | None = None
    concat_function: _CompiledFunctionExecutionContext | None = None
    parent_function_contexts: tuple[_CompiledFunctionExecutionContext, ...] = ()
    if isinstance(source, RandomPointsNodeSource):
        root_source_base_rng_path = ("source", "base")
        root_source_function = _compile_function_execution_context(
            source.function,
            root_rng_path=("source", "function"),
        )
    elif isinstance(source, ConcatNodeSource):
        concat_function = _compile_function_execution_context(
            source.function,
            root_rng_path=("function",),
        )
    elif isinstance(source, StackedNodeSource):
        parent_function_contexts = tuple(
            _compile_function_execution_context(
                parent_function,
                root_rng_path=("parent", parent_index),
            )
            for parent_index, parent_function in enumerate(source.parent_functions)
        )

    nested_contexts: list[_CompiledFunctionExecutionContext | None] = [
        None for _ in node_plan.converter_specs
    ]
    for spec_index, converter_plan in enumerate(node_plan.converter_plans):
        if not isinstance(converter_plan, CategoricalConverterPlan):
            continue
        if str(converter_plan.variant) != "center_random_fn" or converter_plan.function is None:
            continue
        nested_contexts[spec_index] = _compile_function_execution_context(
            converter_plan.function,
            root_rng_path=("converter", int(spec_index), "nested_function"),
        )

    cached_rng_paths: list[tuple[str | int, ...]] = [("latent_weights",), ("latent_scale",)]
    if root_source_base_rng_path is not None:
        cached_rng_paths.append(root_source_base_rng_path)
    for function_context in () if root_source_function is None else (root_source_function,):
        cached_rng_paths.extend(function_context.cached_rng_paths)
    for function_context in () if concat_function is None else (concat_function,):
        cached_rng_paths.extend(function_context.cached_rng_paths)
    for function_context in parent_function_contexts:
        cached_rng_paths.extend(function_context.cached_rng_paths)
    for nested_context in nested_contexts:
        if nested_context is not None:
            cached_rng_paths.extend(nested_context.cached_rng_paths)

    deduped_cached_rng_paths: list[tuple[str | int, ...]] = []
    seen_cached_rng_paths: set[tuple[str | int, ...]] = set()
    for path in cached_rng_paths:
        if path in seen_cached_rng_paths:
            continue
        deduped_cached_rng_paths.append(path)
        seen_cached_rng_paths.add(path)

    return _PreparedNodeExecutionContext(
        node_plan=node_plan,
        node_rng_path=("node", int(node_plan.node_index)),
        root_source_base_rng_path=root_source_base_rng_path,
        root_source_function=root_source_function,
        concat_function=concat_function,
        parent_function_contexts=parent_function_contexts,
        converter_nested_function_contexts=tuple(nested_contexts),
        cached_rng_paths=tuple(deduped_cached_rng_paths),
    )


def _prepare_fixed_layout_execution_context(
    layout: LayoutPlan,
    execution_plan: FixedLayoutExecutionPlan,
) -> _PreparedFixedLayoutExecutionContext:
    """Compile reusable execution metadata for one fixed-layout execution plan."""

    return _PreparedFixedLayoutExecutionContext(
        node_contexts=tuple(
            _prepare_node_execution_context(node_plan) for node_plan in execution_plan.node_plans
        ),
        target_ancestor_node_indices=_target_ancestor_node_indices(layout),
    )


def _build_fixed_layout_batch_execution_context(
    prepared_execution_context: _PreparedFixedLayoutExecutionContext,
    batch_rng: FixedLayoutBatchRng,
) -> _FixedLayoutBatchExecutionContext:
    """Build per-batch cached node RNG handles for prepared execution."""

    return _FixedLayoutBatchExecutionContext(
        node_contexts=tuple(
            (
                lambda node_rng: _PreparedBatchNodeExecutionContext(
                    node_rng=node_rng,
                    child_rngs={
                        path: node_rng.keyed(*path) for path in node_context.cached_rng_paths
                    },
                )
            )(batch_rng.keyed(*node_context.node_rng_path))
            for node_context in prepared_execution_context.node_contexts
        )
    )


def _batch_rng_for_dataset_seeds(
    dataset_seeds: list[int] | tuple[int, ...],
    *,
    device: str,
) -> FixedLayoutBatchRng:
    if not dataset_seeds:
        raise ValueError("dataset_seeds must be non-empty.")
    batch_seed = KeyedRng(int(dataset_seeds[0])).child_seed(
        "fixed_layout_chunk",
        len(dataset_seeds),
    )
    return FixedLayoutBatchRng(seed=batch_seed, batch_size=len(dataset_seeds), device=device)


def _build_prepared_batch_node_context(
    node_context: _PreparedNodeExecutionContext,
    batch_rng: FixedLayoutBatchRng,
) -> _PreparedBatchNodeExecutionContext:
    node_rng = batch_rng.keyed(*node_context.node_rng_path)
    return _PreparedBatchNodeExecutionContext(
        node_rng=node_rng,
        child_rngs={path: node_rng.keyed(*path) for path in node_context.cached_rng_paths},
    )


def _is_leaf_function_plan(plan: FixedLayoutFunctionPlan) -> bool:
    return not isinstance(plan, (ProductFunctionPlan, PiecewiseFunctionPlan))


def _function_plan_supports_mixed_execution(plan: FixedLayoutFunctionPlan) -> bool:
    if _is_leaf_function_plan(plan):
        return True
    if isinstance(plan, ProductFunctionPlan):
        return _function_plan_supports_mixed_execution(
            plan.lhs
        ) and _function_plan_supports_mixed_execution(plan.rhs)
    if isinstance(plan, PiecewiseFunctionPlan):
        return _function_plan_supports_mixed_execution(
            plan.lhs
        ) and _function_plan_supports_mixed_execution(plan.rhs)
    return False


def _source_supports_mixed_execution(
    source: RandomPointsNodeSource | ConcatNodeSource | StackedNodeSource,
) -> bool:
    if isinstance(source, RandomPointsNodeSource):
        return _function_plan_supports_mixed_execution(source.function)
    if isinstance(source, ConcatNodeSource):
        return _function_plan_supports_mixed_execution(source.function)
    return all(_function_plan_supports_mixed_execution(plan) for plan in source.parent_functions)


def _activation_kernel_signature(
    plan: FixedLayoutActivationPlan | None,
) -> tuple[object, ...] | None:
    if plan is None:
        return None
    if isinstance(plan, FixedActivationPlan):
        return ("fixed", str(plan.name))
    if isinstance(plan, ParametricActivationPlan):
        return (
            "parametric",
            str(plan.kind),
            None if plan.poly_power is None else int(plan.poly_power),
            None if plan.temperature is None else float(plan.temperature),
        )
    return (type(plan).__name__,)


def _function_family_topology_signature(
    plan: FixedLayoutFunctionPlan,
) -> tuple[object, ...]:
    family_name = _function_family_name(plan)
    if isinstance(plan, ProductFunctionPlan):
        return (
            family_name,
            _function_family_topology_signature(plan.lhs),
            _function_family_topology_signature(plan.rhs),
        )
    if isinstance(plan, PiecewiseFunctionPlan):
        return (
            family_name,
            _function_family_topology_signature(plan.lhs),
            _function_family_topology_signature(plan.rhs),
        )
    return (family_name,)


def _function_apply_kernel_signature(
    plan: FixedLayoutFunctionPlan,
) -> tuple[object, ...]:
    if isinstance(plan, LinearFunctionPlan):
        return ("linear",)
    if isinstance(plan, QuadraticFunctionPlan):
        return ("quadratic",)
    if isinstance(plan, NeuralNetFunctionPlan):
        return (
            "nn",
            int(plan.n_layers),
            int(plan.hidden_width),
            _activation_kernel_signature(plan.input_activation),
            tuple(
                _activation_kernel_signature(activation) for activation in plan.hidden_activations
            ),
            _activation_kernel_signature(plan.output_activation),
        )
    if isinstance(plan, TreeFunctionPlan):
        return ("tree", tuple(int(depth) for depth in plan.depths))
    if isinstance(plan, DiscretizationFunctionPlan):
        return ("discretization", int(plan.n_centers))
    if isinstance(plan, GpFunctionPlan):
        return ("gp", str(plan.branch_kind), str(plan.variant))
    if isinstance(plan, EmFunctionPlan):
        return ("em", int(plan.m_val))
    if isinstance(plan, ProductFunctionPlan):
        return (
            "product",
            _function_apply_kernel_signature(plan.lhs),
            _function_apply_kernel_signature(plan.rhs),
        )
    if isinstance(plan, PiecewiseFunctionPlan):
        return (
            "piecewise",
            float(plan.gate_bias),
            float(plan.gate_temperature),
            _function_apply_kernel_signature(plan.lhs),
            _function_apply_kernel_signature(plan.rhs),
        )
    return (_function_family_name(plan),)


def _compiled_function_apply_kernel_signature(
    context: _CompiledFunctionExecutionContext | None,
) -> tuple[object, ...] | None:
    if context is None:
        return None
    return _function_apply_kernel_signature(context.plan)


def _compiled_converter_group_signature(
    node_context: _PreparedNodeExecutionContext,
    group: FixedLayoutCompiledConverterGroup,
    *,
    coarse: bool,
) -> tuple[object, ...]:
    if isinstance(group, CompiledNumericConverterGroup):
        return (
            "numeric",
            int(len(group.slices)),
            tuple(int(spec.width) for spec in group.slices),
            bool(group.all_unit_width),
        )
    nested_function_signature: tuple[object, ...] | None = None
    if group.uses_center_random_fn:
        nested_context = node_context.converter_nested_function_contexts[int(group.spec_indices[0])]
        nested_function_signature = (
            _function_family_topology_signature(nested_context.plan)
            if coarse and nested_context is not None
            else _compiled_function_apply_kernel_signature(nested_context)
        )
    return (
        "categorical",
        str(group.plan.method),
        str(group.plan.variant),
        int(len(group.slices)),
        tuple(int(spec.width) for spec in group.slices),
        tuple(None if spec.cardinality is None else int(spec.cardinality) for spec in group.slices),
        bool(group.uses_center_random_fn),
        nested_function_signature,
    )


def _node_source_coarse_sketch(
    node_context: _PreparedNodeExecutionContext,
) -> tuple[object, ...]:
    node_plan = node_context.node_plan
    source = node_plan.source
    if isinstance(source, RandomPointsNodeSource):
        return (
            "random_points",
            _function_family_topology_signature(source.function),
        )
    if isinstance(source, ConcatNodeSource):
        return (
            "concat",
            _function_family_topology_signature(source.function),
        )
    return (
        "stacked",
        str(source.aggregation_kind),
        tuple(_function_family_topology_signature(plan) for plan in source.parent_functions),
    )


def mixed_execution_cohort_compatibility_sketch(
    execution_plan: FixedLayoutExecutionPlan,
    prepared_execution_context: _PreparedFixedLayoutExecutionContext,
) -> tuple[object, ...]:
    """Return one coarse compatibility sketch used for heterogeneous physical grouping."""

    node_sketches: list[tuple[object, ...]] = []
    for node_context, node_plan in zip(
        prepared_execution_context.node_contexts,
        execution_plan.node_plans,
        strict=True,
    ):
        converter_groups = _resolved_compiled_converter_groups(node_plan)
        node_sketches.append(
            (
                int(node_plan.latent.total_dim),
                int(len(node_plan.parent_indices)),
                _node_source_coarse_sketch(node_context),
                tuple(
                    _compiled_converter_group_signature(
                        node_context,
                        group,
                        coarse=True,
                    )
                    for group in converter_groups
                ),
                tuple(
                    str(spec.key)
                    for spec in node_plan.converter_specs
                    if str(spec.key).startswith("target")
                ),
            )
        )
    return (
        tuple(node_sketches),
        tuple(int(index) for index in prepared_execution_context.target_ancestor_node_indices),
    )


def _source_bucket_key(
    node_context: _PreparedNodeExecutionContext,
    *,
    parent_widths: tuple[int, ...],
) -> tuple[object, ...]:
    node_plan = node_context.node_plan
    source = node_plan.source
    if isinstance(source, RandomPointsNodeSource):
        source_signature: tuple[object, ...] = (
            "random_points",
            _compiled_function_apply_kernel_signature(node_context.root_source_function),
        )
    elif isinstance(source, ConcatNodeSource):
        source_signature = (
            "concat",
            _compiled_function_apply_kernel_signature(node_context.concat_function),
        )
    else:
        source_signature = (
            "stacked",
            str(source.aggregation_kind),
            tuple(
                _compiled_function_apply_kernel_signature(function_context)
                for function_context in node_context.parent_function_contexts
            ),
        )
    return (
        source_signature,
        int(node_plan.latent.total_dim),
        tuple(int(width) for width in parent_widths),
    )


def _parent_index_lists(layout: LayoutPlan) -> list[list[int]]:
    adjacency = layout.adjacency
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.as_tensor(adjacency, dtype=torch.bool, device="cpu")
    return [
        sorted(int(parent_index) for parent_index in torch.where(adjacency[:, node_index])[0])
        for node_index in range(int(layout.graph_nodes))
    ]


def build_fixed_layout_execution_plan(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    plan_seed: int,
    mechanism_logit_tilt: float,
    stress_profile_name: str | None = None,
) -> FixedLayoutExecutionPlan:
    """Build one reusable per-node execution-plan payload for fixed-layout batches."""

    plan_root = KeyedRng(int(plan_seed))
    task = str(config.dataset.task)
    node_plans: list[FixedLayoutNodePlan] = []
    for node_index, parent_indices in enumerate(_parent_index_lists(layout)):
        spec_root = plan_root.keyed("node_spec", node_index)
        node_root = plan_root.keyed("node_plan", node_index)
        converter_specs = _build_node_specs(node_index, layout, spec_root)
        if int(node_index) == int(layout.target_to_node):
            converter_specs = [*converter_specs, *_build_target_specs(layout, task)]
        parent_output_dims = [
            int(node_plans[parent_index].latent.total_dim) for parent_index in parent_indices
        ]
        node_plans.append(
            _with_compiled_converter_groups(
                sample_node_plan(
                    node_index=int(node_index),
                    parent_indices=parent_indices,
                    parent_output_dims=parent_output_dims,
                    converter_specs=converter_specs,
                    keyed_rng=node_root,
                    device="cpu",
                    mechanism_logit_tilt=mechanism_logit_tilt,
                    function_family_mix=config.mechanism.function_family_mix,
                    stress_profile_name=stress_profile_name,
                )
            )
        )
    return FixedLayoutExecutionPlan(
        node_plans=tuple(node_plans),
        execution_contract=_FIXED_LAYOUT_EXECUTION_CONTRACT,
    )


def fixed_layout_plan_signature(execution_plan: FixedLayoutExecutionPlan) -> str:
    """Return a deterministic signature for one fixed-layout execution plan payload."""

    encoded = json.dumps(
        fixed_layout_signature_payloads(execution_plan),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=16).hexdigest()


def _compiled_converter_slice(
    spec_index: int,
    spec: FixedLayoutConverterSpec,
) -> FixedLayoutCompiledConverterSlice:
    column_start = int(spec.column_start)
    column_end = int(spec.column_end)
    return FixedLayoutCompiledConverterSlice(
        spec_index=int(spec_index),
        key=str(spec.key),
        column_start=column_start,
        column_end=column_end,
        width=max(1, column_end - column_start),
        cardinality=None if spec.cardinality is None else int(spec.cardinality),
    )


def _compile_converter_groups(
    node_plan: FixedLayoutNodePlan,
) -> tuple[FixedLayoutCompiledConverterGroup, ...]:
    compiled_groups: list[FixedLayoutCompiledConverterGroup] = []
    for group in node_plan.converter_groups:
        slices = tuple(
            _compiled_converter_slice(spec_index, node_plan.converter_specs[spec_index])
            for spec_index in group.spec_indices
        )
        if isinstance(group, NumericConverterGroup):
            plans: list[NumericConverterPlan] = []
            for spec_index in group.spec_indices:
                plan = node_plan.converter_plans[spec_index]
                if not isinstance(plan, NumericConverterPlan):
                    raise ValueError(
                        "Numeric converter group must reference numeric converter plans."
                    )
                plans.append(plan)
            compiled_groups.append(
                CompiledNumericConverterGroup(
                    spec_indices=tuple(int(spec_index) for spec_index in group.spec_indices),
                    slices=slices,
                    plans=tuple(plans),
                    warp_enabled=tuple(bool(plan.warp_enabled) for plan in plans),
                    all_unit_width=all(int(spec.width) == 1 for spec in slices),
                )
            )
            continue

        first_index = int(group.spec_indices[0])
        plan = node_plan.converter_plans[first_index]
        if not isinstance(plan, CategoricalConverterPlan):
            raise ValueError("Categorical converter group must reference categorical plans.")
        compiled_groups.append(
            CompiledCategoricalConverterGroup(
                spec_indices=tuple(int(spec_index) for spec_index in group.spec_indices),
                slices=slices,
                plan=plan,
                category_count=max(2, int(slices[0].cardinality or 2)),
                uses_center_random_fn=str(plan.variant) == "center_random_fn",
            )
        )
    return tuple(compiled_groups)


def _with_compiled_converter_groups(node_plan: FixedLayoutNodePlan) -> FixedLayoutNodePlan:
    if node_plan.compiled_converter_groups:
        return node_plan
    return replace(
        node_plan,
        compiled_converter_groups=_compile_converter_groups(node_plan),
    )


def _resolved_compiled_converter_groups(
    node_plan: FixedLayoutNodePlan,
) -> tuple[FixedLayoutCompiledConverterGroup, ...]:
    if node_plan.compiled_converter_groups:
        return node_plan.compiled_converter_groups
    return _compile_converter_groups(node_plan)


def _accumulate_runtime_metric(
    runtime_metrics_out: dict[str, float] | None,
    key: str,
    value: float,
) -> None:
    if runtime_metrics_out is None:
        return
    runtime_metrics_out[key] = float(runtime_metrics_out.get(key, 0.0)) + float(value)


def _function_family_name(plan: FixedLayoutFunctionPlan) -> str:
    if isinstance(plan, LinearFunctionPlan):
        return "linear"
    if isinstance(plan, QuadraticFunctionPlan):
        return "quadratic"
    if isinstance(plan, NeuralNetFunctionPlan):
        return "nn"
    if isinstance(plan, TreeFunctionPlan):
        return "tree"
    if isinstance(plan, DiscretizationFunctionPlan):
        return "discretization"
    if isinstance(plan, GpFunctionPlan):
        return "gp"
    if isinstance(plan, EmFunctionPlan):
        return "em"
    if isinstance(plan, ProductFunctionPlan):
        return "product"
    if isinstance(plan, PiecewiseFunctionPlan):
        return "piecewise"
    return type(plan).__name__.lower()


def _apply_leaf_function_plan_batch(
    y: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: FixedLayoutFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    if isinstance(plan, LinearFunctionPlan):
        return _apply_linear_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, QuadraticFunctionPlan):
        return _apply_quadratic_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, NeuralNetFunctionPlan):
        return _apply_nn_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, TreeFunctionPlan):
        return _apply_tree_batch(y, rng, plan, out_dim=out_dim, noise_spec=noise_spec)
    if isinstance(plan, DiscretizationFunctionPlan):
        return _apply_discretization_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, GpFunctionPlan):
        return _apply_gp_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, EmFunctionPlan):
        return _apply_em_batch(
            y,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    raise ValueError(f"Unsupported fixed-layout leaf function plan: {plan!r}")


def _apply_function_plan_batch_core(
    y: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: FixedLayoutFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    family_name = _function_family_name(plan)
    family_start = time.perf_counter()
    out: torch.Tensor
    try:
        if isinstance(plan, ProductFunctionPlan):
            product_rng = rng.keyed("product")
            branch_input = _batch_standardize(y)
            lhs = _apply_function_plan_batch_core(
                branch_input,
                product_rng.keyed("lhs"),
                plan.lhs,
                out_dim=out_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
            rhs = _apply_function_plan_batch_core(
                branch_input,
                product_rng.keyed("rhs"),
                plan.rhs,
                out_dim=out_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
            out = lhs * rhs
        elif isinstance(plan, PiecewiseFunctionPlan):
            piecewise_rng = rng.keyed("piecewise")
            gate = _apply_piecewise_gate_with_params(
                y,
                plan,
                _sample_piecewise_gate_params(
                    y,
                    piecewise_rng.keyed("gate_matrix"),
                    plan,
                    noise_sigma_multiplier=noise_sigma_multiplier,
                    noise_spec=noise_spec,
                ),
            )
            branch_input = _batch_standardize(y)
            lhs = _apply_function_plan_batch_core(
                branch_input,
                piecewise_rng.keyed("lhs"),
                plan.lhs,
                out_dim=out_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
            rhs = _apply_function_plan_batch_core(
                branch_input,
                piecewise_rng.keyed("rhs"),
                plan.rhs,
                out_dim=out_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
            out = gate * lhs + (1.0 - gate) * rhs
        else:
            out = _apply_leaf_function_plan_batch(
                y,
                rng,
                plan,
                out_dim=out_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
    finally:
        record_runtime_profile_metric(
            f"profile_node_apply_{family_name}_elapsed_seconds",
            time.perf_counter() - family_start,
        )
    out = torch.nan_to_num(out.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    return torch.clamp(out, -1e6, 1e6)


@dataclass(slots=True)
class _CompiledExecutionFrame:
    """Mutable execution frame for iterative compiled higher-order evaluation."""

    node_index: int
    input_tensor: torch.Tensor
    started_at: float
    stage: int = 0
    exclusive_elapsed_seconds: float = 0.0
    branch_input: torch.Tensor | None = None
    gate: torch.Tensor | None = None


def _sanitize_function_output(out: torch.Tensor) -> torch.Tensor:
    """Clamp one function-program output to the shared fixed-layout bounds."""

    out = torch.nan_to_num(out.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    return torch.clamp(out, -1e6, 1e6)


def _resolve_compiled_program_rng(
    node_rng: FixedLayoutBatchRng,
    *,
    path: tuple[str | int, ...],
    cached_child_rngs: dict[tuple[str | int, ...], FixedLayoutBatchRng] | None,
) -> FixedLayoutBatchRng:
    """Resolve one compiled subtree RNG handle, reusing prepared batch caches when available."""

    if cached_child_rngs is not None:
        cached = cached_child_rngs.get(path)
        if cached is not None:
            return cached
    return node_rng.keyed(*path)


def _can_fuse_leaf_pair(
    lhs_step: _CompiledFunctionProgramStep,
    rhs_step: _CompiledFunctionProgramStep,
) -> bool:
    """Return whether one sibling leaf pair can run through the paired exact helper."""

    if lhs_step.lhs_index is not None or lhs_step.rhs_index is not None:
        return False
    if rhs_step.lhs_index is not None or rhs_step.rhs_index is not None:
        return False
    lhs_plan = lhs_step.plan
    rhs_plan = rhs_step.plan
    if isinstance(lhs_plan, LinearFunctionPlan) and isinstance(rhs_plan, LinearFunctionPlan):
        return True
    if isinstance(lhs_plan, QuadraticFunctionPlan) and isinstance(rhs_plan, QuadraticFunctionPlan):
        return True
    if isinstance(lhs_plan, GpFunctionPlan) and isinstance(rhs_plan, GpFunctionPlan):
        return str(lhs_plan.branch_kind) == str(rhs_plan.branch_kind) and str(
            lhs_plan.variant
        ) == str(rhs_plan.variant)
    if isinstance(lhs_plan, DiscretizationFunctionPlan) and isinstance(
        rhs_plan, DiscretizationFunctionPlan
    ):
        return int(lhs_plan.n_centers) == int(rhs_plan.n_centers)
    if isinstance(lhs_plan, EmFunctionPlan) and isinstance(rhs_plan, EmFunctionPlan):
        return int(lhs_plan.m_val) == int(rhs_plan.m_val)
    return False


def _record_compiled_leaf_elapsed_metric(family_name: str, elapsed_seconds: float) -> None:
    record_runtime_profile_metric(
        f"profile_node_apply_{family_name}_elapsed_seconds",
        float(elapsed_seconds),
    )


def _apply_fused_leaf_pair_batch(
    x: torch.Tensor,
    *,
    node_rng: FixedLayoutBatchRng,
    lhs_step: _CompiledFunctionProgramStep,
    rhs_step: _CompiledFunctionProgramStep,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    cached_child_rngs: dict[tuple[str | int, ...], FixedLayoutBatchRng] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one exact fused sibling leaf pair and return lhs/rhs outputs."""

    lhs_plan = lhs_step.plan
    rhs_plan = rhs_step.plan
    plans: tuple[
        LinearFunctionPlan
        | QuadraticFunctionPlan
        | DiscretizationFunctionPlan
        | GpFunctionPlan
        | EmFunctionPlan,
        LinearFunctionPlan
        | QuadraticFunctionPlan
        | DiscretizationFunctionPlan
        | GpFunctionPlan
        | EmFunctionPlan,
    ]
    if isinstance(lhs_plan, LinearFunctionPlan) and isinstance(rhs_plan, LinearFunctionPlan):
        plans = (lhs_plan, rhs_plan)
    elif isinstance(lhs_plan, QuadraticFunctionPlan) and isinstance(
        rhs_plan, QuadraticFunctionPlan
    ):
        plans = (lhs_plan, rhs_plan)
    elif isinstance(lhs_plan, DiscretizationFunctionPlan) and isinstance(
        rhs_plan, DiscretizationFunctionPlan
    ):
        plans = (lhs_plan, rhs_plan)
    elif isinstance(lhs_plan, GpFunctionPlan) and isinstance(rhs_plan, GpFunctionPlan):
        plans = (lhs_plan, rhs_plan)
    elif isinstance(lhs_plan, EmFunctionPlan) and isinstance(rhs_plan, EmFunctionPlan):
        plans = (lhs_plan, rhs_plan)
    else:
        raise ValueError("Unsupported fused leaf pair passed to compiled fixed-layout execution.")

    lhs_rng = _resolve_compiled_program_rng(
        node_rng,
        path=lhs_step.root_rng_path,
        cached_child_rngs=cached_child_rngs,
    )
    rhs_rng = _resolve_compiled_program_rng(
        node_rng,
        path=rhs_step.root_rng_path,
        cached_child_rngs=cached_child_rngs,
    )
    family_name = lhs_step.family_name
    family_start = time.perf_counter()
    outputs = _apply_leaf_pair_batch(
        x,
        rngs=(lhs_rng, rhs_rng),
        plans=plans,
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    _record_compiled_leaf_elapsed_metric(family_name, time.perf_counter() - family_start)
    return _sanitize_function_output(outputs[:, 0, :, :]), _sanitize_function_output(
        outputs[:, 1, :, :]
    )


def _apply_compiled_function_plan_batch_core(
    y: torch.Tensor,
    *,
    node_rng: FixedLayoutBatchRng,
    context: _CompiledFunctionExecutionContext,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    cached_child_rngs: dict[tuple[str | int, ...], FixedLayoutBatchRng] | None = None,
) -> torch.Tensor:
    program_steps = context.program_steps
    if not program_steps:
        raise ValueError("Compiled fixed-layout function context is missing execution steps.")

    results: dict[int, torch.Tensor] = {}
    stack: list[_CompiledExecutionFrame] = [
        _CompiledExecutionFrame(
            node_index=int(context.root_program_index),
            input_tensor=y,
            started_at=time.perf_counter(),
        )
    ]
    while stack:
        frame = stack[-1]
        step = program_steps[frame.node_index]
        if step.lhs_index is None and step.rhs_index is None:
            leaf_start = time.perf_counter()
            leaf_out = _apply_leaf_function_plan_batch(
                frame.input_tensor,
                _resolve_compiled_program_rng(
                    node_rng,
                    path=step.root_rng_path,
                    cached_child_rngs=cached_child_rngs,
                ),
                step.plan,
                out_dim=out_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
            _record_compiled_leaf_elapsed_metric(step.family_name, time.perf_counter() - leaf_start)
            results[frame.node_index] = _sanitize_function_output(leaf_out)
            stack.pop()
            continue

        if step.lhs_index is None or step.rhs_index is None:
            raise ValueError("Compiled higher-order program step is missing child indices.")

        if frame.stage == 0:
            exclusive_start = time.perf_counter()
            frame.branch_input = (
                _batch_standardize(frame.input_tensor)
                if step.standardize_input_boundary
                else frame.input_tensor
            )
            if isinstance(step.plan, PiecewiseFunctionPlan):
                if step.gate_matrix_rng_path is None:
                    raise ValueError("Compiled piecewise program step is missing gate-matrix path.")
                frame.gate = _apply_piecewise_gate_with_params(
                    frame.input_tensor,
                    step.plan,
                    _sample_piecewise_gate_params(
                        frame.input_tensor,
                        _resolve_compiled_program_rng(
                            node_rng,
                            path=step.gate_matrix_rng_path,
                            cached_child_rngs=cached_child_rngs,
                        ),
                        step.plan,
                        noise_sigma_multiplier=noise_sigma_multiplier,
                        noise_spec=noise_spec,
                    ),
                )
            frame.exclusive_elapsed_seconds += time.perf_counter() - exclusive_start
            lhs_step = program_steps[step.lhs_index]
            rhs_step = program_steps[step.rhs_index]
            if frame.branch_input is None:
                raise RuntimeError("Compiled branch frame is missing standardized input.")
            if _can_fuse_leaf_pair(lhs_step, rhs_step):
                pair_start = time.perf_counter()
                lhs_out, rhs_out = _apply_fused_leaf_pair_batch(
                    frame.branch_input,
                    node_rng=node_rng,
                    lhs_step=lhs_step,
                    rhs_step=rhs_step,
                    out_dim=out_dim,
                    noise_sigma_multiplier=noise_sigma_multiplier,
                    noise_spec=noise_spec,
                    cached_child_rngs=cached_child_rngs,
                )
                if isinstance(step.plan, PiecewiseFunctionPlan):
                    if frame.gate is None:
                        raise RuntimeError("Compiled piecewise frame is missing gate values.")
                    combined = frame.gate * lhs_out + (1.0 - frame.gate) * rhs_out
                else:
                    combined = lhs_out * rhs_out
                frame.exclusive_elapsed_seconds += time.perf_counter() - pair_start
                results[frame.node_index] = _sanitize_function_output(combined)
                record_runtime_profile_metric(
                    f"profile_node_apply_{step.family_name}_elapsed_seconds",
                    time.perf_counter() - frame.started_at,
                )
                record_runtime_profile_metric(
                    f"profile_node_apply_{step.family_name}_exclusive_elapsed_seconds",
                    frame.exclusive_elapsed_seconds,
                )
                record_runtime_profile_metric(
                    f"profile_node_apply_{step.family_name}_call_count",
                    1.0,
                )
                stack.pop()
                continue
            frame.stage = 1
            stack.append(
                _CompiledExecutionFrame(
                    node_index=int(step.rhs_index),
                    input_tensor=frame.branch_input,
                    started_at=time.perf_counter(),
                )
            )
            stack.append(
                _CompiledExecutionFrame(
                    node_index=int(step.lhs_index),
                    input_tensor=frame.branch_input,
                    started_at=time.perf_counter(),
                )
            )
            continue

        lhs_out = results.pop(int(step.lhs_index))
        rhs_out = results.pop(int(step.rhs_index))
        combine_start = time.perf_counter()
        if isinstance(step.plan, PiecewiseFunctionPlan):
            if frame.gate is None:
                raise RuntimeError("Compiled piecewise frame is missing gate values.")
            combined = frame.gate * lhs_out + (1.0 - frame.gate) * rhs_out
        else:
            combined = lhs_out * rhs_out
        frame.exclusive_elapsed_seconds += time.perf_counter() - combine_start
        results[frame.node_index] = _sanitize_function_output(combined)
        record_runtime_profile_metric(
            f"profile_node_apply_{step.family_name}_elapsed_seconds",
            time.perf_counter() - frame.started_at,
        )
        record_runtime_profile_metric(
            f"profile_node_apply_{step.family_name}_exclusive_elapsed_seconds",
            frame.exclusive_elapsed_seconds,
        )
        record_runtime_profile_metric(
            f"profile_node_apply_{step.family_name}_call_count",
            1.0,
        )
        stack.pop()

    final_out = results.get(int(context.root_program_index))
    if final_out is None:
        raise RuntimeError("Compiled fixed-layout program did not produce a root output.")
    return _sanitize_function_output(final_out)


def apply_function_plan_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: FixedLayoutFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    standardize_input: bool = True,
) -> torch.Tensor:
    """Apply one frozen function-family plan across a batch of datasets."""

    y = x.to(torch.float32)
    if standardize_input:
        y = _batch_standardize(y)
    return _apply_function_plan_batch_core(
        y,
        rng,
        plan,
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )


def apply_compiled_function_plan_batch(
    x: torch.Tensor,
    node_rng: FixedLayoutBatchRng,
    context: _CompiledFunctionExecutionContext,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    standardize_input: bool = True,
    cached_child_rngs: dict[tuple[str | int, ...], FixedLayoutBatchRng] | None = None,
) -> torch.Tensor:
    """Apply one compiled function subtree across a batch of datasets."""

    y = x.to(torch.float32)
    if standardize_input:
        y = _batch_standardize(y)
    return _apply_compiled_function_plan_batch_core(
        y,
        node_rng=node_rng,
        context=context,
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
        cached_child_rngs=cached_child_rngs,
    )


def _sample_mixed_leaf_function_params(
    x: torch.Tensor,
    *,
    program_index: int,
    cohort_runs: list[_MixedFunctionRun],
    out_dim: int,
    noise_sigma_multiplier: float,
) -> Any:
    representative_step = cohort_runs[0].function_context.program_steps[int(program_index)]
    if representative_step.lhs_index is not None or representative_step.rhs_index is not None:
        raise ValueError("Mixed leaf-parameter sampling requires a leaf program step.")
    return _concat_sampled_leaf_function_params(
        [
            _sample_leaf_function_params_batch(
                x[run.start : run.stop],
                _resolve_compiled_program_rng(
                    run.batch_node_context.node_rng,
                    path=run.function_context.program_steps[int(program_index)].root_rng_path,
                    cached_child_rngs=run.batch_node_context.child_rngs,
                ),
                cast(
                    LinearFunctionPlan
                    | QuadraticFunctionPlan
                    | NeuralNetFunctionPlan
                    | TreeFunctionPlan
                    | DiscretizationFunctionPlan
                    | GpFunctionPlan
                    | EmFunctionPlan,
                    run.function_context.program_steps[int(program_index)].plan,
                ),
                out_dim=out_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=run.noise_spec,
            )
            for run in cohort_runs
        ]
    )


def _sample_mixed_piecewise_gate(
    x: torch.Tensor,
    *,
    program_index: int,
    cohort_runs: list[_MixedFunctionRun],
    noise_sigma_multiplier: float,
) -> torch.Tensor:
    step = cohort_runs[0].function_context.program_steps[int(program_index)]
    if not isinstance(step.plan, PiecewiseFunctionPlan):
        raise ValueError("Mixed piecewise gate sampling requires a piecewise program step.")
    if step.gate_matrix_rng_path is None:
        raise ValueError("Compiled piecewise program step is missing gate-matrix path.")
    params_list: list[_SampledPiecewiseGateParams] = []
    for run in cohort_runs:
        run_step = run.function_context.program_steps[int(program_index)]
        if run_step.gate_matrix_rng_path is None:
            raise ValueError("Mixed piecewise cohort is missing gate-matrix path.")
        params_list.append(
            _sample_piecewise_gate_params(
                x[run.start : run.stop],
                _resolve_compiled_program_rng(
                    run.batch_node_context.node_rng,
                    path=run_step.gate_matrix_rng_path,
                    cached_child_rngs=run.batch_node_context.child_rngs,
                ),
                cast(PiecewiseFunctionPlan, run_step.plan),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=run.noise_spec,
            )
        )
    params = _concat_piecewise_gate_params(params_list)
    return _apply_piecewise_gate_with_params(x, step.plan, params)


def _apply_mixed_compiled_function_plan_batch_core(
    y: torch.Tensor,
    *,
    cohort_runs: list[_MixedFunctionRun],
    context: _CompiledFunctionExecutionContext,
    out_dim: int,
    noise_sigma_multiplier: float,
    program_index: int | None = None,
) -> torch.Tensor:
    program_steps = context.program_steps
    if not program_steps:
        raise ValueError("Compiled fixed-layout function context is missing execution steps.")

    current_program_index = int(
        context.root_program_index if program_index is None else program_index
    )
    step = program_steps[current_program_index]
    family_start = time.perf_counter()
    exclusive_elapsed_seconds = 0.0
    try:
        if step.lhs_index is None and step.rhs_index is None:
            out = _apply_sampled_leaf_function_batch(
                y,
                cast(
                    LinearFunctionPlan
                    | QuadraticFunctionPlan
                    | NeuralNetFunctionPlan
                    | TreeFunctionPlan
                    | DiscretizationFunctionPlan
                    | GpFunctionPlan
                    | EmFunctionPlan,
                    step.plan,
                ),
                _sample_mixed_leaf_function_params(
                    y,
                    program_index=current_program_index,
                    cohort_runs=cohort_runs,
                    out_dim=out_dim,
                    noise_sigma_multiplier=noise_sigma_multiplier,
                ),
            )
        else:
            if step.lhs_index is None or step.rhs_index is None:
                raise ValueError("Compiled higher-order program step is missing child indices.")
            boundary_start = time.perf_counter()
            branch_input = _batch_standardize(y) if step.standardize_input_boundary else y
            exclusive_elapsed_seconds += time.perf_counter() - boundary_start
            gate: torch.Tensor | None = None
            if isinstance(step.plan, PiecewiseFunctionPlan):
                gate_start = time.perf_counter()
                gate = _sample_mixed_piecewise_gate(
                    y,
                    program_index=current_program_index,
                    cohort_runs=cohort_runs,
                    noise_sigma_multiplier=noise_sigma_multiplier,
                )
                exclusive_elapsed_seconds += time.perf_counter() - gate_start
            lhs = _apply_mixed_compiled_function_plan_batch_core(
                branch_input,
                cohort_runs=cohort_runs,
                context=context,
                out_dim=out_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                program_index=int(step.lhs_index),
            )
            rhs = _apply_mixed_compiled_function_plan_batch_core(
                branch_input,
                cohort_runs=cohort_runs,
                context=context,
                out_dim=out_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                program_index=int(step.rhs_index),
            )
            combine_start = time.perf_counter()
            if isinstance(step.plan, PiecewiseFunctionPlan):
                if gate is None:
                    raise RuntimeError("Mixed piecewise execution is missing gate values.")
                out = gate * lhs + (1.0 - gate) * rhs
            else:
                out = lhs * rhs
            exclusive_elapsed_seconds += time.perf_counter() - combine_start
    finally:
        elapsed_seconds = time.perf_counter() - family_start
        record_runtime_profile_metric(
            f"profile_node_apply_{step.family_name}_elapsed_seconds",
            elapsed_seconds,
        )
        if step.lhs_index is not None and step.rhs_index is not None:
            record_runtime_profile_metric(
                f"profile_node_apply_{step.family_name}_exclusive_elapsed_seconds",
                exclusive_elapsed_seconds,
            )
            record_runtime_profile_metric(
                f"profile_node_apply_{step.family_name}_call_count",
                1.0,
            )
    return _sanitize_function_output(out)


def _apply_mixed_compiled_function_plan_batch(
    x: torch.Tensor,
    *,
    cohort_runs: list[_MixedFunctionRun],
    context: _CompiledFunctionExecutionContext,
    out_dim: int,
    noise_sigma_multiplier: float,
    standardize_input: bool = True,
) -> torch.Tensor:
    y = x.to(torch.float32)
    if standardize_input:
        y = _batch_standardize(y)
    return _apply_mixed_compiled_function_plan_batch_core(
        y,
        cohort_runs=cohort_runs,
        context=context,
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
    )


def apply_numeric_converter_plan_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: NumericConverterPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one vector-valued numeric converter plan across a batch of datasets."""

    y = x.to(torch.float32)
    values = y[:, :, 0].clone()
    if not plan.warp_enabled:
        return y, values

    a = rng.keyed("a").log_uniform((y.shape[0],), low=0.2, high=5.0)
    b = rng.keyed("b").log_uniform((y.shape[0],), low=0.2, high=5.0)
    lo = torch.min(y, dim=1, keepdim=True).values
    hi = torch.max(y, dim=1, keepdim=True).values
    scaled = (y - lo) / torch.clamp(hi - lo, min=1e-6)
    warped = 1.0 - torch.pow(
        1.0 - torch.pow(torch.clamp(scaled, 0.0, 1.0), a.view(y.shape[0], 1, 1)),
        b.view(y.shape[0], 1, 1),
    )
    return warped, values


def _apply_numeric_converter_group_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    warp_enabled: torch.Tensor,
    *,
    spec_indices: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = x.to(torch.float32)
    values = y.clone()
    if not bool(torch.any(warp_enabled)):
        return y, values
    if spec_indices is None:
        a = rng.keyed("a").log_uniform((y.shape[0], y.shape[2]), low=0.2, high=5.0)
        b = rng.keyed("b").log_uniform((y.shape[0], y.shape[2]), low=0.2, high=5.0)
    else:
        if len(spec_indices) != int(y.shape[2]):
            raise ValueError("spec_indices must align with numeric converter group width.")
        a = torch.stack(
            [
                rng.keyed("converter", spec_index)
                .keyed("a")
                .log_uniform(
                    (y.shape[0],),
                    low=0.2,
                    high=5.0,
                )
                for spec_index in spec_indices
            ],
            dim=1,
        )
        b = torch.stack(
            [
                rng.keyed("converter", spec_index)
                .keyed("b")
                .log_uniform(
                    (y.shape[0],),
                    low=0.2,
                    high=5.0,
                )
                for spec_index in spec_indices
            ],
            dim=1,
        )
    lo = torch.min(y, dim=1, keepdim=True).values
    hi = torch.max(y, dim=1, keepdim=True).values
    scaled = (y - lo) / torch.clamp(hi - lo, min=1e-6)
    warped = 1.0 - torch.pow(
        1.0 - torch.pow(torch.clamp(scaled, 0.0, 1.0), a.view(y.shape[0], 1, y.shape[2])),
        b.view(y.shape[0], 1, y.shape[2]),
    )
    return torch.where(warp_enabled.view(1, 1, -1), warped, y), values


def _categorical_group_input_views(
    latent: torch.Tensor,
    compiled_slices: tuple[FixedLayoutCompiledConverterSlice, ...],
) -> torch.Tensor:
    views = [
        latent[:, :, int(spec.column_start) : int(spec.column_end)] for spec in compiled_slices
    ]
    return torch.stack(views, dim=2)


def _gather_group_centers(y: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # y: [B, rows, G, D], indices: [B, G, K] -> [B, G, K, D]
    y_perm = y.permute(0, 2, 1, 3)
    return torch.gather(
        y_perm,
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, y.shape[3]),
    )


def _apply_categorical_group_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    converter_plan: CategoricalConverterPlan,
    *,
    n_categories: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    spec_indices: tuple[int, ...] | None = None,
    class_probs_out: dict[str, torch.Tensor] | None = None,
    compiled_nested_function_context: _CompiledFunctionExecutionContext | None = None,
    compiled_nested_function_node_rng: FixedLayoutBatchRng | None = None,
    compiled_nested_function_cached_child_rngs: (
        dict[tuple[str | int, ...], FixedLayoutBatchRng] | None
    ) = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = x.to(torch.float32)
    batch_size, n_rows, group_size, width = y.shape
    category_count = max(2, int(n_categories))
    method = str(converter_plan.method)
    variant = str(converter_plan.variant)
    if spec_indices is not None and len(spec_indices) != int(group_size):
        raise ValueError("spec_indices must align with categorical converter group width.")

    centers: torch.Tensor | None = None
    if method == "neighbor":
        n_centers = min(category_count, n_rows)
        if spec_indices is None:
            center_idx = rng.keyed("center_index").randperm_indices(
                length=n_rows,
                sample_size=n_centers,
                leading_shape=(group_size,),
            )
        else:
            center_idx = torch.stack(
                [
                    rng.keyed("converter", spec_index)
                    .keyed("center_index")
                    .randperm_indices(
                        length=n_rows,
                        sample_size=n_centers,
                    )
                    for spec_index in spec_indices
                ],
                dim=1,
            )
        centers = _gather_group_centers(y, center_idx)
        if spec_indices is None:
            p = rng.keyed("lp_norm").log_uniform((batch_size, group_size), low=0.5, high=4.0)
        else:
            p = torch.stack(
                [
                    rng.keyed("converter", spec_index)
                    .keyed("lp_norm")
                    .log_uniform(
                        (batch_size,),
                        low=0.5,
                        high=4.0,
                    )
                    for spec_index in spec_indices
                ],
                dim=1,
            )
        labels_bg = _nearest_lp_center_indices(
            y.permute(0, 2, 1, 3),
            centers,
            p=p,
        )
        if n_centers < category_count:
            labels_bg = labels_bg % category_count
        labels = labels_bg.permute(0, 2, 1)
        probs = torch.nn.functional.one_hot(
            labels.to(torch.int64),
            num_classes=category_count,
        ).to(torch.float32)
    else:
        if width != category_count:
            if spec_indices is None:
                projections = rng.keyed("projections").normal(
                    (batch_size, group_size, width, category_count)
                )
            else:
                projections = torch.stack(
                    [
                        rng.keyed("converter", spec_index)
                        .keyed("projections")
                        .normal((batch_size, width, category_count))
                        for spec_index in spec_indices
                    ],
                    dim=1,
                )
            logits_in = torch.einsum("brgd,bgdc->brgc", y, projections)
        else:
            logits_in = y
        logits_std = _batch_standardize(logits_in)
        if spec_indices is None:
            a = rng.keyed("softmax_scale").log_uniform((batch_size, group_size), low=0.1, high=10.0)
            w = rng.keyed("softmax_bias").uniform(
                (batch_size, group_size, category_count),
                low=0.0,
                high=1.0,
            )
        else:
            a = torch.stack(
                [
                    rng.keyed("converter", spec_index)
                    .keyed("softmax_scale")
                    .log_uniform(
                        (batch_size,),
                        low=0.1,
                        high=10.0,
                    )
                    for spec_index in spec_indices
                ],
                dim=1,
            )
            w = torch.stack(
                [
                    rng.keyed("converter", spec_index)
                    .keyed("softmax_bias")
                    .uniform(
                        (batch_size, category_count),
                        low=0.0,
                        high=1.0,
                    )
                    for spec_index in spec_indices
                ],
                dim=1,
            )
        b = torch.log(w + 1e-4)
        logits = a.unsqueeze(1).unsqueeze(-1) * logits_std + b.unsqueeze(1)
        probs = torch.softmax(logits, dim=3)
        if spec_indices is None:
            labels = rng.keyed("labels").categorical(probs)
        else:
            labels = torch.stack(
                [
                    rng.keyed("converter", spec_index)
                    .keyed("labels")
                    .categorical(probs[:, :, group_index, :])
                    for group_index, spec_index in enumerate(spec_indices)
                ],
                dim=2,
            )

    if class_probs_out is not None:
        class_probs_out["probs"] = probs.to(torch.float32)

    if variant == "input":
        out = y
    elif variant == "index_repeat":
        out = labels.unsqueeze(-1).repeat(1, 1, 1, width).to(torch.float32)
    elif variant == "center":
        if centers is None:
            out = y
        else:
            labels_bg = labels.permute(0, 2, 1)
            gathered = torch.gather(
                centers,
                2,
                labels_bg.unsqueeze(-1).expand(-1, -1, -1, width),
            )
            out = gathered.permute(0, 2, 1, 3)
    elif variant == "center_random_fn":
        nested_input = y
        if centers is not None:
            labels_bg = labels.permute(0, 2, 1)
            nested_input = torch.gather(
                centers,
                2,
                labels_bg.unsqueeze(-1).expand(-1, -1, -1, width),
            ).permute(0, 2, 1, 3)
        if group_size != 1:
            raise ValueError("center_random_fn converter groups must have size 1.")
        nested_function = converter_plan.function
        if nested_function is None:
            raise ValueError("center_random_fn converter plan requires a nested function.")
        if (
            compiled_nested_function_context is not None
            and compiled_nested_function_node_rng is not None
        ):
            nested_out = apply_compiled_function_plan_batch(
                nested_input[:, :, 0, :],
                compiled_nested_function_node_rng,
                compiled_nested_function_context,
                out_dim=width,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                cached_child_rngs=compiled_nested_function_cached_child_rngs,
            )
        else:
            nested_rng = (
                rng.keyed("nested_function")
                if spec_indices is None
                else rng.keyed("converter", spec_indices[0]).keyed("nested_function")
            )
            nested_out = apply_function_plan_batch(
                nested_input[:, :, 0, :],
                nested_rng,
                nested_function,
                out_dim=width,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
        out = nested_out.unsqueeze(2)
    elif variant == "softmax_points":
        if spec_indices is None:
            points = rng.keyed("softmax_points").normal(
                (batch_size, group_size, category_count, width)
            )
        else:
            points = torch.stack(
                [
                    rng.keyed("converter", spec_index)
                    .keyed("softmax_points")
                    .normal((batch_size, category_count, width))
                    for spec_index in spec_indices
                ],
                dim=1,
            )
        labels_bg = labels.permute(0, 2, 1)
        out = torch.gather(
            points,
            2,
            labels_bg.unsqueeze(-1).expand(-1, -1, -1, width),
        ).permute(0, 2, 1, 3)
    else:
        out = y

    out = torch.nan_to_num(out.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    labels = torch.remainder(labels.to(torch.int64), category_count)
    return out, labels


def _apply_node_plan_batch(
    config: GeneratorConfig | None,
    node_plan: FixedLayoutNodePlan,
    parent_data: list[torch.Tensor],
    *,
    n_rows: int,
    rng: FixedLayoutBatchRng,
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    runtime_metrics_out: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _ = config
    _ = device
    node_start = time.perf_counter()
    node_start_cpu = time.process_time()
    total_dim = int(node_plan.latent.total_dim)
    if parent_data:
        source = node_plan.source
        if isinstance(source, ConcatNodeSource):
            concat = _sanitize_and_batch_standardize(torch.cat(parent_data, dim=2))
            latent = apply_function_plan_batch(
                concat,
                rng.keyed("function"),
                source.function,
                out_dim=total_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                standardize_input=False,
            )
        else:
            if not isinstance(source, StackedNodeSource):
                raise ValueError("Parent-driven fixed-layout node must use a multi-input source.")
            aggregation_kind = source.aggregation_kind
            standardized_parents = [
                _sanitize_and_batch_standardize(parent_tensor) for parent_tensor in parent_data
            ]
            if aggregation_kind == "logsumexp":
                transformed_outputs = [
                    apply_function_plan_batch(
                        standardized_parent,
                        rng.keyed("parent", plan_index),
                        source.parent_functions[plan_index],
                        out_dim=total_dim,
                        noise_sigma_multiplier=noise_sigma_multiplier,
                        noise_spec=noise_spec,
                        standardize_input=False,
                    )
                    for plan_index, standardized_parent in enumerate(standardized_parents)
                ]
                stacked = torch.stack(transformed_outputs, dim=2)
                latent = _aggregate_parent_outputs_batch(
                    stacked,
                    aggregation_kind=source.aggregation_kind,
                )
            else:
                aggregate: torch.Tensor | None = None
                for plan_index, standardized_parent in enumerate(standardized_parents):
                    transformed_output = apply_function_plan_batch(
                        standardized_parent,
                        rng.keyed("parent", plan_index),
                        source.parent_functions[plan_index],
                        out_dim=total_dim,
                        noise_sigma_multiplier=noise_sigma_multiplier,
                        noise_spec=noise_spec,
                        standardize_input=False,
                    )
                    if aggregate is None:
                        aggregate = transformed_output
                    else:
                        aggregate = _aggregate_batch_incrementally(
                            aggregate,
                            transformed_output,
                            aggregation_kind=aggregation_kind,
                        )
                if aggregate is None:
                    raise RuntimeError("Expected at least one parent tensor for stacked node plan.")
                latent = aggregate
    else:
        source = node_plan.source
        if not isinstance(source, RandomPointsNodeSource):
            raise ValueError("Root fixed-layout node must use a random-points source.")
        source_rng = rng.keyed("source")
        base = _sample_random_points_batch(
            source_rng.keyed("base"),
            n_rows=n_rows,
            dim=total_dim,
            base_kind=source.base_kind,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        latent = apply_function_plan_batch(
            base,
            source_rng.keyed("function"),
            source.function,
            out_dim=total_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )

    latent = torch.nan_to_num(latent.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    latent = torch.clamp(latent, -1e6, 1e6)
    latent = _batch_standardize(latent)

    weights = _sample_random_weights_batch(
        rng.keyed("latent_weights"),
        dim=int(latent.shape[2]),
        sigma_multiplier=float(noise_sigma_multiplier),
        noise_spec=noise_spec,
    )
    latent = latent * weights.unsqueeze(1)
    mean_l2 = torch.mean(torch.norm(latent, dim=2), dim=1)
    latent = latent / torch.clamp(mean_l2.view(-1, 1, 1), min=1e-6)

    extracted: dict[str, torch.Tensor] = {}
    converter_start = time.perf_counter()
    converter_start_cpu = time.process_time()
    for group in _resolved_compiled_converter_groups(node_plan):
        if isinstance(group, CompiledNumericConverterGroup):
            _accumulate_runtime_metric(runtime_metrics_out, "mixed_converter_bucket_count", 1.0)
            _accumulate_runtime_metric(
                runtime_metrics_out,
                "mixed_converter_bucket_dataset_sum",
                float(rng.batch_size),
            )
            if group.all_unit_width:
                grouped_input = torch.cat(
                    [
                        latent[:, :, int(spec.column_start) : int(spec.column_end)]
                        for spec in group.slices
                    ],
                    dim=2,
                )
                warp_enabled = torch.tensor(
                    group.warp_enabled,
                    device=latent.device,
                    dtype=torch.bool,
                )
                x_prime, values = _apply_numeric_converter_group_batch(
                    grouped_input,
                    rng,
                    warp_enabled,
                    spec_indices=group.spec_indices,
                )
                for local_index, spec in enumerate(group.slices):
                    start = int(spec.column_start)
                    end = int(spec.column_end)
                    latent[:, :, start:end] = x_prime[:, :, local_index : local_index + 1]
                    extracted[str(spec.key)] = values[:, :, local_index]
                continue
            for spec, plan in zip(group.slices, group.plans, strict=True):
                start = int(spec.column_start)
                end = int(spec.column_end)
                spec_out, values = apply_numeric_converter_plan_batch(
                    latent[:, :, start:end],
                    rng.keyed("converter", spec.spec_index),
                    plan,
                )
                if int(spec_out.shape[2]) != (end - start):
                    if int(spec_out.shape[2]) > (end - start):
                        spec_out = spec_out[:, :, : (end - start)]
                    else:
                        spec_out = torch.nn.functional.pad(
                            spec_out, (0, (end - start) - int(spec_out.shape[2]))
                        )
                latent[:, :, start:end] = spec_out
                extracted[str(spec.key)] = values
            continue

        if not group.uses_center_random_fn:
            x_prime, values = _apply_categorical_group_batch(
                _categorical_group_input_views(latent, group.slices),
                rng,
                group.plan,
                n_categories=group.category_count,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                spec_indices=group.spec_indices,
            )
            for local_index, spec in enumerate(group.slices):
                start = int(spec.column_start)
                end = int(spec.column_end)
                spec_out = x_prime[:, :, local_index, :]
                if int(spec_out.shape[2]) != (end - start):
                    if int(spec_out.shape[2]) > (end - start):
                        spec_out = spec_out[:, :, : (end - start)]
                    else:
                        spec_out = torch.nn.functional.pad(
                            spec_out, (0, (end - start) - int(spec_out.shape[2]))
                        )
                latent[:, :, start:end] = spec_out
                extracted[str(spec.key)] = values[:, :, local_index]
            continue
        for spec in group.slices:
            start = int(spec.column_start)
            end = int(spec.column_end)
            spec_view = latent[:, :, start:end].unsqueeze(2)
            x_prime, values = _apply_categorical_group_batch(
                spec_view,
                rng.keyed("converter", spec.spec_index),
                group.plan,
                n_categories=max(2, int(spec.cardinality or 2)),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
            spec_out = x_prime[:, :, 0, :]
            if int(spec_out.shape[2]) != (end - start):
                if int(spec_out.shape[2]) > (end - start):
                    spec_out = spec_out[:, :, : (end - start)]
                else:
                    spec_out = torch.nn.functional.pad(
                        spec_out, (0, (end - start) - int(spec_out.shape[2]))
                    )
            latent[:, :, start:end] = spec_out
            extracted[str(spec.key)] = values[:, :, 0]

    _accumulate_runtime_metric(
        runtime_metrics_out,
        "converter_elapsed_seconds",
        time.perf_counter() - converter_start,
    )
    _accumulate_runtime_metric(
        runtime_metrics_out,
        "converter_cpu_time_seconds",
        time.process_time() - converter_start_cpu,
    )

    scale = rng.keyed("latent_scale").log_uniform((rng.batch_size,), low=0.1, high=10.0)
    latent = latent * scale.view(-1, 1, 1)
    _accumulate_runtime_metric(
        runtime_metrics_out,
        "node_apply_elapsed_seconds",
        time.perf_counter() - node_start,
    )
    _accumulate_runtime_metric(
        runtime_metrics_out,
        "node_apply_cpu_time_seconds",
        time.process_time() - node_start_cpu,
    )
    return latent, extracted


def _apply_node_plan_batch_prepared(
    config: GeneratorConfig | None,
    node_context: _PreparedNodeExecutionContext,
    parent_data: list[torch.Tensor],
    *,
    n_rows: int,
    batch_node_context: _PreparedBatchNodeExecutionContext,
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    runtime_metrics_out: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply one prepared fixed-layout node plan across a batch of datasets."""

    _ = config
    _ = device
    node_plan = node_context.node_plan
    node_rng = batch_node_context.node_rng
    cached_child_rngs = batch_node_context.child_rngs
    node_start = time.perf_counter()
    node_start_cpu = time.process_time()
    total_dim = int(node_plan.latent.total_dim)
    if parent_data:
        source = node_plan.source
        if isinstance(source, ConcatNodeSource):
            if node_context.concat_function is None:
                raise ValueError("Prepared concat node context is incomplete.")
            concat = _sanitize_and_batch_standardize(torch.cat(parent_data, dim=2))
            latent = apply_compiled_function_plan_batch(
                concat,
                node_rng,
                node_context.concat_function,
                out_dim=total_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                standardize_input=False,
                cached_child_rngs=cached_child_rngs,
            )
        else:
            if not isinstance(source, StackedNodeSource):
                raise ValueError("Parent-driven fixed-layout node must use a multi-input source.")
            aggregation_kind = source.aggregation_kind
            standardized_parents = [
                _sanitize_and_batch_standardize(parent_tensor) for parent_tensor in parent_data
            ]
            if aggregation_kind == "logsumexp":
                transformed_outputs = [
                    apply_compiled_function_plan_batch(
                        standardized_parent,
                        node_rng,
                        node_context.parent_function_contexts[plan_index],
                        out_dim=total_dim,
                        noise_sigma_multiplier=noise_sigma_multiplier,
                        noise_spec=noise_spec,
                        standardize_input=False,
                        cached_child_rngs=cached_child_rngs,
                    )
                    for plan_index, standardized_parent in enumerate(standardized_parents)
                ]
                stacked = torch.stack(transformed_outputs, dim=2)
                latent = _aggregate_parent_outputs_batch(
                    stacked,
                    aggregation_kind=source.aggregation_kind,
                )
            else:
                aggregate: torch.Tensor | None = None
                for plan_index, standardized_parent in enumerate(standardized_parents):
                    transformed_output = apply_compiled_function_plan_batch(
                        standardized_parent,
                        node_rng,
                        node_context.parent_function_contexts[plan_index],
                        out_dim=total_dim,
                        noise_sigma_multiplier=noise_sigma_multiplier,
                        noise_spec=noise_spec,
                        standardize_input=False,
                        cached_child_rngs=cached_child_rngs,
                    )
                    if aggregate is None:
                        aggregate = transformed_output
                    else:
                        aggregate = _aggregate_batch_incrementally(
                            aggregate,
                            transformed_output,
                            aggregation_kind=aggregation_kind,
                        )
                if aggregate is None:
                    raise RuntimeError("Expected at least one parent tensor for stacked node plan.")
                latent = aggregate
    else:
        source = node_plan.source
        if not isinstance(source, RandomPointsNodeSource):
            raise ValueError("Root fixed-layout node must use a random-points source.")
        if (
            node_context.root_source_base_rng_path is None
            or node_context.root_source_function is None
        ):
            raise ValueError("Prepared root-node execution context is incomplete.")
        base = _sample_random_points_batch(
            _resolve_compiled_program_rng(
                node_rng,
                path=node_context.root_source_base_rng_path,
                cached_child_rngs=cached_child_rngs,
            ),
            n_rows=n_rows,
            dim=total_dim,
            base_kind=source.base_kind,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        latent = apply_compiled_function_plan_batch(
            base,
            node_rng,
            node_context.root_source_function,
            out_dim=total_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
            cached_child_rngs=cached_child_rngs,
        )

    latent = torch.nan_to_num(latent.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    latent = torch.clamp(latent, -1e6, 1e6)
    latent = _batch_standardize(latent)

    weights = _sample_random_weights_batch(
        _resolve_compiled_program_rng(
            node_rng,
            path=("latent_weights",),
            cached_child_rngs=cached_child_rngs,
        ),
        dim=int(latent.shape[2]),
        sigma_multiplier=float(noise_sigma_multiplier),
        noise_spec=noise_spec,
    )
    latent = latent * weights.unsqueeze(1)
    mean_l2 = torch.mean(torch.norm(latent, dim=2), dim=1)
    latent = latent / torch.clamp(mean_l2.view(-1, 1, 1), min=1e-6)

    extracted: dict[str, torch.Tensor] = {}
    converter_start = time.perf_counter()
    converter_start_cpu = time.process_time()
    for group in _resolved_compiled_converter_groups(node_plan):
        if isinstance(group, CompiledNumericConverterGroup):
            if group.all_unit_width:
                grouped_input = torch.cat(
                    [
                        latent[:, :, int(spec.column_start) : int(spec.column_end)]
                        for spec in group.slices
                    ],
                    dim=2,
                )
                warp_enabled = torch.tensor(
                    group.warp_enabled,
                    device=latent.device,
                    dtype=torch.bool,
                )
                x_prime, values = _apply_numeric_converter_group_batch(
                    grouped_input,
                    node_rng,
                    warp_enabled,
                    spec_indices=group.spec_indices,
                )
                for local_index, spec in enumerate(group.slices):
                    start = int(spec.column_start)
                    end = int(spec.column_end)
                    latent[:, :, start:end] = x_prime[:, :, local_index : local_index + 1]
                    extracted[str(spec.key)] = values[:, :, local_index]
                continue
            for spec, plan in zip(group.slices, group.plans, strict=True):
                start = int(spec.column_start)
                end = int(spec.column_end)
                spec_out, values = apply_numeric_converter_plan_batch(
                    latent[:, :, start:end],
                    node_rng.keyed("converter", spec.spec_index),
                    plan,
                )
                if int(spec_out.shape[2]) != (end - start):
                    if int(spec_out.shape[2]) > (end - start):
                        spec_out = spec_out[:, :, : (end - start)]
                    else:
                        spec_out = torch.nn.functional.pad(
                            spec_out, (0, (end - start) - int(spec_out.shape[2]))
                        )
                latent[:, :, start:end] = spec_out
                extracted[str(spec.key)] = values
            continue

        if not group.uses_center_random_fn:
            _accumulate_runtime_metric(runtime_metrics_out, "mixed_converter_bucket_count", 1.0)
            _accumulate_runtime_metric(
                runtime_metrics_out,
                "mixed_converter_bucket_dataset_sum",
                float(node_rng.batch_size),
            )
            x_prime, values = _apply_categorical_group_batch(
                _categorical_group_input_views(latent, group.slices),
                node_rng,
                group.plan,
                n_categories=group.category_count,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                spec_indices=group.spec_indices,
            )
            for local_index, spec in enumerate(group.slices):
                start = int(spec.column_start)
                end = int(spec.column_end)
                spec_out = x_prime[:, :, local_index, :]
                if int(spec_out.shape[2]) != (end - start):
                    if int(spec_out.shape[2]) > (end - start):
                        spec_out = spec_out[:, :, : (end - start)]
                    else:
                        spec_out = torch.nn.functional.pad(
                            spec_out, (0, (end - start) - int(spec_out.shape[2]))
                        )
                latent[:, :, start:end] = spec_out
                extracted[str(spec.key)] = values[:, :, local_index]
            continue
        for spec in group.slices:
            start = int(spec.column_start)
            end = int(spec.column_end)
            spec_view = latent[:, :, start:end].unsqueeze(2)
            x_prime, values = _apply_categorical_group_batch(
                spec_view,
                node_rng.keyed("converter", spec.spec_index),
                group.plan,
                n_categories=max(2, int(spec.cardinality or 2)),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                compiled_nested_function_context=node_context.converter_nested_function_contexts[
                    spec.spec_index
                ],
                compiled_nested_function_node_rng=node_rng,
                compiled_nested_function_cached_child_rngs=cached_child_rngs,
            )
            spec_out = x_prime[:, :, 0, :]
            if int(spec_out.shape[2]) != (end - start):
                if int(spec_out.shape[2]) > (end - start):
                    spec_out = spec_out[:, :, : (end - start)]
                else:
                    spec_out = torch.nn.functional.pad(
                        spec_out, (0, (end - start) - int(spec_out.shape[2]))
                    )
            latent[:, :, start:end] = spec_out
            extracted[str(spec.key)] = values[:, :, 0]

    _accumulate_runtime_metric(
        runtime_metrics_out,
        "converter_elapsed_seconds",
        time.perf_counter() - converter_start,
    )
    _accumulate_runtime_metric(
        runtime_metrics_out,
        "converter_cpu_time_seconds",
        time.process_time() - converter_start_cpu,
    )

    scale = _resolve_compiled_program_rng(
        node_rng,
        path=("latent_scale",),
        cached_child_rngs=cached_child_rngs,
    ).log_uniform((node_rng.batch_size,), low=0.1, high=10.0)
    latent = latent * scale.view(-1, 1, 1)
    _accumulate_runtime_metric(
        runtime_metrics_out,
        "node_apply_elapsed_seconds",
        time.perf_counter() - node_start,
    )
    _accumulate_runtime_metric(
        runtime_metrics_out,
        "node_apply_cpu_time_seconds",
        time.process_time() - node_start_cpu,
    )
    return latent, extracted


def _apply_node_converters_and_scale_prepared(
    node_context: _PreparedNodeExecutionContext,
    latent: torch.Tensor,
    *,
    batch_node_context: _PreparedBatchNodeExecutionContext,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    runtime_metrics_out: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply only converter extraction and final latent scaling for one prepared node."""

    node_plan = node_context.node_plan
    node_rng = batch_node_context.node_rng
    cached_child_rngs = batch_node_context.child_rngs
    extracted: dict[str, torch.Tensor] = {}
    converter_start = time.perf_counter()
    converter_start_cpu = time.process_time()
    for group in _resolved_compiled_converter_groups(node_plan):
        if isinstance(group, CompiledNumericConverterGroup):
            if group.all_unit_width:
                grouped_input = torch.cat(
                    [
                        latent[:, :, int(spec.column_start) : int(spec.column_end)]
                        for spec in group.slices
                    ],
                    dim=2,
                )
                warp_enabled = torch.tensor(
                    group.warp_enabled,
                    device=latent.device,
                    dtype=torch.bool,
                )
                x_prime, values = _apply_numeric_converter_group_batch(
                    grouped_input,
                    node_rng,
                    warp_enabled,
                    spec_indices=group.spec_indices,
                )
                for local_index, spec in enumerate(group.slices):
                    start = int(spec.column_start)
                    end = int(spec.column_end)
                    latent[:, :, start:end] = x_prime[:, :, local_index : local_index + 1]
                    extracted[str(spec.key)] = values[:, :, local_index]
                continue
            for spec, plan in zip(group.slices, group.plans, strict=True):
                start = int(spec.column_start)
                end = int(spec.column_end)
                spec_out, values = apply_numeric_converter_plan_batch(
                    latent[:, :, start:end],
                    node_rng.keyed("converter", spec.spec_index),
                    plan,
                )
                if int(spec_out.shape[2]) != (end - start):
                    if int(spec_out.shape[2]) > (end - start):
                        spec_out = spec_out[:, :, : (end - start)]
                    else:
                        spec_out = torch.nn.functional.pad(
                            spec_out, (0, (end - start) - int(spec_out.shape[2]))
                        )
                latent[:, :, start:end] = spec_out
                extracted[str(spec.key)] = values
            continue

        if not group.uses_center_random_fn:
            x_prime, values = _apply_categorical_group_batch(
                _categorical_group_input_views(latent, group.slices),
                node_rng,
                group.plan,
                n_categories=group.category_count,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                spec_indices=group.spec_indices,
            )
            for local_index, spec in enumerate(group.slices):
                start = int(spec.column_start)
                end = int(spec.column_end)
                spec_out = x_prime[:, :, local_index, :]
                if int(spec_out.shape[2]) != (end - start):
                    if int(spec_out.shape[2]) > (end - start):
                        spec_out = spec_out[:, :, : (end - start)]
                    else:
                        spec_out = torch.nn.functional.pad(
                            spec_out, (0, (end - start) - int(spec_out.shape[2]))
                        )
                latent[:, :, start:end] = spec_out
                extracted[str(spec.key)] = values[:, :, local_index]
            continue

        for spec in group.slices:
            _accumulate_runtime_metric(runtime_metrics_out, "mixed_converter_bucket_count", 1.0)
            _accumulate_runtime_metric(
                runtime_metrics_out,
                "mixed_converter_bucket_dataset_sum",
                float(node_rng.batch_size),
            )
            start = int(spec.column_start)
            end = int(spec.column_end)
            spec_view = latent[:, :, start:end].unsqueeze(2)
            x_prime, values = _apply_categorical_group_batch(
                spec_view,
                node_rng.keyed("converter", spec.spec_index),
                group.plan,
                n_categories=max(2, int(spec.cardinality or 2)),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                compiled_nested_function_context=node_context.converter_nested_function_contexts[
                    spec.spec_index
                ],
                compiled_nested_function_node_rng=node_rng,
                compiled_nested_function_cached_child_rngs=cached_child_rngs,
            )
            spec_out = x_prime[:, :, 0, :]
            if int(spec_out.shape[2]) != (end - start):
                if int(spec_out.shape[2]) > (end - start):
                    spec_out = spec_out[:, :, : (end - start)]
                else:
                    spec_out = torch.nn.functional.pad(
                        spec_out, (0, (end - start) - int(spec_out.shape[2]))
                    )
            latent[:, :, start:end] = spec_out
            extracted[str(spec.key)] = values[:, :, 0]

    _accumulate_runtime_metric(
        runtime_metrics_out,
        "converter_elapsed_seconds",
        time.perf_counter() - converter_start,
    )
    _accumulate_runtime_metric(
        runtime_metrics_out,
        "converter_cpu_time_seconds",
        time.process_time() - converter_start_cpu,
    )

    scale = _resolve_compiled_program_rng(
        node_rng,
        path=("latent_scale",),
        cached_child_rngs=cached_child_rngs,
    ).log_uniform((node_rng.batch_size,), low=0.1, high=10.0)
    return latent * scale.view(-1, 1, 1), extracted


def _materialize_fixed_layout_output_batch(
    layout: LayoutPlan,
    *,
    task: str,
    batch_rng: FixedLayoutBatchRng,
    noise_spec: NoiseSamplingSpec | None,
    feature_values: list[torch.Tensor | None],
    target_values: torch.Tensor | None,
    device: str,
    emit_features: bool,
    runtime_metrics_out: dict[str, float] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, list[dict[str, Any]]]:
    """Materialize one per-layout raw feature/target batch from extracted tensors."""

    batch_size = int(batch_rng.batch_size)
    n_rows = int(target_values.shape[1]) if target_values is not None else 0
    num_features = int(layout.n_features)
    aux_meta_batch: list[dict[str, Any]] = [
        {"filter": {"mode": "deferred", "status": "not_run"}} for _ in range(batch_size)
    ]

    x_complete: torch.Tensor | None = None
    if emit_features:
        feature_start = time.perf_counter()
        feature_start_cpu = time.process_time()
        feature_types = list(layout.feature_types)
        card_by_feature = dict(layout.card_by_feature)
        feature_columns: list[torch.Tensor] = []
        for feature_index in range(num_features):
            feature_tensor = feature_values[feature_index]
            if feature_tensor is None:
                if feature_types[feature_index] == "cat":
                    cardinality = int(card_by_feature[feature_index])
                    feature_tensor = batch_rng.randint(0, cardinality, (batch_size, n_rows))
                else:
                    feature_tensor = sample_noise_from_spec(
                        (batch_size, n_rows),
                        generator=batch_rng.torch_generator,
                        device=device,
                        noise_spec=noise_spec,
                    )
            feature_columns.append(feature_tensor.to(torch.float32).unsqueeze(2))
        x_complete = (
            torch.cat(feature_columns, dim=2)
            if feature_columns
            else torch.empty((batch_size, n_rows, 0), dtype=torch.float32, device=device)
        )
        x_complete, _feature_types, _feature_index_map = postprocess_feature_matrix(
            x_complete,
            list(layout.feature_types),
            keyed_rng=None,
            preserve_feature_schema=True,
        )
        _accumulate_runtime_metric(
            runtime_metrics_out,
            "feature_materialization_elapsed_seconds",
            time.perf_counter() - feature_start,
        )
        _accumulate_runtime_metric(
            runtime_metrics_out,
            "feature_materialization_cpu_time_seconds",
            time.process_time() - feature_start_cpu,
        )

    if target_values is None:
        raise RuntimeError("Fixed-layout execution did not extract a latent target value.")
    if task == "classification":
        y = target_values.to(torch.int64) % int(layout.n_classes)
    else:
        y = target_values.to(torch.float32)
    return x_complete if emit_features else None, y, aux_meta_batch


def _build_mixed_supported_source_latent_bucket(
    node_context: _PreparedNodeExecutionContext,
    *,
    parent_data: list[torch.Tensor],
    cohort_runs: list[_MixedNodeRun],
    n_rows: int,
    noise_sigma_multiplier: float,
) -> torch.Tensor:
    """Build one mixed physical bucket for supported compiled node sources."""

    node_plan = node_context.node_plan
    total_dim = int(node_plan.latent.total_dim)
    source = node_plan.source
    if not _source_supports_mixed_execution(source):
        raise ValueError("Mixed supported source bucket requires mixed-compatible sources.")

    if parent_data:
        if isinstance(source, ConcatNodeSource):
            if node_context.concat_function is None:
                raise ValueError("Prepared concat node context is incomplete.")
            combined_input = _sanitize_and_batch_standardize(torch.cat(parent_data, dim=2))
            latent = _apply_mixed_compiled_function_plan_batch(
                combined_input,
                cohort_runs=[
                    _MixedFunctionRun(
                        function_context=run.node_context.concat_function,
                        batch_node_context=run.batch_node_context,
                        start=int(run.start),
                        stop=int(run.stop),
                        noise_spec=run.noise_spec,
                    )
                    for run in cohort_runs
                    if run.node_context.concat_function is not None
                ],
                context=node_context.concat_function,
                out_dim=total_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                standardize_input=False,
            )
        else:
            if not isinstance(source, StackedNodeSource):
                raise ValueError("Parent-driven mixed source bucket requires a multi-input source.")
            standardized_parents = [
                _sanitize_and_batch_standardize(parent_tensor) for parent_tensor in parent_data
            ]
            transformed_outputs: list[torch.Tensor] = []
            for plan_index, standardized_parent in enumerate(standardized_parents):
                parent_function_context = node_context.parent_function_contexts[plan_index]
                transformed_outputs.append(
                    _apply_mixed_compiled_function_plan_batch(
                        standardized_parent,
                        cohort_runs=[
                            _MixedFunctionRun(
                                function_context=run.node_context.parent_function_contexts[
                                    plan_index
                                ],
                                batch_node_context=run.batch_node_context,
                                start=int(run.start),
                                stop=int(run.stop),
                                noise_spec=run.noise_spec,
                            )
                            for run in cohort_runs
                        ],
                        context=parent_function_context,
                        out_dim=total_dim,
                        noise_sigma_multiplier=noise_sigma_multiplier,
                        standardize_input=False,
                    )
                )
            if source.aggregation_kind == "logsumexp":
                latent = _aggregate_parent_outputs_batch(
                    torch.stack(transformed_outputs, dim=2),
                    aggregation_kind=source.aggregation_kind,
                )
            else:
                aggregate: torch.Tensor | None = None
                for transformed in transformed_outputs:
                    aggregate = (
                        transformed
                        if aggregate is None
                        else _aggregate_batch_incrementally(
                            aggregate,
                            transformed,
                            aggregation_kind=source.aggregation_kind,
                        )
                    )
                if aggregate is None:
                    raise RuntimeError("Expected at least one parent tensor for stacked node plan.")
                latent = aggregate
    else:
        if not isinstance(source, RandomPointsNodeSource):
            raise ValueError("Root mixed source bucket must use random-points source.")
        if (
            node_context.root_source_base_rng_path is None
            or node_context.root_source_function is None
        ):
            raise ValueError("Prepared root-node execution context is incomplete.")
        base = torch.cat(
            [
                _sample_random_points_batch(
                    _resolve_compiled_program_rng(
                        run.batch_node_context.node_rng,
                        path=(
                            run.node_context.root_source_base_rng_path
                            if run.node_context.root_source_base_rng_path is not None
                            else ()
                        ),
                        cached_child_rngs=run.batch_node_context.child_rngs,
                    ),
                    n_rows=n_rows,
                    dim=total_dim,
                    base_kind=cast(
                        RandomPointsNodeSource,
                        run.node_context.node_plan.source,
                    ).base_kind,
                    noise_sigma_multiplier=noise_sigma_multiplier,
                    noise_spec=run.noise_spec,
                )
                for run in cohort_runs
            ],
            dim=0,
        )
        standardized_base = _sanitize_and_batch_standardize(base)
        latent = _apply_mixed_compiled_function_plan_batch(
            standardized_base,
            cohort_runs=[
                _MixedFunctionRun(
                    function_context=run.node_context.root_source_function,
                    batch_node_context=run.batch_node_context,
                    start=int(run.start),
                    stop=int(run.stop),
                    noise_spec=run.noise_spec,
                )
                for run in cohort_runs
                if run.node_context.root_source_function is not None
            ],
            context=node_context.root_source_function,
            out_dim=total_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            standardize_input=False,
        )

    latent = torch.nan_to_num(latent.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    latent = torch.clamp(latent, -1e6, 1e6)
    latent = _batch_standardize(latent)
    weights = torch.cat(
        [
            _sample_random_weights_batch(
                _resolve_compiled_program_rng(
                    run.batch_node_context.node_rng,
                    path=("latent_weights",),
                    cached_child_rngs=run.batch_node_context.child_rngs,
                ),
                dim=int(latent.shape[2]),
                sigma_multiplier=float(noise_sigma_multiplier),
                noise_spec=run.noise_spec,
            )
            for run in cohort_runs
        ],
        dim=0,
    )
    latent = latent * weights.unsqueeze(1)
    mean_l2 = torch.mean(torch.norm(latent, dim=2), dim=1)
    return latent / torch.clamp(mean_l2.view(-1, 1, 1), min=1e-6)


def _generate_mixed_fixed_layout_graph_batch_prepared(
    cohorts: list[_MixedPreparedLogicalCohort],
    *,
    device: str,
    noise_sigma_multiplier: float,
    runtime_metrics_out: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    """Generate one mixed physical microbatch from several exact logical cohorts."""

    if not cohorts:
        raise ValueError("cohorts must be non-empty.")
    raw_batch_start = time.perf_counter()
    raw_batch_start_cpu = time.process_time()
    total_items = int(sum(len(cohort.dataset_seeds) for cohort in cohorts))
    batch_rngs = [
        _batch_rng_for_dataset_seeds(cohort.dataset_seeds, device=device) for cohort in cohorts
    ]
    batch_execution_contexts = [
        _build_fixed_layout_batch_execution_context(cohort.prepared_execution_context, batch_rng)
        for cohort, batch_rng in zip(cohorts, batch_rngs, strict=True)
    ]
    cohort_feature_values: list[list[torch.Tensor | None]] = [
        [None] * int(cohort.layout.n_features) for cohort in cohorts
    ]
    cohort_target_values: list[torch.Tensor | None] = [None] * len(cohorts)
    cohort_node_outputs: list[list[torch.Tensor | None]] = [
        [None] * len(cohort.execution_plan.node_plans) for cohort in cohorts
    ]

    max_graph_nodes = max(len(cohort.execution_plan.node_plans) for cohort in cohorts)
    for node_index in range(max_graph_nodes):
        supported_buckets: dict[tuple[object, ...], list[int]] = {}
        unsupported_cohorts: list[int] = []
        active_cohort_indices: list[int] = []
        for cohort_index, cohort in enumerate(cohorts):
            if node_index >= len(cohort.execution_plan.node_plans):
                continue
            active_cohort_indices.append(int(cohort_index))
            node_plan = cohort.execution_plan.node_plans[node_index]
            parent_data = [
                cohort_node_outputs[cohort_index][int(parent_index)]
                for parent_index in node_plan.parent_indices
            ]
            if any(parent is None for parent in parent_data):
                raise RuntimeError("Parent node output is missing for mixed physical microbatch.")
            resolved_parent_data = cast(list[torch.Tensor], parent_data)
            if not _source_supports_mixed_execution(node_plan.source):
                unsupported_cohorts.append(int(cohort_index))
                continue
            supported_buckets.setdefault(
                _source_bucket_key(
                    cohort.prepared_execution_context.node_contexts[node_index],
                    parent_widths=tuple(int(parent.shape[2]) for parent in resolved_parent_data),
                ),
                [],
            ).append(int(cohort_index))

        if active_cohort_indices:
            _accumulate_runtime_metric(runtime_metrics_out, "mixed_source_node_slot_count", 1.0)

        for cohort_index in unsupported_cohorts:
            cohort = cohorts[cohort_index]
            batch_node_context = batch_execution_contexts[cohort_index].node_contexts[node_index]
            parent_data = [
                cohort_node_outputs[cohort_index][int(parent_index)]
                for parent_index in cohort.execution_plan.node_plans[node_index].parent_indices
            ]
            latent, extracted = _apply_node_plan_batch_prepared(
                None,
                cohort.prepared_execution_context.node_contexts[node_index],
                [parent for parent in parent_data if parent is not None],
                n_rows=int(cohort.n_rows),
                batch_node_context=batch_node_context,
                device=device,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=cohort.noise_spec,
                runtime_metrics_out=runtime_metrics_out,
            )
            cohort_node_outputs[cohort_index][node_index] = latent
            for key, values in extracted.items():
                if key.startswith("feature_"):
                    cohort_feature_values[cohort_index][int(key.split("_", 1)[1])] = values
                elif key == "target":
                    cohort_target_values[cohort_index] = values
                else:
                    raise ValueError(f"Unexpected extracted fixed-layout key {key!r}.")

        for cohort_indices in supported_buckets.values():
            bucket_start = time.perf_counter()
            bucket_start_cpu = time.process_time()
            _accumulate_runtime_metric(runtime_metrics_out, "mixed_source_bucket_count", 1.0)
            _accumulate_runtime_metric(
                runtime_metrics_out,
                "mixed_source_bucket_dataset_sum",
                float(len(cohort_indices)),
            )
            if len(cohort_indices) == 1:
                cohort_index = cohort_indices[0]
                cohort = cohorts[cohort_index]
                parent_data = [
                    cohort_node_outputs[cohort_index][int(parent_index)]
                    for parent_index in cohort.execution_plan.node_plans[node_index].parent_indices
                ]
                latent, extracted = _apply_node_plan_batch_prepared(
                    None,
                    cohort.prepared_execution_context.node_contexts[node_index],
                    [parent for parent in parent_data if parent is not None],
                    n_rows=int(cohort.n_rows),
                    batch_node_context=batch_execution_contexts[cohort_index].node_contexts[
                        node_index
                    ],
                    device=device,
                    noise_sigma_multiplier=noise_sigma_multiplier,
                    noise_spec=cohort.noise_spec,
                    runtime_metrics_out=runtime_metrics_out,
                )
                cohort_node_outputs[cohort_index][node_index] = latent
                for key, values in extracted.items():
                    if key.startswith("feature_"):
                        cohort_feature_values[cohort_index][int(key.split("_", 1)[1])] = values
                    elif key == "target":
                        cohort_target_values[cohort_index] = values
                    else:
                        raise ValueError(f"Unexpected extracted fixed-layout key {key!r}.")
                continue
            representative = cohorts[cohort_indices[0]]
            representative_context = representative.prepared_execution_context.node_contexts[
                node_index
            ]
            representative_parent_count = len(
                representative.execution_plan.node_plans[node_index].parent_indices
            )
            bucket_parent_data: list[torch.Tensor] = []
            for parent_slot in range(representative_parent_count):
                parent_slot_outputs = [
                    cohort_node_outputs[cohort_index][
                        int(
                            cohorts[cohort_index]
                            .execution_plan.node_plans[node_index]
                            .parent_indices[parent_slot]
                        )
                    ]
                    for cohort_index in cohort_indices
                ]
                bucket_parent_data.append(
                    torch.cat(cast(list[torch.Tensor], parent_slot_outputs), dim=0)
                )
            cohort_runs: list[_MixedNodeRun] = []
            position = 0
            for cohort_index in cohort_indices:
                cohort = cohorts[cohort_index]
                size = len(cohort.dataset_seeds)
                cohort_runs.append(
                    _MixedNodeRun(
                        node_context=cohort.prepared_execution_context.node_contexts[node_index],
                        batch_node_context=batch_execution_contexts[cohort_index].node_contexts[
                            node_index
                        ],
                        start=int(position),
                        stop=int(position + size),
                        noise_spec=cohort.noise_spec,
                    )
                )
                position += size
            latent = _build_mixed_supported_source_latent_bucket(
                representative_context,
                parent_data=bucket_parent_data,
                cohort_runs=cohort_runs,
                n_rows=int(representative.n_rows),
                noise_sigma_multiplier=noise_sigma_multiplier,
            )
            for run, cohort_index in zip(
                cohort_runs,
                cohort_indices,
                strict=True,
            ):
                cohort_node_context = cohorts[
                    cohort_index
                ].prepared_execution_context.node_contexts[node_index]
                cohort_latent, extracted = _apply_node_converters_and_scale_prepared(
                    cohort_node_context,
                    latent[run.start : run.stop],
                    batch_node_context=run.batch_node_context,
                    noise_sigma_multiplier=noise_sigma_multiplier,
                    noise_spec=cohorts[cohort_index].noise_spec,
                    runtime_metrics_out=runtime_metrics_out,
                )
                cohort_node_outputs[cohort_index][node_index] = cohort_latent
                for key, values in extracted.items():
                    if key.startswith("feature_"):
                        cohort_feature_values[cohort_index][int(key.split("_", 1)[1])] = values
                    elif key == "target":
                        cohort_target_values[cohort_index] = values
                    else:
                        raise ValueError(f"Unexpected extracted fixed-layout key {key!r}.")
            _accumulate_runtime_metric(
                runtime_metrics_out,
                "node_apply_elapsed_seconds",
                time.perf_counter() - bucket_start,
            )
            _accumulate_runtime_metric(
                runtime_metrics_out,
                "node_apply_cpu_time_seconds",
                time.process_time() - bucket_start_cpu,
            )

    x_batches: list[torch.Tensor] = []
    y_batches: list[torch.Tensor] = []
    aux_meta_batch: list[dict[str, object]] = []
    for cohort_index, cohort in enumerate(cohorts):
        x_batch, y_batch, cohort_aux = _materialize_fixed_layout_output_batch(
            cohort.layout,
            task=cohort.task,
            batch_rng=batch_rngs[cohort_index],
            noise_spec=cohort.noise_spec,
            feature_values=cohort_feature_values[cohort_index],
            target_values=cohort_target_values[cohort_index],
            device=device,
            emit_features=True,
            runtime_metrics_out=runtime_metrics_out,
        )
        if x_batch is None:
            raise RuntimeError("Expected mixed fixed-layout feature batch to be materialized.")
        x_batches.append(x_batch)
        y_batches.append(y_batch)
        aux_meta_batch.extend(cohort_aux)

    _accumulate_runtime_metric(
        runtime_metrics_out,
        "raw_batch_elapsed_seconds",
        time.perf_counter() - raw_batch_start,
    )
    _accumulate_runtime_metric(
        runtime_metrics_out,
        "raw_batch_cpu_time_seconds",
        time.process_time() - raw_batch_start_cpu,
    )
    if int(sum(int(x.shape[0]) for x in x_batches)) != total_items:
        raise RuntimeError("Mixed physical microbatch emitted an inconsistent batch size.")
    return torch.cat(x_batches, dim=0), torch.cat(y_batches, dim=0), aux_meta_batch


def _generate_fixed_layout_raw_batch(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    execution_plan: FixedLayoutExecutionPlan,
    dataset_seeds: list[int],
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    emit_features: bool,
    prepared_execution_context: _PreparedFixedLayoutExecutionContext | None = None,
    active_node_indices: tuple[int, ...] | None = None,
    runtime_metrics_out: dict[str, float] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, list[dict[str, Any]]]:
    """Generate one fixed-layout microbatch of complete features and raw targets."""

    if not dataset_seeds:
        raise ValueError("dataset_seeds must be non-empty.")
    raw_batch_start = time.perf_counter()
    raw_batch_start_cpu = time.process_time()
    batch_size = len(dataset_seeds)
    n_rows = int(config.dataset.n_train + config.dataset.n_test)
    num_features = int(layout.n_features)
    dtype = torch.float32
    batch_seed = KeyedRng(int(dataset_seeds[0])).child_seed("fixed_layout_chunk", batch_size)
    rng = FixedLayoutBatchRng(seed=batch_seed, batch_size=batch_size, device=device)
    prepared_batch_context = (
        None
        if prepared_execution_context is None
        else _build_fixed_layout_batch_execution_context(prepared_execution_context, rng)
    )

    node_outputs: list[torch.Tensor | None] = [None] * int(layout.graph_nodes)
    feature_values: list[torch.Tensor | None] = [None] * num_features
    target_values: torch.Tensor | None = None
    aux_meta_batch: list[dict[str, Any]] = [
        {"filter": {"mode": "deferred", "status": "not_run"}} for _ in dataset_seeds
    ]

    node_indices = (
        tuple(int(node_index) for node_index in active_node_indices)
        if active_node_indices is not None
        else tuple(range(len(execution_plan.node_plans)))
    )
    for node_index in node_indices:
        node_plan = execution_plan.node_plans[int(node_index)]
        parent_tensors = []
        for parent_index in node_plan.parent_indices:
            parent_output = node_outputs[int(parent_index)]
            if parent_output is not None:
                parent_tensors.append(parent_output)
        if prepared_execution_context is None or prepared_batch_context is None:
            node_rng = rng.keyed("node", node_index)
            latent, extracted = _apply_node_plan_batch(
                config,
                node_plan,
                parent_tensors,
                n_rows=n_rows,
                rng=node_rng,
                device=device,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                runtime_metrics_out=runtime_metrics_out,
            )
        else:
            latent, extracted = _apply_node_plan_batch_prepared(
                config,
                prepared_execution_context.node_contexts[int(node_index)],
                parent_tensors,
                n_rows=n_rows,
                batch_node_context=prepared_batch_context.node_contexts[int(node_index)],
                device=device,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                runtime_metrics_out=runtime_metrics_out,
            )
        node_outputs[node_index] = latent
        for key, values in extracted.items():
            if key.startswith("feature_"):
                feature_index = int(key.split("_", 1)[1])
                feature_values[feature_index] = values
            elif key == "target":
                if target_values is not None:
                    raise ValueError(
                        "Fixed-layout node extraction produced multiple target values."
                    )
                target_values = values
            else:
                raise ValueError(f"Unexpected extracted fixed-layout key {key!r}.")

    x_complete: torch.Tensor | None = None
    if emit_features:
        feature_start = time.perf_counter()
        feature_start_cpu = time.process_time()
        feature_types = list(layout.feature_types)
        card_by_feature = dict(layout.card_by_feature)
        feature_columns: list[torch.Tensor] = []
        for feature_index in range(num_features):
            feature_tensor = feature_values[feature_index]
            if feature_tensor is None:
                if feature_types[feature_index] == "cat":
                    cardinality = int(card_by_feature[feature_index])
                    feature_tensor = rng.randint(0, cardinality, (batch_size, n_rows))
                else:
                    feature_tensor = sample_noise_from_spec(
                        (batch_size, n_rows),
                        generator=rng.torch_generator,
                        device=device,
                        noise_spec=noise_spec,
                    )
            feature_columns.append(feature_tensor.to(dtype).unsqueeze(2))
        x_complete = (
            torch.cat(feature_columns, dim=2)
            if feature_columns
            else torch.empty((batch_size, n_rows, 0), dtype=dtype, device=device)
        )
        x_complete, _feature_types, _feature_index_map = postprocess_feature_matrix(
            x_complete,
            list(layout.feature_types),
            keyed_rng=None,
            preserve_feature_schema=True,
        )
        _accumulate_runtime_metric(
            runtime_metrics_out,
            "feature_materialization_elapsed_seconds",
            time.perf_counter() - feature_start,
        )
        _accumulate_runtime_metric(
            runtime_metrics_out,
            "feature_materialization_cpu_time_seconds",
            time.process_time() - feature_start_cpu,
        )

    if target_values is None:
        raise RuntimeError("Fixed-layout execution did not extract a latent target value.")
    if str(config.dataset.task) == "classification":
        y = target_values.to(torch.int64) % int(layout.n_classes)
    else:
        y = target_values.to(dtype)
    _accumulate_runtime_metric(
        runtime_metrics_out,
        "raw_batch_elapsed_seconds",
        time.perf_counter() - raw_batch_start,
    )
    _accumulate_runtime_metric(
        runtime_metrics_out,
        "raw_batch_cpu_time_seconds",
        time.process_time() - raw_batch_start_cpu,
    )
    return x_complete if emit_features else None, y, aux_meta_batch


def generate_fixed_layout_graph_batch(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    execution_plan: FixedLayoutExecutionPlan,
    dataset_seeds: list[int],
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    runtime_metrics_out: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    """Generate one fixed-layout microbatch of raw `x`/`y` tensors."""

    x, y, aux_meta_batch = _generate_fixed_layout_raw_batch(
        config,
        layout,
        execution_plan=execution_plan,
        dataset_seeds=dataset_seeds,
        device=device,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
        emit_features=True,
        runtime_metrics_out=runtime_metrics_out,
    )
    if x is None:
        raise RuntimeError("Expected fixed-layout feature batch to be materialized.")
    return x, y, aux_meta_batch


def generate_fixed_layout_label_batch(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    execution_plan: FixedLayoutExecutionPlan,
    dataset_seeds: list[int],
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    runtime_metrics_out: dict[str, float] | None = None,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Generate one fixed-layout microbatch of raw target tensors only."""

    _x, y, aux_meta_batch = _generate_fixed_layout_raw_batch(
        config,
        layout,
        execution_plan=execution_plan,
        dataset_seeds=dataset_seeds,
        device=device,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
        emit_features=False,
        runtime_metrics_out=runtime_metrics_out,
    )
    return y, aux_meta_batch


def _generate_fixed_layout_graph_batch_prepared(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    execution_plan: FixedLayoutExecutionPlan,
    prepared_execution_context: _PreparedFixedLayoutExecutionContext,
    dataset_seeds: list[int],
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    runtime_metrics_out: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    """Generate one fixed-layout graph batch using prepared execution metadata."""

    x, y, aux_meta_batch = _generate_fixed_layout_raw_batch(
        config,
        layout,
        execution_plan=execution_plan,
        dataset_seeds=dataset_seeds,
        device=device,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
        emit_features=True,
        prepared_execution_context=prepared_execution_context,
        runtime_metrics_out=runtime_metrics_out,
    )
    if x is None:
        raise RuntimeError("Expected fixed-layout feature batch to be materialized.")
    return x, y, aux_meta_batch


def _generate_fixed_layout_validation_label_batch(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    execution_plan: FixedLayoutExecutionPlan,
    prepared_execution_context: _PreparedFixedLayoutExecutionContext,
    dataset_seeds: list[int],
    device: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    runtime_metrics_out: dict[str, float] | None = None,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Generate one exact target-only label batch for replay validation."""

    _x, y, aux_meta_batch = _generate_fixed_layout_raw_batch(
        config,
        layout,
        execution_plan=execution_plan,
        dataset_seeds=dataset_seeds,
        device=device,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
        emit_features=False,
        prepared_execution_context=prepared_execution_context,
        active_node_indices=prepared_execution_context.target_ancestor_node_indices,
        runtime_metrics_out=runtime_metrics_out,
    )
    return y, aux_meta_batch


__all__ = [
    "FixedLayoutBatchRng",
    "_apply_activation_plan",
    "_apply_categorical_group_batch",
    "_apply_compiled_function_plan_batch_core",
    "_apply_node_plan_batch",
    "_apply_node_plan_batch_prepared",
    "_compile_function_execution_context",
    "_generate_fixed_layout_graph_batch_prepared",
    "_generate_fixed_layout_raw_batch",
    "_generate_fixed_layout_validation_label_batch",
    "_lp_distances_to_centers",
    "_nearest_lp_center_indices",
    "_prepare_fixed_layout_execution_context",
    "_sample_random_matrix_from_plan_batch",
    "apply_compiled_function_plan_batch",
    "apply_function_plan_batch",
    "apply_numeric_converter_plan_batch",
    "build_fixed_layout_execution_plan",
    "fixed_layout_plan_signature",
    "generate_fixed_layout_graph_batch",
    "generate_fixed_layout_label_batch",
]
