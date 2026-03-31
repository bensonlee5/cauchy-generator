"""Batched fixed-layout execution-plan sampling and generation helpers."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from typing import Any

import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core.execution_semantics import (
    sample_node_plan,
)
from dagzoo.core.layout import _build_node_specs, _build_target_specs
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.postprocess.postprocess import postprocess_feature_matrix
from dagzoo.rng import KeyedRng
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
    _apply_linear_batch,
    _apply_nn_batch,
    _apply_quadratic_batch,
    _apply_tree_batch,
    _sample_random_matrix_from_plan_batch,
    _sample_random_points_batch,
)
from .plan_types import (
    DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT,
    CategoricalConverterPlan,
    CompiledCategoricalConverterGroup,
    CompiledNumericConverterGroup,
    ConcatNodeSource,
    DiscretizationFunctionPlan,
    EmFunctionPlan,
    FixedLayoutCompiledConverterGroup,
    FixedLayoutCompiledConverterSlice,
    FixedLayoutConverterSpec,
    FixedLayoutExecutionPlan,
    FixedLayoutFunctionPlan,
    FixedLayoutNodePlan,
    FixedLayoutTargetHeadPlan,
    GpFunctionPlan,
    LinearFunctionPlan,
    NeuralNetFunctionPlan,
    NumericConverterGroup,
    NumericConverterPlan,
    PiecewiseFunctionPlan,
    ProductFunctionPlan,
    QuadraticFunctionPlan,
    RandomPointsNodeSource,
    StackedNodeSource,
    TreeFunctionPlan,
    fixed_layout_signature_payloads,
)

_FIXED_LAYOUT_EXECUTION_CONTRACT = DEFAULT_FIXED_LAYOUT_EXECUTION_CONTRACT


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
) -> FixedLayoutExecutionPlan:
    """Build one reusable per-node execution-plan payload for fixed-layout batches."""

    plan_root = KeyedRng(int(plan_seed))
    task = str(config.dataset.task)
    node_plans: list[FixedLayoutNodePlan] = []
    for node_index, parent_indices in enumerate(_parent_index_lists(layout)):
        spec_root = plan_root.keyed("node_spec", node_index)
        node_root = plan_root.keyed("node_plan", node_index)
        converter_specs = _build_node_specs(node_index, layout, spec_root)
        node_plans.append(
            _with_compiled_converter_groups(
                sample_node_plan(
                    node_index=int(node_index),
                    parent_indices=parent_indices,
                    converter_specs=converter_specs,
                    keyed_rng=node_root,
                    device="cpu",
                    mechanism_logit_tilt=mechanism_logit_tilt,
                    function_family_mix=config.mechanism.function_family_mix,
                )
            )
        )
    target_head_plan = _with_compiled_converter_groups(
        sample_node_plan(
            node_index=int(layout.graph_nodes),
            parent_indices=tuple(int(index) for index in layout.target_parent_features),
            converter_specs=_build_target_specs(layout, task),
            keyed_rng=plan_root.keyed("target_head"),
            device="cpu",
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=config.mechanism.function_family_mix,
        )
    )
    return FixedLayoutExecutionPlan(
        node_plans=tuple(node_plans),
        target_head_plan=FixedLayoutTargetHeadPlan(
            parent_feature_indices=tuple(int(index) for index in layout.target_parent_features),
            node_plan=target_head_plan,
        ),
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


def _apply_function_plan_batch_core(
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
    if isinstance(plan, ProductFunctionPlan):
        branch_input = _batch_standardize(y)
        lhs = _apply_function_plan_batch_core(
            branch_input,
            rng.keyed("product", "lhs"),
            plan.lhs,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        rhs = _apply_function_plan_batch_core(
            branch_input,
            rng.keyed("product", "rhs"),
            plan.rhs,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        return lhs * rhs
    if isinstance(plan, PiecewiseFunctionPlan):
        gate_matrix = _sample_random_matrix_from_plan_batch(
            plan.gate_matrix,
            out_dim=1,
            in_dim=int(y.shape[2]),
            rng=rng.keyed("piecewise", "gate_matrix"),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        gate_projection = torch.einsum("bni,boi->bno", y, gate_matrix)
        gate = torch.sigmoid(
            (gate_projection + float(plan.gate_bias)) * float(plan.gate_temperature)
        )
        branch_input = _batch_standardize(y)
        lhs = _apply_function_plan_batch_core(
            branch_input,
            rng.keyed("piecewise", "lhs"),
            plan.lhs,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        rhs = _apply_function_plan_batch_core(
            branch_input,
            rng.keyed("piecewise", "rhs"),
            plan.rhs,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        return gate * lhs + (1.0 - gate) * rhs
    raise ValueError(f"Unsupported fixed-layout function plan: {plan!r}")


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
    target_teacher_conditionals_out: dict[str, torch.Tensor] | None = None,
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
            class_probs_payload: dict[str, torch.Tensor] | None = (
                {} if target_teacher_conditionals_out is not None else None
            )
            x_prime, values = _apply_categorical_group_batch(
                _categorical_group_input_views(latent, group.slices),
                rng,
                group.plan,
                n_categories=group.category_count,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                spec_indices=group.spec_indices,
                class_probs_out=class_probs_payload,
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
                if (
                    target_teacher_conditionals_out is not None
                    and str(spec.key) == "target"
                    and class_probs_payload is not None
                    and "probs" in class_probs_payload
                ):
                    target_teacher_conditionals_out["probs"] = class_probs_payload["probs"][
                        :, :, local_index, :
                    ]
            continue
        for spec in group.slices:
            start = int(spec.column_start)
            end = int(spec.column_end)
            spec_view = latent[:, :, start:end].unsqueeze(2)
            single_class_probs_payload: dict[str, torch.Tensor] | None = (
                {} if target_teacher_conditionals_out is not None else None
            )
            x_prime, values = _apply_categorical_group_batch(
                spec_view,
                rng.keyed("converter", spec.spec_index),
                group.plan,
                n_categories=max(2, int(spec.cardinality or 2)),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                class_probs_out=single_class_probs_payload,
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
            if (
                target_teacher_conditionals_out is not None
                and str(spec.key) == "target"
                and single_class_probs_payload is not None
                and "probs" in single_class_probs_payload
            ):
                target_teacher_conditionals_out["probs"] = single_class_probs_payload["probs"][
                    :, :, 0, :
                ]

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

    node_outputs: list[torch.Tensor | None] = [None] * int(layout.graph_nodes)
    feature_values: list[torch.Tensor | None] = [None] * num_features
    aux_meta_batch: list[dict[str, Any]] = [
        {"filter": {"mode": "deferred", "status": "not_run"}} for _ in dataset_seeds
    ]

    for node_index, node_plan in enumerate(execution_plan.node_plans):
        parent_tensors = []
        for parent_index in node_plan.parent_indices:
            parent_output = node_outputs[int(parent_index)]
            if parent_output is not None:
                parent_tensors.append(parent_output)
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
        node_outputs[node_index] = latent
        for key, values in extracted.items():
            if key.startswith("feature_"):
                feature_index = int(key.split("_", 1)[1])
                feature_values[feature_index] = values
            else:
                raise ValueError(f"Unexpected extracted fixed-layout key {key!r}.")

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

    target_head = execution_plan.target_head_plan
    target_values: torch.Tensor | None
    teacher_conditionals_payload: dict[str, torch.Tensor] | None = (
        {}
        if str(config.dataset.task) == "classification"
        and config.diagnostics.teacher_conditional_export
        else None
    )
    if target_head is None:
        target_values = None
    else:
        target_start = time.perf_counter()
        target_start_cpu = time.process_time()
        target_parent_tensors = [
            x_complete[:, :, int(feature_index) : int(feature_index) + 1]
            for feature_index in target_head.parent_feature_indices
        ]
        _target_latent, extracted = _apply_node_plan_batch(
            config,
            target_head.node_plan,
            target_parent_tensors,
            n_rows=n_rows,
            rng=rng.keyed("target_head"),
            device=device,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
            runtime_metrics_out=runtime_metrics_out,
            target_teacher_conditionals_out=teacher_conditionals_payload,
        )
        target_values = extracted.get("target")
        _accumulate_runtime_metric(
            runtime_metrics_out,
            "target_materialization_elapsed_seconds",
            time.perf_counter() - target_start,
        )
        _accumulate_runtime_metric(
            runtime_metrics_out,
            "target_materialization_cpu_time_seconds",
            time.process_time() - target_start_cpu,
        )

    if target_values is None:
        if str(config.dataset.task) == "classification":
            y = rng.randint(0, int(layout.n_classes), (batch_size, n_rows)).to(torch.int64)
        else:
            y = sample_noise_from_spec(
                (batch_size, n_rows),
                generator=rng.torch_generator,
                device=device,
                noise_spec=noise_spec,
            ).to(dtype)
    else:
        if str(config.dataset.task) == "classification":
            y = target_values.to(torch.int64) % int(layout.n_classes)
        else:
            y = target_values.to(dtype)
    if teacher_conditionals_payload is not None and "probs" in teacher_conditionals_payload:
        target_probs = teacher_conditionals_payload["probs"].detach().to(device="cpu")
        class_labels = list(range(int(target_probs.shape[-1])))
        for batch_index in range(batch_size):
            aux_meta_batch[batch_index]["teacher_conditional_class_labels"] = list(class_labels)
            aux_meta_batch[batch_index]["teacher_conditional_full_probs"] = target_probs[
                batch_index
            ].tolist()
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


__all__ = [
    "FixedLayoutBatchRng",
    "_apply_activation_plan",
    "_apply_categorical_group_batch",
    "_apply_node_plan_batch",
    "_generate_fixed_layout_raw_batch",
    "_lp_distances_to_centers",
    "_nearest_lp_center_indices",
    "_sample_random_matrix_from_plan_batch",
    "apply_function_plan_batch",
    "apply_numeric_converter_plan_batch",
    "build_fixed_layout_execution_plan",
    "fixed_layout_plan_signature",
    "generate_fixed_layout_graph_batch",
    "generate_fixed_layout_label_batch",
]
