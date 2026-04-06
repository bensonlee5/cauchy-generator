"""Canonical fixed-layout run preparation and execution orchestration."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core.generation_context import (
    _resolve_device,
    _resolve_run_seed,
    _resolve_split_sizes,
    _torch_dtype,
    _validate_class_split_for_layout,
)
from dagzoo.core.generation_runtime import (
    _build_fixed_schema_finalization_context,
    _finalize_generated_chunk_preserve_schema,
    _finalize_generated_chunk_variable_schema,
    _finalize_generated_tensors,
    _FixedSchemaFinalizationContext,
    _resolve_split_indices,
)
from dagzoo.core.layout import _resample_layout_graph, _sample_layout
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.core.noise_runtime import (
    _noise_sampling_spec,
    _resolve_noise_runtime_selection,
)
from dagzoo.core.shift import ShiftRuntimeParams, resolve_shift_runtime_params
from dagzoo.core.steering import resolve_steering
from dagzoo.core.validation import (
    RECOVERABLE_RETRY_SCOPE_NEXT_PLAN_CANDIDATE,
    RECOVERABLE_RETRY_SCOPE_SAME_PLAN_ATTEMPT,
    InfeasibleStratifiedSplitError,
    RecoverableGenerationFailure,
    _classification_split_valid,
    _stratified_split_indices,
    classify_recoverable_generation_failure,
)
from dagzoo.filtering.structural_validity import (
    StructuralValidityConfig,
    evaluate_layout_structural_validity,
)
from dagzoo.rng import KeyedRng
from dagzoo.types import DatasetBundle

from .batched import (
    _generate_fixed_layout_graph_batch_prepared,
    _generate_fixed_layout_validation_label_batch,
    _prepare_fixed_layout_execution_context,
    build_fixed_layout_execution_plan,
    fixed_layout_plan_signature,
    generate_fixed_layout_graph_batch,
    generate_fixed_layout_label_batch,
)
from .grouped import (
    _GroupedRawBatch,
    _NoiseRuntimeGroup,
)
from .grouped import generate_grouped_raw_batches as _generate_grouped_raw_batches_impl
from .grouped import group_noise_runtime_chunk as _group_noise_runtime_chunk_impl
from .interventions import resolve_fixed_layout_intervention_plan
from .metadata import (
    _annotate_fixed_layout_metadata,
    _extract_emitted_schema_signature,
    _FixedLayoutPlan,
    _layout_signature,
)
from .plan_types import FixedLayoutExecutionPlan
from .prepare import (
    _effective_fixed_layout_target_cells,
    _resolve_fixed_layout_batch_size,
    _validate_fixed_layout_rows_mode,
    realize_generation_config_for_run,
)


@dataclass(slots=True)
class CanonicalFixedLayoutRun:
    """Prepared fixed-layout run context for canonical public generation."""

    config: GeneratorConfig
    plan: _FixedLayoutPlan
    run_seed: int
    requested_device: str
    resolved_device: str
    batch_size: int
    classification_attempt_plan: tuple[int, ...] | None = None


@dataclass(slots=True)
class _ResolvedDatasetDescriptor:
    """Resolved per-dataset generation state for one public runtime path."""

    dataset_index: int
    dataset_root: KeyedRng
    effective_config: GeneratorConfig
    effective_plan: _FixedLayoutPlan
    effective_shift: ShiftRuntimeParams
    finalization_context: _FixedSchemaFinalizationContext | None


class InvalidStructuralLayoutError(ValueError):
    """Raised when a sampled layout violates always-on structural validity rules."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = str(reason)


_MIN_STRUCTURAL_PLAN_CANDIDATE_ATTEMPTS = 5
_STRATIFIED_LOOKAHEAD_MULTIPLIER = 8
_MIN_STRATIFIED_LOOKAHEAD = 32


def _noise_config_signature(config: GeneratorConfig) -> tuple[object, ...]:
    mixture_weights = (
        tuple(
            sorted(
                (str(component), float(weight))
                for component, weight in config.noise.mixture_weights.items()
            )
        )
        if config.noise.mixture_weights is not None
        else None
    )
    return (
        str(config.noise.family),
        float(config.noise.base_scale),
        float(config.noise.student_t_df),
        mixture_weights,
    )


def _raw_generation_cohort_key(
    descriptor: _ResolvedDatasetDescriptor,
) -> tuple[object, ...]:
    """Return the raw-generation contract key for one resolved dataset descriptor."""

    return (
        str(descriptor.effective_plan.layout_signature),
        str(descriptor.effective_plan.plan_signature or ""),
        str(descriptor.effective_config.dataset.task),
        int(descriptor.effective_plan.n_train),
        int(descriptor.effective_plan.n_test),
        *_noise_config_signature(descriptor.effective_config),
        float(descriptor.effective_shift.variance_sigma_multiplier),
    )


def _descriptor_row_count(descriptor: _ResolvedDatasetDescriptor) -> int:
    return int(descriptor.effective_plan.n_train) + int(descriptor.effective_plan.n_test)


def _stratified_descriptor_key(
    descriptor: _ResolvedDatasetDescriptor,
) -> tuple[int, int]:
    return (
        _descriptor_row_count(descriptor),
        int(descriptor.effective_plan.layout.n_features),
    )


def _accumulate_optional_runtime_metric(
    metrics: dict[str, float],
    key: str,
    value: float,
) -> None:
    metrics[key] = float(metrics.get(key, 0.0)) + float(value)


def _size_bucket_metric_key(prefix: str, size: int) -> str:
    if size <= 1:
        return f"{prefix}_bucket_1_count"
    if size <= 3:
        return f"{prefix}_bucket_2_3_count"
    if size <= 7:
        return f"{prefix}_bucket_4_7_count"
    if size <= 15:
        return f"{prefix}_bucket_8_15_count"
    return f"{prefix}_bucket_16_plus_count"


def _structural_validity_checks(config: GeneratorConfig) -> StructuralValidityConfig:
    return StructuralValidityConfig(
        min_target_indegree=int(config.filter.min_target_indegree),
        min_target_relevant_feature_count=int(config.filter.min_target_relevant_feature_count),
        min_target_relevant_feature_fraction=float(
            config.filter.min_target_relevant_feature_fraction
        ),
    )


def _plan_candidate_attempt_budget(config: GeneratorConfig) -> int:
    """Return the internal layout-plan sampling retry budget."""

    return max(_MIN_STRUCTURAL_PLAN_CANDIDATE_ATTEMPTS, int(config.filter.max_attempts))


def _validate_sampled_layout_structure(config: GeneratorConfig, *, layout: LayoutPlan) -> None:
    structural_result = evaluate_layout_structural_validity(
        layout,
        checks=_structural_validity_checks(config),
    )
    if not structural_result.valid:
        raise InvalidStructuralLayoutError(str(structural_result.reason))


def _normalized_keyed_replay_root_path(
    value: object,
    *,
    field_name: str,
) -> list[str | int]:
    """Normalize one keyed-replay root path from emitted bundle metadata."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list.")
    normalized: list[str | int] = []
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, (int, str)):
            raise ValueError(
                f"{field_name}[{index}] must be an int or string path component, got {component!r}."
            )
        normalized.append(int(component) if isinstance(component, int) else str(component))
    return normalized


def _candidate_attempt_from_layout_root_path(layout_root_path: list[str | int]) -> int:
    """Return the fixed-layout candidate attempt encoded in one replay root path."""

    for index in range(len(layout_root_path) - 2):
        if (
            layout_root_path[index] == "plan_candidate"
            and isinstance(layout_root_path[index + 1], int)
            and layout_root_path[index + 2] == "layout"
        ):
            return int(layout_root_path[index + 1])
    return 0


def _steering_candidate_root_paths(
    *,
    dataset_index: int,
    candidate_attempt: int,
) -> tuple[list[str | int], list[str | int]]:
    """Return replay roots for one steering-resampled plan candidate."""

    if int(candidate_attempt) <= 0:
        return (
            ["dataset", int(dataset_index), "steering", "layout"],
            ["dataset", int(dataset_index), "steering", "execution_plan"],
        )
    return (
        ["dataset", int(dataset_index), "steering", "candidate", int(candidate_attempt), "layout"],
        [
            "dataset",
            int(dataset_index),
            "steering",
            "candidate",
            int(candidate_attempt),
            "execution_plan",
        ],
    )


def _replay_emitted_fixed_layout_plan(
    config: GeneratorConfig,
    bundle: DatasetBundle,
) -> _FixedLayoutPlan:
    """Replay one emitted fixed-layout plan from bundle metadata and keyed roots."""

    metadata = bundle.metadata
    run_seed = metadata.get("seed")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise ValueError("metadata.seed must be an integer to replay a fixed-layout plan.")
    keyed_replay = metadata.get("keyed_replay")
    if not isinstance(keyed_replay, dict):
        raise ValueError("metadata.keyed_replay must be a mapping to replay a fixed-layout plan.")

    layout_root_path = _normalized_keyed_replay_root_path(
        keyed_replay.get("layout_root_path"),
        field_name="metadata.keyed_replay.layout_root_path",
    )
    execution_plan_root_path = _normalized_keyed_replay_root_path(
        keyed_replay.get("execution_plan_root_path"),
        field_name="metadata.keyed_replay.execution_plan_root_path",
    )
    steering_layout_root_path = keyed_replay.get("steering_layout_root_path")
    normalized_steering_layout_root_path = (
        None
        if steering_layout_root_path is None
        else _normalized_keyed_replay_root_path(
            steering_layout_root_path,
            field_name="metadata.keyed_replay.steering_layout_root_path",
        )
    )
    steering_execution_plan_root_path = keyed_replay.get("steering_execution_plan_root_path")
    normalized_steering_execution_plan_root_path = (
        None
        if steering_execution_plan_root_path is None
        else _normalized_keyed_replay_root_path(
            steering_execution_plan_root_path,
            field_name="metadata.keyed_replay.steering_execution_plan_root_path",
        )
    )

    config_payload = metadata.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("metadata.config must be a mapping to replay a fixed-layout plan.")
    effective_config = GeneratorConfig.from_dict(config_payload)
    effective_shift = resolve_shift_runtime_params(effective_config)
    stress_profile_name = metadata.get("layout_stress_profile_name")
    if stress_profile_name is not None and (
        not isinstance(stress_profile_name, str) or not stress_profile_name
    ):
        raise ValueError("metadata.layout_stress_profile_name must be a non-empty string when set.")

    run_root = KeyedRng(int(run_seed))
    layout = _sample_layout(
        effective_config,
        run_root.keyed(*layout_root_path),
        "cpu",
        stress_profile_name=None if stress_profile_name is None else str(stress_profile_name),
    )
    if normalized_steering_layout_root_path is not None:
        layout = _resample_layout_graph(
            layout,
            config=effective_config,
            keyed_rng=run_root.keyed(*normalized_steering_layout_root_path),
            edge_logit_bias=float(effective_shift.edge_logit_bias_shift),
            stress_profile_name=None if stress_profile_name is None else str(stress_profile_name),
        )

    effective_execution_plan_root_path = (
        execution_plan_root_path
        if normalized_steering_execution_plan_root_path is None
        else normalized_steering_execution_plan_root_path
    )
    execution_plan = build_fixed_layout_execution_plan(
        effective_config,
        layout,
        plan_seed=run_root.keyed(*effective_execution_plan_root_path).child_seed(),
        mechanism_logit_tilt=float(effective_shift.mechanism_logit_tilt),
        stress_profile_name=None if stress_profile_name is None else str(stress_profile_name),
    )
    prepared_execution_context = _prepare_fixed_layout_execution_context(layout, execution_plan)
    intervention_plan = resolve_fixed_layout_intervention_plan(effective_config, layout)

    requested_device = metadata.get("requested_device")
    if not isinstance(requested_device, str) or not requested_device:
        raise ValueError("metadata.requested_device must be a non-empty string.")
    resolved_device = metadata.get("resolved_device")
    if not isinstance(resolved_device, str) or not resolved_device:
        raise ValueError("metadata.resolved_device must be a non-empty string.")
    layout_plan_seed = metadata.get("layout_plan_seed")
    if isinstance(layout_plan_seed, bool) or not isinstance(layout_plan_seed, int):
        raise ValueError(
            "metadata.layout_plan_seed must be an integer to replay a fixed-layout plan."
        )

    plan = _FixedLayoutPlan(
        layout=layout,
        requested_device=str(requested_device),
        resolved_device=str(resolved_device),
        plan_seed=int(layout_plan_seed),
        n_train=int(effective_config.dataset.n_train),
        n_test=int(effective_config.dataset.n_test),
        layout_signature=_layout_signature(layout),
        candidate_attempt=_candidate_attempt_from_layout_root_path(layout_root_path),
        execution_plan=execution_plan,
        plan_signature=fixed_layout_plan_signature(execution_plan),
        layout_root_path=list(layout_root_path),
        execution_plan_root_path=list(execution_plan_root_path),
        steering_layout_root_path=(
            None
            if normalized_steering_layout_root_path is None
            else list(normalized_steering_layout_root_path)
        ),
        steering_execution_plan_root_path=(
            None
            if normalized_steering_execution_plan_root_path is None
            else list(normalized_steering_execution_plan_root_path)
        ),
        stress_profile_name=None if stress_profile_name is None else str(stress_profile_name),
        intervention_plan=intervention_plan,
        prepared_execution_context=prepared_execution_context,
    )

    emitted_layout_signature = metadata.get("layout_signature")
    if isinstance(emitted_layout_signature, str) and emitted_layout_signature:
        if str(plan.layout_signature) != str(emitted_layout_signature):
            raise ValueError("Replayed fixed-layout plan does not match metadata.layout_signature.")
    emitted_plan_signature = metadata.get("layout_plan_signature")
    if isinstance(emitted_plan_signature, str) and emitted_plan_signature:
        if str(plan.plan_signature) != str(emitted_plan_signature):
            raise ValueError(
                "Replayed fixed-layout plan does not match metadata.layout_plan_signature."
            )
    return plan


def _sample_fixed_layout_candidate(
    config: GeneratorConfig,
    *,
    keyed_rng: KeyedRng,
    rows_seed: int,
    requested_device: str,
    resolved_device: str,
    stress_profile_name: str | None = None,
) -> _FixedLayoutPlan:
    """Sample one fixed-layout plan candidate without replay validation."""

    return _sample_fixed_layout_once(
        config,
        keyed_rng=keyed_rng,
        rows_seed=rows_seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
        stress_profile_name=stress_profile_name,
    )


def _group_noise_runtime_chunk(
    config: GeneratorConfig,
    *,
    dataset_roots: list[KeyedRng],
    attempts: list[int] | None = None,
) -> list[_NoiseRuntimeGroup]:
    return _group_noise_runtime_chunk_impl(
        config,
        dataset_roots=dataset_roots,
        attempts=attempts,
        resolve_noise_runtime_selection=_resolve_noise_runtime_selection,
    )


def _generate_grouped_raw_batches(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    execution_plan: FixedLayoutExecutionPlan,
    intervention_plan,
    grouped_noise_runtime: list[_NoiseRuntimeGroup],
    requested_device: str,
    resolved_device: str,
    noise_sigma_multiplier: float,
) -> list[_GroupedRawBatch]:
    return _generate_grouped_raw_batches_impl(
        config,
        layout,
        execution_plan=execution_plan,
        intervention_plan=intervention_plan,
        grouped_noise_runtime=grouped_noise_runtime,
        resolved_device=resolved_device,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_sampling_spec=_noise_sampling_spec,
        generate_graph_batch=generate_fixed_layout_graph_batch,
    )


def prepare_canonical_fixed_layout_run(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    precompute_classification_attempt_plan: bool = True,
) -> CanonicalFixedLayoutRun:
    """Prepare one internal fixed-layout run context for public generation APIs."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")

    realized_config, run_seed, requested_device, resolved_device, stress_profile_name = (
        realize_generation_config_for_run(
            config,
            seed=seed,
            device=device,
        )
    )
    run_root = KeyedRng(run_seed)
    rows_seed = run_root.child_seed("rows")
    attempts = _plan_candidate_attempt_budget(realized_config)
    last_error = "unknown"
    for attempt in range(attempts):
        candidate_root = run_root.keyed("plan_candidate", attempt)
        try:
            plan = _sample_fixed_layout_candidate(
                realized_config,
                keyed_rng=candidate_root,
                rows_seed=rows_seed,
                requested_device=requested_device,
                resolved_device=resolved_device,
                stress_profile_name=stress_profile_name,
            )
        except InvalidStructuralLayoutError as exc:
            last_error = str(exc.reason)
            continue
        effective_batch_size = _resolve_fixed_layout_batch_size(
            plan,
            num_datasets=max(1, int(num_datasets)),
            batch_size=batch_size,
            target_cells=_effective_fixed_layout_target_cells(realized_config),
            batch_size_cap=realized_config.runtime.fixed_layout_batch_size_cap,
        )
        classification_attempt_plan: tuple[int, ...] | None = None
        if (
            precompute_classification_attempt_plan
            and str(realized_config.dataset.task) == "classification"
        ):
            classification_attempt_plan = _fixed_layout_plan_classification_attempt_plan(
                realized_config,
                plan=plan,
                requested_device=requested_device,
                resolved_device=resolved_device,
                run_root=run_root,
                num_datasets=max(1, int(num_datasets)),
                batch_size=int(effective_batch_size),
            )
            if classification_attempt_plan is None:
                last_error = "invalid_class_split"
                continue
        return CanonicalFixedLayoutRun(
            config=realized_config,
            plan=plan,
            run_seed=int(run_seed),
            requested_device=str(requested_device),
            resolved_device=str(resolved_device),
            batch_size=int(effective_batch_size),
            classification_attempt_plan=classification_attempt_plan,
        )

    raise ValueError(
        "Failed to prepare a fixed-layout run after "
        f"{attempts} attempts. Last reason: {last_error}."
    )


def _sample_fixed_layout_once(
    config: GeneratorConfig,
    *,
    keyed_rng: KeyedRng,
    rows_seed: int,
    requested_device: str,
    resolved_device: str,
    stress_profile_name: str | None = None,
) -> _FixedLayoutPlan:
    """Sample one fixed-layout plan candidate without replay validation retries."""

    _validate_fixed_layout_rows_mode(config)
    layout_seed = keyed_rng.child_seed("layout")
    layout = _sample_layout(
        config,
        keyed_rng.keyed("layout"),
        "cpu",
        stress_profile_name=stress_profile_name,
    )
    _validate_sampled_layout_structure(config, layout=layout)
    n_train, n_test = _resolve_split_sizes(config, dataset_seed=rows_seed)
    _validate_class_split_for_layout(config, layout=layout, n_train=n_train, n_test=n_test)
    shift_params = resolve_shift_runtime_params(config)
    execution_plan_seed = keyed_rng.child_seed("execution_plan")
    path = tuple(keyed_rng.path)
    candidate_attempt = (
        int(path[-1])
        if len(path) >= 2 and path[-2] == "plan_candidate" and isinstance(path[-1], int)
        else 0
    )
    execution_plan = build_fixed_layout_execution_plan(
        config,
        layout,
        plan_seed=execution_plan_seed,
        mechanism_logit_tilt=float(shift_params.mechanism_logit_tilt),
        stress_profile_name=stress_profile_name,
    )
    prepared_execution_context = _prepare_fixed_layout_execution_context(layout, execution_plan)
    intervention_plan = resolve_fixed_layout_intervention_plan(config, layout)
    return _FixedLayoutPlan(
        layout=layout,
        requested_device=requested_device,
        resolved_device=resolved_device,
        plan_seed=int(layout_seed),
        n_train=int(n_train),
        n_test=int(n_test),
        layout_signature=_layout_signature(layout),
        candidate_attempt=candidate_attempt,
        execution_plan=execution_plan,
        plan_signature=fixed_layout_plan_signature(execution_plan),
        stress_profile_name=stress_profile_name,
        intervention_plan=intervention_plan,
        prepared_execution_context=prepared_execution_context,
    )


def _resolve_steered_plan_for_dataset(
    config: GeneratorConfig,
    *,
    base_plan: _FixedLayoutPlan,
    dataset_index: int,
    num_datasets: int,
    dataset_root: KeyedRng,
    candidate_attempt: int = 0,
) -> tuple[GeneratorConfig, _FixedLayoutPlan]:
    """Resolve one effective config and fixed-layout plan for a steered dataset ordinal."""

    resolution = resolve_steering(
        config,
        dataset_index=dataset_index,
        run_num_datasets=num_datasets,
    )
    effective_config = resolution.config
    base_shift = resolve_shift_runtime_params(config)
    effective_shift = resolve_shift_runtime_params(effective_config)
    if math.isclose(
        float(effective_shift.graph_scale),
        float(base_shift.graph_scale),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) and math.isclose(
        float(effective_shift.mechanism_logit_tilt),
        float(base_shift.mechanism_logit_tilt),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return effective_config, base_plan

    steering_layout_root_path, steering_execution_plan_root_path = _steering_candidate_root_paths(
        dataset_index=dataset_index,
        candidate_attempt=candidate_attempt,
    )
    layout_root = dataset_root.keyed(*steering_layout_root_path[2:])
    layout = _resample_layout_graph(
        base_plan.layout,
        config=effective_config,
        keyed_rng=layout_root,
        edge_logit_bias=float(effective_shift.edge_logit_bias_shift),
        stress_profile_name=base_plan.stress_profile_name,
    )
    execution_plan_root = dataset_root.keyed(*steering_execution_plan_root_path[2:])
    execution_plan_seed = execution_plan_root.child_seed()
    execution_plan = build_fixed_layout_execution_plan(
        effective_config,
        layout,
        plan_seed=execution_plan_seed,
        mechanism_logit_tilt=float(effective_shift.mechanism_logit_tilt),
        stress_profile_name=base_plan.stress_profile_name,
    )
    prepared_execution_context = _prepare_fixed_layout_execution_context(layout, execution_plan)
    plan_seed = layout_root.child_seed()
    intervention_plan = resolve_fixed_layout_intervention_plan(effective_config, layout)
    return effective_config, _FixedLayoutPlan(
        layout=layout,
        requested_device=str(base_plan.requested_device),
        resolved_device=str(base_plan.resolved_device),
        plan_seed=int(plan_seed),
        n_train=int(base_plan.n_train),
        n_test=int(base_plan.n_test),
        layout_signature=_layout_signature(layout),
        candidate_attempt=int(base_plan.candidate_attempt),
        execution_plan=execution_plan,
        plan_signature=fixed_layout_plan_signature(execution_plan),
        layout_root_path=(
            None if base_plan.layout_root_path is None else list(base_plan.layout_root_path)
        ),
        execution_plan_root_path=(
            None
            if base_plan.execution_plan_root_path is None
            else list(base_plan.execution_plan_root_path)
        ),
        steering_layout_root_path=list(steering_layout_root_path),
        steering_execution_plan_root_path=list(steering_execution_plan_root_path),
        stress_profile_name=base_plan.stress_profile_name,
        intervention_plan=intervention_plan,
        prepared_execution_context=prepared_execution_context,
    )


def _resolve_steered_dataset_descriptor(
    config: GeneratorConfig,
    *,
    base_plan: _FixedLayoutPlan,
    dataset_index: int,
    num_datasets: int,
    dataset_root: KeyedRng,
) -> _ResolvedDatasetDescriptor:
    """Resolve one per-dataset steering descriptor for batched fixed-layout execution."""

    attempts = _plan_candidate_attempt_budget(config)
    last_error = "unknown"
    effective_config: GeneratorConfig | None = None
    effective_plan: _FixedLayoutPlan | None = None
    resolved = False
    for candidate_attempt in range(attempts):
        effective_config, effective_plan = _resolve_steered_plan_for_dataset(
            config,
            base_plan=base_plan,
            dataset_index=dataset_index,
            num_datasets=num_datasets,
            dataset_root=dataset_root,
            candidate_attempt=candidate_attempt,
        )
        try:
            _validate_sampled_layout_structure(effective_config, layout=effective_plan.layout)
        except InvalidStructuralLayoutError as exc:
            last_error = str(exc.reason)
            continue
        resolved = True
        break
    if not resolved or effective_config is None or effective_plan is None:
        raise ValueError(
            "Failed to resolve a structurally valid steered dataset descriptor after "
            f"{attempts} attempts. Last reason: {last_error}."
        )
    effective_shift = resolve_shift_runtime_params(effective_config)
    finalization_context = _build_fixed_schema_finalization_context(
        effective_config,
        effective_plan.layout,
        n_train=int(effective_plan.n_train),
        n_test=int(effective_plan.n_test),
        shift_params=effective_shift,
    )
    return _ResolvedDatasetDescriptor(
        dataset_index=int(dataset_index),
        dataset_root=dataset_root,
        effective_config=effective_config,
        effective_plan=effective_plan,
        effective_shift=effective_shift,
        finalization_context=finalization_context,
    )


def _resolve_heterogeneous_dataset_descriptor(
    config: GeneratorConfig,
    *,
    requested_device: str,
    resolved_device: str,
    rows_seed: int,
    plan_candidate_attempt: int,
    dataset_index: int,
    num_datasets: int,
    dataset_root: KeyedRng,
    stress_profile_name: str | None = None,
) -> _ResolvedDatasetDescriptor:
    """Resolve one fully heterogeneous per-dataset plan and finalization context."""

    effective_config = resolve_steering(
        config,
        dataset_index=dataset_index,
        run_num_datasets=num_datasets,
    ).config
    candidate_attempts = _plan_candidate_attempt_budget(effective_config)
    candidate_start = max(0, int(plan_candidate_attempt))
    last_error = "unknown"
    effective_plan: _FixedLayoutPlan | None = None
    resolved_candidate_attempt = int(plan_candidate_attempt)
    for candidate_attempt in range(candidate_start, candidate_attempts):
        plan_root = dataset_root.keyed("plan_candidate", int(candidate_attempt))
        try:
            effective_plan = _sample_fixed_layout_once(
                effective_config,
                keyed_rng=plan_root,
                rows_seed=rows_seed,
                requested_device=requested_device,
                resolved_device=resolved_device,
                stress_profile_name=stress_profile_name,
            )
        except InvalidStructuralLayoutError as exc:
            last_error = str(exc.reason)
            continue
        resolved_candidate_attempt = int(candidate_attempt)
        break
    if effective_plan is None:
        raise ValueError(
            "Failed to resolve a structurally valid heterogeneous dataset descriptor after "
            f"{candidate_attempts} attempts. Last reason: {last_error}."
        )
    effective_plan.layout_root_path = [
        "dataset",
        int(dataset_index),
        "plan_candidate",
        int(resolved_candidate_attempt),
        "layout",
    ]
    effective_plan.execution_plan_root_path = [
        "dataset",
        int(dataset_index),
        "plan_candidate",
        int(resolved_candidate_attempt),
        "execution_plan",
    ]
    effective_shift = resolve_shift_runtime_params(effective_config)
    return _ResolvedDatasetDescriptor(
        dataset_index=int(dataset_index),
        dataset_root=dataset_root,
        effective_config=effective_config,
        effective_plan=effective_plan,
        effective_shift=effective_shift,
        finalization_context=None,
    )


def _generate_heterogeneous_bundle_with_plan_candidates(
    config: GeneratorConfig,
    *,
    requested_device: str,
    resolved_device: str,
    rows_seed: int,
    dataset_index: int,
    num_datasets: int,
    dataset_root: KeyedRng,
    initial_descriptor: _ResolvedDatasetDescriptor | None = None,
    initial_start_attempt: int = 0,
    start_candidate_attempt: int = 0,
    on_raw_batch_metrics: Callable[[dict[str, float]], None] | None = None,
) -> tuple[_ResolvedDatasetDescriptor, DatasetBundle]:
    """Generate one heterogeneous bundle, resampling plan candidates when needed."""

    last_error: Exception | None = None
    if initial_descriptor is not None:
        try:
            return initial_descriptor, _generate_fixed_layout_bundle_with_retries(
                initial_descriptor.effective_config,
                plan=initial_descriptor.effective_plan,
                dataset_root=initial_descriptor.dataset_root,
                requested_device=requested_device,
                resolved_device=resolved_device,
                preserve_feature_schema=False,
                start_attempt=initial_start_attempt,
                finalization_context=initial_descriptor.finalization_context,
                on_raw_batch_metrics=on_raw_batch_metrics,
            )
        except ValueError as exc:
            last_error = exc

    candidate_attempts = _plan_candidate_attempt_budget(config)
    candidate_start = max(0, int(start_candidate_attempt))
    for candidate_attempt in range(candidate_start, candidate_attempts):
        descriptor_start = time.perf_counter()
        descriptor_start_cpu = time.process_time()
        descriptor = _resolve_heterogeneous_dataset_descriptor(
            config,
            requested_device=requested_device,
            resolved_device=resolved_device,
            rows_seed=rows_seed,
            plan_candidate_attempt=candidate_attempt,
            dataset_index=dataset_index,
            num_datasets=num_datasets,
            dataset_root=dataset_root,
        )
        if on_raw_batch_metrics is not None:
            on_raw_batch_metrics(
                {
                    "heterogeneous_descriptor_resolution_elapsed_seconds": (
                        time.perf_counter() - descriptor_start
                    ),
                    "heterogeneous_descriptor_resolution_cpu_time_seconds": (
                        time.process_time() - descriptor_start_cpu
                    ),
                }
            )
        try:
            bundle = _generate_fixed_layout_bundle_with_retries(
                descriptor.effective_config,
                plan=descriptor.effective_plan,
                dataset_root=descriptor.dataset_root,
                requested_device=requested_device,
                resolved_device=resolved_device,
                preserve_feature_schema=False,
                start_attempt=0,
                finalization_context=descriptor.finalization_context,
                on_raw_batch_metrics=on_raw_batch_metrics,
            )
            return descriptor, bundle
        except ValueError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise ValueError("Failed to generate a heterogeneous dataset: no plan candidates were tried.")


def _next_plan_candidate_attempt_for_descriptor(
    descriptor: _ResolvedDatasetDescriptor,
) -> int:
    """Return the next plan-candidate attempt after one resolved descriptor."""

    return int(descriptor.effective_plan.candidate_attempt) + 1


def _generate_batch_with_dynamic_steering_iter(
    config: GeneratorConfig,
    *,
    base_plan: _FixedLayoutPlan,
    num_datasets: int,
    seed: int | None = None,
    batch_size: int | None = None,
    classification_attempt_plan: tuple[int, ...] | None = None,
    on_raw_batch_metrics: Callable[[dict[str, float]], None] | None = None,
) -> Iterator[DatasetBundle]:
    """Yield steering-enabled datasets while preserving grouped fixed-layout batching."""

    requested_device = str(base_plan.requested_device)
    validated_resolved_device = str(base_plan.resolved_device)
    run_seed = _resolve_run_seed(config, seed)
    run_root = KeyedRng(run_seed)
    dtype = _torch_dtype(config)
    expected_schema: tuple[int, tuple[str, ...], tuple[int, ...], int] | None = None
    effective_batch_size = _resolve_fixed_layout_batch_size(
        base_plan,
        num_datasets=num_datasets,
        batch_size=batch_size,
        target_cells=_effective_fixed_layout_target_cells(config),
        batch_size_cap=config.runtime.fixed_layout_batch_size_cap,
    )

    dataset_index = 0
    while dataset_index < num_datasets:
        chunk_size = min(effective_batch_size, num_datasets - dataset_index)
        descriptors = [
            _resolve_steered_dataset_descriptor(
                config,
                base_plan=base_plan,
                dataset_index=dataset_index + offset,
                num_datasets=num_datasets,
                dataset_root=run_root.keyed("dataset", dataset_index + offset),
            )
            for offset in range(chunk_size)
        ]
        chunk_attempts = (
            list(classification_attempt_plan[dataset_index : dataset_index + chunk_size])
            if classification_attempt_plan is not None
            else [0] * chunk_size
        )
        raw_batch_by_offset: list[DatasetBundle | None] = [None] * chunk_size
        recoverable_failure_by_offset: list[RecoverableGenerationFailure | None] = [
            None
        ] * chunk_size
        finalized_offsets: set[int] = set()
        zero_attempt_offsets = [
            offset for offset, attempt in enumerate(chunk_attempts) if int(attempt) == 0
        ]
        retry_offsets = [
            offset for offset, attempt in enumerate(chunk_attempts) if int(attempt) != 0
        ]
        retry_offset_set = set(retry_offsets)
        cohort_offsets_by_key: dict[tuple[object, ...], list[int]] = {}
        cohort_order: list[tuple[object, ...]] = []
        for offset in zero_attempt_offsets:
            cohort_key = _raw_generation_cohort_key(descriptors[offset])
            if cohort_key not in cohort_offsets_by_key:
                cohort_offsets_by_key[cohort_key] = []
                cohort_order.append(cohort_key)
            cohort_offsets_by_key[cohort_key].append(int(offset))

        for cohort_key in cohort_order:
            cohort_offsets = cohort_offsets_by_key[cohort_key]
            cohort_descriptors = [descriptors[offset] for offset in cohort_offsets]
            representative = cohort_descriptors[0]
            grouped_noise_runtime = _group_noise_runtime_chunk(
                representative.effective_config,
                dataset_roots=[entry.dataset_root for entry in cohort_descriptors],
                attempts=[0] * len(cohort_descriptors),
            )
            try:
                grouped_raw_batches = _generate_grouped_raw_batches(
                    representative.effective_config,
                    representative.effective_plan.layout,
                    execution_plan=representative.effective_plan.execution_plan,
                    intervention_plan=representative.effective_plan.intervention_plan,
                    grouped_noise_runtime=grouped_noise_runtime,
                    requested_device=requested_device,
                    resolved_device=validated_resolved_device,
                    noise_sigma_multiplier=float(
                        representative.effective_shift.variance_sigma_multiplier
                    ),
                )
            except ValueError as exc:
                recoverable_failure = classify_recoverable_generation_failure(
                    exc,
                    degeneracy_retry_scope=RECOVERABLE_RETRY_SCOPE_NEXT_PLAN_CANDIDATE,
                )
                if recoverable_failure is None:
                    raise
                for offset in cohort_offsets:
                    recoverable_failure_by_offset[offset] = recoverable_failure
                continue
            for grouped_batch in grouped_raw_batches:
                if on_raw_batch_metrics is not None:
                    on_raw_batch_metrics(dict(getattr(grouped_batch, "runtime_metrics", {})))
                group_dataset_offsets = [
                    cohort_offsets[int(chunk_offset)]
                    for chunk_offset in grouped_batch.chunk_offsets
                ]
                group_descriptors = [descriptors[offset] for offset in group_dataset_offsets]
                group_dataset_roots = [entry.dataset_root for entry in group_descriptors]
                resolved_split_indices: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
                if str(representative.effective_config.dataset.task) == "classification":
                    resolved_split_indices = []
                    for local_index, entry in enumerate(group_descriptors):
                        attempt_root = entry.dataset_root.keyed("attempt", grouped_batch.attempt)
                        try:
                            split_indices = _resolve_split_indices(
                                grouped_batch.y_batch[local_index],
                                task=entry.effective_config.dataset.task,
                                n_train=int(entry.effective_plan.n_train),
                                keyed_rng=attempt_root.keyed("split"),
                            )
                        except InfeasibleStratifiedSplitError:
                            split_indices = None
                        resolved_split_indices.append(split_indices)
                representative_context = representative.finalization_context
                raw_contexts_by_batch = [entry.finalization_context for entry in group_descriptors]
                if representative_context is None or any(
                    context is None for context in raw_contexts_by_batch
                ):
                    raise RuntimeError(
                        "Fixed-layout steering descriptors must include finalization contexts."
                    )
                contexts_by_batch = [
                    context for context in raw_contexts_by_batch if context is not None
                ]
                finalized_group = _finalize_generated_chunk_preserve_schema(
                    representative.effective_config,
                    representative.effective_plan.layout,
                    context=representative_context,
                    contexts_by_batch=contexts_by_batch,
                    configs_by_batch=[entry.effective_config for entry in group_descriptors],
                    dataset_roots=group_dataset_roots,
                    attempt=grouped_batch.attempt,
                    attempts_used=grouped_batch.attempt + 1,
                    device=grouped_batch.effective_resolved_device,
                    n_train=int(representative.effective_plan.n_train),
                    n_test=int(representative.effective_plan.n_test),
                    requested_device=requested_device,
                    resolved_device=grouped_batch.effective_resolved_device,
                    device_fallback_reason=grouped_batch.device_fallback_reason,
                    x=grouped_batch.x_batch,
                    y=grouped_batch.y_batch,
                    aux_meta_batch=grouped_batch.aux_meta_batch,
                    noise_runtime_selection=grouped_batch.selection,
                    dtype=dtype,
                    resolved_split_indices=resolved_split_indices,
                )
                for local_index, offset in enumerate(group_dataset_offsets):
                    raw_batch_by_offset[offset] = finalized_group[local_index]
                    finalized_offsets.add(int(offset))

        for offset, descriptor in enumerate(descriptors):
            if offset in retry_offset_set:
                bundle = _generate_fixed_layout_bundle_with_retries(
                    descriptor.effective_config,
                    plan=descriptor.effective_plan,
                    dataset_root=descriptor.dataset_root,
                    requested_device=requested_device,
                    resolved_device=validated_resolved_device,
                    preserve_feature_schema=True,
                    start_attempt=chunk_attempts[offset],
                    finalization_context=descriptor.finalization_context,
                    on_raw_batch_metrics=on_raw_batch_metrics,
                )
            else:
                if offset not in finalized_offsets:
                    raise RuntimeError(
                        "Missing grouped raw batch entry for steering-enabled fixed-layout chunk offset."
                    )
                candidate_bundle = raw_batch_by_offset[offset]
                if candidate_bundle is None:
                    bundle = _generate_fixed_layout_bundle_with_retries(
                        descriptor.effective_config,
                        plan=descriptor.effective_plan,
                        dataset_root=descriptor.dataset_root,
                        requested_device=requested_device,
                        resolved_device=validated_resolved_device,
                        preserve_feature_schema=True,
                        start_attempt=int(chunk_attempts[offset]),
                        finalization_context=descriptor.finalization_context,
                        on_raw_batch_metrics=on_raw_batch_metrics,
                    )
                else:
                    bundle = candidate_bundle
            _annotate_fixed_layout_metadata(bundle, plan=descriptor.effective_plan)
            schema = _extract_emitted_schema_signature(bundle)
            if expected_schema is None:
                expected_schema = schema
            elif schema != expected_schema:
                raise ValueError(
                    "Fixed-layout schema mismatch: emitted dataset does not match "
                    "the first fixed-layout bundle schema."
                )
            yield bundle
        dataset_index += chunk_size


def _raw_classification_labels_support_split(
    y: torch.Tensor,
    *,
    dataset_root: KeyedRng,
    attempt: int,
    n_train: int,
) -> bool:
    """Return whether one raw classification label vector can satisfy split constraints."""

    labels = y.to(device="cpu", dtype=torch.int64)
    split_generator = dataset_root.keyed("attempt", attempt, "split").torch_rng(device="cpu")
    try:
        train_idx_cpu, test_idx_cpu = _stratified_split_indices(
            labels,
            int(n_train),
            split_generator,
            "cpu",
        )
    except InfeasibleStratifiedSplitError:
        return False
    return _classification_split_valid(labels[train_idx_cpu], labels[test_idx_cpu])


def _resolve_heterogeneous_batch_size(
    config: GeneratorConfig,
    *,
    rows_seed: int,
    num_datasets: int,
    batch_size: int | None,
) -> int:
    """Resolve one conservative chunk size for heterogeneous public generation."""

    effective_cap = (
        None
        if config.runtime.fixed_layout_batch_size_cap is None
        else max(1, int(config.runtime.fixed_layout_batch_size_cap))
    )
    if batch_size is not None:
        resolved = max(1, min(int(batch_size), int(num_datasets)))
        return resolved if effective_cap is None else min(resolved, effective_cap)

    n_train, n_test = _resolve_split_sizes(config, dataset_seed=rows_seed)
    per_dataset_cells = max(
        1,
        int(n_train + n_test) * max(1, int(config.dataset.n_features_max)),
    )
    auto_batch = max(1, int(_effective_fixed_layout_target_cells(config)) // per_dataset_cells)
    resolved = max(1, min(int(num_datasets), int(auto_batch)))
    return resolved if effective_cap is None else min(resolved, effective_cap)


def _resolve_stratified_microbatch_size(
    config: GeneratorConfig,
    *,
    n_rows: int,
    n_features: int,
    num_datasets: int,
    batch_size: int | None,
) -> int:
    """Resolve one exact-stratum microbatch size from the target-cell budget."""

    effective_cap = (
        None
        if config.runtime.fixed_layout_batch_size_cap is None
        else max(1, int(config.runtime.fixed_layout_batch_size_cap))
    )
    if batch_size is not None:
        resolved = max(1, min(int(batch_size), int(num_datasets)))
        return resolved if effective_cap is None else min(resolved, effective_cap)

    per_dataset_cells = max(1, int(n_rows) * max(1, int(n_features)))
    auto_batch = max(1, int(_effective_fixed_layout_target_cells(config)) // per_dataset_cells)
    resolved = max(1, min(int(num_datasets), int(auto_batch)))
    return resolved if effective_cap is None else min(resolved, effective_cap)


def _resolve_stratified_lookahead_size(
    config: GeneratorConfig,
    *,
    rows_seed: int,
    num_datasets: int,
    batch_size: int | None,
) -> int:
    """Resolve the descriptor lookahead window for stratified scheduling."""

    base_batch_size = _resolve_heterogeneous_batch_size(
        config,
        rows_seed=rows_seed,
        num_datasets=num_datasets,
        batch_size=batch_size,
    )
    return max(
        1,
        min(
            int(num_datasets),
            max(_MIN_STRATIFIED_LOOKAHEAD, int(base_batch_size) * _STRATIFIED_LOOKAHEAD_MULTIPLIER),
        ),
    )


def _resolve_heterogeneous_descriptor_window(
    config: GeneratorConfig,
    *,
    requested_device: str,
    resolved_device: str,
    rows_seed: int,
    dataset_index: int,
    window_size: int,
    num_datasets: int,
    run_root: KeyedRng,
    stress_profile_name: str | None = None,
) -> tuple[list[_ResolvedDatasetDescriptor], dict[str, float]]:
    """Resolve one contiguous descriptor window for heterogeneous-style execution."""

    descriptor_start = time.perf_counter()
    descriptor_start_cpu = time.process_time()
    descriptors = [
        _resolve_heterogeneous_dataset_descriptor(
            config,
            requested_device=requested_device,
            resolved_device=resolved_device,
            rows_seed=rows_seed,
            plan_candidate_attempt=0,
            dataset_index=dataset_index + offset,
            num_datasets=num_datasets,
            dataset_root=run_root.keyed("dataset", dataset_index + offset),
            stress_profile_name=stress_profile_name,
        )
        for offset in range(window_size)
    ]
    return descriptors, {
        "heterogeneous_descriptor_resolution_elapsed_seconds": (
            time.perf_counter() - descriptor_start
        ),
        "heterogeneous_descriptor_resolution_cpu_time_seconds": (
            time.process_time() - descriptor_start_cpu
        ),
    }


def _execute_heterogeneous_descriptor_chunk(
    config: GeneratorConfig,
    *,
    descriptors: list[_ResolvedDatasetDescriptor],
    requested_device: str,
    resolved_device: str,
    rows_seed: int,
    num_datasets: int,
    dtype: torch.dtype,
    layout_mode: str,
    on_raw_batch_metrics: Callable[[dict[str, float]], None] | None = None,
) -> list[tuple[_ResolvedDatasetDescriptor, DatasetBundle]]:
    """Execute one already-resolved heterogeneous-style descriptor chunk."""

    chunk_size = len(descriptors)
    raw_batch_by_offset: list[DatasetBundle | None] = [None] * chunk_size
    recoverable_failure_by_offset: list[RecoverableGenerationFailure | None] = [None] * chunk_size
    cohort_offsets_by_key: dict[tuple[object, ...], list[int]] = {}
    cohort_order: list[tuple[object, ...]] = []
    for offset, descriptor in enumerate(descriptors):
        cohort_key = _raw_generation_cohort_key(descriptor)
        if cohort_key not in cohort_offsets_by_key:
            cohort_offsets_by_key[cohort_key] = []
            cohort_order.append(cohort_key)
        cohort_offsets_by_key[cohort_key].append(int(offset))

    for cohort_key in cohort_order:
        cohort_offsets = cohort_offsets_by_key[cohort_key]
        cohort_descriptors = [descriptors[offset] for offset in cohort_offsets]
        representative = cohort_descriptors[0]
        grouped_noise_runtime = _group_noise_runtime_chunk(
            representative.effective_config,
            dataset_roots=[entry.dataset_root for entry in cohort_descriptors],
            attempts=[0] * len(cohort_descriptors),
        )
        try:
            grouped_raw_batches = _generate_grouped_raw_batches(
                representative.effective_config,
                representative.effective_plan.layout,
                execution_plan=representative.effective_plan.execution_plan,
                intervention_plan=representative.effective_plan.intervention_plan,
                grouped_noise_runtime=grouped_noise_runtime,
                requested_device=requested_device,
                resolved_device=resolved_device,
                noise_sigma_multiplier=float(
                    representative.effective_shift.variance_sigma_multiplier
                ),
            )
        except ValueError as exc:
            recoverable_failure = classify_recoverable_generation_failure(exc)
            if recoverable_failure is None:
                raise
            for offset in cohort_offsets:
                recoverable_failure_by_offset[offset] = recoverable_failure
            continue
        for grouped_batch in grouped_raw_batches:
            group_dataset_offsets = [
                cohort_offsets[int(chunk_offset)] for chunk_offset in grouped_batch.chunk_offsets
            ]
            group_descriptors = [descriptors[offset] for offset in group_dataset_offsets]
            group_runtime_metrics = dict(getattr(grouped_batch, "runtime_metrics", {}))
            finalized_group, finalized_failures = _finalize_generated_chunk_variable_schema(
                representative.effective_plan.layout,
                configs_by_batch=[entry.effective_config for entry in group_descriptors],
                shift_params_by_batch=[entry.effective_shift for entry in group_descriptors],
                dataset_roots=[entry.dataset_root for entry in group_descriptors],
                attempt=grouped_batch.attempt,
                attempts_used=grouped_batch.attempt + 1,
                device=grouped_batch.effective_resolved_device,
                n_train=int(representative.effective_plan.n_train),
                n_test=int(representative.effective_plan.n_test),
                requested_device=requested_device,
                resolved_device=grouped_batch.effective_resolved_device,
                device_fallback_reason=grouped_batch.device_fallback_reason,
                x=grouped_batch.x_batch,
                y=grouped_batch.y_batch,
                aux_meta_batch=grouped_batch.aux_meta_batch,
                noise_runtime_selection=grouped_batch.selection,
                dtype=dtype,
                runtime_metrics_out=group_runtime_metrics,
            )
            if on_raw_batch_metrics is not None:
                on_raw_batch_metrics(group_runtime_metrics)
            for local_index, offset in enumerate(group_dataset_offsets):
                raw_batch_by_offset[offset] = finalized_group[local_index]
                recoverable_failure_by_offset[offset] = finalized_failures[local_index]

    completed: list[tuple[_ResolvedDatasetDescriptor, DatasetBundle]] = []
    for offset, descriptor in enumerate(descriptors):
        bundle = raw_batch_by_offset[offset]
        if bundle is None:
            recoverable_failure = recoverable_failure_by_offset[offset]
            if (
                recoverable_failure is not None
                and recoverable_failure.retry_scope == RECOVERABLE_RETRY_SCOPE_NEXT_PLAN_CANDIDATE
            ):
                descriptor, bundle = _generate_heterogeneous_bundle_with_plan_candidates(
                    config,
                    requested_device=requested_device,
                    resolved_device=resolved_device,
                    rows_seed=rows_seed,
                    dataset_index=int(descriptor.dataset_index),
                    num_datasets=num_datasets,
                    dataset_root=descriptor.dataset_root,
                    initial_descriptor=None,
                    initial_start_attempt=0,
                    start_candidate_attempt=_next_plan_candidate_attempt_for_descriptor(descriptor),
                    on_raw_batch_metrics=on_raw_batch_metrics,
                )
            else:
                descriptor, bundle = _generate_heterogeneous_bundle_with_plan_candidates(
                    config,
                    requested_device=requested_device,
                    resolved_device=resolved_device,
                    rows_seed=rows_seed,
                    dataset_index=int(descriptor.dataset_index),
                    num_datasets=num_datasets,
                    dataset_root=descriptor.dataset_root,
                    initial_descriptor=descriptor,
                    initial_start_attempt=1,
                    start_candidate_attempt=_next_plan_candidate_attempt_for_descriptor(descriptor),
                    on_raw_batch_metrics=on_raw_batch_metrics,
                )
            descriptors[offset] = descriptor
        _annotate_fixed_layout_metadata(
            bundle,
            plan=descriptor.effective_plan,
            layout_mode=str(layout_mode),
        )
        completed.append((descriptor, bundle))
    return completed


def _generate_batch_with_heterogeneous_layout_iter(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    on_raw_batch_metrics: Callable[[dict[str, float]], None] | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets from a fully heterogeneous per-dataset plan-sampling run."""

    realized_config, run_seed, requested_device, validated_resolved_device, stress_profile_name = (
        realize_generation_config_for_run(
            config,
            seed=seed,
            device=device,
            prefer_cpu_for_mps_auto=True,
        )
    )
    run_root = KeyedRng(run_seed)
    rows_seed = run_root.child_seed("rows")
    dtype = _torch_dtype(realized_config)
    effective_batch_size = _resolve_heterogeneous_batch_size(
        realized_config,
        rows_seed=rows_seed,
        num_datasets=num_datasets,
        batch_size=batch_size,
    )

    dataset_index = 0
    while dataset_index < num_datasets:
        chunk_size = min(effective_batch_size, num_datasets - dataset_index)
        descriptors, descriptor_metrics = _resolve_heterogeneous_descriptor_window(
            realized_config,
            requested_device=requested_device,
            resolved_device=validated_resolved_device,
            rows_seed=rows_seed,
            dataset_index=dataset_index,
            window_size=chunk_size,
            num_datasets=num_datasets,
            run_root=run_root,
            stress_profile_name=stress_profile_name,
        )
        if on_raw_batch_metrics is not None:
            on_raw_batch_metrics(descriptor_metrics)
        for _descriptor, bundle in _execute_heterogeneous_descriptor_chunk(
            realized_config,
            descriptors=descriptors,
            requested_device=requested_device,
            resolved_device=validated_resolved_device,
            rows_seed=rows_seed,
            num_datasets=num_datasets,
            dtype=dtype,
            layout_mode="heterogeneous",
            on_raw_batch_metrics=on_raw_batch_metrics,
        ):
            yield bundle
        dataset_index += chunk_size


def _generate_batch_with_stratified_layout_iter(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    on_raw_batch_metrics: Callable[[dict[str, float]], None] | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets from a stratified heterogeneous scheduler."""

    realized_config, run_seed, requested_device, validated_resolved_device, stress_profile_name = (
        realize_generation_config_for_run(
            config,
            seed=seed,
            device=device,
            prefer_cpu_for_mps_auto=True,
        )
    )
    run_root = KeyedRng(run_seed)
    rows_seed = run_root.child_seed("rows")
    dtype = _torch_dtype(realized_config)
    lookahead_size = _resolve_stratified_lookahead_size(
        realized_config,
        rows_seed=rows_seed,
        num_datasets=num_datasets,
        batch_size=batch_size,
    )

    dataset_index = 0
    while dataset_index < num_datasets:
        window_size = min(int(lookahead_size), num_datasets - dataset_index)
        descriptors, descriptor_metrics = _resolve_heterogeneous_descriptor_window(
            realized_config,
            requested_device=requested_device,
            resolved_device=validated_resolved_device,
            rows_seed=rows_seed,
            dataset_index=dataset_index,
            window_size=window_size,
            num_datasets=num_datasets,
            run_root=run_root,
            stress_profile_name=stress_profile_name,
        )
        if on_raw_batch_metrics is not None:
            on_raw_batch_metrics(descriptor_metrics)

        scheduler_start = time.perf_counter()
        scheduler_start_cpu = time.process_time()
        stratum_offsets_by_key: dict[tuple[int, int], list[int]] = {}
        stratum_first_index_by_key: dict[tuple[int, int], int] = {}
        for offset, descriptor in enumerate(descriptors):
            stratum_key = _stratified_descriptor_key(descriptor)
            stratum_offsets_by_key.setdefault(stratum_key, []).append(int(offset))
            stratum_first_index_by_key.setdefault(stratum_key, int(descriptor.dataset_index))

        ordered_strata = sorted(
            stratum_offsets_by_key,
            key=lambda key: (
                -len(stratum_offsets_by_key[key]),
                int(stratum_first_index_by_key[key]),
            ),
        )

        window_metrics: dict[str, float] = {
            "stratified_descriptor_window_fill_ratio_sum": (
                float(window_size) / float(max(1, int(lookahead_size)))
            ),
            "stratified_descriptor_window_count": 1.0,
        }
        bundles_by_dataset_index: dict[int, DatasetBundle] = {}

        for stratum_key in ordered_strata:
            stratum_offsets = stratum_offsets_by_key[stratum_key]
            stratum_size = len(stratum_offsets)
            _accumulate_optional_runtime_metric(
                window_metrics, "stratified_stratum_size_sum", stratum_size
            )
            _accumulate_optional_runtime_metric(window_metrics, "stratified_stratum_count", 1.0)
            _accumulate_optional_runtime_metric(
                window_metrics,
                _size_bucket_metric_key("stratified_stratum_size", stratum_size),
                1.0,
            )
            n_rows, n_features = stratum_key
            microbatch_size = _resolve_stratified_microbatch_size(
                realized_config,
                n_rows=int(n_rows),
                n_features=int(n_features),
                num_datasets=stratum_size,
                batch_size=batch_size,
            )
            for start in range(0, stratum_size, max(1, int(microbatch_size))):
                micro_offsets = stratum_offsets[start : start + max(1, int(microbatch_size))]
                micro_size = len(micro_offsets)
                _accumulate_optional_runtime_metric(
                    window_metrics,
                    "stratified_executed_microbatch_size_sum",
                    micro_size,
                )
                _accumulate_optional_runtime_metric(
                    window_metrics,
                    "stratified_executed_microbatch_count",
                    1.0,
                )
                _accumulate_optional_runtime_metric(
                    window_metrics,
                    _size_bucket_metric_key("stratified_microbatch_size", micro_size),
                    1.0,
                )
                if micro_size == 1:
                    _accumulate_optional_runtime_metric(
                        window_metrics,
                        "stratified_scalar_fallback_dataset_count",
                        1.0,
                    )
                micro_descriptors = [descriptors[offset] for offset in micro_offsets]
                for descriptor, bundle in _execute_heterogeneous_descriptor_chunk(
                    realized_config,
                    descriptors=micro_descriptors,
                    requested_device=requested_device,
                    resolved_device=validated_resolved_device,
                    rows_seed=rows_seed,
                    num_datasets=num_datasets,
                    dtype=dtype,
                    layout_mode="stratified",
                    on_raw_batch_metrics=on_raw_batch_metrics,
                ):
                    bundles_by_dataset_index[int(descriptor.dataset_index)] = bundle

        window_metrics["stratified_scheduler_elapsed_seconds"] = (
            time.perf_counter() - scheduler_start
        )
        window_metrics["stratified_scheduler_cpu_time_seconds"] = (
            time.process_time() - scheduler_start_cpu
        )
        if on_raw_batch_metrics is not None:
            on_raw_batch_metrics(window_metrics)

        for current_index in range(dataset_index, dataset_index + window_size):
            bundle_candidate = bundles_by_dataset_index.get(int(current_index))
            if bundle_candidate is None:
                raise RuntimeError(
                    "Missing stratified heterogeneous bundle for resolved dataset window."
                )
            yield bundle_candidate
        dataset_index += window_size


def _first_valid_classification_attempt_for_dataset(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    dataset_root: KeyedRng,
    requested_device: str,
    resolved_device: str,
    start_attempt: int = 0,
    attempt_budget: int | None = None,
) -> int | None:
    """Return the first valid replay attempt for one dataset seed, if any."""

    shift_params = resolve_shift_runtime_params(config)
    noise_runtime_selection = _resolve_noise_runtime_selection(
        config,
        keyed_rng=dataset_root.keyed("noise_runtime"),
    )
    noise_spec = _noise_sampling_spec(noise_runtime_selection)
    attempts = (
        max(1, int(attempt_budget))
        if attempt_budget is not None
        else _plan_candidate_attempt_budget(config)
    )

    for attempt in range(max(0, int(start_attempt)), attempts):
        if plan.prepared_execution_context is not None:
            y_batch, _aux_meta_batch = _generate_fixed_layout_validation_label_batch(
                config,
                plan.layout,
                execution_plan=plan.execution_plan,
                prepared_execution_context=plan.prepared_execution_context,
                intervention_plan=plan.intervention_plan,
                dataset_seeds=[
                    dataset_root.keyed("attempt", attempt, "raw_generation").child_seed()
                ],
                device=resolved_device,
                noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
                noise_spec=noise_spec,
            )
        else:
            y_batch, _aux_meta_batch = generate_fixed_layout_label_batch(
                config,
                plan.layout,
                execution_plan=plan.execution_plan,
                intervention_plan=plan.intervention_plan,
                dataset_seeds=[
                    dataset_root.keyed("attempt", attempt, "raw_generation").child_seed()
                ],
                device=resolved_device,
                noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
                noise_spec=noise_spec,
            )
        if _raw_classification_labels_support_split(
            y_batch[0],
            dataset_root=dataset_root,
            attempt=attempt,
            n_train=int(plan.n_train),
        ):
            return int(attempt)
    return None


def _grouped_validation_labels_for_attempts(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    dataset_roots: list[KeyedRng],
    attempts: list[int],
    resolved_device: str,
    noise_sigma_multiplier: float,
) -> list[tuple[torch.Tensor, int]]:
    """Return grouped exact validation-label batches indexed back to the input dataset order."""

    grouped_noise_runtime = _group_noise_runtime_chunk(
        config,
        dataset_roots=dataset_roots,
        attempts=attempts,
    )
    raw_batch_by_offset: list[tuple[torch.Tensor, int] | None] = [None] * len(dataset_roots)
    for group in grouped_noise_runtime:
        noise_spec = _noise_sampling_spec(group.selection)
        if plan.prepared_execution_context is not None:
            y_batch, _aux_meta_batch = _generate_fixed_layout_validation_label_batch(
                config,
                plan.layout,
                execution_plan=plan.execution_plan,
                prepared_execution_context=plan.prepared_execution_context,
                intervention_plan=plan.intervention_plan,
                dataset_seeds=group.generation_seeds,
                device=resolved_device,
                noise_sigma_multiplier=float(noise_sigma_multiplier),
                noise_spec=noise_spec,
            )
        else:
            y_batch, _aux_meta_batch = generate_fixed_layout_label_batch(
                config,
                plan.layout,
                execution_plan=plan.execution_plan,
                intervention_plan=plan.intervention_plan,
                dataset_seeds=group.generation_seeds,
                device=resolved_device,
                noise_sigma_multiplier=float(noise_sigma_multiplier),
                noise_spec=noise_spec,
            )
        for local_index, chunk_offset in enumerate(group.chunk_offsets):
            raw_batch_by_offset[int(chunk_offset)] = (y_batch, int(local_index))

    resolved_batches: list[tuple[torch.Tensor, int]] = []
    for chunk_offset, raw_batch_entry in enumerate(raw_batch_by_offset):
        if raw_batch_entry is None:
            raise RuntimeError(
                "Missing grouped validation-label batch entry for fixed-layout chunk offset "
                f"{chunk_offset}."
            )
        resolved_batches.append(raw_batch_entry)
    return resolved_batches


def _batched_valid_classification_attempts_for_datasets(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    dataset_roots: list[KeyedRng],
    requested_device: str,
    resolved_device: str,
    start_attempt: int = 1,
    attempt_budget: int | None = None,
) -> list[int | None]:
    """Return first-valid replay attempts for a dataset group using exact grouped validation."""

    attempts = (
        max(1, int(attempt_budget))
        if attempt_budget is not None
        else _plan_candidate_attempt_budget(config)
    )
    if not dataset_roots:
        return []
    if len(dataset_roots) == 1:
        return [
            _first_valid_classification_attempt_for_dataset(
                config,
                plan=plan,
                dataset_root=dataset_roots[0],
                requested_device=requested_device,
                resolved_device=resolved_device,
                start_attempt=start_attempt,
                attempt_budget=attempts,
            )
        ]

    shift_params = resolve_shift_runtime_params(config)
    unresolved_offsets = list(range(len(dataset_roots)))
    resolved_attempts: list[int | None] = [None] * len(dataset_roots)
    for attempt in range(max(0, int(start_attempt)), attempts):
        if not unresolved_offsets:
            break
        pending_dataset_roots = [dataset_roots[offset] for offset in unresolved_offsets]
        pending_attempts = [int(attempt)] * len(pending_dataset_roots)
        pending_batches = _grouped_validation_labels_for_attempts(
            config,
            plan=plan,
            dataset_roots=pending_dataset_roots,
            attempts=pending_attempts,
            resolved_device=resolved_device,
            noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
        )
        next_unresolved_offsets: list[int] = []
        for pending_offset, dataset_root in enumerate(pending_dataset_roots):
            y_batch, local_index = pending_batches[pending_offset]
            if _raw_classification_labels_support_split(
                y_batch[local_index],
                dataset_root=dataset_root,
                attempt=attempt,
                n_train=int(plan.n_train),
            ):
                resolved_attempts[unresolved_offsets[pending_offset]] = int(attempt)
            else:
                next_unresolved_offsets.append(unresolved_offsets[pending_offset])
        unresolved_offsets = next_unresolved_offsets
    return resolved_attempts


def _fixed_layout_plan_classification_attempt_plan(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    requested_device: str,
    resolved_device: str,
    run_root: KeyedRng,
    num_datasets: int = 1,
    batch_size: int = 1,
) -> tuple[int, ...] | None:
    """Return the first-valid attempt per dataset for a replayable classification run."""

    shift_params = resolve_shift_runtime_params(config)
    replay_attempt_budget = max(1, int(config.filter.max_attempts))
    effective_batch_size = max(1, int(batch_size))
    dataset_index = 0
    attempt_plan: list[int] = []
    while dataset_index < num_datasets:
        chunk_size = min(effective_batch_size, num_datasets - dataset_index)
        dataset_roots = [
            run_root.keyed("dataset", dataset_index + offset) for offset in range(chunk_size)
        ]
        raw_batch_by_offset = _grouped_validation_labels_for_attempts(
            config,
            plan=plan,
            dataset_roots=dataset_roots,
            attempts=[0] * chunk_size,
            resolved_device=resolved_device,
            noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
        )
        invalid_offsets: list[int] = []
        for offset, dataset_root in enumerate(dataset_roots):
            y_batch, local_index = raw_batch_by_offset[offset]
            if _raw_classification_labels_support_split(
                y_batch[local_index],
                dataset_root=dataset_root,
                attempt=0,
                n_train=int(plan.n_train),
            ):
                attempt_plan.append(0)
                continue
            invalid_offsets.append(int(offset))
            attempt_plan.append(-1)
        if invalid_offsets:
            replay_attempts = _batched_valid_classification_attempts_for_datasets(
                config,
                plan=plan,
                dataset_roots=[dataset_roots[offset] for offset in invalid_offsets],
                requested_device=requested_device,
                resolved_device=resolved_device,
                start_attempt=1,
                attempt_budget=replay_attempt_budget,
            )
            for offset, replay_attempt in zip(
                invalid_offsets,
                replay_attempts,
                strict=True,
            ):
                if replay_attempt is None:
                    return None
                attempt_plan[dataset_index + offset] = int(replay_attempt)
        dataset_index += chunk_size

    return tuple(attempt_plan)


def _fixed_layout_plan_supports_classification_run(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    requested_device: str,
    resolved_device: str,
    run_root: KeyedRng,
    num_datasets: int = 1,
    batch_size: int = 1,
) -> bool:
    """Return whether a classification plan can replay for the full requested run."""

    return (
        _fixed_layout_plan_classification_attempt_plan(
            config,
            plan=plan,
            requested_device=requested_device,
            resolved_device=resolved_device,
            run_root=run_root,
            num_datasets=num_datasets,
            batch_size=batch_size,
        )
        is not None
    )


def _fixed_layout_plan_supports_classification_replay(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    requested_device: str,
    resolved_device: str,
    validation_root: KeyedRng,
) -> bool:
    """Return whether a classification plan can replay under the fixed-layout engine."""

    return _fixed_layout_plan_supports_classification_run(
        config,
        plan=plan,
        requested_device=requested_device,
        resolved_device=resolved_device,
        run_root=validation_root,
        num_datasets=1,
        batch_size=1,
    )


def _sample_fixed_layout(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> _FixedLayoutPlan:
    """Sample one in-process layout plan for canonical fixed-layout generation."""

    run_seed = _resolve_run_seed(config, seed)
    run_root = KeyedRng(run_seed)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)
    stress_profile_name = None if config.stress.profile is None else str(config.stress.profile)
    rows_seed = run_root.child_seed("rows")
    attempts = max(1, int(config.filter.max_attempts))
    last_error = "unknown"

    for attempt in range(attempts):
        candidate_root = run_root.keyed("plan_candidate", attempt)
        try:
            plan = _sample_fixed_layout_once(
                config,
                keyed_rng=candidate_root,
                rows_seed=rows_seed,
                requested_device=requested_device,
                resolved_device=resolved_device,
                stress_profile_name=stress_profile_name,
            )
        except InvalidStructuralLayoutError as exc:
            last_error = str(exc.reason)
            continue
        if str(config.dataset.task) != "classification":
            return plan
        valid = False
        for validation_attempt in range(attempts):
            if _fixed_layout_plan_supports_classification_replay(
                config,
                plan=plan,
                requested_device=requested_device,
                resolved_device=resolved_device,
                validation_root=candidate_root.keyed("validation", validation_attempt),
            ):
                valid = True
                break
        if valid:
            return plan
        last_error = "invalid_class_split"

    raise ValueError(
        "Failed to sample a replayable fixed-layout classification plan after "
        f"{attempts} attempts. Last reason: {last_error}."
    )


def _generate_fixed_layout_graph_batch_with_runtime_metrics(
    config: GeneratorConfig,
    plan: _FixedLayoutPlan,
    *,
    dataset_seeds: list[int],
    resolved_device: str,
    noise_sigma_multiplier: float,
    noise_spec,
    runtime_metrics_out: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    if plan.prepared_execution_context is not None:
        return _generate_fixed_layout_graph_batch_prepared(
            config,
            plan.layout,
            execution_plan=plan.execution_plan,
            prepared_execution_context=plan.prepared_execution_context,
            intervention_plan=plan.intervention_plan,
            dataset_seeds=dataset_seeds,
            device=resolved_device,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
            runtime_metrics_out=runtime_metrics_out,
        )
    return generate_fixed_layout_graph_batch(
        config,
        plan.layout,
        execution_plan=plan.execution_plan,
        intervention_plan=plan.intervention_plan,
        dataset_seeds=dataset_seeds,
        device=resolved_device,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
        runtime_metrics_out=runtime_metrics_out,
    )


def _generate_fixed_layout_bundle_with_retries(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    dataset_root: KeyedRng,
    requested_device: str,
    resolved_device: str,
    preserve_feature_schema: bool,
    start_attempt: int = 0,
    finalization_context: _FixedSchemaFinalizationContext | None = None,
    on_raw_batch_metrics: Callable[[dict[str, float]], None] | None = None,
) -> DatasetBundle:
    dataset_seed = dataset_root.child_seed()
    shift_params = resolve_shift_runtime_params(config)
    noise_runtime_selection = _resolve_noise_runtime_selection(
        config,
        keyed_rng=dataset_root.keyed("noise_runtime"),
    )
    noise_spec = _noise_sampling_spec(noise_runtime_selection)
    dtype = _torch_dtype(config)
    attempts = max(1, int(config.filter.max_attempts))
    initial_attempt = max(0, int(start_attempt))
    if initial_attempt >= attempts:
        raise ValueError(
            "Fixed-layout retry start attempt exceeds configured retry budget: "
            f"start_attempt={initial_attempt} max_attempts={attempts}."
        )
    last_error: str = "unknown"

    for attempt in range(initial_attempt, attempts):
        try:
            runtime_metrics: dict[str, float] = {}
            (
                x_batch,
                y_batch,
                aux_meta_batch,
            ) = _generate_fixed_layout_graph_batch_with_runtime_metrics(
                config,
                plan,
                dataset_seeds=[
                    dataset_root.keyed("attempt", attempt, "raw_generation").child_seed()
                ],
                resolved_device=resolved_device,
                noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
                noise_spec=noise_spec,
                runtime_metrics_out=runtime_metrics,
            )
            bundle = _finalize_generated_tensors(
                config,
                plan.layout,
                dataset_seed=dataset_seed,
                attempt=attempt,
                attempts_used=attempt + 1,
                dataset_root=dataset_root,
                device=resolved_device,
                n_train=int(plan.n_train),
                n_test=int(plan.n_test),
                requested_device=requested_device,
                resolved_device=resolved_device,
                device_fallback_reason=None,
                x=x_batch[0],
                y=y_batch[0],
                aux_meta=aux_meta_batch[0],
                shift_params=shift_params,
                noise_runtime_selection=noise_runtime_selection,
                dtype=dtype,
                preserve_feature_schema=preserve_feature_schema,
                finalization_context=finalization_context,
                runtime_metrics_out=runtime_metrics,
            )
            if on_raw_batch_metrics is not None:
                on_raw_batch_metrics(dict(runtime_metrics))
            return bundle
        except ValueError as exc:
            recoverable_failure = classify_recoverable_generation_failure(exc)
            if recoverable_failure is None:
                raise
            last_error = recoverable_failure.reason
            if recoverable_failure.retry_scope == RECOVERABLE_RETRY_SCOPE_SAME_PLAN_ATTEMPT:
                continue
            break

    raise ValueError(
        "Failed to generate a valid fixed-layout dataset after "
        f"{attempts} attempts. Last reason: {last_error}."
    )


def _generate_batch_with_plan_iter(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    num_datasets: int,
    seed: int | None = None,
    batch_size: int | None = None,
    classification_attempt_plan: tuple[int, ...] | None = None,
    on_raw_batch_metrics: Callable[[dict[str, float]], None] | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets for one in-process fixed-layout plan."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return
    if classification_attempt_plan is not None and len(classification_attempt_plan) != num_datasets:
        raise ValueError(
            "Fixed-layout classification attempt plan length must match num_datasets: "
            f"attempt_plan={len(classification_attempt_plan)} num_datasets={num_datasets}"
        )
    if config.steering.enabled:
        yield from _generate_batch_with_dynamic_steering_iter(
            config,
            base_plan=plan,
            num_datasets=num_datasets,
            seed=seed,
            batch_size=batch_size,
            classification_attempt_plan=classification_attempt_plan,
            on_raw_batch_metrics=on_raw_batch_metrics,
        )
        return

    requested_device = str(plan.requested_device)
    validated_resolved_device = str(plan.resolved_device)
    run_seed = _resolve_run_seed(config, seed)
    run_root = KeyedRng(run_seed)
    dtype = _torch_dtype(config)
    shift_params = resolve_shift_runtime_params(config)
    expected_schema: tuple[int, tuple[str, ...], tuple[int, ...], int] | None = None
    finalization_context = _build_fixed_schema_finalization_context(
        config,
        plan.layout,
        n_train=int(plan.n_train),
        n_test=int(plan.n_test),
        shift_params=shift_params,
    )

    effective_batch_size = _resolve_fixed_layout_batch_size(
        plan,
        num_datasets=num_datasets,
        batch_size=batch_size,
        target_cells=_effective_fixed_layout_target_cells(config),
        batch_size_cap=config.runtime.fixed_layout_batch_size_cap,
    )
    dataset_index = 0
    while dataset_index < num_datasets:
        chunk_size = min(effective_batch_size, num_datasets - dataset_index)
        dataset_roots = [
            run_root.keyed("dataset", dataset_index + offset) for offset in range(chunk_size)
        ]
        chunk_attempts = (
            list(classification_attempt_plan[dataset_index : dataset_index + chunk_size])
            if classification_attempt_plan is not None
            else [0] * chunk_size
        )
        raw_batch_by_offset: list[DatasetBundle | None] = [None] * chunk_size
        finalized_offsets: set[int] = set()
        zero_attempt_offsets = [
            offset for offset, attempt in enumerate(chunk_attempts) if int(attempt) == 0
        ]
        retry_offsets = [
            offset for offset, attempt in enumerate(chunk_attempts) if int(attempt) != 0
        ]
        retry_offset_set = set(retry_offsets)
        grouped_attempt_zero_offsets = (
            list(range(chunk_size))
            if classification_attempt_plan is not None
            else zero_attempt_offsets
        )

        if grouped_attempt_zero_offsets:
            grouped_attempt_zero_dataset_roots = [
                dataset_roots[offset] for offset in grouped_attempt_zero_offsets
            ]
            grouped_noise_runtime = _group_noise_runtime_chunk(
                config,
                dataset_roots=grouped_attempt_zero_dataset_roots,
                attempts=[0] * len(grouped_attempt_zero_dataset_roots),
            )
        else:
            grouped_noise_runtime = []
        grouped_raw_batches = _generate_grouped_raw_batches(
            config,
            plan.layout,
            execution_plan=plan.execution_plan,
            intervention_plan=plan.intervention_plan,
            grouped_noise_runtime=grouped_noise_runtime,
            requested_device=requested_device,
            resolved_device=validated_resolved_device,
            noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
        )
        for grouped_batch in grouped_raw_batches:
            if on_raw_batch_metrics is not None:
                on_raw_batch_metrics(dict(getattr(grouped_batch, "runtime_metrics", {})))
            group_dataset_offsets = [
                grouped_attempt_zero_offsets[int(chunk_offset)]
                for chunk_offset in grouped_batch.chunk_offsets
            ]
            group_dataset_roots = [
                dataset_roots[int(chunk_offset)] for chunk_offset in group_dataset_offsets
            ]
            resolved_split_indices: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
            if str(config.dataset.task) == "classification":
                resolved_split_indices = []
                for local_index, dataset_root in enumerate(group_dataset_roots):
                    attempt_root = dataset_root.keyed("attempt", grouped_batch.attempt)
                    try:
                        split_indices = _resolve_split_indices(
                            grouped_batch.y_batch[local_index],
                            task=config.dataset.task,
                            n_train=int(plan.n_train),
                            keyed_rng=attempt_root.keyed("split"),
                        )
                    except InfeasibleStratifiedSplitError:
                        split_indices = None
                    resolved_split_indices.append(split_indices)
            finalized_group = _finalize_generated_chunk_preserve_schema(
                config,
                plan.layout,
                context=finalization_context,
                dataset_roots=group_dataset_roots,
                attempt=grouped_batch.attempt,
                attempts_used=grouped_batch.attempt + 1,
                device=grouped_batch.effective_resolved_device,
                n_train=int(plan.n_train),
                n_test=int(plan.n_test),
                requested_device=requested_device,
                resolved_device=grouped_batch.effective_resolved_device,
                device_fallback_reason=grouped_batch.device_fallback_reason,
                x=grouped_batch.x_batch,
                y=grouped_batch.y_batch,
                aux_meta_batch=grouped_batch.aux_meta_batch,
                noise_runtime_selection=grouped_batch.selection,
                dtype=dtype,
                resolved_split_indices=resolved_split_indices,
            )
            for local_index, chunk_offset in enumerate(group_dataset_offsets):
                offset = int(chunk_offset)
                raw_batch_by_offset[offset] = finalized_group[local_index]
                finalized_offsets.add(offset)
        for offset, dataset_root in enumerate(dataset_roots):
            if offset in retry_offset_set:
                bundle = _generate_fixed_layout_bundle_with_retries(
                    config,
                    plan=plan,
                    dataset_root=dataset_root,
                    requested_device=requested_device,
                    resolved_device=validated_resolved_device,
                    preserve_feature_schema=True,
                    start_attempt=chunk_attempts[offset],
                    finalization_context=finalization_context,
                    on_raw_batch_metrics=on_raw_batch_metrics,
                )
                _annotate_fixed_layout_metadata(bundle, plan=plan)
                schema = _extract_emitted_schema_signature(bundle)
                if expected_schema is None:
                    expected_schema = schema
                elif schema != expected_schema:
                    raise ValueError(
                        "Fixed-layout schema mismatch: emitted dataset does not match "
                        "the first fixed-layout bundle schema."
                    )
                yield bundle
                continue

            if offset not in finalized_offsets:
                raise RuntimeError("Missing grouped raw batch entry for fixed-layout chunk offset.")
            grouped_bundle = raw_batch_by_offset[offset]
            if grouped_bundle is None:
                grouped_bundle = _generate_fixed_layout_bundle_with_retries(
                    config,
                    plan=plan,
                    dataset_root=dataset_root,
                    requested_device=requested_device,
                    resolved_device=validated_resolved_device,
                    preserve_feature_schema=True,
                    start_attempt=int(chunk_attempts[offset]),
                    finalization_context=finalization_context,
                    on_raw_batch_metrics=on_raw_batch_metrics,
                )
            _annotate_fixed_layout_metadata(grouped_bundle, plan=plan)
            schema = _extract_emitted_schema_signature(grouped_bundle)
            if expected_schema is None:
                expected_schema = schema
            elif schema != expected_schema:
                raise ValueError(
                    "Fixed-layout schema mismatch: emitted dataset does not match "
                    "the first fixed-layout bundle schema."
                )
            yield grouped_bundle
        dataset_index += chunk_size


__all__ = [
    "CanonicalFixedLayoutRun",
    "_fixed_layout_plan_supports_classification_run",
    "_generate_batch_with_heterogeneous_layout_iter",
    "_generate_batch_with_plan_iter",
    "_generate_batch_with_stratified_layout_iter",
    "_replay_emitted_fixed_layout_plan",
    "_resolve_fixed_layout_batch_size",
    "_sample_fixed_layout",
    "prepare_canonical_fixed_layout_run",
    "realize_generation_config_for_run",
]
