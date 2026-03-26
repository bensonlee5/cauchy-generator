"""Canonical fixed-layout run preparation and execution orchestration."""

from __future__ import annotations

import math
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
    InfeasibleStratifiedSplitError,
    InvalidClassSplitError,
    _classification_split_valid,
    _stratified_split_indices,
)
from dagzoo.rng import KeyedRng
from dagzoo.types import DatasetBundle

from .batched import (
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


@dataclass(slots=True)
class _SteeredDatasetDescriptor:
    """Resolved steering state and fixed-layout runtime inputs for one dataset."""

    dataset_index: int
    dataset_root: KeyedRng
    effective_config: GeneratorConfig
    effective_plan: _FixedLayoutPlan
    effective_shift: ShiftRuntimeParams
    finalization_context: _FixedSchemaFinalizationContext


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


def _steered_raw_generation_cohort_key(
    descriptor: _SteeredDatasetDescriptor,
) -> tuple[object, ...]:
    """Return the raw-generation contract key for one steered dataset descriptor."""

    return (
        str(descriptor.effective_plan.layout_signature),
        str(descriptor.effective_plan.plan_signature or ""),
        str(descriptor.effective_config.dataset.task),
        int(descriptor.effective_plan.n_train),
        int(descriptor.effective_plan.n_test),
        *_noise_config_signature(descriptor.effective_config),
        float(descriptor.effective_shift.variance_sigma_multiplier),
    )


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

    if (
        len(layout_root_path) >= 3
        and layout_root_path[0] == "plan_candidate"
        and isinstance(layout_root_path[1], int)
        and layout_root_path[2] == "layout"
    ):
        return int(layout_root_path[1])
    return 0


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

    run_root = KeyedRng(int(run_seed))
    layout = _sample_layout(config, run_root.keyed(*layout_root_path), "cpu")
    if normalized_steering_layout_root_path is not None:
        layout = _resample_layout_graph(
            layout,
            keyed_rng=run_root.keyed(*normalized_steering_layout_root_path),
            edge_logit_bias=float(effective_shift.edge_logit_bias_shift),
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
    )

    requested_device = metadata.get("requested_device")
    if not isinstance(requested_device, str) or not requested_device:
        raise ValueError("metadata.requested_device must be a non-empty string.")
    resolved_device = metadata.get("resolved_device")
    if not isinstance(resolved_device, str) or not resolved_device:
        raise ValueError("metadata.resolved_device must be a non-empty string.")
    layout_plan_seed = metadata.get("layout_plan_seed")
    if isinstance(layout_plan_seed, bool) or not isinstance(layout_plan_seed, int):
        plan_seed_root_path = (
            layout_root_path
            if normalized_steering_layout_root_path is None
            else normalized_steering_layout_root_path
        )
        layout_plan_seed = run_root.keyed(*plan_seed_root_path).child_seed()

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
) -> _FixedLayoutPlan:
    """Sample one fixed-layout plan candidate without replay validation."""

    return _sample_fixed_layout_once(
        config,
        keyed_rng=keyed_rng,
        rows_seed=rows_seed,
        requested_device=requested_device,
        resolved_device=resolved_device,
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
    grouped_noise_runtime: list[_NoiseRuntimeGroup],
    requested_device: str,
    resolved_device: str,
    noise_sigma_multiplier: float,
) -> list[_GroupedRawBatch]:
    return _generate_grouped_raw_batches_impl(
        config,
        layout,
        execution_plan=execution_plan,
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
) -> CanonicalFixedLayoutRun:
    """Prepare one internal fixed-layout run context for public generation APIs."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")

    realized_config, run_seed, requested_device, resolved_device = (
        realize_generation_config_for_run(
            config,
            seed=seed,
            device=device,
        )
    )
    run_root = KeyedRng(run_seed)
    rows_seed = run_root.child_seed("rows")
    attempts = max(1, int(realized_config.filter.max_attempts))
    last_error = "unknown"
    for attempt in range(attempts):
        candidate_root = run_root.keyed("plan_candidate", attempt)
        plan = _sample_fixed_layout_candidate(
            realized_config,
            keyed_rng=candidate_root,
            rows_seed=rows_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
        )
        effective_batch_size = _resolve_fixed_layout_batch_size(
            plan,
            num_datasets=max(1, int(num_datasets)),
            batch_size=batch_size,
            target_cells=_effective_fixed_layout_target_cells(realized_config),
        )
        break
    else:
        raise ValueError(
            "Failed to prepare a fixed-layout run after "
            f"{attempts} attempts. Last reason: {last_error}."
        )
    return CanonicalFixedLayoutRun(
        config=realized_config,
        plan=plan,
        run_seed=int(run_seed),
        requested_device=str(requested_device),
        resolved_device=str(resolved_device),
        batch_size=int(effective_batch_size),
    )


def _sample_fixed_layout_once(
    config: GeneratorConfig,
    *,
    keyed_rng: KeyedRng,
    rows_seed: int,
    requested_device: str,
    resolved_device: str,
) -> _FixedLayoutPlan:
    """Sample one fixed-layout plan candidate without replay validation retries."""

    _validate_fixed_layout_rows_mode(config)
    layout_seed = keyed_rng.child_seed("layout")
    layout = _sample_layout(config, keyed_rng.keyed("layout"), "cpu")
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
    )
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
    )


def _resolve_steered_plan_for_dataset(
    config: GeneratorConfig,
    *,
    base_plan: _FixedLayoutPlan,
    dataset_index: int,
    num_datasets: int,
    dataset_root: KeyedRng,
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

    layout_root = dataset_root.keyed("steering", "layout")
    layout = _resample_layout_graph(
        base_plan.layout,
        keyed_rng=layout_root,
        edge_logit_bias=float(effective_shift.edge_logit_bias_shift),
    )
    execution_plan_root = dataset_root.keyed("steering", "execution_plan")
    execution_plan_seed = execution_plan_root.child_seed()
    execution_plan = build_fixed_layout_execution_plan(
        effective_config,
        layout,
        plan_seed=execution_plan_seed,
        mechanism_logit_tilt=float(effective_shift.mechanism_logit_tilt),
    )
    plan_seed = layout_root.child_seed()
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
        steering_layout_root_path=["dataset", int(dataset_index), "steering", "layout"],
        steering_execution_plan_root_path=[
            "dataset",
            int(dataset_index),
            "steering",
            "execution_plan",
        ],
    )


def _resolve_steered_dataset_descriptor(
    config: GeneratorConfig,
    *,
    base_plan: _FixedLayoutPlan,
    dataset_index: int,
    num_datasets: int,
    dataset_root: KeyedRng,
) -> _SteeredDatasetDescriptor:
    """Resolve one per-dataset steering descriptor for batched fixed-layout execution."""

    effective_config, effective_plan = _resolve_steered_plan_for_dataset(
        config,
        base_plan=base_plan,
        dataset_index=dataset_index,
        num_datasets=num_datasets,
        dataset_root=dataset_root,
    )
    effective_shift = resolve_shift_runtime_params(effective_config)
    finalization_context = _build_fixed_schema_finalization_context(
        effective_config,
        effective_plan.layout,
        n_train=int(effective_plan.n_train),
        n_test=int(effective_plan.n_test),
        shift_params=effective_shift,
    )
    return _SteeredDatasetDescriptor(
        dataset_index=int(dataset_index),
        dataset_root=dataset_root,
        effective_config=effective_config,
        effective_plan=effective_plan,
        effective_shift=effective_shift,
        finalization_context=finalization_context,
    )


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
    expected_schema: tuple[int, tuple[str, ...], tuple[int, ...]] | None = None
    effective_batch_size = _resolve_fixed_layout_batch_size(
        base_plan,
        num_datasets=num_datasets,
        batch_size=batch_size,
        target_cells=_effective_fixed_layout_target_cells(config),
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
            cohort_key = _steered_raw_generation_cohort_key(descriptors[offset])
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
            grouped_raw_batches = _generate_grouped_raw_batches(
                representative.effective_config,
                representative.effective_plan.layout,
                execution_plan=representative.effective_plan.execution_plan,
                grouped_noise_runtime=grouped_noise_runtime,
                requested_device=requested_device,
                resolved_device=validated_resolved_device,
                noise_sigma_multiplier=float(
                    representative.effective_shift.variance_sigma_multiplier
                ),
            )
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
                finalized_group = _finalize_generated_chunk_preserve_schema(
                    representative.effective_config,
                    representative.effective_plan.layout,
                    context=representative.finalization_context,
                    contexts_by_batch=[entry.finalization_context for entry in group_descriptors],
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
                bundle = _generate_bundle_with_retries_compat(
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
                    bundle = _generate_bundle_with_retries_compat(
                        descriptor.effective_config,
                        plan=descriptor.effective_plan,
                        dataset_root=descriptor.dataset_root,
                        requested_device=requested_device,
                        resolved_device=validated_resolved_device,
                        preserve_feature_schema=True,
                        start_attempt=max(1, int(chunk_attempts[offset])),
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


def _first_valid_classification_attempt_for_dataset(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    dataset_root: KeyedRng,
    requested_device: str,
    resolved_device: str,
    start_attempt: int = 0,
) -> int | None:
    """Return the first valid replay attempt for one dataset seed, if any."""

    shift_params = resolve_shift_runtime_params(config)
    noise_runtime_selection = _resolve_noise_runtime_selection(
        config,
        keyed_rng=dataset_root.keyed("noise_runtime"),
    )
    noise_spec = _noise_sampling_spec(noise_runtime_selection)
    attempts = max(1, int(config.filter.max_attempts))

    for attempt in range(max(0, int(start_attempt)), attempts):
        y_batch, _aux_meta_batch = generate_fixed_layout_label_batch(
            config,
            plan.layout,
            execution_plan=plan.execution_plan,
            dataset_seeds=[dataset_root.keyed("attempt", attempt, "raw_generation").child_seed()],
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
    effective_batch_size = max(1, int(batch_size))
    dataset_index = 0
    attempt_plan: list[int] = []
    while dataset_index < num_datasets:
        chunk_size = min(effective_batch_size, num_datasets - dataset_index)
        dataset_roots = [
            run_root.keyed("dataset", dataset_index + offset) for offset in range(chunk_size)
        ]
        grouped_noise_runtime = _group_noise_runtime_chunk(
            config,
            dataset_roots=dataset_roots,
        )
        raw_batch_by_offset: list[tuple[torch.Tensor, int] | None] = [None] * chunk_size
        for group in grouped_noise_runtime:
            noise_spec = _noise_sampling_spec(group.selection)
            y_batch, _aux_meta_batch = generate_fixed_layout_label_batch(
                config,
                plan.layout,
                execution_plan=plan.execution_plan,
                dataset_seeds=group.generation_seeds,
                device=resolved_device,
                noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
                noise_spec=noise_spec,
            )
            for local_index, chunk_offset in enumerate(group.chunk_offsets):
                raw_batch_by_offset[chunk_offset] = (y_batch, int(local_index))
        for offset, dataset_root in enumerate(dataset_roots):
            raw_batch_entry = raw_batch_by_offset[offset]
            if raw_batch_entry is None:
                raise RuntimeError("Missing grouped raw batch entry for fixed-layout chunk offset.")
            y_batch, local_index = raw_batch_entry
            if _raw_classification_labels_support_split(
                y_batch[local_index],
                dataset_root=dataset_root,
                attempt=0,
                n_train=int(plan.n_train),
            ):
                attempt_plan.append(0)
                continue
            replay_attempt = _first_valid_classification_attempt_for_dataset(
                config,
                plan=plan,
                dataset_root=dataset_root,
                requested_device=requested_device,
                resolved_device=resolved_device,
                start_attempt=1,
            )
            if replay_attempt is None:
                return None
            attempt_plan.append(int(replay_attempt))
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
    rows_seed = run_root.child_seed("rows")
    attempts = max(1, int(config.filter.max_attempts))
    last_error = "unknown"

    for attempt in range(attempts):
        candidate_root = run_root.keyed("plan_candidate", attempt)
        plan = _sample_fixed_layout_once(
            config,
            keyed_rng=candidate_root,
            rows_seed=rows_seed,
            requested_device=requested_device,
            resolved_device=resolved_device,
        )
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
    try:
        return generate_fixed_layout_graph_batch(
            config,
            plan.layout,
            execution_plan=plan.execution_plan,
            dataset_seeds=dataset_seeds,
            device=resolved_device,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
            runtime_metrics_out=runtime_metrics_out,
        )
    except TypeError as exc:
        if "runtime_metrics_out" not in str(exc):
            raise
        return generate_fixed_layout_graph_batch(
            config,
            plan.layout,
            execution_plan=plan.execution_plan,
            dataset_seeds=dataset_seeds,
            device=resolved_device,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )


def _generate_bundle_with_retries_compat(
    config: GeneratorConfig,
    *,
    plan: _FixedLayoutPlan,
    dataset_root: KeyedRng,
    requested_device: str,
    resolved_device: str,
    preserve_feature_schema: bool,
    start_attempt: int,
    finalization_context: _FixedSchemaFinalizationContext,
    on_raw_batch_metrics: Callable[[dict[str, float]], None] | None,
) -> DatasetBundle:
    try:
        return _generate_fixed_layout_bundle_with_retries(
            config,
            plan=plan,
            dataset_root=dataset_root,
            requested_device=requested_device,
            resolved_device=resolved_device,
            preserve_feature_schema=preserve_feature_schema,
            start_attempt=start_attempt,
            finalization_context=finalization_context,
            on_raw_batch_metrics=on_raw_batch_metrics,
        )
    except TypeError as exc:
        if "on_raw_batch_metrics" not in str(exc):
            raise
        return _generate_fixed_layout_bundle_with_retries(
            config,
            plan=plan,
            dataset_root=dataset_root,
            requested_device=requested_device,
            resolved_device=resolved_device,
            preserve_feature_schema=preserve_feature_schema,
            start_attempt=start_attempt,
            finalization_context=finalization_context,
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
        runtime_metrics: dict[str, float] = {}
        (
            x_batch,
            y_batch,
            aux_meta_batch,
        ) = _generate_fixed_layout_graph_batch_with_runtime_metrics(
            config,
            plan,
            dataset_seeds=[dataset_root.keyed("attempt", attempt, "raw_generation").child_seed()],
            resolved_device=resolved_device,
            noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
            noise_spec=noise_spec,
            runtime_metrics_out=runtime_metrics,
        )
        if on_raw_batch_metrics is not None:
            on_raw_batch_metrics(dict(runtime_metrics))
        try:
            return _finalize_generated_tensors(
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
            )
        except InvalidClassSplitError:
            last_error = "invalid_class_split"
            continue

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
    expected_schema: tuple[int, tuple[str, ...], tuple[int, ...]] | None = None
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

        if zero_attempt_offsets:
            zero_attempt_dataset_roots = [dataset_roots[offset] for offset in zero_attempt_offsets]
            grouped_noise_runtime = _group_noise_runtime_chunk(
                config,
                dataset_roots=zero_attempt_dataset_roots,
                attempts=[0] * len(zero_attempt_dataset_roots),
            )
        else:
            grouped_noise_runtime = []
        grouped_raw_batches = _generate_grouped_raw_batches(
            config,
            plan.layout,
            execution_plan=plan.execution_plan,
            grouped_noise_runtime=grouped_noise_runtime,
            requested_device=requested_device,
            resolved_device=validated_resolved_device,
            noise_sigma_multiplier=float(shift_params.variance_sigma_multiplier),
        )
        for grouped_batch in grouped_raw_batches:
            if on_raw_batch_metrics is not None:
                on_raw_batch_metrics(dict(getattr(grouped_batch, "runtime_metrics", {})))
            group_dataset_offsets = [
                zero_attempt_offsets[int(chunk_offset)]
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
                bundle = _generate_bundle_with_retries_compat(
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
                grouped_bundle = _generate_bundle_with_retries_compat(
                    config,
                    plan=plan,
                    dataset_root=dataset_root,
                    requested_device=requested_device,
                    resolved_device=validated_resolved_device,
                    preserve_feature_schema=True,
                    start_attempt=max(1, int(chunk_attempts[offset])),
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
    "_generate_batch_with_plan_iter",
    "_replay_emitted_fixed_layout_plan",
    "_resolve_fixed_layout_batch_size",
    "_sample_fixed_layout",
    "prepare_canonical_fixed_layout_run",
    "realize_generation_config_for_run",
]
