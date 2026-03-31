"""Shared generation finalization helpers for canonical fixed-layout execution."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core.config_predicates import missingness_enabled as _is_missingness_enabled
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.core.metadata import _build_lineage_metadata, _build_shift_metadata
from dagzoo.core.noise_runtime import (
    NoiseRuntimeSelection,
    _build_noise_distribution_metadata,
)
from dagzoo.core.shift import ShiftRuntimeParams
from dagzoo.core.validation import (
    InfeasibleStratifiedSplitError,
    InvalidClassSplitError,
    _classification_split_valid,
    _stratified_split_indices,
)
from dagzoo.postprocess.postprocess import (
    inject_missingness,
    postprocess_dataset,
    postprocess_fixed_schema_target_batch,
    postprocess_targets,
)
from dagzoo.rng import KeyedRng
from dagzoo.types import DatasetBundle


@dataclass(slots=True)
class _FixedSchemaFinalizationContext:
    """Cached metadata used by fixed-schema chunk finalization."""

    config_payload: dict[str, Any]
    shift_metadata: dict[str, Any]
    feature_types: list[str]
    feature_index_map: list[int]
    base_metadata: dict[str, Any]
    missingness_enabled: bool


def _config_payload_for_metadata(
    config: GeneratorConfig,
    *,
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    """Serialize config metadata while omitting schema-only runtime-irrelevant sections."""

    config_payload = asdict(config)
    dataset_payload = config_payload.get("dataset")
    if isinstance(dataset_payload, dict):
        dataset_payload["n_train"] = int(n_train)
        dataset_payload["n_test"] = int(n_test)
    config_payload.pop("steering", None)
    config_payload.pop("stress", None)

    runtime_payload = config_payload.get("runtime")
    if (
        isinstance(runtime_payload, dict)
        and runtime_payload.get("fixed_layout_target_cells") is None
    ):
        runtime_payload.pop("fixed_layout_target_cells", None)
    if (
        isinstance(runtime_payload, dict)
        and runtime_payload.get("fixed_layout_batch_size_cap") is None
    ):
        runtime_payload.pop("fixed_layout_batch_size_cap", None)
    return config_payload


def _classification_class_structure(
    *,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    n_classes_sampled: int,
) -> dict[str, Any]:
    """Build classification label-structure metadata for one emitted bundle."""

    y_train_i64 = y_train.to(torch.int64)
    y_test_i64 = y_test.to(torch.int64)
    y_all = torch.cat([y_train_i64, y_test_i64], dim=0)
    unique_all = torch.unique(y_all, sorted=True)
    n_classes_realized = int(unique_all.numel())
    labels_contiguous = bool(
        torch.equal(
            unique_all,
            torch.arange(n_classes_realized, dtype=unique_all.dtype, device=unique_all.device),
        )
    )
    train_classes = torch.unique(y_train_i64, sorted=True)
    test_classes = torch.unique(y_test_i64, sorted=True)

    return {
        "n_classes_sampled": int(n_classes_sampled),
        "n_classes_realized": int(n_classes_realized),
        "labels_contiguous": bool(labels_contiguous),
        "train_test_class_match": bool(torch.equal(train_classes, test_classes)),
        "min_label": int(unique_all[0].item()) if n_classes_realized > 0 else None,
        "max_label": int(unique_all[-1].item()) if n_classes_realized > 0 else None,
    }


def _resolve_split_indices(
    y: torch.Tensor,
    *,
    task: str,
    n_train: int,
    keyed_rng: KeyedRng,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve one dataset's train/test split indices with CUDA preference and CPU fallback."""

    device_candidates = ("cuda", "cpu") if str(y.device.type) == "cuda" else ("cpu",)
    last_error: Exception | None = None
    for split_device in device_candidates:
        try:
            return _resolve_split_indices_for_device(
                y,
                task=task,
                n_train=n_train,
                keyed_rng=keyed_rng,
                device=split_device,
            )
        except InfeasibleStratifiedSplitError:
            raise
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to resolve train/test split indices.")


def _resolve_split_indices_for_device(
    y: torch.Tensor,
    *,
    task: str,
    n_train: int,
    keyed_rng: KeyedRng,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve one dataset's train/test split indices on one requested split device."""

    generator = keyed_rng.torch_rng(device=device)
    split_y = y if str(y.device.type) == device else y.to(device=device)
    if task == "classification":
        return _stratified_split_indices(
            split_y,
            n_train,
            generator,
            device,
        )

    total_rows = int(split_y.shape[0])
    order = torch.randperm(
        total_rows,
        generator=generator,
        device=device,
    )
    return order[:n_train], order[n_train:]


def _split_raw_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one raw feature/target pair using indices applied on the tensor device."""

    train_index = train_idx if train_idx.device == x.device else train_idx.to(device=x.device)
    test_index = test_idx if test_idx.device == x.device else test_idx.to(device=x.device)
    return x[train_index], y[train_index], x[test_index], y[test_index]


def _normalized_filter_metadata(aux_meta: dict[str, Any]) -> dict[str, Any]:
    """Return the emitted filter metadata shape for one dataset."""

    filter_metadata = aux_meta.get("filter", {})
    if isinstance(filter_metadata, dict):
        return dict(filter_metadata)
    return {"mode": "deferred", "status": "not_run"}


def _base_bundle_metadata_for_layout(
    layout: LayoutPlan,
    *,
    feature_types: list[str],
    feature_index_map: list[int],
) -> dict[str, Any]:
    """Build invariant metadata shared across bundles with one emitted feature schema."""

    return {
        "backend": "torch",
        "compute_backend": "torch_appendix_full",
        "n_features": int(len(feature_types)),
        "n_categorical_features": int(sum(1 for t in feature_types if t == "cat")),
        "graph_nodes": int(layout.graph_nodes),
        "graph_edges": int(layout.graph_edges),
        "graph_depth_nodes": int(layout.graph_depth_nodes),
        "graph_edge_density": float(layout.graph_edge_density),
        "lineage": _build_lineage_metadata(layout, feature_index_map=feature_index_map),
    }


def _build_fixed_schema_finalization_context(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    n_train: int,
    n_test: int,
    shift_params: ShiftRuntimeParams,
    feature_types: list[str] | None = None,
    feature_index_map: list[int] | None = None,
) -> _FixedSchemaFinalizationContext:
    """Build cached metadata for fixed-schema bundle finalization."""

    config_payload = _config_payload_for_metadata(
        config,
        n_train=n_train,
        n_test=n_test,
    )
    resolved_feature_types = (
        [str(feature_type) for feature_type in list(layout.feature_types)]
        if feature_types is None
        else [str(feature_type) for feature_type in feature_types]
    )
    resolved_feature_index_map = (
        [int(i) for i in range(int(layout.n_features))]
        if feature_index_map is None
        else [int(i) for i in feature_index_map]
    )
    return _FixedSchemaFinalizationContext(
        config_payload=config_payload,
        shift_metadata=_build_shift_metadata(
            shift_params=shift_params,
            function_family_mix=config.mechanism.function_family_mix,
        ),
        feature_types=resolved_feature_types,
        feature_index_map=resolved_feature_index_map,
        base_metadata=_base_bundle_metadata_for_layout(
            layout,
            feature_types=resolved_feature_types,
            feature_index_map=resolved_feature_index_map,
        ),
        missingness_enabled=_is_missingness_enabled(config),
    )


def _build_bundle_metadata(
    context: _FixedSchemaFinalizationContext,
    *,
    seed: int,
    attempt: int,
    attempts_used: int,
    device: str,
    requested_device: str,
    resolved_device: str,
    device_fallback_reason: str | None,
    aux_meta: dict[str, Any],
    noise_runtime_selection: NoiseRuntimeSelection,
    missingness_summary: dict[str, Any] | None,
    class_structure: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build emitted dataset metadata for scalar and batched finalization paths."""

    n_classes = None if class_structure is None else int(class_structure["n_classes_realized"])
    metadata = dict(context.base_metadata)
    metadata["lineage"] = copy.deepcopy(context.base_metadata["lineage"])
    metadata.update(
        {
            "device": str(device),
            "requested_device": str(requested_device),
            "resolved_device": str(resolved_device),
            "device_fallback_reason": device_fallback_reason,
            "n_classes": n_classes,
            "seed": int(seed),
            "attempt_used": int(attempt),
            "filter": _normalized_filter_metadata(aux_meta),
            "shift": copy.deepcopy(context.shift_metadata),
            "noise_distribution": _build_noise_distribution_metadata(noise_runtime_selection),
            "generation_attempts": {
                "total_attempts": int(attempts_used),
                "retry_count": int(max(0, attempts_used - 1)),
                "filter_attempts": 0,
                "filter_rejections": 0,
                "filter_rejection_rate": None,
            },
            "prior": {
                "factorization": "independent_p_x_complete_and_p_y_given_x_complete",
                "target_head": "latent_complete_x_conditional",
                "feature_generator": "latent_dag",
                "missingness_stage": "post_target_observation",
                "classification_validity_policy": "retry_only",
                "localization_mode": "none",
                "n_adaptation": "none",
            },
            "config": copy.deepcopy(context.config_payload),
        }
    )
    if missingness_summary is not None:
        metadata["missingness"] = missingness_summary
    if class_structure is not None:
        metadata["class_structure"] = class_structure
    return metadata


def _finalize_processed_bundle(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    context: _FixedSchemaFinalizationContext,
    dataset_seed: int,
    attempt: int,
    attempts_used: int,
    attempt_root: KeyedRng,
    device: str,
    requested_device: str,
    resolved_device: str,
    device_fallback_reason: str | None,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    aux_meta: dict[str, Any],
    noise_runtime_selection: NoiseRuntimeSelection,
    dtype: torch.dtype,
) -> DatasetBundle:
    """Finalize one already-postprocessed dataset into the emitted bundle contract."""

    if context.missingness_enabled:
        x_train, x_test, missingness_summary = inject_missingness(
            x_train,
            x_test,
            dataset_cfg=config.dataset,
            keyed_rng=attempt_root.keyed("missingness"),
            device=device,
        )
    else:
        missingness_summary = None

    if config.dataset.task == "classification" and not _classification_split_valid(y_train, y_test):
        raise InvalidClassSplitError("invalid_class_split")

    x_train = x_train.to(device=device, dtype=dtype)
    x_test = x_test.to(device=device, dtype=dtype)
    y_dtype = torch.int64 if config.dataset.task == "classification" else dtype
    y_train = y_train.to(device=device, dtype=y_dtype)
    y_test = y_test.to(device=device, dtype=y_dtype)

    class_structure: dict[str, Any] | None = None
    if config.dataset.task == "classification":
        class_structure = _classification_class_structure(
            y_train=y_train,
            y_test=y_test,
            n_classes_sampled=int(layout.n_classes),
        )

    metadata = _build_bundle_metadata(
        context,
        seed=dataset_seed,
        attempt=attempt,
        attempts_used=attempts_used,
        device=device,
        requested_device=requested_device,
        resolved_device=resolved_device,
        device_fallback_reason=device_fallback_reason,
        aux_meta=aux_meta,
        noise_runtime_selection=noise_runtime_selection,
        missingness_summary=missingness_summary,
        class_structure=class_structure,
    )
    return DatasetBundle(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        feature_types=list(context.feature_types),
        metadata=metadata,
        runtime_metrics={},
    )


def _finalize_generated_chunk_preserve_schema(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    context: _FixedSchemaFinalizationContext,
    contexts_by_batch: Sequence[_FixedSchemaFinalizationContext] | None = None,
    configs_by_batch: Sequence[GeneratorConfig] | None = None,
    dataset_roots: list[KeyedRng],
    attempt: int,
    attempts_used: int,
    device: str,
    n_train: int,
    n_test: int,
    requested_device: str,
    resolved_device: str,
    device_fallback_reason: str | None,
    x: torch.Tensor,
    y: torch.Tensor,
    aux_meta_batch: list[dict[str, Any]],
    noise_runtime_selection: NoiseRuntimeSelection,
    dtype: torch.dtype,
    resolved_split_indices: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
) -> list[DatasetBundle | None]:
    """Finalize one fixed-schema raw chunk while preserving scalar retry semantics."""

    if int(x.shape[0]) != len(dataset_roots) or int(y.shape[0]) != len(dataset_roots):
        raise ValueError("Chunk tensors must align with provided dataset roots.")
    if contexts_by_batch is not None and len(contexts_by_batch) != len(dataset_roots):
        raise ValueError("contexts_by_batch must align with provided dataset roots.")
    if configs_by_batch is not None and len(configs_by_batch) != len(dataset_roots):
        raise ValueError("configs_by_batch must align with provided dataset roots.")
    if resolved_split_indices is not None and len(resolved_split_indices) != len(dataset_roots):
        raise ValueError("resolved_split_indices must align with provided dataset roots.")

    results: list[DatasetBundle | None] = [None] * len(dataset_roots)
    valid_positions: list[int] = []
    train_idx_list: list[torch.Tensor] = []
    test_idx_list: list[torch.Tensor] = []
    postprocess_roots: list[KeyedRng] = []
    for batch_index, dataset_root in enumerate(dataset_roots):
        if resolved_split_indices is not None:
            split_indices = resolved_split_indices[batch_index]
            if split_indices is None:
                continue
            train_idx, test_idx = split_indices
        else:
            attempt_root = dataset_root.keyed("attempt", attempt)
            try:
                train_idx, test_idx = _resolve_split_indices(
                    y[batch_index],
                    task=config.dataset.task,
                    n_train=n_train,
                    keyed_rng=attempt_root.keyed("split"),
                )
            except InfeasibleStratifiedSplitError:
                continue

        valid_positions.append(int(batch_index))
        train_idx_list.append(train_idx)
        test_idx_list.append(test_idx)
        postprocess_roots.append(dataset_root.keyed("attempt", attempt, "postprocess"))

    if not valid_positions:
        return results

    valid_index = torch.as_tensor(valid_positions, dtype=torch.long, device=x.device)
    x_valid = x.index_select(0, valid_index)
    y_valid = y.index_select(0, valid_index)
    train_idx = torch.stack([idx.to(device=x.device) for idx in train_idx_list])
    test_idx = torch.stack([idx.to(device=x.device) for idx in test_idx_list])

    x_train_t = torch.gather(
        x_valid,
        1,
        train_idx.unsqueeze(-1).expand(-1, -1, int(x.shape[2])),
    )
    x_test_t = torch.gather(
        x_valid,
        1,
        test_idx.unsqueeze(-1).expand(-1, -1, int(x.shape[2])),
    )
    y_train_t = torch.gather(y_valid, 1, train_idx)
    y_test_t = torch.gather(y_valid, 1, test_idx)

    x_train = x_train_t
    x_test = x_test_t
    y_train, y_test = postprocess_fixed_schema_target_batch(
        y_train_t,
        y_test_t,
        config.dataset.task,
        postprocess_roots=postprocess_roots,
    )
    for local_index, batch_index in enumerate(valid_positions):
        try:
            dataset_root = dataset_roots[batch_index]
            dataset_seed = dataset_root.child_seed()
            per_dataset_context = (
                context if contexts_by_batch is None else contexts_by_batch[batch_index]
            )
            per_dataset_config = (
                config if configs_by_batch is None else configs_by_batch[batch_index]
            )
            results[batch_index] = _finalize_processed_bundle(
                per_dataset_config,
                layout,
                context=per_dataset_context,
                dataset_seed=dataset_seed,
                attempt=attempt,
                attempts_used=attempts_used,
                attempt_root=dataset_root.keyed("attempt", attempt),
                device=device,
                requested_device=requested_device,
                resolved_device=resolved_device,
                device_fallback_reason=device_fallback_reason,
                x_train=x_train[local_index],
                y_train=y_train[local_index],
                x_test=x_test[local_index],
                y_test=y_test[local_index],
                aux_meta=aux_meta_batch[batch_index],
                noise_runtime_selection=noise_runtime_selection,
                dtype=dtype,
            )
        except InvalidClassSplitError:
            continue

    return results


def _finalize_generated_tensors(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    dataset_seed: int,
    attempt: int,
    attempts_used: int,
    dataset_root: KeyedRng,
    device: str,
    n_train: int,
    n_test: int,
    requested_device: str,
    resolved_device: str,
    device_fallback_reason: str | None,
    x: torch.Tensor,
    y: torch.Tensor,
    aux_meta: dict[str, Any],
    shift_params: ShiftRuntimeParams,
    noise_runtime_selection: NoiseRuntimeSelection,
    dtype: torch.dtype,
    preserve_feature_schema: bool = False,
    finalization_context: _FixedSchemaFinalizationContext | None = None,
) -> DatasetBundle:
    """Finalize one raw `x`/`y` pair into the standard dataset bundle contract."""

    attempt_root = dataset_root.keyed("attempt", attempt)
    try:
        train_idx, test_idx = _resolve_split_indices(
            y,
            task=config.dataset.task,
            n_train=n_train,
            keyed_rng=attempt_root.keyed("split"),
        )
    except InfeasibleStratifiedSplitError as exc:
        raise InvalidClassSplitError("invalid_class_split") from exc

    x_train_t, y_train_t, x_test_t, y_test_t = _split_raw_tensors(
        x,
        y,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    feature_types: list[str]
    feature_index_map: list[int]
    if preserve_feature_schema:
        x_train = x_train_t
        x_test = x_test_t
        y_train, y_test = postprocess_targets(
            y_train_t,
            y_test_t,
            config.dataset.task,
            keyed_rng=attempt_root.keyed("postprocess"),
        )
        feature_types = [str(feature_type) for feature_type in layout.feature_types]
        feature_index_map = [int(i) for i in range(int(x.shape[1]))]
    else:
        (
            x_train,
            y_train,
            x_test,
            y_test,
            postprocessed_feature_types,
            feature_index_map,
        ) = postprocess_dataset(
            x_train_t,
            y_train_t,
            x_test_t,
            y_test_t,
            list(layout.feature_types),
            config.dataset.task,
            attempt_root.keyed("postprocess"),
            device,
            return_feature_index_map=True,
            preserve_feature_schema=False,
        )
        feature_types = [str(feature_type) for feature_type in postprocessed_feature_types]
    if (
        finalization_context is not None
        and list(feature_types) == list(finalization_context.feature_types)
        and [int(i) for i in feature_index_map] == list(finalization_context.feature_index_map)
    ):
        context = finalization_context
    else:
        context = _build_fixed_schema_finalization_context(
            config,
            layout,
            n_train=n_train,
            n_test=n_test,
            shift_params=shift_params,
            feature_types=feature_types,
            feature_index_map=feature_index_map,
        )
    return _finalize_processed_bundle(
        config,
        layout,
        context=context,
        dataset_seed=dataset_seed,
        attempt=attempt,
        attempts_used=attempts_used,
        attempt_root=attempt_root,
        device=device,
        requested_device=requested_device,
        resolved_device=resolved_device,
        device_fallback_reason=device_fallback_reason,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        aux_meta=aux_meta,
        noise_runtime_selection=noise_runtime_selection,
        dtype=dtype,
    )
