"""Seeded synthetic dataset generation public entrypoints."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from dagzoo.config import (
    INTERVENTION_MODE_OBSERVATIONAL,
    LAYOUT_MODE_FIXED,
    LAYOUT_MODE_STRATIFIED,
    GeneratorConfig,
)
from dagzoo.config.models import SteeringStageConfig, steering_stage_definitions
from dagzoo.core.fixed_layout.prepare import (
    normalize_fixed_layout_target_cells,
    realize_generation_config_for_run,
)
from dagzoo.core.fixed_layout.runtime import (
    CanonicalFixedLayoutRun,
    _generate_batch_with_heterogeneous_layout_iter,
    _generate_batch_with_plan_iter,
    _generate_batch_with_stratified_layout_iter,
)
from dagzoo.core.identity import (
    canonical_dataset_id,
    canonical_layout_plan_split_group,
    canonical_request_run_provenance,
    canonical_request_run_split_group,
    heterogeneous_cohort_split_group,
    heterogeneous_dataset_id,
    heterogeneous_request_run_split_group,
)
from dagzoo.core.shift import resolve_shift_runtime_params
from dagzoo.rng import KeyedRng
from dagzoo.types import DatasetBundle

__all__ = [
    "generate_batch",
    "generate_batch_iter",
    "generate_one",
]


def _validate_public_generation_config(config: GeneratorConfig) -> None:
    """Reject public generation configs that are intentionally unsupported."""

    if bool(config.filter.enabled):
        raise ValueError(
            "Inline filtering has been removed from generate. Set filter.enabled=false and run "
            "`dagzoo filter --in <shard_dir> --out <out_dir>` after generation. "
            "Generation still uses filter.min_target_* and filter.max_attempts while "
            "resampling structurally valid layouts."
        )
    if str(config.runtime.layout_mode) == str(LAYOUT_MODE_FIXED):
        raise ValueError(
            "Public `runtime.layout_mode: fixed` has been removed. Use "
            "`runtime.layout_mode: stratified` for throughput-sensitive heterogeneous runs."
        )


def _require_metadata_string(metadata: Mapping[str, Any], *, key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Canonical generation metadata is missing required string field {key!r}.")
    return value


def _request_run_provenance_for_config(
    config: GeneratorConfig,
    *,
    resolved_device: str,
) -> dict[str, Any]:
    noise_distribution = _noise_distribution_provenance_for_config(config)
    provenance_payload: dict[str, Any] = {
        "config": {
            "dataset": {
                "task": str(config.dataset.task),
                "n_train": int(config.dataset.n_train),
                "n_test": int(config.dataset.n_test),
                "missing_rate": float(config.dataset.missing_rate),
                "missing_mechanism": str(config.dataset.missing_mechanism),
                "missing_mar_observed_fraction": float(
                    config.dataset.missing_mar_observed_fraction
                ),
                "missing_mar_logit_scale": float(config.dataset.missing_mar_logit_scale),
                "missing_mnar_logit_scale": float(config.dataset.missing_mnar_logit_scale),
            },
            "runtime": {
                "fixed_layout_target_cells": config.runtime.fixed_layout_target_cells,
                "torch_dtype": str(config.runtime.torch_dtype),
            },
        },
        "noise_distribution": noise_distribution,
        "shift": {
            "variance_sigma_multiplier": float(
                resolve_shift_runtime_params(config).variance_sigma_multiplier
            )
        },
        "prior": {
            "target_derivation": "tabiclv2_latent_node",
        },
        "resolved_device": str(resolved_device),
        "compute_backend": "torch_appendix_full",
    }
    if str(config.intervention.mode) != INTERVENTION_MODE_OBSERVATIONAL:
        signature = config.intervention.signature
        if not isinstance(signature, str) or not signature:
            raise ValueError(
                "Hard-interventional request-run provenance requires intervention.signature."
            )
        provenance_payload["intervention"] = {
            "mode": str(config.intervention.mode),
            "signature": str(signature),
        }
    return canonical_request_run_provenance(provenance_payload)


def _noise_distribution_provenance_for_config(config: GeneratorConfig) -> dict[str, Any]:
    """Return normalized run-level noise provenance for one generation config."""

    noise_distribution: dict[str, Any] = {
        "family_requested": str(config.noise.family),
        "base_scale": float(config.noise.base_scale),
    }
    if str(config.noise.family) == "student_t":
        noise_distribution["student_t_df"] = float(config.noise.student_t_df)
    elif str(config.noise.family) == "mixture" and config.noise.mixture_weights is not None:
        noise_distribution["mixture_weights"] = {
            str(component): float(weight)
            for component, weight in sorted(config.noise.mixture_weights.items())
        }
    return noise_distribution


def _normalized_float_band(band: list[float] | None) -> list[float] | None:
    """Return a stable two-endpoint float band payload."""

    if band is None:
        return None
    return [float(band[0]), float(band[1])]


def _steering_stage_provenance(stage: SteeringStageConfig) -> dict[str, Any]:
    """Return stable request-run provenance for one steering stage definition."""

    payload: dict[str, Any] = {
        "name": str(stage.name),
        "fraction": float(stage.fraction),
    }
    if stage.missing_rate is not None:
        payload["missing_rate"] = _normalized_float_band(stage.missing_rate)
    if stage.missing_mechanism is not None:
        payload["missing_mechanism"] = str(stage.missing_mechanism)
    if stage.missing_mar_observed_fraction is not None:
        payload["missing_mar_observed_fraction"] = float(stage.missing_mar_observed_fraction)
    if stage.missing_mar_logit_scale is not None:
        payload["missing_mar_logit_scale"] = float(stage.missing_mar_logit_scale)
    if stage.missing_mnar_logit_scale is not None:
        payload["missing_mnar_logit_scale"] = float(stage.missing_mnar_logit_scale)
    if stage.shift_mode is not None:
        payload["shift_mode"] = str(stage.shift_mode)
    if stage.shift_graph_scale is not None:
        payload["shift_graph_scale"] = _normalized_float_band(stage.shift_graph_scale)
    if stage.shift_variance_scale is not None:
        payload["shift_variance_scale"] = _normalized_float_band(stage.shift_variance_scale)
    if stage.noise_family is not None:
        payload["noise_family"] = str(stage.noise_family)
    if stage.noise_student_t_df is not None:
        payload["noise_student_t_df"] = float(stage.noise_student_t_df)
    if stage.noise_mixture_weights is not None:
        payload["noise_mixture_weights"] = {
            str(component): _normalized_float_band(stage.noise_mixture_weights[component])
            for component in sorted(stage.noise_mixture_weights)
        }
    return payload


def _steering_provenance_for_config(config: GeneratorConfig) -> dict[str, Any]:
    """Return stable request-run provenance for dynamic steering authoring."""

    if not config.steering.enabled:
        return {"enabled": False}

    return {
        "enabled": True,
        "preset": config.steering.preset,
        "stages": [
            _steering_stage_provenance(stage)
            for stage in steering_stage_definitions(config.steering)
        ],
    }


def _heterogeneous_request_run_provenance_for_config(
    config: GeneratorConfig,
    *,
    resolved_device: str,
) -> dict[str, Any]:
    """Return the heterogeneous request-run provenance payload for one public run."""

    shift_params = resolve_shift_runtime_params(config)
    provenance = _request_run_provenance_for_config(config, resolved_device=resolved_device)
    provenance["dataset_structure"] = {
        "n_features_min": int(config.dataset.n_features_min),
        "n_features_max": int(config.dataset.n_features_max),
        "n_classes_min": int(config.dataset.n_classes_min),
        "n_classes_max": int(config.dataset.n_classes_max),
        "categorical_ratio_min": float(config.dataset.categorical_ratio_min),
        "categorical_ratio_max": float(config.dataset.categorical_ratio_max),
        "max_categorical_cardinality": int(config.dataset.max_categorical_cardinality),
    }
    provenance["graph"] = {
        "n_nodes_min": int(config.graph.n_nodes_min),
        "n_nodes_max": int(config.graph.n_nodes_max),
        "target_depth_nodes_min": (
            None
            if config.graph.target_depth_nodes_min is None
            else int(config.graph.target_depth_nodes_min)
        ),
        "target_depth_nodes_max": (
            None
            if config.graph.target_depth_nodes_max is None
            else int(config.graph.target_depth_nodes_max)
        ),
    }
    provenance["mechanism"] = {
        "function_family_mix": (
            None
            if config.mechanism.function_family_mix is None
            else {
                str(family): float(weight)
                for family, weight in sorted(config.mechanism.function_family_mix.items())
            }
        )
    }
    provenance["shift"] = {
        **dict(provenance["shift"]),
        "enabled": bool(shift_params.enabled),
        "mode": str(shift_params.mode),
        "graph_scale": float(shift_params.graph_scale),
        "mechanism_scale": float(shift_params.mechanism_scale),
        "variance_scale": float(shift_params.variance_scale),
        "edge_logit_bias_shift": float(shift_params.edge_logit_bias_shift),
        "mechanism_logit_tilt": float(shift_params.mechanism_logit_tilt),
    }
    provenance["steering"] = _steering_provenance_for_config(config)
    provenance["runtime"] = {
        **dict(provenance["runtime"]),
        "fixed_layout_target_cells": normalize_fixed_layout_target_cells(
            config.runtime.fixed_layout_target_cells
        ),
    }
    return provenance


def _heterogeneous_cohort_payload(bundle: DatasetBundle) -> dict[str, Any]:
    metadata = bundle.metadata
    config_payload = metadata.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("Heterogeneous bundle is missing metadata.config.")
    dataset_payload = config_payload.get("dataset")
    if not isinstance(dataset_payload, Mapping):
        raise ValueError("Heterogeneous bundle is missing metadata.config.dataset.")
    noise_distribution = metadata.get("noise_distribution")
    if not isinstance(noise_distribution, Mapping):
        raise ValueError("Heterogeneous bundle is missing metadata.noise_distribution.")
    shift_payload = metadata.get("shift")
    if not isinstance(shift_payload, Mapping):
        raise ValueError("Heterogeneous bundle is missing metadata.shift.")
    mixture_weights_raw = noise_distribution.get("mixture_weights")
    mixture_weights = (
        None
        if not isinstance(mixture_weights_raw, Mapping)
        else {
            str(component): float(mixture_weights_raw[component])
            for component in sorted(mixture_weights_raw)
        }
    )
    return {
        "layout_signature": _require_metadata_string(metadata, key="layout_signature"),
        "layout_plan_signature": _require_metadata_string(metadata, key="layout_plan_signature"),
        "layout_execution_contract": _require_metadata_string(
            metadata,
            key="layout_execution_contract",
        ),
        "task": str(dataset_payload["task"]),
        "n_train": int(dataset_payload["n_train"]),
        "n_test": int(dataset_payload["n_test"]),
        "noise": {
            "family_requested": str(noise_distribution["family_requested"]),
            "base_scale": float(noise_distribution["base_scale"]),
            "student_t_df": float(noise_distribution.get("student_t_df", 0.0)),
            "mixture_weights": mixture_weights,
        },
        "shift": {
            "variance_sigma_multiplier": float(shift_payload["variance_sigma_multiplier"]),
        },
    }


def _annotate_canonical_batch_metadata(
    bundle: DatasetBundle,
    *,
    run_seed: int,
    dataset_index: int,
    run_num_datasets: int,
    request_run_provenance: Mapping[str, Any],
) -> DatasetBundle:
    """Rewrite canonical bundle metadata to preserve run-level replay information."""

    dataset_seed = bundle.metadata.get("seed")
    if not isinstance(dataset_seed, int) or isinstance(dataset_seed, bool):
        dataset_seed = KeyedRng(int(run_seed)).child_seed("dataset", int(dataset_index))
    keyed_replay = bundle.metadata.get("keyed_replay")
    if not isinstance(keyed_replay, dict):
        keyed_replay = {}
    keyed_replay["dataset_root_path"] = ["dataset", int(dataset_index)]
    layout_signature = _require_metadata_string(bundle.metadata, key="layout_signature")
    layout_plan_signature = _require_metadata_string(bundle.metadata, key="layout_plan_signature")
    layout_execution_contract = _require_metadata_string(
        bundle.metadata,
        key="layout_execution_contract",
    )
    split_groups = {
        "request_run": canonical_request_run_split_group(
            seed=int(run_seed),
            run_num_datasets=int(run_num_datasets),
            layout_signature=layout_signature,
            layout_plan_signature=layout_plan_signature,
            request_run_provenance=dict(request_run_provenance),
        ),
        "layout_plan": canonical_layout_plan_split_group(
            layout_signature=layout_signature,
            layout_plan_signature=layout_plan_signature,
            layout_execution_contract=layout_execution_contract,
        ),
    }
    bundle.metadata["seed"] = int(run_seed)
    bundle.metadata["dataset_seed"] = int(dataset_seed)
    bundle.metadata["dataset_index"] = int(dataset_index)
    bundle.metadata["dataset_id"] = canonical_dataset_id(
        request_run_split_group=split_groups["request_run"],
        layout_plan_split_group=split_groups["layout_plan"],
        dataset_index=int(dataset_index),
        dataset_seed=int(dataset_seed),
    )
    bundle.metadata["run_num_datasets"] = int(run_num_datasets)
    bundle.metadata["split_groups"] = split_groups
    bundle.metadata["keyed_replay"] = keyed_replay
    return bundle


def _annotate_heterogeneous_batch_metadata(
    bundle: DatasetBundle,
    *,
    run_seed: int,
    dataset_index: int,
    run_num_datasets: int,
    request_run_split_group: str,
) -> DatasetBundle:
    """Rewrite heterogeneous bundle metadata to preserve run-level replay information."""

    dataset_seed = bundle.metadata.get("seed")
    if not isinstance(dataset_seed, int) or isinstance(dataset_seed, bool):
        dataset_seed = KeyedRng(int(run_seed)).child_seed("dataset", int(dataset_index))
    keyed_replay = bundle.metadata.get("keyed_replay")
    if not isinstance(keyed_replay, dict):
        keyed_replay = {}
    keyed_replay["dataset_root_path"] = ["dataset", int(dataset_index)]
    cohort_split_group = heterogeneous_cohort_split_group(
        cohort_payload=_heterogeneous_cohort_payload(bundle)
    )
    split_groups = {
        "request_run": str(request_run_split_group),
        "cohort": str(cohort_split_group),
    }
    bundle.metadata["seed"] = int(run_seed)
    bundle.metadata["dataset_seed"] = int(dataset_seed)
    bundle.metadata["dataset_index"] = int(dataset_index)
    bundle.metadata["dataset_id"] = heterogeneous_dataset_id(
        request_run_split_group=str(request_run_split_group),
        cohort_split_group=str(cohort_split_group),
        dataset_index=int(dataset_index),
        dataset_seed=int(dataset_seed),
    )
    bundle.metadata["run_num_datasets"] = int(run_num_datasets)
    bundle.metadata["split_groups"] = split_groups
    bundle.metadata["keyed_replay"] = keyed_replay
    return bundle


def generate_one(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> DatasetBundle:
    """Generate one dataset bundle using the configured public generation mode."""

    return next(generate_batch_iter(config, num_datasets=1, seed=seed, device=device))


def generate_batch(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
) -> list[DatasetBundle]:
    """Generate a batch of datasets using deterministic per-dataset child seeds."""

    return list(
        generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        )
    )


def generate_batch_iter(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets from one public generation run."""

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return
    _validate_public_generation_config(config)

    realized_config, run_seed, _requested_device, resolved_device, _carried_stress_profile = (
        realize_generation_config_for_run(
            config,
            seed=seed,
            device=device,
            prefer_cpu_for_mps_auto=True,
        )
    )
    request_run_split_group = heterogeneous_request_run_split_group(
        seed=int(run_seed),
        run_num_datasets=int(num_datasets),
        request_run_provenance=_heterogeneous_request_run_provenance_for_config(
            realized_config,
            resolved_device=resolved_device,
        ),
    )
    for dataset_index, bundle in enumerate(
        (
            _generate_batch_with_stratified_layout_iter(
                config,
                num_datasets=num_datasets,
                seed=seed,
                device=device,
            )
            if str(config.runtime.layout_mode) == str(LAYOUT_MODE_STRATIFIED)
            else _generate_batch_with_heterogeneous_layout_iter(
                config,
                num_datasets=num_datasets,
                seed=seed,
                device=device,
            )
        )
    ):
        yield _annotate_heterogeneous_batch_metadata(
            bundle,
            run_seed=int(run_seed),
            dataset_index=dataset_index,
            run_num_datasets=num_datasets,
            request_run_split_group=request_run_split_group,
        )


def _iter_prepared_canonical_batch_iter(
    prepared: CanonicalFixedLayoutRun,
    *,
    num_datasets: int,
    on_raw_batch_metrics: Callable[[dict[str, float]], None] | None = None,
) -> Iterator[DatasetBundle]:
    """Yield annotated bundles from one already-prepared canonical fixed-layout run."""

    request_run_provenance = _request_run_provenance_for_config(
        prepared.config,
        resolved_device=prepared.resolved_device,
    )

    for dataset_index, bundle in enumerate(
        _generate_batch_with_plan_iter(
            prepared.config,
            plan=prepared.plan,
            num_datasets=num_datasets,
            seed=prepared.run_seed,
            batch_size=prepared.batch_size,
            classification_attempt_plan=prepared.classification_attempt_plan,
            on_raw_batch_metrics=on_raw_batch_metrics,
        )
    ):
        yield _annotate_canonical_batch_metadata(
            bundle,
            run_seed=prepared.run_seed,
            dataset_index=dataset_index,
            run_num_datasets=num_datasets,
            request_run_provenance=request_run_provenance,
        )
