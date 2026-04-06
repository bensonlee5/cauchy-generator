"""Grouped raw-batch helpers for fixed-layout generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.core.noise_runtime import NoiseRuntimeSelection
from dagzoo.rng import KeyedRng
from dagzoo.sampling.noise import NoiseSamplingSpec

from .plan_types import FixedLayoutExecutionPlan


@dataclass(slots=True)
class _NoiseRuntimeGroup:
    """One chunk subgroup that shares the same raw noise sampling contract."""

    chunk_offsets: list[int]
    generation_seeds: list[int]
    selection: NoiseRuntimeSelection
    attempt: int


@dataclass(slots=True)
class _GroupedRawBatch:
    """One raw fixed-layout subgroup generation result."""

    chunk_offsets: list[int]
    selection: NoiseRuntimeSelection
    attempt: int
    x_batch: torch.Tensor
    y_batch: torch.Tensor
    aux_meta_batch: list[dict[str, Any]]
    effective_resolved_device: str
    device_fallback_reason: str | None
    runtime_metrics: dict[str, float]


def _noise_runtime_selection_key(
    selection: NoiseRuntimeSelection,
) -> tuple[str, float, float]:
    return (
        str(selection.family_sampled),
        float(selection.base_scale),
        float(selection.student_t_df),
    )


def group_noise_runtime_chunk(
    config: GeneratorConfig,
    *,
    dataset_roots: list[KeyedRng],
    attempts: list[int] | None = None,
    resolve_noise_runtime_selection: Callable[..., NoiseRuntimeSelection],
) -> list[_NoiseRuntimeGroup]:
    """Group datasets that share the same sampled noise runtime selection."""

    effective_attempts = attempts or [0] * len(dataset_roots)
    if len(effective_attempts) != len(dataset_roots):
        raise ValueError(
            "Fixed-layout attempt plan length must match chunk dataset count: "
            f"dataset_roots={len(dataset_roots)} attempts={len(effective_attempts)}"
        )

    grouped: dict[tuple[str, float, float, int], _NoiseRuntimeGroup] = {}
    ordered_keys: list[tuple[str, float, float, int]] = []
    for chunk_offset, (dataset_root, attempt) in enumerate(
        zip(dataset_roots, effective_attempts, strict=True)
    ):
        if attempt < 0:
            raise ValueError(f"Fixed-layout attempt plan entries must be >= 0, got {attempt}.")
        selection = resolve_noise_runtime_selection(
            config,
            keyed_rng=dataset_root.keyed("noise_runtime"),
        )
        generation_seed = dataset_root.keyed("attempt", int(attempt), "raw_generation").child_seed()
        key = _noise_runtime_selection_key(selection) + (int(attempt),)
        if key not in grouped:
            grouped[key] = _NoiseRuntimeGroup(
                chunk_offsets=[],
                generation_seeds=[],
                selection=selection,
                attempt=int(attempt),
            )
            ordered_keys.append(key)
        group = grouped[key]
        group.chunk_offsets.append(int(chunk_offset))
        group.generation_seeds.append(int(generation_seed))
    return [grouped[key] for key in ordered_keys]


def generate_grouped_raw_batches(
    config: GeneratorConfig,
    layout: LayoutPlan,
    *,
    execution_plan: FixedLayoutExecutionPlan,
    intervention_plan: Any | None = None,
    grouped_noise_runtime: list[_NoiseRuntimeGroup],
    resolved_device: str,
    noise_sigma_multiplier: float,
    noise_sampling_spec: Callable[[NoiseRuntimeSelection], NoiseSamplingSpec | None],
    generate_graph_batch: Callable[..., tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]],
) -> list[_GroupedRawBatch]:
    """Generate raw grouped batches for one fixed-layout chunk."""

    grouped_batches: list[_GroupedRawBatch] = []
    for group in grouped_noise_runtime:
        noise_spec = noise_sampling_spec(group.selection)
        runtime_metrics: dict[str, float] = {}
        x_batch, y_batch, aux_meta_batch = generate_graph_batch(
            config,
            layout,
            execution_plan=execution_plan,
            intervention_plan=intervention_plan,
            dataset_seeds=group.generation_seeds,
            device=resolved_device,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
            runtime_metrics_out=runtime_metrics,
        )
        grouped_batches.append(
            _GroupedRawBatch(
                chunk_offsets=list(group.chunk_offsets),
                selection=group.selection,
                attempt=int(group.attempt),
                x_batch=x_batch,
                y_batch=y_batch,
                aux_meta_batch=aux_meta_batch,
                effective_resolved_device=str(resolved_device),
                device_fallback_reason=None,
                runtime_metrics=dict(runtime_metrics),
            )
        )
    return grouped_batches


__all__ = [
    "_GroupedRawBatch",
    "_NoiseRuntimeGroup",
    "generate_grouped_raw_batches",
    "group_noise_runtime_chunk",
]
