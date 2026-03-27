"""Dynamic steering resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass

from dagzoo.config import (
    NOISE_MIXTURE_COMPONENT_GAUSSIAN,
    NOISE_MIXTURE_COMPONENT_LAPLACE,
    NOISE_MIXTURE_COMPONENT_STUDENT_T,
    SHIFT_MODE_CUSTOM,
    SHIFT_MODE_GRAPH_DRIFT,
    SHIFT_MODE_MIXED,
    SHIFT_MODE_NOISE_DRIFT,
    GeneratorConfig,
    SteeringConfig,
    StressConfig,
    clone_generator_config,
)
from dagzoo.config.models import SteeringStageConfig, steering_stage_definitions

_STEERING_MIXTURE_COMPONENTS = (
    NOISE_MIXTURE_COMPONENT_GAUSSIAN,
    NOISE_MIXTURE_COMPONENT_LAPLACE,
    NOISE_MIXTURE_COMPONENT_STUDENT_T,
)


@dataclass(slots=True, frozen=True)
class SteeringResolution:
    """Resolved per-dataset steering state."""

    config: GeneratorConfig
    dataset_index: int
    run_num_datasets: int
    progress: float
    stage_index: int | None
    stage_name: str | None
    stage_progress: float | None


def _interpolate_band(band: list[float], t: float) -> float:
    start = float(band[0])
    end = float(band[1])
    return float(start + ((end - start) * t))


def _global_progress(*, dataset_index: int, run_num_datasets: int) -> float:
    if run_num_datasets <= 1:
        return 0.0
    return float(max(0.0, min(1.0, float(dataset_index) / float(run_num_datasets - 1))))


def _resolve_active_stage(
    stages: list[SteeringStageConfig],
    *,
    progress: float,
) -> tuple[int, SteeringStageConfig, float]:
    cumulative = 0.0
    last_index = len(stages) - 1
    for idx, stage in enumerate(stages):
        start = cumulative
        end = cumulative + float(stage.fraction)
        cumulative = end
        if progress < end or idx == last_index:
            width = max(float(stage.fraction), 1e-12)
            local_progress = float(max(0.0, min(1.0, (progress - start) / width)))
            return int(idx), stage, local_progress
    raise RuntimeError("Failed to resolve active steering stage.")


def _apply_stage_to_config(
    config: GeneratorConfig,
    stage: SteeringStageConfig,
    *,
    t: float,
) -> None:
    if stage.missing_rate is not None:
        config.dataset.missing_rate = _interpolate_band(stage.missing_rate, t)
    if stage.missing_mechanism is not None:
        config.dataset.missing_mechanism = stage.missing_mechanism
    if stage.missing_mar_observed_fraction is not None:
        config.dataset.missing_mar_observed_fraction = float(stage.missing_mar_observed_fraction)
    if stage.missing_mar_logit_scale is not None:
        config.dataset.missing_mar_logit_scale = float(stage.missing_mar_logit_scale)
    if stage.missing_mnar_logit_scale is not None:
        config.dataset.missing_mnar_logit_scale = float(stage.missing_mnar_logit_scale)

    if (
        stage.shift_mode is not None
        or stage.shift_graph_scale is not None
        or stage.shift_variance_scale is not None
    ):
        config.shift.enabled = True
        if stage.shift_mode is not None:
            config.shift.mode = stage.shift_mode
        shift_mode = str(config.shift.mode)
        # Steering stages do not author mechanism drift, so always discard any
        # inherited mechanism-scale override before applying graph/noise ramps.
        config.shift.mechanism_scale = None
        if shift_mode == SHIFT_MODE_GRAPH_DRIFT:
            config.shift.variance_scale = None
        elif shift_mode == SHIFT_MODE_NOISE_DRIFT:
            config.shift.graph_scale = None
        elif shift_mode in {SHIFT_MODE_MIXED, SHIFT_MODE_CUSTOM}:
            config.shift.mechanism_scale = 0.0
        if stage.shift_graph_scale is not None:
            config.shift.graph_scale = _interpolate_band(stage.shift_graph_scale, t)
        if stage.shift_variance_scale is not None:
            config.shift.variance_scale = _interpolate_band(stage.shift_variance_scale, t)

    if (
        stage.noise_family is not None
        or stage.noise_student_t_df is not None
        or stage.noise_mixture_weights is not None
    ):
        if stage.noise_family is not None:
            config.noise.family = stage.noise_family
        if stage.noise_student_t_df is not None:
            config.noise.student_t_df = float(stage.noise_student_t_df)
        if stage.noise_mixture_weights is not None:
            resolved_weights = {
                component: (
                    _interpolate_band(stage.noise_mixture_weights[component], t)
                    if component in stage.noise_mixture_weights
                    else 0.0
                )
                for component in _STEERING_MIXTURE_COMPONENTS
            }
            total = float(sum(resolved_weights.values()))
            if total > 0.0:
                resolved_weights = {
                    component: (float(weight) / total)
                    for component, weight in resolved_weights.items()
                }
            config.noise.mixture_weights = resolved_weights


def resolve_steering(
    config: GeneratorConfig,
    *,
    dataset_index: int,
    run_num_datasets: int,
) -> SteeringResolution:
    """Resolve dynamic steering into one effective config for one dataset ordinal."""

    effective = clone_generator_config(config, revalidate=False)
    if not config.steering.enabled:
        return SteeringResolution(
            config=effective,
            dataset_index=int(dataset_index),
            run_num_datasets=int(run_num_datasets),
            progress=_global_progress(
                dataset_index=dataset_index, run_num_datasets=run_num_datasets
            ),
            stage_index=None,
            stage_name=None,
            stage_progress=None,
        )

    stages = steering_stage_definitions(config.steering)
    progress = _global_progress(dataset_index=dataset_index, run_num_datasets=run_num_datasets)
    active_stage_index, active_stage, active_stage_progress = _resolve_active_stage(
        stages,
        progress=progress,
    )

    for stage_index, stage in enumerate(stages):
        if stage_index < active_stage_index:
            _apply_stage_to_config(effective, stage, t=1.0)
            continue
        if stage_index == active_stage_index:
            _apply_stage_to_config(effective, stage, t=active_stage_progress)
        break

    effective.steering = SteeringConfig()
    effective.stress = StressConfig()
    effective.validate_generation_constraints()
    return SteeringResolution(
        config=effective,
        dataset_index=int(dataset_index),
        run_num_datasets=int(run_num_datasets),
        progress=float(progress),
        stage_index=int(active_stage_index),
        stage_name=str(active_stage.name),
        stage_progress=float(active_stage_progress),
    )


__all__ = ["SteeringResolution", "resolve_steering"]
