"""Shared helpers for meta-feature target specs and diagnostics bands."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from dagzoo.config import clone_generator_config
from dagzoo.diagnostics import CoverageAggregationConfig

if TYPE_CHECKING:
    from dagzoo.config import DiagnosticsConfig, GeneratorConfig

TargetBand = tuple[float, float]


def coerce_target_bands(raw: object) -> dict[str, TargetBand]:
    """Normalize target band mappings into finite `(lo, hi)` tuples."""

    normalized: dict[str, TargetBand] = {}
    if not isinstance(raw, dict):
        return normalized
    for metric_name, value in raw.items():
        if not isinstance(metric_name, str):
            continue
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        try:
            lo = float(value[0])
            hi = float(value[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lo) and math.isfinite(hi)):
            continue
        normalized[metric_name] = (lo, hi) if lo <= hi else (hi, lo)
    return normalized


def merge_target_bands(*raw_values: object) -> dict[str, TargetBand]:
    """Merge target bands in order, where later entries win."""

    merged: dict[str, TargetBand] = {}
    for raw in raw_values:
        merged.update(coerce_target_bands(raw))
    return merged


def coerce_quantiles(raw: object) -> tuple[float, ...]:
    """Normalize quantile payload into finite float values."""

    if not isinstance(raw, (list, tuple)):
        return ()
    normalized: list[float] = []
    for item in raw:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            normalized.append(value)
    return tuple(normalized)


def build_diagnostics_aggregation_config(
    config: GeneratorConfig | DiagnosticsConfig,
) -> CoverageAggregationConfig:
    """Build diagnostics coverage aggregation config from diagnostics settings."""

    if hasattr(config, "diagnostics"):
        generator_config = cast("GeneratorConfig", config)
        diagnostics_config = generator_config.diagnostics
    else:
        generator_config = None
        diagnostics_config = cast("DiagnosticsConfig", config)

    return CoverageAggregationConfig(
        include_spearman=bool(diagnostics_config.include_spearman),
        histogram_bins=max(1, int(diagnostics_config.histogram_bins)),
        quantiles=coerce_quantiles(diagnostics_config.quantiles),
        underrepresented_threshold=float(diagnostics_config.underrepresented_threshold),
        max_values_per_metric=diagnostics_config.max_values_per_metric,
        target_bands=merge_target_bands(diagnostics_config.meta_feature_targets),
        steering_config=(
            clone_generator_config(generator_config, revalidate=False)
            if generator_config is not None
            else None
        ),
    )
