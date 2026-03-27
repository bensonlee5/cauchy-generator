"""Shared config state predicates used across runtime and benchmark code."""

from __future__ import annotations

from dagzoo.config import (
    MISSINGNESS_MECHANISM_NONE,
    NOISE_FAMILY_GAUSSIAN,
    GeneratorConfig,
)


def missingness_enabled(config: GeneratorConfig) -> bool:
    """Return whether missingness injection can mutate emitted tensors."""

    return bool(
        float(config.dataset.missing_rate) > 0.0
        and str(config.dataset.missing_mechanism).strip().lower() != MISSINGNESS_MECHANISM_NONE
    )


def shift_enabled(config: GeneratorConfig) -> bool:
    """Return whether shift controls are enabled in config."""

    return bool(config.shift.enabled)


def non_gaussian_noise_enabled(config: GeneratorConfig) -> bool:
    """Return whether non-Gaussian noise controls are enabled in config."""

    return str(config.noise.family).strip().lower() != NOISE_FAMILY_GAUSSIAN
