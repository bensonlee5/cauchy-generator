"""Shared math and random-matrix utilities."""

from .utils import (
    coerce_optional_finite_float,
    log_uniform,
    normalize_positive_weights,
    row_normalize,
    sanitize_and_standardize,
    sanitize_json,
    standardize,
    to_numpy,
)

__all__ = [
    "coerce_optional_finite_float",
    "log_uniform",
    "normalize_positive_weights",
    "row_normalize",
    "sanitize_and_standardize",
    "sanitize_json",
    "standardize",
    "to_numpy",
]
