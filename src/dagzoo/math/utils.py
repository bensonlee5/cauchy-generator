"""Shared math utilities used across the generation pipeline."""

from __future__ import annotations

import math

import numpy as np
import torch


def normalize_positive_weights[KT: str](
    weights: dict[KT, float],
    *,
    field_name: str = "weights",
) -> dict[KT, float]:
    """Numerically stable normalization: filter positive, scale-by-max, fsum, normalize."""

    positive = {k: v for k, v in weights.items() if v > 0.0}
    if not positive:
        raise ValueError(f"{field_name} must have a positive total weight.")

    max_weight = max(positive.values())
    scaled = {k: v / max_weight for k, v in positive.items()}
    total = float(math.fsum(scaled.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{field_name} must have a positive total weight.")

    return {k: v / total for k, v in scaled.items()}


def row_normalize(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Normalize nonzero rows exactly to unit norm while leaving zero rows at zero."""

    norms = torch.linalg.norm(x, dim=dim, keepdim=True)
    nonzero = norms > 0.0
    safe_norms = torch.where(nonzero, norms, torch.ones_like(norms))
    normalized = x / safe_norms
    return torch.where(nonzero, normalized, torch.zeros_like(normalized))


def log_uniform(generator: torch.Generator, low: float, high: float, device: str) -> float:
    """Sample from a log-uniform distribution using torch."""
    low_log = math.log(low)
    high_log = math.log(high)
    u = torch.empty(1, device=device).uniform_(low_log, high_log, generator=generator)
    return float(torch.exp(u).item())


def standardize(x: torch.Tensor) -> torch.Tensor:
    """Standardize columns with epsilon-protected standard deviations in torch."""
    mu = torch.mean(x, dim=0, keepdim=True)
    sigma = torch.std(x, dim=0, keepdim=True)
    return (x - mu) / torch.clamp(sigma, min=1e-6)


def sanitize_and_standardize(x: torch.Tensor) -> torch.Tensor:
    """Clip non-finite/extreme values before column standardization."""
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    x = torch.clamp(x, -1e6, 1e6)
    return standardize(x)


def to_numpy(value: object) -> np.ndarray:
    """Convert tensors or array-like values to NumPy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sanitize_json(value: object) -> object:
    """Recursively sanitize metadata for strict JSON serialization."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v) for v in value]
    return value


def coerce_optional_finite_float(value: object) -> float | None:
    """Return a finite float for numeric scalar inputs; otherwise return None."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    if not math.isfinite(as_float):
        return None
    return as_float
