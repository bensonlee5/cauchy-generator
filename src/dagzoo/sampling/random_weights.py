"""Random positive weights sampling."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

from dagzoo.math import log_uniform as _log_uniform
from dagzoo.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec

_LOG_WEIGHT_CLAMP = 60.0


def _validate_random_weight_args(dim: int, *, sigma_multiplier: float) -> None:
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")
    if not math.isfinite(float(sigma_multiplier)) or float(sigma_multiplier) <= 0.0:
        raise ValueError(f"sigma_multiplier must be a finite value > 0, got {sigma_multiplier!r}")


def _random_weight_q_low(dim: int, *, min_q_scale: float) -> float:
    return float(min_q_scale) / math.log(float(dim) + 1.0)


def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in shape)


def _coerce_parameter_tensor(
    value: torch.Tensor | float,
    *,
    shape: tuple[int, ...],
    device: str,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.float32)
        if tuple(int(dim) for dim in tensor.shape) != shape:
            raise ValueError(
                f"{name} tensor shape must be {shape}, got {tuple(int(dim) for dim in tensor.shape)}"
            )
        return tensor
    return torch.full(shape, float(value), dtype=torch.float32, device=device)


def resolve_random_weight_parameters(
    dim: int,
    generator: torch.Generator,
    device: str,
    *,
    min_q_scale: float = 0.1,
    max_q: float = 6.0,
    sigma_min: float = 1e-4,
    sigma_max: float = 10.0,
    q: float | None = None,
    sigma: float | None = None,
    sigma_multiplier: float = 1.0,
) -> tuple[float, float]:
    """Resolve scalar ``q``/``sigma`` parameters for random-weight sampling."""

    _validate_random_weight_args(dim, sigma_multiplier=sigma_multiplier)
    resolved_q = (
        float(q)
        if q is not None
        else float(
            _log_uniform(
                generator,
                _random_weight_q_low(dim, min_q_scale=min_q_scale),
                max_q,
                device,
            )
        )
    )
    resolved_sigma = (
        float(sigma)
        if sigma is not None
        else float(_log_uniform(generator, sigma_min, sigma_max, device))
    )
    return resolved_q, resolved_sigma


def resolve_random_weight_parameter_tensors(
    *,
    dim: int,
    shape: Sequence[int],
    device: str,
    q: torch.Tensor | float | None = None,
    sigma: torch.Tensor | float | None = None,
    q_sampler: Callable[[tuple[int, ...]], torch.Tensor | float] | None = None,
    sigma_sampler: Callable[[tuple[int, ...]], torch.Tensor | float] | None = None,
    sigma_multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve tensor-shaped ``q``/``sigma`` parameters for random-weight sampling."""

    _validate_random_weight_args(dim, sigma_multiplier=sigma_multiplier)
    normalized_shape = _normalize_shape(shape)
    if q is None:
        if q_sampler is None:
            raise ValueError("q_sampler is required when q is not provided.")
        q_value = q_sampler(normalized_shape)
    else:
        q_value = q
    if sigma is None:
        if sigma_sampler is None:
            raise ValueError("sigma_sampler is required when sigma is not provided.")
        sigma_value = sigma_sampler(normalized_shape)
    else:
        sigma_value = sigma
    resolved_q = _coerce_parameter_tensor(
        q_value,
        shape=normalized_shape,
        device=device,
        name="q",
    )
    resolved_sigma = _coerce_parameter_tensor(
        sigma_value,
        shape=normalized_shape,
        device=device,
        name="sigma",
    )
    return resolved_q, resolved_sigma


def sample_random_weight_tensor(
    *,
    dim: int,
    device: str,
    noise_generator: torch.Generator,
    perm_generator: torch.Generator,
    leading_shape: Sequence[int] = (),
    parameter_shape: Sequence[int] | None = None,
    q: torch.Tensor | float | None = None,
    sigma: torch.Tensor | float | None = None,
    q_sampler: Callable[[tuple[int, ...]], torch.Tensor | float] | None = None,
    sigma_sampler: Callable[[tuple[int, ...]], torch.Tensor | float] | None = None,
    sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Sample positive normalized random weights for scalar or batched leading shapes."""

    _validate_random_weight_args(dim, sigma_multiplier=sigma_multiplier)
    normalized_leading_shape = _normalize_shape(leading_shape)
    normalized_parameter_shape = (
        normalized_leading_shape if parameter_shape is None else _normalize_shape(parameter_shape)
    )
    if len(normalized_parameter_shape) > len(normalized_leading_shape):
        raise ValueError("parameter_shape cannot be longer than leading_shape.")
    if normalized_leading_shape[: len(normalized_parameter_shape)] != normalized_parameter_shape:
        raise ValueError("parameter_shape must be a prefix of leading_shape.")

    resolved_q, resolved_sigma = resolve_random_weight_parameter_tensors(
        dim=dim,
        shape=normalized_parameter_shape,
        device=device,
        q=q,
        sigma=sigma,
        q_sampler=q_sampler,
        sigma_sampler=sigma_sampler,
        sigma_multiplier=sigma_multiplier,
    )

    full_shape = (*normalized_leading_shape, int(dim))
    base_noise = sample_noise_from_spec(
        full_shape,
        generator=noise_generator,
        device=device,
        noise_spec=noise_spec,
        scale_multiplier=float(sigma_multiplier),
    )
    broadcast_tail = len(normalized_leading_shape) - len(normalized_parameter_shape) + 1
    q_view = resolved_q.reshape(*normalized_parameter_shape, *([1] * broadcast_tail))
    sigma_view = resolved_sigma.reshape(*normalized_parameter_shape, *([1] * broadcast_tail))
    noise = base_noise * sigma_view
    ranks = torch.arange(1, dim + 1, dtype=torch.float32, device=device).view(
        *([1] * len(normalized_leading_shape)),
        dim,
    )
    log_w = (-q_view * torch.log(ranks)) + noise
    log_w = torch.nan_to_num(
        log_w,
        nan=0.0,
        posinf=_LOG_WEIGHT_CLAMP,
        neginf=-_LOG_WEIGHT_CLAMP,
    )
    log_w = torch.clamp(log_w, min=-_LOG_WEIGHT_CLAMP, max=_LOG_WEIGHT_CLAMP)
    log_w = log_w - torch.max(log_w, dim=-1, keepdim=True).values
    weights = torch.clamp(torch.exp(log_w), min=1e-12)
    weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-12)
    perm_scores = torch.empty(full_shape, device=device).uniform_(
        0.0,
        1.0,
        generator=perm_generator,
    )
    perm = torch.argsort(perm_scores, dim=-1)
    return torch.gather(weights, -1, perm)


def sample_random_weights(
    dim: int,
    generator: torch.Generator,
    device: str,
    *,
    min_q_scale: float = 0.1,
    max_q: float = 6.0,
    sigma_min: float = 1e-4,
    sigma_max: float = 10.0,
    q: float | None = None,
    sigma: float | None = None,
    sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Sample positive normalized weights using torch."""

    resolved_q, resolved_sigma = resolve_random_weight_parameters(
        dim,
        generator,
        device,
        min_q_scale=min_q_scale,
        max_q=max_q,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        q=q,
        sigma=sigma,
        sigma_multiplier=sigma_multiplier,
    )
    return sample_random_weight_tensor(
        dim=dim,
        device=device,
        noise_generator=generator,
        perm_generator=generator,
        q=resolved_q,
        sigma=resolved_sigma,
        sigma_multiplier=sigma_multiplier,
        noise_spec=noise_spec,
    )
