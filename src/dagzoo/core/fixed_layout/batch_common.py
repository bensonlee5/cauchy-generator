"""Shared fixed-layout batched tensor and RNG helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from dagzoo.core.layout_types import AggregationKind
from dagzoo.math import row_normalize
from dagzoo.rng import KeyedRng
from dagzoo.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec

_PAIRWISE_CENTER_BLOCK_SIZE = 32


def _batch_standardize(x: torch.Tensor) -> torch.Tensor:
    correction = 1 if int(x.shape[1]) > 1 else 0
    var, mean = torch.var_mean(x, dim=1, keepdim=True, correction=correction)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    return (x - mean) / torch.clamp(std, min=1e-6)


def _sanitize_and_batch_standardize(x: torch.Tensor) -> torch.Tensor:
    y = torch.nan_to_num(x.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    y = torch.clamp(y, -1e6, 1e6)
    return _batch_standardize(y)


def _aggregate_batch_incrementally(
    aggregate: torch.Tensor,
    transformed: torch.Tensor,
    *,
    aggregation_kind: AggregationKind,
) -> torch.Tensor:
    if aggregation_kind == "sum":
        return aggregate + transformed
    if aggregation_kind == "product":
        return aggregate * transformed
    if aggregation_kind == "max":
        return torch.maximum(aggregate, transformed)
    raise ValueError(f"Unknown aggregation kind: {aggregation_kind!r}")


def _aggregate_parent_outputs_batch(
    stacked: torch.Tensor,
    *,
    aggregation_kind: AggregationKind,
) -> torch.Tensor:
    if aggregation_kind == "sum":
        return torch.sum(stacked, dim=2)
    if aggregation_kind == "product":
        return torch.prod(stacked, dim=2)
    if aggregation_kind == "max":
        return torch.max(stacked, dim=2).values
    if aggregation_kind == "logsumexp":
        return torch.logsumexp(stacked, dim=2)
    raise ValueError(f"Unknown aggregation kind: {aggregation_kind!r}")


def _flatten_leading_dims(
    x: torch.Tensor, *, trailing_dims: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    leading_shape = tuple(int(dim) for dim in x.shape[:-trailing_dims])
    flat = max(1, math.prod(leading_shape))
    return x.reshape(flat, *x.shape[-trailing_dims:]), leading_shape


def _nearest_lp_center_indices(
    x: torch.Tensor,
    centers: torch.Tensor,
    *,
    p: torch.Tensor,
    block_size: int = _PAIRWISE_CENTER_BLOCK_SIZE,
) -> torch.Tensor:
    x_flat, leading_shape = _flatten_leading_dims(x, trailing_dims=2)
    centers_flat, _ = _flatten_leading_dims(centers, trailing_dims=2)
    p_flat = p.reshape(-1)

    best_distance: torch.Tensor | None = None
    best_indices: torch.Tensor | None = None
    for start in range(0, int(centers_flat.shape[1]), max(1, int(block_size))):
        stop = min(start + max(1, int(block_size)), int(centers_flat.shape[1]))
        center_block = centers_flat[:, start:stop, :]
        diff = torch.abs(x_flat.unsqueeze(2) - center_block.unsqueeze(1))
        block_distances = torch.pow(diff, p_flat.view(-1, 1, 1, 1)).sum(dim=3)
        local_distance, local_indices = torch.min(block_distances, dim=2)
        local_indices = local_indices + int(start)
        if best_distance is None or best_indices is None:
            best_distance = local_distance
            best_indices = local_indices
            continue
        use_local = local_distance < best_distance
        best_distance = torch.where(use_local, local_distance, best_distance)
        best_indices = torch.where(use_local, local_indices, best_indices)

    if best_indices is None:
        raise RuntimeError("Expected at least one center when computing nearest center indices.")
    return best_indices.reshape(*leading_shape, int(x.shape[-2]))


def _lp_distances_to_centers(
    x: torch.Tensor,
    centers: torch.Tensor,
    *,
    p: torch.Tensor,
    take_root: bool,
    block_size: int = _PAIRWISE_CENTER_BLOCK_SIZE,
) -> torch.Tensor:
    x_flat, leading_shape = _flatten_leading_dims(x, trailing_dims=2)
    centers_flat, _ = _flatten_leading_dims(centers, trailing_dims=2)
    p_flat = p.reshape(-1)
    blocks: list[torch.Tensor] = []
    for start in range(0, int(centers_flat.shape[1]), max(1, int(block_size))):
        stop = min(start + max(1, int(block_size)), int(centers_flat.shape[1]))
        center_block = centers_flat[:, start:stop, :]
        diff = torch.abs(x_flat.unsqueeze(2) - center_block.unsqueeze(1))
        block_distances = torch.pow(diff, p_flat.view(-1, 1, 1, 1)).sum(dim=3)
        if take_root:
            block_distances = torch.pow(
                block_distances,
                p_flat.reciprocal().view(-1, 1, 1),
            )
        blocks.append(block_distances)
    if not blocks:
        raise RuntimeError("Expected at least one center block when computing distances.")
    distances = torch.cat(blocks, dim=2)
    return distances.reshape(*leading_shape, int(x.shape[-2]), int(centers.shape[-2]))


@dataclass(slots=True)
class FixedLayoutBatchRng:
    """Chunk-scoped RNG for fixed-layout batched generation."""

    seed: int | None
    batch_size: int
    device: str
    generator: torch.Generator | None = field(default=None, repr=False)
    keyed_root: KeyedRng | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.keyed_root is None and self.seed is not None:
            self.keyed_root = KeyedRng(int(self.seed))
        if self.generator is None:
            if self.seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(self.seed))
                self.generator = generator
            elif self.keyed_root is not None:
                self.generator = self.keyed_root.torch_rng(device=self.device)
            else:
                raise ValueError("FixedLayoutBatchRng requires either seed or generator.")

    @classmethod
    def from_generator(
        cls,
        generator: torch.Generator,
        *,
        batch_size: int,
        device: str,
    ) -> FixedLayoutBatchRng:
        return cls(
            seed=None,
            batch_size=int(batch_size),
            device=device,
            generator=generator,
        )

    @classmethod
    def from_keyed_rng(
        cls,
        keyed_rng: KeyedRng,
        *,
        batch_size: int,
        device: str,
    ) -> FixedLayoutBatchRng:
        return cls(
            seed=None,
            batch_size=int(batch_size),
            device=device,
            keyed_root=keyed_rng,
        )

    @property
    def torch_generator(self) -> torch.Generator:
        if self.generator is None:
            raise RuntimeError("FixedLayoutBatchRng generator was not initialized.")
        return self.generator

    def keyed(self, *components: str | int) -> FixedLayoutBatchRng:
        if self.keyed_root is None:
            return self
        return FixedLayoutBatchRng.from_keyed_rng(
            self.keyed_root.keyed(*components),
            batch_size=self.batch_size,
            device=self.device,
        )

    def normal(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=self.generator, device=self.device)

    def uniform(self, shape: tuple[int, ...], *, low: float, high: float) -> torch.Tensor:
        return torch.empty(shape, device=self.device).uniform_(
            float(low), float(high), generator=self.generator
        )

    def randint(self, low: int, high: int, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(
            int(low),
            int(high),
            shape,
            generator=self.generator,
            device=self.device,
        )

    def log_uniform(self, shape: tuple[int, ...], *, low: float, high: float) -> torch.Tensor:
        samples = self.uniform(
            shape,
            low=math.log(float(low)),
            high=math.log(float(high)),
        )
        return torch.exp(samples)

    def randperm_indices(
        self,
        *,
        length: int,
        sample_size: int,
        leading_shape: tuple[int, ...] = (),
    ) -> torch.Tensor:
        scores = self.uniform(
            (self.batch_size, *leading_shape, int(length)),
            low=0.0,
            high=1.0,
        )
        return torch.topk(
            scores,
            k=int(sample_size),
            dim=-1,
            largest=False,
            sorted=True,
        ).indices.to(torch.long)

    def categorical(self, probs: torch.Tensor) -> torch.Tensor:
        flat = probs.reshape(-1, probs.shape[-1])
        sampled = torch.multinomial(flat, 1, generator=self.generator).squeeze(1)
        return sampled.reshape(probs.shape[:-1]).to(torch.long)


def _row_normalize_batch(matrix: torch.Tensor) -> torch.Tensor:
    return row_normalize(matrix, dim=-1)


def _sample_random_weights_batch(
    rng: FixedLayoutBatchRng,
    *,
    dim: int,
    leading_shape: tuple[int, ...] = (),
    parameter_shape: tuple[int, ...] | None = None,
    sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    q: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")
    normalized_leading_shape = tuple(int(value) for value in leading_shape)
    normalized_parameter_shape = (
        normalized_leading_shape
        if parameter_shape is None
        else tuple(int(value) for value in parameter_shape)
    )
    if len(normalized_parameter_shape) > len(normalized_leading_shape):
        raise ValueError("parameter_shape cannot be longer than leading_shape.")

    leading = (rng.batch_size, *normalized_leading_shape, int(dim))
    q_shape = (rng.batch_size, *normalized_parameter_shape)
    if q is None:
        q_low = 0.1 / math.log(dim + 1.0)
        q = rng.keyed("q").log_uniform(q_shape, low=q_low, high=6.0)
    if sigma is None:
        sigma = rng.keyed("sigma").log_uniform(q_shape, low=1e-4, high=10.0)
    base_noise = sample_noise_from_spec(
        leading,
        generator=rng.keyed("noise").torch_generator,
        device=rng.device,
        noise_spec=noise_spec,
        scale_multiplier=float(sigma_multiplier),
    )
    broadcast_tail = len(normalized_leading_shape) - len(normalized_parameter_shape) + 1
    q_view = q.view(rng.batch_size, *normalized_parameter_shape, *([1] * broadcast_tail))
    sigma_view = sigma.view(rng.batch_size, *normalized_parameter_shape, *([1] * broadcast_tail))
    noise = base_noise * sigma_view
    ranks = torch.arange(1, dim + 1, dtype=torch.float32, device=rng.device).view(
        *([1] * (len(normalized_leading_shape) + 1)),
        dim,
    )
    log_w = (-q_view * torch.log(ranks)) + noise
    log_w = torch.nan_to_num(log_w, nan=0.0, posinf=60.0, neginf=-60.0)
    log_w = torch.clamp(log_w, min=-60.0, max=60.0)
    log_w = log_w - torch.max(log_w, dim=-1, keepdim=True).values
    weights = torch.clamp(torch.exp(log_w), min=1e-12)
    weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-12)
    perm = torch.argsort(rng.keyed("perm").uniform(leading, low=0.0, high=1.0), dim=-1)
    return torch.gather(weights, -1, perm)


__all__ = [
    "FixedLayoutBatchRng",
    "_aggregate_batch_incrementally",
    "_aggregate_parent_outputs_batch",
    "_batch_standardize",
    "_flatten_leading_dims",
    "_lp_distances_to_centers",
    "_nearest_lp_center_indices",
    "_row_normalize_batch",
    "_sample_random_weights_batch",
    "_sanitize_and_batch_standardize",
]
