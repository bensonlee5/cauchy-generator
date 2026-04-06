"""Correlated scalar and categorical sampling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, TypeVar

import torch

from dagzoo.rng import KeyedRng

_T = TypeVar("_T")
_CATEGORY_WEIGHT_LOG_LOW = math.log(0.25)
_CATEGORY_WEIGHT_LOG_HIGH = math.log(4.0)


@dataclass(slots=True)
class _NumericParams:
    alpha: float
    beta: float


@dataclass(slots=True)
class CorrelatedSamplingContext:
    """Memoize dataset-scoped correlated sampling draws by semantic name."""

    shared_root: KeyedRng
    _numeric_params: dict[tuple[str, str], _NumericParams]
    _categorical_weights: dict[tuple[str, str, str], float]
    _categorical_weight_tensors: dict[tuple[str, str, tuple[str, ...]], torch.Tensor]

    @classmethod
    def from_keyed_rng(cls, keyed_rng: KeyedRng) -> "CorrelatedSamplingContext":
        return cls(
            shared_root=_shared_correlation_root(keyed_rng),
            _numeric_params={},
            _categorical_weights={},
            _categorical_weight_tensors={},
        )

    def matches(self, keyed_rng: KeyedRng) -> bool:
        shared_root = _shared_correlation_root(keyed_rng)
        return int(self.shared_root.seed) == int(shared_root.seed) and tuple(
            self.shared_root._ambient_nonce
        ) == tuple(shared_root._ambient_nonce)

    def numeric_params(self, *, name: str, device: str) -> _NumericParams:
        key = (str(device), str(name))
        cached = self._numeric_params.get(key)
        if cached is not None:
            return cached
        generator = self.shared_root.torch_rng("numeric_params", name, device=device)
        params = _draw_numeric_params(generator, device=device)
        self._numeric_params[key] = params
        return params

    def categorical_weight(self, *, name: str, label: str, device: str) -> float:
        key = (str(device), str(name), str(label))
        cached = self._categorical_weights.get(key)
        if cached is not None:
            return cached
        generator = self.shared_root.torch_rng(
            "categorical_weight",
            name,
            label,
            device=device,
        )
        log_weight = torch.empty(1, device=_tensor_device(device)).uniform_(
            _CATEGORY_WEIGHT_LOG_LOW,
            _CATEGORY_WEIGHT_LOG_HIGH,
            generator=generator,
        )
        value = float(torch.exp(log_weight).item())
        self._categorical_weights[key] = value
        return value

    def categorical_weight_tensor(
        self,
        *,
        name: str,
        labels: tuple[str, ...],
        device: str,
    ) -> torch.Tensor:
        key = (str(device), str(name), tuple(str(label) for label in labels))
        cached = self._categorical_weight_tensors.get(key)
        if cached is not None:
            return cached
        weights = torch.tensor(
            [self.categorical_weight(name=name, label=label, device=device) for label in labels],
            dtype=torch.float32,
            device=_tensor_device(device),
        )
        self._categorical_weight_tensors[key] = weights
        return weights


_CURRENT_CORRELATED_SAMPLING_CONTEXT: ContextVar[CorrelatedSamplingContext | None] = ContextVar(
    "dagzoo_correlated_sampling_context",
    default=None,
)


def _shared_correlation_root(keyed_rng: KeyedRng) -> KeyedRng:
    """Return a keyed RNG rooted at the dataset seed, ignoring local call-site path."""

    return KeyedRng(
        seed=int(keyed_rng.seed),
        _ambient_nonce=tuple(int(value) for value in keyed_rng._ambient_nonce),
    )


def current_correlated_sampling_context() -> CorrelatedSamplingContext | None:
    """Return the active correlated-sampling cache, if any."""

    return _CURRENT_CORRELATED_SAMPLING_CONTEXT.get()


@contextmanager
def correlated_sampling_scope(keyed_rng: KeyedRng) -> Iterator[CorrelatedSamplingContext]:
    """Install one correlated-sampling cache for compatible keyed roots."""

    current = current_correlated_sampling_context()
    if current is not None and current.matches(keyed_rng):
        yield current
        return
    context = CorrelatedSamplingContext.from_keyed_rng(keyed_rng)
    token = _CURRENT_CORRELATED_SAMPLING_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_CORRELATED_SAMPLING_CONTEXT.reset(token)


def _tensor_device(device: str) -> str:
    normalized = str(device)
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return normalized


def _draw_numeric_params(generator: torch.Generator, *, device: str) -> _NumericParams:
    tensor_device = _tensor_device(device)
    t = torch.empty(1, device=tensor_device).uniform_(0.0, 1.0, generator=generator).item()
    log_s = (
        torch.empty(1, device=tensor_device)
        .uniform_(math.log(0.1), math.log(10_000.0), generator=generator)
        .item()
    )
    s = math.exp(log_s)
    return _NumericParams(
        alpha=float(s * t),
        beta=float(s * (1.0 - t)),
    )


def numeric_params_for_name(
    keyed_rng: KeyedRng,
    name: str,
    *,
    device: str,
) -> _NumericParams:
    """Return shared Beta parameters for a semantic scalar variable name."""

    context = current_correlated_sampling_context()
    if context is not None and context.matches(keyed_rng):
        return context.numeric_params(name=name, device=device)
    generator = _shared_correlation_root(keyed_rng).torch_rng("numeric_params", name, device=device)
    return _draw_numeric_params(generator, device=device)


def sample_correlated_unit_interval_tensor(
    keyed_rng: KeyedRng,
    *,
    name: str,
    shape: Sequence[int] = (),
    device: str,
) -> torch.Tensor:
    """Sample tensor-shaped correlated Beta draws for one semantic variable name."""

    normalized_shape = tuple(int(value) for value in shape)
    params = numeric_params_for_name(keyed_rng, name, device=device)
    generator = keyed_rng.torch_rng("numeric_draw", name, device="cpu")
    tensor_device = _tensor_device(device)
    concentration = torch.tensor(
        [params.alpha, params.beta],
        dtype=torch.float64,
        device="cpu",
    ).expand(*normalized_shape, 2)
    probs = torch._sample_dirichlet(concentration, generator=generator)
    return probs[..., 0].to(device=tensor_device, dtype=torch.float32)


def sample_correlated_num(
    keyed_rng: KeyedRng,
    *,
    name: str,
    low: float,
    high: float,
    device: str,
    log_scale: bool = False,
    as_int: bool = False,
) -> float | int:
    """Sample one correlated scalar for a semantic variable name."""

    u = float(
        sample_correlated_unit_interval_tensor(
            keyed_rng,
            name=name,
            shape=(),
            device=device,
        ).item()
    )
    if log_scale:
        value = math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
    else:
        value = low + u * (high - low)
    if as_int:
        return int(math.floor(value))
    return float(value)


def sample_correlated_num_tensor(
    keyed_rng: KeyedRng,
    *,
    name: str,
    shape: Sequence[int],
    low: float,
    high: float,
    device: str,
    log_scale: bool = False,
) -> torch.Tensor:
    """Sample a tensor of correlated numeric draws for one semantic variable name."""

    u = sample_correlated_unit_interval_tensor(
        keyed_rng,
        name=name,
        shape=shape,
        device=device,
    ).to(dtype=torch.float32)
    if log_scale:
        log_low = float(math.log(low))
        log_high = float(math.log(high))
        return torch.exp(log_low + u * (log_high - log_low))
    return float(low) + u * float(high - low)


def _shared_category_weight(
    keyed_rng: KeyedRng,
    *,
    name: str,
    label: str,
    device: str,
) -> float:
    context = current_correlated_sampling_context()
    if context is not None and context.matches(keyed_rng):
        return context.categorical_weight(name=name, label=label, device=device)
    generator = _shared_correlation_root(keyed_rng).torch_rng(
        "categorical_weight",
        name,
        label,
        device=device,
    )
    log_weight = torch.empty(1, device=_tensor_device(device)).uniform_(
        _CATEGORY_WEIGHT_LOG_LOW,
        _CATEGORY_WEIGHT_LOG_HIGH,
        generator=generator,
    )
    return float(torch.exp(log_weight).item())


def sample_correlated_choice(
    keyed_rng: KeyedRng,
    *,
    name: str,
    values: Sequence[_T],
    device: str,
    base_probs: Sequence[float] | None = None,
) -> _T:
    """Sample one value using shared per-label weights and local call-site randomness."""

    normalized_values = tuple(values)
    if not normalized_values:
        raise ValueError("sample_correlated_choice requires at least one candidate value.")
    if base_probs is None:
        probs = torch.full(
            (len(normalized_values),),
            1.0 / float(len(normalized_values)),
            dtype=torch.float32,
            device=_tensor_device(device),
        )
    else:
        if len(base_probs) != len(normalized_values):
            raise ValueError("base_probs length must match values length.")
        probs = torch.as_tensor(base_probs, dtype=torch.float32, device=_tensor_device(device))
        probs = torch.clamp(probs, min=0.0)
        total = float(probs.sum().item())
        if total <= 0.0:
            raise ValueError("sample_correlated_choice requires at least one positive base_prob.")
        probs = probs / total

    labels = tuple(str(value) for value in normalized_values)
    context = current_correlated_sampling_context()
    if context is not None and context.matches(keyed_rng):
        shared_weights = context.categorical_weight_tensor(
            name=name,
            labels=labels,
            device=device,
        )
    else:
        shared_weights = torch.tensor(
            [
                _shared_category_weight(
                    keyed_rng,
                    name=name,
                    label=label,
                    device=device,
                )
                for label in labels
            ],
            dtype=torch.float32,
            device=_tensor_device(device),
        )
    adjusted = torch.clamp(probs * shared_weights, min=1e-8)
    adjusted = adjusted / torch.clamp(adjusted.sum(), min=1e-12)
    generator = keyed_rng.torch_rng("categorical_draw", name, *labels, device=device)
    index = int(torch.multinomial(adjusted, 1, generator=generator).item())
    return normalized_values[index]


class CorrelatedSampler:
    """Name-keyed sampler with shared latent parameters per variable name."""

    def __init__(self, keyed_rng: KeyedRng, device: str):
        """Initialize correlated sampler state for one dataset generation run."""

        self._keyed_rng = keyed_rng
        self._device = device
        self._numeric_params: dict[str, _NumericParams] = {}
        self._categorical_weights: dict[tuple[str, int], torch.Tensor] = {}
        self._numeric_draw_counts: dict[str, int] = {}
        self._categorical_draw_counts: dict[tuple[str, int], int] = {}

    def _get_numeric_params(self, name: str) -> _NumericParams:
        """Create/retrieve latent beta parameters shared by a scalar variable name."""

        if name not in self._numeric_params:
            self._numeric_params[name] = numeric_params_for_name(
                self._keyed_rng,
                name,
                device=self._device,
            )
        return self._numeric_params[name]

    def _sample_beta(self, alpha: float, beta: float, *, name: str) -> float:
        """Sample a scalar from Beta(alpha, beta) using keyed per-name draws."""

        draw_index = int(self._numeric_draw_counts.get(name, 0))
        self._numeric_draw_counts[name] = draw_index + 1
        local_generator = self._keyed_rng.torch_rng(
            "numeric_draw",
            name,
            draw_index,
            device="cpu",
        )
        concentration = torch.tensor([alpha, beta], dtype=torch.float64, device="cpu")
        probs = torch._sample_dirichlet(concentration, generator=local_generator)
        return float(probs[0].item())

    def _categorical_draw_generator(self, name: str, n_categories: int) -> torch.Generator:
        key = (name, int(n_categories))
        draw_index = int(self._categorical_draw_counts.get(key, 0))
        self._categorical_draw_counts[key] = draw_index + 1
        return self._keyed_rng.torch_rng(
            "categorical_draw",
            name,
            int(n_categories),
            draw_index,
            device=self._device,
        )

    def sample_num(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log_scale: bool = False,
        as_int: bool = False,
    ) -> float | int:
        """Sample a correlated scalar for a named variable."""

        params = self._get_numeric_params(name)
        u = self._sample_beta(params.alpha, params.beta, name=name)
        if log_scale:
            value = math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
        else:
            value = low + u * (high - low)
        if as_int:
            return int(math.floor(value))
        return float(value)

    def sample_category(self, name: str, n_categories: int) -> int:
        """Sample a correlated categorical value for a named variable."""

        key = (name, n_categories)
        if key not in self._categorical_weights:
            generator = self._keyed_rng.torch_rng(
                "categorical_weights",
                name,
                int(n_categories),
                device=self._device,
            )
            raw = (
                torch.empty(n_categories, device=self._device).uniform_(0, 1, generator=generator)
                + 1e-6
            )
            self._categorical_weights[key] = raw / raw.sum()
        probs = self._categorical_weights[key]
        return int(
            torch.multinomial(
                probs,
                1,
                generator=self._categorical_draw_generator(name, n_categories),
            ).item()
        )
