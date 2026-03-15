import pytest
import torch
from conftest import make_generator as _make_generator
from hypothesis import given, settings
from hypothesis import strategies as st

import dagzoo.sampling.random_weights as random_weights_mod
from dagzoo.rng import SEED32_MAX
from dagzoo.sampling.noise import NoiseSamplingSpec
from dagzoo.sampling.random_weights import sample_random_weights

_MIXTURE_COMPONENTS = ("gaussian", "laplace", "student_t")
_SEED32_STRATEGY = st.integers(min_value=0, max_value=SEED32_MAX)
_DIM_STRATEGY = st.integers(min_value=1, max_value=64)
_Q_STRATEGY = st.floats(
    min_value=0.125,
    max_value=6.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_SIGMA_STRATEGY = st.floats(
    min_value=1e-4,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
)
_SIGMA_MULTIPLIER_STRATEGY = st.floats(
    min_value=0.125,
    max_value=4.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_SCALE_STRATEGY = st.floats(
    min_value=0.125,
    max_value=4.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_STUDENT_T_DF_STRATEGY = st.floats(
    min_value=2.25,
    max_value=12.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_POSITIVE_WEIGHT_STRATEGY = st.floats(
    min_value=0.125,
    max_value=8.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_MIXTURE_WEIGHTS_STRATEGY = st.dictionaries(
    keys=st.sampled_from(_MIXTURE_COMPONENTS),
    values=_POSITIVE_WEIGHT_STRATEGY,
    min_size=1,
    max_size=len(_MIXTURE_COMPONENTS),
)


@st.composite
def _noise_spec_strategy(draw: st.DrawFn) -> NoiseSamplingSpec | None:
    if draw(st.booleans()):
        return None

    family = draw(st.sampled_from(("gaussian", "laplace", "student_t", "mixture")))
    scale = draw(_SCALE_STRATEGY)
    if family == "student_t":
        return NoiseSamplingSpec(
            family=family,
            scale=scale,
            student_t_df=draw(_STUDENT_T_DF_STRATEGY),
        )
    if family == "mixture":
        return NoiseSamplingSpec(
            family=family,
            scale=scale,
            student_t_df=draw(_STUDENT_T_DF_STRATEGY),
            mixture_weights=draw(_MIXTURE_WEIGHTS_STRATEGY),
        )
    return NoiseSamplingSpec(family=family, scale=scale)


def _entropy(weights: torch.Tensor) -> float:
    probs = torch.clamp(weights, min=1e-12)
    return float(-(probs * torch.log(probs)).sum().item())


def _assert_valid_weights(weights: torch.Tensor, *, dim: int) -> None:
    assert weights.shape == (dim,)
    assert torch.all(torch.isfinite(weights))
    assert torch.all(weights > 0)
    torch.testing.assert_close(
        weights.sum(),
        torch.tensor(1.0, dtype=weights.dtype, device=weights.device),
        atol=1e-5,
        rtol=1e-5,
    )


def test_random_weights_normalized() -> None:
    g = _make_generator(42)
    w = sample_random_weights(32, g, "cpu")
    assert w.shape == (32,)
    assert torch.all(w > 0)
    torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_random_weights_deterministic() -> None:
    a = sample_random_weights(16, _make_generator(0), "cpu")
    b = sample_random_weights(16, _make_generator(0), "cpu")
    torch.testing.assert_close(a, b)


def test_random_weights_positive() -> None:
    g = _make_generator(7)
    w = sample_random_weights(64, g, "cpu")
    assert torch.all(w > 0)


def test_random_weights_sigma_multiplier_increases_peakedness() -> None:
    low = sample_random_weights(
        64,
        _make_generator(314),
        "cpu",
        q=0.0,
        sigma=1.0,
        sigma_multiplier=1.0,
    )
    high = sample_random_weights(
        64,
        _make_generator(314),
        "cpu",
        q=0.0,
        sigma=1.0,
        sigma_multiplier=1.5,
    )
    assert _entropy(high) < _entropy(low)


def test_random_weights_nonlegacy_noise_remains_finite() -> None:
    w = sample_random_weights(
        128,
        _make_generator(101),
        "cpu",
        noise_spec=NoiseSamplingSpec(family="student_t", student_t_df=5.0),
    )
    assert torch.all(torch.isfinite(w))
    assert torch.all(w > 0)
    torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_random_weights_handles_nonfinite_noise_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _bad_noise(*_args, **_kwargs) -> torch.Tensor:
        return torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0])

    monkeypatch.setattr(random_weights_mod, "sample_noise_from_spec", _bad_noise)
    w = sample_random_weights(4, _make_generator(202), "cpu", q=1.0, sigma=1.0)
    assert torch.all(torch.isfinite(w))
    assert torch.all(w > 0)
    torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


@settings(max_examples=75, deadline=None)
@given(
    dim=_DIM_STRATEGY,
    seed=_SEED32_STRATEGY,
    q=_Q_STRATEGY,
    sigma=_SIGMA_STRATEGY,
    sigma_multiplier=_SIGMA_MULTIPLIER_STRATEGY,
    noise_spec=_noise_spec_strategy(),
)
def test_random_weights_hypothesis_explicit_params_are_valid_and_deterministic(
    dim: int,
    seed: int,
    q: float,
    sigma: float,
    sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> None:
    first = sample_random_weights(
        dim,
        _make_generator(seed),
        "cpu",
        q=q,
        sigma=sigma,
        sigma_multiplier=sigma_multiplier,
        noise_spec=noise_spec,
    )
    second = sample_random_weights(
        dim,
        _make_generator(seed),
        "cpu",
        q=q,
        sigma=sigma,
        sigma_multiplier=sigma_multiplier,
        noise_spec=noise_spec,
    )

    _assert_valid_weights(first, dim=dim)
    torch.testing.assert_close(first, second)


@settings(max_examples=75, deadline=None)
@given(
    dim=_DIM_STRATEGY,
    seed=_SEED32_STRATEGY,
    sigma_multiplier=_SIGMA_MULTIPLIER_STRATEGY,
    noise_spec=_noise_spec_strategy(),
)
def test_random_weights_hypothesis_internal_q_sigma_path_is_valid_and_deterministic(
    dim: int,
    seed: int,
    sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> None:
    first = sample_random_weights(
        dim,
        _make_generator(seed),
        "cpu",
        sigma_multiplier=sigma_multiplier,
        noise_spec=noise_spec,
    )
    second = sample_random_weights(
        dim,
        _make_generator(seed),
        "cpu",
        sigma_multiplier=sigma_multiplier,
        noise_spec=noise_spec,
    )

    _assert_valid_weights(first, dim=dim)
    torch.testing.assert_close(first, second)


@settings(max_examples=75, deadline=None)
@given(
    seed=_SEED32_STRATEGY,
    q=_Q_STRATEGY,
    sigma=_SIGMA_STRATEGY,
    sigma_multiplier=_SIGMA_MULTIPLIER_STRATEGY,
    noise_spec=_noise_spec_strategy(),
)
def test_random_weights_hypothesis_dim_one_is_exactly_one(
    seed: int,
    q: float,
    sigma: float,
    sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> None:
    weights = sample_random_weights(
        1,
        _make_generator(seed),
        "cpu",
        q=q,
        sigma=sigma,
        sigma_multiplier=sigma_multiplier,
        noise_spec=noise_spec,
    )

    _assert_valid_weights(weights, dim=1)
    torch.testing.assert_close(
        weights,
        torch.ones(1, dtype=weights.dtype, device=weights.device),
        atol=1e-6,
        rtol=1e-6,
    )


@settings(max_examples=75, deadline=None)
@given(
    dim=_DIM_STRATEGY,
    seed=_SEED32_STRATEGY,
    q=_Q_STRATEGY,
    sigma=_SIGMA_STRATEGY,
    low_sigma_multiplier=_SIGMA_MULTIPLIER_STRATEGY,
    high_sigma_multiplier=_SIGMA_MULTIPLIER_STRATEGY,
    noise_spec=_noise_spec_strategy(),
)
def test_random_weights_hypothesis_larger_sigma_multiplier_preserves_validity(
    dim: int,
    seed: int,
    q: float,
    sigma: float,
    low_sigma_multiplier: float,
    high_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> None:
    low, high = sorted((low_sigma_multiplier, high_sigma_multiplier))
    low_weights = sample_random_weights(
        dim,
        _make_generator(seed),
        "cpu",
        q=q,
        sigma=sigma,
        sigma_multiplier=low,
        noise_spec=noise_spec,
    )
    high_weights = sample_random_weights(
        dim,
        _make_generator(seed),
        "cpu",
        q=q,
        sigma=sigma,
        sigma_multiplier=high,
        noise_spec=noise_spec,
    )

    _assert_valid_weights(low_weights, dim=dim)
    _assert_valid_weights(high_weights, dim=dim)


@settings(max_examples=75, deadline=None)
@given(
    dim=_DIM_STRATEGY,
    seed=_SEED32_STRATEGY,
    q=_Q_STRATEGY,
    sigma=_SIGMA_STRATEGY,
    sigma_multiplier=_SIGMA_MULTIPLIER_STRATEGY,
)
def test_random_weights_hypothesis_sanitizes_nonfinite_noise_samples(
    dim: int,
    seed: int,
    q: float,
    sigma: float,
    sigma_multiplier: float,
) -> None:
    def _bad_noise(shape: tuple[int, ...], **_kwargs) -> torch.Tensor:
        numel = shape[0]
        pattern = torch.tensor(
            [float("inf"), float("-inf"), float("nan"), 0.0],
            dtype=torch.float32,
        )
        repeats = (numel + len(pattern) - 1) // len(pattern)
        return pattern.repeat(repeats)[:numel]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(random_weights_mod, "sample_noise_from_spec", _bad_noise)
        weights = sample_random_weights(
            dim,
            _make_generator(seed),
            "cpu",
            q=q,
            sigma=sigma,
            sigma_multiplier=sigma_multiplier,
        )

    _assert_valid_weights(weights, dim=dim)
