"""Tests for math/random_matrices.py."""

import pytest
import torch
from conftest import make_generator as _make_generator
from hypothesis import given, settings
from hypothesis import strategies as st

from dagzoo.math.random_matrices import sample_random_matrix
from dagzoo.rng import SEED32_MAX
from dagzoo.sampling.noise import NoiseSamplingSpec

_SEED32_STRATEGY = st.integers(min_value=0, max_value=SEED32_MAX)
_OUT_DIM_STRATEGY = st.integers(min_value=1, max_value=24)
_IN_DIM_STRATEGY = st.integers(min_value=1, max_value=24)
_KIND_STRATEGY = st.sampled_from(("gaussian", "weights", "singular_values", "kernel", "activation"))
_NORMALIZING_KIND_STRATEGY = st.sampled_from(("weights", "kernel", "activation"))
_NOISE_SIGMA_MULTIPLIER_STRATEGY = st.floats(
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
    keys=st.sampled_from(("gaussian", "laplace", "student_t")),
    values=_POSITIVE_WEIGHT_STRATEGY,
    min_size=1,
    max_size=3,
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


def _assert_valid_matrix(matrix: torch.Tensor, *, out_dim: int, in_dim: int) -> None:
    assert matrix.shape == (out_dim, in_dim)
    assert torch.all(torch.isfinite(matrix))
    norms = torch.linalg.norm(matrix, dim=1)
    torch.testing.assert_close(
        norms,
        torch.ones(out_dim, dtype=matrix.dtype, device=matrix.device),
        atol=1e-4,
        rtol=1e-4,
    )


def test_output_shape() -> None:
    g = _make_generator(0)
    m = sample_random_matrix(5, 3, g, "cpu")
    assert m.shape == (5, 3)


def test_rows_unit_normalized() -> None:
    g = _make_generator(1)
    m = sample_random_matrix(4, 6, g, "cpu")
    norms = torch.linalg.norm(m, dim=1)
    torch.testing.assert_close(norms, torch.ones(4), atol=1e-4, rtol=1e-4)


def test_deterministic() -> None:
    a = sample_random_matrix(3, 4, _make_generator(42), "cpu")
    b = sample_random_matrix(3, 4, _make_generator(42), "cpu")
    torch.testing.assert_close(a, b)


def test_kernel_single_column_rows_are_unit_normalized() -> None:
    matrix = sample_random_matrix(6, 1, _make_generator(0), "cpu", kind="kernel")
    norms = torch.linalg.norm(matrix, dim=1)
    torch.testing.assert_close(norms, torch.ones(6), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kind", ["gaussian", "weights", "singular_values", "kernel", "activation"])
def test_each_kind(kind: str) -> None:
    g = _make_generator(10)
    m = sample_random_matrix(4, 5, g, "cpu", kind=kind)
    assert m.shape == (4, 5)
    assert torch.all(torch.isfinite(m))


def test_invalid_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown matrix kind"):
        sample_random_matrix(2, 2, _make_generator(0), "cpu", kind="bogus")


@settings(max_examples=60, deadline=None)
@given(
    out_dim=_OUT_DIM_STRATEGY,
    in_dim=_IN_DIM_STRATEGY,
    seed=_SEED32_STRATEGY,
    kind=_KIND_STRATEGY,
    noise_sigma_multiplier=_NOISE_SIGMA_MULTIPLIER_STRATEGY,
    noise_spec=_noise_spec_strategy(),
)
def test_random_matrix_hypothesis_supported_kinds_are_finite_normalized_and_deterministic(
    out_dim: int,
    in_dim: int,
    seed: int,
    kind: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> None:
    first = sample_random_matrix(
        out_dim,
        in_dim,
        _make_generator(seed),
        "cpu",
        kind=kind,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    second = sample_random_matrix(
        out_dim,
        in_dim,
        _make_generator(seed),
        "cpu",
        kind=kind,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )

    _assert_valid_matrix(first, out_dim=out_dim, in_dim=in_dim)
    torch.testing.assert_close(first, second)


@settings(max_examples=60, deadline=None)
@given(
    out_dim=_OUT_DIM_STRATEGY,
    in_dim=_IN_DIM_STRATEGY,
    seed=_SEED32_STRATEGY,
    noise_sigma_multiplier=_NOISE_SIGMA_MULTIPLIER_STRATEGY,
    noise_spec=_noise_spec_strategy(),
)
def test_random_matrix_hypothesis_kind_none_is_deterministic_and_valid(
    out_dim: int,
    in_dim: int,
    seed: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> None:
    first = sample_random_matrix(
        out_dim,
        in_dim,
        _make_generator(seed),
        "cpu",
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    second = sample_random_matrix(
        out_dim,
        in_dim,
        _make_generator(seed),
        "cpu",
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )

    _assert_valid_matrix(first, out_dim=out_dim, in_dim=in_dim)
    torch.testing.assert_close(first, second)


@settings(max_examples=60, deadline=None)
@given(
    out_dim=_OUT_DIM_STRATEGY,
    in_dim=_IN_DIM_STRATEGY,
    seed=_SEED32_STRATEGY,
    kind=_NORMALIZING_KIND_STRATEGY,
    noise_sigma_multiplier=_NOISE_SIGMA_MULTIPLIER_STRATEGY,
    noise_spec=_noise_spec_strategy(),
)
def test_random_matrix_hypothesis_transformed_kinds_preserve_row_normalization(
    out_dim: int,
    in_dim: int,
    seed: int,
    kind: str,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> None:
    matrix = sample_random_matrix(
        out_dim,
        in_dim,
        _make_generator(seed),
        "cpu",
        kind=kind,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )

    _assert_valid_matrix(matrix, out_dim=out_dim, in_dim=in_dim)
