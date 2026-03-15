"""Tests for deterministic MCAR/MAR/MNAR missingness mask sampling."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from dagzoo.config import (
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MNAR,
    MISSINGNESS_MECHANISM_NONE,
    DatasetConfig,
    MissingnessMechanism,
)
from dagzoo.rng import SEED32_MAX, KeyedRng
from dagzoo.sampling import sample_missingness_mask

_MECHANISM_STRATEGY = st.sampled_from(
    (MISSINGNESS_MECHANISM_MCAR, MISSINGNESS_MECHANISM_MAR, MISSINGNESS_MECHANISM_MNAR)
)
_SEED32_STRATEGY = st.integers(min_value=0, max_value=SEED32_MAX)
_ROWS_STRATEGY = st.integers(min_value=64, max_value=512)
_COLS_STRATEGY = st.integers(min_value=2, max_value=16)
_EMPIRICAL_ROWS_STRATEGY = st.integers(min_value=256, max_value=512)
_EMPIRICAL_COLS_STRATEGY = st.integers(min_value=8, max_value=16)
_MISSING_RATE_STRATEGY = st.floats(
    min_value=0.05,
    max_value=0.95,
    allow_nan=False,
    allow_infinity=False,
)
_MAR_OBSERVED_FRACTION_STRATEGY = st.floats(
    min_value=0.1,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)
_LOGIT_SCALE_STRATEGY = st.floats(
    min_value=0.25,
    max_value=3.0,
    allow_nan=False,
    allow_infinity=False,
)


def _feature_matrix(n_rows: int = 512, n_cols: int = 12) -> torch.Tensor:
    """Create a deterministic, non-trivial feature matrix for sampler tests."""

    grid = torch.linspace(-3.0, 3.0, steps=n_rows, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(1, n_cols + 1, dtype=torch.float32).unsqueeze(0)
    return torch.sin(grid * freqs * 0.7) + torch.cos(grid * (freqs + 0.5) * 0.4)


def _cfg(
    mechanism: MissingnessMechanism,
    *,
    missing_rate: float = 0.35,
    missing_mar_observed_fraction: float = 0.5,
    missing_mar_logit_scale: float = 1.5,
    missing_mnar_logit_scale: float = 1.5,
) -> DatasetConfig:
    return DatasetConfig(
        missing_rate=missing_rate,
        missing_mechanism=mechanism,
        missing_mar_observed_fraction=missing_mar_observed_fraction,
        missing_mar_logit_scale=missing_mar_logit_scale,
        missing_mnar_logit_scale=missing_mnar_logit_scale,
    )


def test_missingness_mask_shape_and_dtype() -> None:
    x = _feature_matrix(64, 7)
    cfg = _cfg(MISSINGNESS_MECHANISM_MCAR, missing_rate=0.2)
    mask = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(7), device="cpu")
    assert mask.shape == x.shape
    assert mask.dtype == torch.bool


@pytest.mark.parametrize(
    "mechanism",
    [MISSINGNESS_MECHANISM_NONE, MISSINGNESS_MECHANISM_MCAR, MISSINGNESS_MECHANISM_MAR],
)
def test_missingness_rate_zero_returns_all_false(mechanism: str) -> None:
    x = _feature_matrix(32, 5)
    cfg = _cfg(mechanism, missing_rate=0.0)
    mask = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(1), device="cpu")
    assert not torch.any(mask)


def test_mcar_rate_one_returns_all_true() -> None:
    x = _feature_matrix(32, 5)
    cfg = _cfg(MISSINGNESS_MECHANISM_MCAR, missing_rate=1.0)
    mask = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(1), device="cpu")
    assert torch.all(mask)


def test_mcar_empirical_rate_close_to_target() -> None:
    x = _feature_matrix(2000, 10)
    target_rate = 0.30
    cfg = _cfg(MISSINGNESS_MECHANISM_MCAR, missing_rate=target_rate)
    mask = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(11), device="cpu")
    observed_rate = float(mask.float().mean().item())
    assert abs(observed_rate - target_rate) < 0.03


@pytest.mark.parametrize(
    "mechanism",
    [MISSINGNESS_MECHANISM_MCAR, MISSINGNESS_MECHANISM_MAR, MISSINGNESS_MECHANISM_MNAR],
)
def test_sampler_is_deterministic_for_fixed_seed(mechanism: str) -> None:
    x = _feature_matrix(257, 13)
    cfg = _cfg(mechanism, missing_rate=0.35)
    mask_a = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(123), device="cpu")
    mask_b = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(123), device="cpu")
    assert torch.equal(mask_a, mask_b)


@pytest.mark.parametrize(
    "mechanism",
    [MISSINGNESS_MECHANISM_MCAR, MISSINGNESS_MECHANISM_MAR, MISSINGNESS_MECHANISM_MNAR],
)
def test_sampler_changes_when_seed_changes(mechanism: str) -> None:
    x = _feature_matrix(257, 13)
    cfg = _cfg(mechanism, missing_rate=0.35)
    mask_a = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(123), device="cpu")
    mask_b = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(124), device="cpu")
    assert not torch.equal(mask_a, mask_b)


@pytest.mark.parametrize(
    "mechanism",
    [MISSINGNESS_MECHANISM_MAR, MISSINGNESS_MECHANISM_MNAR],
)
def test_mar_and_mnar_empirical_rate_close_to_target(mechanism: str) -> None:
    x = _feature_matrix(2048, 16)
    target_rate = 0.25
    cfg = _cfg(mechanism, missing_rate=target_rate)
    mask = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(19), device="cpu")
    observed_rate = float(mask.float().mean().item())
    assert abs(observed_rate - target_rate) < 0.05


def test_mar_mask_depends_on_observed_feature_values() -> None:
    x = _feature_matrix(512, 12)
    perturb = torch.linspace(-1.5, 1.5, steps=x.shape[0], dtype=torch.float32).unsqueeze(1)
    scales = torch.arange(1, x.shape[1] + 1, dtype=torch.float32).unsqueeze(0) / float(x.shape[1])
    x_shifted = x + perturb * scales

    cfg = _cfg(MISSINGNESS_MECHANISM_MAR, missing_rate=0.35)
    seed = 909
    mask_a = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu")
    mask_b = sample_missingness_mask(
        x_shifted, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu"
    )
    assert not torch.equal(mask_a, mask_b)


def test_mnar_mask_depends_on_feature_self_values() -> None:
    x = _feature_matrix(512, 12)
    x_mutated = x.clone()
    x_mutated[:, 0] = x_mutated[:, 0] ** 3 + 0.5 * x_mutated[:, 0]

    cfg = _cfg(MISSINGNESS_MECHANISM_MNAR, missing_rate=0.35)
    seed = 1201
    mask_a = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu")
    mask_b = sample_missingness_mask(
        x_mutated, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu"
    )

    assert int(torch.count_nonzero(mask_a ^ mask_b).item()) > 0
    assert int(torch.count_nonzero(mask_a[:, 0] ^ mask_b[:, 0]).item()) > 0


def test_sampler_rejects_non_2d_input() -> None:
    x = torch.ones(10)
    cfg = _cfg(MISSINGNESS_MECHANISM_MCAR, missing_rate=0.2)
    with pytest.raises(ValueError, match="expects a 2D tensor"):
        sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(1), device="cpu")


@settings(max_examples=50, deadline=None)
@given(
    mechanism=_MECHANISM_STRATEGY,
    n_rows=_ROWS_STRATEGY,
    n_cols=_COLS_STRATEGY,
    missing_rate=_MISSING_RATE_STRATEGY,
    missing_mar_observed_fraction=_MAR_OBSERVED_FRACTION_STRATEGY,
    missing_mar_logit_scale=_LOGIT_SCALE_STRATEGY,
    missing_mnar_logit_scale=_LOGIT_SCALE_STRATEGY,
    seed=_SEED32_STRATEGY,
)
def test_missingness_mask_shape_and_dtype_hypothesis(
    mechanism: MissingnessMechanism,
    n_rows: int,
    n_cols: int,
    missing_rate: float,
    missing_mar_observed_fraction: float,
    missing_mar_logit_scale: float,
    missing_mnar_logit_scale: float,
    seed: int,
) -> None:
    x = _feature_matrix(n_rows, n_cols)
    cfg = _cfg(
        mechanism,
        missing_rate=missing_rate,
        missing_mar_observed_fraction=missing_mar_observed_fraction,
        missing_mar_logit_scale=missing_mar_logit_scale,
        missing_mnar_logit_scale=missing_mnar_logit_scale,
    )

    mask = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu")

    assert mask.shape == x.shape
    assert mask.dtype == torch.bool


@settings(max_examples=50, deadline=None)
@given(
    mechanism=_MECHANISM_STRATEGY,
    n_rows=_ROWS_STRATEGY,
    n_cols=_COLS_STRATEGY,
    missing_rate=_MISSING_RATE_STRATEGY,
    missing_mar_observed_fraction=_MAR_OBSERVED_FRACTION_STRATEGY,
    missing_mar_logit_scale=_LOGIT_SCALE_STRATEGY,
    missing_mnar_logit_scale=_LOGIT_SCALE_STRATEGY,
    seed=_SEED32_STRATEGY,
)
def test_missingness_mask_is_deterministic_for_identical_inputs_hypothesis(
    mechanism: MissingnessMechanism,
    n_rows: int,
    n_cols: int,
    missing_rate: float,
    missing_mar_observed_fraction: float,
    missing_mar_logit_scale: float,
    missing_mnar_logit_scale: float,
    seed: int,
) -> None:
    x = _feature_matrix(n_rows, n_cols)
    cfg = _cfg(
        mechanism,
        missing_rate=missing_rate,
        missing_mar_observed_fraction=missing_mar_observed_fraction,
        missing_mar_logit_scale=missing_mar_logit_scale,
        missing_mnar_logit_scale=missing_mnar_logit_scale,
    )

    first = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu")
    second = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu")

    assert torch.equal(first, second)


@settings(max_examples=50, deadline=None)
@given(
    mechanism=_MECHANISM_STRATEGY,
    n_rows=_ROWS_STRATEGY,
    n_cols=_COLS_STRATEGY,
    missing_rate_pair=st.tuples(_MISSING_RATE_STRATEGY, _MISSING_RATE_STRATEGY).map(
        lambda pair: tuple(sorted(pair))
    ),
    missing_mar_observed_fraction=_MAR_OBSERVED_FRACTION_STRATEGY,
    missing_mar_logit_scale=_LOGIT_SCALE_STRATEGY,
    missing_mnar_logit_scale=_LOGIT_SCALE_STRATEGY,
    seed=_SEED32_STRATEGY,
)
def test_missingness_mask_rate_is_monotonic_for_same_seed_hypothesis(
    mechanism: MissingnessMechanism,
    n_rows: int,
    n_cols: int,
    missing_rate_pair: tuple[float, float],
    missing_mar_observed_fraction: float,
    missing_mar_logit_scale: float,
    missing_mnar_logit_scale: float,
    seed: int,
) -> None:
    low_rate, high_rate = missing_rate_pair
    x = _feature_matrix(n_rows, n_cols)
    common_kwargs = {
        "missing_mar_observed_fraction": missing_mar_observed_fraction,
        "missing_mar_logit_scale": missing_mar_logit_scale,
        "missing_mnar_logit_scale": missing_mnar_logit_scale,
    }
    low_mask = sample_missingness_mask(
        x,
        dataset_cfg=_cfg(mechanism, missing_rate=low_rate, **common_kwargs),
        keyed_rng=KeyedRng(seed),
        device="cpu",
    )
    high_mask = sample_missingness_mask(
        x,
        dataset_cfg=_cfg(mechanism, missing_rate=high_rate, **common_kwargs),
        keyed_rng=KeyedRng(seed),
        device="cpu",
    )

    assert not bool(torch.any(low_mask & ~high_mask).item())


@settings(max_examples=50, deadline=None)
@given(
    mechanism=_MECHANISM_STRATEGY,
    n_rows=_EMPIRICAL_ROWS_STRATEGY,
    n_cols=_EMPIRICAL_COLS_STRATEGY,
    missing_rate=_MISSING_RATE_STRATEGY,
    missing_mar_observed_fraction=_MAR_OBSERVED_FRACTION_STRATEGY,
    missing_mar_logit_scale=_LOGIT_SCALE_STRATEGY,
    missing_mnar_logit_scale=_LOGIT_SCALE_STRATEGY,
    seed=_SEED32_STRATEGY,
)
def test_missingness_mask_empirical_rate_tracks_target_hypothesis(
    mechanism: MissingnessMechanism,
    n_rows: int,
    n_cols: int,
    missing_rate: float,
    missing_mar_observed_fraction: float,
    missing_mar_logit_scale: float,
    missing_mnar_logit_scale: float,
    seed: int,
) -> None:
    x = _feature_matrix(n_rows, n_cols)
    cfg = _cfg(
        mechanism,
        missing_rate=missing_rate,
        missing_mar_observed_fraction=missing_mar_observed_fraction,
        missing_mar_logit_scale=missing_mar_logit_scale,
        missing_mnar_logit_scale=missing_mnar_logit_scale,
    )

    mask = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu")
    observed_rate = float(mask.float().mean().item())
    tolerance = 0.05 if mechanism == MISSINGNESS_MECHANISM_MCAR else 0.08

    assert abs(observed_rate - missing_rate) <= tolerance


@settings(max_examples=50, deadline=None)
@given(
    mechanism=_MECHANISM_STRATEGY,
    n_rows=_ROWS_STRATEGY,
    n_cols=_COLS_STRATEGY,
    missing_rate=_MISSING_RATE_STRATEGY,
    missing_mar_observed_fraction=_MAR_OBSERVED_FRACTION_STRATEGY,
    missing_mar_logit_scale=_LOGIT_SCALE_STRATEGY,
    missing_mnar_logit_scale=_LOGIT_SCALE_STRATEGY,
    seed=_SEED32_STRATEGY,
)
def test_missingness_mask_sanitizes_nonfinite_inputs_hypothesis(
    mechanism: MissingnessMechanism,
    n_rows: int,
    n_cols: int,
    missing_rate: float,
    missing_mar_observed_fraction: float,
    missing_mar_logit_scale: float,
    missing_mnar_logit_scale: float,
    seed: int,
) -> None:
    x = _feature_matrix(n_rows, n_cols).clone()
    x[0, 0] = float("nan")
    x[1, 1 % n_cols] = float("inf")
    x[2, 0] = -float("inf")
    cfg = _cfg(
        mechanism,
        missing_rate=missing_rate,
        missing_mar_observed_fraction=missing_mar_observed_fraction,
        missing_mar_logit_scale=missing_mar_logit_scale,
        missing_mnar_logit_scale=missing_mnar_logit_scale,
    )

    first = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu")
    second = sample_missingness_mask(x, dataset_cfg=cfg, keyed_rng=KeyedRng(seed), device="cpu")

    assert first.shape == x.shape
    assert first.dtype == torch.bool
    assert torch.equal(first, second)
