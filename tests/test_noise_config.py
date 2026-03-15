import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dagzoo.config import (
    NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_LAPLACE,
    NOISE_FAMILY_MIXTURE,
    NOISE_FAMILY_STUDENT_T,
    GeneratorConfig,
)

_NOISE_FAMILIES = (
    NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_LAPLACE,
    NOISE_FAMILY_STUDENT_T,
    NOISE_FAMILY_MIXTURE,
)
_NON_MIXTURE_NOISE_FAMILIES = (
    NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_LAPLACE,
    NOISE_FAMILY_STUDENT_T,
)
_MIXTURE_COMPONENTS = (
    NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_LAPLACE,
    NOISE_FAMILY_STUDENT_T,
)
_TRIM_STRATEGY = st.sampled_from(("", " ", "  ", "\t"))
_POSITIVE_FLOAT_STRATEGY = st.floats(
    min_value=0.125,
    max_value=8.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_STUDENT_T_DF_STRATEGY = st.floats(
    min_value=2.25,
    max_value=16.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _styled_token(draw: st.DrawFn, token: str) -> str:
    chars: list[str] = []
    for char in token:
        if char.isalpha():
            chars.append(draw(st.sampled_from((char.lower(), char.upper()))))
        else:
            chars.append(char)
    prefix = draw(_TRIM_STRATEGY)
    suffix = draw(_TRIM_STRATEGY)
    return f"{prefix}{''.join(chars)}{suffix}"


@st.composite
def _styled_noise_family_strategy(draw: st.DrawFn) -> tuple[str, str]:
    family = draw(st.sampled_from(_NOISE_FAMILIES))
    return _styled_token(draw, family), family


@st.composite
def _styled_non_mixture_noise_family_strategy(draw: st.DrawFn) -> tuple[str, str]:
    family = draw(st.sampled_from(_NON_MIXTURE_NOISE_FAMILIES))
    return _styled_token(draw, family), family


@st.composite
def _styled_mixture_weight_map_strategy(
    draw: st.DrawFn,
) -> tuple[dict[str, float], dict[str, float]]:
    components = draw(
        st.lists(
            st.sampled_from(_MIXTURE_COMPONENTS),
            unique=True,
            min_size=1,
            max_size=len(_MIXTURE_COMPONENTS),
        )
    )
    positive_components = set(
        draw(
            st.lists(
                st.sampled_from(tuple(components)),
                unique=True,
                min_size=1,
                max_size=len(components),
            )
        )
    )
    raw: dict[str, float] = {}
    canonical: dict[str, float] = {}
    for component in components:
        weight = draw(_POSITIVE_FLOAT_STRATEGY) if component in positive_components else 0.0
        canonical[component] = weight
        raw[_styled_token(draw, component)] = weight
    return raw, canonical


def test_noise_config_defaults_from_default_yaml() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    assert cfg.noise.family == NOISE_FAMILY_GAUSSIAN
    assert cfg.noise.base_scale == pytest.approx(1.0)
    assert cfg.noise.student_t_df == pytest.approx(5.0)
    assert cfg.noise.mixture_weights is None


@pytest.mark.parametrize(
    "family",
    [
        NOISE_FAMILY_GAUSSIAN,
        NOISE_FAMILY_LAPLACE,
        NOISE_FAMILY_STUDENT_T,
        NOISE_FAMILY_MIXTURE,
    ],
)
def test_noise_config_accepts_supported_families(family: str) -> None:
    cfg = GeneratorConfig.from_dict({"noise": {"family": family}})
    assert cfg.noise.family == family


def test_noise_config_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="Unsupported noise.family"):
        GeneratorConfig.from_dict({"noise": {"family": "exponential"}})


def test_noise_config_rejects_removed_legacy_family() -> None:
    with pytest.raises(ValueError, match="Unsupported noise.family"):
        GeneratorConfig.from_dict({"noise": {"family": "legacy"}})


def test_noise_config_rejects_student_t_df_at_or_below_two() -> None:
    with pytest.raises(ValueError, match=r"noise\.student_t_df must be a finite value > 2"):
        GeneratorConfig.from_dict({"noise": {"family": "student_t", "student_t_df": 2.0}})


def test_noise_config_rejects_mixture_weights_when_family_not_mixture() -> None:
    with pytest.raises(ValueError, match="noise.mixture_weights is only allowed"):
        GeneratorConfig.from_dict(
            {
                "noise": {
                    "family": "gaussian",
                    "mixture_weights": {"gaussian": 0.5, "laplace": 0.5},
                }
            }
        )


def test_noise_config_normalizes_mixture_weights() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "noise": {
                "family": "mixture",
                "mixture_weights": {"gaussian": 2.0, "laplace": 1.0, "student_t": 1.0},
            }
        }
    )
    weights = cfg.noise.mixture_weights
    assert weights is not None
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["gaussian"] == pytest.approx(0.5)
    assert weights["laplace"] == pytest.approx(0.25)
    assert weights["student_t"] == pytest.approx(0.25)


def test_noise_config_rejects_mixture_weights_with_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unsupported noise.mixture_weights key"):
        GeneratorConfig.from_dict(
            {
                "noise": {
                    "family": "mixture",
                    "mixture_weights": {"gaussian": 1.0, "exponential": 1.0},
                }
            }
        )


def test_noise_config_rejects_mixture_weights_with_nonpositive_total() -> None:
    with pytest.raises(ValueError, match="positive total weight"):
        GeneratorConfig.from_dict(
            {
                "noise": {
                    "family": "mixture",
                    "mixture_weights": {"gaussian": 0.0, "laplace": 0.0, "student_t": 0.0},
                }
            }
        )


def test_noise_config_stably_normalizes_large_mixture_weights() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "noise": {
                "family": "mixture",
                "mixture_weights": {"gaussian": 1e308, "laplace": 1e308},
            }
        }
    )
    weights = cfg.noise.mixture_weights
    assert weights is not None
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["gaussian"] == pytest.approx(0.5)
    assert weights["laplace"] == pytest.approx(0.5)


@settings(max_examples=100, deadline=None)
@given(
    family_case=_styled_noise_family_strategy(),
    base_scale=_POSITIVE_FLOAT_STRATEGY,
    student_t_df=_STUDENT_T_DF_STRATEGY,
)
def test_noise_config_hypothesis_normalizes_supported_families_and_scalars(
    family_case: tuple[str, str],
    base_scale: float,
    student_t_df: float,
) -> None:
    raw_family, canonical_family = family_case
    cfg = GeneratorConfig.from_dict(
        {
            "noise": {
                "family": raw_family,
                "base_scale": base_scale,
                "student_t_df": student_t_df,
            }
        }
    )
    assert cfg.noise.family == canonical_family
    assert cfg.noise.base_scale == pytest.approx(base_scale)
    assert cfg.noise.student_t_df == pytest.approx(student_t_df)


@settings(max_examples=100, deadline=None)
@given(mixture_case=_styled_mixture_weight_map_strategy())
def test_noise_config_hypothesis_normalizes_mixture_weights_and_prunes_zeros(
    mixture_case: tuple[dict[str, float], dict[str, float]],
) -> None:
    raw_weights, canonical_weights = mixture_case
    cfg = GeneratorConfig.from_dict(
        {
            "noise": {
                "family": "mixture",
                "mixture_weights": raw_weights,
            }
        }
    )
    normalized = cfg.noise.mixture_weights
    assert normalized is not None

    expected_positive = {key: value for key, value in canonical_weights.items() if value > 0.0}
    total = math.fsum(expected_positive.values())
    assert set(normalized) == set(expected_positive)
    assert math.isclose(math.fsum(normalized.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    for key, value in expected_positive.items():
        assert normalized[key] == pytest.approx(value / total)


@settings(max_examples=100, deadline=None)
@given(
    family_case=_styled_non_mixture_noise_family_strategy(),
    mixture_case=_styled_mixture_weight_map_strategy(),
)
def test_noise_config_hypothesis_rejects_mixture_weights_for_non_mixture_families(
    family_case: tuple[str, str],
    mixture_case: tuple[dict[str, float], dict[str, float]],
) -> None:
    raw_family, _ = family_case
    raw_weights, _ = mixture_case
    with pytest.raises(ValueError, match="noise.mixture_weights is only allowed"):
        GeneratorConfig.from_dict(
            {
                "noise": {
                    "family": raw_family,
                    "mixture_weights": raw_weights,
                }
            }
        )
