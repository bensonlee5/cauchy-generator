import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dagzoo.config import GeneratorConfig

_MECHANISM_FAMILIES = (
    "nn",
    "tree",
    "discretization",
    "gp",
    "linear",
    "quadratic",
    "em",
    "product",
    "piecewise",
)
_HIGHER_ORDER_FAMILIES = ("product", "piecewise")
_COMPONENT_FAMILIES = ("tree", "discretization", "gp", "linear", "quadratic")
_TRIM_STRATEGY = st.sampled_from(("", " ", "  ", "\t"))
_POSITIVE_FLOAT_STRATEGY = st.floats(
    min_value=0.125,
    max_value=8.0,
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
def _valid_family_mix_case_strategy(
    draw: st.DrawFn,
) -> tuple[dict[str, float], dict[str, float]]:
    families = set(
        draw(
            st.lists(
                st.sampled_from(_MECHANISM_FAMILIES),
                unique=True,
                min_size=1,
                max_size=len(_MECHANISM_FAMILIES),
            )
        )
    )
    positive_families = set(
        draw(
            st.lists(
                st.sampled_from(tuple(families)),
                unique=True,
                min_size=1,
                max_size=len(families),
            )
        )
    )
    if positive_families.intersection(
        _HIGHER_ORDER_FAMILIES
    ) and not positive_families.intersection(_COMPONENT_FAMILIES):
        component = draw(st.sampled_from(_COMPONENT_FAMILIES))
        families.add(component)
        positive_families.add(component)

    raw: dict[str, float] = {}
    canonical: dict[str, float] = {}
    for family in sorted(families):
        weight = draw(_POSITIVE_FLOAT_STRATEGY) if family in positive_families else 0.0
        canonical[family] = weight
        raw[_styled_token(draw, family)] = weight
    return raw, canonical


@st.composite
def _invalid_higher_order_mix_case_strategy(
    draw: st.DrawFn,
) -> tuple[dict[str, float], str]:
    family = draw(st.sampled_from(_HIGHER_ORDER_FAMILIES))
    supporting = draw(
        st.lists(
            st.sampled_from(("nn", "em")),
            unique=True,
            min_size=0,
            max_size=2,
        )
    )
    raw = {_styled_token(draw, family): draw(_POSITIVE_FLOAT_STRATEGY)}
    for other in supporting:
        raw[_styled_token(draw, other)] = draw(_POSITIVE_FLOAT_STRATEGY)
    return raw, family


def test_mechanism_family_mix_defaults_to_none() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    assert cfg.mechanism.function_family_mix is None


def test_mechanism_family_mix_accepts_partial_map_and_normalizes() -> None:
    cfg = GeneratorConfig.from_dict(
        {"mechanism": {"function_family_mix": {"NN": 3.0, "linear": 1.0, "em": 0.0}}}
    )
    mix = cfg.mechanism.function_family_mix
    assert mix is not None
    assert set(mix) == {"nn", "linear"}
    assert mix["nn"] == pytest.approx(0.75)
    assert mix["linear"] == pytest.approx(0.25)
    assert sum(mix.values()) == pytest.approx(1.0)


def test_mechanism_family_mix_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unsupported mechanism.function_family_mix key"):
        GeneratorConfig.from_dict({"mechanism": {"function_family_mix": {"bogus": 1.0}}})


@pytest.mark.parametrize("value", [[], 1, "not-a-map", True])
def test_mechanism_family_mix_rejects_non_mapping(value: object) -> None:
    with pytest.raises(ValueError, match="mechanism.function_family_mix must be a mapping"):
        GeneratorConfig.from_dict({"mechanism": {"function_family_mix": value}})


def test_mechanism_family_mix_rejects_empty_mapping() -> None:
    with pytest.raises(
        ValueError,
        match="mechanism.function_family_mix must include at least one supported family",
    ):
        GeneratorConfig.from_dict({"mechanism": {"function_family_mix": {}}})


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan"), True])
def test_mechanism_family_mix_rejects_invalid_weights(value: float | bool) -> None:
    with pytest.raises(
        ValueError,
        match=r"mechanism\.function_family_mix\.nn must be a finite value >= 0",
    ):
        GeneratorConfig.from_dict({"mechanism": {"function_family_mix": {"nn": value}}})


def test_mechanism_family_mix_rejects_nonpositive_total_weight() -> None:
    with pytest.raises(
        ValueError,
        match="mechanism.function_family_mix must have a positive total weight",
    ):
        GeneratorConfig.from_dict(
            {"mechanism": {"function_family_mix": {"nn": 0.0, "linear": 0.0}}}
        )


def test_mechanism_family_mix_rejects_product_without_component_family() -> None:
    with pytest.raises(
        ValueError,
        match="assigns positive weight to 'product' but none of its component families are enabled",
    ):
        GeneratorConfig.from_dict(
            {"mechanism": {"function_family_mix": {"product": 0.7, "nn": 0.3}}}
        )


def test_mechanism_family_mix_accepts_product_with_component_family() -> None:
    cfg = GeneratorConfig.from_dict(
        {"mechanism": {"function_family_mix": {"product": 0.4, "linear": 0.6}}}
    )
    mix = cfg.mechanism.function_family_mix
    assert mix is not None
    assert math.isclose(sum(mix.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert set(mix) == {"product", "linear"}


@pytest.mark.parametrize(
    ("family_mix", "family"),
    [
        ({"piecewise": 1.0}, "piecewise"),
        ({"piecewise": 0.7, "nn": 0.3}, "piecewise"),
    ],
)
def test_mechanism_family_mix_rejects_piecewise_without_component_family(
    family_mix: dict[str, float],
    family: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"assigns positive weight to '{family}' but none of its component families are enabled",
    ):
        GeneratorConfig.from_dict({"mechanism": {"function_family_mix": family_mix}})


def test_mechanism_family_mix_accepts_piecewise_with_component_family() -> None:
    cfg = GeneratorConfig.from_dict(
        {"mechanism": {"function_family_mix": {"piecewise": 3.0, "linear": 1.0}}}
    )
    mix = cfg.mechanism.function_family_mix
    assert mix is not None
    assert set(mix) == {"piecewise", "linear"}
    assert mix["piecewise"] == pytest.approx(0.75)
    assert mix["linear"] == pytest.approx(0.25)


@settings(max_examples=100, deadline=None)
@given(mix_case=_valid_family_mix_case_strategy())
def test_mechanism_family_mix_hypothesis_normalizes_and_prunes_zero_weights(
    mix_case: tuple[dict[str, float], dict[str, float]],
) -> None:
    raw_mix, canonical_mix = mix_case
    cfg = GeneratorConfig.from_dict({"mechanism": {"function_family_mix": raw_mix}})
    normalized = cfg.mechanism.function_family_mix
    assert normalized is not None

    expected_positive = {key: value for key, value in canonical_mix.items() if value > 0.0}
    total = math.fsum(expected_positive.values())
    assert set(normalized) == set(expected_positive)
    assert math.isclose(math.fsum(normalized.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    for key, value in expected_positive.items():
        assert normalized[key] == pytest.approx(value / total)


@settings(max_examples=100, deadline=None)
@given(mix_case=_invalid_higher_order_mix_case_strategy())
def test_mechanism_family_mix_hypothesis_rejects_missing_component_family_dependencies(
    mix_case: tuple[dict[str, float], str],
) -> None:
    raw_mix, family = mix_case
    with pytest.raises(
        ValueError,
        match=rf"assigns positive weight to '{family}' but none of its component families are enabled",
    ):
        GeneratorConfig.from_dict({"mechanism": {"function_family_mix": raw_mix}})
