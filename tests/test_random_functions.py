import pytest
import torch
from conftest import make_generator as _make_generator

import dagzoo.functions.random_functions as random_functions_mod
from dagzoo.core.execution_semantics import sample_function_family
from dagzoo.core.fixed_layout.plan_types import GaussianMatrixPlan, LinearFunctionPlan
from dagzoo.core.validation import RetryableDegeneracyError
from dagzoo.functions.random_functions import (
    MechanismFamily,
    _sample_function_family,
    apply_random_function,
)
from dagzoo.rng import KeyedRng


def test_tree_family_survives_nan_feature() -> None:
    g = _make_generator(42)
    x = torch.randn(64, 4, generator=g)
    x[:, 2] = float("nan")

    y = apply_random_function(x, _make_generator(42), out_dim=2, function_type="tree")
    assert y.shape == (64, 2)
    assert torch.all(torch.isfinite(y))

    x = torch.ones(32, 3)
    y = apply_random_function(x, _make_generator(7), out_dim=1, function_type="tree")
    assert y.shape == (32, 1)
    assert torch.all(torch.isfinite(y))


def test_output_shape() -> None:
    g = _make_generator(0)
    x = torch.randn(64, 4, generator=g)
    y = apply_random_function(x, g, out_dim=5)
    assert y.shape == (64, 5)


def test_deterministic() -> None:
    x = torch.randn(32, 4)
    y1 = apply_random_function(x.clone(), _make_generator(0), out_dim=3)
    y2 = apply_random_function(x.clone(), _make_generator(0), out_dim=3)
    torch.testing.assert_close(y1, y2)


def test_deterministic_with_shift_tilt_and_noise_multiplier() -> None:
    x = torch.randn(32, 4)
    y1 = apply_random_function(
        x.clone(),
        _make_generator(8),
        out_dim=3,
        mechanism_logit_tilt=0.7,
        noise_sigma_multiplier=1.35,
    )
    y2 = apply_random_function(
        x.clone(),
        _make_generator(8),
        out_dim=3,
        mechanism_logit_tilt=0.7,
        noise_sigma_multiplier=1.35,
    )
    torch.testing.assert_close(y1, y2)


def test_apply_random_function_output_depends_on_generator_root_nonce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = iter(
        [
            KeyedRng(seed=17, _ambient_nonce=(101, 102, 103)),
            KeyedRng(seed=17, _ambient_nonce=(201, 202, 203)),
        ]
    )
    monkeypatch.setattr(
        random_functions_mod,
        "keyed_rng_from_generator",
        lambda *_args, **_kwargs: next(roots),
    )
    monkeypatch.setattr(
        random_functions_mod,
        "sample_function_plan_for_family",
        lambda *_args, **_kwargs: LinearFunctionPlan(matrix=GaussianMatrixPlan()),
    )

    x = torch.randn(32, 4, generator=_make_generator(11))
    first = apply_random_function(x.clone(), _make_generator(12), out_dim=3, function_type="linear")
    second = apply_random_function(
        x.clone(),
        _make_generator(12),
        out_dim=3,
        function_type="linear",
    )

    assert not torch.equal(first, second)


@pytest.mark.parametrize(
    "family",
    ["linear", "quadratic", "discretization", "gp", "em", "product", "piecewise"],
)
def test_non_finite_inputs_are_sanitized(family: MechanismFamily) -> None:
    x = torch.tensor(
        [
            [0.0, float("nan"), 1.0],
            [float("inf"), -1.0, 2.0],
            [-float("inf"), 0.5, 3.0],
            [1.5, -2.0, 4.0],
            [2.0, 3.0, -5.0],
            [-3.0, 1.0, 6.0],
        ],
        dtype=torch.float32,
    )

    y = apply_random_function(x, _make_generator(9), out_dim=2, function_type=family)
    assert y.shape == (6, 2)
    assert torch.all(torch.isfinite(y))


@pytest.mark.parametrize(
    "family",
    ["nn", "tree", "discretization", "gp", "linear", "quadratic", "em", "product", "piecewise"],
)
def test_each_family(family: MechanismFamily) -> None:
    g = _make_generator(10)
    x = torch.randn(64, 4, generator=g)
    y = apply_random_function(x, _make_generator(10), out_dim=3, function_type=family)
    assert y.shape == (64, 3)
    assert torch.all(torch.isfinite(y))


@pytest.mark.parametrize("family", ["discretization", "quadratic"])
def test_explicit_family_is_deterministic(family: MechanismFamily) -> None:
    x = torch.randn(64, 4, generator=_make_generator(14))
    y1 = apply_random_function(
        x.clone(),
        _make_generator(15),
        out_dim=3,
        function_type=family,
    )
    y2 = apply_random_function(
        x.clone(),
        _make_generator(15),
        out_dim=3,
        function_type=family,
    )
    torch.testing.assert_close(y1, y2)


def test_invalid_type_raises() -> None:
    g = _make_generator()
    x = torch.randn(32, 4, generator=g)
    with pytest.raises(ValueError, match="Unknown random function family"):
        apply_random_function(x, g, out_dim=2, function_type="bogus")  # type: ignore[arg-type]


def test_sampled_family_respects_family_mix_allowlist() -> None:
    g = _make_generator(3)
    mix = {"linear": 1.0}
    for _ in range(16):
        sampled = sample_function_family(
            g,
            mechanism_logit_tilt=0.9,
            function_family_mix=mix,  # type: ignore[arg-type]
        )
        assert sampled == "linear"


def test_sampled_family_skips_zero_probability_families_on_zero_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _make_generator(33)
    mix = {"linear": 1.0}
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.0]))
    sampled = sample_function_family(
        g,
        mechanism_logit_tilt=0.9,
        function_family_mix=mix,  # type: ignore[arg-type]
    )
    assert sampled == "linear"


def test_disallowed_explicit_family_raises_when_mix_is_present() -> None:
    g = _make_generator(4)
    x = torch.randn(32, 4, generator=g)
    with pytest.raises(ValueError, match="is not enabled by mechanism.function_family_mix"):
        apply_random_function(
            x,
            g,
            out_dim=2,
            function_type="gp",
            function_family_mix={"linear": 1.0},  # type: ignore[arg-type]
        )


def test_product_family_without_enabled_component_raises_when_mix_is_present() -> None:
    g = _make_generator(5)
    x = torch.randn(32, 4, generator=g)
    with pytest.raises(ValueError, match="enables 'product' but disables all product component"):
        apply_random_function(
            x,
            g,
            out_dim=2,
            function_type="product",
            function_family_mix={"product": 1.0},  # type: ignore[arg-type]
        )


def test_piecewise_family_without_enabled_component_raises_when_mix_is_present() -> None:
    g = _make_generator(6)
    x = torch.randn(32, 4, generator=g)
    with pytest.raises(
        ValueError,
        match="enables 'piecewise' but disables all piecewise component",
    ):
        apply_random_function(
            x,
            g,
            out_dim=2,
            function_type="piecewise",
            function_family_mix={"piecewise": 1.0},  # type: ignore[arg-type]
        )


def test_deterministic_with_family_mix() -> None:
    x = torch.randn(32, 4)
    mix = {"nn": 0.5, "linear": 0.5}
    y1 = apply_random_function(
        x.clone(),
        _make_generator(6),
        out_dim=3,
        mechanism_logit_tilt=0.7,
        function_family_mix=mix,  # type: ignore[arg-type]
    )
    y2 = apply_random_function(
        x.clone(),
        _make_generator(6),
        out_dim=3,
        mechanism_logit_tilt=0.7,
        function_family_mix=mix,  # type: ignore[arg-type]
    )
    torch.testing.assert_close(y1, y2)


def test_default_sampled_family_never_emits_piecewise() -> None:
    g = _make_generator(16)
    for _ in range(128):
        sampled = sample_function_family(
            g,
            mechanism_logit_tilt=1.0,
            function_family_mix=None,
        )
        assert sampled != "piecewise"


def test_sampled_family_can_emit_piecewise_when_mix_enables_it() -> None:
    g = _make_generator(17)
    mix = {"piecewise": 1.0}
    for _ in range(16):
        sampled = sample_function_family(
            g,
            mechanism_logit_tilt=0.8,
            function_family_mix=mix,  # type: ignore[arg-type]
        )
        assert sampled == "piecewise"


def test_piecewise_explicit_family_with_mix_is_deterministic() -> None:
    x = torch.randn(64, 4, generator=_make_generator(18))
    mix = {"piecewise": 0.5, "linear": 0.5}
    y1 = apply_random_function(
        x.clone(),
        _make_generator(19),
        out_dim=3,
        function_type="piecewise",
        function_family_mix=mix,  # type: ignore[arg-type]
    )
    y2 = apply_random_function(
        x.clone(),
        _make_generator(19),
        out_dim=3,
        function_type="piecewise",
        function_family_mix=mix,  # type: ignore[arg-type]
    )
    torch.testing.assert_close(y1, y2)


def test_random_functions_module_retains_family_sampling_patch_point() -> None:
    g = _make_generator(21)
    sampled = _sample_function_family(
        g,
        mechanism_logit_tilt=0.0,
        function_family_mix={"linear": 1.0},  # type: ignore[arg-type]
    )
    assert sampled == "linear"


def test_apply_random_function_retries_retryable_degeneracy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def _stub_sample_function_plan_for_family(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RetryableDegeneracyError("degenerate_tree_plan")
        return LinearFunctionPlan(matrix=GaussianMatrixPlan())

    monkeypatch.setattr(
        random_functions_mod,
        "sample_function_plan_for_family",
        _stub_sample_function_plan_for_family,
    )

    x = torch.randn(32, 4, generator=_make_generator(22))
    y = apply_random_function(x, _make_generator(23), out_dim=3, function_type="linear")

    assert call_count == 2
    assert y.shape == (32, 3)
    assert torch.all(torch.isfinite(y))
