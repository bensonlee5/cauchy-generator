from collections.abc import Sequence
from dataclasses import dataclass

import pytest
import torch
from conftest import make_generator as _make_generator
from conftest import make_keyed_rng as _make_keyed_rng

import dagzoo.converters.categorical as categorical_mod
import dagzoo.converters.numeric as numeric_mod
import dagzoo.core.execution_sampling_common as execution_sampling_common_mod
import dagzoo.core.execution_semantics as execution_semantics_mod
import dagzoo.core.fixed_layout.batched as fixed_layout_batched_mod
import dagzoo.core.node_pipeline as node_pipeline_mod
import dagzoo.functions.random_functions as random_functions_mod
import dagzoo.sampling.random_points as random_points_mod
from dagzoo.converters.categorical import apply_categorical_converter
from dagzoo.converters.numeric import apply_numeric_converter
from dagzoo.core.execution_semantics import typed_converter_specs
from dagzoo.core.fixed_layout.batched import (
    FixedLayoutBatchRng,
    _apply_categorical_group_batch,
    _apply_node_plan_batch,
    _sample_random_points_batch,
    apply_function_plan_batch,
    apply_numeric_converter_plan_batch,
)
from dagzoo.core.fixed_layout.plan_types import (
    CategoricalConverterPlan,
    DiscretizationFunctionPlan,
    EmFunctionPlan,
    FixedActivationPlan,
    FixedLayoutConverterPlan,
    FixedLayoutExecutionPlan,
    FixedLayoutLatentPlan,
    FixedLayoutNodePlan,
    GaussianMatrixPlan,
    GpFunctionPlan,
    LinearFunctionPlan,
    NeuralNetFunctionPlan,
    NumericConverterPlan,
    ParametricActivationPlan,
    PiecewiseFunctionPlan,
    ProductFunctionPlan,
    QuadraticFunctionPlan,
    RandomPointsNodeSource,
    StackedNodeSource,
    TreeFunctionPlan,
    fixed_layout_converter_groups,
    fixed_layout_signature_payloads,
)
from dagzoo.core.layout_types import MechanismFamily
from dagzoo.core.node_pipeline import apply_node_pipeline
from dagzoo.functions.random_functions import apply_random_function
from dagzoo.rng import KeyedRng
from dagzoo.sampling.random_points import sample_random_points


@dataclass(slots=True)
class ConverterSpec:
    key: str
    kind: str
    dim: int
    cardinality: int | None = None


@pytest.mark.parametrize(
    ("family", "plan"),
    [
        ("linear", LinearFunctionPlan(matrix=GaussianMatrixPlan())),
        ("quadratic", QuadraticFunctionPlan(matrix=GaussianMatrixPlan())),
        (
            "nn",
            NeuralNetFunctionPlan(
                n_layers=2,
                hidden_width=5,
                input_activation=FixedActivationPlan(name="relu"),
                output_activation=None,
                layer_matrices=(GaussianMatrixPlan(), GaussianMatrixPlan()),
                hidden_activations=(FixedActivationPlan(name="tanh"),),
            ),
        ),
        ("tree", TreeFunctionPlan(n_trees=2, depths=(2, 3))),
        (
            "discretization",
            DiscretizationFunctionPlan(
                n_centers=4,
                linear_matrix=GaussianMatrixPlan(),
            ),
        ),
        ("gp", GpFunctionPlan(branch_kind="ha")),
        (
            "em",
            EmFunctionPlan(
                m_val=4,
                linear_matrix=GaussianMatrixPlan(),
            ),
        ),
        (
            "product",
            ProductFunctionPlan(
                lhs=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
                rhs=QuadraticFunctionPlan(matrix=GaussianMatrixPlan()),
            ),
        ),
        (
            "piecewise",
            PiecewiseFunctionPlan(
                gate_matrix=GaussianMatrixPlan(),
                gate_bias=0.25,
                gate_temperature=4.0,
                lhs=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
                rhs=QuadraticFunctionPlan(matrix=GaussianMatrixPlan()),
            ),
        ),
    ],
)
def test_apply_random_function_matches_explicit_plan(
    family: str,
    plan: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(32, 4, generator=_make_generator(1))
    monkeypatch.setattr(
        random_functions_mod,
        "sample_function_plan_for_family",
        lambda *_args, **_kwargs: plan,
    )

    actual_generator = _make_generator(2)
    reference_generator = _make_generator(2)
    actual = apply_random_function(x.clone(), actual_generator, out_dim=3, function_type=family)  # type: ignore[arg-type]

    root = _make_keyed_rng(reference_generator, "apply_random_function")
    rng = FixedLayoutBatchRng.from_keyed_rng(root.keyed("execution"), batch_size=1, device="cpu")
    expected = apply_function_plan_batch(
        random_functions_mod.sanitize_and_standardize(x).unsqueeze(0),
        rng,
        plan,  # type: ignore[arg-type]
        out_dim=3,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
        standardize_input=False,
    ).squeeze(0)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


@pytest.mark.parametrize("family", ["nn", "tree", "discretization", "em"])
def test_sample_function_plan_for_family_uses_generator_device_for_log_uniform(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _make_generator(123)
    calls: list[str] = []
    monkeypatch.setattr(
        execution_semantics_mod,
        "_log_uniform",
        lambda *_args: calls.append(_args[3]) or 5.0,
    )
    monkeypatch.setattr(execution_sampling_common_mod, "_generator_device", lambda *_args: "cuda")
    monkeypatch.setattr(
        execution_semantics_mod.KeyedRng,
        "torch_rng",
        lambda _self, *args, **kwargs: _make_generator(1000),
    )
    monkeypatch.setattr(execution_semantics_mod, "_randint_scalar", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(execution_semantics_mod, "_sample_bool", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        execution_semantics_mod,
        "sample_activation_plan",
        lambda *_args, **_kwargs: FixedActivationPlan(name="relu"),
    )
    monkeypatch.setattr(
        execution_semantics_mod,
        "sample_matrix_plan",
        lambda *_args, **_kwargs: GaussianMatrixPlan(),
    )

    execution_semantics_mod.sample_function_plan_for_family(
        generator,
        family=family,  # type: ignore[arg-type]
        out_dim=4,
        mechanism_logit_tilt=0.0,
        function_family_mix=None,
    )

    assert calls == ["cuda"]


def test_keyed_plan_sampling_uses_explicit_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices: list[str] = []

    monkeypatch.setattr(
        execution_semantics_mod.KeyedRng,
        "torch_rng",
        lambda _self, *args, **kwargs: (
            devices.append(str(kwargs["device"])) or _make_generator(1000)
        ),
    )
    monkeypatch.setattr(
        execution_semantics_mod,
        "sample_function_plan",
        lambda *_args, **_kwargs: LinearFunctionPlan(matrix=GaussianMatrixPlan()),
    )
    monkeypatch.setattr(execution_semantics_mod, "_rand_scalar", lambda *_args, **_kwargs: 1.0)

    root = KeyedRng(124)
    execution_semantics_mod.sample_function_family(
        keyed_rng=root.keyed("family"),
        mechanism_logit_tilt=0.0,
        function_family_mix=None,
        device="cuda",
    )
    execution_semantics_mod.sample_matrix_plan(keyed_rng=root.keyed("matrix"), device="cuda")
    execution_semantics_mod.sample_activation_plan(
        keyed_rng=root.keyed("activation"), device="cuda"
    )
    execution_semantics_mod.sample_converter_plan(
        ConverterSpec(key="feature", kind="cat", dim=3, cardinality=5),
        keyed_rng=root.keyed("converter"),
        mechanism_logit_tilt=0.0,
        function_family_mix=None,
        device="cuda",
    )
    execution_semantics_mod.sample_multi_source_plan(
        keyed_rng=root.keyed("multi"),
        parent_count=2,
        out_dim=3,
        mechanism_logit_tilt=0.0,
        function_family_mix=None,
        device="cuda",
    )
    execution_semantics_mod.sample_root_source_plan(
        keyed_rng=root.keyed("root"),
        out_dim=3,
        mechanism_logit_tilt=0.0,
        function_family_mix=None,
        device="cuda",
    )

    assert devices
    assert all(device == "cuda" for device in devices)


def test_sample_activation_plan_can_emit_gumbel_softmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution_semantics_mod, "_rand_scalar", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(execution_semantics_mod, "_randint_scalar", lambda *_args, **_kwargs: 4)
    monkeypatch.setattr(execution_semantics_mod, "_log_uniform", lambda *_args, **_kwargs: 0.75)

    plan = execution_semantics_mod.sample_activation_plan(
        keyed_rng=KeyedRng(41),
        device="cpu",
    )

    assert isinstance(plan, execution_semantics_mod.ParametricActivationPlan)
    assert plan.kind == "gumbel_softmax"
    assert plan.temperature == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("variant_index", "expected_variant"),
    [
        (0, "standard"),
        (1, "periodic"),
        (2, "multiscale"),
    ],
)
def test_sample_function_plan_for_gp_can_emit_variants(
    variant_index: int,
    expected_variant: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution_semantics_mod, "_sample_bool", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        execution_semantics_mod,
        "_randint_scalar",
        lambda *_args, **_kwargs: variant_index,
    )

    plan = execution_semantics_mod.sample_function_plan_for_family(
        keyed_rng=KeyedRng(51 + variant_index),
        family="gp",
        out_dim=4,
        mechanism_logit_tilt=0.0,
        function_family_mix=None,
        device="cpu",
    )

    assert isinstance(plan, GpFunctionPlan)
    assert plan.branch_kind == "ha"
    assert plan.variant == expected_variant


def test_fixed_layout_signature_payloads_include_gp_variant_and_activation_temperature() -> None:
    execution_plan = FixedLayoutExecutionPlan(
        node_plans=(
            FixedLayoutNodePlan(
                node_index=0,
                parent_indices=(),
                converter_specs=(),
                converter_plans=(),
                converter_groups=(),
                latent=FixedLayoutLatentPlan(required_dim=2, extra_dim=0, total_dim=2),
                source=RandomPointsNodeSource(
                    base_kind="normal",
                    function=NeuralNetFunctionPlan(
                        n_layers=1,
                        hidden_width=3,
                        input_activation=ParametricActivationPlan(
                            kind="gumbel_softmax",
                            temperature=0.75,
                        ),
                        output_activation=None,
                        layer_matrices=(GaussianMatrixPlan(),),
                        hidden_activations=(),
                    ),
                ),
            ),
            FixedLayoutNodePlan(
                node_index=1,
                parent_indices=(0,),
                converter_specs=(),
                converter_plans=(),
                converter_groups=(),
                latent=FixedLayoutLatentPlan(required_dim=2, extra_dim=0, total_dim=2),
                source=RandomPointsNodeSource(
                    base_kind="normal",
                    function=GpFunctionPlan(branch_kind="projected", variant="multiscale"),
                ),
            ),
        )
    )

    payloads = fixed_layout_signature_payloads(execution_plan)

    assert payloads[0]["function"]["input_activation"]["temperature"] == pytest.approx(0.75)
    assert payloads[1]["function"]["variant"] == "multiscale"


def test_keyed_product_rhs_sampling_is_independent_of_lhs_draw_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rhs_probe(lhs_extra_draws: int) -> tuple[int, ...]:
        matrix_call_index = 0
        rhs_tokens: list[tuple[int, ...]] = []

        def fake_sample_matrix_plan(
            generator: torch.Generator | None = None,
            *,
            keyed_rng: KeyedRng | None = None,
            device: str | None = None,
        ) -> GaussianMatrixPlan:
            nonlocal matrix_call_index
            rng, _ = execution_semantics_mod._resolve_sampling_generator(
                generator=generator,
                keyed_rng=keyed_rng,
                device=device,
            )
            if matrix_call_index == 0:
                _ = torch.rand(lhs_extra_draws, generator=rng, device=rng.device)
            else:
                rhs_tokens.append(
                    tuple(
                        int(value)
                        for value in torch.randint(
                            0,
                            2**31,
                            (4,),
                            generator=rng,
                            device=rng.device,
                        ).tolist()
                    )
                )
            matrix_call_index += 1
            return GaussianMatrixPlan()

        monkeypatch.setattr(
            execution_semantics_mod,
            "_sample_product_component_family",
            lambda *args, **kwargs: "linear",
        )
        monkeypatch.setattr(execution_semantics_mod, "sample_matrix_plan", fake_sample_matrix_plan)

        plan = execution_semantics_mod.sample_function_plan_for_family(
            keyed_rng=KeyedRng(125),
            family="product",
            out_dim=4,
            mechanism_logit_tilt=0.0,
            function_family_mix=None,
            device="cpu",
        )

        assert isinstance(plan, ProductFunctionPlan)
        assert isinstance(plan.lhs, LinearFunctionPlan)
        assert isinstance(plan.rhs, LinearFunctionPlan)
        return rhs_tokens[0]

    assert rhs_probe(32) == rhs_probe(1)


def test_sample_function_plan_for_piecewise_uses_non_product_branch_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mix = {"piecewise": 0.2, "linear": 0.4, "quadratic": 0.4}
    sampled_families = iter(["linear", "quadratic"])
    seen_mixes: list[dict[str, float] | None] = []

    def fake_sample_piecewise_component_family(*args, **kwargs):
        seen_mixes.append(kwargs.get("function_family_mix"))
        return next(sampled_families)

    monkeypatch.setattr(
        execution_semantics_mod,
        "_sample_piecewise_component_family",
        fake_sample_piecewise_component_family,
    )
    monkeypatch.setattr(
        execution_semantics_mod,
        "sample_matrix_plan",
        lambda *args, **kwargs: GaussianMatrixPlan(),
    )
    monkeypatch.setattr(execution_semantics_mod, "_log_uniform", lambda *_args: 4.0)
    monkeypatch.setattr(execution_semantics_mod, "_rand_scalar", lambda *_args: 0.75)

    plan = execution_semantics_mod.sample_function_plan_for_family(
        keyed_rng=KeyedRng(129),
        family="piecewise",
        out_dim=3,
        mechanism_logit_tilt=0.0,
        function_family_mix=mix,
        device="cpu",
    )

    assert isinstance(plan, PiecewiseFunctionPlan)
    assert plan.gate_temperature == pytest.approx(4.0)
    assert plan.gate_bias == pytest.approx(0.75)
    assert isinstance(plan.lhs, LinearFunctionPlan)
    assert isinstance(plan.rhs, QuadraticFunctionPlan)
    assert seen_mixes == [mix, mix]


def test_sample_function_plan_for_piecewise_requires_enabled_component_family() -> None:
    with pytest.raises(
        ValueError,
        match="enables 'piecewise' but disables all piecewise component families",
    ):
        execution_semantics_mod.sample_function_plan_for_family(
            keyed_rng=KeyedRng(129),
            family="piecewise",
            out_dim=3,
            mechanism_logit_tilt=0.0,
            function_family_mix={"piecewise": 1.0},
            device="cpu",
        )


def test_keyed_sample_function_plan_preserves_product_subroots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rhs_probe(lhs_extra_draws: int) -> tuple[int, ...]:
        matrix_call_index = 0
        rhs_tokens: list[tuple[int, ...]] = []

        def fake_sample_matrix_plan(
            generator: torch.Generator | None = None,
            *,
            keyed_rng: KeyedRng | None = None,
            device: str | None = None,
        ) -> GaussianMatrixPlan:
            nonlocal matrix_call_index
            rng, _ = execution_semantics_mod._resolve_sampling_generator(
                generator=generator,
                keyed_rng=keyed_rng,
                device=device,
            )
            if matrix_call_index == 0:
                _ = torch.rand(lhs_extra_draws, generator=rng, device=rng.device)
            else:
                rhs_tokens.append(
                    tuple(
                        int(value)
                        for value in torch.randint(
                            0,
                            2**31,
                            (4,),
                            generator=rng,
                            device=rng.device,
                        ).tolist()
                    )
                )
            matrix_call_index += 1
            return GaussianMatrixPlan()

        monkeypatch.setattr(
            execution_semantics_mod,
            "sample_function_family",
            lambda *args, **kwargs: "product",
        )
        monkeypatch.setattr(
            execution_semantics_mod,
            "_sample_product_component_family",
            lambda *args, **kwargs: "linear",
        )
        monkeypatch.setattr(execution_semantics_mod, "sample_matrix_plan", fake_sample_matrix_plan)

        plan = execution_semantics_mod.sample_function_plan(
            keyed_rng=KeyedRng(128),
            out_dim=4,
            mechanism_logit_tilt=0.0,
            function_family_mix=None,
            device="cpu",
        )

        assert isinstance(plan, ProductFunctionPlan)
        assert isinstance(plan.lhs, LinearFunctionPlan)
        assert isinstance(plan.rhs, LinearFunctionPlan)
        return rhs_tokens[0]

    assert rhs_probe(32) == rhs_probe(1)


def test_keyed_multi_parent_sampling_is_independent_across_parent_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sample_bool = execution_semantics_mod._sample_bool

    def parent_probe(first_parent_extra_draws: int) -> tuple[int, ...]:
        parent_call_index = 0
        second_parent_tokens: list[tuple[int, ...]] = []
        combine_call = 0

        def force_stack(generator: torch.Generator, *, p: float = 0.5) -> bool:
            nonlocal combine_call
            if combine_call == 0:
                combine_call += 1
                return False
            return original_sample_bool(generator, p=p)

        def fake_sample_function_plan(
            generator: torch.Generator | None = None,
            *,
            keyed_rng: KeyedRng | None = None,
            out_dim: int,
            mechanism_logit_tilt: float,
            function_family_mix: dict[MechanismFamily, float] | None,
            device: str | None = None,
        ) -> LinearFunctionPlan:
            nonlocal parent_call_index
            rng, _ = execution_semantics_mod._resolve_sampling_generator(
                generator=generator,
                keyed_rng=keyed_rng,
                device=device,
            )
            if parent_call_index == 0:
                _ = torch.rand(first_parent_extra_draws, generator=rng, device=rng.device)
            else:
                second_parent_tokens.append(
                    tuple(
                        int(value)
                        for value in torch.randint(
                            0,
                            2**31,
                            (4,),
                            generator=rng,
                            device=rng.device,
                        ).tolist()
                    )
                )
            parent_call_index += 1
            return LinearFunctionPlan(matrix=GaussianMatrixPlan())

        monkeypatch.setattr(execution_semantics_mod, "_sample_bool", force_stack)
        monkeypatch.setattr(
            execution_semantics_mod,
            "sample_function_plan",
            fake_sample_function_plan,
        )

        source = execution_semantics_mod.sample_multi_source_plan(
            keyed_rng=KeyedRng(126),
            parent_count=2,
            out_dim=4,
            aggregation_kind="sum",
            mechanism_logit_tilt=0.0,
            function_family_mix=None,
            device="cpu",
        )

        assert isinstance(source, StackedNodeSource)
        return second_parent_tokens[0]

    assert parent_probe(32) == parent_probe(1)


def test_keyed_node_plan_sampling_keeps_later_converters_and_source_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter_specs = [
        ConverterSpec(key="feature_0", kind="cat", dim=3, cardinality=5),
        ConverterSpec(key="feature_1", kind="num", dim=1),
    ]

    def sample_with_first_converter_draws(
        first_converter_extra_draws: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        converter_call_index = 0
        second_converter_tokens: list[tuple[int, ...]] = []
        source_tokens: list[tuple[int, ...]] = []

        def fake_sample_latent_plan(
            converter_specs: Sequence[execution_semantics_mod.ConverterSpecLike],
            *,
            generator: torch.Generator | None = None,
            keyed_rng: KeyedRng | None = None,
            device: str,
        ) -> FixedLayoutLatentPlan:
            del converter_specs, generator, keyed_rng, device
            return FixedLayoutLatentPlan(required_dim=4, extra_dim=2, total_dim=6)

        def fake_sample_converter_plan(
            spec: ConverterSpec,
            generator: torch.Generator | None = None,
            *,
            keyed_rng: KeyedRng | None = None,
            mechanism_logit_tilt: float,
            function_family_mix: dict[MechanismFamily, float] | None,
            method_override: str | None = None,
            device: str | None = None,
        ) -> FixedLayoutConverterPlan:
            del spec, mechanism_logit_tilt, function_family_mix, method_override
            nonlocal converter_call_index
            rng, _ = execution_semantics_mod._resolve_sampling_generator(
                generator=generator,
                keyed_rng=keyed_rng,
                device=device,
            )
            if converter_call_index == 0:
                _ = torch.rand(first_converter_extra_draws, generator=rng, device=rng.device)
                plan: FixedLayoutConverterPlan = CategoricalConverterPlan(
                    kind="cat",
                    method="neighbor",
                    variant="center",
                )
            else:
                second_converter_tokens.append(
                    tuple(
                        int(value)
                        for value in torch.randint(
                            0,
                            2**31,
                            (4,),
                            generator=rng,
                            device=rng.device,
                        ).tolist()
                    )
                )
                plan = NumericConverterPlan(kind="num", warp_enabled=False)
            converter_call_index += 1
            return plan

        def fake_sample_root_source_plan(
            generator: torch.Generator | None = None,
            *,
            keyed_rng: KeyedRng | None = None,
            out_dim: int,
            mechanism_logit_tilt: float,
            function_family_mix: dict[MechanismFamily, float] | None,
            device: str | None = None,
        ) -> RandomPointsNodeSource:
            del out_dim, mechanism_logit_tilt, function_family_mix
            rng, _ = execution_semantics_mod._resolve_sampling_generator(
                generator=generator,
                keyed_rng=keyed_rng,
                device=device,
            )
            source_tokens.append(
                tuple(
                    int(value)
                    for value in torch.randint(
                        0,
                        2**31,
                        (4,),
                        generator=rng,
                        device=rng.device,
                    ).tolist()
                )
            )
            return RandomPointsNodeSource(
                base_kind="normal",
                function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            )

        monkeypatch.setattr(
            execution_semantics_mod,
            "sample_latent_plan",
            fake_sample_latent_plan,
        )
        monkeypatch.setattr(
            execution_semantics_mod,
            "sample_converter_plan",
            fake_sample_converter_plan,
        )
        monkeypatch.setattr(
            execution_semantics_mod,
            "sample_root_source_plan",
            fake_sample_root_source_plan,
        )

        node_plan = execution_semantics_mod.sample_node_plan(
            node_index=0,
            parent_indices=(),
            converter_specs=converter_specs,
            keyed_rng=KeyedRng(127),
            device="cpu",
            mechanism_logit_tilt=0.0,
            function_family_mix=None,
        )

        assert node_plan.converter_groups
        return second_converter_tokens[0], source_tokens[0]

    assert sample_with_first_converter_draws(32) == sample_with_first_converter_draws(1)


def test_apply_numeric_converter_matches_explicit_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(24, 1, generator=_make_generator(10))
    plan = NumericConverterPlan(kind="num", warp_enabled=True)
    monkeypatch.setattr(numeric_mod, "sample_converter_plan", lambda *_args, **_kwargs: plan)

    actual_generator = _make_generator(11)
    reference_generator = _make_generator(11)
    actual_x, actual_values = apply_numeric_converter(x.clone(), actual_generator)

    root = _make_keyed_rng(reference_generator, "apply_numeric_converter")
    rng = FixedLayoutBatchRng.from_keyed_rng(root.keyed("execution"), batch_size=1, device="cpu")
    expected_x, expected_values = apply_numeric_converter_plan_batch(
        x.unsqueeze(0),
        rng,
        plan,
    )

    torch.testing.assert_close(actual_x, expected_x.squeeze(0))
    torch.testing.assert_close(actual_values, expected_values.squeeze(0))
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_apply_categorical_converter_matches_explicit_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(24, 3, generator=_make_generator(20))
    plan = CategoricalConverterPlan(
        kind="cat",
        method="neighbor",
        variant="center_random_fn",
        function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
    )
    monkeypatch.setattr(categorical_mod, "sample_converter_plan", lambda *_args, **_kwargs: plan)

    actual_generator = _make_generator(21)
    reference_generator = _make_generator(21)
    actual_x, actual_labels = apply_categorical_converter(
        x.clone(), actual_generator, n_categories=5
    )

    root = _make_keyed_rng(reference_generator, "apply_categorical_converter")
    rng = FixedLayoutBatchRng.from_keyed_rng(root.keyed("execution"), batch_size=1, device="cpu")
    expected_x, expected_labels = _apply_categorical_group_batch(
        x.unsqueeze(0).unsqueeze(2),
        rng,
        plan,
        n_categories=5,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(actual_x, expected_x[0, :, 0, :])
    torch.testing.assert_close(actual_labels, expected_labels[0, :, 0])
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_sample_random_points_matches_explicit_root_source_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = RandomPointsNodeSource(
        base_kind="uniform",
        function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
    )
    monkeypatch.setattr(
        random_points_mod, "sample_root_source_plan", lambda *_args, **_kwargs: source
    )

    actual_generator = _make_generator(30)
    reference_generator = _make_generator(30)
    actual = sample_random_points(32, 4, actual_generator, "cpu")

    root = _make_keyed_rng(reference_generator, "sample_random_points")
    rng = FixedLayoutBatchRng.from_keyed_rng(
        root.keyed("execution", "source"),
        batch_size=1,
        device="cpu",
    )
    base = _sample_random_points_batch(
        rng.keyed("base"),
        n_rows=32,
        dim=4,
        base_kind=source.base_kind,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    expected = apply_function_plan_batch(
        base,
        rng.keyed("function"),
        source.function,
        out_dim=4,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    ).squeeze(0)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_apply_node_pipeline_matches_explicit_node_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [
        ConverterSpec(key="feature_0", kind="num", dim=1),
        ConverterSpec(key="feature_1", kind="cat", dim=3, cardinality=4),
    ]
    typed_specs = typed_converter_specs(specs)
    converter_plans = (
        NumericConverterPlan(kind="num", warp_enabled=True),
        CategoricalConverterPlan(
            kind="cat",
            method="softmax",
            variant="softmax_points",
        ),
    )
    node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(0,),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=4, extra_dim=2, total_dim=6),
        source=StackedNodeSource(
            aggregation_kind="sum",
            parent_functions=(LinearFunctionPlan(matrix=GaussianMatrixPlan()),),
        ),
    )
    monkeypatch.setattr(node_pipeline_mod, "sample_node_plan", lambda **_kwargs: node_plan)

    parent = torch.randn(16, 4, generator=_make_generator(40))
    actual_generator = _make_generator(41)
    reference_generator = _make_generator(41)
    actual_latent, actual_extracted = apply_node_pipeline(
        [parent.clone()],
        16,
        typed_converter_specs(specs),
        actual_generator,
        "cpu",
    )

    root = _make_keyed_rng(reference_generator, "apply_node_pipeline")
    rng = FixedLayoutBatchRng.from_keyed_rng(root.keyed("execution"), batch_size=1, device="cpu")
    expected_latent, expected_extracted = _apply_node_plan_batch(
        None,
        node_plan,
        [parent.unsqueeze(0)],
        n_rows=16,
        rng=rng,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(actual_latent, expected_latent.squeeze(0))
    for key, value in actual_extracted.items():
        torch.testing.assert_close(value, expected_extracted[key].squeeze(0))
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


@pytest.mark.parametrize(("kind", "key"), [("num", "value"), ("target_reg", "target")])
def test_apply_node_plan_replaces_full_multi_column_numeric_slices(
    kind: str,
    key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [ConverterSpec(key=key, kind=kind, dim=3)]  # type: ignore[arg-type]
    typed_specs = typed_converter_specs(specs)
    converter_plans = (NumericConverterPlan(kind=kind, warp_enabled=True),)  # type: ignore[arg-type]
    node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(0,),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=3, extra_dim=0, total_dim=3),
        source=StackedNodeSource(
            aggregation_kind="sum",
            parent_functions=(LinearFunctionPlan(matrix=GaussianMatrixPlan()),),
        ),
    )
    pattern = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).view(1, 1, 3)
    monkeypatch.setattr(
        fixed_layout_batched_mod,
        "apply_numeric_converter_plan_batch",
        lambda x, _rng, _plan: (
            pattern.expand(x.shape[0], x.shape[1], -1).to(x.device),
            torch.full((x.shape[0], x.shape[1]), 5.0, device=x.device),
        ),
    )

    rng = FixedLayoutBatchRng.from_generator(_make_generator(45), batch_size=1, device="cpu")
    latent, extracted = _apply_node_plan_batch(
        None,
        node_plan,
        [torch.randn(1, 8, 3, generator=_make_generator(46))],
        n_rows=8,
        rng=rng,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(latent[0, :, 1], 2.0 * latent[0, :, 0])
    torch.testing.assert_close(latent[0, :, 2], 3.0 * latent[0, :, 0])
    torch.testing.assert_close(
        extracted[key],
        torch.full((1, 8), 5.0, dtype=torch.float32),
    )
