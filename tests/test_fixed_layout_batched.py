"""Tests for internal fixed-layout batched helpers."""

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core.execution_semantics import typed_converter_specs
from dagzoo.core.fixed_layout.batched import (
    FixedLayoutBatchRng,
    _apply_activation_plan,
    _apply_node_plan_batch,
    _generate_fixed_layout_raw_batch,
    _lp_distances_to_centers,
    _nearest_lp_center_indices,
    _sample_random_matrix_from_plan_batch,
    apply_function_plan_batch,
    build_fixed_layout_execution_plan,
    generate_fixed_layout_graph_batch,
    generate_fixed_layout_label_batch,
)
from dagzoo.core.fixed_layout.plan_types import (
    ActivationMatrixPlan,
    CategoricalConverterGroup,
    CategoricalConverterPlan,
    FixedActivationPlan,
    FixedLayoutExecutionPlan,
    FixedLayoutLatentPlan,
    FixedLayoutNodePlan,
    GaussianMatrixPlan,
    GpFunctionPlan,
    KernelMatrixPlan,
    LinearFunctionPlan,
    NumericConverterGroup,
    NumericConverterPlan,
    ParametricActivationPlan,
    PiecewiseFunctionPlan,
    RandomPointsNodeSource,
    SingularValuesMatrixPlan,
    WeightsMatrixPlan,
    fixed_layout_converter_groups,
)
from dagzoo.core.fixed_layout.runtime import _sample_fixed_layout
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.functions.activations import _fixed_activation, _gumbel_softmax_activation
from dagzoo.rng import KeyedRng


@dataclass(slots=True)
class ConverterSpec:
    key: str
    kind: str
    dim: int
    cardinality: int | None = None


@pytest.mark.parametrize(
    "name",
    ["relu_sq", "softmax", "onehot_argmax", "argsort", "rank"],
)
def test_apply_activation_plan_fixed_variants_match_flat_reference(name: str) -> None:
    x = torch.randn(2, 5, 4, generator=torch.Generator(device="cpu").manual_seed(7))
    rng = FixedLayoutBatchRng(seed=11, batch_size=2, device="cpu")
    out = _apply_activation_plan(
        x,
        rng,
        FixedActivationPlan(name=name),
        with_standardize=False,
    )
    expected = _fixed_activation(x.reshape(-1, x.shape[-1]), name).reshape_as(x)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("kind", ["relu_pow", "signed_pow", "inv_pow"])
def test_apply_activation_plan_parametric_variants_broadcast_across_matrix_count(
    kind: str,
) -> None:
    x = torch.tensor(
        [
            [
                [[-1.5, -0.5, 0.25], [1.0, 2.0, 3.0]],
                [[-2.0, -1.0, 0.5], [0.75, 1.5, 2.5]],
                [[-3.0, -1.5, 0.75], [1.25, 2.5, 4.0]],
            ],
            [
                [[-1.25, -0.25, 0.4], [1.5, 2.5, 3.5]],
                [[-2.5, -0.75, 0.6], [1.0, 1.75, 2.75]],
                [[-3.5, -1.25, 0.8], [0.5, 1.25, 2.25]],
            ],
        ],
        dtype=torch.float32,
    )
    q = torch.tensor(
        [
            [0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0],
        ],
        dtype=torch.float32,
    )
    rng = FixedLayoutBatchRng(seed=17, batch_size=2, device="cpu")
    with patch.object(
        FixedLayoutBatchRng,
        "log_uniform",
        autospec=True,
        return_value=q,
    ) as mocked_log_uniform:
        out = _apply_activation_plan(
            x,
            rng,
            ParametricActivationPlan(kind=kind),
            with_standardize=False,
        )

    mocked_log_uniform.assert_called_once()
    called_rng, called_shape = mocked_log_uniform.call_args.args
    assert called_shape == (2, 3)
    assert called_rng.batch_size == rng.batch_size
    assert called_rng.device == rng.device
    assert called_rng.keyed_root == rng.keyed_root.keyed(kind)
    assert mocked_log_uniform.call_args.kwargs == {"low": 0.1, "high": 10.0}
    q_view = q.unsqueeze(-1).unsqueeze(-1)
    if kind == "relu_pow":
        expected = torch.pow(torch.clamp(x, min=0.0), q_view)
    elif kind == "signed_pow":
        expected = torch.sign(x) * torch.pow(torch.abs(x), q_view)
    else:
        expected = torch.pow(torch.abs(x) + 1e-3, -q_view)
    torch.testing.assert_close(out, expected)


def test_apply_activation_plan_gumbel_softmax_uses_temperature_and_noise() -> None:
    x = torch.tensor(
        [
            [
                [[-1.5, -0.5, 0.25], [1.0, 2.0, 3.0]],
                [[-2.0, -1.0, 0.5], [0.75, 1.5, 2.5]],
            ],
            [
                [[-1.25, -0.25, 0.4], [1.5, 2.5, 3.5]],
                [[-2.5, -0.75, 0.6], [1.0, 1.75, 2.75]],
            ],
        ],
        dtype=torch.float32,
    )
    uniform = torch.tensor(
        [
            [
                [[0.11, 0.21, 0.31], [0.41, 0.51, 0.61]],
                [[0.71, 0.81, 0.19], [0.29, 0.39, 0.49]],
            ],
            [
                [[0.59, 0.69, 0.79], [0.89, 0.17, 0.27]],
                [[0.37, 0.47, 0.57], [0.67, 0.77, 0.87]],
            ],
        ],
        dtype=torch.float32,
    )
    rng = FixedLayoutBatchRng(seed=19, batch_size=2, device="cpu")
    with patch.object(
        FixedLayoutBatchRng,
        "uniform",
        autospec=True,
        return_value=uniform,
    ) as mocked_uniform:
        out = _apply_activation_plan(
            x,
            rng,
            ParametricActivationPlan(kind="gumbel_softmax", temperature=0.75),
            with_standardize=False,
        )

    mocked_uniform.assert_called_once()
    expected = _gumbel_softmax_activation(
        x,
        temperature=0.75,
        uniform_noise=uniform,
        dim=-1,
    )
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(out.sum(dim=-1), torch.ones_like(out.sum(dim=-1)))


@pytest.mark.parametrize("branch_kind", ["ha", "projected"])
@pytest.mark.parametrize("variant", ["standard", "periodic", "multiscale"])
def test_apply_function_plan_batch_supports_gp_variants_deterministically(
    branch_kind: str,
    variant: str,
) -> None:
    x = torch.randn(2, 12, 4, generator=torch.Generator(device="cpu").manual_seed(29))
    plan = GpFunctionPlan(branch_kind=branch_kind, variant=variant)

    out_a = apply_function_plan_batch(
        x,
        FixedLayoutBatchRng(seed=43, batch_size=2, device="cpu"),
        plan,
        out_dim=3,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    out_b = apply_function_plan_batch(
        x,
        FixedLayoutBatchRng(seed=43, batch_size=2, device="cpu"),
        plan,
        out_dim=3,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    assert out_a.shape == (2, 12, 3)
    assert torch.all(torch.isfinite(out_a))
    torch.testing.assert_close(out_a, out_b)


@pytest.mark.parametrize(
    "plan",
    [
        GaussianMatrixPlan(),
        WeightsMatrixPlan(),
        SingularValuesMatrixPlan(),
        KernelMatrixPlan(),
        ActivationMatrixPlan(base_kind="gaussian", activation=FixedActivationPlan(name="relu")),
    ],
)
def test_sample_random_matrix_from_plan_batch_supports_matrix_count(
    plan: object,
) -> None:
    rng = FixedLayoutBatchRng(seed=13, batch_size=3, device="cpu")
    matrices = _sample_random_matrix_from_plan_batch(
        plan,
        out_dim=4,
        in_dim=3,
        rng=rng,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
        matrix_count=2,
    )
    assert matrices.shape == (3, 2, 4, 3)
    assert torch.all(torch.isfinite(matrices))


def test_sample_random_matrix_from_plan_batch_supports_parametric_activation_with_matrix_count() -> (
    None
):
    rng = FixedLayoutBatchRng(seed=19, batch_size=2, device="cpu")
    matrices = _sample_random_matrix_from_plan_batch(
        ActivationMatrixPlan(
            base_kind="gaussian",
            activation=ParametricActivationPlan(kind="relu_pow"),
        ),
        out_dim=4,
        in_dim=3,
        rng=rng,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
        matrix_count=5,
    )
    assert matrices.shape == (2, 5, 4, 3)
    assert torch.all(torch.isfinite(matrices))


def test_sample_random_matrix_from_plan_batch_kernel_single_column_rows_are_unit_normalized() -> (
    None
):
    rng = FixedLayoutBatchRng(seed=0, batch_size=2, device="cpu")
    matrices = _sample_random_matrix_from_plan_batch(
        KernelMatrixPlan(),
        out_dim=6,
        in_dim=1,
        rng=rng,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    norms = torch.linalg.norm(matrices, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)


def test_apply_function_plan_batch_supports_piecewise() -> None:
    x = torch.randn(2, 12, 4, generator=torch.Generator(device="cpu").manual_seed(37))
    plan = PiecewiseFunctionPlan(
        gate_matrix=GaussianMatrixPlan(),
        gate_bias=0.2,
        gate_temperature=3.0,
        lhs=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        rhs=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
    )

    out_a = apply_function_plan_batch(
        x,
        FixedLayoutBatchRng(seed=41, batch_size=2, device="cpu"),
        plan,
        out_dim=3,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    out_b = apply_function_plan_batch(
        x,
        FixedLayoutBatchRng(seed=41, batch_size=2, device="cpu"),
        plan,
        out_dim=3,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    assert out_a.shape == (2, 12, 3)
    assert torch.all(torch.isfinite(out_a))
    torch.testing.assert_close(out_a, out_b)


def test_fixed_layout_batch_rng_keyed_is_stable_and_flat_equivalent() -> None:
    chained = FixedLayoutBatchRng(seed=23, batch_size=2, device="cpu").keyed("parent").keyed(1)
    flat = FixedLayoutBatchRng(seed=23, batch_size=2, device="cpu").keyed("parent", 1)
    sibling = FixedLayoutBatchRng(seed=23, batch_size=2, device="cpu").keyed("parent", 2)

    torch.testing.assert_close(
        chained.uniform((2, 4), low=0.0, high=1.0),
        flat.uniform((2, 4), low=0.0, high=1.0),
    )
    assert chained.keyed_root == flat.keyed_root
    assert chained.keyed_root != sibling.keyed_root
    assert not torch.equal(
        FixedLayoutBatchRng(seed=23, batch_size=2, device="cpu")
        .keyed("parent", 1)
        .uniform((2, 4), low=0.0, high=1.0),
        FixedLayoutBatchRng(seed=23, batch_size=2, device="cpu")
        .keyed("parent", 2)
        .uniform((2, 4), low=0.0, high=1.0),
    )


def test_fixed_layout_batch_rng_seed_matches_manual_seed_root_stream() -> None:
    seeded = FixedLayoutBatchRng(seed=29, batch_size=2, device="cpu")
    manual_generator = torch.Generator(device="cpu")
    manual_generator.manual_seed(29)
    manual = FixedLayoutBatchRng.from_generator(manual_generator, batch_size=2, device="cpu")

    torch.testing.assert_close(seeded.normal((2, 3)), manual.normal((2, 3)))
    torch.testing.assert_close(
        seeded.uniform((2, 3), low=-1.0, high=1.0),
        manual.uniform((2, 3), low=-1.0, high=1.0),
    )
    torch.testing.assert_close(seeded.randint(0, 7, (2, 3)), manual.randint(0, 7, (2, 3)))
    assert seeded.keyed_root == KeyedRng(29)


def test_build_fixed_layout_execution_plan_uses_keyed_node_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GeneratorConfig.from_yaml("configs/default.yaml")
    config.dataset.task = "regression"

    layout = LayoutPlan(
        n_features=1,
        n_cat=1,
        cat_idx=[0],
        cardinalities=[4],
        card_by_feature={0: 4},
        n_classes=3,
        feature_types=["cat"],
        graph_nodes=2,
        graph_edges=1,
        graph_depth_nodes=2,
        graph_edge_density=0.5,
        adjacency=torch.tensor([[False, True], [False, False]], dtype=torch.bool),
        feature_node_assignment=[0],
        target_node_assignment=1,
    )
    observed_spec_roots: list[tuple[int, int]] = []
    observed_plan_roots: list[tuple[int, int]] = []

    def fake_build_node_specs(
        node_index: int,
        _layout: LayoutPlan,
        task: str,
        keyed_rng: KeyedRng,
    ) -> list[ConverterSpec]:
        assert task == "regression"
        observed_spec_roots.append((node_index, keyed_rng.child_seed("probe")))
        return [ConverterSpec(key=f"feature_{node_index}", kind="num", dim=1)]

    def fake_sample_node_plan(
        *,
        node_index: int,
        parent_indices: tuple[int, ...] | list[int],
        converter_specs: list[ConverterSpec],
        generator: torch.Generator | None = None,
        keyed_rng: KeyedRng | None = None,
        device: str,
        mechanism_logit_tilt: float,
        function_family_mix: dict[str, float] | None,
    ) -> FixedLayoutNodePlan:
        del device, mechanism_logit_tilt, function_family_mix
        assert generator is None
        assert keyed_rng is not None
        observed_plan_roots.append((node_index, keyed_rng.child_seed("probe")))
        typed_specs = typed_converter_specs(converter_specs)
        converter_plans = (NumericConverterPlan(kind="num", warp_enabled=False),)
        return FixedLayoutNodePlan(
            node_index=node_index,
            parent_indices=tuple(int(parent_index) for parent_index in parent_indices),
            converter_specs=typed_specs,
            converter_plans=converter_plans,
            converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
            latent=FixedLayoutLatentPlan(required_dim=1, extra_dim=1, total_dim=2),
            source=RandomPointsNodeSource(
                base_kind="normal",
                function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            ),
        )

    monkeypatch.setattr("dagzoo.core.fixed_layout.batched._build_node_specs", fake_build_node_specs)
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.batched.sample_node_plan",
        fake_sample_node_plan,
    )

    execution_plan = build_fixed_layout_execution_plan(
        config,
        layout,
        plan_seed=31,
        mechanism_logit_tilt=0.0,
    )

    assert len(execution_plan.node_plans) == 2
    assert observed_spec_roots == [
        (0, KeyedRng(31).child_seed("node_spec", 0, "probe")),
        (1, KeyedRng(31).child_seed("node_spec", 1, "probe")),
    ]
    assert observed_plan_roots == [
        (0, KeyedRng(31).child_seed("node_plan", 0, "probe")),
        (1, KeyedRng(31).child_seed("node_plan", 1, "probe")),
    ]


def test_apply_node_plan_batch_grouped_numeric_converters_match_split_execution() -> None:
    typed_specs = typed_converter_specs(
        [
            ConverterSpec(key="feature_0", kind="num", dim=1),
            ConverterSpec(key="feature_1", kind="num", dim=1),
        ]
    )
    converter_plans = (
        NumericConverterPlan(kind="num", warp_enabled=True),
        NumericConverterPlan(kind="num", warp_enabled=False),
    )
    grouped_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=2, extra_dim=1, total_dim=3),
        source=RandomPointsNodeSource(
            base_kind="normal",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    split_plan = FixedLayoutNodePlan(
        node_index=grouped_plan.node_index,
        parent_indices=grouped_plan.parent_indices,
        converter_specs=grouped_plan.converter_specs,
        converter_plans=grouped_plan.converter_plans,
        converter_groups=(
            NumericConverterGroup(spec_indices=(0,)),
            NumericConverterGroup(spec_indices=(1,)),
        ),
        latent=grouped_plan.latent,
        source=grouped_plan.source,
    )

    grouped_latent, grouped_extracted = _apply_node_plan_batch(
        None,
        grouped_plan,
        [],
        n_rows=16,
        rng=FixedLayoutBatchRng(seed=37, batch_size=1, device="cpu"),
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    split_latent, split_extracted = _apply_node_plan_batch(
        None,
        split_plan,
        [],
        n_rows=16,
        rng=FixedLayoutBatchRng(seed=37, batch_size=1, device="cpu"),
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(grouped_latent, split_latent)
    assert set(grouped_extracted) == set(split_extracted) == {"feature_0", "feature_1"}
    for key in grouped_extracted:
        torch.testing.assert_close(grouped_extracted[key], split_extracted[key])


def test_apply_node_plan_batch_keeps_scalar_numeric_groups_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    typed_specs = typed_converter_specs(
        [
            ConverterSpec(key="feature_0", kind="num", dim=1),
            ConverterSpec(key="feature_1", kind="num", dim=1),
        ]
    )
    converter_plans = (
        NumericConverterPlan(kind="num", warp_enabled=True),
        NumericConverterPlan(kind="num", warp_enabled=False),
    )
    grouped_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=2, extra_dim=1, total_dim=3),
        source=RandomPointsNodeSource(
            base_kind="normal",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    calls: list[dict[str, object]] = []

    def _stub_group_batch(
        x: torch.Tensor,
        _rng: FixedLayoutBatchRng,
        warp_enabled: torch.Tensor,
        *,
        spec_indices: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(
            {
                "shape": tuple(int(dim) for dim in x.shape),
                "warp_enabled": warp_enabled.tolist(),
                "spec_indices": spec_indices,
            }
        )
        return x, x[:, :, : x.shape[2]]

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.batched._apply_numeric_converter_group_batch",
        _stub_group_batch,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.batched.apply_numeric_converter_plan_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("per-spec path should not run")
        ),
    )

    _apply_node_plan_batch(
        None,
        grouped_plan,
        [],
        n_rows=8,
        rng=FixedLayoutBatchRng(seed=43, batch_size=1, device="cpu"),
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    assert calls == [{"shape": (1, 8, 2), "warp_enabled": [True, False], "spec_indices": (0, 1)}]


def test_apply_node_plan_batch_grouped_categorical_converters_match_split_execution() -> None:
    typed_specs = typed_converter_specs(
        [
            ConverterSpec(key="feature_0", kind="cat", dim=3, cardinality=5),
            ConverterSpec(key="feature_1", kind="cat", dim=3, cardinality=5),
        ]
    )
    converter_plans = (
        CategoricalConverterPlan(kind="cat", method="softmax", variant="softmax_points"),
        CategoricalConverterPlan(kind="cat", method="softmax", variant="softmax_points"),
    )
    grouped_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=6, extra_dim=2, total_dim=8),
        source=RandomPointsNodeSource(
            base_kind="normal",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    split_plan = FixedLayoutNodePlan(
        node_index=grouped_plan.node_index,
        parent_indices=grouped_plan.parent_indices,
        converter_specs=grouped_plan.converter_specs,
        converter_plans=grouped_plan.converter_plans,
        converter_groups=(
            CategoricalConverterGroup(spec_indices=(0,)),
            CategoricalConverterGroup(spec_indices=(1,)),
        ),
        latent=grouped_plan.latent,
        source=grouped_plan.source,
    )

    grouped_latent, grouped_extracted = _apply_node_plan_batch(
        None,
        grouped_plan,
        [],
        n_rows=16,
        rng=FixedLayoutBatchRng(seed=41, batch_size=1, device="cpu"),
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    split_latent, split_extracted = _apply_node_plan_batch(
        None,
        split_plan,
        [],
        n_rows=16,
        rng=FixedLayoutBatchRng(seed=41, batch_size=1, device="cpu"),
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(grouped_latent, split_latent)
    assert set(grouped_extracted) == set(split_extracted) == {"feature_0", "feature_1"}
    for key in grouped_extracted:
        torch.testing.assert_close(grouped_extracted[key], split_extracted[key])


def test_apply_node_plan_batch_keeps_categorical_groups_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    typed_specs = typed_converter_specs(
        [
            ConverterSpec(key="feature_0", kind="cat", dim=3, cardinality=5),
            ConverterSpec(key="feature_1", kind="cat", dim=3, cardinality=5),
        ]
    )
    converter_plans = (
        CategoricalConverterPlan(kind="cat", method="softmax", variant="softmax_points"),
        CategoricalConverterPlan(kind="cat", method="softmax", variant="softmax_points"),
    )
    grouped_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=6, extra_dim=2, total_dim=8),
        source=RandomPointsNodeSource(
            base_kind="normal",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    calls: list[dict[str, object]] = []

    def _stub_group_batch(
        x: torch.Tensor,
        _rng: FixedLayoutBatchRng,
        _plan: CategoricalConverterPlan,
        *,
        n_categories: int,
        noise_sigma_multiplier: float,
        noise_spec,
        spec_indices: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (noise_sigma_multiplier, noise_spec)
        calls.append(
            {
                "shape": tuple(int(dim) for dim in x.shape),
                "n_categories": n_categories,
                "spec_indices": spec_indices,
            }
        )
        labels = torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2]), dtype=torch.int64, device=x.device
        )
        return x, labels

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.batched._apply_categorical_group_batch",
        _stub_group_batch,
    )

    _apply_node_plan_batch(
        None,
        grouped_plan,
        [],
        n_rows=8,
        rng=FixedLayoutBatchRng(seed=47, batch_size=1, device="cpu"),
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    assert calls == [{"shape": (1, 8, 2, 3), "n_categories": 5, "spec_indices": (0, 1)}]


def test_apply_node_plan_batch_keeps_center_random_fn_groups_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    typed_specs = typed_converter_specs(
        [
            ConverterSpec(key="feature_0", kind="cat", dim=3, cardinality=5),
            ConverterSpec(key="feature_1", kind="cat", dim=3, cardinality=5),
        ]
    )
    converter_plans = (
        CategoricalConverterPlan(
            kind="cat",
            method="neighbor",
            variant="center_random_fn",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
        CategoricalConverterPlan(
            kind="cat",
            method="neighbor",
            variant="center_random_fn",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    grouped_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=6, extra_dim=2, total_dim=8),
        source=RandomPointsNodeSource(
            base_kind="normal",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    group_sizes: list[int] = []

    def _stub_group_batch(
        x: torch.Tensor,
        _rng: FixedLayoutBatchRng,
        _plan: CategoricalConverterPlan,
        *,
        n_categories: int,
        noise_sigma_multiplier: float,
        noise_spec,
        spec_indices: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (n_categories, noise_sigma_multiplier, noise_spec, spec_indices)
        group_sizes.append(int(x.shape[2]))
        labels = torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2]), dtype=torch.int64, device=x.device
        )
        return x, labels

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.batched._apply_categorical_group_batch",
        _stub_group_batch,
    )

    _apply_node_plan_batch(
        None,
        grouped_plan,
        [],
        n_rows=8,
        rng=FixedLayoutBatchRng(seed=53, batch_size=1, device="cpu"),
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    assert group_sizes == [1, 1]


def test_generate_fixed_layout_raw_batch_keys_seeded_batch_rng_per_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2

    layout = LayoutPlan(
        n_features=0,
        n_cat=0,
        cat_idx=[],
        cardinalities=[],
        card_by_feature={},
        n_classes=3,
        feature_types=[],
        graph_nodes=2,
        graph_edges=0,
        graph_depth_nodes=2,
        graph_edge_density=0.0,
        adjacency=torch.zeros((2, 2), dtype=torch.bool),
        feature_node_assignment=[],
        target_node_assignment=1,
    )
    node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(),
        converter_specs=(),
        converter_plans=(),
        converter_groups=(),
        latent=FixedLayoutLatentPlan(required_dim=0, extra_dim=1, total_dim=1),
        source=RandomPointsNodeSource(
            base_kind="normal",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    execution_plan = FixedLayoutExecutionPlan(
        node_plans=(node_plan, node_plan),
    )
    keyed_paths: list[tuple[str | int, ...]] = []

    def _stub_apply_node_plan_batch(
        _config,
        _node_plan,
        _parent_data,
        *,
        n_rows: int,
        rng: FixedLayoutBatchRng,
        device: str,
        noise_sigma_multiplier: float,
        noise_spec,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _ = (_config, _node_plan, _parent_data, device, noise_sigma_multiplier, noise_spec)
        assert rng.keyed_root is not None
        keyed_paths.append(rng.keyed_root.path)
        return torch.zeros((rng.batch_size, n_rows, 1), device=rng.device), {}

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.batched._apply_node_plan_batch",
        _stub_apply_node_plan_batch,
    )

    _generate_fixed_layout_raw_batch(
        cfg,
        layout,
        execution_plan=execution_plan,
        dataset_seeds=[101, 102],
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
        emit_features=False,
    )

    assert keyed_paths == [("node", 0), ("node", 1)]


def test_nearest_lp_center_indices_matches_dense_reference() -> None:
    generator = torch.Generator(device="cpu").manual_seed(23)
    x = torch.randn(2, 3, 7, 4, generator=generator)
    centers = torch.randn(2, 3, 5, 4, generator=generator)
    p = torch.tensor([[0.5, 1.5, 2.0], [3.0, 4.0, 1.25]], dtype=torch.float32)

    out = _nearest_lp_center_indices(x, centers, p=p)
    dense = torch.pow(
        torch.abs(x.unsqueeze(3) - centers.unsqueeze(2)),
        p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
    ).sum(dim=4)
    expected = torch.argmin(dense, dim=3)

    torch.testing.assert_close(out, expected)


def test_lp_distances_to_centers_matches_dense_reference() -> None:
    generator = torch.Generator(device="cpu").manual_seed(29)
    x = torch.randn(2, 11, 3, generator=generator)
    centers = torch.randn(2, 6, 3, generator=generator)
    p = torch.tensor([1.25, 3.0], dtype=torch.float32)

    out = _lp_distances_to_centers(x, centers, p=p, take_root=True)
    expected = torch.pow(
        torch.pow(
            torch.abs(x.unsqueeze(2) - centers.unsqueeze(1)),
            p.view(-1, 1, 1, 1),
        ).sum(dim=3),
        1.0 / p.view(-1, 1, 1),
    )

    torch.testing.assert_close(out, expected)


def test_generate_fixed_layout_label_batch_matches_graph_batch_targets() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "classification"
    cfg.dataset.n_train = 6
    cfg.dataset.n_test = 4
    cfg.dataset.n_features_min = 4
    cfg.dataset.n_features_max = 4
    cfg.dataset.n_classes_min = 3
    cfg.dataset.n_classes_max = 3
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 3
    plan = _sample_fixed_layout(cfg, seed=123, device="cpu")
    dataset_seeds = [901, 902]

    x_batch, y_batch, _aux = generate_fixed_layout_graph_batch(
        cfg,
        plan.layout,
        execution_plan=plan.execution_plan,
        dataset_seeds=dataset_seeds,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    label_batch, _aux_only = generate_fixed_layout_label_batch(
        cfg,
        plan.layout,
        execution_plan=plan.execution_plan,
        dataset_seeds=dataset_seeds,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    assert x_batch.shape[0] == label_batch.shape[0] == len(dataset_seeds)
    torch.testing.assert_close(label_batch, y_batch)
