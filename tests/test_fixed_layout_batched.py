"""Tests for internal fixed-layout batched helpers."""

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
from conftest import load_repo_config

import dagzoo.core.fixed_layout.batched as fixed_layout_batched
from dagzoo.core.execution_semantics import typed_converter_specs
from dagzoo.core.fixed_layout.batch_functions import (
    _apply_leaf_pair_batch as _real_apply_leaf_pair_batch,
)
from dagzoo.core.fixed_layout.batched import (
    FixedLayoutBatchRng,
    _apply_activation_plan,
    _apply_node_plan_batch,
    _compile_function_execution_context,
    _generate_fixed_layout_graph_batch_prepared,
    _generate_fixed_layout_raw_batch,
    _generate_fixed_layout_validation_label_batch,
    _generate_mixed_fixed_layout_graph_batch_prepared,
    _lp_distances_to_centers,
    _MixedPreparedLogicalCohort,
    _nearest_lp_center_indices,
    _prepare_fixed_layout_execution_context,
    _sample_random_matrix_from_plan_batch,
    apply_compiled_function_plan_batch,
    apply_function_plan_batch,
    build_fixed_layout_execution_plan,
    generate_fixed_layout_graph_batch,
    generate_fixed_layout_label_batch,
    mixed_execution_cohort_compatibility_sketch,
)
from dagzoo.core.fixed_layout.plan_types import (
    ActivationMatrixPlan,
    CategoricalConverterGroup,
    CategoricalConverterPlan,
    ConcatNodeSource,
    DiscretizationFunctionPlan,
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
    ProductFunctionPlan,
    QuadraticFunctionPlan,
    RandomPointsNodeSource,
    SingularValuesMatrixPlan,
    StackedNodeSource,
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


def _mixed_executor_test_config():
    cfg = load_repo_config()
    cfg.dataset.task = "regression"
    cfg.filter.enabled = False
    cfg.dataset.n_train = 8
    cfg.dataset.n_test = 4
    cfg.dataset.n_features_min = 5
    cfg.dataset.n_features_max = 5
    cfg.graph.n_nodes_min = 4
    cfg.graph.n_nodes_max = 4
    cfg.mechanism.function_family_mix = {
        "linear": 1.0,
        "quadratic": 1.0,
        "nn": 1.0,
        "tree": 1.0,
        "discretization": 1.0,
        "gp": 1.0,
        "em": 1.0,
    }
    return cfg


def _distinct_leaf_only_plans(cfg, *, count: int) -> list:
    plans = []
    seen_signatures: set[str] = set()
    for seed in range(100, 4096):
        plan = _sample_fixed_layout(cfg, seed=seed, device="cpu")
        signature = str(plan.plan_signature)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        plans.append(plan)
        if len(plans) == count:
            return plans
    raise AssertionError(f"Expected {count} distinct fixed-layout plans for mixed executor test.")


def _layout_stub(
    *,
    feature_types: list[str],
    graph_nodes: int,
    adjacency: torch.Tensor,
    feature_node_assignment: list[int],
    target_node_assignment: int,
) -> LayoutPlan:
    graph_edges = int(adjacency.to(dtype=torch.int64).sum().item())
    density_denominator = graph_nodes * max(graph_nodes - 1, 1)
    graph_edge_density = float(graph_edges) / float(density_denominator) if graph_nodes > 1 else 0.0
    return LayoutPlan(
        n_features=len(feature_types),
        n_cat=0,
        cat_idx=[],
        cardinalities=[],
        card_by_feature={},
        n_classes=3,
        feature_types=list(feature_types),
        graph_nodes=int(graph_nodes),
        graph_edges=graph_edges,
        graph_depth_nodes=int(graph_nodes),
        graph_edge_density=graph_edge_density,
        adjacency=adjacency,
        feature_node_assignment=list(feature_node_assignment),
        target_to_node=int(target_node_assignment),
    )


def _product_function_variant(variant_index: int) -> ProductFunctionPlan:
    variants = (
        ProductFunctionPlan(
            lhs=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            rhs=QuadraticFunctionPlan(matrix=SingularValuesMatrixPlan()),
        ),
        ProductFunctionPlan(
            lhs=GpFunctionPlan(branch_kind="projected", variant="periodic"),
            rhs=LinearFunctionPlan(matrix=WeightsMatrixPlan()),
        ),
        ProductFunctionPlan(
            lhs=DiscretizationFunctionPlan(
                n_centers=4,
                linear_matrix=GaussianMatrixPlan(),
            ),
            rhs=QuadraticFunctionPlan(matrix=WeightsMatrixPlan()),
        ),
    )
    return variants[int(variant_index) % len(variants)]


def _piecewise_function_variant(variant_index: int) -> PiecewiseFunctionPlan:
    variants = (
        PiecewiseFunctionPlan(
            gate_matrix=GaussianMatrixPlan(),
            gate_bias=0.15,
            gate_temperature=1.2,
            lhs=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            rhs=LinearFunctionPlan(matrix=SingularValuesMatrixPlan()),
        ),
        PiecewiseFunctionPlan(
            gate_matrix=WeightsMatrixPlan(),
            gate_bias=-0.25,
            gate_temperature=1.6,
            lhs=GpFunctionPlan(branch_kind="projected", variant="periodic"),
            rhs=QuadraticFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
        PiecewiseFunctionPlan(
            gate_matrix=GaussianMatrixPlan(),
            gate_bias=0.4,
            gate_temperature=0.9,
            lhs=DiscretizationFunctionPlan(
                n_centers=3,
                linear_matrix=WeightsMatrixPlan(),
            ),
            rhs=GpFunctionPlan(branch_kind="ha", variant="multiscale"),
        ),
    )
    return variants[int(variant_index) % len(variants)]


def _nested_piecewise_product_variant(variant_index: int) -> PiecewiseFunctionPlan:
    return PiecewiseFunctionPlan(
        gate_matrix=GaussianMatrixPlan() if int(variant_index) % 2 == 0 else WeightsMatrixPlan(),
        gate_bias=(-0.2 if int(variant_index) % 2 == 0 else 0.35),
        gate_temperature=(1.4 if int(variant_index) % 2 == 0 else 0.85),
        lhs=_product_function_variant(int(variant_index)),
        rhs=_product_function_variant(int(variant_index) + 1),
    )


def _kernel_compatible_piecewise_variant(variant_index: int) -> PiecewiseFunctionPlan:
    return PiecewiseFunctionPlan(
        gate_matrix=GaussianMatrixPlan()
        if int(variant_index) % 2 == 0
        else KernelMatrixPlan(
            gamma=0.75 + 0.1 * int(variant_index), signed=bool(variant_index % 2)
        ),
        gate_bias=0.2,
        gate_temperature=1.1,
        lhs=LinearFunctionPlan(
            matrix=GaussianMatrixPlan() if int(variant_index) % 2 == 0 else WeightsMatrixPlan()
        ),
        rhs=QuadraticFunctionPlan(
            matrix=SingularValuesMatrixPlan()
            if int(variant_index) % 2 == 0
            else KernelMatrixPlan(gamma=1.3, signed=False)
        ),
    )


def _numeric_node_plan(
    node_index: int,
    *,
    parent_indices: tuple[int, ...],
    source,
    feature_key: str | None,
    target_key: str | None,
    latent_total_dim: int,
) -> FixedLayoutNodePlan:
    spec_defs: list[ConverterSpec] = []
    converter_plans = []
    if feature_key is not None:
        spec_defs.append(ConverterSpec(key=feature_key, kind="num", dim=1))
        converter_plans.append(NumericConverterPlan(kind="num", warp_enabled=False))
    if target_key is not None:
        spec_defs.append(ConverterSpec(key=target_key, kind="target_reg", dim=1))
        converter_plans.append(NumericConverterPlan(kind="target_reg", warp_enabled=False))
    converter_specs = typed_converter_specs(spec_defs)
    converter_plan_tuple = tuple(converter_plans)
    required_dim = sum(int(spec.dim) for spec in converter_specs)
    return FixedLayoutNodePlan(
        node_index=int(node_index),
        parent_indices=tuple(int(parent_index) for parent_index in parent_indices),
        converter_specs=converter_specs,
        converter_plans=converter_plan_tuple,
        converter_groups=fixed_layout_converter_groups(converter_specs, converter_plan_tuple),
        latent=FixedLayoutLatentPlan(
            required_dim=int(required_dim),
            extra_dim=max(0, int(latent_total_dim) - int(required_dim)),
            total_dim=int(latent_total_dim),
        ),
        source=source,
    )


def _random_points_execution_plan(
    function_plan,
    *,
    latent_total_dim: int = 3,
) -> tuple[LayoutPlan, FixedLayoutExecutionPlan]:
    layout = _layout_stub(
        feature_types=["num"],
        graph_nodes=1,
        adjacency=torch.zeros((1, 1), dtype=torch.bool),
        feature_node_assignment=[0],
        target_node_assignment=0,
    )
    node_plan = _numeric_node_plan(
        0,
        parent_indices=(),
        source=RandomPointsNodeSource(
            base_kind="normal",
            function=function_plan,
        ),
        feature_key="feature_0",
        target_key="target",
        latent_total_dim=latent_total_dim,
    )
    return layout, FixedLayoutExecutionPlan(node_plans=(node_plan,))


def _multi_input_execution_plan(
    *,
    source_kind: str,
    function_plan=None,
    parent_functions: tuple | None = None,
    latent_total_dim: int = 3,
) -> tuple[LayoutPlan, FixedLayoutExecutionPlan]:
    layout = _layout_stub(
        feature_types=["num", "num", "num"],
        graph_nodes=3,
        adjacency=torch.tensor(
            [
                [False, False, True],
                [False, False, True],
                [False, False, False],
            ],
            dtype=torch.bool,
        ),
        feature_node_assignment=[0, 1, 2],
        target_node_assignment=2,
    )
    root_a = _numeric_node_plan(
        0,
        parent_indices=(),
        source=RandomPointsNodeSource(
            base_kind="normal",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
        feature_key="feature_0",
        target_key=None,
        latent_total_dim=2,
    )
    root_b = _numeric_node_plan(
        1,
        parent_indices=(),
        source=RandomPointsNodeSource(
            base_kind="uniform",
            function=QuadraticFunctionPlan(matrix=WeightsMatrixPlan()),
        ),
        feature_key="feature_1",
        target_key=None,
        latent_total_dim=2,
    )
    if source_kind == "concat":
        source = ConcatNodeSource(function=function_plan)
    elif source_kind == "stacked":
        if parent_functions is None:
            raise ValueError("stacked execution plans require parent_functions.")
        source = StackedNodeSource(
            aggregation_kind="sum",
            parent_functions=parent_functions,
        )
    else:
        raise ValueError(f"Unsupported multi-input source kind: {source_kind!r}")
    target_node = _numeric_node_plan(
        2,
        parent_indices=(0, 1),
        source=source,
        feature_key="feature_2",
        target_key="target",
        latent_total_dim=latent_total_dim,
    )
    return layout, FixedLayoutExecutionPlan(node_plans=(root_a, root_b, target_node))


def _assert_mixed_prepared_matches_exact_cohort_batches(
    cfg,
    cohort_specs: list[tuple[LayoutPlan, FixedLayoutExecutionPlan, tuple[int, ...]]],
) -> None:
    expected_x_batches: list[torch.Tensor] = []
    expected_y_batches: list[torch.Tensor] = []
    expected_aux_meta: list[dict[str, object]] = []
    mixed_cohorts: list[_MixedPreparedLogicalCohort] = []
    for layout, execution_plan, dataset_seeds in cohort_specs:
        prepared = _prepare_fixed_layout_execution_context(layout, execution_plan)
        x_batch, y_batch, aux_meta_batch = _generate_fixed_layout_graph_batch_prepared(
            cfg,
            layout,
            execution_plan=execution_plan,
            prepared_execution_context=prepared,
            dataset_seeds=list(dataset_seeds),
            device="cpu",
            noise_sigma_multiplier=1.0,
            noise_spec=None,
        )
        expected_x_batches.append(x_batch)
        expected_y_batches.append(y_batch)
        expected_aux_meta.extend(aux_meta_batch)
        mixed_cohorts.append(
            _MixedPreparedLogicalCohort(
                layout=layout,
                execution_plan=execution_plan,
                prepared_execution_context=prepared,
                dataset_seeds=tuple(int(seed) for seed in dataset_seeds),
                noise_spec=None,
                task="regression",
                n_rows=int(cfg.dataset.n_train + cfg.dataset.n_test),
            )
        )

    observed_x, observed_y, observed_aux_meta = _generate_mixed_fixed_layout_graph_batch_prepared(
        mixed_cohorts,
        device="cpu",
        noise_sigma_multiplier=1.0,
        runtime_metrics_out={},
    )

    torch.testing.assert_close(observed_x, torch.cat(expected_x_batches, dim=0))
    torch.testing.assert_close(observed_y, torch.cat(expected_y_batches, dim=0))
    assert observed_aux_meta == expected_aux_meta


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


def test_apply_compiled_function_plan_batch_matches_recursive_higher_order_execution() -> None:
    x = torch.randn(2, 12, 4, generator=torch.Generator(device="cpu").manual_seed(31))
    plan = PiecewiseFunctionPlan(
        gate_matrix=GaussianMatrixPlan(),
        gate_bias=0.15,
        gate_temperature=1.2,
        lhs=ProductFunctionPlan(
            lhs=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            rhs=QuadraticFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
        rhs=ProductFunctionPlan(
            lhs=GpFunctionPlan(branch_kind="projected", variant="periodic"),
            rhs=LinearFunctionPlan(matrix=SingularValuesMatrixPlan()),
        ),
    )
    compiled = _compile_function_execution_context(plan, root_rng_path=("function",))

    out_recursive = apply_function_plan_batch(
        x,
        FixedLayoutBatchRng(seed=43, batch_size=2, device="cpu").keyed("function"),
        plan,
        out_dim=3,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    out_compiled = apply_compiled_function_plan_batch(
        x,
        FixedLayoutBatchRng(seed=43, batch_size=2, device="cpu"),
        compiled,
        out_dim=3,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(out_recursive, out_compiled)


@pytest.mark.parametrize(
    ("plan", "out_dim"),
    [
        (
            ProductFunctionPlan(
                lhs=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
                rhs=LinearFunctionPlan(matrix=SingularValuesMatrixPlan()),
            ),
            3,
        ),
        (
            PiecewiseFunctionPlan(
                gate_matrix=GaussianMatrixPlan(),
                gate_bias=-0.2,
                gate_temperature=1.4,
                lhs=GpFunctionPlan(branch_kind="projected", variant="periodic"),
                rhs=GpFunctionPlan(branch_kind="projected", variant="periodic"),
            ),
            2,
        ),
    ],
)
def test_apply_compiled_function_plan_batch_uses_fused_leaf_pairs_without_output_drift(
    monkeypatch: pytest.MonkeyPatch,
    plan,
    out_dim: int,
) -> None:
    x = torch.randn(2, 10, 4, generator=torch.Generator(device="cpu").manual_seed(37))
    compiled = _compile_function_execution_context(plan, root_rng_path=("function",))
    fused_family_calls: list[tuple[str, str]] = []

    def _tracking_apply_leaf_pair_batch(*args, **kwargs):
        fused_family_calls.append(
            (
                type(kwargs["plans"][0]).__name__,
                type(kwargs["plans"][1]).__name__,
            )
        )
        return _real_apply_leaf_pair_batch(*args, **kwargs)

    monkeypatch.setattr(
        fixed_layout_batched,
        "_apply_leaf_pair_batch",
        _tracking_apply_leaf_pair_batch,
    )

    out_recursive = apply_function_plan_batch(
        x,
        FixedLayoutBatchRng(seed=41, batch_size=2, device="cpu").keyed("function"),
        plan,
        out_dim=out_dim,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    out_compiled = apply_compiled_function_plan_batch(
        x,
        FixedLayoutBatchRng(seed=41, batch_size=2, device="cpu"),
        compiled,
        out_dim=out_dim,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    assert fused_family_calls
    torch.testing.assert_close(out_recursive, out_compiled)


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


def test_sample_random_matrix_from_plan_batch_kernel_plan_uses_gamma_and_signed() -> None:
    unsigned_a = _sample_random_matrix_from_plan_batch(
        KernelMatrixPlan(gamma=0.25, signed=False),
        out_dim=4,
        in_dim=3,
        rng=FixedLayoutBatchRng(seed=23, batch_size=2, device="cpu"),
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    unsigned_b = _sample_random_matrix_from_plan_batch(
        KernelMatrixPlan(gamma=0.25, signed=False),
        out_dim=4,
        in_dim=3,
        rng=FixedLayoutBatchRng(seed=23, batch_size=2, device="cpu"),
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    higher_gamma = _sample_random_matrix_from_plan_batch(
        KernelMatrixPlan(gamma=4.0, signed=False),
        out_dim=4,
        in_dim=3,
        rng=FixedLayoutBatchRng(seed=23, batch_size=2, device="cpu"),
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    signed = _sample_random_matrix_from_plan_batch(
        KernelMatrixPlan(gamma=0.25, signed=True),
        out_dim=4,
        in_dim=3,
        rng=FixedLayoutBatchRng(seed=23, batch_size=2, device="cpu"),
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(unsigned_a, unsigned_b)
    assert torch.all(torch.isfinite(unsigned_a))
    assert not torch.allclose(unsigned_a, higher_gamma)
    assert not torch.allclose(unsigned_a, signed)


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


def test_fixed_layout_batch_rng_keyed_reuses_templates_without_advancing_repeat_calls() -> None:
    root = FixedLayoutBatchRng(seed=23, batch_size=2, device="cpu")
    first = root.keyed("parent", 1)
    _ = first.uniform((2, 4), low=0.0, high=1.0)

    draws_a = root.keyed("parent", 1).uniform((2, 4), low=0.0, high=1.0)
    draws_b = root.keyed("parent", 1).uniform((2, 4), low=0.0, high=1.0)

    torch.testing.assert_close(draws_a, draws_b)


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
    config = load_repo_config()
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
        target_to_node=1,
    )
    observed_spec_roots: list[tuple[int, int]] = []
    observed_plan_roots: list[tuple[int, int]] = []

    def fake_build_node_specs(
        node_index: int,
        _layout: LayoutPlan,
        keyed_rng: KeyedRng,
    ) -> list[ConverterSpec]:
        observed_spec_roots.append((node_index, keyed_rng.child_seed("probe")))
        return [ConverterSpec(key=f"feature_{node_index}", kind="num", dim=1)]

    def fake_sample_node_plan(
        *,
        node_index: int,
        parent_indices: tuple[int, ...] | list[int],
        parent_output_dims: tuple[int, ...] | list[int] | None = None,
        converter_specs: list[ConverterSpec],
        generator: torch.Generator | None = None,
        keyed_rng: KeyedRng | None = None,
        device: str,
        mechanism_logit_tilt: float,
        function_family_mix: dict[str, float] | None,
        stress_profile_name: str | None = None,
    ) -> FixedLayoutNodePlan:
        del (
            parent_output_dims,
            device,
            mechanism_logit_tilt,
            function_family_mix,
            stress_profile_name,
        )
        assert generator is None
        assert keyed_rng is not None
        observed_plan_roots.append((node_index, keyed_rng.child_seed("probe")))
        typed_specs = typed_converter_specs(converter_specs)
        converter_plans = tuple(
            NumericConverterPlan(
                kind="target_reg" if spec.kind == "target_reg" else "num",
                warp_enabled=False,
            )
            for spec in typed_specs
        )
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
    assert execution_plan.node_plans[0].compiled_converter_groups
    assert execution_plan.node_plans[0].compiled_converter_groups[0].spec_indices == (0,)
    assert [spec.key for spec in execution_plan.node_plans[1].converter_specs] == [
        "feature_1",
        "target",
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
        class_probs_out: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (noise_sigma_multiplier, noise_spec, class_probs_out)
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
        class_probs_out: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (n_categories, noise_sigma_multiplier, noise_spec, spec_indices, class_probs_out)
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
    cfg = load_repo_config()
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
        target_to_node=1,
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
    target_node_plan = FixedLayoutNodePlan(
        node_index=1,
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
        node_plans=(node_plan, target_node_plan),
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
        runtime_metrics_out=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _ = (
            _config,
            _node_plan,
            _parent_data,
            device,
            noise_sigma_multiplier,
            noise_spec,
            runtime_metrics_out,
        )
        assert rng.keyed_root is not None
        keyed_paths.append(rng.keyed_root.path)
        extracted = {"target": torch.zeros((rng.batch_size, n_rows), device=rng.device)}
        if int(_node_plan.node_index) != int(layout.target_to_node):
            extracted = {}
        return torch.zeros((rng.batch_size, n_rows, 1), device=rng.device), extracted

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


def test_generate_fixed_layout_raw_batch_reports_runtime_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_repo_config()
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2

    layout = LayoutPlan(
        n_features=1,
        n_cat=0,
        cat_idx=[],
        cardinalities=[],
        card_by_feature={},
        n_classes=3,
        feature_types=["num"],
        graph_nodes=1,
        graph_edges=0,
        graph_depth_nodes=1,
        graph_edge_density=0.0,
        adjacency=torch.zeros((1, 1), dtype=torch.bool),
        feature_node_assignment=[0],
        target_to_node=0,
    )
    typed_specs = typed_converter_specs(
        [
            ConverterSpec(key="feature_0", kind="num", dim=1),
            ConverterSpec(key="target", kind="target_reg", dim=1),
        ]
    )
    node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(),
        converter_specs=typed_specs,
        converter_plans=(
            NumericConverterPlan(kind="num", warp_enabled=False),
            NumericConverterPlan(kind="target_reg", warp_enabled=False),
        ),
        converter_groups=(NumericConverterGroup(spec_indices=(0, 1)),),
        latent=FixedLayoutLatentPlan(required_dim=2, extra_dim=0, total_dim=2),
        source=RandomPointsNodeSource(
            base_kind="normal",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    execution_plan = FixedLayoutExecutionPlan(node_plans=(node_plan,))

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
        runtime_metrics_out=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _ = (
            _config,
            _node_plan,
            _parent_data,
            device,
            noise_sigma_multiplier,
            noise_spec,
        )
        if runtime_metrics_out is not None:
            runtime_metrics_out["node_apply_elapsed_seconds"] = (
                float(runtime_metrics_out.get("node_apply_elapsed_seconds", 0.0)) + 1.25
            )
            runtime_metrics_out["node_apply_cpu_time_seconds"] = (
                float(runtime_metrics_out.get("node_apply_cpu_time_seconds", 0.0)) + 0.75
            )
            runtime_metrics_out["converter_elapsed_seconds"] = (
                float(runtime_metrics_out.get("converter_elapsed_seconds", 0.0)) + 0.25
            )
            runtime_metrics_out["converter_cpu_time_seconds"] = (
                float(runtime_metrics_out.get("converter_cpu_time_seconds", 0.0)) + 0.1
            )
        return (
            torch.zeros((rng.batch_size, n_rows, 2), device=rng.device),
            {
                "feature_0": torch.zeros((rng.batch_size, n_rows), device=rng.device),
                "target": torch.zeros((rng.batch_size, n_rows), device=rng.device),
            },
        )

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.batched._apply_node_plan_batch",
        _stub_apply_node_plan_batch,
    )

    runtime_metrics: dict[str, float] = {}
    x, y, _aux_meta_batch = _generate_fixed_layout_raw_batch(
        cfg,
        layout,
        execution_plan=execution_plan,
        dataset_seeds=[201, 202],
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
        emit_features=True,
        runtime_metrics_out=runtime_metrics,
    )

    assert x is not None
    assert x.shape == (2, 6, 1)
    assert y.shape == (2, 6)
    assert runtime_metrics["node_apply_elapsed_seconds"] == pytest.approx(1.25)
    assert runtime_metrics["node_apply_cpu_time_seconds"] == pytest.approx(0.75)
    assert runtime_metrics["converter_elapsed_seconds"] == pytest.approx(0.25)
    assert runtime_metrics["converter_cpu_time_seconds"] == pytest.approx(0.1)
    assert runtime_metrics["feature_materialization_elapsed_seconds"] >= 0.0
    assert runtime_metrics["feature_materialization_cpu_time_seconds"] >= 0.0
    assert runtime_metrics["raw_batch_elapsed_seconds"] >= 0.0
    assert runtime_metrics["raw_batch_cpu_time_seconds"] >= 0.0


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
    cfg = load_repo_config()
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


def test_generate_fixed_layout_graph_batch_prepared_matches_public_graph_batch() -> None:
    cfg = load_repo_config()
    cfg.dataset.task = "regression"
    cfg.filter.enabled = False
    cfg.dataset.n_train = 12
    cfg.dataset.n_test = 4
    cfg.dataset.n_features_min = 5
    cfg.dataset.n_features_max = 5
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 5
    plan = _sample_fixed_layout(cfg, seed=211, device="cpu")
    assert plan.prepared_execution_context is not None
    dataset_seeds = [1001, 1002]

    expected_x, expected_y, _ = generate_fixed_layout_graph_batch(
        cfg,
        plan.layout,
        execution_plan=plan.execution_plan,
        dataset_seeds=dataset_seeds,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    observed_x, observed_y, _ = _generate_fixed_layout_graph_batch_prepared(
        cfg,
        plan.layout,
        execution_plan=plan.execution_plan,
        prepared_execution_context=plan.prepared_execution_context,
        dataset_seeds=dataset_seeds,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(observed_x, expected_x)
    torch.testing.assert_close(observed_y, expected_y)


def test_mixed_execution_cohort_compatibility_sketch_collapses_matrix_only_variants() -> None:
    layout_a, execution_plan_a = _random_points_execution_plan(
        LinearFunctionPlan(matrix=GaussianMatrixPlan())
    )
    layout_b, execution_plan_b = _random_points_execution_plan(
        LinearFunctionPlan(matrix=KernelMatrixPlan(gamma=2.5, signed=False))
    )

    sketch_a = mixed_execution_cohort_compatibility_sketch(
        execution_plan_a,
        _prepare_fixed_layout_execution_context(layout_a, execution_plan_a),
    )
    sketch_b = mixed_execution_cohort_compatibility_sketch(
        execution_plan_b,
        _prepare_fixed_layout_execution_context(layout_b, execution_plan_b),
    )

    assert sketch_a == sketch_b


def test_generate_mixed_fixed_layout_graph_batch_prepared_matches_exact_cohort_batches() -> None:
    cfg = _mixed_executor_test_config()
    plans = _distinct_leaf_only_plans(cfg, count=3)
    cohort_specs = [
        (plans[0], (1501, 1502)),
        (plans[1], (2501,)),
        (plans[2], (3501,)),
    ]

    expected_x_batches: list[torch.Tensor] = []
    expected_y_batches: list[torch.Tensor] = []
    expected_aux_meta: list[dict[str, object]] = []
    mixed_cohorts: list[_MixedPreparedLogicalCohort] = []
    for plan, dataset_seeds in cohort_specs:
        assert plan.prepared_execution_context is not None
        x_batch, y_batch, aux_meta_batch = _generate_fixed_layout_graph_batch_prepared(
            cfg,
            plan.layout,
            execution_plan=plan.execution_plan,
            prepared_execution_context=plan.prepared_execution_context,
            dataset_seeds=list(dataset_seeds),
            device="cpu",
            noise_sigma_multiplier=1.0,
            noise_spec=None,
        )
        expected_x_batches.append(x_batch)
        expected_y_batches.append(y_batch)
        expected_aux_meta.extend(aux_meta_batch)
        mixed_cohorts.append(
            _MixedPreparedLogicalCohort(
                layout=plan.layout,
                execution_plan=plan.execution_plan,
                prepared_execution_context=plan.prepared_execution_context,
                dataset_seeds=tuple(int(seed) for seed in dataset_seeds),
                noise_spec=None,
                task="regression",
                n_rows=int(cfg.dataset.n_train + cfg.dataset.n_test),
            )
        )

    observed_x, observed_y, observed_aux_meta = _generate_mixed_fixed_layout_graph_batch_prepared(
        mixed_cohorts,
        device="cpu",
        noise_sigma_multiplier=1.0,
        runtime_metrics_out={},
    )

    torch.testing.assert_close(observed_x, torch.cat(expected_x_batches, dim=0))
    torch.testing.assert_close(observed_y, torch.cat(expected_y_batches, dim=0))
    assert observed_aux_meta == expected_aux_meta


def test_generate_mixed_fixed_layout_graph_batch_prepared_matches_exact_for_kernel_compatible_linear_sources() -> (
    None
):
    cfg = _mixed_executor_test_config()
    _assert_mixed_prepared_matches_exact_cohort_batches(
        cfg,
        [
            (
                *_random_points_execution_plan(LinearFunctionPlan(matrix=GaussianMatrixPlan())),
                (3601, 3602),
            ),
            (
                *_random_points_execution_plan(
                    LinearFunctionPlan(matrix=KernelMatrixPlan(gamma=1.7, signed=False))
                ),
                (3701,),
            ),
            (
                *_random_points_execution_plan(
                    LinearFunctionPlan(
                        matrix=ActivationMatrixPlan(
                            base_kind="weights",
                            activation=FixedActivationPlan(name="tanh"),
                        )
                    )
                ),
                (3801,),
            ),
        ],
    )


def test_generate_mixed_fixed_layout_graph_batch_prepared_matches_exact_for_product_sources() -> (
    None
):
    cfg = _mixed_executor_test_config()
    _assert_mixed_prepared_matches_exact_cohort_batches(
        cfg,
        [
            (*_random_points_execution_plan(_product_function_variant(0)), (4101, 4102)),
            (*_random_points_execution_plan(_product_function_variant(1)), (4201,)),
            (*_random_points_execution_plan(_product_function_variant(2)), (4301,)),
        ],
    )


def test_generate_mixed_fixed_layout_graph_batch_prepared_matches_exact_for_piecewise_sources() -> (
    None
):
    cfg = _mixed_executor_test_config()
    _assert_mixed_prepared_matches_exact_cohort_batches(
        cfg,
        [
            (*_random_points_execution_plan(_piecewise_function_variant(0)), (5101, 5102)),
            (*_random_points_execution_plan(_piecewise_function_variant(1)), (5201,)),
            (*_random_points_execution_plan(_piecewise_function_variant(2)), (5301,)),
        ],
    )


def test_generate_mixed_fixed_layout_graph_batch_prepared_matches_exact_for_kernel_compatible_piecewise_sources() -> (
    None
):
    cfg = _mixed_executor_test_config()
    _assert_mixed_prepared_matches_exact_cohort_batches(
        cfg,
        [
            (*_random_points_execution_plan(_kernel_compatible_piecewise_variant(0)), (5601, 5602)),
            (*_random_points_execution_plan(_kernel_compatible_piecewise_variant(1)), (5701,)),
            (*_random_points_execution_plan(_kernel_compatible_piecewise_variant(2)), (5801,)),
        ],
    )


def test_generate_mixed_fixed_layout_graph_batch_prepared_matches_exact_for_nested_piecewise_product_sources() -> (
    None
):
    cfg = _mixed_executor_test_config()
    _assert_mixed_prepared_matches_exact_cohort_batches(
        cfg,
        [
            (*_random_points_execution_plan(_nested_piecewise_product_variant(0)), (6101, 6102)),
            (*_random_points_execution_plan(_nested_piecewise_product_variant(1)), (6201,)),
            (*_random_points_execution_plan(_nested_piecewise_product_variant(2)), (6301,)),
        ],
    )


@pytest.mark.parametrize("source_kind", ["concat", "stacked"])
def test_generate_mixed_fixed_layout_graph_batch_prepared_matches_exact_for_higher_order_multi_input_sources(
    source_kind: str,
) -> None:
    cfg = _mixed_executor_test_config()
    if source_kind == "concat":
        cohort_specs = [
            (
                *_multi_input_execution_plan(
                    source_kind="concat", function_plan=_nested_piecewise_product_variant(0)
                ),
                (7101, 7102),
            ),
            (
                *_multi_input_execution_plan(
                    source_kind="concat", function_plan=_nested_piecewise_product_variant(1)
                ),
                (7201,),
            ),
            (
                *_multi_input_execution_plan(
                    source_kind="concat", function_plan=_nested_piecewise_product_variant(2)
                ),
                (7301,),
            ),
        ]
    else:
        cohort_specs = [
            (
                *_multi_input_execution_plan(
                    source_kind="stacked",
                    parent_functions=(
                        _product_function_variant(0),
                        _piecewise_function_variant(0),
                    ),
                ),
                (8101, 8102),
            ),
            (
                *_multi_input_execution_plan(
                    source_kind="stacked",
                    parent_functions=(
                        _nested_piecewise_product_variant(1),
                        _product_function_variant(1),
                    ),
                ),
                (8201,),
            ),
            (
                *_multi_input_execution_plan(
                    source_kind="stacked",
                    parent_functions=(
                        _piecewise_function_variant(2),
                        _nested_piecewise_product_variant(2),
                    ),
                ),
                (8301,),
            ),
        ]
    _assert_mixed_prepared_matches_exact_cohort_batches(cfg, cohort_specs)


def test_generate_fixed_layout_validation_label_batch_matches_public_label_batch() -> None:
    cfg = load_repo_config()
    cfg.dataset.task = "classification"
    cfg.filter.enabled = False
    cfg.dataset.n_train = 12
    cfg.dataset.n_test = 4
    cfg.dataset.n_features_min = 5
    cfg.dataset.n_features_max = 5
    cfg.dataset.n_classes_min = 4
    cfg.dataset.n_classes_max = 4
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 6
    plan = _sample_fixed_layout(cfg, seed=311, device="cpu")
    assert plan.prepared_execution_context is not None
    dataset_seeds = [1101, 1102, 1103]

    expected_y, _ = generate_fixed_layout_label_batch(
        cfg,
        plan.layout,
        execution_plan=plan.execution_plan,
        dataset_seeds=dataset_seeds,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    observed_y, _ = _generate_fixed_layout_validation_label_batch(
        cfg,
        plan.layout,
        execution_plan=plan.execution_plan,
        prepared_execution_context=plan.prepared_execution_context,
        dataset_seeds=dataset_seeds,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(observed_y, expected_y)


def test_generate_fixed_layout_validation_label_batch_skips_non_ancestor_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_repo_config()
    cfg.dataset.task = "classification"
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    layout = LayoutPlan(
        n_features=1,
        n_cat=0,
        cat_idx=[],
        cardinalities=[],
        card_by_feature={},
        n_classes=3,
        feature_types=["num"],
        graph_nodes=3,
        graph_edges=1,
        graph_depth_nodes=2,
        graph_edge_density=1.0 / 3.0,
        adjacency=torch.tensor(
            [
                [False, False, True],
                [False, False, False],
                [False, False, False],
            ],
            dtype=torch.bool,
        ),
        feature_node_assignment=[0],
        target_to_node=2,
    )
    feature_specs = typed_converter_specs([ConverterSpec(key="feature_0", kind="num", dim=1)])
    target_specs = typed_converter_specs([ConverterSpec(key="target", kind="target_cls", dim=1)])
    root_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(),
        converter_specs=feature_specs,
        converter_plans=(NumericConverterPlan(kind="num", warp_enabled=False),),
        converter_groups=fixed_layout_converter_groups(
            feature_specs,
            (NumericConverterPlan(kind="num", warp_enabled=False),),
        ),
        latent=FixedLayoutLatentPlan(required_dim=1, extra_dim=0, total_dim=1),
        source=RandomPointsNodeSource(
            base_kind="normal",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    skipped_plan = FixedLayoutNodePlan(
        node_index=1,
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
    target_plan = FixedLayoutNodePlan(
        node_index=2,
        parent_indices=(0,),
        converter_specs=target_specs,
        converter_plans=(NumericConverterPlan(kind="target_reg", warp_enabled=False),),
        converter_groups=fixed_layout_converter_groups(
            target_specs,
            (NumericConverterPlan(kind="target_reg", warp_enabled=False),),
        ),
        latent=FixedLayoutLatentPlan(required_dim=1, extra_dim=0, total_dim=1),
        source=ConcatNodeSource(
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    execution_plan = FixedLayoutExecutionPlan(node_plans=(root_plan, skipped_plan, target_plan))
    prepared = _prepare_fixed_layout_execution_context(layout, execution_plan)
    observed_node_indices: list[int] = []

    def _stub_apply_node_plan_batch_prepared(
        _config,
        node_context,
        _parent_data,
        *,
        n_rows: int,
        batch_node_context,
        device: str,
        noise_sigma_multiplier: float,
        noise_spec,
        runtime_metrics_out=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _ = (device, noise_sigma_multiplier, noise_spec, runtime_metrics_out)
        observed_node_indices.append(int(node_context.node_plan.node_index))
        node_rng = batch_node_context.node_rng
        latent = torch.full(
            (node_rng.batch_size, n_rows, 1), float(node_context.node_plan.node_index)
        )
        extracted: dict[str, torch.Tensor] = {}
        if int(node_context.node_plan.node_index) == 2:
            extracted["target"] = torch.ones((node_rng.batch_size, n_rows), dtype=torch.float32)
        return latent, extracted

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.batched._apply_node_plan_batch_prepared",
        _stub_apply_node_plan_batch_prepared,
    )

    y_batch, _ = _generate_fixed_layout_validation_label_batch(
        cfg,
        layout,
        execution_plan=execution_plan,
        prepared_execution_context=prepared,
        dataset_seeds=[1201, 1202],
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    assert observed_node_indices == [0, 2]
    assert y_batch.shape == (2, 6)
