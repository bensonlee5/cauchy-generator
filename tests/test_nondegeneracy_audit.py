from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from conftest import load_repo_config
from conftest import make_generator as _make_generator

import dagzoo.core.node_pipeline as node_pipeline_mod
from dagzoo.core.dataset import generate_batch, generate_one
from dagzoo.core.execution_semantics import typed_converter_specs
from dagzoo.core.fixed_layout.plan_types import (
    CategoricalConverterPlan,
    ConcatNodeSource,
    FixedLayoutLatentPlan,
    FixedLayoutNodePlan,
    GaussianMatrixPlan,
    LinearFunctionPlan,
    NumericConverterPlan,
    RandomPointsNodeSource,
    StackedNodeSource,
    fixed_layout_converter_groups,
)
from dagzoo.core.node_pipeline import apply_node_pipeline
from dagzoo.functions.random_functions import apply_random_function
from dagzoo.math.random_matrices import sample_random_matrix


@dataclass(slots=True)
class ConverterSpec:
    key: str
    kind: str
    dim: int
    cardinality: int | None = None


def _assert_not_fully_collapsed(tensor: torch.Tensor) -> None:
    values = tensor.to(torch.float32)
    assert torch.all(torch.isfinite(values))
    if values.dim() == 1:
        values = values.unsqueeze(1)
    channel_variance = torch.var(values, dim=0, correction=0)
    assert bool(torch.any(channel_variance > 0.0))


def _assert_bundle_has_varying_feature(bundle) -> None:
    combined = torch.cat([bundle.X_train.to(torch.float32), bundle.X_test.to(torch.float32)], dim=0)
    assert torch.all(torch.isfinite(combined))
    assert bool(torch.any(torch.var(combined, dim=0, correction=0) > 0.0))


def _structured_input(rows: int, dim: int) -> torch.Tensor:
    base = torch.linspace(-1.0, 1.0, steps=rows, dtype=torch.float32)
    columns = [base]
    for index in range(1, dim):
        columns.append(torch.cos(base * float(index + 1)) + (0.1 * index))
    return torch.stack(columns, dim=1)


def _fixed_audit_config():
    cfg = load_repo_config()
    cfg.runtime.layout_mode = "stratified"
    cfg.runtime.device = "cpu"
    cfg.filter.enabled = False
    cfg.filter.max_attempts = 16
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 24
    cfg.dataset.n_test = 8
    cfg.dataset.n_features_min = 4
    cfg.dataset.n_features_max = 4
    cfg.dataset.categorical_ratio_min = 0.0
    cfg.dataset.categorical_ratio_max = 0.0
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 4
    return cfg


def _heterogeneous_audit_config():
    cfg = _fixed_audit_config()
    cfg.runtime.layout_mode = "heterogeneous"
    cfg.dataset.n_features_min = 4
    cfg.dataset.n_features_max = 6
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    return cfg


@pytest.mark.parametrize(
    ("kind", "out_dim", "in_dim"),
    [
        ("gaussian", 1, 1),
        ("weights", 1, 4),
        ("singular_values", 4, 1),
        ("kernel", 4, 4),
        ("activation", 3, 5),
    ],
)
def test_random_matrix_audit_covers_supported_pathways(
    kind: str,
    out_dim: int,
    in_dim: int,
) -> None:
    matrix = sample_random_matrix(
        out_dim, in_dim, _make_generator(100 + out_dim + in_dim), "cpu", kind=kind
    )

    assert matrix.shape == (out_dim, in_dim)
    assert torch.all(torch.isfinite(matrix))
    norms = torch.linalg.norm(matrix, dim=1)
    torch.testing.assert_close(norms, torch.ones(out_dim), atol=1e-4, rtol=1e-4)
    if matrix.numel() > 1:
        assert float(torch.var(matrix.reshape(-1), correction=0)) > 0.0


@pytest.mark.parametrize(
    ("family", "input_dim", "out_dim"),
    [
        ("linear", 1, 1),
        ("quadratic", 1, 4),
        ("nn", 4, 1),
        ("tree", 4, 4),
        ("discretization", 1, 1),
        ("gp", 1, 4),
        ("em", 4, 1),
        ("product", 4, 4),
        ("piecewise", 1, 4),
    ],
)
def test_random_function_audit_covers_supported_families(
    family: str,
    input_dim: int,
    out_dim: int,
) -> None:
    x = _structured_input(rows=48, dim=input_dim)
    y = apply_random_function(
        x,
        _make_generator(200 + input_dim + out_dim),
        out_dim=out_dim,
        function_type=family,  # type: ignore[arg-type]
    )

    assert y.shape == (48, out_dim)
    _assert_not_fully_collapsed(y)


@pytest.mark.parametrize(
    ("variant", "method", "source_kind"),
    [
        ("input", "neighbor", "root"),
        ("index_repeat", "neighbor", "concat"),
        ("center", "neighbor", "stacked"),
        ("center_random_fn", "neighbor", "root"),
        ("softmax_points", "softmax", "concat"),
    ],
)
def test_node_pipeline_audit_covers_source_and_converter_pathways(
    variant: str,
    method: str,
    source_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = typed_converter_specs(
        [
            ConverterSpec(key="feature_num", kind="num", dim=1),
            ConverterSpec(key="feature_cat", kind="cat", dim=3, cardinality=5),
        ]
    )
    converter_plans = (
        NumericConverterPlan(kind="num", warp_enabled=True),
        CategoricalConverterPlan(
            kind="cat",
            method=method,  # type: ignore[arg-type]
            variant=variant,  # type: ignore[arg-type]
            function=(
                LinearFunctionPlan(matrix=GaussianMatrixPlan())
                if variant == "center_random_fn"
                else None
            ),
        ),
    )
    if source_kind == "root":
        source = RandomPointsNodeSource(
            base_kind="normal",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        )
        parents: list[torch.Tensor] = []
        parent_dims: tuple[int, ...] = ()
    else:
        parents = [_structured_input(48, 3), _structured_input(48, 2)]
        parent_dims = tuple(int(parent.shape[1]) for parent in parents)
        source = StackedNodeSource(
            aggregation_kind="sum" if source_kind == "stacked" else "max",
            parent_functions=(
                LinearFunctionPlan(matrix=GaussianMatrixPlan()),
                LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            ),
        )
        if source_kind == "concat":
            source = ConcatNodeSource(  # type: ignore[assignment]
                function=LinearFunctionPlan(matrix=GaussianMatrixPlan())
            )
    node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=tuple(range(len(parent_dims))),
        converter_specs=specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=4, extra_dim=2, total_dim=6),
        source=source,
    )
    monkeypatch.setattr(node_pipeline_mod, "sample_node_plan", lambda **_kwargs: node_plan)

    latent, extracted = apply_node_pipeline(
        parents,
        48,
        specs,
        _make_generator(300 + len(parents)),
        "cpu",
    )

    _assert_not_fully_collapsed(latent)
    assert set(extracted) == {"feature_num", "feature_cat"}
    assert torch.all(torch.isfinite(extracted["feature_num"]))
    assert bool(torch.var(extracted["feature_num"].to(torch.float32), correction=0) > 0.0)
    assert bool(torch.var(extracted["feature_cat"].to(torch.float32), correction=0) > 0.0)


def test_generate_one_fixed_audit_bundle_is_not_degenerate() -> None:
    bundle = generate_one(_fixed_audit_config(), seed=401, device="cpu")

    _assert_bundle_has_varying_feature(bundle)


def test_generate_batch_fixed_audit_bundles_are_not_degenerate() -> None:
    bundles = generate_batch(_fixed_audit_config(), num_datasets=2, seed=402, device="cpu")

    assert len(bundles) == 2
    for bundle in bundles:
        _assert_bundle_has_varying_feature(bundle)


def test_generate_batch_heterogeneous_audit_bundles_are_not_degenerate() -> None:
    bundles = generate_batch(_heterogeneous_audit_config(), num_datasets=2, seed=403, device="cpu")

    assert len(bundles) == 2
    for bundle in bundles:
        _assert_bundle_has_varying_feature(bundle)
