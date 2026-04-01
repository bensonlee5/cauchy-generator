from dataclasses import dataclass

import pytest
import torch
from conftest import make_generator as _make_generator
from conftest import make_keyed_rng as _make_keyed_rng

import dagzoo.core.node_pipeline as node_pipeline_mod
from dagzoo.core.execution_semantics import typed_converter_specs
from dagzoo.core.fixed_layout.batched import FixedLayoutBatchRng, _apply_node_plan_batch
from dagzoo.core.fixed_layout.plan_types import (
    CategoricalConverterGroup,
    CategoricalConverterPlan,
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
from dagzoo.core.validation import RetryableDegeneracyError


@dataclass(slots=True)
class ConverterSpec:
    key: str
    kind: str
    dim: int
    cardinality: int | None = None


def test_node_pipeline_extracts_requested_columns() -> None:
    g = _make_generator(123)
    parents = [torch.randn(128, 8, generator=g)]
    specs = [
        ConverterSpec(key="feature_0", kind="num", dim=1),
        ConverterSpec(key="feature_1", kind="cat", dim=4, cardinality=5),
        ConverterSpec(key="target", kind="target_cls", dim=3, cardinality=3),
    ]

    x_node, extracted = apply_node_pipeline(parents, 128, typed_converter_specs(specs), g, "cpu")
    assert x_node.shape[0] == 128
    assert "feature_0" in extracted
    assert "feature_1" in extracted
    assert "target" in extracted
    assert extracted["feature_0"].shape == (128,)
    assert extracted["feature_1"].dtype == torch.int64


def test_torch_output_shapes() -> None:
    g = _make_generator()
    parents = [torch.randn(64, 6, generator=g)]
    specs = [
        ConverterSpec(key="f0", kind="num", dim=1),
        ConverterSpec(key="f1", kind="cat", dim=3, cardinality=4),
    ]
    x, ext = apply_node_pipeline(parents, 64, typed_converter_specs(specs), g, "cpu")
    assert x.shape[0] == 64
    assert "f0" in ext
    assert ext["f0"].shape == (64,)
    assert "f1" in ext
    assert ext["f1"].shape == (64,)


def test_torch_no_parents() -> None:
    g = _make_generator(7)
    specs = [ConverterSpec(key="v", kind="num", dim=1)]
    x, ext = apply_node_pipeline([], 64, typed_converter_specs(specs), g, "cpu")
    assert x.shape[0] == 64
    assert "v" in ext


def test_torch_deterministic() -> None:
    specs = [ConverterSpec(key="v", kind="num", dim=1)]
    parents = [torch.randn(32, 4)]

    g1 = _make_generator(0)
    x1, e1 = apply_node_pipeline(parents, 32, typed_converter_specs(specs), g1, "cpu")

    g2 = _make_generator(0)
    x2, e2 = apply_node_pipeline(parents, 32, typed_converter_specs(specs), g2, "cpu")

    torch.testing.assert_close(x1, x2)
    torch.testing.assert_close(e1["v"], e2["v"])


def test_node_pipeline_sanitizes_non_finite_parent_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [ConverterSpec(key="v", kind="num", dim=1)]
    typed_specs = typed_converter_specs(specs)
    converter_plans = (NumericConverterPlan(kind="num", warp_enabled=False),)
    node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(0,),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=1, extra_dim=1, total_dim=2),
        source=StackedNodeSource(
            aggregation_kind="sum",
            parent_functions=(LinearFunctionPlan(matrix=GaussianMatrixPlan()),),
        ),
    )
    monkeypatch.setattr(node_pipeline_mod, "sample_node_plan", lambda **_kwargs: node_plan)

    parents = [
        torch.tensor(
            [[0.0, float("nan")], [float("inf"), -1.0], [-float("inf"), 2.0]],
            dtype=torch.float32,
        )
    ]

    x, ext = apply_node_pipeline(
        parents, 3, typed_converter_specs(specs), _make_generator(5), "cpu"
    )

    assert torch.all(torch.isfinite(x))
    assert torch.all(torch.isfinite(ext["v"]))


def test_node_pipeline_splits_grouped_center_random_fn_converters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [
        ConverterSpec(key="feature_0", kind="cat", dim=3, cardinality=5),
        ConverterSpec(key="feature_1", kind="cat", dim=3, cardinality=5),
    ]
    typed_specs = typed_converter_specs(specs)
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
    grouped_converter_groups = fixed_layout_converter_groups(typed_specs, converter_plans)
    assert grouped_converter_groups == (CategoricalConverterGroup(spec_indices=(0, 1)),)

    grouped_node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=grouped_converter_groups,
        latent=FixedLayoutLatentPlan(required_dim=6, extra_dim=2, total_dim=8),
        source=RandomPointsNodeSource(
            base_kind="normal",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    split_node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=(
            CategoricalConverterGroup(spec_indices=(0,)),
            CategoricalConverterGroup(spec_indices=(1,)),
        ),
        latent=grouped_node_plan.latent,
        source=grouped_node_plan.source,
    )
    monkeypatch.setattr(node_pipeline_mod, "sample_node_plan", lambda **_kwargs: grouped_node_plan)

    actual_generator = _make_generator(17)
    reference_generator = _make_generator(17)
    actual_latent, actual_extracted = apply_node_pipeline(
        [],
        12,
        typed_converter_specs(specs),
        actual_generator,
        "cpu",
    )
    root = _make_keyed_rng(reference_generator, "apply_node_pipeline")
    expected_latent, expected_extracted = _apply_node_plan_batch(
        None,
        split_node_plan,
        [],
        n_rows=12,
        rng=FixedLayoutBatchRng.from_keyed_rng(root.keyed("execution"), batch_size=1, device="cpu"),
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    assert torch.all(torch.isfinite(actual_latent))
    assert set(actual_extracted) == {"feature_0", "feature_1"}
    torch.testing.assert_close(actual_latent, expected_latent.squeeze(0))
    for key, value in actual_extracted.items():
        assert torch.all(torch.isfinite(value))
        torch.testing.assert_close(value, expected_extracted[key].squeeze(0))
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_apply_node_pipeline_retries_retryable_degeneracy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [ConverterSpec(key="v", kind="num", dim=1)]
    typed_specs = typed_converter_specs(specs)
    converter_plans = (NumericConverterPlan(kind="num", warp_enabled=False),)
    node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=1, extra_dim=1, total_dim=2),
        source=RandomPointsNodeSource(
            base_kind="normal",
            function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    call_count = 0

    def _stub_sample_node_plan(**_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RetryableDegeneracyError("constant_pathway_output")
        return node_plan

    monkeypatch.setattr(node_pipeline_mod, "sample_node_plan", _stub_sample_node_plan)

    latent, extracted = apply_node_pipeline([], 32, typed_specs, _make_generator(24), "cpu")

    assert call_count == 2
    assert latent.shape[0] == 32
    assert torch.all(torch.isfinite(latent))
    assert torch.all(torch.isfinite(extracted["v"]))
