from __future__ import annotations

from dataclasses import dataclass

import torch
from conftest import make_generator as _make_generator

from dagzoo.core.execution_semantics import sample_function_plan_for_family, typed_converter_specs
from dagzoo.core.fixed_layout.batched import FixedLayoutBatchRng, apply_function_plan_batch
from dagzoo.core.fixed_layout.plan_types import GaussianMatrixPlan, LinearFunctionPlan
from dagzoo.core.node_pipeline import apply_node_pipeline
from dagzoo.functions.random_functions import apply_random_function
from dagzoo.sampling.random_points import sample_random_points


@dataclass(slots=True)
class _ConverterSpec:
    key: str
    kind: str
    dim: int
    cardinality: int | None = None


def test_benchmark_execution_semantics_sampling(benchmark) -> None:
    def _run():
        return sample_function_plan_for_family(
            _make_generator(11),
            family="linear",
            out_dim=3,
            mechanism_logit_tilt=0.0,
            function_family_mix=None,
        )

    plan = benchmark(_run)
    assert isinstance(plan, LinearFunctionPlan)


def test_benchmark_fixed_layout_batch_apply(benchmark) -> None:
    x = torch.randn(2, 64, 4, generator=_make_generator(12))
    plan = LinearFunctionPlan(matrix=GaussianMatrixPlan())

    def _run():
        return apply_function_plan_batch(
            x,
            FixedLayoutBatchRng(seed=13, batch_size=2, device="cpu"),
            plan,
            out_dim=3,
            noise_sigma_multiplier=1.0,
            noise_spec=None,
        )

    result = benchmark(_run)
    assert result.shape == (2, 64, 3)


def test_benchmark_node_pipeline(benchmark) -> None:
    parents = [torch.randn(128, 6, generator=_make_generator(14))]
    specs = typed_converter_specs((_ConverterSpec(key="feature", kind="num", dim=1),))

    def _run():
        return apply_node_pipeline(parents, 128, specs, _make_generator(15), "cpu")

    latent, extracted = benchmark(_run)
    assert latent.shape[0] == 128
    assert extracted["feature"].shape == (128,)


def test_benchmark_random_functions(benchmark) -> None:
    x = torch.randn(128, 4, generator=_make_generator(16))

    def _run():
        return apply_random_function(
            x,
            _make_generator(17),
            out_dim=3,
            function_type="linear",
        )

    result = benchmark(_run)
    assert result.shape == (128, 3)


def test_benchmark_random_points(benchmark) -> None:
    def _run():
        return sample_random_points(128, 4, _make_generator(18), "cpu")

    result = benchmark(_run)
    assert result.shape == (128, 4)
