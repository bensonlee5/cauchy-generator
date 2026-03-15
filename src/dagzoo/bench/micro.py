"""Component-level microbenchmarks for hot path sanity checks."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from dagzoo.bench.constants import (
    MICRO_CONFIG_N_FEATURES_CAP,
    MICRO_CONFIG_N_NODES_CAP,
    MICRO_CONFIG_N_TEST_CAP,
    MICRO_CONFIG_N_TRAIN_CAP,
    MICROBENCH_DEFAULT_REPEATS,
    MICROBENCH_PARENT_DIM,
    MICROBENCH_SYNTH_FEATURES,
    MICROBENCH_SYNTH_ROWS,
    MILLISECONDS_PER_SECOND,
)
from dagzoo.config import GeneratorConfig, clone_generator_config
from dagzoo.core.dataset import generate_one
from dagzoo.core.fixed_layout.plan_types import FixedLayoutConverterSpec
from dagzoo.core.node_pipeline import apply_node_pipeline
from dagzoo.functions.random_functions import apply_random_function
from dagzoo.rng import KeyedRng

from .runtime_support import _synchronize_accelerator


def _time_ms(func: Callable[[], None], repeats: int, *, device: str | None = None) -> float:
    """Run ``func`` repeatedly and return the average runtime in milliseconds."""

    if device is not None:
        _synchronize_accelerator(device)
    start = time.perf_counter()
    for _ in range(max(1, repeats)):
        func()
    if device is not None:
        _synchronize_accelerator(device)
    elapsed = time.perf_counter() - start
    return (elapsed / max(1, repeats)) * MILLISECONDS_PER_SECOND


def _micro_config(config: GeneratorConfig) -> GeneratorConfig:
    """Create a lightweight config variant for inexpensive microbenchmarks."""

    c = clone_generator_config(config, revalidate=False)
    c.dataset.n_train = min(MICRO_CONFIG_N_TRAIN_CAP, c.dataset.n_train)
    c.dataset.n_test = min(MICRO_CONFIG_N_TEST_CAP, c.dataset.n_test)
    c.dataset.n_features_min = min(MICRO_CONFIG_N_FEATURES_CAP, c.dataset.n_features_min)
    c.dataset.n_features_max = min(MICRO_CONFIG_N_FEATURES_CAP, c.dataset.n_features_max)
    c.graph.n_nodes_min = min(MICRO_CONFIG_N_NODES_CAP, c.graph.n_nodes_min)
    c.graph.n_nodes_max = min(MICRO_CONFIG_N_NODES_CAP, c.graph.n_nodes_max)
    return c


def run_microbenchmarks(
    config: GeneratorConfig,
    *,
    device: str | None = None,
    repeats: int = MICROBENCH_DEFAULT_REPEATS,
    include_generate_one: bool = True,
) -> dict[str, float | int | None]:
    """Run targeted microbenchmarks for core generation components."""

    micro_root = KeyedRng(int(config.seed)).keyed("bench", "micro")
    x = torch.randn(
        MICROBENCH_SYNTH_ROWS,
        MICROBENCH_SYNTH_FEATURES,
        generator=micro_root.keyed("inputs", "x").torch_rng(device="cpu"),
    )

    def run_linear() -> None:
        _ = apply_random_function(
            x,
            micro_root.keyed("random_function").torch_rng(device="cpu"),
            out_dim=MICROBENCH_SYNTH_FEATURES,
            function_type="linear",
        )

    parent_data = [
        torch.randn(
            MICROBENCH_SYNTH_ROWS,
            MICROBENCH_PARENT_DIM,
            generator=micro_root.keyed("inputs", "parent", 0).torch_rng(device="cpu"),
        ),
        torch.randn(
            MICROBENCH_SYNTH_ROWS,
            MICROBENCH_PARENT_DIM,
            generator=micro_root.keyed("inputs", "parent", 1).torch_rng(device="cpu"),
        ),
    ]
    specs = [
        FixedLayoutConverterSpec(
            key="feature_0", kind="num", dim=1, cardinality=None, column_start=0, column_end=1
        ),
        FixedLayoutConverterSpec(
            key="feature_1", kind="cat", dim=4, cardinality=6, column_start=1, column_end=5
        ),
        FixedLayoutConverterSpec(
            key="target", kind="target_cls", dim=3, cardinality=3, column_start=5, column_end=8
        ),
    ]

    def run_node_pipeline() -> None:
        _ = apply_node_pipeline(
            parent_data,
            MICROBENCH_SYNTH_ROWS,
            specs,
            micro_root.keyed("node_pipeline").torch_rng(device="cpu"),
            "cpu",
        )

    micro_cfg = _micro_config(config)
    generate_one_ms: float | None
    if include_generate_one:

        def run_generate_one() -> None:
            _ = generate_one(
                micro_cfg,
                seed=micro_root.child_seed("generate_one"),
                device=device,
            )

        generate_one_ms = _time_ms(
            run_generate_one, repeats, device=device or micro_cfg.runtime.device
        )
    else:
        generate_one_ms = None

    return {
        "micro_repeats": int(max(1, repeats)),
        "micro_random_function_linear_ms": _time_ms(run_linear, repeats),
        "micro_node_pipeline_ms": _time_ms(run_node_pipeline, repeats),
        "micro_generate_one_ms": generate_one_ms,
    }
