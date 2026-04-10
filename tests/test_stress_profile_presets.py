from __future__ import annotations

import pytest

from dagzoo.config import GeneratorConfig


@pytest.mark.parametrize(
    ("config_path", "profile", "out_dir"),
    [
        (
            "configs/preset_stress_classification_slice_generate_smoke.yaml",
            "anti_memorization_piecewise_classification_slice_v1",
            "data/run_stress_classification_slice_smoke",
        ),
        (
            "configs/preset_stress_graph_breadth_generate_smoke.yaml",
            "anti_memorization_piecewise_classification_graph_breadth_slice_v1",
            "data/run_stress_graph_breadth_smoke",
        ),
        (
            "configs/preset_stress_compositional_generate_smoke.yaml",
            "anti_memorization_piecewise_classification_compositional_slice_v1",
            "data/run_stress_compositional_smoke",
        ),
        (
            "configs/preset_stress_categorical_cardinality_generate_smoke.yaml",
            "anti_memorization_piecewise_classification_categorical_cardinality_slice_v1",
            "data/run_stress_categorical_cardinality_smoke",
        ),
        (
            "configs/preset_stress_hybrid_generate_smoke.yaml",
            "anti_memorization_piecewise_classification_hybrid_slice_v1",
            "data/run_stress_hybrid_smoke",
        ),
        (
            "configs/preset_stress_robustness_composition_generate_smoke.yaml",
            "anti_memorization_piecewise_classification_robustness_composition_slice_v1",
            "data/run_stress_robustness_composition_smoke",
        ),
    ],
)
def test_stress_generate_presets_load_with_expected_profile(
    config_path: str,
    profile: str,
    out_dir: str,
) -> None:
    cfg = GeneratorConfig.from_yaml(config_path)

    assert cfg.stress.profile == profile
    assert cfg.runtime.device == "cpu"
    assert cfg.output.out_dir == out_dir
    assert cfg.diagnostics.enabled is True
    assert cfg.filter.enabled is False


@pytest.mark.parametrize(
    ("config_path", "profile", "preset_name", "out_dir"),
    [
        (
            "configs/preset_stress_classification_slice_benchmark_smoke.yaml",
            "anti_memorization_piecewise_classification_slice_v1",
            "stress_classification_slice_smoke",
            "benchmarks/results/smoke_stress_classification_slice",
        ),
        (
            "configs/preset_stress_graph_breadth_benchmark_smoke.yaml",
            "anti_memorization_piecewise_classification_graph_breadth_slice_v1",
            "stress_graph_breadth_smoke",
            "benchmarks/results/smoke_stress_graph_breadth",
        ),
        (
            "configs/preset_stress_compositional_benchmark_smoke.yaml",
            "anti_memorization_piecewise_classification_compositional_slice_v1",
            "stress_compositional_smoke",
            "benchmarks/results/smoke_stress_compositional",
        ),
        (
            "configs/preset_stress_categorical_cardinality_benchmark_smoke.yaml",
            "anti_memorization_piecewise_classification_categorical_cardinality_slice_v1",
            "stress_categorical_cardinality_smoke",
            "benchmarks/results/smoke_stress_categorical_cardinality",
        ),
        (
            "configs/preset_stress_hybrid_benchmark_smoke.yaml",
            "anti_memorization_piecewise_classification_hybrid_slice_v1",
            "stress_hybrid_smoke",
            "benchmarks/results/smoke_stress_hybrid",
        ),
        (
            "configs/preset_stress_robustness_composition_benchmark_smoke.yaml",
            "anti_memorization_piecewise_classification_robustness_composition_slice_v1",
            "stress_robustness_composition_smoke",
            "benchmarks/results/smoke_stress_robustness_composition",
        ),
    ],
)
def test_stress_benchmark_presets_load_with_expected_profile(
    config_path: str,
    profile: str,
    preset_name: str,
    out_dir: str,
) -> None:
    cfg = GeneratorConfig.from_yaml(config_path)

    assert cfg.stress.profile == profile
    assert cfg.runtime.device == "cpu"
    assert cfg.output.out_dir == out_dir
    assert cfg.diagnostics.enabled is False
    assert cfg.benchmark.preset_name == preset_name
    assert cfg.benchmark.suite == "smoke"
    assert cfg.benchmark.collect_memory is False
    assert preset_name in cfg.benchmark.presets
    assert cfg.filter.enabled is False


def test_default_config_leaves_stress_profile_unset() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")

    assert cfg.stress.profile is None
