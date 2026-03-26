from __future__ import annotations

from dagzoo.config import GeneratorConfig


def test_steering_generate_preset_loads_with_expected_profile() -> None:
    cfg = GeneratorConfig.from_yaml("configs/preset_steering_anti_memorization_generate_smoke.yaml")
    assert cfg.steering.enabled is True
    assert cfg.steering.preset == "anti_memorization_piecewise_v1"
    assert cfg.diagnostics.enabled is True
    assert cfg.diagnostics.histogram_bins == 8
    assert cfg.filter.enabled is False


def test_steering_benchmark_preset_loads_with_expected_profile() -> None:
    cfg = GeneratorConfig.from_yaml(
        "configs/preset_steering_anti_memorization_benchmark_smoke.yaml"
    )
    assert cfg.steering.enabled is True
    assert cfg.steering.preset == "anti_memorization_piecewise_v1"
    assert cfg.benchmark.preset_name == "steering_smoke"
    assert "steering_smoke" in cfg.benchmark.presets
    assert cfg.diagnostics.enabled is False
    assert cfg.diagnostics.histogram_bins == 8
    assert cfg.filter.enabled is False
