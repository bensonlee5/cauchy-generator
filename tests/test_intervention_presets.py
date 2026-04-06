from __future__ import annotations

import pytest

from dagzoo.config import GeneratorConfig


@pytest.mark.parametrize(
    ("config_path", "target_kind", "target_index", "target_value", "out_dir"),
    [
        (
            "configs/preset_intervention_target_generate_smoke.yaml",
            "target",
            None,
            2.5,
            "data/run_intervention_target_smoke",
        ),
        (
            "configs/preset_intervention_feature_node_generate_smoke.yaml",
            "feature_node",
            1,
            -1.25,
            "data/run_intervention_feature_node_smoke",
        ),
        (
            "configs/preset_intervention_latent_node_generate_smoke.yaml",
            "latent_node",
            0,
            1.75,
            "data/run_intervention_latent_node_smoke",
        ),
    ],
)
def test_intervention_generate_presets_load_with_expected_selector(
    config_path: str,
    target_kind: str,
    target_index: int | None,
    target_value: float,
    out_dir: str,
) -> None:
    cfg = GeneratorConfig.from_yaml(config_path)

    assert cfg.dataset.task == "regression"
    assert cfg.intervention.mode == "hard_interventional"
    assert len(cfg.intervention.targets) == 1
    assert cfg.intervention.signature is not None
    target = cfg.intervention.targets[0]
    assert str(target.target_kind) == target_kind
    assert target.index == target_index
    assert float(target.value) == pytest.approx(target_value)
    assert cfg.runtime.device == "cpu"
    assert cfg.output.out_dir == out_dir
    assert cfg.filter.enabled is False
