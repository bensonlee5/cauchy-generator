import pytest

from dagzoo.config import NOISE_FAMILY_MIXTURE, GeneratorConfig
from dagzoo.core.steering import resolve_steering


def test_resolve_steering_disabled_returns_base_config() -> None:
    cfg = GeneratorConfig.from_dict({})

    resolved = resolve_steering(cfg, dataset_index=0, run_num_datasets=3)

    assert resolved.stage_name is None
    assert resolved.stage_index is None
    assert resolved.stage_progress is None
    assert resolved.progress == pytest.approx(0.0)
    assert resolved.config.dataset.missing_rate == pytest.approx(cfg.dataset.missing_rate)


def test_resolve_steering_preset_progression_carries_forward_prior_stage_state() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "steering": {
                "enabled": True,
                "preset": "anti_memorization_piecewise_v1",
            }
        }
    )

    resolved = [
        resolve_steering(cfg, dataset_index=index, run_num_datasets=5) for index in range(5)
    ]

    assert [entry.stage_name for entry in resolved] == [
        "missingness_ramp",
        "graph_excursion_out",
        "graph_to_noise_handoff",
        "mixture_noise_ramp",
        "mixture_noise_ramp",
    ]

    assert resolved[0].config.dataset.missing_rate == pytest.approx(0.0)
    assert resolved[1].config.dataset.missing_rate == pytest.approx(0.25)
    assert resolved[1].config.shift.mode == "graph_drift"
    assert resolved[1].config.shift.graph_scale == pytest.approx(0.0)

    assert resolved[2].config.shift.mode == "mixed"
    assert resolved[2].config.shift.graph_scale == pytest.approx(0.5)
    assert resolved[2].config.shift.variance_scale == pytest.approx(0.0)

    assert resolved[3].config.shift.mode == "mixed"
    assert resolved[3].config.shift.graph_scale == pytest.approx(0.0)
    assert resolved[3].config.shift.variance_scale == pytest.approx(0.5)
    assert resolved[3].config.noise.family == NOISE_FAMILY_MIXTURE
    assert resolved[3].config.noise.mixture_weights == {
        "gaussian": pytest.approx(1.0),
        "laplace": pytest.approx(0.0),
        "student_t": pytest.approx(0.0),
    }

    assert resolved[4].config.noise.family == NOISE_FAMILY_MIXTURE
    assert resolved[4].config.noise.mixture_weights == {
        "gaussian": pytest.approx(0.5),
        "laplace": pytest.approx(0.3),
        "student_t": pytest.approx(0.2),
    }


def test_resolve_steering_num_datasets_one_uses_zero_progress() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "steering": {
                "enabled": True,
                "stages": [
                    {
                        "name": "descending_graph",
                        "fraction": 1.0,
                        "shift": {
                            "mode": "graph_drift",
                            "graph_scale": [0.5, 0.0],
                        },
                    }
                ],
            }
        }
    )

    resolved = resolve_steering(cfg, dataset_index=0, run_num_datasets=1)

    assert resolved.progress == pytest.approx(0.0)
    assert resolved.stage_name == "descending_graph"
    assert resolved.config.shift.graph_scale == pytest.approx(0.5)
