import pytest

from dagzoo.config import NOISE_FAMILY_MIXTURE, GeneratorConfig
from dagzoo.core.shift import resolve_shift_runtime_params
from dagzoo.core.steering import _resolve_active_stage, resolve_steering


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
    assert resolve_shift_runtime_params(resolved[2].config).mechanism_logit_tilt == pytest.approx(
        0.0
    )

    assert resolved[3].config.shift.mode == "mixed"
    assert resolved[3].config.shift.graph_scale == pytest.approx(0.0)
    assert resolved[3].config.shift.variance_scale == pytest.approx(0.5)
    assert resolve_shift_runtime_params(resolved[3].config).mechanism_logit_tilt == pytest.approx(
        0.0
    )
    assert resolved[3].config.noise.family == NOISE_FAMILY_MIXTURE
    assert resolved[3].config.noise.mixture_weights == {"gaussian": pytest.approx(1.0)}

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


def test_resolve_active_stage_rejects_empty_stage_list() -> None:
    with pytest.raises(RuntimeError, match="Failed to resolve active steering stage"):
        _resolve_active_stage([], progress=0.5)


def test_resolve_steering_applies_missingness_auxiliary_fields() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "steering": {
                "enabled": True,
                "stages": [
                    {
                        "name": "mar_aux",
                        "fraction": 1.0,
                        "dataset": {
                            "missing_rate": [0.1, 0.2],
                            "missing_mechanism": "mar",
                            "missing_mar_observed_fraction": 0.75,
                            "missing_mar_logit_scale": 1.5,
                            "missing_mnar_logit_scale": 2.5,
                        },
                    }
                ],
            }
        }
    )

    resolved = resolve_steering(cfg, dataset_index=1, run_num_datasets=2)

    assert resolved.config.dataset.missing_rate == pytest.approx(0.2)
    assert resolved.config.dataset.missing_mechanism == "mar"
    assert resolved.config.dataset.missing_mar_observed_fraction == pytest.approx(0.75)
    assert resolved.config.dataset.missing_mar_logit_scale == pytest.approx(1.5)
    assert resolved.config.dataset.missing_mnar_logit_scale == pytest.approx(2.5)


def test_resolve_steering_graph_stage_clears_inherited_noise_scale() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "shift": {
                "enabled": True,
                "mode": "custom",
                "variance_scale": 0.4,
            },
            "steering": {
                "enabled": True,
                "stages": [
                    {
                        "name": "graph_only",
                        "fraction": 1.0,
                        "shift": {
                            "mode": "graph_drift",
                            "graph_scale": [0.0, 0.5],
                        },
                    }
                ],
            },
        }
    )

    resolved = resolve_steering(cfg, dataset_index=1, run_num_datasets=2)
    shift_params = resolve_shift_runtime_params(resolved.config)

    assert resolved.config.shift.mode == "graph_drift"
    assert resolved.config.shift.graph_scale == pytest.approx(0.5)
    assert resolved.config.shift.variance_scale is None
    assert resolved.config.shift.mechanism_scale is None
    assert shift_params.variance_sigma_multiplier == pytest.approx(1.0)


def test_resolve_steering_graph_stage_does_not_keep_prior_mixed_noise_scale() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "steering": {
                "enabled": True,
                "stages": [
                    {
                        "name": "mixed_first",
                        "fraction": 0.5,
                        "shift": {
                            "mode": "mixed",
                            "graph_scale": [0.0, 0.5],
                            "variance_scale": [0.0, 0.5],
                        },
                    },
                    {
                        "name": "graph_second",
                        "fraction": 0.5,
                        "shift": {
                            "mode": "graph_drift",
                            "graph_scale": [0.5, 1.0],
                        },
                    },
                ],
            }
        }
    )

    resolved = resolve_steering(cfg, dataset_index=3, run_num_datasets=4)
    shift_params = resolve_shift_runtime_params(resolved.config)

    assert resolved.stage_name == "graph_second"
    assert resolved.config.shift.mode == "graph_drift"
    assert resolved.config.shift.graph_scale == pytest.approx(1.0)
    assert resolved.config.shift.variance_scale is None
    assert resolved.config.shift.mechanism_scale is None
    assert shift_params.variance_sigma_multiplier == pytest.approx(1.0)
    assert shift_params.mechanism_logit_tilt == pytest.approx(0.0)
