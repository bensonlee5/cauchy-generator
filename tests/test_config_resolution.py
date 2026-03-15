import pytest
from conftest import load_repo_config
from hypothesis import given, settings
from hypothesis import strategies as st

from dagzoo.bench.constants import (
    SMOKE_N_FEATURES_CAP,
    SMOKE_N_NODES_CAP,
    SMOKE_N_TEST_CAP,
    SMOKE_N_TRAIN_CAP,
)
from dagzoo.config import (
    DATASET_ROWS_MAX_TOTAL,
    DATASET_ROWS_MIN_TOTAL,
    DatasetRowsSpec,
    GeneratorConfig,
    dataset_rows_bounds,
    normalize_dataset_rows,
)
from dagzoo.core.config_resolution import (
    BenchmarkSmokeCaps,
    cap_rows_spec_to_total,
    resolve_benchmark_preset_config,
    resolve_generate_config,
    serialize_resolution_events,
)
from dagzoo.hardware import HardwareInfo

_ROW_TOTAL_STRATEGY = st.integers(
    min_value=DATASET_ROWS_MIN_TOTAL,
    max_value=DATASET_ROWS_MAX_TOTAL,
)
_TOTAL_ROWS_CAP_STRATEGY = st.integers(
    min_value=DATASET_ROWS_MIN_TOTAL,
    max_value=DATASET_ROWS_MAX_TOTAL,
)


@st.composite
def _cap_rows_spec_strategy(draw: st.DrawFn) -> DatasetRowsSpec:
    mode = draw(st.sampled_from(("fixed", "range", "choices")))
    if mode == "fixed":
        value = draw(_ROW_TOTAL_STRATEGY)
        return DatasetRowsSpec(mode="fixed", value=value)
    if mode == "range":
        start = draw(
            st.integers(
                min_value=DATASET_ROWS_MIN_TOTAL,
                max_value=DATASET_ROWS_MAX_TOTAL - 1,
            )
        )
        stop = draw(st.integers(min_value=start + 1, max_value=DATASET_ROWS_MAX_TOTAL))
        return DatasetRowsSpec(mode="range", start=start, stop=stop)

    choices = draw(st.lists(_ROW_TOTAL_STRATEGY, min_size=2, max_size=5, unique=True))
    return DatasetRowsSpec(mode="choices", choices=choices)


def _mock_cuda_h100(_requested_device: str) -> HardwareInfo:
    return HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="NVIDIA H100 SXM",
        total_memory_gb=80.0,
        peak_flops=989e12,
        tier="cuda_h100",
    )


def test_resolve_generate_config_applies_cli_overrides() -> None:
    cfg = load_repo_config()
    resolved = resolve_generate_config(
        cfg,
        device_override="cpu",
        rows=None,
        hardware_policy="none",
        missing_rate=0.2,
        missing_mechanism="mnar",
        missing_mar_observed_fraction=0.7,
        missing_mar_logit_scale=1.5,
        missing_mnar_logit_scale=2.5,
        diagnostics_enabled=True,
    )
    assert cfg.runtime.device == "auto"
    assert resolved.requested_device == "cpu"
    assert resolved.config.runtime.device == "cpu"
    assert resolved.config.dataset.missing_rate == pytest.approx(0.2)
    assert resolved.config.dataset.missing_mechanism == "mnar"
    assert resolved.config.dataset.missing_mar_observed_fraction == pytest.approx(0.7)
    assert resolved.config.dataset.missing_mar_logit_scale == pytest.approx(1.5)
    assert resolved.config.dataset.missing_mnar_logit_scale == pytest.approx(2.5)
    assert resolved.config.diagnostics.enabled is True

    trace = serialize_resolution_events(resolved.trace_events)
    assert any(
        event["path"] == "runtime.device" and event["source"] == "cli.device" for event in trace
    )
    assert any(
        event["path"] == "dataset.missing_rate" and event["source"] == "cli.missingness_override"
        for event in trace
    )
    assert any(
        event["path"] == "diagnostics.enabled" and event["source"] == "cli.diagnostics"
        for event in trace
    )


def test_resolve_generate_config_applies_rows_override() -> None:
    cfg = load_repo_config()
    resolved = resolve_generate_config(
        cfg,
        device_override="cpu",
        rows="400..60000",
        hardware_policy="none",
        missing_rate=None,
        missing_mechanism=None,
        missing_mar_observed_fraction=None,
        missing_mar_logit_scale=None,
        missing_mnar_logit_scale=None,
        diagnostics_enabled=False,
    )
    assert resolved.config.dataset.rows is not None
    assert resolved.config.dataset.rows.mode == "range"
    assert resolved.config.dataset.rows.start == 400
    assert resolved.config.dataset.rows.stop == 60000

    trace = serialize_resolution_events(resolved.trace_events)
    assert any(event["path"] == "dataset.rows" and event["source"] == "cli.rows" for event in trace)


def test_cap_rows_spec_to_total_clears_rows_when_cap_is_below_min_total() -> None:
    cfg = load_repo_config()
    cfg.dataset.rows = 1024  # type: ignore[assignment]

    cap_rows_spec_to_total(cfg, total_rows_cap=160)

    assert cfg.dataset.rows is None


@pytest.mark.parametrize(
    ("rows_spec", "expected_mode", "expected_payload"),
    [
        (1024, "fixed", {"value": 500}),
        ("400..60000", "range", {"start": 400, "stop": 500}),
        ("400,600,800", "choices", {"choices": [400, 500]}),
    ],
)
def test_cap_rows_spec_to_total_caps_fixed_range_and_choice_rows(
    rows_spec: object,
    expected_mode: str,
    expected_payload: dict[str, int | list[int]],
) -> None:
    cfg = load_repo_config()
    cfg.dataset.rows = rows_spec  # type: ignore[assignment]

    cap_rows_spec_to_total(cfg, total_rows_cap=500)

    assert cfg.dataset.rows is not None
    assert cfg.dataset.rows.mode == expected_mode
    for field_name, expected_value in expected_payload.items():
        assert getattr(cfg.dataset.rows, field_name) == expected_value


@settings(max_examples=100, deadline=None)
@given(rows_spec=_cap_rows_spec_strategy(), total_rows_cap=_TOTAL_ROWS_CAP_STRATEGY)
def test_cap_rows_spec_to_total_preserves_mode_contracts_hypothesis(
    rows_spec: DatasetRowsSpec,
    total_rows_cap: int,
) -> None:
    cfg = GeneratorConfig.from_dict({})
    cfg.dataset.rows = rows_spec

    cap_rows_spec_to_total(cfg, total_rows_cap=total_rows_cap)

    assert cfg.dataset.rows is not None
    capped_rows = normalize_dataset_rows(cfg.dataset.rows)
    assert capped_rows is not None

    bounds = dataset_rows_bounds(capped_rows)
    assert bounds is not None
    assert bounds[0] <= total_rows_cap
    assert bounds[1] <= total_rows_cap

    if rows_spec.mode == "fixed":
        assert rows_spec.value is not None
        assert capped_rows.mode == "fixed"
        assert capped_rows.value == min(int(rows_spec.value), int(total_rows_cap))
        return

    if rows_spec.mode == "range":
        assert rows_spec.start is not None and rows_spec.stop is not None
        expected_start = min(int(rows_spec.start), int(total_rows_cap))
        expected_stop = min(int(rows_spec.stop), int(total_rows_cap))
        if expected_start == expected_stop:
            assert capped_rows.mode == "fixed"
            assert capped_rows.value == expected_stop
            return

        assert capped_rows.mode == "range"
        assert capped_rows.start == expected_start
        assert capped_rows.stop == expected_stop
        return

    expected_choices = sorted(
        {min(int(choice), int(total_rows_cap)) for choice in rows_spec.choices}
    )
    if len(expected_choices) == 1:
        assert capped_rows.mode == "fixed"
        assert capped_rows.value == expected_choices[0]
        return

    assert capped_rows.mode == "choices"
    assert capped_rows.choices == expected_choices
    assert all(choice <= total_rows_cap for choice in capped_rows.choices)


def test_resolve_generate_config_rejects_invalid_missingness_combination() -> None:
    cfg = load_repo_config()
    with pytest.raises(
        ValueError,
        match="dataset.missing_mechanism must be mcar, mar, or mnar when dataset.missing_rate > 0",
    ):
        resolve_generate_config(
            cfg,
            device_override="cpu",
            rows=None,
            hardware_policy="none",
            missing_rate=0.2,
            missing_mechanism="none",
            missing_mar_observed_fraction=None,
            missing_mar_logit_scale=None,
            missing_mnar_logit_scale=None,
            diagnostics_enabled=False,
        )


def test_resolve_generate_config_treats_null_runtime_device_as_auto() -> None:
    cfg = load_repo_config()
    cfg.runtime.device = None  # type: ignore[assignment]

    resolved = resolve_generate_config(
        cfg,
        device_override=None,
        rows=None,
        hardware_policy="none",
        missing_rate=None,
        missing_mechanism=None,
        missing_mar_observed_fraction=None,
        missing_mar_logit_scale=None,
        missing_mnar_logit_scale=None,
        diagnostics_enabled=False,
    )

    assert resolved.requested_device == "auto"
    assert resolved.config.runtime.device == "auto"


def test_resolve_generate_config_applies_rows_override_after_policy_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dagzoo.core.config_resolution.detect_hardware", _mock_cuda_h100)
    cfg = load_repo_config()
    cfg.dataset.rows = "400..60000"  # type: ignore[assignment]

    resolved = resolve_generate_config(
        cfg,
        device_override="cuda",
        rows="2000..60000",
        hardware_policy="cuda_tiered_v1",
        missing_rate=None,
        missing_mechanism=None,
        missing_mar_observed_fraction=None,
        missing_mar_logit_scale=None,
        missing_mnar_logit_scale=None,
        diagnostics_enabled=False,
    )

    assert resolved.config.dataset.rows is not None
    assert resolved.config.dataset.rows.mode == "range"
    assert resolved.config.dataset.rows.start == 2000
    assert resolved.config.dataset.rows.stop == 60000
    assert resolved.config.dataset.n_test == 1024
    assert resolved.config.runtime.fixed_layout_target_cells == 160_000_000

    trace = serialize_resolution_events(resolved.trace_events)
    assert any(
        event["path"] == "runtime.fixed_layout_target_cells"
        and event["source"] == "hardware_policy.cuda_tiered_v1"
        and event["new_value"] == 160_000_000
        for event in trace
    )


def test_resolve_generate_config_applies_default_cuda_fixed_layout_floor_without_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dagzoo.core.config_resolution.detect_hardware", _mock_cuda_h100)
    cfg = load_repo_config()
    assert cfg.runtime.fixed_layout_target_cells is None

    resolved = resolve_generate_config(
        cfg,
        device_override="cuda",
        rows=None,
        hardware_policy="none",
        missing_rate=None,
        missing_mechanism=None,
        missing_mar_observed_fraction=None,
        missing_mar_logit_scale=None,
        missing_mnar_logit_scale=None,
        diagnostics_enabled=False,
    )

    assert resolved.config.runtime.fixed_layout_target_cells == 160_000_000
    trace = serialize_resolution_events(resolved.trace_events)
    assert any(
        event["path"] == "runtime.fixed_layout_target_cells"
        and event["source"] == "hardware.default_cuda_fixed_layout_target_cells"
        and event["old_value"] is None
        and event["new_value"] == 160_000_000
        for event in trace
    )


def test_resolve_generate_config_preserves_explicit_cuda_target_without_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dagzoo.core.config_resolution.detect_hardware", _mock_cuda_h100)
    cfg = load_repo_config()
    cfg.runtime.fixed_layout_target_cells = 32_000_000

    resolved = resolve_generate_config(
        cfg,
        device_override="cuda",
        rows=None,
        hardware_policy="none",
        missing_rate=None,
        missing_mechanism=None,
        missing_mar_observed_fraction=None,
        missing_mar_logit_scale=None,
        missing_mnar_logit_scale=None,
        diagnostics_enabled=False,
    )

    assert resolved.config.runtime.fixed_layout_target_cells == 32_000_000
    trace = serialize_resolution_events(resolved.trace_events)
    assert not any(
        event["path"] == "runtime.fixed_layout_target_cells"
        and event["source"] == "hardware.default_cuda_fixed_layout_target_cells"
        for event in trace
    )


def test_resolve_benchmark_preset_config_applies_smoke_caps() -> None:
    cfg = load_repo_config("benchmark_cpu.yaml")
    assert cfg.dataset.n_train > SMOKE_N_TRAIN_CAP
    assert cfg.dataset.n_test == SMOKE_N_TEST_CAP
    assert cfg.dataset.n_features_max > SMOKE_N_FEATURES_CAP
    assert cfg.graph.n_nodes_max > SMOKE_N_NODES_CAP

    resolved = resolve_benchmark_preset_config(
        preset_key="cpu",
        config=cfg,
        preset_device="cpu",
        suite="smoke",
        hardware_policy="none",
        smoke_caps=BenchmarkSmokeCaps(
            n_train=SMOKE_N_TRAIN_CAP,
            n_test=SMOKE_N_TEST_CAP,
            n_features=SMOKE_N_FEATURES_CAP,
            n_nodes=SMOKE_N_NODES_CAP,
        ),
    )
    assert cfg.dataset.n_train > SMOKE_N_TRAIN_CAP
    assert resolved.config.dataset.n_train == SMOKE_N_TRAIN_CAP
    assert resolved.config.dataset.n_test == SMOKE_N_TEST_CAP
    assert resolved.config.dataset.n_features_max == SMOKE_N_FEATURES_CAP
    assert resolved.config.graph.n_nodes_max == SMOKE_N_NODES_CAP

    trace = serialize_resolution_events(resolved.trace_events)
    assert any(
        event["path"] == "dataset.n_train"
        and event["source"] == "benchmark.suite_smoke_caps"
        and event["new_value"] == SMOKE_N_TRAIN_CAP
        for event in trace
    )


def test_resolve_benchmark_preset_config_revalidates_after_smoke_caps() -> None:
    cfg = GeneratorConfig.from_dict(
        {
            "dataset": {
                "task": "classification",
                "n_train": 32,
                "n_test": 32,
                "n_classes_min": 2,
                "n_classes_max": 16,
            }
        }
    )

    with pytest.raises(
        ValueError,
        match=r"dataset classification split constraints for n_classes_max",
    ):
        resolve_benchmark_preset_config(
            preset_key="cpu",
            config=cfg,
            preset_device="cpu",
            suite="smoke",
            hardware_policy="none",
            smoke_caps=BenchmarkSmokeCaps(
                n_train=8,
                n_test=8,
                n_features=SMOKE_N_FEATURES_CAP,
                n_nodes=SMOKE_N_NODES_CAP,
            ),
        )


def test_resolve_benchmark_preset_config_treats_null_runtime_device_as_auto() -> None:
    cfg = load_repo_config("benchmark_cpu.yaml")
    cfg.runtime.device = None  # type: ignore[assignment]

    resolved = resolve_benchmark_preset_config(
        preset_key="cpu",
        config=cfg,
        preset_device=None,
        suite="standard",
        hardware_policy="none",
        smoke_caps=None,
    )

    assert resolved.requested_device == "auto"
    assert resolved.config.runtime.device == "auto"


def test_resolve_benchmark_preset_config_requires_smoke_caps_for_smoke_suite() -> None:
    cfg = load_repo_config("benchmark_cpu.yaml")
    with pytest.raises(
        ValueError, match="Benchmark smoke suite config resolution requires smoke cap values"
    ):
        resolve_benchmark_preset_config(
            preset_key="cpu",
            config=cfg,
            preset_device="cpu",
            suite="smoke",
            hardware_policy="none",
            smoke_caps=None,
        )


def test_resolve_benchmark_preset_config_preserves_dataset_rows_for_standard_suite() -> None:
    cfg = load_repo_config("benchmark_cpu.yaml")
    cfg.dataset.rows = "2000..60000"  # type: ignore[assignment]

    resolved = resolve_benchmark_preset_config(
        preset_key="cpu",
        config=cfg,
        preset_device="cpu",
        suite="standard",
        hardware_policy="none",
        smoke_caps=None,
    )

    assert resolved.config.dataset.rows is not None
    assert resolved.config.dataset.rows.mode == "range"
    assert resolved.config.dataset.rows.start == 2000
    assert resolved.config.dataset.rows.stop == 60000


def test_resolve_benchmark_preset_config_preserves_dataset_rows_after_policy_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dagzoo.core.config_resolution.detect_hardware", _mock_cuda_h100)
    cfg = load_repo_config("benchmark_cpu.yaml")
    cfg.dataset.rows = "2000..60000"  # type: ignore[assignment]

    resolved = resolve_benchmark_preset_config(
        preset_key="cpu",
        config=cfg,
        preset_device="cuda",
        suite="standard",
        hardware_policy="cuda_tiered_v1",
        smoke_caps=None,
    )

    # Benchmark preset resolution now permits variable rows; smoke-suite row
    # capping and one-time run realization happen later in benchmark orchestration.
    assert resolved.config.dataset.rows is not None
    assert resolved.config.dataset.rows.mode == "range"
    assert resolved.config.dataset.rows.start == 2000
    assert resolved.config.dataset.rows.stop == 60000
    assert resolved.requested_device == "cuda"
    assert resolved.config.dataset.n_test == 1024
    assert resolved.config.runtime.fixed_layout_target_cells == 160_000_000


def test_resolve_benchmark_preset_config_preserves_explicit_cuda_budget_without_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dagzoo.core.config_resolution.detect_hardware", _mock_cuda_h100)
    cfg = load_repo_config("benchmark_cuda_h100.yaml")
    assert cfg.runtime.fixed_layout_target_cells == 32_000_000

    resolved = resolve_benchmark_preset_config(
        preset_key="cuda_h100",
        config=cfg,
        preset_device=None,
        suite="standard",
        hardware_policy="none",
        smoke_caps=None,
    )

    assert resolved.requested_device == "cuda"
    assert resolved.config.runtime.fixed_layout_target_cells == 32_000_000
    trace = serialize_resolution_events(resolved.trace_events)
    assert not any(
        event["path"] == "runtime.fixed_layout_target_cells"
        and event["source"] == "hardware.default_cuda_fixed_layout_target_cells"
        for event in trace
    )


def test_resolve_benchmark_preset_config_applies_default_cuda_floor_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dagzoo.core.config_resolution.detect_hardware", _mock_cuda_h100)
    cfg = load_repo_config("benchmark_cuda_h100.yaml")
    cfg.runtime.fixed_layout_target_cells = None

    resolved = resolve_benchmark_preset_config(
        preset_key="cuda_h100",
        config=cfg,
        preset_device=None,
        suite="standard",
        hardware_policy="none",
        smoke_caps=None,
    )

    assert resolved.requested_device == "cuda"
    assert resolved.config.runtime.fixed_layout_target_cells == 160_000_000
    trace = serialize_resolution_events(resolved.trace_events)
    assert any(
        event["path"] == "runtime.fixed_layout_target_cells"
        and event["source"] == "hardware.default_cuda_fixed_layout_target_cells"
        and event["old_value"] is None
        and event["new_value"] == 160_000_000
        for event in trace
    )
