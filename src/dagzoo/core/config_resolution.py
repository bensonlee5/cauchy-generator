"""Shared config-resolution helpers for generate and benchmark command paths."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from dagzoo.config import (
    DATASET_ROWS_MIN_TOTAL,
    DatasetRowsSpec,
    GeneratorConfig,
    clone_generator_config,
    materialize_stress_profile,
    normalize_dataset_rows,
)
from dagzoo.hardware import HardwareInfo, detect_hardware
from dagzoo.hardware_policy import (
    apply_hardware_policy,
    resolve_cuda_fixed_layout_target_cells_limits,
)

_MISSING_VALUE = "<missing>"
_DEFAULT_CUDA_FIXED_LAYOUT_TARGET_SOURCE = "hardware.default_cuda_fixed_layout_target_cells"

type ResolutionEvent = dict[str, Any]
type ResolvedConfigBundle = dict[str, Any]


@dataclass(slots=True, frozen=True)
class BenchmarkSmokeCaps:
    """Hard caps applied to benchmark preset configs in smoke mode."""

    n_train: int
    n_test: int
    n_features: int
    n_nodes: int


def _append_event(
    *,
    events: list[ResolutionEvent],
    path: str,
    source: str,
    old_value: Any,
    new_value: Any,
) -> None:
    """Append one trace event when values differ."""

    if old_value == new_value:
        return
    events.append(
        {
            "path": path,
            "source": source,
            "old_value": old_value,
            "new_value": new_value,
        }
    )


def _set_config_path(
    config: GeneratorConfig,
    *,
    path: str,
    value: Any,
    source: str,
    events: list[ResolutionEvent],
) -> None:
    """Set a dotted dataclass path and emit one trace event when changed."""

    parts = path.split(".")
    target: Any = config
    for part in parts[:-1]:
        target = getattr(target, part)
    field_name = parts[-1]
    old_value = getattr(target, field_name)
    if old_value == value:
        return
    setattr(target, field_name, value)
    _append_event(
        events=events,
        path=path,
        source=source,
        old_value=old_value,
        new_value=value,
    )


def _append_diff_events(
    before: Any,
    after: Any,
    *,
    source: str,
    events: list[ResolutionEvent],
    path: str = "",
) -> None:
    """Append trace events by diffing serialized config payloads."""

    if isinstance(before, dict) and isinstance(after, dict):
        keys = sorted(set(before) | set(after))
        for key in keys:
            child_path = key if not path else f"{path}.{key}"
            _append_diff_events(
                before.get(key, _MISSING_VALUE),
                after.get(key, _MISSING_VALUE),
                source=source,
                events=events,
                path=child_path,
            )
        return
    _append_event(
        events=events,
        path=path,
        source=source,
        old_value=before,
        new_value=after,
    )


def _apply_default_cuda_fixed_layout_target_floor(
    config: GeneratorConfig,
    *,
    hw: HardwareInfo,
    events: list[ResolutionEvent],
) -> None:
    """Fill the fixed-layout target from the default CUDA floor when the config leaves it unset."""

    target_floor, _ = resolve_cuda_fixed_layout_target_cells_limits(hw)
    if target_floor is None or config.runtime.fixed_layout_target_cells is not None:
        return
    _set_config_path(
        config,
        path="runtime.fixed_layout_target_cells",
        value=int(target_floor),
        source=_DEFAULT_CUDA_FIXED_LAYOUT_TARGET_SOURCE,
        events=events,
    )


def _apply_smoke_caps(
    config: GeneratorConfig,
    *,
    smoke_caps: BenchmarkSmokeCaps,
    events: list[ResolutionEvent],
) -> None:
    """Apply benchmark smoke caps and trace each field-level cap event."""

    cap_specs = (
        ("dataset.n_train", int(smoke_caps.n_train)),
        ("dataset.n_test", int(smoke_caps.n_test)),
        ("dataset.n_features_min", int(smoke_caps.n_features)),
        ("dataset.n_features_max", int(smoke_caps.n_features)),
        ("graph.n_nodes_min", int(smoke_caps.n_nodes)),
        ("graph.n_nodes_max", int(smoke_caps.n_nodes)),
    )
    for path, cap_value in cap_specs:
        parts = path.split(".")
        target: Any = config
        for part in parts[:-1]:
            target = getattr(target, part)
        field_name = parts[-1]
        old_value = int(getattr(target, field_name))
        new_value = min(old_value, cap_value)
        if old_value == new_value:
            continue
        setattr(target, field_name, new_value)
        _append_event(
            events=events,
            path=path,
            source="benchmark.suite_smoke_caps",
            old_value=old_value,
            new_value=new_value,
        )


def cap_rows_spec_to_total(config: GeneratorConfig, *, total_rows_cap: int) -> None:
    """Cap ``dataset.rows`` to ``total_rows_cap`` while preserving fixed/range row modes."""

    if int(total_rows_cap) < int(DATASET_ROWS_MIN_TOTAL):
        config.dataset.rows = None
        return

    normalized_rows = normalize_dataset_rows(config.dataset.rows)
    if normalized_rows is None:
        return

    if normalized_rows.mode == "fixed":
        assert normalized_rows.value is not None
        config.dataset.rows = DatasetRowsSpec(
            mode="fixed",
            value=min(int(normalized_rows.value), int(total_rows_cap)),
        )
        return

    assert normalized_rows.start is not None and normalized_rows.stop is not None
    capped_start = min(int(normalized_rows.start), int(total_rows_cap))
    capped_stop = min(int(normalized_rows.stop), int(total_rows_cap))
    if capped_start >= capped_stop:
        config.dataset.rows = DatasetRowsSpec(mode="fixed", value=capped_stop)
        return
    config.dataset.rows = DatasetRowsSpec(mode="range", start=capped_start, stop=capped_stop)


def _resolve_config_with_policy(
    config: GeneratorConfig,
    *,
    device_override: str | None,
    device_source: str,
    hardware_policy: str,
    before_materialization_hook: Callable[[GeneratorConfig, list[ResolutionEvent]], None] | None,
    after_policy_hook: Callable[[GeneratorConfig, list[ResolutionEvent]], None] | None,
) -> ResolvedConfigBundle:
    """Resolve shared device/materialization/policy flow for command paths."""

    resolved = clone_generator_config(config, revalidate=True)
    trace_events: list[ResolutionEvent] = []

    if device_override is not None:
        requested_device = str(device_override).strip().lower() or "auto"
    elif resolved.runtime.device is None:
        requested_device = "auto"
    else:
        requested_device = str(resolved.runtime.device).strip().lower() or "auto"
    _set_config_path(
        resolved,
        path="runtime.device",
        value=requested_device,
        source=device_source,
        events=trace_events,
    )
    if before_materialization_hook is not None:
        before_materialization_hook(resolved, trace_events)

    resolved.validate_generation_constraints()
    carried_stress_profile = (
        None if resolved.stress.profile is None else str(resolved.stress.profile)
    )
    materialized = materialize_stress_profile(
        resolved,
        revalidate=False,
        clear_selector=True,
    )
    append_config_diff_events(
        resolved,
        materialized,
        source="stress.profile_materialization",
        events=trace_events,
    )
    resolved = materialized

    hw = detect_hardware(requested_device)
    before_policy = resolved.to_dict()
    resolved = apply_hardware_policy(
        resolved,
        hw,
        policy_name=hardware_policy,
        validate=False,
    )
    after_policy = resolved.to_dict()
    _append_diff_events(
        before_policy,
        after_policy,
        source=f"hardware_policy.{str(hardware_policy).strip().lower()}",
        events=trace_events,
    )
    _apply_default_cuda_fixed_layout_target_floor(resolved, hw=hw, events=trace_events)
    if after_policy_hook is not None:
        after_policy_hook(resolved, trace_events)
    resolved.validate_generation_constraints()
    return {
        "config": resolved,
        "hardware": hw,
        "requested_device": requested_device,
        "carried_stress_profile": carried_stress_profile,
        "trace_events": trace_events,
    }


def resolve_generate_config(
    config: GeneratorConfig,
    *,
    device_override: str | None,
    rows: object | None,
    rows_source: str = "cli.rows",
    hardware_policy: str,
    diagnostics_enabled: bool,
    path_overrides: Sequence[tuple[str, Any]] | None = None,
    post_policy_hook: Callable[[GeneratorConfig, list[ResolutionEvent]], None] | None = None,
) -> ResolvedConfigBundle:
    """Resolve effective config for one generate command invocation."""

    def _before_materialization(
        resolved: GeneratorConfig,
        trace_events: list[ResolutionEvent],
    ) -> None:
        if rows is not None:
            _set_config_path(
                resolved,
                path="dataset.rows",
                value=rows,
                source=rows_source,
                events=trace_events,
            )
        if path_overrides:
            for path, value in path_overrides:
                _set_config_path(
                    resolved,
                    path=path,
                    value=value,
                    source="cli.set",
                    events=trace_events,
                )
        if diagnostics_enabled:
            _set_config_path(
                resolved,
                path="diagnostics.enabled",
                value=True,
                source="cli.diagnostics",
                events=trace_events,
            )

    return _resolve_config_with_policy(
        config,
        device_override=device_override,
        device_source="cli.device",
        hardware_policy=hardware_policy,
        before_materialization_hook=_before_materialization,
        after_policy_hook=post_policy_hook,
    )


def resolve_benchmark_preset_config(
    *,
    preset_key: str,
    config: GeneratorConfig,
    preset_device: str | None,
    suite: str,
    hardware_policy: str,
    smoke_caps: BenchmarkSmokeCaps | None,
) -> ResolvedConfigBundle:
    """Resolve effective config for one benchmark preset run."""

    def _after_policy(resolved: GeneratorConfig, trace_events: list[ResolutionEvent]) -> None:
        if str(suite).strip().lower() != "smoke":
            return
        if smoke_caps is None:
            raise ValueError("Benchmark smoke suite config resolution requires smoke cap values.")
        _apply_smoke_caps(resolved, smoke_caps=smoke_caps, events=trace_events)

    resolved = _resolve_config_with_policy(
        config,
        device_override=preset_device,
        device_source="benchmark.preset_device",
        hardware_policy=hardware_policy,
        before_materialization_hook=None,
        after_policy_hook=_after_policy,
    )
    return {
        "preset_key": preset_key,
        **resolved,
    }


def serialize_resolution_events(events: list[ResolutionEvent]) -> list[dict[str, Any]]:
    """Convert trace payloads into JSON/YAML-safe dictionaries."""

    return [dict(event) for event in events]


def append_config_diff_events(
    before: GeneratorConfig,
    after: GeneratorConfig,
    *,
    source: str,
    events: list[ResolutionEvent],
) -> None:
    """Append trace events describing config differences between two states."""

    _append_diff_events(
        before.to_dict(),
        after.to_dict(),
        source=source,
        events=events,
    )


__all__ = [
    "BenchmarkSmokeCaps",
    "append_config_diff_events",
    "cap_rows_spec_to_total",
    "resolve_benchmark_preset_config",
    "resolve_generate_config",
    "serialize_resolution_events",
]
