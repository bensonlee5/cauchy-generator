"""Hardware policy registry for explicit config scaling rules."""

from __future__ import annotations

import math
from collections.abc import Callable

from dagzoo.config import GeneratorConfig, clone_generator_config
from dagzoo.hardware import HardwareInfo

HardwarePolicy = Callable[[GeneratorConfig, HardwareInfo], GeneratorConfig]

_FIXED_LAYOUT_TARGET_CELL_QUANTUM = 8_000_000
_FIXED_LAYOUT_TARGET_CELLS_PER_GB = 2_000_000
_FIXED_LAYOUT_TARGET_CELLS_MIN_BY_TIER: dict[str, int] = {
    "cuda_desktop": 32_000_000,
    "cuda_datacenter": 64_000_000,
    "cuda_h100": 240_000_000,
    "cuda_unknown_fallback": 32_000_000,
}
_FIXED_LAYOUT_TARGET_CELLS_MAX_BY_TIER: dict[str, int] = {
    "cuda_desktop": 96_000_000,
    "cuda_datacenter": 192_000_000,
    "cuda_h100": 256_000_000,
    "cuda_unknown_fallback": 96_000_000,
}
_FIXED_LAYOUT_TARGET_CELLS_DEFAULT_BY_TIER: dict[str, int] = {
    "cuda_desktop": 48_000_000,
    "cuda_datacenter": 96_000_000,
    "cuda_h100": 240_000_000,
    "cuda_unknown_fallback": 48_000_000,
}


def _policy_none(config: GeneratorConfig, hw: HardwareInfo) -> GeneratorConfig:
    """No-op policy that preserves the provided configuration."""

    _ = hw
    return config


def _round_up_to_multiple(value: int, *, multiple: int) -> int:
    """Round ``value`` up to the next positive multiple."""

    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple!r}.")
    if value <= 0:
        return int(multiple)
    return int(((value + multiple - 1) // multiple) * multiple)


def round_fixed_layout_target_cells(value: int) -> int:
    """Round one fixed-layout target-cells value to the shared quantum."""

    return _round_up_to_multiple(int(value), multiple=_FIXED_LAYOUT_TARGET_CELL_QUANTUM)


def resolve_cuda_fixed_layout_target_cells_limits(
    hw: HardwareInfo,
) -> tuple[int | None, int | None]:
    """Return the CUDA fixed-layout target floor and cap for the detected hardware."""

    if hw.backend != "cuda":
        return None, None

    tier = str(hw.tier).strip().lower()
    default_target = _FIXED_LAYOUT_TARGET_CELLS_DEFAULT_BY_TIER.get(tier)
    max_target = _FIXED_LAYOUT_TARGET_CELLS_MAX_BY_TIER.get(tier)
    if default_target is None:
        return None, None
    if max_target is None:
        return None, None

    total_memory_gb = hw.total_memory_gb
    if total_memory_gb is None or total_memory_gb <= 0.0:
        return int(default_target), int(max_target)

    scaled_target = round_fixed_layout_target_cells(
        int(math.ceil(float(total_memory_gb) * _FIXED_LAYOUT_TARGET_CELLS_PER_GB)),
    )
    min_target = _FIXED_LAYOUT_TARGET_CELLS_MIN_BY_TIER[tier]
    return int(max(min_target, min(max_target, scaled_target))), int(max_target)


def _apply_fixed_layout_target_cells_floor(config: GeneratorConfig, *, target_cells: int) -> None:
    """Raise the configured auto-batch target to at least ``target_cells``."""

    current_target = config.runtime.fixed_layout_target_cells
    current_value = 0 if current_target is None else int(current_target)
    config.runtime.fixed_layout_target_cells = max(current_value, int(target_cells))


def _policy_cuda_tiered_v1(config: GeneratorConfig, hw: HardwareInfo) -> GeneratorConfig:
    """Apply coarse CUDA tier overrides modeled after prior runtime defaults."""

    if hw.backend != "cuda":
        return config

    fixed_layout_target_cells, _ = resolve_cuda_fixed_layout_target_cells_limits(hw)

    if hw.tier == "cuda_h100":
        config.benchmark.preset_name = "cuda_h100_auto"
        config.dataset.n_train = max(config.dataset.n_train, 4096)
        config.dataset.n_test = max(config.dataset.n_test, 1024)
        config.dataset.n_features_max = max(config.dataset.n_features_max, 192)
        config.graph.n_nodes_max = max(config.graph.n_nodes_max, 64)
        config.output.shard_size = max(config.output.shard_size, 512)
        config.benchmark.num_datasets = max(config.benchmark.num_datasets, 5000)
        config.benchmark.warmup_datasets = max(config.benchmark.warmup_datasets, 50)
        if fixed_layout_target_cells is not None:
            _apply_fixed_layout_target_cells_floor(
                config,
                target_cells=fixed_layout_target_cells,
            )
        return config

    if hw.tier == "cuda_datacenter":
        config.benchmark.preset_name = "cuda_datacenter_auto"
        config.dataset.n_train = max(config.dataset.n_train, 1536)
        config.dataset.n_test = max(config.dataset.n_test, 512)
        config.dataset.n_features_max = max(config.dataset.n_features_max, 96)
        config.graph.n_nodes_max = max(config.graph.n_nodes_max, 48)
        config.output.shard_size = max(config.output.shard_size, 256)
        config.benchmark.num_datasets = max(config.benchmark.num_datasets, 2500)
        config.benchmark.warmup_datasets = max(config.benchmark.warmup_datasets, 30)
        if fixed_layout_target_cells is not None:
            _apply_fixed_layout_target_cells_floor(
                config,
                target_cells=fixed_layout_target_cells,
            )
        return config

    if hw.tier == "cuda_desktop":
        config.benchmark.preset_name = "cuda_desktop_auto"
        if fixed_layout_target_cells is not None:
            _apply_fixed_layout_target_cells_floor(
                config,
                target_cells=fixed_layout_target_cells,
            )
        return config

    config.benchmark.preset_name = "cuda_unknown_fallback"
    if fixed_layout_target_cells is not None:
        _apply_fixed_layout_target_cells_floor(
            config,
            target_cells=fixed_layout_target_cells,
        )
    return config


_HARDWARE_POLICY_REGISTRY: dict[str, HardwarePolicy] = {
    "none": _policy_none,
    "cuda_tiered_v1": _policy_cuda_tiered_v1,
}


def list_hardware_policies() -> tuple[str, ...]:
    """Return all registered hardware policy names."""

    return tuple(sorted(_HARDWARE_POLICY_REGISTRY))


def register_hardware_policy(
    name: str,
    policy: HardwarePolicy,
    *,
    overwrite: bool = False,
) -> None:
    """Register a named hardware policy callback."""

    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("hardware policy name must be a non-empty string.")
    if normalized_name in _HARDWARE_POLICY_REGISTRY and not overwrite:
        raise ValueError(
            f"hardware policy '{normalized_name}' already exists; pass overwrite=True to replace it."
        )
    _HARDWARE_POLICY_REGISTRY[normalized_name] = policy


def apply_hardware_policy(
    config: GeneratorConfig,
    hw: HardwareInfo,
    *,
    policy_name: str = "none",
    validate: bool = True,
) -> GeneratorConfig:
    """Apply a named hardware policy without mutating the input config."""

    normalized_name = str(policy_name).strip().lower()
    policy = _HARDWARE_POLICY_REGISTRY.get(normalized_name)
    if policy is None:
        raise ValueError(
            "Unknown hardware policy "
            f"{policy_name!r}. Available: {', '.join(list_hardware_policies()) or '(none)'}."
        )

    base = clone_generator_config(config, revalidate=False)
    result = policy(base, hw)
    if not isinstance(result, GeneratorConfig):
        raise TypeError(
            f"hardware policy '{normalized_name}' must return GeneratorConfig, got {type(result)!r}."
        )
    if validate:
        result.validate_generation_constraints()
    return result


__all__ = [
    "HardwarePolicy",
    "apply_hardware_policy",
    "list_hardware_policies",
    "register_hardware_policy",
    "resolve_cuda_fixed_layout_target_cells_limits",
    "round_fixed_layout_target_cells",
]
