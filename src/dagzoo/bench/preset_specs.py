"""Benchmark preset spec resolution and smoke-cap helpers."""

from __future__ import annotations

from dataclasses import dataclass

from dagzoo.config import (
    GeneratorConfig,
    clone_generator_config,
)
from dagzoo.core.config_resolution import BenchmarkSmokeCaps, cap_rows_spec_to_total
from dagzoo.recipes import load_config_reference, parse_recipe_reference

from .constants import (
    SMOKE_N_FEATURES_CAP,
    SMOKE_N_NODES_CAP,
    SMOKE_N_TEST_CAP,
    SMOKE_N_TRAIN_CAP,
)

DEFAULT_PRESET_CONFIGS: dict[str, str] = {
    "cpu": "configs/benchmark_cpu.yaml",
    "cuda_desktop": "configs/benchmark_cuda_desktop.yaml",
    "cuda_h100": "configs/benchmark_cuda_h100.yaml",
}
CPU_BENCHMARK_ROW_TOTALS: tuple[int, ...] = (1024, 4096, 8192)


@dataclass(slots=True)
class PresetRunSpec:
    """Benchmark execution spec for one preset/config pair."""

    key: str
    config: GeneratorConfig
    device: str | None = None


def _copy_runtime_config(config: GeneratorConfig) -> GeneratorConfig:
    """Copy an already validated runtime config without re-running schema validation."""

    return clone_generator_config(config, revalidate=False)


def _cpu_row_profile_key(total_rows: int) -> str:
    """Return the derived preset key for one built-in CPU row profile."""

    return f"cpu_rows{int(total_rows)}"


def _is_builtin_cpu_row_profile_key(preset_key: str) -> bool:
    """Return whether `preset_key` is one derived built-in CPU row-profile key."""

    return str(preset_key).startswith("cpu_rows")


def _split_total_rows(total_rows: int) -> tuple[int, int]:
    """Split total rows into the repo-standard 3:1 train/test ratio."""

    n_test = max(1, int(total_rows) // 4)
    n_train = max(1, int(total_rows) - n_test)
    return n_train, n_test


def _cap_smoke_rows_spec(config: GeneratorConfig) -> None:
    """Cap benchmark rows spec to the already-capped smoke split total."""

    cap_rows_spec_to_total(
        config,
        total_rows_cap=int(config.dataset.n_train + config.dataset.n_test),
    )


def _expand_builtin_cpu_run_specs(
    config: GeneratorConfig, *, device: str | None
) -> list[PresetRunSpec]:
    """Expand the built-in CPU preset into explicit row-profile run specs."""

    base_preset = dict(config.benchmark.presets.get("cpu", {}))
    expanded: list[PresetRunSpec] = []
    for total_rows in CPU_BENCHMARK_ROW_TOTALS:
        derived = _copy_runtime_config(config)
        n_train, n_test = _split_total_rows(total_rows)
        derived.dataset.n_train = int(n_train)
        derived.dataset.n_test = int(n_test)
        derived_key = _cpu_row_profile_key(total_rows)
        derived.benchmark.preset_name = derived_key
        derived.benchmark.presets[derived_key] = dict(base_preset)
        expanded.append(PresetRunSpec(key=derived_key, config=derived, device=device))
    return expanded


def _smoke_caps_for_spec(spec: PresetRunSpec) -> BenchmarkSmokeCaps:
    """Resolve smoke-suite caps for one preset spec."""

    if _is_builtin_cpu_row_profile_key(spec.key):
        return BenchmarkSmokeCaps(
            n_train=int(spec.config.dataset.n_train),
            n_test=int(spec.config.dataset.n_test),
            n_features=SMOKE_N_FEATURES_CAP,
            n_nodes=SMOKE_N_NODES_CAP,
        )
    return BenchmarkSmokeCaps(
        n_train=SMOKE_N_TRAIN_CAP,
        n_test=SMOKE_N_TEST_CAP,
        n_features=SMOKE_N_FEATURES_CAP,
        n_nodes=SMOKE_N_NODES_CAP,
    )


def resolve_preset_run_specs(
    *,
    preset_keys: list[str] | None,
    config_path: str | None,
) -> list[PresetRunSpec]:
    """Resolve requested preset keys into concrete benchmark run specs."""

    keys = list(preset_keys or [])
    if "all" in keys:
        keys = ["cpu", "cuda_desktop", "cuda_h100"]

    if not keys:
        keys = ["custom"] if config_path else ["cpu"]

    resolved: list[PresetRunSpec] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)

        if key == "custom":
            if not config_path:
                raise ValueError("Preset 'custom' requires --config.")
            config = load_config_reference(config_path)
            recipe_name = parse_recipe_reference(config_path)
            preset_key = recipe_name or config.benchmark.preset_name or "custom"
            resolved.append(
                PresetRunSpec(key=preset_key, config=config, device=config.runtime.device)
            )
            continue

        config_file = DEFAULT_PRESET_CONFIGS.get(key)
        if not config_file:
            raise ValueError(f"Unknown benchmark preset key: {key}")

        config = GeneratorConfig.from_yaml(config_file)
        preset_device = str(
            config.benchmark.presets.get(key, {}).get("device", config.runtime.device)
        )
        if key == "cpu":
            resolved.extend(_expand_builtin_cpu_run_specs(config, device=preset_device))
        else:
            resolved.append(PresetRunSpec(key=key, config=config, device=preset_device))

    return resolved
