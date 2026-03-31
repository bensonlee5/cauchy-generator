"""Fixed-layout run preparation helpers."""

from __future__ import annotations

from dagzoo.config import (
    DatasetRowsSpec,
    GeneratorConfig,
    clone_generator_config,
    dataset_rows_is_variable,
    materialize_stress_profile,
    resolve_dataset_total_rows,
)
from dagzoo.core.generation_context import _resolve_device, _resolve_run_seed
from dagzoo.rng import KeyedRng

from .metadata import _FixedLayoutPlan

_FIXED_LAYOUT_TARGET_CELLS = 4_000_000


def normalize_fixed_layout_target_cells(target_cells: int | None) -> int:
    """Return the effective fixed-layout target cell budget for a runtime value."""

    if target_cells is None:
        return int(_FIXED_LAYOUT_TARGET_CELLS)
    return int(target_cells)


def _validate_fixed_layout_rows_mode(config: GeneratorConfig) -> None:
    if dataset_rows_is_variable(config.dataset.rows):
        raise ValueError(
            "Fixed-layout generation requires a fixed split size; variable dataset.rows "
            "modes (range/choices) are not supported."
        )


def _effective_fixed_layout_target_cells(config: GeneratorConfig) -> int:
    """Return the configured fixed-layout auto-batch target cell budget."""

    return normalize_fixed_layout_target_cells(config.runtime.fixed_layout_target_cells)


def _resolve_fixed_layout_batch_size(
    plan: _FixedLayoutPlan,
    *,
    num_datasets: int,
    batch_size: int | None,
    target_cells: int | None = None,
    batch_size_cap: int | None = None,
) -> int:
    effective_cap = None if batch_size_cap is None else max(1, int(batch_size_cap))
    if batch_size is not None:
        resolved = max(1, min(int(batch_size), int(num_datasets)))
        return resolved if effective_cap is None else min(resolved, effective_cap)
    per_dataset_cells = max(
        1, int(plan.n_train + plan.n_test) * max(1, int(plan.layout.n_features))
    )
    auto_batch = max(1, int(target_cells or _FIXED_LAYOUT_TARGET_CELLS) // per_dataset_cells)
    resolved = max(1, min(int(num_datasets), int(auto_batch)))
    return resolved if effective_cap is None else min(resolved, effective_cap)


def realize_generation_config_for_run(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> tuple[GeneratorConfig, int, str, str]:
    """Resolve one canonical single-run config with rows fixed for the full run."""

    run_seed = _resolve_run_seed(config, seed)
    requested_device = (device or config.runtime.device or "auto").lower()
    resolved_device = _resolve_device(config, device)

    if config.stress.profile is not None:
        config.validate_generation_constraints()
        realized = materialize_stress_profile(
            config,
            revalidate=False,
            clear_selector=True,
        )
    else:
        realized = clone_generator_config(config, revalidate=False)
    rows_seed = KeyedRng(run_seed).child_seed("rows")
    total_rows = resolve_dataset_total_rows(realized.dataset.rows, dataset_seed=rows_seed)
    if total_rows is not None:
        n_test = int(realized.dataset.n_test)
        n_train = int(total_rows) - n_test
        if n_train <= 0:
            raise ValueError(
                "Resolved rows split is invalid: total rows must be > dataset.n_test "
                f"(total_rows={int(total_rows)}, n_test={n_test})."
            )
        realized.dataset.n_train = int(n_train)
        realized.dataset.rows = DatasetRowsSpec(mode="fixed", value=int(total_rows))

    return realized, int(run_seed), str(requested_device), str(resolved_device)


__all__ = [
    "_effective_fixed_layout_target_cells",
    "_resolve_fixed_layout_batch_size",
    "_validate_fixed_layout_rows_mode",
    "normalize_fixed_layout_target_cells",
    "realize_generation_config_for_run",
]
