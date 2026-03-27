"""Dataset row-spec models and helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass

from dagzoo.rng import derive_seed

from .constants import (
    DATASET_ROWS_MAX_TOTAL,
    DATASET_ROWS_MIN_TOTAL,
    RowsMode,
)
from .scalars import _validate_int_field


@dataclass(slots=True)
class DatasetRowsSpec:
    """Normalized dataset total-row sampling spec."""

    mode: RowsMode
    value: int | None = None
    start: int | None = None
    stop: int | None = None


def _validate_rows_total(
    *,
    field_name: str,
    value: object,
) -> int:
    """Validate one dataset total-rows value against supported bounds."""

    return _validate_int_field(
        field_name=field_name,
        value=value,
        minimum=DATASET_ROWS_MIN_TOTAL,
        maximum=DATASET_ROWS_MAX_TOTAL,
    )


def normalize_dataset_rows(value: object | None) -> DatasetRowsSpec | None:
    """Normalize dataset.rows into a validated internal row-spec representation."""

    if value is None:
        return None
    if isinstance(value, dict):
        mode = value.get("mode")
        if mode == "fixed":
            return normalize_dataset_rows(DatasetRowsSpec(mode="fixed", value=value.get("value")))
        if mode == "range":
            return normalize_dataset_rows(
                DatasetRowsSpec(
                    mode="range",
                    start=value.get("start"),
                    stop=value.get("stop"),
                )
            )
        raise ValueError("dataset.rows mapping mode must be 'fixed' or 'range'.")
    if isinstance(value, DatasetRowsSpec):
        if value.mode == "fixed":
            if value.value is None:
                raise ValueError("dataset.rows fixed mode requires a value.")
            return DatasetRowsSpec(
                mode="fixed",
                value=_validate_rows_total(field_name="dataset.rows", value=value.value),
            )
        if value.mode == "range":
            if value.start is None or value.stop is None:
                raise ValueError("dataset.rows range mode requires start/stop.")
            start = _validate_rows_total(field_name="dataset.rows.start", value=value.start)
            stop = _validate_rows_total(field_name="dataset.rows.stop", value=value.stop)
            if start > stop:
                raise ValueError(
                    f"dataset.rows range start must be <= stop, got start={start} stop={stop}."
                )
            if start == stop:
                return DatasetRowsSpec(mode="fixed", value=start)
            return DatasetRowsSpec(mode="range", start=start, stop=stop)
        raise ValueError(f"dataset.rows mode must be fixed or range (got {value.mode!r}).")

    if isinstance(value, bool):
        raise ValueError("dataset.rows must be an integer, range string, or null.")
    if isinstance(value, int):
        return DatasetRowsSpec(
            mode="fixed", value=_validate_rows_total(field_name="dataset.rows", value=value)
        )
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("dataset.rows must be a non-empty integer or range string.")
        if ".." in normalized:
            parts = normalized.split("..")
            if len(parts) != 2:
                raise ValueError(
                    "dataset.rows range must use 'start..stop' with one '..' delimiter."
                )
            start = _validate_rows_total(field_name="dataset.rows.start", value=parts[0].strip())
            stop = _validate_rows_total(field_name="dataset.rows.stop", value=parts[1].strip())
            if start > stop:
                raise ValueError(
                    f"dataset.rows range start must be <= stop, got start={start} stop={stop}."
                )
            if start == stop:
                return DatasetRowsSpec(mode="fixed", value=start)
            return DatasetRowsSpec(mode="range", start=start, stop=stop)
        return DatasetRowsSpec(
            mode="fixed",
            value=_validate_rows_total(field_name="dataset.rows", value=normalized),
        )
    raise ValueError("dataset.rows must be an integer, range string, or null.")


def dataset_rows_bounds(rows: object | None) -> tuple[int, int] | None:
    """Return min/max total rows for a normalized rows spec."""

    normalized_rows = normalize_dataset_rows(rows)
    if normalized_rows is None:
        return None
    if normalized_rows.mode == "fixed":
        assert normalized_rows.value is not None
        return int(normalized_rows.value), int(normalized_rows.value)
    if normalized_rows.mode == "range":
        assert normalized_rows.start is not None and normalized_rows.stop is not None
        return int(normalized_rows.start), int(normalized_rows.stop)
    raise ValueError(f"Unsupported dataset.rows mode {normalized_rows.mode!r}.")


def dataset_rows_is_variable(rows: object | None) -> bool:
    """Return whether rows spec varies per dataset."""

    normalized_rows = normalize_dataset_rows(rows)
    return normalized_rows is not None and normalized_rows.mode == "range"


def resolve_dataset_total_rows(
    rows: object | None,
    *,
    dataset_seed: int | None,
) -> int | None:
    """Resolve one total-rows value from a normalized rows spec."""

    normalized_rows = normalize_dataset_rows(rows)
    if normalized_rows is None:
        return None
    if normalized_rows.mode == "fixed":
        assert normalized_rows.value is not None
        return int(normalized_rows.value)
    if dataset_seed is None:
        raise ValueError(
            "Variable dataset.rows modes require dataset seed context to resolve total rows."
        )
    selector_seed = derive_seed(int(dataset_seed), "rows")
    rng = random.Random(selector_seed)
    if normalized_rows.mode == "range":
        assert normalized_rows.start is not None and normalized_rows.stop is not None
        return int(rng.randint(int(normalized_rows.start), int(normalized_rows.stop)))
    raise ValueError(f"Unsupported dataset.rows mode {normalized_rows.mode!r}.")


def validate_class_split_feasibility(
    *,
    n_classes: int,
    n_train: int,
    n_test: int,
    context: str,
) -> None:
    """Validate whether split sizes can represent all classes in both train and test."""

    if n_classes > n_train or n_classes > n_test:
        raise ValueError(
            f"{context}: infeasible class/split combination for classification "
            f"(n_classes={n_classes}, n_train={n_train}, n_test={n_test}). "
            "Require n_train and n_test to each be >= n_classes."
        )
