"""Deferred filter staged-output helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from dagzoo.core.staged_artifacts import staged_output_path as _staged_output_path
from dagzoo.io.parquet_writer import (
    _close_packed_shard_handles,
    _PackedShardState,
    _write_packed_split,
)
from dagzoo.io.shard_contract import DATASET_CATALOG_FILENAME


@dataclass(slots=True)
class _CuratedShardWriter:
    """Incremental writer state for one curated accepted-only shard."""

    shard_state: _PackedShardState
    final_shard_dir: Path

    @property
    def shard_dir(self) -> Path:
        return self.shard_state.shard_dir


def _ensure_curated_output_dir_safe(out_dir: Path) -> None:
    """Fail fast when curated output already contains shard artifacts."""

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return

    stale = next(out_dir.glob("shard_*"), None)
    if stale is not None:
        raise RuntimeError(
            f"Curated output directory already contains shard data: {out_dir}. "
            "Choose a new --curated-out directory or remove existing shard_* folders first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)


def _create_curated_shard_writer(
    *,
    curated_out_dir: Path,
    shard_name: str,
    staging_token: str,
) -> _CuratedShardWriter:
    """Initialize incremental writer state for one curated shard."""

    shard_dir = _staged_output_path(
        parent_dir=curated_out_dir,
        final_name=shard_name,
        staging_token=staging_token,
    )
    shard_dir.mkdir(parents=True, exist_ok=False)
    final_shard_dir = curated_out_dir / shard_name
    return _CuratedShardWriter(
        shard_state=_PackedShardState(
            shard_dir=shard_dir,
            train_path=shard_dir / "train.parquet",
            test_path=shard_dir / "test.parquet",
            metadata_path=shard_dir / DATASET_CATALOG_FILENAME,
        ),
        final_shard_dir=final_shard_dir,
    )


def _write_curated_split(
    *,
    state: _CuratedShardWriter,
    split: str,
    dataset_index: int,
    x: np.ndarray,
    y: np.ndarray,
    compression: str,
) -> None:
    """Append one accepted dataset split into a curated shard parquet file."""

    _write_packed_split(
        state=state.shard_state,
        split=split,
        dataset_index=dataset_index,
        x=x,
        y=y,
        compression=compression,
    )


def _write_curated_dataset(
    *,
    state: _CuratedShardWriter,
    dataset_index: int,
    train_split: Any,
    test_split: Any,
    record: Mapping[str, Any],
) -> None:
    """Append one accepted dataset to curated shard outputs."""

    _write_curated_split(
        state=state,
        split="train",
        dataset_index=dataset_index,
        x=train_split.x,
        y=train_split.y,
        compression="zstd",
    )
    _write_curated_split(
        state=state,
        split="test",
        dataset_index=dataset_index,
        x=test_split.x,
        y=test_split.y,
        compression="zstd",
    )
    state.shard_state.catalog_records.append(cast(dict[str, Any], dict(record)))


def _close_curated_shard_writer(state: _CuratedShardWriter | None) -> None:
    """Close open parquet and metadata handles for one curated shard."""

    if state is None:
        return
    _close_packed_shard_handles(state.shard_state)


def _consume_expected_split(
    split_iter: Iterator[Any],
    *,
    expected_dataset_index: int,
    split_path: Path,
) -> Any:
    """Consume the next packed split group and validate dataset alignment."""

    try:
        split_dataset = next(split_iter)
    except StopIteration as exc:
        raise ValueError(
            "Missing packed split rows for deferred filtering: "
            f"split={split_path} dataset_index={expected_dataset_index}"
        ) from exc

    if split_dataset.dataset_index != expected_dataset_index:
        raise ValueError(
            "Packed split coverage mismatch for deferred filtering: "
            f"split={split_path} expected dataset_index={expected_dataset_index} "
            f"got={split_dataset.dataset_index}"
        )
    return split_dataset


def _ensure_split_iter_exhausted(
    split_iter: Iterator[Any],
    *,
    split_path: Path,
) -> None:
    """Ensure a packed split iterator has no extra dataset groups beyond metadata."""

    try:
        extra_split = next(split_iter)
    except StopIteration:
        return
    raise ValueError(
        "Packed split contains extra dataset rows beyond metadata coverage: "
        f"split={split_path} dataset_index={extra_split.dataset_index}"
    )
