"""Deferred CPU filtering over persisted shard outputs."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dagzoo.config import FilterConfig
from dagzoo.core.staged_artifacts import cleanup_path as _cleanup_path
from dagzoo.core.staged_artifacts import promote_staged_path as _promote_staged_path
from dagzoo.core.staged_artifacts import staged_output_path as _staged_output_path
from dagzoo.filtering.deferred_filter_artifacts import (
    _close_curated_shard_writer,
    _consume_expected_split,
    _create_curated_shard_writer,
    _CuratedShardWriter,
    _ensure_curated_output_dir_safe,
    _ensure_split_iter_exhausted,
    _write_curated_dataset,
    _write_ndjson_record,
)
from dagzoo.filtering.deferred_filter_replay import (
    _build_filter_metadata,
    _resolve_filter_config,
    _resolve_filter_seed,
)
from dagzoo.filtering.structural_filter import STRUCTURAL_FILTER_MODE, apply_structural_filter
from dagzoo.io.parquet_writer import (
    _require_pyarrow,
    pq,
)
from dagzoo.io.shard_contract import (
    DATASET_CATALOG_FILENAME,
    REPLAY_CATALOG_FILENAME,
    internal_shard_dir,
    iter_ndjson_records,
    resolve_internal_root,
)
from dagzoo.math import sanitize_json as _sanitize_json

MANIFEST_FILENAME = "filter_manifest.ndjson"
SUMMARY_FILENAME = "filter_summary.json"


@dataclass(slots=True)
class DeferredFilterRunResult:
    """Result payload for one deferred filter command execution."""

    manifest_path: Path
    summary_path: Path
    total_datasets: int
    accepted_datasets: int
    rejected_datasets: int
    elapsed_seconds: float
    datasets_per_minute: float
    curated_out_dir: Path | None = None
    curated_accepted_datasets: int = 0


@dataclass(slots=True)
class _PackedSplitDataset:
    """One dataset worth of packed parquet rows for a single split."""

    dataset_index: int
    x: np.ndarray
    y: np.ndarray


def _discover_shard_dirs(input_path: Path) -> list[Path]:
    """Resolve shard directories from a root output dir or a direct shard path."""

    if input_path.is_dir() and _catalog_path_for_shard(input_path) is not None:
        return [input_path]

    shards = sorted(p for p in input_path.glob("shard_*") if p.is_dir())
    shards = [p for p in shards if _catalog_path_for_shard(p) is not None]
    if shards:
        return shards

    raise FileNotFoundError(
        "No shard directories found under input path. "
        f"Expected either <dir>/{DATASET_CATALOG_FILENAME} "
        f"or <dir>/shard_*/{DATASET_CATALOG_FILENAME}: {input_path}"
    )


def _catalog_path_for_shard(shard_dir: Path) -> Path | None:
    """Return the public catalog path for one shard."""

    candidate = shard_dir / DATASET_CATALOG_FILENAME
    return candidate if candidate.exists() else None


def _replay_catalog_path_for_shard(shard_dir: Path, *, internal_root: Path) -> Path:
    """Return the replay metadata path for one shard."""

    private_candidate = (
        internal_shard_dir(
            internal_root=internal_root,
            shard_name=shard_dir.name,
        )
        / REPLAY_CATALOG_FILENAME
    )
    if private_candidate.exists():
        return private_candidate
    raise FileNotFoundError(
        f"Missing replay metadata for shard {shard_dir}. Expected {private_candidate}."
    )


def _ensure_filter_output_dir_safe(out_dir: Path) -> None:
    """Fail fast when deferred-filter output already contains prior artifacts."""

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return

    stale_paths = [out_dir / MANIFEST_FILENAME, out_dir / SUMMARY_FILENAME]
    stale = next((path for path in stale_paths if path.exists()), None)
    if stale is not None:
        raise RuntimeError(
            f"Deferred filter output directory already contains prior artifacts: {out_dir}. "
            f"Remove {stale.name} or choose a new --out directory."
        )

    out_dir.mkdir(parents=True, exist_ok=True)


def _iter_metadata_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield NDJSON records for one shard."""

    yield from iter_ndjson_records(path)


def _build_packed_split_dataset(
    *,
    dataset_index: int,
    x_chunks: list[np.ndarray],
    y_chunks: list[np.ndarray],
    split_path: Path,
) -> _PackedSplitDataset:
    """Convert one accumulated packed split group into NumPy arrays."""

    if not x_chunks:
        x_np = np.empty((0, 0), dtype=np.float32)
    elif len(x_chunks) == 1:
        x_np = np.asarray(x_chunks[0], dtype=np.float32, copy=False)
    else:
        x_np = np.concatenate(x_chunks, axis=0).astype(np.float32, copy=False)
    if not y_chunks:
        y_np = np.empty((0,), dtype=np.float32)
    elif len(y_chunks) == 1:
        y_np = np.asarray(y_chunks[0])
    else:
        y_np = np.concatenate(y_chunks, axis=0)
    if x_np.ndim != 2:
        raise ValueError(
            "Invalid packed feature shape while replaying deferred filter: "
            f"split={split_path} dataset_index={dataset_index} shape={x_np.shape}"
        )
    if y_np.ndim != 1:
        y_np = np.asarray(y_np).reshape(-1)
    return _PackedSplitDataset(dataset_index=dataset_index, x=x_np, y=y_np)


def _packed_feature_column_to_numpy_matrix(
    *,
    feature_column: Any,
    split_path: Path,
    dataset_index: int,
) -> np.ndarray:
    """Convert one packed list column batch into a dense 2D NumPy matrix."""

    offsets = np.asarray(feature_column.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    n_rows = max(0, int(offsets.shape[0] - 1))
    if n_rows == 0:
        return np.empty((0, 0), dtype=np.float32)

    base_offset = int(offsets[0])
    normalized_offsets = offsets - base_offset
    row_widths = np.diff(normalized_offsets)
    if row_widths.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    expected_width = int(row_widths[0])
    if np.any(row_widths != expected_width):
        raise ValueError(
            "Invalid packed feature shape while replaying deferred filter: "
            f"split={split_path} dataset_index={dataset_index} shape=ragged"
        )

    total_values = int(normalized_offsets[-1])
    values = np.asarray(
        feature_column.values.slice(base_offset, total_values).to_numpy(zero_copy_only=False),
        dtype=np.float32,
    )
    if expected_width == 0:
        return np.empty((n_rows, 0), dtype=np.float32)
    return values.reshape(n_rows, expected_width)


def _iter_packed_split_datasets(split_path: Path) -> Iterator[_PackedSplitDataset]:
    """Yield packed split rows one dataset at a time, validating shard ordering."""

    _require_pyarrow()

    parquet_file = pq.ParquetFile(split_path)
    required_columns = {"dataset_index", "row_index", "x", "y"}
    missing_columns = sorted(required_columns.difference(parquet_file.schema_arrow.names))
    if missing_columns:
        raise ValueError(
            f"Packed split is missing required columns in {split_path}: {missing_columns}"
        )

    current_dataset_index: int | None = None
    expected_row_index = 0
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []

    for batch in parquet_file.iter_batches(columns=["dataset_index", "row_index", "x", "y"]):
        dataset_indices = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int64)
        if dataset_indices.size == 0:
            continue
        row_indices = np.asarray(batch.column(1).to_numpy(zero_copy_only=False), dtype=np.int64)
        feature_column = batch.column(2)
        batch_y_rows = np.asarray(batch.column(3).to_numpy(zero_copy_only=False))

        group_starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(np.diff(dataset_indices) != 0) + 1)
        )
        group_ends = np.concatenate(
            (group_starts[1:], np.array([dataset_indices.size], dtype=np.int64))
        )

        for start_raw, end_raw in zip(group_starts, group_ends, strict=True):
            start = int(start_raw)
            end = int(end_raw)
            dataset_index = int(dataset_indices[start])
            group_row_indices = row_indices[start:end]

            if current_dataset_index is None:
                current_dataset_index = dataset_index
                expected_row_index = 0
            elif dataset_index < current_dataset_index:
                raise ValueError(
                    "Packed split rows must be grouped by monotonically increasing dataset_index: "
                    f"split={split_path} saw dataset_index={dataset_index} after "
                    f"{current_dataset_index}"
                )
            elif dataset_index != current_dataset_index:
                yield _build_packed_split_dataset(
                    dataset_index=current_dataset_index,
                    x_chunks=x_chunks,
                    y_chunks=y_chunks,
                    split_path=split_path,
                )
                current_dataset_index = dataset_index
                expected_row_index = 0
                x_chunks = []
                y_chunks = []

            expected_last_row_index = expected_row_index + int(group_row_indices.size) - 1
            group_is_contiguous = bool(
                group_row_indices.size == 0
                or (
                    int(group_row_indices[0]) == expected_row_index
                    and int(group_row_indices[-1]) == expected_last_row_index
                    and (
                        group_row_indices.size == 1 or bool(np.all(np.diff(group_row_indices) == 1))
                    )
                )
            )
            if not group_is_contiguous:
                actual_row_index = int(group_row_indices[0]) if group_row_indices.size > 0 else -1
                raise ValueError(
                    "Packed split rows must have contiguous row_index values starting at 0: "
                    f"split={split_path} dataset_index={dataset_index} "
                    f"expected_row_index={expected_row_index} got={actual_row_index}"
                )

            group_x_rows = _packed_feature_column_to_numpy_matrix(
                feature_column=feature_column.slice(start, end - start),
                split_path=split_path,
                dataset_index=dataset_index,
            )

            x_chunks.append(group_x_rows)
            y_chunks.append(batch_y_rows[start:end])
            expected_row_index += int(group_row_indices.size)

    if current_dataset_index is not None:
        yield _build_packed_split_dataset(
            dataset_index=current_dataset_index,
            x_chunks=x_chunks,
            y_chunks=y_chunks,
            split_path=split_path,
        )


def _filter_dataset(
    *,
    lineage_payload: Mapping[str, Any] | None,
    lineage_base_dir: Path | None,
    filter_cfg: FilterConfig,
) -> tuple[bool, dict[str, Any], float]:
    """Replay structural filter on persisted lineage metadata for one dataset."""

    start = time.perf_counter()
    accepted, details = apply_structural_filter(
        lineage_payload=lineage_payload,
        lineage_base_dir=lineage_base_dir,
        min_target_indegree=filter_cfg.min_target_indegree,
        min_target_relevant_feature_count=filter_cfg.min_target_relevant_feature_count,
        min_target_relevant_feature_fraction=filter_cfg.min_target_relevant_feature_fraction,
    )
    elapsed_seconds = max(0.0, time.perf_counter() - start)
    return bool(accepted), dict(details), float(elapsed_seconds)


def run_deferred_filter(
    *,
    in_dir: str | Path,
    out_dir: str | Path,
    curated_out_dir: str | Path | None = None,
    path_overrides: tuple[tuple[str, Any], ...] = (),
) -> DeferredFilterRunResult:
    """Replay structural filtering over persisted shard outputs."""

    _require_pyarrow()

    input_path = Path(in_dir)
    output_path = Path(out_dir)
    _ensure_filter_output_dir_safe(output_path)

    curated_path: Path | None = None
    if curated_out_dir is not None:
        curated_path = Path(curated_out_dir)
        _ensure_curated_output_dir_safe(curated_path)

    shard_dirs = _discover_shard_dirs(input_path)
    internal_root = resolve_internal_root(input_path)

    rejected_reason_counts: Counter[str] = Counter()

    accepted_total = 0
    rejected_total = 0
    total_elapsed_seconds = 0.0
    curated_accepted_total = 0

    manifest_path = output_path / MANIFEST_FILENAME
    summary_path = output_path / SUMMARY_FILENAME
    staging_token = str(time.time_ns())
    staged_manifest_path = _staged_output_path(
        parent_dir=output_path,
        final_name=MANIFEST_FILENAME,
        staging_token=staging_token,
    )
    staged_summary_path = _staged_output_path(
        parent_dir=output_path,
        final_name=SUMMARY_FILENAME,
        staging_token=staging_token,
    )
    staged_curated_dirs: list[Path] = []
    promotable_curated_dirs: list[tuple[Path, Path]] = []
    promoted_final_paths: list[Path] = []

    try:
        with staged_manifest_path.open("w", encoding="utf-8") as manifest_file:
            for shard_dir in shard_dirs:
                catalog_path = _catalog_path_for_shard(shard_dir)
                if catalog_path is None:
                    raise FileNotFoundError(
                        f"Shard directory is missing a public catalog file: {shard_dir}"
                    )
                replay_path = _replay_catalog_path_for_shard(
                    shard_dir,
                    internal_root=internal_root,
                )
                train_path = shard_dir / "train.parquet"
                test_path = shard_dir / "test.parquet"
                if not train_path.exists() or not test_path.exists():
                    raise FileNotFoundError(
                        "Shard directory is missing required artifacts "
                        f"({DATASET_CATALOG_FILENAME}/train.parquet/test.parquet): "
                        f"{shard_dir}"
                    )

                train_iter = _iter_packed_split_datasets(train_path)
                test_iter = _iter_packed_split_datasets(test_path)
                last_dataset_index = -1
                curated_writer: _CuratedShardWriter | None = None
                curated_written = 0

                try:
                    replay_records = _iter_metadata_records(replay_path)
                    for catalog_record in _iter_metadata_records(catalog_path):
                        dataset_index_raw = catalog_record.get("dataset_index")
                        if dataset_index_raw is None or isinstance(dataset_index_raw, bool):
                            raise ValueError(
                                "Invalid dataset_index in metadata record: "
                                f"shard={shard_dir} dataset_index={dataset_index_raw!r}"
                            )
                        try:
                            dataset_index = int(dataset_index_raw)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                "Invalid dataset_index in metadata record: "
                                f"shard={shard_dir} dataset_index={dataset_index_raw!r}"
                            ) from exc

                        if dataset_index <= last_dataset_index:
                            raise ValueError(
                                "Metadata records must use strictly increasing dataset_index values: "
                                f"shard={shard_dir} dataset_index={dataset_index}"
                            )
                        last_dataset_index = dataset_index

                        try:
                            replay_record = next(replay_records)
                        except StopIteration as exc:
                            raise ValueError(
                                "Replay sidecar has fewer records than the public catalog: "
                                f"shard={shard_dir} dataset_index={dataset_index}"
                            ) from exc
                        replay_dataset_index = replay_record.get("dataset_index")
                        if (
                            replay_dataset_index is None
                            or int(replay_dataset_index) != dataset_index
                        ):
                            raise ValueError(
                                "Replay sidecar dataset_index does not align with public catalog: "
                                f"shard={shard_dir} expected={dataset_index} got={replay_dataset_index!r}"
                            )

                        metadata_payload = replay_record.get("metadata")
                        if not isinstance(metadata_payload, Mapping):
                            raise ValueError(
                                "Invalid metadata payload for deferred filtering: "
                                f"shard={shard_dir} dataset_index={dataset_index}"
                            )

                        train_split = _consume_expected_split(
                            train_iter,
                            expected_dataset_index=dataset_index,
                            split_path=train_path,
                        )
                        test_split = _consume_expected_split(
                            test_iter,
                            expected_dataset_index=dataset_index,
                            split_path=test_path,
                        )

                        filter_cfg = _resolve_filter_config(
                            metadata_payload=metadata_payload,
                            path_overrides=path_overrides,
                        )
                        seed = _resolve_filter_seed(metadata_payload, dataset_index=dataset_index)

                        accepted, filter_details, elapsed_seconds = _filter_dataset(
                            lineage_payload=(
                                metadata_payload.get("lineage")
                                if isinstance(metadata_payload.get("lineage"), Mapping)
                                else None
                            ),
                            lineage_base_dir=replay_path.parent,
                            filter_cfg=filter_cfg,
                        )
                        total_elapsed_seconds += elapsed_seconds

                        filter_metadata = _build_filter_metadata(
                            existing_filter=metadata_payload.get("filter"),
                            accepted=accepted,
                            filter_details=filter_details,
                        )

                        reason_value = filter_details.get("reason")
                        reason = (
                            str(reason_value)
                            if isinstance(reason_value, str) and reason_value
                            else None
                        )
                        if not accepted:
                            rejected_total += 1
                            rejected_reason_counts[reason or "rejected"] += 1
                        else:
                            accepted_total += 1
                            if curated_path is not None:
                                if curated_writer is None:
                                    curated_writer = _create_curated_shard_writer(
                                        curated_out_dir=curated_path,
                                        shard_name=shard_dir.name,
                                        staging_token=staging_token,
                                    )
                                    staged_curated_dirs.append(curated_writer.shard_dir)
                                _write_curated_dataset(
                                    state=curated_writer,
                                    dataset_index=dataset_index,
                                    train_split=train_split,
                                    test_split=test_split,
                                    record=dict(catalog_record),
                                )
                                curated_written += 1

                        _write_ndjson_record(
                            manifest_file,
                            {
                                "dataset_index": dataset_index,
                                "seed": seed,
                                "source_shard": shard_dir.name,
                                "accepted": bool(accepted),
                                "status": "accepted" if accepted else "rejected",
                                "reason": reason,
                                "elapsed_seconds": float(elapsed_seconds),
                                "filter": filter_metadata,
                            },
                        )

                    _ensure_split_iter_exhausted(train_iter, split_path=train_path)
                    _ensure_split_iter_exhausted(test_iter, split_path=test_path)
                    try:
                        next(replay_records)
                    except StopIteration:
                        pass
                    else:
                        raise ValueError(
                            "Replay sidecar contains extra records beyond public catalog coverage: "
                            f"shard={shard_dir}"
                        )
                finally:
                    _close_curated_shard_writer(curated_writer)

                if curated_writer is not None and curated_written > 0:
                    promotable_curated_dirs.append(
                        (curated_writer.shard_dir, curated_writer.final_shard_dir)
                    )
                    curated_accepted_total += curated_written

        total_datasets = accepted_total + rejected_total
        datasets_per_minute = (
            (float(total_datasets) / float(total_elapsed_seconds)) * 60.0
            if total_elapsed_seconds > 0.0
            else 0.0
        )

        summary_payload: dict[str, Any] = {
            "input_dir": str(input_path.resolve()),
            "out_dir": str(output_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "filter_mode": STRUCTURAL_FILTER_MODE,
            "total_datasets": int(total_datasets),
            "accepted_datasets": int(accepted_total),
            "rejected_datasets": int(rejected_total),
            "acceptance_rate": (
                float(accepted_total) / float(total_datasets) if total_datasets > 0 else None
            ),
            "rejected_reason_counts": {
                key: int(rejected_reason_counts[key]) for key in sorted(rejected_reason_counts)
            },
            "elapsed_seconds": float(total_elapsed_seconds),
            "datasets_per_minute": float(datasets_per_minute),
            "curated_out_dir": str(curated_path.resolve()) if curated_path is not None else None,
            "curated_accepted_datasets": int(curated_accepted_total),
        }
        if path_overrides:
            summary_payload["path_overrides"] = [
                {"path": path, "value": value} for path, value in path_overrides
            ]
        staged_summary_path.write_text(
            json.dumps(_sanitize_json(summary_payload), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

        for staged_dir, final_dir in promotable_curated_dirs:
            _promote_staged_path(staged_path=staged_dir, final_path=final_dir)
            promoted_final_paths.append(final_dir)
        _promote_staged_path(staged_path=staged_manifest_path, final_path=manifest_path)
        promoted_final_paths.append(manifest_path)
        _promote_staged_path(staged_path=staged_summary_path, final_path=summary_path)
        promoted_final_paths.append(summary_path)
    except Exception:
        for path in reversed(promoted_final_paths):
            _cleanup_path(path)
        raise
    finally:
        _cleanup_path(staged_manifest_path)
        _cleanup_path(staged_summary_path)
        for staged_dir in staged_curated_dirs:
            _cleanup_path(staged_dir)

    return DeferredFilterRunResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        total_datasets=int(total_datasets),
        accepted_datasets=int(accepted_total),
        rejected_datasets=int(rejected_total),
        elapsed_seconds=float(total_elapsed_seconds),
        datasets_per_minute=float(datasets_per_minute),
        curated_out_dir=curated_path,
        curated_accepted_datasets=int(curated_accepted_total),
    )
