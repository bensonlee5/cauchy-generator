"""Parquet writer utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TextIO, cast

import numpy as np

from dagzoo.io.lineage_artifact import (
    pack_upper_triangle_adjacency,
    sha256_hex,
    upper_triangle_bit_length,
)
from dagzoo.io.lineage_schema import (
    LINEAGE_ADJACENCY_ENCODING,
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION_COMPACT,
    LINEAGE_SCHEMA_VERSION_DENSE,
    validate_lineage_payload,
)
from dagzoo.io.shard_contract import (
    DATASET_CATALOG_FILENAME,
    REPLAY_CATALOG_FILENAME,
    build_dataset_catalog_record,
    internal_shard_dir,
    resolve_internal_root,
    write_dataset_catalog_records,
)
from dagzoo.math import sanitize_json as _sanitize_json
from dagzoo.math import to_numpy as _to_numpy
from dagzoo.types import DatasetBundle

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pa = None
    pq = None


def _require_pyarrow() -> None:
    """Require pyarrow before accessing parquet writer helpers."""

    if pa is None or pq is None:
        raise RuntimeError(
            "pyarrow is required for Parquet output. Install project dependencies with uv."
        )


def _shard_id_for_dataset(*, dataset_index: int, shard_size: int) -> int:
    """Return shard id for a dataset index under configured shard size."""

    return dataset_index // max(1, shard_size)


def _ensure_output_dir_safe(out_dir: Path) -> None:
    """Fail fast when output directory already contains prior shard outputs."""

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return
    stale = next(out_dir.glob("shard_*"), None)
    if stale is not None:
        raise RuntimeError(
            f"Output directory already contains shard data: {out_dir}. "
            "Choose a new --out directory or remove existing shard_* folders first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def _ensure_internal_output_dir_safe(internal_root: Path) -> None:
    """Fail fast when internal sidecar directory already contains prior shard outputs."""

    if not internal_root.exists():
        internal_root.mkdir(parents=True, exist_ok=True)
        return
    stale = next(internal_root.glob("shard_*"), None)
    if stale is not None:
        raise RuntimeError(
            f"Internal output directory already contains shard data: {internal_root}. "
            "Choose a new output root or remove existing internal/shard_* folders first."
        )
    internal_root.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class _PackedShardState:
    """Mutable packed-output state for one shard."""

    shard_dir: Path
    train_path: Path
    test_path: Path
    metadata_path: Path
    train_writer: Any | None = None
    test_writer: Any | None = None
    catalog_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class _ReplayShardState:
    """Mutable private sidecar state for one shard."""

    shard_dir: Path
    replay_path: Path
    replay_file: TextIO | None = None


@dataclass(slots=True)
class _ShardLineageState:
    """Mutable lineage artifact state for one output shard."""

    blob_path: Path
    index_path: Path
    blob_file: BinaryIO | None = None
    byte_offset: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)


def _packed_state_for_shard(
    *,
    shard_id: int,
    out_dir: Path,
    shard_states: dict[int, _PackedShardState],
) -> _PackedShardState:
    """Get or initialize packed output state for one shard."""

    state = shard_states.get(shard_id)
    if state is not None:
        return state

    shard_dir = out_dir / f"shard_{shard_id:05d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    state = _PackedShardState(
        shard_dir=shard_dir,
        train_path=shard_dir / "train.parquet",
        test_path=shard_dir / "test.parquet",
        metadata_path=shard_dir / DATASET_CATALOG_FILENAME,
    )
    shard_states[shard_id] = state
    return state


def _replay_state_for_shard(
    *,
    shard_id: int,
    internal_root: Path,
    replay_states: dict[int, _ReplayShardState],
) -> _ReplayShardState:
    """Get or initialize private replay sidecar state for one shard."""

    state = replay_states.get(shard_id)
    if state is not None:
        return state

    shard_dir = internal_shard_dir(
        internal_root=internal_root,
        shard_name=f"shard_{shard_id:05d}",
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    state = _ReplayShardState(
        shard_dir=shard_dir,
        replay_path=shard_dir / REPLAY_CATALOG_FILENAME,
    )
    replay_states[shard_id] = state
    return state


def _lineage_state_for_shard(
    *,
    shard_id: int,
    shard_dir: Path,
    lineage_states: dict[int, _ShardLineageState],
) -> _ShardLineageState:
    """Get or initialize lineage artifact state for one shard."""

    state = lineage_states.get(shard_id)
    if state is not None:
        return state

    lineage_dir = shard_dir / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    state = _ShardLineageState(
        blob_path=lineage_dir / "adjacency.bitpack.bin",
        index_path=lineage_dir / "adjacency.index.json",
    )
    lineage_states[shard_id] = state
    return state


def _build_compact_lineage_payload(
    lineage: Mapping[str, Any],
    *,
    dataset_index: int,
    bit_offset: int,
    bit_length: int,
    edge_count: int,
    sha256: str,
    blob_path: str,
    index_path: str,
    n_nodes: int,
) -> dict[str, Any]:
    """Build compact lineage payload for persisted metadata."""

    assignments = cast(Mapping[str, Any], lineage["assignments"])
    feature_to_node = list(cast(list[int], assignments["feature_to_node"]))
    payload: dict[str, Any] = {
        "schema_name": LINEAGE_SCHEMA_NAME,
        "schema_version": LINEAGE_SCHEMA_VERSION_COMPACT,
        "graph": {
            "n_nodes": int(n_nodes),
            "edge_count": int(edge_count),
            "adjacency_ref": {
                "encoding": LINEAGE_ADJACENCY_ENCODING,
                "blob_path": blob_path,
                "index_path": index_path,
                "dataset_index": int(dataset_index),
                "bit_offset": int(bit_offset),
                "bit_length": int(bit_length),
                "sha256": sha256,
            },
        },
        "assignments": {
            "feature_to_node": feature_to_node,
        },
    }
    assignments_payload = cast(dict[str, Any], payload["assignments"])
    if "target_to_node" in assignments:
        assignments_payload["target_to_node"] = cast(int, assignments["target_to_node"])
    for key in (
        "target_relevant_features",
        "target_relevant_feature_count",
        "target_relevant_feature_fraction",
    ):
        if key in assignments:
            assignments_payload[key] = assignments[key]
    return payload


def _ensure_shard_blob_open(state: _ShardLineageState) -> BinaryIO:
    """Return an append-ready shard blob handle, opening it once per shard."""

    blob_file = state.blob_file
    if blob_file is not None and not blob_file.closed:
        return blob_file

    opened = state.blob_path.open("ab")
    state.blob_file = opened
    return opened


def _ensure_replay_file_open(state: _ReplayShardState) -> TextIO:
    """Return an append-ready replay sidecar handle, opening it once per shard."""

    replay_file = state.replay_file
    if replay_file is not None and not replay_file.closed:
        return replay_file
    opened = state.replay_path.open("a", encoding="utf-8")
    state.replay_file = opened
    return opened


def _close_shard_blob(state: _ShardLineageState) -> None:
    """Close a single shard blob handle if open."""

    blob_file = state.blob_file
    if blob_file is None:
        return
    try:
        blob_file.close()
    finally:
        state.blob_file = None


def _close_packed_shard_handles(state: _PackedShardState) -> None:
    """Close open parquet and catalog handles for one shard."""

    train_writer = state.train_writer
    if train_writer is not None:
        train_writer.close()
        state.train_writer = None

    test_writer = state.test_writer
    if test_writer is not None:
        test_writer.close()
        state.test_writer = None

    if state.catalog_records:
        write_dataset_catalog_records(state.metadata_path, state.catalog_records)
        state.catalog_records = []


def _close_replay_shard_handles(state: _ReplayShardState) -> None:
    """Close open replay sidecar handles for one shard."""

    replay_file = state.replay_file
    if replay_file is not None:
        replay_file.close()
        state.replay_file = None


def _close_shard_lineage_files(lineage_states: Mapping[int, _ShardLineageState]) -> None:
    """Close open shard blob handles."""

    for state in lineage_states.values():
        _close_shard_blob(state)


def _close_packed_shard_files(shard_states: Mapping[int, _PackedShardState]) -> None:
    """Close open packed shard handles."""

    for state in shard_states.values():
        _close_packed_shard_handles(state)


def _close_replay_shard_files(replay_states: Mapping[int, _ReplayShardState]) -> None:
    """Close open replay sidecar handles."""

    for state in replay_states.values():
        _close_replay_shard_handles(state)


def _persist_lineage_artifact_for_dataset(
    metadata: dict[str, Any],
    *,
    dataset_index: int,
    shard_id: int,
    shard_dir: Path,
    lineage_states: dict[int, _ShardLineageState],
) -> dict[str, Any]:
    """Convert dense lineage payload to compact shard artifact pointers for one dataset."""

    lineage_raw = metadata.get("lineage")
    if not isinstance(lineage_raw, Mapping):
        return metadata
    lineage = cast(Mapping[str, Any], lineage_raw)
    validate_lineage_payload(lineage)

    schema_version = cast(str, lineage["schema_version"])
    if schema_version != LINEAGE_SCHEMA_VERSION_DENSE:
        return metadata

    graph = cast(Mapping[str, Any], lineage["graph"])
    adjacency_raw = graph["adjacency"]
    n_nodes, edge_count, packed = pack_upper_triangle_adjacency(adjacency_raw)
    bit_length = upper_triangle_bit_length(n_nodes)

    state = _lineage_state_for_shard(
        shard_id=shard_id,
        shard_dir=shard_dir,
        lineage_states=lineage_states,
    )
    bit_offset = state.byte_offset * 8
    state.byte_offset += len(packed)

    blob_file = _ensure_shard_blob_open(state)
    blob_file.write(packed)
    checksum = sha256_hex(packed)

    blob_path = str(Path("lineage") / state.blob_path.name)
    index_path = str(Path("lineage") / state.index_path.name)
    compact_lineage = _build_compact_lineage_payload(
        lineage,
        dataset_index=dataset_index,
        bit_offset=bit_offset,
        bit_length=bit_length,
        edge_count=edge_count,
        sha256=checksum,
        blob_path=blob_path,
        index_path=index_path,
        n_nodes=n_nodes,
    )
    validate_lineage_payload(compact_lineage)
    metadata["lineage"] = compact_lineage

    state.records.append(
        {
            "dataset_index": int(dataset_index),
            "n_nodes": int(n_nodes),
            "edge_count": int(edge_count),
            "bit_offset": int(bit_offset),
            "bit_length": int(bit_length),
            "sha256": checksum,
        }
    )
    return metadata


def _write_shard_lineage_indexes(lineage_states: Mapping[int, _ShardLineageState]) -> None:
    """Write shard-level lineage index files after all datasets are persisted."""

    for state in lineage_states.values():
        if not state.records:
            continue
        payload = {
            "schema_name": LINEAGE_SCHEMA_NAME,
            "schema_version": LINEAGE_SCHEMA_VERSION_COMPACT,
            "encoding": LINEAGE_ADJACENCY_ENCODING,
            "records": state.records,
        }
        with state.index_path.open("w", encoding="utf-8") as f:
            json.dump(
                _sanitize_json(payload),
                f,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )


def _finalize_lineage_states(
    lineage_states: Mapping[int, _ShardLineageState],
    *,
    strict_index_write: bool,
) -> None:
    """Close handles and flush lineage indexes.

    When `strict_index_write` is False, index write failures are swallowed so the
    original generation exception remains the primary error.
    """

    _close_shard_lineage_files(lineage_states)
    if strict_index_write:
        _write_shard_lineage_indexes(lineage_states)
        return
    try:
        _write_shard_lineage_indexes(lineage_states)
    except Exception:
        return


def _build_split_table(*, dataset_index: int, x: np.ndarray, y: np.ndarray) -> Any:
    """Build packed row-wise table for one split of one dataset."""

    _require_pyarrow()

    if x.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"Expected 1D targets, got shape={y.shape}.")

    n_rows, n_features = x.shape
    if y.shape[0] != n_rows:
        raise ValueError(
            f"Mismatched split sizes: features rows={n_rows} targets rows={y.shape[0]}."
        )

    x_item_type = pa.float64() if x.dtype == np.float64 else pa.float32()
    x_contig = np.ascontiguousarray(x)
    y_contig = np.ascontiguousarray(y)

    x_values = pa.array(x_contig.reshape(-1), type=x_item_type)
    if n_features > 0:
        offsets_np = np.arange(0, (n_rows + 1) * n_features, n_features, dtype=np.int64)
    else:
        offsets_np = np.zeros(n_rows + 1, dtype=np.int64)
    x_column = pa.ListArray.from_arrays(
        pa.array(offsets_np),
        x_values,
        type=pa.list_(x_item_type),
    )
    data = {
        "dataset_index": pa.array(np.full(n_rows, dataset_index, dtype=np.int64)),
        "row_index": pa.array(np.arange(n_rows, dtype=np.int64)),
        "x": x_column,
        "y": pa.array(y_contig),
    }
    return pa.table(data)


def _ensure_split_writer(
    *,
    state: _PackedShardState,
    split: str,
    schema: Any,
    compression: str,
) -> Any:
    """Get or initialize a split writer for one shard."""

    _require_pyarrow()

    if split == "train":
        writer = state.train_writer
        path = state.train_path
    else:
        writer = state.test_writer
        path = state.test_path

    if writer is None:
        writer = pq.ParquetWriter(path, schema=schema, compression=compression)
        if split == "train":
            state.train_writer = writer
        else:
            state.test_writer = writer
    return writer


def _write_packed_split(
    *,
    state: _PackedShardState,
    split: str,
    dataset_index: int,
    x: np.ndarray,
    y: np.ndarray,
    compression: str,
) -> None:
    """Append one dataset split into shard-level packed parquet."""

    table = _build_split_table(dataset_index=dataset_index, x=x, y=y)
    writer = _ensure_split_writer(
        state=state,
        split=split,
        schema=table.schema,
        compression=compression,
    )
    if not table.schema.equals(writer.schema, check_metadata=False):
        split_path = state.train_path if split == "train" else state.test_path
        raise ValueError(
            "Incompatible packed "
            f"{split} schema in shard output '{split_path}': "
            f"expected {writer.schema}, got {table.schema} "
            f"(dataset_index={dataset_index}). "
            "Mixed feature/target dtypes within one shard are not supported."
        )
    writer.write_table(table)


def _write_metadata_record(
    *,
    state: _PackedShardState,
    dataset_index: int,
    feature_types: list[str],
    metadata: Mapping[str, Any],
    n_train: int,
    n_test: int,
    n_features: int,
) -> None:
    """Append one public dataset-catalog record to the shard catalog stream."""

    payload = build_dataset_catalog_record(
        dataset_index=dataset_index,
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        feature_types=feature_types,
        metadata=metadata,
    )
    state.catalog_records.append(cast(dict[str, Any], _sanitize_json(payload)))


def _write_replay_record(
    *,
    state: _ReplayShardState,
    dataset_index: int,
    feature_types: list[str],
    metadata: Mapping[str, Any],
    n_train: int,
    n_test: int,
    n_features: int,
) -> None:
    """Append one private replay-sidecar record with full metadata."""

    replay_file = _ensure_replay_file_open(state)
    payload = {
        "dataset_index": int(dataset_index),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_features": int(n_features),
        "feature_types": list(feature_types),
        "metadata": dict(metadata),
    }
    replay_file.write(
        json.dumps(
            _sanitize_json(payload),
            sort_keys=True,
            allow_nan=False,
        )
    )
    replay_file.write("\n")


def _write_bundle_to_shard(
    bundle: DatasetBundle,
    *,
    out_dir: Path,
    internal_root: Path,
    dataset_index: int,
    shard_size: int,
    compression: str,
    shard_states: dict[int, _PackedShardState],
    replay_states: dict[int, _ReplayShardState],
    lineage_states: dict[int, _ShardLineageState],
) -> None:
    """Write one dataset bundle into packed shard artifacts."""

    shard_id = _shard_id_for_dataset(dataset_index=dataset_index, shard_size=shard_size)
    shard_state = _packed_state_for_shard(
        shard_id=shard_id,
        out_dir=out_dir,
        shard_states=shard_states,
    )
    replay_state = _replay_state_for_shard(
        shard_id=shard_id,
        internal_root=internal_root,
        replay_states=replay_states,
    )

    x_train = _to_numpy(bundle.X_train)
    y_train = _to_numpy(bundle.y_train)
    x_test = _to_numpy(bundle.X_test)
    y_test = _to_numpy(bundle.y_test)

    _write_packed_split(
        state=shard_state,
        split="train",
        dataset_index=dataset_index,
        x=x_train,
        y=y_train,
        compression=compression,
    )
    _write_packed_split(
        state=shard_state,
        split="test",
        dataset_index=dataset_index,
        x=x_test,
        y=y_test,
        compression=compression,
    )

    metadata = deepcopy(bundle.metadata)
    metadata = _persist_lineage_artifact_for_dataset(
        metadata,
        dataset_index=dataset_index,
        shard_id=shard_id,
        shard_dir=replay_state.shard_dir,
        lineage_states=lineage_states,
    )
    _write_metadata_record(
        state=shard_state,
        dataset_index=dataset_index,
        feature_types=list(bundle.feature_types),
        metadata=metadata,
        n_train=int(x_train.shape[0]),
        n_test=int(x_test.shape[0]),
        n_features=int(x_train.shape[1]),
    )
    _write_replay_record(
        state=replay_state,
        dataset_index=dataset_index,
        feature_types=list(bundle.feature_types),
        metadata=metadata,
        n_train=int(x_train.shape[0]),
        n_test=int(x_test.shape[0]),
        n_features=int(x_train.shape[1]),
    )


def write_packed_parquet_shards_stream(
    bundles: Iterable[DatasetBundle],
    out_dir: str | Path,
    *,
    shard_size: int = 128,
    compression: str = "zstd",
    internal_root: str | Path | None = None,
) -> int:
    """Write bundles into packed shard outputs and return dataset count."""

    out = Path(out_dir)
    _ensure_output_dir_safe(out)
    resolved_internal_root = resolve_internal_root(out, explicit_internal_root=internal_root)
    _ensure_internal_output_dir_safe(resolved_internal_root)

    shard_states: dict[int, _PackedShardState] = {}
    replay_states: dict[int, _ReplayShardState] = {}
    lineage_states: dict[int, _ShardLineageState] = {}
    active_shard_id: int | None = None
    written = 0
    success = False

    try:
        for idx, bundle in enumerate(bundles):
            shard_id = _shard_id_for_dataset(dataset_index=idx, shard_size=shard_size)
            if active_shard_id is not None and shard_id != active_shard_id:
                previous_shard = shard_states.get(active_shard_id)
                if previous_shard is not None:
                    _close_packed_shard_handles(previous_shard)
                previous_replay = replay_states.get(active_shard_id)
                if previous_replay is not None:
                    _close_replay_shard_handles(previous_replay)
                previous_lineage = lineage_states.get(active_shard_id)
                if previous_lineage is not None:
                    _close_shard_blob(previous_lineage)
            active_shard_id = shard_id
            _write_bundle_to_shard(
                bundle,
                out_dir=out,
                internal_root=resolved_internal_root,
                dataset_index=idx,
                shard_size=shard_size,
                compression=compression,
                shard_states=shard_states,
                replay_states=replay_states,
                lineage_states=lineage_states,
            )
            written = idx + 1
        success = True
        return written
    finally:
        _close_packed_shard_files(shard_states)
        _close_replay_shard_files(replay_states)
        _finalize_lineage_states(lineage_states, strict_index_write=success)
