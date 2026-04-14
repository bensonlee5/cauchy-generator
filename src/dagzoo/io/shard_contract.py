"""Public shard contract helpers and internal sidecar layout."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency in some read-only contexts
    pa = None
    pq = None


DATASET_CATALOG_FILENAME = "dataset_catalog.parquet"
INTERNAL_DIRNAME = "internal"
REPLAY_CATALOG_FILENAME = "replay_catalog.parquet"
RUN_CONTEXT_FILENAME = "run_context.json"
_BLAKE2S_HEX_LENGTH = 32
_JSON_RECORD_PARQUET_BATCH_SIZE = 1024


def _require_pyarrow() -> None:
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required for parquet-backed shard catalogs.")


def _is_direct_shard_input(path: Path) -> bool:
    """Return whether a public path names one shard directory directly."""

    return path.name.startswith("shard_")


def _internal_root_candidates(public_root: Path) -> list[Path]:
    """Return candidate internal roots for a public run root or shard path."""

    if _is_direct_shard_input(public_root):
        public_run_root = public_root.parent
        candidates: list[Path] = []
        if public_run_root.name == "generated":
            candidates.append(public_run_root.parent / INTERNAL_DIRNAME)
        candidates.append(public_run_root / INTERNAL_DIRNAME)
        return candidates
    return [
        public_root / INTERNAL_DIRNAME,
        public_root.parent / INTERNAL_DIRNAME,
    ]


def infer_task_from_metadata(metadata: Mapping[str, Any]) -> str:
    """Infer task label from metadata payloads."""

    config = metadata.get("config")
    if isinstance(config, Mapping):
        dataset = config.get("dataset")
        if isinstance(dataset, Mapping):
            task = dataset.get("task")
            if isinstance(task, str) and task.strip():
                normalized = task.strip().lower()
                if normalized in {"classification", "regression"}:
                    return normalized
    return "classification" if metadata.get("n_classes") is not None else "regression"


def _normalize_intervention_summary(value: object, *, path: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    mode = value.get("mode")
    signature = value.get("signature")
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(f"{path}.mode must be a non-empty string.")
    if not isinstance(signature, str) or not signature.strip():
        raise ValueError(f"{path}.signature must be a non-empty string.")
    normalized_signature = signature.strip()
    if len(normalized_signature) != _BLAKE2S_HEX_LENGTH or any(
        ch not in "0123456789abcdef" for ch in normalized_signature
    ):
        raise ValueError(
            f"{path}.signature must be a {_BLAKE2S_HEX_LENGTH}-character lowercase hexadecimal string."
        )
    return {
        "mode": mode.strip(),
        "signature": normalized_signature,
    }


def build_dataset_catalog_record(
    *,
    dataset_index: int,
    n_train: int,
    n_test: int,
    n_features: int,
    feature_types: list[str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one minimal public dataset-catalog record."""

    record: dict[str, Any] = {
        "dataset_index": int(dataset_index),
        "dataset_id": metadata.get("dataset_id"),
        "task": infer_task_from_metadata(metadata),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_features": int(n_features),
        "feature_types": list(feature_types),
        "n_classes": metadata.get("n_classes"),
    }
    split_groups = metadata.get("split_groups")
    if isinstance(split_groups, Mapping):
        group_ids = {}
        request_run = split_groups.get("request_run")
        if request_run is not None:
            group_ids["request_run"] = request_run
        cohort = split_groups.get("cohort")
        if cohort is not None:
            group_ids["cohort"] = cohort
        layout_plan = split_groups.get("layout_plan")
        if layout_plan is not None:
            group_ids["layout_plan"] = layout_plan
        if group_ids:
            record["group_ids"] = group_ids
    prior = metadata.get("prior")
    if isinstance(prior, Mapping):
        target_derivation = prior.get("target_derivation")
        if isinstance(target_derivation, str) and target_derivation.strip():
            record["target_derivation"] = target_derivation
    lineage = metadata.get("lineage")
    if isinstance(lineage, Mapping):
        assignments = lineage.get("assignments")
        if isinstance(assignments, Mapping):
            target_relevant_feature_count = assignments.get("target_relevant_feature_count")
            target_relevant_feature_fraction = assignments.get("target_relevant_feature_fraction")
            if (
                not isinstance(target_relevant_feature_count, bool)
                and isinstance(target_relevant_feature_count, int)
                and not isinstance(target_relevant_feature_fraction, bool)
                and isinstance(target_relevant_feature_fraction, (int, float))
            ):
                record["target_relevance"] = {
                    "feature_count": int(target_relevant_feature_count),
                    "feature_fraction": float(target_relevant_feature_fraction),
                }
    intervention = metadata.get("intervention")
    if intervention is not None:
        record["intervention"] = _normalize_intervention_summary(
            intervention,
            path="metadata.intervention",
        )
    return record


def _canonical_record_json(record: Mapping[str, Any]) -> str:
    return json.dumps(dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _build_json_record_parquet_row(record: Mapping[str, Any]) -> dict[str, Any]:
    dataset_index = record.get("dataset_index")
    if dataset_index is None or isinstance(dataset_index, bool):
        raise ValueError("json-record parquet rows require an integer dataset_index.")
    record_json = _canonical_record_json(record)
    return {
        "dataset_index": int(dataset_index),
        "record_json": record_json,
        "record_sha256": sha256(record_json.encode("utf-8")).hexdigest(),
    }


def json_record_parquet_schema():
    _require_pyarrow()
    return pa.schema(
        [
            pa.field("dataset_index", pa.int64()),
            pa.field("record_json", pa.large_string()),
            pa.field("record_sha256", pa.string()),
        ]
    )


def build_dataset_catalog_parquet_row(record: Mapping[str, Any]) -> dict[str, Any]:
    record_json = _canonical_record_json(record)
    metadata = record.get("metadata")
    metadata_mapping = metadata if isinstance(metadata, Mapping) else None
    filter_payload = (
        metadata_mapping.get("filter") if isinstance(metadata_mapping, Mapping) else None
    )
    split_groups = record.get("group_ids")
    request_run = split_groups.get("request_run") if isinstance(split_groups, Mapping) else None
    teacher_conditionals = record.get("teacher_conditionals")
    return {
        "dataset_index": int(record["dataset_index"]),
        "record_json": record_json,
        "record_sha256": sha256(record_json.encode("utf-8")).hexdigest(),
        "resolved_dataset_id": record.get("dataset_id"),
        "resolved_request_run": request_run,
        "resolved_task": str(
            record.get("task") or infer_task_from_metadata(metadata_mapping or {})
        ),
        "resolved_n_train": int(record["n_train"]),
        "resolved_n_test": int(record["n_test"]),
        "resolved_n_features": int(record["n_features"]),
        "resolved_n_classes": (
            None if record.get("n_classes") is None else int(record["n_classes"])
        ),
        "resolved_filter_mode": (
            filter_payload.get("mode") if isinstance(filter_payload, Mapping) else None
        ),
        "resolved_filter_status": (
            filter_payload.get("status") if isinstance(filter_payload, Mapping) else None
        ),
        "resolved_filter_accepted": (
            filter_payload.get("accepted")
            if isinstance(filter_payload, Mapping)
            and isinstance(filter_payload.get("accepted"), bool)
            else None
        ),
        "teacher_conditionals_available": bool(
            isinstance(teacher_conditionals, Mapping)
            and teacher_conditionals.get("available") is True
        ),
    }


def dataset_catalog_schema():
    _require_pyarrow()
    return pa.schema(
        [
            pa.field("dataset_index", pa.int64()),
            pa.field("record_json", pa.large_string()),
            pa.field("record_sha256", pa.string()),
            pa.field("resolved_dataset_id", pa.string()),
            pa.field("resolved_request_run", pa.string()),
            pa.field("resolved_task", pa.string()),
            pa.field("resolved_n_train", pa.int64()),
            pa.field("resolved_n_test", pa.int64()),
            pa.field("resolved_n_features", pa.int64()),
            pa.field("resolved_n_classes", pa.int64()),
            pa.field("resolved_filter_mode", pa.string()),
            pa.field("resolved_filter_status", pa.string()),
            pa.field("resolved_filter_accepted", pa.bool_()),
            pa.field("teacher_conditionals_available", pa.bool_()),
        ]
    )


def write_dataset_catalog_records(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    _require_pyarrow()
    rows = [build_dataset_catalog_parquet_row(record) for record in records]
    table = pa.Table.from_pylist(rows, schema=dataset_catalog_schema())
    pq.write_table(table, Path(path), compression="zstd")


def write_json_record_parquet_records(
    path: str | Path, records: Sequence[Mapping[str, Any]]
) -> None:
    _require_pyarrow()
    rows = [_build_json_record_parquet_row(record) for record in records]
    table = pa.Table.from_pylist(rows, schema=json_record_parquet_schema())
    pq.write_table(table, Path(path), compression="zstd")


@dataclass(slots=True)
class BufferedJsonRecordParquetWriter:
    """Incremental parquet writer for JSON-backed record streams."""

    path: Path
    compression: str = "zstd"
    batch_size: int = _JSON_RECORD_PARQUET_BATCH_SIZE
    _writer: Any | None = None
    _rows: list[dict[str, Any]] = field(default_factory=list)

    def write(self, record: Mapping[str, Any]) -> None:
        self._rows.append(_build_json_record_parquet_row(record))
        if len(self._rows) >= int(self.batch_size):
            self.flush()

    def flush(self) -> None:
        _require_pyarrow()
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=json_record_parquet_schema())
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.path,
                table.schema,
                compression=self.compression,
            )
        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if self._writer is not None:
                self._writer.close()
                self._writer = None


def resolve_internal_root(
    public_root: str | Path,
    *,
    explicit_internal_root: str | Path | None = None,
) -> Path:
    """Resolve the internal sidecar root for one public run root or shard path."""

    if explicit_internal_root is not None:
        return Path(explicit_internal_root)

    resolved_public_root = Path(public_root)
    candidates = _internal_root_candidates(resolved_public_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def internal_shard_dir(
    *,
    internal_root: str | Path,
    shard_name: str,
) -> Path:
    """Return the internal sidecar directory for one public shard."""

    return Path(internal_root) / str(shard_name)


def iter_parquet_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON record payloads from a parquet-backed record stream."""

    _require_pyarrow()
    resolved_path = Path(path)
    if resolved_path.suffix != ".parquet":
        raise ValueError(f"legacy NDJSON record streams are unsupported: {resolved_path}")
    rows = pq.read_table(
        resolved_path,
        columns=["record_json", "record_sha256"],
    ).to_pylist()
    for row_number, row in enumerate(rows, start=1):
        record_json = row.get("record_json")
        if not isinstance(record_json, str):
            raise ValueError(
                f"Invalid parquet catalog row in {path}:{row_number}: missing record_json."
            )
        expected_sha = row.get("record_sha256")
        if not isinstance(expected_sha, str) or not expected_sha:
            raise ValueError(
                f"Invalid parquet catalog row in {path}:{row_number}: missing record_sha256."
            )
        actual_sha = sha256(record_json.encode("utf-8")).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"Invalid parquet catalog row in {path}:{row_number}: record_sha256 mismatch."
            )
        payload = json.loads(record_json)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid parquet catalog row in {path}:{row_number}: expected object."
            )
        yield payload
