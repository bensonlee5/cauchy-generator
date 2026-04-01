"""Generate handoff manifest helpers for downstream corpus consumers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from time import time_ns
from typing import Any, NoReturn, cast

from dagzoo.core.identity import stable_blake2s_hex
from dagzoo.core.staged_artifacts import cleanup_path, promote_staged_path, staged_output_path
from dagzoo.io.lineage_artifact import sha256_hex
from dagzoo.io.shard_contract import DATASET_CATALOG_FILENAME, iter_ndjson_records
from dagzoo.math import sanitize_json

HANDOFF_MANIFEST_FILENAME = "handoff_manifest.json"
GENERATE_HANDOFF_SCHEMA_NAME = "dagzoo_generate_handoff_manifest"
GENERATE_HANDOFF_SCHEMA_VERSION = 4
HANDOFF_SOURCE_FAMILY_FIXED = "dagzoo.fixed_layout_scm"
HANDOFF_SOURCE_FAMILY_HETEROGENEOUS = "dagzoo.heterogeneous_scm"
HANDOFF_SOURCE_FAMILIES = (
    HANDOFF_SOURCE_FAMILY_FIXED,
    HANDOFF_SOURCE_FAMILY_HETEROGENEOUS,
)
_BLAKE2S_HEX_LENGTH = 32
_GENERATE_OVERRIDE_KEYS = (
    "num_datasets",
    "seed",
    "rows",
    "device",
    "hardware_policy",
    "missing_rate",
    "missing_mechanism",
    "missing_mar_observed_fraction",
    "missing_mar_logit_scale",
    "missing_mnar_logit_scale",
    "diagnostics",
    "diagnostics_out_dir",
    "handoff_root",
)
_OPTIONAL_GENERATE_OVERRIDE_KEYS = ("set_overrides",)


def _raise(path: str, message: str) -> NoReturn:
    raise ValueError(f"{path}: {message}")


def _require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise(path, "must be a mapping")
    return cast(Mapping[str, Any], value)


def _require_non_empty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _raise(path, "must be a non-empty string")
    return cast(str, value)


def _require_optional_string(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, path=path)


def _require_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        _raise(path, "must be a boolean")
    return cast(bool, value)


def _require_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(path, "must be an integer")
    return int(cast(int, value))


def _require_optional_int(value: object, *, path: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, path=path)


def _require_non_negative_float(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _raise(path, "must be a finite non-negative number")
    number = float(cast(int | float, value))
    if not math.isfinite(number) or number < 0.0:
        _raise(path, "must be a finite non-negative number")
    return number


def _require_optional_float(value: object, *, path: str) -> float | None:
    if value is None:
        return None
    return _require_non_negative_float(value, path=path)


def _require_hex_string(value: object, *, path: str, expected_length: int) -> str:
    text = _require_non_empty_string(value, path=path)
    if len(text) != expected_length or any(ch not in "0123456789abcdef" for ch in text):
        _raise(path, f"must be a {expected_length}-character lowercase hexadecimal string")
    return text


def _require_rows_override(value: object, *, path: str) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        _raise(path, "must be a string, integer, or null")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str) and value.strip():
        return value
    _raise(path, "must be a string, integer, or null")


def _require_set_overrides(value: object, *, path: str) -> list[tuple[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        _raise(path, "must be a list of [path, value] pairs")

    normalized: list[tuple[str, Any]] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            _raise(item_path, "must be a [path, value] pair")
        override_path = item[0]
        if not isinstance(override_path, str) or not override_path.strip():
            _raise(f"{item_path}[0]", "must be a non-empty string")
        normalized.append((override_path, item[1]))
    return normalized


def _resolve_path_str(path: str | Path) -> str:
    return str(Path(path).resolve())


def _resolve_optional_path_str(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return _resolve_path_str(path)


def _read_sha256(path: str | Path) -> str:
    return sha256_hex(Path(path).read_bytes())


def _relative_posix_path(path: str | Path, *, start: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(start).resolve()).as_posix()


def _datasets_per_minute(*, datasets: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0.0:
        return 0.0
    return (float(datasets) / float(elapsed_seconds)) * 60.0


def _validate_generate_overrides(overrides: Mapping[str, Any], *, path: str) -> None:
    expected_keys = set(_GENERATE_OVERRIDE_KEYS)
    optional_keys = set(_OPTIONAL_GENERATE_OVERRIDE_KEYS)
    actual_keys = set(overrides)
    unexpected_keys = sorted(actual_keys - expected_keys - optional_keys)
    if unexpected_keys:
        _raise(path, f"contains unknown keys: {', '.join(unexpected_keys)}")
    missing_keys = sorted(expected_keys - actual_keys)
    if missing_keys:
        _raise(path, f"is missing required keys: {', '.join(missing_keys)}")

    _require_int(overrides.get("num_datasets"), path=f"{path}.num_datasets")
    _require_optional_int(overrides.get("seed"), path=f"{path}.seed")
    _require_rows_override(overrides.get("rows"), path=f"{path}.rows")
    _require_optional_string(overrides.get("device"), path=f"{path}.device")
    _require_non_empty_string(overrides.get("hardware_policy"), path=f"{path}.hardware_policy")
    _require_optional_float(overrides.get("missing_rate"), path=f"{path}.missing_rate")
    _require_optional_string(
        overrides.get("missing_mechanism"),
        path=f"{path}.missing_mechanism",
    )
    _require_optional_float(
        overrides.get("missing_mar_observed_fraction"),
        path=f"{path}.missing_mar_observed_fraction",
    )
    _require_optional_float(
        overrides.get("missing_mar_logit_scale"),
        path=f"{path}.missing_mar_logit_scale",
    )
    _require_optional_float(
        overrides.get("missing_mnar_logit_scale"),
        path=f"{path}.missing_mnar_logit_scale",
    )
    _require_bool(overrides.get("diagnostics"), path=f"{path}.diagnostics")
    _require_optional_string(
        overrides.get("diagnostics_out_dir"),
        path=f"{path}.diagnostics_out_dir",
    )
    _require_non_empty_string(overrides.get("handoff_root"), path=f"{path}.handoff_root")
    _require_set_overrides(overrides.get("set_overrides"), path=f"{path}.set_overrides")


def _generated_catalog_paths(generated_dir: str | Path) -> list[Path]:
    """Return public shard-catalog paths for one generated corpus."""

    catalog_paths = sorted(Path(generated_dir).glob(f"shard_*/{DATASET_CATALOG_FILENAME}"))
    return catalog_paths


def _catalog_record_identity_fields(
    *,
    record: Mapping[str, Any],
    catalog_path: Path,
) -> tuple[str, str]:
    """Extract dataset and request-run identities from one catalog record."""

    split_groups = _require_mapping(
        record.get("group_ids"),
        path=f"{catalog_path}.group_ids",
    )
    return (
        _require_hex_string(
            split_groups.get("request_run"),
            path=f"{catalog_path}.group_ids.request_run",
            expected_length=_BLAKE2S_HEX_LENGTH,
        ),
        _require_hex_string(
            record.get("dataset_id"),
            path=f"{catalog_path}.dataset_id",
            expected_length=_BLAKE2S_HEX_LENGTH,
        ),
    )


def _load_generated_identity(
    *,
    generated_dir: str | Path,
    expected_datasets: int,
) -> tuple[str, str]:
    """Read canonical corpus ids from the public dataset catalog."""

    catalog_paths = _generated_catalog_paths(generated_dir)
    if not catalog_paths:
        _raise(
            "generated_dir",
            f"must contain shard catalogs under shard_*/{DATASET_CATALOG_FILENAME}",
        )

    generate_run_id: str | None = None
    dataset_ids: list[str] = []
    for catalog_path in catalog_paths:
        for record in iter_ndjson_records(catalog_path):
            current_generate_run_id, dataset_id = _catalog_record_identity_fields(
                record=record,
                catalog_path=catalog_path,
            )
            if generate_run_id is None:
                generate_run_id = current_generate_run_id
            elif current_generate_run_id != generate_run_id:
                _raise(
                    str(catalog_path),
                    "contains multiple request_run identities; expected one canonical run",
                )
            dataset_ids.append(dataset_id)

    if generate_run_id is None:
        _raise("generated_dir", "must contain at least one dataset catalog record")

    if len(dataset_ids) != int(expected_datasets):
        _raise(
            "generated_dir",
            "dataset catalog record count does not match generated_datasets "
            f"(records={len(dataset_ids)}, generated_datasets={int(expected_datasets)})",
        )

    generated_corpus_id = stable_blake2s_hex(
        {
            "generate_run_id": generate_run_id,
            "dataset_ids": dataset_ids,
        }
    )
    return generate_run_id, generated_corpus_id


def _load_generated_provenance(
    *,
    generated_dir: str | Path,
) -> dict[str, Any]:
    """Read one minimal generated-corpus provenance summary from the public dataset catalog."""

    catalog_paths = _generated_catalog_paths(generated_dir)
    if not catalog_paths:
        _raise(
            "generated_dir",
            f"must contain shard catalogs under shard_*/{DATASET_CATALOG_FILENAME}",
        )

    target_derivations: set[str] = set()
    target_relevant_feature_counts: list[int] = []
    target_relevant_feature_fractions: list[float] = []
    for catalog_path in catalog_paths:
        for record in iter_ndjson_records(catalog_path):
            target_derivation = record.get("target_derivation")
            if isinstance(target_derivation, str) and target_derivation.strip():
                target_derivations.add(target_derivation)
            target_relevance = record.get("target_relevance")
            if isinstance(target_relevance, Mapping):
                feature_count = target_relevance.get("feature_count")
                feature_fraction = target_relevance.get("feature_fraction")
                if not isinstance(feature_count, bool) and isinstance(feature_count, int):
                    target_relevant_feature_counts.append(int(feature_count))
                if not isinstance(feature_fraction, bool) and isinstance(
                    feature_fraction, (int, float)
                ):
                    target_relevant_feature_fractions.append(float(feature_fraction))
    payload: dict[str, Any] = {}
    if len(target_derivations) == 1:
        payload["target_derivation"] = next(iter(target_derivations))
    elif len(target_derivations) > 1:
        _raise("generated_dir", "contains mixed target derivation summaries; expected one")
    if target_relevant_feature_counts:
        payload["target_relevant_feature_count_range"] = {
            "min": int(min(target_relevant_feature_counts)),
            "max": int(max(target_relevant_feature_counts)),
        }
    if target_relevant_feature_fractions:
        payload["target_relevant_feature_fraction_range"] = {
            "min": float(min(target_relevant_feature_fractions)),
            "max": float(max(target_relevant_feature_fractions)),
        }
    return payload


def build_generate_handoff_manifest(
    *,
    config_path: str | Path,
    generate_invocation_overrides: Mapping[str, Any],
    run_root: str | Path,
    generated_dir: str | Path,
    effective_config_path: str | Path,
    effective_config_trace_path: str | Path,
    generated_datasets: int,
    generation_elapsed_seconds: float,
    requested_device: str,
    resolved_device: str,
    hardware_backend: str,
    hardware_device_name: str,
    hardware_tier: str,
    hardware_policy: str,
    source_family: str = HANDOFF_SOURCE_FAMILY_FIXED,
    diversity_summary_json_path: str | Path | None = None,
    diversity_summary_md_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the machine-readable generate handoff manifest payload."""

    generate_run_id, generated_corpus_id = _load_generated_identity(
        generated_dir=generated_dir,
        expected_datasets=int(generated_datasets),
    )
    provenance = _load_generated_provenance(generated_dir=generated_dir)
    run_root_path = Path(run_root).resolve()
    curated_dir = run_root_path / "curated"
    has_curated_dir = curated_dir.exists() and any(curated_dir.glob("shard_*"))
    payload: dict[str, Any] = {
        "schema_name": GENERATE_HANDOFF_SCHEMA_NAME,
        "schema_version": GENERATE_HANDOFF_SCHEMA_VERSION,
        "identity": {
            "source_family": str(source_family),
            "generate_run_id": generate_run_id,
            "generated_corpus_id": generated_corpus_id,
        },
        "artifacts_relative": {
            "generated_dir": _relative_posix_path(generated_dir, start=run_root),
        },
        "summary": {
            "generated_datasets": int(generated_datasets),
        },
    }
    if has_curated_dir:
        payload["artifacts_relative"]["curated_dir"] = _relative_posix_path(
            curated_dir, start=run_root
        )
    if provenance:
        payload["provenance"] = provenance
    validate_generate_handoff_manifest(payload)
    return payload


def validate_generate_handoff_manifest(payload: Mapping[str, Any]) -> None:
    """Validate the generate handoff manifest wire shape."""

    root = _require_mapping(payload, path="handoff_manifest")
    schema_name = _require_non_empty_string(
        root.get("schema_name"),
        path="handoff_manifest.schema_name",
    )
    if schema_name != GENERATE_HANDOFF_SCHEMA_NAME:
        _raise(
            "handoff_manifest.schema_name",
            f"must equal {GENERATE_HANDOFF_SCHEMA_NAME!r}",
        )
    schema_version = _require_int(
        root.get("schema_version"),
        path="handoff_manifest.schema_version",
    )
    if schema_version != GENERATE_HANDOFF_SCHEMA_VERSION:
        _raise(
            "handoff_manifest.schema_version",
            f"must equal {GENERATE_HANDOFF_SCHEMA_VERSION}",
        )

    identity = _require_mapping(root.get("identity"), path="handoff_manifest.identity")
    source_family = _require_non_empty_string(
        identity.get("source_family"),
        path="handoff_manifest.identity.source_family",
    )
    if source_family not in HANDOFF_SOURCE_FAMILIES:
        _raise(
            "handoff_manifest.identity.source_family",
            f"must equal one of {HANDOFF_SOURCE_FAMILIES!r}",
        )
    _require_hex_string(
        identity.get("generate_run_id"),
        path="handoff_manifest.identity.generate_run_id",
        expected_length=_BLAKE2S_HEX_LENGTH,
    )
    _require_hex_string(
        identity.get("generated_corpus_id"),
        path="handoff_manifest.identity.generated_corpus_id",
        expected_length=_BLAKE2S_HEX_LENGTH,
    )

    artifacts_relative = _require_mapping(
        root.get("artifacts_relative"),
        path="handoff_manifest.artifacts_relative",
    )
    for key in ("generated_dir",):
        _require_non_empty_string(
            artifacts_relative.get(key),
            path=f"handoff_manifest.artifacts_relative.{key}",
        )
    _require_optional_string(
        artifacts_relative.get("curated_dir"),
        path="handoff_manifest.artifacts_relative.curated_dir",
    )

    summary = _require_mapping(root.get("summary"), path="handoff_manifest.summary")
    _require_int(
        summary.get("generated_datasets"), path="handoff_manifest.summary.generated_datasets"
    )
    provenance = root.get("provenance")
    if provenance is not None:
        provenance_mapping = _require_mapping(provenance, path="handoff_manifest.provenance")
        target_derivation = provenance_mapping.get("target_derivation")
        if target_derivation is not None:
            _require_non_empty_string(
                target_derivation,
                path="handoff_manifest.provenance.target_derivation",
            )
        for key in (
            "target_relevant_feature_count_range",
            "target_relevant_feature_fraction_range",
        ):
            range_payload = provenance_mapping.get(key)
            if range_payload is None:
                continue
            range_mapping = _require_mapping(
                range_payload,
                path=f"handoff_manifest.provenance.{key}",
            )
            _require_non_negative_float(
                range_mapping.get("min"),
                path=f"handoff_manifest.provenance.{key}.min",
            )
            _require_non_negative_float(
                range_mapping.get("max"),
                path=f"handoff_manifest.provenance.{key}.max",
            )


def write_generate_handoff_manifest(
    *,
    config_path: str | Path,
    generate_invocation_overrides: Mapping[str, Any],
    run_root: str | Path,
    generated_dir: str | Path,
    effective_config_path: str | Path,
    effective_config_trace_path: str | Path,
    generated_datasets: int,
    generation_elapsed_seconds: float,
    requested_device: str,
    resolved_device: str,
    hardware_backend: str,
    hardware_device_name: str,
    hardware_tier: str,
    hardware_policy: str,
    source_family: str = HANDOFF_SOURCE_FAMILY_FIXED,
    diversity_summary_json_path: str | Path | None = None,
    diversity_summary_md_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Write the generate handoff manifest to disk and return its path."""

    manifest_path = (
        Path(out_path) if out_path is not None else Path(run_root) / HANDOFF_MANIFEST_FILENAME
    )
    payload = build_generate_handoff_manifest(
        config_path=config_path,
        generate_invocation_overrides=generate_invocation_overrides,
        run_root=run_root,
        generated_dir=generated_dir,
        effective_config_path=effective_config_path,
        effective_config_trace_path=effective_config_trace_path,
        generated_datasets=generated_datasets,
        generation_elapsed_seconds=generation_elapsed_seconds,
        requested_device=requested_device,
        resolved_device=resolved_device,
        hardware_backend=hardware_backend,
        hardware_device_name=hardware_device_name,
        hardware_tier=hardware_tier,
        hardware_policy=hardware_policy,
        source_family=source_family,
        diversity_summary_json_path=diversity_summary_json_path,
        diversity_summary_md_path=diversity_summary_md_path,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staged_manifest_path = staged_output_path(
        parent_dir=manifest_path.parent,
        final_name=manifest_path.name,
        staging_token=str(time_ns()),
    )
    try:
        staged_manifest_path.write_text(
            json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        promote_staged_path(staged_path=staged_manifest_path, final_path=manifest_path)
    finally:
        cleanup_path(staged_manifest_path)
    return manifest_path


__all__ = [
    "GENERATE_HANDOFF_SCHEMA_NAME",
    "GENERATE_HANDOFF_SCHEMA_VERSION",
    "HANDOFF_MANIFEST_FILENAME",
    "HANDOFF_SOURCE_FAMILY_FIXED",
    "HANDOFF_SOURCE_FAMILY_HETEROGENEOUS",
    "build_generate_handoff_manifest",
    "validate_generate_handoff_manifest",
    "write_generate_handoff_manifest",
]
