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
from dagzoo.math import sanitize_json
from dagzoo.recipes import serialize_config_reference

HANDOFF_MANIFEST_FILENAME = "handoff_manifest.json"
GENERATE_HANDOFF_SCHEMA_NAME = "dagzoo_generate_handoff_manifest"
GENERATE_HANDOFF_SCHEMA_VERSION = 1
HANDOFF_SOURCE_FAMILY = "dagzoo.fixed_layout_scm"
HANDOFF_RECOMMENDED_TRAINING_CORPUS = "generated"
HANDOFF_RECOMMENDED_TRAINING_ARTIFACT_KEY = "generated_dir"
HANDOFF_CURATION_POLICY = "none"
_BLAKE2S_HEX_LENGTH = 32
_SHA256_HEX_LENGTH = 64
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


def _load_generated_identity(
    *,
    generated_dir: str | Path,
    expected_datasets: int,
) -> tuple[str, str]:
    """Read canonical corpus ids from emitted metadata instead of filesystem paths."""

    metadata_paths = sorted(Path(generated_dir).glob("shard_*/metadata.ndjson"))
    if not metadata_paths:
        _raise("generated_dir", "must contain shard metadata under shard_*/metadata.ndjson")

    generate_run_id: str | None = None
    dataset_ids: list[str] = []
    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    _raise(
                        f"{metadata_path}:{line_number}",
                        f"contains invalid JSON ({exc.msg})",
                    )
                record = _require_mapping(payload, path=f"{metadata_path}:{line_number}")
                metadata = _require_mapping(
                    record.get("metadata"),
                    path=f"{metadata_path}:{line_number}.metadata",
                )
                split_groups = _require_mapping(
                    metadata.get("split_groups"),
                    path=f"{metadata_path}:{line_number}.metadata.split_groups",
                )
                current_generate_run_id = _require_hex_string(
                    split_groups.get("request_run"),
                    path=f"{metadata_path}:{line_number}.metadata.split_groups.request_run",
                    expected_length=_BLAKE2S_HEX_LENGTH,
                )
                dataset_id = _require_hex_string(
                    metadata.get("dataset_id"),
                    path=f"{metadata_path}:{line_number}.metadata.dataset_id",
                    expected_length=_BLAKE2S_HEX_LENGTH,
                )
                if generate_run_id is None:
                    generate_run_id = current_generate_run_id
                elif current_generate_run_id != generate_run_id:
                    _raise(
                        str(metadata_path),
                        "contains multiple request_run identities; expected one canonical run",
                    )
                dataset_ids.append(dataset_id)

    if generate_run_id is None:
        _raise("generated_dir", "must contain at least one metadata record")

    if len(dataset_ids) != int(expected_datasets):
        _raise(
            "generated_dir",
            "metadata record count does not match generated_datasets "
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
    """Read posterior-predictive provenance from emitted metadata records."""

    metadata_paths = sorted(Path(generated_dir).glob("shard_*/metadata.ndjson"))
    if not metadata_paths:
        _raise("generated_dir", "must contain shard metadata under shard_*/metadata.ndjson")

    factorization: str | None = None
    teacher_conditional_export: bool | None = None
    metric_definition: str | None = None
    target_parent_prior: str | None = None
    target_parent_near_max_band_min_fraction: float | None = None
    target_parent_below_sqrt_prob: float | None = None
    target_parent_midrange_prob: float | None = None
    target_parent_counts: list[int] = []
    target_parent_fractions: list[float] = []
    target_parent_regimes: set[str] = set()
    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    _raise(
                        f"{metadata_path}:{line_number}",
                        f"contains invalid JSON ({exc.msg})",
                    )
                record = _require_mapping(payload, path=f"{metadata_path}:{line_number}")
                metadata = _require_mapping(
                    record.get("metadata"),
                    path=f"{metadata_path}:{line_number}.metadata",
                )
                prior = _require_mapping(
                    metadata.get("prior"),
                    path=f"{metadata_path}:{line_number}.metadata.prior",
                )
                current_factorization = _require_non_empty_string(
                    prior.get("factorization"),
                    path=f"{metadata_path}:{line_number}.metadata.prior.factorization",
                )
                posterior_predictive = _require_mapping(
                    metadata.get("posterior_predictive"),
                    path=f"{metadata_path}:{line_number}.metadata.posterior_predictive",
                )
                current_teacher_export = _require_bool(
                    posterior_predictive.get("teacher_conditional_export_enabled"),
                    path=(
                        f"{metadata_path}:{line_number}."
                        "metadata.posterior_predictive.teacher_conditional_export_enabled"
                    ),
                )
                current_metric_definition = _require_non_empty_string(
                    posterior_predictive.get("metric_definition"),
                    path=(
                        f"{metadata_path}:{line_number}."
                        "metadata.posterior_predictive.metric_definition"
                    ),
                )
                if factorization is None:
                    factorization = current_factorization
                elif factorization != current_factorization:
                    _raise(
                        str(metadata_path),
                        "contains multiple posterior predictive factorizations; expected one",
                    )
                if teacher_conditional_export is None:
                    teacher_conditional_export = current_teacher_export
                elif teacher_conditional_export != current_teacher_export:
                    _raise(
                        str(metadata_path),
                        "contains mixed teacher-conditional export settings; expected one",
                    )
                if metric_definition is None:
                    metric_definition = current_metric_definition
                elif metric_definition != current_metric_definition:
                    _raise(
                        str(metadata_path),
                        "contains mixed posterior predictive metric definitions; expected one",
                    )
                config = _require_mapping(
                    metadata.get("config"),
                    path=f"{metadata_path}:{line_number}.metadata.config",
                )
                dataset = _require_mapping(
                    config.get("dataset"),
                    path=f"{metadata_path}:{line_number}.metadata.config.dataset",
                )
                current_target_parent_prior = _require_non_empty_string(
                    dataset.get("target_parent_prior"),
                    path=f"{metadata_path}:{line_number}.metadata.config.dataset.target_parent_prior",
                )
                current_near_max_band_min_fraction = _require_non_negative_float(
                    dataset.get("target_parent_near_max_band_min_fraction"),
                    path=(
                        f"{metadata_path}:{line_number}."
                        "metadata.config.dataset.target_parent_near_max_band_min_fraction"
                    ),
                )
                current_below_sqrt_prob = _require_non_negative_float(
                    dataset.get("target_parent_below_sqrt_prob"),
                    path=(
                        f"{metadata_path}:{line_number}."
                        "metadata.config.dataset.target_parent_below_sqrt_prob"
                    ),
                )
                current_midrange_prob = _require_non_negative_float(
                    dataset.get("target_parent_midrange_prob"),
                    path=(
                        f"{metadata_path}:{line_number}."
                        "metadata.config.dataset.target_parent_midrange_prob"
                    ),
                )
                lineage = _require_mapping(
                    metadata.get("lineage"),
                    path=f"{metadata_path}:{line_number}.metadata.lineage",
                )
                assignments = _require_mapping(
                    lineage.get("assignments"),
                    path=f"{metadata_path}:{line_number}.metadata.lineage.assignments",
                )
                current_target_parent_count = _require_int(
                    assignments.get("target_parent_count"),
                    path=(
                        f"{metadata_path}:{line_number}."
                        "metadata.lineage.assignments.target_parent_count"
                    ),
                )
                current_target_parent_fraction = _require_non_negative_float(
                    assignments.get("target_parent_fraction"),
                    path=(
                        f"{metadata_path}:{line_number}."
                        "metadata.lineage.assignments.target_parent_fraction"
                    ),
                )
                current_target_parent_regime = _require_non_empty_string(
                    assignments.get("target_parent_regime"),
                    path=(
                        f"{metadata_path}:{line_number}."
                        "metadata.lineage.assignments.target_parent_regime"
                    ),
                )
                if target_parent_prior is None:
                    target_parent_prior = current_target_parent_prior
                elif target_parent_prior != current_target_parent_prior:
                    _raise(str(metadata_path), "contains mixed target-parent priors; expected one")
                if target_parent_near_max_band_min_fraction is None:
                    target_parent_near_max_band_min_fraction = current_near_max_band_min_fraction
                elif not math.isclose(
                    target_parent_near_max_band_min_fraction,
                    current_near_max_band_min_fraction,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    _raise(
                        str(metadata_path),
                        "contains mixed target-parent near-max band minimum fractions; expected one",
                    )
                if target_parent_below_sqrt_prob is None:
                    target_parent_below_sqrt_prob = current_below_sqrt_prob
                elif not math.isclose(
                    target_parent_below_sqrt_prob,
                    current_below_sqrt_prob,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    _raise(
                        str(metadata_path),
                        "contains mixed target-parent below-sqrt probabilities; expected one",
                    )
                if target_parent_midrange_prob is None:
                    target_parent_midrange_prob = current_midrange_prob
                elif not math.isclose(
                    target_parent_midrange_prob,
                    current_midrange_prob,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    _raise(
                        str(metadata_path),
                        "contains mixed target-parent midrange probabilities; expected one",
                    )
                target_parent_counts.append(int(current_target_parent_count))
                target_parent_fractions.append(float(current_target_parent_fraction))
                target_parent_regimes.add(str(current_target_parent_regime))

    if (
        factorization is None
        or teacher_conditional_export is None
        or metric_definition is None
        or target_parent_prior is None
        or target_parent_near_max_band_min_fraction is None
        or target_parent_below_sqrt_prob is None
        or target_parent_midrange_prob is None
        or not target_parent_counts
        or not target_parent_fractions
    ):
        _raise("generated_dir", "must contain posterior predictive provenance metadata")
    return {
        "posterior_predictive_factorization": factorization,
        "teacher_conditional_export": teacher_conditional_export,
        "teacher_conditional_metric_definition": metric_definition,
        "target_parent_prior": target_parent_prior,
        "target_parent_count_range": {
            "min": int(min(target_parent_counts)),
            "max": int(max(target_parent_counts)),
        },
        "target_parent_fraction_range": {
            "min": float(min(target_parent_fractions)),
            "max": float(max(target_parent_fractions)),
        },
        "target_parent_regimes_present": sorted(target_parent_regimes),
        "target_parent_near_max_band_min_fraction": float(target_parent_near_max_band_min_fraction),
        "target_parent_below_sqrt_prob": float(target_parent_below_sqrt_prob),
        "target_parent_midrange_prob": float(target_parent_midrange_prob),
    }


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
    diversity_summary_json_path: str | Path | None = None,
    diversity_summary_md_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the machine-readable generate handoff manifest payload."""

    overrides_payload = dict(generate_invocation_overrides)
    _validate_generate_overrides(overrides_payload, path="generate_invocation.overrides")
    generate_invocation = {
        "config_path": serialize_config_reference(config_path),
        "overrides": overrides_payload,
    }

    effective_config_sha256 = _read_sha256(effective_config_path)
    effective_config_trace_sha256 = _read_sha256(effective_config_trace_path)
    generate_run_id, generated_corpus_id = _load_generated_identity(
        generated_dir=generated_dir,
        expected_datasets=int(generated_datasets),
    )
    provenance = _load_generated_provenance(generated_dir=generated_dir)
    payload: dict[str, Any] = {
        "schema_name": GENERATE_HANDOFF_SCHEMA_NAME,
        "schema_version": GENERATE_HANDOFF_SCHEMA_VERSION,
        "identity": {
            "source_family": HANDOFF_SOURCE_FAMILY,
            "generate_run_id": generate_run_id,
            "generated_corpus_id": generated_corpus_id,
        },
        "generate_invocation": generate_invocation,
        "artifacts": {
            "run_root": _resolve_path_str(run_root),
            "generated_dir": _resolve_path_str(generated_dir),
            "effective_config_path": _resolve_path_str(effective_config_path),
            "effective_config_trace_path": _resolve_path_str(effective_config_trace_path),
        },
        "artifacts_relative": {
            "run_root": ".",
            "generated_dir": _relative_posix_path(generated_dir, start=run_root),
            "effective_config_path": _relative_posix_path(effective_config_path, start=run_root),
            "effective_config_trace_path": _relative_posix_path(
                effective_config_trace_path, start=run_root
            ),
        },
        "checksums": {
            "effective_config_sha256": effective_config_sha256,
            "effective_config_trace_sha256": effective_config_trace_sha256,
        },
        "summary": {
            "generated_datasets": int(generated_datasets),
        },
        "provenance": provenance,
        "throughput": {
            "generation_stage": {
                "generated_datasets": int(generated_datasets),
                "elapsed_seconds": float(generation_elapsed_seconds),
                "datasets_per_minute": _datasets_per_minute(
                    datasets=int(generated_datasets),
                    elapsed_seconds=float(generation_elapsed_seconds),
                ),
            },
        },
        "hardware": {
            "requested_device": str(requested_device),
            "resolved_device": str(resolved_device),
            "backend": str(hardware_backend),
            "device_name": str(hardware_device_name),
            "tier": str(hardware_tier),
            "hardware_policy": str(hardware_policy),
        },
        "diversity_artifacts": {
            "summary_json_path": _resolve_optional_path_str(diversity_summary_json_path),
            "summary_md_path": _resolve_optional_path_str(diversity_summary_md_path),
        },
        "defaults": {
            "recommended_training_corpus": HANDOFF_RECOMMENDED_TRAINING_CORPUS,
            "recommended_training_artifact_key": HANDOFF_RECOMMENDED_TRAINING_ARTIFACT_KEY,
            "curation_policy": HANDOFF_CURATION_POLICY,
        },
    }
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
    if source_family != HANDOFF_SOURCE_FAMILY:
        _raise(
            "handoff_manifest.identity.source_family",
            f"must equal {HANDOFF_SOURCE_FAMILY!r}",
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

    generate_invocation = _require_mapping(
        root.get("generate_invocation"),
        path="handoff_manifest.generate_invocation",
    )
    _require_non_empty_string(
        generate_invocation.get("config_path"),
        path="handoff_manifest.generate_invocation.config_path",
    )
    overrides = _require_mapping(
        generate_invocation.get("overrides"),
        path="handoff_manifest.generate_invocation.overrides",
    )
    _validate_generate_overrides(overrides, path="handoff_manifest.generate_invocation.overrides")

    artifacts = _require_mapping(root.get("artifacts"), path="handoff_manifest.artifacts")
    for key in (
        "run_root",
        "generated_dir",
        "effective_config_path",
        "effective_config_trace_path",
    ):
        _require_non_empty_string(
            artifacts.get(key),
            path=f"handoff_manifest.artifacts.{key}",
        )

    artifacts_relative = _require_mapping(
        root.get("artifacts_relative"),
        path="handoff_manifest.artifacts_relative",
    )
    run_root_relative = _require_non_empty_string(
        artifacts_relative.get("run_root"),
        path="handoff_manifest.artifacts_relative.run_root",
    )
    if run_root_relative != ".":
        _raise("handoff_manifest.artifacts_relative.run_root", "must equal '.'")
    for key in (
        "generated_dir",
        "effective_config_path",
        "effective_config_trace_path",
    ):
        _require_non_empty_string(
            artifacts_relative.get(key),
            path=f"handoff_manifest.artifacts_relative.{key}",
        )

    checksums = _require_mapping(root.get("checksums"), path="handoff_manifest.checksums")
    for key in (
        "effective_config_sha256",
        "effective_config_trace_sha256",
    ):
        _require_hex_string(
            checksums.get(key),
            path=f"handoff_manifest.checksums.{key}",
            expected_length=_SHA256_HEX_LENGTH,
        )

    summary = _require_mapping(root.get("summary"), path="handoff_manifest.summary")
    _require_int(
        summary.get("generated_datasets"), path="handoff_manifest.summary.generated_datasets"
    )
    provenance = _require_mapping(root.get("provenance"), path="handoff_manifest.provenance")
    _require_non_empty_string(
        provenance.get("posterior_predictive_factorization"),
        path="handoff_manifest.provenance.posterior_predictive_factorization",
    )
    _require_bool(
        provenance.get("teacher_conditional_export"),
        path="handoff_manifest.provenance.teacher_conditional_export",
    )
    _require_optional_string(
        provenance.get("teacher_conditional_metric_definition"),
        path="handoff_manifest.provenance.teacher_conditional_metric_definition",
    )
    _require_optional_string(
        provenance.get("target_parent_prior"),
        path="handoff_manifest.provenance.target_parent_prior",
    )
    target_parent_count_range = provenance.get("target_parent_count_range")
    if target_parent_count_range is not None:
        target_parent_count_range_mapping = _require_mapping(
            target_parent_count_range,
            path="handoff_manifest.provenance.target_parent_count_range",
        )
        _require_int(
            target_parent_count_range_mapping.get("min"),
            path="handoff_manifest.provenance.target_parent_count_range.min",
        )
        _require_int(
            target_parent_count_range_mapping.get("max"),
            path="handoff_manifest.provenance.target_parent_count_range.max",
        )
    target_parent_fraction_range = provenance.get("target_parent_fraction_range")
    if target_parent_fraction_range is not None:
        target_parent_fraction_range_mapping = _require_mapping(
            target_parent_fraction_range,
            path="handoff_manifest.provenance.target_parent_fraction_range",
        )
        _require_non_negative_float(
            target_parent_fraction_range_mapping.get("min"),
            path="handoff_manifest.provenance.target_parent_fraction_range.min",
        )
        _require_non_negative_float(
            target_parent_fraction_range_mapping.get("max"),
            path="handoff_manifest.provenance.target_parent_fraction_range.max",
        )
    target_parent_regimes_present = provenance.get("target_parent_regimes_present")
    if target_parent_regimes_present is not None:
        if not isinstance(target_parent_regimes_present, list):
            _raise(
                "handoff_manifest.provenance.target_parent_regimes_present",
                "must be a list of strings",
            )
        for index, value in enumerate(target_parent_regimes_present):
            _require_non_empty_string(
                value,
                path=f"handoff_manifest.provenance.target_parent_regimes_present[{index}]",
            )
    _require_optional_float(
        provenance.get("target_parent_near_max_band_min_fraction"),
        path="handoff_manifest.provenance.target_parent_near_max_band_min_fraction",
    )
    _require_optional_float(
        provenance.get("target_parent_below_sqrt_prob"),
        path="handoff_manifest.provenance.target_parent_below_sqrt_prob",
    )
    _require_optional_float(
        provenance.get("target_parent_midrange_prob"),
        path="handoff_manifest.provenance.target_parent_midrange_prob",
    )

    throughput = _require_mapping(root.get("throughput"), path="handoff_manifest.throughput")
    generation_stage = _require_mapping(
        throughput.get("generation_stage"),
        path="handoff_manifest.throughput.generation_stage",
    )
    _require_int(
        generation_stage.get("generated_datasets"),
        path="handoff_manifest.throughput.generation_stage.generated_datasets",
    )
    _require_non_negative_float(
        generation_stage.get("elapsed_seconds"),
        path="handoff_manifest.throughput.generation_stage.elapsed_seconds",
    )
    _require_non_negative_float(
        generation_stage.get("datasets_per_minute"),
        path="handoff_manifest.throughput.generation_stage.datasets_per_minute",
    )

    hardware = _require_mapping(root.get("hardware"), path="handoff_manifest.hardware")
    for key in (
        "requested_device",
        "resolved_device",
        "backend",
        "device_name",
        "tier",
        "hardware_policy",
    ):
        _require_non_empty_string(
            hardware.get(key),
            path=f"handoff_manifest.hardware.{key}",
        )

    diversity_artifacts = _require_mapping(
        root.get("diversity_artifacts"),
        path="handoff_manifest.diversity_artifacts",
    )
    _require_optional_string(
        diversity_artifacts.get("summary_json_path"),
        path="handoff_manifest.diversity_artifacts.summary_json_path",
    )
    _require_optional_string(
        diversity_artifacts.get("summary_md_path"),
        path="handoff_manifest.diversity_artifacts.summary_md_path",
    )

    defaults = _require_mapping(root.get("defaults"), path="handoff_manifest.defaults")
    recommended_training_corpus = _require_non_empty_string(
        defaults.get("recommended_training_corpus"),
        path="handoff_manifest.defaults.recommended_training_corpus",
    )
    if recommended_training_corpus != HANDOFF_RECOMMENDED_TRAINING_CORPUS:
        _raise(
            "handoff_manifest.defaults.recommended_training_corpus",
            f"must equal {HANDOFF_RECOMMENDED_TRAINING_CORPUS!r}",
        )
    recommended_artifact_key = _require_non_empty_string(
        defaults.get("recommended_training_artifact_key"),
        path="handoff_manifest.defaults.recommended_training_artifact_key",
    )
    if recommended_artifact_key != HANDOFF_RECOMMENDED_TRAINING_ARTIFACT_KEY:
        _raise(
            "handoff_manifest.defaults.recommended_training_artifact_key",
            f"must equal {HANDOFF_RECOMMENDED_TRAINING_ARTIFACT_KEY!r}",
        )
    curation_policy = _require_non_empty_string(
        defaults.get("curation_policy"),
        path="handoff_manifest.defaults.curation_policy",
    )
    if curation_policy != HANDOFF_CURATION_POLICY:
        _raise(
            "handoff_manifest.defaults.curation_policy",
            f"must equal {HANDOFF_CURATION_POLICY!r}",
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
    "build_generate_handoff_manifest",
    "validate_generate_handoff_manifest",
    "write_generate_handoff_manifest",
]
