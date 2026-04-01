"""Stable identity helpers for persisted request, corpus, and dataset artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from dagzoo.core.fixed_layout.prepare import normalize_fixed_layout_target_cells
from dagzoo.math import sanitize_json


def stable_blake2s_hex(payload: Mapping[str, Any], *, digest_size: int = 16) -> str:
    """Return a stable BLAKE2s hex digest for sanitized JSON-compatible payloads."""

    encoded = json.dumps(
        sanitize_json(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=int(digest_size)).hexdigest()


def _require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return value


def _require_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string.")
    return value


def _require_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    return int(value)


def _require_optional_int(value: object, *, path: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, path=path)


def _require_number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be a finite number.")
    return number


def _normalized_float_mapping(value: object, *, path: str) -> dict[str, float]:
    mapping = _require_mapping(value, path=path)
    normalized: dict[str, float] = {}
    for key in sorted(mapping):
        normalized[str(key)] = _require_number(mapping[key], path=f"{path}.{key}")
    return normalized


def canonical_request_run_provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical run-level provenance payload used for dataset identities."""

    config = _require_mapping(metadata.get("config"), path="metadata.config")
    dataset = _require_mapping(config.get("dataset"), path="metadata.config.dataset")
    runtime = _require_mapping(config.get("runtime"), path="metadata.config.runtime")
    noise_distribution = _require_mapping(
        metadata.get("noise_distribution"),
        path="metadata.noise_distribution",
    )
    shift = _require_mapping(metadata.get("shift"), path="metadata.shift")

    missing_rate = _require_number(
        dataset.get("missing_rate"),
        path="metadata.config.dataset.missing_rate",
    )
    missing_mechanism = _require_string(
        dataset.get("missing_mechanism"),
        path="metadata.config.dataset.missing_mechanism",
    )
    if missing_rate <= 0.0:
        missing_mechanism = "none"
    missingness_payload: dict[str, Any] = {
        "missing_rate": missing_rate,
        "missing_mechanism": missing_mechanism,
    }
    if missing_mechanism == "mar":
        missingness_payload["missing_mar_observed_fraction"] = _require_number(
            dataset.get("missing_mar_observed_fraction"),
            path="metadata.config.dataset.missing_mar_observed_fraction",
        )
        missingness_payload["missing_mar_logit_scale"] = _require_number(
            dataset.get("missing_mar_logit_scale"),
            path="metadata.config.dataset.missing_mar_logit_scale",
        )
    elif missing_mechanism == "mnar":
        missingness_payload["missing_mnar_logit_scale"] = _require_number(
            dataset.get("missing_mnar_logit_scale"),
            path="metadata.config.dataset.missing_mnar_logit_scale",
        )

    noise_family = _require_string(
        noise_distribution.get("family_requested"),
        path="metadata.noise_distribution.family_requested",
    )
    noise_payload: dict[str, Any] = {
        "family_requested": noise_family,
        "base_scale": _require_number(
            noise_distribution.get("base_scale"),
            path="metadata.noise_distribution.base_scale",
        ),
    }
    if noise_family == "student_t":
        noise_payload["student_t_df"] = _require_number(
            noise_distribution.get("student_t_df"),
            path="metadata.noise_distribution.student_t_df",
        )
    elif noise_family == "mixture":
        noise_payload["mixture_weights"] = _normalized_float_mapping(
            noise_distribution.get("mixture_weights"),
            path="metadata.noise_distribution.mixture_weights",
        )

    return {
        "dataset": {
            "task": _require_string(
                dataset.get("task"),
                path="metadata.config.dataset.task",
            ),
            "n_train": _require_int(
                dataset.get("n_train"),
                path="metadata.config.dataset.n_train",
            ),
            "n_test": _require_int(
                dataset.get("n_test"),
                path="metadata.config.dataset.n_test",
            ),
            "target_derivation": _require_string(
                _require_mapping(
                    metadata.get("prior"),
                    path="metadata.prior",
                ).get("target_derivation"),
                path="metadata.prior.target_derivation",
            ),
            "missingness": missingness_payload,
        },
        "noise": noise_payload,
        "shift": {
            "variance_sigma_multiplier": _require_number(
                shift.get("variance_sigma_multiplier"),
                path="metadata.shift.variance_sigma_multiplier",
            )
        },
        "runtime": {
            "fixed_layout_target_cells": normalize_fixed_layout_target_cells(
                _require_optional_int(
                    runtime.get("fixed_layout_target_cells"),
                    path="metadata.config.runtime.fixed_layout_target_cells",
                )
            ),
            "resolved_device": _require_string(
                metadata.get("resolved_device"),
                path="metadata.resolved_device",
            ),
            "compute_backend": _require_string(
                metadata.get("compute_backend"),
                path="metadata.compute_backend",
            ),
            "torch_dtype": _require_string(
                runtime.get("torch_dtype"),
                path="metadata.config.runtime.torch_dtype",
            ),
        },
    }


def canonical_request_run_split_group(
    *,
    seed: int,
    run_num_datasets: int,
    layout_signature: str,
    layout_plan_signature: str,
    request_run_provenance: Mapping[str, Any],
) -> str:
    """Return a stable split-group key for one canonical request run."""

    return stable_blake2s_hex(
        {
            "seed": int(seed),
            "run_num_datasets": int(run_num_datasets),
            "layout_signature": str(layout_signature),
            "layout_plan_signature": str(layout_plan_signature),
            "request_run_provenance": dict(request_run_provenance),
        }
    )


def canonical_layout_plan_split_group(
    *,
    layout_signature: str,
    layout_plan_signature: str,
    layout_execution_contract: str,
) -> str:
    """Return a stable split-group key for one fixed-layout execution plan."""

    return stable_blake2s_hex(
        {
            "layout_signature": str(layout_signature),
            "layout_plan_signature": str(layout_plan_signature),
            "layout_execution_contract": str(layout_execution_contract),
        }
    )


def canonical_dataset_id(
    *,
    request_run_split_group: str,
    layout_plan_split_group: str,
    dataset_index: int,
    dataset_seed: int,
) -> str:
    """Return a stable dataset identifier for one canonical dataset bundle."""

    return stable_blake2s_hex(
        {
            "request_run_split_group": str(request_run_split_group),
            "layout_plan_split_group": str(layout_plan_split_group),
            "dataset_index": int(dataset_index),
            "dataset_seed": int(dataset_seed),
        }
    )


def heterogeneous_request_run_split_group(
    *,
    seed: int,
    run_num_datasets: int,
    request_run_provenance: Mapping[str, Any],
) -> str:
    """Return a stable split-group key for one heterogeneous request run."""

    return stable_blake2s_hex(
        {
            "seed": int(seed),
            "run_num_datasets": int(run_num_datasets),
            "layout_mode": "heterogeneous",
            "request_run_provenance": dict(request_run_provenance),
        }
    )


def heterogeneous_cohort_split_group(
    *,
    cohort_payload: Mapping[str, Any],
) -> str:
    """Return a stable grouping key for one heterogeneous raw-generation cohort."""

    return stable_blake2s_hex(
        {
            "layout_mode": "heterogeneous",
            "cohort_payload": dict(cohort_payload),
        }
    )


def heterogeneous_dataset_id(
    *,
    request_run_split_group: str,
    cohort_split_group: str,
    dataset_index: int,
    dataset_seed: int,
) -> str:
    """Return a stable dataset identifier for one heterogeneous dataset bundle."""

    return stable_blake2s_hex(
        {
            "request_run_split_group": str(request_run_split_group),
            "cohort_split_group": str(cohort_split_group),
            "dataset_index": int(dataset_index),
            "dataset_seed": int(dataset_seed),
        }
    )


__all__ = [
    "canonical_dataset_id",
    "canonical_layout_plan_split_group",
    "canonical_request_run_provenance",
    "canonical_request_run_split_group",
    "heterogeneous_cohort_split_group",
    "heterogeneous_dataset_id",
    "heterogeneous_request_run_split_group",
    "stable_blake2s_hex",
]
