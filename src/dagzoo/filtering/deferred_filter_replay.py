"""Deferred filter replay helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from dagzoo.config import FilterConfig, normalize_filter_config
from dagzoo.rng import SEED32_MAX, SEED32_MIN

_REMOVED_FILTER_THRESHOLD_MESSAGE = (
    "filter.threshold has been removed. Use filter.easy_skill_threshold, "
    "filter.easy_gain_threshold, and filter.hard_skill_threshold instead."
)


def _coerce_seed(raw_seed: object, *, dataset_index: int) -> int:
    """Resolve a valid seed32 for filter replay."""

    if isinstance(raw_seed, bool):
        raw_seed = None

    if isinstance(raw_seed, float):
        if math.isfinite(raw_seed) and float(raw_seed).is_integer():
            raw_seed = int(raw_seed)
        else:
            raw_seed = None

    if isinstance(raw_seed, int):
        if SEED32_MIN <= raw_seed <= SEED32_MAX:
            return int(raw_seed)

    return int(dataset_index % (SEED32_MAX + 1))


def _resolve_filter_seed(metadata_payload: Mapping[str, Any], *, dataset_index: int) -> int:
    """Resolve filter replay seed from persisted metadata with child-seed preference."""

    return _coerce_seed(
        metadata_payload.get("dataset_seed", metadata_payload.get("seed")),
        dataset_index=dataset_index,
    )


def _resolve_task_and_filter_config(
    *,
    metadata_payload: Mapping[str, Any],
    path_overrides: tuple[tuple[str, Any], ...],
) -> tuple[str, FilterConfig]:
    """Resolve task + filter config for one dataset record."""

    task: str | None = None
    embedded_filter: Mapping[str, Any] | None = None

    config_payload = metadata_payload.get("config")
    if isinstance(config_payload, Mapping):
        dataset_payload = config_payload.get("dataset")
        if isinstance(dataset_payload, Mapping):
            dataset_task = dataset_payload.get("task")
            if isinstance(dataset_task, str) and dataset_task.strip():
                task = dataset_task.strip().lower()

        filter_payload = config_payload.get("filter")
        if isinstance(filter_payload, Mapping):
            embedded_filter = filter_payload

    if task not in {"classification", "regression"}:
        raise ValueError(
            "Deferred filter requires embedded metadata.config.dataset.task in shard metadata."
        )

    if embedded_filter is not None:
        embedded_filter_payload = dict(embedded_filter)
        if "threshold" in embedded_filter_payload:
            raise ValueError(_REMOVED_FILTER_THRESHOLD_MESSAGE)
        filter_cfg = FilterConfig(**embedded_filter_payload)
    else:
        raise ValueError(
            "Deferred filter requires embedded metadata.config.filter in shard metadata."
        )

    filter_cfg.enabled = True
    for path, value in path_overrides:
        if not path.startswith("filter."):
            raise ValueError(f"filter --set only supports filter.<field> paths, got {path!r}.")
        field_name = path.split(".", 1)[1]
        if not hasattr(filter_cfg, field_name):
            raise ValueError(f"Unsupported filter override path {path!r}.")
        setattr(filter_cfg, field_name, value)
    normalize_filter_config(filter_cfg)

    return task, filter_cfg


def _build_filter_metadata(
    *,
    existing_filter: object,
    accepted: bool,
    filter_details: Mapping[str, Any],
) -> dict[str, Any]:
    """Build normalized filter metadata payload for deferred status."""

    payload: dict[str, Any] = dict(existing_filter) if isinstance(existing_filter, Mapping) else {}
    payload["mode"] = "deferred"
    payload["status"] = "accepted" if accepted else "rejected"
    payload["enabled"] = True
    payload["accepted"] = bool(accepted)
    payload.update(dict(filter_details))
    return payload
