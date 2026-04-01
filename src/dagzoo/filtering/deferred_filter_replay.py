"""Deferred filter replay helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from dagzoo.config import FilterConfig, normalize_filter_config
from dagzoo.config.models import _REMOVED_FILTER_FIELDS, _raise_removed_filter_fields
from dagzoo.rng import SEED32_MAX, SEED32_MIN


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


def _resolve_filter_config(
    *,
    metadata_payload: Mapping[str, Any],
    path_overrides: tuple[tuple[str, Any], ...],
) -> FilterConfig:
    """Resolve filter config for one dataset record."""

    embedded_filter: Mapping[str, Any] | None = None

    config_payload = metadata_payload.get("config")
    if isinstance(config_payload, Mapping):
        filter_payload = config_payload.get("filter")
        if isinstance(filter_payload, Mapping):
            embedded_filter = filter_payload

    if embedded_filter is not None:
        embedded_filter_payload = dict(embedded_filter)
        removed_filter_fields = set(embedded_filter_payload).intersection(_REMOVED_FILTER_FIELDS)
        if removed_filter_fields:
            _raise_removed_filter_fields(removed_filter_fields)
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
        if field_name in _REMOVED_FILTER_FIELDS:
            _raise_removed_filter_fields({field_name})
        if not hasattr(filter_cfg, field_name):
            raise ValueError(f"Unsupported filter override path {path!r}.")
        setattr(filter_cfg, field_name, value)
    normalize_filter_config(filter_cfg)

    return filter_cfg


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
