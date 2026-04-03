"""Cycle-free helpers for stable hashed identity payloads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

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


__all__ = ["stable_blake2s_hex"]
