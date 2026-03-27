"""Effective-config rendering and persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dagzoo.config import GeneratorConfig, effective_config_payload


def effective_config_payload_yaml(payload: dict[str, Any]) -> str:
    """Render an already-materialized effective config payload as YAML text."""

    return yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
    )


def effective_config_yaml(config: GeneratorConfig) -> str:
    """Render an effective config payload as YAML text."""

    return effective_config_payload_yaml(effective_config_payload(config))


def write_effective_config_payload(payload: dict[str, Any], path: Path) -> Path:
    """Persist one already-materialized effective config payload as YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(effective_config_payload_yaml(payload), encoding="utf-8")
    return path


def write_effective_config(config: GeneratorConfig, path: Path) -> Path:
    """Persist effective config YAML to disk."""

    return write_effective_config_payload(effective_config_payload(config), path)


def effective_resolution_trace_yaml(trace_payload: list[dict[str, Any]]) -> str:
    """Render a field-level config resolution trace as YAML text."""

    return yaml.safe_dump(
        trace_payload,
        sort_keys=False,
        default_flow_style=False,
    )


def write_effective_config_trace(trace_payload: list[dict[str, Any]], path: Path) -> Path:
    """Persist effective config resolution trace YAML to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(effective_resolution_trace_yaml(trace_payload), encoding="utf-8")
    return path


def print_effective_config(config: GeneratorConfig, *, header: str) -> None:
    """Print effective config YAML to stdout with a short header."""

    print_effective_config_payload(effective_config_payload(config), header=header)


def print_effective_config_payload(payload: dict[str, Any], *, header: str) -> None:
    """Print one already-materialized effective config payload as YAML."""

    print(header)
    print(effective_config_payload_yaml(payload).rstrip())


def print_resolution_trace(trace_payload: list[dict[str, Any]], *, header: str) -> None:
    """Print config resolution trace YAML to stdout with a short header."""

    print(header)
    print(effective_resolution_trace_yaml(trace_payload).rstrip())
