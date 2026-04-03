"""Shared CLI helpers."""

from __future__ import annotations

from typing import NoReturn

import click

from dagzoo.config import GeneratorConfig
from dagzoo.recipes import load_config_reference


def raise_usage_error(message: str) -> NoReturn:
    """Raise a Click usage error."""

    raise click.UsageError(message)


def load_config_or_usage_error(config_ref: str) -> GeneratorConfig:
    """Load one config ref or raise a CLI usage error with its parse message."""

    try:
        return load_config_reference(config_ref)
    except (OSError, TypeError, ValueError) as exc:
        raise_usage_error(str(exc))
    raise AssertionError("unreachable")
