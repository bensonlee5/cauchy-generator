"""Shared CLI helpers."""

from __future__ import annotations

import sys
from typing import NoReturn

from dagzoo.config import GeneratorConfig


def raise_usage_error(message: str) -> NoReturn:
    """Exit with argparse-compatible usage error semantics."""

    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def load_config_or_usage_error(path: str) -> GeneratorConfig:
    """Load one config file or raise a CLI usage error with its parse message."""

    try:
        return GeneratorConfig.from_yaml(path)
    except (TypeError, ValueError) as exc:
        raise_usage_error(str(exc))
    raise AssertionError("unreachable")
