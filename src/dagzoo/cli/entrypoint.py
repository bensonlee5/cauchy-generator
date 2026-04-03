"""CLI main entrypoint."""

from __future__ import annotations

import sys

import click

from .parser import build_cli


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    cli = build_cli()
    try:
        result = cli.main(args=argv, prog_name="dagzoo", standalone_mode=False)
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except click.exceptions.Exit as exc:
        if exc.exit_code == 0:
            return 0
        raise SystemExit(exc.exit_code) from exc
    except click.Abort as exc:
        raise SystemExit(1) from exc
    return 0 if result is None else int(result)
