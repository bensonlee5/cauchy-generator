#!/usr/bin/env python3
"""Validate internal links in built Hugo output.

Checks all HTML files in a built output directory (for example `site/public/`) and
fails when:
- absolute internal links ignore the configured base path
- internal links point to non-existent files/routes in the built output
- GitHub source/edit links still point at `/.generated/content/` paths
"""

from __future__ import annotations

import posixpath
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import click

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_ROOT = REPO_ROOT / "site"
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

ATTR_RE = re.compile(
    r"(?:\bhref|\bsrc)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))",
    re.IGNORECASE,
)

SKIP_PREFIXES = ("mailto:", "tel:", "javascript:", "data:", "//")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_base_path() -> str:
    text = _read_text(SITE_ROOT / "hugo.yaml")
    match = re.search(r"^baseURL:\s*(.+)$", text, re.MULTILINE)
    if not match:
        return ""
    parsed = urlparse(match.group(1).strip())
    path = parsed.path.rstrip("/")
    return "" if path == "/" else path


BASE_PATH = _read_base_path()


def _iter_html_files(output_dir: Path) -> Iterable[Path]:
    for path in output_dir.rglob("*.html"):
        if path.is_file():
            yield path


def _normalize_attr_target(raw_target: str) -> str:
    return raw_target.strip().strip('"').strip("'")


def _extract_targets(html: str) -> list[str]:
    targets: list[str] = []
    for match in ATTR_RE.finditer(html):
        targets.append(match.group(1) or match.group(2) or match.group(3) or "")
    return targets


def _resolve_internal_path(output_dir: Path, html_file: Path, target: str) -> tuple[str, str]:
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc:
        return ("skip", "")

    path = parsed.path
    if not path or path.startswith("#") or path.startswith(SKIP_PREFIXES):
        return ("skip", "")

    if path.startswith("/"):
        if BASE_PATH:
            if path == BASE_PATH or path == f"{BASE_PATH}/":
                return ("ok", "")
            prefix = f"{BASE_PATH}/"
            if path.startswith(prefix):
                return ("ok", path[len(BASE_PATH) :].lstrip("/"))
            return ("bad_prefix", path)
        return ("ok", path.lstrip("/"))

    current_rel_dir = html_file.relative_to(output_dir).parent.as_posix()
    joined = posixpath.normpath(posixpath.join(current_rel_dir, path))
    if joined == ".":
        joined = ""
    if joined.startswith("../"):
        return ("escape_root", path)
    return ("ok", joined)


def _is_generated_source_link(target: str) -> bool:
    parsed = urlparse(target)
    return bool(parsed.scheme and parsed.netloc and "/.generated/content/" in parsed.path)


def _built_target_exists(output_dir: Path, rel_path: str) -> bool:
    if rel_path == "":
        return (output_dir / "index.html").exists()

    candidate = output_dir / rel_path
    if candidate.exists():
        return True

    if rel_path.endswith("/"):
        return (output_dir / rel_path / "index.html").exists()

    if candidate.suffix:
        return False

    return any(
        path.exists()
        for path in (
            output_dir / rel_path / "index.html",
            (output_dir / rel_path).with_suffix(".html"),
        )
    )


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("output_dir", required=False)
def cli(*, output_dir: str | None) -> int:
    """Validate internal links in built Hugo output."""

    output_dir = (REPO_ROOT / (output_dir or "site/public")).resolve()
    if not output_dir.exists():
        print(f"Built output directory does not exist: {output_dir}")
        return 1

    prefix_errors: list[tuple[Path, str]] = []
    missing_errors: list[tuple[Path, str]] = []
    escape_errors: list[tuple[Path, str]] = []
    generated_source_errors: list[tuple[Path, str]] = []

    for html_file in _iter_html_files(output_dir):
        text = _read_text(html_file)
        for raw_target in _extract_targets(text):
            target = _normalize_attr_target(raw_target)
            if not target:
                continue
            if _is_generated_source_link(target):
                generated_source_errors.append((html_file, target))
                continue

            status, resolved = _resolve_internal_path(output_dir, html_file, target)
            if status == "skip":
                continue
            if status == "bad_prefix":
                prefix_errors.append((html_file, target))
                continue
            if status == "escape_root":
                escape_errors.append((html_file, target))
                continue

            if status == "ok" and not _built_target_exists(output_dir, resolved):
                missing_errors.append((html_file, target))

    if prefix_errors or missing_errors or escape_errors or generated_source_errors:
        if prefix_errors:
            print("Base-path prefix violations:")
            for html_file, target in prefix_errors:
                rel = html_file.relative_to(output_dir)
                print(f"- {rel} -> {target}")
            print()

        if escape_errors:
            print("Links escaping output root:")
            for html_file, target in escape_errors:
                rel = html_file.relative_to(output_dir)
                print(f"- {rel} -> {target}")
            print()

        if generated_source_errors:
            print("Generated-source GitHub link violations:")
            for html_file, target in generated_source_errors:
                rel = html_file.relative_to(output_dir)
                print(f"- {rel} -> {target}")
            print()

        if missing_errors:
            print("Unresolved internal links:")
            for html_file, target in missing_errors:
                rel = html_file.relative_to(output_dir)
                print(f"- {rel} -> {target}")

        return 1

    print("Built-output link check passed.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    try:
        result = cli.main(
            args=list(argv) if argv is not None else None,
            prog_name="check_built_output_links.py",
            standalone_mode=False,
        )
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return int(exc.exit_code)
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.Abort:
        return 1
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
