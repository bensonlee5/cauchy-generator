#!/usr/bin/env python3
"""Check repo-root Markdown path references for stale or missing targets."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import click

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = (
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs",
    "scripts/README.md",
)
REPO_PATH_PREFIXES = (
    "src/",
    "docs/",
    "scripts/",
    "configs/",
    "recipes/",
    "reference/",
    "site/",
)
REPO_PATH_EXACT = frozenset(
    {
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "AGENTS.md",
        "pyproject.toml",
        ".pre-commit-config.yaml",
    }
)
LEGACY_PATH_DENYLIST = frozenset(
    {
        "src/dagzoo/cli.py",
        "src/dagzoo/config.py",
        "src/dagzoo/meta_targets.py",
        "src/dagzoo/core/parallel_generation.py",
        "src/dagzoo/core/worker_partition.py",
        "src/dagzoo/core/generation_engine.py",
    }
)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() == ".md":
            yield root
        return
    for path in root.rglob("*.md"):
        if path.is_file():
            yield path


def _iter_inline_code_spans(path: Path) -> Iterable[tuple[int, str]]:
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in INLINE_CODE_RE.finditer(line):
            yield lineno, match.group(1).strip()


def _iter_markdown_link_targets(path: Path) -> Iterable[tuple[int, str]]:
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line_without_code = INLINE_CODE_RE.sub("", line)
        for match in MARKDOWN_LINK_RE.finditer(line_without_code):
            yield lineno, match.group(1).strip()


def _is_repo_path(candidate: str) -> bool:
    return candidate in REPO_PATH_EXACT or candidate.startswith(REPO_PATH_PREFIXES)


def _normalize_repo_path_candidate(candidate: str) -> str | None:
    stripped = candidate.strip().rstrip(".,:;")
    if any(char in stripped for char in "*?[]{}"):
        return None
    if (REPO_ROOT / stripped).exists():
        return stripped
    token = stripped.split()[0]
    if any(char in token for char in "*?[]{}"):
        return None
    return token


def _normalize_markdown_link_candidate(candidate: str, *, source_path: Path) -> str | None:
    stripped = candidate.strip()
    if not stripped:
        return None
    if stripped.startswith("<") and stripped.endswith(">"):
        stripped = stripped[1:-1].strip()
    if not stripped:
        return None
    if URI_SCHEME_RE.match(stripped):
        return None
    if stripped.startswith(("#", "/")):
        return None

    link_target = stripped.split(maxsplit=1)[0]
    path_only = re.split(r"[?#]", link_target, maxsplit=1)[0]
    if not path_only or any(char in path_only for char in "*?[]{}"):
        return None

    resolved = (source_path.parent / path_only).resolve(strict=False)
    if not resolved.is_relative_to(REPO_ROOT):
        return None
    return resolved.relative_to(REPO_ROOT).as_posix()


def _scan_file(path: Path) -> list[tuple[int, str]]:
    errors: list[tuple[int, str]] = []
    for lineno, candidate in _iter_inline_code_spans(path):
        normalized = _normalize_repo_path_candidate(candidate)
        if normalized is None or not _is_repo_path(normalized):
            continue
        if normalized in LEGACY_PATH_DENYLIST:
            errors.append((lineno, f"{normalized} (legacy path)"))
            continue
        if not (REPO_ROOT / normalized).exists():
            errors.append((lineno, f"{normalized} (missing path)"))
    for lineno, candidate in _iter_markdown_link_targets(path):
        normalized = _normalize_markdown_link_candidate(candidate, source_path=path)
        if normalized is None or not _is_repo_path(normalized):
            continue
        if normalized in LEGACY_PATH_DENYLIST:
            errors.append((lineno, f"{normalized} (legacy path)"))
            continue
        if not (REPO_ROOT / normalized).exists():
            errors.append((lineno, f"{normalized} (missing path)"))
    return errors


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("roots", nargs=-1)
def cli(*, roots: tuple[str, ...]) -> int:
    """Check repo-root Markdown path references for stale or missing targets."""

    scan_roots = roots or DEFAULT_ROOTS
    missing_roots: list[str] = []
    all_errors: list[tuple[Path, int, str]] = []
    for root_rel in scan_roots:
        root = REPO_ROOT / root_rel
        if not root.exists():
            missing_roots.append(root_rel)
            continue
        for path in _iter_markdown_files(root):
            for lineno, target in _scan_file(path):
                all_errors.append((path, lineno, target))

    if missing_roots or all_errors:
        if missing_roots:
            print("Missing Markdown scan roots:")
            for root_rel in missing_roots:
                print(f"- {root_rel}")
        if all_errors:
            print("Broken repo path references found:")
            for path, lineno, target in all_errors:
                print(f"- {path.relative_to(REPO_ROOT)}:{lineno} -> {target}")
        return 1

    print("Repo path check passed.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    try:
        result = cli.main(
            args=list(argv) if argv is not None else None,
            prog_name="check_repo_paths.py",
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
