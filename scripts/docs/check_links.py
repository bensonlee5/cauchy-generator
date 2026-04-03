#!/usr/bin/env python3
"""Link checker for source docs and generated Hugo content.

Checks local links in:
- `README.md`
- `docs/**`
- `site/content/**`
- `site/.generated/content/**`

Validation includes:
- file/route existence checks for local links
- strict guard for authored `site/content/**` links:
  internal root-absolute links must include the configured Hugo base path
  (for this repo: `/dagzoo/...`)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import click

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_ROOT = REPO_ROOT / "site"
SITE_CONTENT_ROOT = SITE_ROOT / "content"

DEFAULT_ROOTS = [
    "README.md",
    "docs",
    "site/content",
    "site/.generated/content",
]

# Markdown inline links/images: [label](target) / ![alt](target)
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

# HTML href/src values (quoted or unquoted)
HTML_LINK_RE = re.compile(
    r"(?:\bhref|\bsrc)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))",
    re.IGNORECASE,
)

SKIP_PREFIXES = (
    "http://",
    "https://",
    "mailto:",
    "tel:",
    "javascript:",
    "data:",
    "//",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_base_path() -> str:
    """Extract URL path prefix from Hugo baseURL config."""
    text = _read_text(SITE_ROOT / "hugo.yaml")
    match = re.search(r"^baseURL:\s*(.+)$", text, re.MULTILINE)
    if not match:
        return ""
    parsed = urlparse(match.group(1).strip())
    path = parsed.path.rstrip("/")
    return "" if path == "/" else path


def _read_base_url() -> str:
    text = _read_text(SITE_ROOT / "hugo.yaml")
    match = re.search(r"^baseURL:\s*(.+)$", text, re.MULTILINE)
    if not match:
        return ""
    return match.group(1).strip().rstrip("/")


BASE_PATH = _read_base_path()
BASE_URL = _read_base_url()
BASE_URL_PARSED = urlparse(BASE_URL) if BASE_URL else None
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _iter_doc_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    if root.is_file():
        if root.suffix.lower() in {".md", ".html"}:
            yield root
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".html"}:
            yield path


def _normalize_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if target.startswith(("{{<", "{{%")) and target.endswith((">}}", "%}}")):
        return target
    if " " in target and not target.startswith(("http://", "https://")):
        # Handles inline title form: path "title"
        target = target.split(" ", 1)[0]
    return target


def _strip_base_path(target: str) -> str | None:
    """Map '/<base>/x' -> '/x'. Return None when target doesn't include base prefix."""
    if not BASE_PATH:
        return target
    if target == BASE_PATH or target == f"{BASE_PATH}/":
        return "/"
    prefix = f"{BASE_PATH}/"
    if target.startswith(prefix):
        return target[len(BASE_PATH) :]
    return None


def _normalize_repo_site_target(target: str) -> str:
    parsed = urlparse(target)
    if BASE_URL_PARSED is None or not parsed.scheme or not parsed.netloc:
        return target
    if parsed.scheme != BASE_URL_PARSED.scheme or parsed.netloc != BASE_URL_PARSED.netloc:
        return target

    if BASE_PATH:
        if parsed.path != BASE_PATH and not parsed.path.startswith(f"{BASE_PATH}/"):
            return target

    normalized = parsed.path or "/"
    if parsed.query:
        normalized = f"{normalized}?{parsed.query}"
    if parsed.fragment:
        normalized = f"{normalized}#{parsed.fragment}"
    return normalized


def _route_candidates(route: str) -> list[Path]:
    route = route.strip("/")
    if not route:
        return [
            REPO_ROOT / "site/content/_index.md",
            REPO_ROOT / "site/.generated/content/_index.md",
        ]

    candidates: list[Path] = []
    for content_root in (REPO_ROOT / "site/content", REPO_ROOT / "site/.generated/content"):
        route_path = content_root / route
        candidates.extend(
            [
                route_path.with_suffix(".md"),
                route_path / "index.md",
                route_path / "_index.md",
                route_path.with_suffix(".html"),
                route_path / "index.html",
            ]
        )
    return candidates


def _is_authored_content(path: Path) -> bool:
    try:
        path.relative_to(SITE_CONTENT_ROOT)
        return True
    except ValueError:
        return False


def _is_root_absolute_policy_violation(source: Path, target: str) -> bool:
    if not _is_authored_content(source):
        return False
    if not BASE_PATH:
        return False
    if not target.startswith("/"):
        return False

    # In authored site content, forbid internal root-absolute links
    # that ignore the configured base path.
    return not (target == BASE_PATH or target.startswith(f"{BASE_PATH}/"))


def _exists_target(source: Path, target: str) -> bool:
    if target.startswith("/"):
        normalized = _strip_base_path(target)
        if normalized is None:
            normalized = target
        return any(candidate.exists() for candidate in _route_candidates(normalized))

    target_path = source.parent / target
    if target_path.exists():
        return True

    if target_path.suffix:
        return False

    candidates = [
        target_path.with_suffix(".md"),
        target_path.with_suffix(".html"),
        target_path / "index.md",
        target_path / "_index.md",
        target_path / "index.html",
    ]
    return any(candidate.exists() for candidate in candidates)


def _collect_targets(line: str, suffix: str) -> list[str]:
    if suffix == ".md":
        targets = [match.group(1) for match in MD_LINK_RE.finditer(line)]
        for match in HTML_LINK_RE.finditer(line):
            targets.append(match.group(1) or match.group(2) or match.group(3) or "")
        return targets

    targets: list[str] = []
    for match in HTML_LINK_RE.finditer(line):
        targets.append(match.group(1) or match.group(2) or match.group(3) or "")
    return targets


def _scan_file(path: Path) -> list[tuple[int, str]]:
    errors: list[tuple[int, str]] = []
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    for lineno, line in enumerate(text.splitlines(), start=1):
        for raw_target in _collect_targets(line, suffix):
            target = _normalize_target(raw_target)
            if not target or target.startswith("#"):
                continue
            if target.startswith(("{{<", "{{%")):
                # Hugo shortcode-generated links (e.g. relref) are resolved at render time.
                continue

            target = _normalize_repo_site_target(target)
            if target.startswith(SKIP_PREFIXES):
                continue

            target_path = target.split("#", 1)[0].split("?", 1)[0]
            if not target_path:
                continue

            if _is_root_absolute_policy_violation(path, target_path):
                errors.append(
                    (
                        lineno,
                        (
                            f"{target} (root-absolute internal links in site/content "
                            f"must include base path '{BASE_PATH}/' or use relref)"
                        ),
                    )
                )
                continue

            if not _exists_target(path, target_path):
                errors.append((lineno, target))

    return errors


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("roots", nargs=-1)
def cli(*, roots: tuple[str, ...]) -> int:
    """Link checker for source docs and generated Hugo content."""

    all_errors: list[tuple[Path, int, str]] = []
    for root_rel in roots or tuple(DEFAULT_ROOTS):
        root = REPO_ROOT / root_rel
        for path in _iter_doc_files(root):
            for lineno, target in _scan_file(path):
                all_errors.append((path, lineno, target))

    if all_errors:
        print("Broken links found:")
        for path, lineno, target in all_errors:
            rel = path.relative_to(REPO_ROOT)
            print(f"- {rel}:{lineno} -> {target}")
        return 1

    print("Link check passed.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    try:
        result = cli.main(
            args=list(argv) if argv is not None else None,
            prog_name="check_links.py",
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
