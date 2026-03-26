#!/usr/bin/env python3
"""Sync docs sources into the Hugo docs site.

This script keeps canonical docs in `docs/` and renders generated site inputs
under `site/.generated/`:

- Markdown user guides -> `site/.generated/content/docs/**`
- Hugo-rendered reference docs -> `site/.generated/content/docs/**`
- Development links page -> `site/.generated/content/docs/development.md`

Use `--check` in CI to fail when generated site inputs are out of date.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
SITE_ROOT = REPO_ROOT / "site"
GENERATED_ROOT = SITE_ROOT / ".generated"
CONTENT_DOCS_ROOT = GENERATED_ROOT / "content" / "docs"
LEGACY_GENERATED_PATHS = [
    GENERATED_ROOT / "static" / "canonical",
]

GITHUB_DEV_DOCS_BASE = "https://github.com/bensonlee5/dagzoo/blob/main/docs/development"

USER_MD_SOURCES = [
    "start.md",
    "reference-packs.md",
    "how-it-works.md",
    "transforms.md",
    "usage-guide.md",
    "output-format.md",
    "features/diagnostics.md",
    "features/missingness.md",
    "features/many-class.md",
    "features/shift.md",
    "features/steering.md",
    "features/noise.md",
    "features/benchmark-guardrails.md",
    "features/mechanism-diversity.md",
]


@dataclass(frozen=True, slots=True)
class PageMeta:
    weight: int
    description: str | None = None
    aliases: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)


PAGE_METADATA: dict[str, PageMeta] = {
    "start.md": PageMeta(
        weight=10,
        description="Install dagzoo, inspect curated recipes, and generate the first reproducible run.",
    ),
    "reference-packs.md": PageMeta(
        weight=20,
        description="Published dagzoo recipe packs, confidence tiers, and citation-oriented usage.",
    ),
    "how-it-works.md": PageMeta(
        weight=30,
        description="End-to-end runtime behavior, core concepts, and pipeline walkthrough.",
        aliases=("/canonical/how-it-works.html",),
        params={"mermaid": True},
    ),
    "transforms.md": PageMeta(
        weight=40,
        description="Mathematical reference for generation transforms.",
        aliases=("/canonical/transforms.html",),
        params={"math": True},
    ),
    "usage-guide.md": PageMeta(
        weight=50,
        description="Advanced controls and custom YAML workflows beyond the curated recipe catalog.",
    ),
    "output-format.md": PageMeta(
        weight=60,
        description="Artifact schema, API contract, and handoff metadata layout.",
    ),
    "features/diagnostics.md": PageMeta(
        weight=60,
        description="Runtime observability metrics and diagnostic outputs.",
    ),
    "features/missingness.md": PageMeta(
        weight=61,
        description="Controlled injection of missing values into generated datasets.",
    ),
    "features/many-class.md": PageMeta(
        weight=62,
        description="Multi-class target generation with configurable class counts.",
    ),
    "features/shift.md": PageMeta(
        weight=63,
        description="Distribution-shift controls for graph, mechanism, and noise.",
    ),
    "features/steering.md": PageMeta(
        weight=64,
        description="Meta-feature steering presets for auditable harder-front workflows.",
    ),
    "features/noise.md": PageMeta(
        weight=65,
        description="Noise family selection, mixture modes, and per-dataset resolution.",
    ),
    "features/benchmark-guardrails.md": PageMeta(
        weight=66,
        description="Automated quality checks for benchmark suite runs.",
    ),
    "features/mechanism-diversity.md": PageMeta(
        weight=67,
        description="Opt-in mechanism-family rollout workflows and diagnostics checks.",
    ),
}

# Standard Markdown inline-link pattern: [label](target)
MD_LINK_RE = re.compile(r"(!?\[[^\]]+\]\()([^\)]+)(\))")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_base_path() -> str:
    """Extract URL path prefix from Hugo baseURL config."""
    text = _read_text(SITE_ROOT / "hugo.yaml")
    match = re.search(r"^baseURL:\s*(.+)$", text, re.MULTILINE)
    if not match:
        return ""
    return urlparse(match.group(1).strip()).path.rstrip("/")


def _title_from_markdown(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            title = re.sub(r"\s+\{#.*\}\s*$", "", title)
            return title
    return fallback


def _strip_matching_h1(content: str, title: str) -> str:
    lines = content.splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    if idx < len(lines) and lines[idx].strip().startswith("# "):
        heading = lines[idx].strip()[2:].strip()
        heading = re.sub(r"\s+\{#.*\}\s*$", "", heading)
        if heading.lower() == title.lower():
            del lines[idx]
            if idx < len(lines) and not lines[idx].strip():
                del lines[idx]

    stripped = "\n".join(lines).rstrip()
    return f"{stripped}\n" if stripped else ""


def _slug_title(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def _front_matter(
    title: str,
    meta: PageMeta,
    *,
    extra_params: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "title": title,
        "weight": meta.weight,
    }
    if meta.description:
        payload["description"] = meta.description
    if meta.aliases:
        payload["aliases"] = list(meta.aliases)
    payload.update(meta.params)
    if extra_params:
        payload.update(extra_params)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False).strip()
    return f"---\n{text}\n---\n\n"


def _normalize_link(source_rel: str, target: str) -> str:
    source_dir = posixpath.dirname(source_rel)
    joined = posixpath.join(source_dir, target)
    return posixpath.normpath(joined)


def _rewrite_markdown_links(content: str, source_rel: str, route_map: dict[str, str]) -> str:
    def replacer(match: re.Match[str]) -> str:
        prefix, raw_target, suffix = match.groups()
        target = raw_target.strip()

        if not target:
            return match.group(0)

        if target.startswith(("http://", "https://", "mailto:", "tel:", "#", "/")):
            return match.group(0)

        path, anchor = (target.split("#", 1) + [""])[:2]
        normalized = _normalize_link(source_rel, path)

        new_target = route_map.get(normalized)
        if new_target is None and normalized.startswith("development/"):
            new_target = f"{GITHUB_DEV_DOCS_BASE}/{normalized.removeprefix('development/')}"

        if new_target is None:
            return match.group(0)

        if anchor:
            new_target = f"{new_target}#{anchor}"

        return f"{prefix}{new_target}{suffix}"

    return MD_LINK_RE.sub(replacer, content)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sync_text(path: Path, expected: str, check: bool, changed: list[Path]) -> None:
    _ensure_parent(path)
    current = _read_text(path) if path.exists() else None
    if current == expected:
        return
    changed.append(path)
    if not check:
        path.write_text(expected, encoding="utf-8")


def _remove_stale_path(path: Path, check: bool, changed: list[Path]) -> None:
    if not path.exists():
        return
    changed.append(path)
    if check:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _build_route_map(base_path: str) -> dict[str, str]:
    route_map: dict[str, str] = {
        "how-it-works.html": f"{base_path}/docs/how-it-works/",
        "transforms.html": f"{base_path}/docs/transforms/",
    }

    for rel in USER_MD_SOURCES:
        slug = Path(rel).stem
        if rel.startswith("features/"):
            route_map[rel] = f"{base_path}/docs/features/{slug}/"
        else:
            route_map[rel] = f"{base_path}/docs/{slug}/"

    return route_map


def _sync_user_markdown(route_map: dict[str, str], check: bool, changed: list[Path]) -> None:
    for rel in USER_MD_SOURCES:
        src = DOCS_ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing source Markdown file: {src}")

        source_text = _read_text(src)
        rewritten = _rewrite_markdown_links(source_text, rel, route_map)

        title = _title_from_markdown(rewritten, fallback=_slug_title(Path(rel).stem))
        rewritten = _strip_matching_h1(rewritten, title=title)
        meta = PAGE_METADATA.get(rel, PageMeta(weight=100))
        out_text = (
            _front_matter(
                title=title,
                meta=meta,
                extra_params={"canonical_repo_path": f"docs/{rel}"},
            )
            + rewritten
        )

        dest = CONTENT_DOCS_ROOT / rel
        _sync_text(dest, out_text, check=check, changed=changed)


def _humanize_dev_name(filename: str) -> str:
    return filename.replace("_", " ").replace("-", " ").replace(".md", "").title()


def _sync_development_links_page(check: bool, changed: list[Path]) -> None:
    dev_dir = DOCS_ROOT / "development"
    dev_docs = sorted(p.name for p in dev_dir.glob("*.md"))
    if not dev_docs:
        raise FileNotFoundError(f"No development docs found in: {dev_dir}")

    lines: list[str] = [
        _front_matter(
            title="Development References",
            meta=PageMeta(weight=90),
            extra_params={"hide_repo_links": True},
        ),
        "Phase 1 keeps development docs canonical in repo Markdown and links them directly from the docs site.\n",
        "",
    ]
    for filename in dev_docs:
        label = _humanize_dev_name(filename)
        url = f"{GITHUB_DEV_DOCS_BASE}/{filename}"
        lines.append(f"- [{label}]({url})")

    lines.append("")
    text = "\n".join(lines)
    _sync_text(CONTENT_DOCS_ROOT / "development.md", text, check=check, changed=changed)


def sync(check: bool) -> list[Path]:
    changed: list[Path] = []
    base_path = _read_base_path()
    route_map = _build_route_map(base_path)

    _sync_user_markdown(route_map=route_map, check=check, changed=changed)
    _sync_development_links_page(check=check, changed=changed)
    for path in LEGACY_GENERATED_PATHS:
        _remove_stale_path(path, check=check, changed=changed)

    return changed


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write files; fail if outputs are out of sync.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    changed = sync(check=args.check)

    if args.check:
        if changed:
            print("Docs sync is out of date. Regenerate with:")
            print("  .venv/bin/python scripts/docs/sync_hugo_content.py")
            print("\nFiles needing update:")
            for path in changed:
                print(f"- {path.relative_to(REPO_ROOT)}")
            return 1
        print("Docs sync check passed.")
        return 0

    if changed:
        print(f"Updated {len(changed)} file(s):")
        for path in changed:
            print(f"- {path.relative_to(REPO_ROOT)}")
    else:
        print("No updates needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
