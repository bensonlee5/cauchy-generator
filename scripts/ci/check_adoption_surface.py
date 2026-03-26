#!/usr/bin/env python3
"""Validate that dagzoo's public adoption surface stays coherent."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from devlib.common import python_tool  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dagzoo.recipes import iter_recipe_specs  # noqa: E402

REQUIRED_FILES = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "start.md",
    REPO_ROOT / "docs" / "reference-packs.md",
    REPO_ROOT / "docs" / "usage-guide.md",
    REPO_ROOT / "docs" / "output-format.md",
    REPO_ROOT / "recipes" / "README.md",
    REPO_ROOT / "CITATION.cff",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "SECURITY.md",
)
REQUIRED_README_SNIPPETS = (
    "dagzoo recipe list",
    "recipe:default-baseline",
    "build_dataloader(",
)
REQUIRED_START_SNIPPETS = (
    "dagzoo recipe list",
    "recipe:default-baseline",
    "recipe:tabpfn-v1-prior-approx",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_version_from_pyproject() -> str:
    text = _read_text(REPO_ROOT / "pyproject.toml")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    if match is None:
        raise ValueError("Could not find version in pyproject.toml.")
    return match.group(1)


def _extract_version_from_changelog() -> str:
    text = _read_text(REPO_ROOT / "CHANGELOG.md")
    match = re.search(r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\]", text, re.MULTILINE)
    if match is None:
        raise ValueError("Could not find top release heading in CHANGELOG.md.")
    return match.group(1)


def _extract_version_from_citation() -> str:
    text = _read_text(REPO_ROOT / "CITATION.cff")
    match = re.search(r"^version:\s*([0-9]+\.[0-9]+\.[0-9]+)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError("Could not find version in CITATION.cff.")
    return match.group(1)


def _require(condition: bool, message: str, *, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _build_wheel(out_dir: Path) -> Path:
    uv = python_tool("uv")
    result = subprocess.run(
        (uv, "build", "--wheel", "--out-dir", str(out_dir)),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "uv build failed"
        raise RuntimeError(detail)
    wheels = sorted(out_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one wheel in {out_dir}, found {len(wheels)}.")
    return wheels[0]


def main() -> int:
    errors: list[str] = []

    for path in REQUIRED_FILES:
        _require(path.exists(), f"Missing required public-surface file: {path.name}", errors=errors)

    readme_path = REPO_ROOT / "README.md"
    readme_text = _read_text(readme_path) if readme_path.exists() else ""
    for snippet in REQUIRED_README_SNIPPETS:
        _require(snippet in readme_text, f"README.md must mention `{snippet}`.", errors=errors)

    start_path = REPO_ROOT / "docs" / "start.md"
    start_text = _read_text(start_path) if start_path.exists() else ""
    for snippet in REQUIRED_START_SNIPPETS:
        _require(snippet in start_text, f"docs/start.md must mention `{snippet}`.", errors=errors)

    reference_packs_path = REPO_ROOT / "docs" / "reference-packs.md"
    recipes_readme_path = REPO_ROOT / "recipes" / "README.md"
    reference_packs_text = _read_text(reference_packs_path) if reference_packs_path.exists() else ""
    recipes_readme_text = _read_text(recipes_readme_path) if recipes_readme_path.exists() else ""
    for spec in iter_recipe_specs():
        _require(
            bool(spec.confidence_tier.strip()),
            f"Recipe `{spec.name}` is missing a confidence tier.",
            errors=errors,
        )
        _require(
            bool(spec.expected_regime.strip()),
            f"Recipe `{spec.name}` is missing an expected-regime description.",
            errors=errors,
        )
        _require(
            bool(spec.citations),
            f"Recipe `{spec.name}` is missing citations.",
            errors=errors,
        )
        _require(
            (REPO_ROOT / spec.repo_path).exists(),
            f"Recipe YAML is missing for `{spec.name}` at `{spec.repo_path}`.",
            errors=errors,
        )
        _require(
            spec.name in reference_packs_text,
            f"docs/reference-packs.md must mention `{spec.name}`.",
            errors=errors,
        )
        _require(
            spec.name in recipes_readme_text,
            f"recipes/README.md must mention `{spec.name}`.",
            errors=errors,
        )

    pyproject_version = _extract_version_from_pyproject()
    changelog_version = _extract_version_from_changelog()
    citation_version = _extract_version_from_citation()
    _require(
        pyproject_version == changelog_version,
        "pyproject.toml and CHANGELOG.md versions must match.",
        errors=errors,
    )
    _require(
        pyproject_version == citation_version,
        "pyproject.toml and CITATION.cff versions must match.",
        errors=errors,
    )

    with tempfile.TemporaryDirectory(prefix="dagzoo_adoption_surface_") as tmp_dir:
        wheel_path = _build_wheel(Path(tmp_dir))
        with zipfile.ZipFile(wheel_path) as archive:
            names = set(archive.namelist())
        for spec in iter_recipe_specs():
            resource_path = f"dagzoo/recipes/resources/{spec.resource_name}"
            _require(
                resource_path in names,
                f"Wheel is missing packaged recipe resource `{resource_path}`.",
                errors=errors,
            )

    if errors:
        print("Adoption surface check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Adoption surface check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
