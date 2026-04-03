#!/usr/bin/env python3
"""Resolve whether one package version should publish to PyPI."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import click

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from devlib.semver import SemVer, allowed_release_successors, read_pyproject_version  # noqa: E402

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@dataclass(frozen=True)
class PublishDecision:
    should_publish: bool
    version: str
    reason: str


def parse_tag_version(tag_name: str) -> str:
    stripped = str(tag_name).strip()
    if stripped.startswith("refs/tags/"):
        stripped = stripped[len("refs/tags/") :]
    if not stripped.startswith("v"):
        raise ValueError(f"Tag name must start with 'v', got {tag_name!r}")
    return stripped[1:]


def allowed_successors(previous: SemVer) -> set[SemVer]:
    return allowed_release_successors(previous)


def fetch_published_versions(package_name: str) -> set[str]:
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return set()
        raise
    releases = payload.get("releases", {})
    if not isinstance(releases, dict):
        raise ValueError("Invalid PyPI response: releases must be a mapping.")
    versions: set[str] = set()
    for key in releases:
        try:
            SemVer.parse(str(key))
        except ValueError:
            continue
        versions.add(str(key))
    return versions


def resolve_publish_decision(
    *,
    current_version: str,
    published_versions: set[str],
    expected_version: str | None = None,
    tag_name: str | None = None,
) -> PublishDecision:
    current = SemVer.parse(current_version)
    if expected_version is not None and expected_version.strip():
        expected = SemVer.parse(expected_version)
        if current != expected:
            raise ValueError(
                f"pyproject.toml version {current} does not match expected version {expected}."
            )
    if tag_name is not None and tag_name.strip():
        tag_version = SemVer.parse(parse_tag_version(tag_name))
        if current != tag_version:
            raise ValueError(
                f"pyproject.toml version {current} does not match tag version {tag_version}."
            )

    parsed_published = {SemVer.parse(version) for version in published_versions}
    if current in parsed_published:
        return PublishDecision(
            should_publish=False,
            version=str(current),
            reason="already_published",
        )

    if not parsed_published:
        return PublishDecision(
            should_publish=True,
            version=str(current),
            reason="first_release",
        )

    previous = max(parsed_published)
    allowed = allowed_release_successors(previous)
    if current not in allowed:
        allowed_list = ", ".join(str(version) for version in sorted(allowed))
        raise ValueError(
            f"Version {current} is not a valid single-step successor of published version "
            f"{previous}. Allowed next versions: {allowed_list}."
        )

    return PublishDecision(
        should_publish=True,
        version=str(current),
        reason=f"next_release_after_{previous}",
    )


def _write_github_outputs(decision: PublishDecision) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    tag_name = f"v{decision.version}"
    with Path(github_output).open("a", encoding="utf-8") as handle:
        handle.write(f"should_publish={'true' if decision.should_publish else 'false'}\n")
        handle.write(f"version={decision.version}\n")
        handle.write(f"reason={decision.reason}\n")
        handle.write(f"tag_name={tag_name}\n")


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--package-name", required=True)
@click.option("--pyproject", default="pyproject.toml", show_default=True)
@click.option("--expected-version", default=None)
@click.option("--tag-name", default=None)
def cli(
    *,
    package_name: str,
    pyproject: str,
    expected_version: str | None,
    tag_name: str | None,
) -> int:
    """Resolve whether one package version should publish to PyPI."""

    current_version = read_pyproject_version(pyproject)
    published_versions = fetch_published_versions(package_name)
    decision = resolve_publish_decision(
        current_version=current_version,
        published_versions=published_versions,
        expected_version=expected_version,
        tag_name=tag_name,
    )
    _write_github_outputs(decision)
    json.dump(
        {
            "should_publish": decision.should_publish,
            "version": decision.version,
            "reason": decision.reason,
        },
        sys.stdout,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        result = cli.main(args=argv, prog_name="publish_plan.py", standalone_mode=False)
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
