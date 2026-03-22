from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_PYPROJECT_VERSION_RE = re.compile(r'^version = "([^"]+)"$', re.MULTILINE)


@dataclass(frozen=True, order=True)
class SemVer:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> "SemVer":
        match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", str(value).strip())
        if match is None:
            raise ValueError(f"Unsupported version format: {value!r}")
        return cls(*(int(part) for part in match.groups()))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def read_pyproject_version_text(text: str, *, source: str) -> str:
    match = _PYPROJECT_VERSION_RE.search(text)
    if match is None:
        raise ValueError(f"Could not parse version from {source!r}")
    return match.group(1)


def read_pyproject_version(pyproject_path: str) -> str:
    return read_pyproject_version_text(
        Path(pyproject_path).read_text(encoding="utf-8"),
        source=pyproject_path,
    )


def allowed_release_successors(previous: SemVer) -> set[SemVer]:
    return {
        SemVer(previous.major, previous.minor, previous.patch + 1),
        SemVer(previous.major, previous.minor + 1, 0),
        SemVer(previous.major + 1, 0, 0),
    }


def allowed_pr_successors(previous: SemVer) -> set[SemVer]:
    return {
        SemVer(previous.major, previous.minor, previous.patch + 1),
        SemVer(previous.major, previous.minor + 1, 0),
    }
