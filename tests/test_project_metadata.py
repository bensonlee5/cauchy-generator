from __future__ import annotations

import pathlib
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_project_license_metadata_includes_notice_files() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["license"] == "Apache-2.0"
    assert set(project["license-files"]) >= {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"}
    assert (REPO_ROOT / "NOTICE").exists()
    assert (REPO_ROOT / "THIRD_PARTY_NOTICES.md").exists()
