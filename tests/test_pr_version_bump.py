from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def pr_version_module():
    from conftest import load_script_module

    return load_script_module("pr_version_bump", "scripts/ci/check_pr_version_bump.py")


def test_pr_version_checker_allows_unchanged_version(pr_version_module) -> None:
    decision = pr_version_module.resolve_pr_version_decision(
        head_version="0.10.3",
        base_version="0.10.3",
    )
    assert decision.ok is True
    assert decision.reason == "unchanged"


def test_pr_version_checker_allows_next_patch(pr_version_module) -> None:
    decision = pr_version_module.resolve_pr_version_decision(
        head_version="0.10.4",
        base_version="0.10.3",
    )
    assert decision.ok is True
    assert decision.reason == "next_patch"


def test_pr_version_checker_allows_next_minor(pr_version_module) -> None:
    decision = pr_version_module.resolve_pr_version_decision(
        head_version="0.11.0",
        base_version="0.10.3",
    )
    assert decision.ok is True
    assert decision.reason == "next_minor"


@pytest.mark.parametrize("head_version", ["1.0.0", "0.10.5", "0.12.0", "0.10.2"])
def test_pr_version_checker_rejects_invalid_bumps(
    pr_version_module,
    head_version: str,
) -> None:
    with pytest.raises(ValueError, match="must be unchanged or one allowed step"):
        pr_version_module.resolve_pr_version_decision(
            head_version=head_version,
            base_version="0.10.3",
        )
