from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def publish_plan_module():
    from conftest import load_script_module

    return load_script_module("release_publish_plan", "scripts/release/publish_plan.py")


def test_allowed_successors_include_one_patch_minor_and_major_step(publish_plan_module) -> None:
    previous = publish_plan_module.SemVer.parse("0.9.8")
    allowed = publish_plan_module.allowed_successors(previous)
    assert allowed == {
        publish_plan_module.SemVer.parse("0.9.9"),
        publish_plan_module.SemVer.parse("0.10.0"),
        publish_plan_module.SemVer.parse("1.0.0"),
    }


def test_resolve_publish_decision_allows_next_patch_release(publish_plan_module) -> None:
    decision = publish_plan_module.resolve_publish_decision(
        current_version="0.9.9",
        published_versions={"0.9.8"},
    )
    assert decision.should_publish is True
    assert decision.reason == "next_release_after_0.9.8"


def test_resolve_publish_decision_allows_next_minor_release(publish_plan_module) -> None:
    decision = publish_plan_module.resolve_publish_decision(
        current_version="0.10.0",
        published_versions={"0.9.9"},
    )
    assert decision.should_publish is True
    assert decision.reason == "next_release_after_0.9.9"


def test_resolve_publish_decision_skips_already_published_version(publish_plan_module) -> None:
    decision = publish_plan_module.resolve_publish_decision(
        current_version="0.9.8",
        published_versions={"0.9.8"},
    )
    assert decision.should_publish is False
    assert decision.reason == "already_published"


def test_resolve_publish_decision_rejects_non_adjacent_version_gap(publish_plan_module) -> None:
    with pytest.raises(ValueError, match="not a valid single-step successor"):
        publish_plan_module.resolve_publish_decision(
            current_version="0.9.11",
            published_versions={"0.9.9"},
        )


def test_resolve_publish_decision_validates_expected_version(publish_plan_module) -> None:
    with pytest.raises(ValueError, match="does not match expected version 0.10.0"):
        publish_plan_module.resolve_publish_decision(
            current_version="0.10.1",
            published_versions={"0.10.0"},
            expected_version="0.10.0",
        )


def test_resolve_publish_decision_validates_tag_version(publish_plan_module) -> None:
    with pytest.raises(ValueError, match="does not match tag version 0.10.0"):
        publish_plan_module.resolve_publish_decision(
            current_version="0.10.1",
            published_versions={"0.10.0"},
            tag_name="v0.10.0",
        )
