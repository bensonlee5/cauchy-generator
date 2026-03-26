from __future__ import annotations

from pathlib import Path

import pytest

from dagzoo.recipes import (
    iter_recipe_specs,
    load_config_reference,
    serialize_config_reference,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_recipe_catalog_entries_have_required_metadata() -> None:
    specs = iter_recipe_specs()

    assert len(specs) == 5
    for spec in specs:
        assert spec.confidence_tier
        assert spec.expected_regime
        assert spec.citations
        assert (REPO_ROOT / spec.repo_path).exists()


def test_load_config_reference_supports_recipe_reference_and_repo_yaml() -> None:
    from_recipe = load_config_reference("recipe:default-baseline")
    from_path = load_config_reference(REPO_ROOT / "recipes" / "default-baseline.yaml")

    assert from_recipe.to_dict() == from_path.to_dict()


def test_load_config_reference_rejects_unknown_recipe() -> None:
    with pytest.raises(ValueError, match="dagzoo recipe list"):
        load_config_reference("recipe:not-a-recipe")


def test_serialize_config_reference_keeps_recipe_literal() -> None:
    assert serialize_config_reference("recipe:default-baseline") == "recipe:default-baseline"
