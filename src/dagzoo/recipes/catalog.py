"""Shared recipe metadata and config-reference loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

RECIPE_REF_PREFIX = "recipe:"

if TYPE_CHECKING:
    from dagzoo.config import GeneratorConfig


@dataclass(frozen=True, slots=True)
class RecipeSpec:
    """Public metadata for one curated dagzoo recipe."""

    name: str
    title: str
    summary: str
    confidence_tier: str
    category: str
    expected_regime: str
    citations: tuple[str, ...]
    resource_name: str

    @property
    def reference(self) -> str:
        """Return the stable CLI config reference for this recipe."""

        return recipe_reference(self.name)

    @property
    def repo_path(self) -> str:
        """Return the repo-local YAML path for this recipe."""

        return f"recipes/{self.resource_name}"


_RECIPES: dict[str, RecipeSpec] = {
    "default-baseline": RecipeSpec(
        name="default-baseline",
        title="Default Baseline",
        summary=(
            "Balanced mixed-type baseline with the default factorized "
            "feature-prior plus observed-X target-head semantics."
        ),
        confidence_tier="baseline",
        category="reference prior",
        expected_regime="General mixed-type classification with no added stress regime.",
        citations=("Dagzoo packaged baseline recipe.",),
        resource_name="default-baseline.yaml",
    ),
    "tabpfn-v1-prior-approx": RecipeSpec(
        name="tabpfn-v1-prior-approx",
        title="TabPFN v1 Prior Approximation",
        summary=(
            "Paper-backed numeric-heavy classification prior with the default "
            "factorized observed-X target-head semantics."
        ),
        confidence_tier="paper-backed approximation",
        category="reference prior",
        expected_regime="Numeric-only small-table classification with moderate feature and class counts.",
        citations=(
            "Accurate predictions on small data with a tabular foundation model.",
            "TabICLv2: A better, faster, scalable, and open tabular foundation model.",
        ),
        resource_name="tabpfn-v1-prior-approx.yaml",
    ),
    "high-cardinality-stress": RecipeSpec(
        name="high-cardinality-stress",
        title="High Cardinality Stress",
        summary="Categorical-heavy stress pack for regimes with wider cardinality envelopes.",
        confidence_tier="stress profile",
        category="stress pack",
        expected_regime="Feature spaces dominated by categorical columns with larger cardinalities.",
        citations=(
            "Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks.",
        ),
        resource_name="high-cardinality-stress.yaml",
    ),
    "missingness-robustness": RecipeSpec(
        name="missingness-robustness",
        title="Missingness Robustness",
        summary="Missingness-first stress pack for robustness experiments with structured missing values.",
        confidence_tier="stress profile",
        category="stress pack",
        expected_regime="Moderate-to-heavy missingness with explicit MNAR controls.",
        citations=(
            "A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities.",
            "TabICLv2: A better, faster, scalable, and open tabular foundation model.",
        ),
        resource_name="missingness-robustness.yaml",
    ),
    "shift-stress": RecipeSpec(
        name="shift-stress",
        title="Shift Stress",
        summary="Mixed graph-and-noise drift stress pack for controlled train/test shift experiments.",
        confidence_tier="stress profile",
        category="stress pack",
        expected_regime="Controlled mixed drift with graph and variance shifts turned on.",
        citations=(
            "Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data.",
        ),
        resource_name="shift-stress.yaml",
    ),
}


def recipe_reference(name: str) -> str:
    """Return the canonical config reference string for `name`."""

    return f"{RECIPE_REF_PREFIX}{str(name).strip()}"


def supported_recipe_names() -> tuple[str, ...]:
    """Return the supported public recipe names in stable order."""

    return tuple(_RECIPES)


def iter_recipe_specs() -> tuple[RecipeSpec, ...]:
    """Return the public recipe catalog in stable order."""

    return tuple(_RECIPES.values())


def _normalize_recipe_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if not normalized:
        supported = ", ".join(supported_recipe_names())
        raise ValueError(f"Recipe name must be non-empty. Expected one of: {supported}.")
    return normalized


def parse_recipe_reference(reference: str | Path) -> str | None:
    """Return the recipe name for one `recipe:<name>` reference, else `None`."""

    if not isinstance(reference, str):
        return None
    stripped = reference.strip()
    if not stripped.startswith(RECIPE_REF_PREFIX):
        return None
    return _normalize_recipe_name(stripped.removeprefix(RECIPE_REF_PREFIX))


def get_recipe_spec(name: str) -> RecipeSpec:
    """Return one recipe spec by name."""

    normalized = _normalize_recipe_name(name)
    try:
        return _RECIPES[normalized]
    except KeyError as exc:
        supported = ", ".join(supported_recipe_names())
        raise ValueError(
            f"Unsupported recipe {normalized!r}. Expected one of: {supported}. "
            "Use `dagzoo recipe list` to inspect the curated catalog."
        ) from exc


def _repo_recipe_path(resource_name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "recipes" / resource_name


def _read_recipe_text(resource_name: str) -> str:
    repo_path = _repo_recipe_path(resource_name)
    if repo_path.exists():
        return repo_path.read_text(encoding="utf-8")
    resource = files("dagzoo.recipes.resources").joinpath(resource_name)
    return resource.read_text(encoding="utf-8")


def load_recipe_config(name: str) -> GeneratorConfig:
    """Load one curated recipe config into `GeneratorConfig`."""

    from dagzoo.config import GeneratorConfig

    spec = get_recipe_spec(name)
    loaded = yaml.safe_load(_read_recipe_text(spec.resource_name)) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Recipe {spec.reference} must be a mapping at the top level.")
    return GeneratorConfig.from_dict(loaded)


def serialize_config_reference(config: str | Path) -> str:
    """Return one stable config reference string for artifacts and reports."""

    recipe_name = parse_recipe_reference(config)
    if recipe_name is not None:
        return recipe_reference(recipe_name)
    return str(Path(config).resolve())


def load_config_reference(config: str | Path) -> GeneratorConfig:
    """Load one config path or `recipe:<name>` reference."""

    from dagzoo.config import GeneratorConfig

    recipe_name = parse_recipe_reference(config)
    if recipe_name is not None:
        return load_recipe_config(recipe_name)
    return GeneratorConfig.from_yaml(config)
