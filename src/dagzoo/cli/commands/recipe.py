"""Recipe catalog command handlers."""

from __future__ import annotations

from dagzoo.recipes import iter_recipe_specs


def run_recipe_list_command() -> int:
    """Print the curated public recipe catalog."""

    print("Curated dagzoo recipes")
    print("Use with: dagzoo generate --config recipe:<name> --num-datasets <n> --out <dir>")
    for spec in iter_recipe_specs():
        print(
            f"- {spec.reference}: {spec.summary} "
            f"[category={spec.category}; confidence={spec.confidence_tier}; "
            f"regime={spec.expected_regime}]"
        )
    return 0
