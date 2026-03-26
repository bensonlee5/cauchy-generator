"""Recipe catalog command handlers."""

from __future__ import annotations

import argparse

from dagzoo.recipes import iter_recipe_specs


def run_recipe_list_command(_args: argparse.Namespace) -> int:
    """Print the curated public recipe catalog."""

    print("Curated dagzoo recipes")
    print("Use with: dagzoo generate --config recipe:<name> --num-datasets <n> --out <dir>")
    for spec in iter_recipe_specs():
        print(
            f"- {spec.reference}: {spec.summary} "
            f"[confidence={spec.confidence_tier}; regime={spec.expected_regime}]"
        )
    return 0
