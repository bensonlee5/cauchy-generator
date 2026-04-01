#!/usr/bin/env python3
"""Render dagzoo's export-contract field catalog from the checked-in inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO_ROOT / "reference" / "export_contract_inventory.yaml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "docs" / "export-contract-fields.md"


def _normalize_inventory_path(value: object) -> str:
    path = str(value)
    if len(path) >= 2 and path[0] == "'" and path[-1] == "'":
        return path[1:-1]
    return path


def _load_inventory() -> dict[str, object]:
    payload = yaml.safe_load(INVENTORY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{INVENTORY_PATH} must contain a mapping.")
    return payload


def render_markdown() -> str:
    payload = _load_inventory()
    schema = payload.get("schema")
    artifacts = payload.get("artifacts")
    entries = payload.get("entries")
    if not isinstance(schema, str):
        raise ValueError("Inventory schema must be a string.")
    if not isinstance(artifacts, dict):
        raise ValueError("Inventory artifacts must be a mapping.")
    if not isinstance(entries, list):
        raise ValueError("Inventory entries must be a list.")

    grouped: dict[str, list[dict[str, str]]] = {artifact: [] for artifact in artifacts}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("Inventory entry must be a mapping.")
        artifact = raw_entry.get("artifact")
        if not isinstance(artifact, str) or artifact not in grouped:
            raise ValueError(f"Inventory entry references unknown artifact: {artifact!r}")
        normalized = {key: str(value) for key, value in raw_entry.items()}
        normalized["path"] = _normalize_inventory_path(raw_entry.get("path"))
        grouped[artifact].append(normalized)

    lines = [
        "# Export Contract Fields",
        "",
        f"- Inventory schema: `{schema}`",
        f"- Machine-readable source of truth: `{INVENTORY_PATH.relative_to(REPO_ROOT)}`",
        "- Path patterns use `*` for dynamic map keys and `[]` for list item shapes.",
        "- `audit_status` is a field-review classification only; it does not change the live export surface.",
        "",
    ]

    for artifact, meta in artifacts.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Artifact metadata for {artifact!r} must be a mapping.")
        label = meta.get("label", artifact)
        description = meta.get("description", "")
        lines.extend([f"## {label}", "", str(description), ""])
        lines.append(
            "| Path | Type | Presence | Stability | Producer | Audit | Known Consumer / Rationale |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for entry in sorted(grouped[artifact], key=lambda item: item["path"]):
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{entry['path']}`",
                        f"`{entry['type']}`",
                        entry["presence"],
                        f"`{entry['stability_tier']}`",
                        f"`{entry['producer']}`",
                        f"`{entry['audit_status']}`",
                        entry["rationale"],
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    rendered = render_markdown()
    if args.check:
        existing = args.output.read_text(encoding="utf-8")
        if existing != rendered:
            print(
                f"{args.output} is out of date with {INVENTORY_PATH.relative_to(REPO_ROOT)}.",
                file=sys.stderr,
            )
            return 1
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
