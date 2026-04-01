# Contributing

`dagzoo` optimizes for a clean public surface, reproducible generation, and fast
iteration. Contributions should make the user-facing story clearer, not just
add more knobs.

## Development setup

```bash
./scripts/dev bootstrap
source .venv/bin/activate
.venv/bin/nox -s quick
```

Use `.venv/` for commands and tests in this repo.

## Public surface expectations

- Treat `recipe:<name>` references and documented artifact contracts as the
  stable adoption layer.
- Keep curated recipes under `recipes/` discoverable, documented, and packaged.
- Keep advanced/internal authoring under `configs/` available without promoting
  it as the default first-touch surface.
- If a CLI flag, persisted metadata field, or artifact schema changes, call it
  out explicitly as a user-facing change.
- If the export contract changes, update `reference/export_contract_inventory.yaml`,
  `docs/output-format.md`, `docs/export-contract-fields.md`, and the export-contract
  tests in the same PR.

## Docs boundaries

- `README.md` and the public docs under `docs/` should stay user-facing.
- `AGENTS.md` is the canonical operating contract for autonomous contributors.
- Internal maintainer workflows, tracker operations, and deeper design notes
  belong in `docs/development/`.

## Verification

Canonical local verification:

```bash
.venv/bin/nox -s quick
```

Useful follow-ons:

```bash
./scripts/dev impact
.venv/bin/nox -s docs
.venv/bin/nox -s bench_smoke
.venv/bin/nox -s full
```

Before review, compare your branch with `main` and confirm that intended
changes are present and unintended changes are not.

## Versioning and changelog

- Changes under `src/dagzoo` usually require a version bump in `pyproject.toml`.
- Update `CHANGELOG.md` in the same change.
- If you update `docs/development/roadmap.md`, also keep the linked GitHub
  issues and roadmap section references aligned.

## Docs and recipe parity

Curated recipes, packaged resources, README examples, and docs entrypoints are
kept in sync by repo checks. If you add or rename a recipe, update:

- `recipes/`
- `src/dagzoo/recipes/catalog.py`
- `docs/reference-packs.md`
- `recipes/README.md`
- any impacted README and site entrypoints
