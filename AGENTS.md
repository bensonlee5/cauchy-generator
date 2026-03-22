# Development Patterns

## Environment

- Use `.venv/` for commands and tests in this repo.

## Architecture and Code Organization

- Prefer breaking dependency cycles and centralizing shared wiring in `src/dagzoo/core`; avoid "legacy" pathways, duplicate pathways, and shims.
- Do not introduce parallel implementations of the same logic in different layers of the codebase.
- We prefer shared utility packages over hand-rolled helpers to keep invariants centralized.

## Compatibility and Change Scope

- We optimize for iteration speed: internal Python APIs and internal config structure may change without backward-compat guarantees.
- If CLI flags, persisted metadata schema, or dataset artifact contract changes, treat it as a user-facing break and call it out explicitly.

## Data Handling

- We don’t probe data “YOLO-style”—we validate boundaries or rely on typed SDKs.

## Verification and Review

- Canonical local verification is `./scripts/dev verify quick`; use `./scripts/dev impact` when you need a dependency-aware ripple check before broader refactors.
- Prior to declaring a branch ready for review, compare branch to main and verify that all intended changes are included and no unintended changes are included.

## Versioning and Changelog

- For behavior/schema changes under `src/dagzoo`, bump version in `pyproject.toml` just before merging into main so that the version reflects the latest changes (patch by default; minor for intentionally broad user-facing breaks). Docs/tests-only changes do not require a bump.
- On every version bump, update `CHANGELOG.md` in the same PR.
