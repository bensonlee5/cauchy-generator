# AGENTS

`AGENTS.md` is the canonical operating contract for autonomous contributors in
this repo. Keep agent-only workflow guidance here or in internal maintainer
docs under `docs/development/`. Keep `README.md` and the user docs focused on
how to install, run, and understand `dagzoo`.

## Working Surface

- Use `.venv/` for commands and tests in this repo.
- Treat `recipe:<name>`, documented CLI/API behavior, and artifact contracts as
  the stable public surface.
- Repo-local `configs/`, internal Python APIs, and internal metadata details
  may change faster than the named recipe surface.
- If a change affects CLI flags, persisted metadata, or dataset artifact
  contracts, treat it as user-facing and call it out explicitly.

## Bootstrap And Verification

Bootstrap a fresh checkout with:

```bash
./scripts/dev bootstrap
source .venv/bin/activate
```

Canonical verification commands:

- `.venv/bin/nox -s quick`
- `.venv/bin/nox -s docs`
- `.venv/bin/nox -s full`
- `.venv/bin/nox -s bench_smoke`
- `./scripts/dev impact` for dependency-aware ripple checks before broader
  refactors

Before declaring work ready for review:

- compare the branch against `main`
- confirm all intended changes are present
- confirm unrelated changes are not included

## Architecture And Code Organization

- Prefer breaking dependency cycles and centralizing shared wiring in
  `src/dagzoo/core`.
- Do not introduce long-lived "legacy" pathways, duplicate implementations, or
  compatibility shims without an explicit reason.
- Prefer shared utility packages over hand-rolled helpers so invariants stay
  centralized.
- Validate data boundaries; do not probe data "YOLO-style" when typed or
  validated interfaces are available.
- Preserve one canonical execution path whenever possible rather than building
  parallel flows in different layers of the codebase.

## Docs Boundary

- `README.md` and the public docs under `docs/` should stay user-facing.
- Research framing, internal design rationale, tracker procedures, and
  automation runbooks belong in `AGENTS.md` or `docs/development/`.
- When a workflow change affects both humans and autonomous contributors, keep
  `README.md`, `CONTRIBUTING.md`, public docs, and `AGENTS.md` aligned.
- Do not surface maintainer-only runbooks from public doc entrypoints.

## Change Classification

- Internal Python APIs and internal config structure can change without
  backward-compat guarantees.
- Behavior or schema changes under `src/dagzoo` usually require a version bump
  in `pyproject.toml` just before merging to `main`.
- Patch bumps are the default; use a minor bump for intentionally broad
  user-facing breaks.
- Docs-only and tests-only changes do not require a version bump.
- Every version bump must update `CHANGELOG.md` in the same PR.

## Tracker And Issue Hygiene

- Implementation-ready issues should include `Summary`, `Why`, `Scope`,
  `Acceptance Criteria`, and `Validation`.
- Keep one coherent behavior change or refactor seam per issue when possible.
- If a change is user-facing, the issue should say so and include docs-update
  expectations.
- If `docs/development/roadmap.md` changes, update the linked GitHub issues and
  keep roadmap section references aligned in both directions.
- If your response would close a GitHub issue, say so explicitly and reference
  the issue number.

Canonical tracker states:

- `Backlog`
- `Todo`
- `In Progress`
- `Human Review`
- `Rework`
- `Merging`
- `Done`

Detailed tracker operations live in `docs/development/linear.md`.

## Weekly Repo Audit

The recurring repo audit should verify:

- `README.md`, `AGENTS.md`, and active docs still describe the real workflow
- public vs internal surfaces are clearly separated
- canonical bootstrap and verification commands still work and are discoverable
- architecture guidance still matches the actual module graph
- stale docs, scripts, or local-output assumptions are removed or tracked
- tracker remediation work is deduplicated and written with clear acceptance
  criteria
