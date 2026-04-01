# Linear Tracker Operations

`dagzoo` uses Linear as its live tracker. This document covers the current
tracker contract and repo-owned guidance that contributors should follow while
working against the project.

The historical GitHub-to-Linear migration and bootstrap scripts were removed
after the cutover completed. Tracker work now happens directly in Linear rather
than through repo-owned one-shot tooling.

## Canonical Tracker State

- Linear project URL:
  `https://linear.app/bl-personal/project/dagzoo-4867d49bb182/overview`
- Linear project slug ID: `4867d49bb182`
- Linear team key: `BL`

Canonical workflow states for this repo:

- `Backlog`
- `Todo`
- `In Progress`
- `Human Review`
- `Rework`
- `Merging`
- `Done`

## Issue Hygiene

Implementation-ready issues should follow
[`docs/development/issue_authoring.md`](issue_authoring.md) and include:

- `Summary`
- `Why`
- `Scope`
- `Acceptance Criteria`
- `Validation`

Additional repo expectations:

- Keep one coherent behavior change or refactor seam per issue when possible.
- Call out user-facing changes explicitly, including docs-update expectations.
- Keep roadmap references in
  [`docs/development/roadmap.md`](roadmap.md) aligned with the active tracker.

## Weekly Repo Audit

The repo-owned weekly audit rubric lives at
[`docs/development/harness_audit.md`](harness_audit.md).

Default recurring audit contract:

- Linear issue title: `ops(harness): weekly full-repo harness audit`
- Schedule: Friday, 10:00 PM `America/Los_Angeles`
- Creation state: `Todo`
- Remediation issues: `Backlog`, label `harness`

When the weekly audit finds a new gap:

1. Search the current Linear project for an open issue covering the same work.
2. Reuse that issue if it already exists.
3. Otherwise create a new remediation issue that references the audit, starts
   in `Backlog`, and includes acceptance criteria plus validation.

## Related Docs

- `AGENTS.md`: canonical operating contract for autonomous contributors.
- [`docs/development/harness_audit.md`](harness_audit.md): weekly repo-audit
  rubric.
- [`docs/development/issue_authoring.md`](issue_authoring.md): issue-writing
  standard for implementation-ready work.
- [`docs/development/roadmap.md`](roadmap.md): canonical planning state and
  tracker-link inventory.
