# Issue Authoring Standard

This document defines how `dagzoo` issues should be written so implementation
work can proceed without inventing missing scope.

## Goals

- Keep issues small enough for one coherent implementation loop.
- Make acceptance criteria and validation non-optional.
- Reduce ambiguity before code edits begin.

## Required Sections

Every implementation-ready issue should include:

- `Summary`
  - the requested behavior or refactor in one short paragraph
- `Why`
  - the concrete problem or risk being addressed
- `Scope`
  - what is in and out of scope
- `Acceptance Criteria`
  - objective pass/fail conditions
- `Validation`
  - exact commands or checks expected before handoff

Optional but recommended:

- `Non-goals`
- `Risks`
- `Follow-up work`

## Sizing Rules

- Prefer one issue per coherent behavior change or refactor seam.
- Split work when one ticket would require multiple unrelated validation loops.
- Use an epic only when the work truly needs coordination across multiple child
  tickets.

## Validation Rules

- If the change affects CLI, persisted metadata, or dataset artifacts, the issue
  must say so explicitly.
- If the change is user-facing, the issue must include docs-update
  expectations.
- If the change is primarily internal, the issue must still name the required
  tests or checks.

## Tracker Conventions

- `Backlog`: not yet active
- `Todo`: ready for implementation
- `In Progress`: actively being implemented
- `Human Review`: waiting on a human decision
- `Rework`: changes requested
- `Merging`: approved and ready to land
- `Done`: merged or intentionally completed

## Anti-Patterns

Do not open tickets that are:

- vague requests without a concrete outcome
- multiple unrelated changes bundled together
- missing validation
- missing acceptance criteria
- pure reminders with no action shape
