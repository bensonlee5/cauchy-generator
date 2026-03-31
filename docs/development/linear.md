# Linear Tracker Operations

`dagzoo` uses Linear as its live tracker. This document covers the repo-owned
tracker tooling for migration, workflow-state bootstrap, and recurring repo
audit seeding.

## What lives in this repo

- `scripts/linear/github_to_linear.py`: one-shot GitHub Issues -> Linear
  migration and GitHub cutover tool.
- `scripts/linear/seed_harness_backlog.py`: seeds the harness-engineering epic,
  child tickets, and weekly audit issue body.
- `AGENTS.md`: canonical operating contract for autonomous contributors.
- `docs/development/harness_audit.md`: weekly repo-audit rubric.
- `docs/development/issue_authoring.md`: issue-writing standard for
  implementation-ready work.

## Required environment

- A Linear API key file passed with `--linear-api-key-file`.
- `uv`: used to run the repo scripts in the project environment.
- `gh`: required by the GitHub migration/cutover flow.

## Target tracker

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

The migration tool bootstraps any missing workflow states on the owning team
before importing issues.

## Weekly Repo Audit

The repo-owned weekly audit rubric lives at
[`docs/development/harness_audit.md`](harness_audit.md).

Default recurring audit contract:

- Linear issue title: `ops(harness): weekly full-repo harness audit`
- Schedule: Friday, 10:00 PM `America/Los_Angeles`
- Creation state: `Todo`
- Remediation issues: `Backlog`, label `harness`

Use `scripts/linear/seed_harness_backlog.py` to seed the harness epic, child
tickets, and weekly audit issue body. If Linear recurrence is not available
through the API, configure the final recurrence in the Linear UI after seeding.

Dry-run the seed flow first:

```bash
uv run python scripts/linear/seed_harness_backlog.py \
  --linear-api-key-file ~/.linear/linear_api_key.txt \
  --project-slug 4867d49bb182 \
  --dry-run
```

## Migrating GitHub issues to Linear

Dry-run a subset first:

```bash
uv run python scripts/linear/github_to_linear.py \
  --repo bensonlee5/dagzoo \
  --linear-api-key-file ~/.linear/linear_api_key.txt \
  --project-slug 4867d49bb182 \
  --mapping-path reference/linear_issue_map_2026-03-08.json \
  --issue-number 148 \
  --issue-number 146 \
  --issue-number 175 \
  --dry-run
```

Run the full migration and GitHub cutover:

```bash
uv run python scripts/linear/github_to_linear.py \
  --repo bensonlee5/dagzoo \
  --linear-api-key-file ~/.linear/linear_api_key.txt \
  --project-slug 4867d49bb182 \
  --mapping-path reference/linear_issue_map_2026-03-08.json \
  --mode all
```

## Migration defaults

- All open GitHub issues migrate to Linear `Backlog`.
- All closed GitHub issues migrate to Linear `Done`.
- GitHub labels `P0`, `P1`, and `P2` map to Linear priorities `Urgent`,
  `High`, and `Normal`.
- Non-priority GitHub labels are created or reused as Linear labels.
- GitHub issues labeled `epic` become parent issues when their bodies
  explicitly reference child issue numbers.
- After cutover, GitHub issues are commented with the Linear successor URL and
  remaining open issues are closed.
