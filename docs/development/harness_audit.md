# Weekly Repo Audit

This document is the repo-owned source of truth for the recurring `dagzoo`
repo audit.

Primary reference:

- OpenAI, [Harness Engineering](https://openai.com/index/harness-engineering/)

The weekly audit translates the article's repo-relevant themes into an explicit
operating checklist for `dagzoo`:

- repo as system of record
- automation and contributor readiness
- canonical interfaces
- architecture and taste enforcement
- entropy and garbage collection
- safe autonomy and reliable handoffs

## Recurring Ticket Contract

- Linear issue title:
  `ops(harness): weekly full-repo harness audit`
- Linear project:
  `dagzoo`
- Linear state on creation:
  `Todo`
- Recurrence:
  weekly, Friday at 10:00 PM `America/Los_Angeles`
- Execution mode:
  scheduled automation or manual maintainer run

## Audit Checklist

Each weekly audit must inspect the current repo state, not stale assumptions.

### 1. Repo As System Of Record

- `README.md`, `AGENTS.md`, and active docs still describe the real workflow.
- User-facing contract changes are reflected in docs and, when required, version
  bump / changelog policy.
- Linear backlog and roadmap references still align.

### 2. Automation And Contributor Readiness

- `AGENTS.md` is sufficient for an unfamiliar autonomous contributor to
  understand how to work in the repo.
- Canonical commands are discoverable without reconstructing them from CI files.
- Public vs internal surfaces are legible in docs or codebase-navigation docs.

### 3. Canonical Interfaces

- There is a minimal, canonical bootstrap path for a fresh workspace.
- There is a minimal, canonical verification path for a normal code change.
- Scripts and wrappers are current and do not point to stale behavior.

### 4. Architecture And Structural Enforcement

- Dependency direction still matches the intended `src/dagzoo/core`-centric
  wiring model.
- No new long-lived shims or obvious cyclic dependency pressure has appeared.
- Architectural guidance in docs still matches the actual module graph.

### 5. CI And Review Surface

- Required CI still reflects the current quality bar.
- PR/review expectations are enforced by repo-owned guidance/templates.
- Failing or flaky validation is captured as tracker work, not tribal knowledge.

### 6. Entropy And Garbage Collection

- Stale docs, scripts, generated outputs, or dead local-output assumptions are
  identified and either removed or ticketed.
- Known drift areas, such as stale public-output paths or unused helper flows,
  are explicitly checked.

### 7. Tracker Hygiene

- Oversized or ambiguous tickets are split or rewritten.
- Duplicate remediation tickets are not created when an open ticket already
  covers the same problem.
- Tickets include acceptance criteria and validation expectations.

## Remediation Issue Rules

When the audit finds an actionable gap:

1. Search the current `dagzoo` Linear project for an open issue covering the
   same gap.
1. Reuse the existing issue if it already exists.
1. Otherwise, create a new issue:
   - in state `Backlog`
   - with label `harness`
   - with no default priority assignment
   - linked back to the weekly audit issue
1. Each remediation issue must include:
   - the concrete repo fact that triggered it
   - the target behavior
   - acceptance criteria
   - validation expectation

## Completion Rule

The weekly audit issue is complete only when:

- it contains a concise audit summary, and
- every net-new actionable gap is linked to a remediation issue, or
- it explicitly records that no action was required this week.
