# Mission-Aligned Roadmap (2026Q1)

This is the canonical roadmap for `dagzoo`.

It maps the mission and strategic pillars in `README.md` to:

- current implemented capabilities
- known gaps
- prioritized roadmap items with explicit exit criteria
- canonical status/rank sequencing and tracker links

Related docs:

- Decision rubric and go/no-go gates: `docs/development/backlog_decision_rules.md`
- Evidence appendix: `reference/literature_evidence_2026.md`
- System behavior walkthrough: `docs/how-it-works.md`
- Output contract: `docs/output-format.md`

## Status Labels

- `implemented`: available in current code and exposed through config/CLI.
- `partial`: some building blocks exist, but mission-level claim is not fully met.
- `planned`: scoped and prioritized, not implemented.
- `research`: exploratory with higher uncertainty or risk.
- `retired`: historical item that no longer represents active roadmap work.

## Canonical Planning Metadata

`docs/development/roadmap.md` is the single source of truth for planning state. Every active item is tracked here with:

- status and milestone lane
- priority rank
- active Linear issue mapping plus historical GitHub references for completed work
- dependencies and exit criteria

If any other document disagrees with this file, this file is authoritative.

Active execution now lives in the Linear project `dagzoo`
(`https://linear.app/bl-personal/project/dagzoo-4867d49bb182/overview`).
The GitHub-to-Linear migration map is committed at
`reference/linear_issue_map_2026-03-08.json`.

## PFN Utility Prioritization Lens

Roadmap ranking is currently optimized for downstream PFN utility:

- primary: filtered dataset throughput and acceptance yield while preserving or improving effective diversity for classification/regression corpora
- implemented baseline: downstream handoff now ships through `dagzoo generate --handoff-root` plus `handoff_manifest.json` so downstream repos such as `tab-foundry` can consume generated corpora without a parallel request-only contract
- deferred: closed-loop feedback from downstream model predictions back into generation policy after the one-way handoff is stable

## Canonical Priority Queue

Lower rank means higher priority. Rank `0` is reserved for completed items retained for traceability.

| Rank | Roadmap ID | Item                                                                   | Status      | Milestone | Tracker Links                                                                             |
| ---- | ---------- | ---------------------------------------------------------------------- | ----------- | --------- | ----------------------------------------------------------------------------------------- |
| 0    | RD-001     | Ground-truth DAG artifact export                                       | implemented | Now       | `#44 -> #45 -> #46 -> #47 -> #48` (completed)                                             |
| 0    | RD-003     | Missingness generation (MCAR/MAR/MNAR)                                 | implemented | Now       | `#15 -> #17 -> #18` (completed)                                                           |
| 0    | RD-004     | Shift-aware SCM generation                                             | implemented | Now       | `#64 -> #72 -> #73 -> #74 -> #75` (completed)                                             |
| 0    | RD-006     | Curriculum complexity scaling (features + graph)                       | retired     | Now       | `#49 -> #50 -> #51 -> #90 -> #52 -> #53` (historical), `#142` (replacement)               |
| 0    | RD-007     | Many-class rollout envelope (`<=32` classes)                           | implemented | Now       | `BL-17 -> BL-18 -> BL-19 -> BL-20 -> BL-21` (completed), `BL-31` (closure)                |
| 0    | RD-008     | Meta-feature coverage steering (retired)                               | retired     | Now       | `#9` (historical)                                                                         |
| 0    | RD-012     | Noise family diversification for synthetic generation                  | implemented | Now       | `#24 -> #25 -> #26 -> #27` (completed)                                                    |
| 0    | RD-014     | Stage-level benchmark observability and telemetry                      | implemented | Now       | `BL-82` (completed historical delivery under `BL-49`)                                     |
| 0    | RD-015     | Keyed RNG semantic reproducibility                                     | implemented | Now       | `BL-90 -> BL-133 -> BL-134 -> BL-135 -> BL-136 -> BL-137`                                 |
| 0    | RD-009     | Filtered dataset throughput and deferred-filter scaling                | implemented | Now       | `BL-49 -> BL-148 -> BL-149 -> BL-150` (completed), `BL-84 -> BL-85` (deferred follow-ons) |
| 0    | RD-016     | Generate-handoff manifest and one-way `tab-foundry` handoff            | implemented | Now       | `BL-143 -> BL-144 -> BL-145 -> BL-146 -> BL-147` (completed)                              |
| 3    | RD-011     | Mechanism diversity expansion with measurable effective-diversity gain | planned     | Now       | `BL-26 -> BL-151 -> BL-29 -> BL-30`                                                       |
| 8    | RD-005     | Robustness stress profiles (hard-task/adversarial regimes)             | research    | Later     | `BL-48 -> BL-59 -> BL-62 -> BL-61 -> BL-60`                                               |
| 9    | RD-013     | Time-series generation tracks for PFN pretraining                      | research    | Later     | `BL-73 -> BL-74 -> BL-75 -> BL-76 -> BL-77`                                               |
| 10   | RD-002     | Interventional and counterfactual generation modes                     | research    | Later     | `BL-50 -> BL-67 -> BL-68 -> BL-69 -> BL-70`                                               |
| 11   | RD-010     | Hardware-adaptive autotuning beyond coarse FLOPs tiers                 | planned     | Later     | `BL-42 -> BL-43 -> BL-44 -> BL-53 -> BL-45 -> BL-54 -> BL-46`                             |

## Current Capability Matrix

| README Mission/Pillar Claim                                         | Current State | Evidence in Repo                                                                                                                                                                                                                                                                         | Gap                                                                                                                                              | Roadmap IDs    |
| ------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------- |
| Foundation model pretraining with diverse structural priors         | `partial`     | Canonical fixed-layout generation, `dataset.rows`, deferred filtering, effective-diversity audits, diagnostics coverage aggregation, explicit noise/shift controls, stage-level throughput metrics, generate-handoff manifests, and the shipped `piecewise` control path are implemented | Widened `gp` variants still need measured diversity evidence and time-series generation is not implemented end-to-end                            | RD-011, RD-013 |
| Causal discovery with ground-truth DAGs and interventional datasets | `partial`     | DAG lineage metadata is emitted per dataset and persisted as compact shard-level artifacts with schema validation and benchmark guardrails                                                                                                                                               | Interventional/counterfactual generation semantics are not implemented                                                                           | RD-002         |
| Robustness testing with hard tasks, shifts, adversarial regimes     | `partial`     | Deferred filtering, diagnostics proxies, missingness mechanisms, explicit noise-family controls, and shift/drift controls are implemented with deterministic controls and benchmark guardrails                                                                                           | No explicit hard-task/adversarial profile suite                                                                                                  | RD-004, RD-005 |
| Causal structural integrity (hierarchical dependencies)             | `implemented` | Graph-driven node pipeline, canonical fixed-layout execution, DAG lineage artifacts, and keyed RNG semantic reproducibility are implemented                                                                                                                                              | Remaining work is diversity-oriented expansion, not structural correctness                                                                       | RD-011         |
| Tabular realism (mixed type + postprocess hooks)                    | `partial`     | Numeric/categorical converters, postprocess hooks, many-class rollout within the current `<=32` class envelope, configurable missingness, explicit noise families, and canonical fixed-layout generation are implemented                                                                 | Broader mechanism diversity and optional future expansion beyond the current many-class envelope remain deferred                                 | RD-007, RD-011 |
| PFN task coverage (classification, regression, time-series)         | `partial`     | Classification and regression generation pipelines are fully supported with deterministic seeds, keyed replay metadata, and benchmark workflows                                                                                                                                          | No time-series generation mode, temporal metadata contract, or temporal diagnostics/guardrails                                                   | RD-013         |
| Staged complexity scaling (features/nodes/samples)                  | `retired`     | Historical staged-complexity implementation (RD-006) has been retired in favor of explicit split sizing and fixed-layout generation                                                                                                                                                      | Not active                                                                                                                                       | RD-006         |
| Hardware-native performance (Torch + hardware-aware tuning)         | `partial`     | Torch CPU/CUDA/MPS path, hardware detection, coarse profile-based tuning, benchmark suite, and stage-level generation/write/filter metrics are implemented                                                                                                                               | Hardware-adaptive autotuning is not implemented; any further throughput work is deferred to later follow-ons after the completed RD-009 baseline | RD-010, RD-014 |
| Downstream synthetic-corpus handoff                                 | `implemented` | `dagzoo generate --handoff-root` emits `handoff_manifest.json`, writes generated handoff artifacts under `generated/`, and the docs include a reproducible one-way `dagzoo -> tab-foundry` smoke workflow                                                                                | Closed-loop downstream feedback remains intentionally deferred beyond the one-way handoff baseline                                               | RD-016         |

## Current Implementation Baseline

This roadmap does not duplicate the current codebase map or public contract.
Use the canonical docs instead:

- system/data-flow walkthrough: `docs/how-it-works.md`
- output and artifact contract: `docs/output-format.md`
- package/module structure: `docs/development/codebase-navigation.md`
- generated import/dependency map: `docs/development/module-dependency-map.md`
- packaged/runtime profiles: `configs/`
- source references: `reference/PAPERS.md` for TabICLv2 Appendix E (`E.2`-`E.14`),
  "A Closer Look at TabPFN v2", and
  "Accurate predictions on small data with a tabular foundation model"

### Reproducibility Strategy

1. Global run seed -> per-dataset seed -> per-component derived seeds.
1. `KeyedRng` provides the semantic RNG contract for generation/runtime code,
   while preserving deterministic child-seed derivation for replay and
   compatibility workflows.
1. Document expected backend variation (best effort, not strict bitwise
   determinism).

### Validation and Benchmarks

#### Correctness

- Unit invariants for ranges, shapes, DAG validity, converter class ranges, and
  matrix normalization.
- Unit/integration coverage for missingness mask invariants, deterministic
  behavior, and end-to-end metadata emission.
- Integration tests for end-to-end classification/regression paths.

#### Reproducibility

- Fixed seed should reproduce metadata exactly and numeric outputs within
  tolerance.

#### Performance

- Benchmark suites: `smoke`, `standard`, `full`.
- Artifacts: JSON + Markdown summaries under
  `benchmarks/results/<timestamp>/`.
- Soft regression gate: warn at configurable threshold, fail only on severe
  regression with `--fail-on-regression`.

## Roadmap Items

### RD-001: Ground-Truth DAG Artifact Export

- Status: `implemented`
- Milestone: `Now` (completed via epics/issues `#44`, `#45`, `#46`, `#47`, `#48`)
- Mission alignment: causal discovery
- Pillar alignment: causal structural integrity
- Goal: persist full adjacency matrix and node assignment lineage as stable dataset artifacts.
- Repo touchpoints: `src/dagzoo/core/dataset.py`, `src/dagzoo/core/layout.py`, `src/dagzoo/io/parquet_writer.py`, `src/dagzoo/types.py`
- Delivered scope:
  - Every generated dataset emits lineage metadata with adjacency + assignment lineage and deterministic seed behavior.
  - Persisted shard outputs rewrite dense adjacency into compact bit-packed artifacts with per-shard index files.
  - Validator enforces versioned dense/compact lineage schemas and compatibility rules.
  - Benchmark profiles report `lineage_guardrails` for export-overhead checks using warn/fail thresholds.
- Completion evidence:
  - Docs and config presets include lineage workflow and benchmark examples.
  - Integration tests cover classification and regression generation + artifact persistence.
  - Existing config defaults remain backward-compatible.

### RD-002: Interventional and Counterfactual Generation Modes

- Status: `research`
- Milestone: `Later`
- Mission alignment: causal discovery
- Pillar alignment: causal structural integrity
- Goal: support observational + interventional sampling tracks with explicit intervention specs.
- Linear tracking: epic `BL-50`; dependency chain `BL-67 -> BL-68 -> BL-69 -> BL-70`
- Repo touchpoints: `src/dagzoo/config/`, `src/dagzoo/core/dataset.py`, `src/dagzoo/core/node_pipeline.py`, `src/dagzoo/cli/`
- Exit criteria:
  - Config supports opt-in intervention mode with safe default (`off`).
  - Generated artifacts contain intervention set and pre/post intervention metadata.
  - Acceptance tests verify truncated-factorization behavior for fixed interventions.

### RD-003: Missingness Generation (MCAR/MAR/MNAR)

- Status: `implemented`
- Milestone: `Now` (completed via epics/issues `#17` and `#18`)
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: tabular realism
- Goal: provide configurable missing-data mechanisms with deterministic seeded behavior and benchmark-time acceptance/runtime guardrails.
- Delivered scope:
  - `DatasetConfig` supports missingness controls (`missing_rate`, mechanism, MAR/MNAR scales). See [docs/how-it-works.md](../how-it-works.md) for MCAR/MAR/MNAR mechanism definitions.
  - `dagzoo generate` supports missingness CLI overrides.
  - Generation path injects deterministic missingness masks and emits per-bundle metadata.
  - Benchmark profiles emit `missingness_guardrails` including metadata coverage, realized-rate accuracy, and runtime degradation checks.
- Repo touchpoints: `src/dagzoo/config/`, `src/dagzoo/sampling/missingness.py`, `src/dagzoo/postprocess/postprocess.py`, `src/dagzoo/core/dataset.py`, `src/dagzoo/cli/`, `src/dagzoo/bench/suite.py`
- Completion evidence:
  - Config and CLI support opt-in mechanism selection and missing rate controls.
  - Tests validate expected missing-rate and dependency behavior.
  - Benchmark summaries include missingness guardrail metrics and status.

### RD-004: Shift-Aware SCM Generation

- Status: `implemented`
- Milestone: `Now` (completed via epics/issues `#64`, `#72`, `#73`, `#74`, and `#75`)
- Mission alignment: robustness testing, causal discovery
- Pillar alignment: causal structural integrity, tabular realism
- Goal: introduce controlled distribution-shift/drift modes in graph and mechanism sampling.
- GitHub tracking: epic `#64`; dependency chain `#72 -> #73 -> #74 -> #75`
- Repo touchpoints: `src/dagzoo/config/`, `src/dagzoo/core/dataset.py`, `src/dagzoo/core/shift.py`, `src/dagzoo/diagnostics/`, `src/dagzoo/bench/`, `configs/preset_shift_*.yaml`
- Delivered scope:
  - Shift controls are integrated into graph/mechanism/noise sampling with deterministic seeded behavior.
  - Per-bundle metadata and diagnostics expose resolved shift settings and observability signals.
  - Discoverable shift presets are available for generation and benchmark smoke workflows.
  - Benchmark profiles emit `shift_guardrails` with runtime, metadata-coverage, and directional checks against shift-disabled controls.
- Completion evidence:
  - Shift workflows are runnable directly from preset configs and documented in user-facing guides.
  - Integration tests cover shift metadata/diagnostics propagation and preset/CLI execution paths.
  - Benchmark summaries include `shift_guardrails` alongside existing guardrail families.

### RD-005: Robustness Stress Profiles (Hard-Task/Adversarial Regimes)

- Status: `research`
- Milestone: `Later`
- Mission alignment: robustness testing
- Pillar alignment: tabular realism
- Goal: define reproducible stress presets (low-SNR, class imbalance, harder interactions).
- Linear tracking: epic `BL-48`; dependency chain `BL-59 -> BL-62 -> BL-61 -> BL-60`
- Repo touchpoints: `src/dagzoo/config/`, `src/dagzoo/functions/random_functions.py`, `src/dagzoo/postprocess/postprocess.py`, `src/dagzoo/bench/`
- Exit criteria:
  - Presets are selectable via config/CLI and remain opt-in.
  - Benchmarks and diagnostics confirm regimes differ from baseline in intended directions.
  - Reproducibility tests pass for fixed seed runs.

### RD-006: Staged Complexity Scaling (Features + Graph)

- Status: `retired`
- Milestone: `Now` (completed via epics/issues `#50`, `#51`, `#90`, `#52`, `#53`)
- Mission alignment: foundation model pretraining
- Pillar alignment: tabular realism
- Goal: historical; staged complexity controls have been removed in favor of
  explicit split sizing and fixed-layout generation.
- GitHub tracking: epic `#49`; dependency chain `#50 -> #51 -> #90 -> #52 -> #53`
- Repo touchpoints (historical): `src/dagzoo/config/`,
  `src/dagzoo/core/dataset.py`

### RD-007: Many-Class Rollout Envelope (`<=32` Classes)

- Status: `implemented`
- Milestone: `Now` (completed via `BL-17`, `BL-18`, `BL-19`, `BL-20`, `BL-21`, with `BL-31` retained as closure note)
- Mission alignment: foundation model pretraining
- Pillar alignment: tabular realism, causal structural integrity
- Goal: land a stable many-class rollout envelope while keeping filter behavior and label handling interpretable.
- Linear tracking: historical epic `BL-17`; completion chain `BL-18 -> BL-19 -> BL-20 -> BL-21`; closure note `BL-31`
- Repo touchpoints: `src/dagzoo/config/`, `src/dagzoo/converters/categorical.py`, `src/dagzoo/filtering/extra_trees_filter.py`, `docs/features/many-class.md`
- Delivered scope:
  - `dataset.n_classes_max <= 32` is enforced as the supported rollout envelope.
  - Converter and postprocess paths are hardened for the current many-class range.
  - Deferred filter diagnostics make accept/reject behavior interpretable within the current envelope.
  - Discoverable many-class generation and benchmark smoke workflows are documented.
- Completion evidence:
  - Config validation and user docs align on the supported envelope.
  - Integration and CLI tests cover the current many-class workflows.
  - Broader expansion beyond the current envelope is not an active roadmap item.

### RD-008: Meta-Feature Coverage Steering

- Status: `retired`
- Milestone: `Now`
- Mission alignment: foundation model pretraining
- Pillar alignment: tabular realism
- Goal: feature retired; diagnostics target bands remain as reporting-only metadata.
- Repo touchpoints: `src/dagzoo/diagnostics/coverage.py`, `src/dagzoo/core/dataset.py`, `src/dagzoo/cli/`
- Notes:
  - Steering selection logic was removed to simplify generation semantics.
  - `diagnostics.meta_feature_targets` remains supported for coverage summaries.

### RD-009: Filtered Dataset Throughput and Deferred-Filter Scaling

- Status: `implemented`
- Milestone: `Now`
- Mission alignment: foundation model pretraining
- Pillar alignment: hardware-native performance
- Goal: improve accepted-corpus throughput on the canonical `generate -> filter` pipeline while preserving or improving effective diversity.
- Linear tracking: epic `BL-49`; completed chain `BL-148 -> BL-149 -> BL-150`; deferred follow-ons `BL-84 -> BL-85`; historical completed work `BL-82`, `BL-83`, `BL-86`
- Repo touchpoints: `src/dagzoo/filtering/deferred_filter.py`, `src/dagzoo/bench/stage_metrics.py`, `src/dagzoo/bench/suite.py`, `src/dagzoo/cli/`, `src/dagzoo/io/parquet_writer.py`
- Delivered scope:
  - `BL-148`: optimized deferred-filter replay throughput on canonical shard metadata without reviving worker orchestration.
  - `BL-149`: promoted filtered-corpus throughput and acceptance yield into first-class benchmark outputs and artifacts.
  - `BL-150`: added diversity-aware filter calibration guardrails so throughput tuning can be evaluated against diversity regression signals.
- Completion evidence:
  - Filter-enabled benchmark flows show filtered-corpus throughput alongside generation, write, and filter stage rates plus acceptance-yield signals.
  - Filter calibration and audit workflows can compare throughput or yield against diversity regression signals.
  - No new public worker-count or worker-index surface was added.
  - The active RD-009 workstream is complete; `BL-84` and `BL-85` remain deferred follow-ons rather than part of the completed critical path.

### RD-010: Hardware-Adaptive Autotuning Beyond Coarse FLOPs Tiers

- Status: `planned`
- Milestone: `Later`
- Mission alignment: foundation model pretraining
- Pillar alignment: hardware-native performance
- Goal: evolve hardware-aware scaling from static coarse profile tiers to bounded adaptive tuning based on observed throughput/memory behavior when throughput/cost becomes a practical bottleneck.
- Linear tracking: epic `BL-42`; dependency chain `BL-43 -> BL-44 -> BL-53 -> BL-45 -> BL-54 -> BL-46`
- Repo touchpoints: `src/dagzoo/hardware.py`, `src/dagzoo/config/`, `src/dagzoo/cli/`, `src/dagzoo/bench/suite.py`, `src/dagzoo/bench/report.py`
- Exit criteria:
  - Adaptive mode improves throughput versus profile baseline on at least one CUDA hardware class without violating memory guardrails.
  - Unknown CUDA devices can run adaptive tuning without relying only on static fallback tiers.
  - Fixed seed + fixed hardware signature reproduces selected tuning settings within declared deterministic behavior.
  - Opt-out mode preserves current profile-only behavior.

### RD-011: Mechanism Diversity Expansion With Measurable Effective-Diversity Gain

- Status: `planned`
- Milestone: `Now`
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: causal structural integrity, tabular realism
- Goal: build on the already-landed mechanism-family mix surface by widening the current `gp` family, preserving the shipped `piecewise` control, and proving that the widened path increases emitted diversity without unacceptable throughput or filter-yield regressions.
- Linear tracking: epic `BL-26`; landed baseline plumbing and shipped control are historical `BL-27`; active closeout is `BL-151`; remaining closeout items are `BL-29 -> BL-30`; canceled historical tickets `BL-28 -> BL-51 -> BL-52` are no longer the implementation path
- Repo touchpoints: `src/dagzoo/config/`, `src/dagzoo/functions/random_functions.py`, `src/dagzoo/core/node_pipeline.py`, `src/dagzoo/core/dataset.py`, `src/dagzoo/bench/suite.py`
- Delivered baseline inside this roadmap item:
  - `mechanism.function_family_mix` is already live on the canonical fixed-layout path.
  - `piecewise` shipped as a public, mix-controlled mechanism family and serves as the current control path.
  - Bundle metadata, diagnostics coverage, and diversity-audit artifacts already expose realized mechanism-family coverage.
- Active closeout scope:
  - widen `gp` through internal `standard`, `periodic`, and `multiscale` variants without adding a new public config knob
  - expose additive mechanism-variant observability in emitted metadata and downstream summaries
  - add GP-focused presets/docs so the candidate path can be evaluated against baseline and the shipped `piecewise` control
- Exit criteria:
  - Widened `gp` variants are exercised through the existing family-mix surface and covered by unit/integration tests.
  - Metadata, diagnostics, and diversity-audit artifacts expose realized mechanism-family and mechanism-variant coverage beyond config echo.
  - Diversity-audit runs provide evidence for the widened `gp` path versus baseline, with the shipped `piecewise` path retained as the control comparator.
  - Presets/docs/bench guardrails exist for contributors.

### RD-012: Noise Family Diversification for Synthetic Generation

- Status: `implemented`
- Milestone: `Now` (completed via epics/issues `#24`, `#25`, `#26`, `#27`)
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: tabular realism
- Goal: complete lean, low-complexity integration of explicit noise-family controls and mixtures to diversify residual/noise behavior without broad generator refactors.
- GitHub tracking: epic `#24`; dependency chain `#25 -> #26 -> #27` (completed)
- Repo touchpoints: `src/dagzoo/config/`, `src/dagzoo/sampling/`, `src/dagzoo/core/dataset.py`, `src/dagzoo/bench/suite.py`
- Delivered scope:
  - Config supports `gaussian`, `laplace`, `student_t`, and `mixture` families with safety validation.
  - Runtime sampling and generation metadata report requested/effective family settings.
  - Benchmark guardrails include metadata validation and runtime delta checks versus gaussian-noise controls.
  - Presets/docs/tests cover family-specific generation and benchmark workflows.
- Completion evidence:
  - All delivery issues in the chain are closed (`#25`, `#26`, `#27`).
  - End-user docs include noise workflow guidance and benchmark examples.

### RD-013: Time-Series Generation Tracks for PFN Pretraining

- Status: `research`
- Milestone: `Later`
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: tabular realism
- Goal: add an opt-in temporal generation track for sequence datasets so PFN pretraining workflows cover classification/regression/time-series under one reproducible generator framework.
- Linear tracking: epic `BL-73`; dependency chain `BL-74 -> BL-75 -> BL-76 -> BL-77`
- Repo touchpoints: `src/dagzoo/config/`, `src/dagzoo/core/dataset.py`, `src/dagzoo/core/node_pipeline.py`, `src/dagzoo/diagnostics/`, `src/dagzoo/bench/`, `docs/`
- Exit criteria:
  - Temporal mode is opt-in and backward-compatible (`off` by default).
  - Fixed seed + config reproducibility is preserved for temporal generation.
  - Sequence metadata/diagnostics contracts are emitted and test-covered.
  - Presets/docs/bench guardrails provide discoverable temporal workflows.

### RD-014: Stage-Level Benchmark Observability and Telemetry

- Status: `implemented`
- Milestone: `Now` (completed historically via `BL-82`)
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: hardware-native performance
- Goal: expose stage-level throughput telemetry for canonical benchmark runs so bottlenecks can be attributed before further runtime work is prioritized.
- Linear tracking: historical delivery `BL-82` under `BL-49`
- Repo touchpoints: `src/dagzoo/bench/stage_metrics.py`, `src/dagzoo/bench/suite.py`, `src/dagzoo/cli/`, `docs/features/benchmark-guardrails.md`
- Delivered scope:
  - Benchmark/report artifacts expose generation, write, and optional filter stage throughput instead of only total runtime.
  - Filter rejection and retry signals are surfaced for filter-enabled benchmark runs.
  - Stage metrics are stable enough to use in regression triage and backlog prioritization.
- Completion evidence:
  - Benchmark CLI summaries print stage-level throughput fields.
  - Benchmark JSON and Markdown artifacts persist stage-level metrics.
  - Subsequent throughput planning now relies on those stage metrics rather than inferred bottlenecks.

### RD-015: Keyed RNG Semantic Reproducibility

- Status: `implemented`
- Milestone: `Now` (completed via `BL-90`, `BL-133`, `BL-134`, `BL-135`, `BL-136`, and `BL-137`)
- Mission alignment: foundation model pretraining, robustness testing
- Pillar alignment: causal structural integrity, hardware-native performance
- Goal: replace order-coupled ambient RNG usage with keyed semantic namespaces so regrouping, retries, and scalar-vs-batched path changes preserve the intended reproducibility contract.
- Linear tracking: epic `BL-90`; dependency chain `BL-133 -> BL-134 -> BL-135 -> BL-136 -> BL-137`
- Repo touchpoints: `src/dagzoo/rng.py`, `src/dagzoo/core/`, `src/dagzoo/postprocess/`, `src/dagzoo/sampling/`, `src/dagzoo/bench/`, `docs/`
- Delivered scope:
  - Core generation/runtime randomness is keyed by semantic namespace rather than draw order or offset-only coupling.
  - `KeyedRng` is the semantic RNG surface used by runtime code.
  - Retrying one stage does not perturb sibling-stage randomness.
  - Scalar and batched typed-plan execution preserve semantic equivalence under the keyed contract.
  - Reproducibility docs describe `keyed_replay` and the intended non-goals of the contract.
- Completion evidence:
  - Repo code and tests exercise `KeyedRng` across generation, postprocess, missingness, and benchmark flows.
  - Generated bundle metadata includes keyed replay paths for canonical batch replay.
  - The Linear implementation chain is closed and moved to historical traceability.

### RD-016: Generate-Handoff Manifest and One-Way `tab-foundry` Handoff

- Status: `implemented`
- Milestone: `Now`
- Mission alignment: foundation model pretraining
- Pillar alignment: hardware-native performance, tabular realism
- Goal: let downstream repos consume generated corpora plus machine-readable handoff metadata directly from the canonical generate workflow, without a parallel request-only contract.
- Linear tracking: epic `BL-143`; dependency chain `BL-144 -> BL-145 -> BL-146 -> BL-147`
- Repo touchpoints: `src/dagzoo/cli/`, `src/dagzoo/core/`, `docs/`, downstream `tab-foundry`
- Exit criteria:
  - `dagzoo generate --handoff-root` publishes a versioned handoff manifest without a separate request-file schema.
  - Handoff execution stays on the canonical generate path with effective-config traceability.
  - A machine-readable handoff manifest exposes generated-corpus paths, stage-throughput context, invocation metadata, and diversity-artifact paths.
  - Docs include at least one reproducible `dagzoo -> tab-foundry` smoke workflow.
  - Closed-loop feedback ingestion remains explicitly out of scope until the one-way handoff is stable.
- Completion evidence:
  - `dagzoo generate --handoff-root` rejects stale handoff roots before execution.
  - Handoff runs publish a versioned `handoff_manifest.json` at the handoff root with generate invocation metadata, artifact paths, throughput context, hardware metadata, and nullable diversity-artifact paths.
  - Public docs cover the handoff artifact layout, manifest contract, and one-way downstream smoke workflow.
  - The Linear implementation chain `BL-143 -> BL-144 -> BL-145 -> BL-146 -> BL-147` is closed.

## Milestone Board

### Implemented

- RD-001 ground-truth DAG artifact export, completed via `#44`, `#45`, `#46`, `#47`, and `#48`
- RD-003 missingness generation (MCAR/MAR/MNAR), completed via `#17` and `#18`
- RD-004 shift-aware SCM generation, completed via `#64`, `#72`, `#73`, `#74`, and `#75`
- RD-007 many-class rollout envelope, completed via `BL-17`, `BL-18`, `BL-19`, `BL-20`, and `BL-21`
- RD-008 meta-feature coverage steering (retired)
- RD-006 staged complexity scaling (retired), completed via `#49`, `#50`, `#51`, `#90`, `#52`, and `#53`
- RD-012 noise family diversification, completed via `#24`, `#25`, `#26`, and `#27`
- RD-014 stage-level benchmark observability and telemetry, completed via `BL-82`
- RD-015 keyed RNG semantic reproducibility, completed via `BL-90`, `BL-133`, `BL-134`, `BL-135`, `BL-136`, and `BL-137`
- RD-016 generate-handoff manifest and one-way `tab-foundry` handoff, completed via `BL-143`, `BL-144`, `BL-145`, `BL-146`, and `BL-147`

### Now

- RD-011 mechanism diversity expansion with measurable effective-diversity gain

### Later

- RD-005 robustness stress profiles
- RD-013 time-series generation tracks
- RD-002 interventional and counterfactual generation modes
- RD-010 hardware-adaptive autotuning

## Dependencies and Sequencing

- RD-014 is implemented and provided the stage-level evidence that justified centering RD-009 on deferred-filter throughput rather than speculative public parallelism.
- RD-015 is implemented and provides the semantic RNG contract that active throughput or handoff work must preserve.
- RD-009 is implemented and now serves as the baseline canonical `generate -> filter` pipeline for later handoff and runtime work.
- RD-016 is implemented on top of the canonical `generate -> filter` pipeline from RD-009 and does not introduce a parallel configuration surface that the repo has already removed.
- RD-011 is the active diversity lever, but it should be evaluated against RD-009's acceptance-yield and diversity guardrails rather than in isolation.
- RD-012 is implemented and provides explicit noise-family controls that RD-005 can consume later for stress-profile composition.
- RD-005 now primarily depends on RD-004, RD-003, and the filter/density observability already in the repo.
- RD-013 remains later because the near-term downstream contract is one-way tabular corpus handoff, not temporal generation.
- RD-002 builds on completed RD-001 lineage artifacts for intervention metadata extensions.
- RD-010 remains opt-in and benchmark-guarded, but is sequenced later because downstream handoff and mechanism-diversity expansion are more urgent than adaptive tuning.

## Guardrails

- New public config surfaces remain opt-in by default; internal widening behind an existing public family label must be explicitly documented when it changes emitted behavior.
- Existing config files remain valid unless explicitly versioned.
- Reproducibility expectations are mandatory for every roadmap item.
- Benchmark warn/fail thresholds remain the performance gate.
