# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Historical note: entries below are ordered by release version. This file also
contains imported legacy history, so date order is not strictly monotonic:
`0.3.0` records the older `cauchy-generator -> dagzoo` rename, while `0.5.0`
records the later `dagsynth -> dagzoo` rename on the current release line.

## [0.19.2] - 2026-04-02

### Added

- Added repo-local RD-005 stress-profile smoke presets plus a public stress
  profiles guide covering generate, benchmark, and diversity-audit workflows.
- Added a repo-local Hugo installer used by GitHub Actions so docs jobs validate
  against the pinned `0.152.2` extended build without deprecated Node 20 action
  runtimes.

### Fixed

- Fixed fixed-layout benchmark retry planning so precomputed classification
  attempt plans respect `filter.max_attempts` and never emit replay indices that
  runtime retries cannot honor.
- Fixed the docs sync manifest so the public stress-profile feature page is
  published into the Hugo site and linked docs builds stay green.
- Fixed coverage-summary contract fixtures and goldens for the new
  relationship-structure diversity metrics.

### Changed

- Updated docs and test workflows to `actions/setup-node@v6` with installed Node
  `22`, and replaced `peaceiris/actions-hugo@v3` with the repo-local installer
  path.

## [0.19.1] - 2026-04-02

### Added

- Added relationship-structure diagnostics for indegree/outdegree dispersion,
  reachability, ancestor overlap, target-ancestor breadth, and mechanism-family
  cooccurrence so diversity audits can measure graph-structure variation
  directly.
- Added named RD-005 stress profiles
  `anti_memorization_piecewise_classification_graph_breadth_slice_v1` and
  `anti_memorization_piecewise_classification_compositional_slice_v1`.
- Added maintainer-only RD-005 tooling and notes, including a TabICLv2 parity
  audit and a handoff-root Pareto evaluation script for downstream-score,
  diversity-shift, and throughput comparisons.

### Changed

- Expanded internal correlated sampling reuse beyond `categorical_ratio` to
  cover categorical cardinalities, mechanism-family draws, multi-parent
  aggregation choices, converter and GP variants, and fixed-layout
  random-weight decay parameters.

## [0.19.0] - 2026-04-01

### Changed

- **BREAKING:** Deferred filtering is structural-only. `dagzoo filter`,
  benchmark filter replay, and diversity-audit filter replay now accept or
  reject datasets from lineage reachability and target-validity checks only.
- **BREAKING:** Removed the learned-filter config fields
  `threshold`, `n_estimators`, `max_depth`, `min_samples_leaf`,
  `max_leaf_nodes`, `max_features`, `n_bootstrap`, `ease_k_small`,
  `easy_skill_threshold`, `easy_gain_threshold`, `hard_skill_threshold`,
  `stump_skill_threshold`, `use_lineage_veto`,
  `classification_kappa_threshold`,
  `classification_require_prediction_diversity`, and `n_jobs`. Config loading
  and `dagzoo filter --set` now fail fast when those fields are present.
- Filter-enabled replay now requires `metadata.lineage` in shard sidecars
  because the structural path no longer falls back to a learned probe.
- Deferred-filter manifests, summaries, and benchmark filter summaries now use
  `filter_mode=structural_v1` and emit structural reason counts instead of
  learned-filter skill/stump diagnostics.

## [0.18.0] - 2026-04-01

### Changed

- **BREAKING:** Public generation no longer supports `runtime.layout_mode: fixed`.
  Use `runtime.layout_mode: stratified` for throughput-sensitive heterogeneous
  runs while preserving per-dataset layout and plan diversity.
- Added public `runtime.layout_mode: stratified`, a rolling-window
  heterogeneous scheduler that groups datasets by exact `(n_rows, n_features)`
  strata, sizes stratum microbatches from the target-cell budget, and records
  stratification fill/fallback metrics for throughput probes.
- Added `run_stratified_throughput_benchmark(...)` so throughput tooling can
  measure the new public stratified path separately from the fully
  heterogeneous baseline.
- Deferred filtering now defaults `filter.n_jobs` to `1` instead of `-1`.
  Explicit `filter.n_jobs: -1` remains supported for users who still want the
  prior all-core sklearn behavior.

## [0.17.1] - 2026-04-01

### Changed

- Heterogeneous public generation on Apple hardware now resolves
  `device=auto` to CPU instead of MPS. Explicit `device=mps` requests are
  unchanged, and fixed-layout generation keeps the existing device-resolution
  path.
- Heterogeneous grouped execution now defers per-dataset finalization-context
  construction until after postprocessing, avoiding eager metadata work that
  was usually invalidated by feature dropping and permutation.
- Heterogeneous runtime instrumentation now records descriptor-resolution,
  split-resolution, postprocess, and metadata-finalization stage timings for
  throughput probes.
- Added heterogeneous-specific throughput and microbenchmark coverage so the
  benchmark helpers can expose the public heterogeneous bottlenecks alongside
  the existing fixed-layout measurements.

## [0.17.0] - 2026-03-31

### Changed

- **BREAKING:** Public generation now defaults to fully heterogeneous
  per-dataset layout/plan sampling. `generate_one`, `generate_batch`,
  `generate_batch_iter`, `dagzoo generate`, and `build_dataloader(...)` no
  longer imply one shared fixed-layout plan unless
  `runtime.layout_mode: fixed` is set explicitly.
- Added `runtime.layout_mode` with `heterogeneous | fixed` values. Fixed mode
  preserves the prior shared fixed-layout behavior for schema-aligned batching
  and throughput-sensitive workflows.
- Generated dataset catalogs now emit `group_ids.cohort` for heterogeneous
  runs, while `group_ids.layout_plan` is fixed-mode-only.
- Generate handoff manifests now use schema version `4` and accept
  mode-specific `identity.source_family` values:
  `dagzoo.heterogeneous_scm` and `dagzoo.fixed_layout_scm`.
- Fixed-layout-specific throughput and microbenchmark paths now pin
  `runtime.layout_mode: fixed` explicitly instead of inheriting the public
  default.
- **BREAKING:** The repo now hard-cuts to latent-node target semantics. The
  canonical generator derives `y` from one selected latent DAG node, and the
  public docs no longer describe a separate `p(y | X_complete)` target head.
- **BREAKING:** Lineage schema support is now limited to the current
  `target_to_node` shape. Dense lineage payloads use schema version `1.4.0`,
  compact persisted payloads use `1.5.0`, and legacy `target_mode` /
  `target_parent_*` payloads are rejected.
- **BREAKING:** Generated-corpus readers now require `dataset_catalog.ndjson`
  plus the current replay sidecars; compatibility fallback to
  `metadata.ndjson` has been removed from handoff and deferred-filter paths.
- Public artifact docs now match the emitted shard layout: public shards write
  `dataset_catalog.ndjson` and compact lineage artifacts, while replay sidecars
  remain under `internal/`.

## [0.16.1] - 2026-03-31

### Added

- Classification generation can now optionally export teacher-conditional
  test-split probabilities and optimal log-loss metadata when
  `diagnostics.teacher_conditional_export` is enabled.
- Generate handoff manifests now include posterior-predictive provenance
  fields derived from emitted dataset metadata.
- Dataset config now supports target-parent prior controls for lineage
  sampling, including explicit count bounds and near-max mixture weights.
- Emitted lineage metadata and generate-handoff manifests now record
  target-parent selections, fractions, regimes, and prior provenance for
  downstream adequacy checks.

### Changed

- Diagnostics config normalization now validates
  `diagnostics.teacher_conditional_export` as a boolean.
- `README.md` and the published docs now stay user-facing, while autonomous
  contributor workflow guidance is consolidated into `AGENTS.md` and internal
  maintainer docs.
- Fixed-layout schema consistency checks now include target-parent feature
  assignments when validating reused layouts and grouped batch outputs.
- Throughput benchmark preparation now skips eager run-wide classification
  attempt-plan precomputation while preserving the stricter default
  classification replay validation used by canonical dataset generation.

## [0.16.0] - 2026-03-31

### Changed

- **BREAKING:** The default synthetic prior is now factorized. Canonical
  generation samples complete features `X_complete` from the latent DAG first,
  then generates `y` from a separately sampled conditional head over
  `X_complete`, and only later applies optional missingness as an observation
  process instead of extracting the target from a latent DAG node.
- **BREAKING:** Emitted metadata now includes `metadata.prior`, lineage
  assignments now mark the target as `latent_complete_x_conditional`, and canonical
  fixed-layout contract metadata now uses
  `layout_plan_schema_version=9` with
  `layout_execution_contract=chunk_batched_v3`.
- Fixed-layout replay and plan signatures now include the explicit target-head
  payload, classification generation now uses retry-only invalid-split
  handling, and missingness is sampled once over the full emitted feature
  matrix rather than independently per split.
- README, runtime docs, transform notes, output-format docs, missingness docs,
  and curated recipe descriptions now describe the factorized
  `p(x_complete)` plus `p(y | x_complete)` default and call out that the repo
  does not currently implement Nagler-style localization or explicit
  `n`-adaptive prior semantics.

## [0.15.2] - 2026-03-30

### Added

- Added optional `runtime.fixed_layout_batch_size_cap` so fixed-layout
  generation can cap automatically resolved batch sizes across canonical,
  plan-backed, and steering execution paths.

### Changed

- Runtime config normalization now validates
  `runtime.fixed_layout_batch_size_cap` as a positive integer, and emitted
  metadata config payloads continue to omit the field when it is unset.

## [0.15.1] - 2026-03-27

### Changed

- Streamlined internal generation and benchmark plumbing by removing
  fixed-layout compatibility shims, consolidating config-resolution and
  effective-config serialization helpers, routing the PyTorch bridge through
  the canonical batch iterator, and centralizing shared benchmark/runtime
  predicate and count helpers without changing the public CLI or artifact
  contracts.

## [0.15.0] - 2026-03-27

### Changed

- **BREAKING:** Simplified generation overrides around one repeatable
  `--set path=value` surface and removed the granular generate/filter CLI
  flags for missingness and deferred-filter threshold tuning.
- **BREAKING:** Simplified `dataset.rows` support to fixed totals and
  `start..stop` ranges only; list, CSV, mapping, and public `choices`-mode
  row specs are no longer accepted.
- **BREAKING:** Flattened `steering.stages[*]` onto direct stage fields
  (`missing_*`, `shift_*`, `noise_*`) instead of nested `dataset`, `shift`,
  and `noise` blocks.
- **BREAKING:** Simplified config-resolution results and trace artifacts to
  plain dictionaries, including `{path, source, old_value, new_value}` trace
  events for generate and benchmark resolution flows.
- Stress-profile locked-path validation now derives the protected field set
  from the built-in profile definition instead of a hand-maintained string
  table.
- Fixed-layout execution semantics now use the shared sampling helper module
  directly, and the fixed-layout function/matrix plan IR now relies on class
  types instead of redundant matrix/function discriminator fields.
- Filtering now treats deferred replay as a supported runtime path and no
  longer exposes the stale unsupported-feature availability shim.
- **BREAKING:** Removed the runtime-package `dagzoo.bench.h100_validation`
  entrypoint. The supported H100 validation path is now
  `./.venv/bin/python scripts/ci/h100_validation.py`.
- **BREAKING:** Benchmark preset results now emit a unified `scenarios`
  mapping with per-scenario metrics, optional control metrics, and issues
  instead of collector-specific guardrail payloads.
- **BREAKING:** Diagnostics coverage summaries now emit the curated metric
  subset used by product docs and benchmarks and no longer include the
  top-level `steering` payload.
- **BREAKING:** Repo-local dev tooling now keeps `impact`, `contract`, and
  `review-base`, while verification/dependency/environment wrappers move to
  direct `nox` and `import-linter` entrypoints.
- Added repo-level `nox` sessions (`quick`, `full`, `docs`, `bench_smoke`) as
  the canonical verification surface on top of the shared `.venv/`.

## [0.14.5] - 2026-03-27

### Added

- Added a new config-level carried-slice selector under `stress.profile` with
  the first shipped built-in slice
  `anti_memorization_piecewise_classification_slice_v1`.
- Added strict validation and config-resolution materialization for
  `stress.profile` so generation and benchmark flows can reuse the existing
  steering, missingness, shift, and noise plumbing without a parallel runtime.
- Added coverage for stress-profile parsing, conflict validation, config
  resolution, generation metadata stripping, and CLI override rejection.

### Changed

- Effective generate and benchmark configs now record stress-profile
  materialization in their resolution traces, materialize stress selectors
  into concrete effective configs, and continue to omit `stress` from emitted
  per-dataset `metadata.config`.
- Stress-profile resolution now validates authored selectors before
  materialization, then allows repo-owned smoke caps, CUDA hardware policy
  expansion, and microbenchmark downsizing to operate on the concrete config.
- Removed one unused internal stress-profile helper so the shipped RD-005
  selector branch stays clean under dead-code enforcement.
- Roadmap and reference-pack docs now distinguish curated recipe `stress profile` examples from the first RD-005 carried-slice contract used for
  downstream fixed-slice comparisons.

## [0.14.4] - 2026-03-26

### Added

- Added a curated public recipe catalog under `recipes/` with the first
  shipped reference packs: `default-baseline`, `tabpfn-v1-prior-approx`,
  `high-cardinality-stress`, `missingness-robustness`, and `shift-stress`.
- Added `dagzoo recipe list` plus `recipe:<name>` config references for the
  curated recipe surface.
- Added citation, contribution, and security repo metadata via `CITATION.cff`,
  `CONTRIBUTING.md`, and `SECURITY.md`.
- Added adoption-surface parity checks so recipe docs, packaged wheel
  resources, and release metadata cannot silently drift.

### Changed

- `dagzoo generate`, `dagzoo benchmark --preset custom`, `dagzoo diversity-audit`,
  and `build_dataloader(...)` now accept curated `recipe:<name>` references in
  addition to YAML paths.
- Generate handoff manifests now preserve literal `recipe:<name>` references in
  `generate_invocation.config_path` instead of forcing them into filesystem-like
  absolute paths.
- Recipe-backed `dagzoo benchmark --preset custom` runs now persist the bare
  recipe name (for example, `default-baseline`) as the benchmark `preset_key`
  in suite summaries, diagnostics artifact names, and saved baseline payloads
  instead of collapsing all recipe-backed runs onto the default config preset
  key.
- README and user docs now lead with Start, Reference Packs, Advanced Controls,
  and Artifacts/API instead of assuming repo-local config authoring as the
  first-touch workflow.
- Public docs and roadmap status now reflect the shipped steering surface and
  the supported deferred-filter replay workflow.

## [0.14.3] - 2026-03-26

### Added

- Added a PyTorch-native in-process bridge under `dagzoo.pytorch` with
  `DagzooDataset` and `build_dataloader(...)` for streaming canonical
  `DatasetBundle` outputs directly into researcher-facing training loops.

### Changed

- The package root now re-exports `DagzooDataset`, `DagzooSample`, and
  `build_dataloader(...)` for shorter PyTorch bridge imports.

## [0.14.2] - 2026-03-25

### Added

- Added discoverable steering smoke presets for generation and benchmark
  workflows under `configs/preset_steering_anti_memorization_*`.
- Added the steering feature guide and usage-guide / benchmark-guide links for
  the shipped `anti_memorization_piecewise_v1` workflow.

### Changed

- Steering benchmark smoke documentation now points users to
  `diagnostics_artifacts` and the existing `coverage_summary.json` /
  `coverage_summary.md` contract instead of introducing a new benchmark
  `steering_guardrails` summary field.

## [0.14.1] - 2026-03-25

### Changed

- Diagnostics coverage artifacts now include an additive top-level `steering`
  section that reports requested stage authoring, per-stage realized
  missingness/shift/noise summaries, and resolution-consistency checks for
  steering-enabled runs.
- `dagzoo generate --diagnostics` and diversity-audit coverage aggregation now
  carry steering-aware coverage config so the existing `coverage_summary.json`
  and `coverage_summary.md` artifacts can audit curriculum movement without
  changing the per-dataset `metadata.ndjson` contract.
- Diagnostics docs and the output-format contract now document the steering
  analysis section in `coverage_summary.*`.

## [0.14.0] - 2026-03-25

### Added

- Added dynamic `steering.stages` curricula for deterministic per-dataset
  piecewise interpolation of missingness, graph drift, noise drift, and
  mixture-noise weights, plus the built-in
  `anti_memorization_piecewise_v1` preset for richer multi-epoch corpora.

### Changed

- **BREAKING:** Fixed-layout `generate_batch` runs with graph-steered stages
  may now emit multiple `layout_signature` and `layout_plan_signature` groups
  within one corpus while preserving a stable column schema.
- Steering now resolves effective missingness, shift, and noise settings
  against corpus progress on every dataset, carrying prior stage end-states
  forward across later stages instead of selecting one static recipe per
  slice.
- Steering-enabled generation now records resolved per-dataset missingness,
  shift, and noise-distribution metadata while continuing to omit top-level
  `steering` from emitted `metadata.config`.
- Steering-enabled fixed-layout generation now keeps grouped batching for
  matching effective layout and noise cohorts instead of scalarizing every
  dataset when steering is enabled.
- Steering stage resolution now clears incompatible inherited shift scales
  when later stages switch modes, so authored graph-only or noise-only
  stages do not silently keep stale drift from prior stages.
- Graph-steered bundles now keep base fixed-layout replay roots under
  `metadata.keyed_replay.layout_root_path` and
  `metadata.keyed_replay.execution_plan_root_path`, while additive
  steering-specific replay roots capture graph/execution overrides without
  breaking reproducible layout replay.
- `steering.target_metrics` is no longer accepted in generator configs until
  the runtime can honor it; config loading now raises an explicit error
  instead of silently accepting a no-op field.

## [0.13.0] - 2026-03-25

### Changed

- **BREAKING:** Removed the unsupported `dagzoo filter-calibration` command and
  deleted its calibration-only internals and artifact writers. Use
  `dagzoo diversity-audit` for accepted-corpus comparisons between configs.
- **BREAKING:** Removed thin convenience wrapper scripts under `scripts/`.
  Canonical entrypoints are now the packaged `dagzoo` CLI and
  `python -m dagzoo.bench.h100_validation` for the H100 validation workflow.
- **BREAKING:** Removed the `dagzoo.cli` compatibility re-export surface and
  switched the packaged console entrypoint to `dagzoo.cli.entrypoint:main`.
  Internal callers and tests now import direct owner modules instead of going
  through CLI shims.
- `dagzoo generate --help` now states that `--handoff-root` cannot be combined
  with `--out` or `--no-dataset-write`, and `dagzoo benchmark --help` now
  states that `--device` only applies to preset `custom` or a single resolved
  preset.
- Repo docs now advertise only the direct CLI/module entrypoints and no longer
  describe deleted wrapper scripts or the removed calibration workflow.

## [0.12.0] - 2026-03-24

### Changed

- **BREAKING:** Removed the threshold-based ExtraTrees deferred filter. `dagzoo filter`
  now uses a small-shot ease filter with held-out ExtraTrees skill, an optional
  stump veto, and a lineage no-path veto. Public filter cutoffs are now
  `ease_k_small`, `easy_skill_threshold`, `easy_gain_threshold`,
  `hard_skill_threshold`, `stump_skill_threshold`, and `use_lineage_veto`.
- **BREAKING:** Removed the `--threshold` deferred-filter CLI override and the
  persisted `filter.threshold` config key. Old configs now fail with a clear
  migration error pointing to the new ease-filter parameters.
- `filter-calibration` remains exposed as a command entrypoint, but it now
  fails immediately with an explicit unsupported-mode error until a dedicated
  calibration workflow exists for the small-shot ease filter.
- Deferred-filter manifests, summaries, and benchmark replay telemetry now
  record small-shot ease metrics (`skill_small`, `skill_full`, `skill_gain`,
  bootstrap bounds, stump skill, and lineage-veto status) instead of OOB
  threshold-era `wins_ratio` diagnostics.
- Added explicit `dagzoo filter` CLI overrides for both `--lineage-veto` and
  `--no-lineage-veto`, and documented the replay-time lineage-veto control.
- Tuned the canonical `preset_filter_benchmark_smoke` benchmark preset to
  disable the lineage veto, enable a stump veto, and use a deterministic seed
  that yields both accepted and rejected datasets in the smoke-stage replay
  sample.
- Corrected diagnostics `wins_ratio_proxy` back to a direct bootstrap
  win-ratio estimate over held-out ExtraTrees predictions instead of deriving
  it indirectly from the small-shot ease score.
- Deferred filtering now verifies compact lineage adjacency blob checksums
  before applying the no-path veto so stale or corrupted shard lineage
  artifacts fail fast instead of silently changing filter outcomes.
- Removed dead threshold-era calibration helpers and zero-bypass constants that
  were still tripping `vulture` after the small-shot ease filter rewrite.
- Updated the docs to reflect the current branch state: `dagzoo filter` is
  supported again as a deferred small-shot ease replay stage, while
  `filter-calibration` remains unsupported until a dedicated calibration
  workflow exists.

## [0.11.0] - 2026-03-21

### Changed

- `dagzoo filter` and `dagzoo filter-calibration` are supported again. Deferred
  filtering now accepts an explicit `--threshold` override, and
  `threshold=0.0` runs as a provenance-preserving bypass that accepts every
  dataset without fitting ExtraTrees.
- **BREAKING:** filter thresholds are now constrained to the public range
  `[0.0, 1.0]` across config validation, deferred filter CLI overrides, and
  filter-calibration threshold sweeps.
- Pull request CI now validates `pyproject.toml` version bumps against
  `origin/main` and only allows unchanged versions or a single patch/minor step
  per PR.

## [0.10.3] - 2026-03-19

### Changed

- Added `./scripts/dev verify affected` plus impact-aware pytest target
  selection so local and CI verification can choose narrower trusted test
  slices for many code changes.
- PR test workflow now runs an affected-change fast path, emits coverage XML,
  and enforces changed-line coverage with `diff-cover` against `origin/main`.
- Added normalized contract goldens for benchmark summaries, handoff manifests,
  and diagnostics coverage artifacts, plus stricter import-linter product
  surface contracts for `config`, `diagnostics`, and `bench`.
- Benchmark PR workflow now renders comment markdown, step-summary markdown,
  and `benchmark_delta.json` from benchmark summaries and checked-in baselines.

## [0.10.2] - 2026-03-15

### Changed

- Benchmark throughput, latency, microbenchmark, and CUDA peak-memory timing
  now synchronize accelerator work before measuring elapsed time so GPU runs
  report device-complete timings instead of kernel-launch timings.
- Added `scripts/benchmark-gpu-validation.sh` as the canonical local H100
  validation runner for the primary CUDA benchmark plus feature-smoke GPU
  coverage, with the primary standard baseline saved as a run artifact.
- Benchmark suite summaries now record direct bottleneck evidence for
  generation CPU time, write/filter replay timing, fixed-layout batch/chunking,
  and CUDA memory headroom instead of relying only on aggregate throughput.
- The H100 validation runner now adds a large-shape stress phase, bounded
  saturation candidates, and host-level `nvidia-smi` telemetry artifacts for
  the primary H100 performance phases.
- Canonical fixed-layout classification runs now emit datasets as they
  complete instead of prevalidating the entire run before the first bundle, so
  late invalid-class-split failures no longer block earlier successful output.
- Fixed-layout grouped generation now reuses the first split-resolution pass
  during chunk finalization and retries only the dataset offsets that actually
  fail, reducing CPU-heavy preparation overhead on CUDA paths.
- Throughput and suite benchmark artifacts now record preparation-stage wall
  time and CPU time separately from total generation time, and the suite
  report surfaces that preparation evidence directly.
- Fixed-schema batched classification postprocessing now prefers device-local
  label permutations on CUDA batches instead of always generating the class
  permutation on CPU.
- H100 hardware defaults now target `240_000_000` fixed-layout cells, and the
  saturation validation runner keeps its explicit target-cell sweep by running
  those generated configs without the tiered hardware policy floor.
- Fixed-layout batched execution now caches compiled converter-group metadata,
  reuses shared nested standardization work across higher-order function
  branches, and assembles emitted feature tensors by ordered column
  concatenation to cut repeated CPU work in the H100 path.
- Throughput benchmarks and suite reports now record raw-batch substage timing
  for node application, converter execution, and feature materialization so
  H100 bottleneck analysis has direct stage evidence instead of only aggregate
  throughput.
- Fixed-layout runtime metric hooks now degrade cleanly when older internal
  helper stubs do not accept the new optional timing kwargs.

## [0.10.1] - 2026-03-15

### Changed

- Fixed random matrix row normalization so nonzero rows now reach exact unit
  L2 norm even in tiny-magnitude edge cases, including the fixed-layout batch
  matrix path used during generation.
- Added final property-based validation coverage for lineage schema payloads
  and noise/mechanism config normalization to harden merge-time invariants.

## [0.10.0] - 2026-03-14

### Changed

- **BREAKING:** `dagzoo filter` and `dagzoo filter-calibration` now fail with
  an explicit unsupported-feature error while deferred filtering is temporarily
  disabled.
- Filter-enabled `dagzoo benchmark` and `dagzoo diversity-audit` workflows
  remain supported because they replay filter metrics analytically rather than
  producing deferred-filter artifacts.
- **BREAKING:** Removed the `dagzoo request` subcommand and the separate
  request-file contract. Downstream handoff runs now use
  `dagzoo generate --handoff-root <dir>`.
- **BREAKING:** Handoff manifests now use schema
  `dagzoo_generate_handoff_manifest` with a `generate_invocation` block instead
  of request-file echo data.
- `dagzoo generate --handoff-root` now writes generated shards,
  `effective_config.yaml`, and `effective_config_trace.yaml` under
  `<handoff_root>/generated`, publishes `handoff_manifest.json` at the handoff
  root, and rejects stale handoff roots before execution.
- Canonical `dataset_id` and `split_groups.request_run` now incorporate the
  effective `runtime.fixed_layout_target_cells` budget, including values
  applied by hardware policies, so distinct generated corpora no longer share
  the same identity when only auto-batching changes.

## [0.9.11] - 2026-03-14

### Changed

- **BREAKING:** Canonical generated bundle metadata now emits stable
  path-independent `dataset_id` plus `split_groups.request_run` and
  `split_groups.layout_plan` fields for downstream split assignment and corpus
  relocation safety. `request_run` and `dataset_id` now include stable
  non-plan run provenance that changes when emitted data changes outside the
  fixed-layout execution plan, while `layout_plan` remains the narrower
  shared-plan grouping key.
- Shared identity hashing now lives under `src/dagzoo/core/identity.py`, so
  request handoff identity and canonical dataset metadata use the same stable
  JSON normalization rules.
- Output-format docs, packed-shard persistence coverage, deferred-filter
  preservation coverage, and request-run integration tests now treat
  `dataset_id` and `split_groups.*` as the canonical downstream grouping
  surface.

## [0.9.9] - 2026-03-14

### Changed

- **BREAKING:** `dagzoo request` handoff manifests now use schema version `2`
  and add stable request-run and corpus identity, manifest-relative artifact
  paths, checksum metadata, and explicit curated-corpus defaults for downstream
  consumers.
- Request handoff docs now direct portable downstream consumers to the new
  relative artifact contract and clarify that the accepted-only curated corpus
  is the canonical training target.

## [0.9.8] - 2026-03-13

### Changed

- Improved PyPI package metadata with a clearer summary, explicit Markdown
  README rendering metadata, author attribution, search keywords, trove
  classifiers, and project links for docs, issues, repository, and changelog.

## [0.9.7] - 2026-03-12

### Added

- Added widened internal `gp` variants on the canonical fixed-layout path:
  `gp.standard`, `gp.periodic`, and `gp.multiscale`.
- Added `gumbel_softmax` as a sampled parametric activation family for random
  activation plans and fixed-layout execution metadata.
- Added curated GP-focused mechanism-diversity smoke presets for direct
  generation, diversity-audit, and filter-calibration workflows.

### Changed

- Default behavior for runs that sample the existing public `gp` family now
  broadens across internal GP variants; no new CLI flag or config section was
  added.
- Fixed-layout bundle metadata now emits additive
  `mechanism_families.sampled_variant_counts` and
  `mechanism_families.variants_present` fields alongside the existing
  family-level counts.
- Coverage summaries, diversity-audit reports, and filter-calibration reports
  now expose additive GP variant observability through
  `mechanism_family_summary.sampled_variant_counts` and
  `mechanism_family_summary.dataset_presence_rate_by_variant`.
- Fixed-layout metadata schema version is now `6` to reflect the widened
  `mechanism_families` payload on generated bundles.
- Public docs and roadmap guidance now treat shipped `piecewise` as the control
  path and widened `gp` as the active RD-011 candidate path.

## [0.9.6] - 2026-03-11

### Added

- Added an opt-in `piecewise` mechanism family to the fixed-layout sampler and
  batched execution path. It is available only through
  `mechanism.function_family_mix.piecewise` in this release, paired with at
  least one explicit branch family; the default sampler order is unchanged.
- Added curated mechanism-diversity smoke presets plus a dedicated feature
  guide for `piecewise` generate, diversity-audit, and filter-calibration
  workflows.

### Changed

- Fixed-layout bundle metadata now emits `mechanism_families` with realized
  sampled family counts, families present, and total realized function-plan
  count for each dataset bundle.
- Coverage summaries, diversity-audit reports, and filter-calibration reports
  now emit `mechanism_family_summary` in `summary.json` and render the same
  realized mechanism-family usage in `summary.md`.
- Fixed-layout metadata schema version is now `5` to reflect the new
  `mechanism_families` payload on generated bundles.

## [0.9.5] - 2026-03-11

### Added

- `dagzoo request` now writes a versioned `handoff_manifest.json` artifact at
  the request `output_root`, giving downstream consumers one machine-readable
  entrypoint for the curated filtered corpus, effective-config traceability,
  filter summaries, and request echo metadata.

### Changed

- `dagzoo request` CLI output now prints the handoff-manifest path alongside
  the existing generated/filter artifact locations.
- `dagzoo request` handoff manifests now report end-to-end wall-clock
  deferred-filter stage timing for request runs instead of the filter
  implementation's compute-only timing fields.
- `dagzoo request` now publishes `handoff_manifest.json` atomically via a
  staged temp file plus rename, avoiding truncated-manifest races for
  downstream consumers that watch the request `output_root`.
- Request workflow docs now describe the request-run artifact layout and the
  one-way `dagzoo -> tab-foundry` smoke handoff flow; closed-loop downstream
  feedback remains explicitly out of scope.

## [0.9.4] - 2026-03-11

### Changed

- Cleaned up internal technical debt by removing trivial dead wrappers,
  centralizing release-risk review policy, and adding stale docs-path checks to
  the repo verification workflow.
- Regrouped the fixed-layout execution stack under `src/dagzoo/core/fixed_layout/`
  and consolidated shared numeric helpers plus random matrix sampling under
  `src/dagzoo/math/`, removing the old flat module layout.
- Added shared test helpers for repo config loading, YAML/config file writing,
  and script-module loading, then migrated repeated test setup to those shared
  helpers.
- Normalized import ordering and sorted exported symbol lists across the repo,
  and refreshed docs references to the current `src/dagzoo/cli/` and
  `src/dagzoo/config/` package layout.

## [0.9.3] - 2026-03-11

### Added

- Added `dagzoo request --request <path>` as a request-file execution workflow
  that resolves a v1 request into the canonical `generate -> deferred filter`
  pipeline.

### Changed

- Request execution now writes artifacts under one stable `output_root`
  layout with `generated/`, `filter/`, and `curated/` subdirectories, and
  persists request effective-config plus resolution-trace artifacts under
  `generated/`.
- Config resolution now supports request-owned precedence for
  default/smoke profiles, public `rows` overrides, missingness-profile
  overlays, and `output_root` mapping to request-run generation output.

### Fixed

- `dagzoo request` now loads its built-in default and missingness preset
  configs from packaged wheel resources instead of repo-relative paths, so
  installed consumers can run request files outside a source checkout.
- `dagzoo request` now preserves the smoke profile envelope under CUDA hardware
  policies by reapplying the smoke caps before request config validation and by
  capping smoke rows before one-time run realization.
- Non-smoke `dagzoo request` runs now reapply request-owned `rows` specs after
  hardware-policy transforms, so supported CUDA policies preserve the same row
  precedence as `dagzoo generate --rows`.
- `dagzoo request` now converts malformed request YAML into the normal
  argparse-style usage error path instead of printing a traceback.

## [0.9.2] - 2026-03-11

### Changed

- Split the monolithic `dagzoo.cli`, `dagzoo.config`, `dagzoo.bench.suite`,
  `dagzoo.filtering.deferred_filter`, and fixed-layout core modules into
  smaller internal packages/modules while preserving the existing public CLI,
  config import surface, benchmark summary contracts, and deferred-filter
  artifact schema.
- Extracted fixed-layout batching, execution-semantic sampling, and runtime
  preparation/grouping helpers into dedicated internal modules so the core
  generation paths keep one canonical implementation without parallel legacy
  code paths.

## [0.9.1] - 2026-03-10

### Added

- Added `dagzoo filter-calibration` as a threshold-sweep workflow for one
  filter-enabled config, with consolidated `summary.json` / `summary.md`
  artifacts that rank accepted-corpus throughput against diversity shift.

### Fixed

- `dagzoo filter-calibration` now preserves distinct candidate labels for
  fine-grained threshold sweeps, reconstructs candidate rows without relying
  on label-keyed maps, and keeps full normalized threshold precision in CLI
  and markdown status output.
- `dagzoo diversity-audit` now uses one shared probe seed from the baseline
  config across the baseline and every variant, so seed drift is no longer
  reported as a diversity regression.
- `dagzoo diversity-audit` and `dagzoo filter-calibration` now reuse one
  baseline-derived diagnostics coverage aggregation config for the whole run,
  so diagnostics-only settings drift cannot create synthetic diversity shifts.
- `dagzoo diversity-audit` and `dagzoo filter-calibration` now reject
  non-finite or swapped `--warn-threshold-pct` / `--fail-threshold-pct`
  values so CLI guardrails cannot be silently disabled or inverted by bad
  input.
- `dagzoo filter-calibration` now rejects non-finite or out-of-range baseline
  `filter.threshold` values from the resolved config before building sweep
  candidates or artifacts.

## [0.9.0] - 2026-03-10

### Removed

- Replaced the legacy `dagzoo diversity-audit` local/equivalence and arm-set
  workflow. The command no longer accepts `--config`, `--phase`, `--arm-set`,
  `--num-datasets-per-arm`, `--seed`, `--n-seeds`, `--n-rows`, `--n-cols`,
  `--out-dim`, `--nn-degenerate-trials`, `--exact-affine-rmse`,
  `--near-cosine`, `--near-affine-rmse`, `--meaningful-threshold-pct`,
  `--baseline`, or `--save-baseline`.
- Removed legacy diversity-audit artifacts such as `run_summary.json` and
  `equivalence_report.json`; the rewritten audit now persists only
  `summary.json` and `summary.md`.

### Changed

- Rewrote `dagzoo diversity-audit` as a baseline-vs-variant accepted-corpus
  comparison workflow driven by `--baseline-config` plus one or more
  `--variant-config` values.
- Replaced the monolithic effective-diversity implementation with a package
  built around shared corpus probes, accepted-corpus coverage summaries, and
  per-variant diversity shift scoring.
- Diversity-audit now resolves one shared probe size from the baseline config
  for the whole run, replays the measured corpus in a second streaming pass
  instead of buffering full corpora in memory, and treats
  `insufficient_metrics` as a non-passing audit result for
  `--fail-on-regression`.

## [0.8.2] - 2026-03-10

### Changed

- Benchmark preset results now expose
  `filter_accepted_datasets_per_minute` so accepted-corpus throughput is a
  first-class metric alongside filter-stage throughput and dataset-level
  acceptance yield.
- Newly saved benchmark baselines now persist
  `filter_accepted_datasets_per_minute` in the default gating metric set, so
  `--baseline` and `--fail-on-regression` catch accepted-corpus throughput
  regressions by default.
- Added a public filter-enabled smoke benchmark preset and documented the
  canonical benchmark command plus the summary fields to inspect for accepted
  throughput and acceptance/rejection yield.

## [0.8.1] - 2026-03-10

### Changed

- Fixed deferred-filter packed-split replay so one Parquet record batch can
  contain adjacent datasets with different feature widths without raising a
  false ragged-shape error.
- Benchmark preset results now replace the misleading
  `accepted_datasets_measured` field with explicit filter-stage dataset-yield
  telemetry: accepted count, rejected count, acceptance rate, and rejection
  rate. CLI and markdown benchmark summaries now print the new dataset-level
  filter accept/reject percentages alongside the existing attempt-level filter
  rejection pressure metrics.

## [0.8.0] - 2026-03-10

### Removed

- Deleted dead `scripts/benchmark-issue148-cuda-desktop.sh` and its
  `configs/benchmark_filter_enabled.yaml` (issue closed).
- Removed the `dagzoo filter --config` fallback path; deferred filtering now
  replays strictly from embedded shard metadata.
- Removed automatic MPS-to-CPU retry in fixed-layout generation; backend
  runtime failures now surface directly.
- Removed legacy untyped fixed-layout execution-plan coercion from runtime
  generation paths.
- Removed lazy `dagzoo.functions` `apply_*` re-exports.
- Removed trivial `_standardize()` wrapper in `random_functions.py`; callers
  now use `sanitize_and_standardize()` directly.
- Removed redundant local `import math as _math` inside `sanitize_json()`.

### Added

- Clarifying comment on `correction=0` in `missingness._standardize_columns()`.

### Changed

- Broke the `layout` / `node_pipeline` / `fixed_layout_batched` dependency
  cycle by moving fixed-layout node specs onto one typed converter contract.
- Centralized generator-config cloning semantics under
  `clone_generator_config(...)` and reused shared packed-parquet writer helpers
  in deferred filtering.
- `device_fallback_reason` remains in emitted metadata for contract stability,
  but current generation leaves it `null`.

## [0.7.0] - 2026-03-10

### Changed

- Removed the legacy `dagzoo.rng.SeedManager` and `dagzoo.rng.offset_seed32`
  compatibility helpers so `KeyedRng` is now the repo's only semantic RNG
  surface.
- Benchmark throughput, latency, microbench, lineage-guardrail, and
  reproducibility helpers now derive seeds from explicit keyed benchmark
  namespaces instead of offset formulas and manual-seeded compatibility paths.
- Benchmark suite outputs now distinguish exact reproducibility from
  workload-shape stability by emitting both content and workload match signals
  for reproducibility checks.

## [0.6.7] - 2026-03-10

### Changed

- Canonical runtime orchestration, layout sampling, correlated sampling,
  dataset-level noise-family selection, split permutation, missingness, and
  postprocess now derive randomness from explicit keyed namespaces instead of
  offset-only stage seeds and shared compatibility wrappers.
- Fixed-layout plan preparation now keys run-level rows realization, shared
  plan candidates, node-spec sampling, and dataset attempt roots explicitly,
  and canonical bundle metadata now records exact keyed subtree replay paths in
  `keyed_replay` while preserving `dataset_seed` and `layout_plan_seed` as
  stable child-seed identifiers for deferred filter replay and diagnostics.
- Classification fixed-layout replay now keeps cached retry planning and actual
  retry execution on aligned keyed roots, falling back to scalar generation for
  runs that require nonzero per-dataset retry attempts so replay stays correct
  with the current chunk-scoped raw batch kernels.

## [0.6.5] - 2026-03-09

### Changed

- Fixed-layout typed plan sampling and execution now derive randomness from
  semantic keyed namespaces instead of shared ambient draw order, so regrouping
  parent execution, converter execution, and nested function-plan sampling no
  longer perturbs sibling randomness.
- `apply_random_function()`, `apply_multi_function()`, `apply_node_pipeline()`,
  numeric/categorical converter helpers, and random-points sampling now route
  generator-based entrypoints through keyed `plan` and `execution` roots while
  preserving their public call surface.
- Fixed-layout batch execution now keys batched matrix, activation, parent,
  source, product, tree, and converter draws by semantic role, and grouped vs
  split converter execution paths now remain deterministic for the same keyed
  root.

## [0.6.4] - 2026-03-09

### Added

- Added `dagzoo.rng.KeyedRng`, a keyed namespace helper for deriving
  deterministic child seeds and device-aware `torch.Generator` instances from
  one base seed plus a semantic path.

### Changed

- `SeedManager.child()` and `SeedManager.torch_rng()` now delegate through the
  keyed RNG substrate while preserving the existing `derive_seed()` and
  `offset_seed32()` compatibility surface.

## [0.6.3] - 2026-03-09

### Changed

- Deferred filter replay now streams packed shard inputs dataset-by-dataset
  instead of materializing whole shard tables in Python memory, stages
  manifest/curated outputs until shard and run validation succeed, and
  validates packed split ordering and coverage during replay.
- Deferred filter now refuses stale `--out` manifest/summary directories from a
  prior run and rejects symlinked lineage entries when copying curated shard
  lineage artifacts.
- Canonical fixed-layout classification preparation now caches the first valid
  retry attempt per dataset for the requested run so singleton retry fallback
  can skip already-proven invalid attempts without changing emitted metadata or
  dataset artifact layout.
- Removed several runtime-unused private helper paths and duplicate scalar
  implementations across random functions, postprocess aliases, node pipeline,
  converters, and sampling helpers to keep the active execution surface smaller
  and easier to interpret.
- Test execution now relies on the installed package layout instead of
  force-inserting `src/` onto `sys.path`, and CI now runs a Python `3.13` /
  `3.14` matrix with a `vulture` dead-code check plus manual `workflow_dispatch`
  support.

## [0.6.2] - 2026-03-08

### Changed

- Fixed-layout auto batch sizing now accepts an optional
  `runtime.fixed_layout_target_cells` override, and the built-in benchmark CPU,
  CUDA desktop, and CUDA H100 presets pin explicit target-cell budgets instead
  of relying on the single global default.
- CUDA config resolution now fills `runtime.fixed_layout_target_cells` from a
  memory-scaled floor for desktop, datacenter, and H100-class hardware when the
  config leaves that field unset, even when `--hardware-policy none` is used.
  Explicit preset or user-configured values are preserved. `cuda_tiered_v1`
  reuses the same floor logic alongside its broader tier-specific config
  transforms, including the unknown-CUDA fallback tier.
- Added an internal throughput sweep helper that benchmarks a small grid of
  `fixed_layout_target_cells` values and reports the best-performing candidate
  for the detected hardware tier. CUDA sweeps now start at the active
  memory-scaled floor/current target and explore upward instead of using a
  stale low static grid.

### Breaking

- **BREAKING:** Emitted dataset metadata now serializes the optional
  `config.runtime.fixed_layout_target_cells` field inside the persisted config
  payload when present. Consumers that validate the nested config payload
  against an exact schema should allow that key.

## [0.6.1] - 2026-03-08

### Changed

- Fixed-layout mixed-noise runs no longer fall all the way back to singleton
  dataset generation when a chunk samples multiple runtime noise families.
  Chunks now stay batched by subgrouped noise contract and preserve original
  dataset order through finalization.
- Fixed-layout batched tree execution now samples ODT splits, leaf indices, and
  leaf-value gathers in tensorized batch form instead of using per-dataset
  Python loops.
- Fixed-layout batched activation and quadratic matrix paths now avoid
  per-dataset or per-output Python loops, flatten fixed-activation application
  across batch dimensions, and reduce repeated mean/std reduction overhead in
  batched standardization.

## [0.6.0] - 2026-03-07

### Changed

- `generate_one`, `generate_batch`, and `generate_batch_iter` now sample one
  internal fixed-layout plan per run and execute generation through the
  fixed-layout engine instead of the old per-dataset dynamic layout path.
- Variable `dataset.rows` modes now resolve once per run. All datasets emitted
  from the same `generate` run share the same realized `n_train` / `n_test`
  split, and `effective_config.yaml` records that realized split.
- `dagzoo benchmark` now uses the same canonical fixed-layout generation path as
  `dagzoo generate`, reports `generation_mode="fixed_batched"` on that path,
  and caps smoke-suite `dataset.rows` before run realization.
- The old dynamic executor has been removed. Canonical split/finalization logic
  now lives in shared internal runtime helpers used by fixed-layout execution.
- Parallel-generation scaffolding and worker-partition wiring have been removed
  from the active product surface.

### Breaking

- **BREAKING:** Public generation no longer samples a fresh layout per dataset.
  One run now samples one fixed-layout plan and all emitted datasets in that run
  share its layout lineage and execution-plan provenance.
- **BREAKING:** `dataset.rows` `range` / `choices` no longer vary per dataset
  within a run. They now realize once per run.
- **BREAKING:** Generated bundles from the canonical `generate_*` APIs now carry
  fixed-layout provenance metadata (`layout_mode`, `layout_signature`,
  `layout_plan_seed`, `layout_plan_signature`, and execution-contract fields).
- **BREAKING:** The public `dagzoo fixed-layout ...` CLI workflow has been
  removed, and the top-level fixed-layout generation exports are no longer part
  of the public Python package surface.
- **BREAKING:** `runtime.worker_count` and `runtime.worker_index` are no longer
  supported runtime config keys for generation, benchmark, or diversity-audit
  flows.
- **BREAKING:** Local parallel-generation orchestration and the old dynamic
  generation executor have been removed rather than left as dormant product
  paths.

## [0.5.7] - 2026-03-07

### Added

- Added `dagzoo fixed-layout sample` and `dagzoo fixed-layout generate` CLI
  subcommands for sampling reusable fixed-layout plan artifacts and generating
  datasets from saved plans.

### Changed

- Upgraded fixed-layout generation to freeze node execution plans in the saved
  plan artifact and generate raw graph tensors in batched `N x rows x features`
  chunks before per-dataset finalization.
- Fixed-layout plan artifacts now serialize `node_plans`,
  `plan_signature`, `execution_contract`, and `schema_version: 3`, and
  emitted fixed-layout bundles now include `metadata.layout_plan_signature`.
- Built-in CPU benchmark runs (`dagzoo benchmark --preset cpu`) now expand into
  explicit `1024`, `4096`, and `8192` total-row profiles, use the fixed-layout
  batched generator by default for those row profiles, and report
  `generation_mode="fixed_batched"` plus explicit dataset row counts in
  benchmark results.
- Fixed-layout raw generation now uses a chunk-scoped batched RNG contract and
  grouped converter execution in the fixed-layout batched engine.
- Built-in CPU benchmark fixed-layout runs now pin one internal fixed-layout
  batch size per preset run so throughput, reproducibility, and lineage
  guardrail generation use the same chunking contract.
- Emitted fixed-layout bundles now include
  `metadata.layout_plan_schema_version` and
  `metadata.layout_execution_contract`.
- Fixed-layout replay now uses the current requested device instead of the
  plan's sampling backend provenance, and `dagzoo fixed-layout generate --device ...` now controls replay backend selection correctly.
- Fixed-layout generation again retries on CPU when `--device auto` resolves to
  `mps` and the batched fixed-layout runtime hits an unsupported MPS op.

### Breaking

- **BREAKING:** Fixed-layout plan artifacts changed schema in this release.
  Older serialized plan payloads without execution plans or plan signatures are
  no longer accepted by `generate_batch_fixed_layout(_iter)` or the new
  `dagzoo fixed-layout generate` CLI.
- **BREAKING:** The built-in CPU benchmark preset no longer measures the
  dynamic-layout generator path by default and no longer emits a single CPU
  result row. It now emits `cpu_rows1024`, `cpu_rows4096`, and `cpu_rows8192`
  results; benchmark artifacts distinguish `generation_mode="fixed_batched"`
  and `generation_mode="dynamic"` and include explicit dataset row counts.
- **BREAKING:** Fixed-layout plan artifacts now serialize `schema_version: 3`
  and `execution_contract: "chunk_batched_v1"`.
- **BREAKING:** Fixed-layout outputs are no longer batch-size independent. The
  current contract is deterministic for the same `plan + run seed + batch_size`,
  but changing `batch_size` may change the emitted dataset values.

## [0.5.6] - 2026-03-07

### Changed

- Improved practical generation throughput in shared Torch generation paths by
  caching per-layout parent index lists, removing redundant latent-width checks
  in the node pipeline, vectorizing quadratic random-function evaluation, using
  `torch.cdist()` for discretization nearest-center selection, and reducing
  intermediate allocations in multi-input aggregation.
- These changes affect both single-worker generation and the local CPU
  multi-worker benchmark path because they optimize shared generation kernels
  instead of benchmark-only accounting.

## [0.5.5] - 2026-03-06

### Added

- Added process-based local CPU multi-worker benchmark orchestration for
  `dagzoo benchmark` when `runtime.worker_count > 1`,
  `runtime.worker_index == 0`, and `--preset custom` is used.

### Changed

- Benchmark throughput, reproducibility, and lineage guardrail generation paths
  now use the local multi-worker iterator when multi-worker benchmark mode is
  active.
- Local multi-worker benchmark fan-out now caps to host CPU capacity before the
  benchmark coordinator spawns worker processes and bounded IPC queues.
- The process-based local multi-worker coordinator now uses bounded per-worker
  result queues, preserving deterministic ordering without deadlocking on slow
  low-index workers, tolerating clean-exit result/control message reordering,
  cleaning up IPC queues on worker-startup failures, and avoiding unbounded
  out-of-order result accumulation.
- True local multi-worker benchmark runs now coerce `runtime.device: auto` to
  CPU after effective fan-out resolution, while effectively single-worker runs
  keep their requested device behavior.
- Multi-worker benchmark summaries now report latency fields and
  `micro_generate_one_ms` as unavailable when the run actually uses multiple
  active worker partitions, avoiding misleading single-worker timings.
- True local multi-worker benchmark summaries now also report memory fields as
  unavailable, because coordinator-only RSS and CUDA counters do not represent
  child worker processes.
- Effectively single-worker benchmark runs now stay on the sequential
  generation path for both warmup and measured passes even when
  `runtime.worker_count > 1`, avoiding unnecessary coordinator overhead for
  tiny runs.

### Breaking

- **BREAKING:** True local multi-worker `dagzoo benchmark` runs remain CPU-only
  in this release; effectively single-worker runs keep their requested device,
  but explicit `cuda`/`mps` requests are still rejected once effective fan-out
  exceeds one local worker.
- **BREAKING:** `dagzoo generate` still does not orchestrate peer workers, and
  write-enabled multi-worker generate runs remain blocked until shard-writing
  coordination lands.

## [0.5.4] - 2026-03-06

### Changed

- `dagzoo benchmark` now rejects `--device` when multiple `--preset` values are
  selected instead of silently ignoring the override.
- Docs tooling now treats `site/public/` as the canonical built Hugo output in
  local workflows and CI, removing the previous `public/` vs `site/public/`
  ambiguity.
- Developer docs now explicitly document the docs wrapper model, current
  coverage-report omission for `src/dagzoo/bench/*`, and the current
  MPS-to-CPU hardware-tier mapping.
- Coverage reporting now includes `src/dagzoo/cli.py`; only `src/dagzoo/bench/*`
  remains omitted in this pass.
- Script documentation now matches the actual reference-fetch workflow and
  documents the standalone effective-diversity audit helper.
- Added a safe local-artifact cleanup helper for ignored runtime/docs outputs.

## [0.5.3] - 2026-03-05

### Added

- Added worker-partition runtime schema fields:
  `runtime.worker_count` and `runtime.worker_index`.
- Added deterministic round-robin worker partitioning for
  `dagzoo generate --no-dataset-write`, where `num_datasets` remains the global
  dataset-index space and each worker covers a disjoint subset.

### Breaking

- **BREAKING:** `dagzoo generate` now rejects write-enabled runs when
  `runtime.worker_count > 1`. Multi-worker mode is currently supported only with
  `--no-dataset-write` until shard-writing coordination lands.
- **BREAKING:** `dagzoo benchmark` and `dagzoo diversity-audit` now reject
  configs with `runtime.worker_count > 1` until those flows become
  partition-aware.

## [0.5.2] - 2026-03-05

### Fixed

- Benchmark suite generation paths now consistently use a filter-disabled
  runtime config across throughput, latency, reproducibility, microbenchmarks,
  lineage guardrails, and missingness/shift/noise control runs. Presets with
  `filter.enabled=true` no longer crash benchmark execution after throughput.
- Benchmark `filter_retry_dataset_rate` now uses a denominator from the same
  sampled population as replayed filter rejections when deferred filter-stage
  replay metrics are active, preventing systematic underreporting on sampled
  suites.

## [0.5.1] - 2026-03-05

### Added

- Added a new deferred filtering CLI stage:
  `dagzoo filter --in <shard_dir> --out <out_dir>` with optional
  accepted-only curated shard output via `--curated-out`.
- Added deferred filter manifest (`filter_manifest.ndjson`) and summary
  (`filter_summary.json`) artifacts for acceptance auditing and throughput
  reporting.

### Changed

- Generated bundle metadata now always marks deferred filter state using
  `metadata.filter.mode` and `metadata.filter.status` (`not_run` immediately
  after generation).
- Benchmark filter stage throughput now replays deferred filtering over sampled
  bundles and reports replay-derived filter attempts/rejections when the filter
  stage is enabled.

### Breaking

- **BREAKING:** Inline filtering was removed from generation. `dagzoo generate`
  now errors when `filter.enabled=true`; users must run `dagzoo filter` as a
  separate stage.

## [0.5.0] - 2026-03-03

### Changed

- **BREAKING:** Renamed package from `dagsynth` to `dagzoo`. All imports change
  from `from dagsynth ...` to `from dagzoo ...`.
- **BREAKING:** Lineage schema name changed from `dagsynth.dag_lineage` to
  `dagzoo.dag_lineage`. Existing Parquet files with the old schema name will not
  validate against the new constant.

## [0.4.9] - 2026-03-04

### Fixed

- Benchmark runtime guardrail control runs now execute the same per-bundle
  callback instrumentation (stage-sample update + throughput-pressure
  collection) as the primary throughput run, removing callback-overhead skew
  from missingness/shift/noise runtime degradation comparisons.
- Benchmark stage-sample bundles are now released immediately after write/filter
  stage probes, reducing avoidable bundle/tensor retention before latency
  collection and guardrail control runs.
- Filter stage throughput now uses benchmark-only runtime timing captured on
  pre-filter tensors (`bundle.runtime_metrics.filter_elapsed_seconds`), keeping
  emitted metadata deterministic while avoiding replay skew from postprocessing
  and missingness-transformed outputs.

## [0.4.8] - 2026-03-04

### Fixed

- Scalar RNG draws (`torch.randint`, `torch.rand`) now explicitly target the
  generator's device via new `rand_scalar`/`randint_scalar` helpers, fixing
  failures when the generator lives on a non-CPU device.
- Generation retry exception logging now includes the exception message for
  easier debugging.

### Changed

- Benchmark suite summaries now emit stage-level throughput metrics for
  generation, parquet writing, and filter replay:
  `generation_datasets_per_minute`, `write_datasets_per_minute`, and
  `filter_datasets_per_minute` (nullable when filtering is disabled).
- Benchmark suite summaries now emit throughput-pressure metrics required for
  end-to-end interpretation, including attempt pressure and both filter
  rejection views:
  `filter_rejection_rate_attempt_level` and `filter_retry_dataset_rate`.
- Benchmark baseline defaults now gate on stage throughput metrics in addition
  to `datasets_per_minute`.
- Benchmark CLI single-line preset summaries and markdown reports now include
  stage throughput and filter rejection telemetry columns.
- Added RTX 3060 to the hardware peak FLOPS lookup table.

### Breaking

- Generated bundle metadata now includes a new additive
  `metadata.generation_attempts` object. Consumers with strict metadata schemas
  must allow this field.

## [0.4.7] - 2026-03-04

### Fixed

- Generate config resolution now applies CLI `--rows` overrides before final
  post-policy validation, preventing premature failures when CUDA tiered
  policies raise `dataset.n_test`.
- Benchmark preset resolution now rejects `dataset.rows` before hardware-policy
  transforms, so users consistently receive the intended unsupported-feature
  error message.

## [0.4.6] - 2026-03-04

### Added

- Added unified total-row control for generation via `dataset.rows` and
  `dagzoo generate --rows ...`.
- `dataset.rows` supports:
  - fixed totals (for example `1024`)
  - ranges (for example `400..60000`)
  - choice lists (for example `1024,2048,4096` or YAML list form)
- Variable rows specs now resolve deterministically per dataset seed lineage.

### Changed

- When `dataset.rows` is active, split resolution keeps `dataset.n_test` fixed
  and derives `n_train = total_rows - n_test`.
- Fixed-layout APIs now require fixed row modes; variable `dataset.rows`
  (range/choices) fail fast with clear compatibility errors.
- Benchmark config resolution now rejects `dataset.rows` (benchmark paths remain
  on explicit `dataset.n_train` / `dataset.n_test` for now).
- Generated metadata config snapshots now reflect realized split sizes for each
  emitted bundle.

### Removed

- Retired curriculum shell workflow: removed
  `scripts/generate-curriculum.sh` and its test coverage.

### Breaking

- **BREAKING:** Curriculum shell users must migrate to `dataset.rows` /
  `--rows` workflows for staged row-envelope generation.

## [0.4.5] - 2026-03-04

### Fixed

- Effective-diversity baseline regression comparison now fails fast on
  incompatible baselines (schema/version/suite/arm-set/config-fingerprint
  mismatch and missing evaluated-arm baseline entries) instead of silently
  passing.
- Effective-diversity scale audit now builds combined ablation arms from
  runtime-applicable mappings only, so unavailable activations do not force
  combined-arm skips.

## [0.4.4] - 2026-03-04

### Changed

- Added a first-class `dagzoo diversity-audit` CLI command that runs local
  overlap checks and dataset-scale overlap-impact ablations with reproducible
  run artifacts.
- Effective-diversity local auditing now reads activation names from runtime
  source-of-truth (`fixed_activation_names`) to avoid stale claim coverage when
  activation pools change.
- Added scale-impact baseline persistence and regression comparison workflow for
  repeated effective-diversity audits as the repository evolves.

## [0.4.3] - 2026-03-03

### Fixed

- Corrected mechanism-family sampling to iterate only across positive-probability
  families so `mechanism.function_family_mix` remains a strict hard mask even
  when RNG draws are exactly zero.

## [0.4.2] - 2026-03-03

### Changed

- Added `mechanism.function_family_mix` config support with strict
  normalization/validation and active runtime enforcement:
  unspecified families are excluded from sampling, and shift
  `mechanism_scale` reweights only within enabled families.
- Added `swiglu` to the global random activation pool as a shape-preserving
  variant (`x * silu(x)`).

### Breaking

- Generated bundle `metadata.config` now includes the top-level `mechanism`
  section (persisted config-schema extension). Consumers with strict
  schema expectations must allow this new section.

## [0.4.1] - 2026-03-03

### Changed

- Stabilized smoke benchmark CI behavior by disabling lineage runtime-overhead
  guardrail gating for smoke-size samples while retaining lineage metadata
  coverage validation.
- Lineage runtime degradation guardrails remain enabled for larger suites (for
  example `standard`) where sample sizes are sufficient for stable runtime
  comparisons.

## [0.4.0] - 2026-03-03

### Changed

- Renamed shift terminology in config/runtime/metadata:
  - `shift.mode` (was `shift.profile`)
  - `shift.variance_scale` (was `shift.noise_scale`)
  - `variance_sigma_multiplier` (was `noise_sigma_multiplier`) in emitted shift metadata
- Renamed noise amplitude key to `noise.base_scale` (was `noise.scale`) and standardized emitted dataset metadata under `noise_distribution` (was `noise`).
- Renamed benchmark preset terminology across config/CLI/summary payloads:
  - `benchmark.preset_name` (was `benchmark.profile_name`)
  - `benchmark.presets` (was `benchmark.profiles`)
  - summary keys `preset_results` / `preset_key` (was `profile_results` / `profile_key`)
- Renamed benchmark script/docs terminology from profile-centric wording to preset/tier wording.
- Replaced internal layout dict contracts with `LayoutPlan` dataclass wiring in generation/layout/metadata paths.
- Added explicit device-resolution metadata fields on generated bundles:
  - `requested_device`
  - `resolved_device`
  - `device_fallback_reason`
- Renamed diagnostics target bridge module:
  - `meta_targets.py` -> `diagnostics_targets.py`
  - `build_coverage_aggregation_config` -> `build_diagnostics_aggregation_config`

### Breaking

- `dagzoo benchmark --profile ...` was removed; use `--preset ...`.
- `dagzoo generate --no-write` was removed; use `--no-dataset-write`.
- Config files using `shift.profile`, `shift.noise_scale`, `noise.scale`, `benchmark.profile_name`, or `benchmark.profiles` must be migrated to the new keys above.
- Dataset metadata consumers must read `metadata.noise_distribution` (not `metadata.noise`) and `metadata.shift.mode` (not `metadata.shift.profile`).

## [0.3.3] - 2026-03-03

### Changed

- Refactored generation config validation into explicit staged validators in `src/dagzoo/config.py`:
  stage 1 field normalization/typing, stage 2 cross-field constraints, and stage 3 post-override revalidation.
- `resolve_generate_config()` and `resolve_benchmark_profile_config()` now rely on `GeneratorConfig.validate_generation_constraints()` as the single post-override revalidation pass.
- Classification configs now enforce split feasibility for both `dataset.n_classes_min` and `dataset.n_classes_max` (previously only `n_classes_min` was checked).
- `runtime.device: null` is normalized to `auto` during staged validation.
- Added/updated tests for ordering-sensitive constraints and override-driven revalidation behavior.
- Documented validation stages in `docs/config-resolution.md`.

## [0.3.2] - 2026-03-03

### Changed

- Added internal typed layout contracts in `core/layout_types.py`:
  `LayoutPayload`, `FeatureType`, `ConverterKind`, `MechanismFamily`, and
  `AggregationKind`.
- Updated generation/layout/metadata wiring to use typed layout payloads
  instead of untyped `dict[str, Any]` signatures.
- Node pipeline now exposes explicit feature-key parsing (`parse_feature_key`)
  and uses helper boundaries for latent-dimension resolution and latent padding.
- Multi-parent function composition now supports optional explicit aggregation
  selection while preserving random-selection default behavior.
- Clarified DAG adjacency direction contract in generation/docs:
  `adjacency[src, dst]` means `src -> dst`; parent lookup is column-based.
- Standardized seed derivation callsites in `core/dataset.py` behind named
  helper functions for attempts, node-spec sampling, and split permutations.
- Internal naming cleanup across layout/dataset/node codepaths for clearer
  node/feature/dimension semantics.
- No CLI flag changes, no lineage schema version changes, and no dataset
  metadata contract changes.

## [0.3.1] - 2026-03-03

### Changed

- Added a shared config-resolution engine (`core/config_resolution.py`) used by both `generate` and benchmark profile execution paths.
- Added per-run field-level effective config trace artifacts:
  - `generate`: `effective_config_trace.yaml`
  - `benchmark`: `effective_configs/*_trace.yaml`
- Added CLI flags `--print-resolution-trace` for `generate` and `benchmark`.
- Benchmarks now include per-profile `effective_config_trace` payloads in summary output.
- Added canonical precedence documentation: `docs/config-resolution.md`.
- **Breaking:** `dataset.categorical_ratio_min` and `dataset.categorical_ratio_max` now validate strictly in `[0, 1]` at config load time.
- Updated shipped config presets to use in-range categorical ratio bounds (`0.0..1.0`).

## [0.3.0] - 2026-03-03

### Changed

- **Rename: `cauchy-generator` is now `dagzoo`.** Package name, CLI entrypoint, Python import paths, and all internal references updated. The CLI command is now `dagzoo` (was `cauchy-gen`). Python imports change from `from cauchy_generator.…` to `from dagzoo.…`.
- **Breaking:** Lineage schema name changed from `cauchy_generator.dag_lineage` to `dagzoo.dag_lineage`. Existing Parquet files with embedded lineage metadata will fail validation against the new schema name.
- `sample_cauchy_dag` renamed to `sample_dag`; `graph/cauchy_graph.py` renamed to `graph/dag_sampler.py`.
- Noise rollout presets for generate/benchmark smoke workflows (`configs/preset_noise_*.yaml`)
- Noise workflow script wrapper (`scripts/generate-noise.sh`)
- Noise benchmark guardrail reporting (`noise_guardrails`) for runtime delta vs gaussian control and metadata validity
- Noise feature guide (`docs/features/noise.md`) and usage/workflow documentation updates
- Benchmark profile summaries now surface noise guardrail status in CLI and Markdown reports
- **Breaking:** removed `noise.family=legacy`; default noise family is now explicit `gaussian`.
- **Breaking:** removed config/runtime compatibility knobs `runtime.hardware_aware`, CLI `--no-hardware-aware`, and graph aliases `graph.n_nodes_log2_min/max`.
- **Breaking:** `shift.mode` no longer accepts boolean aliases; only explicit string values are allowed.
- Hardware detection and policy are now separated: `hardware.py` handles detection and `hardware_policy.py` handles explicit scaling policy.
- Added hardware policy registry APIs (`apply_hardware_policy`, `register_hardware_policy`, `list_hardware_policies`) with immutable policy application semantics.
- Added CLI `--hardware-policy` and `--print-effective-config` options; generate/benchmark runs now persist resolved effective config artifacts.
- Fixed-layout plan schema now requires a non-null `compatibility_snapshot` payload.

## [0.2.0] - 2026-03-02

### Added

- Shift profile schema, sampling integration, and metadata/diagnostics observability across generation and benchmarks
- Configurable noise family schema/samplers (`legacy`, `gaussian`, `laplace`, `student_t`, `mixture`) with profile/preset coverage
- Many-class generation envelope and class-aware filtering/diagnostics presets for benchmark and generate workflows
- Fixed-layout batch generation APIs with emitted schema stability guarantees
- Curriculum shell runner (`scripts/generate-curriculum.sh`) for staged dataset batches with manifest output
- Additional guidance docs and feature references (`docs/how-it-works.md`, `docs/design-decisions.md`, `docs/glossary.md`, `docs/output-format.md`, `docs/usage-guide.md`)

### Changed

- Dataset generation flow now favors fixed-layout execution paths instead of curriculum staging APIs
- Tiny split/postprocess and DAG sampling steps run on CPU to avoid underutilized GPU kernels
- Benchmark/docs workflows now include shift/noise/many-class/curriculum shell guidance and presets

### Fixed

- Noise sampling stability improvements, including Student-t support on MPS and tighter mixture reproducibility guards
- Fixed-layout compatibility validation hardening for resolved device checks and feature/node bounds
- Curriculum shell manifest accounting/status fixes for aborted and dry-run chunk handling

### Removed

- Legacy steering hard-fail path and associated deprecated hard-fail tests

## [0.1.14] - 2026-02-26

### Added

- Curriculum stage schema/validation and stagewise runtime scaling for feature/node/depth/structure complexity
- Curriculum benchmark presets/guardrails and curriculum metadata diagnostics (`metadata.curriculum.realized_complexity`, `stage_bounds`, `monotonicity_axes`)
- DAG lineage artifact pipeline with compact shard outputs (`adjacency.bitpack.bin`, `adjacency.index.json`) and benchmark lineage guardrails (`lineage_guardrails`)
- Lineage benchmark smoke preset: `configs/preset_lineage_benchmark_smoke.yaml`

### Changed

- Diagnostics extraction now normalizes bundles to CPU and delegates metric computation to `core/steering_metrics.py`, avoiding non-CPU float64 failures (notably MPS)
- Internal refactors removed dataset/benchmark compatibility shims and consolidated duplicated helper/constants wiring across CLI/core modules
- README/roadmap/implementation docs expanded lineage/curriculum workflows and benchmark guardrail guidance

### Fixed

- Curriculum validation hardening across stage bounds, node/depth consistency, singleton DAG handling, and stratified split retry sizing
- Lineage artifact correctness/resilience: assignment alignment with postprocessed features, adjacency type validation, bounded blob descriptors, and robust index flushing on failures
- Benchmark lineage guardrail robustness for non-runtime probe handling and undersized runtime gating

### Removed

- `RuntimeConfig` keys: `generation_engine`, `gpu_name_hint`, `gpu_memory_gb_hint`, `peak_flops_hint`
- `SeedManager.numpy_rng`

### Compatibility

- Configs that still set `runtime.generation_engine` now fail fast during config load (`TypeError`).

## [0.1.13] - 2026-02-24

### Added

- Successful dataset generation now emits versioned DAG lineage payloads (`metadata.lineage`) with adjacency/assignment metadata derived from seeded layout sampling

## [0.1.12] - 2026-02-23

### Added

- Versioned DAG lineage schema validator for `metadata.lineage` (`schema_name=dagzoo.dag_lineage`, `schema_version=1.0.0`) with strict adjacency/assignment checks and backward-compatible optional metadata mode

## [0.1.11] - 2026-02-23

### Added

- Missingness CLI overrides for `generate`: `--missing-rate`, `--missing-mechanism`, `--missing-mar-observed-fraction`, `--missing-mar-logit-scale`, `--missing-mnar-logit-scale`
- Missingness presets: `configs/preset_missingness_mcar.yaml`, `configs/preset_missingness_mar.yaml`, `configs/preset_missingness_mnar.yaml`
- Missingness wrapper script: `scripts/generate-missingness.sh`
- Missingness benchmark guardrails (runtime vs missingness-off control + acceptance checks) surfaced in profile summaries and regression issues

## [0.1.10] - 2026-02-23

### Added

- End-to-end missingness injection in generation/postprocess path with compact metadata summaries for configured and realized rates

## [0.1.9] - 2026-02-23

### Added

- Deterministic MCAR/MAR/MNAR missingness mask samplers with typed config validation coverage

## [0.1.8] - 2026-02-23

### Added

- Architecture documentation with mermaid diagrams (`docs/architecture.md`)
- Torch-native steering metric extractor (`core/steering_metrics.py`) to avoid NumPy conversion during candidate scoring

### Fixed

- Diagnostics coverage aggregation now uses deterministic reservoir sampling per metric with configurable retention cap (`diagnostics.max_values_per_metric`) to bound memory on long runs

## [0.1.7] - 2026-02-22

### Changed

- Eliminated NumPy bottlenecks across generation pipeline: `functions/`, `linalg/`, and converters now use torch-native implementations
- Steering candidate scoring uses torch-native metric path and torch softmax selection to avoid CPU/NumPy round trips on accelerator runs

### Fixed

- Correlated beta sampling no longer reseeds process-wide Torch RNG state; draws now use a local seed-derived generator for reproducibility isolation

## [0.1.6] - 2026-02-22

### Added

- Opt-in soft meta-feature steering with bounded deterministic candidate selection
- New generation CLI controls: `--diagnostics`, `--steer-meta`, and repeatable `--meta-target key=min:max[:weight]`
- Steering metadata payload propagation on generated bundles when steering is enabled
- Benchmark diagnostics collection controls: `dagzoo benchmark --diagnostics [--diagnostics-out-dir ...]`
- Per-profile benchmark diagnostics artifacts and summary pointers under `diagnostics/<sanitized_preset_key>_<hash>/`
- New presets: `configs/preset_diagnostics_on.yaml` and `configs/preset_steering_conservative.yaml`

### Changed

- Added top-level `GeneratorConfig.meta_feature_targets` and `SteeringConfig` with disabled-by-default safe defaults
- Target resolution now merges legacy `diagnostics.meta_feature_targets` with top-level targets (top-level precedence)
- Benchmark markdown reports now surface diagnostics state and artifact pointers per profile
- Script/README benchmark workflow examples now include diagnostics and conservative steering presets

## [0.1.5] - 2026-02-22

### Added

- Diagnostics coverage aggregation reports with bin target-band analysis
- Dataset-level diagnostics metric extractors
- Staged curriculum CLI workflow with presets and documentation
- Three-stage curriculum core implementation
- CI/CD workflow, AGENTS.md docs, and config/test improvements
- Torch determinism tests and tightened postprocess assertions
- Expanded unit coverage and CI coverage checks

### Changed

- Switched to torch-first runtime with streaming generation
- Removed sklearn dependency; replaced with torch-native implementations

### Fixed

- Bin target-band coverage on configured range
- Coverage robustness and band accounting
- Null diagnostics coverage config values handling

## 0.1.0 – 0.1.4

Early development releases. No structured changelog was maintained for these versions.
