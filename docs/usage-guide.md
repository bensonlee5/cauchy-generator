# Advanced Controls

Use this page after you already have a working recipe-based run.

The default public path is:

- `dagzoo recipe list`
- `dagzoo generate --config recipe:<name>`
- `build_dataloader("recipe:<name>", ...)`

This guide covers the custom controls you reach for when the curated recipe
catalog is not enough and you want to author or modify repo-local configs under
`configs/`.

______________________________________________________________________

## Prerequisites

Examples on this page use repo-local configs under `configs/`.

For a repo checkout:

```bash
uv sync --group dev
source .venv/bin/activate
```

For a global CLI install that still uses the curated catalog:

```bash
uv tool install dagzoo
dagzoo recipe list
```

______________________________________________________________________

## 1. Custom generation from YAML

Use this when you want to author a YAML config directly instead of starting from
`recipe:<name>`.

```bash
dagzoo generate --config configs/default.yaml --num-datasets 10 --out data/run1
```

Each generate run writes `effective_config.yaml` and `effective_config_trace.yaml`
under the resolved output directory.
`dagzoo generate` defaults to fully heterogeneous per-dataset plan sampling, so
datasets in the same run may differ in feature schema, lineage assignments, and
target node choice. Set `runtime.layout_mode: stratified` when you want the
generator to batch large heterogeneous runs by exact `(n_rows, n_features)`
strata while keeping per-dataset layouts and execution plans independent. On
Apple hardware, heterogeneous and stratified
`device=auto` now resolves to CPU instead of MPS because the public
heterogeneous path is typically faster there; pass `--device mps` only when you
want to force the MPS backend explicitly.
Inline filtering is removed from `dagzoo generate`. Keep `filter.enabled: false`
for generate flows, then run `dagzoo filter` as a separate replay stage on the
emitted shards when you want acceptance decisions. Generation still reuses
`filter.min_target_*` and `filter.max_attempts` while resampling layouts for
structural validity.
Generate configs must not include `runtime.worker_count` or
`runtime.worker_index`.

______________________________________________________________________

## 2. Deferred filtering (`dagzoo filter`)

Deferred filtering replays structural lineage-validity checks from shard
metadata and lineage artifacts. The public tuning knobs are
`min_target_indegree`, `min_target_relevant_feature_count`, and
`min_target_relevant_feature_fraction`. `dagzoo filter --set` only accepts
`filter.<field>` overrides for those thresholds plus `filter.enabled` and
`filter.max_attempts`. Filter-enabled replay now requires `metadata.lineage` in
the shard sidecars because there is no learned-filter fallback.

```bash
dagzoo filter --in data/run1 --out data/run1_filter
```

Accepted and rejected outputs are written as a separate curated run; generated
shards themselves still start with `metadata.filter.status=not_run`.

______________________________________________________________________

## 3. Total-row control (`dataset.rows` / `--rows`)

Use `dataset.rows` (or CLI `--rows`) to control total rows with one field:

```bash
dagzoo generate --config configs/default.yaml --rows 1024 --num-datasets 10 --out data/run_rows_fixed
dagzoo generate --config configs/default.yaml --rows 400..60000 --num-datasets 25 --no-dataset-write
dagzoo generate --config configs/default.yaml --rows 1024,2048,4096 --num-datasets 25 --no-dataset-write
```

When rows mode is active, `dataset.n_test` stays fixed and `n_train` is derived as:
`n_train = total_rows - n_test`.
For canonical `generate` runs, `range` and `choices` rows modes are realized
once per run, not once per dataset.

To migrate prior train-row stages:

- old train range `A..B` with fixed `n_test=T` -> new total-row range `(A+T)..(B+T)`
- old train choices `a,b,c` with fixed `n_test=T` -> new total-row choices `(a+T),(b+T),(c+T)`

______________________________________________________________________

## 4. Diagnostics

Use diagnostics to emit per-dataset observability artifacts.

```bash
dagzoo generate \
  --config configs/default.yaml \
  --num-datasets 50 \
  --diagnostics \
  --out data/run_diag
```

Detailed guides:

- [Diagnostics](features/diagnostics.md)

______________________________________________________________________

## 5. Generation modes

Use `dagzoo generate`, `generate_one`, `generate_batch`, or
`generate_batch_iter`; those entrypoints share the same public generation
surface.

Default mode:

- `runtime.layout_mode: heterogeneous`
- samples a layout and execution plan per dataset
- preserves one stable `request_run` id across the run
- emits in-memory `metadata.split_groups.cohort` instead of
  `metadata.split_groups.layout_plan`
- on Apple hardware, `device=auto` prefers CPU over MPS for this mode

Opt-in stratified mode:

- `runtime.layout_mode: stratified`
- keeps per-dataset layout and execution-plan sampling
- batches compatible exact `(n_rows, n_features)` strata within a rolling window
- still emits in-memory `metadata.split_groups.cohort`

When you persist shards, the public dataset catalog projects those in-memory
`split_groups` values into on-disk `group_ids.*` fields.

### PyTorch bridge

Use the PyTorch bridge when you want the same public generation semantics in an
in-process training loop instead of persisted shard outputs. The bridge follows
`runtime.layout_mode`, so it defaults to heterogeneous runs and can be pinned to
stratified mode for large throughput-sensitive heterogeneous corpora.

`build_dataloader(...)` is the recommended public entrypoint for most users.
`DagzooDataset` is the lower-level iterable dataset when you need direct
dataset control. `DagzooSample` is the returned sample shape; it carries
`X_train`, `y_train`, `X_test`, `y_test`, `feature_types`, and `metadata`.

```python
from pathlib import Path

from dagzoo import DagzooDataset, build_dataloader

loader = build_dataloader(
    "recipe:default-baseline",
    num_datasets=10,
    seed=7,
    device="cpu",
)
sample = next(iter(loader))

dataset = DagzooDataset(
    "configs/default.yaml",
    num_datasets=10,
    seed=7,
    device="cpu",
)
```

Bridge input contract:

- `config` may be a `GeneratorConfig`, a `recipe:<name>` reference, a string
  YAML path, or a `Path`
- `num_datasets` is required
- `seed` and `device` are optional
- v1 currently supports `num_workers=0` only

Use the dataloader helper unless you specifically need the iterable dataset
itself.

______________________________________________________________________

## 6. Intervention workflows

Use interventions when you need opt-in hard interventions on the canonical
generation path while keeping observational generation as the default.

```bash
dagzoo generate --config configs/preset_intervention_target_generate_smoke.yaml --num-datasets 25 --out data/run_intervention_target
```

Public artifacts expose only `intervention.mode` and `intervention.signature`;
the authored selector/value payload stays in `effective_config.yaml`.
`counterfactual` remains deferred and unsupported in the current public
surface.

Detailed guide: [Interventions](features/interventions.md)

______________________________________________________________________

## 7. Missingness workflows

Use missingness workflows for MCAR/MAR/MNAR robustness regimes:

```bash
dagzoo generate --config configs/preset_missingness_mar.yaml --num-datasets 25 --out data/run_missing_mar
```

Detailed guide: [Missingness](features/missingness.md)

______________________________________________________________________

## 8. Many-class workflows

Use many-class workflows to exercise the rollout envelope (`n_classes_max <= 32`).

```bash
dagzoo generate --config configs/preset_many_class_generate_smoke.yaml --num-datasets 25 --out data/run_many_class_smoke

dagzoo benchmark \
  --config configs/preset_many_class_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_many_class
```

Note: the built-in CPU benchmark preset (`dagzoo benchmark --preset cpu`) now
measures three explicit row profiles: `1024`, `4096`, and `8192` total rows per
dataset. Those runs report `generation_mode="fixed_batched"` plus explicit row
counts in their summary artifacts and use the same canonical generation path as
`dagzoo generate`.

Custom/standard benchmark presets also support `dataset.rows`. For benchmark
flows, rows specs stay variable through preset config resolution and then
realize once per preset run. Smoke suites cap rows before that run
realization so smoke benchmarks stay within the intended split-size envelope.

Detailed guide: [Many-class](features/many-class.md)

______________________________________________________________________

## 9. Shift workflows

Use shift profiles for controlled graph/mechanism/noise drift:

```bash
dagzoo generate --config configs/preset_shift_mixed_generate_smoke.yaml --num-datasets 25 --out data/run_shift_mixed
```

Detailed guide: [Shift / Drift](features/shift.md)

______________________________________________________________________

## 10. Steering workflows

Use steering workflows when you want one opt-in harder-front preset that
reuses existing missingness, shift, and noise levers:

```bash
dagzoo generate \
  --config configs/preset_steering_anti_memorization_generate_smoke.yaml \
  --num-datasets 25 \
  --out data/run_steering_smoke

dagzoo benchmark \
  --config configs/preset_steering_anti_memorization_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --diagnostics \
  --no-memory \
  --out-dir benchmarks/results/smoke_steering
```

Detailed guide: [Meta-Feature Coverage Steering](features/steering.md)

______________________________________________________________________

## 11. Stress-profile workflows

Use robustness stress profiles when you want one named carried slice rather
than a hand-authored harder config:

```bash
dagzoo generate \
  --config configs/preset_stress_graph_breadth_generate_smoke.yaml \
  --num-datasets 25 \
  --out data/run_stress_graph_breadth_smoke

dagzoo benchmark \
  --config configs/preset_stress_graph_breadth_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --diagnostics \
  --no-memory \
  --out-dir benchmarks/results/smoke_stress_graph_breadth
```

Detailed guide: [Robustness Stress Profiles](features/stress-profiles.md)

______________________________________________________________________

## 12. Noise workflows

Use noise workflows for explicit Gaussian/Laplace/Student-t/mixture regimes:

```bash
dagzoo generate --config configs/preset_noise_mixture_generate_smoke.yaml --num-datasets 25 --out data/run_noise_mixture
```

Detailed guide: [Noise Diversification](features/noise.md)

______________________________________________________________________

## 13. Mechanism-diversity workflows

Use mechanism-diversity workflows when you want to compare the current
baseline sampler against the shipped `piecewise` control and the widened `gp`
candidate path through the existing `mechanism.function_family_mix` surface.
Inspect realized family and variant uptake together with diversity shift,
throughput, and filter yield before deciding whether a candidate is worth
keeping.

```bash
dagzoo generate \
  --config configs/preset_mechanism_gp_generate_smoke.yaml \
  --num-datasets 10 \
  --device cpu \
  --hardware-policy none \
  --out data/run_gp_smoke_local

dagzoo diversity-audit \
  --baseline-config configs/preset_mechanism_baseline_benchmark_smoke.yaml \
  --variant-config configs/preset_mechanism_gp_benchmark_smoke.yaml \
  --suite smoke \
  --num-datasets 10 \
  --warmup 0 \
  --device cpu \
  --out-dir benchmarks/results/diversity_audit_gp

dagzoo diversity-audit \
  --baseline-config configs/preset_mechanism_baseline_benchmark_smoke.yaml \
  --variant-config configs/preset_mechanism_piecewise_benchmark_smoke.yaml \
  --suite smoke \
  --num-datasets 10 \
  --warmup 0 \
  --device cpu \
  --out-dir benchmarks/results/diversity_audit_piecewise_control

```

Detailed guide: [Mechanism Diversity](features/mechanism-diversity.md)

______________________________________________________________________

## 14. Benchmark workflows and guardrails

Use benchmark workflows for smoke checks, feature guardrails, and regression
gating.

```bash
dagzoo benchmark --suite smoke --preset cpu --out-dir benchmarks/results/smoke_cpu

dagzoo benchmark \
  --config configs/preset_filter_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --hardware-policy none \
  --no-memory \
  --out-dir benchmarks/results/smoke_filter
```

`--device` is a single-preset override. When you run multiple `--preset`
values in one command, set device selection in each preset/config instead of
passing a shared CLI device override.

Artifact-producing deferred filtering is disabled, but filter-enabled benchmark
configs and `dagzoo diversity-audit` runs still replay structural filter
metrics analytically. The canonical `preset_filter_benchmark_smoke` run uses
the same structural replay path as `dagzoo filter`, so throughput and
acceptance yield reflect lineage-based acceptance directly.

Detailed guide: [Benchmark Workflows and Guardrails](features/benchmark-guardrails.md)

When you need to compare accepted-corpus diversity between configs, use
`dagzoo diversity-audit` with one `--baseline-config` and one or more
`--variant-config` values. The audit writes `summary.json` and `summary.md`
with per-variant diversity status and throughput deltas.

______________________________________________________________________

## 15. Generate handoff workflows

Use `dagzoo generate --handoff-root` when a downstream consumer needs a stable
handoff root. There is no separate request-file
contract now; the handoff workflow uses the normal internal config plus CLI
overrides.

Example one-way handoff run:

```bash
dagzoo generate \
  --config configs/default.yaml \
  --handoff-root handoffs/smoke_run \
  --num-datasets 2 \
  --rows 1024 \
  --seed 7 \
  --device cpu \
  --hardware-policy none
```

That command writes:

- `handoffs/smoke_run/handoff_manifest.json`
- `handoffs/smoke_run/generated/`
- `handoffs/smoke_run/internal/`

Downstream consumption should start from `handoff_manifest.json`. The manifest
surfaces the generated corpus path and stable corpus identity in one versioned
JSON file:

```bash
./.venv/bin/python -c "import json; from pathlib import Path; payload=json.loads(Path('handoffs/smoke_run/handoff_manifest.json').read_text()); print(payload['artifacts_relative']['generated_dir']); print(payload['summary']['generated_datasets'])"
```

Closed-loop feedback from downstream predictions is still out of scope for this
workflow.

______________________________________________________________________

## Related documents

- Feature deep dives:
  [diagnostics](features/diagnostics.md),
  [missingness](features/missingness.md),
  [many-class](features/many-class.md),
  [stress profiles](features/stress-profiles.md),
  [shift](features/shift.md),
  [steering](features/steering.md),
  [noise](features/noise.md),
  [benchmark guardrails](features/benchmark-guardrails.md)
- Output contract: [output-format.md](output-format.md)
- System guide and terminology: [how-it-works.md](how-it-works.md)
- Quickstart: [start.md](start.md)
- Recipe catalog and citations: [reference-packs.md](reference-packs.md)
