# Usage Guide

Operational entrypoint for end users running generation and benchmarking
workflows. This page stays concise and links to detailed feature guides.

______________________________________________________________________

## Prerequisites

Examples on this page use repo-local presets under `configs/`.

For a repo checkout:

```bash
uv sync --group dev
source .venv/bin/activate
```

For a global CLI install (without repo presets/config files):

```bash
uv tool install dagzoo
```

______________________________________________________________________

## 1. Basic generation

Use this when you want a default high-quality batch.

```bash
dagzoo generate --config configs/default.yaml --num-datasets 10 --out data/run1
```

Each generate run writes `effective_config.yaml` and `effective_config_trace.yaml`
under the resolved output directory.
`dagzoo generate` samples one internal fixed-layout plan per run, so all
datasets emitted in the same run share one layout signature / plan signature.
Inline filtering is removed from `dagzoo generate`. Keep `filter.enabled: false`
for generate flows, then run `dagzoo filter` as a separate replay stage on the
emitted shards when you want acceptance decisions.
Generate configs must not include `runtime.worker_count` or
`runtime.worker_index`.

______________________________________________________________________

## 2. Deferred filtering (`dagzoo filter`)

Deferred filtering replays the small-shot ease filter from shard metadata and
lineage artifacts. The main tuning knobs are `ease_k_small`,
`easy_skill_threshold`, `easy_gain_threshold`, `hard_skill_threshold`, the
optional `stump_skill_threshold`, and `use_lineage_veto`. Replay-time CLI
overrides include both `--lineage-veto` and `--no-lineage-veto` when you need
to force structural veto behavior instead of inheriting the embedded shard
config.

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

## 5. Canonical fixed-layout generation

Use `dagzoo generate`, `generate_one`, `generate_batch`, or
`generate_batch_iter`; those entrypoints all run on the canonical
fixed-layout engine internally.

For persisted outputs, replay canonical batch bundles with the shared run seed
plus `dataset_index` / `run_num_datasets` from the recorded metadata.

### PyTorch bridge

Use the PyTorch bridge when you want the same canonical fixed-layout run
semantics in an in-process training loop instead of persisted shard outputs.
Across one requested run, samples yielded through the bridge still share the
same sampled layout/execution plan.

`build_dataloader(...)` is the recommended public entrypoint for most users.
`DagzooDataset` is the lower-level iterable dataset when you need direct
dataset control. `DagzooSample` is the returned sample shape; it carries
`X_train`, `y_train`, `X_test`, `y_test`, `feature_types`, and `metadata`.

```python
from pathlib import Path

from dagzoo import DagzooDataset, build_dataloader

loader = build_dataloader(
    Path("configs/default.yaml"),
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

- `config` may be a `GeneratorConfig`, a string path, or a `Path`
- `num_datasets` is required
- `seed` and `device` are optional
- v1 currently supports `num_workers=0` only

Use the dataloader helper unless you specifically need the iterable dataset
itself.

______________________________________________________________________

## 6. Missingness workflows

Use missingness workflows for MCAR/MAR/MNAR robustness regimes:

```bash
dagzoo generate --config configs/preset_missingness_mar.yaml --num-datasets 25 --out data/run_missing_mar
```

Detailed guide: [Missingness](features/missingness.md)

______________________________________________________________________

## 7. Many-class workflows

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

## 8. Shift workflows

Use shift profiles for controlled graph/mechanism/noise drift:

```bash
dagzoo generate --config configs/preset_shift_mixed_generate_smoke.yaml --num-datasets 25 --out data/run_shift_mixed
```

Detailed guide: [Shift / Drift](features/shift.md)

______________________________________________________________________

## 9. Steering workflows

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

## 10. Noise workflows

Use noise workflows for explicit Gaussian/Laplace/Student-t/mixture regimes:

```bash
dagzoo generate --config configs/preset_noise_mixture_generate_smoke.yaml --num-datasets 25 --out data/run_noise_mixture
```

Detailed guide: [Noise Diversification](features/noise.md)

______________________________________________________________________

## 11. Mechanism-diversity workflows

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

## 12. Benchmark workflows and guardrails

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
configs and `dagzoo diversity-audit` runs still replay filter metrics
analytically.
The canonical `preset_filter_benchmark_smoke` run intentionally sets
`filter.use_lineage_veto=false` so the benchmark measures learned-filter
throughput and acceptance yield instead of structural no-path rejection.

Detailed guide: [Benchmark Workflows and Guardrails](features/benchmark-guardrails.md)

When you need to compare accepted-corpus diversity between configs, use
`dagzoo diversity-audit` with one `--baseline-config` and one or more
`--variant-config` values. The audit writes `summary.json` and `summary.md`
with per-variant diversity status and throughput deltas.

______________________________________________________________________

## 13. Generate handoff workflows

Use `dagzoo generate --handoff-root` when a downstream repo such as
`tab-foundry` needs a stable handoff root. There is no separate request-file
contract now; the handoff workflow uses the normal internal config plus CLI
overrides.

Example one-way handoff run:

```bash
dagzoo generate \
  --config configs/default.yaml \
  --handoff-root handoffs/tab_foundry_smoke \
  --num-datasets 2 \
  --rows 1024 \
  --seed 7 \
  --device cpu \
  --hardware-policy none
```

That command writes:

- `handoffs/tab_foundry_smoke/handoff_manifest.json`
- `handoffs/tab_foundry_smoke/generated/`

Downstream consumption should start from `handoff_manifest.json`. The manifest
surfaces the generated corpus path, effective-config artifacts, and invocation
metadata in one versioned JSON file:

```bash
./.venv/bin/python -c "import json; from pathlib import Path; payload=json.loads(Path('handoffs/tab_foundry_smoke/handoff_manifest.json').read_text()); print(payload['artifacts']['generated_dir']); print(payload['summary']['generated_datasets'])"
```

Closed-loop feedback from downstream predictions is still out of scope for this
workflow.

______________________________________________________________________

## Related documents

- Feature deep dives:
  [diagnostics](features/diagnostics.md),
  [missingness](features/missingness.md),
  [many-class](features/many-class.md),
  [shift](features/shift.md),
  [steering](features/steering.md),
  [noise](features/noise.md),
  [benchmark guardrails](features/benchmark-guardrails.md)
- Output contract: [output-format.md](output-format.md)
- Config precedence and trace artifacts: [development/config-resolution.md](development/config-resolution.md)
- System guide and terminology: [how-it-works.md](how-it-works.md)
- Architecture rationale and evolution policy: [development/design-decisions.md](development/design-decisions.md)
