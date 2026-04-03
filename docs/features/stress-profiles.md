# Robustness Stress Profiles

`stress.profile` is the carried-slice selector for RD-005 robustness regimes.
It lets you materialize one named stress envelope onto the normal generator
config without creating a second generator subsystem or hand-authoring a large
custom YAML every time.

This surface is different from the curated recipe catalog:

- `recipe:<name>` remains the stable public adoption layer for named reference
  packs under `recipes/`.
- `stress.profile=<name>` is the internal carried-slice selector that resolves
  onto the existing config surface during config resolution.
- Recipe entries labeled `stress profile` are examples; they are not the
  RD-005 carried-slice contract for downstream fixed-regime comparisons.

Use robustness stress profiles when you want reproducible harder-task or
anti-memorization slices while keeping the current missingness, shift, noise,
and diagnostics surfaces intact.

______________________________________________________________________

## When to use

### Why it matters for your prior

- You want one named harder regime that downstream model comparisons can hold
  fixed across runs.
- You need stronger relationship-structure or mechanism-composition pressure
  than the default baseline without opening a parallel config branch.
- You want diagnostics and diversity-audit evidence that the regime differs
  from baseline in intended directions.

### Operational triggers

- You want a reproducible anti-memorization classification slice.
- You need one graph-breadth-heavy slice for relationship-structure audits.
- You want a compositional mechanism slice that pushes family mixing harder
  than the default baseline.

______________________________________________________________________

## Shipped profiles

### `anti_memorization_piecewise_classification_slice_v1`

- Intended regime: baseline carried classification slice with the
  `anti_memorization_piecewise_v1` steering preset turned on.
- Main lever composition:
  - default classification envelope
  - steering-driven missingness/graph/noise progression
  - no extra graph-breadth or compositional mechanism bias

Generate smoke run:

```bash
dagzoo generate \
  --config configs/preset_stress_classification_slice_generate_smoke.yaml \
  --num-datasets 25 \
  --out data/run_stress_classification_slice_smoke
```

Benchmark smoke run:

```bash
dagzoo benchmark \
  --config configs/preset_stress_classification_slice_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --diagnostics \
  --no-memory \
  --out-dir benchmarks/results/smoke_stress_classification_slice
```

Inspect first:

- `coverage_summary.json`
- `metrics.graph_depth_ratio`
- `metrics.graph_reachability_ratio`
- `metrics.graph_target_ancestor_fraction`

### `anti_memorization_piecewise_classification_graph_breadth_slice_v1`

- Intended regime: broader graph/topology slice that increases node count and
  target-ancestor breadth pressure while retaining the anti-memorization
  steering path.
- Main lever composition:
  - larger node envelope
  - stricter target relevance/indegree floor
  - wider emitted feature envelope

Generate smoke run:

```bash
dagzoo generate \
  --config configs/preset_stress_graph_breadth_generate_smoke.yaml \
  --num-datasets 25 \
  --out data/run_stress_graph_breadth_smoke
```

Benchmark smoke run:

```bash
dagzoo benchmark \
  --config configs/preset_stress_graph_breadth_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --diagnostics \
  --no-memory \
  --out-dir benchmarks/results/smoke_stress_graph_breadth
```

Inspect first:

- `coverage_summary.json`
- `metrics.graph_indegree_std`
- `metrics.graph_outdegree_std`
- `metrics.graph_ancestor_overlap_mean`
- `metrics.graph_target_ancestor_fraction`

### `anti_memorization_piecewise_classification_compositional_slice_v1`

- Intended regime: compositional mechanism slice that biases toward
  `piecewise`, `product`, `gp`, and `tree` uptake while retaining the same
  anti-memorization steering path.
- Main lever composition:
  - softened but still non-default mechanism family mix centered on
    `piecewise`, `product`, `gp`, and `tree`
  - broader feature and categorical-cardinality envelope than the carried
    baseline
  - raised graph floor plus a light target relevance floor instead of the
    stricter structural gating used by the graph-breadth slice
  - tuned fixed-layout batch target of `8_000_000` cells for better CPU
    throughput on this heavier compositional regime

Generate smoke run:

```bash
dagzoo generate \
  --config configs/preset_stress_compositional_generate_smoke.yaml \
  --num-datasets 25 \
  --out data/run_stress_compositional_smoke
```

Benchmark smoke run:

```bash
dagzoo benchmark \
  --config configs/preset_stress_compositional_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --diagnostics \
  --no-memory \
  --out-dir benchmarks/results/smoke_stress_compositional
```

Inspect first:

- `coverage_summary.json`
- `mechanism_family_summary`
- `metrics.mechanism_family_cooccurrence_ratio`
- `metrics.graph_ancestor_overlap_mean`

______________________________________________________________________

## Baseline comparison workflow

To compare a stress profile against the baseline, use `dagzoo diversity-audit`
with `configs/default.yaml` as the baseline and one stress benchmark preset as
the variant:

```bash
dagzoo diversity-audit \
  --baseline-config configs/default.yaml \
  --variant-config configs/preset_stress_graph_breadth_benchmark_smoke.yaml \
  --suite smoke \
  --num-datasets 10 \
  --warmup 0 \
  --device cpu \
  --out-dir benchmarks/results/diversity_audit_stress_graph_breadth
```

Swap `--variant-config` to
`configs/preset_stress_classification_slice_benchmark_smoke.yaml` or
`configs/preset_stress_compositional_benchmark_smoke.yaml` for the other
profiles. Inspect `summary.json` and `summary.md` first, then inspect
coverage artifacts from a diagnostics-enabled benchmark run when you need the
relationship-structure or mechanism-family metrics behind the shift.

______________________________________________________________________

## Diagnostics and guardrails

- Generate smoke presets enable diagnostics so they write
  `coverage_summary.json` and `coverage_summary.md`.
- Benchmark smoke presets keep diagnostics off by default. Pass
  `--diagnostics` when you want coverage artifacts in addition to the benchmark
  summary.
- Benchmark summaries stay on the current contract. Steering and
  stress-profile evidence lives in diagnostics artifacts and in
  `dagzoo diversity-audit`, not in a new benchmark-only field family.
- The maintainer-only Pareto loop remains documented in
  [docs/development/rd005_handoff_evaluation.md](../development/rd005_handoff_evaluation.md).

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
- Diagnostics: [diagnostics.md](diagnostics.md)
- Steering: [steering.md](steering.md)
