# Robustness Stress Profiles

`stress.profile` selects named harder-generation regimes that resolve onto the
normal generator config. Use it when you want a reproducible stress envelope
without hand-authoring a large custom YAML for each run.

This surface is different from the curated recipe catalog:

- `recipe:<name>` remains the stable public adoption layer for named reference
  packs under `recipes/`.
- `stress.profile=<name>` is the advanced YAML control for selecting one named
  stress regime inside a repo-local config.
- Recipe entries labeled `stress profile` are ready-made examples of the same
  kind of harder-generation workflow.

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
- You want a categorical/cardinality-heavy lane without hand-authoring the
  converter and cardinality envelope.
- You want one hybrid slice that combines graph/source-shape reuse with
  compositional matrix/kernel reuse.
- You want one robustness-composition slice that explicitly couples missingness,
  shift, and noise with the harder mechanism mix.

______________________________________________________________________

## Shipped core profiles

### `anti_memorization_piecewise_classification_slice_v1`

- Intended regime: default classification envelope with the
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
  - broader feature and categorical-cardinality envelope than the baseline
  - raised graph floor plus a light target relevance floor instead of the
    stricter structural gating used by the graph-breadth slice
  - tuned grouped batch target of `8_000_000` cells for better CPU throughput
    on this heavier compositional regime

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

## Current internal evaluation profiles

These profiles are available through repo-local configs and `stress.profile`,
but they are still evaluation lanes rather than promoted public recipes. Keep
them audit-gated and compare them against baseline and the current
compositional slice before treating them as a broader adoption candidate.

### `anti_memorization_piecewise_classification_categorical_cardinality_slice_v1`

- Intended regime: categorical-heavy harder slice with broader correlated
  categorical-cardinality regimes than the default baseline.
- Main lever composition:
  - raised categorical ratio floor and wider class envelope
  - larger categorical cardinality ceiling tied to the existing
    high-cardinality workflow
  - anti-memorization steering retained as the base harder slice control

Generate smoke run:

```bash
dagzoo generate \
  --config configs/preset_stress_categorical_cardinality_generate_smoke.yaml \
  --num-datasets 25 \
  --out data/run_stress_categorical_cardinality_smoke
```

Benchmark smoke run:

```bash
dagzoo benchmark \
  --config configs/preset_stress_categorical_cardinality_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --diagnostics \
  --no-memory \
  --out-dir benchmarks/results/smoke_stress_categorical_cardinality
```

Inspect first:

- `coverage_summary.json`
- `parity_surface_summary.categorical_cardinality`
- `parity_surface_summary.converter_method_counts`
- `metrics.cat_cardinality_mean`

### `anti_memorization_piecewise_classification_hybrid_slice_v1`

- Intended regime: hybrid structural+compositional slice that combines the
  graph/source-shape reuse pressure from the graph-breadth lane with the
  matrix/kernel/root reuse from the compositional lane.
- Main lever composition:
  - broader node and feature envelope with graph gating enabled
  - parent-arity/source-shape policy reuse
  - correlated matrix/kernel/root-base-kind reuse

Generate smoke run:

```bash
dagzoo generate \
  --config configs/preset_stress_hybrid_generate_smoke.yaml \
  --num-datasets 25 \
  --out data/run_stress_hybrid_smoke
```

Benchmark smoke run:

```bash
dagzoo benchmark \
  --config configs/preset_stress_hybrid_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --diagnostics \
  --no-memory \
  --out-dir benchmarks/results/smoke_stress_hybrid
```

Inspect first:

- `coverage_summary.json`
- `parity_surface_summary.matrix_kind_counts`
- `parity_surface_summary.root_base_kind_counts`
- `parity_surface_summary.source_shape_policy_counts`

### `anti_memorization_piecewise_classification_robustness_composition_slice_v1`

- Intended regime: explicit robustness composition that couples missingness,
  shift, and mixed noise with the anti-memorization harder slice instead of
  leaving those levers as separate recipes.
- Main lever composition:
  - structured MNAR missingness
  - mixed graph/variance drift
  - explicit Gaussian/Laplace/Student-t residual mixture

Generate smoke run:

```bash
dagzoo generate \
  --config configs/preset_stress_robustness_composition_generate_smoke.yaml \
  --num-datasets 25 \
  --out data/run_stress_robustness_composition_smoke
```

Benchmark smoke run:

```bash
dagzoo benchmark \
  --config configs/preset_stress_robustness_composition_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --diagnostics \
  --no-memory \
  --out-dir benchmarks/results/smoke_stress_robustness_composition
```

Inspect first:

- `coverage_summary.json`
- `missingness`
- `shift`
- `noise_distribution`
- `parity_surface_summary.gp_variant_counts`

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

Swap `--variant-config` to any of the other stress benchmark presets for the
other profiles. Inspect `summary.json` and `summary.md` first, then inspect
coverage artifacts from a diagnostics-enabled benchmark run when you need the
relationship-structure or mechanism-family metrics behind the shift.

To turn one diversity-audit run into a parity-focused maintainer report:

```bash
./.venv/bin/python scripts/render_tabiclv2_parity_report.py \
  --summary-json benchmarks/results/diversity_audit_stress_graph_breadth/summary.json \
  --out-dir benchmarks/results/diversity_audit_stress_graph_breadth/parity_report
```

To run the full internal RD-005 promotion suite across the incumbent,
structural control, and current challenger lanes:

```bash
./.venv/bin/python scripts/evaluate_rd005_follow_on_suite.py \
  --baseline-config configs/default.yaml \
  --out-root benchmarks/results/rd005_follow_on \
  --suite smoke \
  --num-datasets 8 \
  --seed 123 \
  --device cpu
```

Read `follow_on_promotion_summary.json` / `follow_on_promotion_summary.md`
first when you want the explicit promotion status for each internal lane.

______________________________________________________________________

## Diagnostics and guardrails

- Generate smoke presets enable diagnostics so they write
  `coverage_summary.json` and `coverage_summary.md`.
- Benchmark smoke presets keep diagnostics off by default. Pass
  `--diagnostics` when you want coverage artifacts in addition to the benchmark
  summary.
- `dagzoo diversity-audit` `summary.json` now carries `parity_surface_summary`
  alongside the mechanism-family summary so the remaining TabICLv2 parity gaps
  can be measured directly instead of inferred indirectly.
- The full RD-005 suite runner keeps the public/internal boundary explicit:
  internal lanes can receive `promote`, `hold_internal`, or
  `structural_control_only`, but public recipe promotion remains a separate
  follow-up step after a winner clears the gate.
- Benchmark summaries stay on the current contract. Steering and
  stress-profile evidence lives in diagnostics artifacts and in
  `dagzoo diversity-audit`, not in a new benchmark-only field family.

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
- Diagnostics: [diagnostics.md](diagnostics.md)
- Steering: [steering.md](steering.md)
