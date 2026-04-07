# Meta-Feature Coverage Steering

Meta-feature coverage steering combines existing missingness, shift/drift, and
noise controls into one preset-driven workflow. Use it when you want a
reproducible progression through harder regions of the dataset space and a
clear diagnostics trail showing what the run actually realized.

Steering resolves onto the same generation surface used by the rest of
`dagzoo`. It is a way to author a staged run, not a separate generator.

______________________________________________________________________

## When to use

### Why it matters for your prior

- You want one discoverable, deterministic harder-front workflow instead of
  hand-authoring separate missingness, shift, and noise runs.
- You need auditable evidence that a run actually moved through the intended
  curriculum rather than only setting a config knob on paper.
- You want to reuse current missingness, shift/drift, and noise controls
  without introducing a parallel curriculum subsystem.
- You want a benchmarkable preset that can be compared over time with the same
  smoke-sized CPU workflow.

### Operational triggers

- You want requested-vs-realized steering evidence in diagnostics artifacts.
- You want one preset that composes missingness, graph drift, mixed drift, and
  mixture noise into a single run.
- You need a documented steering smoke workflow before iterating on new presets.

______________________________________________________________________

## Shipped preset

The built-in preset is `anti_memorization_piecewise_v1`:

- `missingness_ramp`
- `graph_excursion_out`
- `graph_to_noise_handoff`
- `mixture_noise_ramp`

Those stages progressively move the run from light missingness into graph drift,
then a graph-to-noise handoff, and finally a mixture-noise ramp. The preset is
resolved per dataset with fixed-seed determinism.

This preset is built on top of the existing missingness, shift, and noise
controls. The diagnostics output shows the requested stages alongside the
realized run summaries.

______________________________________________________________________

## Preset workflows

Generate smoke run:

```bash
dagzoo generate \
  --config configs/preset_steering_anti_memorization_generate_smoke.yaml \
  --num-datasets 25 \
  --out data/run_steering_smoke
```

The generate smoke preset already enables diagnostics, so the run writes
`coverage_summary.json` and `coverage_summary.md` alongside the generated data.

Benchmark smoke run:

```bash
dagzoo benchmark \
  --config configs/preset_steering_anti_memorization_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --diagnostics \
  --no-memory \
  --out-dir benchmarks/results/smoke_steering
```

Benchmark diagnostics artifacts still require `--diagnostics`. The benchmark
summary stays on the existing contract; steering audit evidence lives in the
diagnostics artifact pointers rather than a new `steering_guardrails` field.

______________________________________________________________________

## What to inspect

- `coverage_summary.json`:
  - `steering.enabled`
  - `steering.authoring_form`
  - `steering.preset`
  - `steering.stage_count`
  - `steering.resolution_checks`
  - `steering.stages[*].requested`
  - `steering.stages[*].requested_effective`
  - `steering.stages[*].realized`
  - `steering.stages[*].metrics`
- `coverage_summary.md`: condensed requested-vs-realized stage movement for fast
  human review.
- Benchmark `summary.json`:
  - `preset_results[*].diagnostics_enabled`
  - `preset_results[*].diagnostics_artifacts.json`
  - `preset_results[*].diagnostics_artifacts.markdown`

Open the benchmark diagnostics artifact path first, then inspect the top-level
`steering` object in `coverage_summary.json`. That is the canonical audit
surface for requested-versus-realized steering movement.

For field definitions, see [output-format.md](../output-format.md). For the
diagnostics artifact workflow, see [diagnostics.md](diagnostics.md).

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Diagnostics artifacts: [diagnostics.md](diagnostics.md)
- Benchmark workflows: [benchmark-guardrails.md](benchmark-guardrails.md)
- Output contract: [output-format.md](../output-format.md)
