# Benchmark Workflows and Guardrails

Use benchmark workflows to validate throughput/latency and enforce regression
guardrails across default and feature-specific configs.

______________________________________________________________________

## When to use

- You need fast smoke checks before wider experimentation.
- You want standardized performance baselines by preset/suite.
- You need CI gating with warn/fail regression thresholds.

______________________________________________________________________

## Baseline workflows

Quick smoke and broader standard runs:

```bash
dagzoo benchmark --suite smoke --preset cpu --out-dir benchmarks/results/smoke_cpu
dagzoo benchmark --suite standard --preset cpu --out-dir benchmarks/results/standard_cpu
```

Diagnostics-enabled benchmark:

```bash
dagzoo benchmark \
  --suite smoke \
  --preset cpu \
  --diagnostics \
  --out-dir benchmarks/results/smoke_cpu_diag
```

Device override note:

- `--device` only applies when a benchmark run selects exactly one `--preset`.
- Multi-preset runs must encode device choice in each preset/config; the CLI now
  rejects ambiguous shared `--device` overrides.

______________________________________________________________________

## Feature-specific guardrail runs

```bash
dagzoo benchmark \
  --config configs/preset_filter_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --hardware-policy none \
  --no-memory \
  --out-dir benchmarks/results/smoke_filter

dagzoo benchmark \
  --config configs/preset_missingness_mar.yaml \
  --preset custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_missing_mar

dagzoo benchmark \
  --config configs/preset_shift_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_shift_guardrails

dagzoo benchmark \
  --config configs/preset_noise_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_noise_guardrails

dagzoo benchmark \
  --config configs/preset_steering_anti_memorization_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --diagnostics \
  --no-memory \
  --out-dir benchmarks/results/smoke_steering
```

______________________________________________________________________

## Filter-enabled benchmark workflow

Use the filter smoke preset when you want one canonical CPU benchmark run that
surfaces filter-stage throughput, accepted-corpus throughput, and acceptance
yield together:

```bash
dagzoo benchmark \
  --config configs/preset_filter_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --hardware-policy none \
  --no-memory \
  --out-dir benchmarks/results/smoke_filter
```

This preset intentionally sets `filter.use_lineage_veto=false` and
`filter.stump_skill_threshold=0.04` so the smoke run measures learned-filter
throughput and yield instead of being dominated by structural no-path vetoes.

Inspect these `summary.json` preset-result fields first:

- `filter_datasets_per_minute`
- `filter_accepted_datasets_per_minute`
- `filter_accepted_datasets_measured`
- `filter_rejected_datasets_measured`
- `filter_acceptance_rate_dataset_level`
- `filter_rejection_rate_dataset_level`
- `filter_rejection_rate_attempt_level`
- `filter_retry_dataset_rate`

The CLI preset line prints the same headline values as `filter/min`,
`filter_accepted/min`, `filter_accept_dataset_pct`, and
`filter_reject_dataset_pct`.

______________________________________________________________________

## Diversity audit workflow

Use `diversity-audit` when you need a baseline-vs-variant comparison of the
accepted corpus, not just benchmark throughput:

```bash
dagzoo diversity-audit \
  --baseline-config configs/default.yaml \
  --variant-config configs/preset_shift_benchmark_smoke.yaml \
  --suite smoke \
  --num-datasets 10 \
  --warmup 0 \
  --device cpu \
  --out-dir benchmarks/results/diversity_audit_shift
```

Inspect these `summary.json` fields first:

- `comparisons[*].diversity_status`
- `comparisons[*].diversity_composite_shift_pct`
- `comparisons[*].datasets_per_minute_delta_pct`
- `comparisons[*].filter_accepted_datasets_per_minute_delta_pct`

The rewritten audit persists `summary.json` and `summary.md` as the canonical
equivalence/local-overlap and cross-run diversity outputs.

______________________________________________________________________

## Regression gating

For CI-like checks:

```bash
dagzoo benchmark \
  --config configs/preset_shift_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --warn-threshold-pct 10 \
  --fail-threshold-pct 20 \
  --fail-on-regression \
  --hardware-policy none \
  --no-memory \
  --out-dir benchmarks/results/ci_smoke_shift_local
```

______________________________________________________________________

## What to inspect

When present in a run summary, inspect:

- `missingness_guardrails`
- `lineage_guardrails`
- `shift_guardrails`
- `noise_guardrails`

Also review throughput/latency aggregates for preset/suite trends.

For steering-enabled runs, inspect `preset_results[*].diagnostics_artifacts`
first and then open the pointed `coverage_summary.json` / `coverage_summary.md`
files. Steering stays on the diagnostics artifact contract; benchmark summaries
do not emit a separate `steering_guardrails` field.

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Output contract: [output-format.md](../output-format.md)
- Noise workflows: [noise.md](noise.md)
- Steering workflows: [steering.md](steering.md)
