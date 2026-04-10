# Diagnostics

Diagnostics adds run-level coverage summaries to a generated corpus so you can
see what the run actually produced. When enabled, `dagzoo` writes
`coverage_summary.json` and `coverage_summary.md` alongside the generated data,
making it easier to inspect feature counts, class counts, mechanism mix, noise,
missingness, parity-surface relationship reuse, and other realized properties.

Use diagnostics when you want to compare recipes, confirm that a preset landed
in the range you expected, or explain why one run behaves differently from
another. Start with the coverage summaries, then drill into
`dataset_catalog.ndjson` or in-process metadata when you need per-dataset
detail.

______________________________________________________________________

## Meta-feature coverage and effective diversity

Effective diversity is **not** the same as the number of datasets or the number
of unique seeds. A corpus of 1 million datasets that all have 10 features,
2 classes, and linear mechanisms has very low effective diversity despite its
scale. A corpus of 10,000 datasets spanning 2--50 features, 2--32 classes,
9 mechanism families, multiple noise profiles, and shift regimes has much higher
effective diversity.

Diagnostics makes this measurable by tracking meta-features (feature count,
class count, mechanism family distribution, noise family, shift presence,
missingness rate) across the corpus and reporting coverage statistics. Optional
target bands let you define expected ranges for specific meta-features and track
what fraction of your corpus falls within those bands, turning effective
diversity from a vague goal into a quantitative metric.

Coverage summaries also persist a `parity_surface_summary` block for the
realized relationship-reuse surface. That additive section reports converter
method/variant frequency, GP variant frequency, kernel `gamma` / `signed`
coverage, matrix-family coverage, root-base-kind coverage, parent-arity counts,
source-shape policy counts, and categorical-cardinality ranges.

______________________________________________________________________

## When to use

### Why it matters for your prior

- You are iterating on your prior configuration and need to measure whether
  changes actually broaden coverage, not just throughput.
- You want to identify specific meta-feature coverage gaps in your corpus --
  for example, finding that your prior undercovers low-feature-count
  high-class-count regimes.
- You are running A/B comparisons between prior configurations and need
  quantitative evidence that one configuration covers more meta-feature space
  than another.
- You want to define target bands for specific meta-features and track what
  fraction of your corpus falls within those bands.

### Operational triggers

- You need the stable public `dataset_catalog.ndjson` plus summary-level metric coverage.
- You are validating whether presets or CLI overrides hit expected ranges.
- You want benchmark runs to include richer context for guardrail review.

______________________________________________________________________

## Quick start

Enable diagnostics directly:

```bash
dagzoo generate \
  --config configs/default.yaml \
  --num-datasets 50 \
  --diagnostics \
  --out data/run_diag
```

Use the discoverable preset:

```bash
dagzoo generate \
  --config configs/preset_diagnostics_on.yaml \
  --num-datasets 25 \
  --diagnostics \
  --out data/run_diag_preset
```

______________________________________________________________________

## Key options

- `--diagnostics`: emit diagnostics artifacts for generated datasets.
- `--out`: output directory containing datasets and diagnostic payloads.

Diagnostics also work with `benchmark`:

```bash
dagzoo benchmark \
  --suite smoke \
  --preset cpu \
  --diagnostics \
  --out-dir benchmarks/results/smoke_cpu_diag
```

______________________________________________________________________

## What to inspect

- Public `dataset_catalog.ndjson` for stable per-dataset identity and emitted schema.
- In-process `DatasetBundle.metadata` when you need rich realized generation parameters.
- Coverage summaries for meta-features, enabled observability metrics, parity-surface summaries, and steering movement when curriculum steering is enabled.
- Benchmark summary guardrail sections that include diagnostics context.

Exact output contracts are documented in
[output-format.md](../output-format.md).

______________________________________________________________________

## Diagnostics target bands

Diagnostics supports optional `diagnostics.meta_feature_targets` to annotate
coverage summaries with in-band counts/fractions for selected metrics.

Target bands do not alter generation; they are reporting metadata only.

When steering is enabled, the same `coverage_summary.json` and
`coverage_summary.md` artifacts add a top-level `steering` section instead of
emitting a separate artifact family. That section reports:

- The steering authoring form (`preset` or explicit stages).
- Requested stage definitions and fractions.
- Per-stage realized missingness, shift, and noise summaries.
- Resolution-consistency checks comparing requested steering resolution against
  emitted metadata for the generated run.

This steering analysis is additive: it does not add new CLI flags and does not
change the public per-dataset `dataset_catalog.ndjson` contract.

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- System terminology: [how-it-works.md](../how-it-works.md)
