# Diagnostics

Effective diversity -- the breadth of meta-feature space actually covered by a
generated corpus -- is the central quality metric for a synthetic tabular prior.
A corpus might contain millions of datasets, but if they all cluster in a narrow
region of meta-feature space (similar feature counts, similar class counts,
similar functional complexity), the foundation model sees a narrow prior and
generalizes poorly to tasks outside that region. Diagnostics provides the
observability layer that makes effective diversity measurable. Without it,
researchers cannot determine whether a config change actually broadened the
prior or merely produced more of the same.

Recent work has shown that meta-feature coverage of weak regimes improves
model reliability -- directly motivating the ability to measure which
meta-feature regions a corpus covers and which it misses. Synthetic prior
quality and scale are central to tabular foundation model performance, making
corpus-level observability a prerequisite for principled prior engineering.

Use diagnostics when you want per-dataset observability artifacts to verify
coverage, spot drift, and debug generation behavior.

______________________________________________________________________

## Effective diversity: what it means and why it matters

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

______________________________________________________________________

## When to use

### Why it matters for your prior

- You are iterating on your prior configuration and need to measure whether
  changes actually improve effective diversity, not just throughput.
- You want to identify specific meta-feature coverage gaps in your corpus --
  for example, finding that your prior undercovers low-feature-count
  high-class-count regimes.
- You are running A/B comparisons between prior configurations and need
  quantitative evidence that one configuration covers more meta-feature space
  than another.
- You want to define target bands for specific meta-features and track what
  fraction of your corpus falls within those bands.

### Operational triggers

- You need per-dataset records in shard `metadata.ndjson` and summary-level metric coverage.
- You are validating whether presets or CLI overrides hit expected ranges.
- You want benchmark runs to include richer context for guardrail triage.

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

- Per-dataset `metadata.ndjson` records for realized generation parameters.
- Coverage summaries for meta-features and enabled observability metrics.
- Benchmark summary guardrail sections that include diagnostics context.

Exact output contracts are documented in
[output-format.md](../output-format.md).

______________________________________________________________________

## Diagnostics target bands

Diagnostics supports optional `diagnostics.meta_feature_targets` to annotate
coverage summaries with in-band counts/fractions for selected metrics.

Target bands do not alter generation; they are reporting metadata only.

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- System terminology: [how-it-works.md](../how-it-works.md)
