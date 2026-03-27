# Many-Class

Most tabular classification benchmarks focus on binary or low-cardinality tasks
(2--5 classes), yet real-world applications frequently involve 10--100+ classes:
medical diagnosis codes, product category prediction, document classification,
species identification. This creates a coverage gap:

```
Typical synthetic prior:       2–5 classes  →  model is well-calibrated here
Real deployment tasks:         10–100+ classes

                               ← gap →

If the prior never includes many-class tasks, the model's in-context
learning for high-cardinality classification is extrapolating into a
regime it has no prior experience with.
```

Many-class settings need dedicated handling and scaling strategies, and
higher-cardinality regimes present known scaling challenges. dagzoo supports
class
counts up to 32 in the current rollout envelope, with explicit presets and
guardrails for stress-testing the cardinality frontier.

Use many-class workflows to generate and benchmark classification datasets near
the current rollout envelope (`n_classes_max <= 32`).

______________________________________________________________________

## When to use

### Why it matters for your prior

- Your model will encounter high-cardinality classification at inference time
  — including many-class tasks in the synthetic prior gives it explicit prior
  exposure to that regime.
- You want to measure whether in-context learning degrades gracefully with
  increasing class count, or whether there is a cardinality cliff where
  performance drops sharply.
- You are investigating the interaction between class cardinality and other
  prior axes (mechanism complexity, noise, missingness) to find where the
  cardinality frontier lies for your model.

### Operational triggers

- You are stress-testing multi-class performance beyond low-class regimes.
- You need smoke-stable presets for higher class cardinality.
- You want guardrail visibility during many-class benchmarking.

______________________________________________________________________

## Class count ranges

The `dataset.n_classes_max` config controls the upper bound on sampled class
count. The actual class count for each dataset is sampled uniformly between
2 and `n_classes_max`:

```
n_classes_max =  5  →  classes sampled from {2, 3, 4, 5}       (standard low-cardinality)
n_classes_max = 10  →  classes sampled from {2, 3, ..., 10}    (moderate)
n_classes_max = 20  →  classes sampled from {2, 3, ..., 20}    (high-cardinality stress)
n_classes_max = 32  →  classes sampled from {2, 3, ..., 32}    (current rollout envelope)
```

Higher class counts interact with sample size: a 32-class dataset with 200
training rows has ~6 samples per class on average, creating a small-shot
many-class regime that is particularly challenging for in-context learning.

______________________________________________________________________

## Generation workflow

```bash
dagzoo generate \
  --config configs/preset_many_class_generate_smoke.yaml \
  --num-datasets 25 \
  --out data/run_many_class_smoke
```

______________________________________________________________________

## Benchmark workflow

```bash
dagzoo benchmark \
  --config configs/preset_many_class_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_many_class
```

Benchmark summaries include throughput/latency plus per-scenario benchmark
status under `preset_results[*].scenarios`.

______________________________________________________________________

## What to inspect

- Class count and target distribution in emitted metadata.
- Benchmark summary sections for latency, throughput, and guardrails.

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
