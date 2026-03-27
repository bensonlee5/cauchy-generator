# Noise Diversification

Real-world tabular data rarely has Gaussian residuals. Financial returns
exhibit fat tails; sensor readings show double-exponential error profiles;
biological measurements often have heavy-tailed, heteroscedastic variation.
A synthetic prior that restricts noise to Gaussian systematically undercovers
these regimes — the foundation model never sees heavy-tailed stochasticity
during pretraining, so its in-context learning on tasks with non-Gaussian noise
must generalize from zero prior exposure.

Prior design realism — including noise diversity — materially affects tabular
foundation model performance. dagzoo exposes four noise regimes so researchers
can control and
measure the noise axis of their prior independently from mechanism complexity:

```
Gaussian    →  light tails, symmetric     → the "easy" baseline
Laplace     →  sharper peak, heavier tails → double-exponential residuals
Student-t   →  heavy tails, occasional outliers (df > 2 for finite variance)
mixture     →  per-dataset draw from the above → corpus spans all three
```

Use noise-family workflows when you want non-Gaussian stochastic regimes while
retaining deterministic seed behavior and explicit metadata reporting.

______________________________________________________________________

## When to use

### Why it matters for your prior

- Your synthetic prior currently uses only Gaussian noise, which means every
  dataset's residuals have light, symmetric tails — a coverage gap for the
  many real tasks with heavy-tailed or asymmetric noise.
- You want to measure whether your model's in-context learning is robust to
  the *shape* of noise (tail weight, kurtosis), not just noise *magnitude*.
- You are investigating effective diversity and suspect Gaussian-only noise
  is a meta-feature coverage gap — coverage of weak meta-feature regimes
  has been shown to improve reliability.
- You want to ablate the noise axis in isolation: hold mechanism complexity
  and graph structure constant, change only the noise family, and attribute
  downstream performance changes to noise distributional effects specifically.

### Operational triggers

- You want heavier-tail stochasticity than the Gaussian default.
- You need deterministic comparisons across Gaussian/Laplace/Student-t regimes.
- You want benchmark guardrails for runtime impact and metadata validity.

______________________________________________________________________

## Supported families

- `gaussian`: default Gaussian sampling. Light, symmetric tails — the most
  benign noise regime. P(|x| > 3σ) ≈ 0.3%.
- `laplace`: double-exponential noise. Sharper peak and heavier tails than
  Gaussian — P(|x| > 3σ) ≈ 1.2%, about 4x more outliers than Gaussian at
  the same scale.
- `student_t`: heavy-tailed Student-t. Requires `df > 2` (finite variance).
  At `df = 3`, tail weight is substantial — roughly 10x more probability
  mass beyond 3σ than Gaussian. At `df = 10`, behavior approaches Gaussian.
  Lower `df` → heavier tails → more extreme outliers.
- `mixture`: per-dataset weighted draw from the above three families. Each
  dataset in the corpus gets one family (resolved at generation time), so a
  corpus generated with `mixture` naturally spans all three noise regimes
  without separate generate runs.

______________________________________________________________________

## Preset workflows

Generate smoke datasets for each family:

```bash
dagzoo generate --config configs/preset_noise_gaussian_generate_smoke.yaml --num-datasets 25 --out data/run_noise_gaussian
dagzoo generate --config configs/preset_noise_laplace_generate_smoke.yaml --num-datasets 25 --out data/run_noise_laplace
dagzoo generate --config configs/preset_noise_student_t_generate_smoke.yaml --num-datasets 25 --out data/run_noise_student_t
dagzoo generate --config configs/preset_noise_mixture_generate_smoke.yaml --num-datasets 25 --out data/run_noise_mixture
```

Benchmark guardrail smoke run:

```bash
dagzoo benchmark \
  --config configs/preset_noise_benchmark_smoke.yaml \
  --preset custom \
  --suite smoke \
  --no-memory \
  --out-dir benchmarks/results/smoke_noise
```

______________________________________________________________________

## What to inspect

- Per-dataset `metadata.ndjson` entries:
  - `noise_distribution.family_requested`
  - `noise_distribution.family_sampled`
  - `noise_distribution.sampling_strategy`
  - `noise_distribution.base_scale`
  - `noise_distribution.student_t_df`
  - `noise_distribution.mixture_weights` (when requested family is `mixture`)
- Benchmark summary `preset_results[*].scenarios.noise`:
  - metadata coverage/validity
  - sampled-family counts
  - runtime delta vs gaussian-noise control

For output details, see [output-format.md](../output-format.md).

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
