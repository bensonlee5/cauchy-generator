# Shift / Drift

Distribution shift between training and test data is one of the most pervasive
failure modes in deployed ML. A model that scores well on i.i.d. test sets can
degrade severely when the deployment distribution differs from training —
covariate shift, concept drift, and environmental noise changes account for the
majority of production ML failures.

A synthetic prior that always generates train and test from the *same*
distribution produces a foundation model that has never seen shift during
pretraining. dagzoo implements three independent shift axes, each mapping to a
distinct real-world drift pattern:

```
graph_scale      → the causal structure itself changes between train and test
                   (e.g., new regulations alter which variables are causally relevant)

mechanism_scale  → the functional relationships change while the graph stays the same
                   (e.g., customer behavior shifts but the same features remain relevant)

variance_scale   → the stochastic variation changes in magnitude
                   (e.g., sensor degradation increases measurement noise over time)
```

Shift-aware constructions in the synthetic prior have been shown to help the
resulting model handle temporal distribution shifts at inference time.

Use shift workflows when you want controlled distribution drift while preserving
deterministic seeds and interpretable scale semantics.

______________________________________________________________________

## When to use

### Why it matters for your prior

- Your model will encounter distribution shift at inference time — if the
  prior never includes shifted train/test splits, the model has no prior
  experience to draw on for robustness.
- You want to ablate which shift axis matters most for downstream robustness:
  hold two axes at zero while sweeping the third, and measure the model's
  degradation curve.
- You want to generate shifted splits that mimic specific real deployment
  scenarios (temporal drift in financial data, population drift in clinical
  trials) within a fully controlled synthetic setting.
- You need interpretable, monotonic shift controls with known mathematical
  semantics — `variance_scale = 0.5` always means +1.5 dB noise variance,
  regardless of the base config.

### Operational triggers

- You need train/test distribution shift for robustness evaluation.
- You want independent control over graph, mechanism, and noise drift.
- You want shift-aware observability in metadata and diagnostics coverage.

______________________________________________________________________

## Shift modes

Mode-only examples:

```yaml
shift:
  enabled: true
  mode: graph_drift
```

```yaml
shift:
  enabled: true
  mode: mechanism_drift
```

```yaml
shift:
  enabled: true
  mode: noise_drift
```

```yaml
shift:
  enabled: true
  mode: mixed
```

Custom mode with explicit scales:

```yaml
shift:
  enabled: true
  mode: custom
  graph_scale: 0.6
  mechanism_scale: 0.2
  variance_scale: 0.4
```

______________________________________________________________________

## Scale interpretation

All three scales use 0 = no drift. Positive values increase drift; the
mapping from scale to runtime effect is deterministic.

**`graph_scale`** — shifts the edge-logit bias, changing how likely edges are
in the DAG. The runtime edge-odds multiplier is `exp(ln(2) * graph_scale)`:

```
graph_scale = 0.0  →  edge-odds multiplier = 1.0×   (no change)
graph_scale = 0.5  →  edge-odds multiplier ≈ 1.41×  (moderate: ~41% more likely per edge)
graph_scale = 1.0  →  edge-odds multiplier = 2.0×   (each edge is twice as likely)
graph_scale = 2.0  →  edge-odds multiplier = 4.0×   (heavy structural drift)
```

**`mechanism_scale`** — tilts the mechanism-family sampling distribution toward
nonlinear families (nn, tree, gp, product) and away from simpler families
(linear, quadratic). The tilt is applied as a logit reweighting within the
enabled family support:

```
mechanism_scale = 0.0  →  no tilt; families sampled at baseline mix weights
mechanism_scale = 0.5  →  moderate tilt toward nonlinear families
mechanism_scale = 1.0  →  strong tilt; linear/quadratic become rare
mechanism_scale = 2.0  →  very strong tilt; almost all mechanisms are nonlinear
```

**`variance_scale`** — multiplies the noise standard deviation globally. The
runtime variance multiplier is `2^variance_scale`:

```
variance_scale = 0.0  →  variance multiplier = 1.0×   (no change)
variance_scale = 0.5  →  variance multiplier ≈ 1.41×  (+1.5 dB; moderate noise increase)
variance_scale = 1.0  →  variance multiplier = 2.0×   (+3 dB; noise variance doubles)
variance_scale = 2.0  →  variance multiplier = 4.0×   (+6 dB; very noisy regime)
```

______________________________________________________________________

## Generation workflows

Run any shift-enabled config:

```bash
dagzoo generate --config path/to/shift_config.yaml --num-datasets 25 --out data/run_shift
```

Use discoverable smoke presets:

```bash
dagzoo generate --config configs/preset_shift_graph_drift_generate_smoke.yaml --num-datasets 25 --out data/run_shift_graph
dagzoo generate --config configs/preset_shift_mechanism_drift_generate_smoke.yaml --num-datasets 25 --out data/run_shift_mechanism
dagzoo generate --config configs/preset_shift_noise_drift_generate_smoke.yaml --num-datasets 25 --out data/run_shift_noise
dagzoo generate --config configs/preset_shift_mixed_generate_smoke.yaml --num-datasets 25 --out data/run_shift_mixed
```

______________________________________________________________________

## What to inspect

- Per-dataset `metadata.ndjson` records include resolved mode/scales and derived
  multipliers (`edge_odds_multiplier`, `noise_variance_multiplier`,
  `mechanism_nonlinear_mass`).
- Diagnostics coverage summaries include shift observability metrics such as
  `shift_graph_scale`, `shift_edge_odds_multiplier`,
  `shift_mechanism_nonlinear_mass`, and `shift_noise_variance_multiplier`.

Benchmark runs can surface `shift_guardrails` in summaries.

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
