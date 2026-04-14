# Missingness

Real-world tabular datasets are almost never complete. Missing data is endemic
across domains, and the *mechanism* of missingness determines its statistical
impact:

```
MCAR  (Missing Completely At Random)
      Each value is independently missing with some probability.
      Example: random sensor packet loss, database write failures.
      → Simplest mechanism.  Reduces sample size but does not bias estimates.

MAR   (Missing At Random)
      Missingness depends on *observed* values.
      Example: older survey respondents skip digital-interaction questions;
               patients with certain diagnoses are less likely to have lab X recorded.
      → Missingness patterns carry information about the observed features.
         A model that ignores this loses signal.

MNAR  (Missing Not At Random)
      Missingness depends on the *missing value itself*.
      Example: sicker patients have more missing lab results (the very values
               that would be most informative are the ones most likely absent).
      → Hardest mechanism.  Ignoring it introduces systematic bias.
```

A foundation model trained on a synthetic prior with no missing values has
never seen incomplete inputs during pretraining. At inference, it must handle
missingness zero-shot with no prior exposure. Incorporating missingness into
synthetic priors has demonstrated performance gains on real datasets with
incomplete features, and covering missingness regimes improves reliability in
weak meta-feature settings.

Use missingness workflows to inject deterministic synthetic null patterns for
robustness testing under MCAR, MAR, and MNAR regimes.

In `dagzoo`, missingness is an observation model applied after target
generation. The default prior first samples a latent DAG, emits complete
features from node-assigned converters, emits `y` from one selected latent
target node, then samples a missingness process and emits
`X_obs = mask(X_complete, m)`. This keeps latent target derivation separate
from the later censoring process.

______________________________________________________________________

## When to use

### Why it matters for your prior

- Your synthetic prior should reflect the near-universal presence of missing
  data in real tabular tasks — without it, the model's first encounter with
  incomplete inputs is at inference time.
- You want to test whether MNAR-aware pretraining produces more robust
  in-context learning than MCAR-only priors — a key question for
  medical/clinical tabular applications where informative missingness is the
  norm.
- You want to measure how effective diversity changes when missingness is
  present versus absent in the prior, as a targeted ablation of this axis.
- You need to control the missingness mechanism (MCAR/MAR/MNAR)
  independently from noise family and mechanism complexity to isolate its
  contribution to downstream performance.

### Operational triggers

- You want realistic training/evaluation with incomplete tabular data.
- You need controlled ablations across missingness mechanisms.
- You want benchmark guardrails to include missingness-aware checks.

______________________________________________________________________

## Preset workflows

Use presets for standard mechanism runs:

```bash
dagzoo generate --config configs/preset_missingness_mcar.yaml --num-datasets 25 --out data/run_missing_mcar
dagzoo generate --config configs/preset_missingness_mar.yaml --num-datasets 25 --out data/run_missing_mar
dagzoo generate --config configs/preset_missingness_mnar.yaml --num-datasets 25 --out data/run_missing_mnar
```

______________________________________________________________________

## Targeted MAR calibration via CLI

```bash
dagzoo generate \
  --config configs/default.yaml \
  --num-datasets 25 \
  --device cpu \
  --set dataset.missing_rate=0.25 \
  --set dataset.missing_mechanism=mar \
  --set dataset.missing_mar_observed_fraction=0.6 \
  --set dataset.missing_mar_logit_scale=1.4 \
  --out data/run_missing_cli_mar
```

______________________________________________________________________

## Key options

- `--set dataset.missing_rate=...`: overall missingness probability (fraction of cells that are
  `NaN` in the output). Concrete examples:

  ```
  --set dataset.missing_rate=0.05  →  5% of cells missing  (light; common in clean survey data)
  --set dataset.missing_rate=0.15  →  15% of cells missing (moderate; typical clinical datasets)
  --set dataset.missing_rate=0.30  →  30% of cells missing (heavy; EHR or sensor-network data)
  ```

- `--set dataset.missing_mechanism=...`: which statistical mechanism drives the missingness.
  `mcar` = independent coin flip per cell; `mar` = missingness depends on
  complete feature values through the observation model; `mnar` = missingness depends on
  the value being censored.

- `--set dataset.missing_mar_observed_fraction=...`: fraction of features used to compute MAR
  logits (higher = more features influence which values go missing).

  ```
  --set dataset.missing_mar_observed_fraction=0.3  →  30% of features drive missingness
  --set dataset.missing_mar_observed_fraction=0.8  →  80% of features drive missingness (strong MAR)
  ```

- `--set dataset.missing_mar_logit_scale=...`: MAR logit sensitivity multiplier. Higher values
  make missingness more sharply dependent on the complete features used by the
  observation model.

  ```
  --set dataset.missing_mar_logit_scale=0.5  →  weak MAR signal (missingness is nearly random)
  --set dataset.missing_mar_logit_scale=1.5  →  strong MAR signal (missingness is highly structured)
  ```

______________________________________________________________________

## What to inspect

- In-process `DatasetBundle.metadata["missingness"]` for resolved missingness
  configuration and realized rates.
- In-process `DatasetBundle.metadata["prior"]` for the latent-node target
  semantics versus the later observation-model step.
- Public `dataset_catalog.parquet` for the stable emitted dataset identity and
  schema surface.
- Benchmark summaries for `preset_results[*].scenarios.missingness` (when enabled).

For output details, see
[output-format.md](../output-format.md).

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
