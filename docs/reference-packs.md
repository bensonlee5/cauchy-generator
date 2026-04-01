# Reference Packs

Reference packs are the public, named generation configs for `dagzoo`.

They exist to make the tool usable without reverse-engineering the internal
config surface from scratch. A paper or benchmark can point to one of these
named packs directly, and a new user can run it immediately with:

```bash
dagzoo generate --config recipe:<name> --num-datasets 25 --out data/<run_name>
```

The same YAML files are also checked into the repo under `recipes/`.

______________________________________________________________________

## Stability model

- Stable adoption layer: `recipe:<name>` references and documented artifact contracts
- Faster-moving authoring layer: repo-local `configs/*.yaml`
- Confidence tiers:
  - `baseline`: maintained default starting point
  - `paper-backed approximation`: intended to approximate a published prior without overclaiming exact equivalence
  - `stress profile`: reproducible stress regime rather than a paper-prior claim
- Developer note: these curated recipe labels are not the RD-005 carried-slice
  selector contract; the first fixed-slice selector lives under
  `stress.profile=anti_memorization_piecewise_classification_slice_v1`.

______________________________________________________________________

## Catalog

| Recipe                    | Confidence                   | Expected regime                                                      | Repo YAML                              |
| ------------------------- | ---------------------------- | -------------------------------------------------------------------- | -------------------------------------- |
| `default-baseline`        | `baseline`                   | Balanced mixed-type classification with no extra stress regime       | `recipes/default-baseline.yaml`        |
| `tabpfn-v1-prior-approx`  | `paper-backed approximation` | Small-data, numeric-heavy classification                             | `recipes/tabpfn-v1-prior-approx.yaml`  |
| `high-cardinality-stress` | `stress profile`             | Categorical-heavy tasks with larger cardinality envelopes            | `recipes/high-cardinality-stress.yaml` |
| `missingness-robustness`  | `stress profile`             | Moderate-to-heavy structured missingness with explicit MNAR controls | `recipes/missingness-robustness.yaml`  |
| `shift-stress`            | `stress profile`             | Mixed graph-and-noise drift for controlled shift experiments         | `recipes/shift-stress.yaml`            |

______________________________________________________________________

## Recipe notes and citations

### `default-baseline`

- Purpose: general-purpose starting point for mixed-type classification studies
- Prior note: latent DAG with emitted features assigned to nodes and the target
  emitted from one selected latent node; optional missingness is a later
  observation model over the emitted features
- Citation note: cite `dagzoo` itself plus the specific recipe name

### `tabpfn-v1-prior-approx`

- Purpose: practical approximation for TabPFN-style small-data classification workflows
- Prior note: numeric-heavy latent-node prior with the same selected-node target
  derivation as the rest of the shipped catalog
- Citations:
  - `Accurate predictions on small data with a tabular foundation model`
  - `TabICLv2: A better, faster, scalable, and open tabular foundation model`

### `high-cardinality-stress`

- Purpose: stress categorical-heavy workloads that exceed the lighter default envelope
- Citation:
  - `Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks`

### `missingness-robustness`

- Purpose: force structured missingness into the training prior without hand-authoring the config
- Citations:
  - `A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities`
  - `TabICLv2: A better, faster, scalable, and open tabular foundation model`

### `shift-stress`

- Purpose: reproducible mixed drift for train/test shift stress testing
- Citation:
  - `Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data`

______________________________________________________________________

## Example usage

```bash
dagzoo recipe list
dagzoo generate --config recipe:default-baseline --num-datasets 25 --out data/default_baseline
dagzoo generate --config recipe:high-cardinality-stress --num-datasets 25 --out data/high_cardinality
```

You can also reference the same configs by path inside a repo checkout:

```bash
dagzoo generate --config recipes/default-baseline.yaml --num-datasets 25 --out data/default_baseline
```
