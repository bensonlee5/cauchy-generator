# Dagzoo Recipes

This directory holds the curated public recipe catalog for `dagzoo`.

These YAML files are the researcher-facing entrypoint. They are intended to be:

- discoverable
- reproducible
- stable enough to cite as named generation packs

Use them in either of these forms:

```bash
dagzoo recipe list
dagzoo generate --config recipe:default-baseline --num-datasets 25 --out data/run_default_recipe
dagzoo generate --config recipes/default-baseline.yaml --num-datasets 25 --out data/run_default_recipe
```

Public stability note:

- `recipe:<name>` references and artifact contracts are the stable adoption layer.
- Repo-internal configs under `configs/` remain available for advanced workflows, but they move faster and are not the primary user-facing surface.

## Catalog

| Recipe                    | Type            | Confidence                   | Purpose                                                                           |
| ------------------------- | --------------- | ---------------------------- | --------------------------------------------------------------------------------- |
| `default-baseline`        | Reference prior | `baseline`                   | Balanced mixed-type baseline for general PFN and tabular experiments.             |
| `tabpfn-v1-prior-approx`  | Reference prior | `paper-backed approximation` | Small-data numeric classification approximation inspired by TabPFN-era workflows. |
| `high-cardinality-stress` | Stress pack     | `stress profile`             | Categorical-heavy regimes with larger cardinality envelopes.                      |
| `missingness-robustness`  | Stress pack     | `stress profile`             | Structured missingness robustness workflows with explicit MNAR controls.          |
| `shift-stress`            | Stress pack     | `stress profile`             | Mixed graph-and-noise drift for train/test shift experiments.                     |
