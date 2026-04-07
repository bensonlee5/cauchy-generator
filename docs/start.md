# Start

This is the fastest path from install to usable synthetic tabular data.

The public entrypoint is the curated recipe catalog. Start with `recipe:<name>`
when you want something reproducible, discoverable, and easy to cite. Move to
repo-local `configs/` only when you need custom authoring beyond the published
recipes.

______________________________________________________________________

## 1. Install

Packaged install:

```bash
uv tool install dagzoo
```

Repo checkout:

```bash
./scripts/dev bootstrap
source .venv/bin/activate
```

______________________________________________________________________

## 2. Inspect the recipe catalog

```bash
dagzoo recipe list
```

That command prints the stable recipe names and the regime each one is meant to
approximate or stress. The catalog is the default adoption layer for `dagzoo`.
All shipped recipes use the same public prior shape: sample a latent DAG, emit
features from node-assigned converters, emit `y` from one selected latent node,
then optionally apply missingness as a separate observation process over the
emitted features.

______________________________________________________________________

## 3. Generate your first run

Balanced baseline with the default latent-node target prior:

```bash
dagzoo generate --config recipe:default-baseline --num-datasets 25 --out data/default_baseline
```

TabPFN-inspired numeric-heavy latent-node prior:

```bash
dagzoo generate --config recipe:tabpfn-v1-prior-approx --num-datasets 25 --out data/tabpfn_prior
```

Every generate run writes:

- `effective_config.yaml`
- `effective_config_trace.yaml`

`dagzoo generate` only generates. If you want accept/reject decisions, run
`dagzoo filter` as a separate replay stage over the emitted shards.
The same filter thresholds also guide generation-time structural retries.
Observational generation is the default. Hard interventions are opt-in through
`intervention.mode: hard_interventional`; when that section is absent, public
artifacts omit intervention metadata entirely.

______________________________________________________________________

## 4. Use the same recipe in process

```python
from dagzoo import build_dataloader

loader = build_dataloader(
    "recipe:default-baseline",
    num_datasets=10,
    seed=7,
    device="cpu",
)
sample = next(iter(loader))
```

`build_dataloader(...)` is the recommended programmatic entrypoint. It uses the
same config surface as the CLI: either `recipe:<name>` or a YAML path.
Public generation defaults to fully heterogeneous per-dataset layouts; set
`runtime.layout_mode: stratified` when you want large-run heterogeneous
generation to batch compatible `(n_rows, n_features)` strata without forcing a
shared layout. On Apple hardware, heterogeneous and stratified `device=auto`
now prefer CPU over MPS; pass `device="mps"` explicitly when you want the MPS
backend.

______________________________________________________________________

## 5. Where to go next

- Want the published catalog and citations: [reference-packs.md](reference-packs.md)
- Need custom generation controls: [usage-guide.md](usage-guide.md)
- Need observational vs hard-interventional workflows: [features/interventions.md](features/interventions.md)
- Need artifact and API contracts: [output-format.md](output-format.md)
- Need the exhaustive field catalog: [export-contract-fields.md](export-contract-fields.md)
- Want the runtime model: [how-it-works.md](how-it-works.md)
