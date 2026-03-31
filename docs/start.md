# Start

This is the fastest path from install to usable synthetic tabular data.

The public entrypoint is the curated recipe catalog. Start with `recipe:<name>`
references when you want something reproducible, discoverable, and easy to cite.
Move to repo-local `configs/` only when you need custom authoring beyond the
published recipes.

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
approximate or stress. All shipped recipes now use the default factorized prior:
sample observed `X` first, then sample `y | X`.

______________________________________________________________________

## 3. Generate your first run

Balanced baseline with the default factorized `p(x)` plus `p(y | x)` prior:

```bash
dagzoo generate --config recipe:default-baseline --num-datasets 25 --out data/default_baseline
```

TabPFN-inspired numeric-heavy factorized prior:

```bash
dagzoo generate --config recipe:tabpfn-v1-prior-approx --num-datasets 25 --out data/tabpfn_prior
```

Every generate run writes:

- `effective_config.yaml`
- `effective_config_trace.yaml`

`dagzoo generate` now only generates. If you want accept/reject decisions, run
`dagzoo filter` as a separate replay stage over the emitted shards.

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

______________________________________________________________________

## 5. Where to go next

- Want the published catalog and citations: [reference-packs.md](reference-packs.md)
- Need custom generation controls: [usage-guide.md](usage-guide.md)
- Need artifact and API contracts: [output-format.md](output-format.md)
- Want the runtime model: [how-it-works.md](how-it-works.md)
