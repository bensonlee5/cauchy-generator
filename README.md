# dagzoo

`dagzoo` generates reproducible synthetic tabular corpora from sampled causal
structure. The stable adoption layer is a small set of named recipe packs plus
stable artifact contracts; repo-internal authoring under `configs/` remains
available for advanced work, but it is not the primary public entrypoint.

## Start

Use the packaged CLI when you want the public workflow without a repo checkout:

```bash
uv tool install dagzoo
dagzoo recipe list
dagzoo generate --config recipe:default-baseline --num-datasets 25 --out data/default_baseline
dagzoo generate --config recipe:tabpfn-v1-prior-approx --num-datasets 25 --out data/tabpfn_prior
dagzoo filter --in data/default_baseline --out data/default_baseline_filter
```

Use a repo checkout when you want to edit configs, run docs tooling, or work on
the codebase:

```bash
./scripts/dev bootstrap
source .venv/bin/activate
./scripts/dev verify quick
```

For in-process training loops, use the same recipe references through the
PyTorch bridge:

```python
from dagzoo import build_dataloader

loader = build_dataloader(
    "recipe:default-baseline",
    num_datasets=10,
    seed=7,
    device="cpu",
)
sample = next(iter(loader))
print(sample["X_train"].shape)
```

## Public Surface

- `dagzoo recipe list` shows the curated public catalog.
- `dagzoo generate --config recipe:<name>` is the primary reproducible CLI path.
- `build_dataloader("recipe:<name>", ...)` is the programmatic equivalent.
- `recipes/*.yaml` are the published recipe sources behind those stable names.
- `configs/*.yaml` remain useful for advanced/internal authoring, but they move
  faster than the named recipe surface.

## Docs

- [Start](docs/start.md)
- [Reference Packs](docs/reference-packs.md)
- [Advanced Controls](docs/usage-guide.md)
- [Artifacts & API](docs/output-format.md)
- [How It Works](docs/how-it-works.md)
- [Feature Guides](https://bensonlee5.github.io/dagzoo/docs/features/)
- [Roadmap](docs/development/roadmap.md)

## Community

- [CITATION.cff](CITATION.cff)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
