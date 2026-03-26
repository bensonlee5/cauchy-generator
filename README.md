# dagzoo

`dagzoo` generates reproducible synthetic tabular corpora from sampled causal
structure. Most users should start with the named `recipe:<name>` configs and
the standard output layout they produce. Repo-local authoring under `configs/`
is still available for advanced work, but it is not the main public entrypoint.

```mermaid
flowchart LR
    classDef setup fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b
    classDef core fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100
    classDef out fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c

    Seed([Root Seed]) --> RNG[Deterministic Seeding]
    RNG --> Layout[Layout & DAG Sampling]
    Layout --> Mechanisms[Random Functional Mechanisms]
    Mechanisms --> Converters[Feature/Target Converters]
    Converters --> Bundle[[DatasetBundle: X, y, Metadata]]

    class Seed,RNG setup
    class Layout,Mechanisms,Converters core
    class Bundle out
```

### From Latent DAG to Tabular Data

Unlike generators that treat each column as independent noise, `dagzoo`
generates data from a latent causal structure. One node in the sampled graph
can branch into multiple observable features, which preserves dependency
patterns in the emitted table.

```mermaid
flowchart LR
    classDef latent fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b,stroke-dasharray: 5 5
    classDef observable fill:#f5f5f5,stroke:#212121,stroke-width:2px,color:#212121

    subgraph LatentSpace [Latent Causal DAG]
        NodeA((Node A)) --> NodeB((Node B))
    end

    subgraph ObservableSpace [Tabular Dataset Layout]
        Feat1[Feature 1: Numeric]
        Feat2[Feature 2: Categorical]
        Feat3[Feature 3: Numeric]
        Target[Target Variable]
    end

    NodeA -. mapping .-> Feat1
    NodeA -. mapping .-> Feat2
    NodeB -. mapping .-> Feat3
    NodeB -. mapping .-> Target

    class NodeA,NodeB latent
    class Feat1,Feat2,Feat3,Target observable

    style LatentSpace fill:#f0faff,stroke:#01579b,stroke-dasharray: 5 5
    style ObservableSpace fill:#fafafa,stroke:#212121
```

## Start

Use the packaged CLI when you want the public workflow without a repo checkout.
These are the main `dagzoo` commands most users start with:

```bash
uv tool install dagzoo

# Inspect the curated recipe catalog and see the stable public names.
dagzoo recipe list

# Generate a general-purpose baseline run under data/default_baseline/.
dagzoo generate --config recipe:default-baseline --num-datasets 25 --out data/default_baseline

# Generate a smaller numeric-heavy run with the published TabPFN-style recipe.
dagzoo generate --config recipe:tabpfn-v1-prior-approx --num-datasets 25 --out data/tabpfn_prior
```

Use a repo checkout when you want to edit configs, run docs tooling, or work on
the codebase:

```bash
./scripts/dev bootstrap
source .venv/bin/activate
./scripts/dev verify quick
```

For in-process training loops, use the same recipe references through the
PyTorch bridge. `build_dataloader(...)` is the in-process equivalent of running
`dagzoo generate --config recipe:<name>` from the CLI:

```python
from dagzoo import build_dataloader

# Load the same baseline recipe directly into a training loop.
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

If you're new, start with the named recipes. The public surface is small on
purpose:

- `dagzoo recipe list` shows the curated recipe catalog.
- `dagzoo generate --config recipe:<name>` generates datasets from one of those
  published recipes.
- `build_dataloader("recipe:<name>", ...)` gives you the same recipe surface
  inside Python.

`recipe:<name>` is the stable public config handle most users should reach for
first. `recipes/*.yaml` are the published YAML files behind those names, so you
can inspect exactly what a recipe contains. Repo-local `configs/*.yaml` are for
custom authoring and internal iteration, and they move faster than the named
recipe surface.

For example, this command generates 25 datasets from the baseline recipe:

```bash
# recipe:default-baseline is the named public config.
# --out chooses the run directory on disk.
dagzoo generate --config recipe:default-baseline --num-datasets 25 --out data/default_baseline
```

That run lands under `data/default_baseline/` because the path is passed to
`--out`.

### What Lands on Disk

After that generate command finishes, this is the kind of layout you should
expect under the run root:

```text
data/default_baseline/
  effective_config.yaml
  effective_config_trace.yaml
  shard_00000/
    train.parquet
    test.parquet
    metadata.ndjson
    lineage/
      adjacency.bitpack.bin
      adjacency.index.json
```

The `shard_*` directories hold the generated datasets. `effective_config.yaml`
records the fully resolved config for the run, and
`effective_config_trace.yaml` records where overrides came from so the run is
reproducible. The full artifact contract lives in `docs/output-format.md`.

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
