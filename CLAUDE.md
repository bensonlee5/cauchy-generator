# CLAUDE.md

This file supplements `AGENTS.md` for Claude Code users working in this
repository. `AGENTS.md` remains the canonical operating contract.

## What This Is

`dagzoo` is a synthetic tabular data generator built around latent causal
structure. The public adoption layer is small on purpose:

- named `recipe:<name>` configs
- the packaged `dagzoo` CLI
- stable artifact contracts
- the PyTorch bridge exports in `src/dagzoo/__init__.py`

Repo-local `configs/` and internal Python modules move faster than that public
surface.

## Commands

```bash
# Setup
./scripts/dev bootstrap

# Pre-review flow
./scripts/dev review-base
./scripts/dev impact

# Canonical verification
.venv/bin/nox -s quick
.venv/bin/nox -s docs
.venv/bin/nox -s full

# Tests
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_generate.py -q
.venv/bin/python -m pytest tests/test_generate.py::test_name -q
.venv/bin/python -m pytest -n auto -q

# Lint and format
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m ruff format src tests scripts

# Type check
.venv/bin/python -m mypy src

# Dead code detection
.venv/bin/python -m vulture src/dagzoo tests --ignore-names __getattr__

# Dependency and architecture checks
.venv/bin/python -m deptry .
.venv/bin/lint-imports
```

## Architecture

### Layer Model

The codebase enforces import boundaries via import-linter (`.importlinter`):

```text
Product surfaces (cli, bench, diagnostics)
        -> depend on
Core (dagzoo.core)
        -> depends on
Libraries (functions, converters, sampling, io, filtering, graph, linalg, postprocess)
```

Libraries and core cannot import product surfaces.

### Public API

Defined in `src/dagzoo/__init__.py`:

- `GeneratorConfig`, `DatasetBundle`
- `DagzooDataset`, `DagzooSample`, `build_dataloader()`
- `generate_one()`, `generate_batch()`, `generate_batch_iter()`
- `apply_hardware_policy()`, `list_hardware_policies()`, `register_hardware_policy()`
- `get_peak_flops()`

### Public Generation Semantics

- Public generation defaults to heterogeneous per-dataset layout and plan
  sampling.
- `runtime.layout_mode: stratified` is the throughput-sensitive public option
  for batching compatible exact `(n_rows, n_features)` strata.
- Public `runtime.layout_mode: fixed` has been removed.
- `dagzoo generate` only generates. `dagzoo filter` is a separate deferred
  replay stage over emitted shards.

### Internal Runtime Shape

The public entrypoints live in `src/dagzoo/core/dataset.py`. They validate the
public config surface, normalize request-run identity, and route into the
shared runtime helpers under `src/dagzoo/core/fixed_layout/`.

That module family name is historical implementation detail, not the public
contract. The current public behavior is heterogeneous-by-default even though
the internal planning and grouped execution helpers still live under
`fixed_layout`.

High-level flow:

```text
YAML path or recipe reference
  -> config resolution and hardware policy
  -> core/dataset.py public entrypoints
  -> shared planning/runtime helpers
  -> postprocess and optional missingness
  -> DatasetBundle or shard artifacts
  -> optional later dagzoo filter replay stage
```

### Key Directories

| Path                      | Role                                                       |
| ------------------------- | ---------------------------------------------------------- |
| `src/dagzoo/core/`        | Generation orchestration and public runtime entrypoints    |
| `src/dagzoo/functions/`   | Mechanism families                                         |
| `src/dagzoo/converters/`  | Latent-to-observable converters                            |
| `src/dagzoo/sampling/`    | Noise families, missingness, correlated sampling           |
| `src/dagzoo/filtering/`   | Deferred structural filtering                              |
| `src/dagzoo/io/`          | Parquet writer, lineage artifacts, schema                  |
| `src/dagzoo/bench/`       | Benchmark suite and regression detection                   |
| `src/dagzoo/diagnostics/` | Coverage aggregation and effective-diversity audit         |
| `scripts/dev.py`          | Dev tooling: bootstrap, impact, contract, review-base      |

## Development Rules

- No legacy pathways, duplicate pathways, or compatibility shims without an
  explicit reason.
- Internal Python APIs may change freely. CLI flags, metadata schema, and
  artifact contract changes are user-facing and must be called out explicitly.
- Changes under `src/dagzoo` usually require a version bump in
  `pyproject.toml` plus a `CHANGELOG.md` update in the same PR. Docs-only and
  tests-only changes do not.
- Run `.venv/bin/nox -s quick` before declaring work ready.
- Use `./scripts/dev impact` before broad refactors.

## Test Fixtures

`tests/conftest.py` provides:

- `make_generator(seed=42)` for a seeded CPU `torch.Generator`
- `make_keyed_rng(generator, *components)` for derived `KeyedRng` instances
