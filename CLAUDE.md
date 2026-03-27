# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

dagzoo is a high-throughput synthetic tabular data generator built around causal structure. It generates reproducible datasets from latent DAGs mapped to observable tabular features. Core dependencies: numpy, torch, pyarrow, scikit-learn, pyyaml.

## Commands

```bash
# Setup
./scripts/dev bootstrap

# Pre-review flow
./scripts/dev review-base
./scripts/dev impact

# Canonical verification (run before any PR)
.venv/bin/nox -s quick

# Tests
.venv/bin/python -m pytest -q                             # all tests
.venv/bin/python -m pytest tests/test_generate.py -q      # single file
.venv/bin/python -m pytest tests/test_generate.py::test_name -q
.venv/bin/python -m pytest -n auto -q                     # parallel
.venv/bin/nox -s full                                     # repo-wide checks + tests

# Lint & format
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m ruff format src tests scripts

# Type check
.venv/bin/python -m mypy src

# Dead code detection
.venv/bin/python -m vulture src/dagzoo tests --ignore-names __getattr__

# Dependency & architecture checks
.venv/bin/python -m deptry .
.venv/bin/lint-imports

# Change impact analysis (before broad refactors)
./scripts/dev impact
```

## Architecture

### Layer Model

The codebase enforces strict import boundaries via import-linter (`.importlinter`):

```
Product surfaces (cli, bench, diagnostics)
        ↓ depend on
Core (dagzoo.core)
        ↓ depends on
Libraries (functions, converters, sampling, io, filtering, graph, linalg, postprocess)
```

**Libraries and core cannot import product surfaces.** This is enforced by CI.

### Public API

Defined in `src/dagzoo/__init__.py` — intentionally minimal:

- `GeneratorConfig`, `DatasetBundle`
- `generate_one()`, `generate_batch()`, `generate_batch_iter()`
- `apply_hardware_policy()`, `list_hardware_policies()`, `register_hardware_policy()`

### Generation Pipeline (data flow)

```
YAML config → config_resolution.py (hardware detect + policy + CLI overrides)
  → layout.py (sample DAG, assign features to nodes)
  → fixed_layout_batched.py (build execution plans, topological node traversal)
  → noise_runtime.py + postprocess.py (noise injection, train/test split, missingness)
  → filtering/deferred_filter.py (optional ExtraTrees acceptance)
  → io/parquet_writer.py (write shards + metadata)
```

The canonical entry is `core/dataset.py` → `core/fixed_layout_runtime.py`. One layout+plan is sampled per run and reused across all emitted datasets for schema stability.

### Key Design Patterns

- **KeyedRng** (`rng.py`): Deterministic reproducibility via blake2s-hashed semantic RNG namespaces. Each component derives its own keyed seed path — no ambient generator state leakage. See `docs/development/keyed-rng.md`.
- **Execution plans**: Plans encode node specs without executing, enabling pre-validation (e.g., classification split feasibility) and batched optimization.
- **Typed config** (`config.py`): Deeply nested dataclasses. Resolution precedence: YAML → focused CLI overrides / `--set` path overrides → hardware policy → defaults. Effective resolution traces are serialized as `{path, source, old_value, new_value}` events.

### Key Directories

| Path                      | Role                                                       |
| ------------------------- | ---------------------------------------------------------- |
| `src/dagzoo/core/`        | Generation pipeline orchestration (20 modules)             |
| `src/dagzoo/functions/`   | Mechanism families (linear, nn, tree, gp, etc.)            |
| `src/dagzoo/converters/`  | Latent-to-observable converters (numeric, categorical)     |
| `src/dagzoo/sampling/`    | Noise families, missingness, correlated sampling           |
| `src/dagzoo/filtering/`   | Deferred ExtraTrees filtering                              |
| `src/dagzoo/io/`          | Parquet writer, lineage artifacts, schema                  |
| `src/dagzoo/bench/`       | Benchmark suite, scenario evaluation, regression detection |
| `src/dagzoo/diagnostics/` | Coverage aggregation, effective diversity audit            |
| `scripts/dev.py`          | Dev tooling: bootstrap, impact, contract, review-base      |

## Development Rules

- No legacy pathways, duplicate pathways, or shims. No parallel implementations of the same logic.
- Internal Python APIs may change freely. CLI flags, metadata schema, or artifact contract changes are user-facing breaks — call them out explicitly.
- Version bump in `pyproject.toml` (patch default, minor for broad breaks) + `CHANGELOG.md` update in the same PR for behavior/schema changes. Docs/tests-only changes skip bumps.
- Run `.venv/bin/nox -s quick` before declaring a branch ready.
- Use `./scripts/dev impact` for dependency-aware ripple checks before broad refactors.

## Test Fixtures

`tests/conftest.py` provides:

- `make_generator(seed=42)` — seeded `torch.Generator` on CPU
- `make_keyed_rng(generator, *components)` — derives `KeyedRng` from generator
