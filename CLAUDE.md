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
./scripts/dev ready

# Canonical verification (run before any PR)
./scripts/dev verify quick

# Tests
uv run pytest -q                                          # all tests
uv run pytest tests/test_generate.py -q                   # single file
uv run pytest tests/test_generate.py::test_name -q        # single test
uv run pytest --testmon -q                                # incremental (changed only)
uv run pytest -n auto -q                                  # parallel
uv run pytest --cov=dagzoo --cov-report=term-missing -q   # with coverage (85% minimum)

# Lint & format
uv run ruff check src tests scripts
uv run ruff format src tests scripts

# Type check
uv run mypy src

# Dead code detection
uv run vulture src/dagzoo tests --ignore-names __getattr__

# Dependency & architecture checks
uv run deptry .
uv run lint-imports

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
- **Typed config** (`config.py`): Deeply nested dataclasses. Resolution precedence: YAML → CLI overrides → hardware policy → defaults. All resolution events traced in `effective_config_trace.yaml`.

### Key Directories

| Path                      | Role                                                   |
| ------------------------- | ------------------------------------------------------ |
| `src/dagzoo/core/`        | Generation pipeline orchestration (20 modules)         |
| `src/dagzoo/functions/`   | Mechanism families (linear, nn, tree, gp, etc.)        |
| `src/dagzoo/converters/`  | Latent-to-observable converters (numeric, categorical) |
| `src/dagzoo/sampling/`    | Noise families, missingness, correlated sampling       |
| `src/dagzoo/filtering/`   | Deferred ExtraTrees filtering                          |
| `src/dagzoo/io/`          | Parquet writer, lineage artifacts, schema              |
| `src/dagzoo/bench/`       | Benchmark suite, guardrails, regression detection      |
| `src/dagzoo/diagnostics/` | Coverage aggregation, effective diversity audit        |
| `scripts/dev.py`          | Dev tooling: doctor, deps, impact, contract, verify    |

## Development Rules

- No legacy pathways, duplicate pathways, or shims. No parallel implementations of the same logic.
- Internal Python APIs may change freely. CLI flags, metadata schema, or artifact contract changes are user-facing breaks — call them out explicitly.
- Version bump in `pyproject.toml` (patch default, minor for broad breaks) + `CHANGELOG.md` update in the same PR for behavior/schema changes. Docs/tests-only changes skip bumps.
- Run `./scripts/dev ready` before declaring a branch ready.
- Use `./scripts/dev impact` for dependency-aware ripple checks before broad refactors.

## Test Fixtures

`tests/conftest.py` provides:

- `make_generator(seed=42)` — seeded `torch.Generator` on CPU
- `make_keyed_rng(generator, *components)` — derives `KeyedRng` from generator
