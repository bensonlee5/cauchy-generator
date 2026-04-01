# Codebase Navigation

The project is organized into functional modules that manage the lifecycle
of a synthetic dataset, from configuration and causal graph sampling to
node execution and quality filtering.

For dependency-aware ripple analysis, see
[`module-dependency-map.md`](module-dependency-map.md). It complements this file
with a generated package graph plus hotspot reverse-dependency summaries. For
example, changes in `src/dagzoo/core/execution_semantics.py` currently cascade
into `fixed_layout/batched.py`, `node_pipeline.py`, converter modules, function
modules, and then outward into `fixed_layout/runtime.py`, benchmarks, and CLI
surfaces.

## 1. Entry Points & Orchestration

The high-level logic that bridges CLI/API requests to the canonical generation
runtime.

- [`src/dagzoo/cli/`](../../src/dagzoo/cli/__init__.py): Maps CLI flags to `GeneratorConfig` and handles command dispatch.
- [`src/dagzoo/core/dataset.py`](../../src/dagzoo/core/dataset.py): Public generation facade. Resolves canonical runs, annotates replay metadata, and streams emitted bundles.
- [`src/dagzoo/core/generation_runtime.py`](../../src/dagzoo/core/generation_runtime.py): Shared split/finalization helpers used by canonical fixed-layout generation.
- [`src/dagzoo/core/fixed_layout/runtime.py`](../../src/dagzoo/core/fixed_layout/runtime.py): Canonical fixed-layout run preparation, classification replay validation, and batched execution orchestration.
- [`src/dagzoo/core/fixed_layout/metadata.py`](../../src/dagzoo/core/fixed_layout/metadata.py): Shared fixed-layout metadata helpers and layout signatures.
- [`src/dagzoo/core/config_resolution.py`](../../src/dagzoo/core/config_resolution.py): Layered config resolution, produces `effective_config_trace.yaml`.

## 2. The Generation Pipeline (The "Assembly Line")

Follow this sequence to understand how a latent causal structure becomes a realized dataset.

- **Structure ([`graph/`](../../src/dagzoo/graph/)):** Samples the underlying Directed Acyclic Graph (DAG).
- **Layout ([`core/layout.py`](../../src/dagzoo/core/layout.py)):** Maps features and targets to DAG nodes and assigns data types.
- **Execution ([`core/fixed_layout/batched.py`](../../src/dagzoo/core/fixed_layout/batched.py)):** Samples typed node plans and executes them in batched topological order.
- **Leaf node helper ([`core/node_pipeline.py`](../../src/dagzoo/core/node_pipeline.py)):** Isolated node-plan execution helper used by tests and microbenchmarks.
- **Mechanisms ([`functions/`](../../src/dagzoo/functions/)):** Contains the mathematical families (linear, non-linear, mixture) that define how nodes interact.
- **Conversion ([`converters/`](../../src/dagzoo/converters/)):** Transforms latent continuous values into observable numeric or categorical data.
- **Sampling ([`sampling/`](../../src/dagzoo/sampling/)):** Noise-family primitives (gaussian/laplace/student_t), random-point geometry, correlated sampling.
- **Math ([`math/`](../../src/dagzoo/math/__init__.py)):** Shared numeric helpers and random matrix families used by mechanism functions.

## 3. Control, Integrity & Quality

Infrastructure that ensures reproducibility, deterministic behavior, and data quality.

- [`src/dagzoo/rng.py`](../../src/dagzoo/rng.py): `KeyedRng` is the repo's single semantic RNG surface for deterministic child seeds and keyed torch generators.
- [`docs/development/keyed-rng.md`](keyed-rng.md): Design note for the keyed RNG migration and semantic reproducibility contract.
- [`src/dagzoo/filtering/`](../../src/dagzoo/filtering/): Implements deferred structural filtering (`dagzoo filter`) and filter replay utilities.
- [`src/dagzoo/core/metrics_torch.py`](../../src/dagzoo/core/metrics_torch.py): Unified torch-native metric extraction used by diagnostics and generation telemetry.
- [`src/dagzoo/postprocess/`](../../src/dagzoo/postprocess/): Handles final-stage transformations; deterministic missingness (MCAR/MAR/MNAR) logic lives in [`sampling/missingness.py`](../../src/dagzoo/sampling/missingness.py).
- [`src/dagzoo/core/shift.py`](../../src/dagzoo/core/shift.py): Shift mode resolution and scale coefficients.
- [`src/dagzoo/core/noise_runtime.py`](../../src/dagzoo/core/noise_runtime.py): Per-dataset noise family resolution.

## 4. Configuration, Hardware & Architecture

- [`src/dagzoo/config/`](../../src/dagzoo/config/__init__.py): The source of truth for all generator settings, implemented as strongly-typed dataclasses.
- [`src/dagzoo/hardware.py`](../../src/dagzoo/hardware.py) + [`hardware_policy.py`](../../src/dagzoo/hardware_policy.py): Tier detection and extensible policy registry.
- [`docs/development/design-decisions.md`](design-decisions.md): Rationale behind the architectural choices and reproducibility guarantees.
- [`docs/development/keyed-rng.md`](keyed-rng.md): Active migration plan for keyed RNG namespaces and semantic reproducibility.

## 5. Output & Benchmarking

- [`src/dagzoo/io/`](../../src/dagzoo/io/): Parquet shard writer, lineage bit-packing and schema.
- [`src/dagzoo/bench/`](../../src/dagzoo/bench/): Benchmark suite orchestration, guardrails (missingness/lineage/shift/noise), baseline regression.

## 6. Current Developer Notes

- Docs source of truth:
  - authored docs live under `docs/`
  - the generated Hugo input and built-output directories inside `site/` are described in `site/README.md` and regenerated or validated by the docs scripts
  - top-level `public/` is stale local output from the older build flow
- Docs architecture:
  - `docs/how-it-works.md` and `docs/transforms.md` are canonical reference docs rendered as normal Hugo pages
- Hardware policy nuance:
  - `runtime.device: mps` is detected and preserved as an MPS backend
  - hardware tiering still maps MPS to the `cpu` tier because only CUDA has tiered policy rules today
- Local cleanup:
  - use `scripts/cleanup_local_artifacts.py` to dry-run or remove ignored runtime/docs outputs without touching tracked files

______________________________________________________________________

### Tip: Audit via Verification

To verify the current repo state while exploring the code:

```bash
./scripts/dev review-base
./.venv/bin/nox -s quick
```

For more targeted local checks:

```bash
./scripts/dev impact
./.venv/bin/nox -s quick
```
