# Design Decisions

Lightweight Architecture Decision Records (ADRs) for foundational choices
in dagzoo.

______________________________________________________________________

## 1. Latent variable edge sampling for DAG structure

### Context

The generator builds random DAGs to define causal structure. Each potential
edge needs a probability. Standard Erdős–Rényi graphs sample each edge with
the same fixed probability, producing homogeneous structure with thin-tailed
degree distributions.

### Decision

Edge probabilities follow an additive logit model using latent variables
drawn from a Cauchy distribution:

```
p_ij = sigmoid(A + B_i + C_j + edge_logit_bias)
```

where A is a global latent scalar, B_i is a per-source-node latent variable,
and C_j is a per-target-node latent variable, all drawn from the standard
Cauchy distribution. Only upper-triangular entries are kept to enforce
acyclicity.

This documents the current baseline implementation of **latent variable edge
sampling** and may be extended via future roadmap work while preserving
backward-compatible defaults.

### Rationale

- **Heavy tails via Cauchy latents** — using the Cauchy distribution for the
  latent variables produces occasional extreme logit values, creating natural
  variability in graph density within and across datasets. Some nodes become
  hubs; others stay sparse.
- **Node-level heterogeneity** — separate per-row (B_i) and per-column (C_j)
  terms let individual nodes have distinct connectivity profiles.
- **Global sparsity control** — the `edge_logit_bias` additive term shifts
  the entire probability surface up or down, enabling controlled graph-density
  variation across generated datasets.
- **Theoretical grounding** — directly implements the mechanism described in
  TabICLv2 Appendix E.4.

### Alternatives considered

- **Erdős–Rényi (uniform p)** — too homogeneous; every node looks the same.
- **Power-law degree sequences** — adds structural diversity but requires a
  separate degree-to-DAG conversion step and does not naturally give per-edge
  control.
- **Normal distribution for logits** — lighter tails suppress the extreme hub
  / isolate patterns that create interesting downstream data.

______________________________________________________________________

## 2. Torch-only generation pipeline (no NumPy)

### Context

The data generation pipeline involves sampling random graphs, applying
function families, running converters, and postprocessing. These operations
could use NumPy, PyTorch, or a mix.

### Decision

All tensor computation in the generation pipeline uses PyTorch exclusively.
NumPy is used only at the I/O boundary (Parquet serialization).

This reflects the current baseline execution path; future roadmap work may
expand mechanism/noise controls (RD-011/RD-012) while keeping generation
backend strategy explicit.

### Rationale

- **Device-agnostic** — the same code path runs on CPU, CUDA, and MPS
  without if/else branching for backend.
- **No transfer bottlenecks** — a mixed pipeline requires CPU↔GPU copies at
  every NumPy↔Torch boundary. Keeping everything in Torch avoids this.
- **Single RNG system** — `torch.Generator` provides deterministic,
  device-aware random number generation. Mixing in `numpy.random` would
  require maintaining two RNG states and reasoning about their interaction.

### Alternatives considered

- **NumPy-first** — simpler for CPU-only use, but forfeits GPU acceleration
  entirely.
- **Mixed NumPy + Torch** — maximizes library ergonomics per operation but
  introduces transfer overhead and dual-RNG complexity.

______________________________________________________________________

## 3. BLAKE2s for seed derivation

### Context

The generator uses a tree of deterministic seeds: one base seed spawns child
seeds for each component (graph sampling, function selection, data generation,
missingness, fixed-layout planning, etc.). The derivation function must map
`(base_seed, component_path)` to a child seed without collisions.

### Decision

Child seeds are derived via BLAKE2s hashing: encode the base seed and path
components as UTF-8, hash them, and extract a 32-bit integer from the digest.

This is the current seeded reproducibility baseline. `KeyedRng` is the
canonical semantic RNG surface built on top of this hash primitive; see
[`keyed-rng.md`](keyed-rng.md) for the repo's namespace and replay
contract.

### Rationale

- **No collisions** — a cryptographic hash ensures that nearby inputs (e.g.,
  seeds 1000 and 1001, or paths "feature/0" and "feature/1") produce
  uncorrelated outputs. Simple arithmetic (addition, XOR) leaks input
  structure into the derived seed.
- **Uniform output** — BLAKE2s output bits are statistically uniform, so
  derived seeds cover the full 32-bit range evenly.
- **Fast** — BLAKE2s is specifically optimized for short inputs. The overhead
  per derivation is negligible compared to the tensor operations that follow.
- **Deterministic** — given the same base seed and component path, the
  derived seed is always identical, which is the foundation of reproducibility.

### Alternatives considered

- **Linear combination / XOR** — fast but produces structured collisions
  (e.g., `seed ^ 1` and `(seed+1) ^ 0` can alias).
- **SHA-256** — equally correct but slower than BLAKE2s for the short-input
  use case.
- **Counter-based (Philox/Threefry)** — good for bulk random streams but
  less natural for tree-structured seed hierarchies with string-named
  components.

______________________________________________________________________

## 4. Canonical fixed-layout execution model

### Context

The project needs one deterministic execution model that preserves emitted
schema alignment across a run, keeps replay behavior stable, and avoids
maintaining separate dynamic and fixed-layout runtimes.

### Decision

Canonical `generate_one`, `generate_batch`, and `generate_batch_iter` use the
fixed-layout engine internally. The old dynamic executor and explicit
fixed-layout public workflow have been removed.

### Rationale

- **Lower branching complexity** — one execution model replaces dual dynamic
  and fixed-layout runtimes.
- **Emitted-schema contract** — fixed-layout batches guarantee aligned emitted
  columns (feature count/order and lineage mapping), so index-based downstream
  consumers can safely stack bundles.
- **Deterministic reproducibility** — one run seed yields one stable shared
  layout signature, while dataset child seeds still vary value realizations.
  Under `chunk_batched_v1`, canonical outputs are deterministic for the same
  run seed and realized run shape.
- **Batch completion guarantee** — classification runs validate the requested
  canonical run before emission so public batches do not fail after partial
  output.

### Alternatives considered

- **Keep separate dynamic and fixed-layout runtimes** — rejected because it
  preserves duplicate orchestration, testing, and documentation burden.
- **Config-only fixed-layout mode** — not pursued.

______________________________________________________________________

## 5. `slots=True` dataclasses

### Context

The generator creates thousands of intermediate dataclass instances per batch
(dataset bundles, converter specs, seed managers, metrics containers, etc.).
Python dataclasses default to storing attributes in a per-instance `__dict__`.

### Decision

All dataclasses in the codebase use `@dataclass(slots=True)`.

This is a codebase-wide convention today, not a protocol contract.

### Rationale

- **Memory** — eliminates the per-instance `__dict__` allocation. When
  generating thousands of dataset bundles in a batch, the cumulative savings
  are meaningful.
- **Speed** — slot-based attribute access is a direct offset lookup instead
  of a dictionary lookup.
- **Safety** — prevents accidental attribute addition (e.g., typos like
  `bundle.metdata = ...`), signaling that these are structured containers
  with a fixed schema.

### Alternatives considered

- **NamedTuples** — immutable and memory-efficient, but lack default values
  and mutation support that dataclasses provide.
- **Plain dicts** — no attribute typo protection and no type annotation
  support.
- **`__slots__` without dataclasses** — equivalent runtime behavior but
  requires manual `__init__`, `__repr__`, etc.

______________________________________________________________________

## 6. Current function-family baseline

### Context

Each DAG node applies a random function to its parent values to produce a
latent representation. The choice of available function families determines
the space of possible data-generating processes.

### Decision

Current default family set includes eight families: neural network, tree ensemble, discretization (nearest
center), Gaussian process (random Fourier features), linear projection,
quadratic forms, EM-style soft assignment, and product (elementwise product
of two sub-families). Multi-parent nodes additionally use aggregation
strategies (sum, product, max, logsumexp).

Family and parameterization expansion is an explicit roadmap direction
(RD-011), with related noise-family expansion tracked separately (RD-012).

### Rationale

- **Theoretical grounding** — implements the mechanism families from
  TabICLv2 Appendix E.8.
- **Diversity across axes** — linear vs. nonlinear, smooth vs. piecewise,
  sparse vs. dense interactions. This diversity is necessary for the
  generated data to cover the space of real-world tabular relationships.
- **Multiplicative interactions** — the product family creates interaction
  effects not achievable by any single family, which is critical for
  modeling feature interactions in tabular data.
- **Composability** — multi-parent aggregation and the product family enable
  higher-order compositions without an exponential explosion in family count.

### Alternatives considered

- **Fewer families (e.g., NN-only)** — simpler but produces a narrower
  distribution of data characteristics. Neural networks alone struggle to
  produce sharp piecewise or nearest-neighbor-like structures.
- **More families** — diminishing returns on diversity; each new family adds
  code and testing surface. The current set covers the major structural
  archetypes.
- **Symbolic regression / GP trees** — expressive but hard to control for
  numerical stability and output distribution.

______________________________________________________________________

## 7. Noise family selection for RD-012 phase 1

### Context

Current generation uses implicit Gaussian-driven stochasticity throughout
matrix, weight, and point sampling, with optional global variance scaling via
`variance_sigma_multiplier` from shift controls. Epic `#24` and issue `#25`
introduce explicit user-facing noise-family configuration.

### Decision

Issue `#25` should start with four selectable modes:

- `gaussian`
- `laplace`
- `student_t`
- `mixture`

`mixture` is constrained to weighted combinations of Gaussian, Laplace, and
Student-t components only.

### Rationale

- **Simple baseline default** — `gaussian` remains the default and preserves
  seeded reproducibility with an explicit family label.
- **Coverage without excessive surface area** — Gaussian, Laplace, and
  Student-t provide progressively heavier tails with interpretable behavior.
- **Controlled flexibility** — `mixture` allows blended regimes without opening
  an unbounded family space in phase 1.
- **Numerical stability guardrails** — `student_t` should require `df > 2`
  to ensure finite variance and avoid unstable extreme draws.

### Alternatives considered

- **Expose Cauchy directly in phase 1** — rejected for now due infinite
  variance and high instability risk in downstream transforms.
- **Single-family only (Gaussian)** — rejected because it does not close the
  realism/robustness coverage gap targeted by RD-012.
- **Large family menu initially** — rejected to keep validation, docs, and
  benchmarking tractable for first delivery.

______________________________________________________________________

## 8. Single-source docs with Hugo-rendered reference pages

### Context

The docs site combines authored Markdown guides, two heavyweight technical
reference pages (`how-it-works` and `transforms`), and a Hugo/Docsy frontend.
Without a clear boundary between authored sources, generated Hugo inputs, and
deployable build output, contributors can end up editing the wrong files or
validating the wrong build tree.

### Decision

Maintain a single-source docs model with explicit generated boundaries:

- canonical authored docs live under `docs/`
- the generated Hugo input directory inside `site/` is described in
  `site/README.md`, regenerated by `scripts/docs/sync_hugo_content.py`, and
  ignored by git
- `how-it-works.md` and `transforms.md` remain canonical reference sources in
  `docs/`
- Hugo renders those two pages directly as normal docs pages
- canonical built output lives under the Hugo app in `site/`; see `site/README.md`

Top-level `public/` is treated as stale local output from the older build flow,
not the deployment source of truth.

### Rationale

- **Clear edit boundaries** — contributors know whether they should edit
  `docs/`, generated Hugo inputs, or only local build artifacts.
- **Preserve canonical technical references** — the two heavyweight reference
  pages stay authored once and render inside the normal docs site without a
  separate static/iframe layer.
- **Deterministic docs automation** — `scripts/docs/sync_hugo_content.py` owns
  generated inputs and CI validates the same canonical build tree used for
  Pages deploys.

### Alternatives considered

- **Keep both `public/` and the Hugo app's built output under `site/` as equivalent outputs** — rejected
  because it creates ambiguity about link checking, deploy inputs, and local
  validation.
- **Keep a separate static canonical HTML layer** — rejected because it
  duplicates deployment paths and creates unnecessary wrapper/static plumbing.

______________________________________________________________________

## 9. Factorized complete-data prior semantics

### Context

`dagzoo` now documents a default prior where the latent DAG generates complete
features first, the target is sampled from that complete feature table, and
optional missingness is applied later as an observation process. This is an
important internal modeling choice, but the research framing around it is too
deep for the first-read user path in `README.md` and the public docs.

### Decision

Keep the public docs focused on the observable behavior of the shipped prior:

- latent DAG -> complete features `X_complete`
- separate target head samples `y | X_complete`
- optional missingness masks the emitted feature table afterward

Keep the deeper research framing in internal docs only:

- the default prior is factorized in the sense of Nagler section 2.2 when `X`
  is interpreted as complete covariates
- `localization_mode` and `n_adaptation` remain `none` in the shipped recipes
- the current implementation should not be read as making direct monotone
  variance-or-bias-versus-`n` claims

### Rationale

- **Cleaner user path** — users need to understand what the generator does and
  what contracts are stable, not the full research caveat stack behind the
  prior.
- **Preserve maintainer context** — contributors still need the theoretical
  framing to evaluate future prior work and interpret fields like
  `localization_mode` and `n_adaptation`.
- **Avoid accidental overclaiming** — keeping the caveat explicit in internal
  docs reduces the chance that future work treats the shipped prior as if it
  already implemented localization or dataset-size adaptation.

### Alternatives considered

- **Keep the full theory note in user-facing docs** — rejected because it makes
  the first-read path feel like maintainer commentary.
- **Delete the caveat entirely** — rejected because the distinction matters for
  future prior design and internal review.

______________________________________________________________________

## Evolution Policy

- These ADRs document the current baseline implementation and rationale.
- Roadmap items may supersede ADR specifics; `development/roadmap.md` is authoritative
  for planned evolution.
- When roadmap delivery changes a decision detail materially, update this file
  in the same PR and record the rationale.
