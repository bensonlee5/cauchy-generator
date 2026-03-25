# Mechanism Diversity

The functional relationships in real tabular data span an enormous range —
linear regressions, tree-like decision boundaries, smooth nonlinear surfaces,
clustered assignments, multiplicative interactions. A synthetic prior that uses
only one or two mechanism families systematically undercovers this space:

```
Prior with only linear + NN:
  ✓ smooth low-complexity      (linear)
  ✓ smooth high-complexity     (neural network)
  ✗ piecewise discontinuous    (tree, discretization)  ← coverage gap
  ✗ smooth periodic/multi-scale (GP variants)           ← coverage gap
  ✗ soft mixture assignments   (EM)                     ← coverage gap
  ✗ multiplicative interactions (product)               ← coverage gap

Prior with all 9 families:
  ✓ all of the above — the foundation model sees the full functional spectrum
```

Mechanism diversity is one of the most direct levers for effective diversity —
the breadth of meta-feature space covered by the corpus. If all datasets have
the same functional complexity profile, the model sees a narrow prior and
generalizes poorly to tasks with different functional character. dagzoo uses
9 families specifically because each contributes qualitatively different latent
structure. Prior realism — including functional diversity — materially affects
downstream performance.

Use mechanism-diversity workflows when you want to exercise the existing
family-mix surface, compare candidate mechanism behavior against the current
baseline, and verify that the generated bundles actually realize the intended
families or variants.

______________________________________________________________________

## When to use

### Why it matters for your prior

- You suspect your prior has a functional complexity gap and want to measure
  whether widening mechanism families improves effective diversity (e.g.,
  adding GP variants to a prior that currently undercovers smooth periodic
  surfaces).
- You are comparing prior configurations to find the mechanism mix that
  maximizes meta-feature coverage for your downstream task distribution.
- You want to verify that a new mechanism variant (e.g., `gp.periodic`,
  `gp.multiscale`) actually contributes distinct latent behavior to the
  prior, not just redundant functional patterns.
- You are investigating whether the mechanism family axis or the noise axis
  has more impact on effective diversity and need to control each
  independently.

### Operational triggers

- You want to compare the current baseline sampler against the shipped
  `piecewise` control or the widened `gp` candidate path.
- You need realized mechanism-family and mechanism-variant counts in bundle
  metadata and audit reports.
- You want diversity-audit evidence before treating a new mechanism path as
  stable. Deferred filter calibration is temporarily unsupported.

______________________________________________________________________

## Understanding `function_family_mix`

The mix is a dictionary mapping family names to relative weights. Families
with weight 0 (or omitted) are excluded entirely; positive weights are
normalized to probabilities. Examples:

```yaml
# Default: all families equally likely
mechanism:
  function_family_mix: {}          # → each of the 9 families gets ~11% probability

# Heavy NN + some linear: generates mostly smooth nonlinear data
mechanism:
  function_family_mix:
    nn: 0.7
    linear: 0.3                    # → 70% NN, 30% linear; all other families excluded

# Broad coverage with GP emphasis:
mechanism:
  function_family_mix:
    nn: 1.0
    tree: 1.0
    gp: 3.0
    linear: 1.0
    quadratic: 1.0                 # → GP gets 3/7 ≈ 43%; others each get 1/7 ≈ 14%

# Isolate piecewise for controlled comparison:
mechanism:
  function_family_mix:
    piecewise: 1.0
    linear: 1.0                    # → 50/50 piecewise vs linear
```

When `shift.mechanism_scale > 0`, the `mechanism_logit_tilt` reweights *within*
the enabled families toward nonlinear ones (nn, tree, gp, product have higher
base logits than linear, quadratic). At `mechanism_scale = 1.0`, the tilt is
strong enough that linear mechanisms become rare even if they have significant
mix weight.

______________________________________________________________________

## Public interface rule

This workflow intentionally keeps the config surface narrow:

- No new config sections.
- No family-specific scalar knobs.
- No new CLI flags.
- The public surface remains `mechanism.function_family_mix`; the widened `gp`
  behavior is an internal variant expansion behind the existing `gp` family
  label, while `piecewise` remains an explicit mix-controlled family.
- `mechanism.function_family_mix.piecewise` must still be paired with at least
  one explicit branch family from `tree`, `discretization`, `gp`, `linear`, or
  `quadratic`.

The curated smoke presets now cover two roles:

- `piecewise` remains the shipped control path with the explicit `piecewise` +
  `linear` staged mix.
- `gp` presets isolate the widened `gp` family so diversity evidence can be
  attributed to `gp.standard`, `gp.periodic`, and `gp.multiscale`.

______________________________________________________________________

## Generate with widened `gp`

Use the curated GP smoke preset for direct generation:

```bash
dagzoo generate \
  --config configs/preset_mechanism_gp_generate_smoke.yaml \
  --num-datasets 10 \
  --device cpu \
  --hardware-policy none \
  --out data/run_gp_smoke_local
```

Inspect shard `metadata.ndjson` for:

- `mechanism_families.sampled_family_counts`
- `mechanism_families.families_present`
- `mechanism_families.sampled_variant_counts`
- `mechanism_families.variants_present`
- `mechanism_families.total_function_plans`

______________________________________________________________________

## Diversity-audit workflow

Compare the matched baseline preset against the widened `gp` preset:

```bash
dagzoo diversity-audit \
  --baseline-config configs/preset_mechanism_baseline_benchmark_smoke.yaml \
  --variant-config configs/preset_mechanism_gp_benchmark_smoke.yaml \
  --suite smoke \
  --num-datasets 10 \
  --warmup 0 \
  --device cpu \
  --out-dir benchmarks/results/diversity_audit_gp
```

Inspect `summary.json` and `summary.md` for:

- `comparisons[*].diversity_composite_shift_pct`
- `baseline.mechanism_family_summary`
- `variants[*].mechanism_family_summary`
- `variants[*].mechanism_family_summary.sampled_variant_counts`
- `variants[*].mechanism_family_summary.dataset_presence_rate_by_variant`

The audit status thresholds treat larger diversity shift as divergence, so use
the raw shift percentages together with throughput and acceptance-yield metrics
instead of treating `pass`/`warn`/`fail` as a standalone go/no-go decision.

`piecewise` remains the shipped control. Keep the matched control audit handy:

```bash
dagzoo diversity-audit \
  --baseline-config configs/preset_mechanism_baseline_benchmark_smoke.yaml \
  --variant-config configs/preset_mechanism_piecewise_benchmark_smoke.yaml \
  --suite smoke \
  --num-datasets 10 \
  --warmup 0 \
  --device cpu \
  --out-dir benchmarks/results/diversity_audit_piecewise_control
```

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
- Diagnostics: [diagnostics.md](diagnostics.md)
