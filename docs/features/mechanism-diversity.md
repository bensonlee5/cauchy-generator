# Mechanism Diversity

Use mechanism-diversity workflows when you want to exercise the existing
family-mix surface, compare candidate mechanism behavior against the current
baseline, and verify that the generated bundles actually realize the intended
families or variants.

______________________________________________________________________

## When to use

- You want to compare the current baseline sampler against the shipped
  `piecewise` control or the widened `gp` candidate path.
- You need realized mechanism-family and mechanism-variant counts in bundle
  metadata and audit reports.
- You want diversity-audit evidence before treating a new mechanism path as
  stable. Deferred filter calibration is temporarily unsupported.

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

______________________________________________________________________

## Filter-calibration workflow

`dagzoo filter-calibration` is temporarily unsupported until a calibration
workflow exists for the small-shot ease filter. Use `dagzoo diversity-audit`
for current filter-enabled mechanism comparisons; that workflow still replays
accepted-corpus metrics without requiring deferred-filter artifacts.

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Benchmark guardrails: [benchmark-guardrails.md](benchmark-guardrails.md)
- Diagnostics: [diagnostics.md](diagnostics.md)
