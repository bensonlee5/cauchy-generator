# TabICLv2 Parity Audit

This note tracks the TabICLv2 Appendix E parity surface that matters for
`RD-005`. The goal is not exact reproduction. The goal is to identify which
correlated generation behaviors are already present in `dagzoo`, which are now
partially matched, and which still need explicit work before broader prior
changes are justified.

## Status

| Surface | Status | Notes |
| --- | --- | --- |
| Cauchy DAG prior | `matched` | `dagzoo` already samples upper-triangular DAGs from shared global/source/target Cauchy latents. |
| Topological node execution | `matched` | Node plans are sampled and executed in DAG order. |
| Correlated scalar primitive | `matched` | Name-keyed correlated Beta sampling already existed and remains the base scalar mechanism. |
| Correlated categorical ratio | `matched` | The `categorical_ratio` draw was already using the correlated scalar primitive. |
| Categorical cardinalities | `partial` | Cardinalities now use correlated discrete choice, but the exact envelope still differs from TabICLv2. |
| Mechanism-family draws | `partial` | Family selection now uses correlated label weights layered on top of the configured family mix. |
| Multi-parent aggregation choice | `partial` | `concat` vs `stack` and stack aggregation kind are now correlated, but the surrounding parent-arity policy still differs. |
| Converter variants | `partial` | Joint categorical converter variants now use correlated choice, but the converter surface is broader than TabICLv2. |
| GP variants | `partial` | GP branch/variant selection now uses correlated choice; the remaining gap is broader structural policy rather than GP-local variant sampling. |
| Random-weight decay parameters (`q`, `sigma`) | `matched` | `sampling.random_weights`, `math.random_matrices`, and fixed-layout batch execution now share one canonical decay implementation. |
| Kernel-family hyperparameters | `partial` | Fixed-layout kernel plans now carry plan-time `gamma`/`signed`, and the compositional stress slice correlates them, but the broader matrix surface still differs from TabICLv2. |
| Global relationship-policy reuse across all plan families | `partial` | Matrix-family, activation-base-kind, and root-base-kind reuse now exist in the compositional slice, while parent-arity/source-shape policy remains deferred to `#293`. |

## RD-005 Read

- The main gap was narrower correlation reuse, not absence of graph correlation.
- `dagzoo` is now materially closer on the requested relationship surfaces:
  cardinalities, mechanism-family draws, aggregation choice, converter/GP
  variants, and random-weight decay parameters.
- The next remaining high-value gap is graph/source-shape policy (`#293`), not a
  second random-weight or kernel-local implementation split.
- Remaining `partial` items should only expand further if the diversity and
  downstream handoff evaluations show that the current correlated reuse is still
  insufficient.
- `RD-002` stays deferred. Hard interventions can be evaluated later without
  changing the current `RD-005` comparison loop.
