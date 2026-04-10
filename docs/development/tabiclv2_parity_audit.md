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
| Multi-parent aggregation choice | `partial` | `concat` vs `stack` and stack aggregation kind are correlated globally, and the graph-breadth slice now adds parent-arity-aware source-shape reuse; the remaining gap is that baseline and the compositional slice intentionally do not use that broader policy. |
| Converter variants | `partial` | Joint categorical converter variants now use correlated choice, but the converter surface is broader than TabICLv2. |
| GP variants | `partial` | GP branch/variant selection now uses correlated choice; the remaining gap is broader structural policy rather than GP-local variant sampling. |
| Random-weight decay parameters (`q`, `sigma`) | `matched` | `sampling.random_weights`, `math.random_matrices`, and fixed-layout batch execution now share one canonical decay implementation. |
| Kernel-family hyperparameters | `partial` | Fixed-layout kernel plans now carry plan-time `gamma`/`signed`, and the compositional stress slice correlates them, but the broader matrix surface still differs from TabICLv2. |
| Global relationship-policy reuse across all plan families | `partial` | Matrix-family, activation-base-kind, and root-base-kind reuse exist in the compositional slice, and parent-arity/source-shape policy now exists in the graph-breadth slice; the remaining gap is that the full reuse surface is intentionally split across opt-in RD-005 lanes rather than applied globally. |

## Evidence Snapshot

- Treat realized artifacts, not this static table, as the current source of
  truth. Start with `dagzoo diversity-audit` `summary.json` / `summary.md`.
- Read `parity_surface_summary` first when you want the remaining parity gaps
  directly: converter methods/variants, GP variants, kernel `gamma` / `signed`,
  matrix kinds, root base kinds, parent arity, source-shape policy, and
  categorical cardinality now surface as first-class summary fields.
- Use [rd005_handoff_evaluation.md](rd005_handoff_evaluation.md) plus
  [`scripts/evaluate_rd005_follow_on_suite.py`](../../scripts/evaluate_rd005_follow_on_suite.py)
  when you want the canonical promotion decision across all current internal
  RD-005 lanes.
- Use [`scripts/evaluate_handoff_pareto.py`](../../scripts/evaluate_handoff_pareto.py)
  when you want only the lower-level structural-diversity/throughput/downstream
  ranking loop without the full suite orchestration.
- Use [`scripts/render_tabiclv2_parity_report.py`](../../scripts/render_tabiclv2_parity_report.py)
  when you want one maintainer-facing markdown/json snapshot from a single
  diversity-audit run.
- Current maintained read: the compositional slice is the promotion candidate,
  graph breadth is the structural extreme, and parity means same-or-better
  realized diversity under the shipped contracts rather than exact Appendix E
  reproduction.

## RD-005 Read

- The main gap was narrower correlation reuse, not absence of graph correlation.
- `dagzoo` is now materially closer on the requested relationship surfaces:
  cardinalities, mechanism-family draws, aggregation choice, converter/GP
  variants, and random-weight decay parameters.
- The graph/source-shape policy lane (`#293`) is now implemented in the
  graph-breadth slice, so the remaining work is no longer “add parent-arity
  policy at all,” but to decide whether any of that opt-in structure should be
  broadened beyond the graph-structure lane.
- Remaining `partial` items should only expand further if the diversity and
  downstream handoff evaluations show that the current correlated reuse is still
  insufficient.
- `RD-002` hard-intervention semantics and artifact contracts are now
  implemented. The remaining follow-on question is whether any future
  counterfactual paired-output surface belongs in the same parity frame rather
  than in the shipped hard-intervention lane.
