# RD-005 Handoff Evaluation

`RD-005` should be promoted by structural diversity first, then throughput
stability, with the lightweight downstream probe used only as an anti-triviality
ceiling. The supported maintainer loop for the full follow-on suite is:

```bash
./.venv/bin/python scripts/evaluate_rd005_follow_on_suite.py \
  --baseline-config configs/default.yaml \
  --out-root benchmarks/results/rd005_follow_on \
  --suite smoke \
  --num-datasets 8 \
  --seed 123 \
  --device cpu
```

This writes one promotion decision bundle at the suite root:

- `follow_on_promotion_summary.json`
- `follow_on_promotion_summary.md`

alongside the supporting artifacts under:

- `diversity_audit/summary.json`
- `diversity_audit/summary.md`
- `diversity_audit/parity_report/parity_report.json`
- `diversity_audit/parity_report/parity_report.md`
- `handoff_pareto/pareto_summary.json`
- `handoff_pareto/pareto_summary.md`

When you want to run just the handoff comparison loop without the suite
orchestration, the lower-level Pareto helper remains:

```bash
./.venv/bin/python scripts/evaluate_handoff_pareto.py \
  --baseline-config configs/default.yaml \
  --stress-profile anti_memorization_piecewise_classification_compositional_slice_v1 \
  --out-root benchmarks/results/rd005_pareto \
  --num-datasets 8 \
  --seed 123 \
  --device cpu
```

## What It Does

- Materializes the carried internal RD-005 candidate set:
  - `compositional` as the incumbent
  - `graph-breadth` as the structural control
  - `categorical-cardinality`, `hybrid`, and `robustness-composition` as
    challengers
- Runs matched-budget `dagzoo diversity-audit` across that full lane set.
- Renders one parity report from the diversity-audit summary.
- Runs matched-budget handoff/Pareto evaluation against the same candidate set.
- Joins structural diversity, throughput, downstream ceiling, and parity-surface
  snapshots into one machine-readable promotion decision summary.
- Runs `dagzoo generate --handoff-root` for one baseline plus one variant at a
  time under the same seed, dataset budget, device, and hardware policy.
- Scores each generated corpus with a lightweight downstream linear probe:
  ridge regression for regression tasks and one-vs-rest ridge classification
  for classification tasks.
- Reads `internal/diagnostics_artifacts/coverage_summary.json` when available
  and computes both the full diversity-shift summary and a structural-only
  diversity summary against the baseline.
- Writes:
  - `pareto_summary.json`
  - `pareto_summary.md`

When you already have one matched-budget `dagzoo diversity-audit` run and want a
maintainer parity snapshot rather than a new handoff evaluation, render:

```bash
./.venv/bin/python scripts/render_tabiclv2_parity_report.py \
  --summary-json benchmarks/results/diversity_audit_stress_graph_breadth/summary.json \
  --out-dir benchmarks/results/diversity_audit_stress_graph_breadth/parity_report
```

## Decision Rule

- Rank lanes by higher `structural_diversity_composite_shift_pct` first, then
  higher `datasets_per_minute`, then lower downstream mean.
- Treat `diversity_status` and the full `diversity_composite_shift_pct` as
  compatibility/reporting fields only; they still mean “distance from baseline,”
  not “better” or “worse.”
- Reject variants that exceed the easy-task ceiling
  `baseline_downstream_mean + 0.10`.
- For promotion runs, require:
  - positive structural diversity shift
  - challengers to beat the compositional incumbent on the same protocol
  - throughput at least 85% of baseline
  - downstream mean at or below the easy-task ceiling
- Persist one machine-readable status per lane:
  - `promote`
  - `hold_internal`
  - `structural_control_only`
- If no lane clears the gate, the correct output is `no_promotion`.

## Notes

- `scripts/evaluate_rd005_follow_on_suite.py` is the canonical maintainer entry
  point for this decision. It should be the artifact root linked from tracker
  updates when someone argues for public promotion.
- The script is repo workflow tooling, not a packaged `dagzoo` CLI command.
- `--stress-profile` variants are synthesized by applying the profile to the
  baseline config and writing a temporary config file under the evaluation root.
- The report now includes:
  - `structural_diversity_metric_shift_pct`
  - `structural_diversity_composite_shift_pct`
  - `easy_task_ceiling_pass`
  - `priority_variant_labels`
  - `promotion_status`
  - `promotion_failure_reasons`
- For the graph/source-shape lane (`#293`), compare baseline against
  `anti_memorization_piecewise_classification_graph_breadth_slice_v1` and read
  `graph_target_ancestor_fraction`, `graph_ancestor_overlap_mean`,
  `graph_reachability_ratio`, and `graph_depth_ratio` before looking at
  throughput.
- `--reuse-existing` can be used to avoid regenerating already-materialized
  handoff roots.
- For the matrix/kernel correlation lane (`#294`), start with the compositional
  slice only so the comparison isolates plan-family reuse instead of mixing in
  the broader graph-structure changes that belong to `#293`.
