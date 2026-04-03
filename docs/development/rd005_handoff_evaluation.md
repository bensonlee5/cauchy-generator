# RD-005 Handoff Evaluation

`RD-005` should be promoted by structural diversity first, then throughput
stability, with the lightweight downstream probe used only as an anti-triviality
ceiling. The supported maintainer loop for that comparison is:

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

## Decision Rule

- Rank variants by higher `structural_diversity_composite_shift_pct` first, then
  higher `datasets_per_minute`, then lower downstream mean.
- Treat `diversity_status` and the full `diversity_composite_shift_pct` as
  compatibility/reporting fields only; they still mean “distance from baseline,”
  not “better” or “worse.”
- Reject variants that exceed the easy-task ceiling
  `baseline_downstream_mean + 0.10`.
- For confirmation runs, require:
  - positive structural diversity shift
  - structural diversity exceeding the current promoted slice on the same protocol
  - throughput at least 85% of baseline
  - downstream mean at or below the easy-task ceiling
- Look at the reported RD-005 priority order and the structural frontier before
  promoting any slice into a benchmark/default workflow.

## Notes

- The script is repo workflow tooling, not a packaged `dagzoo` CLI command.
- `--stress-profile` variants are synthesized by applying the profile to the
  baseline config and writing a temporary config file under the evaluation root.
- The report now includes:
  - `structural_diversity_metric_shift_pct`
  - `structural_diversity_composite_shift_pct`
  - `easy_task_ceiling_pass`
  - `priority_variant_labels`
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
