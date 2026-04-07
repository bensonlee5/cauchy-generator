# Export Contract Fields

- Inventory schema: `dagzoo-export-contract-inventory-v1`
- Machine-readable source of truth: `reference/export_contract_inventory.yaml`
- Path patterns use `*` for dynamic map keys and `[]` for list item shapes.
- `audit_status` is a field-review classification only; it does not change the live export surface.

## dataset_catalog.ndjson record

One minimal per-dataset record written under each generated shard.

| Path | Type | Presence | Stability | Producer | Audit | Known Consumer / Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `dataset_id` | `string` | always | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | stable canonical dataset id |
| `dataset_index` | `int` | always | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | record index within the generated run |
| `feature_types` | `list[string]` | always | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | emitted feature typing |
| `feature_types[]` | `string` | when feature count > 0 | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | per-column emitted feature type |
| `group_ids` | `object` | when run grouping ids are available | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | stable downstream grouping keys |
| `group_ids.cohort` | `string` | when heterogeneous cohort grouping ids are available | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | shared heterogeneous raw-generation cohort key |
| `group_ids.layout_plan` | `string` | when execution-plan grouping ids are available | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | shared execution-plan grouping key |
| `group_ids.request_run` | `string` | when group_ids present | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | request-run grouping key |
| `intervention` | `object` | when hard-intervention metadata is available | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | summary-only intervention contract for downstream consumers |
| `intervention.mode` | `string` | when intervention present | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | emitted intervention regime |
| `intervention.signature` | `string` | when intervention present | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | stable intervention identity summary |
| `n_classes` | `int|null` | always | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | emitted class count for classification tasks |
| `n_features` | `int` | always | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | persisted feature count |
| `n_test` | `int` | always | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | persisted test row count |
| `n_train` | `int` | always | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | persisted train row count |
| `target_derivation` | `string` | when target derivation metadata is available | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | current target-derivation contract marker |
| `target_relevance` | `object` | when target relevance audit metadata is available | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | observed feature relevance summary for the latent target |
| `target_relevance.feature_count` | `int` | when target_relevance present | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | count of emitted features with a path into the target node |
| `target_relevance.feature_fraction` | `float` | when target_relevance present | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | feature_count normalized by emitted feature count |
| `task` | `string` | always | `stable` | `dagzoo.io.shard_contract.build_dataset_catalog_record` | `keep` | dataset task label |

## Packed parquet split row

Row schema shared by shard train.parquet and test.parquet.

| Path | Type | Presence | Stability | Producer | Audit | Known Consumer / Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `dataset_index` | `int64` | always | `stable` | `dagzoo.io.parquet_writer._build_split_table` | `keep` | row-to-dataset join key |
| `row_index` | `int64` | always | `stable` | `dagzoo.io.parquet_writer._build_split_table` | `keep` | row position within one split |
| `x` | `list[float32|float64]` | always | `stable` | `dagzoo.io.parquet_writer._build_split_table` | `keep` | packed feature vector |
| `x[]` | `float32|float64` | when feature count > 0 | `stable` | `dagzoo.io.parquet_writer._build_split_table` | `keep` | emitted feature cell value |
| `y` | `int64|float` | always | `stable` | `dagzoo.io.parquet_writer._build_split_table` | `keep` | emitted target value |

## Generate handoff manifest

Minimal downstream handoff manifest written by dagzoo generate --handoff-root.

| Path | Type | Presence | Stability | Producer | Audit | Known Consumer / Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `artifacts_relative` | `object` | always | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | manifest-relative artifact paths |
| `artifacts_relative.curated_dir` | `string` | when curated accepted-only shards exist | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | relative curated-dir path |
| `artifacts_relative.generated_dir` | `string` | always | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | relative generated-dir path |
| `identity` | `object` | always | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | stable generated corpus identity payload |
| `identity.generate_run_id` | `string` | always | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | stable generate-run id |
| `identity.generated_corpus_id` | `string` | always | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | stable generated corpus id |
| `identity.source_family` | `string` | always | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | handoff source family tag |
| `provenance` | `object` | when generated catalogs expose corpus-level provenance | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | summarized latent-target provenance for downstream corpus consumers |
| `provenance.intervention` | `object` | when hard-intervention provenance is present | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | summary-only intervention provenance for downstream corpora |
| `provenance.intervention.mode` | `string` | when provenance.intervention present | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | generated corpus intervention regime |
| `provenance.intervention.signature` | `string` | when provenance.intervention present | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | stable intervention identity summary for the generated corpus |
| `provenance.target_derivation` | `string` | when provenance present | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | current target-derivation contract marker |
| `provenance.target_relevant_feature_count_range` | `object` | when provenance present | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | min/max relevant feature counts across the generated corpus |
| `provenance.target_relevant_feature_count_range.max` | `int` | when provenance.target_relevant_feature_count_range present | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | upper bound of observed relevant feature counts |
| `provenance.target_relevant_feature_count_range.min` | `int` | when provenance.target_relevant_feature_count_range present | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | lower bound of observed relevant feature counts |
| `provenance.target_relevant_feature_fraction_range` | `object` | when provenance present | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | min/max relevant feature fractions across the generated corpus |
| `provenance.target_relevant_feature_fraction_range.max` | `float` | when provenance.target_relevant_feature_fraction_range present | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | upper bound of observed relevant feature fractions |
| `provenance.target_relevant_feature_fraction_range.min` | `float` | when provenance.target_relevant_feature_fraction_range present | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | lower bound of observed relevant feature fractions |
| `schema_name` | `string` | always | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | handoff manifest schema identifier |
| `schema_version` | `int` | always | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | handoff manifest schema version |
| `summary` | `object` | always | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | generated dataset count summary |
| `summary.generated_datasets` | `int` | always | `stable` | `dagzoo.core.generate_handoff.write_generate_handoff_manifest` | `keep` | number of generated datasets |

## Coverage summary JSON

Stable top-level diagnostics coverage summary fields.

| Path | Type | Presence | Stability | Producer | Audit | Known Consumer / Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `generated_at` | `string` | always | `stable` | `dagzoo.diagnostics.coverage.write_coverage_summary_json` | `keep` | diagnostics summary timestamp |
| `histogram_bins` | `int` | always | `stable` | `dagzoo.diagnostics.coverage.write_coverage_summary_json` | `keep` | histogram bin count used for diagnostics |
| `num_datasets` | `int` | always | `stable` | `dagzoo.diagnostics.coverage.write_coverage_summary_json` | `keep` | number of aggregated datasets |
| `quantiles` | `list[float]` | always | `stable` | `dagzoo.diagnostics.coverage.write_coverage_summary_json` | `keep` | requested quantile levels |
| `quantiles[]` | `float` | when quantiles present | `stable` | `dagzoo.diagnostics.coverage.write_coverage_summary_json` | `keep` | one requested quantile level |
| `task_counts` | `object` | always | `stable` | `dagzoo.diagnostics.coverage.write_coverage_summary_json` | `keep` | aggregated task counts |
| `task_counts.*` | `int` | when task_counts present | `stable` | `dagzoo.diagnostics.coverage.write_coverage_summary_json` | `keep` | one aggregated task count |
