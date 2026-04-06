# Interventions

`dagzoo` now supports two generation regimes on the canonical path:

- `observational`: the safe default; generation behaves exactly like the
  baseline prior and public artifacts omit intervention metadata
- `hard_interventional`: fixed interventions overwrite one or more resolved
  targets during generation and emit a stable summary identity downstream

This is intentionally narrower than a full causal-effects toolbox. The shipped
surface covers observational generation plus fixed hard interventions. It does
not imply paired factual/counterfactual export support.

______________________________________________________________________

## When to use

- You want one corpus that holds all else fixed except a targeted causal
  intervention regime.
- You need stable downstream auditability for intervention-vs-observational
  runs without inventing a separate generator subsystem.
- You want public artifacts to expose only the regime summary while keeping the
  authored selector/value payload in `effective_config.yaml`.

______________________________________________________________________

## Supported selector shapes

`hard_interventional` supports these selector kinds:

- `target`: directly overwrites emitted `y` after postprocess
- `feature_node`: resolves an emitted feature index back to its latent node,
  then clamps that node and descendants during fixed-layout execution
- `latent_node`: directly clamps a latent DAG node and descendants during
  fixed-layout execution

Multiple targets are allowed when they resolve cleanly. Colliding selectors are
rejected instead of silently picking one.

______________________________________________________________________

## Preset workflows

Use the smoke presets when you want runnable examples for each supported
selector shape:

```bash
dagzoo generate --config configs/preset_intervention_target_generate_smoke.yaml --num-datasets 25 --out data/run_intervention_target
dagzoo generate --config configs/preset_intervention_feature_node_generate_smoke.yaml --num-datasets 25 --out data/run_intervention_feature_node
dagzoo generate --config configs/preset_intervention_latent_node_generate_smoke.yaml --num-datasets 25 --out data/run_intervention_latent_node
```

______________________________________________________________________

## Custom authoring

Target intervention:

```yaml
intervention:
  mode: hard_interventional
  targets:
    - target_kind: target
      value: 2.5
```

Feature-node intervention:

```yaml
intervention:
  mode: hard_interventional
  targets:
    - target_kind: feature_node
      index: 1
      value: -1.25
```

Latent-node intervention:

```yaml
intervention:
  mode: hard_interventional
  targets:
    - target_kind: latent_node
      index: 0
      value: 1.75
```

______________________________________________________________________

## Artifact expectations

- `effective_config.yaml` keeps the canonicalized `intervention.targets` list
  plus the derived `intervention.signature`
- in-process `DatasetBundle.metadata` adds a top-level
  `metadata.intervention = {mode, signature}` summary for hard-interventional
  runs
- public `dataset_catalog.ndjson` records expose only that summary object
- internal `replay_catalog.ndjson` entries inherit the same summary via the
  full `metadata` payload
- `handoff_manifest.json` aggregates one optional
  `provenance.intervention = {mode, signature}` summary per generated corpus
- observational runs omit intervention fields from public artifacts entirely

______________________________________________________________________

## Guardrails

- Omit the `intervention` section entirely when you want default observational
  behavior.
- Public artifacts never expose authored selector/value payloads; use the
  authored config or `effective_config.yaml` when you need the full intervention
  spec.
- `target_kind: target` must not set `index`.
- `target_kind: feature_node` requires `index < dataset.n_features_min`.
- `target_kind: latent_node` requires `index < graph.n_nodes_min`.
- Classification target interventions coerce the authored value modulo the
  realized class count.
- `intervention.mode: counterfactual` is unsupported and rejected; paired
  counterfactual exports remain deferred until there is a separate contract.

______________________________________________________________________

## Related docs

- Workflow hub: [usage-guide.md](../usage-guide.md)
- Artifact contract: [output-format.md](../output-format.md)
