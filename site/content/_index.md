---
title: "dagzoo"
linkTitle: "Home"
---

{{< blocks/cover title="dagzoo" image_anchor="top" height="med" color="dark" >}}
<p class="lead mt-3">Reproducible synthetic tabular data from latent causal structure.</p>
<a class="btn btn-lg btn-primary me-3 mb-4" href="{{< relref "/docs" >}}">
  Start Here
</a>
<a class="btn btn-lg btn-secondary me-3 mb-4" href="https://github.com/bensonlee5/dagzoo">
  GitHub <i class="fab fa-github ms-2"></i>
</a>
{{< /blocks/cover >}}

{{% blocks/section color="white" %}}

{{% blocks/feature icon="fas fa-rocket" title="Quickstart" url="docs/start/" %}}
Install dagzoo, inspect the recipe catalog, generate a first run, and use the same recipe in Python.
{{% /blocks/feature %}}

{{% blocks/feature icon="fas fa-layer-group" title="Reference Packs" url="docs/reference-packs/" %}}
Start from named `recipe:<name>` configs and published recipes you can inspect, reuse, and cite.
{{% /blocks/feature %}}

{{% blocks/feature icon="fas fa-file-export" title="Output Format" url="docs/output-format/" %}}
Stable shard layout, metadata contract, and in-memory sample shape.
{{% /blocks/feature %}}

{{% /blocks/section %}}

{{% blocks/section color="primary" %}}

{{% blocks/feature icon="fas fa-book" title="Advanced Controls" url="docs/usage-guide/" %}}
Move from named recipes to repo-local YAML authoring, optional `dagzoo filter` runs, and deeper workflow controls.
{{% /blocks/feature %}}

{{% blocks/feature icon="fas fa-project-diagram" title="How It Works" url="docs/how-it-works/" %}}
See the runtime model, terminology, and the latent-DAG-to-tabular generation flow.
{{% /blocks/feature %}}

{{% blocks/feature icon="fas fa-cogs" title="Features" url="docs/features/" %}}
Explore focused guides for diagnostics, interventions, missingness, shift, noise, and benchmark guardrails.
{{% /blocks/feature %}}

{{% /blocks/section %}}
