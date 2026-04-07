# Dagzoo Transform Reference

<p>This document is the formal reference for the generation transforms used by
<code>dagzoo</code>. The equations are written to match the current runtime in
<code>src/dagzoo</code>.</p>

<h2 id="purpose-and-research-context">Using This Reference</h2>

<p>Use this page when you want the mathematical definitions behind the
generator. Each section covers one part of the runtime: DAG sampling, shift,
mechanism families, node execution, converters, and noise. Together they
describe how <code>dagzoo</code> maps a latent DAG into emitted tabular
features, targets, and metadata.</p>

<ul>
<li><strong>Section 1 (DAG structure)</strong> — topology of causal dependencies</li>
<li><strong>Section 2 (Shift parameters)</strong> — train/test distribution drift</li>
<li><strong>Section 3 (Mechanism families)</strong> — functional complexity of variable relationships</li>
<li><strong>Section 4 (Node pipeline)</strong> — execution mechanics from latent to converter-ready state</li>
<li><strong>Section 5 (Mechanism definitions)</strong> — catalog of functional transform families</li>
<li><strong>Section 6 (Converters)</strong> — mapping from latent state to observed tabular features</li>
<li><strong>Section 7 (Noise)</strong> — stochastic variation character</li>
</ul>

<p>The sections are organized so you can either read end to end or jump
directly to the transform family you need. When you are checking a config or a
runtime behavior, the practical question is: <em>which transform changes, and
how does that change show up in the emitted dataset?</em></p>
<h2 id="notation-and-symbols">Notation and Symbols</h2>
<p><strong>Primary Variable:</strong> Symbol map (this section's
notation table).<br> <strong>Dependency Map:</strong> all symbols used
in later equations -&gt; definitions in this table.<br> <strong>Path to
Final Output:</strong> unambiguous symbols -&gt; unambiguous equations
-&gt; correct interpretation of how transforms produce <code>X</code>,
<code>y</code>, and metadata.<br></p>
<p><strong>Rationale.</strong> This table fixes symbol meanings up front
so equations stay unambiguous across sections. Symbol names mirror
runtime naming in core modules, making the spec easier to trace back to
code.</p>
<table>
<thead>
<tr>
<th>Symbol</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><span class="math inline">\(N\)</span></td>
<td>Number of latent DAG nodes.</td>
</tr>
<tr>
<td><span class="math inline">\(n\)</span></td>
<td>Number of rows sampled for one dataset attempt (<span
class="math inline">\(n = n_{\text{train}} +
n_{\text{test}}\)</span>).</td>
</tr>
<tr>
<td><span class="math inline">\(G \in {0,1}^{N \times N}\)</span></td>
<td>DAG adjacency matrix with convention <code>G[src, dst]</code>.</td>
</tr>
<tr>
<td><span class="math inline">\(i, j\)</span></td>
<td>Node indices in <span
class="math inline">\({0,\dots,N-1}\)</span>.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{pa}(j)\)</span></td>
<td>Parent index set for node <span class="math inline">\(j\)</span>:
<span class="math inline">\({i : G_{ij}=1}\)</span>.</td>
</tr>
<tr>
<td><span class="math inline">\(\sigma(x)\)</span></td>
<td>Logistic sigmoid <span class="math inline">\((1 +
e^{-x})^{-1}\)</span>.</td>
</tr>
<tr>
<td><span class="math inline">\(A, B_i, C_j\)</span></td>
<td>DAG latent logits: <span class="math inline">\(A\)</span> is global
edge propensity, <span class="math inline">\(B_i\)</span> is source-node
effect, <span class="math inline">\(C_j\)</span> is destination-node
effect.</td>
</tr>
<tr>
<td><span class="math inline">\(\beta_{\text{edge}}\)</span></td>
<td>Edge logit bias applied in DAG sampling.</td>
</tr>
<tr>
<td><span class="math inline">\(\tau\)</span></td>
<td>Mechanism logit tilt (<code>mechanism_logit_tilt</code>).</td>
</tr>
<tr>
<td><span class="math inline">\(\lambda_f\)</span></td>
<td>Raw mechanism family weight for family <span
class="math inline">\(f\)</span> from
<code>mechanism.function_family_mix</code>.</td>
</tr>
<tr>
<td><span class="math inline">\(\pi_f^{(0)}\)</span></td>
<td>Baseline mechanism-family probability before logit tilt.</td>
</tr>
<tr>
<td><span class="math inline">\(\pi_f\)</span></td>
<td>Final sampling probability for mechanism family <span
class="math inline">\(f\)</span>.</td>
</tr>
<tr>
<td><span class="math inline">\(\ell_f\)</span></td>
<td>Family base logit for family <span class="math inline">\(f\)</span>
(from <code>MECHANISM_FAMILY_BASE_LOGITS</code>).</td>
</tr>
<tr>
<td><span class="math inline">\(\tilde{\ell}_f\)</span></td>
<td>Centered family base logit: <span class="math inline">\(\ell_f -
\frac{1}{\lvert\mathcal{F}\rvert}\sum_{g\in\mathcal{F}}\ell_g\)</span>.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathcal{F}\)</span></td>
<td>Ordered mechanism family set:
<code>(nn, tree, discretization, gp, linear, quadratic, em, product, piecewise)</code>.</td>
</tr>
<tr>
<td><span class="math inline">\(Z_j \in \mathbb{R}^{n \times
d}\)</span></td>
<td>Node-<span class="math inline">\(j\)</span> latent tensor after node
pipeline transforms.</td>
</tr>
<tr>
<td><span class="math inline">\(d_{\text{req}}\)</span></td>
<td>Required latent width from converter specs: <span
class="math inline">\(\sum_s \max(1, d_s)\)</span>.</td>
</tr>
<tr>
<td><span class="math inline">\(d_{\text{extra}}\)</span></td>
<td>Additional sampled latent width: <span class="math inline">\(\max(1,
\lfloor \mathsf{LogUniform}(1,32) \rfloor)\)</span>.</td>
</tr>
<tr>
<td><span class="math inline">\(d\)</span></td>
<td>Node latent width used for mechanism output: <span
class="math inline">\(d = d_{\text{req}} +
d_{\text{extra}}\)</span>.</td>
</tr>
<tr>
<td><span class="math inline">\(d_s\)</span></td>
<td>Width requested by converter spec <span
class="math inline">\(s\)</span> before max-with-1 clamp.</td>
</tr>
<tr>
<td><span class="math inline">\(w \in \mathbb{R}^d\)</span></td>
<td>Positive random weight vector (normalized to sum 1, then
permuted).</td>
</tr>
<tr>
<td><span class="math inline">\(X\)</span></td>
<td>Generic mechanism input matrix.</td>
</tr>
<tr>
<td><span class="math inline">\(Y\)</span></td>
<td>Generic mechanism output matrix.</td>
</tr>
<tr>
<td><span class="math inline">\(p_{ij}\)</span></td>
<td>DAG edge probability for ordered pair <span
class="math inline">\((i,j)\)</span> with <span class="math inline">\(i
&lt; j\)</span>.</td>
</tr>
<tr>
<td><span class="math inline">\(\gamma_\sigma\)</span></td>
<td>Noise standard-deviation multiplier from shift runtime
(<code>variance_sigma_multiplier</code>).</td>
</tr>
<tr>
<td><span class="math inline">\(\gamma_{\text{var}}\)</span></td>
<td>Noise variance multiplier: <span
class="math inline">\(\gamma_{\text{var}} =
\gamma_\sigma^2\)</span>.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathcal{D}_{\text{noise}}\)</span></td>
<td>Active runtime noise family (<code>gaussian</code>,
<code>laplace</code>, or <code>student_t</code> after runtime
selection).</td>
</tr>
<tr>
<td><span class="math inline">\(C\)</span></td>
<td>Categorical cardinality for a converter target.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{LSE}\)</span></td>
<td>LogSumExp along parent axis.</td>
</tr>
<tr>
<td><span class="math inline">\(T\)</span></td>
<td>Number of trees in a sampled tree ensemble.</td>
</tr>
<tr>
<td><span class="math inline">\(P\)</span></td>
<td>Number of random features in GP approximation (fixed at 256).</td>
</tr>
<tr>
<td><span class="math inline">\(s_j\)</span></td>
<td>Final node-level multiplicative latent scale.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathcal{O}\)</span></td>
<td>Final attempt output tuple: <span class="math inline">\((X, y,
\text{metadata})\)</span>.</td>
</tr>
</tbody>
</table>
<h3 id="operator-reference">Operator Reference</h3>
<table>
<thead>
<tr>
<th>Operator</th>
<th>Definition</th>
<th>Input/Output Shape Semantics</th>
<th>Where Used</th>
<th>Output Impact</th>
</tr>
</thead>
<tbody>
<tr>
<td><span class="math inline">\(\mathsf{pa}(j)\)</span></td>
<td>Parent index set of node <span class="math inline">\(j\)</span>:
<span class="math inline">\({i: G_{ij}=1}\)</span>.</td>
<td>Input: scalar node index <span class="math inline">\(j\)</span> and
adjacency <span class="math inline">\(G\)</span>; Output: set of parent
indices.</td>
<td>Section 1 and Section 4.</td>
<td>Determines which upstream node tensors are composed for node <span
class="math inline">\(j\)</span>.</td>
</tr>
<tr>
<td><span
class="math inline">\(\mathsf{Agg}(Y_1,\dots,Y_k)\)</span></td>
<td>Multi-parent aggregation operator sampled from <span
class="math inline">\({\sum,\prod,\max,\mathsf{LSE}}\)</span>.</td>
<td>Input: parent-aligned tensors <span class="math inline">\((n \times
d)\)</span>; Output: one <span class="math inline">\((n \times
d)\)</span> tensor.</td>
<td>Section 4.2.</td>
<td>Changes how parent effects combine before downstream
mechanism/converter steps.</td>
</tr>
<tr>
<td><span
class="math inline">\(\mathsf{LSE}(Y_1,\dots,Y_k)\)</span></td>
<td>LogSumExp over parent axis: <span class="math inline">\(\log\sum_r
e^{Y_r}\)</span>.</td>
<td>Input: stacked parent outputs; Output: aggregated tensor <span
class="math inline">\((n \times d)\)</span>.</td>
<td>Section 4.2.</td>
<td>Smoothly emphasizes larger parent responses while retaining
contributions from others.</td>
</tr>
<tr>
<td><span class="math inline">\(\arg\min_k\)</span></td>
<td>Index of minimum value over candidate index <span
class="math inline">\(k\)</span>.</td>
<td>Input: candidate scores/distances; Output: integer index per
row.</td>
<td>Sections 5.5 and 6.2.</td>
<td>Produces center assignments that drive discrete or categorical
structure.</td>
</tr>
<tr>
<td><span class="math inline">\(\sum,\prod,\max\)</span></td>
<td>Sum/product/max reductions over specified axis.</td>
<td>Input: tensors with reduction axis; Output: reduced tensor.</td>
<td>Sections 4.2, 4.3, 5.2, 5.4, 5.7.</td>
<td>Controls interaction form (additive, multiplicative, extremal) in
latent construction.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{Converter}_s(V_s)\)</span></td>
<td>Converter function for spec <span class="math inline">\(s\)</span>,
returning transformed slice and extracted value.</td>
<td>Input: latent view <span class="math inline">\(V_s\)</span>; Output:
<span class="math inline">\((X&#39;_s, v_s)\)</span>.</td>
<td>Section 4.3 and Section 6.</td>
<td>Directly emits observed feature values that become final
complete-data <code>X</code>; the target is then generated by the
conditional complete-data head, and optional missingness may later mask
the emitted features.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{Linear}(\cdot)\)</span></td>
<td>Random linear map to the configured output width.</td>
<td>Input: matrix <span class="math inline">\((n \times
d_{in})\)</span>; Output: matrix <span class="math inline">\((n \times
d_{out})\)</span>.</td>
<td>Sections 5.5 and 5.7.</td>
<td>Projects intermediate representations into node latent channels used
by converters.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{leaf}(X)\)</span></td>
<td>Oblivious-tree leaf index determined by sampled
splits/thresholds.</td>
<td>Input: row vector(s) and tree splits; Output: leaf id(s).</td>
<td>Section 5.4.</td>
<td>Selects leaf values that define piecewise latent behavior.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{logit}_{ik}\)</span></td>
<td>Pre-softmax score for row <span class="math inline">\(i\)</span>,
component/class <span class="math inline">\(k\)</span>.</td>
<td>Input: row-to-center distances and scale terms; Output: score
matrix.</td>
<td>Section 5.7.</td>
<td>Determines assignment probabilities that shape EM-style latent
outputs.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{softmax}_k(\cdot)\)</span></td>
<td>Normalize exponentiated logits over index <span
class="math inline">\(k\)</span> to probabilities.</td>
<td>Input: logits per row; Output: probabilities summing to 1 across
<span class="math inline">\(k\)</span>.</td>
<td>Sections 5.7 and 6.2.</td>
<td>Converts scores into stochastic assignments that drive
categorical/mixture behavior.</td>
</tr>
<tr>
<td><span
class="math inline">\(\mathsf{sample\_noise}(\cdot)\)</span></td>
<td>Noise sampler with configurable family and scale.</td>
<td>Input: shape + family params; Output: random tensor with requested
shape.</td>
<td>Section 7.1.</td>
<td>Injects stochastic variation into points, matrices, weights, and
node outputs.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{clip}(x,a,b)\)</span></td>
<td>Clamp each entry of <span class="math inline">\(x\)</span> into
<span class="math inline">\([a,b]\)</span>.</td>
<td>Input: tensor; Output: tensor with same shape.</td>
<td>Section 4.3.</td>
<td>Prevents extreme values from destabilizing downstream
standardization/conversion.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{nan\_to\_num}(x)\)</span></td>
<td>Replace NaN/Inf entries with finite values.</td>
<td>Input: tensor; Output: finite tensor with same shape.</td>
<td>Section 4.3.</td>
<td>Stabilizes latent values before weighting and conversion.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{standardize}(x)\)</span></td>
<td>Column-wise centering and scaling to unit variance (with epsilon
guard).</td>
<td>Input: matrix <span class="math inline">\((n \times d)\)</span>;
Output: matrix <span class="math inline">\((n \times d)\)</span>.</td>
<td>Sections 4.3 and 6.2.</td>
<td>Controls scale so mechanism outputs/converters are numerically
comparable.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{Cauchy}(0,1)\)</span></td>
<td>Standard Cauchy distribution.</td>
<td>Input: location/scale; Output: scalar or tensor draws.</td>
<td>Section 1.</td>
<td>Provides heavy-tailed latent logits for heterogeneous graph
connectivity.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{Bernoulli}(p)\)</span></td>
<td>Bernoulli draw with success probability <span
class="math inline">\(p\)</span>.</td>
<td>Input: probability <span class="math inline">\(p \in [0,1]\)</span>;
Output: binary draw.</td>
<td>Section 1.</td>
<td>Converts edge probabilities into realized adjacency <span
class="math inline">\(G\)</span>.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{Uniform}(a,b)\)</span></td>
<td>Continuous uniform distribution on <span
class="math inline">\([a,b]\)</span>.</td>
<td>Input: bounds <span class="math inline">\((a,b)\)</span>; Output:
scalar/tensor draws.</td>
<td>Sections 4.1 and 7.1.</td>
<td>Drives root sampling and inverse-CDF noise construction.</td>
</tr>
<tr>
<td><span
class="math inline">\(\mathsf{UniformInt}\{m,\dots,n\}\)</span></td>
<td>Discrete uniform integer sampler on inclusive range.</td>
<td>Input: integer bounds; Output: integer draws.</td>
<td>Section 5.4.</td>
<td>Samples tree depths and related discrete structure choices.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{LogUniform}(a,b)\)</span></td>
<td>Distribution with <span class="math inline">\(\log x\)</span>
uniform on <span class="math inline">\([\log a,\log b]\)</span>.</td>
<td>Input: positive bounds <span class="math inline">\((a,b)\)</span>;
Output: positive scalar draws.</td>
<td>Sections 4.3, 5.5, 6.1, 6.2.</td>
<td>Controls scales/exponents over multiple orders of magnitude.</td>
</tr>
<tr>
<td><span class="math inline">\(\mathsf{Categorical}(p)\)</span></td>
<td>Discrete draw from class probabilities <span
class="math inline">\(p\)</span>.</td>
<td>Input: probability vector per row; Output: class/label index.</td>
<td>Section 6.2.</td>
<td>Emits categorical feature labels that populate final observed
features.</td>
</tr>
</tbody>
</table>
<p>Conventions:</p>
<ul>
<li>Unless noted otherwise, operations are row-wise over batch dimension
<span class="math inline">\(n\)</span> and preserve row count.</li>
<li>In <span class="math inline">\(\mathsf{Agg}\)</span>, the
aggregation kind is sampled once per multi-parent composition call and
then applied elementwise across parents.</li>
<li>In noise sampling, <code>mixture</code> in Section 7.1 samples
component assignment per element, while Section 7.2 resolves one family
per dataset attempt before downstream draws.</li>
</ul>
<h3 id="end-to-end-primary-variable-map">End-to-End Primary Variable
Map</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(\mathcal{O} = (X, y,
\text{metadata})\)</span>.<br> <strong>Dependency Map:</strong> <span
class="math inline">\(s_{\text{shift}}, A, B_i, C_j, \lambda_f,
\tilde{\ell}_f, \text{noise\_spec}, \epsilon, G, \pi_f, Z_{\text{base}},
Z_{\text{comp}}, Y, Z_{\text{post}}, (X&#39;_s, v_s) \rightarrow
\mathcal{O}\)</span>.<br> <strong>Path to Final Output:</strong> shift
runtime values define graph, mechanism, and noise controls; these drive
node latents and converter emissions that become <span
class="math inline">\((X,y)\)</span>, while runtime selections and
graph/mechanism/noise summaries become metadata in <span
class="math inline">\(\mathcal{O}\)</span>.<br></p>
<p><strong>Rationale.</strong> This section is the single whole-picture
view that ties section-level primary variables into one output contract.
It complements the per-section formal details by making global dataflow
explicit.</p>
<p>Pipeline skeleton:</p>
<p><span class="math display">\[s_{\text{shift}} \rightarrow
(\beta_{\text{edge}}, \tau, \gamma_{\sigma},
\gamma_{\text{var}})\]</span> <span
class="math display">\[\left(\beta_{\text{edge}}, A, B_i, C_j\right)
\rightarrow G\]</span> <span class="math display">\[\left(\lambda_f,
\tau, \tilde{\ell}_f\right) \rightarrow \pi_f\]</span> <span
class="math display">\[\text{noise\_spec} \rightarrow
\epsilon_{\text{family}} \rightarrow \epsilon\]</span> <span
class="math display">\[\left(G, \pi_f, \epsilon, Z_{\text{base}},
Z_{\text{comp}}, Y, Z_{\text{post}}\right) \rightarrow
{Z_j}_{j=0}^{N-1}\]</span> <span class="math display">\[\left({Z_j},
\mathsf{Converter}_s\right) \rightarrow {(X&#39;_s, v_s)}_s \rightarrow
(X, y)\]</span> <span class="math display">\[\left(G, \pi_f,
\text{noise\_spec}, s_{\text{shift}}\right) \rightarrow
\text{metadata}\]</span> <span class="math display">\[\mathcal{O} = (X,
y, \text{metadata})\]</span></p>
<p>Operator role notes:</p>
<ul>
<li><span class="math inline">\(\mathsf{Agg}\)</span> is the bridge from
multiple parent tensors to <span
class="math inline">\(Z_{\text{comp}}\)</span> (Section 4.2).</li>
<li><span class="math inline">\(\mathsf{Converter}_s\)</span> is the
bridge from latent state to emitted observable values (Section 4.3 and
Section 6).</li>
</ul>
<p>Primary variable crosswalk:</p>
<table>
<thead>
<tr>
<th>Section</th>
<th>Primary Variable</th>
<th>Immediate Inputs</th>
<th>Immediate Output</th>
<th>Contribution to <span
class="math inline">\(\mathcal{O}\)</span></th>
</tr>
</thead>
<tbody>
<tr>
<td>1. DAG Structure Sampling</td>
<td><span class="math inline">\(G\)</span></td>
<td><span class="math inline">\(A, B_i, C_j,
\beta_{\text{edge}}\)</span></td>
<td>Parent sets <span class="math inline">\(\mathsf{pa}(j)\)</span></td>
<td>Controls execution order and parent flow that shapes <span
class="math inline">\((X,y)\)</span> and graph metadata.</td>
</tr>
<tr>
<td>2. Shift Runtime Parameters</td>
<td><span class="math inline">\(s_{\text{shift}}\)</span></td>
<td>Resolved shift config</td>
<td><span class="math inline">\((\beta_{\text{edge}}, \tau,
\gamma_{\sigma}, \gamma_{\text{var}})\)</span></td>
<td>Couples drift controls into graph density, mechanism usage, and
noise magnitude in outputs/metadata.</td>
</tr>
<tr>
<td>3. Mechanism Family Sampling</td>
<td><span class="math inline">\(\pi_f\)</span></td>
<td><span class="math inline">\(\lambda_f, \tau, \tilde{\ell}_f,
\mathcal{F}\)</span></td>
<td>Family draw distribution per mechanism call</td>
<td>Determines which transform family generates latent structure before
conversion to <span class="math inline">\((X,y)\)</span>.</td>
</tr>
<tr>
<td>4. Node Generation Pipeline</td>
<td><span class="math inline">\(Z_j\)</span></td>
<td>Parent/root inputs, width terms, sampled mechanisms, weights,
scale</td>
<td>Node-level latent tensor</td>
<td>Shared latent state from which converter slices emit observed
features/targets.</td>
</tr>
<tr>
<td>4.1 Root-node base sampling</td>
<td><span class="math inline">\(Z_{\text{base}}\)</span></td>
<td>Base-kind selection, distribution params, <span
class="math inline">\(\mathcal{D}_{\text{noise}}\)</span></td>
<td>Initial latent input for root mechanisms</td>
<td>Seeds root latent variability that propagates to downstream nodes
and final observables.</td>
</tr>
<tr>
<td>4.2 Parent composition</td>
<td><span class="math inline">\(Z_{\text{comp}}\)</span></td>
<td>Parent tensors, concat/aggregation path, <span
class="math inline">\(\mathsf{Agg}\)</span></td>
<td>Composed latent input for node transform</td>
<td>Defines multi-parent interaction pattern that feeds mechanism
outputs used in <span class="math inline">\((X,y)\)</span>.</td>
</tr>
<tr>
<td>4.3 Post-processing and slicing</td>
<td><span class="math inline">\(Z_{\text{post}}\)</span></td>
<td>Raw latent, sanitize/standardize, weighting, padding, slice
writeback</td>
<td>Converter-ready latent state</td>
<td>Directly feeds converter extraction and latent feedback that
determines emitted values.</td>
</tr>
<tr>
<td>5. Mechanism Definitions</td>
<td><span class="math inline">\(Y = f(X)\)</span></td>
<td>Mechanism parameters and input <span
class="math inline">\(X\)</span></td>
<td>Mechanism output tensor</td>
<td>Family-level nonlinear map that populates latent channels later
converted to outputs.</td>
</tr>
<tr>
<td>5.1 Linear</td>
<td><span class="math inline">\(Y_{\text{linear}}\)</span></td>
<td><span class="math inline">\(X, M\)</span></td>
<td>Linear latent channels</td>
<td>Baseline low-complexity latent effects converted into
numeric/categorical outputs.</td>
</tr>
<tr>
<td>5.2 Quadratic</td>
<td><span class="math inline">\(Y_{\text{quad}}\)</span></td>
<td><span class="math inline">\(X_{\text{sub}}, X_{\text{aug}},
M_t\)</span></td>
<td>Interaction-heavy latent channels</td>
<td>Adds pairwise interaction structure reflected in emitted
features/targets.</td>
</tr>
<tr>
<td>5.3 Neural network</td>
<td><span class="math inline">\(Y_{\text{nn}}\)</span></td>
<td><span class="math inline">\(X, {M_{\ell}},
{\phi_{\ell}}\)</span></td>
<td>Deep nonlinear latent channels</td>
<td>Produces high-capacity nonlinear signals passed through converters
into <span class="math inline">\((X,y)\)</span>.</td>
</tr>
<tr>
<td>5.4 Tree ensemble</td>
<td><span class="math inline">\(Y_{\text{tree}}\)</span></td>
<td>Splits, thresholds, leaf values</td>
<td>Piecewise latent channels</td>
<td>Introduces discontinuous region-based structure visible in output
distributions.</td>
</tr>
<tr>
<td>5.5 Discretization</td>
<td><span class="math inline">\(Y_{\text{disc}}\)</span></td>
<td>Centers, distance exponent, nearest-center assignment</td>
<td>Clustered latent channels</td>
<td>Creates quantized/cluster-like effects in converted
observables.</td>
</tr>
<tr>
<td>5.6 GP/RFF</td>
<td><span class="math inline">\(Y_{\text{gp}}\)</span></td>
<td><span class="math inline">\(X_{\text{proj}}, \Omega, b,
V\)</span></td>
<td>Smooth kernel-like latent channels</td>
<td>Adds smooth nonlinear components that converters emit as continuous
structure.</td>
</tr>
<tr>
<td>5.7 EM assignment</td>
<td><span class="math inline">\(Y_{\text{em}}\)</span></td>
<td>Centers, scales, exponents, assignment probabilities</td>
<td>Mixture-like latent channels</td>
<td>Encodes local mixture behavior that propagates to final emitted
values.</td>
</tr>
<tr>
<td>5.8 Product family</td>
<td><span class="math inline">\(Y_{\text{prod}}\)</span></td>
<td>Two component family outputs <span class="math inline">\(f(X),
g(X)\)</span></td>
<td>Multiplicative latent channels</td>
<td>Adds higher-order multiplicative effects before converter
extraction.</td>
</tr>
<tr>
<td>6. Converter Definitions</td>
<td><span class="math inline">\((X&#39;_s, v_s)\)</span></td>
<td>Latent slice, converter kind/params</td>
<td>Updated latent slice plus extracted value</td>
<td>Defines the emission boundary from latent state to observed feature
values.</td>
</tr>
<tr>
<td>6.1 Numeric converter</td>
<td><span class="math inline">\((X&#39;_{\text{num}},
v_{\text{num}})\)</span></td>
<td>Numeric slice, warp params, branch choice</td>
<td>Numeric extracted value and feedback slice</td>
<td>Produces numeric entries in <span
class="math inline">\((X,y)\)</span> and latent feedback for downstream
coupling.</td>
</tr>
<tr>
<td>6.2 Categorical converter</td>
<td><span class="math inline">\((X&#39;_{\text{cat}},
\ell)\)</span></td>
<td>Method/variant, cardinality, logits/centers</td>
<td>Categorical label and feedback slice</td>
<td>Produces categorical feature/class outputs and associated latent
feedback.</td>
</tr>
<tr>
<td>7. Noise Sampling and Selection</td>
<td><span class="math inline">\(\epsilon\)</span></td>
<td>Family, scale, family-specific params</td>
<td>Runtime noise draws</td>
<td>Injects stochastic variation throughout latent generation and
converter inputs.</td>
</tr>
<tr>
<td>7.1 Primitive family samplers</td>
<td><span class="math inline">\(\epsilon_{\text{family}}\)</span></td>
<td>Primitive family choice and parameters</td>
<td>Family-specific noise samples</td>
<td>Supplies concrete perturbation draws used by
samplers/mechanisms.</td>
</tr>
<tr>
<td>7.2 Dataset-level family resolution</td>
<td><code>noise_spec</code></td>
<td>Requested family, mixture weights, seeded RNG</td>
<td>One attempt-level family spec</td>
<td>Determines dataset-level noise identity reflected in outputs and
metadata.</td>
</tr>
</tbody>
</table>
<h2 id="1-dag-structure-sampling">1. DAG Structure Sampling</h2>
<p><strong>Motivation</strong>  The graph structure is the causal skeleton that
determines <em>which variables influence which</em>.  Consider three corpora:</p>
<pre><code>Corpus A (Erdős–Rényi, p=0.3):   every DAG looks like a medium-density mesh
                                   → all datasets have ~similar dependency depth

Corpus B (Erdős–Rényi, varied p): sparser and denser meshes, but still homogeneous
                                   per-node — no hub nodes, no isolated chains

Corpus C (Cauchy latents):         some DAGs are star-shaped (one hub drives everything),
                                   some are sparse chains (A→B→C→D), some are dense webs
                                   → the prior covers qualitatively different causal regimes
</code></pre>
<p>Corpus C is what this section produces.  The Cauchy distribution's heavy tails
mean that occasionally one node's latent <span class="math inline">\(B_i\)</span>
is extreme, creating a natural hub; another node's <span class="math inline">\(C_j\)</span>
may be deeply negative, making it nearly parentless.  This heterogeneity is
deliberate — real tabular data comes from causal structures that range from
near-independent features to deeply entangled causal chains, and the prior must
cover that full range.  The <code>edge_logit_bias</code>
(<span class="math inline">\(\beta_{\text{edge}}\)</span>) shifts the overall
density surface up or down, acting as the "how connected" knob for the graph
axis of effective diversity.</p>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(G\)</span> (DAG adjacency matrix).<br>
<strong>Dependency Map:</strong> <span class="math inline">\(A, B_i,
C_j, \beta_{\text{edge}} \rightarrow p_{ij} \rightarrow
G_{ij}\)</span>.<br> <strong>Path to Final Output:</strong> <span
class="math inline">\(G \rightarrow \mathsf{pa}(j)\)</span> for each
node -&gt; node execution order and parent inputs -&gt; latent tensors
-&gt; emitted <code>X</code>, <code>y</code>, and graph
metadata.<br></p>
<p><strong>Rationale.</strong> The latent-logit Cauchy construction
yields heterogeneous edge probabilities while strict upper-triangular
masking guarantees acyclicity. Writing <span
class="math inline">\(p_{ij}\)</span> and <span
class="math inline">\(\beta_{\text{edge}}\)</span> explicitly keeps
graph-drift behavior inspectable and testable.</p>
<p><code>sample_dag</code> draws a strict upper-triangular adjacency
matrix.</p>
<p><span class="math display">\[A \sim \mathsf{Cauchy}(0,1), \quad B_i
\sim \mathsf{Cauchy}(0,1), \quad C_j \sim
\mathsf{Cauchy}(0,1)\]</span></p>
<p>Interpretation:</p>
<ul>
<li><span class="math inline">\(A\)</span>: global intercept-like term
shifting all edge logits together.</li>
<li><span class="math inline">\(B_i\)</span>: source-node effect for
node <span class="math inline">\(i\)</span> (how strongly node <span
class="math inline">\(i\)</span> tends to emit edges).</li>
<li><span class="math inline">\(C_j\)</span>: destination-node effect
for node <span class="math inline">\(j\)</span> (how strongly node <span
class="math inline">\(j\)</span> tends to receive edges).</li>
</ul>
<p>For each <span class="math inline">\(i &lt; j\)</span>:</p>
<p><span class="math display">\[p_{ij} = \sigma(A + B_i + C_j +
\beta_{\text{edge}}), \quad G_{ij} \sim
\mathsf{Bernoulli}(p_{ij})\]</span></p>
<p>and for <span class="math inline">\(i \ge j\)</span>, <span
class="math inline">\(G_{ij}=0\)</span>.</p>
<h2 id="2-shift-runtime-parameters">2. Shift Runtime Parameters</h2>
<p><strong>Motivation</strong>  A model trained on i.i.d. synthetic data
has never seen train/test mismatch during pretraining.  At inference, real
tasks routinely exhibit shift — and the model has no prior experience to fall
back on.  These three shift axes map to distinct real-world drift patterns:</p>
<pre><code>graph_scale  → causal structure itself changes    (e.g., new regulations alter which
               between train and test                variables are causally relevant)

mechanism_scale → functional relationships change  (e.g., customer behavior shifts while
                  while the graph stays the same     the same features remain relevant)

variance_scale  → stochastic variation changes     (e.g., sensor degradation increases
                  in magnitude                       measurement noise over time)
</code></pre>
<p>Because the three axes are independent, a researcher can ablate each in
isolation: hold graph and mechanism fixed while sweeping noise drift to measure
noise-robustness, or hold noise fixed while tilting mechanism complexity.
Shift-aware constructions in the prior have been shown to improve robustness
to temporal distribution shifts.  The
deterministic mappings from config scales to runtime coefficients
(<span class="math inline">\(\beta_{\text{edge}}\)</span>,
<span class="math inline">\(\tau\)</span>,
<span class="math inline">\(\gamma_\sigma\)</span>) keep each axis
interpretable and independently testable.</p>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(s_{\text{shift}} = (\text{graph\_scale},
\text{mechanism\_scale}, \text{variance\_scale})\)</span>.<br>
<strong>Dependency Map:</strong> <span
class="math inline">\(s_{\text{shift}} \rightarrow (\beta_{\text{edge}},
\tau, \gamma_{\sigma}, \gamma_{\text{var}})\)</span> via deterministic
runtime mappings.<br> <strong>Path to Final Output:</strong> <span
class="math inline">\(\beta_{\text{edge}} \rightarrow G\)</span>, <span
class="math inline">\(\tau \rightarrow \pi_f\)</span>, and <span
class="math inline">\(\gamma_{\sigma} \rightarrow\)</span> noise
amplitude; these three paths converge in node generation and sampling to
shape final <code>X</code>, <code>y</code>, and shift metadata.<br></p>
<p><strong>Rationale.</strong> Shift controls are defined as
deterministic mappings from config scales to runtime coefficients so
drift remains interpretable. Separating graph, mechanism, and variance
mappings keeps each drift axis analyzable in isolation.</p>
<p>With resolved shift scales
<code>(graph_scale, mechanism_scale, variance_scale)</code>:</p>
<p><span class="math display">\[\beta_{\text{edge}} = \ln(2) \cdot
\text{graph\_scale}\]</span> <span class="math display">\[\tau =
\text{mechanism\_scale}\]</span> <span
class="math display">\[\gamma_\sigma = \exp\left(\frac{\ln(2)}{2} \cdot
\text{variance\_scale}\right), \quad \gamma_{\text{var}} =
\gamma_\sigma^2 = 2^{\text{variance\_scale}}\]</span></p>
<p>So positive <code>variance_scale</code> increases global noise
variance multiplicatively.</p>
<h2 id="3-mechanism-family-sampling-with-mix--tilt">3. Mechanism Family
Sampling with Mix + Tilt</h2>
<p><strong>Motivation</strong>  Real tabular relationships span a wide
functional spectrum.  A prior restricted to, say, linear + neural-network
mechanisms will never produce piecewise decision boundaries (tree-like),
cluster-assignment patterns (discretization), or smooth periodic surfaces (GP)
— and the foundation model trained on it will struggle when encountering those
patterns at inference time.</p>
<pre><code>linear         →  y ≈ Wx + b                    (low complexity, smooth)
quadratic      →  y ≈ x'Ax + Wx + b             (pairwise interactions)
neural network →  y = σ(W₂ σ(W₁x + b₁) + b₂)   (high-capacity smooth nonlinearity)
tree ensemble  →  y = Σ leaf_values[split_path]  (piecewise constant, discontinuous)
discretization →  y = f(nearest_center(x))       (clustered, Voronoi-like)
GP / RFF       →  y = V cos(Ωx + b)             (smooth kernel, possibly periodic)
EM assignment  →  y = Σ p(k|x) · μ_k            (soft mixture, local structure)
product        →  y = f(x) · g(x)               (multiplicative interaction)
piecewise      →  y = staged_branches(x)         (explicit control path)
</code></pre>
<p>The <code>function_family_mix</code> controls <em>which</em> families are
present (support) and their baseline probabilities.  The
<code>mechanism_logit_tilt</code> (<span class="math inline">\(\tau\)</span>)
controls <em>how much</em> the distribution shifts toward nonlinear families
within that support.  Together they form the most direct lever for changing the
functional complexity profile of the prior.  If your model underperforms on
tasks with tree-like decision
boundaries, increasing tree weight in the mix and measuring the downstream
effect is the first experiment to try.</p>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(\pi_f\)</span> (mechanism-family sampling
distribution).<br> <strong>Dependency Map:</strong> <span
class="math inline">\(\lambda_f, \tau, \tilde{\ell}_f, \mathcal{F}
\rightarrow \pi_f\)</span>.<br> <strong>Path to Final Output:</strong>
<span class="math inline">\(\pi_f \rightarrow\)</span> sampled mechanism
family at each function draw -&gt; latent transform type and
nonlinearity -&gt; converted observable values in
<code>X</code>/<code>y</code> and nonlinear-mass-related
metadata.<br></p>
<p><strong>Rationale.</strong> This parameterization separates family
support control (<code>function_family_mix</code>) from directional
drift (<code>mechanism_logit_tilt</code>). Centered logits make tilt a
relative reweighting mechanism rather than an arbitrary global scaling
change.</p>
<p>Let raw family weights be:</p>
<p><span class="math display">\[\lambda_f = \begin{cases} 1, &amp;
\text{if no function\_family\_mix} \\ \text{configured positive weight},
&amp; \text{otherwise} \end{cases}\]</span></p>
<p>Normalize over positive-weight families:</p>
<p><span class="math display">\[\pi_f^{(0)} = \frac{\lambda_f}{\sum_{g
\in \mathcal{F}} \lambda_g}\]</span></p>
<p><span class="math inline">\(\pi_f^{(0)}\)</span> is the baseline
family distribution after applying <code>function_family_mix</code>
support/weights and before any tilt.</p>
<p>If <span class="math inline">\(\tau \le 0\)</span>, then <span
class="math inline">\(\pi_f = \pi_f^{(0)}\)</span>. If <span
class="math inline">\(\tau &gt; 0\)</span>, logits are tilted using
centered family base logits:</p>
<p><span class="math display">\[\pi_f \propto \exp\left(\ln \pi_f^{(0)}
+ \tau \tilde{\ell}_f\right)\]</span></p>
<p>then normalized to sum to 1.</p>
<p>Configuration intuition:</p>
<ul>
<li><code>mechanism.function_family_mix</code> controls support and
baseline weights through <span
class="math inline">\(\lambda_f\)</span>.</li>
<li><code>shift.mechanism_scale</code> sets <span
class="math inline">\(\tau\)</span> (Section 2), which reweights within
that support.</li>
<li>Increasing <span class="math inline">\(\tau\)</span> shifts mass
toward families with higher <span
class="math inline">\(\tilde{\ell}_f\)</span> and away from families
with lower <span class="math inline">\(\tilde{\ell}_f\)</span>, changing
mechanism usage frequencies that shape final
<code>X</code>/<code>y</code>.</li>
<li>Example: with mix <code>{nn: 0.7, linear: 0.3}</code>, <span
class="math inline">\(\tau=0\)</span> gives <span
class="math inline">\(\pi^{(0)}_{nn}=0.7,\pi^{(0)}_{linear}=0.3\)</span>;
with <span class="math inline">\(\tau&gt;0\)</span>, <span
class="math inline">\(nn\)</span> gains additional mass because its
centered base logit is higher than linear's.</li>
</ul>
<p>Base logits used in <span
class="math inline">\(\tilde{\ell}_f\)</span> (<strong>ONLY IF <span
class="math inline">\(\tau &gt; 0\)</span></strong>):</p>
<table>
<thead>
<tr>
<th style="text-align: left;">Family <span
class="math inline">\(f\)</span></th>
<th style="text-align: left;"><span
class="math inline">\(\ell_f\)</span></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><code>nn</code></td>
<td style="text-align: left;">0.7</td>
</tr>
<tr>
<td style="text-align: left;"><code>tree</code></td>
<td style="text-align: left;">0.7</td>
</tr>
<tr>
<td style="text-align: left;"><code>discretization</code></td>
<td style="text-align: left;">0.5</td>
</tr>
<tr>
<td style="text-align: left;"><code>gp</code></td>
<td style="text-align: left;">0.5</td>
</tr>
<tr>
<td style="text-align: left;"><code>linear</code></td>
<td style="text-align: left;">-0.8</td>
</tr>
<tr>
<td style="text-align: left;"><code>quadratic</code></td>
<td style="text-align: left;">-0.6</td>
</tr>
<tr>
<td style="text-align: left;"><code>em</code></td>
<td style="text-align: left;">-0.3</td>
</tr>
<tr>
<td style="text-align: left;"><code>product</code></td>
<td style="text-align: left;">0.9</td>
</tr>
</tbody>
</table>
<h4 id="rationale-for-non-uniform-base-logits">Rationale for Non-uniform
Base Logits</h4>
<p>The base logits define the <strong>direction of mechanism
drift</strong>, not the default family distribution.</p>
<ul>
<li>When <code>mechanism_logit_tilt &lt;= 0</code>, family probabilities
are just normalized family weights (uniform when no
<code>function_family_mix</code> is provided), so base logits have no
effect.</li>
<li>When <code>mechanism_logit_tilt &gt; 0</code>, centered base logits
reweight probabilities toward families with larger centered logits and
away from families with smaller centered logits.</li>
<li>In the current defaults, this shifts mass toward nonlinear families
(<code>nn</code>, <code>tree</code>, <code>discretization</code>,
<code>gp</code>, <code>product</code>) and away from simpler families
(<code>linear</code>, <code>quadratic</code>), increasing nonlinear
mechanism mass under positive tilt.</li>
<li><code>function_family_mix</code> still controls support and baseline
weights; base logits only reweight <strong>within enabled
families</strong>.</li>
</ul>
<p>Implementation note: <code>product</code> is only valid when at least
one of its component families is enabled (<code>tree</code>,
<code>discretization</code>, <code>gp</code>, <code>linear</code>,
<code>quadratic</code>).</p>
<h2 id="4-node-generation-pipeline">4. Node Generation Pipeline</h2>
<p><strong>Motivation</strong>  The node pipeline is where latent causal
structure becomes concrete tensors.  Each step — root sampling, parent
composition, sanitization, weighting, converter slicing — adds a controlled
layer of variation.  The design is ordered so that:</p>
<pre><code>root nodes   → fresh latent geometry (normal, uniform, unit-ball, correlated)
child nodes  → inherit + transform parent signals via multi-parent composition
all nodes    → sanitize → standardize → weight → slice for converters → scale
</code></pre>
<p>Multiple root-base geometries (Section 4.1) ensure that even before any
mechanism fires, latent inputs already vary in shape — uniform grids, spherical
point clouds, correlated ellipsoids.  Multi-parent composition (Section 4.2)
controls how upstream causal effects combine: additive (sum), multiplicative
(product), extremal (max), or smooth-extremal (logsumexp).  These choices
determine the interaction patterns visible in the final tabular data — a sum
aggregation produces additive parent effects, while a product aggregation
creates multiplicative interactions that no single-parent mechanism can mimic.</p>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(Z_j\)</span> (node-<span
class="math inline">\(j\)</span> latent tensor after pipeline
transforms).<br> <strong>Dependency Map:</strong> root/parent inputs,
mechanism draws, width terms <span
class="math inline">\((d_{\text{req}}, d_{\text{extra}})\)</span>,
random weights <span class="math inline">\(w\)</span>, and scale <span
class="math inline">\(s_j \rightarrow Z_j\)</span>.<br> <strong>Path to
Final Output:</strong> <span class="math inline">\(Z_j
\rightarrow\)</span> converter slices, extracted feature values, and one
selected target-node conversion -&gt; final
<code>X</code>/<code>y</code>.<br></p>
<p><strong>Rationale.</strong> Node generation is documented as an
ordered contract because converter extraction and metadata semantics
depend on stable execution order. The split into root sampling, parent
composition, and latent refinement mirrors the actual runtime path.</p>
<h3 id="41-root-node-base-sampling">4.1 Root-node base sampling</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(Z_{\text{base}}\)</span>.<br> <strong>Dependency
Map:</strong> base-kind choice + distribution parameters + active noise
family <span class="math inline">\(\mathcal{D}_{\text{noise}}
\rightarrow Z_{\text{base}}\)</span>.<br> <strong>Path to Final
Output:</strong> <span class="math inline">\(Z_{\text{base}} \rightarrow
f(Z_{\text{base}})\)</span> (initial node latent) -&gt; downstream node
pipeline -&gt; feature converter extraction -&gt;
<code>X</code>/<code>y</code>.<br></p>
<p><strong>Rationale.</strong> Multiple root base geometries increase
latent diversity before any parent effects exist. Passing every sampled
base through <span class="math inline">\(f(\cdot)\)</span> keeps root
and non-root nodes aligned under the same mechanism-family controls.</p>
<p>For root nodes (<span class="math inline">\(\mathsf{pa}(j) =
\varnothing\)</span>), one base kind is chosen uniformly from:</p>
<ol type="1">
<li><code>normal</code>: <span class="math inline">\(Z_{\text{base}}
\sim \mathcal{D}_{\text{noise}}^{n \times d}\)</span>.</li>
<li><code>uniform</code>: elementwise <span
class="math inline">\(\mathsf{Uniform}(-1,1)\)</span>.</li>
<li><code>unit_ball</code>: for each row, <span class="math display">\[v
\sim \mathcal{N}(0, I_d), \quad u \sim \mathsf{Uniform}(0,1), \quad z =
\frac{v}{|v|_2} u^{1/d}\]</span></li>
<li><code>normal_cov</code>: <span class="math display">\[E \sim
\mathcal{D}_{\text{noise}}^{n \times d}, \quad A \sim
\mathcal{D}_{\text{noise}}^{d \times d}, \quad Z_{\text{base}} = (E
\odot w^\top) A^\top\]</span> where <span
class="math inline">\(w\)</span> is sampled by
<code>sample_random_weights</code> (a positive, sum-to-one, randomly
permuted weight vector following a noisy power-law decay).</li>
</ol>
<p>Then a random mechanism is applied:</p>
<p><span class="math display">\[Z_j = f(Z_{\text{base}})\]</span></p>
<h3 id="42-parent-composition-non-root-nodes">4.2 Parent composition
(non-root nodes)</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(Z_{\text{comp}}\)</span> (composed parent latent
input to node transform).<br> <strong>Dependency Map:</strong> parent
tensors <span class="math inline">\({Z_{p_r}}\)</span> + composition
path choice + aggregation operator <span
class="math inline">\(\mathsf{Agg} \rightarrow
Z_{\text{comp}}\)</span>.<br> <strong>Path to Final Output:</strong>
<span class="math inline">\(Z_{\text{comp}} \rightarrow\)</span> node
mechanism output -&gt; updated <span class="math inline">\(Z_j\)</span>
-&gt; feature converter extraction plus target-node conversion -&gt;
emitted <code>X</code>/<code>y</code>.<br></p>
<p><strong>Rationale.</strong> The 50/50 concat-versus-aggregation
branch injects both feature-mixing and commutative interaction patterns.
This broadens functional regimes without changing the node interface
contract.</p>
<p>If <span class="math inline">\(|\mathsf{pa}(j)|=1\)</span>, apply one
random mechanism to that parent tensor.</p>
<p>If <span class="math inline">\(|\mathsf{pa}(j)| \ge 2\)</span>,
choose one path with equal probability:</p>
<ol type="1">
<li><strong>Concatenation path</strong>: <span
class="math display">\[Z_{\text{in}} = [Z_{p_1} | Z_{p_2} | \dots |
Z_{p_k}], \quad Z_j = f(Z_{\text{in}})\]</span></li>
<li><strong>Aggregation path</strong>: <span class="math display">\[Y_r
= f_r(Z_{p_r}), \quad Z_j = \mathsf{Agg}(Y_1, \dots, Y_k)\]</span> with
<span class="math inline">\(\mathsf{Agg}\)</span> sampled uniformly
from: <span class="math display">\[\sum_r Y_r, \quad \prod_r Y_r, \quad
\max_r Y_r, \quad \mathsf{LSE}(Y_1, \dots, Y_k)\]</span></li>
</ol>
<p>Concatenation semantics (implementation-faithful):</p>
<ul>
<li>If parent tensors are <span class="math inline">\(Z_{p_r} \in
\mathbb{R}^{n \times d_r}\)</span>, concatenation is along feature axis:
<span class="math display">\[Z_{\text{in}} \in \mathbb{R}^{n \times
\sum_{r=1}^k d_r}.\]</span></li>
<li>The concatenated tensor is passed through one random mechanism <span
class="math inline">\(f(\cdot)\)</span> with node output width target
<span class="math inline">\(d\)</span> (the node <code>total_dim</code>
from Section 4.3), so: <span class="math display">\[f: \mathbb{R}^{n
\times \sum_r d_r} \rightarrow \mathbb{R}^{n \times d}.\]</span></li>
<li>In this branch, aggregation kind is not used; <code>Agg</code> only
applies in the non-concatenation branch.</li>
</ul>
<p>Example:</p>
<ul>
<li>Three parents with shapes <span
class="math inline">\((n,4)\)</span>, <span
class="math inline">\((n,6)\)</span>, <span
class="math inline">\((n,3)\)</span> concatenate to <span
class="math inline">\((n,13)\)</span>.</li>
<li>If node target width is <span class="math inline">\(d=9\)</span>,
the mechanism maps <span
class="math inline">\((n,13)\rightarrow(n,9)\)</span> before
post-processing and converter slicing.</li>
</ul>
<h3 id="43-latent-post-processing-and-converter-slicing">4.3 Latent
post-processing and converter slicing</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(Z_{\text{post}}\)</span> (post-processed latent
tensor used for conversion).<br> <strong>Dependency Map:</strong> raw
node latent + sanitize/standardize operations + random weights <span
class="math inline">\(w\)</span> + normalization + padding noise +
slice/writeback updates <span class="math inline">\(\rightarrow
Z_{\text{post}}\)</span>.<br> <strong>Path to Final Output:</strong>
<span class="math inline">\(Z_{\text{post}} \rightarrow\)</span>
per-spec extracted values <span class="math inline">\((v_s)\)</span> and
rewritten latent slices -&gt; assigned feature outputs plus one selected
target-node conversion -&gt; final <code>X</code>/<code>y</code>.<br></p>
<p><strong>Rationale.</strong> Sanitization, standardization, weighting,
and norm scaling are applied before conversion to keep numerics stable
across mechanism families. The slice-and-writeback loop formalizes how
observable values are extracted while preserving latent continuity.</p>
<p><strong>Current default semantic note.</strong> In the shipped default
prior, both feature converters and the target converter are attached to
sampled latent nodes. One selected latent node emits <code>y</code>
directly, and optional missingness acts later as an observation process
over emitted features only.</p>
<p>For each node, converter specs define required width:</p>
<p><span class="math display">\[d_{\text{req}} = \sum_s \max(1, d_s),
\quad d = d_{\text{req}} + d_{\text{extra}}, \quad d_{\text{extra}} =
\max(1, \lfloor \mathsf{LogUniform}(1, 32) \rfloor)\]</span></p>
<p>After mechanism output:</p>
<p><span class="math display">\[Z_j \leftarrow
\mathsf{clip}(\mathsf{nan\_to\_num}(Z_j), -10^6, 10^6)\]</span> <span
class="math display">\[Z_j \leftarrow \mathsf{standardize}(Z_j)\]</span>
<span class="math display">\[Z_j \leftarrow Z_j \odot w^\top\]</span>
<span class="math display">\[Z_j \leftarrow
\frac{Z_j}{\max\left(\frac{1}{n} \sum_{r=1}^n |Z_{j,r:}|_2,
10^{-6}\right)}\]</span></p>
<p>Converter loop semantics:</p>
<ol type="1">
<li>Slice per-spec view <span class="math inline">\(V_s\)</span> from
current cursor over columns.</li>
<li>Pad missing columns with sampled noise if needed.</li>
<li>Apply converter: <span class="math inline">\((X&#39;_s, v_s) =
\mathsf{Converter}_s(V_s)\)</span>.</li>
<li>Write <span class="math inline">\(X&#39;_s\)</span> back into the
same slice (shape-padded/truncated as needed).</li>
<li>Store <span class="math inline">\(v_s\)</span> as extracted
observable value for the feature key.</li>
</ol>
<p>Final node scaling:</p>
<p><span class="math display">\[Z_j \leftarrow s_j Z_j, \quad s_j \sim
\mathsf{LogUniform}(0.1, 10)\]</span></p>
<h2 id="5-mechanism-family-definitions">5. Mechanism Family
Definitions</h2>
<p><strong>Motivation</strong>  This catalog is intentionally broad because
each family occupies a qualitatively distinct region of function space.
Narrowing the catalog narrows the prior:</p>
<pre><code>Remove trees       → prior cannot produce sharp, axis-aligned decision boundaries
Remove GP          → prior cannot produce smooth periodic or multi-scale surfaces
Remove quadratic   → prior loses explicit pairwise feature interactions
Remove product     → prior loses multiplicative cross-family effects
</code></pre>
<p>The breadth of this catalog is what makes mechanism diversity a meaningful
axis of effective diversity.  Each family's parameterization (random weights,
depths, kernel frequencies, etc.) adds within-family variation on top of the
between-family structural differences.  For a researcher evaluating whether to
add or remove a family, the question is: <em>does this family produce latent
structure that no other family can approximate?</em>  If yes, removing it
creates a coverage gap; if a proposed new family largely duplicates an existing
one, adding it increases code surface without improving diversity.</p>
<p><strong>Primary Variable:</strong> <span class="math inline">\(Y =
f(X)\)</span> (family-specific mechanism output).<br> <strong>Dependency
Map:</strong> mechanism-family parameters + input <span
class="math inline">\(X \rightarrow Y\)</span>.<br> <strong>Path to
Final Output:</strong> <span class="math inline">\(Y
\rightarrow\)</span> node latent updates and feedback slices -&gt;
converter-extracted observables -&gt; final
<code>X</code>/<code>y</code>.<br></p>
<p><strong>Rationale.</strong> The mechanism catalog intentionally spans
smooth, piecewise, assignment-based, and compositional transforms to
vary functional complexity under drift and mix controls. Each family
maps to a concrete sampler implementation in
<code>random_functions.py</code>.</p>
<p>Inputs to mechanism families are first sanitized and
standardized.</p>
<h3 id="51-linear">5.1 Linear</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(Y_{\text{linear}}\)</span>.<br> <strong>Dependency
Map:</strong> <span class="math inline">\(X, M \rightarrow
Y_{\text{linear}} = X M^\top\)</span>.<br> <strong>Path to Final
Output:</strong> <span class="math inline">\(Y_{\text{linear}}
\rightarrow\)</span> latent columns for the node -&gt; downstream
converter extraction -&gt; numeric/categorical outputs in <code>X</code>
or <code>y</code>.<br></p>
<p><strong>Rationale.</strong> Linear maps act as a low-complexity
baseline and calibration point for drift against nonlinear families.
Row-normalized random matrices preserve directional mixing while
limiting runaway scale.</p>
<p><span class="math display">\[Y = X M^\top\]</span></p>
<p>where <span class="math inline">\(M\)</span> is a sampled random
matrix.</p>
<h3 id="52-quadratic">5.2 Quadratic</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(Y_{\text{quad}}\)</span>.<br> <strong>Dependency
Map:</strong> <span class="math inline">\(X_{\text{sub}},
X_{\text{aug}}, M_t \rightarrow Y_{\text{quad}}\)</span>.<br>
<strong>Path to Final Output:</strong> <span
class="math inline">\(Y_{\text{quad}} \rightarrow\)</span>
interaction-heavy latent signal -&gt; converter extraction -&gt;
nonlinear structure in emitted features and in any downstream target-node
ancestry.<br></p>
<p><strong>Rationale.</strong> Quadratic forms add pairwise interactions
not available in pure linear maps. Subspace capping and bias
augmentation balance expressivity with numerical stability and compute
control.</p>
<p>Let <span class="math inline">\(X_{\text{sub}}\)</span> be either all
columns or a random 20-column subset if input width exceeds 20. Augment
with bias column: <span class="math inline">\(X_{\text{aug}} =
[X_{\text{sub}}, \mathbf{1}]\)</span>. For each output channel <span
class="math inline">\(t\)</span>:</p>
<p><span class="math display">\[Y_t = \mathsf{diag}(X_{\text{aug}}, M_t,
X_{\text{aug}}^\top)\]</span></p>
<p>or equivalently per row: <span class="math inline">\(y_i =
x_{\text{aug},i}^\top M_t, x_{\text{aug},i}\)</span> — a general (not
necessarily symmetric) quadratic form. The bias augmentation means this
captures quadratic + linear + constant terms.</p>
<h3 id="53-neural-network">5.3 Neural network</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(Y_{\text{nn}}\)</span>.<br> <strong>Dependency
Map:</strong> <span class="math inline">\(X\)</span>, sampled layer
matrices <span class="math inline">\({M_\ell}\)</span>, and sampled
activations <span class="math inline">\({\phi_\ell}\)</span> <span
class="math inline">\(\rightarrow Y_{\text{nn}}\)</span>.<br>
<strong>Path to Final Output:</strong> <span
class="math inline">\(Y_{\text{nn}} \rightarrow\)</span> high-capacity
latent states -&gt; converter extraction -&gt; complex nonlinear
observables in <code>X</code>/<code>y</code>.<br></p>
<p><strong>Rationale.</strong> Random-depth MLPs provide high-capacity
nonlinear transforms with stochastic activation structure. Optional
pre/post activation steps increase shape diversity while retaining a
simple sampled-layer pipeline.</p>
<p>A random depth/width MLP with number of weight matrices sampled
uniformly from <span class="math inline">\({1, 2, 3}\)</span> (0–2
hidden layers) and hidden width from <span class="math inline">\(\lfloor
\mathsf{LogUniform}(1, 127) \rfloor\)</span>:</p>
<p><span class="math display">\[Y^{(0)} = X \text{ (optional
pre-activation with prob 0.5)}\]</span> <span
class="math display">\[Y^{(\ell+1)} = \phi_\ell(Y^{(\ell)}
M_\ell^\top)\]</span></p>
<p>with activation after hidden layers, optional final activation (prob
0.5).</p>
<p>Activation sampler includes fixed and parametric families.</p>
<ul>
<li>Fixed activations: <code>tanh</code>, <code>leaky_relu</code>,
<code>elu</code>, <code>identity</code>, <code>silu</code>,
<code>swiglu</code>, <code>relu</code>, <code>relu_sq</code>,
<code>softplus</code>, <code>sign</code>, <code>gauss</code>,
<code>exp</code>, <code>sin</code>, <code>square</code>,
<code>abs</code>, <code>softmax</code>, <code>logsigmoid</code>,
<code>logabs</code>, <code>sigmoid</code>, <code>round</code>,
<code>mod1</code>, <code>selu</code>, <code>relu6</code>,
<code>hardtanh</code>, <code>indicator_01</code>,
<code>onehot_argmax</code>, <code>argsort</code>,
<code>rank</code>.</li>
<li>Parametric families: <code>relu_pow</code>, <code>signed_pow</code>,
<code>inv_pow</code>, <code>poly</code>, <code>gumbel_softmax</code>.</li>
</ul>
<p><code>gumbel_softmax</code> applies row-wise Gumbel perturbations followed by
temperature-scaled softmax. When activation standardization is enabled later in
the pipeline, those simplex-like row sums are no longer guaranteed to remain
exactly 1 in the emitted latent state.</p>
<h3 id="54-tree-ensemble-oblivious-decision-trees">5.4 Tree ensemble
(oblivious decision trees)</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(Y_{\text{tree}}\)</span>.<br> <strong>Dependency
Map:</strong> split features, thresholds, leaf indices, and leaf values
<span class="math inline">\(\rightarrow Y_{\text{tree}}\)</span>.<br>
<strong>Path to Final Output:</strong> <span
class="math inline">\(Y_{\text{tree}} \rightarrow\)</span> piecewise
latent regions -&gt; converter extraction -&gt; discontinuous/tabular
decision patterns in <code>X</code>/<code>y</code>.<br></p>
<p><strong>Rationale.</strong> Oblivious trees introduce thresholded
piecewise behavior that complements smooth families. Variance-weighted
split sampling biases random partitions toward informative dimensions
without requiring supervised fitting.</p>
<p>Number of trees: <span class="math inline">\(T = \max(1, \lfloor
\mathsf{LogUniform}(1, 32) \rfloor)\)</span>.</p>
<p>For each tree <span class="math inline">\(t\)</span>:</p>
<ol type="1">
<li>Depth <span class="math inline">\(d_t \sim \mathsf{UniformInt}\{1,
\dots, 7\}\)</span>.</li>
<li>Sample split feature indices (variance-weighted probabilities when
available).</li>
<li>Sample thresholds by indexing sampled training rows.</li>
<li>Compute leaf index via packed split bits.</li>
<li>Sample leaf values <span class="math inline">\(V_t \in
\mathbb{R}^{2^{d_t} \times d_{\text{out}}}\)</span> from noise
sampler.</li>
</ol>
<p>Prediction is averaged:</p>
<p><span class="math display">\[Y = \frac{1}{T} \sum_{t=1}^T
V_t[\mathsf{leaf}(X)]\]</span></p>
<h3 id="55-discretization">5.5 Discretization</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(Y_{\text{disc}}\)</span>.<br> <strong>Dependency
Map:</strong> sampled centers + distance exponent <span
class="math inline">\(p\)</span> + nearest-center index + linear map
<span class="math inline">\(\rightarrow Y_{\text{disc}}\)</span>.<br>
<strong>Path to Final Output:</strong> <span
class="math inline">\(Y_{\text{disc}} \rightarrow\)</span> clustered
latent representation -&gt; converter extraction -&gt;
quantized/cluster-like observables in
<code>X</code>/<code>y</code>.<br></p>
<p><strong>Rationale.</strong> Nearest-center assignment creates
clustered discontinuities and quantization effects that smooth
mechanisms do not capture. A final linear map preserves shared
output-shape contracts across families.</p>
<p>Number of centers: <span class="math inline">\(K =
\mathsf{clamp}(\lfloor \mathsf{LogUniform}(2, 128) \rfloor, 2,
n)\)</span>.</p>
<p>Sample centers from rows, compute <span
class="math inline">\(L_p\)</span>-style distances (with sampled <span
class="math inline">\(p \sim \mathsf{LogUniform}(0.5, 4)\)</span>),
assign nearest center, then apply linear map:</p>
<p><span class="math display">\[k^{\ast}(i) = \arg\min_k \sum_q |X_{iq}
- c_{kq}|^p, \quad Y = \mathsf{Linear}(c_{k^{\ast}(i)})\]</span></p>
<h3 id="56-gprff-approximation">5.6 GP/RFF approximation</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(Y_{\text{gp}}\)</span>.<br> <strong>Dependency
Map:</strong> <span class="math inline">\(X_{\text{proj}}, \Omega, b, V
\rightarrow \Phi \rightarrow Y_{\text{gp}}\)</span>.<br> <strong>Path to
Final Output:</strong> <span class="math inline">\(Y_{\text{gp}}
\rightarrow\)</span> smooth kernel-like latent behavior -&gt; converter
extraction -&gt; continuous nonlinear structure in
<code>X</code>/<code>y</code>.<br></p>
<p><strong>Rationale.</strong> Random Fourier features approximate
kernel-style smooth nonlinearities at fixed computational cost. The widened
<code>gp</code> family now keeps one output interface while varying both the
frequency-sampling branch and the internal spectral variant, which broadens
the kinds of smooth or oscillatory behavior that can appear without adding a
new public config surface.</p>
<p>Uses <span class="math inline">\(P=256\)</span> random features and
two branches for sampling frequency matrix <span
class="math inline">\(\Omega\)</span>. Core output:</p>
<p><span class="math display">\[\Phi = \cos(X_{\text{proj}} \Omega^\top
+ b), \quad Y = \frac{1}{\sqrt{P}} \Phi V^\top\]</span></p>
<p>where <span class="math inline">\(b\)</span> is random phase and
<span class="math inline">\(V\)</span> is sampled from noise.</p>
<p>Variant-specific widening keeps the same readout contract:</p>
<ul>
<li><code>standard</code>: uses the base branch-specific frequency draw with the
same cosine readout shown above.</li>
<li><code>multiscale</code>: rescales one half of the random features toward a
lower spectral band and the other half toward a higher spectral band before the
cosine readout, increasing multi-resolution behavior.</li>
<li><code>periodic</code>: applies sampled integer harmonics <span
class="math inline">\(h_p \in \{1,\dots,5\}\)</span> to the phase logits and
uses a mixed cosine/sine readout
<span class="math display">\[\Phi_p = \frac{\cos(h_p \theta_p + b_p) +
\sin(h_p \theta_p - b_p)}{\sqrt{2}}\]</span>
to bias the latent toward more explicitly oscillatory structure.</li>
</ul>
<h3 id="57-em-style-assignment">5.7 EM-style assignment</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(Y_{\text{em}}\)</span>.<br> <strong>Dependency
Map:</strong> centers <span class="math inline">\(c_k\)</span>, scales
<span class="math inline">\(\sigma_k\)</span>, exponents <span
class="math inline">\((p,q)\)</span>, and assignment probabilities <span
class="math inline">\(P_{ik} \rightarrow Y_{\text{em}}\)</span>.<br>
<strong>Path to Final Output:</strong> <span
class="math inline">\(Y_{\text{em}} \rightarrow\)</span> mixture-like
local latent effects -&gt; converter extraction -&gt; locally structured
observables in <code>X</code>/<code>y</code>.<br></p>
<p><strong>Rationale.</strong> Soft center assignments add mixture-like
local behavior and scale-sensitive responses. The linear readout over
assignment probabilities keeps compatibility with common downstream
handling.</p>
<p>Sample centers <span class="math inline">\(c_k\)</span>, scales <span
class="math inline">\(\sigma_k\)</span>, and exponents <span
class="math inline">\(p, q\)</span>:</p>
<ul>
<li>Centers: <span class="math inline">\(K = \max(2, \lfloor
\mathsf{LogUniform}(2, \max(16, 2 d_{\text{out}}))
\rfloor)\)</span></li>
<li>Scales: <span class="math inline">\(\sigma_k = \exp(\text{noise}
\cdot 0.1)\)</span> (log-normal, concentrated near 1)</li>
<li>Exponents: <span class="math inline">\(p \sim \mathsf{LogUniform}(1,
4),\; q \sim \mathsf{LogUniform}(1, 2)\)</span></li>
</ul>
<p><span class="math display">\[d_{ik} = \left(\sum_q |X_{iq} -
c_{kq}|^p\right)^{1/p}\]</span> <span
class="math display">\[\mathsf{logit}_{ik} = -\frac{1}{2}
\log(2\pi\sigma_k^2) - \left(\frac{d_{ik}}{\max(\sigma_k,
10^{-6})}\right)^q\]</span> <span class="math display">\[P_{ik} =
\mathsf{softmax}_k(\mathsf{logit}_{ik}), \quad Y =
\mathsf{Linear}(P)\]</span></p>
<h3 id="58-product-family">5.8 Product family</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(Y_{\text{prod}}\)</span>.<br> <strong>Dependency
Map:</strong> component family draws <span
class="math inline">\(f(X)\)</span> and <span
class="math inline">\(g(X)\)</span> from allowed support <span
class="math inline">\(\rightarrow Y_{\text{prod}} = f(X)\odot
g(X)\)</span>.<br> <strong>Path to Final Output:</strong> <span
class="math inline">\(Y_{\text{prod}} \rightarrow\)</span>
multiplicative latent interactions -&gt; converter extraction -&gt;
higher-order effects in final <code>X</code>/<code>y</code>.<br></p>
<p><strong>Rationale.</strong> Multiplying two component mechanisms
introduces multiplicative interactions and sharper nonlinear responses
than additive composition alone. Restricting component candidates to
vetted families maintains stability under family-mix masking.</p>
<p>Pick two component families <span class="math inline">\(f, g\)</span>
from allowed set
<code>(tree, discretization, gp, linear, quadratic)</code> (filtered by
enabled family mix), then:</p>
<p><span class="math display">\[Y = f(X) \odot g(X)\]</span></p>
<h2 id="6-converter-definitions">6. Converter Definitions</h2>
<p><strong>Motivation</strong>  Converters are the emission boundary — they
translate latent causal state into the tabular features a
downstream model actually sees.  Even if the latent mechanisms are diverse,
monotonous converters produce monotonous observed distributions:</p>
<pre><code>latent Z ──→ numeric converter ──→ observed feature column
             ├─ identity pass-through    → Gaussian-looking marginals
             ├─ Kumaraswamy warp         → skewed / beta-like marginals
             └─ power / log transform    → heavy-tailed or compressed marginals

latent Z ──→ categorical converter ──→ observed class labels
             ├─ argmax on logits         → hard cluster boundaries
             ├─ softmax + Categorical    → soft probabilistic assignment
             └─ k-means style centers    → Voronoi-like class regions
</code></pre>
<p>The numeric converter's optional Kumaraswamy warp varies marginal
distributions beyond the standard Gaussian shape; the categorical converter's
multiple method/variant combinations produce diverse class-assignment patterns.
Without converter diversity, even mechanistically diverse latent signals could
produce tabular data where every numeric column looks Gaussian and every
categorical column uses the same assignment logic — a subtle effective-diversity
gap that is easy to overlook.</p>
<p><strong>Current default semantic note.</strong> The default target path
attaches a target converter to one selected latent DAG node rather than
sampling <code>y</code> from a separate post-feature head.</p>
<p><strong>Primary Variable:</strong> <span
class="math inline">\((X&#39;_s, v_s)\)</span> (converter output state
and extracted observable value for spec <span
class="math inline">\(s\)</span>).<br> <strong>Dependency Map:</strong>
latent slice, converter kind, and converter parameters <span
class="math inline">\(\rightarrow (X&#39;_s, v_s)\)</span>.<br>
<strong>Path to Final Output:</strong> extracted <span
class="math inline">\(v_s\)</span> values are assigned to feature
slots and become emitted feature values; one selected latent node emits
<code>y</code> through its target converter; later missingness may censor
emitted <code>X</code>; <span
class="math inline">\(X&#39;_s\)</span> is written back to latent for
downstream coupling.<br></p>
<p><strong>Rationale.</strong> Converters are separated from mechanism
families because observable typing is an output-contract concern, not a
latent-mechanism concern. This decoupling lets one latent pipeline serve
multiple feature schemas.</p>
<h3 id="61-numeric-converter">6.1 Numeric converter</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\((X&#39;_{\text{num}},
v_{\text{num}})\)</span>.<br> <strong>Dependency Map:</strong> input
slice <span class="math inline">\(X\)</span>, warp parameters <span
class="math inline">\((a,b)\)</span>, and identity/warp branch choice
<span class="math inline">\(\rightarrow (X&#39;_{\text{num}},
v_{\text{num}})\)</span>.<br> <strong>Path to Final Output:</strong>
<span class="math inline">\(v_{\text{num}}\)</span> is emitted as
numeric feature value in <code>X</code>; <span
class="math inline">\(X&#39;_{\text{num}}\)</span> feeds back into node
latent state.<br></p>
<p><strong>Rationale.</strong> The identity branch preserves raw latent
structure when extra warping is unnecessary. The optional bounded
Kumaraswamy-like warp provides controllable marginal shaping while
staying numerically stable.</p>
<p>Given input slice <span class="math inline">\(X\)</span> and raw
extracted scalar <span class="math inline">\(v = X_{:,0}\)</span>:</p>
<ul>
<li>with probability 0.5: identity output (no warp),</li>
<li>otherwise sample <span class="math inline">\(a,b \sim
\mathsf{LogUniform}(0.2,5)\)</span> and apply columnwise min-max +
warp:</li>
</ul>
<p><span class="math display">\[\hat{X} = \frac{X -
X_{\min}}{\max(X_{\max} - X_{\min}, 10^{-6})}\]</span> <span
class="math display">\[X&#39; = 1 - \left(1 - \mathsf{clip}(\hat{X}, 0,
1)^a\right)^b\]</span></p>
<p>Converter returns <span class="math inline">\((X&#39;,
v)\)</span>.</p>
<h3 id="62-categorical-converter">6.2 Categorical converter</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\((X&#39;_{\text{cat}}, \ell)\)</span> where <span
class="math inline">\(\ell\)</span> is categorical label output.<br>
<strong>Dependency Map:</strong> method/variant choice, cardinality
<span class="math inline">\(C\)</span>, centers or logits/probabilities
<span class="math inline">\(\rightarrow (X&#39;_{\text{cat}},
\ell)\)</span>.<br> <strong>Path to Final Output:</strong> <span
class="math inline">\(\ell\)</span> is emitted as categorical
feature/class target in <code>X</code>/<code>y</code>; <span
class="math inline">\(X&#39;_{\text{cat}}\)</span> is fed back into
latent columns.<br></p>
<p><strong>Rationale.</strong> Joint method-variant sampling separates
class assignment logic from latent representation feedback into node
state. This increases categorical behavior diversity while preserving a
consistent <code>(out, labels)</code> converter contract.</p>
<p>A joint <code>(method, variant)</code> pair is sampled from:</p>
<ul>
<li><code>neighbor</code>: <code>input</code>,
<code>index_repeat</code>, <code>center</code>,
<code>center_random_fn</code></li>
<li><code>softmax</code>: <code>input</code>, <code>index_repeat</code>,
<code>softmax_points</code></li>
</ul>
<p>Let target cardinality be <span class="math inline">\(C = \max(2,
n_{\text{categories}})\)</span>.</p>
<p>Neighbor method:</p>
<p><span class="math display">\[\text{label}_i = \arg\min_k \sum_q
|X_{iq} - c_{kq}|^p, \quad p \sim \mathsf{LogUniform}(0.5,
4)\]</span></p>
<p>Softmax method:</p>
<p><span class="math display">\[L = a
\mathsf{standardize}(X_{\text{proj}}) + b, \quad a \sim
\mathsf{LogUniform}(0.1, 10), \quad b_k = \ln(u_k + 10^{-4})\]</span>
<span class="math display">\[\text{label}_i \sim
\mathsf{Categorical}(\mathsf{softmax}(L_i))\]</span></p>
<p>Output representation <span class="math inline">\(X&#39;\)</span> is
variant-dependent (input copy, repeated index, center lookup, center
transformed by random function, or random softmax points). Labels are
finally reduced modulo <span class="math inline">\(C\)</span>.</p>
<h2 id="7-noise-sampling-and-runtime-selection">7. Noise Sampling and
Runtime Selection</h2>
<p><strong>Motivation</strong>  Noise is the stochastic axis of the prior,
independent of mechanism complexity.  Restricting noise to Gaussian produces
datasets whose residuals always have light, symmetric tails — but real tabular
residuals are often heavy-tailed (financial returns), asymmetric (truncated
sensor readings), or heteroscedastic.  The three primitive families span a
tail-weight gradient:</p>
<pre><code>Gaussian    →  light tails, exp(-x²/2)        → most benign residual regime
Laplace     →  moderate tails, exp(-|x|)       → double-exponential, sharper peak
Student-t   →  heavy tails, (1+x²/ν)^(-(ν+1)/2)  → occasional extreme outliers (ν &gt; 2)
mixture     →  per-dataset draw from the above → corpus spans all three regimes
</code></pre>
<p>The <code>mixture</code> mode resolves one family per dataset attempt
(Section 7.2), so a corpus generated with <code>mixture</code> naturally
contains Gaussian-noise datasets, Laplace-noise datasets, and Student-t-noise
datasets — broadening the noise axis of effective diversity without requiring
separate generate runs.  Noise diversity in the prior has been shown to improve
performance on real datasets with non-Gaussian residual structure.</p>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(\epsilon\)</span> (active noise draw process for
the attempt).<br> <strong>Dependency Map:</strong> noise family, scale,
and family-specific parameters <span class="math inline">\(\rightarrow
\epsilon\)</span>.<br> <strong>Path to Final Output:</strong> <span
class="math inline">\(\epsilon\)</span> perturbs points, matrices,
weights, and explicit noise draws -&gt; latent variability and converter
inputs -&gt; final <code>X</code>/<code>y</code> dispersion and noise
metadata.<br></p>
<p><strong>Rationale.</strong> Noise is modeled as its own axis so
distributional perturbation can be controlled independently from
mechanism complexity. Distinguishing primitive samplers from runtime
selection clarifies where stochasticity enters.</p>
<h3 id="71-primitive-family-samplers">7.1 Primitive family samplers</h3>
<p><strong>Primary Variable:</strong> <span
class="math inline">\(\epsilon_{\text{family}}\)</span> (draws from one
resolved primitive family).<br> <strong>Dependency Map:</strong>
selected primitive family + shape + scale (+ df or mixture weights when
applicable) <span class="math inline">\(\rightarrow
\epsilon_{\text{family}}\)</span>.<br> <strong>Path to Final
Output:</strong> <span
class="math inline">\(\epsilon_{\text{family}}\)</span> is used by
samplers/mechanisms across generation -&gt; affects latent tensors and
emitted <code>X</code>/<code>y</code>.<br></p>
<p><strong>Rationale.</strong> Multiple primitive families cover
light-tail, double-exponential, and heavy-tail regimes under one API
surface. Explicit formulas keep distributional assumptions auditable
across backends.</p>
<p>For <span class="math inline">\(\mathsf{sample\_noise}(\text{shape,
family, scale, }\dots)\)</span>:</p>
<ul>
<li><strong>Gaussian</strong>: i.i.d. <span
class="math inline">\(\mathcal{N}(0, 1)\)</span></li>
<li><strong>Laplace</strong> (inverse CDF): <span
class="math display">\[U \sim \mathsf{Uniform}(\epsilon, 1-\epsilon),
\quad X = \begin{cases} \ln(2U), &amp; U &lt; 0.5 \\ -\ln(2(1-U)), &amp;
U \ge 0.5 \end{cases}\]</span></li>
<li><strong>Student-t</strong>: <span class="math display">\[X =
\frac{Z}{\sqrt{\chi^2_\nu / \nu}}, \quad Z \sim \mathcal{N}(0, 1), \quad
\nu &gt; 2\]</span></li>
<li><strong>Mixture</strong>: sample component assignment per element
from normalized weights over <code>{gaussian, laplace, student_t}</code>
and draw from that component.</li>
</ul>
<p>All outputs are scaled by <code>scale</code>.</p>
<h3 id="72-dataset-level-noise-family-resolution">7.2 Dataset-level
noise-family resolution</h3>
<p><strong>Primary Variable:</strong> <code>noise_spec</code> (resolved
attempt-level noise sampling specification).<br> <strong>Dependency
Map:</strong> requested family, mixture weights, and deterministic seed
stream <span class="math inline">\(\rightarrow\)</span> runtime
selection <span class="math inline">\(\rightarrow\)</span>
<code>noise_spec</code>.<br> <strong>Path to Final Output:</strong>
<code>noise_spec</code> parameterizes all downstream noise draws in the
attempt -&gt; dataset-level distributional character in final
<code>X</code>/<code>y</code> plus noise-distribution metadata.<br></p>
<p><strong>Rationale.</strong> In mixture mode, sampling one family per
dataset attempt preserves coherent dataset-level identity instead of
per-element family switching. Seeded selection guarantees
reproducibility and metadata traceability.</p>
<p>Generation runtime resolves one <code>NoiseRuntimeSelection</code>
per dataset attempt:</p>
<ul>
<li>If requested family is not <code>mixture</code>: sampled family
equals requested family.</li>
<li>If requested family is <code>mixture</code>:
<ol type="1">
<li>Normalize configured mixture weights.</li>
<li>Use deterministic keyed RNG stream
<code>KeyedRng(run_seed).keyed("dataset", i, "noise_runtime", "family_selection").torch_rng(...)</code>.</li>
<li>Sample exactly one component family for the dataset.</li>
</ol></li>
</ul>
<p>The resulting <code>noise_spec</code> uses that sampled family and
base scale for all downstream draws in the attempt. Shift noise drift
applies through multiplicative scale factor <span
class="math inline">\(\gamma_\sigma\)</span>.</p>
