# Mathematical specification for the clean rewrite

## 1. Scientific question

The project asks one primary question:

> Does the topology of short acceleration and gyroscope trajectories contain
> activity information that transfers to people not used for training?

Two secondary questions make the topological claim interpretable:

1. Does the information come from temporal order, rather than only the
   unordered distribution of sensor samples?
2. Can a persistent one-dimensional class be interpreted as a circular motion
   coordinate for periodic activity?

The aim is not to beat the engineered UCI HAR classifier. The aim is to show
what the selected topological representation measures, whether it generalizes
across subjects, and where it loses information.

## 2. Data and unit of analysis

For sensor type $s\in\{a,g\}$, one observation is a sampled path

$$
x^{(s)}:\{0,\ldots,127\}\longrightarrow\mathbb R^3,
$$

where $a$ denotes body acceleration, $g$ denotes body angular velocity,
and the sampling frequency is 50 Hz. Each path is therefore a 2.56-second
window with one activity label and one subject identifier.

The supplied UCI split is retained exactly:

- the development set contains 21 subjects;
- the official test set contains 9 different subjects;
- windows from a subject may never occur in both fitting and validation data.

Balanced accuracy is the primary score because the six activity counts are
unequal. Macro F1 is reported as a secondary score.

## 3. From a path to a finite metric space

### 3.1 Delay coordinates

For embedding dimension $m$ and lag $\ell$, define

$$
E_{m,\ell}(x)_t=
\bigl(x_t,x_{t+\ell},\ldots,x_{t+(m-1)\ell}\bigr)
\in\mathbb R^{3m},
$$

for

$$
t=0,\ldots,127-(m-1)\ell.
$$

The resulting finite point cloud is

$$
X_{m,\ell}(x)=
\{E_{m,\ell}(x)_t\mid 0\leq t\leq127-(m-1)\ell\}.
$$

The primary values are $m=3$ and $\ell=15$. They produce 98 points in
$\mathbb R^9$. The lag is justified by the first non-positive value of the
mean normalized acceleration autocorrelation on the development data. It is
not tuned against classification accuracy.

The instantaneous control uses $m=1,\ell=1$, giving the unordered 128 sensor
samples in $\mathbb R^3$.

### 3.2 Translation and scale normalization

For a cloud $X=\{z_1,\ldots,z_n\}$, define its centroid and RMS radius by

$$
\bar z=\frac1n\sum_{i=1}^n z_i,
\qquad
r(X)=\left(\frac1n\sum_{i=1}^n\lVert z_i-\bar z\rVert_2^2\right)^{1/2}.
$$

The normalized cloud is

$$
\widetilde X=\left\{
\frac{z_i-\bar z}{r(X)}:i=1,\ldots,n
\right\}.
$$

This makes the primary topology invariant to translation and positive global
rescaling of a window. The discarded radii
$r(X^{(a)})$ and $r(X^{(g)})$ are retained as two explicit physical-scale
features in one comparison. They are never silently mixed into the normalized
topological vector.

The metric on each normalized cloud is Euclidean distance.

## 4. Vietoris-Rips persistence

For a finite metric space $(X,d)$ and threshold $\epsilon\geq0$, the
Vietoris-Rips complex is

$$
\operatorname{VR}_\epsilon(X)=
\{\sigma\subseteq X:\ d(x,y)\leq\epsilon
\text{ for every }x,y\in\sigma\}.
$$

Because $X$ is finite, this filtration is tame. Apply homology over
$\mathbb F_{47}$ to obtain persistence diagrams

$$
D_0(X),\qquad D_1(X).
$$

Here $H_0$ records the merging of connected components and $H_1$ records
loops. `ripser` performs this calculation. The coefficient field is fixed to
47 because the circular extension requires a sufficiently fine finite-field
cocycle; using the same field lets one persistence call supply both the stable
ranks and circular dynamics. Infinite $H_0$ classes are removed before
vectorization.

## 5. Stable-rank representation

For a finite persistence diagram $D_k$, define the lifetime stable rank

$$
\rho_k(u)=
\#\{(b,d)\in D_k:d-b>u\},\qquad u\geq0.
$$

Evaluate it on the fixed grid

$$
u_j=\frac{2j}{63},\qquad j=0,\ldots,63.
$$

`stablebear` evaluates the stable-rank function. For one window, concatenate

$$
\Phi_{\mathrm{SR}}(x)=
[\rho^{(a)}_0(u_j),\rho^{(a)}_1(u_j),
  \rho^{(g)}_0(u_j),\rho^{(g)}_1(u_j)]_{j=0}^{63}
\in\mathbb R^{256}.
$$

The block order is fixed and tested:

1. acceleration $H_0$;
2. acceleration $H_1$;
3. gyroscope $H_0$;
4. gyroscope $H_1$.

The numerical stable-rank distance used for activity prototypes is the
left-Riemann approximation

$$
d_{\mathrm{SR}}(f,g)=
\frac{2}{63}\sum_{j=0}^{62}|f(u_j)-g(u_j)|.
$$

For concatenated sensor/homology blocks, this integral is computed separately
for every 64-coordinate block and then summed. This avoids accidentally
dropping only one endpoint from the entire 256-vector.

## 6. Temporal-order control

For each window independently, draw a deterministic permutation
$\pi$ from seed $2956+i$, where $i$ is the window index, and form

$$
x^\pi_t=x_{\pi(t)}.
$$

Apply the same $(m,\ell)=(3,15)$ delay construction and persistence pipeline
to $x^\pi$. This preserves the multiset of three-dimensional samples while
destroying temporal adjacency. A performance decrease from delay topology to
shuffled-delay topology supports a temporal interpretation. It does not prove
that the representation recovers the complete time ordering.

## 7. Compact persistence and circular dynamics

Stable rank retains lifetime counts but discards individual birth values and
the identity of a particular loop. The extension therefore summarizes each
sensor diagram by six transparent quantities:

$$
\begin{aligned}
q_1 &= \text{mean finite }H_0\text{ lifetime},\\
q_2 &= \text{maximum finite }H_0\text{ lifetime},\\
q_3 &= |D_1|,\\
q_4 &= \max_{(b,d)\in D_1}(d-b),\\
q_5 &= \sum_{(b,d)\in D_1}(d-b),\\
q_6 &= \text{birth of the longest }H_1\text{ class}.
\end{aligned}
$$

If $D_1\neq\varnothing$, select the longest class $(b^*,d^*)$ and the
corresponding cocycle returned by `ripser` over $\mathbb F_{47}$. Evaluate
the cocycle on the Rips graph at

$$
\epsilon^*=b^*+\tfrac12(d^*-b^*).
$$

Lift its coefficients to centered integers and let $B$ be the oriented
edge-vertex incidence matrix. Anchor the first vertex and solve

$$
\widehat\phi=
\arg\min_{\phi:\phi_0=0}\lVert B\phi+\alpha\rVert_2^2
$$

with SciPy's sparse `lsqr`. The circle-valued coordinate is

$$
\theta_i=\widehat\phi_i\bmod1.
$$

After unwrapping $2\pi\theta_i$ in the original delay-point order, regress it
on time $t_i=i/50$. Record

$$
q_7=|\widehat\beta_1|/(2\pi)
$$

as circular frequency in hertz and

$$
q_8=1-
\frac{\sum_i(\varphi_i-\widehat\varphi_i)^2}
     {\sum_i(\varphi_i-\bar\varphi)^2}
$$

as phase linearity. If no finite $H_1$ class exists, set $q_3,\ldots,q_8$
consistently to zero where undefined.

Concatenating the eight quantities for acceleration and gyroscope gives

$$
\Phi_{\mathrm{dyn}}(x)\in\mathbb R^{16}.
$$

This extension is evaluated separately. It is not concatenated with every
other representation merely to create more experiments.

## 8. Representations and explicit sensor-weight analyses

The classification comparison contains these feature sets:

1. **Instantaneous TDA:** stable ranks from $m=1,\ell=1$.
2. **Delay TDA:** stable ranks from $m=3,\ell=15$.
3. **Shuffled-delay TDA:** the temporal-order control.
4. **Delay TDA + scale:** the 256 stable-rank coordinates plus two RMS radii.
5. **Topological dynamics:** the compact 16-dimensional extension.
6. **Joint-sensor TDA:** one weighted metric on synchronized, separately
   normalized acceleration and gyroscope delay clouds. For acceleration weight
   $w$, concatenate $\sqrt{w}\,\widetilde X^{(a)}$ and
   $\sqrt{1-w}\,\widetilde X^{(g)}$. This gives
   $d_w^2=w d_a^2+(1-w)d_g^2$ and a 128-dimensional joint $H_0,H_1$
   stable-rank vector.
7. **Hybrid TDA:** concatenate the original 256 separate-sensor stable ranks
   with the 128 joint-sensor stable ranks, giving
   $[\Phi_{\mathrm{SR}},\Phi_{\mathrm{joint}}^{(w)}]\in\mathbb R^{384}$.
8. **Hybrid TDA + scale:** append the acceleration and gyroscope RMS radii to
   the selected hybrid vector, giving 386 coordinates. The hybrid weight is not
   reselected after adding scale.

For joint-sensor TDA only, evaluate $w=0,0.1,\ldots,1$ using the same grouped
development folds. Select the highest development balanced accuracy, breaking
exact ties toward $w=0.5$, and evaluate only that selected weight on the
official test set. Apply the same development-only weight selection separately
to the hybrid representation. Both weight curves and selection rules are
reported explicitly; the official test labels do not influence either choice.

The supplied 561-dimensional UCI feature vector is reported once as a
non-topological reference baseline, not as part of the main TDA claim.

Every representation uses the same fixed model:

$$
\operatorname{StandardScaler}
\longrightarrow
\operatorname{RBF\ SVM}(C=10,\gamma=0.001,
\text{class-weight=balanced}).
$$

These parameters are frozen from the earlier development analysis. There is no
nested hyperparameter search in the rewrite and no representation-specific
tuning. This substantially reduces computation and makes the comparison easy
to audit.

Development performance is obtained from five stratified subject-grouped
folds with seed 2956. The model is then fitted once to all development subjects
and evaluated once on the official disjoint test subjects.

## 9. Structural analysis without another classifier

For every development subject $p$ and activity $c$, compute the prototype

$$
\mu_{p,c}=\frac{1}{N_{p,c}}
\sum_{i:\,p_i=p,\,y_i=c}\Phi_{\mathrm{SR}}(x_i).
$$

Report the $6\times6$ matrix whose $(c,c')$ entry is the mean
$d_{\mathrm{SR}}(\mu_{p,c},\mu_{q,c'})$ over distinct subjects. This asks
whether activity structure is visible across people without fitting another
predictive model.

Average-linkage clustering of the six activity means is included as a direct
course-style structural visualization. It is descriptive and is not reported
as a new accuracy result.

## 10. Robustness claim

Use one bounded perturbation experiment rather than a large sensitivity grid.
Select a deterministic balanced sample of development windows. Add independent
Gaussian noise at 5% of each window/channel standard deviation, recompute delay
topology, and report the distribution of stable-rank displacement.

This finite experiment illustrates the course stability mechanism: nearby
maps induce Rips maps that agree up to a filtration shift. It is not presented
as a proof of a stability theorem.

## 11. Software architecture

The implementation must remain small:

- `data.py`: download and load the supplied UCI HAR data;
- `tda.py`: delay clouds, persistence, stable ranks, circular coordinates;
- `evaluate.py`: the single subject-grouped evaluation protocol;
- `run_analysis.py`: explicitly compute the representations, select the joint
  sensor weight on grouped development folds, and save numerical results;
- `test_math.py`: focused mathematical and data-contract tests;
- `build_notebook.py`: created only after all preceding code runs successfully.

No experiment framework, plugin system, model serialization, class hierarchy,
configuration package, or general-purpose cache abstraction is permitted.
NumPy arrays are cached with direct `np.savez_compressed` calls in
`run_analysis.py`. `ripser`, `stablebear`, SciPy, and scikit-learn perform the
specialized work.

The handwritten Python implementation, excluding tests and the eventual
notebook builder, should stay below 650 nonblank lines. If it exceeds this,
scope must be removed before abstraction is added.

## 12. Execution order and acceptance criteria

Development proceeds in this order:

1. unit-test the mathematical primitives on synthetic data;
2. run a small real-data smoke analysis;
3. run the complete feature extraction and evaluation scripts;
4. verify all arrays are finite and every subject split is disjoint;
5. compare the qualitative ordering with the earlier results;
6. only then generate a linear explanatory notebook from the tested functions.

The rewrite is accepted only when:

- a clean environment can run the tests with one documented command;
- the smoke analysis completes in under two minutes;
- the complete pipeline resumes from direct numerical caches;
- the notebook is generated from already tested code and runs top to bottom;
- no scientific computation exists only inside the notebook;
- the report is not rebuilt until the user approves the working analysis;
- nothing is committed or pushed without explicit approval.

## 13. Earlier findings retained as sanity checks

The earlier implementation reported the following balanced accuracies on the
official test subjects:

- instantaneous TDA: 0.558;
- delay TDA: 0.580;
- shuffled-delay TDA: 0.503;
- delay TDA plus scale: 0.640;
- topological dynamics: 0.611;
- supplied UCI features: 0.954.

Because the rewrite freezes one model across all representations instead of
performing representation-specific model selection, exact equality is not
required. The expected qualitative checks are:

1. delay topology should outperform shuffled-delay topology;
2. restoring physical scale should improve normalized delay topology;
3. compact dynamics should be competitive with sampled stable ranks;
4. the engineered UCI baseline should remain substantially stronger.

Failure of one of these checks is investigated and reported; it is not hidden
by adding new parameter searches.
