"""Generate the explanatory notebook after the tested analysis has completed."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).parent


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(r"""
# Topological signatures of human activity

**Question.** Can the topology of short acceleration and gyroscope trajectories
identify activities performed by people who were not used for training?

The argument is developed in one direction:

1. treat each labelled sensor window as a sampled path;
2. turn the path into a normalized delay-coordinate point cloud;
3. turn persistent homology into fixed-length stable-rank vectors;
4. select the fusion weights using grouped development folds;
5. compare every representation on subjects excluded from model fitting.

All numerical results come from the tested functions in the project. This
notebook explains and visualizes that analysis; it does not contain a second
implementation.
"""),
    markdown(r"""
## 0. Load the fixed analysis

The UCI development and official test splits contain disjoint subjects. The
saved arrays below were produced by `run_analysis.py`.
"""),
    code("""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser

from data import ACTIVITIES, load_split
from tda import (
    GRID,
    delay_cloud,
    joint_delay_cloud,
    mean_autocorrelation,
    normalize_cloud,
    stable_rank,
)

ROOT = Path.cwd()
if not (ROOT / "ANALYSIS_SPEC.md").exists():
    raise RuntimeError("Start Jupyter from the project directory")
if not (ROOT / "output/results.json").exists():
    raise RuntimeError("Run `.venv/bin/python run_analysis.py` before this notebook")

results = json.loads((ROOT / "output/results.json").read_text())
cache = ROOT / "output/cache"
delay = np.load(cache / "delay_m3_lag15_f47.npz")
instantaneous = np.load(cache / "instantaneous_m1_lag1_f47.npz")
shuffled = np.load(cache / "shuffled_m3_lag15_seed2956_f47.npz")
train, labels, subjects = load_split("train")
test, test_labels, test_subjects = load_split("test")

print("Development windows:", len(labels))
print("Official test windows:", len(test_labels))
print("Subjects are disjoint:", set(subjects).isdisjoint(test_subjects))
"""),
    markdown(r"""
## 1. One observation is a sampled path

For sensor $s\in\{a,g\}$, one observation is

$$
x^{(s)}:\{0,\ldots,127\}\to\mathbb R^3.
$$

Each observation contains synchronized acceleration and angular-velocity
paths. The 128 samples are recorded at 50 Hz, so one labelled window lasts
2.56 seconds. A window—not an individual time sample—is classified.
"""),
    code("""
index = np.flatnonzero((subjects == 1) & (labels == 1))[0]
time = np.arange(128) / 50
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharex=True)
for axis, sensor in zip(axes, ("body_acc", "body_gyro")):
    axis.plot(time, train[sensor][index])
    axis.set_title(sensor.replace("_", " ").title())
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Sensor value")
fig.suptitle("One walking window from development subject 1")
fig.tight_layout()
"""),
    markdown(r"""
## 2. From a time series to a normalized point cloud

### 2.1 Choose the lag from temporal dependence

For embedding dimension $m$ and lag $\ell$,

$$
E_{m,\ell}(x)_t=(x_t,x_{t+\ell},\ldots,x_{t+(m-1)\ell})
\in\mathbb R^{3m}.
$$

We use $m=3$. The lag is fixed at the first non-positive mean acceleration
autocorrelation on development data, rather than selected by classification
performance. The result is $\ell=15$ samples, or 0.30 seconds. Each sensor path
then becomes $128-2(15)=98$ points in $\mathbb R^9$.
"""),
    code("""
autocorrelation = mean_autocorrelation(train["body_acc"], max_lag=40)
chosen_lag = int(np.flatnonzero(autocorrelation[1:] <= 0)[0] + 1)
plt.figure(figsize=(7, 3.5))
plt.plot(np.arange(41), autocorrelation, marker="o", markersize=3)
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(chosen_lag, color="tab:red", linestyle="--", label=f"lag = {chosen_lag}")
plt.xlabel("Lag (samples)")
plt.ylabel("Mean normalized autocorrelation")
plt.legend()
plt.tight_layout()
print("Chosen lag:", chosen_lag)
"""),
    markdown(r"""
### 2.2 Center and normalize each sensor cloud

For a delay cloud $X=\{z_i\}_{i=1}^n$, define

$$
r(X)=\left(\frac1n\sum_i\lVert z_i-\bar z\rVert_2^2\right)^{1/2},
\qquad
\widetilde X=\left\{\frac{z_i-\bar z}{r(X)}\right\}.
$$

This separates normalized **shape** from physical **scale**. Persistence uses
$\widetilde X$; the discarded acceleration and gyro radii are saved as two
explicit scale features.
"""),
    code("""
raw_cloud = delay_cloud(train["body_acc"][index], m=3, lag=15)
cloud, radius = normalize_cloud(raw_cloud)
fig = plt.figure(figsize=(6, 5))
axis = fig.add_subplot(projection="3d")
axis.plot(cloud[:, 0], cloud[:, 1], cloud[:, 2], marker="o", markersize=2)
axis.set_title("First three coordinates of one normalized delay cloud")
axis.set_xlabel("coordinate 1")
axis.set_ylabel("coordinate 2")
axis.set_zlabel("coordinate 3")
print("Cloud shape:", cloud.shape, "raw RMS radius:", round(radius, 4))
"""),
    markdown(r"""
### 2.3 Construct the weighted joint cloud

The alternative construction joins acceleration and gyroscope at the beginning,
before persistence. For the same time index $t$, first construct the two delay
vectors

$$
A_t=(a_t,a_{t+15},a_{t+30})\in\mathbb R^9,
\qquad
G_t=(g_t,g_{t+15},g_{t+30})\in\mathbb R^9.
$$

Normalize the two clouds separately:

$$
\widetilde A_t=\frac{A_t-\bar A}{r_A},
\qquad
\widetilde G_t=\frac{G_t-\bar G}{r_G}.
$$

For acceleration weight $w\in[0,1]$, join synchronized points as

$$
Z_t^{(w)}=
[\sqrt{w}\,\widetilde A_t,\sqrt{1-w}\,\widetilde G_t]
\in\mathbb R^{18}.
$$

The ordinary Euclidean distance between these joined points is therefore

$$
d_w(i,j)=\sqrt{
w\lVert\widetilde A_i-\widetilde A_j\rVert_2^2+
(1-w)\lVert\widetilde G_i-\widetilde G_j\rVert_2^2}.
$$

Thus $w=0$ is gyro only, $w=1$ is acceleration only, and $w=0.5$ gives
equal weighting. Because both sensor clouds have RMS radius one and the two
weights sum to one, the joined cloud also has RMS radius one for every $w$.
"""),
    code("""
equal_joint_cloud, joint_radii = joint_delay_cloud(
    train["body_acc"][index],
    train["body_gyro"][index],
    acceleration_weight=0.5,
    m=3,
    lag=15,
)
print("Separate acceleration cloud:", cloud.shape)
print("Joint acceleration-gyroscope cloud:", equal_joint_cloud.shape)
print("Acceleration and gyroscope radii:", np.round(joint_radii, 4))

acceleration_cloud, _ = normalize_cloud(
    delay_cloud(train["body_acc"][index], m=3, lag=15)
)
gyroscope_cloud, _ = normalize_cloud(
    delay_cloud(train["body_gyro"][index], m=3, lag=15)
)
acceleration_distance = np.linalg.norm(acceleration_cloud[0] - acceleration_cloud[1])
gyroscope_distance = np.linalg.norm(gyroscope_cloud[0] - gyroscope_cloud[1])
formula_distance = np.sqrt(
    0.5 * acceleration_distance**2 + 0.5 * gyroscope_distance**2
)
computed_distance = np.linalg.norm(equal_joint_cloud[0] - equal_joint_cloud[1])
print("Example distance from formula:", round(formula_distance, 8))
print("Example distance in joined cloud:", round(computed_distance, 8))
"""),
    markdown(r"""
## 3. From point clouds to classification vectors

### 3.1 Persistence and the original separate-sensor vector

For the normalized finite metric space $(X,d)$,

$$
\operatorname{VR}_\epsilon(X)=
\{\sigma\subseteq X:d(x,y)\leq\epsilon\text{ for every }x,y\in\sigma\}.
$$

We compute $H_0$ and $H_1$ persistence over $\mathbb F_{47}$. The field
is shared with the circular cocycle extension, allowing one persistence pass
to produce both representations.

For finite diagram $D_k$, the lifetime stable rank is

$$
\rho_k(u)=\#\{(b,d)\in D_k:d-b>u\}.
$$

It is evaluated at 64 fixed thresholds on $[0,2]$. Keeping the two sensor
topologies separate gives

$$
\Phi_{\mathrm{SR}}=
[\rho_0^{(a)},\rho_1^{(a)},\rho_0^{(g)},\rho_1^{(g)}]
\in\mathbb R^{256}.
$$

This is the original delay-TDA representation.
"""),
    code("""
diagram = ripser(cloud, maxdim=1, coeff=47)["dgms"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for dimension, axis in enumerate(axes):
    finite = diagram[dimension][np.isfinite(diagram[dimension][:, 1])]
    limit = max(1.0, finite.max() * 1.05) if len(finite) else 1.0
    axis.scatter(finite[:, 0], finite[:, 1], s=18)
    axis.plot((0, limit), (0, limit), color="black", linewidth=0.8)
    axis.set(title=f"$H_{dimension}$ persistence", xlabel="Birth", ylabel="Death")
fig.tight_layout()
"""),
    code("""
ranks = delay["train_ranks"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
titles = ("Acceleration H0", "Acceleration H1", "Gyroscope H0", "Gyroscope H1")
for block, (axis, title) in enumerate(zip(axes.flat, titles)):
    start = block * len(GRID)
    for activity, name in ACTIVITIES.items():
        axis.plot(GRID, ranks[labels == activity, start:start + len(GRID)].mean(axis=0),
                  label=name)
    axis.set_title(title)
    axis.set_ylabel("Mean stable rank")
    axis.set_xlabel("Lifetime threshold")
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
"""),
    markdown(r"""
### 3.2 Joint-sensor persistence

The joint cloud has one persistence diagram for $H_0$ and one for $H_1$.
For each $w$, calculate

$$
(D_0^{(w)},D_1^{(w)})=
\operatorname{PH}_{0,1}(\operatorname{VR}(Z^{(w)},d_w)).
$$

Turn each diagram into a stable rank

$$
\rho_k^{(w)}(u)=
\#\{(b,d)\in D_k^{(w)}:d-b>u\},
$$

and evaluate it at the same 64 thresholds $u_0,\ldots,u_{63}$. The final
classification vector is

$$
\Phi_{\mathrm{joint}}^{(w)}=
[\rho_0^{(w)}(u_0),\ldots,\rho_0^{(w)}(u_{63}),
  \rho_1^{(w)}(u_0),\ldots,\rho_1^{(w)}(u_{63})]
\in\mathbb R^{128}.
$$

This is distinct from the existing 256-dimensional vector formed by
concatenating four separately computed sensor/homology blocks.
"""),
    code("""
joint_diagrams = ripser(equal_joint_cloud, maxdim=1, coeff=47)["dgms"]
equal_joint_vector = np.concatenate([
    stable_rank(joint_diagrams[0]),
    stable_rank(joint_diagrams[1]),
])
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for dimension, axis in enumerate(axes):
    finite = joint_diagrams[dimension][
        np.isfinite(joint_diagrams[dimension][:, 1])
    ]
    limit = max(1.0, finite.max() * 1.05) if len(finite) else 1.0
    axis.scatter(finite[:, 0], finite[:, 1], s=18)
    axis.plot((0, limit), (0, limit), color="black", linewidth=0.8)
    axis.set(title=f"Joint H{dimension} persistence", xlabel="Birth", ylabel="Death")
fig.tight_layout()
print("Joint stable-rank vector shape:", equal_joint_vector.shape)
"""),
    markdown(r"""
### 3.3 Preserve separate topology and append joint topology

The hybrid alternative retains the original separate acceleration and
gyroscope topology and appends the joint topology:

$$
\Phi_{\mathrm{hybrid}}^{(w)}=
[\underbrace{\Phi_{\mathrm{acc}}}_{128},
 \underbrace{\Phi_{\mathrm{gyro}}}_{128},
 \underbrace{\Phi_{\mathrm{joint}}^{(w)}}_{128}]
\in\mathbb R^{384}.
$$

The first 256 coordinates are identical for every $w$; only the final joint
block changes. At $w=0$ the last block duplicates gyro topology, and at $w=1$
it duplicates acceleration topology. Interior weights add a genuinely joint
topological description.
"""),
    code("""
equal_hybrid_vector = np.concatenate((
    delay["train_ranks"][index],
    equal_joint_vector,
))
print("Separate-sensor vector shape:", delay["train_ranks"][index].shape)
print("Equal-weight joint vector shape:", equal_joint_vector.shape)
print("Equal-weight hybrid vector shape:", equal_hybrid_vector.shape)
"""),
    markdown(r"""
### 3.4 Restore the two physical scales

Separate normalization deliberately removed the two sensor amplitudes. Add
them back explicitly, without changing the selected joint weight:

$$
\Phi_{\mathrm{hybrid+scale}}^{(w)}=
[\Phi_{\mathrm{hybrid}}^{(w)},r_A,r_G]\in\mathbb R^{386}.
$$

Here $r_A$ and $r_G$ are the RMS radii calculated before normalizing the
acceleration and gyroscope delay clouds. They tell the classifier how large the
motion was, while the first 384 coordinates describe normalized shape.
"""),
    code("""
equal_hybrid_scale_vector = np.concatenate((
    equal_hybrid_vector,
    delay["train_scales"][index],
))
print("Hybrid shape coordinates:", equal_hybrid_vector.shape)
print("Appended sensor radii:", np.round(delay["train_scales"][index], 4))
print("Hybrid plus scale vector shape:", equal_hybrid_scale_vector.shape)
"""),
    markdown(r"""
## 4. Select weights, then classify unseen subjects

### 4.1 One evaluation protocol for every representation

The classifier is deliberately fixed for every representation:

$$
\operatorname{StandardScaler}\to
\operatorname{RBF\ SVM}(C=10,\gamma=0.001).
$$

Development estimates use five stratified subject-grouped folds. All windows
from one person stay in the same fold. The fixed SVM is fitted once more on all
21 development subjects and evaluated on the nine different official-test
subjects.

The shuffled control independently permutes the 128 samples of every window
before delay embedding. It preserves the multiset of sensor values while
destroying temporal adjacency.

The only representation search concerns the acceleration weight. On the fixed
grid $\mathcal W=\{0,0.1,\ldots,1\}$, select

$$
\widehat w_R=
\underset{w\in\mathcal W}{\operatorname{argmax}}
\;\operatorname{BA}_{\mathrm{grouped\ CV}}(\Phi_R^{(w)}),
\qquad R\in\{\mathrm{joint},\mathrm{hybrid}\}.
$$

The official test labels never enter this selection. Hybrid plus scale reuses
$\widehat w_{\mathrm{hybrid}}$ rather than starting another search.
"""),
    markdown(r"""
### 4.2 Joint topology by itself

This curve applies the rule above to the 128-dimensional joint vector. It asks
whether one fused topology is sufficient on its own.
"""),
    code("""
weight_analysis = results["joint_sensor_weight_search"]
weight_grid = weight_analysis["grid"]
weights = np.asarray([row["acceleration_weight"] for row in weight_grid])
weight_scores = np.asarray([row["balanced_accuracy"] for row in weight_grid])
selected_weight = weight_analysis["selected_acceleration_weight"]

plt.figure(figsize=(7, 3.8))
plt.plot(weights, weight_scores, marker="o")
plt.axvline(0.5, color="gray", linestyle=":", label="Equal weighting")
plt.scatter(
    [selected_weight],
    [weight_scores[np.flatnonzero(np.isclose(weights, selected_weight))[0]]],
    color="tab:red",
    zorder=3,
    label=f"Selected weight = {selected_weight:.1f}",
)
plt.xlabel("Acceleration weight (gyro weight = 1 - acceleration weight)")
plt.ylabel("Grouped development balanced accuracy")
plt.xticks(weights)
plt.legend()
plt.tight_layout()
print("Selected acceleration weight:", selected_weight)
print("Selected gyroscope weight:", weight_analysis["selected_gyroscope_weight"])
if np.isclose(selected_weight, 1.0):
    print("Result: acceleration-only was selected; joint fusion did not improve development accuracy.")
elif np.isclose(selected_weight, 0.0):
    print("Result: gyroscope-only was selected; joint fusion did not improve development accuracy.")
else:
    print("Result: a genuinely joint acceleration-gyroscope weighting was selected.")
"""),
    markdown(r"""
### 4.3 Joint topology appended to separate topology

This curve applies the same rule to the 384-dimensional hybrid vector. It asks
whether joint topology adds information after the two separate sensor
topologies are already present.
"""),
    code("""
hybrid_analysis = results["hybrid_sensor_weight_search"]
hybrid_grid = hybrid_analysis["grid"]
hybrid_weights = np.asarray([row["acceleration_weight"] for row in hybrid_grid])
hybrid_scores = np.asarray([row["balanced_accuracy"] for row in hybrid_grid])
selected_hybrid_weight = hybrid_analysis["selected_acceleration_weight"]

plt.figure(figsize=(7, 3.8))
plt.plot(hybrid_weights, hybrid_scores, marker="o")
plt.axvline(0.5, color="gray", linestyle=":", label="Equal-weight joint block")
selected_position = np.flatnonzero(
    np.isclose(hybrid_weights, selected_hybrid_weight)
)[0]
plt.scatter(
    [selected_hybrid_weight], [hybrid_scores[selected_position]],
    color="tab:red", zorder=3,
    label=f"Selected weight = {selected_hybrid_weight:.1f}",
)
plt.xlabel("Acceleration weight inside appended joint block")
plt.ylabel("Grouped development balanced accuracy")
plt.xticks(hybrid_weights)
plt.legend()
plt.tight_layout()
print("Selected hybrid acceleration weight:", selected_hybrid_weight)
print("Selected hybrid gyro weight:", hybrid_analysis["selected_gyroscope_weight"])
hybrid_model = results["models"]["Hybrid TDA"]
hybrid_scale_model = results["models"]["Hybrid TDA + scale"]
print(
    "Hybrid balanced accuracy (development, official test):",
    round(hybrid_model["development"]["balanced_accuracy"], 3),
    round(hybrid_model["official_test"]["balanced_accuracy"], 3),
)
print(
    "Hybrid plus scale balanced accuracy (development, official test):",
    round(hybrid_scale_model["development"]["balanced_accuracy"], 3),
    round(hybrid_scale_model["official_test"]["balanced_accuracy"], 3),
)
"""),
    markdown(r"""
### 4.4 Final subject-independent comparison

The table below summarizes exactly what enters the same fixed classifier.

| Representation | Information retained | Dimension |
|---|---|---:|
| Instantaneous TDA | unordered samples, separate sensors | 256 |
| Delay TDA | temporal shape, separate sensors | 256 |
| Joint-sensor TDA | one selected joint topology | 128 |
| Hybrid TDA | separate plus selected joint topology | 384 |
| Hybrid TDA + scale | hybrid shape plus two radii | 386 |
| Shuffled-delay TDA | time-destroyed control | 256 |
| Delay TDA + scale | separate shape plus two radii | 258 |
| Topological dynamics | compact persistence and circular motion | 16 |
| UCI baseline | supplied engineered features | 561 |
"""),
    code("""
names = list(results["models"])
development = [results["models"][name]["development"]["balanced_accuracy"] for name in names]
official = [results["models"][name]["official_test"]["balanced_accuracy"] for name in names]
x = np.arange(len(names))
plt.figure(figsize=(10, 4.5))
plt.bar(x - 0.18, development, 0.36, label="Grouped development")
plt.bar(x + 0.18, official, 0.36, label="Official test")
plt.axhline(1 / 6, color="black", linestyle=":", label="Chance")
plt.xticks(x, names, rotation=25, ha="right")
plt.ylabel("Balanced accuracy")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
for name in names:
    score = results["models"][name]["official_test"]
    print(f"{name:25s} BA={score['balanced_accuracy']:.3f}  F1={score['macro_f1']:.3f}")
"""),
    markdown(r"""
Read the comparisons in order:

1. Delay TDA exceeds shuffled-delay TDA: temporal adjacency contributes.
2. Joint topology alone performs poorly and selects acceleration only.
3. Appending a genuinely joint $0.4/0.6$ block improves separate delay TDA.
4. Restoring the two radii gives the strongest TDA result, 0.663 test balanced
   accuracy.
5. The 0.954 engineered UCI baseline remains much stronger; topology is an
   interpretable motion representation, not the best available classifier.
"""),
    markdown(r"""
## 5. Cross-subject activity geometry

For subject $p$ and activity $c$, let

$$
\mu_{p,c}=\frac1{N_{p,c}}\sum_{i:p_i=p,y_i=c}\Phi_{\mathrm{SR}}(x_i).
$$

The matrix below averages blockwise stable-rank $L^1$ distances between
subject-activity prototypes. It describes structure without fitting another
classifier.
"""),
    code("""
matrix = np.asarray(results["activity_distance_matrix"])
plt.figure(figsize=(6, 5))
plt.imshow(matrix, cmap="magma")
plt.colorbar(label="Mean stable-rank distance")
activity_names = list(ACTIVITIES.values())
plt.xticks(range(6), activity_names, rotation=35, ha="right")
plt.yticks(range(6), activity_names)
plt.tight_layout()
"""),
    markdown(r"""
## 6. Persistent loops as circular motion coordinates

For the longest $H_1$ bar $(b^*,d^*)$, use its Ripser cocycle on the Rips
graph at

$$
\epsilon^*=b^*+\tfrac12(d^*-b^*).
$$

After lifting the cocycle $\alpha$, anchor one vertex and solve

$$
\widehat\phi=\arg\min_{\phi:\phi_0=0}\lVert B\phi+\alpha\rVert_2^2,
\qquad \theta_i=\widehat\phi_i\bmod1.
$$

The slope of unwrapped $2\pi\theta_i$ gives circular frequency; regression
$R^2$ gives phase linearity. Together with six direct diagram summaries per
sensor this produces a compact 16-dimensional representation.
"""),
    code("""
dynamics = delay["train_dynamics"]
columns = (6, 7, 14, 15)
titles = ("Acceleration frequency", "Acceleration linearity",
          "Gyroscope frequency", "Gyroscope linearity")
means = np.asarray([
    [dynamics[labels == activity, column].mean() for activity in ACTIVITIES]
    for column in columns
])
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for axis, values, title in zip(axes.flat, means, titles):
    axis.bar(range(6), values)
    axis.set_title(title)
    axis.set_xticks(range(6), list(ACTIVITIES.values()), rotation=25, ha="right")
fig.tight_layout()
"""),
    markdown(r"""
## 7. Bounded perturbation check

On a deterministic balanced sample, independent Gaussian noise equal to 5%
of each window/channel standard deviation is added before recomputing delay
topology. This illustrates gradual displacement under perturbation; it is not
presented as a proof of the course stability theorem.
"""),
    code("""
noise = results["noise_5_percent"]
print("Windows:", noise["sample_size"])
print("Mean stable-rank displacement:", round(noise["mean_distance"], 3))
print("95% empirical interval:",
      (round(noise["q025"], 3), round(noise["q975"], 3)))
"""),
    markdown(r"""
## 8. Conclusion

The clean analysis supports a narrow result. Delay-cloud topology transfers
activity information across subjects and performs better than topology after
temporal shuffling. Normalization removes highly predictive physical scale,
which becomes visible when the two radii are restored. The compact
persistence-and-dynamics vector is interpretable and competitive, but the
engineered UCI representation remains substantially stronger.

The hybrid representation selects a genuinely joint block with acceleration
weight 0.4 and gyroscope weight 0.6. Appending that block raises official-test
balanced accuracy from 0.580 for separate delay TDA to 0.611 for hybrid TDA.
Appending the two physical scales raises it further to 0.663, compared with
0.639 for delay TDA plus scale. Hybrid TDA plus scale is therefore the strongest
topological representation in this comparison. Its development advantage over
delay TDA plus scale is only 0.005, however, so the incremental gain should be
treated as modest rather than definitive.

These conclusions use one fixed model, one subject-grouped development
protocol, and one official disjoint test. The joint and hybrid sensor weights
are explicit representation parameter searches and are selected without using
the official test labels.
"""),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3 (SF2956 TDA)",
    "language": "python",
    "name": "sf2956-tda",
}
notebook.metadata["language_info"] = {"name": "python", "version": "3.13"}
nbf.write(notebook, ROOT / "analysis.ipynb")
print("Wrote analysis.ipynb")
