# SF2956 Human Activity TDA - Project Protocol

## Research question

The analysis separates one broad question into five testable ones:

1. Do stable ranks of delay-embedded sensor signals distinguish activities for unseen subjects?
2. Is any separation caused by temporal order?
3. Can a persistent loop be converted into an interpretable circular motion coordinate for all six activities?
4. Do the resulting topology-and-phase features improve classification, alone or together with stable ranks?
5. Do Euclidean and max-product sensor filtrations contain synchronized acceleration--gyroscope information absent from the separate clouds?

## Data contract

- A **subject** is one of the 30 people in the experiment.
- An **observation** is one 2.56-second window from one subject.
- Each observation contains 128 samples recorded at 50 Hz and one activity label.
- Consecutive windows overlap by 50%, so windows from the same subject are dependent.
- The subject, not the window, is therefore the unit used for train/test splitting and uncertainty estimates.
- The seven primary representations were fixed before the official test split was evaluated. The circular-coordinate branch and later test-set diagnostics are reported as post-hoc.

### Verified split audit

| Split | Windows | Subjects | Walking | Upstairs | Downstairs | Sitting | Standing | Lying |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training | 7,352 | 21 | 1,226 | 1,073 | 986 | 1,286 | 1,374 | 1,407 |
| Test | 2,947 | 9 | 496 | 471 | 420 | 491 | 532 | 537 |

The training and test subject sets are disjoint, and every subject contains all six activities. The class counts are not equal, so later classification results must not be summarized by raw accuracy alone.

For one observation $d$, the two sensor views are

\[
a_d(t_i),g_d(t_i)\in\mathbb{R}^3,\qquad i=1,\ldots,128,
\]

where the coordinates are the three body-acceleration and body-angular-velocity axes.

## Frozen geometric representation

For either sensor view $x_d$, the primary point cloud is the multivariate delay embedding

\[
X_d^{m,\ell}=\{(x_d(t_i),x_d(t_{i+\ell}),\ldots,x_d(t_{i+(m-1)\ell}))\}.
\]

The training-set mean acceleration autocorrelation fixes $\ell=15$ samples, and the primary embedding uses $m=3$. One point therefore contains the triaxial measurements at times $i$, $i+15$, and $i+30$: a 0.60-second motion fragment in $\mathbb{R}^9$. Each sensor cloud contains 98 such points.

The cloud is centered and divided by its RMS radius before computing Euclidean Vietoris--Rips persistence in $H_0$ and $H_1$. Stable ranks are sampled at 64 lifetime thresholds in $[0,2]$. The stored vector is

\[
[\text{acceleration }H_0\mid\text{acceleration }H_1\mid
  \text{gyroscope }H_0\mid\text{gyroscope }H_1].
\]

Distances between these vectors approximate the truncated function distance $L^1_{[0,2]}$. In the primary cache, all $H_1$ curves have reached zero at the right endpoint, and 99.7% of windows have zero in both $H_0$ blocks there.

This follows the course pipeline directly: Points 2--3 define a finite representing set $X_d$ for each datum; Points 14--15 explain why a (pseudo)metric is the input to stable constructions; Points 158--160 give the bar decomposition of a tame parameterized vector space; Points 187, 192, and 193 define the tame nested Vietoris--Rips filtration, while Point 194 shows that the full family retains the distance; Points 200, 203, and 207 define homology and interpret $H_0$ as connected components. Stable rank is the subsequent barcode vectorization and is cited separately in the report.

## Prespecified comparisons

Development performance is estimated with five outer subject-grouped folds. Inside every outer training fold, four further subject-grouped folds choose the SVM parameters. A separate five-fold search on all 21 development subjects selects the model fitted before the official test prediction. We compare:

1. compact conventional signal summaries;
2. the supplied 561 UCI features;
3. instantaneous unordered topology;
4. normalized delay topology;
5. delay topology with the two removed cloud scales restored;
6. time-shuffled delay topology;
7. delay topology concatenated with compact summaries.

The primary metric is balanced accuracy; macro-F1 is secondary. Confidence intervals bootstrap subjects, not overlapping windows. The same subject folds are used for every representation.

## Exact control for unordered clouds

Let $A\in\mathbb{R}^{128\times 3}$ contain one signal window row by row, and let $P$ be a permutation matrix. Row shuffling produces $A'=PA$. If $D$ is the Euclidean pairwise-distance matrix, then

\[
D'=PDP^{\mathsf T}.
\]

The equality holds because permuting rows only permutes both indices of every pairwise distance. The two Vietoris--Rips filtrations therefore differ only by vertex names, so their persistence barcodes agree.

The automated test suite verifies the claim using one fixed random permutation:

1. compute the $H_0$ and $H_1$ diagrams of one window;
2. shuffle only its rows;
3. compute the diagrams again;
4. compare the sorted birth-death pairs numerically.

This is the exact negative control for the unordered representation. The test passes.

The Euclidean delay construction has a different exact invariance. Reversing a whole window reverses the order of the $m$ three-coordinate blocks in each delay vector and relabels the 98 points. Because coordinate permutations preserve Euclidean distances, the persistence diagrams are unchanged. Delay topology therefore encodes local temporal adjacency but cannot identify the arrow of time. This claim is also verified by the test suite.

## Post-hoc all-activity circular-coordinate branch

For every normalized acceleration and gyroscope delay cloud, the branch records six persistence-diagram summaries: mean and final $H_0$ merge scale, finite $H_1$ count, longest $H_1$ lifetime, total $H_1$ persistence, and the birth of the longest $H_1$ class. It then takes the corresponding persistent cocycle $\alpha$ over $\mathbb F_{47}$ and computes a smoothed circular coordinate

\[
\widehat f=\operatorname*{arg\,min}_f\|\alpha+\delta f\|_2^2,
\qquad
\theta_i=\widehat f_i\bmod 1.
\]

After ordering the delay points by their original time index, a linear fit to the unwrapped $2\pi\theta_i$ gives circular frequency and $R^2$ phase linearity. The identical eight-feature map is used for each sensor, producing a 16-dimensional vector for every window and every activity. The branch compares diagram summaries alone, phase alone, all 16 features, and the 16 features appended to the 256 stable-rank coordinates. It repeats nested subject-grouped model selection and evaluates the official subject-disjoint split, but its results remain exploratory because the branch was designed after the primary analysis.

## Post-hoc joint sensor representation

The primary representation computes acceleration and gyroscope persistence separately. That keeps their physical units apart, but it cannot retain which acceleration history occurred at the same time as which gyroscope history. The post-hoc extension normalizes the two delay clouds separately and defines

\[
q_i^{(\lambda)}=
\left(\sqrt{\lambda}\,\widetilde\Phi_i^a,
\sqrt{1-\lambda}\,\widetilde\Phi_i^g\right)\in\mathbb R^{18}.
\]

Consequently,

\[
\|q_i^{(\lambda)}-q_j^{(\lambda)}\|_2^2
=\lambda\|\widetilde\Phi_i^a-\widetilde\Phi_j^a\|_2^2
+(1-\lambda)\|\widetilde\Phi_i^g-\widetilde\Phi_j^g\|_2^2.
\]

At $m=3$ and $\ell=15$, one window becomes a $98\times18$ joint cloud. Equal sensor weight uses $\lambda=0.5$. Its $H_0$ and $H_1$ stable ranks form a 128-dimensional vector; combining this with the 256 separate-sensor coordinates gives 384 features. We compare $\lambda\in\{0.25,0.50,0.75\}$ using grouped development folds and the primary frozen SVM settings. The nine official test subjects are not used.

### Exact marginal-preserving synchronization control

For each window, let $\pi$ be a seeded derangement of the 98 delay-point indices and define

\[
q_i^{\mathrm{broken}}=
\left(\sqrt{\lambda}\,\widetilde\Phi_i^a,
\sqrt{1-\lambda}\,\widetilde\Phi_{\pi(i)}^g\right).
\]

The acceleration block is unchanged and the gyroscope block is only row-permuted. Thus both marginal point clouds, their pairwise distances, and their separate persistence diagrams are exactly preserved. What changes is which gyroscope distance is combined with each acceleration distance in the joint metric. Seed 2956 is the declared paired comparison; seeds 2957--2960 check sensitivity to the arbitrary derangement. The test suite verifies the marginal-set equality directly.

### Max-coupled filtration

The Euclidean product allows one sensor distance to compensate for the other. The second coupled construction instead defines

\[
d_{\lambda}^{\max}(i,j)
=\max\left\{\sqrt{2\lambda}\,d_a(i,j),
\sqrt{2(1-\lambda)}\,d_g(i,j)\right\},
\]

where $d_a$ and $d_g$ are Euclidean distances in the separately normalized delay clouds. This is a metric. At $\lambda=1/2$,

\[
d_{1/2}^{\max}(i,j)\leq t
\quad\Longleftrightarrow\quad
d_a(i,j)\leq t\text{ and }d_g(i,j)\leq t.
\]

Thus every Vietoris--Rips edge requires both sensor histories to be close. The two marginal thresholds define a bifiltration; the three weights $\lambda\in\{0.25,0.50,0.75\}$ select one-parameter slices that can be processed by Ripser and StableBear. The same five marginal-preserving derangements are used as synchronization controls.

## Final evidence

Official-test balanced accuracy is 0.580 for normalized delay topology, compared with 0.558 for instantaneous topology and 0.503 after destroying temporal order. The paired subject-bootstrap interval for delay minus shuffled topology is $[0.037,0.108]$, and the exact two-sided sign-flip value over the nine subject effects is $p=0.0117$. For delay minus instantaneous topology, the corresponding interval includes zero and the sign-flip value is $p=0.125$. Thus the Takens-type construction captures temporal adjacency that generalizes across subjects, while evidence against every unordered topological representation is weaker.

However, compact summaries reach 0.852 and the supplied UCI features reach 0.954. Adding scale improves delay topology to 0.640, while concatenating topology with compact summaries reaches only 0.794. The supported conclusion is therefore interpretive rather than competitive: stable-rank topology detects reproducible motion geometry but is not the strongest classifier for this dataset.

The all-activity circular-coordinate branch supplies the main interpretive extension. Gyroscope circular frequency has partial $\eta^2=0.844$ for activity after removing subject means, followed by acceleration circular linearity ($0.787$) and acceleration longest-loop lifetime ($0.754$). Diagram summaries alone score 0.577 on the official test, circular phase alone 0.418, and their 16-feature combination 0.611. The latter exceeds the original Delay TDA score by a mean 0.031 across test subjects, with paired subject-bootstrap interval $[0.020,0.040]$. Appending the 256 stable-rank coordinates lowers the score to 0.593; combined minus 16-feature dynamics has interval $[-0.037,0.000]$. The concise vector is therefore the strongest tested topology-only representation, while ordinary features remain clearly better.

Parameter, cityblock-metric, and Gaussian-noise checks leave this conclusion unchanged. Full details are in `report/report.pdf`; machine-readable outputs are in `results/`.

Post-hoc diagnostics further show that both sensors and both homology degrees contribute. Average-linkage clustering of subject-balanced stable-rank prototypes recovers separate static and dynamic activity families. A grouped nearest-prototype classifier scores 0.476 at the subject-activity level, whereas 2,000 independent within-subject label permutations have mean 0.168, 95% interval $[0.095,0.246]$, and no value as large as observed ($p=0.0005$). These analyses are interpretive extensions rather than new confirmatory test-set claims.

The joint sensor analysis adds a distinct representation result. On the 21 development subjects, separate topology scores 0.533 balanced accuracy, joint topology 0.438, and their concatenation 0.560. The subject-level combined-minus-separate mean is 0.028 with paired bootstrap 95% interval $[0.012,0.044]$. Joint $H_0$ alone scores 0.321 and joint $H_1$ alone 0.379. Acceleration weights 0.25, 0.50, and 0.75 score 0.416, 0.438, and 0.456.

The synchronization control limits that interpretation. Joint topology after broken pairing scores 0.417--0.427 across five seeds. Correct minus broken pairing has subject-level mean 0.022 and interval $[-0.006,0.049]$ for seed 2956. Separate plus broken-pairing topology scores 0.563--0.576, compared with 0.560 for separate plus correctly paired topology. For seed 2956, correct minus broken combined topology has mean $-0.005$ and interval $[-0.021,0.011]$. The supported conclusion is therefore narrower: the joint product metric provides a complementary transformed view, but these experiments do not show that correct acceleration--gyroscope synchronization is responsible for the improvement.

The max-coupled result reaches 0.425 alone and 0.557 when appended to separate topology. Its combined-minus-separate subject mean is 0.025 with interval $[0.010,0.039]$. Broken max coupling scores 0.398--0.406 alone and 0.571--0.576 after concatenation. For seed 2956, correct minus broken has mean 0.027 with interval $[-0.006,0.058]$ alone and $-0.014$ with interval $[-0.029,0.001]$ after concatenation. Requiring both sensor thresholds therefore does not rescue the synchronization claim; it provides a second, course-driven product-metric analysis with the same scientific conclusion.
