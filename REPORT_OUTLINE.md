# Report argument map

The paper should be readable as four questions, not as a tour through TDA terminology. Exact design decisions are recorded in [PROJECT_PROTOCOL.md](PROJECT_PROTOCOL.md).

## Question 1: do the stable ranks contain activity information?

The signal-to-topology construction must be concrete enough to reconstruct from the text:

- one observation is a 128-by-3 window from one subject;
- one delay point joins samples $i$, $i+15$, and $i+30$, so it is a 0.60-second fragment in $\mathbb R^9$;
- the 98 points receive Euclidean distance;
- Vietoris--Rips persistence gives $H_0$ and $H_1$ bars;
- stable rank counts bars whose lifetime exceeds $t$;
- the final vector contains acceleration $H_0,H_1$ followed by gyroscope $H_0,H_1$.

Evidence: nested subject-wise CV, official subject-disjoint evaluation, cross-subject prototype distances, unsupervised activity clustering, and the subject-wise permutation test.

## Question 2: is the information temporal?

The instantaneous cloud is exactly invariant to sample order. The delay cloud is not: shuffling changes which measurements become one point. This is the reason for the two controls, not merely another baseline table.

Evidence: exact permutation invariance for the unordered cloud, paired delay-minus-shuffle performance by test subject, and an exact sign-flip test over the nine paired subject effects. Full time reversal is a separate exact invariance of Euclidean delay persistence, so the method does not recover the arrow of time.

## Question 3: does TDA improve recognition?

Normalization deliberately removes amplitude. Restoring the two RMS radii tests its cost. The compact 66-feature baseline tests whether topology contributes more than elementary signal summaries; the supplied 561 features show the dataset's established prediction level.

Evidence: delay + scale, compact summaries, UCI 561, and delay + summaries. The result is negative and should be stated directly.

## Question 4: do coupled sensor filtrations add information?

The separate acceleration and gyroscope barcodes contain only marginal sensor geometry. The Euclidean-joint cloud averages their squared distances. The max-product metric instead requires proximity in both sensors: at equal weight, $d_{\max}(i,j)\leq t$ exactly when $d_a(i,j)\leq t$ and $d_g(i,j)\leq t$. This is a one-parameter slice of a two-threshold sensor bifiltration.

Evidence: grouped development scores for separate, Euclidean-joint, max-coupled, and concatenated topology; paired subject-bootstrap intervals; sensitivity to acceleration weights 0.25, 0.50, and 0.75; and synchronization controls that derange gyroscope point pairing while preserving both marginal clouds exactly. This analysis is explicitly post-hoc and does not reuse the official test subjects.

## Theory that belongs in the paper

- Course notes Points 2--3: data as finite representing sets and the analysis pipeline.
- Points 14--15: metric and pseudometric spaces and their stability role.
- Points 158--160: existence and uniqueness of the bar decomposition for tame parameterized vector spaces.
- Points 187, 192, and 194: Vietoris--Rips complexes, nesting, and recovery of the finite metric.
- Points 200, 203, and 207: homology and connected components.
- Takens: motivation for delay coordinates, with no claim that its theorem applies literally to these short windows.
- Chachólski--Riihimäki: stable rank, cited separately because it is not defined in the supplied course-note PDF.

## Claims to avoid

- Do not call an $H_1$ class a literal circular body movement. It is a loop in the reconstructed motion cloud.
- Do not claim that TDA improves classification; it does not beat either conventional baseline.
- Do not call later diagnostics confirmatory after the test results were opened.
- Do not call the numerical distance the full $L^1$ distance. It is the integral approximation on $[0,2]$.
- Do not interpret a smooth noise curve as a proof of a persistence-stability theorem.
- Do not attribute the combined-feature gain to sensor synchronization: broken pairing performs equally well when appended to the separate features.

## Final one-sentence answer

Delay-embedded stable ranks contain subject-independent and time-order-dependent activity structure; Euclidean and max-product sensor filtrations add complementary transformations, but marginal-preserving controls find no evidence that correct cross-sensor synchronization causes that gain, and the topological vectorizations remain weaker than ordinary signal features.
