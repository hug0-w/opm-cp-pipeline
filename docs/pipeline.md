# The pipeline, stage by stage

This document explains what the code does and why it is built the way it is. For how to
run it, see the [README](../README.md) and
[`examples/ers_erd_cross_subject_clustering.ipynb`](../examples/ers_erd_cross_subject_clustering.ipynb).

## The problem

A source-space time-frequency analysis of an MEG recording produces, per subject, a
three-dimensional array of power: **parcel × frequency × time**, normalised against a
pre-event baseline so that it reads as a log ratio — positive where power rose above
baseline, negative where it fell.

Two questions follow, and conventional analysis answers them awkwardly.

*What are the dominant patterns?* The usual approach picks a frequency band and a time
window in advance, averages within them, and tests parcels. That works only when you
already know what you are looking for, and it discards the fact that a real
event-related response is a *joint* pattern: particular parcels, at particular
frequencies, over a particular interval. A tensor decomposition recovers exactly that
joint structure without being told where to look.

*Is a pattern shared between subjects?* Each subject is decomposed separately, so the
components come out in an arbitrary order with no correspondence between subjects.
Establishing that correspondence — deciding which of subject 2's components, if any, is
"the same" as a given component of subject 1 — is most of what this code does.

## Stage 1 — reading the data (`io.py`)

The time-frequency estimation happens upstream in MATLAB; this package reads what it
wrote. The files are HDF5 (MATLAB v7.3), and h5py presents v7.3 arrays with their
dimensions reversed relative to MATLAB, so every array is transposed back on the way in.
That transpose is the only subtlety in the module, and getting it wrong produces arrays
of plausible shape with meaningless contents, so it is asserted against the axis vectors
stored alongside.

`load_tensor` returns three views of the same recording:

| key | shape | what it is |
|---|---|---|
| `pow_norm` | (ROI, Freq, Time) | trial-averaged, baseline-normalised power, `ln(power / baseline_mean)` — **the tensor that gets decomposed** |
| `pow_banded` | (ROI, Band, Time) | `pow_norm` averaged into six canonical bands |
| `pow_trials` | (Trial, ROI, Freq, Time) | raw single-trial power |

`pow_trials` is kept because any surrogate or permutation null has to be built at the
trial level. A CP decomposition is invariant to permuting indices *within* a mode, so
circularly shifting the time axis of the averaged tensor leaves the factors and R²
exactly unchanged — a degenerate null with p ≈ 1. A valid null shifts each trial's time
axis by its own random lag and re-averages, which really does produce a different tensor.

## Stage 2 — building the tensors (`tensors.py`)

Three steps, in order.

**ROI selection.** The atlas covers the whole brain including cerebellum and vermis,
which OPM arrays cover poorly. Restricting to a cortical subset stops the decomposition
spending components on parcels the sensors cannot see. `select_rois` matches on
case-insensitive substrings of the parcel names, so `'occipital'` keeps every occipital
parcel in both hemispheres.

**Time cropping.** The first and last few time points of a time-frequency estimate are
contaminated by the taper reaching past the edge of the epoch. Those edge artefacts are
large enough to attract a component of their own, so `n_crop` points are dropped from
each end.

**The ERS/ERD split.** `pow_norm` is signed. Non-negative CP needs non-negative input,
but the better reason for splitting is physiological: a synchronisation (power rising)
and a desynchronisation (power falling) in the same parcel are different phenomena, and
forcing them to share a component buys nothing. `split_ers_erd` returns the positive part
(**ERS**) and the magnitude of the negative part (**ERD**), so that `ers - erd == data`
exactly and no sample is counted twice. Each is then decomposed independently, and every
later stage runs once per type.

The alternative is to leave the tensor signed and constrain only *some* modes to be
non-negative — `decomposition.run_cp_semi_nn`, with the time mode left free so one
component can carry a decrease followed by a rebound. That is a different analysis, not a
step in this one.

## Stage 3 — the decomposition (`decomposition.py`)

A rank-R CP (PARAFAC) model writes the tensor as a sum of R rank-one terms, each the
outer product of one ROI loading vector, one frequency profile and one time course. With
every factor non-negative the terms are additive and read directly as *this much power,
in these parcels, at these frequencies, at these times*.

Two facts about the fit drive the design.

**The objective is non-convex.** A single fit lands in whatever local minimum its random
initialisation leads to. Every fit here is therefore best-of-N restarts (`run_cp_nn`),
scored by reconstruction error. With too few restarts, a reported component may be an
artefact of one unlucky starting point.

**Rank is not identifiable from fit quality.** R² increases monotonically with rank, so it
cannot choose a rank on its own. `rank_selection` pairs it with the **stability** measure
of Williams et al. (2018): refit the same rank from many initialisations, take the
best-fitting model as reference, and score the others against it with `similarity_score`
— a Hungarian-matched mean of per-component similarities, since components come out in
arbitrary order. Stability stays near 1 while the rank is supported by the data and
collapses once it is not. Take the largest rank that is still both stable and explains a
useful share of the variance; plot both panels with `plotting.plot_rank_selection` and
read them together.

## Stage 4 — comparing components across subjects (`similarity.py`)

All subjects' components are flattened into one indexable list (a `ComponentSet`, ordered
subject-major) and every pair is scored. The score is the **product** of the per-mode
similarities, not their mean: a pair must agree in anatomy, in frequency *and* in time to
score highly. A shared rhythm in unrelated parcels is not a shared component, and a
multiplicative score says so.

ROI and frequency modes are compared by plain cosine similarity. The time mode is not,
and this is the one genuinely non-obvious choice in the pipeline:

> The same physiological event does not occur at the same latency in two subjects. Scored
> by a zero-lag cosine, two genuinely matching components whose time courses are offset by
> 100 ms score poorly, and clusters that should hold one component per subject fragment
> into singletons. So the time term is the **maximum normalised cross-correlation over
> lags within ±`lag_ms`** (`time_xcorr_sim`). The overlapping segments are re-normalised
> at each lag, so the result stays in [0, 1] and remains directly comparable to the two
> cosines — but it no longer penalises a shift the experiment never controlled.

`lag_ms` is the knob: wide enough to absorb genuine inter-subject latency jitter and no
wider, because at large values unrelated transients begin matching each other. The
default is 100 ms. `time_weight`, the exponent on the time term, exists for checking how
much a grouping rests on timing at all — 1 gives it full weight, 0 ignores time entirely.

The metric is stored as a `SimilarityConfig` on the returned `Similarity` object, so the
clustering, the cluster means and the quality report all score pairs exactly the way the
matrices were built. A downstream step quietly using a different metric is the most
common way this kind of analysis goes wrong.

## Stage 5 — clustering (`clustering.py`)

Two methods, answering slightly different questions. Both use the convention that
clusters are numbered 1..K and **0 means unassigned**.

### `align_components` — greedy alignment into slots

Seed a set of slots from one reference subject, then match every other subject's
components one-to-one onto those slots by linear assignment (the Hungarian algorithm),
maximising total similarity. A component whose best available slot scores below
`threshold` opens a new slot instead, while slots remain; otherwise it is left
unassigned. Each slot's template is the running mean of its members, so a slot drifts
towards the shared pattern as subjects join rather than staying anchored to the reference.

Fast and easy to read, with two weaknesses: the result depends on which subject seeds the
slots, and the one-to-one constraint can force a poor match when a subject genuinely
lacks a pattern. Use it to get oriented.

### `consensus_clustering` — exact clustering, stability-checked

Each clustering is solved as an integer program (`optimal_constrained_clustering`) that
maximises the total within-cluster similarity subject to three constraints:

- **threshold** — a pair scoring below `threshold` cannot be linked;
- **cannot-link** — two components of the *same* subject cannot share a cluster, encoding
  the assumption of one component per subject per phenomenon;
- **transitivity** — if *i*–*j* and *j*–*k* are linked then *i*–*k* must be, so the
  pairwise decisions form a genuine partition rather than an inconsistent set of
  overlapping pairs.

This is correlation clustering with cannot-link constraints. The number of clusters is
not specified — it falls out of the objective — and because the solution is globally
optimal, no reference subject is needed.

Being exact, re-running it on the same matrix returns the same answer, so a stability
check has to perturb the *input*. That is the resampling scheme of **Monti et al.
(2003)**: add symmetric Gaussian noise to the similarity matrix, re-solve, and record
which pairs landed together; repeat `n_resamples` times and average. The resulting
**consensus matrix** separates pairs that cluster because the data say so (entries near 1)
from pairs that cluster because of a near-tie in the objective (fractional entries). The
reported clusters are a final exact clustering of that matrix at 0.5 — pairs that
clustered together in a majority of rounds — and `m(k)`, the mean within-cluster
consensus, is how stable each cluster is. **This is the result to report.**

The transitivity constraints make the program O(n³) in size, which is fine for the few
tens of components a study of this kind produces and will not scale to hundreds.

### Judging the result

`cluster_quality` prints per-mode diagnostics and flags two specific failures:

- **A weak ROI score alongside high frequency and time scores** usually means a forced
  one-to-one match — an occipital component sitting in a motor cluster because something
  had to go there.
- **A subject whose best-matching component is not the one in the cluster.** Each of that
  subject's components is scored against the template built from the *other* members. If
  the best scorer sits elsewhere, the assignment is suspect; if none of them score well,
  that subject simply does not express the pattern — a finding, not a bug.

## Figures (`plotting.py`)

Conventions shared by every cluster figure:

- **ERS is red, ERD is blue**, so a figure is identifiable out of context.
- **Loadings are max-normalised before plotting.** Component scale in CP is arbitrary and
  absolute power differs between subjects; without normalisation one high-amplitude
  subject dominates every panel. The cost is that each member is pinned to 1 at its own
  peak, so any spread band is narrow near the peaks *by construction* — read those bands
  off-peak.
- **The ROI mode is drawn as a glass brain** where an atlas is available, and as a
  labelled bar panel otherwise.

Two figures per cluster, deliberately:

`plot_cluster` shows every member separately — all members' frequency and time profiles
overlaid, coloured by subject, then one ROI panel each. This is the figure for *judging* a
cluster.

`plot_cluster_mean` collapses it to one consensus profile. Three choices there are worth
stating, because at two or three members a shaded band is easy to over-read:

1. **The band is the member min-max range, not ±1 SD.** A sample SD from two points is
   just `|a−b|/√2`, a range wearing a distribution's clothes. The members are drawn over
   the mean so the figure shows the spread it actually has. A range of non-negative
   loadings also cannot dip below zero, which a symmetric mean ± SD band does at
   low-loading bins.
2. **Time courses are lag-aligned before averaging**, onto the cluster medoid (the member
   most similar to the rest, so the mean is not dragged onto an outlier's latency) and
   within the same ±`lag_ms` the clustering tolerates. Unaligned, an inter-subject latency
   shift smears the mean and reappears as spurious band width. Samples shifted in from
   beyond the recording are left NaN rather than invented, so the band stops at those
   edges. Pass `align_time=False` to compare.
3. **Members are max-normalised**, as above, so the panels read on the same 0–1 scale as
   `plot_cluster`.

`plot_connectogram_grid` shows the whole similarity structure before any clustering is
imposed on it: every component is a node on a ring, split into one sector per subject, and
every pair is a chord weighted by similarity. Comparing the per-mode panels shows which
mode carries the cross-subject structure.

`plot_tfr_roi` plots the un-decomposed data, and is worth using before and after: a CP
component should correspond to something visible in the raw time-frequency maps.

## Validating the method (`simulate.py`)

`simulate.py` generates data from known separable components, with every kind of
between-subject variability the real analysis has to survive: per-subject latency jitter
(the variability the lag-tolerant time similarity exists to absorb), amplitude,
frequency and spatial jitter, subject-specific nuisance components present in one subject
only, and additive noise.

Because the ground truth is known, the clustering can be checked rather than merely
inspected: each shared pattern should come back as one cluster holding one component per
subject, and each subject-specific pattern should be left unassigned.
`simulate.truth_table` prints what went in, for comparison with what came out.

The output dicts have the same keys as `io.load_tensor`, so moving the example onto real
data is a one-line change.

## References

- Williams, A. H., Kim, H., Wang, F., Vyas, S., Ryu, S. I., Shenoy, K. V., Schnitzer, M.,
  Kolda, T. G., & Ganguli, S. (2018). Unsupervised discovery of demixed, low-dimensional
  neural dynamics across multiple timescales through tensor component analysis.
  *Neuron*, 98(6), 1099–1115. — CP for neural data; the stability-based rank selection and
  the model similarity score.
- Monti, S., Tamayo, P., Mesirov, J., & Golub, T. (2003). Consensus clustering: a
  resampling-based method for class discovery and visualization of gene expression
  microarray data. *Machine Learning*, 52, 91–118. — the consensus matrix and `m(k)`.
- Rolls, E. T., Joliot, M., & Tzourio-Mazoyer, N. (2015). Implementation of a new
  parcellation of the orbitofrontal cortex in the automated anatomical labeling atlas.
  *NeuroImage*, 122, 1–5. — the AAL2 atlas.
- Kossaifi, J., Panagakis, Y., Anandkumar, A., & Pantic, M. (2019). TensorLy: Tensor
  learning in Python. *JMLR*, 20(26), 1–6. — the decomposition backend.
