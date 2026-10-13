"""Grouping components across subjects.

Each subject was decomposed independently, so component 0 of one subject has nothing to
do with component 0 of another. The question this module answers is which components
from *different* subjects describe the same phenomenon. Two methods are provided, and
they answer slightly different questions.

**`align_components` — greedy alignment into slots.** Seed a set of slots from one
reference subject, then match every other subject's components one-to-one onto those
slots by linear assignment, opening a new slot when nothing passes `threshold`. Fast,
produces at most one component per subject per slot, and is easy to read. Its weaknesses
are that it depends on which subject seeds the slots, and that the one-to-one constraint
can force a poor match when a subject genuinely lacks a pattern. Use it to get oriented.

**`consensus_clustering` — exact clustering, stability-checked.** Each clustering is
solved as an integer program (`optimal_constrained_clustering`) that maximises total
within-cluster similarity subject to three constraints: pairs below `threshold` cannot be
linked, two components of the *same* subject cannot share a cluster (a cannot-link
constraint — one component per subject per phenomenon), and the pairwise relation must be
transitive so that the result is a genuine partition. Because that is globally optimal,
no reference subject is needed. It is then wrapped in the resampling scheme of Monti et
al. (2003): perturb the similarity matrix with noise many times, re-cluster, and average
how often each pair ends up together. The resulting consensus matrix separates pairs that
cluster because the data say so from pairs that cluster because of a near-tie in the
objective. This is the result to report.

Cluster ids follow one convention throughout: clusters are numbered 1..K and 0 means
unassigned (a singleton, or a component no slot would take).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components


# ── Greedy alignment ─────────────────────────────────────────────────────────

def align_components(sim, n_slots=None, threshold=0.5, ref_subject=None):
    """Align each subject's components into cross-subject slots by linear assignment.

    sim           a `similarity.Similarity`
    n_slots       slots available; may exceed the rank, letting subjects contribute
                  patterns the reference subject does not have. Defaults to the rank.
    threshold     minimum whole-network similarity to accept an assignment
    ref_subject   subject seeding the slots; defaults to the first in the set

    Each slot's template is the running mean of its members, so a slot drifts towards
    the shared pattern as subjects join rather than staying anchored to the reference.
    A component whose best available slot scores below `threshold` opens a new slot
    instead, while slots remain; otherwise it is left unassigned (cluster 0).

    Returns an integer label per component, aligned with `sim`'s rows.
    """
    cs = sim.components
    rank, cfg = cs.rank, sim.config
    n_slots = rank if n_slots is None else n_slots
    if n_slots < rank:
        raise ValueError(f'n_slots ({n_slots}) must be >= rank ({rank})')

    subjects = cs.subjects
    ref = subjects[0] if ref_subject is None else ref_subject
    comp = {s: [cs.vecs[cs.index_of(s, c)] for c in range(rank)] for s in subjects}
    n_modes = len(cs.mode_names)

    templates = [[v.copy() for v in comp[ref][c]] for c in range(rank)]
    counts, n_active = [1] * rank, rank
    clusters = np.zeros(len(cs), dtype=int)
    for c in range(rank):
        clusters[cs.index_of(ref, c)] = c + 1          # the reference fills its own slots

    for s in [q for q in subjects if q != ref]:
        S = np.array([[cfg.whole_similarity(comp[s][c], templates[k])
                       for k in range(n_active)] for c in range(rank)])
        rows, cols = linear_sum_assignment(-S)         # maximise the total similarity
        assigned = {int(c): int(k) for c, k in zip(rows, cols)}
        for c in range(rank):
            k = assigned.get(c)
            if k is not None and S[c, k] >= threshold:
                clusters[cs.index_of(s, c)] = k + 1
                counts[k] += 1
                for m in range(n_modes):               # running-mean template update
                    templates[k][m] += (comp[s][c][m] - templates[k][m]) / counts[k]
            elif n_active < n_slots:
                templates.append([v.copy() for v in comp[s][c]])
                counts.append(1)
                n_active += 1
                clusters[cs.index_of(s, c)] = n_active
    return clusters


# ── Exact constrained clustering ─────────────────────────────────────────────

def _solver():
    """The MILP solver to use. HiGHS is preferred because the CBC binary bundled with
    pulp is x86-only and fails outright on Apple Silicon."""
    import pulp
    available = pulp.listSolvers(onlyAvailable=True)
    if 'HiGHS' in available:
        return pulp.HiGHS(msg=False)
    if 'HiGHS_CMD' in available:
        return pulp.HiGHS_CMD(msg=False)
    return pulp.PULP_CBC_CMD(msg=False)


def optimal_constrained_clustering(similarity_matrix, groups, threshold, solver=None):
    """Globally optimal cannot-link clustering, as an integer program.

    Maximises the sum of within-cluster similarities subject to:

    - **threshold**    — a pair with similarity < `threshold` cannot be linked;
    - **group**        — two items with the same entry in `groups` cannot share a
                         cluster (here: two components of the same subject);
    - **transitivity** — if i-j and j-k are linked then i-k must be, so the pairwise
                         decisions form a valid partition rather than an inconsistent
                         set of overlapping pairs.

    This is correlation clustering with cannot-link constraints: the number of clusters
    is *not* specified, it falls out of the objective. Returns one integer label per item
    (labels are arbitrary; use `relabel_by_size` for the 1..K / 0 convention).

    The transitivity constraints make this O(n³) in problem size, which is fine for the
    few tens of components a study of this kind produces and will not scale to hundreds.
    Requires pulp (`pip install "opm-cp[consensus]"`).
    """
    import pulp

    S = np.asarray(similarity_matrix, float)
    n = len(groups)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    prob = pulp.LpProblem('constrained_clustering', pulp.LpMaximize)
    y = pulp.LpVariable.dicts('y', pairs, cat=pulp.LpBinary)
    prob += pulp.lpSum(S[i, j] * y[(i, j)] for i, j in pairs)

    for i, j in pairs:
        if S[i, j] < threshold or groups[i] == groups[j]:
            prob += y[(i, j)] == 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                prob += y[(i, j)] + y[(j, k)] - y[(i, k)] <= 1
                prob += y[(i, j)] + y[(i, k)] - y[(j, k)] <= 1
                prob += y[(i, k)] + y[(j, k)] - y[(i, j)] <= 1

    prob.solve(solver if solver is not None else _solver())
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise RuntimeError(f'solver returned {pulp.LpStatus[prob.status]}')

    adj = np.eye(n, dtype=bool)
    for (i, j), v in y.items():
        if v.varValue and v.varValue > 0.5:
            adj[i, j] = adj[j, i] = True
    return connected_components(adj, directed=False)[1]


def relabel_by_size(labels, min_size=2):
    """Renumber arbitrary cluster labels to the package convention: clusters with at
    least `min_size` members become 1..K, everything smaller becomes 0 (unassigned).

    A single-subject "cluster" carries no cross-subject evidence, which is why
    singletons are set aside rather than reported as results.
    """
    labels = np.asarray(labels)
    out = np.zeros(len(labels), int)
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        if len(idx) >= min_size:
            out[idx] = out.max() + 1
    return out


def consensus_clustering(sim, threshold=0.6, n_resamples=500, noise_sd=0.05, seed=0,
                         min_size=2, verbose=True):
    """Consensus clustering over `optimal_constrained_clustering` (Monti et al. 2003).

    threshold     cannot-link threshold, on the same scale as `sim.whole`
    n_resamples   number of perturbed clusterings to average (H in the paper)
    noise_sd      SD of the symmetric Gaussian noise added to the similarity matrix
    min_size      smallest cluster reported; smaller ones become unassigned (0)

    The clustering is exact, so repeating it on the same matrix returns the same answer —
    resampling has to come from perturbing the *input*. Each round adds symmetric
    Gaussian noise to the similarity matrix, re-solves, and records which pairs landed
    together; the average over rounds is the consensus matrix M. Pairs that only cluster
    by a hair's breadth get a fractional entry, pairs the data genuinely support get ~1.
    The reported clusters are a final exact clustering of M at 0.5, i.e. pairs that
    clustered together in the majority of rounds.

    Returns (consensus_matrix, clusters, cluster_consensus) where `cluster_consensus`
    is m(k), the mean off-diagonal consensus within cluster k — read it as how stable
    each reported cluster is, in [0, 1].
    """
    S, groups = sim.whole, sim.components.subject_of
    n = len(groups)
    rng = np.random.default_rng(seed)
    solver = _solver()

    M = np.zeros((n, n))
    for _ in range(n_resamples):
        E = rng.normal(0, noise_sd, (n, n))
        E = 0.5 * (E + E.T)                     # keep the perturbed matrix symmetric
        np.fill_diagonal(E, 0.0)
        lab = optimal_constrained_clustering(np.clip(S + E, 0, 1), groups, threshold,
                                             solver=solver)
        M += lab[:, None] == lab[None, :]
    M /= n_resamples

    clusters = relabel_by_size(
        optimal_constrained_clustering(M, groups, 0.5, solver=solver), min_size=min_size)

    cluster_consensus = {}
    for k in cluster_ids(clusters):
        idx = np.where(clusters == k)[0]
        sub = M[np.ix_(idx, idx)]
        cluster_consensus[k] = float(sub[~np.eye(len(idx), dtype=bool)].mean())

    if verbose:
        print(format_clusters(sim, clusters,
                              f'consensus clusters (H={n_resamples}, noise={noise_sd}, '
                              f'threshold={threshold})'))
        for k, m in cluster_consensus.items():
            print(f'    cluster {k}: m(k)={m:.2f}')
    return M, clusters, cluster_consensus


# ── Reporting ────────────────────────────────────────────────────────────────

def cluster_ids(clusters):
    """The assigned cluster ids present, ascending. Excludes 0 (unassigned)."""
    return sorted({int(x) for x in np.asarray(clusters) if x > 0})


def cluster_blocks(ordered_clusters):
    """Contiguous (start, length, cluster_id) runs in an already cluster-ordered label
    array — used to draw the block outlines on a reordered similarity matrix."""
    blocks, start = [], 0
    for i in range(1, len(ordered_clusters) + 1):
        if i == len(ordered_clusters) or ordered_clusters[i] != ordered_clusters[start]:
            blocks.append((start, i - start, ordered_clusters[start]))
            start = i
    return blocks


def cluster_order(clusters):
    """Row order that groups a similarity matrix by cluster, unassigned pushed to the
    end. Stable, so subject order is preserved within a cluster."""
    clusters = np.asarray(clusters)
    return np.argsort(np.where(clusters > 0, clusters, clusters.max() + 1), kind='stable')


def format_clusters(sim, clusters, header=''):
    """Cluster membership as printable text."""
    labs = np.array(sim.labels)
    clusters = np.asarray(clusters)
    named = lambda mask: [str(x) for x in labs[mask]]
    lines = [f'\n=== {header} ===' if header else '']
    for k in cluster_ids(clusters):
        lines.append(f'  cluster {k}: {named(clusters == k)}')
    if (clusters == 0).any():
        lines.append(f'  unassigned: {named(clusters == 0)}')
    return '\n'.join(lines)


def cluster_quality(sim, clusters, weak_roi_threshold=0.5):
    """Per-mode diagnostics for each cluster, as printable text.

    Two patterns are worth looking for, and the report flags both:

    - **A weak ROI score with high Freq/Time scores** usually means a forced one-to-one
      match — an occipital component sitting in a motor cluster because something had to
      go there. Flagged when the mean first-mode similarity to the other members falls
      below `weak_roi_threshold`.
    - **A subject whose best-matching component is not the one in the cluster.** For each
      subject, every one of its components is scored against the template formed from the
      *other* subjects' members. If the best scorer is a component assigned elsewhere, the
      assignment is suspect; if all of them score poorly, that subject simply does not
      express the pattern, which is a finding rather than a bug.
    """
    cs, cfg = sim.components, sim.config
    labs, vecs, po = np.array(sim.labels), cs.vecs, cs.subject_of
    n_modes = len(cs.mode_names)
    out = [f'\n===== cluster quality =====']

    for k in cluster_ids(clusters):
        members = np.where(np.asarray(clusters) == k)[0]
        out.append(f'\ncluster {k}: {[str(x) for x in labs[members]]}')
        for i in members:
            others = [j for j in members if j != i]
            if not others:
                out.append(f'  {labs[i]:>8}: (single member)')
                continue
            per_mode = [np.mean([cfg.mode_similarity(vecs[i][m], vecs[j][m], m)
                                 for j in others]) for m in range(n_modes)]
            flag = '   <-- weak ROI (forced match?)' \
                if per_mode[0] < weak_roi_threshold else ''
            out.append(f'  {labs[i]:>8}: ' + '  '.join(
                f'{name}={v:.2f}' for name, v in zip(cs.mode_names, per_mode)) + flag)

        for s in cs.subjects:
            external = [j for j in members if po[j] != s]
            if not external:
                continue
            template = [np.mean([vecs[j][m] for j in external], axis=0)
                        for m in range(n_modes)]
            scores = [cfg.whole_similarity(vecs[cs.index_of(s, c)], template)
                      for c in range(cs.rank)]
            best = int(np.argmax(scores))
            here = [int(cs.comp_of[i]) for i in members if po[i] == s]
            note = '' if best in here else \
                '   <-- a different comp of this subject fits better'
            out.append(f'    S{s}: best comp={best} ({scores[best]:.2f})  '
                       f'in-cluster={here}{note}')
    return '\n'.join(out)
