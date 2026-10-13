"""Figures for every stage of the pipeline.

The conventions here are deliberate and shared across all the cluster figures:

- **ERS is red, ERD is blue** throughout, so a figure is identifiable out of context.
- **Loadings are max-normalised before plotting.** CP component scale is arbitrary and
  absolute power differs between subjects, so without normalisation one high-amplitude
  subject dominates every panel. The cost is that each member is pinned to 1 at its own
  peak, which narrows any spread band near the peaks by construction — read those bands
  off-peak.
- **The ROI mode is drawn as a glass brain** where an atlas is available, and as a bar
  panel otherwise. Pass `atlas=None` (the default) to get the bars; the brain needs
  nibabel and nilearn (`pip install "opm-cp[brain]"`) plus an `io.Atlas`.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

from .clustering import cluster_blocks, cluster_ids, cluster_order
from .similarity import best_time_lag, shift_onto_reference, time_xcorr_sim

BAND_LABELS = {
    'theta':      'θ (4-7 Hz)',
    'alpha':      'α (8-12 Hz)',
    'low_beta':   'Low β (13-20 Hz)',
    'high_beta':  'High β (21-30 Hz)',
    'low_gamma':  'Low γ (31-45 Hz)',
    'high_gamma': 'High γ (46-70 Hz)',
}

TYPE_COLORS = {'ERS': 'crimson', 'ERD': 'steelblue'}
TYPE_CMAPS = {'ERS': 'Reds', 'ERD': 'Blues'}

# One linestyle per cluster member, so members stay distinguishable where two subjects
# share a colour-blind-unfriendly pair or print in greyscale.
LINESTYLES = ['dotted', 'dashdot', 'dashed', 'solid', (0, (3, 1, 1, 1))]

# Canonical band edges, drawn as ticks above the frequency panels.
BAND_EDGES_HZ = (4, 8, 13, 30)
BAND_MARKS = ((6, 'θ'), (10, 'α'), (18, 'β'), (32, 'γ'))


def to_numpy(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def subject_colors(subjects, cmap='tab10'):
    """A stable {subject: colour} map, so every figure in a session agrees."""
    palette = plt.get_cmap(cmap)
    return {s: palette(i % 10) for i, s in enumerate(subjects)}


def _norm(v):
    """Max-normalise a loading vector. See the module docstring on why."""
    v = np.asarray(v, float)
    return v / (np.nanmax(v) + 1e-12)


def _roi_display_name(name):
    """'Precentral_L' -> 'L | Precentral', which sorts and reads better in a legend."""
    if name.endswith('_L'):
        return f"L | {name[:-2].replace('_', ' ')}"
    if name.endswith('_R'):
        return f"R | {name[:-2].replace('_', ' ')}"
    return name.replace('_', ' ')


# ── Per-subject factor plots ─────────────────────────────────────────────────

def _roi_bars(ax, weights, color='black'):
    ax.bar(np.arange(len(weights)), weights, color=color)
    ax.set_xticks([])
    ax.set_ylabel('Loading')
    ax.set_title('ROI')


def _freq_bars(ax, freq_vec, weights, color='black'):
    fv = to_numpy(freq_vec)
    width = (fv[1] - fv[0]) if len(fv) > 1 else 1.0
    ax.bar(fv, weights, width=width, color=color)
    ax.set_xticks([])
    ax.set_title('Frequency')


def _band_bars(ax, band_labels, weights, color='black'):
    ax.bar(np.arange(len(band_labels)), weights, color=color)
    ax.set_xticks([])
    ax.set_ylabel('Loading', fontsize=14)
    ax.set_title('Frequency Band')


def _time_bars(ax, time_vec, loadings, color='black', time_in_ms=True):
    tv = to_numpy(time_vec) * 1000 if time_in_ms else to_numpy(time_vec)
    y = to_numpy(loadings)
    width = (tv[1] - tv[0]) if len(tv) > 1 else 1.0
    ax.bar(tv, y, width=width, color=color)
    ax.axhline(0, color='black', lw=1.0)
    ax.set_xticks([])
    ax.set_title('Time')


def plot_cp3_full(factors, freq_vec, time_vec, plot_type='ERS', time_in_ms=True,
                  color='black'):
    """One row per component of a (ROI × Freq × Time) model: ROI, Frequency, Time.

    The quick look at a single subject's decomposition — no atlas, no clustering, just
    the raw factors as fitted. ROI labels are omitted because 50+ parcels will not fit on
    an axis; `plot_component` prints the top ROIs by name instead.
    """
    A, B, C = [to_numpy(f) for f in factors]
    for r in range(A.shape[1]):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f'{plot_type} — Component {r + 1}', fontsize=13, fontweight='bold',
                     color=TYPE_COLORS.get(plot_type, color))
        _roi_bars(axes[0], A[:, r], color)
        _freq_bars(axes[1], freq_vec, B[:, r], color)
        _time_bars(axes[2], time_vec, C[:, r], color, time_in_ms)
        plt.tight_layout()
        plt.show()


def plot_cp3_banded(factors, band_names, time_vec, plot_type='ERS', time_in_ms=True):
    """`plot_cp3_full` for a (ROI × Band × Time) tensor, where the frequency mode holds
    the six averaged bands rather than individual bins."""
    A, B, C = [to_numpy(f) for f in factors]
    color = TYPE_COLORS.get(plot_type, 'black')
    labels = [BAND_LABELS.get(b, b) for b in band_names]
    for r in range(A.shape[1]):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f'{plot_type} — Component {r + 1}', fontsize=13, fontweight='bold',
                     color=color)
        _roi_bars(axes[0], A[:, r], color)
        _band_bars(axes[1], labels, B[:, r], color)
        _time_bars(axes[2], time_vec, C[:, r], color, time_in_ms)
        plt.tight_layout()
        plt.show()


def print_top_rois(roi_loadings, roi_labels, top_n=10, indent='    '):
    """The `top_n` highest-loading ROIs by name — the readout the ROI bar panel cannot
    give, and the one actually quoted when describing a component."""
    roi = np.asarray(roi_loadings, float)
    for i in np.argsort(roi)[::-1][:top_n]:
        print(f'{indent}{roi_labels[i]:<28} {roi[i]:.4f}')


def plot_rank_selection(results, title_prefix=''):
    """Stability and R² against rank, from `decomposition.rank_selection`.

    Read the two panels together: R² alone rises with rank forever, so the rank to take
    is the largest one before stability falls away.
    """
    ranks = results['ranks']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    y = np.asarray(results['mean_stabilities'])
    yerr = np.asarray(results['std_stabilities'])
    axes[0].fill_between(ranks, y - yerr, y + yerr, color='lightcoral', alpha=0.35)
    axes[0].plot(ranks, y, 's-', color='0.1', markersize=4)
    axes[0].set(title=f'{title_prefix} Stability', xlabel='Rank',
                ylabel='Stability score', xticks=ranks, ylim=(0.1, 1.15))

    axes[1].plot(ranks, np.asarray(results['best_r2']), 's-', color='0.1', markersize=4)
    axes[1].set(title=f'{title_prefix} Explained Variance', xlabel='Rank', ylabel='R²',
                xticks=ranks, ylim=(0.1, 1.02))

    plt.tight_layout()
    plt.show()


# ── Model- and component-level similarity ────────────────────────────────────

def plot_model_similarity(cp_models, subjects, score_fn, types=('ERS', 'ERD')):
    """Whole-model similarity between every pair of subjects, one panel per type.

    This is the coarse version of the question the clustering answers properly: do these
    subjects' decompositions resemble each other at all, before worrying about which
    component matches which. `score_fn` is normally
    `decomposition.cosine_similarity_score`.
    """
    fig, axes = plt.subplots(1, len(types), figsize=(5.5 * len(types), 5), squeeze=False)
    axes = axes.ravel()
    im = None
    for ax, t in zip(axes, types):
        M = np.array([[score_fn(cp_models[t][a], cp_models[t][b]) for b in subjects]
                      for a in subjects])
        im = ax.imshow(M, cmap='magma', vmin=0, vmax=1)
        ax.set_title(t)
        ax.set_xticks(range(len(subjects)), [f'S{s}' for s in subjects])
        ax.set_yticks(range(len(subjects)), [f'S{s}' for s in subjects])
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color='white' if v < 0.8 else 'black')
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7)
    plt.show()


def plot_similarity_matrices(sim, clusters, colors, title=None):
    """Per-mode and whole-network similarity matrices, ordered by cluster.

    Each cluster's block is outlined, so a good clustering shows bright blocks on the
    diagonal and dark elsewhere. Unassigned components (cluster 0) are pushed to the end
    and left unboxed. Tick labels are coloured by subject, which makes it obvious at a
    glance whether a bright block spans subjects or is one subject's components matching
    each other.
    """
    whole, per_mode = sim.whole, sim.per_mode
    labels = np.array(sim.labels)
    subject_of = np.asarray(sim.subject_of)
    clusters = np.asarray(clusters)

    order = cluster_order(clusters)
    ord_clusters, ord_labels, ord_subject = clusters[order], labels[order], subject_of[order]
    n_slots = len(cluster_ids(clusters))
    palette = plt.get_cmap('tab10', max(n_slots, 10))

    mode_names = list(sim.mode_names)
    mats = [per_mode[name] for name in mode_names] + [whole]
    titles = [f'{name} similarity' for name in mode_names] + ['Whole-network similarity']

    fig, axes = plt.subplots(1, len(mats), figsize=(5 * len(mats), 6.5),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    im = None
    for ax, M, ttl in zip(axes, mats, titles):
        im = ax.imshow(M[np.ix_(order, order)], cmap='magma', vmin=0, vmax=1,
                       aspect='equal', interpolation='nearest')
        ax.set_title(ttl, fontsize=15, pad=8)
        ax.set_xticks(range(len(order)), ord_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(order)), ord_labels, fontsize=6)
        for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
            for tick, s in zip(ticks, ord_subject):
                tick.set_color(colors[s])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for start, length, cid in cluster_blocks(ord_clusters):
            if cid == 0:
                continue                       # don't box the unassigned tail
            ax.add_patch(mpatches.Rectangle((start - 0.5, start - 0.5), length, length,
                                            fill=False, edgecolor=palette(cid - 1), lw=2.5))

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02, shrink=0.6)
    cbar.set_label('Similarity', fontsize=14)
    if title:
        fig.suptitle(title, fontsize=18)
    plt.show()


def plot_consensus_matrix(M, sim, clusters, colors, title=None):
    """A consensus matrix in the style of Monti et al. (2003) figure 2.

    Solid red blocks are clusters that survived every perturbation; fractional (pale)
    entries are pairings that only sometimes held, i.e. the clustering was near a tie
    there. Singletons are pushed to the end.
    """
    labels, subject_of = np.array(sim.labels), np.asarray(sim.subject_of)
    order = cluster_order(clusters)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M[np.ix_(order, order)], cmap='Reds', vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), labels[order], rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), labels[order], fontsize=7)
    for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
        for tick, gi in zip(ticks, order):
            tick.set_color(colors[subject_of[gi]])
    ax.set_title(title or 'Consensus matrix')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Co-clustering frequency')
    plt.tight_layout()
    plt.show()


# ── ROI mode: glass brain, or bars where no atlas is available ───────────────

def roi_to_nifti(roi_loadings, atlas, roi_labels):
    """Paint an ROI loading vector onto an atlas volume.

    Returns (Nifti1Image, n_mapped). ROIs missing from `atlas.intensity` are skipped
    rather than guessed at — check `n_mapped` against `len(roi_labels)` if a brain looks
    emptier than expected.
    """
    import nibabel as nib

    volume = np.zeros(atlas.data.shape, dtype=float)
    n_mapped = 0
    for i, name in enumerate(roi_labels):
        intensity = atlas.intensity.get(name)
        if intensity is None:
            continue
        volume[atlas.data == intensity] = roi_loadings[i]
        n_mapped += 1
    return nib.Nifti1Image(volume, atlas.affine), n_mapped


def _roi_panel(ax, roi_loadings, roi_labels, atlas, cmap, threshold_percentile=80,
               display_mode='lyrz', top_n_labelled=8):
    """Draw the ROI mode: a glass brain if `atlas` is given, else a labelled bar panel.

    The threshold is a percentile of the *non-zero* loadings, not of all of them, so the
    sparsity of a component does not silently change what the threshold means.
    """
    roi = np.nan_to_num(np.asarray(roi_loadings, float), nan=0.0)
    roi_norm = roi / roi.max() if roi.max() > 0 else roi

    if atlas is None:
        order = np.argsort(roi_norm)[::-1]
        bar_color = plt.get_cmap(cmap)(0.65)
        ax.bar(np.arange(len(roi_norm)), roi_norm[order], color=bar_color)
        ax.set_xticks(range(min(top_n_labelled, len(order))))
        ax.set_xticklabels([_roi_display_name(roi_labels[i])
                            for i in order[:top_n_labelled]],
                           rotation=40, ha='right', fontsize=8)
        ax.set_ylabel('Normalised loading')
        ax.set_title('ROI loadings (no atlas supplied — see atlas/README.md)',
                     fontsize=10, loc='left')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        return

    from nilearn import plotting as nlplotting

    nifti, _ = roi_to_nifti(roi_norm, atlas, roi_labels)
    nonzero = roi_norm[roi_norm != 0]
    threshold = np.percentile(nonzero, threshold_percentile) if len(nonzero) else 0.01
    nlplotting.plot_glass_brain(nifti, axes=ax, display_mode=display_mode,
                                threshold=threshold, colorbar=True, plot_abs=True,
                                vmax=1.0, cmap=cmap, alpha=0.8)


# ── Single component ─────────────────────────────────────────────────────────

def _style_freq_axis(ax):
    ax.set_xlim(5, 70)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(range(0, 75, 5))
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    for hz, name in BAND_MARKS:
        ax.text(hz, 1.1, name, fontsize=10)
    for hz in BAND_EDGES_HZ:
        ax.vlines(hz, 1.1, 1.2, ls='--', color='black', alpha=0.5)


def plot_component(factors, comp_idx, roi_labels, freq_vec, time_vec, atlas=None,
                   plot_type='ERS', threshold_percentile=80, display_mode='lyrz',
                   top_n=10, time_in_ms=False, title=None, save_path=None):
    """One component of one subject: frequency and time profiles above, ROI below.

    The single-subject counterpart of `plot_cluster` — same layout and same conventions,
    without any clustering bookkeeping.
    """
    A, B, C = [to_numpy(f) for f in factors]
    color = TYPE_COLORS.get(plot_type, 'black')
    cmap = TYPE_CMAPS.get(plot_type, 'Greys')
    tv = to_numpy(time_vec) * 1000 if time_in_ms else to_numpy(time_vec)

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4])
    axB, axT = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for ax in (axB, axT):
        ax.set_facecolor('#FAFAFA')
        ax.set_ylabel('Normalised Loading')
        ax.minorticks_on()

    axB.plot(to_numpy(freq_vec), _norm(B[:, comp_idx]), color=color, drawstyle='steps-mid')
    _style_freq_axis(axB)
    axT.plot(tv, _norm(C[:, comp_idx]), color=color, drawstyle='steps-mid')
    axT.set_xlabel('Time (ms)' if time_in_ms else 'Time (s)', fontsize=14)

    _roi_panel(fig.add_subplot(gs[1, :]), A[:, comp_idx], roi_labels, atlas, cmap,
               threshold_percentile, display_mode)

    fig.suptitle(title or f'{plot_type} — Component {comp_idx + 1}', fontsize=15, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f'  top {top_n} ROIs:')
    print_top_rois(A[:, comp_idx], roi_labels, top_n)


# ── Clusters ─────────────────────────────────────────────────────────────────

def cluster_report(sim, clusters, cluster_id):
    """Text summary of one cluster: members and their internal similarity.

    The mean off-diagonal whole-network similarity is the number to quote for a cluster —
    it says how alike the members actually are, as opposed to merely having been grouped.
    """
    idx = np.where(np.asarray(clusters) == cluster_id)[0]
    labels = [str(x) for x in np.array(sim.labels)[idx]]
    lines = [f'cluster {cluster_id}: {labels}  (n={len(idx)})']
    if len(idx) > 1:
        sub = sim.whole[np.ix_(idx, idx)]
        off = sub[~np.eye(len(idx), dtype=bool)]
        lines.append(f'  mean off-diagonal whole-network similarity: {off.mean():.3f} '
                     f'(min {off.min():.3f}, max {off.max():.3f})')
    return '\n'.join(lines)


def plot_cluster(sim, clusters, cluster_id, roi_labels, freq_vec, time_vec, colors,
                 atlas=None, plot_type='ERS', threshold_percentile=80,
                 display_mode='lyrz', linestyles=None, save_path=None):
    """One figure per cluster showing every member separately.

    Frequency and time profiles of all members are overlaid on the top row (coloured by
    subject, one linestyle each), then one ROI panel per member below. This is the figure
    for judging a cluster: if the members' profiles genuinely overlap, the cluster is
    real; if one line wanders off, the quality report will usually have flagged it.

    `plot_cluster_mean` is the same cluster collapsed to one profile.
    """
    cs = sim.components
    clusters = np.asarray(clusters)
    idx = np.where(clusters == cluster_id)[0]
    if len(idx) == 0:
        print(f'{plot_type}: no components in cluster {cluster_id}')
        return
    linestyles = LINESTYLES if linestyles is None else linestyles
    members = [str(x) for x in np.array(sim.labels)[idx]]
    print(cluster_report(sim, clusters, cluster_id))

    cmap = TYPE_CMAPS.get(plot_type, 'Greys')
    n = len(idx)
    fig = plt.figure(figsize=(10, 4 + 3 * n))
    gs = fig.add_gridspec(1 + n, 2)
    axB, axT = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for ax in (axB, axT):
        ax.set_facecolor('#FAFAFA')
        ax.set_ylabel('Normalised Loading')
        ax.minorticks_on()

    for k, gi in enumerate(idx):
        s = cs.subject_of[gi]
        col, ls = colors[s], linestyles[k % len(linestyles)]
        _, freq, tim = cs.vecs[gi]
        axB.plot(freq_vec, _norm(freq), color=col, alpha=0.7, drawstyle='steps-mid',
                 linestyle=ls)
        axT.plot(time_vec, _norm(tim), color=col, alpha=0.7, drawstyle='steps-mid',
                 linestyle=ls, label=members[k])
    _style_freq_axis(axB)
    axT.set_xlabel('Time (s)', fontsize=14)
    axT.legend(fontsize=10, framealpha=0)

    for row, gi in enumerate(idx):
        ax = fig.add_subplot(gs[1 + row, :])
        _roi_panel(ax, cs.vecs[gi][0], roi_labels, atlas, cmap, threshold_percentile,
                   display_mode)
        pos = ax.get_position()
        fig.text(pos.x0, pos.y1, f'{members[row]}', fontsize=10,
                 color=colors[cs.subject_of[gi]], ha='left', va='bottom', zorder=10)

    fig.suptitle(f'{plot_type} Cluster {cluster_id}', fontsize=15, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def cluster_member_loadings(sim, clusters, cluster_id, align_time=True):
    """Max-normalised loadings of every member of one cluster, stacked per mode.

    Returns (labels, {mode_name: (n_members, n_bins)}, lag_ms).

    With `align_time`, the time courses are first shifted onto the cluster *medoid* — the
    member most similar to the rest, chosen so the mean is not dragged onto an outlier's
    latency — within the same ±lag the clustering tolerates. Without this, an
    inter-subject latency shift smears the mean and reappears as spurious band width.
    Samples shifted in from beyond the recording are left NaN rather than invented, so
    the band simply stops at those edges.
    """
    cs, cfg = sim.components, sim.config
    idx = np.where(np.asarray(clusters) == cluster_id)[0]
    labels = [str(x) for x in np.array(sim.labels)[idx]]

    stacks = {name: np.asarray([_norm(np.nan_to_num(cs.vecs[gi][m], nan=0.0))
                                for gi in idx])
              for m, name in enumerate(cs.mode_names)}

    lags = np.zeros(len(idx), int)
    if align_time and len(idx) > 1:
        T = stacks[cfg.time_mode]
        xc = np.array([[time_xcorr_sim(T[i], T[j], cfg.max_lag_samples)
                        for j in range(len(T))] for i in range(len(T))])
        ref = int(np.argmax(xc.sum(1)))            # medoid; its own lag stays 0
        lags = np.array([best_time_lag(T[i], T[ref], cfg.max_lag_samples)
                         for i in range(len(T))])
        stacks[cfg.time_mode] = np.array([shift_onto_reference(T[i], lags[i])
                                          for i in range(len(T))])
    return labels, stacks, lags * cfg.dt_ms


def mean_range(Y):
    """Across-member mean, min and max, NaN-safe for the lag-aligned edges.

    The spread shown in the cluster-mean figures is this **min-max range, not ±1 SD**,
    and deliberately so: with two or three members a sample SD is just |a-b|/√2, a range
    wearing a distribution's clothes. A range of non-negative loadings also cannot dip
    below zero, which a symmetric mean ± SD band does at low-loading bins.
    """
    return np.nanmean(Y, 0), np.nanmin(Y, 0), np.nanmax(Y, 0)


def plot_cluster_mean(sim, clusters, cluster_id, roi_labels, freq_vec, time_vec, colors,
                      atlas=None, plot_type='ERS', threshold_percentile=80,
                      display_mode='lyrz', top_n=5, align_time=True, linestyles=None,
                      save_path=None):
    """One figure per cluster: the cluster mean, with the member spread and the members.

    The counterpart to `plot_cluster` — same clusters, collapsed to one consensus profile
    instead of one line per member. The shaded band is the member min-max range (see
    `mean_range`) and the members are drawn over the mean, so the figure shows the spread
    it actually has rather than implying a distribution from two points.
    """
    cs = sim.components
    clusters = np.asarray(clusters)
    idx = np.where(clusters == cluster_id)[0]
    if len(idx) == 0:
        print(f'{plot_type}: no components in cluster {cluster_id}')
        return
    linestyles = LINESTYLES if linestyles is None else linestyles
    labels, stacks, lag_ms = cluster_member_loadings(sim, clusters, cluster_id, align_time)

    print(cluster_report(sim, clusters, cluster_id))
    if align_time and len(idx) > 1:
        print('  time lag applied vs medoid: ' +
              ', '.join(f'{l} {v:+.0f} ms' for l, v in zip(labels, lag_ms)))

    color = TYPE_COLORS.get(plot_type, 'black')
    cmap = TYPE_CMAPS.get(plot_type, 'Greys')
    # Modes are taken by position (ROI, Freq, Time) rather than by name, so a dataset
    # that names its modes differently still plots.
    roi_mode, freq_mode, time_mode = cs.mode_names[0], cs.mode_names[1], cs.mode_names[2]

    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4])
    axB, axT = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for ax, x, mode, xlabel in ((axB, freq_vec, freq_mode, 'Frequency (Hz)'),
                                (axT, time_vec, time_mode, 'Time (s)')):
        Y = stacks[mode]
        mu, lo, hi = mean_range(Y)
        # Legend on the time panel only — the same members appear in both.
        label_for = (lambda s: s) if mode == time_mode else (lambda s: None)
        ax.set_facecolor('#FAFAFA')
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
        for k in range(len(Y)):
            ax.plot(x, Y[k], color=colors[cs.subject_of[idx[k]]], lw=0.9, alpha=0.65,
                    linestyle=linestyles[k % len(linestyles)], label=label_for(labels[k]))
        ax.plot(x, mu, color=color, lw=2.2, label=label_for(f'Mean (n={len(Y)})'))
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylim(0, 1.2)
        ax.minorticks_on()
    _style_freq_axis(axB)
    axB.set_ylabel('Normalised Loading', fontsize=16)
    axT.legend(fontsize=12, framealpha=0)

    roi_mu, roi_lo, roi_hi = mean_range(stacks[roi_mode])
    _roi_panel(fig.add_subplot(gs[1, :]), roi_mu, roi_labels, atlas, cmap,
               threshold_percentile, display_mode)

    fig.suptitle(f'{plot_type} Cluster {cluster_id} — mean, band = member range',
                 fontsize=15, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f'  top {top_n} ROIs of the mean (mean [member min-max]):')
    for i in np.argsort(roi_mu)[::-1][:top_n]:
        print(f'    {roi_labels[i]:<28} {roi_mu[i]:.3f}  '
              f'[{roi_lo[i]:.3f}-{roi_hi[i]:.3f}]')


# ── Connectogram ─────────────────────────────────────────────────────────────

def _circular_positions(subject_of, subjects, gap_deg=22.0, r=1.0, start_deg=90):
    """Angle (rad) and (x, y) per node, grouped into one contiguous arc per subject."""
    n = len(subject_of)
    span = (360.0 - len(subjects) * gap_deg) / n
    ang, cur = {}, float(start_deg)
    for s in subjects:                              # subject_of is subject-major
        for i in np.where(subject_of == s)[0]:
            ang[int(i)] = np.deg2rad(cur + span / 2.0)
            cur += span
        cur += gap_deg
    return ang, {i: (r * np.cos(a), r * np.sin(a)) for i, a in ang.items()}


def plot_connectogram(sim, matrix, ax, colors, title='', gamma=3.0, min_alpha=0.25,
                      gap_deg=22.0, cross_color='black'):
    """Every component as a node on a ring, every pair as a chord weighted by similarity.

    The ring is split into one sector per subject, so cross-subject structure is visible
    as chords between sectors — drawn black and on top — while within-subject similarity
    stays in the subject's own colour. Chord width and opacity follow `w ** gamma`, so
    raising gamma suppresses the mass of weak-but-nonzero edges and leaves only the
    strong ones legible. This shows the whole similarity structure at once, before any
    clustering has been imposed on it.
    """
    cs = sim.components
    labels, subject_of = np.array(sim.labels), np.asarray(cs.subject_of)
    subjects = cs.subjects
    n = len(labels)
    ang, pos = _circular_positions(subject_of, subjects, gap_deg)
    half = (360.0 - len(subjects) * gap_deg) / n / 2.0

    for s in subjects:
        a = np.rad2deg([ang[i] for i in range(n) if subject_of[i] == s])
        ax.add_patch(mpatches.Wedge((0, 0), 1.20, a.min() - half, a.max() + half,
                                    width=0.17, facecolor=colors[s], alpha=0.25, lw=0))
        amid = np.deg2rad(a.mean())
        ax.text(1.40 * np.cos(amid), 1.40 * np.sin(amid), f'S{s}', ha='center',
                va='center', fontsize=14, fontweight='bold', color=colors[s])

    edges = sorted((matrix[i, j], i, j) for i in range(n) for j in range(i + 1, n))
    for w, i, j in edges:                           # ascending weight -> strong drawn last
        wn = max(w, 0.0) ** gamma
        same = subject_of[i] == subject_of[j]
        d = abs(ang[i] - ang[j]) % (2 * np.pi)
        d = min(d, 2 * np.pi - d)
        pull = 0.15 + 0.55 * (d / np.pi)            # neighbours stay shallow, far pairs arc in
        p0, p1 = pos[i], pos[j]
        ctrl = ((p0[0] + p1[0]) / 2.0 * pull, (p0[1] + p1[1]) / 2.0 * pull)
        path = Path([p0, ctrl, p1], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        ax.add_patch(mpatches.PathPatch(
            path, fill=False, edgecolor=colors[subject_of[i]] if same else cross_color,
            lw=(0.4 + 3.5 * wn) if same else (0.6 + 5.5 * wn),
            alpha=float(np.clip(min_alpha + (0.45 if same else 0.65) * wn, 0, 1)),
            capstyle='round', zorder=1 if same else 2))

    for i in range(n):
        x, y = pos[i]
        ax.scatter([x], [y], s=260, color=colors[subject_of[i]], edgecolors='black',
                   linewidths=1.0, zorder=3)
        ax.text(1.13 * np.cos(ang[i]), 1.13 * np.sin(ang[i]), labels[i], ha='center',
                va='center', fontsize=10, rotation=np.rad2deg(ang[i]) - 90,
                rotation_mode='anchor')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=18)


def plot_connectogram_grid(sims, colors, gamma=3.0):
    """A connectogram per (tensor type × mode), plus the whole-network score.

    `sims` is {type: Similarity}. Comparing the per-mode panels shows which mode is
    carrying the cross-subject structure: a strong Freq panel with a weak ROI panel means
    subjects share a rhythm but not the anatomy it lives in.
    """
    types = list(sims)
    panels = list(sims[types[0]].mode_names) + ['Whole']
    fig, axes = plt.subplots(len(types), len(panels),
                             figsize=(6.0 * len(panels), 6.0 * len(types)), squeeze=False)
    for r, t in enumerate(types):
        for c, name in enumerate(panels):
            sim = sims[t]
            matrix = sim.whole if name == 'Whole' else sim.per_mode[name]
            plot_connectogram(sim, matrix, axes[r, c], colors, gamma=gamma,
                              title=f'{t} — {"Whole-network" if name == "Whole" else name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


# ── Raw time-frequency inspection ────────────────────────────────────────────

def _bin_edges(v):
    """Cell edges from (possibly unevenly spaced) bin centres, for pcolormesh."""
    v = np.asarray(v, float)
    if len(v) == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    mid = 0.5 * (v[1:] + v[:-1])
    return np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])


def resolve_rois(rois, roi_labels):
    """Resolve ROI selectors to indices into `roi_labels`.

    Each selector is an int index, an exact label, or a case-insensitive substring
    (e.g. 'occipital' -> every occipital ROI). Returns the matched indices, in order and
    deduplicated.
    """
    if isinstance(rois, (str, int, np.integer)):
        rois = [rois]
    out = []
    for r in rois:
        if isinstance(r, (int, np.integer)):
            out.append(int(r))
        elif r in roi_labels:
            out.append(roi_labels.index(r))
        else:
            hits = [i for i, n in enumerate(roi_labels) if r.lower() in n.lower()]
            if not hits:
                raise KeyError(f'no ROI matches {r!r}')
            out.extend(hits)
    seen = set()
    return [i for i in out if not (i in seen or seen.add(i))]


def plot_tfr_roi(pow_norm, roi_labels, rois, freq_vec, time_vec, tlim=None, flim=None,
                 vmax=None, vmax_percentile=99, average=False, cmap='RdBu_r', title=None,
                 row_height=2.8, width=12, save_path=None):
    """Time-frequency map of the normalised power for one or more ROIs.

    This is the un-decomposed data, and worth looking at before and after: a CP component
    should correspond to something visible here. `pow_norm` is a signed log ratio to
    baseline, so the colour scale is diverging and symmetric about 0 — red is power above
    baseline (ERS), blue below (ERD).

    rois        int / label / case-insensitive substring, or a list of them
    tlim, flim  (lo, hi) windows in s / Hz; None keeps the full axis
    vmax        colour limit; by default `vmax_percentile` of |values| in the window
    average     one panel of the mean over the selected ROIs instead of one panel each

    Returns (fig, axes) so the caller can annotate further.
    """
    pow_norm = to_numpy(pow_norm)
    fv, tv = to_numpy(freq_vec), to_numpy(time_vec)
    idx = resolve_rois(rois, roi_labels)

    fsel = slice(None) if flim is None else slice(*np.searchsorted(fv, flim))
    tmask = np.ones(len(tv), bool) if tlim is None else (tv >= tlim[0]) & (tv <= tlim[1])
    fv_p, tv_p = fv[fsel], tv[tmask]

    maps = pow_norm[np.ix_(idx, np.arange(len(fv))[fsel], np.where(tmask)[0])]
    names = [_roi_display_name(roi_labels[i]) for i in idx]
    if average:
        maps, names = maps.mean(axis=0, keepdims=True), [f'mean of {len(idx)} ROIs']
    if vmax is None:
        vmax = float(np.percentile(np.abs(maps), vmax_percentile))

    fig, axes = plt.subplots(len(maps), 1, figsize=(width, row_height * len(maps)),
                             squeeze=False, sharex=True)
    axes = axes.ravel()
    xe, ye = _bin_edges(tv_p), _bin_edges(fv_p)
    for ax, m, name in zip(axes, maps, names):
        mesh = ax.pcolormesh(xe, ye, m, cmap=cmap, vmin=-vmax, vmax=vmax, shading='flat',
                             rasterized=True)
        for hz in (8, 13, 30):                      # band boundaries, kept recessive
            if ye[0] < hz < ye[-1]:
                ax.axhline(hz, color='0.35', lw=0.6, ls='--', alpha=0.6)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(name, fontsize=11, loc='left')
        ax.tick_params(length=3)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        cb = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025)
        cb.set_label('log ratio to baseline', fontsize=9)
        cb.ax.tick_params(labelsize=8)
    axes[-1].set_xlabel('Time (s)')
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98 if title else 1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig, axes
