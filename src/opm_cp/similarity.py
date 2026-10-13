"""Comparing single components *across* subjects.

`decomposition` compares whole models; this module compares the individual components
inside them, which is what the clustering needs. The components of all subjects are
flattened into one list (a `ComponentSet`) and every pair is scored, giving the square
similarity matrices the clustering consumes.

The score is the product of the per-mode similarities, so a pair must agree in anatomy,
in frequency *and* in time to score highly. Two of the three modes are compared by plain
cosine. The time mode is not:

**Why the time mode is lag-tolerant.** The same physiological event — a beta
desynchronisation around movement, say — does not occur at the same latency in two
subjects. Scored by a zero-lag cosine, two genuinely matching components with time
courses offset by 100 ms score poorly, and clusters that should hold one component per
subject fragment into singletons. So the time term is the *maximum* normalised
cross-correlation over lags within ±`lag_ms` (`time_xcorr_sim`). It stays in [0, 1] and
so remains directly comparable to the two cosines, but it no longer penalises a shift
the experiment never controlled. `lag_ms` is the one knob: it should be wide enough to
absorb genuine inter-subject latency jitter and no wider, since at large values
unrelated transients start matching each other.

`time_weight` is the exponent on the time term, for checking how much the grouping rests
on timing at all: 1 gives it full weight, 0 ignores the time mode entirely.
"""

from dataclasses import dataclass

import numpy as np


# ── Per-mode similarity primitives ───────────────────────────────────────────

def cosine(a, b):
    """Cosine similarity of two vectors; 0 if either is all-zero."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def time_xcorr_sim(a, b, max_lag):
    """Max normalised cross-correlation of two time courses over lags in ±`max_lag`.

    Lags are in samples. The overlapping segments are re-normalised at each lag, so the
    result stays in [0, 1] and is directly comparable to a cosine. Shorter overlaps at
    large lags are compared on their own terms rather than zero-padded, which would
    otherwise make a large shift look artificially good.
    """
    a, b, n = np.asarray(a, float), np.asarray(b, float), len(a)
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        av, bv = (a[lag:], b[:n - lag]) if lag >= 0 else (a[:n + lag], b[-lag:])
        na, nb = np.linalg.norm(av), np.linalg.norm(bv)
        if av.size and na > 0 and nb > 0:
            best = max(best, float(av @ bv / (na * nb)))
    return best


def best_time_lag(a, b, max_lag):
    """The lag (samples) at which `a` best matches `b`, in `time_xcorr_sim`'s convention:
    `a[lag:]` lines up with `b[:n - lag]`, i.e. `a` trails `b` by `lag`.

    Used when averaging a cluster's time courses: the members are first shifted onto one
    reference, so an inter-subject latency difference does not smear the mean.
    """
    a, b, n = np.asarray(a, float), np.asarray(b, float), len(a)
    best, best_lag = -1.0, 0
    for lag in range(-max_lag, max_lag + 1):
        av, bv = (a[lag:], b[:n - lag]) if lag >= 0 else (a[:n + lag], b[-lag:])
        na, nb = np.linalg.norm(av), np.linalg.norm(bv)
        if av.size and na > 0 and nb > 0:
            c = float(av @ bv / (na * nb))
            if c > best:
                best, best_lag = c, lag
    return best_lag


def shift_onto_reference(v, lag):
    """`v` re-indexed onto a reference's time base, so `out[t] = v[t + lag]`.

    Samples shifted in from outside the recording are left NaN rather than invented, so
    a cluster mean stops where its evidence stops instead of tapering into zeros.
    """
    out = np.full(len(v), np.nan)
    if lag >= 0:
        out[:len(v) - lag] = v[lag:]
    else:
        out[-lag:] = v[:len(v) + lag]
    return out


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class SimilarityConfig:
    """How components are compared. Carried on the `Similarity` result so that the
    clustering, the cluster means and the quality report all score pairs exactly the way
    the similarity matrices were built — the single most common way these analyses go
    quietly wrong is a downstream step using a different metric."""
    mode_names: tuple = ('ROI', 'Freq', 'Time')
    time_mode: str = 'Time'
    max_lag_samples: int = 0      # ±lag for the time cross-correlation
    time_weight: float = 1.0      # exponent on the time term
    lag_ms: float = 0.0           # max_lag_samples in ms, for reporting
    dt_ms: float = 0.0            # sampling interval of the time mode

    @property
    def time_mode_idx(self):
        return list(self.mode_names).index(self.time_mode)

    def mode_similarity(self, va, vb, mode_idx):
        """Similarity of one mode's loading vectors. Cosines are clipped at 0: with
        non-negative factors a negative cosine cannot occur, and clipping keeps the
        product well-behaved if the metric is ever reused on signed factors."""
        if mode_idx == self.time_mode_idx:
            return time_xcorr_sim(va, vb, self.max_lag_samples)
        return max(cosine(va, vb), 0.0)

    def whole_similarity(self, vecs_a, vecs_b):
        """Product of the per-mode similarities, with the time term raised to
        `time_weight`. Multiplicative, not additive, so a component that matches in two
        modes but not the third scores low — which is the behaviour wanted: a shared
        frequency band in unrelated parcels is not a shared component."""
        s = 1.0
        for m, (va, vb) in enumerate(zip(vecs_a, vecs_b)):
            c = self.mode_similarity(va, vb, m)
            s *= c ** self.time_weight if m == self.time_mode_idx else c
        return s


# ── Component sets and their similarity matrices ─────────────────────────────

@dataclass
class ComponentSet:
    """Every component of every subject, flattened into one indexable list.

    Order is subject-major: subject `subjects[0]`'s components 0..R-1 come first, then
    the next subject's, and so on. Row/column `i` of every matrix in this module refers
    to `labels[i]`, the component from subject `subject_of[i]`, column `comp_of[i]` of
    that subject's factors. `index_of(subject, comp)` goes the other way.

    vecs[i][m] is component i's loading vector in mode m.
    """
    labels: list
    subject_of: np.ndarray
    comp_of: np.ndarray
    vecs: list
    mode_names: tuple
    subjects: list
    rank: int

    def __len__(self):
        return len(self.labels)

    def index_of(self, subject, comp):
        """Row index of component `comp` of `subject`."""
        return self.subjects.index(subject) * self.rank + comp

    def members_of(self, idx):
        """(subject, comp) pairs for a list of row indices."""
        return [(self.subject_of[i], int(self.comp_of[i])) for i in np.asarray(idx)]


@dataclass
class Similarity:
    """Similarity matrices over a `ComponentSet`.

    per_mode  {mode_name: (n, n)} — one similarity matrix per mode
    whole     (n, n) — the combined score the clustering actually uses
    """
    components: ComponentSet
    per_mode: dict
    whole: np.ndarray
    config: SimilarityConfig

    # Convenience passthroughs, so callers rarely need to reach into `.components`.
    @property
    def labels(self):
        return self.components.labels

    @property
    def subject_of(self):
        return self.components.subject_of

    @property
    def mode_names(self):
        return self.components.mode_names

    def vecs_of(self, i):
        return self.components.vecs[i]

    def cross_subject_values(self):
        """The off-diagonal whole-network similarities between *different* subjects.

        These are the only values the clustering can act on (same-subject pairs are
        forbidden from sharing a cluster), so their spread is the right scale on which
        to set a threshold — hence `summary()` reporting their median.
        """
        po = self.components.subject_of
        return self.whole[np.triu(po[:, None] != po[None, :], 1)]

    def summary(self):
        cross = self.cross_subject_values()
        return (f'{self.whole.shape} similarity over {len(self.labels)} components\n'
                f'  cross-subject whole-network similarity: median '
                f'{np.median(cross):.3f} (max {cross.max():.3f}, n={cross.size})')


def build_component_set(factors_by_subject, subjects=None, rank=None,
                        mode_names=('ROI', 'Freq', 'Time')):
    """Flatten {subject: [mode factor arrays]} into a `ComponentSet`.

    `subjects` restricts and orders the subjects included — the hook for excluding one
    subject's decomposition from the clustering (a fit that failed to converge, or a
    tensor type one subject simply does not express) while keeping it available for
    per-subject figures. Defaults to every key of `factors_by_subject`.
    """
    subjects = list(factors_by_subject) if subjects is None else list(subjects)
    rank = factors_by_subject[subjects[0]][0].shape[1] if rank is None else rank

    vecs, labels, subject_of, comp_of = [], [], [], []
    for s in subjects:
        f = factors_by_subject[s]
        if f[0].shape[1] < rank:
            raise ValueError(f'subject {s} has {f[0].shape[1]} components, rank={rank}')
        for c in range(rank):
            vecs.append([np.asarray(f[m][:, c], float) for m in range(len(mode_names))])
            labels.append(f'S{s}.{c}')
            subject_of.append(s)
            comp_of.append(c)

    return ComponentSet(labels=labels, subject_of=np.array(subject_of),
                        comp_of=np.array(comp_of), vecs=vecs,
                        mode_names=tuple(mode_names), subjects=subjects, rank=rank)


def compute_similarity(component_set, dt_ms, lag_ms=100.0, time_weight=1.0,
                       time_mode='Time', verbose=True):
    """Score every pair of components in a `ComponentSet`.

    dt_ms         sampling interval of the time mode (`Dataset.dt_ms`), which converts
                  `lag_ms` into the sample lag the cross-correlation searches
    lag_ms        max latency difference tolerated between matching time courses
    time_weight   exponent on the time term (1 = full weight, 0 = ignore time)

    Returns a `Similarity` holding the per-mode matrices, the combined `whole` matrix,
    and the `SimilarityConfig` that produced them.
    """
    mode_names = component_set.mode_names
    max_lag = int(round(lag_ms / dt_ms))
    cfg = SimilarityConfig(mode_names=mode_names, time_mode=time_mode,
                           max_lag_samples=max_lag, time_weight=time_weight,
                           lag_ms=lag_ms, dt_ms=dt_ms)
    if verbose:
        print(f'Time resolution {dt_ms:.1f} ms/sample  ->  '
              f'±{lag_ms:.0f} ms = ±{max_lag} samples')

    n = len(component_set)
    per_mode = {name: np.zeros((n, n)) for name in mode_names}
    for i in range(n):
        for j in range(i, n):                     # every mode similarity is symmetric
            for m, name in enumerate(mode_names):
                v = cfg.mode_similarity(component_set.vecs[i][m],
                                        component_set.vecs[j][m], m)
                per_mode[name][i, j] = per_mode[name][j, i] = v

    whole = np.ones((n, n))
    for m, name in enumerate(mode_names):
        whole *= per_mode[name] ** cfg.time_weight if m == cfg.time_mode_idx \
            else per_mode[name]

    sim = Similarity(components=component_set, per_mode=per_mode, whole=whole, config=cfg)
    if verbose:
        print(sim.summary())
    return sim
