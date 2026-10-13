"""Synthetic data with known ground truth, so the pipeline runs with no subject data.

This exists for two reasons. First, the example notebook has to run for someone who has
just cloned the repository and has no recordings. Second — and more usefully — a
simulation with known components is the only way to check that the clustering recovers
what it is supposed to: here the components *are* known, so a cluster either contains one
component per subject from the same ground-truth pattern or the method has failed.

What the generator produces is a signed `pow_norm`-equivalent tensor, built as a sum of
separable (ROI × Freq × Time) patterns, with every kind of between-subject variability
the real analysis has to survive:

- **latency jitter** — each subject's copy of a pattern is shifted in time by up to
  `jitter_ms`. This is the variability `similarity.compute_similarity` tolerates by
  cross-correlating the time mode, and the reason that design exists.
- **amplitude, frequency and spatial jitter** — patterns differ in strength, peak
  frequency and exactly which parcels they load on.
- **subject-specific nuisance components** — structure present in one subject only, which
  should end up as unassigned singletons rather than being forced into a cluster.
- **additive noise.**

The output dicts have the same keys as `io.load_tensor`, so switching the example to real
data is a one-line change: replace `simulate_dataset(...)` with a dict of `load_tensor`
calls and everything downstream is unchanged.
"""

from dataclasses import dataclass, field

import numpy as np

# The 120 AAL2 cortical/subcortical region names, in atlas order. Names only — these come
# from the public AAL2 label table and are what `io.load_aal_atlas` keys on, so a
# simulated dataset can be plotted on a real glass brain if the atlas is available.
AAL2_LABELS = [
    "Precentral_L",
    "Precentral_R",
    "Frontal_Sup_2_L",
    "Frontal_Sup_2_R",
    "Frontal_Mid_2_L",
    "Frontal_Mid_2_R",
    "Frontal_Inf_Oper_L",
    "Frontal_Inf_Oper_R",
    "Frontal_Inf_Tri_L",
    "Frontal_Inf_Tri_R",
    "Frontal_Inf_Orb_2_L",
    "Frontal_Inf_Orb_2_R",
    "Rolandic_Oper_L",
    "Rolandic_Oper_R",
    "Supp_Motor_Area_L",
    "Supp_Motor_Area_R",
    "Olfactory_L",
    "Olfactory_R",
    "Frontal_Sup_Medial_L",
    "Frontal_Sup_Medial_R",
    "Frontal_Med_Orb_L",
    "Frontal_Med_Orb_R",
    "Rectus_L",
    "Rectus_R",
    "OFCmed_L",
    "OFCmed_R",
    "OFCant_L",
    "OFCant_R",
    "OFCpost_L",
    "OFCpost_R",
    "OFClat_L",
    "OFClat_R",
    "Insula_L",
    "Insula_R",
    "Cingulate_Ant_L",
    "Cingulate_Ant_R",
    "Cingulate_Mid_L",
    "Cingulate_Mid_R",
    "Cingulate_Post_L",
    "Cingulate_Post_R",
    "Hippocampus_L",
    "Hippocampus_R",
    "ParaHippocampal_L",
    "ParaHippocampal_R",
    "Amygdala_L",
    "Amygdala_R",
    "Calcarine_L",
    "Calcarine_R",
    "Cuneus_L",
    "Cuneus_R",
    "Lingual_L",
    "Lingual_R",
    "Occipital_Sup_L",
    "Occipital_Sup_R",
    "Occipital_Mid_L",
    "Occipital_Mid_R",
    "Occipital_Inf_L",
    "Occipital_Inf_R",
    "Fusiform_L",
    "Fusiform_R",
    "Postcentral_L",
    "Postcentral_R",
    "Parietal_Sup_L",
    "Parietal_Sup_R",
    "Parietal_Inf_L",
    "Parietal_Inf_R",
    "SupraMarginal_L",
    "SupraMarginal_R",
    "Angular_L",
    "Angular_R",
    "Precuneus_L",
    "Precuneus_R",
    "Paracentral_Lobule_L",
    "Paracentral_Lobule_R",
    "Caudate_L",
    "Caudate_R",
    "Putamen_L",
    "Putamen_R",
    "Pallidum_L",
    "Pallidum_R",
    "Thalamus_L",
    "Thalamus_R",
    "Heschl_L",
    "Heschl_R",
    "Temporal_Sup_L",
    "Temporal_Sup_R",
    "Temporal_Pole_Sup_L",
    "Temporal_Pole_Sup_R",
    "Temporal_Mid_L",
    "Temporal_Mid_R",
    "Temporal_Pole_Mid_L",
    "Temporal_Pole_Mid_R",
    "Temporal_Inf_L",
    "Temporal_Inf_R",
    "Cerebelum_Crus1_L",
    "Cerebelum_Crus1_R",
    "Cerebelum_Crus2_L",
    "Cerebelum_Crus2_R",
    "Cerebelum_3_L",
    "Cerebelum_3_R",
    "Cerebelum_4_5_L",
    "Cerebelum_4_5_R",
    "Cerebelum_6_L",
    "Cerebelum_6_R",
    "Cerebelum_7b_L",
    "Cerebelum_7b_R",
    "Cerebelum_8_L",
    "Cerebelum_8_R",
    "Cerebelum_9_L",
    "Cerebelum_9_R",
    "Cerebelum_10_L",
    "Cerebelum_10_R",
    "Vermis_1_2",
    "Vermis_3",
    "Vermis_4_5",
    "Vermis_6",
    "Vermis_7",
    "Vermis_8",
    "Vermis_9",
    "Vermis_10",
]


@dataclass
class SyntheticComponent:
    """One ground-truth separable pattern.

    name          label used when reporting what the clustering recovered
    sign          +1 for a synchronisation (ERS), -1 for a desynchronisation (ERD)
    roi_patterns  ROI name substrings the pattern loads on (as `tensors.select_rois`)
    freq_hz       peak frequency, and `freq_sd` its Gaussian width in Hz
    time_s        peak latency, and `time_sd` its Gaussian width in seconds
    amplitude     peak magnitude in log-ratio units, before per-subject jitter
    """
    name: str
    sign: int
    roi_patterns: list
    freq_hz: float
    freq_sd: float
    time_s: float
    time_sd: float
    amplitude: float = 1.0


# The default ground truth: three desynchronisations and three synchronisations, chosen to
# look like a movement/visual task — a motor beta ERD around the event with a later beta
# rebound (ERS), an occipital alpha ERD, and so on. The pair of motor patterns at similar
# frequencies in overlapping parcels is the interesting case: they are separated only
# because they differ in *time* and in sign, which is exactly what the ERS/ERD split plus
# the time mode are there to resolve.
DEFAULT_COMPONENTS = [
    SyntheticComponent('motor beta ERD', -1,
                       ['precentral', 'postcentral', 'supp_motor_area'],
                       freq_hz=18, freq_sd=5.0, time_s=0.00, time_sd=0.22, amplitude=1.00),
    SyntheticComponent('occipital alpha ERD', -1,
                       ['calcarine', 'cuneus', 'occipital', 'lingual'],
                       freq_hz=10, freq_sd=2.2, time_s=0.12, time_sd=0.20, amplitude=0.85),
    SyntheticComponent('frontal theta ERD', -1,
                       ['frontal_sup_2', 'frontal_mid_2', 'frontal_sup_medial'],
                       freq_hz=6, freq_sd=1.6, time_s=0.30, time_sd=0.18, amplitude=0.55),
    SyntheticComponent('motor beta rebound ERS', +1,
                       ['precentral', 'postcentral', 'supp_motor_area'],
                       freq_hz=21, freq_sd=4.5, time_s=0.60, time_sd=0.20, amplitude=0.90),
    SyntheticComponent('temporal alpha ERS', +1,
                       ['temporal_sup', 'temporal_mid', 'heschl'],
                       freq_hz=11, freq_sd=2.4, time_s=0.25, time_sd=0.16, amplitude=0.70),
    SyntheticComponent('parietal gamma ERS', +1,
                       ['parietal_sup', 'parietal_inf', 'supramarginal'],
                       freq_hz=40, freq_sd=7.0, time_s=0.18, time_sd=0.14, amplitude=0.60),
]


@dataclass
class SimulationConfig:
    """Axes and the amount of between-subject variability.

    The defaults match the geometry of the recordings this pipeline was built for:
    120 AAL2 parcels, 1 Hz frequency bins from 5-70 Hz, and 10 ms time steps over
    -1 to +1 s around the event.
    """
    n_freq_lo: float = 5.0
    n_freq_hi: float = 70.0
    freq_step: float = 1.0
    time_lo: float = -1.0
    time_hi: float = 1.0
    time_step: float = 0.01

    jitter_ms: float = 80.0          # max per-subject latency shift of a shared pattern
    amp_jitter: float = 0.35         # fractional spread of per-subject amplitude
    freq_jitter_hz: float = 1.5      # per-subject shift of a pattern's peak frequency
    roi_jitter: float = 0.30         # fractional spread of per-parcel loadings
    noise_sd: float = 0.04           # additive Gaussian noise, in log-ratio units
    n_nuisance: int = 1              # subject-specific components, present in one subject
    nuisance_amplitude: float = 0.45

    baseline_window: tuple = (-1.0, -0.5)
    analysis_window: tuple = (-0.5, 1.0)
    band_edges: tuple = ((4, 7), (8, 12), (13, 20), (21, 30), (31, 45), (46, 70))
    n_trials: int = 100              # metadata only; single trials are not simulated

    def freq_vec(self):
        return np.arange(self.n_freq_lo, self.n_freq_hi + 1e-9, self.freq_step)

    def time_vec(self):
        return np.arange(self.time_lo, self.time_hi + 1e-9, self.time_step)


def _gaussian(x, centre, sd):
    return np.exp(-0.5 * ((np.asarray(x, float) - centre) / sd) ** 2)


def _roi_profile(roi_labels, patterns, rng, roi_jitter):
    """Non-negative loading over parcels: 1 on the named parcels, jittered, 0 elsewhere.

    The jitter is multiplicative and clipped at 0, so a parcel can be weakly or strongly
    involved but never negatively — matching the non-negative factors the decomposition
    will recover.
    """
    lowered = [p.lower() for p in patterns]
    hit = np.array([any(p in name.lower() for p in lowered) for name in roi_labels])
    if not hit.any():
        raise ValueError(f'no ROI matches any of {patterns}')
    profile = np.zeros(len(roi_labels))
    profile[hit] = np.clip(1.0 + rng.normal(0, roi_jitter, hit.sum()), 0.05, None)
    return profile / profile.max()


def simulate_subject(subject, components=None, config=None, roi_labels=None, seed=None,
                     verbose=False):
    """One subject's signed (ROI × Freq × Time) tensor, in `io.load_tensor`'s format.

    Each shared component is rebuilt with this subject's own latency, amplitude, peak
    frequency and spatial loadings, then `config.n_nuisance` components unique to this
    subject are added, then noise. The returned dict carries a `truth` key describing
    what went in — the per-subject latency shifts especially, which are what the lag-
    tolerant similarity has to undo.

    `seed` defaults to the subject id, so a given subject is reproducible and two
    subjects differ.
    """
    cfg = SimulationConfig() if config is None else config
    components = DEFAULT_COMPONENTS if components is None else components
    roi_labels = list(AAL2_LABELS) if roi_labels is None else list(roi_labels)
    rng = np.random.default_rng(subject if seed is None else seed)

    freq_vec, time_vec = cfg.freq_vec(), cfg.time_vec()
    pow_norm = np.zeros((len(roi_labels), len(freq_vec), len(time_vec)))
    truth = []

    for comp in components:
        lag_s = rng.uniform(-cfg.jitter_ms, cfg.jitter_ms) / 1e3
        amp = comp.amplitude * max(1.0 + rng.normal(0, cfg.amp_jitter), 0.15)
        f0 = comp.freq_hz + rng.normal(0, cfg.freq_jitter_hz)

        roi = _roi_profile(roi_labels, comp.roi_patterns, rng, cfg.roi_jitter)
        freq = _gaussian(freq_vec, f0, comp.freq_sd)
        time = _gaussian(time_vec, comp.time_s + lag_s, comp.time_sd)

        pow_norm += comp.sign * amp * roi[:, None, None] * freq[None, :, None] \
            * time[None, None, :]
        truth.append(dict(name=comp.name, type='ERS' if comp.sign > 0 else 'ERD',
                          lag_ms=lag_s * 1e3, amplitude=amp, freq_hz=f0,
                          time_s=comp.time_s + lag_s, shared=True))

    # Subject-specific structure. These have no counterpart in any other subject, so a
    # correct clustering leaves them unassigned; they are here so that outcome is tested.
    for k in range(cfg.n_nuisance):
        sign = 1 if rng.random() < 0.5 else -1
        idx = rng.choice(len(roi_labels), size=max(2, len(roi_labels) // 20), replace=False)
        roi = np.zeros(len(roi_labels))
        roi[idx] = rng.uniform(0.3, 1.0, len(idx))
        f0 = rng.uniform(freq_vec[0] + 5, freq_vec[-1] - 5)
        t0 = rng.uniform(time_vec[0] + 0.3, time_vec[-1] - 0.3)
        freq = _gaussian(freq_vec, f0, rng.uniform(2.0, 6.0))
        time = _gaussian(time_vec, t0, rng.uniform(0.10, 0.25))

        pow_norm += sign * cfg.nuisance_amplitude * roi[:, None, None] \
            * freq[None, :, None] * time[None, None, :]
        truth.append(dict(name=f'nuisance {k} (subject {subject} only)',
                          type='ERS' if sign > 0 else 'ERD', lag_ms=0.0,
                          amplitude=cfg.nuisance_amplitude, freq_hz=f0, time_s=t0,
                          shared=False))

    pow_norm += rng.normal(0, cfg.noise_sd, pow_norm.shape)

    # pow_banded, as the MATLAB side would produce it: pow_norm averaged within bands.
    band_names = ['theta', 'alpha', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma']
    pow_banded = np.stack(
        [pow_norm[:, (freq_vec >= lo) & (freq_vec <= hi), :].mean(axis=1)
         for lo, hi in cfg.band_edges], axis=1)

    if verbose:
        shared = sum(t['shared'] for t in truth)
        print(f'Subject {subject}: {pow_norm.shape} | {shared} shared + '
              f'{len(truth) - shared} subject-specific components | latency shifts ' +
              ', '.join(f'{t["lag_ms"]:+.0f}ms' for t in truth if t['shared']))

    return dict(
        pow_norm=pow_norm,
        pow_banded=pow_banded,
        pow_trials=None,            # single trials are not simulated — see the docstring
        toi_vec_trimmed=time_vec.copy(),
        time_vec=time_vec,
        freq_vec=freq_vec,
        baseline_window=np.array(cfg.baseline_window),
        analysis_window=np.array(cfg.analysis_window),
        band_edges=[list(e) for e in cfg.band_edges],
        band_names=band_names,
        n_trials=cfg.n_trials,
        roi_labels=roi_labels,
        truth=truth,
    )


def simulate_dataset(subjects=(1, 2, 3), components=None, config=None, roi_labels=None,
                     verbose=True):
    """{subject: data dict} for several subjects, ready for `tensors.build_dataset`.

    All subjects share the ground-truth component list and the axes; they differ in the
    per-subject jitter drawn inside `simulate_subject`.
    """
    return {s: simulate_subject(s, components=components, config=config,
                                roi_labels=roi_labels, verbose=verbose)
            for s in subjects}


def truth_table(loaded):
    """The ground truth of a simulated dataset as printable text.

    Compare it with the recovered clusters: each shared pattern should appear as one
    cluster holding one component per subject, and each subject-specific pattern should
    be left unassigned.
    """
    lines = ['ground truth', '  (shared patterns should each form one cross-subject '
             'cluster; subject-specific ones should stay unassigned)']
    for s, d in loaded.items():
        lines.append(f'\n  subject {s}:')
        for t in d['truth']:
            tag = 'shared ' if t['shared'] else 'UNIQUE '
            lines.append(f'    {tag}{t["type"]}  {t["name"]:<34} '
                         f'{t["freq_hz"]:5.1f} Hz  peak {t["time_s"]:+.2f} s  '
                         f'(lag {t["lag_ms"]:+.0f} ms)  amp {t["amplitude"]:.2f}')
    return '\n'.join(lines)
