"""Turning loaded files into the tensors that get decomposed.

Three things happen here, in order:

1. **ROI selection.** The atlas covers the whole brain including cerebellum and vermis,
   which OPM arrays cover poorly. Keeping a cortical subset stops the decomposition
   spending components on parcels the sensors cannot see.
2. **Time cropping.** The first and last few time points of a time-frequency estimate
   are contaminated by the taper reaching past the edge of the epoch, and those edge
   artefacts are large enough to attract a component of their own.
3. **ERS/ERD split.** `pow_norm` is a signed log ratio to baseline. Non-negative CP
   needs non-negative input, and — more importantly — a synchronisation (power up) and
   a desynchronisation (power down) in the same parcel are different phenomena that
   should not be forced to share a component. So the tensor is split into its positive
   part (ERS) and the magnitude of its negative part (ERD), and each is decomposed
   separately.
"""

from dataclasses import dataclass, field

import numpy as np


def split_ers_erd(data):
    """Split a signed baseline-normalised tensor into (ERS, ERD), both non-negative.

    ERS is the positive part, ERD the magnitude of the negative part, so
    `ers - erd == data` exactly and no sample is counted twice.
    """
    return np.maximum(data, 0), np.abs(np.minimum(data, 0))


def select_rois(roi_labels, keep_patterns):
    """Indices and names of the ROIs whose label contains any of `keep_patterns`.

    Matching is case-insensitive substring matching, so 'occipital' keeps every
    occipital parcel and both hemispheres at once. Atlas order is preserved, which
    matters because the kept indices are reused to index the atlas lookup later.
    """
    lowered = [p.lower() for p in keep_patterns]
    keep_idx = [i for i, name in enumerate(roi_labels)
                if any(p in name.lower() for p in lowered)]
    return keep_idx, [roi_labels[i] for i in keep_idx]


@dataclass
class Dataset:
    """Per-subject ERS/ERD tensors sharing one set of axes.

    tensors      {'ERS': {subject: (ROI, Freq, Time) array}, 'ERD': {...}}
    roi_labels   the kept ROI names, in atlas order — rows of the ROI mode
    freq_vec     Hz per frequency bin
    time_vec     seconds per time point, already cropped
    """
    tensors: dict
    roi_labels: list
    freq_vec: np.ndarray
    time_vec: np.ndarray
    subjects: list
    mode_names: list = field(default_factory=lambda: ['ROI', 'Freq', 'Time'])

    @property
    def types(self):
        """The tensor types present, conventionally ('ERS', 'ERD')."""
        return list(self.tensors)

    @property
    def dt_ms(self):
        """Sampling interval of the time mode, in ms."""
        return 1e3 * float(np.median(np.diff(self.time_vec)))

    def shape(self):
        first = self.tensors[self.types[0]][self.subjects[0]]
        return tuple(first.shape)


def build_dataset(loaded, keep_patterns=None, n_crop=0, dtype=np.float32, verbose=True):
    """Build a `Dataset` of ERS/ERD tensors from per-subject `io.load_tensor` dicts.

    loaded          {subject: dict} as returned by `io.load_tensor` (or
                    `simulate.simulate_subject`); axes are taken from the first subject
                    and every other subject is asserted to match it
    keep_patterns   ROI name substrings to keep (see `select_rois`); None keeps all ROIs
    n_crop          time points dropped from each end, for time-frequency edge artefacts

    Every subject must share the atlas ordering and the frequency/time axes — these come
    from the same upstream pipeline, so a mismatch means two differently-processed files
    have been mixed, which the assertions below catch rather than silently averaging over.
    """
    subjects = list(loaded)
    probe = loaded[subjects[0]]
    roi_labels_full = list(probe['roi_labels'])
    freq_vec = np.asarray(probe['freq_vec'])
    full_time = np.asarray(probe['time_vec'])
    time_vec = full_time[n_crop:len(full_time) - n_crop] if n_crop else full_time

    if keep_patterns is None:
        keep_idx, roi_labels = list(range(len(roi_labels_full))), roi_labels_full
    else:
        keep_idx, roi_labels = select_rois(roi_labels_full, keep_patterns)
    if not keep_idx:
        raise ValueError('keep_patterns matched no ROI labels')

    tensors = {'ERS': {}, 'ERD': {}}
    for s in subjects:
        d = loaded[s]
        assert list(d['roi_labels']) == roi_labels_full, f'ROI ordering differs for subject {s}'
        assert d['pow_norm'].shape[1] == len(freq_vec), f'frequency axis differs for subject {s}'
        assert d['pow_norm'].shape[2] == len(time_vec) + 2 * n_crop, \
            f'time axis differs for subject {s}'

        pn = d['pow_norm'][keep_idx][:, :, n_crop:d['pow_norm'].shape[2] - n_crop] \
            if n_crop else d['pow_norm'][keep_idx]
        ers, erd = split_ers_erd(pn)
        tensors['ERS'][s] = np.ascontiguousarray(ers, dtype=dtype)
        tensors['ERD'][s] = np.ascontiguousarray(erd, dtype=dtype)
        if verbose:
            print(f'Subject {s}: tensor {pn.shape} | '
                  f'ERS nonzero {(ers > 0).mean() * 100:.1f}%  '
                  f'ERD nonzero {(erd > 0).mean() * 100:.1f}%')

    ds = Dataset(tensors=tensors, roi_labels=roi_labels, freq_vec=freq_vec,
                 time_vec=time_vec, subjects=subjects)
    if verbose:
        print(f'\nKept {len(keep_idx)} / {len(roi_labels_full)} ROIs')
        print(f'time: {time_vec[0]:.3f}-{time_vec[-1]:.3f} s ({len(time_vec)} pts, '
              f'cropped {n_crop}/end, {ds.dt_ms:.1f} ms/sample)')
        print(f'freq: {freq_vec[0]:.1f}-{freq_vec[-1]:.1f} Hz ({len(freq_vec)} bins)')
    return ds
