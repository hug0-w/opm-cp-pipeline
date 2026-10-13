"""Readers for the MATLAB time-frequency files and for the AAL2 atlas.

The time-frequency decomposition itself is done upstream in MATLAB (FieldTrip); this
module only reads what it wrote. The files are HDF5 (MATLAB v7.3), and h5py presents
v7.3 arrays with their dimension order reversed relative to MATLAB, so every array is
transposed back on the way in — that transpose is the one thing worth knowing about
these loaders.
"""

from dataclasses import dataclass

import h5py
import numpy as np

# The six bands `pow_banded` is averaged into, in file order. Edges live in the file
# itself (`band_edges`); these are just the names.
BAND_NAMES = ['theta', 'alpha', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma']


def _char_refs_to_strings(f, refs):
    """MATLAB cellstr -> list[str]. Each cell is an object reference to a char array."""
    return [''.join(chr(c) for c in f[ref][:].flatten()) for ref in refs.flatten()]


def load_tensor(filename, verbose=True):
    """Load an epoched time-frequency file.

    Returns a dict with, among the axis vectors and metadata:

    pow_trials  (Trial, ROI, Freq, Time)  raw single-trial power on `toi_vec_trimmed`
    pow_norm    (ROI, Freq, Time)         trial-averaged, baseline-normalised power,
                                          ln(power / baseline_mean) — signed, so
                                          positive = ERS and negative = ERD
    pow_banded  (ROI, Band, Time)         `pow_norm` averaged within the six bands

    `pow_norm` is what the decompositions are run on (after `tensors.split_ers_erd`);
    `pow_trials` is kept for trial-level work such as surrogate/permutation nulls, which
    have to be built at the trial level rather than on the average — a CP decomposition
    is invariant to permutations within a mode, so shuffling the averaged tensor gives a
    degenerate null.
    """
    with h5py.File(filename, 'r') as f:
        pow_trials_raw = np.array(f['pow_single_trial'])
        pow_norm_raw = np.array(f['pow_norm'])
        pow_banded_raw = np.array(f['pow_banded'])
        toi_vec_trimmed = np.array(f['toi_vec_trimmed']).flatten()
        time_vec = np.array(f['time_vec']).flatten()
        freq_vec = np.array(f['freq_vec']).flatten()
        baseline_window = np.array(f['baseline_window']).flatten()
        analysis_window = np.array(f['analysis_window']).flatten()
        band_edges_raw = np.array(f['band_edges'])
        n_trials = int(np.array(f['n_trials']).flatten()[0])
        roi_labels = _char_refs_to_strings(f, f['roi_labels'][:])

    # h5py reads v7.3 arrays with reversed dim order — restore the MATLAB layout.
    pow_trials = (pow_trials_raw.transpose(3, 2, 1, 0)
                  if pow_trials_raw.shape[0] != n_trials else pow_trials_raw)
    pow_norm = pow_norm_raw.transpose(2, 1, 0)
    pow_banded = pow_banded_raw.transpose(2, 1, 0)
    band_edges = (band_edges_raw.T.tolist() if band_edges_raw.shape[0] == 2
                  else band_edges_raw.tolist())

    assert pow_trials.shape[3] == len(toi_vec_trimmed), (
        f'time mismatch: pow_trials has {pow_trials.shape[3]} time pts, '
        f'toi_vec_trimmed has {len(toi_vec_trimmed)}')

    data = dict(
        pow_trials=pow_trials,
        pow_norm=pow_norm,
        pow_banded=pow_banded,
        toi_vec_trimmed=toi_vec_trimmed,
        time_vec=time_vec,
        freq_vec=freq_vec,
        baseline_window=baseline_window,
        analysis_window=analysis_window,
        band_edges=band_edges,
        band_names=list(BAND_NAMES),
        n_trials=n_trials,
        roi_labels=roi_labels,
    )
    if verbose:
        describe_tensor(data)
    return data


def describe_tensor(d):
    """Print the shapes and axis ranges of a `load_tensor` dict — the first sanity check
    on a new file, and cheap enough to leave on by default."""
    print(f'pow_trials:       {d["pow_trials"].shape}  (Trial x ROI x Freq x Time)')
    print(f'  -> time axis:   toi_vec_trimmed  {d["toi_vec_trimmed"][0]:.3f} to '
          f'{d["toi_vec_trimmed"][-1]:.3f} s ({len(d["toi_vec_trimmed"])} pts)')
    print(f'pow_norm:         {d["pow_norm"].shape}  (ROI x Freq x Time)')
    print(f'pow_banded:       {d["pow_banded"].shape}  (ROI x Band x Time)')
    print(f'  -> time axis:   time_vec         {d["time_vec"][0]:.3f} to '
          f'{d["time_vec"][-1]:.3f} s ({len(d["time_vec"])} pts)')
    print(f'freq_vec:         {d["freq_vec"][0]:.0f} to {d["freq_vec"][-1]:.0f} Hz '
          f'({len(d["freq_vec"])} bins)')
    print(f'baseline_window:  {d["baseline_window"]}')
    print(f'analysis_window:  {d["analysis_window"]}')
    print(f'band_edges:       {d["band_edges"]}')
    print(f'n_trials:         {d["n_trials"]}')
    print(f'ROIs:             {len(d["roi_labels"])}')


def load_tensor_continuous(filename, verbose=True):
    """Load a continuous (non-epoched) time-frequency file.

    These files hold one uninterrupted recording block rather than epoched trials, so
    there is no trial axis and no pre-stimulus baseline window: `pow_norm` is the log
    ratio of `pow_raw` to the per-(ROI, freq) median over the whole block
    (`baseline_median`). Everything downstream of `tensors.split_ers_erd` applies
    unchanged, since the result is still a signed (ROI, Freq, Time) tensor.
    """
    with h5py.File(filename, 'r') as f:
        pow_raw = np.array(f['pow_raw']).transpose(2, 1, 0)       # -> ROI x Freq x Time
        pow_norm = np.array(f['pow_norm']).transpose(2, 1, 0)
        baseline_median = np.array(f['baseline_median']).T        # -> ROI x Freq
        time_vec = np.array(f['time_vec']).flatten()
        freq_vec = np.array(f['freq_vec']).flatten()
        toi_step = float(np.array(f['toi_step']).flatten()[0])
        edge_trim = float(np.array(f['edge_trim']).flatten()[0])
        roi_labels = _char_refs_to_strings(f, f['roi_labels'][:])

    assert pow_norm.shape == (len(roi_labels), len(freq_vec), len(time_vec)), (
        f'axis mismatch: pow_norm {pow_norm.shape} vs '
        f'({len(roi_labels)}, {len(freq_vec)}, {len(time_vec)})')

    if verbose:
        print(f'pow_raw / pow_norm:  {pow_norm.shape}  (ROI x Freq x Time)')
        print(f'  -> time axis:      time_vec  {time_vec[0]:.3f} to {time_vec[-1]:.3f} s '
              f'({len(time_vec)} pts, step {toi_step * 1e3:.0f} ms)')
        print(f'freq_vec:            {freq_vec[0]:.1f} to {freq_vec[-1]:.1f} Hz '
              f'({len(freq_vec)} bins)')
        print(f'baseline_median:     {baseline_median.shape}  (ROI x Freq, whole-block median)')
        print(f'edge_trim:           {edge_trim} s already removed each end')
        print(f'ROIs:                {len(roi_labels)}')

    return dict(
        pow_raw=pow_raw,
        pow_norm=pow_norm,
        baseline_median=baseline_median,
        time_vec=time_vec,
        freq_vec=freq_vec,
        toi_step=toi_step,
        edge_trim=edge_trim,
        roi_labels=roi_labels,
    )


# ── AAL2 atlas ───────────────────────────────────────────────────────────────

@dataclass
class Atlas:
    """A parcellation volume plus the name -> voxel-intensity lookup that paints an ROI
    loading vector back onto it. Built by `load_aal_atlas`; consumed by the glass-brain
    panels in `plotting`."""
    data: np.ndarray          # the label volume
    affine: np.ndarray        # voxel -> world transform
    intensity: dict           # ROI name -> intensity in `data`


def load_aal_atlas(nifti_path, label_txt_path, verbose=True):
    """Load the AAL2 volume and its label table.

    The label table is the `aal2.nii.txt` shipped with the atlas: `index name code`
    per line. The volume distributed alongside it labels regions *sequentially* 1..120
    (the first column), **not** by the 2001/2002/... codes in the third column — so
    names map to the 1-based row index. Check `missing` in the printed summary if a
    glass brain comes out empty; a mismatch there is the usual cause.

    Requires nibabel (`pip install "opm-cp[brain]"`).
    """
    import nibabel as nib

    intensity = {}
    with open(label_txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                intensity[parts[1]] = int(parts[0])

    img = nib.load(nifti_path)
    atlas = Atlas(data=img.get_fdata(), affine=img.affine, intensity=intensity)
    if verbose:
        print(f'AAL2: volume {atlas.data.shape}, {len(intensity)} labelled regions')
    return atlas


def check_atlas_coverage(atlas, roi_labels):
    """ROI names with no entry in the atlas lookup. Empty list = every ROI can be drawn."""
    return [r for r in roi_labels if r not in atlas.intensity]
