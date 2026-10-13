# Atlas files (not included)

The glass-brain panels need the **AAL2** parcellation, which is not redistributed here —
it has its own licence and citation requirement. Everything else in the pipeline runs
without it: pass `atlas=None` (the default) and the ROI mode is drawn as a labelled bar
panel instead of a brain.

To enable the brain panels, put two files in this directory:

| file | what it is |
|---|---|
| `aal2.nii` | the parcellation volume |
| `aal2.nii.txt` | the label table — one `index name code` per line |

Both ship with the AAL2 release (and with the SPM AAL toolbox). Then:

```python
atlas = opm_cp.load_aal_atlas('atlas/aal2.nii', 'atlas/aal2.nii.txt')
print(opm_cp.check_atlas_coverage(atlas, dataset.roi_labels))   # [] = every ROI drawable
```

and pass `atlas=atlas` to `plot_cluster`, `plot_cluster_mean` or `plot_component`.

**One gotcha.** The volume distributed with AAL2 labels regions *sequentially* 1..120 —
the **first** column of the label table — not by the 2001/2002/… codes in the third
column. `load_aal_atlas` keys on the first column accordingly. If a glass brain comes out
empty, check `check_atlas_coverage` first: a name mismatch between the table and your
parcel labels is the usual cause.

Also note that `atlas/*.nii` and `atlas/*.txt` are listed in `.gitignore`, so these files
will not be committed if you add them here.

Cite: Rolls, E. T., Joliot, M., & Tzourio-Mazoyer, N. (2015). Implementation of a new
parcellation of the orbitofrontal cortex in the automated anatomical labeling atlas.
*NeuroImage*, 122, 1–5.
