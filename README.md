# Note from author: Agentic AI was used to compile this git repository, however the original code used in complilation was produced by Hugo Wharton with assistance from AI in selected coding tasks (e.g. code review and debugging). 

# Source-space MEG CPD

Data-driven discovery of event-related brain activity in source-space MEG, using
non-negative tensor decomposition.

## About

Analysing event-related changes in oscillatory power usually means committing to a
frequency band and a time window before you look — averaging within them and testing
across parcels. That works when you already know what you are looking for, but a real
neural response is a *joint* pattern: particular brain regions, at particular frequencies,
over a particular interval. Fixing two of those three in advance discards the structure
that makes the response interesting.

This package takes the whole **region × frequency × time** array of baseline-normalised
power and factorises it, recovering each dominant pattern as a single component with its
own spatial map, spectral profile and time course. Nothing is specified in advance; the
patterns come out of the data.

It then addresses the harder question that follows. Each subject is decomposed separately,
so their components arrive in arbitrary order with no correspondence between them —
knowing that a pattern is reproducible across a group means first working out which
component in one subject corresponds to which in another. The package does that
automatically, and is deliberately tolerant of the fact that the same physiological
response does not occur at the same moment in two people.

Built for OPM-MEG recordings parcellated with the AAL2 atlas, but nothing in it is
specific to OPM: any signed region × frequency × time tensor will work.

## Features

- Non-negative CP (PARAFAC) decomposition with multiple restarts, so results do not hinge
  on one random initialisation
- Principled model-order selection, based on component reproducibility rather than fit
  alone
- Separate treatment of power increases and decreases, which are distinct phenomena
- Cross-subject component matching that tolerates inter-subject latency differences
- Two clustering methods, including an exact formulation with a resampling-based stability
  measure
- Diagnostics that say *why* a cluster looks the way it does, not just what it contains
- Publication-oriented figures at every stage: factor profiles, similarity matrices, glass
  brains, connectograms, time-frequency maps
- A simulator with known ground truth, so the method can be validated rather than trusted

## Installation

```bash
git clone <this repo> && cd opm-cp-pipeline
pip install -e ".[all]"
```

The core install needs only numpy, scipy, h5py, torch, tensorly and matplotlib. Three
optional extras — `brain` (glass brains), `consensus` (exact clustering) and `notebook` —
are grouped under `all`; the pipeline degrades gracefully without any of them.

On Apple Silicon, install the `consensus` extra rather than pulp alone, so you get a
solver that runs natively.

The AAL2 atlas is not bundled, as it carries its own licence and citation requirement.
See [`atlas/README.md`](atlas/README.md) if you want the glass-brain figures.

## Getting started

[`examples/ers_erd_cross_subject_clustering.ipynb`](examples/ers_erd_cross_subject_clustering.ipynb)
runs the whole pipeline end to end and is the place to start. It uses simulated data with
known ground truth, so it works from a fresh clone with no recordings of your own, and its
final section checks what the method recovered against what actually went in. One clearly
marked cell is all you change to point it at your own data.

```python
import opm_cp as cp
from opm_cp import plotting as vz, simulate

# Load and prepare (swap simulate_dataset for cp.load_tensor on real data)
loaded = simulate.simulate_dataset(subjects=(1, 2, 3))
ds = cp.build_dataset(loaded, keep_patterns=['precentral', 'occipital'], n_crop=5)

# Decompose each subject independently
cp_models, factors = cp.decompose_dataset(ds, rank=3, n_runs=25)

# Match components across subjects, then group them
components = cp.build_component_set(factors['ERD'], subjects=ds.subjects, rank=3)
sim = cp.compute_similarity(components, dt_ms=ds.dt_ms, lag_ms=150)
M, clusters, stability = cp.consensus_clustering(sim, threshold=0.5)
```

## Project structure

```
src/opm_cp/
  io.py             reading time-frequency files and the atlas
  tensors.py        preparing the tensors to be decomposed
  decomposition.py  fitting CP models and choosing a model order
  similarity.py     comparing components across subjects
  clustering.py     grouping matching components
  plotting.py       figures for every stage
  simulate.py       synthetic data with known ground truth
docs/pipeline.md    how the method works, and why
examples/           a single worked run, end to end
```

## Documentation

**[`docs/pipeline.md`](docs/pipeline.md)** explains the method stage by stage: what each
step does, which choices are substantive, and how to read the output. Worth reading before
interpreting results. Every module and public function is also documented in place.

## References

The method draws on:

- Williams et al. (2018), *Neuron* 98(6), 1099–1115 — tensor decomposition of neural data
- Monti et al. (2003), *Machine Learning* 52, 91–118 — consensus clustering
- Rolls et al. (2015), *NeuroImage* 122, 1–5 — the AAL2 atlas
- Kossaifi et al. (2019), *JMLR* 20(26), 1–6 — TensorLy

## Licence

MIT. The AAL2 atlas is not included and carries its own terms.
