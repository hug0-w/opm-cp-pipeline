"""opm-cp — non-negative CP decomposition of source-space MEG time-frequency power,
with cross-subject component clustering.

The pipeline runs in five stages, and the modules map one-to-one onto them:

    io          read the MATLAB time-frequency files (and the AAL2 atlas)
    tensors     select ROIs, crop edges, split the signed tensor into ERS and ERD
    decomposition   fit a non-negative CP model per subject, per tensor type
    similarity  score every pair of components across subjects, tolerating latency shifts
    clustering  group matching components into cross-subject clusters

`plotting` covers all five, and `simulate` generates data with known ground truth so the
whole thing runs without any recordings. See docs/pipeline.md for the method, and
examples/ for a single worked run.

The names below are the public API; everything is also reachable through its own module.
"""

from .io import (Atlas, check_atlas_coverage, describe_tensor, load_aal_atlas,
                 load_tensor, load_tensor_continuous)
from .tensors import Dataset, build_dataset, select_rois, split_ers_erd
from .decomposition import (as_tensor, cosine_similarity_score, decompose_dataset,
                            extract_factors, get_similarity_matrix, r_squared, rel_error,
                            run_cp_nn, run_cp_semi_nn, rank_selection,
                            rank_stability_and_r2, similarity_score)
from .similarity import (ComponentSet, Similarity, SimilarityConfig, best_time_lag,
                         build_component_set, compute_similarity, cosine,
                         shift_onto_reference, time_xcorr_sim)
from .clustering import (align_components, cluster_blocks, cluster_ids, cluster_order,
                         cluster_quality, consensus_clustering, format_clusters,
                         optimal_constrained_clustering, relabel_by_size)
from . import plotting, simulate

__version__ = '0.1.0'

__all__ = [
    # io
    'Atlas', 'check_atlas_coverage', 'describe_tensor', 'load_aal_atlas', 'load_tensor',
    'load_tensor_continuous',
    # tensors
    'Dataset', 'build_dataset', 'select_rois', 'split_ers_erd',
    # decomposition
    'as_tensor', 'cosine_similarity_score', 'decompose_dataset', 'extract_factors',
    'get_similarity_matrix', 'r_squared', 'rel_error', 'run_cp_nn', 'run_cp_semi_nn',
    'rank_selection', 'rank_stability_and_r2', 'similarity_score',
    # similarity
    'ComponentSet', 'Similarity', 'SimilarityConfig', 'best_time_lag',
    'build_component_set', 'compute_similarity', 'cosine', 'shift_onto_reference',
    'time_xcorr_sim',
    # clustering
    'align_components', 'cluster_blocks', 'cluster_ids', 'cluster_order',
    'cluster_quality', 'consensus_clustering', 'format_clusters',
    'optimal_constrained_clustering', 'relabel_by_size',
    # modules
    'plotting', 'simulate', '__version__',
]
