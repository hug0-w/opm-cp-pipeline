"""Non-negative CP (PARAFAC) decomposition and the diagnostics used to choose a rank.

A rank-R CP model writes an (ROI, Freq, Time) tensor as a sum of R rank-one terms,
each the outer product of one ROI loading vector, one frequency profile and one time
course. With every factor constrained non-negative the terms are additive and read
directly as "this much power, in these parcels, at these frequencies, at these times".

Two facts about the fit drive everything in this module:

- **The objective is non-convex**, so a single run lands in whatever local minimum its
  random initialisation leads to. Every fit here is therefore best-of-N restarts.
- **Rank is not identifiable from fit alone.** R² rises monotonically with rank, so it
  cannot pick a rank on its own. `rank_selection` pairs it with the stability of
  Williams et al. (2018): refit the same rank from many initialisations and measure how
  reproducible the recovered components are. Stability collapses once the rank exceeds
  what the data support, and the chosen rank is the largest one that is still both
  stable and explains a useful share of the variance.
"""

import numpy as np
import torch
import tensorly as tl
from scipy.optimize import linear_sum_assignment
from tensorly.cp_tensor import cp_normalize
from tensorly.decomposition import (constrained_parafac, non_negative_parafac)

tl.set_backend('pytorch')
torch.set_default_dtype(torch.float32)


def as_tensor(array):
    """NumPy array -> backend tensor of the working dtype, ready to decompose."""
    return tl.tensor(np.ascontiguousarray(array, dtype=np.float32))


def to_numpy(x):
    """Backend tensor (or anything array-like) -> NumPy array."""
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.asarray(x)


# ── Fit quality ──────────────────────────────────────────────────────────────

def rel_error(tensor_data, cp_tensor):
    """Relative reconstruction error, ||X - X̂|| / ||X||."""
    reconstructed = tl.cp_to_tensor(cp_tensor)
    return torch.norm(tensor_data - reconstructed) / torch.norm(tensor_data)


def r_squared(tensor_data, cp_tensor):
    """Variance explained by the model, 1 - SS_res / SS_tot."""
    reconstructed = tl.cp_to_tensor(cp_tensor)
    ss_res = torch.sum((tensor_data - reconstructed) ** 2)
    ss_tot = torch.sum((tensor_data - torch.mean(tensor_data)) ** 2)
    return (1 - (ss_res / ss_tot)).item()


# ── Model-to-model similarity (Williams et al. 2018) ─────────────────────────

def _normalised_factors(cp_tensor):
    """Weights and factors of a CP model as float64 NumPy, with unit-norm columns."""
    weights, factors = cp_normalize(cp_tensor)
    return to_numpy(weights).astype(np.float64), \
        [to_numpy(f).astype(np.float64) for f in factors]


def get_similarity_matrix(cp_tensor_A, cp_tensor_B, weight_term=True):
    """Component-by-component similarity between two CP models.

    Entry (i, j) is the product of the per-mode cosine similarities of component i of A
    and component j of B — a component only scores highly if it matches in *every* mode.
    With `weight_term` the product is further scaled by how close the two components'
    weights are, which is what distinguishes a shared dominant component from a shared
    negligible one.

    Both models must have the same number of modes; their ranks may differ.
    """
    lambdaA, factorsA = _normalised_factors(cp_tensor_A)
    lambdaB, factorsB = _normalised_factors(cp_tensor_B)

    rank_A, rank_B = factorsA[0].shape[1], factorsB[0].shape[1]
    n_modes = len(factorsA)
    sim_matrix = np.zeros((rank_A, rank_B), dtype=np.float64)

    for i in range(rank_A):
        for j in range(rank_B):
            if weight_term:
                max_lam = max(lambdaA[i], lambdaB[j])
                score = 1.0 if max_lam == 0 else \
                    1.0 - abs(lambdaA[i] - lambdaB[j]) / max_lam
            else:
                score = 1.0

            for mode in range(n_modes):
                vec_a, vec_b = factorsA[mode][:, i], factorsB[mode][:, j]
                na, nb = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
                # Non-negative factors give non-negative cosines, so no abs() is needed
                # here — unlike unconstrained CP, where sign is arbitrary per component.
                score *= float(vec_a @ vec_b / (na * nb)) if na > 0 and nb > 0 else 0.0

            sim_matrix[i, j] = score
    return sim_matrix


def similarity_score(cp_tensor_A, cp_tensor_B, weight_term=True):
    """Whole-model similarity in [0, 1]: equation 14 of Williams et al. (2018).

    CP components come out in arbitrary order, so the two models are first matched
    one-to-one by the Hungarian algorithm (maximising total similarity) and the score is
    the mean similarity of the matched pairs. `weight_term=False` drops the component-
    magnitude term and compares factor shapes only.
    """
    sim_matrix = get_similarity_matrix(cp_tensor_A, cp_tensor_B, weight_term=weight_term)
    rows, cols = linear_sum_assignment(sim_matrix, maximize=True)
    return float(sim_matrix[rows, cols].mean())


def cosine_similarity_score(cp_tensor_A, cp_tensor_B):
    """`similarity_score` without the component-magnitude term — shapes only."""
    return similarity_score(cp_tensor_A, cp_tensor_B, weight_term=False)


# ── Fitting ──────────────────────────────────────────────────────────────────

def run_parafac(tensor_data, rank, random_state=None, n_iter_max=2000, tol=1e-8,
                init='random'):
    """A single non-negative CP fit. Thin wrapper over tensorly, kept so that the fit
    options used across this package are defined in one place."""
    return non_negative_parafac(
        tensor_data, rank=rank, init=init, n_iter_max=n_iter_max, tol=tol,
        random_state=random_state, normalize_factors=True,
    )


def run_cp_nn(tensor, rank, n_runs=25, n_iter_max=10000, tol=1e-10, verbose=True):
    """Non-negative CP, best of `n_runs` random initialisations.

    "Best" is lowest reconstruction error. Restarts are the main defence against the
    non-convex objective: with too few, a reported component may be an artefact of one
    unlucky starting point. Seeds are 0..n_runs-1, so a given call is reproducible.
    """
    best_err, best_cp = float('inf'), None
    for i in range(n_runs):
        cp_result = run_parafac(tensor, rank=rank, random_state=i,
                                n_iter_max=n_iter_max, tol=tol)
        err = tl.norm(tensor - tl.cp_to_tensor(cp_result)).item()
        if err < best_err:
            best_err, best_cp = err, cp_result
        if verbose and (i + 1) % 5 == 0:
            print(f'  Run {i + 1}/{n_runs}: best error = {best_err:.6f}')

    if verbose:
        print(f'  Final R² = {r_squared(tensor, best_cp):.4f}')
    return best_cp


def run_cp_semi_nn(tensor, rank, nn_modes, smoothness=None, n_runs=25, n_iter_max=10000,
                   tol_outer=1e-10, tol_inner=1e-6, verbose=True):
    """Partially non-negative CP, best of `n_runs` initialisations.

    Non-negativity applies only to the modes in `nn_modes`; the rest are left free. The
    use for this is an unsplit signed tensor, where leaving the *time* mode free lets one
    component carry a power decrease followed by a rebound rather than needing two. Note
    this gives up the ERS/ERD separation that `tensors.split_ers_erd` buys, so it is an
    alternative to the split pipeline, not a step in it.

    `smoothness` penalises sample-to-sample differences within a factor, e.g.
    `{2: 1.0}` smooths mode 2. Larger values smooth more, at a small cost in R².
    """
    nn_dict = {m: True for m in nn_modes}
    best_err, best_cp = float('inf'), None
    for i in range(n_runs):
        cp_result = constrained_parafac(
            tensor, rank=rank, non_negative=nn_dict, smoothness=smoothness,
            n_iter_max=n_iter_max, n_iter_max_inner=10, init='random',
            tol_outer=tol_outer, tol_inner=tol_inner, random_state=i,
        )
        err = tl.norm(tensor - tl.cp_to_tensor(cp_result)).item()
        if err < best_err:
            best_err, best_cp = err, cp_result
        if verbose and (i + 1) % 5 == 0:
            print(f'  Run {i + 1}/{n_runs}: best error = {best_err:.6f}')

    if verbose:
        print(f'  Final R² = {r_squared(tensor, best_cp):.4f}')
    return best_cp


def extract_factors(cp_tensor, sort=True):
    """CP factors as a list of NumPy (n_bins, rank) arrays, one per mode.

    With `sort`, components are reordered by descending weight, so component 0 is the
    largest contributor in every model. The downstream clustering does not depend on the
    order, but a consistent one makes per-subject figures comparable by eye.
    """
    factors = [to_numpy(f) for f in cp_tensor[1]]
    if not sort:
        return factors
    # lambda_r, the scale of rank-one term r — the product of its columns' norms, which
    # is what cp_normalize would pull out as the weight.
    lam = np.prod([np.linalg.norm(f, axis=0) for f in factors], axis=0)
    order = np.argsort(lam)[::-1]
    return [f[:, order] for f in factors]


# ── Rank selection ───────────────────────────────────────────────────────────

def rank_stability_and_r2(tensor_data, rank, n_repeats=10, verbose=0):
    """Stability and R² at one rank, following Williams et al. (2018).

    Fits the rank `n_repeats` times from different initialisations, takes the lowest-error
    fit as the reference, and scores every other fit against it with `similarity_score`.
    Stability is the mean of those scores: ~1 means every restart found the same
    components, low means the rank is not reproducibly identifiable in this tensor.

    Returns (mean_stability, std_stability, best_r2, similarities, errors, r2_scores).
    """
    if tensor_data.dtype != torch.float32:
        tensor_data = tensor_data.to(torch.float32)
    if verbose:
        print(f'--- Testing Rank {rank} with {n_repeats} repeats ---')

    models, errors, r2_scores = [], [], []
    for i in range(n_repeats):
        try:
            cp_tensor = run_parafac(tensor_data, rank=rank, random_state=i,
                                    tol=1e-6, n_iter_max=1500)
            errors.append(float(to_numpy(rel_error(tensor_data, cp_tensor))))
            r2_scores.append(r_squared(tensor_data, cp_tensor))
            # Keep the normalised model on the CPU: the comparisons below are NumPy, and
            # holding n_repeats models on an accelerator is pointless memory pressure.
            norm_weights, norm_factors = cp_normalize(cp_tensor)
            models.append((
                norm_weights.detach().cpu() if hasattr(norm_weights, 'cpu') else norm_weights,
                [f.detach().cpu() if hasattr(f, 'cpu') else f for f in norm_factors],
            ))
        except Exception as e:
            # A failed restart is information (some ranks are ill-conditioned), not a
            # reason to abandon the sweep — record it and carry on.
            if verbose:
                print(f'  Run {i} failed: {e}')
            errors.append(np.inf)
            r2_scores.append(np.nan)
            models.append(None)

    errors, r2_scores = np.array(errors), np.array(r2_scores)
    valid = np.where(np.isfinite(errors))[0]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0, [], [], []

    best_idx = valid[np.argmin(errors[valid])]
    similarities = [similarity_score(models[best_idx], models[i])
                    for i in valid if i != best_idx]
    mean_stability = float(np.mean(similarities)) if similarities else np.nan
    std_stability = float(np.std(similarities)) if similarities else np.nan

    if verbose:
        print(f'  {len(valid)} models converged')
        print(f'  Rank {rank}: Best Error={errors[best_idx]:.4f} | '
              f'Best R²={r2_scores[best_idx]:.4f}')
        print(f'  Mean Stability={mean_stability:.4f} ± {std_stability:.4f}')

    return (mean_stability, std_stability, float(r2_scores[best_idx]), similarities,
            errors[valid].tolist(), r2_scores[valid].tolist())


def rank_selection(tensor_data, ranks=range(1, 11), n_repeats=10, verbose=1):
    """Sweep `rank_stability_and_r2` over `ranks`.

    Returns a dict keyed 'ranks', 'mean_stabilities', 'std_stabilities', 'best_r2' and
    the per-rank raw values, ready for `plotting.plot_rank_selection`.
    """
    results = {'ranks': list(ranks), 'mean_stabilities': [], 'std_stabilities': [],
               'best_r2': [], 'all_similarities': {}, 'all_errors': {}, 'all_r2': {}}
    for rank in results['ranks']:
        if verbose:
            print(f'\n=== Rank {rank} ===')
        stab, stab_std, best_r2, sims, errs, r2s = rank_stability_and_r2(
            tensor_data, rank, n_repeats=n_repeats, verbose=verbose)
        results['mean_stabilities'].append(stab)
        results['std_stabilities'].append(stab_std)
        results['best_r2'].append(best_r2)
        results['all_similarities'][rank] = sims
        results['all_errors'][rank] = errs
        results['all_r2'][rank] = r2s
    return results


def decompose_dataset(dataset, rank, n_runs=25, n_iter_max=10000, verbose=True):
    """Fit one non-negative CP model per (tensor type, subject) in a `tensors.Dataset`.

    Each subject is decomposed independently — no spatial or temporal correspondence
    between subjects is assumed at this stage. Establishing that correspondence is the
    job of `similarity` and `clustering`, which work on the factors returned here.

    Returns (cp_models, factors), both {type: {subject: ...}}.
    """
    cp_models = {t: {} for t in dataset.types}
    factors = {t: {} for t in dataset.types}
    for t in dataset.types:
        for s in dataset.subjects:
            if verbose:
                print(f'\n=== Subject {s} | {t} ===')
            tensor = as_tensor(dataset.tensors[t][s])
            cp_models[t][s] = run_cp_nn(tensor, rank=rank, n_runs=n_runs,
                                        n_iter_max=n_iter_max, verbose=verbose)
            factors[t][s] = extract_factors(cp_models[t][s])
    return cp_models, factors
