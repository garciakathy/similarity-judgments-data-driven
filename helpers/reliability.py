"""
Noise ceilings for the human similarity RSMs.

One implementation, shared by `dim_selection_train.py` and
`explained_variance.py`, so the train and test ceilings in different figures
are always the same numbers.

Triplet conventions in this project differ by file, and mixing them up fails
silently (see `build_rsm`), so everything here works in one canonical format:
`(i, j, k, chosen_pair_idx)`, where `chosen_pair_idx` names the SIMILAR pair.
Use `load_train_triplets` / `load_test_triplets` and you get that for free.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

N_STIM = 250


def drop_degenerate(trip):
    """Skip triplets that repeat a stimulus, as `build_rsms_and_counts` does."""
    i, j, k = trip[:, 0], trip[:, 1], trip[:, 2]
    return trip[(i != j) & (i != k) & (j != k)]


def odd_one_out_to_pair_idx(trip):
    """(i, j, k, odd-one-out stim id) -> (i, j, k, chosen_pair_idx)."""
    i, j, k, ch = (trip[:, c] for c in range(4))
    return np.column_stack([i, j, k, np.where(ch == i, 2, np.where(ch == j, 1, 0))])


def build_rsm(trip, n=N_STIM, counts=False):
    """
    Vectorised similarity RSM from triplets in (i, j, k, chosen_pair_idx) form,
    where chosen_pair_idx names the SIMILAR pair: 0 -> (i,j), 1 -> (i,k), 2 -> (j,k).

    This is the on-disk convention of `train_triplets.npy`, and it is NOT the
    odd-one-out's position or stimulus id. Feeding an odd-one-out index in here
    produces an RSM that still has a plausible marginal mean but correlates
    r ~= 0.01 with `train_human_sim.npy`, so it fails quietly — assert against
    the on-disk RSM in new code.
    """
    i, j, k, cp = (trip[:, c] for c in range(4))
    total = np.zeros((n, n))
    similar = np.zeros((n, n))
    for a, b in ((i, j), (i, k), (j, k)):
        np.add.at(total, (a, b), 1)
        np.add.at(total, (b, a), 1)
    sa = np.where(cp == 2, j, i)
    sb = np.where(cp == 0, j, k)
    np.add.at(similar, (sa, sb), 1)
    np.add.at(similar, (sb, sa), 1)
    rsm = np.divide(similar, total, out=np.zeros_like(similar), where=total > 0)
    return (rsm, total) if counts else rsm


def load_train_triplets(utils_dir):
    """Training triplets in canonical form. Reproduces `train_human_sim.npy`."""
    arr = np.load(os.path.join(utils_dir, 'train_triplets.npy')).astype(int)
    assert arr[:, 3].max() <= 2, (
        f"train_triplets.npy col 3 max={arr[:, 3].max()}, expected chosen_pair_idx in [0,2]"
    )
    return drop_degenerate(arr)


def load_test_triplets(utils_dir):
    """Dense test triplets in canonical form (the CSV stores an odd-one-out id)."""
    df = pd.read_csv(os.path.join(utils_dir, 'test_triplets.csv'))
    cols = ['stim1_name', 'stim2_name', 'stim3_name', 'choice']
    return odd_one_out_to_pair_idx(drop_degenerate(df[cols].to_numpy().astype(int)))


def split_half_ceiling(trip, ids, n_splits=200, seed=888, n=N_STIM, metric='pearson'):
    """
    Spearman-Brown corrected split-half reliability of the RSM built from
    `trip`, restricted to stimuli `ids`. Bounds the R² any predictor of that
    RSM can reach.

    Returns (mean, (ci_lo, ci_hi)) over `n_splits` random triplet halvings.

    `metric` must match the metric of the curve being bounded — a ceiling has
    to be in the units of the quantity it bounds. The project's published
    0.6658 test ceiling is the 'spearman' variant; Pearson R² curves need
    'pearson'.

    Cells unobserved in a half are excluded (equivalent to
    `min_observations=1` in split_half_reliability_corrected.py). Keeping them
    would score imputed zeros as data: at half density 11% of train cells are
    unobserved versus 0.3% at full density, which drags the train estimate from
    0.50 down to 0.42.
    """
    corr = {'pearson': lambda a, b: pearsonr(a, b)[0],
            'spearman': lambda a, b: spearmanr(a, b).statistic}[metric]
    rng = np.random.default_rng(seed)
    triu = np.triu_indices(len(ids), k=1)
    sel = np.ix_(ids, ids)
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(len(trip))
        ra, ca = build_rsm(trip[perm[:len(trip) // 2]], n=n, counts=True)
        rb, cb = build_rsm(trip[perm[len(trip) // 2:]], n=n, counts=True)
        a, b = ra[sel][triu], rb[sel][triu]
        both = (ca[sel][triu] > 0) & (cb[sel][triu] > 0)
        r = corr(a[both], b[both])
        vals.append(2 * r / (1 + r))          # Spearman-Brown
    vals = np.asarray(vals)
    return vals.mean(), tuple(np.percentile(vals, [2.5, 97.5]))
