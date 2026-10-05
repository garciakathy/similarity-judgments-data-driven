"""
Stimulus-level (video) bootstrap for cross-validated RSA regressions.

The unit of resampling is the video, not the pair: pairs that share a video are
dependent, so resampling the 1,225 pair cells would understate uncertainty.
Each draw resamples n videos with replacement and keeps every pair of two
*different* original videos (pairs of a video with its own duplicate are
dropped, as in rsatoolbox's stimulus bootstrap).

Duplicated videos produce identical pair rows. With plain KFold over pairs one
copy could be trained on and the other tested, inflating r^2, so the CV folds
here are grouped by original pair id: all copies of a pair land in one fold.
Otherwise the estimator is the project's: OLS with intercept, out-of-fold
predictions, Spearman r with the target, squared, averaged over repeats.

Used by explained_variance_shared_only.py --ci (Fig. 2).
"""

import numpy as np
from scipy.stats import rankdata


def pair_lookup(n):
    """(n, n) array: entry [i, j] (i < j) is that pair's index in triu order."""
    lut = -np.ones((n, n), dtype=np.int64)
    iu = np.triu_indices(n, k=1)
    lut[iu] = np.arange(len(iu[0]))
    return lut


def resample_pairs(n, rng, lut=None):
    """One video-bootstrap draw. Returns original-pair indices (with repeats)."""
    lut = pair_lookup(n) if lut is None else lut
    idx = rng.integers(0, n, n)
    a, b = np.triu_indices(n, k=1)
    i, j = idx[a], idx[b]
    keep = i != j
    i, j = i[keep], j[keep]
    return lut[np.minimum(i, j), np.maximum(i, j)]


def _spearman_r(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float(ra @ rb / d) if d > 0 else 0.0


def grouped_folds(groups, n_folds, seed):
    """Fold label per row, assigning whole groups to folds at random."""
    ug, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    fold_of_group = np.empty(len(ug), dtype=np.int64)
    fold_of_group[rng.permutation(len(ug))] = np.arange(len(ug)) % n_folds
    return fold_of_group[inv]


def cv_spearman_r2(X, y, groups=None, n_folds=10, n_repeats=5, seed=0,
                   folds=None):
    """Grouped-CV Spearman r^2 of out-of-fold OLS predictions.

    groups=None treats every row as its own group (plain shuffled K-fold).
    folds: optional precomputed list of fold-label arrays (one per repeat),
    so many models on the same rows share splits.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float)
    if groups is None:
        groups = np.arange(len(y))
    if folds is None:
        folds = [grouped_folds(groups, n_folds, seed + r) for r in range(n_repeats)]
    vals = []
    for fold in folds:
        pred = np.empty_like(y)
        for f in range(n_folds):
            te = fold == f
            tr = ~te
            xm, ym = X[tr].mean(axis=0), y[tr].mean()
            beta = np.linalg.lstsq(X[tr] - xm, y[tr] - ym, rcond=None)[0]
            pred[te] = (X[te] - xm) @ beta + ym
        vals.append(_spearman_r(y, pred) ** 2)
    return float(np.mean(vals))


def percentile_ci(samples, level=0.95):
    """Percentile interval along axis 0."""
    a = (1 - level) / 2
    return (np.nanpercentile(samples, 100 * a, axis=0),
            np.nanpercentile(samples, 100 * (1 - a), axis=0))
