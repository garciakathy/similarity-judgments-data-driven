"""
Replicates Hebart et al. (2020, Nature Human Behaviour) "Reproducibility of
embedding dimensions" procedure exactly, as an alternative to this project's
stricter reciprocal-best-match / bootstrap analysis in
dimension_recoverability.py.

Hebart et al., Methods:
  "...we re-ran the model 20 times, each time with a different random
  initialization. Next, we correlated each of the 49 original dimensions with
  all dimensions of one of the 20 reference embeddings and chose the
  best-fitting dimension across all correlations as the closest match. Then,
  we applied a Fisher z-transform to the correlations, averaged them across
  all 20 reference embeddings and inverted the Fisher z-transform to get a
  mean reliability for each dimension across all 20 embeddings."
  Result: "Pearson r > 0.9 in 34/49 dimensions, Pearson r > 0.6 in 46/49
  dimensions."

Three differences from dimension_recoverability.py's primary analysis, all
of which make that one stricter than this replication:
  1. Seed-only (no bootstrap / data-perturbation arm) -- Hebart's test is
     stability to optimisation noise alone, not to resampled data.
  2. Plain max-correlation match per reference dim, independently, with NO
     uniqueness constraint -- not Hungarian 1-to-1 assignment and not
     reciprocal-best-match. Two reference dims may legitimately claim the
     same refit column as their best match; there is no merge penalty.
  3. Continuous Fisher-z-averaged mean reliability per dim, not a
     %-of-refits->=threshold rate.

`--mode seed` (default) reproduces Hebart's own test exactly (point 1 also
matches theirs). `--mode bootstrap` isolates points 2+3 from point 1: it
applies Hebart's matching rule (plain max-correlation) and aggregation
(Fisher-z mean) to this project's own, harsher bootstrap refits, so any gap
against the seed numbers is attributable to data-resampling instability
alone, not to the merge penalty or the binary threshold used in
dimension_recoverability.py's primary analysis. `--mode both` runs and
compares both.

Reuses the cached seed/bootstrap refits from dimension_recoverability.py
(seed_models.npz / bootstrap_models.npz) -- no retraining. Run
dimension_recoverability.py first if those caches do not exist yet.

Outputs (data/interim/results/stability/recoverability/):
  * recoverability_hebart_replication.csv -- per-dim mean reliability
    (both modes, when --mode both)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_DIR, DIM_LABELS, BEST_DIMS
from analysis.stability.dimension_recoverability import (
    OUT_DIR, SEED_MODELS_CACHE, BOOT_MODELS_CACHE, _corr_cross, SEED_LIST,
)


def hebart_reliability(reference, models):
    """Per-dim Fisher-z-averaged max-correlation reliability, Hebart-style."""
    n_ref = reference.shape[1]
    per_seed_r = np.zeros((len(SEED_LIST), n_ref))
    for i, s in enumerate(SEED_LIST):
        R = _corr_cross(reference, models[s])   # (n_ref, k_rep)
        per_seed_r[i] = R.max(axis=1)           # plain max, no uniqueness constraint
    z = np.arctanh(np.clip(per_seed_r, -0.999999, 0.999999))
    mean_r = np.tanh(z.mean(axis=0))
    return mean_r, per_seed_r


def _print_summary(label, df, col):
    best = df[df['is_best_dim']]
    other = df[~df['is_best_dim']]
    n_dim = len(df)
    print(f"\n--- Summary ({label}) ---")
    print(f"All {n_dim} dims:      r>0.9: {(df[col] > 0.9).sum()}/{n_dim}, "
          f"r>0.6: {(df[col] > 0.6).sum()}/{n_dim}, mean r = {df[col].mean():.3f}")
    print(f"BEST_DIMS (n=11):   r>0.9: {(best[col] > 0.9).sum()}/11, "
          f"r>0.6: {(best[col] > 0.6).sum()}/11, mean r = {best[col].mean():.3f}")
    print(f"Other {len(other)} dims:     r>0.9: {(other[col] > 0.9).sum()}/{len(other)}, "
          f"r>0.6: {(other[col] > 0.6).sum()}/{len(other)}, mean r = {other[col].mean():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['seed', 'bootstrap', 'both'], default='seed',
                        help="seed = Hebart's own procedure exactly (default); "
                             "bootstrap = Hebart's matching+aggregation applied to this "
                             "project's data-perturbation refits, isolating point 1; "
                             "both = run and compare both.")
    args = parser.parse_args()

    reference = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    n_dim = reference.shape[1]
    print(f"Reference embedding: {reference.shape} ({n_dim} dims; Hebart's was 49)")

    dfs = {}
    if args.mode in ('seed', 'both'):
        if not os.path.exists(SEED_MODELS_CACHE):
            raise FileNotFoundError(
                f"{SEED_MODELS_CACHE} not found -- run dimension_recoverability.py first."
            )
        print(f"[seed] {SEED_LIST}")
        seed_models = np.load(SEED_MODELS_CACHE, allow_pickle=True)['models'].item()
        mean_r, _ = hebart_reliability(reference, seed_models)
        dfs['seed'] = mean_r

    if args.mode in ('bootstrap', 'both'):
        if not os.path.exists(BOOT_MODELS_CACHE):
            raise FileNotFoundError(
                f"{BOOT_MODELS_CACHE} not found -- run dimension_recoverability.py first."
            )
        print(f"[bootstrap] {SEED_LIST} (triplets resampled with replacement)")
        boot_models = np.load(BOOT_MODELS_CACHE, allow_pickle=True)['models'].item()
        mean_r, _ = hebart_reliability(reference, boot_models)
        dfs['bootstrap'] = mean_r

    df = pd.DataFrame({
        'dim': np.arange(n_dim),
        'label': [DIM_LABELS.get(d, '') for d in range(n_dim)],
        'is_best_dim': [d in BEST_DIMS for d in range(n_dim)],
    })
    for mode, mean_r in dfs.items():
        df[f'hebart_mean_reliability_{mode}'] = mean_r
    sort_col = f'hebart_mean_reliability_{"bootstrap" if "bootstrap" in dfs else "seed"}'
    df = df.sort_values(sort_col, ascending=False)

    out_csv = os.path.join(OUT_DIR, 'recoverability_hebart_replication.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}\n")

    pd.set_option('display.width', 160)
    pd.set_option('display.max_rows', 50)
    print(df.to_string(index=False))

    for mode in dfs:
        _print_summary(mode, df, f'hebart_mean_reliability_{mode}')

    if 'seed' in dfs and 'bootstrap' in dfs:
        best = df[df['is_best_dim']]
        gap = best['hebart_mean_reliability_seed'].mean() - best['hebart_mean_reliability_bootstrap'].mean()
        print(f"\nBEST_DIMS mean r, seed vs bootstrap (same matching/aggregation "
              f"both sides): {best['hebart_mean_reliability_seed'].mean():.3f} vs "
              f"{best['hebart_mean_reliability_bootstrap'].mean():.3f} (gap = {gap:.3f}) "
              f"-- isolates the effect of data-perturbation (point 1) alone.")


if __name__ == '__main__':
    main()
