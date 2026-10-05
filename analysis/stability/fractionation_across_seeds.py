"""
For each of the 20 bootstrap refits, independently select that refit's own
top-11 dimensions by greedy forward CV against the held-out test RSM
(analysis.dimensions.refit_contrastive_grid.greedy_select_by_test_r2 -- no reference to
BEST_DIMS, no duplicates possible by construction). Then check: within that
refit's own top-11, are there >=2 columns whose single best-correlating
reference BEST_DIM is the SAME one? That is a candidate fractionation event
-- the refit needed more than one of its own axes to cover ground the
reference embedding covers with one labeled dimension.

VALIDATION (separate from classification): for every classified pair we also
compute shared top-20 videos, shared top-10 contrastive words, and the pair's
own Pearson r (analysis.dimensions.refit_contrastive_grid.fractionation_stats), and flag
`validated=True` when the pair sits closer to the Dim0/Dim4
example (r=0.60, 3/20 shared videos, 3/10 shared words) than to a true-merge
control (identical column: r=1.0, 20/20, 10/10) on ALL THREE statistics,
each at the integer/value closest to the midpoint of its two anchors:
  - shared_videos <= 11   (midpoint of 3 and 20 is 11.5; 12 is actually
    closer to 20 than to 3, so 11 -- not 12 -- is the correct cutoff)
  - shared_words  <= 6    (midpoint of 3 and 10 is 6.5 -> 6)
  - r             < 0.80  (exact midpoint of 0.60 and 1.0)
Reported counts (the figure, "N/20 seeds") are the VALIDATED counts, i.e.
classification AND validation both required -- not classification alone.
None of these three cutoffs currently excludes anything: the observed
maxima across all 41 classified pairs are shared_videos=7, shared_words=6,
r=0.47, all well inside the "closer to fractionation" region.

Caches each seed's full greedy_select_by_test_r2 table (with closest_dim_r)
to greedy_top11_by_seed.csv, so retuning MIN_CLOSEST_R does not require
re-running the ~4.5-minute, 20-seed greedy CV selection.

No retraining -- reuses the cached bootstrap refits from
dimension_recoverability.py.

Outputs (data/interim/results/stability/recoverability/):
  * greedy_top11_by_seed.csv -- every seed's own top-11 selection, cached
  * fractionation_across_seeds.csv -- one row per CLASSIFIED candidate pair,
    all seeds, with a `validated` column; plotting/reporting use validated
    rows only (see plot_fractionation_summary.py)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_DIR, DIM_LABELS
from analysis.stability.dimension_recoverability import OUT_DIR, BOOT_MODELS_CACHE, SEED_LIST
from analysis.dimensions.refit_contrastive_grid import (
    load_corpus, greedy_select_by_test_r2, fractionation_stats,
)

MIN_CLOSEST_R = 0.30      # classification floor (~99.7th pct of its own null; see docstring)
VALIDATION_MAX_SHARED_VIDEOS = 11   # nearest integer closer to 3 than to 20 (midpoint 11.5)
VALIDATION_MAX_SHARED_WORDS = 6     # nearest integer closer to 3 than to 10 (midpoint 6.5)
VALIDATION_MAX_R = 0.80             # exact midpoint of 0.60 and 1.0
SELECTION_CACHE = os.path.join(OUT_DIR, 'greedy_top11_by_seed.csv')


def get_selection_table(refresh=False):
    if os.path.exists(SELECTION_CACHE) and not refresh:
        print(f"Loading cached selections: {SELECTION_CACHE}")
        return pd.read_csv(SELECTION_CACHE)

    boot_models = np.load(BOOT_MODELS_CACHE, allow_pickle=True)['models'].item()
    tables = []
    for seed in SEED_LIST:
        sel = greedy_select_by_test_r2(boot_models[seed], n_select=11, verbose=False)
        sel = sel[['refit_col', 'cumulative_test_r2', 'marginal_r2',
                  'closest_dim_id', 'closest_dim_r']].copy()
        sel.insert(0, 'seed', seed)
        tables.append(sel)
        print(f"seed {seed}: selection done")
    full = pd.concat(tables, ignore_index=True)
    full.to_csv(SELECTION_CACHE, index=False)
    print(f"Saved: {SELECTION_CACHE}")
    return full


def main():
    boot_models = np.load(BOOT_MODELS_CACHE, allow_pickle=True)['models'].item()
    selection_df = get_selection_table()

    print(f"\nclosest_dim_r distribution across all {len(selection_df)} "
          f"(seed, selected-column) rows:")
    print(selection_df['closest_dim_r'].describe().to_string())
    print(f"\nApplying MIN_CLOSEST_R={MIN_CLOSEST_R} floor before grouping "
          f"({(selection_df['closest_dim_r'] < MIN_CLOSEST_R).sum()} of "
          f"{len(selection_df)} rows excluded as noise matches)")

    print("\nBuilding caption corpus...")
    corpus, _ = load_corpus()

    rows = []
    for seed in SEED_LIST:
        refit_emb = boot_models[seed]
        sel = selection_df[selection_df['seed'] == seed]
        sel = sel[sel['closest_dim_r'] >= MIN_CLOSEST_R]

        groups = sel.groupby('closest_dim_id')['refit_col'].apply(list)
        candidate_groups = {int(d): cols for d, cols in groups.items() if len(cols) >= 2}

        if not candidate_groups:
            rows.append({'seed': seed, 'ref_dim': None, 'label': None,
                        'col_a': None, 'col_b': None, 'r': None,
                        'shared_videos': None, 'shared_words': None,
                        'validated': False})
            print(f"seed {seed}: no candidate fractionation groups above floor")
            continue

        for ref_dim, cols in candidate_groups.items():
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    a, b = cols[i], cols[j]
                    stats = fractionation_stats(refit_emb, a, b, corpus)
                    validated = (stats['shared_videos'] <= VALIDATION_MAX_SHARED_VIDEOS
                                and stats['shared_words'] <= VALIDATION_MAX_SHARED_WORDS
                                and stats['r'] < VALIDATION_MAX_R)
                    rows.append({
                        'seed': seed, 'ref_dim': ref_dim,
                        'label': DIM_LABELS.get(ref_dim, ''),
                        'col_a': a, 'col_b': b, 'r': stats['r'],
                        'shared_videos': stats['shared_videos'],
                        'shared_words': stats['shared_words'],
                        'validated': validated,
                    })
                    print(f"seed {seed}: cols {a},{b} both ~Dim {ref_dim} ({DIM_LABELS.get(ref_dim,'')})"
                          f"  r={stats['r']:.3f}  shared_videos={stats['shared_videos']}/20"
                          f"  shared_words={stats['shared_words']}/10"
                          f"  validated={validated}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'fractionation_across_seeds.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    events = df.dropna(subset=['ref_dim'])
    n_seeds_classified = events['seed'].nunique()
    print(f"\nCLASSIFICATION (r>={MIN_CLOSEST_R} floor only): {n_seeds_classified}/{len(SEED_LIST)} "
          f"seeds have >=1 candidate fractionation group; {len(events)} candidate pairs total.")

    if not events.empty:
        validated = events[events['validated']]
        print(f"\nVALIDATED (classification AND shared_videos<={VALIDATION_MAX_SHARED_VIDEOS}/20 "
              f"AND shared_words<={VALIDATION_MAX_SHARED_WORDS}/10 AND r<{VALIDATION_MAX_R}, "
              f"vs. merge control 20/20, 10/10, r=1.0): "
              f"{validated['seed'].nunique()}/{len(SEED_LIST)} seeds, "
              f"{len(validated)}/{len(events)} pairs. Reported figures/counts use this set.")
        print("\nBy reference dim (validated pairs only):")
        print(validated.groupby(['ref_dim', 'label']).agg(
            n_seeds=('seed', 'nunique'),
            mean_shared_videos=('shared_videos', 'mean'),
            mean_shared_words=('shared_words', 'mean'),
            mean_r=('r', 'mean'),
        ).to_string())


if __name__ == '__main__':
    main()
