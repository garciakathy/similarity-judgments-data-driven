"""
Combined stability figure: recoverability (Hebart-style, no merge penalty)
alongside fractionation recurrence.

  a) Recoverability -- Hebart-style mean reliability per BEST_DIMS dimension
     (plain max-correlation, Fisher-z averaged over 20 refits; see
     analysis/stability/dimension_recoverability_hebart.py), seed vs bootstrap, in
     canonical BEST_DIMS order. All 11 clear r=0.6 under this procedure
     (11/11).
  b) Fractionation recurrence -- how many of the 20 bootstrap refits show a
     genuine fractionation group for each reference category (from
     analysis/stability/fractionation_across_seeds.py), split by group size.

Together: the underlying geometry/content is recoverable (a), and separately,
independent refits keep re-discovering that broad categories fractionate
into narrower ones -- just not always along the same seam (b).

Reads cached CSVs only -- no retraining, no re-selection.

Outputs (data/interim/results/stability/recoverability/):
  * fractionation_summary.{png,pdf}
  * fractionation_summary_ci.{png,pdf}  (--ci): panel A with 95% CIs of the
    Fisher-z mean over the 20 refits (t, df=19), recomputed from the cached
    refit models; per-refit values in recoverability_hebart_per_refit.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import BEST_DIMS, DIM_LABELS
from analysis.stability.dimension_recoverability import OUT_DIR, SEED_LIST

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

N_SEEDS = len(SEED_LIST)
R_THRESHOLD = 0.6


def per_refit_reliability():
    """(seed, bootstrap) -> (20, n_dims) per-refit max-correlation r, as in
    dimension_recoverability_hebart.hebart_reliability (whose Fisher-z mean the
    bars show)."""
    from config import MODEL_DIR
    from analysis.stability.dimension_recoverability import SEED_MODELS_CACHE, BOOT_MODELS_CACHE
    from analysis.stability.dimension_recoverability_hebart import hebart_reliability
    reference = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    out = {}
    for mode, cache in (('seed', SEED_MODELS_CACHE), ('bootstrap', BOOT_MODELS_CACHE)):
        models = np.load(cache, allow_pickle=True)['models'].item()
        mean_r, per_refit = hebart_reliability(reference, models)
        out[mode] = (mean_r, per_refit)
    return out


def _z_mean_ci(per_refit, level=0.95):
    """Fisher-z mean and t-interval over refits, back-transformed to r."""
    from scipy.stats import t as t_dist
    z = np.arctanh(np.clip(per_refit, -0.999999, 0.999999))
    n = z.shape[0]
    m, se = z.mean(axis=0), z.std(axis=0, ddof=1) / np.sqrt(n)
    h = t_dist.ppf(0.5 + level / 2, n - 1) * se
    return np.tanh(m), np.tanh(m - h), np.tanh(m + h)


def plot_recoverability(ax, ci=None):
    csv_path = os.path.join(OUT_DIR, 'recoverability_hebart_replication.csv')
    df = pd.read_csv(csv_path).set_index('dim')
    df = df.loc[list(BEST_DIMS)]   # canonical BEST_DIMS order

    labels = [f"{d}: {DIM_LABELS.get(d, '')}" for d in BEST_DIMS]
    x = np.arange(len(BEST_DIMS))
    width = 0.38
    err = {}
    if ci is not None:
        # asymmetric error bars in r units, from the Fisher-z t-interval
        for mode in ('seed', 'bootstrap'):
            m, lo, hi = (a[list(BEST_DIMS)] for a in _z_mean_ci(ci[mode][1]))
            assert np.allclose(m, df[f'hebart_mean_reliability_{mode}'].values), \
                f"{mode}: recomputed means differ from the cached CSV"
            err[mode] = np.vstack([m - lo, hi - m])
    ekw = dict(ecolor='0.45', capsize=2.5, error_kw={'elinewidth': 0.9})  # same gray as Fig. 2
    ax.bar(x - width / 2, df['hebart_mean_reliability_seed'], width=width,
          color='tab:blue', label='seed', edgecolor='lightgray',
          yerr=err.get('seed'), **(ekw if err else {}))
    ax.bar(x + width / 2, df['hebart_mean_reliability_bootstrap'], width=width,
          color='tab:green', label='bootstrap', edgecolor='lightgray',
          yerr=err.get('bootstrap'), **(ekw if err else {}))
    ax.axhline(R_THRESHOLD, color='crimson', ls='--', lw=1.2,
              label=f'r = {R_THRESHOLD}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Learned Dimensions')
    ax.set_ylabel('Mean Reliability\n(Fisher-z Pearson, 20 refits)')
    ax.set_ylim(0, 1.05)
    ax.set_title('A', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)


def plot_fractionation_recurrence(ax):
    csv_path = os.path.join(OUT_DIR, 'fractionation_across_seeds.csv')
    df = pd.read_csv(csv_path)
    events = df.dropna(subset=['ref_dim']).copy()
    events['ref_dim'] = events['ref_dim'].astype(int)
    events['col_a'] = events['col_a'].astype(int)
    events['col_b'] = events['col_b'].astype(int)

    # Reported counts use VALIDATED pairs only (classification floor AND the
    # shared-videos/r check against the merge control), not classification alone.
    events = events[events['validated']]

    # Group size = size of the union of columns among a (seed, ref_dim)'s
    # surviving validated pairs -- not the closed-form pair-count inverse,
    # which silently assumes every pair in a group survived validation.
    def _group_size(g):
        return len(set(g['col_a']) | set(g['col_b']))

    groups = (events.groupby(['seed', 'ref_dim', 'label'])
             .apply(_group_size, include_groups=False)
             .reset_index(name='k_cols'))

    order = groups.groupby('label')['seed'].nunique().sort_values(ascending=False).index.tolist()
    order_dim_ids = groups.groupby('label')['ref_dim'].first().reindex(order)
    two_way = groups[groups['k_cols'] == 2].groupby('label')['seed'].nunique().reindex(order, fill_value=0)
    three_way = groups[groups['k_cols'] >= 3].groupby('label')['seed'].nunique().reindex(order, fill_value=0)

    x = np.arange(len(order))
    ax.bar(x, two_way.values, color='cornflowerblue', label='2-way split', edgecolor='lightgray')
    ax.bar(x, three_way.values, bottom=two_way.values, color='tab:pink',
          label='3-way split', edgecolor='lightgray')
    ax.set_xticks(x)
    xtick_labels = [f"{order_dim_ids[l]}: {l}" for l in order]
    ax.set_xticklabels(xtick_labels, rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('Learned Dimensions')
    ax.set_ylabel(f'Fractionation Groups\n({N_SEEDS} bootstrap refits)')
    ax.set_ylim(0, N_SEEDS)
    for xi, tot in zip(x, (two_way + three_way).values):
        ax.text(xi, tot + 0.3, str(int(tot)), ha='center', fontsize=9)
    ax.set_title('B', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ci', action='store_true',
                    help="Add 95%% CIs of the refit mean to panel A; writes "
                         "fractionation_summary_ci.{png,pdf}.")
    args = ap.parse_args()
    ci = None
    if args.ci:
        ci = per_refit_reliability()
        rows = []
        for mode, (_, per_refit) in ci.items():
            m, lo, hi = _z_mean_ci(per_refit)
            for d in BEST_DIMS:
                rows.append({'mode': mode, 'dim': d, 'label': DIM_LABELS.get(d, ''),
                             'mean_r': m[d], 'ci_lo': lo[d], 'ci_hi': hi[d],
                             **{f'refit_{i}': per_refit[i, d] for i in range(per_refit.shape[0])}})
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'recoverability_hebart_per_refit.csv'),
                                  index=False)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    plot_recoverability(axes[0], ci=ci)
    plot_fractionation_recurrence(axes[1])

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fractionation_summary' + ('_ci' if args.ci else ''))
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}.png / .pdf")


if __name__ == '__main__':
    main()
