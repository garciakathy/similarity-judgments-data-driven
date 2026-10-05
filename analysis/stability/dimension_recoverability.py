"""
Per-dimension recoverability of BEST_DIMS across 20 retrains
(Hebart-style reproducibility analysis).

Two modes:
  * seed-only — same triplets, 20 different inits.
                Measures stability to optimisation noise.
  * bootstrap — 20 retrains on triplets resampled with replacement
                (same N, per-seed RNG).  Measures stability to data
                perturbation.

For each retrain, every reference dim is matched 1-to-1 to a column in the
replicate's pruned embedding via Hungarian assignment on Pearson r.  A dim is
called "recoverable" on a given retrain if the matched correlation is at
least R_THRESHOLD (default 0.7).

Matching pool: assignment runs over the FULL reference embedding (all 40
pruned columns), and BEST_DIMS are read off afterwards.  Matching within the
11 BEST_DIMS alone is optimistic — it lets a replicate column that really
corresponds to some other reference dimension be assigned to a BEST_DIM
because no better candidate is in the pool.  Rates here are therefore slightly
lower than a BEST_DIMS-only assignment gives.

A replicate with fewer columns than the reference (some seed-only refits prune
to as few as 32) cannot supply a partner for every reference dim; the surplus
reference dims are left unmatched by the assignment and are scored r = 0
(no corresponding dimension exists in that refit).  Counts are reported.

Outputs (data/interim/results/stability/recoverability/):
  * seed_models.npz, bootstrap_models.npz       — cached pruned embeddings
  * recoverability_seed.npz, recoverability_bootstrap.npz
  * recoverability_summary.csv                  — per-dim stats
  * recoverability_summary.{pdf,png}            — figure
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    UTILS_DIR, MODEL_DIR, RESULTS_DIR,
    SPARSE_LAMBDA, PRUNE_THRESHOLD, EMBEDDING_DIM, N_EPOCHS,
    PATIENCE, LEARNING_RATE, BATCH_SIZE, ENFORCE_POSITIVITY,
    N_VIDEOS, BEST_DIMS, DIM_LABELS,
)
from helpers.data_helpers import train_video_embedding_model

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ============================================================================
# CONFIG
# ============================================================================

N_SEEDS = 20
SEED_LIST = list(range(100, 100 + N_SEEDS))
R_THRESHOLD = 0.7                              # "recoverable" cutoff

OUT_DIR = os.path.join(RESULTS_DIR, 'stability', 'recoverability')
os.makedirs(OUT_DIR, exist_ok=True)

SEED_MODELS_CACHE = os.path.join(OUT_DIR, 'seed_models.npz')
BOOT_MODELS_CACHE = os.path.join(OUT_DIR, 'bootstrap_models.npz')


# ============================================================================
# DATA
# ============================================================================

def _load_triplets_array():
    """Load triplets as (n, 4) int array of (i, j, k, chosen_pair_idx).

    train_triplets.npy is already in this post-processed format (col 3 is
    the position index 0/1/2, not a stim name).
    """
    npy_path = os.path.join(UTILS_DIR, 'train_triplets.npy')

    if os.path.exists(npy_path):
        arr = np.load(npy_path).astype(np.int64)
        assert arr.shape[1] == 4 and arr[:, 3].max() <= 2, (
            f"train_triplets.npy choice column max={arr[:, 3].max()} — "
            "expected pair-index format with values in [0,1,2]"
        )
        return arr

    raise FileNotFoundError("No train triplets in data/utils/")


def _arr_to_tuples(arr):
    return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in arr]


# ============================================================================
# TRAINING WRAPPERS
# ============================================================================

def _train_one(triplets_processed, seed, device):
    (_, pruned_emb, _, _, _, _, _, _, _, _, _, k) = train_video_embedding_model(
        triplets=triplets_processed,
        sparse_lambda=SPARSE_LAMBDA,
        n_videos=N_VIDEOS,
        embedding_dim=EMBEDDING_DIM,
        n_epochs=N_EPOCHS,
        prune_threshold=PRUNE_THRESHOLD,
        enforce_positivity=ENFORCE_POSITIVITY,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        random_seed=seed,
        device=device,
        verbose=False,
    )
    return pruned_emb, k


def get_seed_models():
    """Same triplets, 20 different inits.  Caches to OUT_DIR."""
    if os.path.exists(SEED_MODELS_CACHE):
        print(f"[seed] loading cached models from {SEED_MODELS_CACHE}")
        return np.load(SEED_MODELS_CACHE, allow_pickle=True)['models'].item()

    triplets = _arr_to_tuples(_load_triplets_array())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[seed] training {N_SEEDS} models on {len(triplets)} triplets ({device})")
    out = {}
    for s in tqdm(SEED_LIST, desc="[seed] training"):
        emb, _ = _train_one(triplets, s, device)
        out[s] = emb
    np.savez(SEED_MODELS_CACHE, models=out)
    return out


def get_bootstrap_models():
    """Resample triplets with replacement (per-seed RNG) and retrain."""
    if os.path.exists(BOOT_MODELS_CACHE):
        print(f"[bootstrap] loading cached models from {BOOT_MODELS_CACHE}")
        return np.load(BOOT_MODELS_CACHE, allow_pickle=True)['models'].item()

    arr = _load_triplets_array()
    n = len(arr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[bootstrap] training {N_SEEDS} models on {n} resampled triplets ({device})")

    out = {}
    for s in tqdm(SEED_LIST, desc="[bootstrap] training"):
        rng = np.random.default_rng(s)
        idx = rng.integers(0, n, size=n)
        boot_triplets = _arr_to_tuples(arr[idx])
        emb, _ = _train_one(boot_triplets, s, device)
        out[s] = emb

    np.savez(BOOT_MODELS_CACHE, models=out)
    return out


# ============================================================================
# MATCHING
# ============================================================================

def _corr_cross(A, B):
    """Pearson r between every column of A (n, a) and every column of B (n, b).
    Returns (a, b).  Zero-variance columns yield 0."""
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    sa = np.linalg.norm(A, axis=0)
    sb = np.linalg.norm(B, axis=0)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    R = (A.T @ B) / np.outer(sa, sb)
    return np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)


def match_dims(reference, replicate):
    """Hungarian 1-to-1 match of reference cols to replicate cols (max sum r).

    Reference dims left without a partner (replicate has fewer columns) get
    r = NaN and idx = -1.
    Returns matched_r (n_ref,), matched_idx (n_ref,).
    """
    R = _corr_cross(reference, replicate)        # (n_ref, k_rep)
    row_ind, col_ind = linear_sum_assignment(-R)
    n_ref = reference.shape[1]
    r = np.full(n_ref, np.nan)
    idx = np.full(n_ref, -1, dtype=int)
    r[row_ind] = R[row_ind, col_ind]
    idx[row_ind] = col_ind
    return r, idx


def reciprocal_best_match(reference, replicate):
    """Reciprocal (mutual) best match, the primary measure.

    A refit column counts as a reference dimension's counterpart only if the
    two are each other's best match: the column is the dimension's highest
    correlate, AND no other reference dimension correlates more highly with
    that column.  Otherwise the match is rejected (r = 0, idx = -1).

    This is preferred over Hungarian assignment for per-dimension claims
    because Hungarian maximises the SUM over all dimensions and will move a
    good column away from one dimension to raise another's, penalising an
    individual dimension for reasons unrelated to whether it was recovered.
    It is preferred over plain max-correlation because it rejects merges: if a
    refit collapses two reference dimensions into one column, plain
    max-correlation credits BOTH with recovery, while the reciprocal criterion
    credits only the dimension that best explains the column.

    Note this is not uniformly more permissive than Hungarian — it can reject
    a match outright, which Hungarian never does.
    """
    R = _corr_cross(reference, replicate)
    best_col = R.argmax(axis=1)
    r = np.zeros(reference.shape[1])
    idx = np.full(reference.shape[1], -1, dtype=int)
    for d, c in enumerate(best_col):
        if R[:, c].argmax() == d:                # mutual best match
            r[d] = R[d, c]
            idx[d] = c
    return r, idx


def compute_reciprocal(reference, models, mode):
    n_ref = reference.shape[1]
    match_r = np.zeros((len(SEED_LIST), n_ref))
    match_idx = np.full((len(SEED_LIST), n_ref), -1, dtype=int)
    for i, s in enumerate(tqdm(SEED_LIST, desc=f"[{mode}] reciprocal")):
        r, idx = reciprocal_best_match(reference, models[s])
        match_r[i] = r
        match_idx[i] = idx
    return match_r, match_idx


def compute_recoverability(reference, models, mode):
    n_ref = reference.shape[1]
    match_r = np.full((len(SEED_LIST), n_ref), np.nan)
    match_idx = np.full((len(SEED_LIST), n_ref), -1, dtype=int)
    for i, s in enumerate(tqdm(SEED_LIST, desc=f"[{mode}] match")):
        r, idx = match_dims(reference, models[s])
        match_r[i] = r
        match_idx[i] = idx
    n_unmatched = int((match_idx < 0).sum())
    if n_unmatched:
        print(f"  [{mode}] {n_unmatched} reference dim x refit cells had no "
              f"partner (refit pruned below {n_ref} dims); scored r = 0")
    return match_r, match_idx


def summarize(match_r_dict, unmatched_dict=None):
    rows = []
    for d_idx, dim in enumerate(BEST_DIMS):
        row = {'dim': dim, 'label': DIM_LABELS.get(dim, '')}
        for mode, mr in match_r_dict.items():
            r = mr[:, d_idx]
            row[f'{mode}_median_r'] = float(np.median(r))
            row[f'{mode}_mean_r'] = float(np.mean(r))
            row[f'{mode}_min_r'] = float(np.min(r))
            row[f'{mode}_pct_recoverable'] = float(np.mean(r >= R_THRESHOLD) * 100)
            if unmatched_dict is not None:
                row[f'{mode}_n_unmatched'] = int(unmatched_dict[mode][d_idx])
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# PLOTTING
# ============================================================================

LABELS = [f"{d}: {DIM_LABELS.get(d, '')}" for d in BEST_DIMS]
COLORS = {'seed': 'steelblue', 'bootstrap': 'darkorange'}


def _boxes(ax, data_dict, width=0.38, mask_dict=None):
    """Box plots per dimension.  If mask_dict is given, only retrains where a
    match exists (idx >= 0) contribute, so the boxes describe match quality
    given a match — the frequency of matching is reported separately."""
    x = np.arange(len(BEST_DIMS))
    for offset, mode in zip([-width / 2, width / 2], ['seed', 'bootstrap']):
        mr = data_dict[mode]
        cols = []
        for i in range(len(BEST_DIMS)):
            v = mr[:, i]
            if mask_dict is not None:
                v = v[mask_dict[mode][:, i] >= 0]
            cols.append(v if len(v) else np.array([np.nan]))
        bp = ax.boxplot(cols, positions=x + offset, widths=width * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops={'color': 'black', 'lw': 1.2})
        for box in bp['boxes']:
            box.set_facecolor(COLORS[mode])
            box.set_alpha(0.65)
            box.set_edgecolor(COLORS[mode])
        ax.scatter([], [], s=60, color=COLORS[mode], alpha=0.75, label=mode)
    ax.axhline(R_THRESHOLD, color='red', ls='--', lw=1, label=f'r ≥ {R_THRESHOLD}')
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)


def _bars(ax, values_by_mode, ylabel, width=0.38, ymax=105):
    x = np.arange(len(BEST_DIMS))
    for offset, mode in zip([-width / 2, width / 2], ['seed', 'bootstrap']):
        ax.bar(x + offset, values_by_mode[mode], width=width,
               color=COLORS[mode], alpha=0.85, label=mode, edgecolor='white')
    ax.axhline(100, color='gray', ls=':', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)


def plot_primary(recip_dict, recip_idx_dict):
    """Recoverability rate and merge rate."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.5))

    _bars(axes[0], {m: (recip_dict[m] >= R_THRESHOLD).mean(axis=0) * 100
                    for m in recip_dict},
          f'% retrains recovered (r ≥ {R_THRESHOLD})')
    axes[0].set_title('a   Recoverability', loc='left', fontweight='bold')

    _bars(axes[1], {m: (recip_idx_dict[m] < 0).mean(axis=0) * 100
                    for m in recip_idx_dict},
          '% retrains merged')
    axes[1].set_title('b   Merging', loc='left', fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'recoverability_summary')
    fig.savefig(out + '.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved figure: {out}.pdf")


def plot_match_quality(recip_dict, recip_idx_dict):
    """Match r distributions behind the median values."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    _boxes(ax, recip_dict, mask_dict=recip_idx_dict)
    ax.set_ylabel('Match Pearson r')
    ax.set_title('Match quality', loc='left', fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'recoverability_match_quality')
    fig.savefig(out + '.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {out}.pdf")


def plot(match_r_dict, recip_dict=None):
    """Hungarian assignment (sensitivity analysis)."""
    n_dim = len(BEST_DIMS)
    labels = LABELS
    x = np.arange(n_dim)
    width = 0.38
    colors = COLORS

    n_panels = 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 9),
                             gridspec_kw={'height_ratios': [3, 2]})

    # Panel A — per-dim match r distributions
    ax = axes[0]
    for offset, mode in zip([-width / 2, width / 2], ['seed', 'bootstrap']):
        if mode not in match_r_dict:
            continue
        mr = match_r_dict[mode]
        bp = ax.boxplot(
            [mr[:, i] for i in range(n_dim)],
            positions=x + offset, widths=width * 0.9,
            patch_artist=True, showfliers=False,
            medianprops={'color': 'black', 'lw': 1.2},
        )
        for box in bp['boxes']:
            box.set_facecolor(colors[mode])
            box.set_alpha(0.65)
            box.set_edgecolor(colors[mode])
        ax.scatter([], [], s=60, color=colors[mode], alpha=0.75, label=mode)
    ax.axhline(R_THRESHOLD, color='red', ls='--', lw=1,
               label=f'r ≥ {R_THRESHOLD}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Match Pearson r (reference vs replicate)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Hungarian assignment (sensitivity)', loc='left',
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B — % retrains with r ≥ threshold
    ax = axes[1]
    for offset, mode in zip([-width / 2, width / 2], ['seed', 'bootstrap']):
        if mode not in match_r_dict:
            continue
        pct = (match_r_dict[mode] >= R_THRESHOLD).mean(axis=0) * 100
        ax.bar(x + offset, pct, width=width, color=colors[mode],
               alpha=0.85, label=mode, edgecolor='white')
    ax.axhline(100, color='gray', ls=':', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f'% retrains with r ≥ {R_THRESHOLD}')
    ax.set_ylim(0, 105)
    ax.set_title('Recoverability rate', loc='left', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_pdf = os.path.join(OUT_DIR, 'recoverability_hungarian_sensitivity.pdf')
    out_png = os.path.join(OUT_DIR, 'recoverability_hungarian_sensitivity.png')
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved figure: {out_pdf}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("DIMENSION RECOVERABILITY (BEST_DIMS, Hungarian matching)")
    print(f"Seeds:                      {SEED_LIST}")
    print(f"Reported dims:              {BEST_DIMS}")
    print(f"Recoverability threshold:   r ≥ {R_THRESHOLD}")
    print("=" * 60)

    reference = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    print(f"Reference embedding:        {reference.shape}")
    print(f"Matching pool:              all {reference.shape[1]} reference "
          f"columns (BEST_DIMS read off afterwards)")

    seed_models = get_seed_models()
    boot_models = get_bootstrap_models()

    seed_r, seed_idx = compute_recoverability(reference, seed_models, 'seed')
    boot_r, boot_idx = compute_recoverability(reference, boot_models, 'bootstrap')
    seed_rr, seed_ridx = compute_reciprocal(reference, seed_models, 'seed')
    boot_rr, boot_ridx = compute_reciprocal(reference, boot_models, 'bootstrap')

    np.savez(os.path.join(OUT_DIR, 'recoverability_seed.npz'),
             match_r=seed_r, match_idx=seed_idx,
             seeds=np.array(SEED_LIST),
             reference_dims=np.arange(reference.shape[1]),
             best_dims=np.array(BEST_DIMS))
    np.savez(os.path.join(OUT_DIR, 'recoverability_bootstrap.npz'),
             match_r=boot_r, match_idx=boot_idx,
             seeds=np.array(SEED_LIST),
             reference_dims=np.arange(reference.shape[1]),
             best_dims=np.array(BEST_DIMS))

    # subset to the reported dims; an unmatched cell (NaN) is a failure to
    # recover, so it scores r = 0 rather than being dropped
    unmatched = {'seed': (seed_idx[:, BEST_DIMS] < 0).sum(axis=0),
                 'bootstrap': (boot_idx[:, BEST_DIMS] < 0).sum(axis=0)}
    best = {'seed': np.nan_to_num(seed_r[:, BEST_DIMS]),
            'bootstrap': np.nan_to_num(boot_r[:, BEST_DIMS])}

    df = summarize(best, unmatched)

    # primary measure: reciprocal best match
    recip = {'seed': seed_rr[:, BEST_DIMS], 'bootstrap': boot_rr[:, BEST_DIMS]}
    recip_idx = {'seed': seed_ridx[:, BEST_DIMS],
                 'bootstrap': boot_ridx[:, BEST_DIMS]}
    for mode in ['seed', 'bootstrap']:
        rr, ri = recip[mode], recip_idx[mode]
        df[f'{mode}_recip_pct_recoverable'] = (rr >= R_THRESHOLD).mean(axis=0) * 100
        # median over refits where a reciprocal match exists at all
        df[f'{mode}_recip_median_r'] = [
            float(np.median(rr[ri[:, p] >= 0, p])) if (ri[:, p] >= 0).any() else 0.0
            for p in range(len(BEST_DIMS))]
        df[f'{mode}_pct_merged'] = (ri < 0).mean(axis=0) * 100

    csv_path = os.path.join(OUT_DIR, 'recoverability_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")
    print("\nPRIMARY — reciprocal best match:")
    print(df[['dim', 'label', 'seed_recip_pct_recoverable',
              'bootstrap_recip_pct_recoverable', 'seed_recip_median_r',
              'bootstrap_recip_median_r', 'bootstrap_pct_merged']]
          .to_string(index=False))
    print("\nSENSITIVITY — Hungarian assignment over the full 40-dim reference:")
    print(df[['dim', 'label', 'seed_pct_recoverable',
              'bootstrap_pct_recoverable']].to_string(index=False))
    for mode in ['seed', 'bootstrap']:
        print(f"\n[{mode}] dims at ≥95%: "
              f"reciprocal {int((df[f'{mode}_recip_pct_recoverable'] >= 95).sum())}/11"
              f" (mean {df[f'{mode}_recip_pct_recoverable'].mean():.1f}%)"
              f" | Hungarian {int((df[f'{mode}_pct_recoverable'] >= 95).sum())}/11"
              f" (mean {df[f'{mode}_pct_recoverable'].mean():.1f}%)")

    plot_primary(recip, recip_idx)
    plot_match_quality(recip, recip_idx)
    plot(best)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
