"""
Dimensionality selection via greedy forward selection on the TRAIN RSM.

Train/test split is at the triplet level (Hebart-style): the train RSM is
a full 250x250 matrix built from training triplets covering all 250 videos,
and the test RSM is a dense 50x50 built from held-out triplets on test_idx.
Selection runs on the full 250x250 train RSM so no test-side triplet
information leaks into the elbow.

  - Target y: upper triangle of 250x250 train RSM (training triplets only)
  - Features X: all 250 embedding rows, all dims
  - Elbow: kneedle on cross-validated cumulative R²

Both curves are scored out-of-sample (10-fold x 5 repeats) so the train and
test targets are compared on the same footing. Each is drawn against its own
noise ceiling: the Spearman-Brown corrected split-half reliability of that
target RSM, which bounds the R² any predictor can reach. The ceilings differ
a lot (the 250x250 train RSM has ~4.1 judgments/cell vs ~9.2 for the dense
50x50 test RSM), which is why the raw curves sit at very different heights.

Outputs:
  - data/interim/results/behavioral/dim_selection_train.png/.pdf  (elbow plot)
  - data/interim/results/behavioral/dim_selection_noise_ceilings.csv
  - Prints the recommended BEST_DIMS list for config.py

Flags:
  --refit   re-run greedy forward selection (otherwise reuse the cached
            greedy_forward_r2_train.csv, so the figure can be restyled fast)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
from kneed import KneeLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    UTILS_DIR, MODEL_DIR, RESULTS_BEHAVIORAL_DIR,
    RANDOM_SEED, train_idx, test_idx, BEST_DIMS
)
from helpers.reliability import (
    build_rsm, load_train_triplets, load_test_triplets, split_half_ceiling
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

parser = argparse.ArgumentParser()
parser.add_argument('--refit', action='store_true',
                    help='re-run greedy selection instead of reusing the cached CSV')
parser.add_argument('--n-splits', type=int, default=200,
                    help='split-half repeats for the noise ceiling')
args = parser.parse_args()

N_STIM = 250
GREEDY_CSV = os.path.join(RESULTS_BEHAVIORAL_DIR, 'greedy_forward_r2_train.csv')


# ============================================================
# Load data
# ============================================================
emb_all = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
print(f"Embeddings: {emb_all.shape}")

train_rsm_path = os.path.join(UTILS_DIR, 'train_human_sim.npy')
if not os.path.exists(train_rsm_path):
    train_rsm_path = os.path.join(UTILS_DIR, 'train_human_rsm.csv')
    train_rsm = pd.read_csv(train_rsm_path).to_numpy()
else:
    train_rsm = np.load(train_rsm_path)

print(f"Train RSM: {train_rsm.shape}")

# Use the full 250x250 train RSM: the train/test split is at triplet level,
# so every video (including the 50 test videos) appears in training triplets.
n_stim = train_rsm.shape[0]
assert emb_all.shape[0] == n_stim, (
    f"Embedding rows ({emb_all.shape[0]}) must match train RSM size ({n_stim})"
)

triu_idx = np.triu_indices(n_stim, k=1)
y = train_rsm[triu_idx]
print(f"Train RSM pairs (full {n_stim}x{n_stim}): {len(y)}")

# ============================================================
# Build per-dim similarity vectors (same normalisation as explained_variance.py)
# ============================================================
n_dims = emb_all.shape[1]
X_all = np.zeros((len(y), n_dims))
for d in range(n_dims):
    dist = pdist(emb_all[:, [d]], metric='euclidean')
    dmax = dist.max()
    X_all[:, d] = 1 - (dist / dmax) if dmax != 0 else 1 - dist

# ============================================================
# Cross-validated greedy forward selection (on TRAIN)
# ============================================================
n_folds = 10
n_repeats = 5


def cv_r2(X, y):
    scores = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED + rep)
        scores.extend(cross_val_score(LinearRegression(), X, y, cv=kf, scoring='r2'))
    return float(np.mean(scores))


if not args.refit and os.path.exists(GREEDY_CSV):
    cached = pd.read_csv(GREEDY_CSV)
    cv_selected = cached['dim_id'].tolist()
    cumul_r2_cv = cached['greedy_cumul_r2'].to_numpy()
    print(f"\n--- Reusing cached greedy order: {GREEDY_CSV} (--refit to redo) ---")
    assert len(cv_selected) == n_dims, (
        f"Cached greedy order has {len(cv_selected)} dims but embedding has {n_dims}; "
        "re-run with --refit"
    )
else:
    print("\n--- CV Greedy Forward Selection (train RSM) ---")
    cv_remaining = list(range(n_dims))
    cv_selected = []
    cumul_r2_cv = []

    for step in range(n_dims):
        best_cv_r2, best_dim = -np.inf, None
        for d in cv_remaining:
            mean_cv = cv_r2(X_all[:, cv_selected + [d]], y)
            if mean_cv > best_cv_r2:
                best_cv_r2, best_dim = mean_cv, d

        cv_selected.append(best_dim)
        cv_remaining.remove(best_dim)
        cumul_r2_cv.append(best_cv_r2)

        if step < 20 or (step + 1) % 10 == 0:
            print(f"  Step {step+1:2d}: Dim {best_dim:2d}  CV R²={best_cv_r2:.4f}")

    cumul_r2_cv = np.array(cumul_r2_cv)

    marginal_r2 = np.concatenate([[cumul_r2_cv[0]], np.diff(cumul_r2_cv)])
    pd.DataFrame({
        'dim_id': cv_selected,
        'greedy_step': np.arange(1, n_dims + 1),
        'greedy_marginal_r2': marginal_r2,
        'greedy_cumul_r2': cumul_r2_cv,
    }).to_csv(GREEDY_CSV, index=False)
    print(f"\nSaved train greedy order: {GREEDY_CSV}")

# ============================================================
# Kneedle elbow
# ============================================================
steps = np.arange(1, n_dims + 1)
kneedle = KneeLocator(steps, cumul_r2_cv, curve='concave', direction='increasing')
kneedle_n = int(kneedle.knee) if kneedle.knee is not None else int(np.argmax(cumul_r2_cv)) + 1
kneedle_r2 = cumul_r2_cv[kneedle_n - 1]

train_best_dims = cv_selected[:kneedle_n]

print(f"\nKneedle elbow (train): {kneedle_n} dims  CV R²={kneedle_r2:.4f}")
print(f"Train BEST_DIMS: {train_best_dims}")
print(f"\nCurrent BEST_DIMS (config): {BEST_DIMS}")
print(f"Overlap with current: {sorted(set(train_best_dims) & set(BEST_DIMS))}")

# ============================================================
# Also evaluate train-selected dims on held-out test RSM
# ============================================================
test_rsm_path = os.path.join(UTILS_DIR, 'human_test_rsm.csv')
test_rsm = pd.read_csv(test_rsm_path).to_numpy()

emb_test = emb_all[test_idx, :]
rsm_test_sub = test_rsm  # already 50x50
triu_test = np.triu_indices(len(test_idx), k=1)
y_test = rsm_test_sub[triu_test]

X_test_all = np.zeros((len(y_test), n_dims))
for d in range(n_dims):
    dist = pdist(emb_test[:, [d]], metric='euclidean')
    dmax = dist.max()
    X_test_all[:, d] = 1 - (dist / dmax) if dmax != 0 else 1 - dist

# Cumulative CV R² on the test RSM, using the train-selected dim order.
# Scored out-of-sample like the train curve so the two are directly comparable.
cumul_r2_test = np.array([
    cv_r2(X_test_all[:, cv_selected[:k]], y_test) for k in range(1, n_dims + 1)
])

print(f"\nTest-set CV R² at {kneedle_n} train-selected dims: {cumul_r2_test[kneedle_n-1]:.4f}")
print(f"Test-set CV R² at {len(BEST_DIMS)} current BEST_DIMS: "
      f"{cv_r2(X_test_all[:, BEST_DIMS], y_test):.4f}")

# ============================================================
# Noise ceilings for both targets
# ============================================================
print(f"\n--- Noise ceilings ({args.n_splits} split-halves, Spearman-Brown corrected) ---")

train_trip = load_train_triplets(UTILS_DIR)
test_trip = load_test_triplets(UTILS_DIR)

# Guard: the rebuilt RSMs must be the same matrices the curves are fit against
for label, rebuilt, target in (
    ('train', build_rsm(train_trip)[np.triu_indices(N_STIM, 1)], y),
    ('test', build_rsm(test_trip)[np.ix_(sorted(test_idx), sorted(test_idx))][triu_test], y_test),
):
    assert np.allclose(rebuilt, target, atol=1e-8), (
        f"Rebuilt {label} RSM does not match the regression target — "
        "triplet decoding is wrong, ceiling would be meaningless"
    )

all_ids = np.arange(N_STIM)
test_ids = np.asarray(sorted(test_idx))

nc_train, nc_train_ci = split_half_ceiling(train_trip, all_ids, args.n_splits)
nc_test, nc_test_ci = split_half_ceiling(test_trip, test_ids, args.n_splits)

# each triplet contributes 3 pairs — this density is what drives the ceiling gap
obs_per_cell_train = 3 * len(train_trip) / len(y)
obs_per_cell_test = 3 * len(test_trip) / len(y_test)
print(f"  Judgments per pair — train {obs_per_cell_train:.2f}, test {obs_per_cell_test:.2f}")

print(f"  Train RSM (250x250, {len(train_trip)} triplets): "
      f"{nc_train:.3f}  [{nc_train_ci[0]:.3f}, {nc_train_ci[1]:.3f}]")
print(f"  Test  RSM (50x50,  {len(test_trip)} triplets): "
      f"{nc_test:.3f}  [{nc_test_ci[0]:.3f}, {nc_test_ci[1]:.3f}]")
print(f"  Fraction of ceiling reached at {kneedle_n} dims — "
      f"train {cumul_r2_cv[kneedle_n-1]/nc_train:.2f}, test {cumul_r2_test[kneedle_n-1]/nc_test:.2f}")

pd.DataFrame([
    {'target': 'train_rsm_250', 'n_triplets': len(train_trip), 'n_stimuli': N_STIM,
     'noise_ceiling': nc_train, 'ci_lo': nc_train_ci[0], 'ci_hi': nc_train_ci[1],
     'r2_at_elbow': cumul_r2_cv[kneedle_n - 1]},
    {'target': 'test_rsm_50', 'n_triplets': len(test_trip), 'n_stimuli': len(test_ids),
     'noise_ceiling': nc_test, 'ci_lo': nc_test_ci[0], 'ci_hi': nc_test_ci[1],
     'r2_at_elbow': cumul_r2_test[kneedle_n - 1]},
]).to_csv(os.path.join(RESULTS_BEHAVIORAL_DIR, 'dim_selection_noise_ceilings.csv'), index=False)

# ============================================================
# Plot
# ============================================================
TRAIN_C, TEST_C = '#2C6E9B', '#D97A28'

fig, ax = plt.subplots(figsize=(6.9, 4.3))

# Noise ceiling bands (95% CI over split-halves), drawn behind the curves
# label sides differ so neither collides with the test curve, which passes
# through the train ceiling on its way up
for lo_hi, mean, colour, name, x_text, ha in (
    (nc_test_ci, nc_test, TEST_C, 'Test RSM', n_dims - 0.6, 'right'),
    (nc_train_ci, nc_train, TRAIN_C, 'Train RSM', 1.0, 'left'),
):
    ax.axhspan(lo_hi[0], lo_hi[1], color=colour, alpha=0.13, lw=0, zorder=0)
    ax.axhline(mean, color=colour, ls=(0, (4, 3)), lw=1.1, alpha=0.85, zorder=1)
    ax.text(x_text, lo_hi[1] + 0.008, f'{name} noise ceiling ({mean:.2f})',
            fontsize=8.5, color=colour, va='bottom', ha=ha, zorder=3)

ax.plot(steps, cumul_r2_test, color=TEST_C, lw=2, zorder=4,
        label=f'Test RSM   ({len(test_ids)} videos, {obs_per_cell_test:.1f} judgments/pair)')
ax.plot(steps, cumul_r2_cv, color=TRAIN_C, lw=2, zorder=4,
        label=f'Train RSM ({N_STIM} videos, {obs_per_cell_train:.1f} judgments/pair)')

# Elbow — selection happens on the train curve only. Annotating each curve
# with its fraction of ceiling is the point of the bands: the two targets sit
# at very different heights but the model reaches the same share of each.
ax.axvline(kneedle_n, color='0.35', ls=':', lw=1.2, zorder=2)
for value, ceiling, colour, xy_text, ha in (
    (cumul_r2_test[kneedle_n - 1], nc_test, TEST_C, (kneedle_n - 0.6, 0.60), 'right'),
    (kneedle_r2, nc_train, TRAIN_C, (kneedle_n + 1.2, 0.235), 'left'),
):
    ax.plot(kneedle_n, value, 'o', color=colour, ms=6, mec='white', mew=1.2, zorder=5)
    ax.annotate(f'{value / ceiling:.0%} of ceiling', xy=xy_text,
                fontsize=8.5, color=colour, ha=ha, va='center', zorder=5)
ax.annotate(f'elbow: {kneedle_n} dims', xy=(kneedle_n + 1.2, 0.03),
            fontsize=9, color='0.25', va='bottom', ha='left', zorder=5)

ax.set_xlabel('Number of dimensions (greedy order)', fontsize=10.5)
ax.set_ylabel('Cross-validated R²', fontsize=10.5)
ax.set_xlim(0.5, n_dims + 0.5)
ax.set_ylim(0, max(nc_test_ci[1], nc_train_ci[1]) * 1.10)
ax.tick_params(labelsize=9)
leg = ax.legend(fontsize=9, frameon=False, loc='lower right',
                handlelength=1.8, borderaxespad=0.8, labelspacing=0.4)
leg.set_zorder(6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(RESULTS_BEHAVIORAL_DIR, 'dim_selection_train.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}")