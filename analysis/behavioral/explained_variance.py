"""
Explained variance analysis: behavioral features and embedding dimensions
predicting human similarity judgments.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
from matplotlib.gridspec import GridSpec
from kneed import KneeLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    UTILS_DIR, MODEL_DIR, RESULTS_BEHAVIORAL_DIR, BEHAVIOR_COLS,
    BEST_DIMS, RANDOM_SEED, test_idx
)
from helpers.action_categories import load_action_categories, build_action_rsm
from helpers.variance_partition import (
    cv_r2, cv_spearman_r2, compute_explained_variance,
    compute_explained_variance_spearman, build_behavior_rsms,
    format_behavior_df, get_bar_colors,
)
from helpers.reliability import (
    build_rsm, load_train_triplets, load_test_triplets, split_half_ceiling
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

os.makedirs(RESULTS_BEHAVIORAL_DIR, exist_ok=True)


def main():
    # =========================================================
    # Load data
    # =========================================================
    print("\n" + "=" * 50)
    print("EXPLAINED VARIANCE ANALYSIS")
    print("=" * 50)

    # Load embeddings
    sparse_pruned_emb = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    print(f"Loaded embeddings: {sparse_pruned_emb.shape}")

    # Load annotations
    annotations = pd.read_csv(os.path.join(UTILS_DIR, 'annotations.csv'))
    feature_cols = BEHAVIOR_COLS

    # Build behavioral RSMs (full 250x250)
    rsms = build_behavior_rsms(annotations, feature_cols)

    # Load human test RSM
    test_rsm_path = os.path.join(UTILS_DIR, 'human_test_rsm.csv')
    test_rsm_npy = os.path.join(UTILS_DIR, 'test_human_sim.npy')

    if os.path.exists(test_rsm_path):
        human_test_rsm = pd.read_csv(test_rsm_path).to_numpy()
    elif os.path.exists(test_rsm_npy):
        human_test_rsm = np.load(test_rsm_npy)
    else:
        print("ERROR: No test RSM found")
        return

    print(f"Human test RSM shape: {human_test_rsm.shape}")

    # Determine test indices
    if test_idx is not None:
        t_idx = test_idx
    else:
        # Fall back to RSM size
        n_test = human_test_rsm.shape[0]
        print(f"WARNING: No test_idx from config, assuming first {n_test} videos")
        t_idx = list(range(n_test))

    n_test = len(t_idx)
    print(f"Test set: {n_test} videos")

    # =========================================================
    # Subset to test videos
    # =========================================================
    # Subset embeddings to test videos
    emb_test = sparse_pruned_emb[t_idx, :]
    n_stim, n_dims = emb_test.shape
    print(f"Test embeddings: {emb_test.shape}")

    # Subset behavioral RSMs to test videos
    rsms_test = {}
    for feature in rsms:
        rsms_test[feature] = rsms[feature][np.ix_(t_idx, t_idx)]

    # Target: upper triangle of test human RSM
    triu_idx = np.triu_indices(n_test, k=1)
    y = human_test_rsm[triu_idx]

    # =========================================================
    # LEFT: Behavioral features explained variance
    # =========================================================
    print("\n--- Behavioral Features ---")
    beh_features = {}
    for feature in rsms_test:
        beh_features[feature.title()] = rsms_test[feature][triu_idx]

    # Add action category RSM
    labels_all = load_action_categories(UTILS_DIR)
    labels_test = labels_all[t_idx]
    action_rsm_test = build_action_rsm(labels_test)
    beh_features['Action Categories'] = action_rsm_test[triu_idx]

    beh_data = pd.DataFrame(beh_features)

    r2_full_beh, unique_vars_beh, shared_beh = compute_explained_variance(y, beh_data)
    print(f"[Behavior] Total R2: {r2_full_beh:.4f}")
    for feat, uv in unique_vars_beh.items():
        print(f"  {feat}: {uv:.4f}")
    print(f"  Shared: {shared_beh:.4f}")

    df_beh = format_behavior_df(unique_vars_beh, shared_beh, r2_full_beh)
    beh_colors = get_bar_colors(df_beh["Feature"], "tab:purple")

    # =========================================================
    # RIGHT: Embedding dimensions explained variance
    # =========================================================
    print("\n--- Embedding Dimensions ---")
    emb_features = {}
    for d in range(n_dims):
        dist = pdist(emb_test[:, [d]], metric="euclidean")
        dmax = dist.max()
        sim = 1 - (dist / dmax) if dmax != 0 else 1 - dist
        emb_features[f"Dim {d}"] = sim

    df_emb = pd.DataFrame(emb_features)
    r2_full_emb, unique_vars_emb, shared_emb = compute_explained_variance(y, df_emb)
    print(f"[Embedding] Total R2: {r2_full_emb:.4f}")

    unique_vars_emb["Shared Variance"] = shared_emb
    unique_vars_emb["Total Variance"] = r2_full_emb

    df_var = (
        pd.DataFrame(unique_vars_emb, index=["R2"])
        .T.reset_index()
        .rename(columns={"index": "Dimension"})
    )
    df_dims = df_var[df_var["Dimension"].str.startswith("Dim")]
    df_extra = df_var[~df_var["Dimension"].str.startswith("Dim")]
    df_emb_sorted = pd.concat([df_dims, df_extra], ignore_index=True)
    emb_colors = ["tab:blue"] * len(df_dims) + ["lightgray", "darkgray"]

    # =========================================================
    # Side-by-side plot
    # =========================================================
    noise_ceiling = 0.6658  # Spearman-Brown corrected split-half reliability (dense test triplets)

    x_beh = np.arange(len(df_beh))
    x_emb = np.arange(len(df_emb_sorted))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), sharey=True,
        gridspec_kw={"width_ratios": [1, 2.2]}
    )

    # Left: Behavior
    ax1.bar(x_beh, df_beh["Variance"].values, width=0.5,
            color=beh_colors, edgecolor="black", linewidth=0.8)
    ax1.axhline(y=noise_ceiling, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Behavioral Features", fontsize=14)
    ax1.set_ylabel("Explained Variance (R$^2$)", fontsize=14)
    beh_labels = [f"{f} $^{{\\rm (UV)}}$" if "Shared" not in f and "Total" not in f else f
                  for f in df_beh["Feature"]]
    ax1.set_xticks(x_beh)
    ax1.set_xticklabels(beh_labels, rotation=90, ha="center", fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.25)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Embedding
    ax2.bar(x_emb, df_emb_sorted["R2"].values, width=0.8,
            color=emb_colors, edgecolor="black", linewidth=0.8)
    ax2.axhline(y=noise_ceiling, color="gray", linestyle="--", alpha=0.5)
    ax2.text(x_emb[-1], noise_ceiling, f"Split-Half Reliability ({noise_ceiling:.2f})",
             ha="right", va="bottom", color="gray", alpha=0.7, fontsize=11)
    ax2.set_xlabel("Embedding Dimension", fontsize=14)
    emb_labels = [f"{d} $^{{\\rm (UV)}}$" if "Shared" not in d and "Total" not in d else d
                  for d in df_emb_sorted["Dimension"]]
    ax2.set_xticks(x_emb)
    ax2.set_xticklabels(emb_labels, rotation=90, ha="center", fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.25)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax1.set_ylim(0, noise_ceiling)
    ax2.set_ylim(0, noise_ceiling)

    fig.suptitle("Unique & Shared Explained Variance Predicting Human RSM (Test Set, 50 Videos)",
                 fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(RESULTS_BEHAVIORAL_DIR, 'explained_variance_test_50vids.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close(fig)

    # =========================================================
    # Spearman r² variant
    # =========================================================
    print("\n--- Spearman r² variant ---")

    # Behavioral (Spearman)
    r2_full_beh_sp, unique_vars_beh_sp, shared_beh_sp = compute_explained_variance_spearman(
        y, beh_data
    )
    print(f"[Behavior Spearman] Total r²: {r2_full_beh_sp:.4f}")
    for feat, uv in unique_vars_beh_sp.items():
        print(f"  {feat}: {uv:.4f}")
    print(f"  Shared: {shared_beh_sp:.4f}")

    df_beh_sp = format_behavior_df(dict(unique_vars_beh_sp), shared_beh_sp, r2_full_beh_sp)
    beh_colors_sp = get_bar_colors(df_beh_sp["Feature"], "tab:purple")

    # Embedding (Spearman)
    r2_full_emb_sp, unique_vars_emb_sp, shared_emb_sp = compute_explained_variance_spearman(
        y, df_emb
    )
    print(f"[Embedding Spearman] Total r²: {r2_full_emb_sp:.4f}")

    unique_vars_emb_sp["Shared Variance"] = shared_emb_sp
    unique_vars_emb_sp["Total Variance"] = r2_full_emb_sp

    df_var_sp = (
        pd.DataFrame(unique_vars_emb_sp, index=["R2"])
        .T.reset_index()
        .rename(columns={"index": "Dimension"})
    )
    df_dims_sp = df_var_sp[df_var_sp["Dimension"].str.startswith("Dim")]
    df_extra_sp = df_var_sp[~df_var_sp["Dimension"].str.startswith("Dim")]
    df_emb_sorted_sp = pd.concat([df_dims_sp, df_extra_sp], ignore_index=True)
    emb_colors_sp = ["tab:blue"] * len(df_dims_sp) + ["lightgray", "darkgray"]

    # Plot
    x_beh_sp = np.arange(len(df_beh_sp))
    x_emb_sp = np.arange(len(df_emb_sorted_sp))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), sharey=True,
        gridspec_kw={"width_ratios": [1, 2.2]}
    )

    ax1.bar(x_beh_sp, df_beh_sp["Variance"].values, width=0.5,
            color=beh_colors_sp, edgecolor="black", linewidth=0.8)
    ax1.axhline(y=noise_ceiling, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Behavioral Features", fontsize=14)
    ax1.set_ylabel("Explained Variance (Spearman $r^2$)", fontsize=14)
    beh_labels_sp = [f"{f} $^{{\\rm (UV)}}$" if "Shared" not in f and "Total" not in f else f
                     for f in df_beh_sp["Feature"]]
    ax1.set_xticks(x_beh_sp)
    ax1.set_xticklabels(beh_labels_sp, rotation=90, ha="center", fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.25)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.bar(x_emb_sp, df_emb_sorted_sp["R2"].values, width=0.8,
            color=emb_colors_sp, edgecolor="black", linewidth=0.8)
    ax2.axhline(y=noise_ceiling, color="gray", linestyle="--", alpha=0.5)
    ax2.text(x_emb_sp[-1], noise_ceiling, f"Split-Half Reliability ({noise_ceiling:.2f})",
             ha="right", va="bottom", color="gray", alpha=0.7, fontsize=11)
    ax2.set_xlabel("Embedding Dimension", fontsize=14)
    emb_labels_sp = [f"{d} $^{{\\rm (UV)}}$" if "Shared" not in d and "Total" not in d else d
                     for d in df_emb_sorted_sp["Dimension"]]
    ax2.set_xticks(x_emb_sp)
    ax2.set_xticklabels(emb_labels_sp, rotation=90, ha="center", fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.25)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax1.set_ylim(0, noise_ceiling)
    ax2.set_ylim(0, noise_ceiling)

    fig.suptitle("Unique & Shared Explained Variance Predicting Human RSM "
                 "(Spearman r², Test Set, 50 Videos)",
                 fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path_sp = os.path.join(RESULTS_BEHAVIORAL_DIR,
                                'explained_variance_test_50vids_spearman.png')
    fig.savefig(save_path_sp, dpi=300, bbox_inches='tight')
    fig.savefig(save_path_sp.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"\nSaved: {save_path_sp}")
    plt.close(fig)

    # =========================================================
    # Cumulative R2 plots
    # =========================================================
    print("\n--- Cumulative R2 ---")
    X_all = df_emb.values
    n_pairs, n_dims_emb = X_all.shape

    # Fixed numeric order
    cumul_r2_fixed = []
    for k in range(1, n_dims_emb + 1):
        model_k = LinearRegression().fit(X_all[:, :k], y)
        cumul_r2_fixed.append(r2_score(y, model_k.predict(X_all[:, :k])))
    cumul_r2_fixed = np.array(cumul_r2_fixed)

    # Greedy forward selection (in-sample)
    remaining = list(range(n_dims_emb))
    selected = []
    cumul_r2_forward = []
    while remaining:
        best_r2, best_dim = -np.inf, None
        for d in remaining:
            trial_cols = selected + [d]
            r2 = r2_score(y, LinearRegression().fit(X_all[:, trial_cols], y).predict(X_all[:, trial_cols]))
            if r2 > best_r2:
                best_r2, best_dim = r2, d
        selected.append(best_dim)
        remaining.remove(best_dim)
        cumul_r2_forward.append(best_r2)
    cumul_r2_forward = np.array(cumul_r2_forward)

    # Save greedy order + R² for use in other scripts
    marginal_r2 = np.diff(np.concatenate([[0], cumul_r2_forward]))
    greedy_df = pd.DataFrame({
        'dim_id': selected,
        'greedy_step': list(range(1, len(selected) + 1)),
        'greedy_marginal_r2': marginal_r2,
        'greedy_cumul_r2': cumul_r2_forward,
    })
    greedy_csv_path = os.path.join(RESULTS_BEHAVIORAL_DIR, 'greedy_forward_r2.csv')
    greedy_df.to_csv(greedy_csv_path, index=False)
    print(f"Saved greedy forward R² to {greedy_csv_path}")

    print(f"Best dims (greedy order): {selected[:10]}")

    # Greedy forward selection (cross-validated)
    print("\n--- Cross-Validated Greedy Forward Selection ---")
    n_folds = 10
    n_repeats = 5
    cv_remaining = list(range(n_dims_emb))
    cv_selected = []
    cumul_r2_cv = []

    for step in range(n_dims_emb):
        best_cv_r2, best_dim = -np.inf, None
        for d in cv_remaining:
            trial_cols = cv_selected + [d]
            X_trial = X_all[:, trial_cols]
            # Repeated K-fold for stable estimate
            cv_scores = []
            for rep in range(n_repeats):
                kf = KFold(n_splits=n_folds, shuffle=True,
                           random_state=RANDOM_SEED + rep)
                scores = cross_val_score(
                    LinearRegression(), X_trial, y,
                    cv=kf, scoring='r2'
                )
                cv_scores.extend(scores)
            mean_cv = np.mean(cv_scores)
            if mean_cv > best_cv_r2:
                best_cv_r2, best_dim = mean_cv, d

        cv_selected.append(best_dim)
        cv_remaining.remove(best_dim)
        cumul_r2_cv.append(best_cv_r2)

        if step < 20 or (step + 1) % 10 == 0:
            print(f"  Step {step+1:2d}: Dim {best_dim:2d}  CV R²={best_cv_r2:.4f}")

    cumul_r2_cv = np.array(cumul_r2_cv)

    # Find optimal number of dimensions (peak CV R²)
    optimal_n_dims = int(np.argmax(cumul_r2_cv)) + 1
    optimal_cv_r2 = cumul_r2_cv[optimal_n_dims - 1]
    optimal_dims = cv_selected[:optimal_n_dims]
    print(f"\nOptimal dims (CV): {optimal_n_dims} (CV R²={optimal_cv_r2:.4f})")
    print(f"Selected dimensions: {optimal_dims}")

    # =========================================================
    # Elbow / cutoff detection methods
    # =========================================================
    print("\n--- Elbow / Cutoff Detection ---")
    steps = np.arange(1, n_dims_emb + 1)

    # --- Method 1: Kneedle algorithm ---
    kneedle = KneeLocator(
        steps, cumul_r2_cv, curve='concave', direction='increasing'
    )
    kneedle_n = int(kneedle.knee) if kneedle.knee is not None else optimal_n_dims
    kneedle_r2 = cumul_r2_cv[kneedle_n - 1]
    print(f"  Kneedle elbow: {kneedle_n} dims (CV R²={kneedle_r2:.4f})")

    # --- Method 2: Adjusted R² peak ---
    # Recompute in-sample R² in CV-selected order (penalty meaningful for in-sample)
    insample_r2_cv_order = []
    for k in range(1, n_dims_emb + 1):
        cols = cv_selected[:k]
        r2_k = r2_score(y, LinearRegression().fit(X_all[:, cols], y).predict(X_all[:, cols]))
        insample_r2_cv_order.append(r2_k)
    insample_r2_cv_order = np.array(insample_r2_cv_order)

    n_obs = len(y)
    adj_r2 = 1 - ((1 - insample_r2_cv_order) * (n_obs - 1) / (n_obs - steps - 1))
    adjr2_n = int(np.argmax(adj_r2)) + 1
    adjr2_cv_r2 = cumul_r2_cv[adjr2_n - 1]
    print(f"  Adjusted R² peak: {adjr2_n} dims (CV R²={adjr2_cv_r2:.4f})")

    # --- Method 3: BIC minimum ---
    rss = (1 - insample_r2_cv_order) * np.sum((y - np.mean(y)) ** 2)
    bic = n_obs * np.log(rss / n_obs) + steps * np.log(n_obs)
    bic_n = int(np.argmin(bic)) + 1
    bic_cv_r2 = cumul_r2_cv[bic_n - 1]
    print(f"  BIC minimum: {bic_n} dims (CV R²={bic_cv_r2:.4f})")

    # --- Method 4: Permutation baseline (parallel analysis analog) ---
    print("  Running permutation baseline (100 perms)...")
    rng = np.random.RandomState(RANDOM_SEED)
    n_perms = 100
    null_r2 = np.zeros((n_dims_emb, n_perms))
    for p in range(n_perms):
        y_perm = rng.permutation(y)
        for k in range(1, n_dims_emb + 1):
            cols = cv_selected[:k]
            r2_p = r2_score(
                y_perm,
                LinearRegression().fit(X_all[:, cols], y_perm).predict(X_all[:, cols])
            )
            null_r2[k - 1, p] = r2_p
    null_95 = np.percentile(null_r2, 95, axis=1)

    # Cutoff = last step where observed CV R² exceeds null 95th percentile
    exceeds = cumul_r2_cv > null_95
    if np.any(exceeds):
        perm_n = int(np.where(exceeds)[0][-1]) + 1
    else:
        perm_n = 0
    perm_cv_r2 = cumul_r2_cv[perm_n - 1] if perm_n > 0 else 0.0
    print(f"  Permutation baseline: {perm_n} dims (CV R²={perm_cv_r2:.4f})")

    # --- Summary table ---
    cutoff_methods = {
        'Kneedle (elbow)': (kneedle_n, kneedle_r2),
        'Adjusted R² peak': (adjr2_n, adjr2_cv_r2),
        'BIC minimum': (bic_n, bic_cv_r2),
        'Permutation baseline': (perm_n, perm_cv_r2),
        'Peak CV R²': (optimal_n_dims, optimal_cv_r2),
    }

    method_dims = [v[0] for v in cutoff_methods.values() if v[0] > 0]
    median_dims = int(np.median(method_dims)) if method_dims else optimal_n_dims

    print("\n" + "=" * 60)
    print("ELBOW/CUTOFF DETECTION SUMMARY")
    print("=" * 60)
    print(f"  {'Method':<30s}  {'# Dims':>6s}    {'CV R² at cutoff':>15s}")
    print("-" * 60)
    for name, (nd, r2v) in cutoff_methods.items():
        print(f"  {name:<30s}  {nd:>6d}    {r2v:>15.4f}")
    print("-" * 60)
    print(f"  {'Current BEST_DIMS':<30s}  {len(BEST_DIMS):>6d}")
    print(f"\n  Median across methods: {median_dims} dims")

    # Colors and styles for cutoff lines
    cutoff_styles = {
        'Kneedle (elbow)':      ('tab:green',  '-.',  2.0),
        'Adjusted R² peak':     ('tab:orange', '--',  2.0),
        'BIC minimum':          ('tab:brown',  ':',   2.0),
        'Permutation baseline': ('tab:cyan',   '-',   1.5),
    }

    # =========================================================
    # Cumulative Spearman r2 figure: train-selected order, both targets
    # =========================================================
    print("\n--- Cumulative Spearman r2 (train-selected order, both targets) ---")

    train_rsm_full = np.load(os.path.join(UTILS_DIR, 'train_human_sim.npy'))
    n_stim_train = train_rsm_full.shape[0]
    triu_train = np.triu_indices(n_stim_train, k=1)
    y_train = train_rsm_full[triu_train]

    X_train = np.zeros((len(y_train), n_dims_emb))
    for d in range(n_dims_emb):
        dist = pdist(sparse_pruned_emb[:, [d]], metric='euclidean')
        dmax = dist.max()
        X_train[:, d] = 1 - (dist / dmax) if dmax != 0 else 1 - dist

    # Greedy order selected on the train RSM. dim_selection_train.py caches it;
    # recompute here only if that has never been run.
    train_greedy_csv = os.path.join(RESULTS_BEHAVIORAL_DIR, 'greedy_forward_r2_train.csv')
    if os.path.exists(train_greedy_csv):
        train_order = pd.read_csv(train_greedy_csv)['dim_id'].tolist()
        print(f"  Reusing train greedy order from {os.path.basename(train_greedy_csv)}")
    else:
        # Pearson `cv_r2` on purpose: this fallback must reproduce the order
        # that dim_selection_train.py caches, which is Pearson-selected.
        print("  No cached train greedy order — running selection on the train RSM")
        train_order, rem = [], list(range(n_dims_emb))
        for _ in range(n_dims_emb):
            best_r2, best_d = -np.inf, None
            for d in rem:
                r2_d = cv_r2(X_train[:, train_order + [d]], y_train)
                if r2_d > best_r2:
                    best_r2, best_d = r2_d, d
            train_order.append(best_d)
            rem.remove(best_d)
    assert len(train_order) == n_dims_emb

    # Out-of-sample cumulative curves, both orders x both targets
    fixed_order = list(range(n_dims_emb))
    curves = {}
    for order_name, order in (('fixed', fixed_order), ('greedy', train_order)):
        for tgt_name, X_t, y_t in (('train', X_train, y_train), ('test', X_all, y)):
            curves[(order_name, tgt_name)] = np.array([
                cv_spearman_r2(X_t[:, order[:k]], y_t)
                for k in range(1, n_dims_emb + 1)
            ])

    # Noise ceilings, Spearman to match these Spearman r2 curves. This is the
    # 0.486 / 0.666 pair; the Pearson pair (0.504 / 0.689) bounds R2_det curves.
    train_trip = load_train_triplets(UTILS_DIR)
    test_trip = load_test_triplets(UTILS_DIR)
    test_ids = np.asarray(sorted(t_idx))
    for label, rebuilt, target in (
        ('train', build_rsm(train_trip)[triu_train], y_train),
        ('test', build_rsm(test_trip)[np.ix_(test_ids, test_ids)][triu_idx], y),
    ):
        assert np.allclose(rebuilt, target, atol=1e-8), (
            f"Rebuilt {label} RSM does not match the regression target; "
            "ceiling would be meaningless"
        )
    nc_train, nc_train_ci = split_half_ceiling(
        train_trip, np.arange(n_stim_train), metric='spearman')
    nc_test, nc_test_ci = split_half_ceiling(test_trip, test_ids, metric='spearman')
    print(f"  Ceiling train {nc_train:.3f} [{nc_train_ci[0]:.3f}, {nc_train_ci[1]:.3f}], "
          f"test {nc_test:.3f} [{nc_test_ci[0]:.3f}, {nc_test_ci[1]:.3f}]")

    # Judgments per pair — the quantity that actually sets the ceiling. Stating it in
    # the legend keeps "50 videos beats 250 videos" from reading as a contradiction:
    # the smaller set concentrates a similar judgment budget on 1,225 cells instead of
    # 31,125, so every cell is estimated from ~2x as many responses.
    _, cnt_train = build_rsm(train_trip, counts=True)
    _, cnt_test = build_rsm(test_trip, counts=True)
    jpp_train = cnt_train[triu_train].mean()
    jpp_test = cnt_test[np.ix_(test_ids, test_ids)][triu_idx].mean()
    print(f"  Judgments/pair: train {jpp_train:.2f}, test {jpp_test:.2f}")

    lbl_test = (f'Test RSM — 50 videos, {jpp_test:.1f} judgments/pair '
                f'(ceiling {nc_test:.2f})')
    lbl_train = (f'Train RSM — 250 videos, {jpp_train:.1f} judgments/pair '
                 f'(ceiling {nc_train:.2f})')
    LEG_TITLE = 'Ceiling is set by judgments per pair, not by video count'
    for order_name in ('fixed', 'greedy'):
        print(f"  {order_name:>6s} @ {len(BEST_DIMS)} dims — "
              f"train {curves[(order_name,'train')][len(BEST_DIMS)-1]:.3f} "
              f"({curves[(order_name,'train')][len(BEST_DIMS)-1]/nc_train:.0%} of ceiling), "
              f"test {curves[(order_name,'test')][len(BEST_DIMS)-1]:.3f} "
              f"({curves[(order_name,'test')][len(BEST_DIMS)-1]/nc_test:.0%})")

    TRAIN_C, TEST_C = '#2C6E9B', '#D97A28'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharey=True)
    x = np.arange(1, n_dims_emb + 1)

    panels = (
        (ax1, 'fixed', fixed_order,
         'Fixed numeric order (Dim 0 $\\rightarrow$ 39)', 'Dimension index'),
        (ax2, 'greedy', train_order,
         'Greedy forward order, selected on the train RSM',
         'Greedy step (dimensions ordered by train-RSM gain)'),
    )
    for ax, order_name, order, title, xlabel in panels:
        for lo_hi, mean, colour, name in (
            (nc_test_ci, nc_test, TEST_C, 'Test RSM'),
            (nc_train_ci, nc_train, TRAIN_C, 'Train RSM'),
        ):
            ax.axhspan(lo_hi[0], lo_hi[1], color=colour, alpha=0.13, lw=0, zorder=0)
            ax.axhline(mean, color=colour, ls=(0, (4, 3)), lw=1.1, alpha=0.85, zorder=1)
            ax.text(0.7, lo_hi[1] + 0.008, f'{name} noise ceiling ({mean:.2f})',
                    fontsize=8.5, color=colour, va='bottom', ha='left', zorder=3)

        ax.plot(x, curves[(order_name, 'test')], marker='o', ms=3.5, color=TEST_C,
                lw=1.8, zorder=4, label=lbl_test)
        ax.plot(x, curves[(order_name, 'train')], marker='o', ms=3.5, color=TRAIN_C,
                lw=1.8, zorder=4, label=lbl_train)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Cross-validated Spearman $r^2$", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Dim {d}" for d in order], rotation=90, fontsize=7.5)
        ax.set_xlim(0.4, n_dims_emb + 0.6)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        leg = ax.legend(fontsize=8.5, frameon=False, loc='lower right',
                        title=LEG_TITLE)
        leg.get_title().set_fontsize(8)
        leg.get_title().set_color('0.35')

    # Kneedle-elbow marker, greedy panel only. Labelled "Kneedle" because that is what
    # the line marks; the guard still tests against BEST_DIMS, since the two are the same
    # 11 dims by construction and a mismatch would mean the cached order has drifted.
    n_best = len(BEST_DIMS)
    if set(train_order[:n_best]) == set(BEST_DIMS):
        ax2.axvline(x=n_best, color='0.35', ls=':', lw=1.3, zorder=2)
        ax2.annotate(f'Kneedle: {n_best} Dims', xy=(n_best + 0.6, 0.02),
                     fontsize=9, color='0.25', va='bottom', ha='left', zorder=5)
    else:
        print(f"  NOTE: train_order[:{n_best}] != BEST_DIMS "
              f"({sorted(set(train_order[:n_best]) ^ set(BEST_DIMS))} differ) — "
              "BEST_DIMS line omitted")

    ax1.set_ylim(0, max(nc_test_ci[1], nc_train_ci[1]) * 1.10)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_BEHAVIORAL_DIR, 'cumulative_r2_test_50vids.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close(fig)

    # =========================================================
    # Single-panel version: greedy (kneedle) panel only
    # =========================================================
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for lo_hi, mean, colour, name in (
        (nc_test_ci, nc_test, TEST_C, 'Test RSM'),
        (nc_train_ci, nc_train, TRAIN_C, 'Train RSM'),
    ):
        ax.axhspan(lo_hi[0], lo_hi[1], color=colour, alpha=0.13, lw=0, zorder=0)
        ax.axhline(mean, color=colour, ls=(0, (4, 3)), lw=1.1, alpha=0.85, zorder=1)
        ax.text(0.7, lo_hi[1] + 0.008, f'{name} noise ceiling ({mean:.2f})',
                fontsize=8.5, color=colour, va='bottom', ha='left', zorder=3)

    ax.plot(x, curves[('greedy', 'test')], marker='o', ms=3.5, color=TEST_C,
            lw=1.8, zorder=4, label=lbl_test)
    ax.plot(x, curves[('greedy', 'train')], marker='o', ms=3.5, color=TRAIN_C,
            lw=1.8, zorder=4, label=lbl_train)

    ax.set_title('Greedy forward order, selected on the train RSM', fontsize=12)
    ax.set_xlabel('Greedy step (dimensions ordered by train-RSM gain)', fontsize=11)
    ax.set_ylabel('Cross-validated Spearman $r^2$', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Dim {d}" for d in train_order], rotation=90, fontsize=7.5)
    ax.set_xlim(0.4, n_dims_emb + 0.6)
    ax.set_ylim(0, max(nc_test_ci[1], nc_train_ci[1]) * 1.10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(fontsize=8.5, frameon=False, loc='lower right', title=LEG_TITLE)
    leg.get_title().set_fontsize(8)
    leg.get_title().set_color('0.35')

    if set(train_order[:n_best]) == set(BEST_DIMS):
        ax.axvline(x=n_best, color='0.35', ls=':', lw=1.3, zorder=2)
        ax.annotate(f'Kneedle: {n_best} Dims', xy=(n_best + 0.6, 0.02),
                    fontsize=9, color='0.25', va='bottom', ha='left', zorder=5)

    plt.tight_layout()
    kneedle_path = os.path.join(RESULTS_BEHAVIORAL_DIR,
                                'cumulative_r2_test_50vids_kneedle.png')
    fig.savefig(kneedle_path, dpi=300, bbox_inches='tight')
    fig.savefig(kneedle_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {kneedle_path}")
    plt.close(fig)

    # =========================================================
    # Cross-validated greedy forward selection plot (enhanced)
    # =========================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(steps, cumul_r2_forward, marker='o', color='violet', linewidth=2,
            alpha=0.4, label='In-sample R²')
    ax.plot(steps, cumul_r2_cv, marker='s', color='tab:blue', linewidth=2,
            label='Cross-validated R² (10-fold, 5 repeats)')

    # Permutation null 95th percentile band
    ax.fill_between(steps, 0, null_95, color='tab:cyan', alpha=0.1,
                    label='Permutation null 95th %ile')

    # Mark the peak CV R² optimum
    ax.axvline(x=optimal_n_dims, color='tab:red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.plot(optimal_n_dims, optimal_cv_r2, 's', color='tab:red', markersize=12, zorder=5)
    ax.annotate(f'Peak CV: {optimal_n_dims} dims\nCV R²={optimal_cv_r2:.3f}',
                xy=(optimal_n_dims, optimal_cv_r2),
                xytext=(optimal_n_dims + 2, optimal_cv_r2 + 0.03),
                fontsize=11, color='tab:red',
                arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.5))

    # Cutoff vertical lines
    for name, (nd, _) in cutoff_methods.items():
        if name == 'Peak CV R²':
            continue  # already plotted
        color, ls, lw = cutoff_styles[name]
        ax.axvline(x=nd, color=color, linestyle=ls, linewidth=lw, alpha=0.8,
                   label=f'{name} ({nd} dims)')

    ax.axhline(noise_ceiling, color='gray', linestyle='--', alpha=0.5)
    ax.text(n_dims_emb, noise_ceiling, f'Noise ceiling ({noise_ceiling:.3f})',
            ha='right', va='bottom', color='gray', alpha=0.7, fontsize=10)

    ax.set_xlabel("Number of dimensions (greedy order)", fontsize=13)
    ax.set_ylabel("Explained variance (R²)", fontsize=13)
    ax.set_title("Greedy Forward Selection: In-Sample vs Cross-Validated R²", fontsize=14)
    ax.set_xticks(steps)
    ax.set_xticklabels([f"Dim {d}" for d in cv_selected], rotation=90, fontsize=8)
    ax.grid(axis='both', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.set_ylim(0, 0.73)

    plt.tight_layout()
    save_path_cv = os.path.join(RESULTS_BEHAVIORAL_DIR, 'greedy_cv_r2_test_50vids.png')
    fig.savefig(save_path_cv, dpi=300, bbox_inches='tight')
    fig.savefig(save_path_cv.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path_cv}")
    plt.close(fig)

    # =========================================================
    # Elbow analysis summary figure
    # =========================================================
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}
    )

    # Left panel: CV R² curve with cutoff lines
    ax_left.plot(steps, cumul_r2_cv, marker='s', color='tab:blue', linewidth=2,
                 label='Cross-validated R²')
    ax_left.fill_between(steps, 0, null_95, color='tab:cyan', alpha=0.1,
                         label='Permutation null 95th %ile')

    for name, (nd, _) in cutoff_methods.items():
        if name == 'Peak CV R²':
            color, ls, lw = 'tab:red', ':', 2.0
        else:
            color, ls, lw = cutoff_styles[name]
        ax_left.axvline(x=nd, color=color, linestyle=ls, linewidth=lw, alpha=0.8,
                        label=f'{name} ({nd})')

    ax_left.axhline(noise_ceiling, color='gray', linestyle='--', alpha=0.5)
    ax_left.set_xlabel("Number of dimensions", fontsize=13)
    ax_left.set_ylabel("Cross-validated R²", fontsize=13)
    ax_left.set_title("CV R² Curve with Cutoff Methods", fontsize=13)
    ax_left.legend(fontsize=8, loc='lower right')
    ax_left.grid(axis='both', linestyle='--', alpha=0.3)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.set_ylim(0, 0.73)

    # Right panel: horizontal bar chart of recommended dims
    bar_names = list(cutoff_methods.keys())
    bar_vals = [cutoff_methods[n][0] for n in bar_names]
    bar_colors_list = [
        cutoff_styles.get(n, ('tab:red', '', 0))[0] for n in bar_names
    ]
    bar_y = np.arange(len(bar_names))

    ax_right.barh(bar_y, bar_vals, color=bar_colors_list, edgecolor='black', linewidth=0.8)
    ax_right.axvline(x=len(BEST_DIMS), color='black', linestyle='--', linewidth=1.5,
                     label=f'Current ({len(BEST_DIMS)} dims)')
    ax_right.axvline(x=median_dims, color='gray', linestyle='-', linewidth=1.5, alpha=0.6,
                     label=f'Median ({median_dims} dims)')
    ax_right.set_yticks(bar_y)
    ax_right.set_yticklabels(bar_names, fontsize=10)
    ax_right.set_xlabel("Recommended # dimensions", fontsize=13)
    ax_right.set_title("Method Comparison", fontsize=13)
    ax_right.legend(fontsize=9, loc='lower right')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)

    # Add value labels on bars
    for i, v in enumerate(bar_vals):
        ax_right.text(v + 0.3, i, str(v), va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path_elbow = os.path.join(RESULTS_BEHAVIORAL_DIR, 'elbow_analysis_summary.png')
    fig.savefig(save_path_elbow, dpi=300, bbox_inches='tight')
    fig.savefig(save_path_elbow.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path_elbow}")
    plt.close(fig)

    print("\n" + "=" * 50)
    print("EXPLAINED VARIANCE ANALYSIS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
