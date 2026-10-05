"""
Neural RSA: construct neural RSMs from fMRI data, per-dimension RSA,
and full neural RSA with permutation testing.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    UTILS_DIR, MODEL_DIR, RESULTS_NEURAL_DIR, RSM_DIR,
    ROI_ORDER, SUBJECTS, BEST_DIMS, test_idx, FMRI_EXPORT_DIR
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

os.makedirs(RESULTS_NEURAL_DIR, exist_ok=True)
os.makedirs(RSM_DIR, exist_ok=True)


# ============================================================================
# BUILD BRAIN RSMs FROM fMRI DATA
# ============================================================================

def build_brain_rsms(metadata_path, response_path, subjects, test_idx=None):
    """
    Build neural RSMs from fMRI voxel beta data.

    Args:
        metadata_path: Path to metadata.csv (subj_id, roi_name, voxel_id)
        response_path: Path to response_data.csv.gz (voxel betas, stimuli x voxels)
        subjects: List of subject IDs
        test_idx: Optional list of test video indices to subset

    Returns:
        brain_rsms: dict[str, dict[str, ndarray]] - subject -> ROI -> RSM
    """
    metadata = pd.read_csv(metadata_path)
    response_data = pd.read_csv(response_path)

    brain_rsms = {}
    for sub in subjects:
        sub_str = str(sub)
        sub_meta = metadata[metadata['subj_id'] == sub]
        rois = [r for r in sub_meta['roi_name'].unique() if r != 'none']

        roi_rsms = {}
        response_values = response_data.values  # (n_voxels, n_stimuli)
        for roi in rois:
            roi_meta = sub_meta[sub_meta['roi_name'] == roi]
            voxel_ids = roi_meta['voxel_id'].values

            # Response data is voxels x stimuli (rows=voxels, cols=stimuli)
            # Select rows for this ROI's voxels, then transpose to stimuli x voxels
            valid_ids = voxel_ids[voxel_ids < response_values.shape[0]]
            if len(valid_ids) == 0:
                continue

            betas = response_values[valid_ids, :].T  # (n_stimuli, n_voxels)

            # Subset to test videos if specified
            if test_idx is not None:
                betas = betas[test_idx, :]

            # Compute RSM: 1 - correlation distance
            rsm = 1 - pairwise_distances(betas, metric='correlation')
            roi_rsms[roi] = rsm

        brain_rsms[sub_str] = roi_rsms

    return brain_rsms


def load_precomputed_brain_rsms(rsm_dir, subjects, roi_order):
    """Load precomputed neural RSMs from .npy files."""
    brain_rsms = {}
    for sub in subjects:
        sub_str = str(sub)
        roi_rsms = {}
        for roi in roi_order:
            path = os.path.join(rsm_dir, f'brain_rsm_sub{sub}_{roi}.npy')
            if os.path.exists(path):
                roi_rsms[roi] = np.load(path)
        if roi_rsms:
            brain_rsms[sub_str] = roi_rsms
    return brain_rsms


# ============================================================================
# PER-DIMENSION RSA
# ============================================================================

def _spearman(x, y):
    """Spearman correlation handling scipy version differences."""
    res = spearmanr(x, y, nan_policy="omit")
    return getattr(res, "correlation", res[0])


def compute_per_dim_rsa_against_brain(
    embeddings, brain_rsms, dim_indices,
    metric="euclidean", space="similarity", normalize_dist=True
):
    """
    Per-dimension RSA: Spearman correlation between each embedding dimension's
    RSM and each brain ROI RSM, averaged across subjects.

    Args:
        embeddings: [n_stim, n_dims] array
        brain_rsms: dict[str, dict[str, ndarray]] - subject -> ROI -> RSM
        dim_indices: list of dimension indices to test
        metric: distance metric for 1D embedding
        space: "similarity" or "distance"
        normalize_dist: whether to min-max normalize distances

    Returns:
        rsa_mean: DataFrame (dims x ROIs) with mean Spearman r
        rsa_sem: DataFrame (dims x ROIs) with SEM
        rsa_long: Long-form DataFrame with per-subject values
    """
    emb = np.asarray(embeddings)
    n_stim = emb.shape[0]

    first_subj = next(iter(brain_rsms.keys()))
    roi_names = list(brain_rsms[first_subj].keys())
    iu = np.triu_indices(n_stim, k=1)

    # Vectorize brain RSMs
    brain_vecs = {roi: [] for roi in roi_names}
    subj_ids = []
    for subj_id, roi_dict in brain_rsms.items():
        subj_ids.append(subj_id)
        for roi in roi_names:
            M = np.asarray(roi_dict[roi])
            v = (1.0 - M) if space == "distance" else M
            brain_vecs[roi].append(v[iu])

    brain_vecs = {roi: np.vstack(vs) for roi, vs in brain_vecs.items()}
    n_subj = len(subj_ids)

    def emb_vec_for_dim(d):
        x = emb[:, d].reshape(-1, 1)
        dv = pdist(x, metric=metric)
        if space == "distance":
            return dv
        if normalize_dist:
            dmin, dmax = dv.min(), dv.max()
            if dmax > dmin:
                dv = (dv - dmin) / (dmax - dmin)
            else:
                dv = np.zeros_like(dv)
        return 1.0 - dv

    rows_mean, rows_sem, rows_long = [], [], []
    for d in dim_indices:
        emb_vec = emb_vec_for_dim(d)
        mean_vals, sem_vals = [], []
        for roi in roi_names:
            subj_mat = brain_vecs[roi]
            rs = [_spearman(emb_vec, subj_mat[s_i]) for s_i in range(n_subj)]
            mean_vals.append(np.nanmean(rs))
            sem_vals.append(np.nanstd(rs, ddof=1) / np.sqrt(n_subj) if n_subj > 1 else np.nan)
            for s_i, r in enumerate(rs):
                rows_long.append((f"dim_{d}", roi, subj_ids[s_i], r))
        rows_mean.append(mean_vals)
        rows_sem.append(sem_vals)

    rsa_mean = pd.DataFrame(rows_mean, index=[f"dim_{d}" for d in dim_indices], columns=roi_names)
    rsa_sem = pd.DataFrame(rows_sem, index=[f"dim_{d}" for d in dim_indices], columns=roi_names)
    rsa_long = pd.DataFrame(rows_long, columns=["dim", "roi", "subject", "spearman_r"])
    return rsa_mean, rsa_sem, rsa_long


# ============================================================================
# FULL NEURAL RSA WITH PERMUTATION TESTING
# ============================================================================

def full_neural_rsa(embeddings, brain_rsms, best_dims, test_idx=None,
                    num_permutations=5000, rng_seed=0):
    """
    Full neural RSA: embedding RSM (best_dims) vs brain ROI RSMs
    with permutation testing per subject.

    Args:
        embeddings: [n_videos, n_dims] array
        brain_rsms: dict[str, dict[str, ndarray]]
        best_dims: list of best dimension indices
        test_idx: optional video indices for subsetting embeddings
        num_permutations: number of permutations for p-values
        rng_seed: random seed

    Returns:
        results_df: DataFrame with columns: ROI, Subject, Spearman_r, P_value, is_significant
    """
    rng = np.random.default_rng(rng_seed)

    # Build embedding RSM from best dimensions
    emb = embeddings[test_idx][:, best_dims] if test_idx is not None else embeddings[:, best_dims]
    emb_rsm = 1 - pairwise_distances(emb, metric='correlation')
    emb_rsm_flat = emb_rsm[np.triu_indices_from(emb_rsm, k=1)]

    results = []
    for sub_str, roi_dict in brain_rsms.items():
        for roi, rsm in roi_dict.items():
            neural_flat = rsm[np.triu_indices_from(rsm, k=1)]

            # Observed correlation
            r_obs, _ = spearmanr(neural_flat, emb_rsm_flat)

            # Permutation testing
            perm_rs = np.empty(num_permutations)
            for p in tqdm(range(num_permutations),
                          desc=f"Perm sub{sub_str} {roi}", leave=False):
                perm_idx = rng.permutation(len(neural_flat))
                perm_rs[p], _ = spearmanr(neural_flat[perm_idx], emb_rsm_flat)

            p_value = (np.sum(perm_rs >= r_obs) + 1) / (num_permutations + 1)

            results.append({
                'ROI': roi,
                'Subject': sub_str,
                'Spearman_r': r_obs,
                'P_value': round(p_value, 4),
                'is_significant': p_value <= 0.05,
            })
            print(f"  {roi}, Sub {sub_str}: r={r_obs:.4f}, p={p_value:.4f}")

    return pd.DataFrame(results)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_per_dim_rsa_heatmap(rsa_mean, dim_indices, save_path=None):
    """Heatmap of per-dimension RSA results."""
    import seaborn as sns

    # Reorder ROIs
    desired_order = ROI_ORDER
    col_map = {c.lower(): c for c in rsa_mean.columns}
    present = [col_map[r.lower()] for r in desired_order if r.lower() in col_map]
    rsa_ordered = rsa_mean[present]

    fig, ax = plt.subplots(figsize=(12, max(6, len(dim_indices) * 0.4)))
    sns.heatmap(
        rsa_ordered, annot=True, fmt=".2f",
        cmap="viridis", vmin=0, vmax=1, ax=ax,
        cbar_kws={'label': 'Spearman r (mean across subjects)'}
    )
    ax.set_title("Per-Dimension RSA vs Neural RSMs (mean across subjects)", fontsize=14)
    ax.set_ylabel("Embedding Dimension", fontsize=12)
    ax.set_xlabel("ROI", fontsize=12)
    ax.set_yticklabels([f"{d}" for d in dim_indices], rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_neural_rsa_bars(results_df, metric_col='Spearman_r', ylabel='Spearman r',
                         ylim_max=0.2, save_path=None):
    """
    Bar plot of neural RSA results split into lateral and ventral ROIs.
    Averages across subjects.
    """
    avg = results_df.groupby('ROI', as_index=False).mean(numeric_only=True)
    avg['is_significant'] = avg['P_value'] <= 0.05

    avg['ROI'] = pd.Categorical(avg['ROI'], categories=ROI_ORDER, ordered=True)
    avg = avg.sort_values('ROI').dropna(subset=['ROI'])

    lateral_rois = [r for r in ROI_ORDER if r not in ['FFA', 'PPA']]
    ventral_rois = ['FFA', 'PPA']

    df_main = avg[avg['ROI'].isin(lateral_rois)]
    df_right = avg[avg['ROI'].isin(ventral_rois)]

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(8, 6))

    # Lateral ROIs
    bars1 = ax1.bar(df_main["ROI"].astype(str), df_main[metric_col], color='tab:blue',
                    edgecolor='black', linewidth=0.8)
    ax1.set_xlabel("Lateral ROIs", fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, ylim_max)

    for i, bar in enumerate(bars1):
        if df_main["is_significant"].iloc[i]:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     '*', ha='center', color='gray', fontsize=12)

    # Ventral ROIs
    if not df_right.empty:
        bars2 = ax2.bar(df_right["ROI"].astype(str), df_right[metric_col], color='tab:blue',
                        edgecolor='black', linewidth=0.8)
        ax2.set_xlabel("Ventral ROIs", fontsize=14)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_ylim(0, ylim_max)

        for i, bar in enumerate(bars2):
            if df_right["is_significant"].iloc[i]:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         '*', ha='center', color='gray', fontsize=12)

    fig.text(0.99, 0.01, "* p < 0.05", ha='right', va='center',
             fontsize=10, style='italic')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 50)
    print("NEURAL RSA")
    print("=" * 50)

    # Load embeddings
    sparse_pruned_emb = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    print(f"Loaded embeddings: {sparse_pruned_emb.shape}")

    # Try to load brain RSMs
    metadata_path = os.path.join(FMRI_EXPORT_DIR, 'metadata.csv')
    response_path = os.path.join(FMRI_EXPORT_DIR, 'response_data.csv.gz')

    if os.path.exists(metadata_path) and os.path.exists(response_path):
        print("Building brain RSMs from fMRI data...")
        brain_rsms = build_brain_rsms(
            metadata_path, response_path,
            SUBJECTS, test_idx=test_idx
        )

        # Save RSMs
        for sub_str, roi_dict in brain_rsms.items():
            for roi, rsm in roi_dict.items():
                np.save(os.path.join(RSM_DIR, f'brain_rsm_sub{sub_str}_{roi}.npy'), rsm)
        print(f"Saved brain RSMs to {RSM_DIR}")

    elif any(os.path.exists(os.path.join(RSM_DIR, f'brain_rsm_sub{s}_{ROI_ORDER[0]}.npy'))
             for s in SUBJECTS):
        print("Loading precomputed brain RSMs...")
        brain_rsms = load_precomputed_brain_rsms(RSM_DIR, SUBJECTS, ROI_ORDER)
    else:
        print("ERROR: No neural data found.")
        print(f"  Expected: {metadata_path} and {response_path}")
        print(f"  Or precomputed RSMs in {RSM_DIR}")
        return

    n_subjects = len(brain_rsms)
    n_rois = len(next(iter(brain_rsms.values())))
    print(f"Loaded brain RSMs: {n_subjects} subjects, {n_rois} ROIs")

    # Determine embedding subset
    if test_idx is not None:
        emb_test = sparse_pruned_emb[test_idx, :]
    else:
        emb_test = sparse_pruned_emb

    # Per-dimension RSA
    print("\n--- Per-Dimension RSA ---")
    n_dims = emb_test.shape[1]
    all_dims = list(range(n_dims))

    rsa_mean, rsa_sem, rsa_long = compute_per_dim_rsa_against_brain(
        embeddings=emb_test,
        brain_rsms=brain_rsms,
        dim_indices=all_dims,
        metric="euclidean",
        space="similarity",
        normalize_dist=True
    )

    rsa_mean.to_csv(os.path.join(RESULTS_NEURAL_DIR, 'per_dim_neural_rsa_mean.csv'))
    rsa_long.to_csv(os.path.join(RESULTS_NEURAL_DIR, 'per_dim_neural_rsa_long.csv'), index=False)

    plot_per_dim_rsa_heatmap(
        rsa_mean, all_dims,
        save_path=os.path.join(RESULTS_NEURAL_DIR, 'per_dim_neural_rsa_heatmap.png')
    )
    print(f"Saved per-dimension RSA results to {RESULTS_NEURAL_DIR}")

    # Full neural RSA (embedding RSM vs brain RSMs)
    print("\n--- Full Neural RSA ---")
    neural_results = full_neural_rsa(
        embeddings=sparse_pruned_emb,
        brain_rsms=brain_rsms,
        best_dims=BEST_DIMS,
        test_idx=test_idx,
        num_permutations=5000
    )

    neural_results.to_csv(os.path.join(RESULTS_NEURAL_DIR, 'neural_rsa_results.csv'), index=False)

    # Plot
    plot_neural_rsa_bars(
        neural_results,
        metric_col='Spearman_r',
        ylabel='Spearman r',
        ylim_max=0.2,
        save_path=os.path.join(RESULTS_NEURAL_DIR, 'neural_rsa_bars.png')
    )

    # Also plot R2
    neural_results['R2'] = neural_results['Spearman_r'] ** 2
    plot_neural_rsa_bars(
        neural_results,
        metric_col='R2',
        ylabel='Explained Variance (R$^2$)',
        ylim_max=0.05,
        save_path=os.path.join(RESULTS_NEURAL_DIR, 'neural_rsa_bars_r2.png')
    )

    print(f"\nSaved neural RSA results and plots to {RESULTS_NEURAL_DIR}")

    print("\n" + "=" * 50)
    print("NEURAL RSA COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
