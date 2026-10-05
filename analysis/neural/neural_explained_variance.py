"""
Neural-Behavioral Explained Variance Analysis.

Analysis 1: Behavioral features → each neural ROI's RSM (which behavioral
features explain variance in each brain region).

Analysis 2: Total behavioral vs total embedding R² → each neural ROI's RSM
(grouped bar chart comparing how well behavior and learned embeddings
each explain neural representations).

Analysis 3: Neural ROIs → Learned Embedding RSM.
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
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, pairwise_distances

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    UTILS_DIR, MODEL_DIR, RESULTS_NEURAL_DIR, RSM_DIR,
    BEHAVIOR_COLS, FEATURE_ORDER, FEATURE_REMAPS,
    ROI_ORDER, SUBJECTS, RANDOM_SEED, test_idx, BEST_DIMS, DIM_LABELS,
    FMRI_BETAS_DIR, FMRI_FREESURFER_DIR, FMRI_EXPORT_DIR,
)
from helpers.variance_partition import (
    compute_explained_variance, compute_explained_variance_spearman, cv_r2, cv_spearman_r2, build_behavior_rsms, format_behavior_df, get_bar_colors,
)
from analysis.neural.neural_rsa import load_precomputed_brain_rsms, build_brain_rsms
from helpers.action_categories import load_action_categories, build_action_rsm

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

os.makedirs(RESULTS_NEURAL_DIR, exist_ok=True)


def average_brain_rsms(brain_rsms, roi_order):
    """Average brain RSMs across subjects for each ROI.

    Args:
        brain_rsms: dict[str, dict[str, ndarray]] - subject -> ROI -> RSM
        roi_order: list of ROI names

    Returns:
        dict[str, ndarray] - ROI -> averaged (50x50) RSM
    """
    avg = {}
    for roi in roi_order:
        stacked = []
        for sub_str, roi_dict in brain_rsms.items():
            if roi in roi_dict:
                stacked.append(roi_dict[roi])
        if stacked:
            avg[roi] = np.mean(stacked, axis=0)
    return avg


def noise_ceiling_bounds(brain_rsms, roi_order, squared=True):
    """Noise ceiling bounds per ROI following Nili et al. (2014).

    Upper bound: mean r(subject_i, mean_of_all_subjects), squared.
    Lower bound: mean r(subject_i, mean_of_other_subjects), squared.

    Args:
        brain_rsms: dict[str, dict[str, ndarray]] - subject -> ROI -> RSM
        roi_order: list of ROI names

    Returns:
        dict[str, tuple[float, float]] - ROI -> (lower_R2, upper_R2)
    """
    subjects = list(brain_rsms.keys())

    bounds = {}
    for roi in roi_order:
        sub_rsms = []
        for s in subjects:
            if roi in brain_rsms[s]:
                sub_rsms.append(brain_rsms[s][roi])
        if len(sub_rsms) < 2:
            continue

        stacked = np.array(sub_rsms)
        n = stacked.shape[0]
        triu = np.triu_indices(stacked.shape[1], k=1)
        mean_all = stacked.mean(axis=0)

        upper_rs = []
        lower_rs = []
        for i in range(n):
            subj_vec = stacked[i][triu]

            # Upper: correlate with mean of all subjects
            r_upper, _ = spearmanr(subj_vec, mean_all[triu])
            upper_rs.append(r_upper)

            # Lower: correlate with mean of other subjects (leave-one-out)
            others = np.delete(stacked, i, axis=0).mean(axis=0)
            r_lower, _ = spearmanr(subj_vec, others[triu])
            lower_rs.append(r_lower)

        lo, hi = np.mean(lower_rs), np.mean(upper_rs)
        # squared=False gives the bounds in r units, for figures plotted in r
        bounds[roi] = (lo ** 2, hi ** 2) if squared else (lo, hi)

    return bounds


def plot_per_roi_variance(results_per_roi, save_path):
    """3x3 grid of bar charts showing behavioral explained variance per ROI.

    Args:
        results_per_roi: dict[roi -> (r2_full, unique_vars, shared)]
        save_path: output path for the figure
    """
    n_rois = len(results_per_roi)
    ncols = 3
    nrows = (n_rois + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), sharey=True)
    axes = axes.flatten()

    roi_list = [r for r in ROI_ORDER if r in results_per_roi]

    for i, roi in enumerate(roi_list):
        ax = axes[i]
        r2_full, unique_vars, shared = results_per_roi[roi]

        df = format_behavior_df(dict(unique_vars), shared, r2_full)
        colors = get_bar_colors(df["Feature"], base_color="tab:blue")
        x = np.arange(len(df))

        ax.bar(x, df["Variance"].values, width=0.6,
               color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(roi, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Feature"], rotation=90, ha="center", fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i % ncols == 0:
            ax.set_ylabel("Unique R²", fontsize=11)

    # Hide unused axes
    for j in range(len(roi_list), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Behavioral Features → Neural ROI RSMs (Unique Explained Variance)",
                 fontsize=16, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_joint_variance(roi_total_r2, noise_ceiling_bounds, save_path):
    """Grouped bar chart: total behavioral vs total embedding R² per ROI.

    Args:
        roi_total_r2: list of dicts with keys ROI, Behavioral_R2, Embedding_R2
        noise_ceiling_bounds: dict[roi -> (lower_R2, upper_R2)]
        save_path: output path
    """
    df = pd.DataFrame(roi_total_r2)
    rois = df['ROI'].tolist()
    x = np.arange(len(rois))
    width = 0.35
    bar_half = width + 0.05  # half-width of the band per ROI

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, df['Behavioral_R2'].values, width,
           label='Behavioral Features', color='tab:purple', edgecolor='black', linewidth=0.8)
    ax.bar(x + width / 2, df['Embedding_R2'].values, width,
           label='Learned Dimensions', color='tab:blue', edgecolor='black', linewidth=0.8)

    # Noise ceiling as shaded band (lower to upper bound) per ROI
    for i, roi in enumerate(rois):
        if roi in noise_ceiling_bounds:
            lo, hi = noise_ceiling_bounds[roi]
            ax.fill_between(
                [x[i] - bar_half, x[i] + bar_half], lo, hi,
                color='lightgray', alpha=0.7, zorder=0, edgecolor='none',
            )

    ax.set_xlabel("Neural ROI", fontsize=14)
    ax.set_ylabel("Total Explained Variance (R²)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=0, ha="center", fontsize=12)
    ax.legend(fontsize=12, loc='center', ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.95))
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set y_lim to max noise ceiling upper bound + 0.05
    if noise_ceiling_bounds:
        max_hi = max(hi for lo, hi in noise_ceiling_bounds.values())
        ax.set_ylim(0, max_hi + 0.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_neural_to_embedding_variance(unique_vars, shared, r2_full, save_path,
                                      noise_ceiling=None):
    """Bar chart of neural ROI unique R² predicting the embedding RSM.

    Args:
        unique_vars: dict[roi -> unique R²]
        shared: float, shared variance
        r2_full: float, total R²
        save_path: output path
        noise_ceiling: float, optional y-limit to match other plots
    """
    # Build ordered DataFrame
    data = dict(unique_vars)
    data["Shared Variance"] = shared
    data["Total Variance"] = r2_full

    roi_plot_order = [r for r in ROI_ORDER if r in unique_vars] + ["Shared Variance", "Total Variance"]
    df = (
        pd.DataFrame(data, index=["Variance"])
        .T.reset_index()
        .rename(columns={"index": "Feature"})
    )
    df["Feature"] = pd.Categorical(df["Feature"], categories=roi_plot_order, ordered=True)
    df = df.sort_values("Feature")
    df["Feature"] = df["Feature"].astype(str)

    colors = get_bar_colors(df["Feature"], base_color="tab:green")
    labels = [f"{f} $^{{\\rm (UV)}}$" if "Shared" not in f and "Total" not in f else f
              for f in df["Feature"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    ax.bar(x, df["Variance"].values, width=0.6,
           color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Neural ROI", fontsize=14)
    ax.set_ylabel("Explained Variance (R²)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Neural ROIs → Learned Embedding RSM (Unique Explained Variance, Test Set)",
                 fontsize=14)
    if noise_ceiling is not None:
        ax.set_ylim(0, noise_ceiling)
        ax.axhline(y=noise_ceiling, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_best_per_roi(beh_unique_per_roi, dim_unique_per_roi, noise_ceiling_bounds,
                      save_path, roi_list=None):
    """Bar chart: best single behavioral feature vs best single dimension per ROI.

    Each bar is labeled with the name of the winning feature/dimension.

    Args:
        beh_unique_per_roi: dict[roi -> dict[feature -> unique R²]]
        dim_unique_per_roi: dict[roi -> dict[dim_label -> unique R²]]
        noise_ceiling_bounds: dict[roi -> (lower_R2, upper_R2)]
        save_path: output path
        roi_list: optional subset/order of ROIs to plot
    """
    if roi_list is None:
        roi_list = [r for r in ROI_ORDER if r in beh_unique_per_roi]

    rois = []
    best_beh_vals, best_beh_names = [], []
    best_dim_vals, best_dim_names = [], []

    for roi in roi_list:
        if roi not in beh_unique_per_roi or roi not in dim_unique_per_roi:
            continue
        rois.append(roi)

        beh_uv = beh_unique_per_roi[roi]
        best_b = max(beh_uv, key=beh_uv.get)
        best_beh_names.append(best_b)
        best_beh_vals.append(beh_uv[best_b])

        dim_uv = dim_unique_per_roi[roi]
        best_d = max(dim_uv, key=dim_uv.get)
        best_dim_names.append(best_d)
        best_dim_vals.append(dim_uv[best_d])

    x = np.arange(len(rois))
    width = 0.35
    bar_half = width + 0.05

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_beh = ax.bar(x - width / 2, best_beh_vals, width,
                      label='Best Behavioral Feature', color='tab:purple',
                      edgecolor='black', linewidth=0.8)
    bars_dim = ax.bar(x + width / 2, best_dim_vals, width,
                      label='Best Learned Dimension', color='tab:blue',
                      edgecolor='black', linewidth=0.8)

    for i, roi in enumerate(rois):
        if roi in noise_ceiling_bounds:
            lo, hi = noise_ceiling_bounds[roi]
            ax.fill_between(
                [x[i] - bar_half, x[i] + bar_half], lo, hi,
                color='lightgray', alpha=0.7, zorder=0, edgecolor='none',
            )

    for bar, name in zip(bars_beh, best_beh_names):
        label = name.replace(' - ', '\n').replace(' ', '\n', 1) if ' - ' not in name else name.replace(' - ', '\n')
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                label, ha='center', va='bottom', fontsize=6,
                fontstyle='italic', linespacing=0.9)
    for bar, name in zip(bars_dim, best_dim_names):
        label = name.replace(' - ', '\n')
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                label, ha='center', va='bottom', fontsize=6,
                fontstyle='italic', linespacing=0.9)

    ax.set_xlabel("Neural ROI", fontsize=14)
    ax.set_ylabel("Best Unique Variance (Spearman r²)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=0, ha="center", fontsize=12)
    ax.legend(fontsize=11, loc='center', ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.95))
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if noise_ceiling_bounds:
        relevant = {k: v for k, v in noise_ceiling_bounds.items() if k in rois}
        if relevant:
            max_hi = max(hi for lo, hi in relevant.values())
            ax.set_ylim(0, max_hi + 0.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_best_dim_brain_surface(dim_unique_per_roi, metadata_path, save_path):
    """Plot brain surface with each ROI colored by its best embedding dimension.

    Each ROI is uniformly colored by the dimension with the highest unique
    variance. The unique R² value is annotated on each ROI centroid.

    Args:
        dim_unique_per_roi: dict[roi -> dict[dim_label -> unique R²]]
        metadata_path: path to metadata.csv with voxel coordinates
        save_path: output path for the figure
    """
    import nibabel as nib
    from nilearn import surface, plotting
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    metadata = pd.read_csv(metadata_path)

    base_colors = [
        '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
        '#dcbeff', '#ffe119',
    ]

    all_winning_dims = set()
    roi_best = {}
    for roi, dim_uv in dim_unique_per_roi.items():
        best_label = max(dim_uv, key=dim_uv.get)
        best_val = dim_uv[best_label]
        roi_best[roi] = (best_label, best_val)
        all_winning_dims.add(best_label)

    all_best_dims_sorted = sorted(BEST_DIMS)
    dim_to_idx = {}
    for d in BEST_DIMS:
        label = f"Dim {d} - {DIM_LABELS[d]}" if d in DIM_LABELS else f"Dim {d}"
        dim_to_idx[label] = BEST_DIMS.index(d)

    n_dims = len(BEST_DIMS)
    colors = base_colors[:n_dims]
    dim_cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, n_dims, 1.0)
    norm = BoundaryNorm(bounds, n_dims)

    for sub in SUBJECTS:
        sub_meta = metadata[metadata['subj_id'] == sub]

        beta_path = os.path.join(
            FMRI_BETAS_DIR,
            f'sub-0{sub}_space-T1w_desc-test-fracridge_data.nii.gz')
        if not os.path.exists(beta_path):
            print(f"  Skipping sub-{sub}: no beta NIfTI")
            continue
        ref_img = nib.load(beta_path)
        vol_shape = ref_img.shape[:3]
        affine = ref_img.affine

        roi_vol = np.full(vol_shape, np.nan, dtype=np.float32)

        for roi, (best_label, best_val) in roi_best.items():
            roi_voxels = sub_meta[sub_meta['roi_name'] == roi]
            if len(roi_voxels) == 0:
                continue
            idx = dim_to_idx.get(best_label)
            if idx is None:
                continue
            ii = roi_voxels['i_index'].values
            jj = roi_voxels['j_index'].values
            kk = roi_voxels['k_index'].values
            roi_vol[ii, jj, kk] = idx + 1

        roi_nii = nib.Nifti1Image(roi_vol, affine)

        surfs = {}
        missing = False
        for hemi in ['lh', 'rh']:
            for s in ['inflated', 'pial', 'sulc']:
                p = os.path.join(FMRI_FREESURFER_DIR,
                                 f'sub-0{sub}', 'surf', f'{hemi}.{s}')
                if not os.path.exists(p):
                    print(f"  WARNING: Missing surface {p}")
                    missing = True
                    break
                surfs[f'{hemi}_{s}'] = p
            if missing:
                break
        if missing:
            continue

        views = [
            ('lh', 'lateral'), ('rh', 'lateral'),
            ('lh', 'medial'), ('rh', 'medial'),
            ('lh', 'ventral'), ('rh', 'ventral'),
        ]

        fig, axes = plt.subplots(1, 6, figsize=(26, 4),
                                 subplot_kw={'projection': '3d'})

        for col, (hemi, view) in enumerate(views):
            tex = surface.vol_to_surf(
                roi_nii, surfs[f'{hemi}_pial'],
                interpolation='nearest_most_frequent')

            plotting.plot_surf_stat_map(
                surfs[f'{hemi}_inflated'], tex,
                hemi='left' if hemi == 'lh' else 'right',
                view=view,
                bg_map=surfs[f'{hemi}_sulc'],
                threshold=0.5,
                vmin=0.5,
                vmax=n_dims + 0.5,
                avg_method='median',
                colorbar=False,
                cmap=dim_cmap,
                axes=axes[col],
                figure=fig,
            )

        fig.subplots_adjust(left=0.01, right=0.72, bottom=0.02, top=0.88,
                            wspace=0.02)
        fig.suptitle(f'Best Embedding Dimension per ROI (Sub-{sub})', fontsize=16)

        used_indices = set()
        label_lines = []
        for roi in ROI_ORDER:
            if roi not in roi_best:
                continue
            best_label, best_val = roi_best[roi]
            idx = dim_to_idx.get(best_label)
            if idx is not None:
                used_indices.add(idx)
            label_lines.append(f"{roi}: {best_label} (r²={best_val:.3f})")

        patches = []
        for i, d in enumerate(BEST_DIMS):
            lbl = f"Dim {d} - {DIM_LABELS.get(d, '')}"
            if i in used_indices:
                patches.append(mpatches.Patch(color=colors[i], label=lbl))

        annotation_text = '\n'.join(label_lines)
        fig.text(0.73, 0.5, annotation_text, fontsize=11, va='center',
                 family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                           edgecolor='gray', alpha=0.9))

        fig.legend(handles=patches, loc='upper right',
                   bbox_to_anchor=(0.99, 0.95), fontsize=11, frameon=True,
                   title='Dimension', title_fontsize=12)

        sub_save = save_path.replace('.png', f'_sub{sub}.png')
        fig.savefig(sub_save, dpi=300, bbox_inches='tight')
        fig.savefig(sub_save.replace('.png', '.pdf'), format='pdf',
                    bbox_inches='tight')
        print(f"Saved: {sub_save}")
        plt.close(fig)


def main():
    print("\n" + "=" * 60)
    print("NEURAL-BEHAVIORAL EXPLAINED VARIANCE ANALYSIS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    annotations = pd.read_csv(os.path.join(UTILS_DIR, 'annotations.csv'))
    print(f"Loaded annotations: {annotations.shape}")

    # Human test RSM
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

    # Test indices
    if test_idx is not None:
        t_idx = test_idx
    else:
        n_test = human_test_rsm.shape[0]
        print(f"WARNING: No test_idx from config, assuming first {n_test} videos")
        t_idx = list(range(n_test))
    n_test = len(t_idx)
    print(f"Test set: {n_test} videos")

    # Brain RSMs
    metadata_path = os.path.join(FMRI_EXPORT_DIR, 'metadata.csv')
    response_path = os.path.join(FMRI_EXPORT_DIR, 'response_data.csv.gz')

    if os.path.exists(metadata_path) and os.path.exists(response_path):
        print("Building brain RSMs from fMRI data...")
        brain_rsms = build_brain_rsms(
            metadata_path, response_path, SUBJECTS, test_idx=test_idx
        )
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

    # ------------------------------------------------------------------
    # Build behavioral RSMs and subset to test set
    # ------------------------------------------------------------------
    rsms_full = build_behavior_rsms(annotations, BEHAVIOR_COLS)
    rsms_test = {f: rsms_full[f][np.ix_(t_idx, t_idx)] for f in rsms_full}

    triu_idx = np.triu_indices(n_test, k=1)

    beh_features = {}
    for feature in rsms_test:
        beh_features[feature.title()] = rsms_test[feature][triu_idx]

    # Add action category RSM
    labels_all = load_action_categories(UTILS_DIR)
    labels_test = labels_all[t_idx]
    action_rsm_test = build_action_rsm(labels_test)
    beh_features['Action Categories'] = action_rsm_test[triu_idx]

    beh_df = pd.DataFrame(beh_features)
    print(f"Behavioral features ({len(beh_df.columns)}): {list(beh_df.columns)}")

    # Average brain RSMs across subjects
    avg_brain = average_brain_rsms(brain_rsms, ROI_ORDER)
    print(f"Averaged brain RSMs: {len(avg_brain)} ROIs")

    # Human RSM upper triangle
    y_human = human_test_rsm[triu_idx]

    # Noise ceiling from split-half reliability
    noise_ceiling = 0.6658  # Spearman-Brown corrected split-half reliability (dense test triplets)

    # ==================================================================
    # Analysis 1: Behavioral → each neural ROI
    # ==================================================================
    print("\n" + "-" * 50)
    print("Analysis 1: Behavioral features → Neural ROI RSMs")
    print("-" * 50)

    results_per_roi = {}
    rows_csv = []

    for roi in ROI_ORDER:
        if roi not in avg_brain:
            print(f"  Skipping {roi}: not available")
            continue

        y_roi = avg_brain[roi][triu_idx]
        r2_full, unique_vars, shared = compute_explained_variance(y_roi, beh_df)

        results_per_roi[roi] = (r2_full, unique_vars, shared)
        print(f"  {roi}: Total R²={r2_full:.4f}, Shared={shared:.4f}")
        for feat, uv in unique_vars.items():
            print(f"    {feat}: {uv:.4f}")
            rows_csv.append({'ROI': roi, 'Feature': feat, 'Unique_R2': uv,
                             'Total_R2': r2_full, 'Shared_R2': shared})

    # Save CSV
    csv_path = os.path.join(RESULTS_NEURAL_DIR, 'neural_explained_variance_per_roi.csv')
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot 3x3 grid
    plot_per_roi_variance(
        results_per_roi,
        os.path.join(RESULTS_NEURAL_DIR, 'neural_explained_variance_per_roi.png')
    )

    # ==================================================================
    # Analysis 2: Behavioral vs Embedding total R² → each Neural ROI
    # ==================================================================
    print("\n" + "-" * 50)
    print("Analysis 2: Behavioral vs Embedding → each Neural ROI (Total R²)")
    print("-" * 50)

    # Load embedding and build per-dimension RSMs for test videos
    sparse_pruned_emb = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    emb_test = sparse_pruned_emb[t_idx, :]
    n_dims = emb_test.shape[1]
    print(f"Embedding dimensions: {n_dims}")

    # Build per-dimension similarity vectors (matching explained_variance.py)
    emb_features = {}
    for d in range(n_dims):
        dist = pdist(emb_test[:, [d]], metric="euclidean")
        dmax = dist.max()
        sim = 1 - (dist / dmax) if dmax != 0 else 1 - dist
        emb_features[f"Dim {d}"] = sim

    emb_df = pd.DataFrame(emb_features)

    # Noise ceiling bounds (Nili et al. 2014): lower (leave-one-out) and upper (mean-all)
    print("\n  Computing noise ceiling bounds (Nili et al. 2014)...")
    noise_ceilings = noise_ceiling_bounds(brain_rsms, ROI_ORDER)
    for roi, (lo, hi) in noise_ceilings.items():
        print(f"    {roi}: lower={lo:.4f}, upper={hi:.4f}")

    roi_total_r2 = []
    for roi in ROI_ORDER:
        if roi not in avg_brain:
            print(f"  Skipping {roi}: not available")
            continue

        y_roi = avg_brain[roi][triu_idx]

        # Cross-validated total R² from all behavioral features
        r2_beh = cv_r2(beh_df.values, y_roi)

        # Cross-validated total R² from all embedding dimensions
        r2_emb = cv_r2(emb_df.values, y_roi)

        roi_total_r2.append({
            'ROI': roi, 'Behavioral_R2': r2_beh, 'Embedding_R2': r2_emb
        })
        print(f"  {roi}: Behavioral CV R²={r2_beh:.4f}, Embedding CV R²={r2_emb:.4f}")

    # Plot (Nili et al. noise ceiling)
    plot_joint_variance(
        roi_total_r2, noise_ceilings,
        os.path.join(RESULTS_NEURAL_DIR, 'neural_behavioral_joint_variance.png')
    )

    # Voxel reliability noise ceiling (plotted after Spearman variant below)

    # Save CSV
    csv_path = os.path.join(RESULTS_NEURAL_DIR, 'neural_behavioral_joint_variance.csv')
    pd.DataFrame(roi_total_r2).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ==================================================================
    # Spearman r² variant: consistent metric for bars + noise ceiling
    # ==================================================================
    print("\n" + "-" * 50)
    print("Spearman r² variant (consistent metric throughout)")
    print("-" * 50)

    # -- Analysis 1 Spearman: Behavioral → each neural ROI (unique/shared) --
    print("\n  Analysis 1 (Spearman): Behavioral → Neural ROI RSMs")
    results_per_roi_sp = {}
    rows_csv_sp = []

    for roi in ROI_ORDER:
        if roi not in avg_brain:
            continue
        y_roi = avg_brain[roi][triu_idx]
        r2_full_sp, unique_vars_sp, shared_sp = compute_explained_variance_spearman(
            y_roi, beh_df
        )
        results_per_roi_sp[roi] = (r2_full_sp, unique_vars_sp, shared_sp)
        print(f"  {roi}: Total Spearman r²={r2_full_sp:.4f}, Shared={shared_sp:.4f}")
        for feat, uv in unique_vars_sp.items():
            rows_csv_sp.append({'ROI': roi, 'Feature': feat, 'Unique_R2': uv,
                                'Total_R2': r2_full_sp, 'Shared_R2': shared_sp})

    csv_path = os.path.join(RESULTS_NEURAL_DIR,
                            'neural_explained_variance_per_roi_spearman.csv')
    pd.DataFrame(rows_csv_sp).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    plot_per_roi_variance(
        results_per_roi_sp,
        os.path.join(RESULTS_NEURAL_DIR,
                     'neural_explained_variance_per_roi_spearman.png')
    )

    # -- Analysis 2 Spearman: Behavioral vs Embedding total → each Neural ROI --
    print("\n  Analysis 2 (Spearman): Behavioral vs Embedding → each Neural ROI")
    roi_spearman_r2 = []
    for roi in ROI_ORDER:
        if roi not in avg_brain:
            continue
        y_roi = avg_brain[roi][triu_idx]
        r2_beh_sp = cv_spearman_r2(beh_df.values, y_roi)
        r2_emb_sp = cv_spearman_r2(emb_df.values, y_roi)
        roi_spearman_r2.append({
            'ROI': roi, 'Behavioral_R2': r2_beh_sp, 'Embedding_R2': r2_emb_sp
        })
        print(f"  {roi}: Behavioral Spearman r²={r2_beh_sp:.4f}, "
              f"Embedding Spearman r²={r2_emb_sp:.4f}")

    plot_joint_variance(
        roi_spearman_r2, noise_ceilings,
        os.path.join(RESULTS_NEURAL_DIR,
                     'neural_behavioral_joint_variance_spearman.png')
    )
    csv_path = os.path.join(RESULTS_NEURAL_DIR,
                            'neural_behavioral_joint_variance_spearman.csv')
    pd.DataFrame(roi_spearman_r2).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Filtered variant: exclude face-pSTS, FFA, PPA
    exclude_rois = {'face-pSTS', 'FFA', 'PPA'}
    roi_spearman_r2_filtered = [r for r in roi_spearman_r2
                                if r['ROI'] not in exclude_rois]
    noise_ceilings_filtered = {k: v for k, v in noise_ceilings.items()
                               if k not in exclude_rois}
    plot_joint_variance(
        roi_spearman_r2_filtered, noise_ceilings_filtered,
        os.path.join(RESULTS_NEURAL_DIR,
                     'neural_behavioral_joint_variance_spearman_filtered.png')
    )

    # BEST_DIMS variant: rerun Analysis 2 using only the best dimensions
    best_dim_cols = [f"Dim {d}" for d in BEST_DIMS if f"Dim {d}" in emb_df.columns]
    missing = [d for d in BEST_DIMS if f"Dim {d}" not in emb_df.columns]
    if missing:
        print(f"  WARNING: BEST_DIMS missing from pruned embedding: {missing}")
    emb_df_best = emb_df[best_dim_cols]
    n_best = len(best_dim_cols)
    print(f"\n  Analysis 2 (Spearman, BEST_DIMS): "
          f"Behavioral vs Embedding[{n_best}] → each Neural ROI")
    print(f"  Using {n_best} dims: {best_dim_cols}")

    roi_spearman_r2_best = []
    for roi in ROI_ORDER:
        if roi not in avg_brain:
            continue
        y_roi = avg_brain[roi][triu_idx]
        r2_beh_sp = cv_spearman_r2(beh_df.values, y_roi)
        r2_emb_sp = cv_spearman_r2(emb_df_best.values, y_roi)
        roi_spearman_r2_best.append({
            'ROI': roi, 'Behavioral_R2': r2_beh_sp, 'Embedding_R2': r2_emb_sp
        })
        print(f"  {roi}: Behavioral Spearman r²={r2_beh_sp:.4f}, "
              f"Embedding[{n_best}] Spearman r²={r2_emb_sp:.4f}")

    plot_joint_variance(
        roi_spearman_r2_best, noise_ceilings,
        os.path.join(RESULTS_NEURAL_DIR,
                     f'neural_behavioral_joint_variance_spearman_best{n_best}.png')
    )
    csv_path = os.path.join(RESULTS_NEURAL_DIR,
                            f'neural_behavioral_joint_variance_spearman_best{n_best}.csv')
    pd.DataFrame(roi_spearman_r2_best).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    roi_spearman_r2_best_filtered = [r for r in roi_spearman_r2_best
                                     if r['ROI'] not in exclude_rois]
    plot_joint_variance(
        roi_spearman_r2_best_filtered, noise_ceilings_filtered,
        os.path.join(RESULTS_NEURAL_DIR,
                     f'neural_behavioral_joint_variance_spearman_best{n_best}_filtered.png')
    )

    # ==================================================================
    # Best-per-ROI plot: highest unique variance feature/dim per ROI
    # ==================================================================
    print("\n" + "-" * 50)
    print("Best unique variance per ROI (behavioral vs embedding dims)")
    print("-" * 50)

    beh_unique_per_roi = {}
    for roi in ROI_ORDER:
        if roi not in results_per_roi_sp:
            continue
        _, unique_vars_sp, _ = results_per_roi_sp[roi]
        beh_unique_per_roi[roi] = dict(unique_vars_sp)

    dim_unique_per_roi = {}
    for roi in ROI_ORDER:
        if roi not in avg_brain:
            continue
        y_roi = avg_brain[roi][triu_idx]
        _, dim_uv, _ = compute_explained_variance_spearman(y_roi, emb_df_best)
        labeled = {}
        for col, uv in dim_uv.items():
            d = int(col.split()[-1])
            label = f"Dim {d} - {DIM_LABELS[d]}" if d in DIM_LABELS else col
            labeled[label] = uv
        dim_unique_per_roi[roi] = labeled
        best_d = max(labeled, key=labeled.get)
        best_b = max(beh_unique_per_roi.get(roi, {}),
                     key=beh_unique_per_roi.get(roi, {}).get, default='N/A')
        print(f"  {roi}: best dim = {best_d} ({labeled[best_d]:.4f}), "
              f"best beh = {best_b}")

    plot_best_per_roi(
        beh_unique_per_roi, dim_unique_per_roi, noise_ceilings,
        os.path.join(RESULTS_NEURAL_DIR, 'neural_best_unique_variance_per_roi.png')
    )
    filtered_rois = [r for r in ROI_ORDER if r not in exclude_rois]
    plot_best_per_roi(
        beh_unique_per_roi, dim_unique_per_roi, noise_ceilings_filtered,
        os.path.join(RESULTS_NEURAL_DIR,
                     'neural_best_unique_variance_per_roi_filtered.png'),
        roi_list=filtered_rois
    )

    # Brain surface: each ROI colored by best dimension
    plot_best_dim_brain_surface(
        dim_unique_per_roi, metadata_path,
        os.path.join(RESULTS_NEURAL_DIR, 'neural_best_dim_per_roi_brain.png')
    )

    # Variant: voxel reliability noise ceiling with Spearman r² bars
    metadata = pd.read_csv(metadata_path)
    voxel_reliability = {}
    for roi in ROI_ORDER:
        roi_voxels = metadata[metadata['roi_name'] == roi]['reliability']
        if len(roi_voxels) > 0:
            mean_rel = roi_voxels.mean()
            voxel_reliability[roi] = (mean_rel ** 2, mean_rel ** 2)
    print("\n  Voxel reliability noise ceiling (mean r, squared):")
    for roi, (lo, _) in voxel_reliability.items():
        print(f"    {roi}: R²={lo:.4f}")
    plot_joint_variance(
        roi_spearman_r2, voxel_reliability,
        os.path.join(RESULTS_NEURAL_DIR, 'neural_behavioral_joint_variance_voxel_reliability.png')
    )

    # ==================================================================
    # Analysis 2b: Per-dim Spearman r² for BEST_DIMS → each Neural ROI
    # Same metric as combined plot: single-dim CV linear regression Spearman r²
    # ==================================================================
    print("\n" + "-" * 50)
    print("Analysis 2b: Per-dim Spearman r² (BEST_DIMS) → each Neural ROI")
    print("-" * 50)

    per_dim_rows = []
    for d in BEST_DIMS:
        col = f"Dim {d}"
        X_d = emb_df[[col]].values
        for roi in ROI_ORDER:
            if roi not in avg_brain:
                continue
            y_roi = avg_brain[roi][triu_idx]
            r2_d = cv_spearman_r2(X_d, y_roi)
            per_dim_rows.append({'Dim': d, 'ROI': roi, 'Spearman_R2': r2_d})
            print(f"  Dim {d:2d} x {roi}: {r2_d:.4f}")

    per_dim_df = pd.DataFrame(per_dim_rows)

    # Pivot and rank by mean across ROIs
    pivot = per_dim_df.pivot(index='Dim', columns='ROI', values='Spearman_R2')
    pivot = pivot[[r for r in ROI_ORDER if r in pivot.columns]]
    pivot['Mean_R2'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Mean_R2', ascending=False)
    print("\n  Ranked by mean Spearman r² across ROIs:")
    print(pivot.to_string())

    csv_path = os.path.join(RESULTS_NEURAL_DIR, 'per_dim_neural_spearman_r2.csv')
    pivot.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")

    # ==================================================================
    # Analysis 3: Neural ROIs → Embedding RSM
    # ==================================================================
    print("\n" + "-" * 50)
    print("Analysis 3: Neural ROIs → Learned Embedding RSM")
    print("-" * 50)

    # Build embedding RSM (correlation-based, test videos) — reuse emb_test from Analysis 2
    emb_rsm = 1 - pairwise_distances(emb_test, metric='correlation')
    y_emb = emb_rsm[triu_idx]
    print(f"Embedding RSM shape: {emb_rsm.shape}")

    # Predictors: averaged neural ROI RSMs
    neural_df = pd.DataFrame({
        roi: avg_brain[roi][triu_idx] for roi in ROI_ORDER if roi in avg_brain
    })

    r2_full_ne, unique_vars_ne, shared_ne = compute_explained_variance(y_emb, neural_df)
    print(f"Total R²: {r2_full_ne:.4f}, Shared: {shared_ne:.4f}")
    for roi, uv in sorted(unique_vars_ne.items(), key=lambda x: -x[1]):
        print(f"  {roi}: {uv:.4f}")

    # Plot
    plot_neural_to_embedding_variance(
        unique_vars_ne, shared_ne, r2_full_ne,
        os.path.join(RESULTS_NEURAL_DIR, 'neural_to_embedding_variance.png'),
        noise_ceiling=noise_ceiling
    )

    # Save CSV
    ne_rows = [{'ROI': roi, 'Unique_R2': uv, 'Total_R2': r2_full_ne, 'Shared_R2': shared_ne}
               for roi, uv in unique_vars_ne.items()]
    csv_path = os.path.join(RESULTS_NEURAL_DIR, 'neural_to_embedding_variance.csv')
    pd.DataFrame(ne_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print("\n" + "=" * 60)
    print("NEURAL-BEHAVIORAL EXPLAINED VARIANCE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
