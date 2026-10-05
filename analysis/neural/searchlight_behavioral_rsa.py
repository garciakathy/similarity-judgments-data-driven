"""
Searchlight Behavioral RSA: whole-brain searchlight comparing behavioral
feature RDMs to local neural RDMs, with inflated surface visualization.

Produces:
  1. Combined behavioral RSA map (all BEHAVIOR_COLS) — main figure
  2. Per-feature unique R² maps (each of BEHAVIOR_COLS) — supplement
     Each map shows unique explained variance: R²_full - R²_leave_one_out_feat
  3. Winner-takes-all map: voxels colored by feature with highest unique variance
"""

import os
import sys
import gc
import argparse

# Pin numpy/BLAS to 1 thread per worker so joblib n_jobs controls parallelism
# cleanly without thread oversubscription (set before importing numpy).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from helpers.surfaces import load_surfaces
from config import (
    MODEL_DIR, RESULTS_SEARCHLIGHT_DIR, UTILS_DIR,
    FMRI_BETAS_DIR, FMRI_RELI_DIR,
    SUBJECTS, BEHAVIOR_COLS, FEATURE_REMAPS, test_idx,
    SEARCHLIGHT_RADIUS_VOXELS,
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Pseudo-column for the action-category predictor (not in the annotations CSV;
# built from train/test_categories.csv via helpers.action_categories).
ACTION_COL = 'action category'

# CV folds over stimulus pairs; set from --n-folds in main().
N_FOLDS = 5

# Sanitize column names for use in filenames
def _col_to_key(col):
    return col.replace(' ', '_')

# Display name for a behavioral feature
def _display_name(col):
    if col == ACTION_COL:
        return 'Action Categories'
    key = col.title()
    return FEATURE_REMAPS.get(key, key)


# ============================================================================
# PATHS
# ============================================================================

def _betas_path(sub):
    return os.path.join(FMRI_BETAS_DIR,
                        f'sub-0{sub}_space-T1w_desc-test-fracridge_data.nii.gz')

def _reli_path(sub):
    return os.path.join(FMRI_RELI_DIR,
                        f'sub-0{sub}_space-T1w_desc-test-fracridge_reliability-mask.nii.gz')

# ============================================================================
# BEHAVIORAL MODEL RDVs
# ============================================================================

def build_behavioral_rdvs(test_idx, behavior_cols):
    """
    Build behavioral representational distance vectors (lower triangle).

    Loads annotations, subsets to test videos, computes min-max normalized
    Euclidean distance for each behavioral feature.

    Returns:
        combined_X: (n_pairs, n_features) — per-feature RDVs as columns
        per_feat_rdvs: dict[str, ndarray] — col_key → single-feature RDV
    """
    ann_path = os.path.join(UTILS_DIR, 'annotations.csv')
    annotations = pd.read_csv(ann_path)
    annotations = annotations.sort_values('vid_id').reset_index(drop=True)

    per_feat_rdvs = {}
    dim_columns = []
    for col in behavior_cols:
        if col == ACTION_COL:
            # Binary distance: 0 same action category, 1 different. Already
            # in [0, 1], so min-max normalisation is a no-op.
            from helpers.action_categories import load_action_categories
            labels = load_action_categories(UTILS_DIR)[test_idx]
            dv = pdist(np.asarray(labels).reshape(-1, 1),
                       metric=lambda a, b: float(a[0] != b[0]))
            per_feat_rdvs[_col_to_key(col)] = dv
            dim_columns.append(dv)
            continue
        vals = annotations.loc[test_idx, col].values.astype(float)
        dv = pdist(vals.reshape(-1, 1), metric='euclidean')
        dmin, dmax = dv.min(), dv.max()
        if dmax > dmin:
            dv = (dv - dmin) / (dmax - dmin)
        key = _col_to_key(col)
        per_feat_rdvs[key] = dv
        dim_columns.append(dv)

    combined_X = np.column_stack(dim_columns)
    return combined_X, per_feat_rdvs


# ============================================================================
# SEARCHLIGHT CORE
# ============================================================================

def _cv_spearman_r2(X, y, n_folds=None, seed=0):
    """Out-of-fold Spearman r² using KFold CV.

    Fits OLS on each fold, collects out-of-fold predictions, then computes
    Spearman r between all out-of-fold predictions and actuals, squared.
    Matches the metric used in the ROI-level neural_explained_variance.py.
    """
    if n_folds is None:
        n_folds = N_FOLDS
    try:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=kf)
        r, _ = spearmanr(y, y_pred)
        return float(r ** 2) if not np.isnan(r) else 0.0
    except Exception:
        return 0.0


def _rsa_kernel(center_idx, neighbor_indices, betas_2d,
                combined_X, feat_keys, n_folds=5):
    """
    Regression RSA for a single searchlight sphere.

    Uses N_FOLDS-fold CV (over pairs) Spearman r² (out-of-fold predictions) for the full model
    and leave-one-out unique variance per feature — matching the metric used
    in neural_explained_variance.py for methodological consistency.
    """
    all_names = ['beh_combined'] + [f'beh_{k}' for k in feat_keys]
    local_betas = betas_2d[:, neighbor_indices]

    if local_betas.shape[1] < 2:
        return {name: 0.0 for name in all_names}

    neural_rdv = pdist(local_betas, metric='correlation')
    if np.all(np.isnan(neural_rdv)) or np.std(neural_rdv) == 0:
        return {name: 0.0 for name in all_names}
    neural_rdv = np.nan_to_num(neural_rdv, nan=0.0)

    results = {}

    # Full model: CV Spearman r² (OLS predict neural from all features, out-of-fold)
    r2_full = _cv_spearman_r2(combined_X, neural_rdv, n_folds=n_folds)
    results['beh_combined'] = r2_full

    # Per-feature unique variance: leave-one-out with same CV metric
    n_feats = len(feat_keys)
    for i, key in enumerate(feat_keys):
        try:
            if n_feats == 1:
                results[f'beh_{key}'] = r2_full
            else:
                keep = [j for j in range(n_feats) if j != i]
                r2_reduced = _cv_spearman_r2(combined_X[:, keep], neural_rdv,
                                             n_folds=n_folds)
                results[f'beh_{key}'] = max(r2_full - r2_reduced, 0.0)
        except Exception:
            results[f'beh_{key}'] = 0.0

    return results


def _voxel_size_mm(affine):
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def run_searchlight_behavioral(betas_img, mask_img, combined_X, per_feat_rdvs,
                               radius_voxels, n_jobs=8, desc="Searchlight",
                               n_folds=5):
    betas_4d = np.asarray(betas_img.dataobj, dtype=np.float32)
    mask_3d = np.asarray(mask_img.dataobj).astype(bool)
    affine = betas_img.affine

    voxel_dims = _voxel_size_mm(affine)
    radius_mm = radius_voxels * np.mean(voxel_dims)
    print(f"  Voxel size: {voxel_dims} mm → radius: {radius_voxels} voxels = {radius_mm:.2f} mm")

    mask_ijk = np.array(np.where(mask_3d)).T
    n_centers = mask_ijk.shape[0]
    mask_mm = nib.affines.apply_affine(affine, mask_ijk)

    betas_2d = betas_4d[mask_3d].T
    del betas_4d

    feat_keys = list(per_feat_rdvs.keys())

    tree = KDTree(mask_mm)
    print(f"  Building neighbor lists for {n_centers} centers (radius={radius_mm:.2f}mm)...")
    all_neighbors = tree.query_ball_tree(tree, r=radius_mm)

    all_names = ['beh_combined'] + [f'beh_{k}' for k in feat_keys]
    print(f"  Running searchlight ({n_centers} centers, {len(all_names)} models, {n_jobs} jobs)...")
    verbose_level = 10 if n_centers > 1000 else 5
    results_list = Parallel(n_jobs=n_jobs, backend='loky', max_nbytes='100K',
                            verbose=verbose_level)(
        delayed(_rsa_kernel)(i, all_neighbors[i], betas_2d,
                             combined_X, feat_keys, n_folds)
        for i in range(n_centers)
    )

    results_3d = {}
    for name in all_names:
        vol = np.zeros(mask_3d.shape, dtype=np.float64)
        for i, res in enumerate(results_list):
            vol[mask_ijk[i, 0], mask_ijk[i, 1], mask_ijk[i, 2]] = max(res[name], 0.0)
        results_3d[name] = vol

    return results_3d


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_combined_behavioral_surface(subjects, sl_dir, threshold=0.01, vmax=None, tag=''):
    """Combined behavioral regression RSA map across all features."""
    from nilearn import surface, plotting
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    n_rows = len(subjects)
    fig, axes = plt.subplots(n_rows, 6, figsize=(26, 4 * n_rows),
                             subplot_kw={'projection': '3d'})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    used_vmax = vmax

    for row, sub in enumerate(subjects):
        nii_path = os.path.join(sl_dir, f'sl_beh_combined{tag}_sub{sub}.nii.gz')
        if not os.path.exists(nii_path):
            print(f"  Skipping sub-{sub}: {nii_path} not found")
            continue

        sl_img = nib.load(nii_path)
        surfs = load_surfaces(sub)
        if surfs is None:
            continue

        sl_data = np.asarray(sl_img.dataobj)
        if used_vmax is None:
            used_vmax = np.percentile(sl_data[sl_data > threshold], 99) if np.any(sl_data > threshold) else 0.15

        views = [
            ('lh', 'lateral'), ('rh', 'lateral'),
            ('lh', 'medial'), ('rh', 'medial'),
            ('lh', 'ventral'), ('rh', 'ventral'),
        ]

        for col, (hemi, view) in enumerate(views):
            texture = surface.vol_to_surf(
                sl_img, surfs[f'{hemi}_pial'],
                interpolation='nearest_most_frequent')
            plotting.plot_surf_stat_map(
                surfs[f'{hemi}_inflated'], texture,
                hemi='left' if hemi == 'lh' else 'right',
                view=view,
                bg_map=surfs[f'{hemi}_sulc'],
                threshold=threshold,
                vmax=used_vmax,
                colorbar=False,
                cmap='inferno',
                axes=axes[row, col],
                figure=fig,
                title=f'Sub-{sub}' if col == 0 else '',
            )

    fig.subplots_adjust(left=0.01, right=0.90, bottom=0.02, top=0.92,
                        wspace=0.02, hspace=0.05)
    fig.suptitle('Searchlight Regression RSA: Combined Behavioral Features', fontsize=16)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    norm = Normalize(vmin=0, vmax=used_vmax)
    sm = cm.ScalarMappable(cmap='inferno', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Spearman $r^2$')
    ticks = np.linspace(0, used_vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:.2f}' for t in ticks])

    save_path = os.path.join(sl_dir, f'sl_beh_combined_surface{tag}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved combined behavioral surface plot to {save_path}")


def plot_per_feature_surfaces(subjects, behavior_cols, sl_dir, threshold=0.01, vmax=None, tag=''):
    """Per-feature unique R² maps on inflated surfaces."""
    from nilearn import surface, plotting
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    for col in behavior_cols:
        key = _col_to_key(col)
        display = _display_name(col)
        n_rows = len(subjects)
        fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4 * n_rows),
                                 subplot_kw={'projection': '3d'})
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        used_vmax = vmax

        for row, sub in enumerate(subjects):
            nii_path = os.path.join(sl_dir, f'sl_beh_{key}{tag}_sub{sub}.nii.gz')
            if not os.path.exists(nii_path):
                continue

            sl_img = nib.load(nii_path)
            surfs = load_surfaces(sub)
            if surfs is None:
                continue

            sl_data = np.asarray(sl_img.dataobj)
            if used_vmax is None:
                used_vmax = max(np.percentile(sl_data[sl_data > threshold], 99), 0.05) if np.any(sl_data > threshold) else 0.05

            views = [
                ('lh', 'lateral'), ('rh', 'lateral'),
                ('lh', 'ventral'), ('rh', 'ventral'),
            ]

            for c, (hemi, view) in enumerate(views):
                texture = surface.vol_to_surf(sl_img, surfs[f'{hemi}_pial'],
                                              interpolation='nearest_most_frequent')
                plotting.plot_surf_stat_map(
                    surfs[f'{hemi}_inflated'], texture,
                    hemi='left' if hemi == 'lh' else 'right',
                    view=view,
                    bg_map=surfs[f'{hemi}_sulc'],
                    threshold=threshold,
                    vmax=used_vmax,
                    colorbar=False,
                    cmap='inferno',
                    axes=axes[row, c],
                    figure=fig,
                    title=f'Sub-{sub}' if c == 0 else '',
                )

        fig.subplots_adjust(left=0.01, right=0.88, bottom=0.02, top=0.92,
                            wspace=0.02, hspace=0.05)
        fig.suptitle(f'Searchlight RSA: {display} (Unique $r^2$)', fontsize=14)

        final_vmax = used_vmax if used_vmax is not None else 0.05
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
        norm = Normalize(vmin=0, vmax=final_vmax)
        sm = cm.ScalarMappable(cmap='inferno', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, label='Unique Variance (Spearman $r^2$)')
        ticks = np.linspace(0, final_vmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{t:.2f}' for t in ticks])

        save_path = os.path.join(sl_dir, f'sl_beh_{key}_surface{tag}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {display} surface plot to {save_path}")


def plot_winner_take_all_behavioral(subjects, behavior_cols, sl_dir, threshold=0.01, tag=''):
    """
    Winner-takes-all: each vertex colored by the behavioral feature with
    the highest unique R² at that location.
    """
    from nilearn import surface, plotting
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    n_feats = len(behavior_cols)
    colors = [
        '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
        '#dcbeff', '#ffe119', '#9a6324',
    ][:n_feats]
    feat_cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, n_feats, 1.0)
    norm = BoundaryNorm(bounds, n_feats)

    n_rows = len(subjects)
    fig, axes = plt.subplots(n_rows, 6, figsize=(26, 4 * n_rows),
                             subplot_kw={'projection': '3d'})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    views = [
        ('lh', 'lateral'), ('rh', 'lateral'),
        ('lh', 'medial'), ('rh', 'medial'),
        ('lh', 'ventral'), ('rh', 'ventral'),
    ]

    for row, sub in enumerate(subjects):
        surfs = load_surfaces(sub)
        if surfs is None:
            continue

        feat_imgs = []
        feats_found = []
        for col in behavior_cols:
            key = _col_to_key(col)
            p = os.path.join(sl_dir, f'sl_beh_{key}{tag}_sub{sub}.nii.gz')
            if not os.path.exists(p):
                continue
            feat_imgs.append(nib.load(p))
            feats_found.append(col)

        if len(feat_imgs) != n_feats:
            print(f"  Skipping sub-{sub}: only {len(feat_imgs)}/{n_feats} feature maps found")
            continue

        for col_idx, (hemi, view) in enumerate(views):
            textures = []
            for img in feat_imgs:
                tex = surface.vol_to_surf(
                    img, surfs[f'{hemi}_pial'],
                    interpolation='nearest_most_frequent')
                textures.append(tex)
            stacked = np.stack(textures, axis=0)  # (n_feats, n_vertices)

            winner = np.argmax(stacked, axis=0).astype(float)
            max_val = np.max(stacked, axis=0)
            # vmin/vmax are passed explicitly because nilearn otherwise infers
            # vmin from the data, which shifts every colour off the legend in
            # any panel where the lowest-index category happens not to win.
            # avg_method='median' stops a face straddling a patch border from
            # averaging into a different category's colour bin.
            winner = winner + 1.0
            winner[max_val < threshold] = np.nan

            plotting.plot_surf_stat_map(
                surfs[f'{hemi}_inflated'], winner,
                hemi='left' if hemi == 'lh' else 'right',
                view=view,
                bg_map=surfs[f'{hemi}_sulc'],
                threshold=0.5,
                vmin=0.5,
                vmax=n_feats + 0.5,
                avg_method='median',
                colorbar=False,
                cmap=feat_cmap,
                axes=axes[row, col_idx],
                figure=fig,
                title=f'Sub-{sub}' if col_idx == 0 else '',
            )

    fig.subplots_adjust(left=0.01, right=0.82, bottom=0.02, top=0.92,
                        wspace=0.02, hspace=0.05)
    fig.suptitle('Searchlight RSA: Feature with Highest Unique Variance', fontsize=16)

    patches = [mpatches.Patch(color=colors[i], label=_display_name(behavior_cols[i]))
               for i in range(n_feats)]
    fig.legend(handles=patches, loc='center right',
               bbox_to_anchor=(0.99, 0.5), fontsize=11, frameon=True,
               title='Feature', title_fontsize=13,
               markerscale=2.0, handlelength=2.0, handleheight=1.5,
               borderpad=1.0, labelspacing=0.8)

    save_path = os.path.join(sl_dir, f'sl_beh_winner_take_all{tag}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved behavioral winner-take-all plot to {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global N_FOLDS
    ap = argparse.ArgumentParser()
    ap.add_argument('--include-action', action='store_true',
                    help="Add the action-category RDM as a 13th predictor.")
    ap.add_argument('--n-folds', type=int, default=5,
                    help="KFold splits over the 1225 stimulus pairs (default 5).")
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--subjects', type=int, nargs='+', default=None)
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    N_FOLDS = args.n_folds
    subjects = args.subjects or SUBJECTS
    behavior_cols = list(BEHAVIOR_COLS) + ([ACTION_COL] if args.include_action else [])
    # Defaults (12 features, 5-fold) keep the original filenames.
    tag = ('_action' if args.include_action else '') + \
          (f'_{N_FOLDS}fold' if N_FOLDS != 5 else '')

    print("\n" + "=" * 60)
    print("SEARCHLIGHT BEHAVIORAL RSA")
    print("=" * 60)

    if test_idx is None:
        print("ERROR: test_idx not available. Need data/utils/test.csv")
        return

    # Build behavioral RDVs
    print("\nBuilding behavioral RDVs...")
    combined_X, per_feat_rdvs = build_behavioral_rdvs(test_idx, behavior_cols)
    print(f"  Combined predictor matrix: {combined_X.shape}")
    print(f"  Features: {list(per_feat_rdvs.keys())}")
    print(f"  CV: {N_FOLDS}-fold over pairs; output tag='{tag}'")

    # Run searchlight for each subject
    for sub in subjects:
        print(f"\n--- Subject {sub} ---")

        betas_path = _betas_path(sub)
        reli_path = _reli_path(sub)

        if not os.path.exists(betas_path):
            print(f"  Skipping: {betas_path} not found")
            continue
        if not os.path.exists(reli_path):
            print(f"  Skipping: {reli_path} not found")
            continue

        betas_img = nib.load(betas_path)
        mask_img = nib.load(reli_path)
        print(f"  Betas shape: {betas_img.shape}")
        print(f"  Mask shape: {mask_img.shape}, reliable voxels: {np.asarray(mask_img.dataobj).astype(bool).sum()}")

        combined_path = os.path.join(RESULTS_SEARCHLIGHT_DIR, f'sl_beh_combined{tag}_sub{sub}.nii.gz')
        if os.path.exists(combined_path):
            print(f"  Results already exist at {combined_path}, skipping computation.")
            print(f"  Delete the file to re-run.")
            continue

        results_3d = run_searchlight_behavioral(
            betas_img, mask_img, combined_X, per_feat_rdvs,
            radius_voxels=SEARCHLIGHT_RADIUS_VOXELS,
            n_jobs=args.n_jobs,
            n_folds=N_FOLDS,
            desc=f"Sub-{sub}"
        )

        for name, vol in results_3d.items():
            out_path = os.path.join(RESULTS_SEARCHLIGHT_DIR, f'sl_{name}{tag}_sub{sub}.nii.gz')
            out_img = nib.Nifti1Image(vol, betas_img.affine, betas_img.header)
            nib.save(out_img, out_path)
            print(f"  Saved {out_path}")

        del results_3d, betas_img, mask_img
        gc.collect()

    if args.no_plot:
        return

    # Visualization
    print("\n--- Surface Visualization ---")

    all_vals = []
    for sub in subjects:
        p = os.path.join(RESULTS_SEARCHLIGHT_DIR, f'sl_beh_combined{tag}_sub{sub}.nii.gz')
        if os.path.exists(p):
            data = np.asarray(nib.load(p).dataobj)
            all_vals.append(data[data > 0.01])
    shared_vmax = np.percentile(np.concatenate(all_vals), 99) if all_vals else 0.15

    plot_combined_behavioral_surface(subjects, RESULTS_SEARCHLIGHT_DIR,
                                     threshold=0.01, vmax=shared_vmax, tag=tag)
    plot_per_feature_surfaces(subjects, behavior_cols, RESULTS_SEARCHLIGHT_DIR,
                              threshold=0.01, vmax=shared_vmax, tag=tag)
    plot_winner_take_all_behavioral(subjects, behavior_cols, RESULTS_SEARCHLIGHT_DIR,
                                   threshold=0.01, tag=tag)

    print("\n" + "=" * 60)
    print("SEARCHLIGHT BEHAVIORAL RSA COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
