"""
Searchlight RSA: whole-brain searchlight comparing embedding-derived RSMs
to local neural RSMs, with inflated surface visualization.

Produces:
  1. Combined RSA map (all 11 BEST_DIMS) — main figure
  2. Per-dimension unique R² maps (each of 11 BEST_DIMS) — supplement
     Each dim map shows unique explained variance: R²_full - R²_leave_one_out_d
  3. Winner-takes-all map: voxels colored by dimension with highest unique R²

Uses test-set betas (50 videos) and reliability masks to restrict
searchlight centers to reliable voxels.
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
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from helpers.surfaces import load_surfaces
from config import (
    MODEL_DIR, RESULTS_SEARCHLIGHT_DIR, RESULTS_BEHAVIORAL_DIR,
    FMRI_BETAS_DIR, FMRI_RELI_DIR,
    SUBJECTS, BEST_DIMS, DIM_LABELS, test_idx,
    SEARCHLIGHT_RADIUS_VOXELS,
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


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
# MODEL RDMs
# ============================================================================

def build_model_rdvs(embeddings, test_idx, best_dims):
    """
    Build model representational distance vectors (lower triangle).

    Returns:
        combined_X: (n_pairs, n_best_dims) matrix — per-dim distance vectors
                    as columns, for multivariate regression RSA
        per_dim_rdvs: dict[str, ndarray] — 'dim_X' → single-dim distance vector
    """
    emb_test = embeddings[test_idx]

    # Per-dimension: Euclidean distance, min-max normalized
    per_dim_rdvs = {}
    dim_columns = []
    for d in best_dims:
        dv = pdist(emb_test[:, [d]], metric='euclidean')
        dmin, dmax = dv.min(), dv.max()
        if dmax > dmin:
            dv = (dv - dmin) / (dmax - dmin)
        per_dim_rdvs[f'dim_{d}'] = dv
        dim_columns.append(dv)

    # Combined: matrix of all per-dim distance vectors (for regression RSA)
    combined_X = np.column_stack(dim_columns)

    return combined_X, per_dim_rdvs


# ============================================================================
# SEARCHLIGHT CORE
# ============================================================================

def _cv_spearman_r2(X, y, n_folds=5, seed=0):
    """Out-of-fold Spearman r² using KFold CV.

    Fits OLS on each fold, collects out-of-fold predictions, then computes
    Spearman r between all out-of-fold predictions and actuals, squared.
    Matches the metric used in the ROI-level neural_explained_variance.py.
    """
    try:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=kf)
        r, _ = spearmanr(y, y_pred)
        return float(r ** 2) if not np.isnan(r) else 0.0
    except Exception:
        return 0.0


def cv_projectors(X, n_folds=5, seed=0, combined_only=False):
    """Out-of-fold OLS predictions as fixed linear maps of the target.

    For a fixed design X, cross_val_predict(LinearRegression(), X, y, cv=kf)
    is linear in y: in each test fold the prediction is
    (X_te - mean(X_tr)) @ pinv(X_tr - mean(X_tr)) @ y_tr + mean(y_tr), which is
    exactly what sklearn computes (centered least squares, minimum-norm
    solution, intercept from the means). Stacking the folds gives one
    (n_pairs, n_pairs) matrix P with y_pred = P @ y.

    Returns P of shape (1 + n_dims, n_pairs, n_pairs): index 0 is the full
    model, index 1 + i the model without column i. The searchlight's design
    never changes between spheres, only y does, so this replaces 1 + n_dims
    sklearn CV fits per sphere with one batched matvec.
    """
    n, k = X.shape
    subsets = [list(range(k))]
    if not combined_only and k > 1:
        subsets += [[j for j in range(k) if j != i] for i in range(k)]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(X))
    P = np.zeros((len(subsets), n, n))
    for si, cols in enumerate(subsets):
        Xs = X[:, cols]
        for tr, te in folds:
            xm = Xs[tr].mean(axis=0)
            H = (Xs[te] - xm) @ np.linalg.pinv(Xs[tr] - xm)
            P[si][np.ix_(te, tr)] = H + 1.0 / len(tr)
    return P


def _rsa_kernel_proj(neighbor_indices, betas_2d, P, dim_keys, combined_only,
                     signed=False):
    """Same outputs as _rsa_kernel, using precomputed cv_projectors.

    signed=True (combined_only only) returns the signed out-of-fold Spearman r
    for the full model instead of r^2; r**2 reproduces the default map.
    """
    from scipy.stats import rankdata
    all_names = ['combined'] if combined_only else ['combined'] + list(dim_keys)
    local_betas = betas_2d[:, neighbor_indices]
    if local_betas.shape[1] < 2:
        return {name: 0.0 for name in all_names}
    y = pdist(local_betas, metric='correlation')
    if np.all(np.isnan(y)) or np.std(y) == 0:
        return {name: 0.0 for name in all_names}
    y = np.nan_to_num(y, nan=0.0)

    preds = P @ y                                   # (n_models, n_pairs)
    ry = rankdata(y)
    ry = ry - ry.mean()
    rp = rankdata(preds, axis=1)
    rp = rp - rp.mean(axis=1, keepdims=True)
    denom = np.sqrt((rp ** 2).sum(axis=1) * (ry ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (rp @ ry) / denom
    r2 = np.where(np.isnan(r), 0.0, r ** 2)       # _cv_spearman_r2: NaN -> 0

    if signed:
        return {'combined': float(np.nan_to_num(r[0]))}
    results = {'combined': float(r2[0])}
    if combined_only:
        return results
    if len(dim_keys) == 1:
        results[dim_keys[0]] = float(r2[0])
        return results
    for i, name in enumerate(dim_keys):
        results[name] = max(float(r2[0] - r2[1 + i]), 0.0)
    return results


def _rank_unit(v):
    """Rank-transform, centre, and scale to unit norm.

    With both sides in this form a Spearman correlation is a plain dot product,
    so the marginal readout costs one matvec per searchlight instead of a
    regression fit per dimension.
    """
    from scipy.stats import rankdata
    r = rankdata(v).astype(np.float64)
    r -= r.mean()
    n = np.sqrt((r * r).sum())
    return r / n if n > 0 else r


def _marginal_kernel(center_idx, neighbor_indices, betas_2d, Xr, names):
    """Signed marginal RSA for a single searchlight center.

    Spearman r between each dimension's model RDV *on its own* and the local
    neural RDV -- i.e. "how well does this dimension explain this sphere",
    with no partialling out of the other dimensions. Contrast `_rsa_kernel`,
    which reports unique variance (what a dimension explains that no other one
    can); with collinear predictors the two answer different questions and can
    pick different winners.

    Kept signed: a dimension whose geometry *anti*-predicts the sphere should
    not be able to win an argmax, which squaring would allow.
    """
    local_betas = betas_2d[:, neighbor_indices]
    if local_betas.shape[1] < 2:
        return {n: 0.0 for n in names}

    neural_rdv = pdist(local_betas, metric='correlation')
    if np.all(np.isnan(neural_rdv)) or np.std(neural_rdv) == 0:
        return {n: 0.0 for n in names}

    yr = _rank_unit(np.nan_to_num(neural_rdv, nan=0.0))
    r = Xr.T @ yr
    return {n: float(r[i]) for i, n in enumerate(names)}


def _rsa_kernel(center_idx, neighbor_indices, betas_2d,
                combined_X, per_dim_rdvs, combined_only=False, n_folds=5):
    """
    Regression RSA for a single searchlight center.

    For 'combined': fits OLS from all per-dim model RDVs → neural RDV via
    5-fold CV, then computes Spearman r² on out-of-fold predictions.
    For per-dim: leave-one-out unique variance using the same CV metric.

    Args:
        center_idx: index into the mask-voxel array
        neighbor_indices: indices into betas_2d columns for this sphere
        betas_2d: (n_stimuli, n_mask_voxels) array
        combined_X: (n_pairs, n_dims) matrix of per-dim model RDVs
        per_dim_rdvs: per-dim model names, in combined_X column order (a
                      dict of RDVs also works; only its keys are used)

    Returns:
        dict[str, float] — model_name → Spearman r²
    """
    all_names = ['combined'] if combined_only else ['combined'] + list(per_dim_rdvs)
    local_betas = betas_2d[:, neighbor_indices]  # (n_stimuli, n_neighbors)

    if local_betas.shape[1] < 2:
        return {name: 0.0 for name in all_names}

    # Neural RDV: Pearson correlation distance between stimuli
    neural_rdv = pdist(local_betas, metric='correlation')

    # Handle degenerate cases (constant voxels)
    if np.all(np.isnan(neural_rdv)) or np.std(neural_rdv) == 0:
        return {name: 0.0 for name in all_names}

    neural_rdv = np.nan_to_num(neural_rdv, nan=0.0)

    results = {}

    # Combined: CV Spearman r² (OLS predict neural from all dims, out-of-fold)
    r2_full = _cv_spearman_r2(combined_X, neural_rdv, n_folds=n_folds)
    results['combined'] = r2_full

    if combined_only:
        return results

    # Per-dimension: unique explained variance = R²_full - R²_leave_one_out_d
    dim_keys = list(per_dim_rdvs)
    n_dims_local = len(dim_keys)
    for i, name in enumerate(dim_keys):
        try:
            if n_dims_local == 1:
                results[name] = r2_full
            else:
                keep = [j for j in range(n_dims_local) if j != i]
                r2_reduced = _cv_spearman_r2(combined_X[:, keep], neural_rdv,
                                             n_folds=n_folds)
                results[name] = max(r2_full - r2_reduced, 0.0)
        except Exception:
            results[name] = 0.0

    return results


def _voxel_size_mm(affine):
    """Extract voxel size in mm from a NIfTI affine matrix."""
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def run_searchlight_rsa(betas_img, mask_img, combined_X, per_dim_rdvs,
                        radius_voxels, n_jobs=-1, desc="Searchlight",
                        combined_only=False, readout='unique', n_folds=5,
                        signed=False):
    """
    Run regression-based searchlight RSA across all reliable voxels.

    Args:
        betas_img: nibabel image, 4D (x, y, z, n_stimuli)
        mask_img: nibabel image, 3D boolean mask
        combined_X: (n_pairs, n_dims) matrix of per-dim model RDVs
        per_dim_rdvs: dict[str, ndarray] — per-dim RDV vectors
        radius_voxels: searchlight radius in voxels
        n_jobs: number of parallel jobs (-1 = all cores)
        desc: tqdm description

    Returns:
        results_3d: dict[str, ndarray] — model_name → 3D volume of Spearman r²
    """
    betas_4d = np.asarray(betas_img.dataobj, dtype=np.float32)
    mask_3d = np.asarray(mask_img.dataobj).astype(bool)
    affine = betas_img.affine

    # Convert voxel radius to mm using voxel size
    voxel_dims = _voxel_size_mm(affine)
    radius_mm = radius_voxels * np.mean(voxel_dims)
    print(f"  Voxel size: {voxel_dims} mm → radius: {radius_voxels} voxels = {radius_mm:.2f} mm")

    # Get voxel coordinates of all mask voxels
    mask_ijk = np.array(np.where(mask_3d)).T  # (n_voxels, 3)
    n_centers = mask_ijk.shape[0]

    # Convert voxel coords to mm (world coordinates)
    mask_mm = nib.affines.apply_affine(affine, mask_ijk)  # (n_voxels, 3)

    # Extract betas for all mask voxels: (n_stimuli, n_mask_voxels)
    betas_2d = betas_4d[mask_3d].T  # mask_3d indexes first 3 dims → (n_voxels,n_stim) → transpose
    del betas_4d  # free the full 4D array (~hundreds of MB)

    # Build KDTree for neighbor lookups
    tree = KDTree(mask_mm)

    # Pre-compute neighbor lists for all centers
    print(f"  Building neighbor lists for {n_centers} centers (radius={radius_mm:.2f}mm)...")
    all_neighbors = tree.query_ball_tree(tree, r=radius_mm)

    # Run searchlight in parallel
    if readout == 'marginal':
        # 'combined' is the full-model R^2 and does not depend on the per-dim
        # readout, so it is not recomputed here -- reuse sl_combined_sub{N}.
        all_names = list(per_dim_rdvs.keys())
        Xr = np.column_stack([_rank_unit(v) for v in per_dim_rdvs.values()])
        job = lambda i: _marginal_kernel(i, all_neighbors[i], betas_2d,
                                         Xr, all_names)
    else:
        all_names = (['combined'] if combined_only
                     else ['combined'] + list(per_dim_rdvs.keys()))
    print(f"  Running searchlight ({n_centers} centers, {len(all_names)} models,"
          f" readout={readout}, n_jobs={n_jobs}, n_folds={n_folds})...")
    verbose_level = 10 if n_centers > 1000 else 5
    if readout == 'marginal':
        # one matvec per sphere: cheap enough that threads are fine
        results_list = Parallel(n_jobs=n_jobs, prefer='threads', verbose=verbose_level)(
            delayed(job)(i) for i in range(n_centers)
        )
    else:
        # The regression kernel is mostly GIL-bound sklearn overhead, so it
        # needs processes, not threads. Pass the dim names rather than the
        # RDV dict, and memmap arrays over 100 KB, so each task ships only
        # its neighbor list.
        # Precomputing the CV projectors makes each sphere one batched
        # matvec instead of (1 + n_dims) x n_folds sklearn fits (~50x faster;
        # identical to _rsa_kernel up to float roundoff).
        dim_keys = list(per_dim_rdvs.keys())
        P = cv_projectors(combined_X, n_folds=n_folds, combined_only=combined_only)
        print(f"  CV projectors: {P.shape} ({P.nbytes / 1e6:.0f} MB)")
        results_list = Parallel(n_jobs=n_jobs, backend='loky', max_nbytes='1M',
                                batch_size=256, verbose=verbose_level)(
            delayed(_rsa_kernel_proj)(all_neighbors[i], betas_2d, P,
                                      dim_keys, combined_only, signed)
            for i in range(n_centers)
        )

    # Assemble results into 3D volumes
    results_3d = {}
    for name in all_names:
        vol = np.zeros(mask_3d.shape, dtype=np.float64)
        for i, res in enumerate(results_list):
            v = res[name]
            # unique R^2 is non-negative by construction; marginal r is signed
            # and must stay so, or anti-predicting dims could win an argmax.
            vol[mask_ijk[i, 0], mask_ijk[i, 1], mask_ijk[i, 2]] = (
                v if (readout == 'marginal' or signed) else max(v, 0.0))
        results_3d[name] = vol

    return results_3d


# ============================================================================
# VISUALIZATION: SURFACE PROJECTION
# ============================================================================

def plot_combined_surface(subjects, sl_dir, threshold=0.01, vmax=None, tag=''):
    """
    Plot combined searchlight regression RSA (Spearman r²) on inflated surfaces.
    4 subjects × 6 views. One shared colorbar on the right.
    """
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
        nii_path = os.path.join(sl_dir, f'sl_combined{tag}_sub{sub}.nii.gz')
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
    fig.suptitle('Searchlight Regression RSA: Combined Embedding Dimensions', fontsize=16)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    norm = Normalize(vmin=0, vmax=used_vmax)
    sm = cm.ScalarMappable(cmap='inferno', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Spearman $r^2$')
    ticks = np.linspace(0, used_vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:.2f}' for t in ticks])

    save_path = os.path.join(sl_dir, f'sl_combined_surface{tag}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved combined surface plot to {save_path}")


def plot_per_dim_surfaces(subjects, best_dims, sl_dir, threshold=0.01,
                          vmax=None, tag=''):
    """
    Plot per-dimension searchlight RSA (Spearman r²) on inflated surfaces.
    One figure per dimension: 4 subjects × 4 views.
    """
    from nilearn import surface, plotting
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    for d in best_dims:
        n_rows = len(subjects)
        fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4 * n_rows),
                                 subplot_kw={'projection': '3d'})
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        used_vmax = vmax

        for row, sub in enumerate(subjects):
            nii_path = os.path.join(sl_dir, f'sl_dim_{d}{tag}_sub{sub}.nii.gz')
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

            for col, (hemi, view) in enumerate(views):
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
                    axes=axes[row, col],
                    figure=fig,
                    title=f'Sub-{sub}' if col == 0 else '',
                )

        fig.subplots_adjust(left=0.01, right=0.88, bottom=0.02, top=0.92,
                            wspace=0.02, hspace=0.05)
        fig.suptitle(f'Searchlight RSA: Dimension {d} (Unique $r^2$)', fontsize=14)

        # Shared colorbar
        final_vmax = used_vmax if used_vmax is not None else 0.05
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
        norm = Normalize(vmin=0, vmax=final_vmax)
        sm = cm.ScalarMappable(cmap='inferno', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, label='Unique Variance (Spearman $r^2$)')
        ticks = np.linspace(0, final_vmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{t:.2f}' for t in ticks])

        save_path = os.path.join(sl_dir, f'sl_dim{d}{tag}_surface.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved dim {d} surface plot to {save_path}")


def plot_winner_take_all(subjects, best_dims, sl_dir, threshold=0.01, tag='',
                         title=True, rasterize=False):
    """
    Winner-take-all map: each vertex colored by the dimension with
    the highest Spearman r² at that location. Only vertices where
    the winning dimension exceeds threshold are colored.
    """
    from nilearn import surface, plotting
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    n_dims = len(best_dims)
    # Hand-picked palette for up to 12 dims; fall back to tab20+extension for more
    base_colors = [
        '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
        '#dcbeff', '#ffe119',
    ]
    if n_dims <= len(base_colors):
        colors = base_colors[:n_dims]
    else:
        import matplotlib.cm as _cm
        cmap_src = _cm.get_cmap('tab20', n_dims)
        colors = [matplotlib.colors.to_hex(cmap_src(i)) for i in range(n_dims)]
    dim_cmap = ListedColormap(colors)
    # Boundaries: -0.5, 0.5, 1.5, ..., n_dims-0.5
    bounds = np.arange(-0.5, n_dims, 1.0)
    norm = BoundaryNorm(bounds, n_dims)

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

        # Load all per-dim NIfTIs for this subject
        dim_imgs = []
        dims_found = []
        for d in best_dims:
            p = os.path.join(sl_dir, f'sl_dim_{d}{tag}_sub{sub}.nii.gz')
            if not os.path.exists(p):
                continue
            dim_imgs.append(nib.load(p))
            dims_found.append(d)

        if len(dim_imgs) != n_dims:
            print(f"  Skipping sub-{sub}: only {len(dim_imgs)}/{n_dims} dim maps found")
            continue

        for col, (hemi, view) in enumerate(views):
            # Project each dim to surface, stack
            textures = []
            for img in dim_imgs:
                tex = surface.vol_to_surf(
                    img, surfs[f'{hemi}_pial'],
                    interpolation='nearest_most_frequent')
                textures.append(tex)
            stacked = np.stack(textures, axis=0)  # (n_dims, n_vertices)

            # Winner-take-all: argmax across dims
            winner = np.argmax(stacked, axis=0).astype(float)
            max_val = np.max(stacked, axis=0)

            # Mask vertices below threshold
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
                vmax=n_dims + 0.5,
                avg_method='median',
                colorbar=False,
                cmap=dim_cmap,
                axes=axes[row, col],
                figure=fig,
                title=f'Sub-{sub}' if col == 0 else '',
            )
            if rasterize:
                # Mesh as a 300 dpi image in the PDF; text and legend stay vector
                axes[row, col].set_rasterized(True)

    fig.subplots_adjust(left=0.01, right=0.78, bottom=0.02, top=0.92,
                        wspace=0.02, hspace=0.05)
    if title:
        fig.suptitle('Searchlight RSA: Dimension with Highest Unique Variance', fontsize=16)

    # Legend with descriptive labels (source of truth: config.DIM_LABELS)
    patches = [mpatches.Patch(color=colors[i],
                              label=f'Dim {best_dims[i]}: {DIM_LABELS.get(best_dims[i], "")}')
               for i in range(n_dims)]
    fig.legend(handles=patches, loc='center right',
               bbox_to_anchor=(0.99, 0.5), fontsize=15, frameon=True,
               title='Dimension', title_fontsize=17,
               markerscale=2.5, handlelength=2.5, handleheight=2.0,
               borderpad=1.2, labelspacing=1.0)

    save_path = os.path.join(sl_dir, f'sl_winner_take_all{tag}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved winner-take-all plot to {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', choices=['best12', 'all'], default='best12',
                        help="best12: use config.BEST_DIMS (default). "
                             "all: use every pruned embedding dim.")
    parser.add_argument('--greedy-k', type=int, default=None,
                        help="Use the first K dims of the train-selected greedy "
                             "order (greedy_forward_r2_train.csv) instead of "
                             "--dims. Outputs carry a _dim{K} tag. K=12 is "
                             "BEST_DIMS + Dim 33, the count-matched model in "
                             "ED Fig. 7.")
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                        help="Subset of subjects to run (default: all).")
    parser.add_argument('--combined-only', action='store_true',
                        help="Only compute+save the combined map (skip per-dim "
                             "unique R² and winner-take-all).")
    parser.add_argument('--readout', choices=['unique', 'marginal'],
                        default='unique',
                        help="unique: per-dim unique R² = R²_full - R²_leave_one_out "
                             "(default, the published maps). marginal: signed "
                             "Spearman r between each dim's RDV alone and the "
                             "local neural RDV — 'which dim best explains this "
                             "sphere' rather than 'what only this dim explains'. "
                             "Marginal outputs carry a _marginal suffix.")
    parser.add_argument('--n-jobs', type=int, default=4,
                        help="Parallel workers (default 4; BLAS is pinned to 1 "
                             "thread each, so this is the whole CPU budget).")
    parser.add_argument('--n-folds', type=int, default=5,
                        help="KFold splits over the 1225 stimulus pairs for the "
                             "unique readout (default 5). Non-default values "
                             "add a _{k}fold suffix to every output.")
    parser.add_argument('--no-plot', action='store_true',
                        help="Write NIfTIs only; skip all surface figures.")
    parser.add_argument('--plot', action='store_true',
                        help="Also render whole-brain surface figures. Off by "
                             "default under --readout marginal, where the "
                             "downstream ROI figures are usually what is wanted.")
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else SUBJECTS

    print("\n" + "=" * 60)
    print(f"SEARCHLIGHT RSA (dims={args.dims})")
    print("=" * 60)

    # Load embeddings
    emb_path = os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy')
    embeddings = np.load(emb_path)
    print(f"Loaded embeddings: {embeddings.shape}")

    if test_idx is None:
        print("ERROR: test_idx not available. Need data/utils/test.csv")
        return

    if args.greedy_k is not None:
        greedy_csv = os.path.join(RESULTS_BEHAVIORAL_DIR, 'greedy_forward_r2_train.csv')
        order = pd.read_csv(greedy_csv)['dim_id'].tolist()
        assert order[:len(BEST_DIMS)] == list(BEST_DIMS), \
            "cached greedy order no longer starts with BEST_DIMS"
        dims = order[:args.greedy_k]
        tag = f'_dim{args.greedy_k}'
    elif args.dims == 'best12':
        dims = list(BEST_DIMS)
        tag = ''
    else:
        dims = list(range(embeddings.shape[1]))
        tag = '_all'
    # readout suffix on per-dim outputs only; the combined full-model map is
    # readout-independent and keeps its original name.
    rtag = '_marginal' if args.readout == 'marginal' else ''
    if args.readout == 'unique' and args.n_folds != 5:
        tag += f'_{args.n_folds}fold'
    base_tag = tag
    tag = tag + rtag
    print(f"Using {len(dims)} dims; readout={args.readout}; output tag='{tag}'")

    # Build model RDVs
    print("\nBuilding model RDVs...")
    combined_X, per_dim_rdvs = build_model_rdvs(embeddings, test_idx, dims)
    print(f"  Combined predictor matrix: {combined_X.shape}")
    print(f"  Per-dim models: {list(per_dim_rdvs.keys())}")

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

        # Skip if all expected outputs already exist for this run mode
        combined_path = os.path.join(RESULTS_SEARCHLIGHT_DIR,
                                     f'sl_combined{base_tag}_sub{sub}.nii.gz')
        per_dim_paths = [os.path.join(RESULTS_SEARCHLIGHT_DIR,
                                      f'sl_dim_{d}{tag}_sub{sub}.nii.gz')
                         for d in dims]
        if args.readout == 'marginal':
            expected_paths = per_dim_paths
        elif args.combined_only:
            expected_paths = [combined_path]
        else:
            expected_paths = [combined_path] + per_dim_paths
        if all(os.path.exists(p) for p in expected_paths):
            print(f"  All {len(expected_paths)} expected NIfTIs exist for sub-{sub}, skipping.")
            continue

        # Run searchlight
        results_3d = run_searchlight_rsa(
            betas_img, mask_img, combined_X, per_dim_rdvs,
            radius_voxels=SEARCHLIGHT_RADIUS_VOXELS,
            n_jobs=args.n_jobs,
            desc=f"Sub-{sub}",
            combined_only=args.combined_only,
            readout=args.readout,
            n_folds=args.n_folds,
        )

        # Save NIfTI results
        for name, vol in results_3d.items():
            out_path = os.path.join(RESULTS_SEARCHLIGHT_DIR,
                                    f'sl_{name}{tag}_sub{sub}.nii.gz')
            out_img = nib.Nifti1Image(vol, betas_img.affine, betas_img.header)
            nib.save(out_img, out_path)
            print(f"  Saved {out_path}")

        # Free memory before next subject
        del results_3d, betas_img, mask_img
        gc.collect()

    # Visualization
    if args.no_plot:
        print("\nNIfTIs written; --no-plot, skipping figures.")
        return
    if args.readout == 'marginal' and not args.plot:
        print("\nMarginal NIfTIs written. Whole-brain figures skipped "
              "(pass --plot to render them).")
        print("\n" + "=" * 60)
        print("SEARCHLIGHT RSA COMPLETE")
        print("=" * 60)
        return

    print("\n--- Surface Visualization ---")

    # Determine a shared vmax from all combined results
    all_vals = []
    for sub in subjects:
        p = os.path.join(RESULTS_SEARCHLIGHT_DIR,
                         f'sl_combined{base_tag}_sub{sub}.nii.gz')
        if os.path.exists(p):
            data = np.asarray(nib.load(p).dataobj)
            all_vals.append(data[data > 0.01])
    if all_vals:
        shared_vmax = np.percentile(np.concatenate(all_vals), 99)
    else:
        shared_vmax = 0.15

    # Data is already Spearman r² from the regression RSA kernel
    plot_combined_surface(subjects, RESULTS_SEARCHLIGHT_DIR,
                          threshold=0.01, vmax=shared_vmax, tag=base_tag)
    if not args.combined_only:
        # 0.01 in r^2 units is 0.1 in r units; keep the two comparable
        dim_thr = 0.1 if args.readout == 'marginal' else 0.01
        plot_per_dim_surfaces(subjects, dims, RESULTS_SEARCHLIGHT_DIR,
                              threshold=dim_thr, vmax=shared_vmax, tag=tag)
        plot_winner_take_all(subjects, dims, RESULTS_SEARCHLIGHT_DIR,
                             threshold=dim_thr, tag=tag)

    print("\n" + "=" * 60)
    print("SEARCHLIGHT RSA COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
