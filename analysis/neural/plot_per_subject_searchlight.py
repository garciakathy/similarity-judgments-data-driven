"""
Generate per-subject searchlight plots for all 4 subjects:

1. sl_combined_vs_behavioral_sub{N}.png
   Panel A: Behavioral Features combined r² (inferno)
   Panel B: Learned Dimensions combined r² (inferno)
   Shared colorbar.

2. sl_wta_combined_vs_behavioral_sub{N}.png
   Panel A: Behavioral Features winner-take-all (categorical)
   Panel B: Learned Dimensions winner-take-all (categorical)
   Separate legends.
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from helpers.surfaces import load_surfaces
from config import (
    RESULTS_SEARCHLIGHT_DIR,
    BEST_DIMS, BEHAVIOR_COLS, FEATURE_REMAPS, SUBJECTS,
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def _col_to_key(col):
    return col.replace(' ', '_')


def _display_name(col):
    key = col.title()
    return FEATURE_REMAPS.get(key, key)


COLORS_12 = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#ffe119',
]

DIM_LABELS = {
    1: 'Conversation', 0: 'Caregiving', 17: 'Action sport',
    13: 'Outdoor activity', 2: 'Musical performance', 9: 'Infants crying',
    6: 'Cooking/Eating', 4: 'Infant play', 11: 'Parent-child activity',
    30: 'Parent-child interaction', 8: 'Infant bath/bed', 33: 'Direct interactions',
}


# ============================================================================
# COMBINED R² PLOT (inferno colorbar)
# ============================================================================

def plot_combined_vs_behavioral(sub, sl_dir, threshold=0.01, dim_tag='',
                                out_tag=None, panel_a_title='Behavioral Features',
                                panel_b_title='Learned Dimensions', beh_tag='',
                                font_delta=0, rasterize=False,
                                cbar_label='Spearman $r^2$', positive_only=False,
                                cbar_font_delta=0):
    """Panel A: beh combined r², Panel B: dim combined r². Shared inferno colorbar.

    dim_tag='' uses sl_combined_sub{N}.nii.gz (BEST_DIMS).
    dim_tag='_all' uses sl_combined_all_sub{N}.nii.gz (full pruned dim set).

    out_tag names the output file independently of the input it reads, so a
    figure can record how many dimensions went into it (e.g. '_dim11') without
    that string having to match an on-disk NIfTI. Defaults to dim_tag.
    """
    if out_tag is None:
        out_tag = dim_tag
    from nilearn import surface, plotting

    surfs = load_surfaces(sub)
    if surfs is None:
        return

    beh_path = os.path.join(sl_dir, f'sl_beh_combined{beh_tag}_sub{sub}.nii.gz')
    dim_path = os.path.join(sl_dir, f'sl_combined{dim_tag}_sub{sub}.nii.gz')
    if not os.path.exists(beh_path) or not os.path.exists(dim_path):
        print(f"  Missing combined NIfTI for sub-{sub}")
        return

    beh_img = nib.load(beh_path)
    dim_img = nib.load(dim_path)
    if positive_only:
        # Signed-r maps: draw positive fits only. nilearn thresholds on |value|,
        # so negative r (anti-prediction) would otherwise be painted on a scale
        # meant for positive fit.
        beh_img, dim_img = [nib.Nifti1Image(np.clip(np.asarray(im.dataobj), 0, None),
                                            im.affine, im.header)
                            for im in (beh_img, dim_img)]

    # Determine shared vmax from both maps
    beh_data = np.asarray(beh_img.dataobj)
    dim_data = np.asarray(dim_img.dataobj)
    all_vals = np.concatenate([beh_data[beh_data > threshold],
                               dim_data[dim_data > threshold]])
    vmax = np.percentile(all_vals, 99) if len(all_vals) > 0 else 0.15

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.06,
                           top=0.90, bottom=0.03, left=0.04, right=0.86)

    panels = [
        (gs[0], beh_img, 'A', 'Behavioral Features'),
        (gs[1], dim_img, 'B', 'Learned Dimensions'),
    ]

    view_names = ['lateral', 'medial', 'ventral']
    hemis = ['lh', 'rh']

    for panel_gs, img, panel_letter, panel_title in panels:
        gs_inner = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=panel_gs,
                                                    wspace=0.0, hspace=0.0)
        axes = np.array([[fig.add_subplot(gs_inner[r, c], projection='3d')
                          for c in range(2)] for r in range(3)])

        textures = {}
        for hemi in hemis:
            textures[hemi] = surface.vol_to_surf(
                img, surfs[f'{hemi}_pial'],
                interpolation='nearest_most_frequent')

        for row, view in enumerate(view_names):
            for col, hemi in enumerate(hemis):
                plotting.plot_surf_stat_map(
                    surfs[f'{hemi}_inflated'], textures[hemi],
                    hemi='left' if hemi == 'lh' else 'right',
                    view=view,
                    bg_map=surfs[f'{hemi}_sulc'],
                    threshold=threshold,
                    vmax=vmax,
                    colorbar=False,
                    cmap='inferno',
                    axes=axes[row, col],
                    figure=fig,
                )
                if rasterize:
                    # Mesh as a 300 dpi image in the PDF; text stays vector
                    axes[row, col].set_rasterized(True)

    # Panel titles (centered above each panel)
    fd = font_delta  # added to every text size (ED Fig. 7 uses 3; Fig. 5 uses 0)
    fig.text(0.22, 0.94, panel_a_title, fontsize=14 + fd,
             fontstyle='italic', ha='center', va='top')
    fig.text(0.63, 0.94, panel_b_title, fontsize=14 + fd,
             fontstyle='italic', ha='center', va='top')
    fig.text(0.04, 0.94, 'A', fontsize=18 + fd, fontweight='bold', va='top')
    fig.text(0.45, 0.94, 'B', fontsize=18 + fd, fontweight='bold', va='top')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.70])
    norm = Normalize(vmin=0, vmax=vmax)
    sm = cm.ScalarMappable(cmap='inferno', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label=cbar_label)
    ticks = np.linspace(0, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:.2f}' for t in ticks])
    cfd = fd + cbar_font_delta  # colour bar gets font_delta plus its own extra
    if cfd:
        # matplotlib's default 10 pt otherwise; left untouched at 0
        cbar.set_label(cbar_label, fontsize=10 + cfd)
        cbar.ax.tick_params(labelsize=10 + cfd)

    save_path = os.path.join(sl_dir,
                             f'sl_combined_vs_behavioral{out_tag}_sub{sub}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


# ============================================================================
# WINNER-TAKE-ALL PLOT (categorical legends)
# ============================================================================

def _render_wta_panel(fig, axes, surfs, imgs, n_cats, cmap, threshold=0.01):
    from nilearn import surface, plotting

    hemis = ['lh', 'rh']
    view_names = ['lateral', 'medial', 'ventral']

    for row, hemi in enumerate(hemis):
        textures = []
        for img in imgs:
            tex = surface.vol_to_surf(
                img, surfs[f'{hemi}_pial'],
                interpolation='nearest_most_frequent')
            textures.append(tex)
        stacked = np.stack(textures, axis=0)

        winner = np.argmax(stacked, axis=0).astype(float)
        max_val = np.max(stacked, axis=0)
        # vmin/vmax are passed explicitly because nilearn otherwise infers
        # vmin from the data, which shifts every colour off the legend in
        # any panel where the lowest-index category happens not to win.
        # avg_method='median' stops a face straddling a patch border from
        # averaging into a different category's colour bin.
        winner = winner + 1.0
        winner[max_val < threshold] = np.nan

        for col, view in enumerate(view_names):
            plotting.plot_surf_stat_map(
                surfs[f'{hemi}_inflated'], winner,
                hemi='left' if hemi == 'lh' else 'right',
                view=view,
                bg_map=surfs[f'{hemi}_sulc'],
                threshold=0.5,
                vmin=0.5,
                vmax=n_cats + 0.5,
                avg_method='median',
                colorbar=False,
                cmap=cmap,
                axes=axes[row, col],
                figure=fig,
            )


def plot_wta_combined_vs_behavioral(sub, best_dims, behavior_cols, sl_dir,
                                    threshold=0.01):
    from nilearn import surface, plotting

    surfs = load_surfaces(sub)
    if surfs is None:
        return

    n_dims = len(best_dims)
    n_feats = len(behavior_cols)

    # Load dimension NIfTIs
    dim_imgs = []
    for d in best_dims:
        p = os.path.join(sl_dir, f'sl_dim_{d}_sub{sub}.nii.gz')
        if not os.path.exists(p):
            print(f"  Missing: {p}")
            return
        dim_imgs.append(nib.load(p))

    # Load behavioral NIfTIs
    beh_imgs = []
    for col in behavior_cols:
        p = os.path.join(sl_dir, f'sl_beh_{_col_to_key(col)}_sub{sub}.nii.gz')
        if not os.path.exists(p):
            print(f"  Missing: {p}")
            return
        beh_imgs.append(nib.load(p))

    dim_colors = COLORS_12[:n_dims]
    beh_colors = COLORS_12[:n_feats]
    dim_cmap = ListedColormap(dim_colors)
    beh_cmap = ListedColormap(beh_colors)

    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.15,
                           top=0.94, bottom=0.02, left=0.06, right=0.68)

    # Panel A: Behavioral Features
    gs_a = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0],
                                            wspace=0.02, hspace=0.05)
    axes_a = np.array([[fig.add_subplot(gs_a[r, c], projection='3d')
                         for c in range(3)] for r in range(2)])

    # Panel B: Learned Dimensions
    gs_b = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1],
                                            wspace=0.02, hspace=0.05)
    axes_b = np.array([[fig.add_subplot(gs_b[r, c], projection='3d')
                         for c in range(3)] for r in range(2)])

    _render_wta_panel(fig, axes_a, surfs, beh_imgs, n_feats, beh_cmap, threshold)
    _render_wta_panel(fig, axes_b, surfs, dim_imgs, n_dims, dim_cmap, threshold)

    # Column titles
    col_titles = ['Lateral', 'Medial', 'Ventral']
    for col, title in enumerate(col_titles):
        axes_a[0, col].set_title(title, fontsize=13, fontweight='bold', pad=0)

    # Row labels
    hemi_labels = ['LH', 'RH']
    for axes_panel in [axes_a, axes_b]:
        for row, label in enumerate(hemi_labels):
            axes_panel[row, 0].text2D(-0.15, 0.5, label,
                                       transform=axes_panel[row, 0].transAxes,
                                       fontsize=13, va='center', ha='center')

    # Panel labels
    fig.text(0.02, 0.93, 'A', fontsize=18, fontweight='bold', va='top')
    fig.text(0.06, 0.91, 'Behavioral Features', fontsize=14, fontstyle='italic', va='top')
    fig.text(0.02, 0.49, 'B', fontsize=18, fontweight='bold', va='top')
    fig.text(0.06, 0.47, 'Learned Dimensions', fontsize=14, fontstyle='italic', va='top')

    # Behavioral legend (top right)
    beh_patches = [mpatches.Patch(color=beh_colors[i], label=_display_name(behavior_cols[i]))
                   for i in range(n_feats)]
    leg_a = fig.legend(handles=beh_patches, loc='upper right',
                       bbox_to_anchor=(0.99, 0.92), fontsize=9, frameon=True,
                       title='Behavioral Feature', title_fontsize=10,
                       handlelength=2.0, handleheight=1.3,
                       borderpad=0.8, labelspacing=0.6)

    # Dimension legend (bottom right)
    dim_patches = [mpatches.Patch(color=dim_colors[i],
                                  label=f'Dim {best_dims[i]}: {DIM_LABELS.get(best_dims[i], "")}')
                   for i in range(n_dims)]
    leg_b = fig.legend(handles=dim_patches, loc='lower right',
                       bbox_to_anchor=(0.99, 0.05), fontsize=9, frameon=True,
                       title='Dimension', title_fontsize=10,
                       handlelength=2.0, handleheight=1.3,
                       borderpad=0.8, labelspacing=0.6)
    fig.add_artist(leg_a)

    save_path = os.path.join(sl_dir, f'sl_wta_combined_vs_behavioral_sub{sub}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


# ============================================================================
# DIMENSIONS-ONLY WTA (side-by-side layout matching combined plot)
# ============================================================================

def plot_wta_dims_only(sub, best_dims, sl_dir, threshold=0.01):
    """Dimensions WTA in 3-row (views) × 2-col (hemispheres) layout with legend."""
    from nilearn import surface, plotting

    surfs = load_surfaces(sub)
    if surfs is None:
        return

    n_dims = len(best_dims)

    dim_imgs = []
    for d in best_dims:
        p = os.path.join(sl_dir, f'sl_dim_{d}_sub{sub}.nii.gz')
        if not os.path.exists(p):
            print(f"  Missing: {p}")
            return
        dim_imgs.append(nib.load(p))

    dim_colors = COLORS_12[:n_dims]
    dim_cmap = ListedColormap(dim_colors)

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.0, hspace=-0.25,
                           top=0.95, bottom=0.0, left=0.04, right=0.65)

    axes = np.array([[fig.add_subplot(gs[r, c], projection='3d')
                      for c in range(2)] for r in range(3)])

    view_names = ['lateral', 'medial', 'ventral']
    hemis = ['lh', 'rh']

    for hemi in hemis:
        textures = []
        for img in dim_imgs:
            tex = surface.vol_to_surf(
                img, surfs[f'{hemi}_pial'],
                interpolation='nearest_most_frequent')
            textures.append(tex)
        stacked = np.stack(textures, axis=0)

        winner = np.argmax(stacked, axis=0).astype(float)
        max_val = np.max(stacked, axis=0)
        # vmin/vmax are passed explicitly because nilearn otherwise infers
        # vmin from the data, which shifts every colour off the legend in
        # any panel where the lowest-index category happens not to win.
        # avg_method='median' stops a face straddling a patch border from
        # averaging into a different category's colour bin.
        winner = winner + 1.0
        winner[max_val < threshold] = np.nan

        col = 0 if hemi == 'lh' else 1
        for row, view in enumerate(view_names):
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
            )

    fig.text(0.37, 0.96, 'Learned Dimensions', fontsize=14,
             fontstyle='italic', ha='center', va='top')

    dim_patches = [mpatches.Patch(color=dim_colors[i],
                                  label=f'Dim {best_dims[i]}: {DIM_LABELS.get(best_dims[i], "")}')
                   for i in range(n_dims)]
    fig.legend(handles=dim_patches, loc='center right',
               bbox_to_anchor=(1.05, 0.5), fontsize=9, frameon=True,
               title='Dimension', title_fontsize=10,
               handlelength=2.0, handleheight=1.3,
               borderpad=0.8, labelspacing=0.6)

    save_path = os.path.join(sl_dir, f'sl_wta_dims_only_sub{sub}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', choices=['best12', 'all'], default='best12',
                        help="best12 (default) uses sl_combined_sub{N}.nii.gz; "
                             "all uses sl_combined_all_sub{N}.nii.gz.")
    parser.add_argument('--skip-wta', action='store_true',
                        help="Skip winner-take-all plots.")
    parser.add_argument('--wta-dims-only', action='store_true',
                        help="Generate dimensions-only WTA plot.")
    parser.add_argument('--subjects', type=int, nargs='+', default=None)
    parser.add_argument('--panel-a-title', default='Behavioral Features',
                        help="Italic title over panel A (the behavioral map). "
                             "The manuscript uses 'Pre-labeled Annotations'.")
    parser.add_argument('--panel-b-title', default='Learned Dimensions',
                        help="Italic title over panel B (the embedding map).")
    parser.add_argument('--dim-tag', default=None,
                        help="Embedding NIfTI tag, overriding --dims: reads "
                             "sl_combined{dim_tag}_sub{N}.nii.gz. Fig. 5 uses "
                             "'_all_10fold'; ED Fig. 7 uses '_dim12_10fold'.")
    parser.add_argument('--beh-tag', default='',
                        help="Annotation NIfTI tag: reads "
                             "sl_beh_combined{beh_tag}_sub{N}.nii.gz. The "
                             "manuscript uses '_action_10fold'.")
    parser.add_argument('--threshold', type=float, default=0.01,
                        help="Display threshold (0.01 for r^2 maps; use 0.1 "
                             "for signed-r maps, the same cut in r units).")
    parser.add_argument('--cbar-label', default='Spearman $r^2$')
    parser.add_argument('--positive-only', action='store_true',
                        help="Clip negative values to 0 before drawing (signed-r maps).")
    parser.add_argument('--rasterize', action='store_true',
                        help="Embed the brain surfaces in the PDF as 300 dpi "
                             "images (~10x smaller); text stays vector.")
    parser.add_argument('--cbar-font-delta', type=int, default=0,
                        help="Extra points on the colour-bar label and ticks, on "
                             "top of --font-delta (Fig. 5 uses 3).")
    parser.add_argument('--font-delta', type=int, default=0,
                        help="Points added to every text size in the combined "
                             "figure (ED Fig. 7 uses 3).")
    parser.add_argument('--out-tag', default=None,
                        help="Tag inserted into the output filename instead of "
                             "the input tag, e.g. '_dim11' to record the "
                             "dimension count. Defaults to the input tag.")
    args = parser.parse_args()

    dim_tag = '' if args.dims == 'best12' else '_all'
    if args.dim_tag is not None:
        dim_tag = args.dim_tag
    subjects = args.subjects if args.subjects else SUBJECTS

    for sub in subjects:
        print(f"\n=== Subject {sub} ===")
        plot_combined_vs_behavioral(sub, RESULTS_SEARCHLIGHT_DIR, dim_tag=dim_tag,
                                    out_tag=args.out_tag,
                                    panel_a_title=args.panel_a_title,
                                    panel_b_title=args.panel_b_title,
                                    beh_tag=args.beh_tag,
                                    font_delta=args.font_delta,
                                    rasterize=args.rasterize,
                                    threshold=args.threshold,
                                    cbar_label=args.cbar_label,
                                    positive_only=args.positive_only,
                                    cbar_font_delta=args.cbar_font_delta)
        if not args.skip_wta and args.dims == 'best12' and args.dim_tag is None:
            plot_wta_combined_vs_behavioral(sub, BEST_DIMS, BEHAVIOR_COLS,
                                            RESULTS_SEARCHLIGHT_DIR)
        if args.wta_dims_only:
            plot_wta_dims_only(sub, BEST_DIMS, RESULTS_SEARCHLIGHT_DIR)
