"""
ED Fig. 6: combined searchlight r2 for pre-labeled annotations and learned
dimensions, all subjects, one figure on one colour scale.

  A) Pre-labeled Annotations -- sl_beh_combined{beh_tag}_sub{N}.nii.gz
  B) Learned Dimensions      -- sl_combined{dim_tag}_sub{N}.nii.gz

Reads existing NIfTIs only; no searchlight is recomputed. Surface axes are
rasterised so the PDF is megabytes rather than ~400 MB of mesh.

Outputs (data/interim/results/searchlight/):
  * sl_combined_surface_pair{out_tag}.{png,pdf}
"""

import os
import sys
import argparse

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RESULTS_SEARCHLIGHT_DIR, SUBJECTS
from helpers.surfaces import load_surfaces

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

VIEWS = [
    ('lh', 'lateral'), ('rh', 'lateral'),
    ('lh', 'medial'), ('rh', 'medial'),
    ('lh', 'ventral'), ('rh', 'ventral'),
]


def shared_vmax(paths, threshold):
    """99th percentile of suprathreshold voxels pooled over every map."""
    vals = []
    for p in paths:
        data = np.asarray(nib.load(p).dataobj)
        vals.append(data[data > threshold])
    vals = np.concatenate(vals)
    return float(np.percentile(vals, 99)) if vals.size else 0.15


def plot_pair(subjects, sl_dir, beh_tag, dim_tag, out_tag, threshold=0.01,
              vmax=None, panel_titles=('Pre-labeled Annotations', 'Learned Dimensions'),
              dpi=300):
    from nilearn import surface, plotting

    panels = [
        ('A', panel_titles[0], [os.path.join(sl_dir, f'sl_beh_combined{beh_tag}_sub{s}.nii.gz')
                                for s in subjects]),
        ('B', panel_titles[1], [os.path.join(sl_dir, f'sl_combined{dim_tag}_sub{s}.nii.gz')
                                for s in subjects]),
    ]
    missing = [p for _, _, ps in panels for p in ps if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('Missing NIfTIs:\n  ' + '\n  '.join(missing))

    if vmax is None:
        vmax = shared_vmax([p for _, _, ps in panels for p in ps], threshold)
    print(f"  Shared vmax = {vmax:.3f} (threshold {threshold})")

    n_sub = len(subjects)
    n_rows = 2 * n_sub
    fig, axes = plt.subplots(n_rows, len(VIEWS), figsize=(26, 4 * n_rows),
                             subplot_kw={'projection': '3d'})
    fig.subplots_adjust(left=0.04, right=0.90, bottom=0.01, top=0.97,
                        wspace=0.02, hspace=0.05)

    surfs = {s: load_surfaces(s) for s in subjects}
    for p_i, (letter, title, paths) in enumerate(panels):
        for s_i, (sub, path) in enumerate(zip(subjects, paths)):
            row = p_i * n_sub + s_i
            img = nib.load(path)
            # One projection per hemisphere, reused across its three views
            textures = {h: surface.vol_to_surf(img, surfs[sub][f'{h}_pial'],
                                               interpolation='nearest_most_frequent')
                        for h in ('lh', 'rh')}
            for col, (hemi, view) in enumerate(VIEWS):
                ax = axes[row, col]
                plotting.plot_surf_stat_map(
                    surfs[sub][f'{hemi}_inflated'], textures[hemi],
                    hemi='left' if hemi == 'lh' else 'right',
                    view=view,
                    bg_map=surfs[sub][f'{hemi}_sulc'],
                    threshold=threshold,
                    vmax=vmax,
                    colorbar=False,
                    cmap='inferno',
                    axes=ax,
                    figure=fig,
                )
                ax.set_rasterized(True)
            print(f"  {letter} sub-{sub} done")

            # Subject label at the left edge of the row
            bb = axes[row, 0].get_position()
            fig.text(0.035, (bb.y0 + bb.y1) / 2, f'Sub-{sub}', rotation=90,
                     ha='center', va='center', fontsize=21)

        # Panel letter + italic title above the panel's first row
        top = axes[p_i * n_sub, 0].get_position().y1
        fig.text(0.005, top + 0.004, letter, fontsize=33, fontweight='bold',
                 ha='left', va='bottom')
        fig.text(0.47, top + 0.004, title, fontsize=29, style='italic',
                 ha='center', va='bottom')

    cbar_ax = fig.add_axes([0.92, 0.30, 0.012, 0.40])
    sm = cm.ScalarMappable(cmap='inferno', norm=Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Spearman $r^2$', fontsize=23)
    ticks = np.linspace(0, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:.2f}' for t in ticks])
    cbar.ax.tick_params(labelsize=19)

    save_path = os.path.join(sl_dir, f'sl_combined_surface_pair{out_tag}.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', dpi=dpi,
                bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--beh-tag', default='_action_10fold',
                    help="Annotation NIfTI tag (manuscript: 12 annotations + action, 10-fold).")
    ap.add_argument('--dim-tag', default='_all_10fold',
                    help="Embedding NIfTI tag (manuscript: 40 dims, 10-fold).")
    ap.add_argument('--out-tag', default='_all_action_10fold')
    ap.add_argument('--threshold', type=float, default=0.01)
    ap.add_argument('--vmax', type=float, default=None,
                    help="Override the pooled 99th-percentile vmax.")
    ap.add_argument('--subjects', type=int, nargs='+', default=None)
    args = ap.parse_args()

    plot_pair(args.subjects or SUBJECTS, RESULTS_SEARCHLIGHT_DIR,
              beh_tag=args.beh_tag, dim_tag=args.dim_tag, out_tag=args.out_tag,
              threshold=args.threshold, vmax=args.vmax)


if __name__ == '__main__':
    main()
