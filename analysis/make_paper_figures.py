"""
Render every paper figure at final-artwork size into paper_figures/: 180 mm (or 88 mm) wide, Arial 5-7 pt, vector PDF
with 300-dpi rasters, plus a PNG preview.

Each figure's own plotting code runs unchanged inside helpers.pub_style.capture(),
which stops it writing to results/; the captured figure is then rescaled by
pub_style.finalize(). All inputs are cached results (CSVs, bootstrap/permutation
caches, searchlight NIfTIs), so nothing is recomputed that the original figure
did not also read from cache.

Run:
  .venv/bin/python analysis/make_paper_figures.py            # everything
  .venv/bin/python analysis/make_paper_figures.py fig2 ed1   # a subset
"""

import os
import sys
import argparse
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import pub_style as ps


@contextlib.contextmanager
def _argv(*args):
    old = sys.argv
    sys.argv = ['x', *args]
    try:
        yield
    finally:
        sys.argv = old


def _one(captured, key):
    if key not in captured:
        raise KeyError(f"{key} not captured; got {sorted(captured)}")
    return captured[key]


def _finish(fig, name, **kw):
    ps.finalize(fig, **kw)
    sizes = sorted({round(t.get_fontsize(), 2) for t in ps._visible_texts(fig)})
    pdf = ps.save(fig, name)
    w, h, fonts = ps.report(pdf)
    print(f"{name}: {w:.0f} x {h:.0f} mm, text {sizes[0]}-{sizes[-1]} pt, fonts {fonts}")
    return pdf


# --------------------------------------------------------------------------- figures

def fig2():
    from analysis.behavioral import explained_variance_shared_only as m
    with ps.capture() as cap, _argv('--perm', 'video', '--ci'):
        m.main()
    _finish(_one(cap, 'explained_variance_test_50vids_spearman_emb_shared_total_'
                      'stacked_total_variance_videoperm_ci'), 'Fig2')


def _contrastive_grids():
    from analysis.dimensions import regen_contrastive_grid_all as m
    with ps.capture() as cap, _argv():
        m.main()
    return cap


def fig3_ed3():
    cap = _contrastive_grids()
    _finish(_one(cap, 'contrastive_grid_all_batch0'), 'Fig3')
    # Two columns of 15 rows leave a ~3 mm label column: turn "Dim N" vertical.
    def vertical_dim_labels(t, s):
        if t.get_text().startswith('Dim '):
            t.set_rotation(90)
            t.set_position((0.5, 0.5))
            t.set_ha('center')
            t.set_va('center')
        return s
    _finish(_one(cap, 'contrastive_grid_all_remaining'), 'ED_Fig3',
            size_override=vertical_dim_labels)


def heatmaps():
    from analysis.dimensions import dimension_analysis as m
    with ps.capture() as cap, _argv():
        m.main()
    _finish(_one(cap, 'dim_behavior_spearman_r'), 'Fig4')
    _finish(_one(cap, 'dimension_correlations'), 'ED_Fig4', width=120 * ps.MM)
    _finish(_one(cap, 'dim_behavior_spearman_r_all'), 'ED_Fig5')


def ed1():
    from analysis.behavioral import explained_variance as m
    with ps.capture() as cap, _argv():
        m.main()
    _finish(_one(cap, 'cumulative_r2_test_50vids_kneedle'), 'ED_Fig1')


def ed2():
    from analysis.stability import plot_fractionation_summary as m
    with ps.capture() as cap, _argv('--ci'):
        m.main()
    _finish(_one(cap, 'fractionation_summary_ci'), 'ED_Fig2', width=ps.ONE_COL)


def ed8():
    from analysis.neural import neural_explained_variance_per_subject as m
    with ps.capture() as cap, _argv():
        m.main()
    _finish(_one(cap, 'neural_behavioral_joint_variance_spearman_subjmean_filtered'),
            'ED_Fig8', width=120 * ps.MM)


def _combined_vs_behavioral(dim_tag, out_tag):
    from analysis.neural.plot_per_subject_searchlight import plot_combined_vs_behavioral
    from config import RESULTS_SEARCHLIGHT_DIR
    with ps.capture() as cap:
        plot_combined_vs_behavioral(2, RESULTS_SEARCHLIGHT_DIR, dim_tag=dim_tag,
                                    out_tag=out_tag, beh_tag='_action_10fold',
                                    panel_a_title='Pre-labeled Annotations',
                                    rasterize=True)
    return _one(cap, f'sl_combined_vs_behavioral{out_tag}_sub2')


def fig5():
    _finish(_combined_vs_behavioral('_all_10fold', '_all_action_10fold'), 'Fig5')


def ed6():
    from analysis.neural import plot_combined_surfaces_pair as m
    with ps.capture() as cap, _argv():
        m.main()
    _finish(_one(cap, 'sl_combined_surface_pair_all_action_10fold'), 'ED_Fig6')


def ed7():
    _finish(_combined_vs_behavioral('_dim11_10fold', '_dim11_action_10fold'), 'ED_Fig7')


def ed9():
    from analysis.neural.searchlight_rsa import plot_winner_take_all
    from config import SUBJECTS, BEST_DIMS, RESULTS_SEARCHLIGHT_DIR
    with ps.capture() as cap:
        plot_winner_take_all(SUBJECTS, list(BEST_DIMS), RESULTS_SEARCHLIGHT_DIR,
                             threshold=0.1, tag='_marginal', title=False, rasterize=True)
    _finish(_one(cap, 'sl_winner_take_all_marginal'), 'ED_Fig9')


FIGURES = {'fig2': fig2, 'fig3': fig3_ed3, 'heatmaps': heatmaps,
           'ed1': ed1, 'ed2': ed2, 'ed8': ed8,
           'fig5': fig5, 'ed6': ed6, 'ed7': ed7, 'ed9': ed9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('which', nargs='*', default=list(FIGURES))
    args = ap.parse_args()
    for w in args.which:
        FIGURES[w]()


if __name__ == '__main__':
    main()
