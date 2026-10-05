"""
Pipeline: given one cached seed/bootstrap refit (from dimension_recoverability.py),
render the Hebart-style contrastive-word grid (same visualization as
contrastive_grid_all_batch0.png) for that refit's OWN top-11 dimensions.

Two selection modes (--select):
  * test-cv (default) -- greedy forward selection, scored by 10-fold x
    5-repeat CV R² against the dense 50-video human test RSM, run on THIS
    seed's own embedding columns (not the reference embedding). Take the
    first 11 dims in that greedy order. This is an independent top-11 for
    the refit: it is never told about the reference BEST_DIMS or about which
    columns collided with which reference dims, so it cannot contain
    duplicates by construction (greedy selection never re-adds a column that
    is already selected) and it is not the "same 11 dims we already have" --
    it is whichever 11 columns of THIS refit's own embedding best predict
    held-out human similarity judgments for this refit.
  * reference-match -- match each of the reference
    BEST_DIMS to its best-correlating column in the refit independently (no
    uniqueness constraint), so a collision (merge) plots the same column
    twice.

Reuses cached refits (seed_models.npz / bootstrap_models.npz); no retraining.
Contrastive stats are computed fresh for the refit's own columns (they are
not in contrastive_word_stats.csv, which only covers the reference embedding).

Usage:
    python analysis/dimensions/refit_contrastive_grid.py --seed 101 --refit bootstrap
    python analysis/dimensions/refit_contrastive_grid.py --seed 101 --refit bootstrap --select reference-match

Outputs (data/interim/results/dimensions/):
  * refit_contrastive_grid_{refit}_seed{seed}_batch0.{png,pdf}
  * prints the selected columns (with their test-set greedy CV R² contribution,
    and the closest reference BEST_DIMS label for interpretability only)
"""

import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    UTILS_DIR, MODEL_DIR, RESULTS_DIMENSIONS_DIR, BEST_DIMS, DIM_LABELS,
    RANDOM_SEED, test_idx,
)
from analysis.stability.dimension_recoverability import (
    OUT_DIR, SEED_MODELS_CACHE, BOOT_MODELS_CACHE, _corr_cross, SEED_LIST,
)
from analysis.dimensions.dimension_interpretation import (
    CaptionCorpus, analyze_dimension_contrastive, POS_COLORS,
)

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


def load_corpus():
    raw_df = pd.read_csv(os.path.join(UTILS_DIR, 'frame_data.csv'))[
        ['video_name', 'captions']
    ].reset_index().rename(columns={'index': 'vid_id'})
    raw_images_dir = os.path.join(os.path.dirname(UTILS_DIR), 'raw', 'images')
    rows = []
    for _, row in raw_df.iterrows():
        image_path = os.path.join(raw_images_dir, row['video_name'].replace('.mp4', '.png'))
        try:
            caption_list = ast.literal_eval(row['captions'])
        except (ValueError, SyntaxError):
            caption_list = [str(row['captions'])]
        for cap in caption_list:
            cap = cap.strip()
            if cap:
                rows.append({'vid_id': row['vid_id'], 'captions': cap, 'image_path': image_path})
    caps_df = pd.DataFrame(rows)
    return CaptionCorpus.from_caption_df_with_pos(caps_df), caps_df


def _cv_r2(X, y, n_folds=10, n_repeats=5, seed=RANDOM_SEED):
    """Same CV procedure as dim_selection_train.py's cv_r2: repeated K-fold,
    sklearn r2 scoring."""
    scores = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        scores.extend(cross_val_score(LinearRegression(), X, y, cv=kf, scoring='r2'))
    return float(np.mean(scores))


def greedy_select_by_test_r2(refit_emb, n_select=11, verbose=True):
    """Greedy forward selection of THIS refit's own columns, scored by CV R²
    against the dense 50-video human test RSM (same procedure as
    dim_selection_train.py, applied to a refit instead of the reference
    embedding). Never told about reference dims or BEST_DIMS, so the result
    cannot contain duplicates and is not tied to the canonical top-11.
    """
    test_rsm = pd.read_csv(os.path.join(UTILS_DIR, 'human_test_rsm.csv')).to_numpy()
    idx = np.asarray(sorted(test_idx))
    emb_test = refit_emb[idx, :]
    triu = np.triu_indices(len(idx), k=1)
    y_test = test_rsm[triu]

    n_dims = refit_emb.shape[1]
    X_all = np.zeros((len(y_test), n_dims))
    for d in range(n_dims):
        dist = pdist(emb_test[:, [d]], metric='euclidean')
        dmax = dist.max()
        X_all[:, d] = 1 - (dist / dmax) if dmax != 0 else 1 - dist

    remaining = list(range(n_dims))
    selected = []
    cumul_r2 = []
    for step in range(min(n_select, n_dims)):
        best_r2, best_dim = -np.inf, None
        for d in remaining:
            r2 = _cv_r2(X_all[:, selected + [d]], y_test)
            if r2 > best_r2:
                best_r2, best_dim = r2, d
        selected.append(best_dim)
        remaining.remove(best_dim)
        cumul_r2.append(best_r2)
        if verbose:
            print(f"  step {step + 1:2d}: col {best_dim:2d}  cumulative test CV R²={best_r2:.4f}")

    marginal = np.diff([0.0] + cumul_r2)

    # For interpretability only: closest reference BEST_DIMS label, not used for selection
    reference = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    R = _corr_cross(reference[:, list(BEST_DIMS)], refit_emb[:, selected])
    closest_label, closest_dim_id, closest_dim_r = [], [], []
    for j in range(len(selected)):
        i = R[:, j].argmax()
        closest_label.append(f"~D{BEST_DIMS[i]} {DIM_LABELS.get(BEST_DIMS[i], '')} (r={R[i, j]:.2f})")
        closest_dim_id.append(int(BEST_DIMS[i]))
        closest_dim_r.append(float(R[i, j]))

    row_label = [
        f"Step {i + 1}, col {c}\n{lbl}\nΔR²={m:.3f}"
        for i, (c, lbl, m) in enumerate(zip(selected, closest_label, marginal))
    ]
    df = pd.DataFrame({
        'refit_col': selected,
        'row_label': row_label,
        'cumulative_test_r2': cumul_r2,
        'marginal_r2': marginal,
        'closest_reference_dim': closest_label,
        'closest_dim_id': closest_dim_id,
        'closest_dim_r': closest_dim_r,
        'merged': False,   # by construction: greedy never re-selects a column
    })
    return df


def fractionation_stats(embeddings, col_a, col_b, corpus, top_n_videos=20,
                        contrastive_top_k=8, top_n_words=10):
    """Two parameters that must NOT be conflated: top_n_videos=20 is the window
    for the video-overlap statistic (direct activation ranking); contrastive_
    top_k=8 is the window used to BUILD the contrastive word ranking (matches
    analyze_dimension_contrastive's usage everywhere else in the project --
    dimension_interpretation.py's own grids, regen_contrastive_grid_all.py).

    Two columns are "fractionated" (distinct content, not a redundant
    duplicate) if BOTH the top-N video sets and the top-M contrastive-word
    sets have low overlap despite the columns being at least moderately
    correlated (otherwise they wouldn't plausibly stem from one broad
    category to begin with).
    """
    acts_a, acts_b = embeddings[:, col_a], embeddings[:, col_b]
    top_a = set(np.argsort(-acts_a)[:top_n_videos].tolist())
    top_b = set(np.argsort(-acts_b)[:top_n_videos].tolist())
    shared_videos = len(top_a & top_b)

    cdf_a = analyze_dimension_contrastive(embeddings, corpus, col_a, top_k=contrastive_top_k)
    cdf_b = analyze_dimension_contrastive(embeddings, corpus, col_b, top_k=contrastive_top_k)
    words_a = set(cdf_a[cdf_a['log_ratio'] > 0].head(top_n_words)['word'])
    words_b = set(cdf_b[cdf_b['log_ratio'] > 0].head(top_n_words)['word'])
    shared_words = len(words_a & words_b)

    r = float(np.corrcoef(acts_a, acts_b)[0, 1])
    return {
        'col_a': col_a, 'col_b': col_b, 'r': r,
        'shared_videos': shared_videos, 'n_top_videos': top_n_videos,
        'shared_words': shared_words, 'n_top_words': top_n_words,
        'words_a': sorted(words_a), 'words_b': sorted(words_b),
    }


def match_best_dims_to_refit(reference, refit, best_dims):
    """Plain max-correlation match, independently per reference dim -- no
    uniqueness constraint, so a collision (merge) shows up as a duplicate
    column across two rows."""
    R = _corr_cross(reference, refit)   # (n_ref, k_refit)
    rows = []
    for d in best_dims:
        col = int(R[d].argmax())
        rows.append({'ref_dim': d, 'label': DIM_LABELS.get(d, ''),
                     'refit_col': col, 'r': float(R[d, col])})
    df = pd.DataFrame(rows)
    dup_cols = df['refit_col'][df['refit_col'].duplicated(keep=False)].unique()
    df['merged'] = df['refit_col'].isin(dup_cols)
    df['row_label'] = [
        f"D{row.ref_dim}: {row.label}\ncol {row.refit_col}, r={row.r:.2f}"
        + ("\n[MERGED]" if row.merged else "")
        for row in df.itertuples()
    ]
    return df


def plot_labeled_contrastive_grid(embeddings, corpus, contrastive_by_col, mapping_df,
                                   top_k=8, max_words=20, figsize=(10, 12), save_path=None):
    """Same rendering as dimension_interpretation.plot_contrastive_grid, but
    each row is labeled with the reference dim it stands in for (and its
    match r) rather than the raw refit column index, and merged rows are
    flagged."""
    if not HAS_WORDCLOUD:
        print("WARNING: wordcloud not installed, skipping grid.")
        return

    n_rows = len(mapping_df)
    n_cols = top_k + 3
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.02, hspace=0.005)
    gs.set_width_ratios([1.4, 0.08, 2] + [1] * top_k)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005,
                        wspace=0.02, hspace=0.005)

    for r, row in enumerate(mapping_df.itertuples()):
        col = row.refit_col
        acts = embeddings[:, col]
        top_idx = np.argsort(-acts)[:top_k]

        cdf = contrastive_by_col.get(col, pd.DataFrame())
        if cdf.empty:
            wc_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        else:
            top_words = cdf[cdf['log_ratio'] > 0].head(max_words)
            freqs = {w: lr for w, lr in zip(top_words['word'], top_words['log_ratio']) if lr > 0}
            if freqs:
                pos_lookup = dict(zip(top_words['word'], top_words['pos']))

                def _make_color_func(lookup):
                    def color_func(word, **kwargs):
                        return POS_COLORS.get(lookup.get(word, 'Other'), '#BBBBBB')
                    return color_func

                wc = WordCloud(width=400, height=200, background_color='white',
                              max_words=max_words, color_func=_make_color_func(pos_lookup),
                              prefer_horizontal=0.7)
                wc.generate_from_frequencies(freqs)
                wc_img = wc.to_array()
            else:
                wc_img = np.ones((200, 400, 3), dtype=np.uint8) * 255

        ax_lbl = fig.add_subplot(gs[r, 0])
        ax_lbl.text(0.5, 0.5, row.row_label, ha='center', va='center', fontsize=7.5)
        ax_lbl.axis('off')

        ax_sp = fig.add_subplot(gs[r, 1])
        ax_sp.axis('off')

        ax_wc = fig.add_subplot(gs[r, 2])
        ax_wc.imshow(wc_img)
        ax_wc.axis('off')

        for c, vid in enumerate(top_idx):
            ax = fig.add_subplot(gs[r, c + 3])
            vid = int(vid)
            if vid in corpus.video_df.index:
                img_path = corpus.video_df.loc[vid, 'image_path']
                try:
                    ax.imshow(mpimg.imread(img_path))
                except Exception:
                    ax.text(0.5, 0.5, f"Missing\n{vid}", ha='center', va='center', fontsize=8)
            else:
                ax.text(0.5, 0.5, f"No row\n{vid}", ha='center', va='center', fontsize=8)
            ax.axis('off')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf',
                    bbox_inches='tight', pad_inches=0.02)
        print(f"Saved {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=101,
                        help='Which of the 20 cached refit seeds to use (default 101, '
                             'a bootstrap refit where Dim 0/Dim 4 collide).')
    parser.add_argument('--refit', choices=['seed', 'bootstrap'], default='bootstrap')
    parser.add_argument('--select', choices=['test-cv', 'reference-match'], default='test-cv',
                        help="test-cv (default): greedy forward selection scored against the "
                             "held-out test RSM, on this refit's own columns -- an independent "
                             "top-11, no duplicates possible. reference-match: match each "
                             "reference BEST_DIMS to its best column independently (may show "
                             "duplicates on a merge).")
    parser.add_argument('--n-select', type=int, default=11)
    parser.add_argument('--dims', type=int, nargs='+', default=None,
                        help='(reference-match mode only) reference dims to match; default BEST_DIMS.')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--max-words', type=int, default=20)
    args = parser.parse_args()

    if args.seed not in SEED_LIST:
        raise ValueError(f"--seed must be one of the cached seeds: {SEED_LIST}")

    cache_path = SEED_MODELS_CACHE if args.refit == 'seed' else BOOT_MODELS_CACHE
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} not found -- run dimension_recoverability.py first.")

    models = np.load(cache_path, allow_pickle=True)['models'].item()
    refit_emb = models[args.seed]
    print(f"Refit ({args.refit}, seed {args.seed}): {refit_emb.shape}")

    if args.select == 'test-cv':
        print(f"\nGreedy forward selection on this refit's own {refit_emb.shape[1]} columns, "
              f"scored against the held-out test RSM (top {args.n_select}):")
        mapping_df = greedy_select_by_test_r2(refit_emb, n_select=args.n_select)
        print()
        print(mapping_df[['refit_col', 'closest_reference_dim', 'marginal_r2',
                          'cumulative_test_r2']].to_string(index=False))
        print(f"\n{mapping_df['refit_col'].nunique()} unique columns among "
              f"{len(mapping_df)} selected (no duplicates, by construction).")
    else:
        reference = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
        dims = args.dims if args.dims is not None else list(BEST_DIMS)
        mapping_df = match_best_dims_to_refit(reference, refit_emb, dims)
        print("\nReference dim -> refit column mapping (plain max-correlation, no uniqueness):")
        print(mapping_df.to_string(index=False))
        n_merged = mapping_df['merged'].sum()
        if n_merged:
            print(f"\n{n_merged} of {len(mapping_df)} rows share a column with another row "
                  f"(a merge) -- those rows will render an IDENTICAL panel below.")

    print("\nBuilding caption corpus...")
    corpus, caps_df = load_corpus()
    print(f"Corpus: {len(corpus.video_df)} videos")

    print("Computing contrastive word stats for the selected refit columns...")
    contrastive_by_col = {}
    for col in mapping_df['refit_col'].unique():
        contrastive_by_col[int(col)] = analyze_dimension_contrastive(
            refit_emb, corpus, int(col), top_k=8
        )

    out_name = f"refit_contrastive_grid_{args.refit}_seed{args.seed}_{args.select}_batch0.png"
    save_path = os.path.join(RESULTS_DIMENSIONS_DIR, out_name)
    plot_labeled_contrastive_grid(
        refit_emb, corpus, contrastive_by_col, mapping_df,
        top_k=args.top_k, max_words=args.max_words,
        figsize=(10, 12), save_path=save_path,
    )


if __name__ == '__main__':
    main()
