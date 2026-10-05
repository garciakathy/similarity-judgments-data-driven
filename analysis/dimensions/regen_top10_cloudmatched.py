"""Top-10 contrastive word tables that match the word clouds in Fig. 3 / ED Fig. 3.

The clouds draw 20 words (``max_words=20``) and size each by log_ratio, while
table_contrastive_top10*.tex lists the first 10 in ranking order, so the
largest words in a cloud can be missing from the table. These tables instead
list each cloud's 10 largest words (``cloud_top_words``); POS shares are
recomputed over those words, JS is unchanged (it does not depend on ranking).

Reuses the cached contrastive_word_stats.csv (no recompute) and writes
``_cloudmatched`` files next to the originals, which are left untouched:
    table_contrastive_top10_cloudmatched.{csv,tex}              (BEST_DIMS)
    table_contrastive_top10_all_cloudmatched.{csv,tex}          (all dims)
    table_contrastive_top10_all_cloudmatched_publication.tex    (ED Table 1)
"""
import os
import sys
import ast
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import UTILS_DIR, MODEL_DIR, RESULTS_DIMENSIONS_DIR, BEST_DIMS, DIM_LABELS
from analysis.dimensions.dimension_interpretation import (
    CaptionCorpus,
    build_contrastive_top10_table,
    write_top10_publication_tex,
)

# Must match max_words in the plot_contrastive_grid* calls that drew the figures
CLOUD_MAX_WORDS = 20
SUFFIX = '_cloudmatched'

CAPTION = (r'Top ten contrastive caption words for every learned dimension, as '
           r'shown in the word clouds. For each dimension, the 20 words most '
           r'over-represented in the captions of its 8 highest- vs 8 lowest-activating '
           r'videos (smoothed log-ratio weighted by log frequency, averaged over 200 '
           r'random tie-breaks of the two groups) were drawn as a word cloud with '
           r'size proportional to log-ratio; listed are the ten largest, in order of '
           r'log-ratio. Subscripts give part of speech (N noun, V verb, Adj adjective, '
           r"Adv adverb, O other); POS columns give each part of speech's share of the "
           "listed words' contrast. JS: Jensen--Shannon divergence (bits) between the "
           r'caption word distributions of the two groups. Dimensions are ordered by '
           r'greedy forward selection on the training RSM; the first 11 are those '
           r'retained by the Kneedle elbow.')


def load_corpus():
    """Plain (no POS) caption corpus; enough for the JS column."""
    raw_df = pd.read_csv(os.path.join(UTILS_DIR, 'frame_data.csv'))[
        ['video_name', 'captions']
    ].reset_index().rename(columns={'index': 'vid_id'})
    rows = []
    for _, row in raw_df.iterrows():
        try:
            caption_list = ast.literal_eval(row['captions'])
        except (ValueError, SyntaxError):
            caption_list = [str(row['captions'])]
        for cap in caption_list:
            cap = cap.strip()
            if cap:
                rows.append({'vid_id': row['vid_id'], 'captions': cap, 'image_path': ''})
    return CaptionCorpus.from_caption_df(pd.DataFrame(rows))


def save(table, greedy, name):
    """CSV + plain .tex in train greedy order, floats at 2 dp."""
    table = (table.merge(greedy.rename(columns={'dim_id': 'Dim'}), on='Dim', how='left')
                  .sort_values('greedy_step', kind='stable').reset_index(drop=True))
    out = table.copy()
    for c in out.select_dtypes('float').columns:
        digits = 4 if c == 'greedy_marginal_r2' else 2
        out[c] = [f"{v:.{digits}f}" if pd.notna(v) else '' for v in out[c]]
    out.to_csv(os.path.join(RESULTS_DIMENSIONS_DIR, f'{name}.csv'), index=False)
    cols = [c for c in out.columns if not c.startswith('greedy_')]
    with open(os.path.join(RESULTS_DIMENSIONS_DIR, f'{name}.tex'), 'w') as f:
        f.write(out[cols].to_latex(index=False, escape=True))
    print(f"  Saved {name}.csv / .tex")
    return table


def main():
    emb = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    corpus = load_corpus()

    cstats = pd.read_csv(os.path.join(RESULTS_DIMENSIONS_DIR, 'contrastive_word_stats.csv'))
    if 'signed_contrast' not in cstats:
        raise SystemExit("contrastive_word_stats.csv is single-split (no signed_contrast); "
                         "re-run analysis/dimensions/dimension_interpretation.py first")
    results = [g.reset_index(drop=True) for _, g in cstats.groupby('dim_id', sort=False)]
    all_dims = sorted(cstats['dim_id'].unique().tolist())
    if len(all_dims) != emb.shape[1]:
        raise SystemExit(f"cache has {len(all_dims)} dims, embedding has {emb.shape[1]}")

    greedy = pd.read_csv(os.path.join(os.path.dirname(RESULTS_DIMENSIONS_DIR), 'behavioral',
                                      'greedy_forward_r2_train.csv'))[
        ['dim_id', 'greedy_step', 'greedy_marginal_r2', 'greedy_cumul_r2']]
    best_dims = list(BEST_DIMS)
    assert greedy['dim_id'].tolist()[:len(best_dims)] == best_dims, \
        "greedy order no longer starts with BEST_DIMS"

    common = dict(contrastive_results=results, top_n=10, embeddings=emb, corpus=corpus,
                  cloud_max_words=CLOUD_MAX_WORDS)
    labels = {d: DIM_LABELS.get(d, '') for d in all_dims}

    t = build_contrastive_top10_table(dims_to_plot=best_dims, dim_labels=labels, **common)
    save(t, greedy, f'table_contrastive_top10{SUFFIX}')

    t = build_contrastive_top10_table(dims_to_plot=all_dims, dim_labels=labels, **common)
    t = save(t, greedy, f'table_contrastive_top10_all{SUFFIX}')
    write_top10_publication_tex(
        t, os.path.join(RESULTS_DIMENSIONS_DIR,
                        f'table_contrastive_top10_all{SUFFIX}_publication.tex'),
        best_dims=best_dims, dim_labels=DIM_LABELS, caption=CAPTION,
        label='tab:contrastive_top10_all_cloudmatched')


if __name__ == '__main__':
    main()
