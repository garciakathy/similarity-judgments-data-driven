"""Regenerate contrastive_grid_all batch plots with BEST_DIMS as batch0.

Reuses saved contrastive_word_stats.csv (no recompute), loads embeddings
and builds a minimal caption corpus (no POS/spacy needed for plotting).

Also writes contrastive_grid_all_remaining.{png,pdf}: the 29 non-BEST dims in
one two-column figure (paper ED Fig. 3), in train greedy order, instead of batches 1-3 as separate
pages. --skip-batches writes only that figure and leaves the batch files
(batch0 is paper Fig. 3) untouched.
"""
import argparse
import os
import sys
import ast
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import UTILS_DIR, MODEL_DIR, RESULTS_DIMENSIONS_DIR, BEST_DIMS, RANDOM_SEED
from analysis.dimensions.dimension_interpretation import (
    CaptionCorpus,
    plot_contrastive_grid,
    plot_contrastive_grid_columns,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-batches', action='store_true',
                    help="Only write the combined non-BEST figure (ED Fig. 3).")
    args = ap.parse_args()

    sparse_pruned_emb = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    print(f"Loaded embeddings: {sparse_pruned_emb.shape}")

    raw_df = pd.read_csv(os.path.join(UTILS_DIR, 'frame_data.csv'))[
        ['video_name', 'captions']
    ].reset_index().rename(columns={'index': 'vid_id'})
    raw_images_dir = os.path.join(os.path.dirname(UTILS_DIR), 'raw', 'images')
    rows = []
    for _, row in raw_df.iterrows():
        image_path = os.path.join(
            raw_images_dir, row['video_name'].replace('.mp4', '.png')
        )
        try:
            caption_list = ast.literal_eval(row['captions'])
        except (ValueError, SyntaxError):
            caption_list = [str(row['captions'])]
        for cap in caption_list:
            cap = cap.strip()
            if cap:
                rows.append({
                    'vid_id': row['vid_id'],
                    'captions': cap,
                    'image_path': image_path,
                })
    caps_df = pd.DataFrame(rows)
    corpus = CaptionCorpus.from_caption_df(caps_df)
    print(f"Corpus: {len(corpus.video_df)} videos")

    cstats_path = os.path.join(RESULTS_DIMENSIONS_DIR, 'contrastive_word_stats.csv')
    all_cstats = pd.read_csv(cstats_path)
    print(f"Loaded {cstats_path}: {len(all_cstats)} rows, "
          f"{all_cstats['dim_id'].nunique()} dims")

    n_all = sparse_pruned_emb.shape[1]
    if 'signed_contrast' not in all_cstats:
        raise SystemExit("contrastive_word_stats.csv is single-split (no signed_contrast); "
                         "re-run analysis/dimensions/dimension_interpretation.py first")
    cached_dims = set(all_cstats['dim_id'].unique().tolist())
    missing = [d for d in range(n_all) if d not in cached_dims]
    if missing:
        print(f"Computing missing contrastive results for dims: {missing}")
        from analysis.dimensions.dimension_interpretation import (
            analyze_dimension_contrastive, CONTRASTIVE_N_RESAMPLES)
        corpus_pos = CaptionCorpus.from_caption_df_with_pos(caps_df)
        extra = []
        for d in missing:
            cdf = analyze_dimension_contrastive(
                sparse_pruned_emb, corpus_pos, d, top_k=8,
                n_resamples=CONTRASTIVE_N_RESAMPLES,
            )
            extra.append(cdf)
        if extra:
            all_cstats = pd.concat([all_cstats] + extra, ignore_index=True)

    contrastive_results = [g.reset_index(drop=True) for _, g in
                           all_cstats.groupby('dim_id', sort=False)]

    best_dims = list(BEST_DIMS)
    # Non-BEST dims in train greedy order, matching ED Table 1 and ED Figs. 4-5
    greedy_order = pd.read_csv(os.path.join(os.path.dirname(RESULTS_DIMENSIONS_DIR), 'behavioral',
                                            'greedy_forward_r2_train.csv'))['dim_id'].tolist()
    assert greedy_order[:len(best_dims)] == best_dims, "greedy order no longer starts with BEST_DIMS"
    remaining = [d for d in greedy_order if d not in best_dims]
    ordered_dims = best_dims + remaining

    if not args.skip_batches:
        plot_contrastive_grid(
            embeddings=sparse_pruned_emb,
            corpus=corpus,
            contrastive_results=contrastive_results,
            dims_to_plot=ordered_dims,
            top_k=8,
            max_words=20,
            batch_size=len(best_dims),
            figsize=(10, 12),
            save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'contrastive_grid_all.png'),
            random_state=RANDOM_SEED,
        )
        print(f"\nRegenerated batches (batch0 = {best_dims})")

    # ED Fig. 3: all non-BEST dims on one page, two columns (15 + 14 rows)
    plot_contrastive_grid_columns(
        embeddings=sparse_pruned_emb,
        corpus=corpus,
        contrastive_results=contrastive_results,
        dims_to_plot=remaining,
        n_columns=2,
        top_k=8,
        max_words=20,
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR,
                               'contrastive_grid_all_remaining.png'),
    )


if __name__ == '__main__':
    main()
