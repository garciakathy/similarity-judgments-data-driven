"""
Contrastive caption analysis for the *pre-labeled behavioral annotations*.

Companion to the contrastive analysis of the learned embedding dimensions in
``dimension_interpretation.py``. Same corpus, same tokenizer, same POS tagger,
same log-ratio statistic -- the only thing that changes is what defines the
high- and low-activating video groups:

    embedding dims      videos ranked by  sparse_pruned_emb[:, dim]
    behavioral feats    videos ranked by  annotations[feature]

Tie-aware resampling
--------------------
A single top-8/bottom-8 split is not well defined for either side here:

  * ``indoor`` is binary, so 193 videos tie at the 8th-highest value and the
    "top 8" is an arbitrary subset;
  * every embedding dimension is sparse and non-negative, so 74-107 videos sit
    at exactly 0 and the "bottom 8" is an arbitrary subset of those.

Both groups are therefore drawn ``--n-resamples`` times, breaking ties at
random under a fixed seed, and the per-word statistics are averaged over
draws. With no ties this reduces exactly to the deterministic single split
used by ``dimension_interpretation.analyze_dimension_contrastive``. The same
estimator is applied to the embedding dimensions so the two sides of the
comparison are matched.

Outputs (data/interim/results/dimensions/):
    behavioral_contrastive_word_stats.csv    per-word stats, all features
    table_behavioral_contrastive_top10.csv   top-10 words + POS mass per feature
    behavioral_contrastive_pos_breakdown.png/pdf
    pos_composition_comparison.png/pdf       behavioral vs embedding, side by side
    pos_composition_comparison.csv
"""

import os
import sys
import ast
import argparse
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    UTILS_DIR, MODEL_DIR, RESULTS_DIMENSIONS_DIR,
    BEST_DIMS, DIM_LABELS, BEHAVIOR_COLS, FEATURE_REMAPS,
)
from analysis.dimensions.dimension_interpretation import (
    CaptionCorpus, POS_COLORS,
    _pool_pos_counts, _pool_word_counts, _js_divergence_from_counters,
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

POS_ORDER = ['Noun', 'Verb', 'Adjective', 'Adverb', 'Other']

# Binary annotations: the top-k/bottom-k split is entirely tie-broken for these.
BINARY_FEATURES = {'indoor'}


# ============================================================================
# CORPUS
# ============================================================================

def load_corpus() -> Optional[CaptionCorpus]:
    """Build the POS-tagged caption corpus exactly as dimension_interpretation does."""
    captions_path = os.path.join(UTILS_DIR, 'frame_data.csv')
    if not os.path.exists(captions_path):
        print(f"ERROR: captions not found at {captions_path}")
        return None

    raw_df = pd.read_csv(captions_path)[['video_name', 'captions']].reset_index()
    raw_df = raw_df.rename(columns={'index': 'vid_id'})

    rows = []
    for _, row in raw_df.iterrows():
        try:
            caption_list = ast.literal_eval(row['captions'])
        except (ValueError, SyntaxError):
            caption_list = [str(row['captions'])]
        for cap in caption_list:
            cap = cap.strip()
            if cap:
                rows.append({'vid_id': row['vid_id'], 'captions': cap})

    return CaptionCorpus.from_caption_df_with_pos(pd.DataFrame(rows))


def load_annotations(feature_cols: Sequence[str]) -> pd.DataFrame:
    """Load behavioral annotations, indexed 0..249 to match caption vid_id.

    ``frame_data.csv`` and ``annotations.csv`` list the 250 videos
    in the same order (verified on ``video_name``), so positional index is the
    shared key.
    """
    path = os.path.join(UTILS_DIR, 'annotations.csv')
    ann = pd.read_csv(path)

    frame_path = os.path.join(UTILS_DIR, 'frame_data.csv')
    if os.path.exists(frame_path):
        frame_names = pd.read_csv(frame_path)['video_name'].values
        if not np.array_equal(frame_names, ann['video_name'].values):
            raise ValueError(
                "frame_data.csv and annotations.csv disagree on video "
                "order; positional vid_id would misalign captions with ratings."
            )

    missing = [c for c in feature_cols if c not in ann.columns]
    if missing:
        raise ValueError(f"annotations missing columns: {missing}")

    out = ann[list(feature_cols)].copy()
    out.index = np.arange(len(ann))
    return out


# ============================================================================
# TIE-AWARE CONTRASTIVE ANALYSIS
# ============================================================================

def _tie_group_bounds(scores: np.ndarray, top_k: int):
    """Return (n_ties_high, n_ties_low) at the top-k / bottom-k cut points."""
    srt = np.sort(scores)
    return int((scores == srt[-top_k]).sum()), int((scores == srt[top_k - 1]).sum())


def _draw_groups(scores: np.ndarray, valid: np.ndarray, top_k: int,
                 rng: np.random.Generator):
    """Draw a top-k and bottom-k group, breaking ties at random.

    Videos strictly beyond the cut are always included; the remaining slots are
    filled by sampling without replacement from the tied set. The low group is
    drawn from videos not already taken by the high group, so the two are
    disjoint even for a constant score vector.
    """
    srt = np.sort(scores[valid])
    hi_cut, lo_cut = srt[-top_k], srt[top_k - 1]

    sure_hi = valid[scores[valid] > hi_cut]
    tied_hi = valid[scores[valid] == hi_cut]
    n_fill = top_k - len(sure_hi)
    top = np.concatenate([sure_hi, rng.choice(tied_hi, n_fill, replace=False)])

    remaining = np.setdiff1d(valid, top)
    sure_lo = remaining[scores[remaining] < lo_cut]
    tied_lo = remaining[scores[remaining] == lo_cut]
    n_fill = min(top_k - len(sure_lo), len(tied_lo))
    bottom = np.concatenate([sure_lo, rng.choice(tied_lo, n_fill, replace=False)])
    if len(bottom) > top_k:  # more videos strictly below the cut than slots
        bottom = rng.choice(bottom, top_k, replace=False)

    return [int(v) for v in top], [int(v) for v in bottom]


def contrastive_from_scores(
    scores: np.ndarray,
    corpus: CaptionCorpus,
    group_id,
    top_k: int = 8,
    alpha: float = 0.5,
    n_resamples: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Contrastive log-ratio between the top-k and bottom-k videos on ``scores``.

    Identical statistic to ``analyze_dimension_contrastive`` -- log-ratio of
    smoothed counts, weighted by log total frequency -- but averaged over
    ``n_resamples`` random tie-broken draws of the two groups.

    Columns: group_id, word, pos, count_high, count_low, log_ratio,
    weighted_contrast, signed_contrast, sign_consistency, pct_high, pct_low.

    ``signed_contrast`` is the quantity to rank by: it is the mean of
    ``log_ratio * log(total + 1)`` over draws, so a word whose direction flips
    between draws is penalised rather than credited. For a tie-free score vector
    (one effective draw) ranking by ``signed_contrast`` on the ``log_ratio > 0``
    subset is identical to ranking by ``weighted_contrast``.
    """
    scores = np.asarray(scores, dtype=float)
    valid = np.array([i for i in range(len(scores)) if i in corpus.video_df.index])
    rng = np.random.default_rng(seed)

    # Only the tied slots differ between draws; collapse to one draw if none.
    n_hi, n_lo = _tie_group_bounds(scores[valid], top_k)
    n_draws = 1 if (n_hi == 1 and n_lo == 1) else n_resamples

    sum_lr = Counter()
    sum_wc = Counter()
    sum_signed = Counter()
    sum_pos_sign = Counter()
    sum_ct = Counter()
    sum_cb = Counter()
    sum_pct_hi = Counter()
    sum_pct_lo = Counter()
    js_vals = []
    pos_by_key: Dict = {}

    for _ in range(n_draws):
        top_vids, bottom_vids = _draw_groups(scores, valid, top_k, rng)

        top_counts = _pool_pos_counts(top_vids, corpus)
        bottom_counts = _pool_pos_counts(bottom_vids, corpus)
        n_top = sum(top_counts.values()) or 1
        n_bottom = sum(bottom_counts.values()) or 1

        for key in sorted(set(top_counts) | set(bottom_counts)):
            word, pos = key
            pos_by_key[key] = pos
            ct = top_counts.get(key, 0)
            cb = bottom_counts.get(key, 0)
            log_ratio = np.log((ct + alpha) / (cb + alpha))
            weight = np.log(ct + cb + 1)
            sum_lr[key] += log_ratio
            sum_wc[key] += abs(log_ratio) * weight
            sum_signed[key] += log_ratio * weight
            sum_pos_sign[key] += 1.0 if log_ratio > 0 else 0.0
            sum_ct[key] += ct
            sum_cb[key] += cb
            sum_pct_hi[key] += ct / n_top
            sum_pct_lo[key] += cb / n_bottom

        js = _js_divergence_from_counters(_pool_word_counts(top_vids, corpus),
                                          _pool_word_counts(bottom_vids, corpus))
        js_vals.append(js / np.log(2))

    rows = []
    for key, pos in sorted(pos_by_key.items()):
        rows.append({
            'group_id': group_id,
            'word': key[0],
            'pos': pos,
            'count_high': sum_ct[key] / n_draws,
            'count_low': sum_cb[key] / n_draws,
            'log_ratio': sum_lr[key] / n_draws,
            'weighted_contrast': sum_wc[key] / n_draws,
            'signed_contrast': sum_signed[key] / n_draws,
            'sign_consistency': sum_pos_sign[key] / n_draws,
            'pct_high': sum_pct_hi[key] / n_draws,
            'pct_low': sum_pct_lo[key] / n_draws,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Tie-break on word so the top-N cut is reproducible: with sparse
        # counts many words share a signed contrast exactly, and set iteration
        # order varies with PYTHONHASHSEED between runs.
        df = df.sort_values(['signed_contrast', 'word'], ascending=[False, True]
                            ).reset_index(drop=True)
    df.attrs['js_bits'] = float(np.mean(js_vals)) if js_vals else np.nan
    df.attrs['n_draws'] = n_draws
    df.attrs['n_tied_high'] = n_hi
    df.attrs['n_tied_low'] = n_lo
    return df


def top_words(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Top-n words on the high side, ranked by mean signed contrast."""
    if df.empty:
        return df
    return df[df['signed_contrast'] > 0].head(top_n)


def pos_mass(top: pd.DataFrame) -> Dict[str, float]:
    """Share of |signed contrast| held by each POS among the given words."""
    total = top['signed_contrast'].abs().sum()
    if total <= 0:
        return {p: 0.0 for p in POS_ORDER}
    return {p: round(top.loc[top['pos'] == p, 'signed_contrast'].abs().sum()
                     / total * 100, 1) for p in POS_ORDER}


# ============================================================================
# TABLES
# ============================================================================

def build_top10_table(results: Dict, order: Sequence, labels: Dict,
                      top_n: int = 10) -> pd.DataFrame:
    rows = []
    for gid in order:
        df = results[gid]
        top = top_words(df, top_n)
        if top.empty:
            continue
        row = {'Group': labels.get(gid, str(gid))}
        for i in range(top_n):
            row[f'Word {i+1}'] = (
                f"{top.iloc[i]['word']} ({top.iloc[i]['pos'][:1]})" if i < len(top) else ''
            )
        mass = pos_mass(top)
        row.update({'Noun%': mass['Noun'], 'Verb%': mass['Verb'],
                    'Adj%': mass['Adjective'], 'Adverb%': mass['Adverb'],
                    'Other%': mass['Other']})
        row['n_adjectives'] = int((top['pos'] == 'Adjective').sum())
        row['JS'] = round(df.attrs.get('js_bits', np.nan), 3)
        row['n_draws'] = df.attrs.get('n_draws')
        row['n_tied_high'] = df.attrs.get('n_tied_high')
        row['n_tied_low'] = df.attrs.get('n_tied_low')
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# PLOTS
# ============================================================================

def plot_pos_breakdown(results: Dict, order: Sequence, labels: Dict,
                       top_n: int = 10, save_path: Optional[str] = None,
                       n_cols: int = 2):
    """Horizontal bars of the top contrastive words, coloured by POS.

    Same layout as ``contrastive_pos_breakdown.png`` so the two figures can be
    read side by side.
    """
    n = len(order)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows),
                             squeeze=False)

    x_lim = 1.1 * max(
        (top_words(results[g], top_n)['signed_contrast'].max()
         for g in order if not top_words(results[g], top_n).empty),
        default=1.0,
    )

    for idx, gid in enumerate(order):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        df = results[gid]
        top = top_words(df, top_n)
        title = labels.get(gid, str(gid))

        if top.empty:
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')
            continue

        sub = top.sort_values('signed_contrast')
        colors = [POS_COLORS.get(p, '#BBBBBB') for p in sub['pos']]
        y_pos = np.arange(len(sub))
        ax.barh(y_pos, sub['signed_contrast'].values, color=colors,
                edgecolor='none', height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub['word'].values, fontsize=12)
        ax.set_xlim(0, x_lim)
        ax.set_title(title, fontsize=11, fontweight='bold')

        mass = pos_mass(top)
        note = f"JS = {df.attrs.get('js_bits', np.nan):.2f} bits\nN {mass['Noun']:.0f}% · V {mass['Verb']:.0f}% · A {mass['Adjective']:.0f}%"
        ax.text(0.97, 0.05, note, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=11, fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#CCCCCC', alpha=0.9))

        ax.set_xlabel("Weighted Contrast" if row == n_rows - 1 else "", fontsize=14)

    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')

    legend = [Patch(facecolor=c, label=p) for p, c in POS_COLORS.items()]
    fig.legend(handles=legend, loc='upper center', fontsize=10, frameon=True,
               title='POS', title_fontsize=10, ncol=len(POS_COLORS),
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf',
                    bbox_inches='tight')
        print(f"  Saved {save_path}")
    plt.close(fig)


def plot_pos_comparison(comp: pd.DataFrame, save_path: Optional[str] = None):
    """Stacked POS composition: behavioral features vs embedding dimensions."""
    blocks = [('behavioral', 'Pre-labeled behavioral ratings'),
              ('embedding', 'Learned embedding dimensions')]
    present = [(k, t) for k, t in blocks if (comp['kind'] == k).any()]
    heights = [(comp['kind'] == k).sum() for k, _ in present]

    fig, axes = plt.subplots(
        len(present), 2, figsize=(14, 0.42 * sum(heights) + 3.4),
        gridspec_kw={'height_ratios': heights, 'width_ratios': [2.05, 1]},
        squeeze=False,
    )

    for bi, (kind, title) in enumerate(present):
        sub = comp[comp['kind'] == kind].sort_values('Noun%', ascending=True)
        y = np.arange(len(sub))

        # --- left: stacked POS composition ---
        ax = axes[bi, 0]
        left = np.zeros(len(sub))
        for pos in POS_ORDER:
            vals = sub[f'{pos}%'].values if f'{pos}%' in sub else sub[f'{pos[:3]}%'].values
            ax.barh(y, vals, left=left, color=POS_COLORS[pos], height=0.72,
                    edgecolor='white', linewidth=0.5)
            for yi, (v, l) in enumerate(zip(vals, left)):
                if v >= 6:
                    ax.text(l + v / 2, yi, f"{v:.0f}", ha='center', va='center',
                            fontsize=9, color='white', fontweight='bold')
            left += vals
        ax.set_yticks(y)
        ax.set_yticklabels(sub['label'].values, fontsize=11)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of top-10 contrastive mass" if bi == len(present) - 1 else "",
                      fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')

        # --- right: descriptive share, adjective and adverb kept distinct ---
        ax2 = axes[bi, 1]
        adj, adv = sub['Adjective%'].values, sub['Adverb%'].values
        ax2.barh(y, adj, color=POS_COLORS['Adjective'], height=0.72)
        ax2.barh(y, adv, left=adj, color=POS_COLORS['Adverb'], height=0.72)
        desc = adj + adv
        for yi, v in enumerate(desc):
            ax2.text(v + 0.6, yi, f"{v:.1f}", va='center', fontsize=9)
        ax2.set_yticks(y)
        ax2.set_yticklabels([])
        ax2.set_xlim(0, max(6.0, comp['Adjective%'].add(comp['Adverb%']).max() * 1.25))
        ax2.set_xlabel("% descriptive (adj + adv)" if bi == len(present) - 1 else "",
                       fontsize=12)
        ax2.set_title("Descriptive share", fontsize=11, loc='left')

        for a in (ax, ax2):
            a.spines[['top', 'right']].set_visible(False)

    legend = [Patch(facecolor=POS_COLORS[p], label=p) for p in POS_ORDER]
    fig.legend(handles=legend, loc='upper center', fontsize=10, frameon=True,
               title='POS', title_fontsize=10, ncol=len(POS_ORDER),
               bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=[0, 0, 1, 0.955])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf',
                    bbox_inches='tight')
        print(f"  Saved {save_path}")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--top-k', type=int, default=8,
                    help='videos per group (default 8, matching the dimension analysis)')
    ap.add_argument('--top-n', type=int, default=10,
                    help='contrastive words summarised per group')
    ap.add_argument('--n-resamples', type=int, default=200,
                    help='random tie-broken draws to average over')
    ap.add_argument('--seed', type=int, default=888)
    args = ap.parse_args()

    print("=" * 60)
    print("BEHAVIORAL CONTRASTIVE CAPTION ANALYSIS")
    print("=" * 60)

    corpus = load_corpus()
    if corpus is None:
        return
    print(f"Corpus: {len(corpus.video_df)} videos, {corpus.global_n_tokens} tokens")

    feature_cols = list(BEHAVIOR_COLS)
    ann = load_annotations(feature_cols)

    emb = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    print(f"Embeddings: {emb.shape}")

    def feat_label(col: str) -> str:
        pretty = col.title()
        return FEATURE_REMAPS.get(pretty, pretty)

    # ---- behavioral features -------------------------------------------
    print(f"\n--- Behavioral features (top/bottom {args.top_k}, "
          f"{args.n_resamples} tie-broken draws) ---")
    beh_results, beh_labels = {}, {}
    for col in feature_cols:
        key = f'beh::{col}'
        beh_results[key] = contrastive_from_scores(
            ann[col].values, corpus, key, top_k=args.top_k,
            n_resamples=args.n_resamples, seed=args.seed)
        beh_labels[key] = feat_label(col)
        df = beh_results[key]
        top = top_words(df, args.top_n)
        mass = pos_mass(top)
        flag = ' [binary]' if col in BINARY_FEATURES else ''
        print(f"  {feat_label(col):22s}{flag:9s} ties hi/lo="
              f"{df.attrs['n_tied_high']:3d}/{df.attrs['n_tied_low']:3d}  "
              f"N {mass['Noun']:5.1f}%  V {mass['Verb']:5.1f}%  "
              f"A {mass['Adjective']:5.1f}%  Adv {mass['Adverb']:4.1f}%")
        print(f"    {', '.join(f'{r.word}({r.pos[:1]})' for _, r in top.iterrows())}")

    # ---- embedding dimensions (same estimator) --------------------------
    print(f"\n--- Embedding dimensions (same estimator) ---")
    emb_results, emb_labels = {}, {}
    for d in BEST_DIMS:
        key = f'dim::{d}'
        emb_results[key] = contrastive_from_scores(
            emb[:, d], corpus, key, top_k=args.top_k,
            n_resamples=args.n_resamples, seed=args.seed)
        emb_labels[key] = f"Dim {d}: {DIM_LABELS.get(d, '')}".strip(': ')
        df = emb_results[key]
        top = top_words(df, args.top_n)
        mass = pos_mass(top)
        print(f"  {emb_labels[key]:26s} ties hi/lo="
              f"{df.attrs['n_tied_high']:3d}/{df.attrs['n_tied_low']:3d}  "
              f"N {mass['Noun']:5.1f}%  V {mass['Verb']:5.1f}%  "
              f"A {mass['Adjective']:5.1f}%  Adv {mass['Adverb']:4.1f}%")

    all_results = {**beh_results, **emb_results}
    all_labels = {**beh_labels, **emb_labels}
    beh_order = list(beh_results)
    emb_order = list(emb_results)

    os.makedirs(RESULTS_DIMENSIONS_DIR, exist_ok=True)

    # ---- per-word stats -------------------------------------------------
    long = pd.concat([all_results[g] for g in beh_order + emb_order],
                     ignore_index=True)
    long['kind'] = np.where(long['group_id'].str.startswith('beh::'),
                            'behavioral', 'embedding')
    long['label'] = long['group_id'].map(all_labels)
    p = os.path.join(RESULTS_DIMENSIONS_DIR, 'behavioral_contrastive_word_stats.csv')
    long.to_csv(p, index=False)
    print(f"\n  Saved {p} ({len(long)} rows)")

    # ---- top-10 table ---------------------------------------------------
    tbl = build_top10_table(all_results, beh_order + emb_order, all_labels,
                            top_n=args.top_n)
    tbl.insert(1, 'kind', ['behavioral'] * len(beh_order) + ['embedding'] * len(emb_order))
    p = os.path.join(RESULTS_DIMENSIONS_DIR, 'table_behavioral_contrastive_top10.csv')
    tbl.to_csv(p, index=False)
    print(f"  Saved {p}")
    with open(p.replace('.csv', '.tex'), 'w') as f:
        f.write(tbl.to_latex(index=False, escape=True))

    # ---- POS composition comparison ------------------------------------
    comp_rows = []
    for kind, order in (('behavioral', beh_order), ('embedding', emb_order)):
        for gid in order:
            top = top_words(all_results[gid], args.top_n)
            if top.empty:
                continue
            mass = pos_mass(top)
            comp_rows.append({
                'kind': kind, 'group_id': gid, 'label': all_labels[gid],
                **{f'{p_}%': mass[p_] for p_ in POS_ORDER},
                'n_adjectives': int((top['pos'] == 'Adjective').sum()),
                'js_bits': round(all_results[gid].attrs.get('js_bits', np.nan), 3),
            })
    comp = pd.DataFrame(comp_rows)
    p = os.path.join(RESULTS_DIMENSIONS_DIR, 'pos_composition_comparison.csv')
    comp.to_csv(p, index=False)
    print(f"  Saved {p}")

    # ---- summary --------------------------------------------------------
    print("\n" + "=" * 60)
    print("POS COMPOSITION: behavioral ratings vs embedding dimensions")
    print("=" * 60)
    for kind in ('behavioral', 'embedding'):
        sub = comp[comp['kind'] == kind]
        if sub.empty:
            continue
        nv = sub['Noun%'] + sub['Verb%']
        desc = sub['Adjective%'] + sub['Adverb%']
        print(f"\n{kind} (n={len(sub)} groups)")
        print(f"  noun+verb   mean {nv.mean():5.1f}%  "
              f"[{nv.min():.1f}, {nv.max():.1f}]")
        print(f"  noun        mean {sub['Noun%'].mean():5.1f}%")
        print(f"  verb        mean {sub['Verb%'].mean():5.1f}%")
        print(f"  adjective   mean {sub['Adjective%'].mean():5.1f}%")
        print(f"  adverb      mean {sub['Adverb%'].mean():5.1f}%")
        print(f"  descriptive mean {desc.mean():5.1f}%  "
              f"[{desc.min():.1f}, {desc.max():.1f}]")
        print(f"  groups with zero adjectives among top {args.top_n}: "
              f"{(sub['n_adjectives'] == 0).sum()}/{len(sub)}")
        print(f"  JS (bits)   mean {sub['js_bits'].mean():.3f}")

    b = comp[comp['kind'] == 'behavioral']
    e = comp[comp['kind'] == 'embedding']
    if not b.empty and not e.empty:
        from scipy.stats import mannwhitneyu
        print("\nbehavioral vs embedding (Mann-Whitney, "
              f"n={len(b)} vs {len(e)} groups):")
        tests = [
            ('descriptive %  (adj+adv)', b['Adjective%'] + b['Adverb%'],
             e['Adjective%'] + e['Adverb%']),
            ('noun %', b['Noun%'], e['Noun%']),
            ('verb %', b['Verb%'], e['Verb%']),
            ('noun - verb %', b['Noun%'] - b['Verb%'], e['Noun%'] - e['Verb%']),
            ('JS divergence (bits)', b['js_bits'], e['js_bits']),
        ]
        for name, bv, ev in tests:
            u, pv = mannwhitneyu(bv.values, ev.values, alternative='two-sided')
            print(f"  {name:26s} {bv.mean():6.2f} vs {ev.mean():6.2f}   "
                  f"U={u:5.1f}  p={pv:.4f}")

    # ---- figures --------------------------------------------------------
    print("\n--- Figures ---")
    plot_pos_breakdown(
        beh_results, beh_order, beh_labels, top_n=args.top_n,
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR,
                               'behavioral_contrastive_pos_breakdown.png'))
    plot_pos_comparison(
        comp, save_path=os.path.join(RESULTS_DIMENSIONS_DIR,
                                     'pos_composition_comparison.png'))

    print("\nDONE")


if __name__ == "__main__":
    main()
