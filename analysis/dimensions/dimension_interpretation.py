"""
Dimension interpretation via caption-based token analysis and word clouds.
"""

import os
import sys
import ast
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from collections import Counter
from typing import Optional, List, Tuple, Iterable, Dict

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

try:
    import nltk
    from nltk import pos_tag as nltk_pos_tag
    from nltk.tokenize import word_tokenize as nltk_word_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    UTILS_DIR, MODEL_DIR, RESULTS_DIMENSIONS_DIR, BEST_DIMS, BEHAVIOR_COLS, FEATURE_REMAPS,
    RANDOM_SEED, DIM_LABELS,
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

os.makedirs(RESULTS_DIMENSIONS_DIR, exist_ok=True)

# Tie-broken draws for the contrastive word statistics behind the paper's
# caption figures (Fig. 3, ED Fig. 3) and ED Table 1. Same draws and seed as
# behavioral_contrastive_captions.py, so the two sides use one estimator.
CONTRASTIVE_N_RESAMPLES = 200

# ============================================================================
# STOPWORDS
# ============================================================================

ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn',
    'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
    'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
    'also', 'could', 'would', 'may', 'might', 'shall', 'must',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'second', 'third',
}

_TOKEN_RE = re.compile(r"[a-z]{2,}")

# Lemma map: merge plurals, verb forms, and synonyms into canonical tokens.
# Keeps the most common/readable form as the canonical key.
_LEMMA_MAP = {
    # plurals
    'babies': 'baby', 'children': 'child', 'dogs': 'dog', 'cats': 'cat',
    'men': 'man', 'women': 'woman', 'boys': 'boy', 'girls': 'girl',
    'kids': 'kid', 'persons': 'person', 'people': 'person',
    'hands': 'hand', 'arms': 'arm', 'legs': 'leg', 'eyes': 'eye',
    'faces': 'face', 'heads': 'head', 'feet': 'foot',
    'cars': 'car', 'trucks': 'truck', 'drums': 'drum',
    'trees': 'tree', 'rooms': 'room', 'tables': 'table',
    'chairs': 'chair', 'animals': 'animal', 'toys': 'toy',
    # verb forms
    'playing': 'play', 'played': 'play', 'plays': 'play',
    'talking': 'talk', 'talked': 'talk', 'talks': 'talk',
    'walking': 'walk', 'walked': 'walk', 'walks': 'walk',
    'running': 'run', 'runs': 'run',
    'sitting': 'sit', 'sits': 'sit',
    'standing': 'stand', 'stands': 'stand',
    'holding': 'hold', 'holds': 'hold', 'held': 'hold',
    'eating': 'eat', 'eats': 'eat',
    'cooking': 'cook', 'cooks': 'cook',
    'singing': 'sing', 'sings': 'sing',
    'dancing': 'dance', 'dances': 'dance', 'danced': 'dance',
    'crying': 'cry', 'cries': 'cry', 'cried': 'cry',
    'laughing': 'laugh', 'laughs': 'laugh',
    'smiling': 'smile', 'smiles': 'smile',
    'moving': 'move', 'moves': 'move', 'moved': 'move',
    'lying': 'lie', 'lies': 'lie', 'laying': 'lie',
    'feeding': 'feed', 'feeds': 'feed',
    'kicking': 'kick', 'kicks': 'kick', 'kicked': 'kick',
    'hitting': 'hit', 'hits': 'hit',
    'fishing': 'fish',
    'brushing': 'brush',
    'using': 'use', 'uses': 'use', 'used': 'use',
    'looking': 'look', 'looks': 'look', 'looked': 'look',
    'throwing': 'throw', 'throws': 'throw', 'threw': 'throw',
    'catching': 'catch', 'catches': 'catch', 'caught': 'catch',
    'reading': 'read', 'reads': 'read',
    'swimming': 'swim', 'swims': 'swim',
    'climbing': 'climb', 'climbs': 'climb',
    'jumping': 'jump', 'jumps': 'jump', 'jumped': 'jump',
    'drinking': 'drink', 'drinks': 'drink',
    'riding': 'ride', 'rides': 'ride',
    'driving': 'drive', 'drives': 'drive',
    'hugging': 'hug', 'hugs': 'hug',
    'kissing': 'kiss', 'kisses': 'kiss',
    'sleeping': 'sleep', 'sleeps': 'sleep',
    'fighting': 'fight', 'fights': 'fight',
    'crawling': 'crawl', 'crawls': 'crawl',
    'clapping': 'clap', 'claps': 'clap',
}

# Domain-specific POS overrides for video caption context.
# NLTK often mistaggs common action verbs as nouns in short captions.
# Applied after lemmatization to ensure consistent POS per lemma.
_POS_OVERRIDES = {
    'talk': 'Verb', 'feed': 'Verb', 'brush': 'Verb', 'kick': 'Verb',
    'sing': 'Verb', 'cry': 'Verb', 'play': 'Verb', 'cook': 'Verb',
    'eat': 'Verb', 'laugh': 'Verb', 'lie': 'Verb', 'sit': 'Verb',
    'stand': 'Verb', 'read': 'Verb', 'walk': 'Verb', 'run': 'Verb',
    'throw': 'Verb', 'catch': 'Verb', 'swim': 'Verb', 'dance': 'Verb',
    'climb': 'Verb', 'jump': 'Verb', 'fight': 'Verb', 'hug': 'Verb',
    'kiss': 'Verb', 'sleep': 'Verb', 'drink': 'Verb', 'drive': 'Verb',
    'ride': 'Verb', 'hold': 'Verb', 'move': 'Verb', 'smile': 'Verb',
    'hit': 'Verb', 'look': 'Verb', 'use': 'Verb', 'crawl': 'Verb',
    'clap': 'Verb', 'fish': 'Verb',
}


def tokenize(text: str, extra_stopwords=None) -> list:
    """Simple lowercase tokenizer with stopword removal and lemmatization."""
    stops = ENGLISH_STOPWORDS | (set(extra_stopwords) if extra_stopwords else set())
    tokens = _TOKEN_RE.findall(text.lower())
    return [_LEMMA_MAP.get(w, w) for w in tokens if w not in stops]


# Penn Treebank tag -> coarse POS group
_PTB_TO_GROUP = {}
for _prefix, _group in [('NN', 'Noun'), ('VB', 'Verb'), ('JJ', 'Adjective'), ('RB', 'Adverb')]:
    for _suffix in ('', 'S', 'P', 'Z', 'G', 'D', 'N', 'R', 'T', 'ST'):
        _PTB_TO_GROUP[_prefix + _suffix] = _group


def tokenize_with_pos(text: str, extra_stopwords=None) -> List[Tuple[str, str]]:
    """Tokenize with POS tags using nltk on full sentence for accurate context.

    Returns list of (lemma, pos_group) tuples where pos_group is one of:
    Noun, Verb, Adjective, Adverb, Other.
    """
    if not HAS_NLTK:
        return [(w, 'Other') for w in tokenize(text, extra_stopwords)]

    stops = ENGLISH_STOPWORDS | (set(extra_stopwords) if extra_stopwords else set())

    try:
        words = nltk_word_tokenize(text.lower())
    except LookupError:
        words = text.lower().split()

    try:
        tagged = nltk_pos_tag(words)
    except LookupError:
        tagged = [(w, 'NN') for w in words]

    result = []
    for word, tag in tagged:
        if not _TOKEN_RE.match(word):
            continue
        if word in stops:
            continue
        lemma = _LEMMA_MAP.get(word, word)
        pos_group = _POS_OVERRIDES.get(lemma, _PTB_TO_GROUP.get(tag, 'Other'))
        result.append((lemma, pos_group))
    return result


# ============================================================================
# CAPTION CORPUS
# ============================================================================

class CaptionCorpus:
    """Tokenized caption corpus indexed by video ID."""

    def __init__(self, caption_rows: pd.DataFrame, video_df: pd.DataFrame,
                 global_token_counts: Counter, global_n_tokens: int):
        self.caption_rows = caption_rows
        self.video_df = video_df
        self.global_token_counts = global_token_counts
        self.global_n_tokens = global_n_tokens

    @classmethod
    def from_caption_df(cls, caps_df: pd.DataFrame, extra_stopwords=None):
        """Build from DataFrame with columns: vid_id, captions, (optional) image_path."""
        rows = caps_df.copy()
        rows['tokens'] = rows['captions'].fillna('').apply(
            lambda x: tokenize(x, extra_stopwords)
        )
        rows['token_set'] = rows['tokens'].apply(set)

        # Video-level aggregation
        video_groups = rows.groupby('vid_id')
        video_records = []
        for vid, grp in video_groups:
            all_tokens = []
            for toks in grp['tokens']:
                all_tokens.extend(toks)
            token_set = set(all_tokens)
            img_path = grp['image_path'].iloc[0] if 'image_path' in grp.columns else ''
            video_records.append({
                'vid_id': vid,
                'video_tokens': all_tokens,
                'video_token_set': token_set,
                'image_path': img_path,
            })

        video_df = pd.DataFrame(video_records).set_index('vid_id')

        # Global stats
        global_counts = Counter()
        for toks in rows['tokens']:
            global_counts.update(toks)
        global_n = sum(global_counts.values())

        return cls(rows, video_df, global_counts, global_n)

    @classmethod
    def from_caption_df_with_pos(cls, caps_df: pd.DataFrame, extra_stopwords=None):
        """Build corpus with POS-tagged tokens alongside plain tokens.

        Stores ``tokens_with_pos`` (list of (word, pos) tuples) per caption row
        and ``video_tokens_with_pos`` per video, while keeping the existing
        ``tokens`` field unchanged for backward compatibility.
        """
        rows = caps_df.copy()
        rows['tokens'] = rows['captions'].fillna('').apply(
            lambda x: tokenize(x, extra_stopwords)
        )
        rows['token_set'] = rows['tokens'].apply(set)
        rows['tokens_with_pos'] = rows['captions'].fillna('').apply(
            lambda x: tokenize_with_pos(x, extra_stopwords)
        )

        video_groups = rows.groupby('vid_id')
        video_records = []
        for vid, grp in video_groups:
            all_tokens = []
            all_tokens_pos = []
            for toks in grp['tokens']:
                all_tokens.extend(toks)
            for toks_pos in grp['tokens_with_pos']:
                all_tokens_pos.extend(toks_pos)
            token_set = set(all_tokens)
            img_path = grp['image_path'].iloc[0] if 'image_path' in grp.columns else ''
            video_records.append({
                'vid_id': vid,
                'video_tokens': all_tokens,
                'video_token_set': token_set,
                'video_tokens_with_pos': all_tokens_pos,
                'image_path': img_path,
            })

        video_df = pd.DataFrame(video_records).set_index('vid_id')

        global_counts = Counter()
        for toks in rows['tokens']:
            global_counts.update(toks)
        global_n = sum(global_counts.values())

        obj = cls(rows, video_df, global_counts, global_n)
        obj.has_pos = True
        return obj


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def _entropy_from_counts(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            ent -= p * np.log(p)
    return ent


def _js_divergence_from_counters(c1: Counter, c2: Counter) -> float:
    eps = 1e-12
    t1, t2 = sum(c1.values()), sum(c2.values())
    if t1 == 0 or t2 == 0:
        return 0.0
    vocab = set(c1.keys()) | set(c2.keys())
    js = 0.0
    for w in vocab:
        p = c1.get(w, 0) / t1
        q = c2.get(w, 0) / t2
        m = 0.5 * (p + q)
        if p > 0:
            js += 0.5 * p * np.log((p + eps) / (m + eps))
        if q > 0:
            js += 0.5 * q * np.log((q + eps) / (m + eps))
    return js


def _top_mass_from_counts(counter: Counter, k: int) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return sum(c for _, c in counter.most_common(k)) / total


def analyze_dimension_captions(
    embeddings: np.ndarray,
    corpus: CaptionCorpus,
    dim_id: int,
    top_k: int = 8,
    side: str = "top",
    top_n_words: int = 30,
    eps: float = 1e-12,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute per-word quantitative stats for a single dimension.

    Returns:
        word_stats_df: DataFrame with word-level statistics
        summary: dict with dimension-level summary metrics
    """
    n_videos, D = embeddings.shape
    acts = embeddings[:, dim_id]

    if side == "top":
        sel_vids = np.argsort(-acts)[:top_k]
    else:
        sel_vids = np.argsort(acts)[:top_k]

    sel_vids = [int(v) for v in sel_vids if int(v) in corpus.video_df.index]
    if len(sel_vids) == 0:
        return pd.DataFrame(), {"dim_id": dim_id, "side": side, "n_selected_videos": 0}

    sel_caps = corpus.caption_rows[corpus.caption_rows['vid_id'].isin(sel_vids)]
    sel_video_df = corpus.video_df.loc[sel_vids]

    # Token counts
    sel_token_counts = Counter()
    for toks in sel_caps['tokens']:
        sel_token_counts.update(toks)
    n_tokens = sum(sel_token_counts.values())

    # Caption coverage
    sel_caption_counts = Counter()
    for s in sel_caps['token_set']:
        sel_caption_counts.update(s)
    n_captions = len(sel_caps)

    # Video coverage
    sel_video_counts = Counter()
    for s in sel_video_df['video_token_set']:
        sel_video_counts.update(s)
    n_selected_videos = len(sel_vids)

    # Per-word stats
    rows = []
    for w, c in sel_token_counts.most_common(top_n_words):
        pct_tokens = c / (n_tokens + eps)
        pct_captions = sel_caption_counts.get(w, 0) / (n_captions + eps)
        pct_videos = sel_video_counts.get(w, 0) / (n_selected_videos + eps)
        global_pct = corpus.global_token_counts.get(w, 0) / (corpus.global_n_tokens + eps)
        lift = (pct_tokens + eps) / (global_pct + eps)

        rows.append({
            "dim_id": dim_id, "side": side, "word": w,
            "count": c, "pct_tokens": pct_tokens,
            "n_captions_with_word": sel_caption_counts.get(w, 0),
            "pct_captions": pct_captions,
            "n_videos_with_word": sel_video_counts.get(w, 0),
            "pct_videos": pct_videos,
            "global_pct_tokens": global_pct,
            "lift_vs_global": lift,
        })

    word_stats_df = pd.DataFrame(rows)
    if not word_stats_df.empty:
        word_stats_df["rank"] = np.arange(1, len(word_stats_df) + 1)

    ent = _entropy_from_counts(sel_token_counts)
    js = _js_divergence_from_counters(sel_token_counts, corpus.global_token_counts)

    summary = {
        "dim_id": dim_id, "side": side,
        "n_selected_videos": n_selected_videos,
        "n_selected_captions": n_captions,
        "n_tokens": n_tokens,
        "vocab_size": len(sel_token_counts),
        "entropy_nats": ent,
        "top5_mass": _top_mass_from_counts(sel_token_counts, 5),
        "js_to_global_nats": js,
    }

    return word_stats_df, summary


def analyze_dimensions(
    embeddings: np.ndarray,
    corpus: CaptionCorpus,
    dims: Iterable[int],
    top_k: int = 8,
    sides: Tuple[str, ...] = ("top",),
    top_n_words: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run analysis over multiple dimensions."""
    all_words, all_summaries = [], []

    for d in dims:
        for side in sides:
            ws, summ = analyze_dimension_captions(
                embeddings, corpus, int(d), top_k, side, top_n_words
            )
            if not ws.empty:
                all_words.append(ws)
            all_summaries.append(summ)

    word_stats = pd.concat(all_words, ignore_index=True) if all_words else pd.DataFrame()
    dim_summary = pd.DataFrame(all_summaries)
    return word_stats, dim_summary


def compute_word_overlap_across_dims(
    word_stats: pd.DataFrame,
    rank_cutoff: int = 10,
    side: str = "top",
) -> pd.DataFrame:
    """How shared are top words across dimensions?"""
    if word_stats is None or word_stats.empty:
        return pd.DataFrame(columns=["word", "n_dims", "dims"])

    sub = word_stats[(word_stats["side"] == side) & (word_stats["rank"] <= rank_cutoff)]
    g = sub.groupby("word")["dim_id"].agg(
        [pd.Series.nunique, lambda x: sorted(set(x))]
    ).reset_index()
    g.columns = ["word", "n_dims", "dims"]
    return g.sort_values(["n_dims", "word"], ascending=[False, True]).reset_index(drop=True)


def pretty_top_words_table(word_stats: pd.DataFrame, dim_id: int,
                           side: str = "top", n: int = 15) -> pd.DataFrame:
    """Show top words with interpretable percentages."""
    sub = word_stats[(word_stats["dim_id"] == dim_id) & (word_stats["side"] == side)].copy()
    sub = sub.sort_values("rank").head(n)
    if sub.empty:
        return sub
    out = sub[["rank", "word", "count", "pct_tokens", "pct_captions",
               "pct_videos", "lift_vs_global"]].copy()
    out["pct_tokens"] = (100 * out["pct_tokens"]).round(2)
    out["pct_captions"] = (100 * out["pct_captions"]).round(2)
    out["pct_videos"] = (100 * out["pct_videos"]).round(2)
    out["lift_vs_global"] = out["lift_vs_global"].round(2)
    out = out.rename(columns={
        "pct_tokens": "% tokens", "pct_captions": "% captions",
        "pct_videos": "% videos", "lift_vs_global": "lift vs global",
    })
    return out


# ============================================================================
# WORD CLOUD
# ============================================================================

def get_wordcloud_image(freqs: Counter, max_words: int = 30,
                        width: int = 400, height: int = 200):
    """Generate word cloud image from frequency counter."""
    if not HAS_WORDCLOUD or not freqs:
        # Return blank image
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    wc = WordCloud(
        width=width, height=height,
        background_color='white',
        max_words=max_words,
        colormap='viridis',
    )
    wc.generate_from_frequencies(dict(freqs.most_common(max_words)))
    return wc.to_array()


# ============================================================================
# HEBART-STYLE GRID VISUALIZATION
# ============================================================================

def plot_hebart_style_grid(
    embeddings: np.ndarray,
    corpus: CaptionCorpus,
    top_k: int = 8,
    dims_to_plot: Optional[List[int]] = None,
    figsize=(16, 12),
    max_words: int = 10,
    batch_size: int = 10,
    side: str = "top",
    contrastive_results: Optional[List[pd.DataFrame]] = None,
    save_path: Optional[str] = None,
):
    """Hebart-style grid with word clouds and top-activating video images."""
    if not HAS_WORDCLOUD:
        print("WARNING: wordcloud not installed, skipping grid.")
        return

    n_videos, D = embeddings.shape

    if dims_to_plot is None:
        dims_to_plot = list(range(D))
    else:
        dims_to_plot = [int(d) for d in dims_to_plot if 0 <= int(d) < D]

    # Build contrastive lookup if available
    contrast_by_dim = {}
    if contrastive_results:
        for df in contrastive_results:
            if not df.empty:
                contrast_by_dim[df['dim_id'].iloc[0]] = df

    for start in range(0, len(dims_to_plot), batch_size):
        batch = dims_to_plot[start:start + batch_size]
        n_rows = len(batch)
        n_cols = top_k + 3  # [label] [spacer] [wordcloud] [top_k images]

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.02, hspace=0.02)
        gs.set_width_ratios([0.4, 0.08, 2] + [1] * top_k)

        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                            wspace=0.02, hspace=0.02)

        for r, dim_id in enumerate(batch):
            acts = embeddings[:, dim_id]
            if side == "top":
                top_idx = np.argsort(-acts)[:top_k]
            else:
                top_idx = np.argsort(acts)[:top_k]

            # Build word cloud from contrastive results (log_ratio, POS colors)
            cdf = contrast_by_dim.get(dim_id, pd.DataFrame())
            if not cdf.empty:
                top_words = cdf[cdf['log_ratio'] > 0].head(max_words)
                freqs = {row['word']: row['log_ratio']
                         for _, row in top_words.iterrows()
                         if row['log_ratio'] > 0}

                if freqs:
                    pos_lookup = {row['word']: row['pos']
                                  for _, row in top_words.iterrows()}

                    def _make_color_func(lookup):
                        def color_func(word, **kwargs):
                            pos = lookup.get(word, 'Other')
                            return POS_COLORS.get(pos, '#BBBBBB')
                        return color_func

                    wc = WordCloud(
                        width=400, height=200,
                        background_color='white',
                        max_words=max_words,
                        color_func=_make_color_func(pos_lookup),
                        prefer_horizontal=0.7,
                    )
                    wc.generate_from_frequencies(freqs)
                    wc_img = wc.to_array()
                else:
                    wc_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
            else:
                wc_img = np.ones((200, 400, 3), dtype=np.uint8) * 255

            # Label (col 0)
            ax_lbl = fig.add_subplot(gs[r, 0])
            ax_lbl.text(0.5, 0.5, f"Dim {dim_id}", ha='center', va='center', fontsize=11)
            ax_lbl.axis('off')

            # Spacer (col 1)
            ax_sp = fig.add_subplot(gs[r, 1])
            ax_sp.axis('off')

            # Word cloud (col 2)
            ax_wc = fig.add_subplot(gs[r, 2])
            ax_wc.imshow(wc_img)
            ax_wc.axis('off')

            # Top-k images (cols 3+)
            for c, vid in enumerate(top_idx):
                ax = fig.add_subplot(gs[r, c + 3])
                vid = int(vid)
                if vid in corpus.video_df.index:
                    img_path = corpus.video_df.loc[vid, 'image_path']
                    try:
                        img = mpimg.imread(img_path)
                        ax.imshow(img)
                    except Exception:
                        ax.text(0.5, 0.5, f"Missing\n{vid}",
                                ha='center', va='center', fontsize=8)
                else:
                    ax.text(0.5, 0.5, f"No row\n{vid}",
                            ha='center', va='center', fontsize=8)
                ax.axis('off')

        if save_path:
            batch_path = save_path.replace('.png', f'_batch{start}.png')
            fig.savefig(batch_path, dpi=300, bbox_inches='tight')
            fig.savefig(batch_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)


# ============================================================================
# INTERPRETABILITY TABLE
# ============================================================================

def build_interpretability_table(
    word_stats: pd.DataFrame,
    dim_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a formatted interpretability table with top 3 words per side,
    token mass, and JS divergence for each dimension.
    """
    # Top 3 words with %token share per dimension
    top3 = (
        word_stats.sort_values(["dim_id", "rank"])
        .query("rank <= 3")
        .assign(word_pct=lambda d: d.apply(
            lambda r: f"{r['word']} ({100*r['pct_tokens']:.1f}%)", axis=1))
        .groupby(["dim_id", "side"])["word_pct"]
        .apply(lambda x: ", ".join(x))
        .reset_index(name="top3_words")
    )

    # Build main table
    main = (
        dim_summary.merge(top3, on=["dim_id", "side"], how="left")
        .assign(
            top5_mass_pct=lambda d: 100 * d["top5_mass"],
            js_bits=lambda d: d["js_to_global_nats"] / np.log(2),
        )
        [["dim_id", "side", "top3_words", "top5_mass_pct",
          "js_bits", "n_selected_videos", "n_selected_captions", "n_tokens"]]
    )

    # Nice formatting
    main["top5_mass_pct"] = main["top5_mass_pct"].map(lambda x: f"{x:.1f}%")
    main["js_bits"] = main["js_bits"].map(lambda x: f"{x:.3f}")

    return main


# ============================================================================
# CONTRASTIVE ANALYSIS
# ============================================================================

def _pool_pos_counts(video_ids: List[int], corpus: CaptionCorpus) -> Counter:
    """Pool (word, pos) tuple counts from video-level POS tokens."""
    counts = Counter()
    for vid in video_ids:
        if vid in corpus.video_df.index:
            toks_pos = corpus.video_df.loc[vid, 'video_tokens_with_pos']
            counts.update(toks_pos)
    return counts


def _pool_word_counts(video_ids: List[int], corpus: CaptionCorpus) -> Counter:
    """Pool plain word counts from video-level tokens."""
    counts = Counter()
    for vid in video_ids:
        if vid in corpus.video_df.index:
            counts.update(corpus.video_df.loc[vid, 'video_tokens'])
    return counts


def analyze_dimension_contrastive(
    embeddings: np.ndarray,
    corpus: CaptionCorpus,
    dim_id: int,
    top_k: int = 8,
    alpha: float = 0.5,
    n_resamples: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compute contrastive log-ratio scores between top-k and bottom-k videos.

    Returns DataFrame with columns:
        word, pos, count_high, count_low, log_ratio, weighted_contrast,
        pct_high, pct_low

    n_resamples=None is the original single split: np.argsort picks the top-k
    and bottom-k, which is an arbitrary subset whenever scores tie at the cut
    (74-107 videos sit at exactly 0 on every dimension). An integer delegates to
    behavioral_contrastive_captions.contrastive_from_scores, which averages the
    same statistic over that many random tie-breaks and adds signed_contrast /
    sign_consistency; rows are then sorted by signed_contrast. Requires a POS
    corpus. The paper figures and tables use CONTRASTIVE_N_RESAMPLES.
    """
    if n_resamples:
        # Lazy import: behavioral_contrastive_captions imports from this module
        from analysis.dimensions.behavioral_contrastive_captions import contrastive_from_scores
        df = contrastive_from_scores(embeddings[:, dim_id], corpus, dim_id,
                                     top_k=top_k, alpha=alpha,
                                     n_resamples=n_resamples, seed=seed)
        df = df.rename(columns={'group_id': 'dim_id'})
        # Plain column order for CSVs; the attrs (js_bits, n_draws) are kept
        return df
    acts = embeddings[:, dim_id]
    order = np.argsort(-acts)
    top_vids = [int(v) for v in order[:top_k] if int(v) in corpus.video_df.index]
    bottom_vids = [int(v) for v in order[-top_k:] if int(v) in corpus.video_df.index]

    has_pos = getattr(corpus, 'has_pos', False)

    if has_pos:
        top_counts = _pool_pos_counts(top_vids, corpus)
        bottom_counts = _pool_pos_counts(bottom_vids, corpus)
        vocab = set(top_counts.keys()) | set(bottom_counts.keys())
    else:
        top_word_counts = _pool_word_counts(top_vids, corpus)
        bottom_word_counts = _pool_word_counts(bottom_vids, corpus)
        vocab_words = set(top_word_counts.keys()) | set(bottom_word_counts.keys())
        # Wrap as (word, 'Other') tuples for uniform processing
        top_counts = Counter({(w, 'Other'): c for w, c in top_word_counts.items()})
        bottom_counts = Counter({(w, 'Other'): c for w, c in bottom_word_counts.items()})
        vocab = set(top_counts.keys()) | set(bottom_counts.keys())

    n_top = sum(top_counts.values()) or 1
    n_bottom = sum(bottom_counts.values()) or 1

    rows = []
    for key in vocab:
        word, pos = key
        ct = top_counts.get(key, 0)
        cb = bottom_counts.get(key, 0)
        total = ct + cb
        log_ratio = np.log((ct + alpha) / (cb + alpha))
        weighted_contrast = abs(log_ratio) * np.log(total + 1)
        rows.append({
            'dim_id': dim_id,
            'word': word,
            'pos': pos,
            'count_high': ct,
            'count_low': cb,
            'log_ratio': log_ratio,
            'weighted_contrast': weighted_contrast,
            'pct_high': ct / n_top,
            'pct_low': cb / n_bottom,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('weighted_contrast', ascending=False).reset_index(drop=True)
    return df


def cloud_top_words(cdf: pd.DataFrame, max_words: int, top_n: int) -> pd.DataFrame:
    """The top_n words that render largest in a contrastive word cloud.

    The cloud draws the first ``max_words`` high-side words in ranking order
    and sizes each by log_ratio, so its biggest words are the top_n by
    log_ratio among those -- not the first top_n in ranking order. Mirrors
    ``_contrastive_wordcloud_image`` exactly: a word listed under two POS tags
    is one dict key (first position, last row's values), and WordCloud's
    stable descending sort breaks ties by that key order.
    """
    shown = cdf[cdf['log_ratio'] > 0].head(max_words)
    rows = {}
    for _, r in shown.iterrows():
        rows[r['word']] = r
    ranked = sorted(rows.values(), key=lambda r: r['log_ratio'], reverse=True)
    return pd.DataFrame(ranked[:top_n]).reset_index(drop=True)


def _contrastive_wordcloud_image(cdf, max_words, random_state=None):
    """Word cloud of a dimension's high-side words, sized by log-ratio, coloured by POS."""
    blank = np.ones((200, 400, 3), dtype=np.uint8) * 255
    if cdf.empty:
        return blank
    top_words = cdf[cdf['log_ratio'] > 0].head(max_words)
    freqs = {row['word']: row['log_ratio']
             for _, row in top_words.iterrows()
             if row['log_ratio'] > 0}
    if not freqs:
        return blank
    pos_lookup = {row['word']: row['pos'] for _, row in top_words.iterrows()}

    def color_func(word, **kwargs):
        return POS_COLORS.get(pos_lookup.get(word, 'Other'), '#BBBBBB')

    wc = WordCloud(
        width=400, height=200,
        background_color='white',
        max_words=max_words,
        color_func=color_func,
        prefer_horizontal=0.7,
        random_state=random_state,
    )
    wc.generate_from_frequencies(freqs)
    return wc.to_array()


def _draw_contrastive_row(fig, gs, r, col0, dim_id, embeddings, corpus, cdf,
                          top_k, max_words, random_state=None, label_fontsize=11):
    """One grid row: [label] [spacer] [word cloud] [top_k images], from column col0."""
    acts = embeddings[:, dim_id]
    top_idx = np.argsort(-acts)[:top_k]
    wc_img = _contrastive_wordcloud_image(cdf, max_words, random_state)

    # Label
    ax_lbl = fig.add_subplot(gs[r, col0])
    ax_lbl.text(0.5, 0.5, f"Dim {dim_id}", ha='center', va='center',
                fontsize=label_fontsize)
    ax_lbl.axis('off')

    # Spacer
    ax_sp = fig.add_subplot(gs[r, col0 + 1])
    ax_sp.axis('off')

    # Word cloud
    ax_wc = fig.add_subplot(gs[r, col0 + 2])
    ax_wc.imshow(wc_img)
    ax_wc.axis('off')

    # Top-k images
    for c, vid in enumerate(top_idx):
        ax = fig.add_subplot(gs[r, col0 + 3 + c])
        vid = int(vid)
        if vid in corpus.video_df.index:
            img_path = corpus.video_df.loc[vid, 'image_path']
            try:
                img = mpimg.imread(img_path)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, f"Missing\n{vid}",
                        ha='center', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, f"No row\n{vid}",
                    ha='center', va='center', fontsize=8)
        ax.axis('off')


def _contrast_lookup(contrastive_results):
    """dim_id -> contrastive DataFrame."""
    contrast_by_dim = {}
    for df in contrastive_results:
        if df.empty:
            continue
        contrast_by_dim[df['dim_id'].iloc[0]] = df
    return contrast_by_dim


def plot_contrastive_grid(
    embeddings: np.ndarray,
    corpus: CaptionCorpus,
    contrastive_results: List[pd.DataFrame],
    dims_to_plot: List[int],
    top_k: int = 8,
    max_words: int = 25,
    batch_size: int = 10,
    figsize=(16, 14),
    save_path: Optional[str] = None,
    random_state: Optional[int] = None,
):
    """Hebart-style grid with dual-color contrastive word clouds.

    Red/warm = more frequent in high-activating videos (positive log-ratio).
    Blue/cool = more frequent in low-activating videos (negative log-ratio).
    """
    if not HAS_WORDCLOUD:
        print("WARNING: wordcloud not installed, skipping contrastive grid.")
        return

    dims_to_plot = [int(d) for d in dims_to_plot]
    contrast_by_dim = _contrast_lookup(contrastive_results)

    for start in range(0, len(dims_to_plot), batch_size):
        batch = dims_to_plot[start:start + batch_size]
        n_rows = len(batch)
        n_cols = top_k + 3  # [label] [spacer] [wordcloud] [top_k images]

        scaled_height = figsize[1] * n_rows / batch_size
        fig = plt.figure(figsize=(figsize[0], scaled_height))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.02, hspace=0.005)
        gs.set_width_ratios([0.4, 0.08, 2] + [1] * top_k)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005,
                            wspace=0.02, hspace=0.005)

        for r, dim_id in enumerate(batch):
            _draw_contrastive_row(fig, gs, r, 0, dim_id, embeddings, corpus,
                                  contrast_by_dim.get(dim_id, pd.DataFrame()),
                                  top_k, max_words, random_state=random_state)

        if save_path:
            batch_idx = start // batch_size
            batch_path = save_path.replace('.png', f'_batch{batch_idx}.png')
            fig.savefig(batch_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
            fig.savefig(batch_path.replace('.png', '.pdf'), format='pdf',
                        bbox_inches='tight', pad_inches=0.02)
            print(f"  Saved {batch_path}")
        plt.close(fig)


def plot_contrastive_grid_columns(
    embeddings: np.ndarray,
    corpus: CaptionCorpus,
    contrastive_results: List[pd.DataFrame],
    dims_to_plot: List[int],
    n_columns: int = 2,
    top_k: int = 8,
    max_words: int = 20,
    row_height: float = 12 / 11,
    col_width: float = 10,
    random_state: Optional[int] = RANDOM_SEED,
    save_path: Optional[str] = None,
):
    """Every dimension in one figure, rows filled column-major across n_columns.

    Same row as plot_contrastive_grid, so one page can hold more dimensions than
    a single batch (used for ED Fig. 3, the 29 non-BEST dims). row_height and
    col_width default to the per-row geometry of the (10, 12) / 11-row batches.
    random_state fixes word-cloud placement so re-renders are identical.
    """
    if not HAS_WORDCLOUD:
        print("WARNING: wordcloud not installed, skipping contrastive grid.")
        return

    dims_to_plot = [int(d) for d in dims_to_plot]
    contrast_by_dim = _contrast_lookup(contrastive_results)
    n_rows = int(np.ceil(len(dims_to_plot) / n_columns))
    row_cols = top_k + 3  # [label] [spacer] [wordcloud] [top_k images]
    gutter = 0.25         # relative width of the blank column between blocks

    fig = plt.figure(figsize=(col_width * n_columns, row_height * n_rows))
    width_ratios = []
    for b in range(n_columns):
        if b:
            width_ratios.append(gutter)
        width_ratios += [0.4, 0.08, 2] + [1] * top_k
    gs = gridspec.GridSpec(n_rows, len(width_ratios), figure=fig,
                           width_ratios=width_ratios)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005,
                        wspace=0.02, hspace=0.005)

    for i, dim_id in enumerate(dims_to_plot):
        b, r = divmod(i, n_rows)
        col0 = b * (row_cols + 1)
        _draw_contrastive_row(fig, gs, r, col0, dim_id, embeddings, corpus,
                              contrast_by_dim.get(dim_id, pd.DataFrame()),
                              top_k, max_words, random_state=random_state)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf',
                    bbox_inches='tight', pad_inches=0.02)
        print(f"  Saved {save_path}")
    plt.close(fig)


POS_COLORS = {
    'Noun': '#4477AA',
    'Verb': '#228833',
    'Adjective': '#EE6677',
    'Adverb': '#AA3377',
    'Other': '#BBBBBB',
}


def plot_pos_breakdown(
    contrastive_results: List[pd.DataFrame],
    dims_to_plot: List[int],
    top_n: int = 15,
    embeddings: Optional[np.ndarray] = None,
    corpus: Optional[CaptionCorpus] = None,
    top_k: int = 8,
    save_path: Optional[str] = None,
):
    """Horizontal diverging bar chart of contrastive words colored by POS.

    Bars point right for words more frequent in high-activating videos and
    left for words more frequent in low-activating videos.
    Uses a 2-column grid layout for readability.

    If ``embeddings`` and ``corpus`` are provided, annotates each panel with
    the Jensen-Shannon divergence between the top-k and bottom-k caption
    word distributions.
    """
    from matplotlib.patches import Patch

    contrast_by_dim = {}
    for df in contrastive_results:
        if df.empty:
            continue
        contrast_by_dim[df['dim_id'].iloc[0]] = df

    # Pre-compute JS divergence per dimension
    js_by_dim = {}
    if embeddings is not None and corpus is not None:
        for dim_id in dims_to_plot:
            acts = embeddings[:, dim_id]
            order = np.argsort(-acts)
            top_vids = [int(v) for v in order[:top_k]
                        if int(v) in corpus.video_df.index]
            bottom_vids = [int(v) for v in order[-top_k:]
                           if int(v) in corpus.video_df.index]
            top_counts = _pool_word_counts(top_vids, corpus)
            bottom_counts = _pool_word_counts(bottom_vids, corpus)
            js_nats = _js_divergence_from_counters(top_counts, bottom_counts)
            js_by_dim[dim_id] = js_nats / np.log(2)  # convert to bits

    n_dims = len(dims_to_plot)
    n_cols = 2
    n_rows = int(np.ceil(n_dims / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows),
                             squeeze=False)

    # Find global x-range for shared axis limits (top words only)
    global_max = 0
    for dim_id in dims_to_plot:
        cdf = contrast_by_dim.get(dim_id, pd.DataFrame())
        if not cdf.empty:
            sub = cdf[cdf['log_ratio'] > 0].head(top_n)
            if not sub.empty:
                col = 'signed_contrast' if 'signed_contrast' in sub else 'weighted_contrast'
                global_max = max(global_max, sub[col].max())
    x_lim = global_max * 1.1

    for idx, dim_id in enumerate(dims_to_plot):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        cdf = contrast_by_dim.get(dim_id, pd.DataFrame())
        if cdf.empty:
            ax.set_title(f"Dim {dim_id}", fontsize=11, fontweight='bold')
            ax.axis('off')
            continue

        # top_n words from high-activating direction only
        sub = cdf[cdf['log_ratio'] > 0].head(top_n).copy()
        if 'signed_contrast' not in sub:  # single-split stats
            sub['signed_contrast'] = sub['weighted_contrast']
        sub = sub.sort_values('signed_contrast')

        colors = [POS_COLORS.get(p, '#BBBBBB') for p in sub['pos']]
        y_pos = np.arange(len(sub))

        ax.barh(y_pos, sub['signed_contrast'].values, color=colors,
                edgecolor='none', height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub['word'].values, fontsize=12)
        ax.set_xlim(0, x_lim)
        ax.set_title(f"Dim {dim_id}", fontsize=11, fontweight='bold')

        # JS divergence annotation
        if dim_id in js_by_dim:
            ax.text(0.97, 0.05, f"JS = {js_by_dim[dim_id]:.2f} bits",
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=12, fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#CCCCCC', alpha=0.9))

        # Only show x-label on bottom row
        if row == n_rows - 1:
            ax.set_xlabel("Weighted Contrast", fontsize=14)
        else:
            ax.set_xlabel("")

    # Turn off unused axes
    for idx in range(n_dims, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')

    # Legend at the top
    legend_elements = [Patch(facecolor=c, label=p) for p, c in POS_COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper center', fontsize=10,
               frameon=True, title='POS', title_fontsize=10,
               ncol=len(POS_COLORS), bbox_to_anchor=(0.5, 1.01))

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf',
                    bbox_inches='tight')
        print(f"  Saved {save_path}")
    plt.close(fig)


def plot_contrastive_wordclouds(
    contrastive_results: List[pd.DataFrame],
    dims_to_plot: List[int],
    top_n: int = 10,
    save_path: Optional[str] = None,
):
    """Word cloud grid where word size is scaled by log_ratio (top words only)."""
    if not HAS_WORDCLOUD:
        print("WARNING: wordcloud not installed, skipping contrastive wordclouds.")
        return

    contrast_by_dim = {}
    for df in contrastive_results:
        if df.empty:
            continue
        contrast_by_dim[df['dim_id'].iloc[0]] = df

    n_dims = len(dims_to_plot)
    n_cols = 2
    n_rows = int(np.ceil(n_dims / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), squeeze=False)

    for idx, dim_id in enumerate(dims_to_plot):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        cdf = contrast_by_dim.get(dim_id, pd.DataFrame())
        if cdf.empty:
            ax.set_title(f"Dim {dim_id}", fontsize=11, fontweight='bold')
            ax.axis('off')
            continue

        top = cdf[cdf['log_ratio'] > 0].head(top_n)
        freqs = {r['word']: r['log_ratio'] for _, r in top.iterrows()
                 if r['log_ratio'] > 0}

        if freqs:
            pos_lookup = {r['word']: r['pos'] for _, r in top.iterrows()}

            def _make_color_func(lookup):
                def color_func(word, **kwargs):
                    pos = lookup.get(word, 'Other')
                    return POS_COLORS.get(pos, '#BBBBBB')
                return color_func

            wc = WordCloud(
                width=600, height=300,
                background_color='white',
                max_words=top_n,
                color_func=_make_color_func(pos_lookup),
                prefer_horizontal=0.7,
            )
            wc.generate_from_frequencies(freqs)
            ax.imshow(wc.to_array(), interpolation='bilinear')
        ax.set_title(f"Dim {dim_id}", fontsize=11, fontweight='bold')
        ax.axis('off')

    for idx in range(n_dims, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')

    # POS legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=p) for p, c in POS_COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper center', fontsize=10,
               frameon=True, title='POS', title_fontsize=10,
               ncol=len(POS_COLORS), bbox_to_anchor=(0.5, 1.01))

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf',
                    bbox_inches='tight')
        print(f"  Saved {save_path}")
    plt.close(fig)


def build_contrastive_table(
    contrastive_results: List[pd.DataFrame],
    dims_to_plot: List[int],
    top_n: int = 5,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Build summary table: top/bottom contrastive words per dimension."""
    rows = []
    for df in contrastive_results:
        if df.empty:
            continue
        dim_id = df['dim_id'].iloc[0]
        if dim_id not in dims_to_plot:
            continue

        # Words more frequent in high-activating videos
        pos_df = df[df['log_ratio'] > 0].head(top_n)
        for _, r in pos_df.iterrows():
            rows.append({
                'dim_id': dim_id, 'more_frequent_in': 'high',
                'word': r['word'], 'pos': r['pos'],
                'log_ratio': round(r['log_ratio'], 3),
                'weighted_contrast': round(r['weighted_contrast'], 3),
                'count_high': int(r['count_high']),
                'count_low': int(r['count_low']),
            })

        # Words more frequent in low-activating videos
        neg_df = df[df['log_ratio'] < 0].sort_values('weighted_contrast', ascending=False).head(top_n)
        for _, r in neg_df.iterrows():
            rows.append({
                'dim_id': dim_id, 'more_frequent_in': 'low',
                'word': r['word'], 'pos': r['pos'],
                'log_ratio': round(r['log_ratio'], 3),
                'weighted_contrast': round(r['weighted_contrast'], 3),
                'count_high': int(r['count_high']),
                'count_low': int(r['count_low']),
            })

    table = pd.DataFrame(rows)

    if save_dir and not table.empty:
        csv_path = os.path.join(save_dir, 'table_contrastive_summary.csv')
        table.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")

        tex_path = os.path.join(save_dir, 'table_contrastive_summary.tex')
        with open(tex_path, 'w') as f:
            f.write(table.to_latex(index=False, escape=True))
        print(f"  Saved {tex_path}")

    return table


def _js_top_bottom_bits(scores: np.ndarray, corpus: 'CaptionCorpus', top_k: int = 8,
                        n_resamples: int = 200, seed: int = 888) -> float:
    """Top-k vs bottom-k caption JS divergence in bits, averaged over tie-breaks.

    Sparse dimensions put 74-107 videos at exactly 0, so the bottom-k group is
    otherwise one arbitrary draw. Uses the draw logic of
    ``behavioral_contrastive_captions`` (imported lazily: that module imports
    this one) with a fresh RNG per call, so values match its ``js_bits``.
    """
    from analysis.dimensions.behavioral_contrastive_captions import _draw_groups, _tie_group_bounds
    scores = np.asarray(scores, dtype=float)
    valid = np.array([i for i in range(len(scores)) if i in corpus.video_df.index])
    rng = np.random.default_rng(seed)
    n_hi, n_lo = _tie_group_bounds(scores[valid], top_k)
    n_draws = 1 if (n_hi == 1 and n_lo == 1) else n_resamples
    vals = []
    for _ in range(n_draws):
        top, bottom = _draw_groups(scores, valid, top_k, rng)
        vals.append(_js_divergence_from_counters(_pool_word_counts(top, corpus),
                                                 _pool_word_counts(bottom, corpus)) / np.log(2))
    return float(np.mean(vals))


# Word-level POS tags in the top-10 tables. Adjective and Adverb need distinct
# codes (a first-letter tag makes both "A").
POS_ABBR = {'Noun': 'N', 'Verb': 'V', 'Adjective': 'Adj', 'Adverb': 'Adv'}


def _tex_escape(text: str) -> str:
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
                 ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
                 ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')):
        text = text.replace(a, b)
    return text


def write_top10_publication_tex(table: pd.DataFrame, tex_path: str,
                                best_dims: List[int], dim_labels: Dict[int, str],
                                caption: Optional[str] = None,
                                label: str = 'tab:contrastive_top10_all'):
    """ED Table 1: publication LaTeX for the all-dimension top-10 table.

    ``table`` is the numeric output of build_contrastive_top10_table, already in
    train greedy order. BEST_DIMS rows come first, carry their interpretive
    label, and are separated from the rest by a rule. The ten words go in one
    wrapped column with POS subscripts. Landscape xltabular (longtable with an
    X column that takes the remaining width) so it fits the page and can break
    across pages; needs booktabs, xltabular, array and pdflscape.
    """
    word_cols = [c for c in table.columns if c.startswith('Word ')]
    pct_cols = [('Noun%', 'N'), ('Verb%', 'V'), ('Adj%', 'Adj'),
                ('Adverb%', 'Adv'), ('Other%', 'Other')]
    n_cols = 3 + len(pct_cols) + 1

    def _words(row):
        out = []
        for c in word_cols:
            cell = row[c]
            if not isinstance(cell, str) or not cell:
                continue
            word, tag = cell.rsplit(' (', 1)
            out.append(f"{_tex_escape(word)}\\textsubscript{{{tag.rstrip(')')}}}")
        return ', '.join(out)

    def _row(row):
        d = int(row['Dim'])
        cells = [str(d), _tex_escape(dim_labels.get(d, '')), _words(row)]
        cells += [f"{row[c]:.1f}" for c, _ in pct_cols]
        cells.append(f"{row['JS']:.2f}")
        return ' & '.join(cells) + r' \\'

    header = (r'Dim & Label & Top 10 contrastive words (high-activating side) & '
              + ' & '.join(h for _, h in pct_cols) + r' & JS \\')
    group = (r' & & & \multicolumn{5}{c}{POS share of contrast (\%)} & \\'
             '\n' r'\cmidrule(lr){4-8}')
    caption = caption or (r'Top ten contrastive caption words for every learned dimension. '
               r'For each dimension, the words most over-represented in the '
               r'captions of its 8 highest- vs 8 lowest-activating videos, '
               r'ranked by smoothed log-ratio weighted by log frequency and '
               r'averaged over 200 random tie-breaks of the two groups. '
               r'Subscripts give part of speech (N noun, V verb, Adj adjective, '
               "Adv adverb, O other); POS columns give each part of speech's "
               r'share of the top-ten contrast. JS: Jensen--Shannon divergence '
               r'(bits) between the caption word distributions of the two groups. '
               r'Dimensions are ordered by greedy forward selection on the '
               r'training RSM; the first 11 are those retained by the Kneedle elbow.')

    best = [int(d) for d in best_dims]
    top = table[table['Dim'].isin(best)]
    rest = table[~table['Dim'].isin(best)]

    colspec = (r'r >{\raggedright\arraybackslash}p{2.9cm} '
               r'>{\raggedright\arraybackslash}X r r r r r r')
    lines = [
        '% ED Table 1 -- generated by analysis/dimensions/dimension_interpretation.py',
        '% Requires: \\usepackage{booktabs,xltabular,array,pdflscape}',
        r'\begin{landscape}',
        r'\footnotesize',
        r'\setlength{\tabcolsep}{4pt}',
        r'\setlength{\LTcapwidth}{\linewidth}',
        r'\begin{xltabular}{\linewidth}{' + colspec + '}',
        r'\caption{' + caption + r'}\label{' + label + r'}\\',
        r'\toprule', group, header, r'\midrule', r'\endfirsthead',
        r'\multicolumn{' + str(n_cols) + r'}{l}{\emph{(continued)}}\\',
        r'\toprule', group, header, r'\midrule', r'\endhead',
        r'\bottomrule', r'\endlastfoot',
    ]
    lines += [_row(r) for _, r in top.iterrows()]
    lines += [r'\midrule']
    lines += [_row(r) for _, r in rest.iterrows()]
    lines += [r'\end{xltabular}', r'\end{landscape}']
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved {tex_path}")


def build_contrastive_top10_table(
    contrastive_results: List[pd.DataFrame],
    dims_to_plot: List[int],
    dim_labels: Dict[int, str],
    top_n: int = 10,
    save_dir: Optional[str] = None,
    suffix: str = '',
    embeddings: Optional[np.ndarray] = None,
    corpus: Optional['CaptionCorpus'] = None,
    top_k: int = 8,
    n_resamples: int = 200,
    seed: int = 888,
    cloud_max_words: Optional[int] = None,
) -> pd.DataFrame:
    """Build summary table from top-N contrastive words per dimension (high side).

    cloud_max_words=None takes the first top_n words in ranking order. An
    integer instead lists the top_n words that render largest in a word cloud
    of that many words (``cloud_top_words``), so the table matches the figure.

    Columns: Dim, Word 1..10, Noun%, Verb%, Adj%, Adverb%, Other%, JS.
    Mass% = share of total weighted_contrast from that POS among the top N.
    JS = Jensen-Shannon divergence (bits) between the caption word distributions
    of the top-k and bottom-k videos on the dimension, averaged over
    ``n_resamples`` random tie-breaks -- the same quantity, draws and seed as
    ``behavioral_contrastive_captions.py``, so the two are directly comparable.
    NaN when ``embeddings``/``corpus`` are not given.
    """
    pos_categories = [('Noun', 'Noun%'), ('Verb', 'Verb%'),
                      ('Adjective', 'Adj%'), ('Adverb', 'Adverb%'),
                      ('Other', 'Other%')]

    rows = []
    for df in contrastive_results:
        if df.empty:
            continue
        dim_id = df['dim_id'].iloc[0]
        if dim_id not in dims_to_plot:
            continue

        top = (cloud_top_words(df, cloud_max_words, top_n) if cloud_max_words
               else df[df['log_ratio'] > 0].head(top_n))
        if top.empty:
            continue

        row = {'Dim': dim_id}

        # Word columns
        for i in range(top_n):
            if i < len(top):
                r = top.iloc[i]
                row[f'Word {i+1}'] = f"{r['word']} ({POS_ABBR.get(r['pos'], 'O')})"
            else:
                row[f'Word {i+1}'] = ''

        # POS mass % (share of contrast per POS). Tie-averaged stats carry
        # signed_contrast, which is what they are ranked by; for a tie-free
        # dimension it equals weighted_contrast on the log_ratio > 0 side.
        mass_col = 'signed_contrast' if 'signed_contrast' in top else 'weighted_contrast'
        total_wc = top[mass_col].sum()
        for pos, col_name in pos_categories:
            pos_wc = top[top['pos'] == pos][mass_col].sum()
            row[col_name] = round(pos_wc / total_wc * 100, 1) if total_wc > 0 else 0.0

        # JS divergence: top-k vs bottom-k caption words, tie-averaged
        row['JS'] = (round(_js_top_bottom_bits(embeddings[:, dim_id], corpus, top_k,
                                               n_resamples, seed), 3)
                     if embeddings is not None and corpus is not None else np.nan)

        rows.append(row)

    table = pd.DataFrame(rows)

    if save_dir and not table.empty:
        csv_path = os.path.join(save_dir, f'table_contrastive_top10{suffix}.csv')
        table.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")

        tex_path = os.path.join(save_dir, f'table_contrastive_top10{suffix}.tex')
        with open(tex_path, 'w') as f:
            f.write(table.to_latex(index=False, escape=True))
        print(f"  Saved {tex_path}")

    return table


def build_caption_noun_verb_table(
    embeddings: np.ndarray,
    corpus,
    dims_to_plot: List[int],
    top_k: int = 8,
    top_n: int = 5,
    save_dir: Optional[str] = None,
    suffix: str = '',
) -> pd.DataFrame:
    """Build table of top-5 caption nouns and verbs per dimension (high side only).

    Columns: Dim, Noun 1..5, Noun Mass%, Verb 1..5, Verb Mass%, JS.
    Words are ranked by frequency (pct_tokens) within the top-k high-activating videos.
    JS = Jensen-Shannon distance of word distribution vs uniform (0=uniform, 1=peaked).
    """
    from scipy.spatial.distance import jensenshannon

    rows = []
    for dim_id in dims_to_plot:
        acts = embeddings[:, dim_id]
        order = np.argsort(-acts)
        top_vids = [int(v) for v in order[:top_k] if int(v) in corpus.video_df.index]

        counts = _pool_pos_counts(top_vids, corpus)
        n_tokens = sum(counts.values()) or 1

        # Split by POS
        noun_counts = {w: c for (w, pos), c in counts.items() if pos == 'Noun'}
        verb_counts = {w: c for (w, pos), c in counts.items() if pos == 'Verb'}

        noun_sorted = sorted(noun_counts.items(), key=lambda x: -x[1])[:top_n]
        verb_sorted = sorted(verb_counts.items(), key=lambda x: -x[1])[:top_n]

        row = {'Dim': dim_id}

        noun_mass = 0.0
        for i in range(top_n):
            if i < len(noun_sorted):
                word, count = noun_sorted[i]
                pct = count / n_tokens * 100
                noun_mass += pct
                row[f'Noun {i+1}'] = f"{word} ({pct:.1f}%)"
            else:
                row[f'Noun {i+1}'] = ''
        row['Noun Mass%'] = round(noun_mass, 1)

        verb_mass = 0.0
        for i in range(top_n):
            if i < len(verb_sorted):
                word, count = verb_sorted[i]
                pct = count / n_tokens * 100
                verb_mass += pct
                row[f'Verb {i+1}'] = f"{word} ({pct:.1f}%)"
            else:
                row[f'Verb {i+1}'] = ''
        row['Verb Mass%'] = round(verb_mass, 1)

        # JS from full word distribution
        freqs = np.array(list(counts.values()), dtype=float)
        probs = freqs / freqs.sum() if freqs.sum() > 0 else np.ones(len(freqs)) / len(freqs)

        uniform = np.ones(len(probs)) / len(probs)
        row['JS'] = round(jensenshannon(probs, uniform, base=2), 3)

        rows.append(row)

    table = pd.DataFrame(rows)

    if save_dir and not table.empty:
        csv_path = os.path.join(save_dir, f'table_caption_noun_verb{suffix}.csv')
        table.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")

        tex_path = os.path.join(save_dir, f'table_caption_noun_verb{suffix}.tex')
        with open(tex_path, 'w') as f:
            f.write(table.to_latex(index=False, escape=True))
        print(f"  Saved {tex_path}")

    return table


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 50)
    print("DIMENSION INTERPRETATION")
    print("=" * 50)

    # Load embeddings
    sparse_pruned_emb = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    print(f"Loaded embeddings: {sparse_pruned_emb.shape}")

    # Load captions
    captions_path = os.path.join(UTILS_DIR, 'frame_data.csv')
    if not os.path.exists(captions_path):
        print(f"WARNING: Captions file not found at {captions_path}")
        print("Skipping caption-based analysis. Provide frame_data.csv to enable.")
        return

    raw_df = pd.read_csv(captions_path)[['video_name', 'captions']].reset_index()
    raw_df = raw_df.rename(columns={'index': 'vid_id'})

    # Explode the stringified list of captions into individual rows
    raw_images_dir = os.path.join(os.path.dirname(UTILS_DIR), 'raw', 'images')
    rows = []
    for _, row in raw_df.iterrows():
        vid_id = row['vid_id']
        video_name = row['video_name']
        image_path = os.path.join(raw_images_dir, video_name.replace('.mp4', '.png'))
        try:
            caption_list = ast.literal_eval(row['captions'])
        except (ValueError, SyntaxError):
            caption_list = [str(row['captions'])]
        for cap in caption_list:
            cap = cap.strip()
            if cap:
                rows.append({
                    'vid_id': vid_id,
                    'captions': cap,
                    'image_path': image_path,
                })
    caps_df = pd.DataFrame(rows)

    # Build corpus
    corpus = CaptionCorpus.from_caption_df(caps_df)
    print(f"Corpus: {len(corpus.video_df)} videos, {corpus.global_n_tokens} tokens")

    best_dims = BEST_DIMS

    # Quantitative analysis (both top and bottom)
    print("\n--- Analyzing dimensions ---")
    word_stats, dim_summary = analyze_dimensions(
        embeddings=sparse_pruned_emb,
        corpus=corpus,
        dims=best_dims,
        top_k=4,
        sides=("top", "bottom"),
        top_n_words=50,
    )

    # Save results
    dim_summary.to_csv(os.path.join(RESULTS_DIMENSIONS_DIR, 'dim_caption_summary.csv'), index=False)
    word_stats.to_csv(os.path.join(RESULTS_DIMENSIONS_DIR, 'dim_caption_word_stats.csv'), index=False)
    print(f"Saved caption analysis to {RESULTS_DIMENSIONS_DIR}")

    # Print top words per dimension
    for d in best_dims:
        for side in ("top", "bottom"):
            table = pretty_top_words_table(word_stats, dim_id=d, side=side, n=10)
            if not table.empty:
                print(f"\n  Dim {d} {side} words:")
                for _, row in table.iterrows():
                    print(f"    {row['word']:>15s}: {row['% tokens']:.1f}% tokens, "
                          f"lift={row['lift vs global']:.1f}")

    # Word overlap
    overlap = compute_word_overlap_across_dims(word_stats, rank_cutoff=10, side="top")
    if not overlap.empty:
        print("\n--- Word overlap across dimensions ---")
        print(overlap.head(20).to_string(index=False))

    # =========================================================
    # Interpretability summary table
    # =========================================================
    print("\n--- Building interpretability table ---")
    main_table = build_interpretability_table(word_stats, dim_summary)
    table_csv_path = os.path.join(RESULTS_DIMENSIONS_DIR, 'table_dim_interpretability_main.csv')
    main_table.to_csv(table_csv_path, index=False)
    print(f"\nSaved: {table_csv_path}")
    print(main_table.to_markdown(index=False))

    # LaTeX version
    latex = main_table.to_latex(index=False, escape=False)
    tex_path = os.path.join(RESULTS_DIMENSIONS_DIR, 'table_dim_interpretability_main.tex')
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {tex_path}")

    # =========================================================
    # Contrastive caption analysis
    # =========================================================
    print("\n--- Contrastive caption analysis ---")
    corpus_pos = CaptionCorpus.from_caption_df_with_pos(caps_df)
    print(f"POS corpus: {len(corpus_pos.video_df)} videos, "
          f"has_pos={getattr(corpus_pos, 'has_pos', False)}")

    contrastive_results = []
    for d in best_dims:
        cdf = analyze_dimension_contrastive(sparse_pruned_emb, corpus_pos, d, top_k=8,
                                            n_resamples=CONTRASTIVE_N_RESAMPLES)
        contrastive_results.append(cdf)

    # Save per-word contrastive stats
    all_contrastive = pd.concat(contrastive_results, ignore_index=True)
    all_contrastive.to_csv(
        os.path.join(RESULTS_DIMENSIONS_DIR, 'contrastive_word_stats.csv'),
        index=False)
    print(f"Saved contrastive_word_stats.csv ({len(all_contrastive)} rows)")

    # Print top contrastive words per dimension
    for d in best_dims:
        cdf = all_contrastive[all_contrastive['dim_id'] == d]
        if cdf.empty:
            continue
        top_pos = cdf[cdf['log_ratio'] > 0].head(5)
        top_neg = cdf[cdf['log_ratio'] < 0].sort_values('weighted_contrast', ascending=False).head(5)
        print(f"\n  Dim {d} contrastive:")
        print(f"    High-activating: {', '.join(f'{r.word}({r.pos[:1]})' for _, r in top_pos.iterrows())}")
        print(f"    Low-activating:  {', '.join(f'{r.word}({r.pos[:1]})' for _, r in top_neg.iterrows())}")

    # Hebart-style grid (using contrastive word clouds)
    print("\n--- Generating Hebart-style grid ---")
    plot_hebart_style_grid(
        embeddings=sparse_pruned_emb,
        corpus=corpus_pos,
        top_k=8,
        dims_to_plot=best_dims,
        max_words=10,
        batch_size=len(best_dims),
        figsize=(10, 11),
        side="top",
        contrastive_results=contrastive_results,
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'dimension_interpretation_grid.png'),
    )

    # Contrastive word cloud grid (best dims only)
    print("\n--- Generating contrastive word cloud grid ---")
    plot_contrastive_grid(
        embeddings=sparse_pruned_emb,
        corpus=corpus_pos,
        contrastive_results=contrastive_results,
        dims_to_plot=best_dims,
        top_k=8,
        max_words=20,
        batch_size=len(best_dims),
        figsize=(10, 12),
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'contrastive_grid.png'),
    )

    # Contrastive word cloud grid for ALL dimensions (ordered by greedy R²)
    print("\n--- Generating contrastive grid for ALL dimensions ---")
    greedy_path = os.path.join(os.path.dirname(RESULTS_DIMENSIONS_DIR),
                               'behavioral', 'greedy_forward_r2.csv')
    if os.path.exists(greedy_path):
        greedy_df = pd.read_csv(greedy_path)
        all_dims = greedy_df['dim_id'].tolist()
    else:
        print("WARNING: greedy_forward_r2.csv not found, using default dim order")
        all_dims = list(range(sparse_pruned_emb.shape[1]))
    remaining_dims = [d for d in all_dims if d not in best_dims]
    ordered_dims = list(best_dims) + remaining_dims
    all_contrastive_results = list(contrastive_results)
    for d in remaining_dims:
        cdf = analyze_dimension_contrastive(sparse_pruned_emb, corpus_pos, d, top_k=8,
                                            n_resamples=CONTRASTIVE_N_RESAMPLES)
        all_contrastive_results.append(cdf)
    plot_contrastive_grid(
        embeddings=sparse_pruned_emb,
        corpus=corpus_pos,
        contrastive_results=all_contrastive_results,
        dims_to_plot=ordered_dims,
        top_k=8,
        max_words=20,
        batch_size=len(best_dims),
        figsize=(10, 12),
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'contrastive_grid_all.png'),
    )

    # POS breakdown
    print("\n--- Generating POS breakdown chart ---")
    plot_pos_breakdown(
        contrastive_results=contrastive_results,
        dims_to_plot=best_dims,
        top_n=10,
        embeddings=sparse_pruned_emb,
        corpus=corpus_pos,
        top_k=8,
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'contrastive_pos_breakdown.png'),
    )

    # Contrastive word clouds (log_ratio scaled)
    print("\n--- Generating contrastive word clouds ---")
    plot_contrastive_wordclouds(
        contrastive_results=contrastive_results,
        dims_to_plot=best_dims,
        top_n=10,
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'contrastive_wordclouds.png'),
    )

    # Summary table
    print("\n--- Building contrastive summary table ---")
    build_contrastive_table(
        contrastive_results=contrastive_results,
        dims_to_plot=best_dims,
        top_n=5,
        save_dir=RESULTS_DIMENSIONS_DIR,
    )

    # Load greedy forward R² (TRAIN-set selection, matches BEST_DIMS / kneedle elbow)
    greedy_csv_path = os.path.join(os.path.dirname(RESULTS_DIMENSIONS_DIR), 'behavioral', 'greedy_forward_r2_train.csv')
    greedy_r2_df = None
    if os.path.exists(greedy_csv_path):
        greedy_r2_df = pd.read_csv(greedy_csv_path)[['dim_id', 'greedy_step', 'greedy_marginal_r2', 'greedy_cumul_r2']]
        print(f"Loaded greedy R² from {greedy_csv_path}")
    else:
        print(f"WARNING: greedy_forward_r2_train.csv not found at {greedy_csv_path} — run dim_selection_train.py first to add greedy R² columns")

    def _add_greedy_r2(table, suffix=''):
        """Merge greedy R² columns into a table with a 'Dim' column and re-save CSVs."""
        if greedy_r2_df is None or table.empty:
            return table
        table = table.merge(greedy_r2_df.rename(columns={'dim_id': 'Dim'}), on='Dim', how='left')
        return table.sort_values('greedy_step', kind='stable').reset_index(drop=True)

    def _save_top10(table, suffix=''):
        """Write the top-10 table as CSV + .tex in greedy order, rounded for print.

        Floats are fixed at 2 dp, except greedy_marginal_r2 at 4 dp (its gains
        past step 11 are all < 0.003). The .tex keeps the pre-merge columns.
        """
        def _fmt(col, v):
            if pd.isna(v):
                return ''
            s = f"{v:.{4 if col == 'greedy_marginal_r2' else 2}f}"
            return s[1:] if s.startswith('-') and float(s) == 0 else s  # no "-0.00"

        out = table.copy()
        for c in out.select_dtypes('float').columns:
            out[c] = [_fmt(c, v) for v in out[c]]
        out.to_csv(os.path.join(RESULTS_DIMENSIONS_DIR, f'table_contrastive_top10{suffix}.csv'),
                   index=False)
        cols = [c for c in out.columns if not c.startswith('greedy_')]
        # Formatted numbers are strings now; keep them right-aligned in LaTeX
        col_fmt = ''.join('r' if pd.api.types.is_numeric_dtype(table[c]) else 'l'
                          for c in cols)
        tex_path = os.path.join(RESULTS_DIMENSIONS_DIR, f'table_contrastive_top10{suffix}.tex')
        with open(tex_path, 'w') as f:
            f.write(out[cols].to_latex(index=False, escape=True, column_format=col_fmt))

    # Contrastive top-10 table
    print("\n--- Building contrastive top-10 table ---")
    dim_labels_path = os.path.join(RESULTS_DIMENSIONS_DIR, 'dimension_summaries_noun_verb.csv')
    if os.path.exists(dim_labels_path):
        label_df = pd.read_csv(dim_labels_path)
        dim_labels = dict(zip(label_df['Dim'], label_df['Label']))
    else:
        dim_labels = {d: '' for d in best_dims}
    t = build_contrastive_top10_table(
        contrastive_results=contrastive_results,
        dims_to_plot=best_dims,
        dim_labels=dim_labels,
        top_n=10,
        save_dir=RESULTS_DIMENSIONS_DIR,
        embeddings=sparse_pruned_emb,
        corpus=corpus_pos,
    )
    t = _add_greedy_r2(t)
    if not t.empty:
        _save_top10(t, suffix='')

    # Caption noun/verb table (top side only)
    print("\n--- Building caption noun/verb table ---")
    t = build_caption_noun_verb_table(
        embeddings=sparse_pruned_emb,
        corpus=corpus_pos,
        dims_to_plot=best_dims,
        top_k=8,
        top_n=5,
        save_dir=RESULTS_DIMENSIONS_DIR,
    )
    t = _add_greedy_r2(t)
    if not t.empty:
        t.to_csv(os.path.join(RESULTS_DIMENSIONS_DIR, 'table_caption_noun_verb.csv'), index=False)

    # =========================================================
    # All-dimensions tables (all 40 dims)
    # =========================================================
    n_all_dims = sparse_pruned_emb.shape[1]
    all_dims = list(range(n_all_dims))

    print(f"\n--- Contrastive analysis for all {n_all_dims} dimensions ---")
    contrastive_results_all = []
    for d in all_dims:
        cdf = analyze_dimension_contrastive(sparse_pruned_emb, corpus_pos, d, top_k=8,
                                            n_resamples=CONTRASTIVE_N_RESAMPLES)
        contrastive_results_all.append(cdf)

    # Cache every dimension, so regen_contrastive_grid_all.py never recomputes
    pd.concat(contrastive_results_all, ignore_index=True).to_csv(
        os.path.join(RESULTS_DIMENSIONS_DIR, 'contrastive_word_stats.csv'), index=False)
    print(f"Saved contrastive_word_stats.csv (all {n_all_dims} dims)")

    dim_labels_all = {d: dim_labels.get(d, '') for d in all_dims}

    print("\n--- Building contrastive top-10 table (all dims) ---")
    t = build_contrastive_top10_table(
        contrastive_results=contrastive_results_all,
        dims_to_plot=all_dims,
        dim_labels=dim_labels_all,
        top_n=10,
        save_dir=RESULTS_DIMENSIONS_DIR,
        embeddings=sparse_pruned_emb,
        corpus=corpus_pos,
        suffix='_all',
    )
    t = _add_greedy_r2(t, suffix='_all')
    if not t.empty:
        _save_top10(t, suffix='_all')
        write_top10_publication_tex(
            t, os.path.join(RESULTS_DIMENSIONS_DIR, 'table_contrastive_top10_all_publication.tex'),
            best_dims=best_dims, dim_labels=DIM_LABELS)

    print("\n--- Building caption noun/verb table (all dims) ---")
    t = build_caption_noun_verb_table(
        embeddings=sparse_pruned_emb,
        corpus=corpus_pos,
        dims_to_plot=all_dims,
        top_k=8,
        top_n=5,
        save_dir=RESULTS_DIMENSIONS_DIR,
        suffix='_all',
    )
    t = _add_greedy_r2(t, suffix='_all')
    if not t.empty:
        t.to_csv(os.path.join(RESULTS_DIMENSIONS_DIR, 'table_caption_noun_verb_all.csv'), index=False)

    print("\n" + "=" * 50)
    print("DIMENSION INTERPRETATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
