"""
Dimension analysis: dominant dimensions, inter-dimension correlations,
and dimension-behavior correlations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
from scipy.stats import spearmanr, pearsonr, rankdata
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    UTILS_DIR, MODEL_DIR, RESULTS_DIMENSIONS_DIR, RESULTS_BEHAVIORAL_DIR,
    BEHAVIOR_COLS, FEATURE_REMAPS, BEST_DIMS, test_idx
)
from helpers.action_categories import load_action_categories

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

os.makedirs(RESULTS_DIMENSIONS_DIR, exist_ok=True)

# Ordering + grouping of the pre-labeled annotations, mirroring FEATURE_GROUPS in
# analysis/behavioral/explained_variance_shared_only.py so the annotation axis reads the same
# way in both figures. Names are post-FEATURE_REMAPS display labels.
# Action Categories is nominal, so its column holds a pair-level quantity
# (see action_category_rsm_correspondence) rather than an item-level Pearson r.
ACTION_CATEGORY_LABEL = "Action Categories"

ANNOTATION_GROUPS = [
    ("Scene", ["Indoor/Outdoor", "Spatial Expanse"]),
    ("Bodies", ["Interagent Distance", "Agents Facing"]),
    ("Action + Interaction",
     [ACTION_CATEGORY_LABEL, "Object Directed", "Communicating",
      "Acting Jointly"]),
    ("Social Affective",
     ["Intimacy", "Dominance", "Cooperation", "Valence", "Arousal"]),
]


def action_category_rsm_correspondence(embeddings, action_labels, dim_indices,
                                       n_perm=5000, seed=0, method='pearson'):
    """Signed Pearson r between each dimension's RSM and the same-category RSM.

    method='spearman' rank-transforms each dimension RSM first (the binary
    category vector is rank-invariant up to an affine map), giving Spearman r.

    Action category is nominal, so there is no item-level correlation with a
    continuous dimension. The per-dimension RSM is built as in
    ``correlate_rsm_level``.

    The p-value is from a **stimulus-label permutation** (shuffle video identity
    and rebuild the category RSM), not the parametric one. The 1225 pairs are not
    independent -- each of the 50 videos appears in 49 of them -- so a parametric
    p treats dependent cells as independent and runs orders of magnitude too
    small (e.g. 3.5e-9 for an r of 0.17).

    Returns ``(r, p)``, each of length ``len(dim_indices)``.
    """
    n_vids = embeddings.shape[0]
    triu_idx = np.triu_indices(n_vids, k=1)

    def _cat_vec(labels):
        return ((labels[:, None] == labels[None, :]).astype(float))[triu_idx]

    def _unit(v):
        v = v - v.mean()
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    # Pre-normalise so a Pearson r is a dot product; the permutation then costs
    # one matvec per shuffle rather than 11 correlations.
    dim_mat = []
    for d in dim_indices:
        dist = pdist(embeddings[:, [d]], metric="euclidean")
        dmax = dist.max()
        dim_sim = 1 - (dist / dmax) if dmax != 0 else 1 - dist
        if method == 'spearman':
            dim_sim = rankdata(dim_sim)
        dim_mat.append(_unit(dim_sim))
    dim_mat = np.asarray(dim_mat)

    r = dim_mat @ _unit(_cat_vec(action_labels))

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, len(dim_indices)))
    for k in range(n_perm):
        null[k] = dim_mat @ _unit(_cat_vec(action_labels[rng.permutation(n_vids)]))
    p = ((np.abs(null) >= np.abs(r)).sum(axis=0) + 1) / (n_perm + 1)
    return r, p


def annotation_group_layout(labels):
    """Order ``labels`` by ANNOTATION_GROUPS.

    Returns ``(order, spans)`` where ``order`` indexes into ``labels`` and
    ``spans`` is a list of (group_name, first_col, last_col) in heatmap column
    coordinates, for the bracket annotations. Labels not in any group keep their
    original relative order and are appended, ungrouped, at the right.
    """
    by_label = {lab: i for i, lab in enumerate(labels)}
    order, spans = [], []
    for gname, feats in ANNOTATION_GROUPS:
        present = [f for f in feats if f in by_label]
        if not present:
            continue
        first = len(order)
        order.extend(by_label[f] for f in present)
        spans.append((gname, first, len(order) - 1))
    grouped = {f for _, feats in ANNOTATION_GROUPS for f in feats}
    order.extend(i for i, lab in enumerate(labels) if lab not in grouped)
    return order, spans


def draw_annotation_group_brackets(fig, ax, spans, pad=0.02, gap=0.012,
                                   label_gap=0.05, fontsize=14):
    """Draw a horizontal rule + group name beneath each group's x tick labels.

    x is in ``ax`` data coords and y in figure coords, so a bracket spans exactly
    the columns it belongs to regardless of how long the tick labels are. Call
    after ``tight_layout``; heatmap column centres sit at i + 0.5.

    ``pad``/``gap``/``label_gap`` are figure-fraction offsets; the caller scales
    ``label_gap`` down for taller figures so the same physical gap doesn't turn
    into visible dead space (see ``correlate_embeddings_with_behaviors_pearson``).
    """
    if not spans:
        return None
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ticks = ax.get_xticklabels()
    if ticks:
        bottoms = [fig.transFigure.inverted().transform(
            t.get_window_extent(renderer=renderer)).min(axis=0)[1] for t in ticks]
        y_bracket = min(bottoms) - pad
    else:
        y_bracket = ax.get_position().y0 - pad

    trans = blended_transform_factory(ax.transData, fig.transFigure)
    for gname, c0, c1 in spans:
        x0, x1 = c0 + 0.08, c1 + 0.92
        ax.plot([x0, x1], [y_bracket, y_bracket], transform=trans,
                color="0.35", linewidth=1.2, solid_capstyle="butt",
                clip_on=False, zorder=5)
        ax.text((x0 + x1) / 2, y_bracket - gap, gname, transform=trans,
                ha="center", va="top", fontsize=fontsize, color="0.2",
                clip_on=False)

    # Keep the axis label clear of the brackets.
    pos = ax.get_position()
    y_label = y_bracket - gap - label_gap
    ax.xaxis.set_label_coords(0.5, (y_label - pos.y0) / pos.height)
    return y_bracket


def plot_dominant_dimensions(embeddings, save_path=None):
    """Plot distribution of dominant (argmax) dimensions across videos."""
    dominant_dims = np.argmax(embeddings, axis=1)
    n_dims = embeddings.shape[1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(dominant_dims, bins=np.arange(n_dims + 1) - 0.5, edgecolor='black', color='tab:purple')
    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("# Videos", fontsize=12)
    ax.set_title("Dominant Dimension per Video (argmax activation)", fontsize=14)
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([str(d) for d in range(n_dims)], rotation=90, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    return dominant_dims


def plot_dimension_correlations(embeddings, save_path=None, best_dims=None, order=None):
    """Plot Spearman correlation matrix between embedding dimensions (ED Fig. 4).

    order: rows/columns in this dimension order (the paper uses the train
    greedy order, so BEST_DIMS form the top-left block); a line is drawn after
    the first len(best_dims) entries when they are exactly BEST_DIMS. The
    returned matrix is always in numeric dimension order.

    Spearman, not Pearson: the dimensions are sparse and heavy-tailed, and a
    single video can dominate a Pearson r. Test video 25 holds 97% / 92% of
    Dim 34 / 35's summed activation, giving Pearson r = 1.00 (0.24 without it)
    against Spearman 0.14. Note the flip side: 13-29 of 50 videos sit at exactly
    0 on each dim, and shared tied zeros nudge Spearman slightly positive.
    """
    corr = spearmanr(embeddings).correlation
    n_dims = corr.shape[0]

    order = list(range(n_dims)) if order is None else [int(d) for d in order]
    axis_label = "Dimension" if order == list(range(n_dims)) else "Dimension (greedy order)"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr[np.ix_(order, order)], cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax, square=True, xticklabels=order, yticklabels=order,
                cbar_kws={'label': 'Spearman r'})
    ax.set_xlabel(axis_label, fontsize=14)
    ax.set_ylabel(axis_label, fontsize=14)
    ax.tick_params(labelsize=12)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Spearman r', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    if best_dims is not None and order[:len(best_dims)] == list(best_dims):
        ax.axhline(len(best_dims), color='k', lw=1.2)
        ax.axvline(len(best_dims), color='k', lw=1.2)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)

    # Find most correlated pairs (excluding diagonal)
    np.fill_diagonal(corr, 0)
    triu_idx = np.triu_indices(n_dims, k=1)
    pairs = [(int(triu_idx[0][i]), int(triu_idx[1][i]), corr[triu_idx[0][i], triu_idx[1][i]])
             for i in range(len(triu_idx[0]))]
    pairs.sort(key=lambda x: -abs(x[2]))

    print("\nTop 10 most correlated dimension pairs (Spearman):")
    for d1, d2, r in pairs[:10]:
        print(f"  Dim {d1} <-> Dim {d2}: rho = {r:.4f}")

    # Best dims correlation if provided
    if best_dims is not None:
        emb_best = embeddings[:, best_dims]
        corr_best = spearmanr(emb_best).correlation
        mask = np.triu(np.ones_like(corr_best, dtype=bool))

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(corr_best, mask=mask, cmap='RdBu_r', square=True,
                    vmin=-0.6, vmax=0.6,
                    xticklabels=best_dims, yticklabels=best_dims,
                    cbar_kws={'label': 'Spearman r'},
                    linewidths=0.5, linecolor='white', ax=ax)
        ax.set_xlabel("Dimension", fontsize=12)
        ax.set_ylabel("Dimension", fontsize=12)
        plt.tight_layout()
        if save_path:
            best_path = save_path.replace('.png', '_best_dims.png')
            fig.savefig(best_path, dpi=300, bbox_inches='tight')
            fig.savefig(best_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)

    return corr


def correlate_embeddings_with_behaviors(embeddings, annotations, feature_cols,
                                         best_dims=None, save_path=None):
    """Item-level Spearman R² heatmap: embedding dimensions vs behavioral features.
    If best_dims is provided, only those dimensions are shown."""
    dim_indices = best_dims if best_dims is not None else list(range(embeddings.shape[1]))
    n_dims = len(dim_indices)
    feature_labels = [FEATURE_REMAPS.get(f.title(), f.title()) for f in feature_cols]

    r2_matrix = np.zeros((len(feature_cols), n_dims))
    p_matrix = np.zeros((len(feature_cols), n_dims))

    for i, feature in enumerate(feature_cols):
        for j, d in enumerate(dim_indices):
            r, p = spearmanr(embeddings[:, d], annotations[feature].values)
            r2_matrix[i, j] = r ** 2
            p_matrix[i, j] = p

    # Sort features by total R² descending
    row_order = np.argsort(-r2_matrix.sum(axis=1))
    r2_matrix = r2_matrix[row_order]
    p_matrix = p_matrix[row_order]
    feature_labels = [feature_labels[i] for i in row_order]

    # Build annotation strings with significance stars
    annot = np.empty_like(r2_matrix, dtype=object)
    for i in range(r2_matrix.shape[0]):
        for j in range(r2_matrix.shape[1]):
            val = r2_matrix[i, j]
            star = '*' if p_matrix[i, j] < 0.05 else ''
            annot[i, j] = f"{val:.2f}{star}"

    dim_labels = [str(d) for d in dim_indices]

    fig, ax = plt.subplots(figsize=(max(8, n_dims * 0.8), max(6, len(feature_cols) * 0.55)))
    sns.heatmap(r2_matrix, cmap='viridis', vmin=0, vmax=1.0, ax=ax,
                xticklabels=dim_labels,
                yticklabels=feature_labels,
                annot=annot, fmt='',
                cbar_kws={'label': 'Spearman R²'})
    ax.set_title("Spearman Rank-Explained Variance (R²)", fontsize=14)
    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("Behavioral Feature", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    return r2_matrix


def correlate_embeddings_with_behaviors_pearson(embeddings, annotations, feature_cols,
                                                 best_dims=None, save_path=None,
                                                 action_labels=None, cbar_shrink=None,
                                                 method='pearson', order=None, font_delta=0):
    """Item-level Pearson r heatmap: pre-labeled annotations (x) vs embedding
    dimensions (y). Blue = negative, red = positive.

    Annotations are ordered and bracketed by ``ANNOTATION_GROUPS`` so the x-axis
    matches ``explained_variance_test_50vids_spearman_emb_shared_total_stacked_total_variance``.

    ``action_labels`` (one nominal action category per video, same row order as
    ``embeddings``) adds the Action Categories column. That variable is nominal,
    so no item-level Pearson r exists for it; the column instead holds the
    Pearson r between each dimension's RSM and the binary same-category RSM.

    Asterisks are Bonferroni-corrected within each dimension (row).

    method: 'pearson' (default) or 'spearman' (Fig. 4, ED Fig. 5), applied to
    both the item-level cells and the action-category column. order: row order
    of dimensions when best_dims is None (ED Fig. 5 uses the train greedy
    order). font_delta: added to every text size.
    """
    corr_fn = spearmanr if method == 'spearman' else pearsonr
    r_label = 'Spearman r' if method == 'spearman' else 'Pearson r'
    fd = font_delta
    if best_dims is not None:
        dim_indices = best_dims
    elif order is not None:
        dim_indices = [int(d) for d in order]
    else:
        dim_indices = list(range(embeddings.shape[1]))
    n_dims = len(dim_indices)
    feature_labels = [FEATURE_REMAPS.get(f.title(), f.title()) for f in feature_cols]

    # r/p are computed (dim, feature) so the heatmap is dims on y, annotations on x.
    r_matrix = np.zeros((n_dims, len(feature_cols)))
    p_matrix = np.zeros((n_dims, len(feature_cols)))

    for j, feature in enumerate(feature_cols):
        for i, d in enumerate(dim_indices):
            r, p = corr_fn(embeddings[:, d], annotations[feature].values)
            r_matrix[i, j] = r
            p_matrix[i, j] = p

    if action_labels is not None:
        act_r, act_p = action_category_rsm_correspondence(
            embeddings, np.asarray(action_labels), dim_indices, method=method)
        r_matrix = np.column_stack([r_matrix, act_r])
        p_matrix = np.column_stack([p_matrix, act_p])
        feature_labels = feature_labels + [ACTION_CATEGORY_LABEL]

    # Column order follows the annotation grouping; anything ungrouped is appended.
    col_order, spans = annotation_group_layout(feature_labels)
    r_matrix = r_matrix[:, col_order]
    p_matrix = p_matrix[:, col_order]
    feature_labels = [feature_labels[j] for j in col_order]

    # Bonferroni within each dimension: a row's annotation tests are the family.
    q_matrix = np.minimum(p_matrix * p_matrix.shape[1], 1.0)

    # Build annotation strings with significance stars
    annot = np.empty_like(r_matrix, dtype=object)
    for i in range(r_matrix.shape[0]):
        for j in range(r_matrix.shape[1]):
            val = r_matrix[i, j]
            star = '*' if q_matrix[i, j] <= 0.05 else ''
            annot[i, j] = f"{val:.2f}{star}"

    print(f"  Bonferroni within dimension ({p_matrix.shape[1]} tests/row): "
          f"{int((q_matrix <= 0.05).sum())} of "
          f"{int((p_matrix < 0.05).sum())} uncorrected stars survive")

    dim_labels = [str(d) for d in dim_indices]

    fig, ax = plt.subplots(figsize=(max(8, len(feature_labels) * 1.0),
                                    max(6, n_dims * 0.55)))
    # Group-label / note spacing below the heatmap is tuned against this
    # 11-dim (BEST_DIMS) figure's height. `spacing_scale` is 1.0 there (no
    # change) and shrinks for taller figures (e.g. all 40 dims) so the same
    # physical gap doesn't turn into visible dead space.
    REFERENCE_FIG_H = 6.05
    spacing_scale = min(1.0, REFERENCE_FIG_H / fig.get_size_inches()[1])
    cbar_kws = {'label': r_label}
    if cbar_shrink is not None:
        cbar_kws['shrink'] = cbar_shrink
    sns.heatmap(r_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=feature_labels,
                yticklabels=dim_labels,
                annot=annot, fmt='', annot_kws={'fontsize': 14 + fd},
                cbar_kws=cbar_kws)
    ax.collections[0].colorbar.set_label(r_label, fontsize=14 + fd)
    ax.collections[0].colorbar.ax.tick_params(labelsize=12 + fd)
    ax.set_xlabel("Pre-labeled Annotations", fontsize=18 + fd)
    ylabel = "Learned Dimensions" + (" (greedy order)" if order is not None and best_dims is None else "")
    ax.set_ylabel(ylabel, fontsize=16 + fd)
    ax.set_yticklabels(dim_labels, rotation=0, fontsize=14 + fd)
    ax.set_xticklabels(feature_labels, rotation=90, ha='center', fontsize=14 + fd)
    plt.tight_layout()
    y_bracket = draw_annotation_group_brackets(
        fig, ax, spans, label_gap=0.05 * spacing_scale, fontsize=16 + fd)
    note_gap = 0.10 * spacing_scale
    fig.text(ax.get_position().x1,
             (y_bracket - note_gap) if y_bracket is not None else 0.005,
             r"$*$: Bonferroni $p \leq 0.05$",
             ha="right", va="top", fontsize=12 + fd, color="gray")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    return r_matrix


def correlate_rsm_level(embeddings, annotations, feature_cols, best_dims=None,
                        save_path=None):
    """RSM-level Spearman R²: per-dimension RSM vs behavioral feature RSM.
    If best_dims is provided, only those dimensions are shown."""
    n_vids = embeddings.shape[0]
    triu_idx = np.triu_indices(n_vids, k=1)

    dim_indices = best_dims if best_dims is not None else list(range(embeddings.shape[1]))
    n_dims = len(dim_indices)
    feature_labels = [FEATURE_REMAPS.get(f.title(), f.title()) for f in feature_cols]

    r2_matrix = np.zeros((len(feature_cols), n_dims))
    p_matrix = np.zeros((len(feature_cols), n_dims))

    for i, feature in enumerate(feature_cols):
        feat_vals = annotations[feature].to_numpy().reshape(-1, 1)
        feat_rsm = 1 - pairwise_distances(feat_vals, metric='euclidean')
        feat_vec = feat_rsm[triu_idx]

        for j, d in enumerate(dim_indices):
            dist = pdist(embeddings[:, [d]], metric="euclidean")
            dmax = dist.max()
            dim_sim = 1 - (dist / dmax) if dmax != 0 else 1 - dist
            r, p = spearmanr(dim_sim, feat_vec)
            r2_matrix[i, j] = r ** 2
            p_matrix[i, j] = p

    # Sort features by total R² descending
    row_order = np.argsort(-r2_matrix.sum(axis=1))
    r2_matrix = r2_matrix[row_order]
    p_matrix = p_matrix[row_order]
    feature_labels = [feature_labels[i] for i in row_order]

    # Build annotation strings with significance stars
    annot = np.empty_like(r2_matrix, dtype=object)
    for i in range(r2_matrix.shape[0]):
        for j in range(r2_matrix.shape[1]):
            val = r2_matrix[i, j]
            star = '*' if p_matrix[i, j] < 0.05 else ''
            annot[i, j] = f"{val:.2f}{star}"

    dim_labels = [str(d) for d in dim_indices]

    fig, ax = plt.subplots(figsize=(max(8, n_dims * 0.8), max(6, len(feature_cols) * 0.55)))
    sns.heatmap(r2_matrix, cmap='viridis', vmin=0, vmax=1.0, ax=ax,
                xticklabels=dim_labels,
                yticklabels=feature_labels,
                annot=annot, fmt='',
                cbar_kws={'label': 'Spearman R²'})
    ax.set_title("Spearman Rank-Explained Variance (R²)", fontsize=14)
    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("Behavioral Feature", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    return r2_matrix


def plot_sorted_activations(embeddings, best_dims, video_idx=None, save_path=None):
    """Sorted activation curves per dimension (best dims only).
    For each dimension, sort its activation values descending and plot.
    If video_idx is provided, subset to those videos first."""
    if video_idx is not None:
        embeddings = embeddings[video_idx]
    n_videos = embeddings.shape[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    for d in best_dims:
        vals = np.sort(embeddings[:, d])[::-1]
        ax.plot(np.arange(1, n_videos + 1), vals, label=str(d), linewidth=1.5)

    ax.set_xlabel("Activation rank (1 = highest)", fontsize=12)
    ax.set_ylabel("Activation", fontsize=12)
    ax.set_title("Sorted activations per dimension", fontsize=14)
    ax.legend(title="Dimension", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    print("\n" + "=" * 50)
    print("DIMENSION ANALYSIS")
    print("=" * 50)

    # Load embeddings — subset to test set
    sparse_pruned_emb = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    print(f"Loaded embeddings: {sparse_pruned_emb.shape}")
    emb_test = sparse_pruned_emb[test_idx, :]
    print(f"Test set embeddings: {emb_test.shape} ({len(test_idx)} videos)")

    # Load annotations — subset to test set
    annotations = pd.read_csv(os.path.join(UTILS_DIR, 'annotations.csv'))
    annotations_test = annotations.iloc[test_idx].reset_index(drop=True)

    # Dominant dimensions
    print("\n--- Dominant Dimensions ---")
    dom_dims = plot_dominant_dimensions(
        emb_test,
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'dominant_dimensions.png')
    )

    # Dimension correlations
    print("\n--- Dimension Correlations ---")
    greedy_order = pd.read_csv(os.path.join(RESULTS_BEHAVIORAL_DIR,
                                            'greedy_forward_r2_train.csv'))['dim_id'].tolist()
    corr = plot_dimension_correlations(
        emb_test,
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'dimension_correlations.png'),
        best_dims=BEST_DIMS,
        order=greedy_order,
    )

    # Sorted activations per dimension (test set)
    print("\n--- Sorted Activations ---")
    plot_sorted_activations(
        emb_test, BEST_DIMS,
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'sorted_activations.png')
    )

    # Dimension x Behavior correlations (item-level Spearman R2)
    print("\n--- Dimension x Behavior (Spearman R2) ---")
    r2_matrix = correlate_embeddings_with_behaviors(
        emb_test, annotations_test, BEHAVIOR_COLS,
        best_dims=BEST_DIMS,
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'dim_behavior_spearman_r2.png')
    )

    # Pearson r version (item-level, best dims only)
    print("\n--- Dimension x Behavior (Pearson r) ---")
    correlate_embeddings_with_behaviors_pearson(
        emb_test, annotations_test, BEHAVIOR_COLS,
        best_dims=BEST_DIMS,
        action_labels=load_action_categories(UTILS_DIR)[test_idx],
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'dim_behavior_pearson_r.png')
    )

    # Spearman r version of Fig. 4 (item-level, best dims only), matching ED Fig. 5
    print("\n--- Dimension x Behavior (Spearman r) ---")
    correlate_embeddings_with_behaviors_pearson(
        emb_test, annotations_test, BEHAVIOR_COLS,
        best_dims=BEST_DIMS,
        action_labels=load_action_categories(UTILS_DIR)[test_idx],
        method='spearman',
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'dim_behavior_spearman_r.png')
    )

    # Spearman r version (item-level, all 40 dims, greedy order) -- ED Fig. 5.
    # Spearman for the same reason as ED Fig. 4: single videos dominate Pearson
    # on these sparse dimensions.
    print("\n--- Dimension x Behavior (Spearman r, all dims, greedy order) ---")
    correlate_embeddings_with_behaviors_pearson(
        emb_test, annotations_test, BEHAVIOR_COLS,
        best_dims=None,
        action_labels=load_action_categories(UTILS_DIR)[test_idx],
        cbar_shrink=0.5,
        method='spearman', order=greedy_order, font_delta=2,
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'dim_behavior_spearman_r_all.png')
    )

    # RSM-level Spearman R² (best dims only)
    print("\n--- Dimension x Behavior (RSM-level Spearman R²) ---")
    rsm_corr = correlate_rsm_level(
        emb_test, annotations_test, BEHAVIOR_COLS,
        best_dims=BEST_DIMS,
        save_path=os.path.join(RESULTS_DIMENSIONS_DIR, 'dim_behavior_rsm_spearman.png')
    )

    print("\n" + "=" * 50)
    print("DIMENSION ANALYSIS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
