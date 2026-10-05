"""
Variant of explained_variance plot: keeps per-feature unique variance for
behavioral features, but collapses embedding dimensions to only Shared and
Total variance (Spearman r², 50-video test set).
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    UTILS_DIR, MODEL_DIR, RESULTS_BEHAVIORAL_DIR, BEHAVIOR_COLS, test_idx,
    RANDOM_SEED, BEST_DIMS,
)
from helpers.variance_partition import (
    build_behavior_rsms, compute_explained_variance_spearman, format_behavior_df, get_bar_colors, cv_spearman_r2,
)
from helpers.action_categories import load_action_categories, build_action_rsm
from helpers.video_bootstrap import (pair_lookup, resample_pairs, grouped_folds,
                                     cv_spearman_r2 as _cv_r2_grouped, percentile_ci)


N_PERMS = 1000
SIG_ALPHA = 0.05

# Null used by every permutation test below. 'cell' shuffles the 1225 pair
# values independently, which breaks the dependence between pairs that share
# a video and is anticonservative. 'video' permutes video identity (rows and
# columns of the 50x50 RSM together) and keeps that dependence. Set from
# --perm in main(); 'video' writes to separately suffixed cache and figures.
PERM_MODE = "cell"

# Grouping of the pre-labeled features for the x-axis of the left panel.
# Each group is drawn as a contiguous cluster of bars with a horizontal
# bracket and a group name beneath the (rotated) tick labels.
FEATURE_GROUPS = [
    ("Scene", ["Indoor/Outdoor", "Spatial Expanse"]),
    ("Bodies", ["Interagent Distance", "Agents Facing"]),
    ("Action + Interaction",
     ["Action Categories", "Object Directed", "Communicating",
      "Acting Jointly"]),
    ("Social Affective",
     ["Intimacy", "Dominance", "Cooperation", "Valence", "Arousal"]),
]
GROUP_GAP = 0.9      # extra x-units inserted between groups
SUMMARY_GAP = 1.4    # extra x-units before the Shared/Total summary bars


def group_layout(df_beh, exclude=(), summary_gap=None):
    """Reorder df_beh into FEATURE_GROUPS.

    Returns ``(row_order, x_positions, spans)`` where ``row_order`` indexes
    into ``df_beh`` and ``spans`` is a list of (group_name, x_first, x_last)
    for the bracket annotations. The Shared/Total summary bars carry no
    group and are appended, after an extra gap, at the right.
    """
    by_feat = {f: i for i, f in enumerate(df_beh["Feature"])
               if f not in exclude}
    rows, xs, spans = [], [], []
    x = 0.0
    for gi, (gname, feats) in enumerate(FEATURE_GROUPS):
        present = [f for f in feats if f in by_feat]
        if not present:
            continue
        if gi > 0:
            x += GROUP_GAP
        x_first = x
        for f in present:
            rows.append(by_feat[f])
            xs.append(x)
            x += 1.0
        spans.append((gname, x_first, x - 1.0))
    # Anything not assigned to a group (Shared / Total) goes at the end.
    leftover = [i for f, i in by_feat.items()
                if not any(f in feats for _, feats in FEATURE_GROUPS)]
    if leftover:
        x += SUMMARY_GAP if summary_gap is None else summary_gap
        for i in sorted(leftover):
            rows.append(i)
            xs.append(x)
            x += 1.0
    return rows, np.asarray(xs), spans


def _cv_spearman_r2_single(X, y, n_folds=10, seed=RANDOM_SEED):
    """Single-repeat 10-fold CV Spearman r² (fast variant for permutation null)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(LinearRegression(), X, y, cv=kf)
    r, _ = spearmanr(y, y_pred)
    return float(r ** 2)


def _permute(y, rng):
    """One null draw of the upper-triangle target ``y`` under PERM_MODE."""
    if PERM_MODE == "cell":
        return rng.permutation(y)
    n = int(round((1 + np.sqrt(1 + 8 * len(y))) / 2))
    iu = np.triu_indices(n, k=1)
    M = np.zeros((n, n))
    M[iu] = y
    M = M + M.T
    p = rng.permutation(n)
    return M[np.ix_(p, p)][iu]


def compute_partition_pvalues(y, feature_df, r2_full_obs, unique_obs, shared_obs,
                              n_perms=N_PERMS, seed=RANDOM_SEED):
    """Permutation p-values for each unique/shared/total variance bar."""
    cols = list(feature_df.columns)
    X = feature_df.values
    rng = np.random.RandomState(seed)

    exc_full = 0
    exc_unique = {c: 0 for c in cols}
    exc_shared = 0

    for _ in tqdm(range(n_perms), desc="Permutations"):
        y_perm = _permute(y, rng)
        r2_full_p = _cv_spearman_r2_single(X, y_perm, seed=seed)
        exc_full += int(r2_full_p >= r2_full_obs)

        sum_u = 0.0
        for c in cols:
            X_wo = feature_df.drop(columns=[c]).values
            r2_wo_p = _cv_spearman_r2_single(X_wo, y_perm, seed=seed)
            u_p = r2_full_p - r2_wo_p
            sum_u += u_p
            exc_unique[c] += int(u_p >= unique_obs[c])
        shared_p = r2_full_p - sum_u
        exc_shared += int(shared_p >= shared_obs)

    p_full = (exc_full + 1) / (n_perms + 1)
    p_unique = {c: (exc_unique[c] + 1) / (n_perms + 1) for c in cols}
    p_shared = (exc_shared + 1) / (n_perms + 1)
    return p_full, p_unique, p_shared


def compute_total_pvalue(y, feature_df, r2_full_obs,
                         n_perms=N_PERMS, seed=RANDOM_SEED):
    """Permutation p-value for total Spearman r² only."""
    X = feature_df.values
    rng = np.random.RandomState(seed)
    exc = 0
    for _ in tqdm(range(n_perms), desc="Permutations (emb total)"):
        y_perm = _permute(y, rng)
        r2_full_p = _cv_spearman_r2_single(X, y_perm, seed=seed)
        exc += int(r2_full_p >= r2_full_obs)
    return (exc + 1) / (n_perms + 1)


def compute_marginal_r2(y, feature_df):
    """Total (marginal) variance each feature explains on its own.

    Unlike the unique-variance partition this does not remove what a feature
    shares with the others, so the values overlap and do not sum to the
    full-model r2. Same estimator as everywhere else: 10-fold x 5-repeat CV
    Spearman r2.
    """
    return {c: cv_spearman_r2(feature_df[[c]].values, y)
            for c in feature_df.columns}


def compute_marginal_pvalues(y, feature_df, obs, n_perms=N_PERMS,
                             seed=RANDOM_SEED):
    """Permutation p-values for the per-feature marginal r2 bars."""
    cols = list(feature_df.columns)
    rng = np.random.RandomState(seed)
    exc = {c: 0 for c in cols}
    for _ in tqdm(range(n_perms), desc="Permutations (marginal)"):
        y_perm = _permute(y, rng)
        for c in cols:
            r2_p = _cv_spearman_r2_single(feature_df[[c]].values, y_perm,
                                          seed=seed)
            exc[c] += int(r2_p >= obs[c])
    return {c: (exc[c] + 1) / (n_perms + 1) for c in cols}


def compute_joint_pvalues(y, beh_df, emb_df, obs_u_beh, obs_shared, obs_u_emb,
                          n_perms=N_PERMS, seed=RANDOM_SEED):
    """Permutation p-values for the pre-labeled x learned variance partition.

    Two-set partition: unique_beh = r2_joint - r2_emb, unique_emb =
    r2_joint - r2_beh, shared = r2_beh + r2_emb - r2_joint. All three are
    recomputed on each permuted target so they share one null draw, matching
    how ``compute_partition_pvalues`` handles the per-feature bars.
    """
    Xb, Xe = beh_df.values, emb_df.values
    Xj = np.hstack([Xb, Xe])
    rng = np.random.RandomState(seed)
    exc_u_beh = exc_shared = exc_u_emb = 0
    for _ in tqdm(range(n_perms), desc="Permutations (joint partition)"):
        y_perm = _permute(y, rng)
        r2_b = _cv_spearman_r2_single(Xb, y_perm, seed=seed)
        r2_e = _cv_spearman_r2_single(Xe, y_perm, seed=seed)
        r2_j = _cv_spearman_r2_single(Xj, y_perm, seed=seed)
        exc_u_beh += int((r2_j - r2_e) >= obs_u_beh)
        exc_shared += int((r2_b + r2_e - r2_j) >= obs_shared)
        exc_u_emb += int((r2_j - r2_b) >= obs_u_emb)
    return ((exc_u_beh + 1) / (n_perms + 1),
            (exc_shared + 1) / (n_perms + 1),
            (exc_u_emb + 1) / (n_perms + 1))


def bonferroni(pvals):
    """Bonferroni-adjusted p-values, returned in input order."""
    p = np.asarray(pvals, dtype=float)
    return np.minimum(p * len(p), 1.0)


def sig_stars(p):
    """Return asterisk string for a p-value (single '*' at alpha=0.05)."""
    return "*" if p <= SIG_ALPHA else ""

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


N_BOOT = 1000
CI_COLOR = "0.45"   # error-bar gray, a shade darker than the asterisks


def _boot_one(b, y, beh_X, beh_names, emb_X, best_cols_idx, lut, seed):
    """One video-bootstrap draw of every quantity Fig. 2 draws."""
    rng = np.random.default_rng(seed + b)
    pid = resample_pairs(int(round((1 + np.sqrt(1 + 8 * len(y))) / 2)), rng, lut)
    yb = y[pid]
    folds = [grouped_folds(pid, 10, RANDOM_SEED + r) for r in range(5)]
    r2 = lambda X: _cv_r2_grouped(X, yb, folds=folds, n_folds=10)
    out = {f"marginal::{n}": r2(beh_X[pid, j]) for j, n in enumerate(beh_names)}
    out["full_beh"] = r2(beh_X[pid])
    out["full_emb"] = r2(emb_X[pid])
    out["full_emb_best"] = r2(emb_X[pid][:, best_cols_idx])
    out["joint"] = r2(np.hstack([beh_X[pid], emb_X[pid]]))
    out["u_beh_joint"] = out["joint"] - out["full_emb"]
    out["u_emb_joint"] = out["joint"] - out["full_beh"]
    out["shared_joint"] = out["full_beh"] + out["full_emb"] - out["joint"]
    return out


def video_bootstrap(y, beh_data, df_emb, best_cols, n_boot=N_BOOT, seed=RANDOM_SEED,
                    n_jobs=6):
    """Video-bootstrap distribution (n_boot x quantities) for Fig. 2's bars.

    Resamples the 50 test videos; CV folds grouped by original pair so
    duplicated pairs never straddle train/test (see helpers/video_bootstrap).
    """
    from joblib import Parallel, delayed
    n = int(round((1 + np.sqrt(1 + 8 * len(y))) / 2))
    lut = pair_lookup(n)
    beh_X, emb_X = beh_data.values.astype(float), df_emb.values.astype(float)
    best_idx = [list(df_emb.columns).index(c) for c in best_cols]
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_boot_one)(b, y, beh_X, list(beh_data.columns), emb_X, best_idx,
                           lut, seed) for b in tqdm(range(n_boot), desc="video bootstrap"))
    return pd.DataFrame(rows)


def main():
    global PERM_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", choices=["cell", "video"], default="cell",
                    help="permutation null: shuffle pair cells (original) or "
                         "video labels; 'video' outputs carry a _videoperm suffix")
    ap.add_argument("--ci", action="store_true",
                    help="add 95%% video-bootstrap CIs to every bar and the "
                         "noise-ceiling CI band; renders only the stacked "
                         "total-variance figure, with a _ci suffix")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    PERM_MODE = args.perm
    perm_tag = "" if PERM_MODE == "cell" else "_videoperm"
    sparse_pruned_emb = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
    annotations = pd.read_csv(os.path.join(UTILS_DIR, 'annotations.csv'))

    test_rsm_path = os.path.join(UTILS_DIR, 'human_test_rsm.csv')
    test_rsm_npy = os.path.join(UTILS_DIR, 'test_human_sim.npy')
    if os.path.exists(test_rsm_path):
        human_test_rsm = pd.read_csv(test_rsm_path).to_numpy()
    else:
        human_test_rsm = np.load(test_rsm_npy)

    t_idx = test_idx if test_idx is not None else list(range(human_test_rsm.shape[0]))
    n_test = len(t_idx)
    triu_idx = np.triu_indices(n_test, k=1)
    y = human_test_rsm[triu_idx]

    # Behavioral features
    rsms = build_behavior_rsms(annotations, BEHAVIOR_COLS)
    beh_features = {f.title(): rsms[f][np.ix_(t_idx, t_idx)][triu_idx] for f in rsms}
    labels_all = load_action_categories(UTILS_DIR)
    action_rsm_test = build_action_rsm(labels_all[t_idx])
    beh_features['Action Categories'] = action_rsm_test[triu_idx]
    beh_data = pd.DataFrame(beh_features)

    # Embedding dimensions (RSM per-dim)
    emb_test = sparse_pruned_emb[t_idx, :]
    emb_features = {}
    for d in range(emb_test.shape[1]):
        dist = pdist(emb_test[:, [d]], metric="euclidean")
        dmax = dist.max()
        emb_features[f"Dim {d}"] = 1 - (dist / dmax) if dmax != 0 else 1 - dist
    df_emb = pd.DataFrame(emb_features)

    print("Computing behavioral variance (Spearman)...")
    r2_full_beh, unique_beh, shared_beh = compute_explained_variance_spearman(y, beh_data)
    print(f"  Total: {r2_full_beh:.4f}  Shared: {shared_beh:.4f}")

    print("Computing embedding variance (Spearman)...")
    r2_full_emb, _, shared_emb = compute_explained_variance_spearman(y, df_emb)
    print(f"  Total: {r2_full_emb:.4f}  Shared: {shared_emb:.4f}")

    print("Computing per-feature marginal (total) variance...")
    marginal_beh = compute_marginal_r2(y, beh_data)
    for f, v in sorted(marginal_beh.items(), key=lambda kv: -kv[1]):
        print(f"    marginal {f:24s} {v:.4f}")

    # Same partition restricted to the 11 selected dimensions (BEST_DIMS).
    best_cols = [f"Dim {d}" for d in BEST_DIMS]
    df_emb_best = df_emb[best_cols]
    print(f"Computing embedding variance for BEST_DIMS ({len(BEST_DIMS)} dims)...")
    r2_full_emb_best, _, _ = compute_explained_variance_spearman(y, df_emb_best)
    print(f"  Total: {r2_full_emb_best:.4f}")

    # Two-set variance partition: what the pre-labeled features and the full
    # learned embedding explain uniquely, and what they explain in common.
    print("Computing pre-labeled x learned partition (all dims)...")
    df_joint_x = pd.concat([beh_data, df_emb], axis=1)
    r2_joint = cv_spearman_r2(df_joint_x.values, y)
    u_beh_joint = r2_joint - r2_full_emb
    u_emb_joint = r2_joint - r2_full_beh
    shared_joint = r2_full_beh + r2_full_emb - r2_joint
    print(f"  Joint: {r2_joint:.4f}  unique pre-labeled: {u_beh_joint:.4f}  "
          f"shared: {shared_joint:.4f}  unique learned: {u_emb_joint:.4f}")

    # -----------------------------------------------------------------
    # Permutation p-values (cached to JSON so re-runs are fast)
    # -----------------------------------------------------------------
    pval_cache = os.path.join(RESULTS_BEHAVIORAL_DIR,
                              f'explained_variance_pvalues_spearman{perm_tag}.json')
    if os.path.exists(pval_cache):
        with open(pval_cache) as fh:
            pv = json.load(fh)
        p_full_beh = pv["p_full_beh"]
        p_unique_beh = pv["p_unique_beh"]
        p_shared_beh = pv["p_shared_beh"]
        p_full_emb = pv["p_full_emb"]
        print(f"Loaded cached p-values from {pval_cache}")
        if "p_full_emb_best" in pv:
            p_full_emb_best = pv["p_full_emb_best"]
        else:
            print("  cache predates BEST_DIMS total — running that permutation only...")
            p_full_emb_best = compute_total_pvalue(y, df_emb_best, r2_full_emb_best)
            pv["p_full_emb_best"] = p_full_emb_best
            with open(pval_cache, "w") as fh:
                json.dump(pv, fh, indent=2)
            print(f"  updated {pval_cache}")
        if "p_marginal_beh" in pv:
            p_marginal_beh = pv["p_marginal_beh"]
        else:
            print("  cache predates the marginal bars — running those permutations...")
            p_marginal_beh = compute_marginal_pvalues(y, beh_data, marginal_beh)
            pv["p_marginal_beh"] = p_marginal_beh
            with open(pval_cache, "w") as fh:
                json.dump(pv, fh, indent=2)
            print(f"  updated {pval_cache}")
        if "p_u_beh_joint" in pv and "p_full_joint" in pv:
            p_u_beh_joint = pv["p_u_beh_joint"]
            p_shared_joint = pv["p_shared_joint"]
            p_u_emb_joint = pv["p_u_emb_joint"]
            p_full_joint = pv["p_full_joint"]
        else:
            print("  cache predates the joint partition — running those permutations...")
            p_u_beh_joint, p_shared_joint, p_u_emb_joint = compute_joint_pvalues(
                y, beh_data, df_emb, u_beh_joint, shared_joint, u_emb_joint)
            p_full_joint = compute_total_pvalue(y, df_joint_x, r2_joint)
            pv.update({"p_u_beh_joint": p_u_beh_joint,
                       "p_shared_joint": p_shared_joint,
                       "p_u_emb_joint": p_u_emb_joint,
                       "p_full_joint": p_full_joint})
            with open(pval_cache, "w") as fh:
                json.dump(pv, fh, indent=2)
            print(f"  updated {pval_cache}")
    else:
        print(f"Running permutation tests ({N_PERMS} perms) — this takes a few minutes...")
        p_full_beh, p_unique_beh, p_shared_beh = compute_partition_pvalues(
            y, beh_data, r2_full_beh, dict(unique_beh), shared_beh
        )
        p_marginal_beh = compute_marginal_pvalues(y, beh_data, marginal_beh)
        p_full_emb = compute_total_pvalue(y, df_emb, r2_full_emb)
        p_full_emb_best = compute_total_pvalue(y, df_emb_best, r2_full_emb_best)
        p_u_beh_joint, p_shared_joint, p_u_emb_joint = compute_joint_pvalues(
            y, beh_data, df_emb, u_beh_joint, shared_joint, u_emb_joint)
        p_full_joint = compute_total_pvalue(y, df_joint_x, r2_joint)
        with open(pval_cache, "w") as fh:
            json.dump({
                "n_perms": N_PERMS,
                "p_full_beh": p_full_beh,
                "p_unique_beh": p_unique_beh,
                "p_shared_beh": p_shared_beh,
                "p_marginal_beh": p_marginal_beh,
                "p_full_emb": p_full_emb,
                "p_full_emb_best": p_full_emb_best,
                "p_u_beh_joint": p_u_beh_joint,
                "p_shared_joint": p_shared_joint,
                "p_u_emb_joint": p_u_emb_joint,
                "p_full_joint": p_full_joint,
            }, fh, indent=2)
        print(f"Saved p-values to {pval_cache}")

    print(f"  p(behavior total) = {p_full_beh:.4f}")
    print(f"  p(behavior shared) = {p_shared_beh:.4f}")
    for feat, pv_ in p_unique_beh.items():
        print(f"    p(unique {feat}) = {pv_:.4f}")
    print(f"  p(embedding total) = {p_full_emb:.4f}")
    print(f"  p(embedding total, BEST_DIMS) = {p_full_emb_best:.4f}")
    print(f"  p(unique pre-labeled | joint) = {p_u_beh_joint:.4f}")
    print(f"  p(shared pre-labeled n learned) = {p_shared_joint:.4f}")
    print(f"  p(unique learned | joint) = {p_u_emb_joint:.4f}")
    print(f"  p(joint total) = {p_full_joint:.4f}")

    # -----------------------------------------------------------------
    # Multiple-comparison correction (Bonferroni), applied within
    # two families: the 13 per-annotation unique-variance tests, and the 8
    # summary tests (the two model totals, the pre-labeled shared term, and
    # the four bars of the joint partition). Correcting within families
    # rather than pooling avoids letting the near-zero summary p-values
    # inflate the annotation threshold.
    # -----------------------------------------------------------------
    feat_names = list(p_unique_beh)
    q_unique_beh = dict(zip(feat_names,
                            bonferroni([p_unique_beh[c] for c in feat_names])))
    q_marginal_beh = dict(zip(feat_names,
                              bonferroni([p_marginal_beh[c] for c in feat_names])))
    summary_names = ["shared_beh", "full_beh", "full_emb", "full_emb_best",
                     "u_beh_joint", "shared_joint", "u_emb_joint", "full_joint"]
    summary_p = [p_shared_beh, p_full_beh, p_full_emb, p_full_emb_best,
                 p_u_beh_joint, p_shared_joint, p_u_emb_joint, p_full_joint]
    q_sum = dict(zip(summary_names, bonferroni(summary_p)))

    print("\n  Bonferroni over the 13 annotation tests (p_adj):")
    for c in sorted(feat_names, key=lambda k: p_unique_beh[k]):
        mark = "*" if q_unique_beh[c] <= SIG_ALPHA else " "
        print(f"    {mark} {c:20s} p={p_unique_beh[c]:.4f}  q={q_unique_beh[c]:.4f}")
    print("\n  Bonferroni over the 13 marginal (total) annotation tests (p_adj):")
    for c in sorted(feat_names, key=lambda k: p_marginal_beh[k]):
        mark = "*" if q_marginal_beh[c] <= SIG_ALPHA else " "
        print(f"    {mark} {c:20s} r2={marginal_beh[c]:.4f}  "
              f"p={p_marginal_beh[c]:.4f}  q={q_marginal_beh[c]:.4f}")
    print("  Bonferroni over the 8 summary tests: "
          f"max p_adj = {max(q_sum.values()):.4f}")

    # Map behavioral-feature p-values onto the renamed/ordered df_beh rows.
    # format_behavior_df renames raw column names via FEATURE_REMAPS, then
    # inserts "Behavior Shared" and "Behavior Total" — so look up each row's
    # p-value using the original column label when possible.
    from config import FEATURE_REMAPS
    remap_inv = {v: k for k, v in FEATURE_REMAPS.items()}

    def _pval_for_feature(label, qmap):
        """Bonferroni-adjusted p for a panel-A bar (within its own family)."""
        if "Total" in label:
            return q_sum["full_beh"]
        if "Shared" in label:
            return q_sum["shared_beh"]
        raw = remap_inv.get(label, label)
        return qmap.get(raw, qmap.get(label, 1.0))

    # Format behavioral df (keeps per-feature value + shared + total).
    # Two readouts share this layout: 'unique' (variance a feature explains
    # that no other does) and 'total' (what it explains on its own).
    def _panel_a(values, qmap):
        df = format_behavior_df(dict(values), shared_beh, r2_full_beh)
        colors = get_bar_colors(df["Feature"], "tab:purple")
        pvals = [_pval_for_feature(f, qmap) for f in df["Feature"]]
        # Display-only rename; the underlying "Total Variance" key drives
        # colors, p-value lookup and FEATURE_ORDER sorting above, so rename
        # after those.
        df["Feature"] = df["Feature"].replace(
            {"Shared Variance": "Shared pre-labeled",
             "Total Variance": "Total pre-labeled"})
        # Re-order into conceptual groups (Scene / Bodies / Action +
        # Interaction / Social Affective), with the summary bars last.
        row_order, xs, spans = group_layout(df)
        return (df.iloc[row_order].reset_index(drop=True),
                [colors[i] for i in row_order],
                [pvals[i] for i in row_order], xs, spans)

    panel_a = {
        "unique": _panel_a(unique_beh, q_unique_beh),
        "total": _panel_a(marginal_beh, q_marginal_beh),
    }
    df_beh = panel_a["unique"][0]

    # Embedding reduced to totals only: all dims, then the BEST_DIMS subset.
    df_emb_reduced = pd.DataFrame({
        "Dimension": [f"Total learned ({df_emb.shape[1]} dims)",
                      f"Total learned ({len(BEST_DIMS)} Dims)"],
        "R2": [r2_full_emb, r2_full_emb_best],
    })
    emb_colors = ["darkgray", "darkgray"]

    noise_ceiling = 0.6658

    # -----------------------------------------------------------------
    # Video-bootstrap CIs (--ci), cached so re-renders are instant
    # -----------------------------------------------------------------
    ci = None
    if args.ci:
        boot_cache = os.path.join(RESULTS_BEHAVIORAL_DIR,
                                  'explained_variance_bootstrap_spearman.csv')
        if os.path.exists(boot_cache):
            boot = pd.read_csv(boot_cache)
            print(f"Loaded {len(boot)} bootstrap draws from {boot_cache}")
        else:
            print(f"Video bootstrap ({args.n_boot} draws)...")
            boot = video_bootstrap(y, beh_data, df_emb, best_cols, n_boot=args.n_boot)
            boot.to_csv(boot_cache, index=False)
            print(f"Saved {boot_cache}")
        lo, hi = percentile_ci(boot.values)
        ci = {c: (l, h) for c, l, h in zip(boot.columns, lo, hi)}
        from helpers.reliability import load_test_triplets, split_half_ceiling
        nc_mean, nc_ci = split_half_ceiling(load_test_triplets(UTILS_DIR),
                                            np.asarray(sorted(t_idx)), metric='spearman')
        ci["ceiling"] = nc_ci
        point = {"full_beh": r2_full_beh, "full_emb": r2_full_emb,
                 "full_emb_best": r2_full_emb_best, "joint": r2_joint,
                 "u_beh_joint": u_beh_joint, "u_emb_joint": u_emb_joint,
                 "shared_joint": shared_joint,
                 **{f"marginal::{k}": v for k, v in marginal_beh.items()}}
        summary = pd.DataFrame([{"quantity": k, "estimate": point[k],
                                 "ci_lo": ci[k][0], "ci_hi": ci[k][1]}
                                for k in point])
        summary.to_csv(os.path.join(RESULTS_BEHAVIORAL_DIR,
                                    'explained_variance_bootstrap_ci_spearman.csv'),
                       index=False)
        print(summary.round(4).to_string(index=False))
        print(f"  noise ceiling {nc_mean:.4f} [{nc_ci[0]:.4f}, {nc_ci[1]:.4f}]")

    def _ci_for_label(label):
        """(lo, hi) for a panel-A bar, by display label."""
        if "Total" in label:
            return ci["full_beh"]
        raw = remap_inv.get(label, label)
        return ci.get(f"marginal::{raw}", ci.get(f"marginal::{label}", (np.nan, np.nan)))

    def _render(joint_style, readout="unique"):
        """Draw the whole figure.

        `joint_style` controls the third panel:
        'grouped'  - four side-by-side bars (unique / unique / shared / total)
        'stacked'  - one stacked bar, the two model totals bracketed either side

        `readout` controls what panel A's per-annotation bars show:
        'unique'   - variance that annotation explains and no other one can
        'total'    - variance that annotation explains on its own (these
                     overlap, so they do not sum to the full-model bar)
        """
        src_df, src_colors, src_pvals, x_beh, group_spans = panel_a[readout]
        # Third panel: the two-set partition of pre-labeled vs learned.
        df_joint = pd.DataFrame({
            "Part": ["Unique pre-labeled",
                     f"Unique learned ({df_emb.shape[1]})",
                     "Shared", "Total"],
            "R2": [u_beh_joint, u_emb_joint, shared_joint, r2_joint],
        })
        joint_colors = ["tab:purple", "tab:blue", "lightgray", "darkgray"]
        joint_pvals = [q_sum["u_beh_joint"], q_sum["u_emb_joint"],
                       q_sum["shared_joint"], q_sum["full_joint"]]

        x_emb = np.arange(len(df_emb_reduced)) - (len(df_emb_reduced) - 1) / 2
        x_joint = np.arange(len(df_joint)) - (len(df_joint) - 1) / 2
        # The stacked variant drops the Shared pre-labeled bar, so panel A is
        # laid out again without it rather than left with a hole.
        if joint_style == "stacked":
            keep, panel_x, panel_spans = group_layout(
                src_df, exclude={"Shared pre-labeled"}, summary_gap=0.8)
            panel_df = src_df.iloc[keep].reset_index(drop=True)
            panel_pvals = [src_pvals[i] for i in keep]
        else:
            panel_x, panel_spans = x_beh, group_spans
            panel_df, panel_pvals = src_df, src_pvals
        left_panel_units = panel_x.max() - panel_x.min() + 1

        # Width ratio chosen so that 1 data unit on each axis maps to the same
        # physical width — keeps bar widths visually identical across panels.
        # Right panel gets extra x-range (set via xlim below) to leave room for
        # the "Learned Dimensions" xlabel without cutting it off.
        right_panel_units = 4
        joint_panel_units = 6 if joint_style == "grouped" else 5
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(17, 6), sharey=True,
            gridspec_kw={"width_ratios": [left_panel_units, right_panel_units,
                                          joint_panel_units]},
        )

        # The stacked variant recolors the summary bars so the two model
        # families read as blues; the grouped variant keeps the original grays.
        if joint_style == "stacked":
            summary_recolor = {"Total pre-labeled": "rebeccapurple",
                               "Shared pre-labeled": "mediumpurple"}
            # Per-feature bars go mediumorchid, matching panel B's unique
            # pre-labeled segment; the summary bars keep their own colors.
            panel_beh_colors = [summary_recolor.get(
                                    f, "mediumorchid" if c == "tab:purple" else c)
                                for f, c in zip(panel_df["Feature"],
                                                [src_colors[i] for i in keep])]
            panel_emb_colors = ["tab:blue"] * len(df_emb_reduced)
        else:
            panel_beh_colors, panel_emb_colors = src_colors, emb_colors

        # Left: behavioral features (unique + shared + total)
        bar_width = 0.6
        vals_a = panel_df["Variance"].values
        err_a, top_a = None, np.maximum(vals_a, 0)
        if ci is not None:
            lohi = np.array([_ci_for_label(f) for f in panel_df["Feature"]])
            err_a = np.vstack([np.clip(vals_a - lohi[:, 0], 0, None),
                               np.clip(lohi[:, 1] - vals_a, 0, None)])
            top_a = np.maximum(top_a, lohi[:, 1])
        ax1.bar(panel_x, vals_a, width=bar_width,
                color=panel_beh_colors, edgecolor="black", linewidth=0.8,
                yerr=err_a, capsize=3 if ci is not None else 0,
                error_kw={"elinewidth": 1.0, "ecolor": CI_COLOR})

        # Significance asterisks above each significant bar
        star_offset = noise_ceiling * 0.015
        for xi, val, pv_ in zip(panel_x, top_a, panel_pvals):
            s = sig_stars(pv_)
            if s:
                ax1.text(xi, max(val, 0) + star_offset, s,
                         ha="center", va="bottom", fontsize=14, fontweight="bold",
                         color="gray")

        ax1.axhline(y=noise_ceiling, color="gray", linestyle="--", alpha=0.5)
        variance_word = "Total" if readout == "total" else "Unique"
        ax1.set_xlabel(
            f"Pre-labeled Annotations ({variance_word} variance)"
            if joint_style == "stacked" else "Pre-labeled Features",
            fontsize=16)
        ax1.set_ylabel("Explained Variance (Spearman $r^2$)", fontsize=15)
        ax1.tick_params(axis='y', labelsize=12)
        # Grouped keeps the per-bar (UV) tag; stacked states it once in the
        # axis label instead.
        tag = "TV" if readout == "total" else "UV"
        beh_labels = [f if joint_style == "stacked"
                      or "Shared" in f or "Total" in f
                      else f"{f} $^{{\\rm ({tag})}}$"
                      for f in panel_df["Feature"]]
        ax1.set_xticks(panel_x)
        ax1.set_xticklabels(beh_labels, rotation=90, ha="center", fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.25)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Right: embedding shared + total
        vals_b = df_emb_reduced["R2"].values
        err_b, top_b = None, np.maximum(vals_b, 0)
        if ci is not None:
            lohi = np.array([ci["full_emb"], ci["full_emb_best"]])
            err_b = np.vstack([np.clip(vals_b - lohi[:, 0], 0, None),
                               np.clip(lohi[:, 1] - vals_b, 0, None)])
            top_b = np.maximum(top_b, lohi[:, 1])
        ax2.bar(x_emb, vals_b, width=bar_width,
                color=panel_emb_colors, edgecolor="black", linewidth=0.8,
                yerr=err_b, capsize=3 if ci is not None else 0,
                error_kw={"elinewidth": 1.0, "ecolor": CI_COLOR})
        emb_pvals = [q_sum["full_emb"], q_sum["full_emb_best"]]
        for xi, val, pv_ in zip(x_emb, top_b, emb_pvals):
            s = sig_stars(pv_)
            if s:
                ax2.text(xi, max(val, 0) + star_offset, s,
                         ha="center", va="bottom", fontsize=16, fontweight="bold",
                         color="gray")
        ax2.axhline(y=noise_ceiling, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Learned Dimensions", fontsize=16)
        ax2.set_xticks(x_emb)
        ax2.set_xticklabels(df_emb_reduced["Dimension"], rotation=90, ha="center", fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.25)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Far right: how the pre-labeled features and the full learned embedding
        # divide the variance they jointly explain.
        if joint_style == "grouped":
            ax3.bar(x_joint, df_joint["R2"].values, width=bar_width,
                    color=joint_colors, edgecolor="black", linewidth=0.8)
            for xi, val, pv_ in zip(x_joint, df_joint["R2"].values, joint_pvals):
                s_ = sig_stars(pv_)
                if s_:
                    ax3.text(xi, max(val, 0) + star_offset, s_,
                             ha="center", va="bottom", fontsize=14,
                             fontweight="bold", color="gray")
        else:
            # One stacked bar, bottom to top: unique pre-labeled, unique
            # learned, shared. The upper two are the learned model's total and
            # are boxed; the pre-labeled total (unique pre-labeled + shared) is
            # NOT a contiguous run under this order, so it carries no bracket.
            bar_w = 2.2
            segs = [("Unique pre-labeled", u_beh_joint, "rebeccapurple",
                     q_sum["u_beh_joint"]),
                    (f"Unique learned ({df_emb.shape[1]} dims)", u_emb_joint,
                     "tab:blue", q_sum["u_emb_joint"]),
                    ("Shared", shared_joint, "lightgray",
                     q_sum["shared_joint"])]
            bottom = 0.0
            handles = []
            for name, val, col, pv_ in segs:
                ax3.bar([0], [val], bottom=[bottom], width=bar_w, color=col,
                        edgecolor="black", linewidth=0.8, zorder=2,
                        label=name)
                seg_lbl = f"{val:.2f}" + ("$^{*}$" if sig_stars(pv_) else "")
                ax3.text(0, bottom + val / 2, seg_lbl, ha="center",
                         va="center", fontsize=12,
                         color="white" if col in ("tab:blue",
                                                 "rebeccapurple") else "black",
                         zorder=4)
                handles.append(Patch(facecolor=col, edgecolor="black",
                                     linewidth=0.8, label=name))
                bottom += val

            # The marker capping the bar stands for the joint total.
            handles.append(Line2D([], [], marker="s", color="black",
                                  linestyle="none", markersize=7,
                                  label="Total variance"))
            ax3.legend(handles=handles[::-1], loc="center left",
                       bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=12)

            # Total on top, with the marker + asterisk carried over.
            ax3.plot([0], [bottom + star_offset], marker="s", color="black",
                     markersize=7, zorder=6, clip_on=False)
            lbl_y = bottom + star_offset * 3.5
            if ci is not None:
                lo_j, hi_j = ci["joint"]
                # Just outside the bar's right edge, so it clears the
                # segment labels; the total's label stays on the bar.
                ax3.errorbar([bar_w / 2 + 0.25], [bottom],
                             yerr=[[bottom - lo_j], [hi_j - bottom]],
                             fmt="o", color=CI_COLOR, markersize=3,
                             ecolor=CI_COLOR, elinewidth=1.0, capsize=3,
                             zorder=6, clip_on=False)
            lbl = f"{bottom:.2f}" + ("$^{*}$" if sig_stars(q_sum["full_joint"]) else "")
            ax3.text(0, lbl_y, lbl, ha="center",
                     va="bottom", fontsize=12)
        ax3.axhline(y=noise_ceiling, color="gray", linestyle="--", alpha=0.5)
        if ci is not None:
            for ax_ in (ax1, ax2, ax3):
                ax_.axhspan(ci["ceiling"][0], ci["ceiling"][1], color="gray",
                            alpha=0.15, lw=0, zorder=0)
        ceiling_label = f"Split-Half Reliability ({noise_ceiling:.2f})"
        if joint_style == "stacked":
            # A single label, at the right-hand end of panel A; the ceiling
            # line itself still runs across all three panels.
            ax1.text(panel_x.max() + 0.5, noise_ceiling, ceiling_label,
                     ha="right", va="bottom", color="gray", alpha=0.7,
                     fontsize=12)
        else:
            ax3.text(joint_panel_units / 2, noise_ceiling, ceiling_label,
                     ha="right", va="bottom", color="gray", alpha=0.7,
                     fontsize=12)
        # Wrapped in the stacked variant: with no spacer column, the one-line
        # label is wider than its panel and runs into panel B's.
        ax3.set_xlabel("Pre-labeled and \nLearned Combined"
                       if joint_style == "stacked"
                       else "Pre-labeled and Learned Combined", fontsize=16)
        if joint_style == "grouped":
            ax3.set_xticks(x_joint)
            ax3.set_xticklabels(df_joint["Part"], rotation=90, ha="center",
                                fontsize=12)
        else:
            ax3.set_xticks([])
        ax3.grid(axis='y', linestyle='--', alpha=0.25)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        y_top = max(noise_ceiling, ci["ceiling"][1]) if ci is not None else noise_ceiling
        ax1.set_ylim(0, y_top)
        ax1.set_xlim(panel_x.min() - 0.5, panel_x.max() + 0.5)
        ax2.set_ylim(0, y_top)
        # Center the single bar with x-range matching width_ratio so 1 data unit
        # has the same physical size as the left panel.
        ax2.set_xlim(-right_panel_units / 2, right_panel_units / 2)
        ax3.set_ylim(0, y_top)
        ax3.set_xlim(-joint_panel_units / 2, joint_panel_units / 2)

        plt.tight_layout()
        # Align both xlabels to the lower of the two positions so they sit at
        # the same y and clear the (longer) behavioral tick labels.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        def _label_bottom(ax):
            ticks = ax.get_xticklabels()
            if not ticks:
                return np.inf
            bbox = ticks[0].get_window_extent(renderer=renderer)
            return fig.transFigure.inverted().transform(bbox).min(axis=0)[1]
        y_ticks_bottom = min(_label_bottom(ax1), _label_bottom(ax2),
                             _label_bottom(ax3))

        # Group brackets + names, drawn just beneath the rotated tick labels.
        # x in ax1 data coords, y in figure coords, so the rule spans exactly the
        # bars it belongs to regardless of tick-label length.
        trans = blended_transform_factory(ax1.transData, fig.transFigure)
        y_bracket = y_ticks_bottom - 0.03
        for gname, gx0, gx1 in panel_spans:
            ax1.plot([gx0 - 0.42, gx1 + 0.42], [y_bracket, y_bracket],
                     transform=trans, color="0.35", linewidth=1.2,
                     solid_capstyle="butt", clip_on=False, zorder=5)
            ax1.text((gx0 + gx1) / 2, y_bracket - 0.014, gname, transform=trans,
                     ha="center", va="top", fontsize=13, color="0.2", clip_on=False)

        y_fig = y_bracket - 0.09
        for ax in (ax1, ax2, ax3):
            ax.xaxis.set_label_coords(0.5, (y_fig - ax.get_position().y0) / ax.get_position().height)

        # Significance legend in bottom-right, beneath the xlabel
        ax3_pos = ax3.get_position()
        # The stacked variant's panel-C label wraps to two lines, so the
        # significance note needs to clear the second one.
        fig.text(ax3_pos.x1, y_fig - (0.085 if joint_style == "stacked"
                                      else 0.04), r"$*$: Bonferroni $p \leq 0.05$",
                 ha="right", va="top", fontsize=12, color="gray")

        # Panel letters, stacked variant only: (A) pre-labeled, (B) learned,
        # (C) the two combined.
        for letter, ax in ((("A", ax1), ("B", ax2), ("C", ax3))
                           if joint_style == "stacked" else ()):
            pos = ax.get_position()
            fig.text(pos.x0, pos.y1 + 0.02, letter, ha="left", va="bottom",
                     fontsize=18, fontweight="bold")

        suffix = "" if joint_style == "grouped" else "_stacked"
        if readout == "total":
            suffix += "_total_variance"
        ci_tag = "_ci" if ci is not None else ""
        save_path = os.path.join(
            RESULTS_BEHAVIORAL_DIR,
            f'explained_variance_test_50vids_spearman_emb_shared_total{suffix}{perm_tag}{ci_tag}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        fig.savefig(save_path.replace('.png', '.pdf'), format='pdf',
                    bbox_inches='tight', pad_inches=0.5)
        print(f"Saved: {save_path}")
        plt.close(fig)

    if ci is not None:
        _render("stacked", readout="total")   # the paper's Fig. 2 layout only
        return
    _render("grouped")
    _render("stacked")
    _render("stacked", readout="total")


if __name__ == "__main__":
    main()
