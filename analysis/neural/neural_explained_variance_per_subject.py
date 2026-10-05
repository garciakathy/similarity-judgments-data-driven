"""
Per-subject version of `neural_behavioral_joint_variance_spearman[_filtered]`.

`neural_explained_variance.py` averages the brain RSMs across the 4 subjects
before regressing.
This script runs the identical Analysis-2 computation (cross-validated Spearman
r^2 of behavioral features vs learned embedding dimensions predicting an ROI
RSM) against each subject's *own* RSM, and emits:

  neural_behavioral_joint_variance_spearman_per_subject.{png,pdf,csv}
  neural_behavioral_joint_variance_spearman_per_subject_filtered.{png,pdf}
  neural_behavioral_joint_variance_spearman_subjmean[_filtered].{png,pdf}

The per-subject panels carry NO noise-ceiling band. The Nili bounds used in the
group figure are ceilings for the *group-average* RSM and do not bound a single
subject's. A dashed line marks each subject's leave-one-out between-subject
consistency (Spearman r^2 with the mean of the other subjects) -- that is a
reference for how much of this subject's RSM is shared with the group, not a
reliability ceiling.

--metric r plots cross-validated Spearman r instead of r^2 (same CV folds and
seeds, just not squared), with the Nili bounds and LOO line also left in r.
Its outputs insert "_r" after "spearman" (e.g.
neural_behavioral_joint_variance_spearman_r_subjmean_filtered.png); the default
r^2 outputs and CSV columns are unchanged.

--split train-test fits the regression weights on the 200 TRAIN videos'
neural RSMs (19,900 pairs) and scores them, without refitting, on the 50 test
videos' RSMs (1,225 pairs), a stimulus-level split. The default within-test mode
cross-validates over pairs of the 50 test videos, so every held-out pair's
videos are also in training pairs; predictors concentrated on a few videos (17
of the 40 embedding dims put > 90% of their RDM variance in per-video terms)
can then learn each video's RSM row, and 50 stimulus-free video-identity
regressors reach r = .75-.87. Train-test outputs insert "_traintest" after the
metric tag and never overwrite the within-test files. Train-video RSMs are
built from FMRI_EXPORT_DIR (config) and cached as
rsms/brain_rsm_train_sub{N}_{ROI}.npy. Embedding-dim RDMs are min-max scaled by
the max over all 250 videos so the predictor has the same scale when fit and
when scored (within-test scales by the 50-video max, as before). The CSV also
carries the 11 BEST_DIMS model as a third column.
"""

import argparse

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, ttest_rel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    UTILS_DIR, MODEL_DIR, RESULTS_NEURAL_DIR, RSM_DIR,
    BEHAVIOR_COLS, ROI_ORDER, SUBJECTS, test_idx, train_idx, RANDOM_SEED,
    BEST_DIMS, FMRI_EXPORT_DIR,
)
from helpers.variance_partition import (
    cv_spearman_r2, build_behavior_rsms,
)
from analysis.neural.neural_explained_variance import noise_ceiling_bounds
from analysis.neural.neural_rsa import load_precomputed_brain_rsms, build_brain_rsms
from helpers.action_categories import load_action_categories, build_action_rsm

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

EXCLUDE_ROIS = {'face-pSTS', 'FFA', 'PPA'}   # same filter as the group figure
BEH_COLOR, EMB_COLOR = 'tab:purple', 'tab:blue'

# Set in main() from --metric: value columns, axis label, file-name tag
METRIC = 'r2'
COLS = ('Behavioral_R2', 'Embedding_R2')
YLABEL = "Spearman r²"


def _cv_spearman_r(X, y, n_folds=10, n_repeats=5, seed=RANDOM_SEED):
    """cv_spearman_r2 without the square: mean signed Spearman r between
    out-of-fold OLS predictions and y, over the same repeated K-fold splits."""
    rs = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=kf)
        rs.append(spearmanr(y, y_pred)[0])
    return np.mean(rs)


def train_test_score(X_train, y_train, X_test, y_test, squared=True):
    """OLS fit on train-video pairs, Spearman r (or r^2) on test-video pairs."""
    y_pred = LinearRegression().fit(X_train, y_train).predict(X_test)
    r = spearmanr(y_test, y_pred)[0]
    return r ** 2 if squared else r


def load_train_brain_rsms(subjects, rois):
    """200x200 ROI RSMs for the train videos, cached next to the test ones."""
    paths = {(s, r): os.path.join(RSM_DIR, f'brain_rsm_train_sub{s}_{r}.npy')
             for s in subjects for r in rois}
    if not all(os.path.exists(p) for p in paths.values()):
        print(f"Building train-video brain RSMs from {FMRI_EXPORT_DIR} ...")
        built = build_brain_rsms(os.path.join(FMRI_EXPORT_DIR, 'metadata.csv'),
                                 os.path.join(FMRI_EXPORT_DIR, 'response_data.csv.gz'),
                                 subjects, test_idx=list(train_idx))
        for (s, r), p in paths.items():
            if r in built[str(s)]:
                np.save(p, built[str(s)][r])
    out = {}
    for (s, r), p in paths.items():
        if os.path.exists(p):
            out.setdefault(str(s), {})[r] = np.load(p)
    return out


def predictor_pairs(idx, annotations, action_labels, emb, dim_max):
    """Behavioral (13) and embedding-dim RDM predictors over pairs of `idx`."""
    idx = np.asarray(idx)
    tri = np.triu_indices(len(idx), k=1)
    rsms_full = build_behavior_rsms(annotations, BEHAVIOR_COLS)
    beh = {f.title(): rsms_full[f][np.ix_(idx, idx)][tri] for f in rsms_full}
    beh['Action Categories'] = build_action_rsm(action_labels[idx])[tri]
    e = emb[idx]
    embd = {f"Dim {d}": 1 - pdist(e[:, [d]], metric="euclidean") / dim_max[d]
            if dim_max[d] != 0 else 1 - pdist(e[:, [d]], metric="euclidean")
            for d in range(e.shape[1])}
    return pd.DataFrame(beh), pd.DataFrame(embd)


def loo_consistency(brain_rsms, subj, roi, triu, squared=True):
    """Spearman r^2 (or r) between this subject's ROI RSM and the mean of the others."""
    others = [brain_rsms[s][roi] for s in brain_rsms
              if s != subj and roi in brain_rsms[s]]
    if not others:
        return np.nan
    r, _ = spearmanr(brain_rsms[subj][roi][triu], np.mean(others, axis=0)[triu])
    return r ** 2 if squared else r


def plot_per_subject(df, subjects, rois, save_path, loo=None):
    """One panel per subject: grouped behavioral/embedding bars across ROIs."""
    n = len(subjects)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.2 * nrow),
                             squeeze=False)
    ymax = max(df[COLS[0]].max(), df[COLS[1]].max())
    if loo is not None and len(loo):
        ymax = max(ymax, np.nanmax(list(loo.values())))

    for ax, subj in zip(axes.ravel(), subjects):
        sub = df[df['Subject'] == subj].set_index('ROI').reindex(rois).dropna(how='all')
        present = list(sub.index)
        x = np.arange(len(present))
        w = 0.35
        ax.bar(x - w / 2, sub[COLS[0]].values, w, label='Behavioral Features',
               color=BEH_COLOR, edgecolor='black', linewidth=0.8)
        ax.bar(x + w / 2, sub[COLS[1]].values, w, label='Learned Dimensions',
               color=EMB_COLOR, edgecolor='black', linewidth=0.8)
        if loo is not None:
            for i, roi in enumerate(present):
                v = loo.get((subj, roi), np.nan)
                if np.isfinite(v):
                    ax.plot([x[i] - w - 0.05, x[i] + w + 0.05], [v, v],
                            color='dimgray', linestyle='--', linewidth=1.2, zorder=3)
        ax.set_title(f'Subject {subj}', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(present, fontsize=10)
        ax.set_ylabel(YLABEL, fontsize=12)
        ax.set_ylim(min(0, df[list(COLS)].min().min() * 1.15), ymax * 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if loo is not None:
        handles.append(plt.Line2D([0], [0], color='dimgray', linestyle='--', lw=1.2))
        labels.append('Between-subject consistency (LOO)')
    fig.legend(handles, labels, fontsize=11, ncol=len(labels),
               loc='lower center', frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_subject_mean(df, rois, save_path, ceilings=None):
    """Mean across subjects + individual subject dots, over the Nili ceiling band.

    The Nili bounds are single-subject-referenced (mean over subjects of
    r(subject_i, mean-of-others / mean-of-all), squared), so they bound a
    per-subject score directly. Note they are mean(r)**2 while the bars are
    mean(r**2); by Jensen the band is marginally conservative.
    """
    g = df[df['ROI'].isin(rois)]
    rois = [r for r in rois if r in set(g['ROI'])]
    x = np.arange(len(rois))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    for off, col, color, label in [
        (-w / 2, COLS[0], BEH_COLOR, 'Pre-labeled Annotations'),
        (+w / 2, COLS[1], EMB_COLOR, 'Learned Dimensions'),
    ]:
        m = [g.loc[g['ROI'] == r, col].mean() for r in rois]
        ax.bar(x + off, m, w, label=label,
               color=color, edgecolor='black', linewidth=0.8, zorder=2)
        for i, r in enumerate(rois):
            vals = g.loc[g['ROI'] == r, col].values
            ax.scatter(np.full(len(vals), x[i] + off), vals, s=14,
                       color='black', alpha=0.6, zorder=4)

    # Noise ceiling band (Nili et al. 2014), same style as the group figure
    bar_half = w + 0.05
    if ceilings:
        for i, roi in enumerate(rois):
            if roi in ceilings:
                lo, hi = ceilings[roi]
                ax.fill_between([x[i] - bar_half, x[i] + bar_half], lo, hi,
                                color='lightgray', alpha=0.7, zorder=0,
                                edgecolor='none')

    ax.set_xlabel("Neural ROI", fontsize=14)
    ax.set_ylabel(YLABEL, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(rois, fontsize=12)
    ax.legend(fontsize=12, ncol=2, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ceilings:
        vis = [ceilings[r][1] for r in rois if r in ceilings]
        if vis:
            ax.set_ylim(min(0, g[list(COLS)].min().min() * 1.15), max(vis) + 0.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    global METRIC, COLS, YLABEL
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', choices=['r2', 'r'], default='r2',
                    help="r2 (default, the published figures) or r: unsquared "
                         "cross-validated Spearman r, ceiling and LOO in r too.")
    ap.add_argument('--split', choices=['within-test', 'train-test'], default='within-test',
                    help="within-test (default, the published figures): CV over pairs of "
                         "the 50 test videos. train-test: fit on the 200 train videos' "
                         "RSMs, score on the 50 test videos'.")
    args = ap.parse_args()
    METRIC = args.metric
    if METRIC == 'r':
        COLS, YLABEL = ('Behavioral_r', 'Embedding_r'), "Spearman r"
    squared = METRIC == 'r2'
    score = cv_spearman_r2 if squared else _cv_spearman_r
    tag = ('' if squared else '_r') + ('_traintest' if args.split == 'train-test' else '')
    out = lambda name: os.path.join(RESULTS_NEURAL_DIR, name.replace('_spearman', '_spearman' + tag, 1))

    if test_idx is None:
        print("test_idx unavailable (data/utils/test.csv missing)")
        return
    t_idx = np.asarray(test_idx)
    n_test = len(t_idx)
    triu = np.triu_indices(n_test, k=1)

    brain_rsms = load_precomputed_brain_rsms(RSM_DIR, SUBJECTS, ROI_ORDER)
    if not brain_rsms:
        print(f"No precomputed brain RSMs in {RSM_DIR}")
        return
    print(f"Loaded per-subject RSMs for subjects: {sorted(brain_rsms)}")

    # --- predictors: identical to neural_explained_variance.py Analysis 2 ---
    annotations = pd.read_csv(os.path.join(UTILS_DIR, 'annotations.csv'))
    rsms_full = build_behavior_rsms(annotations, BEHAVIOR_COLS)
    beh_features = {f.title(): rsms_full[f][np.ix_(t_idx, t_idx)][triu]
                    for f in rsms_full}
    labels_test = load_action_categories(UTILS_DIR)[t_idx]
    beh_features['Action Categories'] = build_action_rsm(labels_test)[triu]
    beh_df = pd.DataFrame(beh_features)

    emb = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))[t_idx, :]
    emb_features = {}
    for d in range(emb.shape[1]):
        dist = pdist(emb[:, [d]], metric="euclidean")
        dmax = dist.max()
        emb_features[f"Dim {d}"] = 1 - (dist / dmax) if dmax != 0 else 1 - dist
    emb_df = pd.DataFrame(emb_features)
    print(f"Predictors: {beh_df.shape[1]} behavioral, {emb_df.shape[1]} embedding "
          f"dims over {len(triu[0])} pairs")

    rows, loo = [], {}
    if args.split == 'train-test':
        emb_all = np.load(os.path.join(MODEL_DIR, 'sparse_pruned_emb_sorted.npy'))
        labels_all = load_action_categories(UTILS_DIR)
        dim_max = np.array([pdist(emb_all[:, [d]]).max() for d in range(emb_all.shape[1])])
        beh_tr, emb_tr = predictor_pairs(train_idx, annotations, labels_all, emb_all, dim_max)
        beh_te, emb_te = predictor_pairs(t_idx, annotations, labels_all, emb_all, dim_max)
        best = [f"Dim {d}" for d in BEST_DIMS]
        train_rsms = load_train_brain_rsms(SUBJECTS, ROI_ORDER)
        triu_tr = np.triu_indices(len(train_idx), k=1)
        print(f"Train-test split: fit on {len(triu_tr[0])} train-video pairs, "
              f"score on {len(triu[0])} test-video pairs")
    for subj in sorted(brain_rsms, key=int):
        for roi in ROI_ORDER:
            if roi not in brain_rsms[subj]:
                continue
            y = brain_rsms[subj][roi][triu]
            if args.split == 'train-test':
                y_tr = train_rsms[subj][roi][triu_tr]
                r2_beh = train_test_score(beh_tr.values, y_tr, beh_te.values, y, squared)
                r2_emb = train_test_score(emb_tr.values, y_tr, emb_te.values, y, squared)
                r2_e11 = train_test_score(emb_tr[best].values, y_tr, emb_te[best].values, y, squared)
            else:
                r2_beh = score(beh_df.values, y)
                r2_emb = score(emb_df.values, y)
                r2_e11 = score(emb_df[[f"Dim {d}" for d in BEST_DIMS]].values, y)
            loo[(subj, roi)] = loo_consistency(brain_rsms, subj, roi, triu, squared)
            rows.append({'Subject': subj, 'ROI': roi,
                         COLS[0]: r2_beh, COLS[1]: r2_emb,
                         COLS[1].replace('Embedding', 'Embedding11'): r2_e11,
                         'LOO_consistency_r2' if squared else 'LOO_consistency_r': loo[(subj, roi)]})
            print(f"  sub{subj} {roi}: beh={r2_beh:.4f} emb={r2_emb:.4f} emb11={r2_e11:.4f} "
                  f"loo={loo[(subj, roi)]:.4f}")

    df = pd.DataFrame(rows)
    csv_path = out('neural_behavioral_joint_variance_spearman_per_subject.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    subjects = sorted(df['Subject'].unique(), key=int)
    rois_all = [r for r in ROI_ORDER if r in set(df['ROI'])]
    rois_filt = [r for r in rois_all if r not in EXCLUDE_ROIS]

    plot_per_subject(df, subjects, rois_all,
                     out('neural_behavioral_joint_variance_spearman_per_subject.png'), loo=loo)
    plot_per_subject(df, subjects, rois_filt,
                     out('neural_behavioral_joint_variance_spearman_per_subject_filtered.png'), loo=loo)
    ceilings = noise_ceiling_bounds(brain_rsms, ROI_ORDER, squared=squared)
    print("\nNoise ceiling bounds (Nili et al. 2014):")
    for roi, (lo, hi) in ceilings.items():
        print(f"  {roi}: lower={lo:.4f}, upper={hi:.4f}")

    plot_subject_mean(df, rois_all,
                      out('neural_behavioral_joint_variance_spearman_subjmean.png'), ceilings=ceilings)
    plot_subject_mean(df, rois_filt,
                      out('neural_behavioral_joint_variance_spearman_subjmean_filtered.png'),
                      ceilings=ceilings)

    # --- paired embedding-vs-behavioral test across subjects, per ROI ---
    print("\nPaired t-test across subjects (Embedding − Behavioral), n="
          f"{len(subjects)} subjects — descriptive only at this n:")
    for roi in rois_all:
        s = df[df['ROI'] == roi]
        if len(s) < 3:
            continue
        d = s[COLS[1]].values - s[COLS[0]].values
        t, p = ttest_rel(s[COLS[1]], s[COLS[0]])
        print(f"  {roi}: Δ={d.mean():+.4f} (SEM {d.std(ddof=1)/np.sqrt(len(d)):.4f}), "
              f"t={t:.2f}, p={p:.3f}")


if __name__ == '__main__':
    main()
