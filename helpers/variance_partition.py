"""Cross-validated RSA regression and unique/shared variance partitioning.

Each predictor is an RSM; the target is a similarity RSM (behavioral or neural).
Scores are repeated K-fold OLS, evaluated as R^2 (coefficient of determination)
or as squared Spearman r of the out-of-fold predictions.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold

from config import FEATURE_ORDER, FEATURE_REMAPS, RANDOM_SEED


def cv_r2(X, y, n_folds=10, n_repeats=5, seed=RANDOM_SEED):
    """Cross-validated R² (repeated K-fold)."""
    scores = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        scores.extend(cross_val_score(
            LinearRegression(), X, y, cv=kf, scoring='r2'
        ))
    return np.mean(scores)


def cv_spearman_r2(X, y, n_folds=10, n_repeats=5, seed=RANDOM_SEED):
    """Cross-validated Spearman r² using out-of-fold predictions.

    Fits OLS via repeated K-fold CV, collects out-of-fold predicted values,
    then computes Spearman r between predictions and actual, squared.
    """
    r2_vals = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        y_pred = cross_val_predict(LinearRegression(), X, y, cv=kf)
        r, _ = spearmanr(y, y_pred)
        r2_vals.append(r ** 2)
    return np.mean(r2_vals)


def compute_explained_variance(y, feature_df):
    """Compute total, unique, and shared R2 for a set of features (cross-validated)."""
    X = feature_df.values
    r2_full = cv_r2(X, y)

    unique_vars = {}
    for col in feature_df.columns:
        X_wo = feature_df.drop(columns=[col]).values
        r2_wo = cv_r2(X_wo, y)
        unique_vars[col] = r2_full - r2_wo

    shared = r2_full - sum(unique_vars.values())
    return r2_full, unique_vars, shared


def compute_explained_variance_spearman(y, feature_df):
    """Compute total, unique, and shared Spearman r² (cross-validated).

    Same variance partitioning logic as compute_explained_variance but uses
    Spearman r² (rank-order) instead of OLS R² as the evaluation metric.
    """
    X = feature_df.values
    r2_full = cv_spearman_r2(X, y)

    unique_vars = {}
    for col in feature_df.columns:
        X_wo = feature_df.drop(columns=[col]).values
        r2_wo = cv_spearman_r2(X_wo, y)
        unique_vars[col] = r2_full - r2_wo

    shared = r2_full - sum(unique_vars.values())
    return r2_full, unique_vars, shared


def build_behavior_rsms(annotations, feature_cols):
    """Build per-feature RSMs from behavioral annotations."""
    rsms = {}
    for feature in feature_cols:
        feature_matrix = annotations[[feature]].to_numpy()
        feature_rsm = 1 - pairwise_distances(feature_matrix, metric='euclidean')
        rsms[feature] = feature_rsm
    return rsms


def format_behavior_df(unique_vars, shared, r2_full):
    """Format behavioral variance results with renaming and ordering."""
    unique_vars["Behavior Shared"] = shared
    unique_vars["Behavior Total"] = r2_full

    df = (
        pd.DataFrame(unique_vars, index=["Variance"])
        .T.reset_index()
        .rename(columns={"index": "Feature"})
    )
    df['Feature'] = df['Feature'].replace(FEATURE_REMAPS)
    df["Feature"] = pd.Categorical(df["Feature"], categories=FEATURE_ORDER, ordered=True)
    df = df.sort_values("Feature")
    df["Feature"] = df["Feature"].astype(str)
    df = df.dropna(subset=["Variance"])
    return df


def get_bar_colors(feature_names, base_color="tab:blue"):
    """Generate colors for bar chart (gray for shared/total)."""
    colors = []
    for feat in feature_names:
        if "Shared" in feat:
            colors.append("lightgray")
        elif "Total" in feat:
            colors.append("darkgray")
        else:
            colors.append(base_color)
    return colors
