"""
Split-half reliability calculation using triplet-level splitting.
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import UTILS_DIR, RANDOM_SEED, N_VIDEOS

def build_rsm_from_triplets(triplet_df, n_videos=250):
    """
    Build a similarity RSM from triplet odd-one-out judgments.

    For each triplet (i, j, k, choice):
    - If choice == i: j and k are similar (i is odd)
    - If choice == j: i and k are similar (j is odd)
    - If choice == k: i and j are similar (k is odd)

    Similarity = (times chosen as similar) / (total times pair appeared)

    Parameters:
    -----------
    triplet_df : pd.DataFrame
        Must have columns: stim1_name, stim2_name, stim3_name, choice
    n_videos : int
        Total number of videos (for matrix size)

    Returns:
    --------
    sim_matrix : np.ndarray (n_videos x n_videos)
        Similarity matrix where sim[i,j] = proportion of times i,j were chosen as similar
    count_matrix : np.ndarray (n_videos x n_videos)
        Count of how many triplets each pair appeared in
    """
    # Initialize matrices
    similar_counts = np.zeros((n_videos, n_videos))
    total_counts = np.zeros((n_videos, n_videos))

    # Process each triplet
    for _, row in triplet_df.iterrows():
        i = int(row['stim1_name'])
        j = int(row['stim2_name'])
        k = int(row['stim3_name'])
        choice = int(row['choice'])

        # Skip triplets that repeat a stimulus, as build_rsms_and_counts does
        if len({i, j, k}) != 3:
            continue

        # All three pairs appear in this triplet
        pairs = [(i, j), (i, k), (j, k)]

        # Increment total counts for all pairs
        for p1, p2 in pairs:
            total_counts[p1, p2] += 1
            total_counts[p2, p1] += 1

        # Determine which pair was chosen as similar (non-odd)
        if choice == i:
            # i is odd, so j and k are similar
            similar_counts[j, k] += 1
            similar_counts[k, j] += 1
        elif choice == j:
            # j is odd, so i and k are similar
            similar_counts[i, k] += 1
            similar_counts[k, i] += 1
        elif choice == k:
            # k is odd, so i and j are similar
            similar_counts[i, j] += 1
            similar_counts[j, i] += 1

    # Compute similarity as proportion
    sim_matrix = np.zeros((n_videos, n_videos))
    valid_mask = total_counts > 0
    sim_matrix[valid_mask] = similar_counts[valid_mask] / total_counts[valid_mask]

    # Set diagonal to 1.0 (self-similarity)
    np.fill_diagonal(sim_matrix, 1.0)

    return sim_matrix, total_counts


def calculate_split_half_reliability_triplets(triplet_df, n_videos=250, n_iterations=1000,
                                               min_observations=1, seed=RANDOM_SEED,
                                               verbose=True):
    """
    Calculate split-half reliability by splitting at the triplet level.

    1. Randomly split triplets into two halves
    2. Build RSM from each half
    3. Correlate the two RSMs (only on pairs with data in both)
    4. Apply Spearman-Brown correction

    Parameters:
    -----------
    triplet_df : pd.DataFrame
        Triplet judgments with columns: stim1_name, stim2_name, stim3_name, choice
    n_videos : int
        Number of videos
    n_iterations : int
        Number of random splits to average over
    min_observations : int
        Minimum number of triplet observations per pair to include in correlation
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print diagnostics

    Returns:
    --------
    split_half_reliability : float
        Spearman-Brown corrected split-half reliability (noise ceiling)
    """
    rng = np.random.RandomState(seed)
    n_triplets = len(triplet_df)

    if verbose:
        print(f"\n{'='*60}")
        print("SPLIT-HALF RELIABILITY CALCULATION (TRIPLET-LEVEL SPLIT)")
        print(f"{'='*60}")
        print(f"Total triplets: {n_triplets}")
        print(f"Videos: {n_videos}")
        print(f"Iterations: {n_iterations}")
        print(f"Min observations per pair: {min_observations}")
        print(f"Seed: {seed}")
        print()

    correlations = []
    n_valid_pairs_list = []

    for i in tqdm(range(n_iterations), desc="Computing split-half reliability"):
        # Randomly split triplets into two halves
        indices = np.arange(n_triplets)
        rng.shuffle(indices)
        mid = n_triplets // 2

        half1_indices = indices[:mid]
        half2_indices = indices[mid:]

        triplets_half1 = triplet_df.iloc[half1_indices]
        triplets_half2 = triplet_df.iloc[half2_indices]

        # Build RSMs from each half
        rsm1, counts1 = build_rsm_from_triplets(triplets_half1, n_videos=n_videos)
        rsm2, counts2 = build_rsm_from_triplets(triplets_half2, n_videos=n_videos)

        # Get upper triangle indices (to avoid double-counting and diagonal)
        triu_idx = np.triu_indices(n_videos, k=1)

        # Only include pairs with sufficient observations in BOTH halves
        valid_mask = (counts1[triu_idx] >= min_observations) & (counts2[triu_idx] >= min_observations)

        if valid_mask.sum() > 10:  # Need at least some pairs for correlation
            rsm1_vals = rsm1[triu_idx][valid_mask]
            rsm2_vals = rsm2[triu_idx][valid_mask]

            # Calculate Spearman correlation
            corr, _ = spearmanr(rsm1_vals, rsm2_vals)

            if not np.isnan(corr):
                correlations.append(corr)
                n_valid_pairs_list.append(valid_mask.sum())

    # Calculate mean correlation
    mean_correlation = np.mean(correlations)
    std_correlation = np.std(correlations)

    # Apply Spearman-Brown correction
    # Corrects for the fact that we're correlating half-length tests
    split_half_reliability = (2 * mean_correlation) / (1 + mean_correlation)

    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Valid iterations: {len(correlations)} / {n_iterations}")
        print(f"Mean valid pairs per iteration: {np.mean(n_valid_pairs_list):.0f}")
        print(f"Range of valid pairs: {np.min(n_valid_pairs_list):.0f} - {np.max(n_valid_pairs_list):.0f}")
        print()
        print(f"Raw correlation (half-length):")
        print(f"  Mean: {mean_correlation:.4f}")
        print(f"  Std:  {std_correlation:.4f}")
        print(f"  Range: [{np.min(correlations):.4f}, {np.max(correlations):.4f}]")
        print()
        print(f"Spearman-Brown corrected reliability (noise ceiling): {split_half_reliability:.4f}")
        print(f"{'='*60}")

        # Additional diagnostics on full RSM
        print("\nFull RSM diagnostics (using all triplets):")
        full_rsm, full_counts = build_rsm_from_triplets(triplet_df, n_videos=n_videos)
        triu_idx = np.triu_indices(n_videos, k=1)
        non_zero = full_counts[triu_idx] > 0

        print(f"  Total possible pairs: {len(triu_idx[0])}")
        print(f"  Pairs with data: {non_zero.sum()} ({100*non_zero.sum()/len(triu_idx[0]):.1f}%)")
        print(f"  Mean count (non-zero pairs): {full_counts[triu_idx][non_zero].mean():.2f}")
        print(f"  Median count (non-zero pairs): {np.median(full_counts[triu_idx][non_zero]):.0f}")
        print(f"  Max count: {full_counts[triu_idx].max():.0f}")
        print()
        print(f"  Similarity distribution (non-zero pairs):")
        sim_vals = full_rsm[triu_idx][non_zero]
        print(f"    Mean: {sim_vals.mean():.4f}")
        print(f"    Median: {np.median(sim_vals):.4f}")
        print(f"    Std: {sim_vals.std():.4f}")
        print(f"    Range: [{sim_vals.min():.4f}, {sim_vals.max():.4f}]")

        # Show distribution
        print("\n  Histogram of similarity values:")
        hist, bins = np.histogram(sim_vals, bins=10, range=(0, 1))
        for i in range(len(hist)):
            bar_width = int(50 * hist[i] / hist.max()) if hist.max() > 0 else 0
            bar = '█' * bar_width
            print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:5d} {bar}")

    return split_half_reliability, correlations


def _pair_idx_to_odd_one_out(arr):
    """
    Convert (i, j, k, chosen_pair_idx) -> (i, j, k, odd-one-out stim id).

    train_triplets.npy / all_triplets.npy store column 3 as chosen_pair_idx,
    which names the SIMILAR pair (0 -> (i,j), 1 -> (i,k), 2 -> (j,k)).
    build_rsm_from_triplets expects the odd one out as a stimulus id, so the
    remaining stimulus is what it needs. Feeding it the raw 0/1/2 codes makes
    every `choice == i/j/k` test fail and yields a near-empty RSM.
    """
    i, j, k, cp = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    odd = np.where(cp == 0, k, np.where(cp == 1, j, i))
    return np.column_stack([i, j, k, odd])


def _load_triplet(npy_path, parquet_path):
    """Load triplets from .npy or .parquet, whichever exists."""
    cols = ['stim1_name', 'stim2_name', 'stim3_name', 'choice']
    if os.path.exists(npy_path):
        print(f"Found: {npy_path}")
        arr = np.load(npy_path).astype(np.int64)
        if arr[:, 3].max() <= 2:
            print("  column 3 is chosen_pair_idx -> converting to odd-one-out stim id")
            arr = _pair_idx_to_odd_one_out(arr)
        return pd.DataFrame(arr, columns=cols)
    elif os.path.exists(parquet_path):
        print(f"Found: {parquet_path}")
        return pd.read_parquet(parquet_path)
    return None


def _load_test_triplets():
    """Load the separately-collected dense test triplets (50 test videos only)."""
    cols = ['stim1_name', 'stim2_name', 'stim3_name', 'choice']
    csv_path = os.path.join(UTILS_DIR, 'human_test_triplets.csv')
    if os.path.exists(csv_path):
        print(f"Found: {csv_path}")
        df = pd.read_csv(csv_path)
        return df[cols]
    return None


def main():
    """Run split-half reliability analysis on train, test, or all datasets."""

    parser = argparse.ArgumentParser(description="Split-half reliability for triplet data")
    parser.add_argument(
        '--dataset', choices=['train', 'test', 'all'], nargs='+',
        default=None,
        help="Which dataset(s) to run. Default: all available."
    )
    args = parser.parse_args()

    available = {
        'train': (os.path.join(UTILS_DIR, 'train_triplets.npy'),
                  os.path.join(UTILS_DIR, 'train_triplets.parquet')),
        'all':   (os.path.join(UTILS_DIR, 'all_triplets.npy'),
                  os.path.join(UTILS_DIR, 'all_triplets.parquet')),
    }

    requested = args.dataset if args.dataset else ['train', 'test', 'all']

    datasets = {}
    for name in requested:
        if name == 'test':
            df = _load_test_triplets()
        else:
            npy_path, parquet_path = available[name]
            df = _load_triplet(npy_path, parquet_path)
        if df is not None:
            datasets[name] = df

    if not datasets:
        print(f"\nERROR: No triplet files found for {requested}!")
        print(f"Expected files in {UTILS_DIR}:")
        print("  - train_triplets.npy or train_triplets.parquet")
        print("  - test_triplets.npy or test_triplets.parquet")
        print("  - all_triplets.npy or all_triplets.parquet")
        return

    # Run split-half reliability for each dataset
    results = {}

    for name, df in datasets.items():
        print(f"\n\n{'#'*70}")
        print(f"# DATASET: {name.upper()}")
        print(f"{'#'*70}")

        reliability, correlations = calculate_split_half_reliability_triplets(
            df,
            n_videos=N_VIDEOS,
            n_iterations=1000,
            min_observations=1,
            seed=RANDOM_SEED,
            verbose=True
        )

        results[name] = {
            'reliability': reliability,
            'correlations': correlations,
            'n_triplets': len(df)
        }

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY OF SPLIT-HALF RELIABILITIES (NOISE CEILINGS)")
    print(f"{'='*70}")

    for name, res in results.items():
        print(f"{name.upper():10s}: {res['reliability']:.4f}  (n={res['n_triplets']} triplets)")

    print(f"{'='*70}")
    print("\nNOTE: This is the corrected split-half reliability using triplet-level")
    print("splitting, which is the appropriate method for odd-one-out triplet data.")
    print("Previous pair-level approach was methodologically incorrect.")


if __name__ == "__main__":
    main()
