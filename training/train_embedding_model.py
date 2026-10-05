"""
Train triplet embedding model with sparse, non-negative constraints.
Based on Hebart methodology for interpretable similarity judgments.
Trains on TRAIN triplets only.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import random
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RANDOM_SEED, EMBEDDING_DIM, SPARSE_LAMBDA, PRUNE_THRESHOLD,
    N_EPOCHS, PATIENCE, LEARNING_RATE, BATCH_SIZE, ENFORCE_POSITIVITY,
    N_VIDEOS, UTILS_DIR, MODEL_DIR
)
from helpers.data_helpers import (
    load_and_process_triplet_data,
    train_video_embedding_model,
    log_embedding_run_header,
    plot_embedding_analysis_sorted
)

# ============================================================================
# SETUP
# ============================================================================

ap = argparse.ArgumentParser()
ap.add_argument('--out-dir', default=MODEL_DIR,
                help='where to write model outputs (default: the published model dir)')
ap.add_argument('--lambda', dest='sparse_lambda', type=float, default=SPARSE_LAMBDA)
ap.add_argument('--seed', type=int, default=RANDOM_SEED)
ap.add_argument('--overwrite', action='store_true',
                help='allow writing over an existing model in --out-dir')
args = ap.parse_args()
SPARSE_LAMBDA, RANDOM_SEED = args.sparse_lambda, args.seed

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

OUT_DIR = os.path.abspath(args.out_dir)
if not args.overwrite and any(
        os.path.exists(os.path.join(OUT_DIR, f))
        for f in ('pruned_embeddings.npy', 'sparse_pruned_emb_sorted.npy')):
    sys.exit(f"{OUT_DIR} already holds a model; pass --out-dir elsewhere or --overwrite")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "=" * 50)
print("LOAD TRIPLET DATA")
print("=" * 50)

# train_triplets.npy / val_triplets.npy are the 90/10 train/validation split
# of the 49,484 training-experiment triplets. Column 3 is already
# chosen_pair_idx (0 -> ij, 1 -> ik, 2 -> jk), NOT the odd-one-out id, so it is
# used as-is -- routing it through load_and_process_triplet_data would read it
# as a stimulus id and discard almost every triplet.
train_npy = os.path.join(UTILS_DIR, 'train_triplets.npy')
val_npy = os.path.join(UTILS_DIR, 'val_triplets.npy')
train_triplets = [tuple(int(v) for v in row) for row in np.load(train_npy)]
val_triplets = [tuple(int(v) for v in row) for row in np.load(val_npy)]
for name, trips in (('train', train_triplets), ('val', val_triplets)):
    assert {t[3] for t in trips} <= {0, 1, 2}, f"{name}: column 3 is not chosen_pair_idx"
print(f"Loaded {len(train_triplets)} train and {len(val_triplets)} validation triplets")
test_triplets = val_triplets

print(f"\nTraining on {len(train_triplets)} train triplets only (NOT combined)")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

log_embedding_run_header(
    phase_title="Training Triplet Embedding Model (Train Only)",
    device=device,
    n_dims=EMBEDDING_DIM,
    sparse_lambda=SPARSE_LAMBDA,
    prune_threshold=PRUNE_THRESHOLD,
    enforce_positivity=ENFORCE_POSITIVITY,
    learning_rate=LEARNING_RATE,
    n_epochs=N_EPOCHS,
    patience=PATIENCE
)

# ============================================================================
# TRAIN MODEL
# ============================================================================

(
    sparse_model,
    sparse_pruned_emb,
    sparse_full_embed,
    sparse_keep_mask,
    sparse_kept_indices,
    sparse_old_to_new,
    sparse_new_to_old,
    sparse_train_losses,
    sparse_test_losses,
    sparse_train_accuracies,
    sparse_test_accuracies,
    sparse_k_prune
) = train_video_embedding_model(
    triplets=train_triplets,  # Train on train triplets ONLY
    sparse_lambda=SPARSE_LAMBDA,
    n_videos=N_VIDEOS,
    embedding_dim=EMBEDDING_DIM,
    n_epochs=N_EPOCHS,
    prune_threshold=PRUNE_THRESHOLD,
    enforce_positivity=ENFORCE_POSITIVITY,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    patience=PATIENCE,
    random_seed=RANDOM_SEED,
    device=device,
    val_triplets=val_triplets
)

print(f"\nKept {sparse_k_prune}/{EMBEDDING_DIM} dims.")
print("Example old->new mapping (first 10):",
      dict(list(sparse_old_to_new.items())[:min(10, len(sparse_old_to_new))]))

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "=" * 50)
print("SAVING MODEL OUTPUTS")
print("=" * 50)

np.save(os.path.join(OUT_DIR, 'pruned_embeddings.npy'), sparse_pruned_emb)
np.save(os.path.join(OUT_DIR, 'full_embeddings.npy'), sparse_full_embed)
np.save(os.path.join(OUT_DIR, 'keep_mask.npy'), sparse_keep_mask)
np.save(os.path.join(OUT_DIR, 'kept_indices.npy'), sparse_kept_indices)

with open(os.path.join(OUT_DIR, 'dimension_mappings.json'), 'w') as f:
    json.dump({
        'old_to_new': sparse_old_to_new,
        'new_to_old': sparse_new_to_old,
        'k_pruned': sparse_k_prune,
        'original_dims': EMBEDDING_DIM
    }, f, indent=2)

torch.save(sparse_model.state_dict(), os.path.join(OUT_DIR, 'model_state.pt'))

history_df = pd.DataFrame({
    'train_loss': sparse_train_losses,
    'test_loss': sparse_test_losses,
    'train_accuracy': sparse_train_accuracies,
    'test_accuracy': sparse_test_accuracies
})
history_df.to_csv(os.path.join(OUT_DIR, 'training_history.csv'), index=False)

# Save training summary
with open(os.path.join(OUT_DIR, 'training_summary.txt'), 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("TRIPLET EMBEDDING MODEL - TRAINING SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("HYPERPARAMETERS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Device                    : {device}\n")
    f.write(f"Number of videos          : {N_VIDEOS}\n")
    f.write(f"Embedding dimensions      : {EMBEDDING_DIM}\n")
    f.write(f"Sparsity lambda           : {SPARSE_LAMBDA}\n")
    f.write(f"Prune threshold           : {PRUNE_THRESHOLD}\n")
    f.write(f"Enforce positivity        : {ENFORCE_POSITIVITY}\n")
    f.write(f"Learning rate             : {LEARNING_RATE}\n")
    f.write(f"Max epochs                : {N_EPOCHS}\n")
    f.write(f"Early stopping patience   : {PATIENCE}\n")
    f.write(f"Batch size                : {BATCH_SIZE}\n")
    f.write(f"Random seed               : {RANDOM_SEED}\n\n")

    f.write("DATASET\n")
    f.write("-" * 70 + "\n")
    f.write(f"Training triplets         : {len(train_triplets)}\n")
    f.write(f"Validation triplets       : {len(val_triplets)}\n")
    f.write(f"NOTE: Trained on train triplets ONLY\n\n")

    f.write("TRAINING RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Epochs completed          : {len(sparse_train_losses)}\n")
    f.write(f"Final train loss          : {sparse_train_losses[-1]:.6f}\n")
    f.write(f"Final train accuracy      : {sparse_train_accuracies[-1]:.6f} ({sparse_train_accuracies[-1]*100:.2f}%)\n")
    f.write(f"Final val loss            : {sparse_test_losses[-1]:.6f}\n")
    f.write(f"Final val accuracy        : {sparse_test_accuracies[-1]:.6f} ({sparse_test_accuracies[-1]*100:.2f}%)\n")
    f.write(f"Best val accuracy         : {max(sparse_test_accuracies):.6f} ({max(sparse_test_accuracies)*100:.2f}%)\n")
    f.write(f"Best val accuracy epoch   : {sparse_test_accuracies.index(max(sparse_test_accuracies)) + 1}\n\n")

    f.write("DIMENSION PRUNING\n")
    f.write("-" * 70 + "\n")
    f.write(f"Original dimensions       : {EMBEDDING_DIM}\n")
    f.write(f"Pruned dimensions         : {sparse_k_prune}\n")
    f.write(f"Dimensions kept           : {sparse_k_prune}/{EMBEDDING_DIM} ({sparse_k_prune/EMBEDDING_DIM*100:.1f}%)\n")
    f.write(f"Dimensions removed        : {EMBEDDING_DIM - sparse_k_prune}/{EMBEDDING_DIM} ({(EMBEDDING_DIM-sparse_k_prune)/EMBEDDING_DIM*100:.1f}%)\n\n")

    f.write("KEPT DIMENSION INDICES (Original numbering)\n")
    f.write("-" * 70 + "\n")
    f.write(f"{sparse_kept_indices.tolist()}\n\n")

    f.write("DIMENSION MAPPING (Old -> New)\n")
    f.write("-" * 70 + "\n")
    for old_idx in sorted(sparse_old_to_new.keys()):
        new_idx = sparse_old_to_new[old_idx]
        f.write(f"Dim {old_idx:2d} -> Dim {new_idx:2d}\n")
    f.write("\n")

    f.write("TRAINING HISTORY (All Epochs)\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Epoch':>6} | {'Train Loss':>11} | {'Train Acc':>10} | {'Val Loss':>11} | {'Val Acc':>10}\n")
    f.write("-" * 70 + "\n")
    for i in range(len(sparse_train_losses)):
        f.write(f"{i+1:6d} | {sparse_train_losses[i]:11.6f} | {sparse_train_accuracies[i]:10.6f} | "
                f"{sparse_test_losses[i]:11.6f} | {sparse_test_accuracies[i]:10.6f}\n")

print(f"\nSaved to {OUT_DIR}:")
print("  - pruned_embeddings.npy")
print("  - full_embeddings.npy")
print("  - keep_mask.npy")
print("  - kept_indices.npy")
print("  - dimension_mappings.json")
print("  - model_state.pt")
print("  - training_history.csv")
print("  - training_summary.txt")

# ============================================================================
# GENERATE VISUALIZATION
# ============================================================================

print("\n" + "=" * 50)
print("GENERATING EMBEDDING VISUALIZATION")
print("=" * 50)

try:
    ordered_dims, sparse_pruned_emb_sorted = plot_embedding_analysis_sorted(
        full_embed=sparse_full_embed,
        pruned_embed=sparse_pruned_emb,
        sparse_lambda=SPARSE_LAMBDA,
        prune_threshold=PRUNE_THRESHOLD
    )
    np.save(os.path.join(OUT_DIR, 'sparse_pruned_emb_sorted.npy'), sparse_pruned_emb_sorted)
    print(f"Saved sorted embeddings to {OUT_DIR}/sparse_pruned_emb_sorted.npy")
except Exception as e:
    print(f"Warning: Could not generate visualization: {e}")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)
