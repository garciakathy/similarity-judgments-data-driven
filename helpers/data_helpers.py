"""
Helper functions for triplet embedding model.
Includes raw data loading, RSM building, model definition, training, and visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import string
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================================
# Raw Meadows data loading and RSM construction
# ============================================================================

def load_sim_judg(raw_dir, remove_catch=True, remove_singles=False, remove_test=True,
                  train_idx=None):
    """
    Load raw Meadows triplet judgment data from CSV files, clean, and filter.

    Args:
        raw_dir: Path to directory containing raw Meadows CSV files
        remove_catch: Remove participants who failed catch trials
        remove_test: If True, keep only triplets where all 3 videos are in train_idx
        train_idx: List of video IDs for train set (required if remove_test=True)

    Returns:
        data: Cleaned DataFrame of triplet judgments
        flagged: List of participant IDs removed for failing catch trials
    """
    from dateutil.parser import isoparse as parse_iso_dt

    # Load all annotation batches
    extra_metadata = pd.read_csv(f'{raw_dir}/Meadows_Dynamic-Similarity-Judge-copy_v_v3_metadata.csv')
    extra_annotations = pd.read_csv(f'{raw_dir}/Meadows_Dynamic-Similarity-Judge-copy_v_v3_annotations.csv')
    v2_annot = pd.read_csv(f'{raw_dir}/Meadows_Dynamic-Similarity-Judgements2_v_v1_annotations.csv')
    v2_meta = pd.read_csv(f'{raw_dir}/Meadows_Dynamic-Similarity-Judgements2_v_v1_metadata.csv')
    metadata = pd.read_csv(f'{raw_dir}/Meadows_Dynamic-Similarity-Judgements_v_v23_metadata.csv')

    metadata = pd.concat([metadata, extra_metadata, v2_meta], ignore_index=True)
    metadata = metadata.loc[metadata.status == 'finished']
    name_token = {}
    for _, row in metadata.iterrows():
        name_token[row['name']] = row['token']
    print(f"\n >> The number of participants who finished: {len(metadata.loc[metadata.status == 'finished'].index)}")

    # Data cleaning
    full = pd.read_csv(f'{raw_dir}/Meadows_Dynamic-Similarity-Judgements_v_v23_annotations.csv')
    full = pd.concat([full, extra_annotations, v2_annot], ignore_index=True)
    full.loc[:, 'response_time'] = [
        (parse_iso_dt(trial.time_trial_response) - parse_iso_dt(trial.time_trial_start)).total_seconds()
        for _, trial in full.iterrows()
    ]
    data = full.loc[full['task'].str.contains('set')]
    stim_id_name = {}
    for index, row in data.iterrows():
        for i in range(1, 4):
            stim_id_name[row[f'stim{i}_id']] = row[f'stim{i}_name']
    data['choice'] = data['label'].map(stim_id_name)
    data = data[['participation', 'task', 'trial', 'stim1_name', 'stim2_name', 'stim3_name', 'choice', 'response_time']]

    # Flag catch trials
    data.loc[:, 'catch_trial_flag'] = False
    for idx, row in data.iterrows():
        stim1, stim2, stim3, choice = row['stim1_name'], row['stim2_name'], row['stim3_name'], row['choice']
        stimuli_list = [stim1, stim2, stim3]
        if len(set(stimuli_list)) == 2:
            repeated_stim = max(stimuli_list, key=stimuli_list.count)
            if choice == repeated_stim:
                data.at[idx, 'catch_trial_flag'] = True

    data['token'] = data['participation'].map(name_token)
    data = pd.merge(data, metadata[['name', 'survey_code']], left_on='participation', right_on='name').drop('name', axis=1)
    org_data_len = len(data)
    print(f' >> Total triplet length (including catch trials): {org_data_len}')

    flagged = 0
    if remove_catch:
        flagged = list(set(data.loc[data.catch_trial_flag].participation))
        data = data[~data['participation'].isin(flagged)]
        print(f' >> Number of participants who failed a catch trial: {len(flagged)}')
        print(f' >> New participant count: {len(set(data.participation))}')
        print(f' >> Total triplet length (excluding catch trials): {len(data)}')

    if remove_test and train_idx is not None:
        data = data[
            (data['stim1_name'].isin(train_idx)) &
            (data['stim2_name'].isin(train_idx)) &
            (data['stim3_name'].isin(train_idx))
        ]
        print(f" >> Total triplets without Test videos: {len(data)}")

    return data, flagged


def build_rsms_and_counts(sim_judg, n_videos=250):
    """
    Build similarity RSM and count matrix from triplet judgments.

    Args:
        sim_judg: DataFrame with stim1_name, stim2_name, stim3_name, choice columns
        n_videos: Total number of videos (matrix size)

    Returns:
        sim_rsm: DataFrame (n_videos x n_videos) probability matrix with diagonal=1
        count_df: DataFrame (n_videos x n_videos) count of pair appearances
    """
    similarity_matrix = np.zeros((n_videos, n_videos))
    count_matrix = np.zeros((n_videos, n_videos))

    for idx, row in sim_judg.iterrows():
        stim1, stim2, stim3, choice = row['stim1_name'], row['stim2_name'], row['stim3_name'], row['choice']
        stimuli_list = [stim1, stim2, stim3]
        if len(set(stimuli_list)) != 3:
            continue

        # Track all pairs
        triplet_pairs = [(stim1, stim2), (stim1, stim3), (stim2, stim3)]
        for stim_a, stim_b in triplet_pairs:
            if stim_a == stim_b:
                continue
            count_matrix[stim_a, stim_b] += 1
            count_matrix[stim_b, stim_a] += 1

        try:
            similar_stims = set(stimuli_list) - {choice}
            stim_a, stim_b = similar_stims
            similarity_matrix[stim_a, stim_b] += 1
            similarity_matrix[stim_b, stim_a] += 1
        except ValueError:
            continue

    similarity_df = pd.DataFrame(similarity_matrix, index=range(n_videos), columns=range(n_videos))
    count_df = pd.DataFrame(count_matrix, index=range(n_videos), columns=range(n_videos))

    probability_matrix = np.divide(similarity_matrix, count_matrix, where=count_matrix != 0)
    probability_matrix = np.nan_to_num(probability_matrix)
    np.fill_diagonal(probability_matrix, 1)
    sim_rsm = pd.DataFrame(probability_matrix, index=range(n_videos), columns=range(n_videos))

    return sim_rsm, count_df


# ============================================================================
# Load processed triplet data
# ============================================================================
def load_and_process_triplet_data(file_path, train_set='train'):
    """
    Load and process triplet data from file.

    Args:
        file_path: path to file with triplet data

    Returns:
        processed_triplets: list of (i, j, k, chosen_pair_idx) tuples
        df: original dataframe
    """
    if '.parquet' in file_path:
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    print("\n" + "=" * 40)
    print(f"{train_set.upper()} TRIPLETS")
    print("=" * 40)
    print(f"Loaded {len(df)} raw {train_set} triplet judgments")
    print(f"Participants: {df['participation'].nunique()}")
    print(f"Unique videos: {len(set(df['stim1_name']) | set(df['stim2_name']) | set(df['stim3_name']))}")

    processed_triplets = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing triplets"):
        # Extract video IDs
        i, j, k = int(row['stim1_name']), int(row['stim2_name']), int(row['stim3_name'])
        choice = int(row['choice'])

        # Determine which pair was chosen (not the odd one out)
        if choice == i:
            # i was odd one out, so j and k are similar (pair index 2: jk)
            chosen_pair_idx = 2
        elif choice == j:
            # j was odd one out, so i and k are similar (pair index 1: ik)
            chosen_pair_idx = 1
        elif choice == k:
            # k was odd one out, so i and j are similar (pair index 0: ij)
            chosen_pair_idx = 0
        else:
            print(f"Warning: Invalid choice {choice} for triplet ({i}, {j}, {k})")
            continue

        processed_triplets.append((i, j, k, chosen_pair_idx))

    print(f"Processed {len(processed_triplets)} valid {train_set} triplets")
    return processed_triplets, df

# Triplet embedding model (Hebart-style pruning)
# ----------------------------------------
# Model definition: sparse + non-negative
# ----------------------------------------
class VideoEmbeddingModel(nn.Module):
    def __init__(self, n_videos=250, embedding_dim=90, sparse_lambda=0.008,
                 enforce_positivity=True, device='cpu'):
        super().__init__()
        self.n_videos = n_videos
        self.embedding_dim = embedding_dim
        self.sparse_lambda = sparse_lambda
        self.enforce_positivity = enforce_positivity
        self.device = device

        # Hebart used uniform [0, 1] init for non-negative embeddings
        if enforce_positivity:
            self.embeddings = nn.Parameter(
                torch.rand(n_videos, embedding_dim, device=device) * 0.05
            )
        else:
            self.embeddings = nn.Parameter(
                torch.randn(n_videos, embedding_dim, device=device) * 0.001
            )

    def forward(self, triplets):
        if self.enforce_positivity:
            emb = torch.relu(self.embeddings)
        else:
            emb = self.embeddings

        i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        emb_i, emb_j, emb_k = emb[i], emb[j], emb[k]

        sim_ij = torch.sum(emb_i * emb_j, dim=1)
        sim_ik = torch.sum(emb_i * emb_k, dim=1)
        sim_jk = torch.sum(emb_j * emb_k, dim=1)

        similarities = torch.stack([sim_ij, sim_ik, sim_jk], dim=1)
        log_probs = torch.log_softmax(similarities, dim=1)
        return log_probs, similarities

    def get_sparsity_loss(self):
        return self.sparse_lambda * torch.sum(torch.abs(self.embeddings))

    def get_similarity_matrix(self):
        with torch.no_grad():
            emb = torch.relu(self.embeddings) if self.enforce_positivity else self.embeddings
            sim = torch.mm(emb, emb.t())
        return sim.cpu().numpy()

# ------------------------------------------------------
# Training function: sparse non-negative Hebart-style
# ------------------------------------------------------
def train_video_embedding_model(triplets, n_videos=250, embedding_dim=90,
                                n_epochs=300, learning_rate=0.001, batch_size=100,
                                test_split=0.1, device='cpu', verbose=True,
                                patience=20, min_delta=0.0005, random_seed=42,
                                sparse_lambda=0.008, enforce_positivity=True, prune_threshold=0.1,
                                val_triplets=None):
    """val_triplets: optional explicit validation set in the same (i, j, k,
    chosen_pair_idx) format; when given, `triplets` is used whole for training
    and the random `test_split` is skipped."""
    # Set seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Prepare data
    triplet_data = np.array([[t[0], t[1], t[2]] for t in triplets])
    targets = np.array([t[3] for t in triplets])
    if val_triplets is None:
        X_train, X_test, y_train, y_test = train_test_split(triplet_data, targets, test_size=test_split, random_state=random_seed)
    else:
        X_train, y_train = triplet_data, targets
        X_test = np.array([[t[0], t[1], t[2]] for t in val_triplets])
        y_test = np.array([t[3] for t in val_triplets])

    X_train = torch.tensor(X_train, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test, dtype=torch.long, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    print(f"Training on {len(X_train)} triplets, validating on {len(X_test)} triplets")

    model = VideoEmbeddingModel(
        n_videos=n_videos,
        embedding_dim=embedding_dim,
        sparse_lambda=sparse_lambda,
        enforce_positivity=enforce_positivity,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-07)
    criterion = nn.NLLLoss()

    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    best_test_acc, best_model_state, patience_counter = 0, None, 0
    progress_bar = tqdm(range(n_epochs), desc="Training", disable=not verbose)

    for epoch in progress_bar:
        model.train()
        epoch_loss, epoch_correct, total_grad_norm = 0, 0, 0
        perm = torch.randperm(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            log_probs, _ = model(batch_X)
            current_lambda = sparse_lambda * min(1.0, epoch / 20)  # ramps up over 20 epochs
            loss = criterion(log_probs, batch_y) + current_lambda * model.get_sparsity_loss()
            loss.backward()
            total_grad_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            optimizer.step()

            epoch_loss += loss.item()
            predictions = torch.argmax(log_probs, dim=1)
            epoch_correct += (predictions == batch_y).sum().item()

        train_loss = epoch_loss / (len(X_train) // batch_size)
        train_acc = epoch_correct / len(X_train)

        model.eval()
        with torch.no_grad():
            log_probs_test, _ = model(X_test)
            test_loss = criterion(log_probs_test, y_test).item()
            test_preds = torch.argmax(log_probs_test, dim=1)
            test_acc = (test_preds == y_test).float().mean().item()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if test_acc > best_test_acc + min_delta:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        progress_bar.set_postfix({
            'TrLoss': f'{train_loss:.3f}',
            'TrAcc': f'{train_acc:.3f}',
            'TsLoss': f'{test_loss:.3f}',
            'TsAcc': f'{test_acc:.3f}',
        })

        if patience_counter >= patience:
            tqdm.write(f"\nEarly stopping at epoch {epoch+1} after {patience} stale epochs, best val accuracy: {best_test_acc:.4f}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # === Hebart-style pruning: max activation threshold ===
    with torch.no_grad():
        emb_torch = torch.relu(model.embeddings) if enforce_positivity else model.embeddings
        max_vals = torch.max(emb_torch, dim=0).values.cpu().numpy()

        # Use passed-in threshold (e.g., 0.1) to exclude low-activation dims
        keep_mask = max_vals >= prune_threshold                  # shape: (embedding_dim,)
        kept_indices = np.flatnonzero(keep_mask)                 # original dim ids kept
        k = int(kept_indices.size)

        # Prune to kept dims
        pruned_emb_torch = emb_torch[:, keep_mask]               # (n_videos, k)

        # ---- Renumber dims 0..k-1 ----
        # (Data doesn't change — we just expose a clean index space & mappings.)
        new_dim_ids = np.arange(k)                               # 0..k-1
        old_to_new = {int(old): int(new) for new, old in enumerate(kept_indices)}
        new_to_old = {int(new): int(old) for new, old in enumerate(kept_indices)}

        print(f"\nPruned from {embedding_dim} to {k} dimensions "
              f"using Hebart-style threshold (max activation ≥ {prune_threshold}). "
              f"Renumbered dims to 0..{k-1}.")

        # Move to numpy for returning
        pruned_emb = pruned_emb_torch.detach().cpu().numpy()     # (n_videos, k)
        emb = emb_torch.detach().cpu().numpy()

    return (model, pruned_emb, emb, keep_mask, kept_indices, old_to_new, new_to_old,
            train_losses, test_losses, train_accuracies, test_accuracies, k)


# Plotting and logging

def plot_embedding_analysis(full_embed, pruned_embed, sparse_lambda, prune_threshold, figsize=(14, 6)):

    # Convert to numpy
    full_embed = np.asarray(full_embed)
    pruned_embed = np.asarray(pruned_embed)

    n_dims = full_embed.shape[1]
    k = pruned_embed.shape[1]

    # Compact, explicit labels
    embedding_df = pd.DataFrame(full_embed, columns=[str(i) for i in range(n_dims)])
    pruned_df = pd.DataFrame(pruned_embed, columns=[str(i) for i in range(k)])

    threshold = 1e-3
    activation_counts_full = (full_embed > threshold).sum(axis=0)
    activation_counts_pruned = (pruned_embed > threshold).sum(axis=0)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, width_ratios=[3, 2, 0.1], height_ratios=[2, 1], hspace=0.4, wspace=0.3)

    # A. Full Embedding Heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(embedding_df, aspect='auto', cmap='viridis', vmin=0, vmax=4.5)
    ax_a.set_title(f"Full Embedding Heatmap\nλ={sparse_lambda}, {n_dims} dims")
    ax_a.set_xlabel("Dimension (0-based)")
    ax_a.set_ylabel("Video ID")

    # B. Pruned Embedding Heatmap (labels 0..k-1)
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(pruned_df, aspect='auto', cmap='viridis', vmin=0, vmax=4.5)
    ax_b.set_title(f"Pruned Embedding Heatmap\nλ={sparse_lambda}, Threshold={prune_threshold}, {k}/{n_dims} dims")
    ax_b.set_xlabel("Pruned Dimension (0-based)")
    ax_b.set_ylabel("Video ID")
    ax_b.set_xticks(np.arange(k))
    ax_b.set_xticklabels(pruned_df.columns.tolist(), rotation=90, fontsize=7)
    fig.colorbar(im_b, ax=ax_b, cax=fig.add_subplot(gs[0, 2]), orientation='vertical', label='Activation Value')

    # C. Activation Histogram (Full)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.hist(activation_counts_full, bins=range(0, embedding_df.shape[0] + 10, 10), edgecolor='black')
    ax_c.set_title("Activation Counts (Full)")
    ax_c.set_xlabel(f"# Videos with Activation > {threshold}")
    ax_c.set_ylabel("# Dimensions")
    ax_c.grid(True)

    # D. Activation Histogram (Pruned)
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.hist(activation_counts_pruned, bins=range(0, pruned_df.shape[0] + 10, 10), edgecolor='black')
    ax_d.set_title("Activation Counts (Pruned)")
    ax_d.set_xlabel(f"# Videos with Activation > {threshold}")
    ax_d.set_ylabel("# Dimensions")
    ax_d.grid(True)

    # Labels A–D
    for ax, label in zip([ax_a, ax_b, ax_c, ax_d], string.ascii_uppercase[:4]):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.show()


def log_embedding_run_header(
    phase_title: str,
    device: str,
    n_dims: int,
    sparse_lambda: float,
    prune_threshold: float,
    enforce_positivity: bool,
    learning_rate: float,
    n_epochs: int,
    patience: int,
):
    print("\n" + "=" * 60)
    print(f"{phase_title.upper()}")
    print("=" * 60)
    print(f"- Computing on               : {device}")
    print(f"- Embedding dimensions       : {n_dims}")
    print(f"- Sparsity lambda (λ)        : {sparse_lambda}")
    print(f"- Prune threshold            : {prune_threshold}")
    print(f"- Positivity constraint      : {enforce_positivity}")
    print(f"- Learning rate              : {learning_rate}")
    print(f"- Max epochs                 : {n_epochs}")
    print(f"- Early stopping patience    : {patience} epochs")


def plot_embedding_analysis_sorted(full_embed, pruned_embed, sparse_lambda, prune_threshold, figsize=(14, 6)):
    # Ensure CPU tensors -> numpy
    full_embed = torch.as_tensor(full_embed).float().cpu().numpy()
    pruned_embed = torch.as_tensor(pruned_embed).float().cpu().numpy()

    n_dims = full_embed.shape[1]
    k = pruned_embed.shape[1]
    threshold = 1e-3

    # --- Sort FULL by descending activation counts ---
    activation_counts_full = (full_embed > threshold).sum(axis=0)
    full_sorted_idx = np.argsort(-activation_counts_full)  # descending
    full_embed_sorted = full_embed[:, full_sorted_idx]
    embedding_df = pd.DataFrame(full_embed_sorted, columns=[str(i) for i in full_sorted_idx])

    # --- Sort PRUNED by its own activation counts (independent of original ids) ---
    activation_counts_pruned = (pruned_embed > threshold).sum(axis=0)
    pruned_sorted_idx = np.argsort(-activation_counts_pruned)  # descending
    pruned_sorted = pruned_embed[:, pruned_sorted_idx]

    # Label pruned dims as 0..k-1 (compact ids), even after sorting
    pruned_df = pd.DataFrame(pruned_sorted, columns=[str(i) for i in range(k)])

    # Plotting
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, width_ratios=[3, 2, 0.1], height_ratios=[2, 1], hspace=0.4, wspace=0.3)

    # A. Full Embedding Heatmap (sorted)
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(embedding_df, aspect='auto', cmap='viridis', vmin=0, vmax=4.5)
    ax_a.set_title(f"Full Embedding Heatmap\nλ={sparse_lambda}, {n_dims} dims (sorted)")
    ax_a.set_xlabel("Full Dimensions (sorted by activation)")
    ax_a.set_ylabel("Video ID")

    # B. Pruned Embedding Heatmap (sorted, labels 0..k-1)
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(pruned_df, aspect='auto', cmap='viridis', vmin=0, vmax=4.5)
    ax_b.set_title(f"Pruned Embedding Heatmap\nλ={sparse_lambda}, Threshold={prune_threshold}, {k}/{n_dims} dims (sorted)")
    ax_b.set_xlabel("Pruned Dimensions (0-based, sorted by activation)")
    ax_b.set_ylabel("Video ID")
    ax_b.set_xticks(np.arange(k))
    ax_b.set_xticklabels(pruned_df.columns.tolist(), rotation=90, fontsize=7)
    fig.colorbar(im_b, ax=ax_b, cax=fig.add_subplot(gs[0, 2]), orientation='vertical', label='Activation Value')

    # C. Activation Histogram (Full, sorted data already aggregated)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.hist(activation_counts_full[full_sorted_idx], bins=range(0, full_embed.shape[0] + 10, 10), edgecolor='black')
    ax_c.set_title("Activation Counts (Full)")
    ax_c.set_xlabel(f"# Videos with Activation > {threshold}")
    ax_c.set_ylabel("# Dimensions")
    ax_c.grid(True)

    # D. Activation Histogram (Pruned)
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.hist(activation_counts_pruned[pruned_sorted_idx], bins=range(0, pruned_embed.shape[0] + 10, 10), edgecolor='black')
    ax_d.set_title("Activation Counts (Pruned)")
    ax_d.set_xlabel(f"# Videos with Activation > {threshold}")
    ax_d.set_ylabel("# Dimensions")
    ax_d.grid(True)

    # Subplot labels A–D
    for ax, label in zip([ax_a, ax_b, ax_c, ax_d], string.ascii_uppercase[:4]):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.show()

    # Return the order of compact (0..k-1) ids after sorting
    # (i.e., permutation of 0..k-1 corresponding to the columns as shown)
    ordered_dims = [int(i) for i in range(k)]
    return ordered_dims, pruned_sorted
