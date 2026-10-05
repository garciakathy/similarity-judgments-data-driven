"""Action-category labels for the 250 videos and the binary same-category RSM."""

import os
import pandas as pd


def load_action_categories(utils_dir, annotations_path=None):
    """
    Load action categories for all 250 videos, aligned to video index order.

    Returns:
        labels: array of shape (N_VIDEOS,) with action category strings
    """
    train_acts = pd.read_csv(os.path.join(utils_dir, 'train_categories.csv'))
    test_acts = pd.read_csv(os.path.join(utils_dir, 'test_categories.csv'))
    actions = pd.concat([train_acts, test_acts], ignore_index=True)

    # Align to video index via annotations
    if annotations_path is None:
        annotations_path = os.path.join(utils_dir, 'annotations.csv')
    annotations = pd.read_csv(annotations_path)

    # vid_id is the row index in annotations (0..249)
    if 'vid_id' not in annotations.columns:
        annotations = annotations.reset_index().rename(columns={'index': 'vid_id'})

    lab = (
        annotations[['vid_id', 'video_name']]
        .merge(actions, on='video_name', how='left')
        .sort_values('vid_id')
    )

    return lab['action_categories'].to_numpy()


def build_action_rsm(labels):
    """Build binary RSM: 1 if same category, 0 if different."""
    return (labels[:, None] == labels[None, :]).astype(float)
