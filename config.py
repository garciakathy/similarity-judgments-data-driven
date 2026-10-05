"""
Central configuration for the similarity-judgment data-driven project.
All paths, hyperparameters, seeds, and shared constants in one place.
"""

import os
import pandas as pd

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
UTILS_DIR = os.path.join(DATA_DIR, 'utils')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
MODEL_DIR = os.path.join(INTERIM_DIR, 'model')
RESULTS_DIR = os.path.join(INTERIM_DIR, 'results')
RESULTS_BEHAVIORAL_DIR = os.path.join(RESULTS_DIR, 'behavioral')
RESULTS_DIMENSIONS_DIR = os.path.join(RESULTS_DIR, 'dimensions')
RESULTS_NEURAL_DIR = os.path.join(RESULTS_DIR, 'neural')
RESULTS_ACTION_DIR = os.path.join(RESULTS_DIR, 'action')
RESULTS_RSM_DIR = os.path.join(RESULTS_DIR, 'rsm')
RSM_DIR = os.path.join(INTERIM_DIR, 'rsms')

# fMRI data paths
FMRI_DIR = os.path.join(RAW_DIR, 'fmri')
FMRI_BETAS_DIR = os.path.join(FMRI_DIR, 'fmri_betas')
FMRI_RELI_DIR = os.path.join(FMRI_DIR, 'reliability_mask')
FMRI_FREESURFER_DIR = os.path.join(FMRI_DIR, 'freesurfer')
FMRI_ANATOMY_DIR = os.path.join(FMRI_DIR, 'anatomy')
RESULTS_SEARCHLIGHT_DIR = os.path.join(RESULTS_DIR, 'searchlight')

# Ensure output directories exist
for d in [MODEL_DIR, RESULTS_DIR, RESULTS_BEHAVIORAL_DIR, RESULTS_DIMENSIONS_DIR,
          RESULTS_NEURAL_DIR, RESULTS_ACTION_DIR, RESULTS_RSM_DIR, RSM_DIR,
          RESULTS_SEARCHLIGHT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# RANDOM SEED
# ============================================================================

RANDOM_SEED = 888

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

EMBEDDING_DIM = 90
SPARSE_LAMBDA = 0.011
PRUNE_THRESHOLD = 0.1
N_EPOCHS = 1000
PATIENCE = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 100
ENFORCE_POSITIVITY = True
N_VIDEOS = 250

# ============================================================================
# TRAIN / TEST VIDEO SPLIT
# ============================================================================

def _load_train_test_idx():
    """Load train/test video indices from annotations + train.csv/test.csv."""
    annotations_path = os.path.join(UTILS_DIR, 'annotations.csv')
    train_csv = os.path.join(UTILS_DIR, 'train.csv')
    test_csv = os.path.join(UTILS_DIR, 'test.csv')

    if os.path.exists(annotations_path) and os.path.exists(train_csv) and os.path.exists(test_csv):
        annotations = pd.read_csv(annotations_path)
        train_list = pd.read_csv(train_csv)
        test_list = pd.read_csv(test_csv)

        train_behavior = annotations[annotations['video_name'].isin(train_list['video_name'])]
        test_behavior = annotations[annotations['video_name'].isin(test_list['video_name'])]

        return train_behavior['vid_id'].to_list(), test_behavior['vid_id'].to_list()
    else:
        return None, None

train_idx, test_idx = _load_train_test_idx()

# ============================================================================
# BEST DIMENSIONS (greedy forward selection on 250x250 train RSM,
# Kneedle elbow on 10-fold x 5-rep CV R²; see analysis/behavioral/dim_selection_train.py)
# ============================================================================

BEST_DIMS = [1, 0, 13, 7, 9, 2, 4, 3, 6, 17, 8]

DIM_LABELS = {
    1: 'Conversation',
    0: 'Caregiving',
    13: 'Outdoor activity',
    7: 'Martial arts',
    9: 'Infants crying',
    2: 'Musical performance',
    4: 'Infant play',
    3: 'Children dancing',
    6: 'Cooking/Eating',
    17: 'Action sport',
    8: 'Infant bath/bed',
}

# ============================================================================
# BEHAVIORAL FEATURES
# ============================================================================

BEHAVIOR_COLS = [
    'indoor', 'expanse', 'transitivity', 'agent distance',
    'facingness', 'joint action', 'communication', 'cooperation',
    'dominance', 'intimacy', 'valence', 'arousal',
]

FEATURE_ORDER = [
    'Intimacy', 'Dominance', 'Interagent Distance',
    'Spatial Expanse', 'Valence', 'Indoor/Outdoor', 'Arousal',
    'Object Directed', 'Communicating', 'Cooperation', 'Acting Jointly',
    'Agents Facing', 'Action Categories', 'Shared Variance', 'Total Variance'
]

FEATURE_REMAPS = {
    'Agent Distance': 'Interagent Distance',
    'Facingness': 'Agents Facing',
    'Joint Action': 'Acting Jointly',
    'Communication': 'Communicating',
    'Indoor': 'Indoor/Outdoor',
    'Expanse': 'Spatial Expanse',
    'Transitivity': 'Object Directed',
    'Behavior Shared': 'Shared Variance',
    'Behavior Total': 'Total Variance',
}

# ============================================================================
# NEURAL ANALYSIS
# ============================================================================

ROI_ORDER = ['EVC', 'MT', 'EBA', 'LOC', 'pSTS', 'face-pSTS', 'aSTS', 'FFA', 'PPA']

# ROI voxel responses for all 250 videos (metadata.csv, response_data.csv.gz).
# Not in the repo; only needed to rebuild the ROI RSMs, whose cached copies in
# data/interim/rsms/ are what the analyses read.
FMRI_EXPORT_DIR = os.path.join(FMRI_DIR, 'voxel_responses')
SUBJECTS = [1, 2, 3, 4]

# Searchlight parameters
SEARCHLIGHT_RADIUS_VOXELS = 3  # sphere radius in voxels (converted to mm at runtime)

# ============================================================================
# PLOTTING
# ============================================================================

PDF_FONTTYPE = 42  # keeps text editable in Illustrator
PS_FONTTYPE = 42
