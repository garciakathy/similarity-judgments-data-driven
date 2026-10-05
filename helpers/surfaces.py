"""FreeSurfer surface paths for plotting searchlight maps on the cortex."""

import os

from config import FMRI_FREESURFER_DIR


def surf_path(sub, hemi, surface):
    """hemi: 'lh'|'rh', surface: 'inflated'|'pial'|'sulc'"""
    return os.path.join(FMRI_FREESURFER_DIR,
                        f'sub-0{sub}', 'surf', f'{hemi}.{surface}')


def load_surfaces(sub):
    """Load FreeSurfer surfaces for a subject. Returns dict or None if missing."""
    paths = {}
    for hemi in ['lh', 'rh']:
        for surf in ['inflated', 'pial', 'sulc']:
            p = surf_path(sub, hemi, surf)
            if not os.path.exists(p):
                print(f"  WARNING: Missing surface {p}")
                return None
            paths[f'{hemi}_{surf}'] = p
    return paths
