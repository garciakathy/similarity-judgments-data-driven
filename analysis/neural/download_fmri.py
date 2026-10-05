"""
Download the fMRI files the neural analyses need from the OSF project of
McMahon, Isik & Bonner (2025), https://osf.io/4j29y, into data/raw/fmri/:

  fmri_betas/sub-0N_space-T1w_desc-test-fracridge_data.nii.gz   (test-video betas, ~35 MB each)
  freesurfer/sub-0N/surf/{lh,rh}.{inflated,pial,sulc}            (from freesurfer.zip, 1.2 GB download)

Existing files are left alone, so the script can be re-run to fill in gaps.

Run:
  python analysis/neural/download_fmri.py
  python analysis/neural/download_fmri.py --skip-surfaces   # betas only
"""

import os
import sys
import json
import shutil
import argparse
import tempfile
import zipfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import FMRI_DIR, SUBJECTS

OSF_ROOT = 'https://api.osf.io/v2/nodes/4j29y/files/osfstorage/'
SURFACES = [f'{h}.{s}' for h in ('lh', 'rh') for s in ('inflated', 'pial', 'sulc')]


def _get(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def _find(folder_url, name):
    """Return the OSF entry called `name` inside a folder listing (follows pagination)."""
    url = folder_url
    while url:
        page = _get(url)
        for entry in page['data']:
            if entry['attributes']['name'] == name:
                return entry
        url = page['links'].get('next')
    raise FileNotFoundError(f'{name} not found on OSF')


def _children(entry):
    return entry['relationships']['files']['links']['related']['href']


def _download(entry, dest):
    tmp = dest + '.part'
    print(f'  {os.path.relpath(dest)} ({entry["attributes"]["size"] / 1e6:.0f} MB)')
    with urllib.request.urlopen(entry['links']['download']) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dest)


def download_betas(dest_dir):
    betas = _find(OSF_ROOT, 'betas')
    for sub in SUBJECTS:
        name = f'sub-0{sub}_space-T1w_desc-test-fracridge_data.nii.gz'
        dest = os.path.join(dest_dir, 'fmri_betas', name)
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        _download(_find(_children(_find(_children(betas), f'sub-0{sub}')), name), dest)


def download_surfaces(dest_dir):
    wanted = {f'freesurfer/sub-0{sub}/surf/{s}' for sub in SUBJECTS for s in SURFACES}
    missing = {m for m in wanted if not os.path.exists(os.path.join(dest_dir, m))}
    if not missing:
        return
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, 'freesurfer.zip')
        _download(_find(OSF_ROOT, 'freesurfer.zip'), zip_path)
        with zipfile.ZipFile(zip_path) as z:
            for member in sorted(missing):
                out = os.path.join(dest_dir, member)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with z.open(member) as src, open(out, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
    print(f'  extracted {len(missing)} surface files')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=FMRI_DIR, help='default: data/raw/fmri')
    ap.add_argument('--skip-surfaces', action='store_true',
                    help='betas only (surfaces are needed only to plot brain maps)')
    args = ap.parse_args()
    print(f'Downloading fMRI data from https://osf.io/4j29y into {args.dest}')
    download_betas(args.dest)
    if not args.skip_surfaces:
        download_surfaces(args.dest)
    print('done')


if __name__ == '__main__':
    main()
