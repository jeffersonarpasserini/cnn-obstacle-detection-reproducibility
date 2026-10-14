"""Atomic study artifacts and immutable input identities (no model fitting)."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    compression = {'method': 'gzip', 'mtime': 0} if path.suffix == '.gz' else None
    frame.to_csv(temporary, index=False, float_format='%.17g', compression=compression)
    temporary.replace(path)


def guard_output(path):
    path = Path(path).resolve()
    protected = [ROOT / 'results' / name for name in
                 ('full_search', 'fine_tuning', 'fine_tuning_recovered', 'scene_groups', 'scene_groups_v1')]
    protected += [ROOT / '.venv', ROOT / 'configs', ROOT / 'data', ROOT / 'via-dataset']
    if path in (ROOT, ROOT / 'results') or ROOT not in path.parents:
        raise ValueError('Use a dedicated study output directory inside the repository')
    if any(p == path or p in path.parents or path in p.parents for p in protected):
        raise ValueError('Output would overwrite a protected input or existing study')
    return path


def source_identity():
    files = ['src/artifact.py', 'src/selection.py', 'src/fixed_refit.py', 'src/fine_tuning.py',
             'src/study_io.py', 'src/study_statistics.py', 'src/study_features.py', 'src/study_estimates.py',
             'src/nested_cv.py', 'src/external_test.py', 'src/external_oof.py', 'scripts/clean_external_dataset.py',
             'scripts/compute_tibshirani_bias.py',
             'scripts/analyze_nested_cv.py', 'scripts/analyze_external_test.py',
             'scripts/build_scene_groups.py', 'scripts/scene_groups_geometry.py', 'scripts/scene_groups_graph.py']
    return {name: sha256(ROOT / name) for name in files if (ROOT / name).exists()}


@lru_cache(maxsize=1)
def gpu_inventory():
    try:
        value = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                         '--format=csv,noheader,nounits'], text=True,
                                         stderr=subprocess.DEVNULL, timeout=5)
        return [line.strip() for line in value.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.SubprocessError):
        return []


def runtime():
    packages = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'tensorflow', 'umap-learn',
                 'skrebate', 'torch', 'transformers', 'Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'physical_gpu_inventory': gpu_inventory(), 'sources': source_identity()}


def checked_identity(directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(f'Inputs/code changed; use a new output directory: {directory}')
    if not path.exists():
        write_json(path, identity)


def completed_job(directory, identity):
    marker = Path(directory) / 'complete.json'
    if not marker.exists():
        return False
    value = json.loads(marker.read_text())
    if value['identity'] != identity:
        raise ValueError('Completed job input identity changed')
    for name, checksum in value['artifacts'].items():
        if sha256(Path(directory) / name) != checksum:
            raise ValueError(f'Completed artifact changed: {name}')
    return True


def finish_job(directory, identity, names, **metadata):
    directory = Path(directory)
    write_json(directory / 'complete.json', {'identity': identity, **metadata,
               'artifacts': {name: sha256(directory / name) for name in names}})


def committed_bytes(path):
    """Require current bytes to equal HEAD and return the blob and commit identities."""
    path = Path(path).resolve()
    top = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True).strip())
    relative = path.relative_to(top).as_posix()
    result = subprocess.run(['git', 'show', f'HEAD:{relative}'], cwd=ROOT, capture_output=True)
    if result.returncode or result.stdout != path.read_bytes():
        raise ValueError(f'Protocol/config must be committed without local edits: {relative}')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return {'path': relative, 'sha256': sha256(path), 'commit': commit}
