"""Read-only scientific inputs and separately scoped post-test outputs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import importlib.metadata as metadata
import sysconfig

import numpy as np
import pandas as pd

from .study_io import ROOT, sha256, write_json
from .artifact import scan_dataset

SCOPE = 'exploratory/post hoc'


def configuration():
    return json.loads((ROOT/'configs/stage3.json').read_text())


def table(frame, folder, name):
    folder = Path(folder)/'manuscript_tables'; folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(frame).copy()
    frame['analysis_scope'] = SCOPE
    frame.to_csv(folder/f'{name}.csv', index=False)
    from scripts.build_manuscript_tables import latex_escape
    rows=[]
    for row in frame.itertuples(index=False,name=None):
        cells=[f'{v:.4f}' if isinstance(v,(float,np.floating)) and np.isfinite(v)
               else '--' if v is None or isinstance(v,(float,np.floating)) and np.isnan(v)
               else latex_escape(str(v)) for v in row]
        rows.append(' & '.join(cells)+r' \\')
    text='% '+SCOPE+'\n'+'\n'.join(rows)+'\n'
    (folder/f'{name}.tex').write_text(text)
    (folder/f'{name}_rows.tex').write_text(text)
    return frame


def protected_files():
    paths = set()
    for name in ('results/external_test', 'results/learning_curve', 'results/full_search',
                 'results/fine_tuning_recovered', 'via-dataset', 'external-data/via-dataset-extended',
                 'external-data/scene_union'):
        paths.update(p for p in (ROOT/name).rglob('*') if p.is_file())
    paths.update((ROOT/'external-data').glob('*.csv'))
    paths.update((ROOT/'configs').glob('*'))
    paths.add(ROOT/'data/groups.csv')
    paths.update(ROOT/'src'/n for n in ('artifact.py', 'fine_tuning.py', 'fixed_refit.py'))
    return sorted(p for p in paths if p.is_file())


def snapshot():
    def entry(path):
        return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(entry, protected_files()))
    packages = {d.metadata['Name']: d.version for d in metadata.distributions(path=[sysconfig.get_path('purelib')])}
    return {'analysis_scope': SCOPE, 'files': files, 'training_packages': packages,
            'total_bytes': sum(row['bytes'] for row in files)}


def check_protected(output):
    before = json.loads((output/'protected_before.json').read_text())
    after = snapshot()
    a, b = {r['path']: r for r in before['files']}, {r['path']: r for r in after['files']}
    changed = sorted(name for name in a.keys()|b.keys() if a.get(name) != b.get(name))
    comparison = pd.DataFrame([{'path': name, 'sha256_before': a[name]['sha256'],
        'sha256_after': b.get(name, {}).get('sha256'), 'unchanged': a[name] == b.get(name)} for name in a])
    comparison['analysis_scope'] = SCOPE
    comparison.to_csv(output/'protected_hash_comparison.csv', index=False)
    result = {'analysis_scope': SCOPE, 'files_checked': len(a), 'bytes_checked': before['total_bytes'],
        'changed': changed, 'packages_unchanged': before['training_packages'] == after['training_packages']}
    write_json(output/'protected_after.json', after)
    write_json(output/'protected_verification.json', result)
    if changed or not result['packages_unchanged']:
        raise ValueError('Protected inputs or training environment changed')
    return result


def inputs(smoke=False):
    frame = pd.read_csv(ROOT/'results/external_test/predictions.csv', float_precision='round_trip')
    manifest = pd.read_csv(ROOT/'external-data/external_manifest_clean.csv').sort_values('filename')
    if smoke:
        names = np.r_[manifest[manifest.final_label.eq(0)].filename.iloc[:16],
                      manifest[manifest.final_label.eq(1)].filename.iloc[:16]]
        manifest = manifest[manifest.filename.isin(names)]
        frame = frame[frame.filename.isin(names)]
    names = sorted(manifest.filename)
    aligned = manifest.set_index('filename').loc[names]
    return frame, aligned, aligned.final_label.to_numpy(), aligned.group_id.to_numpy()


def panels(frame, decision='predicted_label_default'):
    for name, rows in frame.groupby('model', sort=True):
        names = sorted(rows.filename.unique())
        scores = rows.pivot(index='filename', columns='seed', values='score_clear').loc[names].to_numpy().T
        decisions = rows.pivot(index='filename', columns='seed', values=decision).loc[names].to_numpy().T
        yield name, rows, decisions, scores


def via_inputs():
    fixed = pd.read_csv(ROOT/'results/external_test/calibrated_via_oof.csv', float_precision='round_trip')
    mapping = json.loads((ROOT/'configs/article_selected_models_10fold.json').read_text())['experiments']
    fixed = pd.concat([fixed[fixed.procedure.eq(e['approach'])].assign(model=e['name'], seed=-1,
        predicted_label_default=lambda d: d.predicted_label_calibrated_default) for e in mapping])
    ft = pd.read_csv(ROOT/'results/fine_tuning_recovered/per_image_predictions.csv', float_precision='round_trip')
    ft = ft[ft.family.eq('fine_tuning')].assign(model=lambda d: 'ft_'+d.architecture,
        native_score=lambda d: d.score_clear, predicted_label_default=lambda d: d.predicted_label)
    return pd.concat([fixed, ft], ignore_index=True)


def embeddings():
    metadata = json.loads((ROOT/'external-data/scene_union/embedding_metadata.json').read_text())
    path = ROOT/'external-data/scene_union/embeddings.npz'
    if sha256(path) != metadata['cache_sha256'] or metadata['model']['id'] != 'facebook/dinov2-small':
        raise ValueError('Frozen DINOv2 cache changed or uses a fallback model')
    names = pd.read_csv(ROOT/'external-data/scene_union/groups.csv').filename.to_numpy()
    # The scene pipeline sorts names globally before storing embeddings/groups.
    if list(names) != sorted(names):
        raise ValueError('DINO union filename order is not globally sorted')
    values = np.load(path, allow_pickle=False)['embeddings']
    if len(names) != len(values): raise ValueError('DINO embedding coverage mismatch')
    values = values/np.linalg.norm(values, axis=1, keepdims=True)
    return names, values


def image_paths():
    paths, labels = scan_dataset(ROOT/'via-dataset')
    manifest = pd.read_csv(ROOT/'external-data/external_manifest_clean.csv').sort_values('filename')
    external = [ROOT/'external-data/via-dataset-extended'/name for name in manifest.filename]
    return paths, labels, external, manifest
