"""Verify the already reproduced canonical article-finalist OOF, without fitting."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .artifact import scan_dataset, dataset_content_fingerprint, prediction_metrics
from .study_io import ROOT, sha256


DESIGN = 'article_original_10fold'


def validate_oof_coverage(frame, folds, expected):
    if frame.filename.duplicated().any() or set(frame.filename) != set(expected):
        raise ValueError('Article OOF must cover each original VIA image exactly once')
    assignments = folds.set_index('filename').test_fold.to_dict()
    for row in frame.itertuples():
        if expected[row.filename] != row.true_label or assignments[row.filename] != row.fold:
            raise ValueError('Article OOF labels/folds differ from the original assignment')
    if not np.isfinite(frame.score_clear).all():
        raise ValueError('Article OOF has nonfinite native scores')


def load_article_oof(config, selection, experiments):
    specification = config['fixed_oof']
    source = ROOT/specification['source']
    fold_path = ROOT/specification['fold_assignments']
    reference_path = ROOT/config['fixed_search_results']/'per_fold_metrics.csv'
    for path, key in [(source, 'predictions_sha256'), (fold_path, 'fold_assignments_sha256'),
                      (reference_path, 'reference_metrics_sha256')]:
        if sha256(path) != specification[key]:
            raise ValueError(f'Original article OOF input changed: {path.name}')
    source_run = source.parent
    metadata = json.loads((source_run/'run_metadata.json').read_text())
    if not metadata.get('completed'):
        raise ValueError('Canonical fixed refits have not completed')
    for filename, key in [('src/fixed_refit.py', 'fixed_refit'), ('src/fine_tuning.py', 'fine_tuning'),
                          ('src/artifact.py', 'artifact')]:
        if sha256(ROOT/filename) != metadata['source_hashes'][key]:
            raise ValueError('Canonical original-refit source identity changed')
    paths, labels = scan_dataset(ROOT/config['via_dataset'])
    if dataset_content_fingerprint(paths) != metadata['dataset_content_sha256']:
        raise ValueError('Original VIA pixels changed after canonical refit')
    expected = {path.name: int(label) for path, label in zip(paths, labels)}
    folds = pd.read_csv(fold_path)
    if folds.filename.duplicated().any() or set(folds.filename) != set(expected):
        raise ValueError('Original fold assignment does not cover VIA')
    raw = pd.read_csv(source, float_precision='round_trip')
    names = set(selection['selected'].values())
    if set(raw.experiment) != names or not set(raw.family) <= {'fixed', 'fixed_features'}:
        raise ValueError('Original fixed OOF finalists/family differ from the article')
    if set(raw.seed) != {config['seed']} or raw.duplicated(['experiment', 'filename']).any():
        raise ValueError('Original fixed OOF seed/coverage is invalid')
    reference = pd.read_csv(reference_path).set_index(['experiment', 'fold'])
    verified_jobs, metric_checks = [], 0
    for name, panel in raw.groupby('experiment'):
        validate_oof_coverage(panel, folds, expected)
        for fold, rows in panel.groupby('fold'):
            job = source_run/'jobs'/f'{name}__fold_{int(fold):02d}__seed_{config["seed"]}'
            manifest = json.loads((job/'manifest.json').read_text())
            if manifest['state'] != 'completed':
                raise ValueError('Original canonical refit job is incomplete')
            for filename in ('predictions.csv', 'metrics.json', 'split_indices.json', 'canonical_features.json'):
                if sha256(job/filename) != manifest['artifacts'][filename]:
                    raise ValueError(f'Canonical OOF job artifact changed: {job.name}/{filename}')
            canonical = json.loads((job/'canonical_features.json').read_text())
            if canonical['policy'] != 'globally_ordered_frozen_features_v1' or canonical['batch_size'] != 4:
                raise ValueError('Original OOF did not use the canonical fixed_refit feature policy')
            request = manifest['request']['fixed_experiment']
            for key in ('approach', 'classifier', 'extractors', 'reduction', 'components'):
                if request.get(key) != experiments[name].get(key):
                    raise ValueError('Canonical refit classifier/reducer differs from selected article configuration')
            stored = pd.read_csv(job/'predictions.csv', float_precision='round_trip').sort_values('filename')
            current = rows.sort_values('filename')
            for key in ('filename', 'true_label', 'predicted_label', 'score_clear', 'fold'):
                equal = (np.allclose(stored[key].to_numpy(), current[key].to_numpy(), atol=1e-14, rtol=1e-14)
                         if key == 'score_clear' else np.array_equal(stored[key].to_numpy(), current[key].to_numpy()))
                if not equal:
                    raise ValueError('Aggregate article OOF differs from its canonical jobs')
            split = json.loads((job/'split_indices.json').read_text())
            if set(split['test']) != set(rows.sample_index) or set(split['outer_train']) & set(split['test']):
                raise ValueError('Canonical original-refit split leakage/coverage')
            values = prediction_metrics(rows.true_label, rows.predicted_label, rows.score_clear)
            for key, value in values.items():
                if key in reference.columns:
                    if not np.isclose(value, reference.loc[(name, fold), key], atol=1e-8, rtol=1e-8, equal_nan=True):
                        raise ValueError(f'Canonical OOF differs from full_search: {name}/{fold}/{key}')
                    metric_checks += 1
            verified_jobs.append({'job': job.name, 'manifest_sha256': sha256(job/'manifest.json'),
                                  'predictions_sha256': sha256(job/'predictions.csv')})
    frames = []
    for procedure, name in {**selection['selected'], 'winner': selection['winner']}.items():
        panel = raw[raw.experiment.eq(name)].sort_values('filename').copy()
        panel['procedure'] = procedure
        panel['score_kind'] = 'margin' if experiments[name]['classifier'] in {'logistic', 'linear_svm', 'rbf_svm'} else 'probability'
        panel['predicted_label_default'] = panel.predicted_label
        panel['source'] = 'via_oof'
        frames.append(panel)
    provenance = {'design': DESIGN, 'source': str(source.relative_to(ROOT)), 'source_sha256': sha256(source),
                  'fold_assignments_sha256': sha256(fold_path), 'reference_metrics_sha256': sha256(reference_path),
                  'original_refit_jobs': verified_jobs, 'canonical_metric_checks': metric_checks,
                  'article_finalists': selection['selected'], 'global_winner': selection['winner'],
                  'selection_optimism': 'Finalists were selected on these same VIA folds; thresholds inherit selection optimism.',
                  'new_fits': 0, 'external_images_used': 0}
    return pd.concat(frames, ignore_index=True), provenance
