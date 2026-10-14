#!/usr/bin/env python3
"""Preregistered VIA final models, single external evaluation and secondary curve.

Real evaluation is gated; CPU smokes train/check only small VIA subsets.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from time import perf_counter

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from src.artifact import (scan_dataset, prediction_metrics, continuous_scores, _tensorflow_model,
                          dataset_content_fingerprint)
from src.fine_tuning import (configure_tensorflow, train_neural, fit_fixed_pipeline, transform_fixed,
                             image_batch, preprocessing, save_deployment, check_reload_values, benchmark_saved)
from src.selection import select_procedures
from src.study_features import CanonicalFeatureStore
from src.study_io import (ROOT, sha256, digest, write_json, atomic_csv, guard_output, runtime,
                          committed_bytes, checked_identity, completed_job, finish_job, source_identity)
from src.study_statistics import safety_threshold, wilson
from src.study_estimates import external_estimate
from src.external_oof import load_article_oof, DESIGN as ARTICLE_OOF_DESIGN


def registered_protocol(config_path):
    config = json.loads(Path(config_path).read_text())
    return {'config': committed_bytes(config_path), 'protocol': committed_bytes(ROOT/config['protocol'])}


def fixed_selection(config):
    records = pd.read_csv(ROOT/config['fixed_search_results']/'per_fold_metrics.csv')
    search = json.loads((ROOT/config['fixed_search_config']).read_text())
    selection = select_procedures(records, search)
    reference = json.loads((ROOT/config['fixed_reference_config']).read_text())
    expected = {e['approach']: e['name'] for e in reference['experiments']}
    if selection['selected'] != expected:
        raise ValueError('Article finalist regression failed')
    archived = pd.read_csv(ROOT/config['fixed_search_results']/'manuscript_tables/selected_approaches.csv')
    if selection['winner'] != archived.iloc[0].experiment:
        raise ValueError('Global winner regression failed')
    return selection, {e['name']: e for e in search['experiments']}


def grouped_validation(labels, groups, seed=1980):
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    train, validation = next(splitter.split(np.zeros(len(labels)), labels, groups))
    if set(groups[train]) & set(groups[validation]):
        raise ValueError('Scene crosses final-model training/early-stopping validation')
    return train, validation


def native_calibration_input(scores, kinds):
    scores, kinds = np.asarray(scores, dtype=float), np.asarray(kinds)
    if not np.isfinite(scores).all() or not set(kinds) <= {'margin', 'probability'}:
        raise ValueError('Invalid native score identity')
    probability = kinds == 'probability'
    result = scores.copy()
    result[probability] = logit(np.clip(scores[probability], 1e-7, 1-1e-7))
    return result[:, None]


def fit_probability_adapter(oof, seed=1980):
    if 'source' not in oof or set(oof.source) != {'via_oof'}:
        raise ValueError('Calibration requires only verified VIA OOF data')
    if oof.filename.duplicated().any() or set(oof.true_label) != {0, 1}:
        raise ValueError('Calibration OOF observations must be unique and include both classes')
    adapter = LogisticRegression(C=1., max_iter=500, random_state=seed)
    adapter.fit(native_calibration_input(oof.score_clear, oof.score_kind), oof.true_label)
    if adapter.coef_[0, 0] <= 0:
        raise ValueError('VIA calibration slope is nonpositive; stop before freezing thresholds')
    return adapter


def adapter_probability(adapter, scores, kind):
    return adapter.predict_proba(native_calibration_input(scores, np.repeat(kind, len(scores))))[:, 1]


class CalibratedDecision:
    """Batch-one benchmark includes the frozen probability adapter and decision."""
    def __init__(self, classifier, adapter):
        self.classifier, self.adapter = classifier, adapter
    def predict(self, prepared):
        scores = continuous_scores(self.classifier, prepared)
        kind = 'margin' if hasattr(self.classifier, 'decision_function') else 'probability'
        return (adapter_probability(self.adapter, scores, kind) >= .5).astype(int)


def via_thresholds(config, output, adapters=None):
    design = config.get('fixed_oof', {}).get('design', 'nested_ungrouped')
    if design == ARTICLE_OOF_DESIGN:
        selection, experiments = fixed_selection(config)
        nested, provenance = load_article_oof(config, selection, experiments)
        atomic_csv(nested, output/'fixed_oof_predictions.csv')
        write_json(output/'fixed_oof_provenance.json', provenance)
        path = output/'fixed_oof_predictions.csv'
    elif design == 'nested_ungrouped':
        nested_dir = ROOT/config['nested_results']
        metadata = json.loads((nested_dir/'run_metadata.json').read_text())
        if not metadata.get('completed') or metadata.get('smoke') or metadata.get('groups_sha256'):
            raise ValueError('Complete ungrouped VIA nested CV is required before final thresholds')
        path = nested_dir/'outer_predictions.csv'
        if sha256(path) != metadata['outputs']['outer_predictions.csv']:
            raise ValueError('Nested OOF predictions changed')
        nested = pd.read_csv(path, float_precision='round_trip')
    else:
        raise ValueError('Unknown fixed VIA OOF design')
    via_paths, via_labels = scan_dataset(ROOT/config['via_dataset'])
    expected = {p.name: int(label) for p, label in zip(via_paths, via_labels)}
    values, calibrated = {}, []
    for procedure in ['A', 'B', 'C', 'D', 'winner']:
        oof = nested[nested.procedure.eq(procedure)].sort_values('filename').copy()
        if set(oof.filename) != set(expected) or oof.filename.duplicated().any() or any(expected[r.filename] != r.true_label for r in oof.itertuples()):
            raise ValueError('Fixed VIA OOF coverage differs from original VIA')
        oof['source'] = 'via_oof'
        adapter = fit_probability_adapter(oof, config['seed']) if adapters is None else adapters[procedure]
        joblib.dump(adapter, output/f'adapter_{procedure}.joblib')
        probabilities = adapter.predict_proba(native_calibration_input(oof.score_clear, oof.score_kind))[:, 1]
        values[procedure] = {**safety_threshold(oof.true_label, probabilities, config['safety_obstructed_recall']),
                             'default': .5, 'adapter': f'adapter_{procedure}.joblib',
                             'adapter_sha256': sha256(output/f'adapter_{procedure}.joblib')}
        panel = oof.copy()
        panel['native_score'] = panel.score_clear
        panel['score_clear'] = probabilities
        panel['predicted_label_calibrated_default'] = (probabilities >= .5).astype(int)
        panel['predicted_label_safety'] = (probabilities > values[procedure]['threshold']).astype(int)
        calibrated.append(panel)
    ft_path = ROOT/config['fine_tuning_results']/'per_image_predictions.csv'
    ft = pd.read_csv(ft_path, float_precision='round_trip')
    for architecture in config['architectures']:
        rows = ft[(ft.family == 'fine_tuning') & ft.architecture.eq(architecture)]
        if rows.duplicated(['filename', 'seed']).any() or set(rows.seed) != set(config['initialization_seeds']):
            raise ValueError('Incomplete fine-tuning VIA OOF seeds')
        if not rows.groupby('filename').size().eq(len(config['initialization_seeds'])).all():
            raise ValueError('Fine-tuning OOF seeds missing images')
        oof = rows.groupby('filename').agg(true_label=('true_label', 'first'), score_clear=('score_clear', 'mean'))
        if set(oof.index) != set(expected) or any(expected[name] != row.true_label for name, row in oof.iterrows()):
            raise ValueError('Fine-tuning OOF coverage/labels differ from VIA')
        values[f'ft_{architecture}'] = {**safety_threshold(oof.true_label, oof.score_clear,
                                      config['safety_obstructed_recall']), 'default': .5}
    payload = {'frozen': True, 'source': 'via_oof_only', 'thresholds': values,
               'fixed_oof_design': design, 'fixed_oof_sha256': sha256(path), 'fine_tuning_oof_sha256': sha256(ft_path),
               'dataset_sha256': dataset_content_fingerprint(via_paths)}
    if design == ARTICLE_OOF_DESIGN:
        payload['original_oof_source_sha256'] = config['fixed_oof']['predictions_sha256']
        payload['selection_optimism_notice'] = config['fixed_oof']['notice']
    else:
        payload['nested_oof_sha256'] = sha256(path)
    atomic_csv(pd.concat(calibrated, ignore_index=True), output/'calibrated_via_oof.csv')
    write_json(output/'thresholds.json', payload)
    return payload


def apply_label_review(directory, *, reviewer, date, audit_reviewed, config_sha256=None):
    if not reviewer.strip() or not date.strip() or not audit_reviewed:
        raise ValueError('Author identity, date and explicit full-audit declaration required')
    directory = Path(directory)
    report = json.loads((directory/'cleaning_report.json').read_text())
    manifest = pd.read_csv(directory/'external_manifest_clean.csv')
    if sha256(directory/'external_manifest_clean.csv') != report['manifest_sha256']:
        raise ValueError('Cleaning manifest changed before review application')
    corrections = pd.read_csv(directory/'label_corrections.csv')
    if corrections.filename.duplicated().any() or not set(corrections.filename) <= set(manifest.filename):
        raise ValueError('Invalid label-correction coverage')
    original = manifest.set_index('filename').original_label.to_dict()
    for row in corrections.itertuples():
        if original[row.filename] != row.original_label or row.corrected_label not in (0, 1):
            raise ValueError('Correction original/new label differs from manifest')
        if not all(str(value).strip() and not pd.isna(value) for value in [row.reason, row.reviewer, row.date]):
            raise ValueError('Each correction requires reason, reviewer and date')
        manifest.loc[manifest.filename.eq(row.filename), 'final_label'] = row.corrected_label
    overlaps = pd.read_csv(directory/'external_excluded_scene_overlap.csv')
    rejection_file = directory/'scene_overlap_rejections.csv'
    rejections = pd.read_csv(rejection_file) if rejection_file.exists() else pd.DataFrame(columns=['filename', 'via_filename', 'reason', 'reviewer', 'date'])
    if rejections.duplicated(['filename', 'via_filename']).any():
        raise ValueError('Duplicate scene-overlap rejections')
    links = set(zip(overlaps.filename, overlaps.via_filename))
    for row in rejections.itertuples():
        if (row.filename, row.via_filename) not in links:
            raise ValueError('Rejected scene link is not in the automatic overlap graph')
        if not all(str(v).strip() and not pd.isna(v) for v in (row.reason, row.reviewer, row.date)):
            raise ValueError('Each scene rejection requires reason, reviewer and date')
        links.remove((row.filename, row.via_filename))
    excluded_names = {name for name, _ in links}
    manifest['excluded'] = manifest.filename.isin(excluded_names)
    manifest['exclusion_reason'] = np.where(manifest.excluded, 'confirmed_scene_overlap', '')
    sample = pd.read_csv(directory/'label_audit_sample.csv')
    changed = corrections[corrections.original_label.ne(corrections.corrected_label)]
    disagreements = int(changed.filename.isin(sample.filename).sum())
    rate = disagreements/len(sample)
    report.update(audit_completed=True, discordance_count=disagreements, discordance_rate=rate,
                  discordance_wilson=list(wilson(disagreements, len(sample))))
    if rate > report['audit_reject_above']:
        report.update(status='stopped_label_discordance', approved_by=None, approved_date=None)
        write_json(directory/'cleaning_report.json', report)
        raise ValueError('STOP: label-audit discordance exceeds 5%')
    atomic_csv(manifest, directory/'external_manifest_clean.csv')
    report.update(status='approved', approved_by=reviewer, approved_date=date,
                  manifest_sha256=sha256(directory/'external_manifest_clean.csv'),
                  label_corrections_sha256=sha256(directory/'label_corrections.csv'),
                  scene_rejections_sha256=sha256(rejection_file) if rejection_file.exists() else None,
                  accepted_scene_links=sorted([list(link) for link in links]),
                  reviewed_scene_overlap_images=len(excluded_names),
                  reviewed_principal_images=int((~manifest.excluded).sum()))
    report.setdefault('automatic_scene_overlap_images', report.get('scene_overlap_images', len(set(overlaps.filename))))
    report.setdefault('automatic_scene_overlap_links', report.get('scene_overlap_links', len(overlaps)))
    report.update(scene_overlap_images=len(excluded_names), scene_overlap_links=len(links),
                  main_external_images=int((~manifest.excluded).sum()))
    report.pop('main_external_images_pending_approval', None)
    if config_sha256 is not None:
        report['external_config_sha256'] = config_sha256
    write_json(directory/'cleaning_report.json', report)


def validate_cleaning(config):
    directory = ROOT/config['cleaning_directory']
    report = json.loads((directory/'cleaning_report.json').read_text())
    if (report.get('status') != 'approved' or not report.get('approved_by') or not report.get('approved_date')
            or not report.get('audit_completed') or report.get('discordance_rate') is None
            or report['discordance_rate'] > config['label_discordance_stop_above'] or report.get('smoke')):
        raise ValueError('Author-approved cleaning and completed label audit are required')
    for filename, key in [('external_manifest_clean.csv', 'manifest_sha256'), ('external_groups.csv', 'groups_sha256'),
                          ('label_corrections.csv', 'label_corrections_sha256'), ('external_manifest_raw.csv', 'raw_manifest_sha256'),
                          ('label_audit_sample.csv', 'audit_sample_sha256')]:
        if sha256(directory/filename) != report.get(key):
            raise ValueError(f'Approved cleaning artifact changed: {filename}')
    via_paths, _ = scan_dataset(ROOT/config['via_dataset'])
    via_hashes = {sha256(p) for p in via_paths}
    manifest = pd.read_csv(directory/'external_manifest_clean.csv')
    if manifest.filename.duplicated().any() or manifest.sha256.duplicated().any() or set(manifest.sha256) & via_hashes:
        raise ValueError('External manifest is not exact-disjoint from VIA')
    overlaps = pd.read_csv(directory/'external_excluded_scene_overlap.csv')
    if report.get('overlap_sha256') and sha256(directory/'external_excluded_scene_overlap.csv') != report['overlap_sha256']:
        raise ValueError('Automatic scene-overlap graph changed after review')
    rejected_file = directory/'scene_overlap_rejections.csv'
    rejected = set()
    if report.get('scene_rejections_sha256'):
        if sha256(rejected_file) != report['scene_rejections_sha256']:
            raise ValueError('Author scene-link decisions changed after review')
        rejection_rows = pd.read_csv(rejected_file)
        rejected = set(zip(rejection_rows.filename, rejection_rows.via_filename))
    accepted_names = {row.filename for row in overlaps.itertuples() if (row.filename, row.via_filename) not in rejected}
    if accepted_names & set(manifest[~manifest.excluded].filename) or set(manifest[manifest.excluded].filename) != accepted_names:
        raise ValueError('Scene-overlap images entered the principal external set')
    groups = pd.read_csv(directory/'external_groups.csv').set_index('filename').group_id
    if any(groups[row.filename] != row.group_id for row in manifest.itertuples()):
        raise ValueError('External group membership differs from frozen cleaning')
    if not set(manifest.final_label) <= {0, 1}:
        raise ValueError('Invalid external final labels')
    for row in manifest.itertuples():
        if Path(row.filename).name != row.filename or sha256(ROOT/config['dataset']/row.filename) != row.sha256:
            raise ValueError('External pixels changed after approval')
    return manifest, report


def evaluation_gate(config_path, output):
    config = json.loads(Path(config_path).read_text())
    registered = registered_protocol(config_path)
    if not (Path(output)/'final_models.json').exists():
        raise ValueError('Real final models have not been trained/frozen yet')
    freeze = json.loads((Path(output)/'final_models.json').read_text())
    if not freeze.get('frozen') or freeze.get('smoke'):
        raise ValueError('Real final models must be frozen before external inference')
    for key in ('config', 'protocol'):
        if freeze['preregistration'][key]['sha256'] != registered[key]['sha256']:
            raise ValueError('Current config/protocol differ from the preregistered model freeze')
    if not freeze.get('models') or 'thresholds.json' not in freeze.get('artifacts', {}):
        raise ValueError('Model freeze is missing its models or VIA-only threshold artifacts')
    if sha256(ROOT/config['groups']) != config['frozen_groups_sha256']:
        raise ValueError('Original VIA scene groups changed')
    for name, checksum in freeze['artifacts'].items():
        path = (Path(output)/name).resolve()
        if Path(output).resolve() not in path.parents or sha256(path) != checksum:
            raise ValueError(f'Frozen model/threshold artifact changed: {name}')
    via_paths, _ = scan_dataset(ROOT/config['via_dataset'])
    if dataset_content_fingerprint(via_paths) != freeze['dataset_sha256']:
        raise ValueError('Final models do not match unchanged VIA pixels')
    if config.get('fixed_oof', {}).get('design') == ARTICLE_OOF_DESIGN:
        thresholds = json.loads((Path(output)/'thresholds.json').read_text())
        if (thresholds.get('fixed_oof_design') != ARTICLE_OOF_DESIGN or
                thresholds.get('original_oof_source_sha256') != config['fixed_oof']['predictions_sha256'] or
                sha256(ROOT/config['fixed_oof']['source']) != config['fixed_oof']['predictions_sha256']):
            raise ValueError('Frozen thresholds differ from the registered original-article OOF')
        for filename in ('fixed_oof_predictions.csv', 'fixed_oof_provenance.json', 'calibrated_via_oof.csv'):
            if filename not in freeze['artifacts']:
                raise ValueError('Original-article OOF/calibration freeze is incomplete')
    else:
        nested = json.loads((ROOT/config['nested_results']/'run_metadata.json').read_text())
        if not nested.get('completed') or nested.get('smoke'):
            raise ValueError('Nested CV must finish before external evaluation')
    manifest, report = validate_cleaning(config)
    if report['external_config_sha256'] != registered['config']['sha256']:
        raise ValueError('Approved cleaning config differs from preregistration')
    return config, freeze, manifest


def verify_prediction_coverage(frame, manifest, models):
    if frame.duplicated(['filename', 'model', 'seed']).any():
        raise ValueError('Duplicate external image/model/seed prediction')
    expected = {(row['model'], row['seed']) for row in models}
    obtained = set(zip(frame.model, frame.seed))
    if expected != obtained:
        raise ValueError('Missing/extra external model/seed panels')
    truth = manifest.set_index('filename').final_label.to_dict()
    for _, panel in frame.groupby(['model', 'seed']):
        if set(panel.filename) != set(manifest.filename) or any(truth[r.filename] != r.true_label for r in panel.itertuples()):
            raise ValueError('External prediction coverage or labels differ from frozen manifest')


def write_once_predictions(frame, path):
    """A repeat evaluates into a candidate and compares bytes; cannot overwrite a first result."""
    path = Path(path)
    candidate = path.with_name('predictions_candidate.csv')
    atomic_csv(frame.sort_values(['model', 'seed', 'filename']), candidate)
    if path.exists():
        if candidate.read_bytes() != path.read_bytes():
            candidate.unlink()
            raise ValueError('External repeat predictions differ byte-for-byte; original results preserved')
        candidate.unlink()
    else:
        candidate.replace(path)


def fit_final_models(config_path, output, *, device='cpu', smoke=False):
    config = json.loads(Path(config_path).read_text())
    registration = None if smoke else registered_protocol(config_path)
    if not smoke:
        print(json.dumps(external_estimate(config, 'final-models'), indent=2), flush=True)
    paths, labels = scan_dataset(ROOT/config['via_dataset'])
    if sha256(ROOT/config['groups']) != config['frozen_groups_sha256']:
        raise ValueError('Frozen VIA groups changed')
    groups = pd.read_csv(ROOT/config['groups']).set_index('filename').loc[[p.name for p in paths], 'group_id'].to_numpy()
    train, validation = grouped_validation(labels, groups, config['seed'])
    selection, experiments = fixed_selection(config)
    identity = {'config_sha256': sha256(config_path), 'dataset_sha256': dataset_content_fingerprint(paths),
                'group_sha256': sha256(ROOT/config['groups']), 'smoke': smoke, 'device': device,
                'sources': source_identity(), 'runtime': runtime()}
    checked_identity(output, identity)
    freeze_path = output/'final_models.json'
    if not smoke and freeze_path.exists():
        previous = json.loads(freeze_path.read_text())
        if previous.get('frozen') and not previous.get('smoke'):
            if previous['identity'] != identity:
                raise ValueError('Frozen final model identity changed')
            for key in ('config', 'protocol'):
                if previous['preregistration'][key]['sha256'] != registration[key]['sha256']:
                    raise ValueError('Frozen protocol/config changed')
            for filename, checksum in previous['artifacts'].items():
                if sha256(output/filename) != checksum:
                    raise ValueError(f'Frozen final artifact changed: {filename}')
            print('Final models already frozen; reuse without training/benchmark changes', flush=True)
            return
    write_json(output/'run_metadata.json', {'completed': False, 'smoke': smoke, 'identity': identity,
               'training_device_requested': device, 'external_classification_predictions': 0})
    if not smoke:
        via_thresholds(config, output)
    atomic_csv(pd.DataFrame({'filename': [p.name for p in paths], 'split': np.where(np.isin(np.arange(len(paths)), validation),
               'validation', 'train'), 'group_id': groups}), output/'validation_assignments.csv')
    fine_config = json.loads((ROOT/config['fine_tuning_config']).read_text())
    store = CanonicalFeatureStore(paths, device=device, threads=config['cpu_threads'])
    models, artifacts, costs = [], {}, []
    fixed_references = {**selection['selected'], 'winner': selection['winner']}
    saved = {}
    for procedure, name in fixed_references.items():
        if name not in saved:
            experiment = dict(experiments[name])
            if smoke:
                fit_indices = np.r_[np.flatnonzero(labels == 0)[:4], np.flatnonzero(labels == 1)[:4]]
                if experiment['reduction'] == 'pca':
                    experiment['components'] = 4
            else:
                fit_indices = np.arange(len(paths))
            job = output/'models'/name
            job_identity = {'run': digest(identity), 'experiment': experiment, 'indices': fit_indices.tolist()}
            checked_identity(job, job_identity)
            if not completed_job(job, job_identity):
                matrices = [store.get(architecture)[fit_indices] for architecture in experiment['extractors']]
                with threadpool_limits(config['cpu_threads']):
                    pipeline, _, reduction, training, _ = fit_fixed_pipeline(experiment, matrices, labels[fit_indices], config['seed'])
                joblib.dump(pipeline, job/'pipeline.joblib')
                before = pipeline['classifier'].predict(transform_fixed(pipeline, matrices))
                restored = joblib.load(job/'pipeline.joblib')
                after = restored['classifier'].predict(transform_fixed(restored, matrices))
                if not np.array_equal(before, after):
                    raise ValueError('Fixed pipeline reload changed VIA decisions')
                write_json(job/'cost.json', {'model': name, 'training_seconds': training, 'reduction_seconds': reduction,
                                           'reload_decisions_identical': True, 'smoke': smoke})
                finish_job(job, job_identity, ['pipeline.joblib', 'cost.json'])
            saved[name] = {'model': name, 'seed': -1, 'family': 'fixed', 'procedure': procedure,
                           'experiment': experiment, 'pipeline': str((job/'pipeline.joblib').relative_to(output))}
            costs.append(json.loads((job/'cost.json').read_text()))
        models.append({**saved[name], 'procedure': procedure})
    # Save complete native encoders once. Smoke verifies canonical cached features, plus FT encoders below.
    tf = configure_tensorflow(config['seed'], device, config['cpu_threads']) if store.tf is None else store.tf
    encoder_files = {}
    if not smoke:
        for architecture in sorted({a for item in saved.values() for a in item['experiment']['extractors']}):
            encoder_path = output/'encoders'/f'{architecture}.keras'
            encoder_path.parent.mkdir(parents=True, exist_ok=True)
            if not encoder_path.exists():
                model, _, _ = _tensorflow_model(architecture)
                model.trainable = False
                save_deployment(model, encoder_path)
                del model
                tf.keras.backend.clear_session()
            encoder_files[architecture] = str(encoder_path.relative_to(output))
    if smoke:
        # All five architectures, one epoch/seed on VIA; these cannot pass the freeze gate.
        train = np.r_[train[labels[train] == 0][:4], train[labels[train] == 1][:4]]
        validation = np.r_[validation[labels[validation] == 0][:2], validation[labels[validation] == 1][:2]]
        fine_config = {**fine_config, 'epochs': 1, 'batch_size': 2, 'verbose': 0}
    seeds = [config['seed']] if smoke else config['initialization_seeds']
    for architecture in config['architectures']:
        for seed in seeds:
            name = f'ft_{architecture}'
            job = output/'models'/f'{name}__seed_{seed}'
            request = {'experiment': name, 'architecture': architecture, 'configuration': 'D',
                       'seed': seed, 'fold': 0, 'config': fine_config, 'family': 'fine_tuning'}
            job_identity = {'run': digest(identity), 'request': request, 'train': train.tolist(), 'validation': validation.tolist()}
            checked_identity(job, job_identity)
            if not completed_job(job, job_identity):
                manifest_path = job/'manifest.json'
                manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'state': 'pending', 'request': request}
                if not smoke:
                    manifest['batch_size'] = config['batch_size_overrides'].get(architecture, fine_config['batch_size'])
                model, cost = train_neural(request, job, manifest, paths, labels, train, validation, validation, tf)
                write_json(job/'cost.json', {'model': name, 'seed': seed, **cost, 'smoke': smoke})
                finish_job(job, job_identity, ['model.keras', 'cost.json'])
                del model
                tf.keras.backend.clear_session()
                gc.collect()
            costs.append(json.loads((job/'cost.json').read_text()))
            models.append({'model': name, 'seed': seed, 'family': 'fine_tuning', 'architecture': architecture,
                           'model_path': str((job/'model.keras').relative_to(output)), 'procedure': name})
            print(f'Completed {name}/seed{seed}; VIA only; smoke={smoke}', flush=True)
    # Winner aliases an existing finalist rather than duplicate external model/seed observations.
    models = list({(item['model'], item['seed']): item for item in reversed(models)}.values())
    if not smoke:
        latency = []
        thresholds = json.loads((output/'thresholds.json').read_text())['thresholds']
        for entry in models:
            request = {'dataset': str(ROOT/config['via_dataset']), 'fold_assignments': str(ROOT/'results/full_search/fold_assignments.csv'),
                       'fold': 1, 'config': {'benchmark': config['benchmark']}}
            pipeline = None
            if entry['family'] == 'fixed':
                request['extractors'] = entry['experiment']['extractors']
                request['architecture'] = request['extractors'][0]
                pipeline = joblib.load(output/entry['pipeline'])
                adapter = joblib.load(output/thresholds[entry['procedure']]['adapter'])
                pipeline['classifier'] = CalibratedDecision(pipeline['classifier'], adapter)
                model_files = [encoder_files[name] for name in request['extractors']]
            else:
                request['architecture'] = entry['architecture']; model_files = [entry['model_path']]
            latency.append({'model': entry['model'], 'seed': entry['seed'],
                            **benchmark_saved(request, output, model_files, tf, pipeline)})
        atomic_csv(pd.DataFrame(latency), output/'latency.csv')
        artifacts['latency.csv'] = sha256(output/'latency.csv')
    for item in models:
        for key in ('pipeline', 'model_path'):
            if key in item:
                artifacts[item[key]] = sha256(output/item[key])
    for name in encoder_files.values():
        artifacts[name] = sha256(output/name)
    if not smoke:
        artifacts['thresholds.json'] = sha256(output/'thresholds.json')
        for filename in ('calibrated_via_oof.csv', 'fixed_oof_predictions.csv', 'fixed_oof_provenance.json'):
            if (output/filename).exists():
                artifacts[filename] = sha256(output/filename)
        for path in output.glob('adapter_*.joblib'):
            artifacts[path.name] = sha256(path)
    write_json(output/'feature_provenance.json', store.provenance)
    atomic_csv(pd.DataFrame(costs), output/'computational_costs.csv')
    write_json(output/'final_models.json', {'frozen': not smoke, 'smoke': smoke, 'identity': identity,
               'preregistration': registration, 'dataset_sha256': dataset_content_fingerprint(paths),
               'models': models, 'encoders': encoder_files, 'artifacts': artifacts,
               'article_finalists': selection['selected'], 'global_winner': selection['winner'],
               'external_classification_predictions': 0})
    write_json(output/'run_metadata.json', {'completed': not smoke, 'requested_run_completed': True,
               'smoke': smoke, 'identity': identity, 'training_device_requested': device,
               'model_panels': len(models), 'final_models_sha256': sha256(output/'final_models.json'),
               'groups_sha256': identity['group_sha256'], 'external_classification_predictions': 0})


def predict_frozen(config, freeze, manifest, output, tf):
    """Only called after all gates; no fitting and no threshold selection."""
    manifest = manifest.sort_values('filename')
    paths = [ROOT/config['dataset']/name for name in manifest.filename]
    threshold_data = json.loads((output/'thresholds.json').read_text())['thresholds']
    rows = []
    for entry in freeze['models']:
        job = output/'prediction_jobs'/f"{entry['model']}__seed_{entry['seed']}"
        job_identity = {'model': entry, 'freeze_sha256': sha256(output/'final_models.json'),
                        'manifest_sha256': digest(manifest.fillna('').to_dict('records')),
                        'sources': source_identity(), 'runtime': runtime()}
        checked_identity(job, job_identity)
        if completed_job(job, job_identity):
            panel = pd.read_csv(job/'predictions.csv', float_precision='round_trip')
            rows.extend(panel.to_dict('records'))
            continue
        first_row = len(rows)
        if entry['family'] == 'fixed':
            pipeline = joblib.load(output/entry['pipeline'])
            parts = []
            for architecture in entry['experiment']['extractors']:
                encoder = tf.keras.models.load_model(output/freeze['encoders'][architecture], compile=False)
                processor, size = preprocessing(architecture)
                matrices = []
                for offset in range(0, len(paths), 4):
                    batch_paths = paths[offset:offset+4]
                    values = encoder.predict(image_batch(batch_paths, processor, size, tf), verbose=0)
                    matrices.append(np.asarray(values).reshape(len(batch_paths), -1))
                parts.append(np.concatenate(matrices))
                del encoder
                tf.keras.backend.clear_session()
            prepared = transform_fixed(pipeline, parts)
            native = continuous_scores(pipeline['classifier'], prepared)
            kind = 'margin' if hasattr(pipeline['classifier'], 'decision_function') else 'probability'
            procedure = entry['procedure']
            adapter = joblib.load(output/threshold_data[procedure]['adapter'])
            probabilities = adapter_probability(adapter, native, kind)
        else:
            model = tf.keras.models.load_model(output/entry['model_path'], compile=False)
            processor, size = preprocessing(entry['architecture'])
            values = []
            for offset in range(0, len(paths), 4):
                values.append(model.predict(image_batch(paths[offset:offset+4], processor, size, tf), verbose=0)[:, 1])
            probabilities = native = np.concatenate(values)
            kind, procedure = 'probability', entry['procedure']
            del model
            tf.keras.backend.clear_session()
        if (len(probabilities) != len(manifest) or not np.isfinite(probabilities).all()
                or not np.isfinite(native).all() or np.any((probabilities < 0) | (probabilities > 1))):
            raise ValueError('External frozen model produced invalid scores/probabilities')
        operating = threshold_data[procedure]['threshold']
        for row, score, probability in zip(manifest.itertuples(), native, probabilities):
            rows.append({'filename': row.filename, 'true_label': int(row.final_label), 'group_id': row.group_id,
                         'model': entry['model'], 'procedure': procedure, 'family': entry['family'], 'seed': entry['seed'],
                         'native_score': float(score), 'score_kind': kind, 'score_clear': float(probability),
                         'predicted_label_default': int(probability >= .5),
                         'predicted_label_safety': int(probability > operating), 'scene_overlap': bool(row.excluded)})
        atomic_csv(pd.DataFrame(rows[first_row:]), job/'predictions.csv')
        finish_job(job, job_identity, ['predictions.csv'])
    frame = pd.DataFrame(rows)
    verify_prediction_coverage(frame, manifest, freeze['models'])
    return frame


def evaluate(config_path, output, *, device):
    config, freeze, manifest = evaluation_gate(config_path, output)
    first_path = output/'first_evaluation.json'
    identity = {'sources': source_identity(), 'config_sha256': sha256(config_path),
                'model_freeze_sha256': sha256(output/'final_models.json'),
                'cleaning_report_sha256': sha256(ROOT/config['cleaning_directory']/'cleaning_report.json'),
                'device': device, 'runtime': runtime()}
    if first_path.exists():
        first = json.loads(first_path.read_text())
        if first['identity'] != identity:
            raise ValueError('Code/runtime/input changed after first external evaluation; record PROVENANCE deviation before any new execution')
        if first.get('predictions_sha256') and sha256(output/'predictions.csv') != first['predictions_sha256']:
            raise ValueError('First external predictions changed')
    else:
        write_json(first_path, {'identity': identity, 'status': 'started'})
    tf = configure_tensorflow(config['seed'], device, config['cpu_threads'])
    frame = predict_frozen(config, freeze, manifest, output, tf)
    write_once_predictions(frame, output/'predictions.csv')
    metrics = []
    for (model, seed), panel in frame.groupby(['model', 'seed']):
        for sensitivity, rows in [('principal', panel[~panel.scene_overlap]), ('include_scene_overlap', panel)]:
            for point in ('default', 'safety'):
                metrics.append({'model': model, 'seed': seed, 'panel': sensitivity, 'operating_point': point,
                                **prediction_metrics(rows.true_label, rows[f'predicted_label_{point}'], rows.score_clear)})
    atomic_csv(pd.DataFrame(metrics), output/'metrics.csv')
    write_json(first_path, {'identity': identity, 'status': 'completed', 'predictions_sha256': sha256(output/'predictions.csv'),
               'metrics_sha256': sha256(output/'metrics.csv')})


def evaluation_smoke(output):
    """Full analysis contract on generated numbers; never opens an external image."""
    config = json.loads((ROOT/'configs/external_test.json').read_text())
    fixed = json.loads((ROOT/config['fixed_reference_config']).read_text())['experiments']
    rng = np.random.default_rng(1980)
    population = pd.DataFrame({'filename': [f'fixture_{i:03d}' for i in range(40)],
                'true_label': np.arange(40)%2, 'group_id': np.repeat(np.arange(20), 2),
                'scene_overlap': np.arange(40) < 2})
    entries = [{'model': e['name'], 'procedure': e['approach'], 'seed': -1, 'family': 'fixed'} for e in fixed]
    entries += [{'model': f'ft_{architecture}', 'procedure': f'ft_{architecture}', 'seed': seed, 'family': 'fine_tuning'}
                for architecture in config['architectures'] for seed in config['initialization_seeds']]
    frames, metrics = [], []
    for entry in entries:
        frame = population.copy()
        for key, value in entry.items(): frame[key] = value
        score = np.clip(.2+.6*frame.true_label+rng.normal(0, .25, len(frame)), .001, .999)
        frame['score_clear'] = score; frame['native_score'] = score; frame['score_kind'] = 'probability'
        frame['predicted_label_default'] = (score >= .5).astype(int)
        frame['predicted_label_safety'] = (score > .8).astype(int)
        frames.append(frame)
        for sensitivity, rows in [('principal', frame[~frame.scene_overlap]), ('include_scene_overlap', frame)]:
            for point in ('default', 'safety'):
                metrics.append({'model': entry['model'], 'seed': entry['seed'], 'panel': sensitivity, 'operating_point': point,
                                **prediction_metrics(rows.true_label, rows[f'predicted_label_{point}'], rows.score_clear)})
    predictions = pd.concat(frames)
    synthetic_manifest = population.rename(columns={'true_label': 'final_label'})
    verify_prediction_coverage(predictions, synthetic_manifest, entries)
    output.mkdir(parents=True, exist_ok=True)
    write_once_predictions(predictions, output/'predictions.csv')
    atomic_csv(pd.DataFrame(metrics), output/'metrics.csv')
    write_json(output/'thresholds.json', {'frozen': False, 'thresholds': {e['procedure']: {'threshold': .8} for e in entries}})
    write_json(output/'first_evaluation.json', {'status': 'completed', 'synthetic_only': True,
               'external_classification_predictions': 0, 'predictions_sha256': sha256(output/'predictions.csv'),
               'metrics_sha256': sha256(output/'metrics.csv'), 'identity': {'source': 'generated-number fixture'}})
    print(f'Evaluation contract smoke passed; synthetic numbers only: {output}')


def learning_split(manifest, seed=1980):
    clean = manifest[~manifest.excluded].sort_values('filename').reset_index(drop=True)
    splitter = StratifiedGroupKFold(2, shuffle=True, random_state=seed)
    pool, test = next(splitter.split(np.zeros(len(clean)), clean.final_label, clean.group_id))
    result = clean.copy()
    result['split'] = np.where(np.isin(np.arange(len(clean)), test), 'test_B', 'pool')
    if set(result[result.split.eq('test_B')].group_id) & set(result[result.split.eq('pool')].group_id):
        raise ValueError('Learning pool overlaps test-B scenes')
    return result


def nested_group_subsets(pool, fractions=(0., .25, .5, 1.), seed=1980):
    rng = np.random.default_rng(seed)
    strata = {}
    for group, rows in pool.groupby('group_id', sort=True):
        strata.setdefault(tuple(sorted(rows.final_label.unique())), []).append(group)
    ordered = {key: list(rng.permutation(groups)) for key, groups in sorted(strata.items())}
    result = {}
    for fraction in fractions:
        selected = []
        for options in ordered.values():
            target = fraction*int(pool[pool.group_id.isin(options)].shape[0])
            count = 0
            for group in options:
                size = int(pool.group_id.eq(group).sum())
                if count >= target:
                    break
                selected.append(group); count += size
        result[float(fraction)] = pool[pool.group_id.isin(selected)].filename.tolist()
    return result


def learning_curve(config_path, final_output, *, device='cpu', smoke=False, output=None):
    """Secondary study after the principal predictions freeze; never selects configurations."""
    config = json.loads(Path(config_path).read_text())
    output = guard_output(output or (config['learning_curve']['output'] if not smoke else 'results/learning_curve_smoke'))
    if smoke:
        # CPU control-flow fixture; no external image is opened/classified/trained.
        fixture = pd.DataFrame({'filename': [f'fixture_{i:03d}' for i in range(40)],
                    'final_label': np.arange(40)%2, 'group_id': np.repeat(np.arange(20), 2), 'excluded': False})
        split = learning_split(fixture, config['seed'])
        subsets = nested_group_subsets(split[split.split.eq('pool')])
        atomic_csv(split, output/'split.csv')
        write_json(output/'subsets.json', {str(k): v for k, v in subsets.items()})
        write_json(output/'smoke_report.json', {'passed': True, 'synthetic_only': True,
                    'external_classification_predictions': 0, 'external_images_trained': 0,
                    'nested_sizes': [len(v) for v in subsets.values()]})
        return
    config, freeze, manifest = evaluation_gate(config_path, final_output)
    first = json.loads((final_output/'first_evaluation.json').read_text())
    if first['status'] != 'completed' or sha256(final_output/'predictions.csv') != first['predictions_sha256']:
        raise ValueError('Principal external evaluation must be frozen before the secondary curve')
    if first['identity']['sources'] != source_identity():
        raise ValueError('Secondary code changed after principal evaluation; register PROVENANCE deviation')
    estimate = external_estimate(config, 'learning-curve')
    print(json.dumps(estimate, indent=2), flush=True)
    split = learning_split(manifest, config['seed'])
    pool, test_b = split[split.split.eq('pool')], split[split.split.eq('test_B')]
    subsets = nested_group_subsets(pool, config['learning_curve']['fractions'], config['seed'])
    identity = {'config_sha256': sha256(config_path), 'principal_predictions': first['predictions_sha256'],
                'models': sha256(final_output/'final_models.json'), 'sources': source_identity(),
                'manifest': sha256(ROOT/config['cleaning_directory']/'external_manifest_clean.csv'),
                'subsets': {str(k): v for k, v in subsets.items()}, 'runtime': runtime(), 'device': device}
    checked_identity(output, identity)
    # Persist before any secondary fitting.
    atomic_csv(split, output/'split.csv')
    write_json(output/'subsets.json', identity['subsets'])
    write_json(output/'run_metadata.json', {'completed': False, 'identity': identity, 'estimate': estimate})
    paths, labels = scan_dataset(ROOT/config['via_dataset'])
    via_count = len(paths)
    via_groups = pd.read_csv(ROOT/config['groups']).set_index('filename').loc[[p.name for p in paths], 'group_id'].to_numpy()
    clean = manifest[~manifest.excluded].sort_values('filename')
    paths += [ROOT/config['dataset']/name for name in clean.filename]
    labels = np.r_[labels, clean.final_label.to_numpy()]
    groups = np.r_[[f'via:{group}' for group in via_groups], [f'new:{group}' for group in clean.group_id]]
    index = {p.name: i for i, p in enumerate(paths)}
    b_indices = np.array([index[name] for name in test_b.filename])
    tf = configure_tensorflow(config['seed'], device, config['cpu_threads'])
    via_store = CanonicalFeatureStore(paths[:via_count], device=device, threads=config['cpu_threads'])
    # Image-only external forwards may include B; reducers/classifiers never fit on B.
    matrices = {}
    for architecture, encoder_name in freeze['encoders'].items():
        cache = output/'feature_cache'/architecture
        cache.mkdir(parents=True, exist_ok=True)
        cache_identity = {'encoder_sha256': sha256(final_output/encoder_name),
                          'ordered_pixels': [{'filename': p.name, 'sha256': sha256(p)} for p in paths[via_count:]],
                          'runtime': runtime(), 'batch_size': 4}
        checked_identity(cache, cache_identity)
        if not completed_job(cache, cache_identity):
            model = tf.keras.models.load_model(final_output/encoder_name, compile=False)
            processor, size = preprocessing(architecture)
            values = []
            for offset in range(via_count, len(paths), 4):
                batch_paths = paths[offset:offset+4]
                values.append(np.asarray(model.predict(image_batch(batch_paths, processor, size, tf), verbose=0)).reshape(len(batch_paths), -1))
            temp = cache/'features.tmp.npy'; np.save(temp, np.concatenate(values)); temp.replace(cache/'features.npy')
            finish_job(cache, cache_identity, ['features.npy'])
            del model; tf.keras.backend.clear_session()
        matrices[architecture] = np.vstack([via_store.get(architecture), np.load(cache/'features.npy', mmap_mode='r')])
    _, experiments = fixed_selection(config)
    thresholds = json.loads((final_output/'thresholds.json').read_text())['thresholds']
    fine_config = json.loads((ROOT/config['fine_tuning_config']).read_text())
    original = pd.read_csv(final_output/'predictions.csv', float_precision='round_trip')
    original = original[original.filename.isin(test_b.filename)]
    all_predictions, all_costs = [], []
    for fraction, filenames in subsets.items():
        train_indices = np.r_[np.arange(via_count), np.array([index[name] for name in filenames], dtype=int)]
        directory = output/f'size_{fraction:.2f}'; directory.mkdir(parents=True, exist_ok=True)
        if fraction == 0:
            frame = original.copy(); frame['pool_fraction'] = 0.; frame['training_images'] = via_count
            atomic_csv(frame, directory/'predictions.csv')
            all_predictions.append(frame)
            continue
        fit_local, validation_local = grouped_validation(labels[train_indices], groups[train_indices], config['seed'])
        fit_train, validation = train_indices[fit_local], train_indices[validation_local]
        if set(groups[train_indices]) & set(groups[b_indices]):
            raise ValueError('A test-B group entered the learning training pool')
        frames = []
        for entry in freeze['models']:
            if entry['family'] == 'fine_tuning' and entry['architecture'] not in estimate['ft_architectures']:
                continue
            job = directory/'jobs'/f"{entry['model']}__seed_{entry['seed']}"
            job_identity = {'study': digest(identity), 'entry': entry, 'train': train_indices.tolist(),
                            'fit_train': fit_train.tolist(), 'validation': validation.tolist(), 'test_B': b_indices.tolist()}
            checked_identity(job, job_identity)
            if not completed_job(job, job_identity):
                started = perf_counter()
                if entry['family'] == 'fixed':
                    experiment = experiments[entry['model']]
                    with threadpool_limits(config['cpu_threads']):
                        pipeline, _, reduction, training, _ = fit_fixed_pipeline(experiment,
                           [matrices[a][train_indices] for a in experiment['extractors']], labels[train_indices], config['seed'])
                    joblib.dump(pipeline, job/'pipeline.joblib')
                    prepared = transform_fixed(pipeline, [matrices[a][b_indices] for a in experiment['extractors']])
                    native = continuous_scores(pipeline['classifier'], prepared)
                    kind = 'margin' if hasattr(pipeline['classifier'], 'decision_function') else 'probability'
                    adapter = joblib.load(final_output/thresholds[entry['procedure']]['adapter'])
                    probability = adapter_probability(adapter, native, kind)
                    cost = {'training_seconds': training+reduction, 'update_reduction_seconds': reduction,
                            'update_classifier_seconds': training}
                    model_files = ['pipeline.joblib']
                else:
                    request = {'experiment': entry['model'], 'architecture': entry['architecture'], 'configuration': 'D',
                               'seed': entry['seed'], 'fold': 0, 'config': fine_config, 'family': 'fine_tuning'}
                    manifest_path = job/'manifest.json'
                    training_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'state': 'pending', 'request': request,
                        'batch_size': config['batch_size_overrides'].get(entry['architecture'], fine_config['batch_size'])}
                    model, cost = train_neural(request, job, training_manifest, paths, labels, fit_train, validation, validation, tf)
                    processor, size = preprocessing(entry['architecture']); values = []
                    for offset in range(0, len(b_indices), 4):
                        batch_paths = [paths[i] for i in b_indices[offset:offset+4]]
                        values.append(model.predict(image_batch(batch_paths, processor, size, tf), verbose=0)[:, 1])
                    probability = native = np.concatenate(values); kind = 'probability'
                    del model; tf.keras.backend.clear_session(); gc.collect(); model_files = ['model.keras']
                frame = pd.DataFrame({'filename': [paths[i].name for i in b_indices], 'true_label': labels[b_indices],
                    'group_id': test_b.set_index('filename').loc[[paths[i].name for i in b_indices], 'group_id'].to_numpy(),
                    'model': entry['model'], 'seed': entry['seed'], 'family': entry['family'], 'procedure': entry['procedure'],
                    'native_score': native, 'score_kind': kind, 'score_clear': probability,
                    'predicted_label_default': (probability >= .5).astype(int),
                    'predicted_label_safety': (probability > thresholds[entry['procedure']]['threshold']).astype(int),
                    'scene_overlap': False, 'pool_fraction': fraction, 'training_images': len(train_indices)})
                atomic_csv(frame, job/'predictions.csv')
                write_json(job/'cost.json', {'model': entry['model'], 'seed': entry['seed'], 'pool_fraction': fraction,
                           'training_images': len(train_indices), 'elapsed_seconds': perf_counter()-started, **cost})
                finish_job(job, job_identity, model_files+['predictions.csv', 'cost.json'])
            frames.append(pd.read_csv(job/'predictions.csv', float_precision='round_trip'))
            all_costs.append(json.loads((job/'cost.json').read_text()))
        frame = pd.concat(frames); atomic_csv(frame, directory/'predictions.csv'); all_predictions.append(frame)
    predictions = pd.concat(all_predictions)
    atomic_csv(predictions, output/'predictions.csv')
    atomic_csv(pd.DataFrame(all_costs), output/'update_costs.csv')
    from src.study_statistics import bootstrap_intervals
    rows = []
    for (fraction, model), panel in predictions.groupby(['pool_fraction', 'model']):
        names = sorted(panel.filename.unique())
        truth = panel.groupby('filename').true_label.first().loc[names].to_numpy()
        group_ids = panel.groupby('filename').group_id.first().loc[names].to_numpy()
        scores = panel.pivot(index='filename', columns='seed', values='score_clear').loc[names].to_numpy().T
        for point in ('default', 'safety'):
            decisions = panel.pivot(index='filename', columns='seed', values=f'predicted_label_{point}').loc[names].to_numpy().T
            metrics = [prediction_metrics(truth, d, s) for d, s in zip(decisions, scores)]
            intervals = bootstrap_intervals(truth, decisions, scores, groups=group_ids,
                iterations=config['bootstrap_iterations'], seed=config['seed'],
                endpoints=['accuracy', 'balanced_accuracy', 'mcc', 'roc_auc', 'obstructed_recall', 'obstructed_false_negative_rate'])
            for endpoint in metrics[0]:
                interval = intervals.get(endpoint, [None, None])
                rows.append({'pool_fraction': fraction, 'model': model, 'training_images': panel.training_images.iloc[0],
                             'operating_point': point, 'endpoint': endpoint,
                             'estimate': float(np.mean([m[endpoint] for m in metrics])), 'ci_low': interval[0], 'ci_high': interval[1]})
    from scripts.analyze_nested_cv import table
    tables = output/'manuscript_tables'; tables.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows); table(summary, tables, 'learning_curve_metrics')
    table(pd.DataFrame(all_costs), tables, 'update_costs')
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, endpoint in zip(axes, ('accuracy', 'obstructed_recall')):
        for model, panel in summary[summary.operating_point.eq('default') & summary.endpoint.eq(endpoint)].groupby('model'):
            panel = panel.sort_values('training_images'); axis.plot(panel.training_images, panel.estimate, marker='o', label=model)
            axis.fill_between(panel.training_images, panel.ci_low, panel.ci_high, alpha=.12)
        axis.set(xlabel='Training images (VIA + external pool)', ylabel=endpoint)
    axes[1].legend(fontsize=6); fig.tight_layout(); fig.savefig(tables/'learning_curve.png', dpi=160); fig.savefig(tables/'learning_curve.pdf'); plt.close(fig)
    write_json(output/'run_metadata.json', {'completed': True, 'identity': identity, 'estimate': estimate,
               'test_B_sha256': sha256(output/'split.csv'), 'no_configuration_selection': True,
               'outputs': {name: sha256(output/name) for name in ('predictions.csv', 'update_costs.csv')}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/external_test.json'))
    parser.add_argument('--stage', choices=['final-models', 'evaluate', 'learning-curve', 'approve-cleaning'], default='final-models')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--estimate-output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--reviewer'); parser.add_argument('--review-date')
    parser.add_argument('--audit-reviewed', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if args.dry_run:
        result = external_estimate(config, args.stage)
        print(json.dumps(result, indent=2))
        if args.estimate_output:
            write_json(args.estimate_output, result)
        return
    output = guard_output(args.output_dir or config['output'])
    if args.smoke:
        names = {'final-models': 'external_models_smoke', 'evaluate': 'external_evaluation_smoke', 'learning-curve': 'learning_curve_smoke'}
        output = guard_output(args.output_dir or f"results/{names.get(args.stage, 'review_smoke')}")
    if args.stage == 'approve-cleaning':
        apply_label_review(ROOT/config['cleaning_directory'], reviewer=args.reviewer or '',
                           date=args.review_date or '', audit_reviewed=args.audit_reviewed,
                           config_sha256=sha256(args.config))
    elif args.stage == 'final-models':
        fit_final_models(args.config, output, device=args.device, smoke=args.smoke)
    elif args.stage == 'evaluate':
        if args.smoke:
            evaluation_smoke(output)
            return
        evaluate(args.config, output, device=args.device)
    else:
        learning_curve(args.config, guard_output(config['output']), device=args.device, smoke=args.smoke,
                       output=output if args.output_dir or args.smoke else None)


if __name__ == '__main__':
    main()
