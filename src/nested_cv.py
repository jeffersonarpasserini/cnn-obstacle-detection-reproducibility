#!/usr/bin/env python3
"""Estimate the full article selection procedure using untouched outer folds."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from src.artifact import (scan_dataset, prepare_fold_features, ReliefRankingCache, build_classifier,
                          continuous_scores, prediction_metrics, preprocessing_key, dataset_content_fingerprint)
from src.fine_tuning import load_outer_folds
from src.run_experiments import group_experiments
from src.selection import select_procedures
from src.study_features import CanonicalFeatureStore
from src.study_io import (ROOT, sha256, digest, write_json, atomic_csv, guard_output, runtime,
                          checked_identity, completed_job, finish_job, source_identity)
from src.study_statistics import safety_threshold
from src.study_estimates import nested_estimate


def outer_splits(paths, labels, config, groups_path=None):
    if groups_path is None:
        return load_outer_folds(ROOT / config['outer_assignments'], paths, labels)
    if sha256(groups_path) != config['frozen_groups_sha256']:
        raise ValueError('Frozen VIA scene groups changed')
    frame = pd.read_csv(groups_path)
    if frame.filename.duplicated().any() or set(frame.filename) != {p.name for p in paths}:
        raise ValueError('Group assignment coverage differs from VIA')
    groups = frame.set_index('filename').loc[[p.name for p in paths], 'group_id'].to_numpy()
    splitter = StratifiedGroupKFold(config['outer_folds'], shuffle=True, random_state=config['seed'])
    splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
    diagnostics = pd.read_csv(ROOT / config['group_diagnostics'])
    for number, (train, test) in enumerate(splits, 1):
        row = diagnostics[diagnostics.fold.eq(number)].iloc[0]
        if (len(test), int(labels[test].sum()), int((labels[test] == 0).sum())) != (row.test_image_count, row.clear, row.nonclear):
            raise ValueError('Grouped outer folds differ from the frozen diagnostics')
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Scene crosses grouped outer split')
    return splits


def inner_splits(outer_train, labels, outer_number, config, groups=None, *, smoke=False):
    n = 2 if smoke else config['inner_folds']
    cls = StratifiedKFold if groups is None else StratifiedGroupKFold
    splitter = cls(n, shuffle=True, random_state=config['seed']+outer_number)
    local = splitter.split(np.zeros(len(outer_train)), labels[outer_train],
                           None if groups is None else groups[outer_train])
    return [(outer_train[train], outer_train[test]) for train, test in local]


def assert_no_leakage(outer_train, outer_test, inner, groups=None):
    coverage = []
    for train, validation in inner:
        if set(train) & set(validation) or (set(train)|set(validation)) != set(outer_train):
            raise ValueError('Inner folds must partition only outer training data')
        if (set(train)|set(validation)) & set(outer_test):
            raise ValueError('Outer test image entered internal selection')
        if groups is not None and set(groups[train]) & set(groups[validation]):
            raise ValueError('Scene crosses inner training/validation split')
        coverage.extend(validation.tolist())
    if sorted(coverage) != sorted(outer_train.tolist()):
        raise ValueError('Inner OOF coverage is not exactly once')


def smoke_search():
    experiments = []
    for approach, extractors in [('A', ['MobileNet']), ('A', ['ResNet50']),
                                 ('B', ['MobileNet']), ('B', ['ResNet50']),
                                 ('C', ['MobileNet', 'ResNet50']), ('D', ['MobileNet', 'ResNet50'])]:
        for count in ([None] if approach == 'A' else [2, 4]):
            for classifier in ('logistic', 'linear_svm'):
                name = f"{approach.lower()}_{'+'.join(extractors).lower()}_{count}_{classifier}"
                experiments.append({'name': name, 'approach': approach, 'extractors': extractors,
                                    'reduction': 'full' if approach == 'A' else 'pca',
                                    'components': count, 'classifier': classifier, 'scale': False})
    return {'seed': 1980, 'experiments': experiments}


def infeasible_reason(experiment, feature_map, train):
    components = experiment.get('components')
    if experiment.get('reduction') == 'pca':
        dimensions = [feature_map[name].shape[1] for name in experiment['extractors']]
        width = sum(dimensions) if experiment['approach'] == 'C' else min(dimensions)
        if components > min(len(train), width):
            return f'PCA components={components} exceeds min(train={len(train)}, features={width})'
    if experiment.get("reduction") == "umap" and components + 1 >= len(train):
        return f"UMAP spectral initialization requires components+1 < training samples: {components} / {len(train)}"
    return None


def score_kind(classifier):
    return 'margin' if hasattr(classifier, 'decision_function') else 'probability'


def classify_job(experiment, x_train, x_test, labels, train, test, seed, inner_number,
                 directory, identity, preparation_seconds):
    if completed_job(directory, identity):
        return pd.read_csv(directory/'metrics.csv').iloc[0].to_dict()
    classifier = build_classifier(experiment['classifier'], seed)
    start = perf_counter()
    classifier.fit(x_train, labels[train])
    training_seconds = perf_counter()-start
    start = perf_counter()
    predicted = classifier.predict(x_test)
    scores = continuous_scores(classifier, x_test)
    if scores is None or not np.isfinite(scores).all():
        raise ValueError('Classifier must produce finite continuous inner OOF scores')
    record = {'experiment': experiment['name'], 'fold': inner_number, 'status': 'valid',
              'training_seconds': training_seconds, 'prediction_seconds': perf_counter()-start,
              'reduction_seconds': preparation_seconds, **prediction_metrics(labels[test], predicted, scores)}
    atomic_csv(pd.DataFrame([record]), directory/'metrics.csv')
    atomic_csv(pd.DataFrame({'index': test, 'true_label': labels[test], 'predicted_label': predicted,
                            'score_clear': scores, 'score_kind': score_kind(classifier)}), directory/'predictions.csv.gz')
    finish_job(directory, identity, ['metrics.csv', 'predictions.csv.gz'])
    return record


def search_inner(search, store, labels, splits, directory, config, run_identity):
    all_records = []
    preprocessing_groups = group_experiments(search['experiments'])
    for inner_number, (train, validation) in enumerate(splits, 1):
        train_identity = digest({'train': train.tolist(), 'validation': validation.tolist()})
        relief = ReliefRankingCache(directory/'relieff'/train_identity, n_jobs=config['relieff_n_jobs'])
        for variants in preprocessing_groups:
            first = variants[0]
            identities = {item['name']: {'run': run_identity, 'split': train_identity, 'experiment': item}
                          for item in variants}
            destinations = {item['name']: directory/'jobs'/f"inner_{inner_number:02d}"/item['name'] for item in variants}
            pending = [item for item in variants if not completed_job(destinations[item['name']], identities[item['name']])]
            if not pending:
                all_records.extend(pd.read_csv(destinations[item['name']]/'metrics.csv').iloc[0].to_dict() for item in variants)
                continue
            feature_map = store.feature_map(first['extractors'])
            reason = infeasible_reason(first, feature_map, train)
            if reason:
                for item in variants:
                    record = {'experiment': item['name'], 'fold': inner_number, 'status': 'infeasible', 'reason': reason}
                    dest = destinations[item['name']]
                    atomic_csv(pd.DataFrame([record]), dest/'metrics.csv')
                    finish_job(dest, identities[item['name']], ['metrics.csv'])
                    all_records.append(record)
                continue
            started = perf_counter()
            x_train, x_validation = prepare_fold_features(first, feature_map, labels, train, validation, config['seed'],
                fold=inner_number, relieff_cache=relief, relieff_max_features=300,
                relieff_n_jobs=config['relieff_n_jobs'])
            elapsed = perf_counter()-started
            budget = config['memory_budget_gib']*2**30
            reserve = sum(value.nbytes for value in feature_map.values())+x_train.nbytes+x_validation.nbytes
            per_fit = max(1, x_train.nbytes*4)
            if reserve+per_fit > budget:
                raise ValueError('A preprocessing/classifier job exceeds the configured memory budget')
            workers = min(config['n_jobs'], max(1, int((budget-reserve)//per_fit)))
            with threadpool_limits(config['cpu_threads_per_job']):
                rows = Parallel(n_jobs=workers, backend='threading')(
                    delayed(classify_job)(item, x_train, x_validation, labels, train, validation, config['seed'],
                        inner_number, destinations[item['name']], identities[item['name']], elapsed) for item in variants)
            all_records.extend(rows)
            print(f"inner {inner_number}/{len(splits)}: {first['name']} ({len(all_records)} attempts)", flush=True)
    records = pd.DataFrame(all_records)
    atomic_csv(records, directory/'inner_metrics.csv.gz')
    valid_counts = records[records.status.eq('valid')].groupby('experiment').size()
    names = set(valid_counts[valid_counts.eq(len(splits))].index)
    eligible_search = {**search, 'experiments': [e for e in search['experiments'] if e['name'] in names]}
    selected = select_procedures(records[records.experiment.isin(names)], eligible_search)
    return records, selected


def refit_outer(selected, search, store, paths, labels, train, test, directory, config, inner_count):
    experiment_map = {e['name']: e for e in search['experiments']}
    selections = {**selected['selected'], 'winner': selected['winner']}
    fitted, predictions, metrics, configurations = {}, [], [], []
    for procedure, name in selections.items():
        experiment = experiment_map[name]
        if name not in fitted:
            frames = [pd.read_csv(directory/'jobs'/f'inner_{i:02d}'/name/'predictions.csv.gz') for i in range(1, inner_count+1)]
            oof = pd.concat(frames).sort_values('index')
            if oof['index'].duplicated().any() or set(oof['index']) != set(train):
                raise ValueError('Selected safety scores are not complete inner OOF')
            operating = safety_threshold(oof.true_label, oof.score_clear, config['safety_obstructed_recall'], source='via_inner_oof')
            relief = ReliefRankingCache(directory/'outer_refit_relieff', n_jobs=config['relieff_n_jobs'])
            x_train, x_test = prepare_fold_features(experiment, store.feature_map(experiment['extractors']), labels,
                train, test, config['seed'], fold=0, relieff_cache=relief, relieff_max_features=300,
                relieff_n_jobs=config['relieff_n_jobs'])
            classifier = build_classifier(experiment['classifier'], config['seed'])
            with threadpool_limits(config['cpu_threads_per_job']):
                classifier.fit(x_train, labels[train])
                default = classifier.predict(x_test)
                scores = continuous_scores(classifier, x_test)
            safety = (scores > operating['threshold']).astype(int)
            fitted[name] = (default, scores, safety, operating, score_kind(classifier),
                            prediction_metrics(oof.true_label, oof.predicted_label)['accuracy'])
        default, scores, safety, operating, kind, inner_accuracy = fitted[name]
        predictions.extend({'filename': paths[i].name, 'index': int(i), 'procedure': procedure,
                            'selected_configuration': name, 'true_label': int(labels[i]),
                            'score_clear': float(score), 'score_kind': kind,
                            'predicted_label_default': int(d), 'predicted_label_safety': int(s)}
                           for i, score, d, s in zip(test, scores, default, safety))
        for point, decisions in [('default', default), ('safety', safety)]:
            metrics.append({'procedure': procedure, 'selected_configuration': name, 'operating_point': point,
                            **prediction_metrics(labels[test], decisions, scores)})
        configurations.append({'procedure': procedure, 'selected_configuration': name, 'inner_oof_accuracy': inner_accuracy,
                               'outer_accuracy': prediction_metrics(labels[test], default)['accuracy'],
                               'selection_optimism': inner_accuracy-prediction_metrics(labels[test], default)['accuracy'],
                               **operating})
    return pd.DataFrame(predictions), pd.DataFrame(metrics), pd.DataFrame(configurations)


def validate_predictions(predictions, paths, labels, splits, requested):
    expected = {paths[i].name for fold in requested for i in splits[fold][1]}
    if predictions.duplicated(['filename', 'procedure']).any():
        raise ValueError('More than one outer prediction per image/procedure')
    if set(predictions.procedure) != set(['A', 'B', 'C', 'D', 'winner']):
        raise ValueError('Missing outer procedures')
    truth = {p.name: int(label) for p, label in zip(paths, labels)}
    for _, rows in predictions.groupby('procedure'):
        if set(rows.filename) != expected or any(truth[r.filename] != r.true_label for r in rows.itertuples()):
            raise ValueError('Outer coverage/labels differ from requested folds')


def run(args):
    config = json.loads(args.config.read_text())
    if args.n_jobs is not None:
        config['n_jobs'] = args.n_jobs
    if config['n_jobs'] < 1 or config['memory_budget_gib'] <= 0:
        raise ValueError('Positive worker count and memory budget are required')
    if args.dry_run:
        result = nested_estimate(config, grouped=args.groups is not None)
        print(json.dumps(result, indent=2))
        if args.estimate_output:
            write_json(args.estimate_output, result)
        return
    output = guard_output(args.output_dir or ('results/nested_cv_grouped' if args.groups else 'results/nested_cv'))
    if args.smoke and output.name in ('nested_cv', 'nested_cv_grouped'):
        output = guard_output(output.with_name(output.name+'_smoke'))
    paths, labels = scan_dataset(args.dataset)
    splits = outer_splits(paths, labels, config, args.groups)
    groups = None
    if args.groups:
        groups = pd.read_csv(args.groups).set_index('filename').loc[[p.name for p in paths], 'group_id'].to_numpy()
    requested = [0] if args.smoke else list(range(len(splits))) if not args.outer_folds else [int(i) for i in args.outer_folds.split(',')]
    if len(set(requested)) != len(requested) or not set(requested) <= set(range(len(splits))):
        raise ValueError('Outer-fold indices must be unique and zero based')
    search = smoke_search() if args.smoke else json.loads((ROOT/config['search_config']).read_text())
    identity = {'config': config, 'search_sha256': digest(search), 'dataset_sha256': dataset_content_fingerprint(paths),
                'groups_sha256': sha256(args.groups) if args.groups else None, 'sources': source_identity(),
                'runtime': runtime(), 'device': args.device, 'smoke': args.smoke}
    checked_identity(output, identity)
    write_json(output/'run_metadata.json', {'completed': False, 'identity': identity, 'smoke': args.smoke,
               'requested_outer_indices': requested, 'groups_sha256': identity['groups_sha256'],
               'estimand': 'performance of the complete configuration-selection procedure'})
    store = CanonicalFeatureStore(paths, cache=ROOT/config['canonical_features'],
                                 max_loaded=config['max_loaded_extractors'], device=args.device,
                                 threads=config['cpu_threads_per_job'])
    all_predictions, all_metrics, all_configs = [], [], []
    for index in requested:
        number = index+1
        train, test = splits[index]
        inner = inner_splits(train, labels, number, config, groups, smoke=args.smoke)
        assert_no_leakage(train, test, inner, groups)
        directory = output/'inner_results'/f'outer_{number:02d}'
        outer_identity = {'run': digest(identity), 'train': train.tolist(), 'test': test.tolist(),
                          'inner': [{'train': t.tolist(), 'validation': v.tolist()} for t, v in inner]}
        checked_identity(directory, outer_identity)
        if not completed_job(directory, outer_identity):
            assignments = [{'filename': paths[i].name, 'index': int(i), 'outer_fold': number, 'inner_fold': j,
                            'split': split} for j, (t, v) in enumerate(inner, 1)
                           for split, indices in [('train', t), ('validation', v)] for i in indices]
            atomic_csv(pd.DataFrame(assignments), directory/'inner_assignments.csv')
            records, selected = search_inner(search, store, labels, inner, directory, config, digest(identity))
            write_json(directory/'selection.json', {'selected': selected['selected'], 'winner': selected['winner'],
                       'attempts': len(records), 'infeasible_attempts': int(records.status.eq('infeasible').sum())})
            pred, metric, choices = refit_outer(selected, search, store, paths, labels, train, test, directory, config, len(inner))
            for frame in (pred, metric, choices):
                frame.insert(0, 'fold', number)
            for frame, name in [(pred, 'outer_predictions.csv'), (metric, 'outer_metrics.csv'), (choices, 'selected_configurations.csv')]:
                atomic_csv(frame, directory/name)
            finish_job(directory, outer_identity, ['inner_assignments.csv', 'inner_metrics.csv.gz', 'selection.json',
                       'outer_predictions.csv', 'outer_metrics.csv', 'selected_configurations.csv'])
        all_predictions.append(pd.read_csv(directory/'outer_predictions.csv'))
        all_metrics.append(pd.read_csv(directory/'outer_metrics.csv'))
        all_configs.append(pd.read_csv(directory/'selected_configurations.csv'))
    predictions = pd.concat(all_predictions).sort_values(['fold', 'procedure', 'filename'])
    validate_predictions(predictions, paths, labels, splits, requested)
    metrics, choices = pd.concat(all_metrics), pd.concat(all_configs)
    atomic_csv(predictions, output/'outer_predictions.csv')
    atomic_csv(metrics, output/'outer_metrics.csv')
    atomic_csv(choices, output/'selected_configurations.csv')
    atomic_csv(choices.groupby(['procedure', 'selected_configuration']).size().reset_index(name='outer_folds'),
               output/'selection_frequency.csv')
    atomic_csv(pd.DataFrame([{'filename': paths[i].name, 'fold': j+1, 'true_label': int(labels[i]),
                             'group_id': None if groups is None else groups[i]}
                            for j, (_, indices) in enumerate(splits) for i in indices]), output/'outer_assignments.csv')
    complete = not args.smoke and set(requested) == set(range(config['outer_folds']))
    write_json(output/'feature_provenance.json', store.provenance)
    write_json(output/'run_metadata.json', {'completed': complete, 'requested_run_completed': True,
               'identity': identity, 'smoke': args.smoke, 'requested_outer_indices': requested,
               'groups_sha256': identity['groups_sha256'], 'outer_prediction_rows': len(predictions),
               'external_images_classified': 0, 'outputs': {name: sha256(output/name) for name in
                 ('outer_predictions.csv', 'outer_metrics.csv', 'selected_configurations.csv', 'outer_assignments.csv')}})
    print(f'Completed diagnostic/partial={not complete}: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/nested_cv.json'))
    parser.add_argument('--dataset', type=Path, default=Path('via-dataset'))
    parser.add_argument('--groups', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--outer-folds')
    parser.add_argument('--n-jobs', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--estimate-output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    run(args)


if __name__ == '__main__':
    main()
