"""Selection regression, split isolation, atomic resume and outer scoring contracts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import nested_cv as nested
from src.artifact import scan_dataset, prediction_metrics
from src.selection import select_procedures
from src.study_io import ROOT, sha256, completed_job, finish_job, write_json
from src.study_statistics import safety_threshold


@pytest.fixture
def config():
    return json.loads((ROOT/'configs/nested_cv.json').read_text())


@pytest.fixture
def via():
    return scan_dataset(ROOT/'via-dataset')


def test_article_selection_full_search_regression():
    records = pd.read_csv(ROOT/'results/full_search/per_fold_metrics.csv')
    search = json.loads((ROOT/'configs/full_search.json').read_text())
    reference = json.loads((ROOT/'configs/article_selected_models_10fold.json').read_text())
    original_columns = records.columns.tolist()
    selected = select_procedures(records, search)
    assert selected['selected'] == {e['approach']: e['name'] for e in reference['experiments']}
    assert selected['winner'] == 'b_efficientnetb3_pca_300_linear_svm'
    archived = pd.read_csv(ROOT/'results/full_search/manuscript_tables/selected_approaches.csv')
    assert selected['winner'] == archived.iloc[0].experiment
    assert records.columns.tolist() == original_columns


def test_original_outer_assignments_are_consumed_exactly(config, via):
    paths, labels = via
    splits = nested.outer_splits(paths, labels, config)
    recorded = pd.read_csv(ROOT/config['outer_assignments']).sort_values('sample_index')
    for number, (train, test) in enumerate(splits, 1):
        np.testing.assert_array_equal(test, np.flatnonzero(recorded.test_fold.to_numpy() == number))
        assert not set(train) & set(test)


def test_grouped_outer_matches_frozen_diagnostics(config, via):
    paths, labels = via
    groups_path = ROOT/'data/groups.csv'
    assert sha256(groups_path) == config['frozen_groups_sha256']
    splits = nested.outer_splits(paths, labels, config, groups_path)
    recorded = pd.read_csv(ROOT/config['group_diagnostics'])
    groups = pd.read_csv(groups_path).set_index('filename').loc[[p.name for p in paths], 'group_id'].to_numpy()
    assert len(set(groups)) == 338
    for number, (train, test) in enumerate(splits, 1):
        row = recorded[recorded.fold.eq(number)].iloc[0]
        assert (len(test), labels[test].sum(), np.sum(labels[test] == 0)) == (row.test_image_count, row.clear, row.nonclear)
        assert not set(groups[train]) & set(groups[test])


@pytest.mark.parametrize('grouped', [False, True])
def test_all_inner_folds_are_isolated_from_outer_test(config, via, grouped):
    paths, labels = via
    group_path = ROOT/'data/groups.csv' if grouped else None
    groups = None if group_path is None else pd.read_csv(group_path).set_index('filename').loc[[p.name for p in paths], 'group_id'].to_numpy()
    for number, (train, test) in enumerate(nested.outer_splits(paths, labels, config, group_path), 1):
        splits = nested.inner_splits(train, labels, number, config, groups)
        nested.assert_no_leakage(train, test, splits, groups)
        assert len(splits) == 5
        again = nested.inner_splits(train, labels, number, config, groups)
        for (a, b), (c, d) in zip(splits, again):
            np.testing.assert_array_equal(a, c); np.testing.assert_array_equal(b, d)


def test_outer_test_contamination_rejected():
    with pytest.raises(ValueError):
        nested.assert_no_leakage(np.array([0, 1, 2, 3]), np.array([4]),
            [(np.array([0, 4]), np.array([2, 3])), (np.array([2, 3]), np.array([0, 1]))])


def test_pca_infeasibility_does_not_clamp_components():
    experiment = {'approach': 'B', 'extractors': ['a'], 'reduction': 'pca', 'components': 300}
    assert 'components=300' in nested.infeasible_reason(experiment, {'a': np.zeros((342, 1000))}, np.arange(245))
    assert experiment['components'] == 300


def test_spectral_umap_high_dimensions_are_explicitly_infeasible():
    experiment = {'approach': 'B', 'extractors': ['a'], 'reduction': 'umap', 'components': 250}
    assert 'spectral' in nested.infeasible_reason(experiment, {'a': np.zeros((342, 1000))}, np.arange(245))


def test_safety_uses_via_inner_oof_and_handles_tied_scores():
    truth, score = np.array([0, 0, 1, 1]), np.array([.2, .6, .6, .9])
    value = safety_threshold(truth, score, target=1, source='via_inner_oof')
    assert value['threshold'] == .6
    assert value['obstructed_recall'] == 1
    assert value['specificity'] == .5
    with pytest.raises(ValueError, match='VIA'):
        safety_threshold(truth, score, source='external')


def test_atomic_resume_rejects_changed_metrics(tmp_path):
    identity = {'train': [0, 1], 'test': [2]}
    write_json(tmp_path/'metrics.json', {'accuracy': 1})
    finish_job(tmp_path, identity, ['metrics.json'])
    assert completed_job(tmp_path, identity)
    with pytest.raises(ValueError, match='identity'):
        completed_job(tmp_path, {'train': [0, 2], 'test': [1]})
    write_json(tmp_path/'metrics.json', {'accuracy': 0})
    with pytest.raises(ValueError, match='artifact'):
        completed_job(tmp_path, identity)


@pytest.mark.parametrize('grouped', [False, True])
def test_cpu_smoke_pipeline_contract_and_recomputed_metrics(tmp_path, monkeypatch, config, via, grouped):
    """Fast fixture-based control-flow test; CLI smokes separately use real cached CNN features."""
    paths, labels = via
    rng = np.random.default_rng(1980)
    matrices = {name: rng.normal(size=(len(paths), 8)).astype(np.float32) for name in ('MobileNet', 'ResNet50')}
    class FixtureStore:
        provenance = {'fixture': 'synthetic features; not a scientific result'}
        def __init__(self, *args, **kwargs): pass
        def feature_map(self, names): return {name: matrices[name] for name in names}
    monkeypatch.setattr(nested, 'CanonicalFeatureStore', FixtureStore)
    monkeypatch.setattr(nested, 'guard_output', lambda path: Path(path))
    args = argparse.Namespace(config=ROOT/'configs/nested_cv.json', n_jobs=1, dry_run=False, estimate_output=None,
        output_dir=tmp_path/'smoke', groups=ROOT/'data/groups.csv' if grouped else None,
        dataset=ROOT/'via-dataset', smoke=True, outer_folds=None, device='cpu')
    nested.run(args)
    predictions = pd.read_csv(args.output_dir/'outer_predictions.csv')
    metrics = pd.read_csv(args.output_dir/'outer_metrics.csv')
    assert len(predictions) == len(nested.outer_splits(paths, labels, config, args.groups)[0][1])*5
    assert not predictions.duplicated(['filename', 'procedure']).any()
    for row in metrics.itertuples():
        panel = predictions[predictions.procedure.eq(row.procedure)]
        recomputed = prediction_metrics(panel.true_label, panel[f'predicted_label_{row.operating_point}'], panel.score_clear)
        for key, value in recomputed.items():
            assert getattr(row, key) == pytest.approx(value, nan_ok=True)
    metadata = json.loads((args.output_dir/'run_metadata.json').read_text())
    assert metadata['smoke'] and not metadata['completed']
    assert metadata['external_images_classified'] == 0
    assert metadata['requested_run_completed']
    nested.run(args)  # Reuses checksummed jobs, without new selection fits.
    assert sha256(args.output_dir/'outer_predictions.csv') == metadata['outputs']['outer_predictions.csv']
