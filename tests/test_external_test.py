"""Exact/scene disjointness, preregistration gates and external analysis contracts."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.clean_external_dataset import exact_clean, inventory, overlap_rows, audit_sample
from src import external_test as external
from src import study_io
from src.artifact import prediction_metrics
from src.study_io import ROOT, sha256, write_json, atomic_csv
from src.study_statistics import (resample_indices, paired_bootstrap, safety_threshold, substitution_status,
                                  weighted_metrics, bootstrap_intervals)


def test_real_external_hash_counts_and_all_known_conflicts():
    # Public distribution contains retained privacy copies; reconstruct the
    # historical exact-cleaning audit from its frozen raw manifest.
    raw = pd.read_csv(ROOT/'external-data/external_manifest_raw.csv')
    via = inventory(ROOT/'via-dataset')
    novel, exclusions, counts = exact_clean(raw, via, expected={'total': 1576, 'clear': 844, 'nonclear': 732})
    assert (len(raw), raw.original_label.sum()) == (1929, 1026)
    assert not set(novel.sha256) & set(via.sha256)
    assert novel.sha256.is_unique
    assert len(exclusions) == 353
    conflicts = exclusions[exclusions.reason.eq('via_exact_label_conflict')]
    assert set(conflicts.filename) == {'clear.0930.jpg', 'clear.0932.jpg', 'clear.0933.jpg', 'clear.0934.jpg'}
    assert set(exclusions[exclusions.reason.eq('external_exact_duplicate')].filename) == {'clear.0380.jpg', 'nonclear.0199.jpg', 'nonclear.0873.jpg'}
    from scripts.verify_external_release import verify_release
    assert verify_release(ROOT)['images'] == len(novel)



def test_exact_count_mismatch_stops_before_scene_work():
    raw = pd.DataFrame([{'filename': 'clear.1.jpg', 'sha256': 'a', 'original_label': 1}])
    via = pd.DataFrame([{'filename': 'nonclear.1.jpg', 'sha256': 'b', 'original_label': 0}])
    with pytest.raises(ValueError, match='STOP'):
        exact_clean(raw, via, expected={'total': 1576, 'clear': 844, 'nonclear': 732})


def test_only_direct_cross_dataset_links_define_scene_exclusions():
    paths = [Path('clear.000.jpg'), Path('nonclear.0200.jpg'), Path('clear.0300.jpg')]
    edges = np.array([[False, True, False], [True, False, True], [False, True, False]])
    result = overlap_rows(paths, edges, {'clear.000.jpg'})
    assert result.filename.tolist() == ['nonclear.0200.jpg']
    assert 'clear.0300.jpg' not in set(result.filename)


def test_label_audit_is_deterministic_stratified_and_unique():
    novel = pd.DataFrame({'filename': [f'img{i:04d}.jpg' for i in range(1576)],
                          'original_label': np.r_[np.ones(844, dtype=int), np.zeros(732, dtype=int)]})
    first, second = audit_sample(novel), audit_sample(novel)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 100 and first.filename.is_unique
    assert first.original_label.sum() == 54


def test_preregistration_requires_committed_exact_bytes(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '--quiet', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'config', 'user.name', 'Fixture'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'config', 'user.email', 'fixture@example.invalid'], check=True)
    monkeypatch.setattr(study_io, 'ROOT', tmp_path)
    path = tmp_path/'protocol.md'; path.write_text('Predeclared protocol\n')
    with pytest.raises(ValueError, match='committed'):
        study_io.committed_bytes(path)
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'protocol.md'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'commit', '--quiet', '-m', 'Preregister fixture'], check=True)
    assert study_io.committed_bytes(path)['sha256'] == sha256(path)
    path.write_text('Changed after commit\n')
    with pytest.raises(ValueError, match='local edits'):
        study_io.committed_bytes(path)


def test_evaluation_refuses_a_diagnostic_model_freeze(tmp_path, monkeypatch):
    config = tmp_path/'config.json'; write_json(config, {})
    monkeypatch.setattr(external, 'registered_protocol', lambda _: {})
    write_json(tmp_path/'final_models.json', {'frozen': False, 'smoke': True})
    with pytest.raises(ValueError, match='final models'):
        external.evaluation_gate(config, tmp_path)


def test_evaluation_refuses_changed_config_hash(tmp_path, monkeypatch):
    config = tmp_path/'config.json'; write_json(config, {})
    registration = {'config': {'sha256': 'new'}, 'protocol': {'sha256': 'p'}}
    monkeypatch.setattr(external, 'registered_protocol', lambda _: registration)
    write_json(tmp_path/'final_models.json', {'frozen': True, 'smoke': False,
        'preregistration': {'config': {'sha256': 'old'}, 'protocol': {'sha256': 'p'}}})
    with pytest.raises(ValueError, match='preregistered'):
        external.evaluation_gate(config, tmp_path)


def test_unapproved_label_audit_cannot_be_evaluated(tmp_path, monkeypatch):
    monkeypatch.setattr(external, 'ROOT', tmp_path)
    write_json(tmp_path/'cleaning_report.json', {'status': 'paused_for_author_review', 'approved_by': None,
                                               'audit_completed': False})
    with pytest.raises(ValueError, match='Author-approved'):
        external.validate_cleaning({'cleaning_directory': '.', 'label_discordance_stop_above': .05})


def test_calibration_and_thresholds_reject_external_sources():
    oof = pd.DataFrame({'filename': ['a', 'b', 'c', 'd'], 'true_label': [0, 0, 1, 1],
                        'score_clear': [-2, -1, 1, 2], 'score_kind': 'margin', 'source': 'external'})
    with pytest.raises(ValueError, match='VIA'):
        external.fit_probability_adapter(oof)
    with pytest.raises(ValueError, match='VIA'):
        safety_threshold(oof.true_label, oof.score_clear, source='external')
    oof['source'] = 'via_oof'
    adapter = external.fit_probability_adapter(oof)
    probabilities = external.adapter_probability(adapter, oof.score_clear, 'margin')
    assert (np.diff(probabilities) > 0).all() and ((probabilities > 0) & (probabilities < 1)).all()


def test_external_repeat_is_byte_identical_or_rejected(tmp_path):
    frame = pd.DataFrame({'model': ['m', 'm'], 'seed': [-1, -1], 'filename': ['b', 'a'], 'score_clear': [.2, .8]})
    path = tmp_path/'predictions.csv'
    external.write_once_predictions(frame, path)
    before = path.read_bytes()
    external.write_once_predictions(frame.iloc[::-1], path)
    assert path.read_bytes() == before
    frame.loc[0, 'score_clear'] = .21
    with pytest.raises(ValueError, match='byte-for-byte'):
        external.write_once_predictions(frame, path)
    assert path.read_bytes() == before


def test_prediction_coverage_requires_one_per_model_seed():
    manifest = pd.DataFrame({'filename': ['a', 'b'], 'final_label': [0, 1]})
    panel = pd.DataFrame({'filename': ['a', 'b'], 'true_label': [0, 1], 'model': 'm', 'seed': 1980})
    external.verify_prediction_coverage(panel, manifest, [{'model': 'm', 'seed': 1980}])
    with pytest.raises(ValueError, match='Duplicate'):
        external.verify_prediction_coverage(pd.concat([panel, panel.iloc[:1]]), manifest, [{'model': 'm', 'seed': 1980}])
    with pytest.raises(ValueError, match='coverage'):
        external.verify_prediction_coverage(panel.iloc[:1], manifest, [{'model': 'm', 'seed': 1980}])


def test_cluster_bootstrap_preserves_entire_mixed_class_groups():
    labels = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    groups = np.array(['mixed_a', 'mixed_a', 'obs', 'obs', 'clear', 'clear', 'mixed_b', 'mixed_b'])
    rng = np.random.default_rng(1980)
    for _ in range(100):
        indices = resample_indices(labels, rng, groups)
        multiplicities = np.bincount(indices, minlength=len(labels))
        for group in np.unique(groups):
            assert len(set(multiplicities[groups == group])) == 1
        assert set(labels[indices]) == {0, 1}


def test_paired_cluster_comparison_averages_individual_seed_metrics():
    truth = np.array([0, 0, 1, 1])
    first = np.array([0, 0, 1, 1])
    seeds = np.array([[0, 1, 1, 1], [0, 0, 0, 1], [1, 0, 1, 1]])
    intervals = paired_bootstrap(truth, first, seeds, groups=np.arange(4), iterations=40)
    expected = prediction_metrics(truth, first)['accuracy'] - np.mean([prediction_metrics(truth, row)['accuracy'] for row in seeds])
    assert intervals['accuracy']['difference'] == pytest.approx(expected)


def test_fnr_substitution_orientation():
    assert substitution_status(-.01, .02) == 'supported'
    assert substitution_status(.04, .08, lower_is_better=True) == 'not_supported'
    assert substitution_status(-.06, .02) == 'inconclusive'


def test_learning_split_and_nested_sizes_preserve_scenes():
    manifest = pd.DataFrame({'filename': [f'image{i:03d}' for i in range(40)],
                            'final_label': np.arange(40) % 2, 'group_id': np.repeat(np.arange(20), 2), 'excluded': False})
    split = external.learning_split(manifest)
    test, pool = split[split.split.eq('test_B')], split[split.split.eq('pool')]
    assert not set(test.group_id) & set(pool.group_id)
    sizes = external.nested_group_subsets(pool)
    assert set(sizes[0]) <= set(sizes[.25]) <= set(sizes[.5]) <= set(sizes[1])
    assert set(sizes[1]) == set(pool.filename)
    assert not set(sizes[1]) & set(test.filename)
    for names in sizes.values():
        for _, rows in pool.groupby('group_id'):
            assert not set(rows.filename) & set(names) or set(rows.filename) <= set(names)


def test_final_validation_is_identical_across_seeds_and_disjoint():
    labels = np.arange(100) % 2
    groups = np.repeat(np.arange(50), 2)
    train, validation = external.grouped_validation(labels, groups)
    for seed in (1980, 1981, 1982):
        again = external.grouped_validation(labels, groups, seed=1980)
        np.testing.assert_array_equal(train, again[0]); np.testing.assert_array_equal(validation, again[1])
    assert not set(groups[train]) & set(groups[validation])


def test_label_discordance_over_five_percent_cannot_approve(tmp_path):
    manifest = pd.DataFrame({'filename': [f'x{i}' for i in range(100)], 'original_label': 0, 'final_label': 0})
    atomic_csv(manifest, tmp_path/'external_manifest_clean.csv')
    atomic_csv(manifest, tmp_path/'label_audit_sample.csv')
    corrections = pd.DataFrame({'filename': manifest.filename[:6], 'original_label': 0, 'corrected_label': 1,
                                 'reason': 'author review', 'reviewer': 'Author', 'date': '2026-10-06'})
    atomic_csv(corrections, tmp_path/'label_corrections.csv')
    atomic_csv(pd.DataFrame(columns=['filename', 'via_filename']), tmp_path/'external_excluded_scene_overlap.csv')
    write_json(tmp_path/'cleaning_report.json', {'manifest_sha256': sha256(tmp_path/'external_manifest_clean.csv'), 'audit_reject_above': .05})
    with pytest.raises(ValueError, match='discordance'):
        external.apply_label_review(tmp_path, reviewer='Author', date='2026-10-06', audit_reviewed=True)
    report = json.loads((tmp_path/'cleaning_report.json').read_text())
    assert report['status'] == 'stopped_label_discordance'
    assert report['approved_by'] is None and report['discordance_rate'] == .06


def test_weighted_bootstrap_auc_matches_explicit_repeated_images():
    truth = np.array([0, 1, 0, 1, 1, 0])
    predicted = np.array([0, 1, 1, 1, 0, 0])
    scores = np.array([.2, .5, .5, .7, .5, .1])  # Cross-class ties must get half credit.
    multiplicities = np.array([2, 1, 0, 3, 2, 1])
    indices = np.repeat(np.arange(len(truth)), multiplicities)
    weighted = weighted_metrics(truth, predicted, scores, multiplicities)
    explicit = prediction_metrics(truth[indices], predicted[indices], scores[indices])
    for key in weighted:
        assert weighted[key] == pytest.approx(explicit[key])
    intervals = bootstrap_intervals(truth, predicted, scores, groups=np.arange(6), iterations=20)
    assert set(intervals) == {'accuracy', 'balanced_accuracy', 'mcc', 'roc_auc'}


def test_metrics_are_recomputed_from_the_synthetic_evaluation_smoke(tmp_path, monkeypatch):
    from scripts.analyze_external_test import verify_external_metrics
    external.evaluation_smoke(tmp_path)
    predictions = pd.read_csv(tmp_path/'predictions.csv')
    metrics = pd.read_csv(tmp_path/'metrics.csv')
    verify_external_metrics(predictions, metrics)
    metrics.loc[0, 'accuracy'] += .01
    with pytest.raises(ValueError, match='metric differs'):
        verify_external_metrics(predictions, metrics)
    assert json.loads((tmp_path/'first_evaluation.json').read_text())['external_classification_predictions'] == 0


def test_scene_review_rejects_links_without_changing_thresholds(tmp_path):
    manifest = pd.DataFrame({'filename': ['a', 'b'], 'original_label': [0, 1], 'final_label': [0, 1],
                             'excluded': True, 'exclusion_reason': 'confirmed_scene_overlap'})
    atomic_csv(manifest, tmp_path/'external_manifest_clean.csv')
    atomic_csv(manifest, tmp_path/'label_audit_sample.csv')
    atomic_csv(pd.DataFrame(columns=['filename', 'original_label', 'corrected_label', 'reason', 'reviewer', 'date']),
               tmp_path/'label_corrections.csv')
    atomic_csv(pd.DataFrame({'filename': ['a', 'a', 'b'], 'via_filename': ['via1', 'via2', 'via1']}),
               tmp_path/'external_excluded_scene_overlap.csv')
    atomic_csv(pd.DataFrame({'filename': ['a', 'b'], 'via_filename': ['via1', 'via1'],
                             'reason': 'different pavement', 'reviewer': 'Author', 'date': '2026-10-06'}),
               tmp_path/'scene_overlap_rejections.csv')
    write_json(tmp_path/'cleaning_report.json', {'manifest_sha256': sha256(tmp_path/'external_manifest_clean.csv'), 'audit_reject_above': .05})
    external.apply_label_review(tmp_path, reviewer='Author', date='2026-10-06', audit_reviewed=True)
    result = pd.read_csv(tmp_path/'external_manifest_clean.csv').set_index('filename')
    assert result.loc['a', 'excluded']  # Its second accepted VIA link still excludes it.
    assert not result.loc['b', 'excluded']
    report = json.loads((tmp_path/'cleaning_report.json').read_text())
    assert report['approved_by'] == 'Author' and report['accepted_scene_links'] == [['a', 'via2']]


def test_prediction_jobs_resume_without_reclassifying_fixture_images(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(external, 'ROOT', tmp_path)
    manifest = pd.DataFrame({'filename': ['a.jpg', 'b.jpg'], 'final_label': [0, 1],
                             'group_id': ['g1', 'g2'], 'excluded': False, 'exclusion_reason': np.nan})
    entry = {'model': 'ft_Fixture', 'seed': 1980, 'family': 'fine_tuning',
             'architecture': 'Fixture', 'model_path': 'fixture.keras', 'procedure': 'ft_Fixture'}
    freeze = {'models': [entry]}
    write_json(tmp_path/'final_models.json', freeze)
    write_json(tmp_path/'thresholds.json', {'thresholds': {'ft_Fixture': {'threshold': .7}}})
    calls = []
    class Model:
        def predict(self, batch, verbose=0):
            calls.append(len(batch)); return np.array([[.7, .3], [.1, .9]])
    tf = SimpleNamespace(keras=SimpleNamespace(models=SimpleNamespace(load_model=lambda *a, **k: Model()),
                                               backend=SimpleNamespace(clear_session=lambda: None)))
    monkeypatch.setattr(external, 'preprocessing', lambda _: (None, (1, 1)))
    monkeypatch.setattr(external, 'image_batch', lambda paths, *args: np.zeros((len(paths), 1, 1, 3)))
    first = external.predict_frozen({'dataset': 'fixture'}, freeze, manifest, tmp_path, tf)
    second = external.predict_frozen({'dataset': 'fixture'}, freeze, manifest, tmp_path, tf)
    assert calls == [2]
    pd.testing.assert_frame_equal(first, second)
