"""Original-fold OOF provenance, fold-wise optimism and mean-seed scene intervals."""
import json

import numpy as np
import pandas as pd
import pytest

from src.external_oof import load_article_oof, validate_oof_coverage
from src.study_io import ROOT
from src.study_statistics import BootstrapSampler, resample_indices, wilson, bootstrap_intervals
from scripts.compute_tibshirani_bias import tibshirani_tables
from src.study_io import atomic_csv, write_json


@pytest.mark.parametrize('groups', [None, np.array(['mixed_a', 'mixed_a', 'obs', 'obs', 'clear', 'clear', 'mixed_b', 'mixed_b'])])
def test_precomputed_sampler_reproduces_entire_original_stratified_draw(groups):
    labels = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    sampler = BootstrapSampler(labels, groups)
    before, after = np.random.default_rng(1980), np.random.default_rng(1980)
    for _ in range(100):
        weights = np.bincount(resample_indices(labels, before, groups), minlength=len(labels))
        np.testing.assert_array_equal(sampler.weights(after), weights)


def test_wilson_zero_discordance_has_exact_boundary():
    assert wilson(0, 100) == pytest.approx((0., .03699349820698568))
    assert wilson(0, 100)[0] == 0.
    assert wilson(100, 100)[1] == 1.


def test_mean_seed_safety_intervals_use_scene_trials():
    truth = np.array([0, 1, 0, 1, 0, 1])
    decisions = np.array([[0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 0, 1], [1, 1, 0, 1, 0, 1]])
    intervals = bootstrap_intervals(truth, decisions, groups=np.repeat(np.arange(3), 2), iterations=80,
                                   endpoints=['obstructed_recall', 'obstructed_false_negative_rate'])
    assert intervals['obstructed_recall'][0] == pytest.approx(1-intervals['obstructed_false_negative_rate'][1])
    assert intervals['obstructed_recall'][1] == pytest.approx(1-intervals['obstructed_false_negative_rate'][0])


def test_original_oof_rejects_missing_images_and_changed_folds():
    folds = pd.DataFrame({'filename': ['a', 'b'], 'test_fold': [1, 2]})
    panel = pd.DataFrame({'filename': ['a', 'b'], 'fold': [1, 2], 'true_label': [0, 1], 'score_clear': [-1, 1]})
    validate_oof_coverage(panel, folds, {'a': 0, 'b': 1})
    with pytest.raises(ValueError, match='exactly once'): validate_oof_coverage(panel.iloc[:1], folds, {'a': 0, 'b': 1})
    panel.loc[1, 'fold'] = 1
    with pytest.raises(ValueError, match='labels/folds'): validate_oof_coverage(panel, folds, {'a': 0, 'b': 1})


def test_archived_article_oof_reproduces_all_forty_canonical_jobs_without_fit():
    config = json.loads((ROOT/'configs/external_test.json').read_text())
    reference = json.loads((ROOT/config['fixed_reference_config']).read_text())
    selected = {e['approach']: e['name'] for e in reference['experiments']}
    experiments = {e['name']: e for e in json.loads((ROOT/config['fixed_search_config']).read_text())['experiments']}
    frame, provenance = load_article_oof(config, {'selected': selected, 'winner': selected['B']}, experiments)
    assert len(frame) == 342*5 and len(provenance['original_refit_jobs']) == 40
    assert provenance['canonical_metric_checks'] >= 640 and provenance['new_fits'] == 0
    assert frame.groupby('procedure').filename.nunique().eq(342).all()
    assert set(frame.score_kind) == {'margin'} and set(frame.source) == {'via_oof'}


def bias_fixture():
    experiments, selected, rows = {}, {}, []
    for approach in 'ABCD':
        selected[approach] = f'{approach}_pca'
        for reduction, values in [('pca', [.8, .7]), ('relieff', [.9, .6])]:
            name = f'{approach}_{reduction}'
            experiments[name] = {'name': name, 'approach': approach, 'reduction': reduction}
            for fold, accuracy in enumerate(values, 1):
                rows.append({'experiment': name, 'fold': fold, 'accuracy': accuracy,
                             'obstructed_recall': .7 if fold == 1 else .8})
    return pd.DataFrame(rows), selected, selected['B'], experiments


def test_fold_optimism_formula_and_relief_comparison_use_entire_search():
    records, selected, winner, experiments = bias_fixture()
    bias, relief, diagnostics = tibshirani_tables(records, selected, winner, experiments)
    assert len(bias) == 5 and len(diagnostics) == 20
    assert bias.accuracy_original.to_numpy() == pytest.approx(np.repeat(.75, 5))
    assert bias.accuracy_bias.to_numpy() == pytest.approx(np.repeat(.05, 5))
    assert bias.accuracy_corrected.to_numpy() == pytest.approx(np.repeat(.70, 5))
    assert set(relief.approach) == set('ABCD') and relief.pca_minus_relieff.eq(0).all()
    with pytest.raises(ValueError, match='entire'): tibshirani_tables(records[records.experiment.ne('A_relieff')], selected, winner, experiments)
    with pytest.raises(ValueError, match='complete'): tibshirani_tables(records.iloc[1:], selected, winner, experiments)


def test_analysis_original_fold_domain_branch_with_numeric_external_fixtures(tmp_path, monkeypatch):
    """Exercise the production analysis branch without opening/classifying any external image."""
    from src.external_test import evaluation_smoke
    from scripts import analyze_external_test as analysis
    from scipy.special import expit
    evaluation_smoke(tmp_path)
    # Adjacent floats near one must remain ordered for AUC after CSV round-trip.
    from src.artifact import prediction_metrics
    predictions = pd.read_csv(tmp_path/'predictions.csv', float_precision='round_trip')
    model = selected_model = predictions.model.iloc[0]
    panel_mask = predictions.model.eq(model)
    for truth, probability in [(0, np.nextafter(1., 0.)), (1, 1.)]:
        position = predictions[panel_mask & predictions.true_label.eq(truth)].index[0]
        predictions.loc[position, 'score_clear'] = probability
        predictions.loc[position, 'predicted_label_default'] = 1
        predictions.loc[position, 'predicted_label_safety'] = 1
    metrics = pd.read_csv(tmp_path/'metrics.csv', float_precision='round_trip')
    for index, row in metrics.iterrows():
        if row.model != selected_model: continue
        panel = predictions[predictions.model.eq(row.model) & predictions.seed.eq(row.seed)]
        if row.panel == 'principal': panel = panel[~panel.scene_overlap]
        recalculated = prediction_metrics(panel.true_label, panel[f'predicted_label_{row.operating_point}'], panel.score_clear)
        for key, value in recalculated.items(): metrics.loc[index, key] = value
    atomic_csv(predictions, tmp_path/'predictions.csv'); atomic_csv(metrics, tmp_path/'metrics.csv')
    metadata = json.loads((tmp_path/'first_evaluation.json').read_text())
    from src.study_io import sha256
    metadata['predictions_sha256'], metadata['metrics_sha256'] = sha256(tmp_path/'predictions.csv'), sha256(tmp_path/'metrics.csv')
    metadata.pop('synthetic_only')
    metadata['test_fixture_only'] = True
    write_json(tmp_path/'first_evaluation.json', metadata)
    config = json.loads((ROOT/'configs/external_test.json').read_text())
    reference = json.loads((ROOT/config['fixed_reference_config']).read_text())
    selected = {e['approach']: e['name'] for e in reference['experiments']}
    experiments = {e['name']: e for e in json.loads((ROOT/config['fixed_search_config']).read_text())['experiments']}
    oof, _ = load_article_oof(config, {'selected': selected, 'winner': selected['B']}, experiments)
    atomic_csv(oof, tmp_path/'fixed_oof_predictions.csv')
    calibrated = oof.copy(); calibrated['score_clear'] = expit(calibrated.score_clear)
    atomic_csv(calibrated, tmp_path/'calibrated_via_oof.csv')
    monkeypatch.setattr(analysis, 'guard_output', lambda path: path)
    analysis.analyze(tmp_path, repetitions=4, smoke=False)
    tables = tmp_path/'manuscript_tables'
    assert len(pd.read_csv(tables/'domain_comparison.csv')) == 9
    assert set(pd.read_csv(tables/'domain_calibration.csv').domain) == {'via_oof', 'external'}
    confidence = pd.read_csv(tables/'cluster_bootstrap_wilson.csv')
    assert {'accuracy', 'balanced_accuracy', 'mcc', 'roc_auc', 'obstructed_recall', 'obstructed_false_negative_rate'} <= set(confidence.endpoint)
    ensemble = pd.read_csv(tables/'ensemble_sensitivity.csv')
    assert 'obstructed_false_negative_rate_ci_low' in ensemble.columns
