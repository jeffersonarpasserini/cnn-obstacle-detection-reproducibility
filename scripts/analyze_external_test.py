#!/usr/bin/env python3
"""External whole-scene intervals, mean-seed comparisons and domain/calibration tables."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from src.artifact import prediction_metrics
from src.external_test import native_calibration_input
from src.study_io import sha256, write_json, guard_output
from src.study_statistics import (bootstrap_intervals, paired_bootstrap, substitution_status, wilson,
                                  resample_indices, weighted_metrics, BootstrapSampler)
from analyze_nested_cv import table
from analyze_fine_tuning import mcnemar_exact
from build_manuscript_tables import holm_adjust

ENDPOINTS = ['accuracy', 'balanced_accuracy', 'mcc', 'roc_auc', 'obstructed_recall', 'obstructed_false_negative_rate']


def verify_external_metrics(predictions, metrics):
    for row in metrics.to_dict('records'):
        panel = predictions[predictions.model.eq(row['model']) & predictions.seed.eq(row['seed'])]
        if row['panel'] == 'principal': panel = panel[~panel.scene_overlap]
        recalculated = prediction_metrics(panel.true_label, panel[f"predicted_label_{row['operating_point']}"], panel.score_clear)
        for key, value in recalculated.items():
            if not np.isclose(value, row[key], equal_nan=True, atol=1e-12):
                raise ValueError(f'Recomputed external metric differs: {key}')


def reliability(truth, probabilities, bins=10):
    truth, probabilities = np.asarray(truth), np.asarray(probabilities)
    assigned = np.minimum((probabilities*bins).astype(int), bins-1)
    return pd.DataFrame([{'bin': j, 'images': int(np.sum(assigned == j)),
                          'mean_probability': float(probabilities[assigned == j].mean()) if np.any(assigned == j) else np.nan,
                          'observed_clear_fraction': float(truth[assigned == j].mean()) if np.any(assigned == j) else np.nan}
                         for j in range(bins)])


def domain_difference(via_truth, via_decisions, external_truth, external_decisions, external_groups,
                      *, repetitions=10000, seed=1980):
    """Separate draws for different images/domains; positive means external minus VIA."""
    rng = np.random.default_rng(seed); samples = []
    keys = ['accuracy', 'balanced_accuracy', 'mcc', 'obstructed_recall']
    via_sampler, external_sampler = BootstrapSampler(via_truth), BootstrapSampler(external_truth, external_groups)
    for _ in range(repetitions):
        a = weighted_metrics(via_truth, via_decisions, None, via_sampler.weights(rng))
        b = weighted_metrics(external_truth, external_decisions, None, external_sampler.weights(rng))
        samples.append([b[key]-a[key] for key in keys])
    a = weighted_metrics(via_truth, via_decisions, None, np.ones(len(via_truth)))
    b = weighted_metrics(external_truth, external_decisions, None, np.ones(len(external_truth)))
    return [{'endpoint': key, 'difference': b[key]-a[key],
             'ci_low': float(np.quantile(np.asarray(samples)[:, j], .025)),
             'ci_high': float(np.quantile(np.asarray(samples)[:, j], .975))} for j, key in enumerate(keys)]


def analyze(results, *, output=None, repetitions=10000, seed=1980, smoke=False):
    results = Path(results)
    metadata = json.loads((results/'first_evaluation.json').read_text())
    if metadata.get('synthetic_only') and not smoke:
        raise ValueError('Synthetic evaluation is a smoke, not a scientific external result')
    if metadata['status'] != 'completed': raise ValueError('External evaluation is incomplete')
    if sha256(results/'predictions.csv') != metadata['predictions_sha256'] or sha256(results/'metrics.csv') != metadata['metrics_sha256']:
        raise ValueError('Frozen external predictions/metrics changed')
    predictions = pd.read_csv(results/'predictions.csv', float_precision='round_trip')
    metrics = pd.read_csv(results/'metrics.csv', float_precision='round_trip')
    verify_external_metrics(predictions, metrics)
    output = guard_output(output or results/'manuscript_tables'); output.mkdir(parents=True, exist_ok=True)
    summaries, confidence, calibration, paired, mcnemar, ensemble_rows = [], [], [], [], [], []
    for sensitivity, population in [('principal', predictions[~predictions.scene_overlap]), ('include_scene_overlap', predictions)]:
        for model, panel in population.groupby('model'):
            filenames = sorted(panel.filename.unique())
            truth = panel.groupby('filename').true_label.first().loc[filenames].to_numpy()
            groups = panel.groupby('filename').group_id.first().loc[filenames].to_numpy()
            scores = panel.pivot(index='filename', columns='seed', values='score_clear').loc[filenames].to_numpy().T
            for point in ('default', 'safety'):
                decisions = panel.pivot(index='filename', columns='seed', values=f'predicted_label_{point}').loc[filenames].to_numpy().T
                individual = [prediction_metrics(truth, row, score) for row, score in zip(decisions, scores)]
                averages = {key: float(np.mean([v[key] for v in individual])) for key in individual[0]}
                summaries.append({'model': model, 'panel': sensitivity, 'operating_point': point,
                                  'analysis': 'individual-seed metric mean' if len(decisions)>1 else 'single fixed model', **averages})
                intervals = bootstrap_intervals(truth, decisions, scores, groups=groups, iterations=repetitions,
                                                seed=seed, endpoints=ENDPOINTS)
                for endpoint, interval in intervals.items():
                    confidence.append({'model': model, 'panel': sensitivity, 'operating_point': point,
                                       'method': 'whole-scene bootstrap', 'analysis': 'individual-seed metric mean' if len(decisions)>1 else 'single fixed model',
                                       'endpoint': endpoint, 'estimate': averages[endpoint], 'ci_low': interval[0], 'ci_high': interval[1]})
                # Wilson per individual seed; a mean of three correlated seed decisions is not a binomial trial.
                for seed_number, value in zip(sorted(panel.seed.unique()), individual):
                    n = int(np.sum(truth == 0))
                    for endpoint, successes in [('obstructed_recall', value['tn_obstructed']),
                                                ('obstructed_false_negative_rate', value['fp_obstructed_as_clear'])]:
                        low, high = wilson(successes, n)
                        confidence.append({'model': model, 'seed': seed_number, 'panel': sensitivity,
                                           'method': 'Wilson', 'analysis': 'individual seed',
                                           'operating_point': point, 'endpoint': endpoint,
                                           'estimate': value[endpoint], 'ci_low': low, 'ci_high': high})
                if len(decisions)>1:
                    mean_prob = scores.mean(axis=0)
                    if point == 'default': ensemble_decision = (mean_prob >= .5).astype(int)
                    else:
                        thresholds = json.loads((results/'thresholds.json').read_text())['thresholds']
                        procedure = panel.procedure.iloc[0]
                        ensemble_decision = (mean_prob > thresholds[procedure]['threshold']).astype(int)
                    ensemble_metrics = prediction_metrics(truth, ensemble_decision, mean_prob)
                    ensemble_intervals = bootstrap_intervals(truth, ensemble_decision, mean_prob, groups=groups,
                        iterations=repetitions, seed=seed, endpoints=ENDPOINTS)
                    ensemble_rows.append({'model': model, 'panel': sensitivity, 'operating_point': point,
                                          'analysis': '3-seed ensemble sensitivity', **ensemble_metrics,
                                          **{f'{endpoint}_ci_{bound}': interval[j] for endpoint, interval in ensemble_intervals.items()
                                             for j,bound in enumerate(('low', 'high'))}})
                    for endpoint, interval in ensemble_intervals.items():
                        confidence.append({'model': model, 'panel': sensitivity, 'operating_point': point,
                                           'method': 'whole-scene bootstrap', 'analysis': '3-seed ensemble sensitivity',
                                           'endpoint': endpoint, 'estimate': ensemble_metrics[endpoint],
                                           'ci_low': interval[0], 'ci_high': interval[1]})
            for seed_number, score in zip(sorted(panel.seed.unique()), scores):
                rel = reliability(truth, score)
                rel['model'], rel['seed'], rel['panel'] = model, seed_number, sensitivity
                rel['brier'] = brier_score_loss(truth, score)
                calibration.append(rel)
        fixed = population[population.family.eq('fixed')]
        neural = population[population.family.eq('fine_tuning')]
        for model, panel in fixed.groupby('model'):
            panel = panel.sort_values('filename')
            for ft_name, ft in neural.groupby('model'):
                for point in ('default', 'safety'):
                    matrix = ft.pivot(index='filename', columns='seed', values=f'predicted_label_{point}').loc[panel.filename].to_numpy().T
                    differences = paired_bootstrap(panel.true_label, panel[f'predicted_label_{point}'], matrix,
                        groups=panel.group_id, iterations=repetitions, seed=seed)
                    for endpoint, interval in differences.items():
                        paired.append({'fixed': model, 'fine_tuning': ft_name, 'panel': sensitivity, 'operating_point': point,
                                       'endpoint': endpoint, **interval,
                                       'substitution': substitution_status(interval['ci_low'], interval['ci_high'],
                                                                           lower_is_better=endpoint.endswith('negative_rate'))})
                    for seed_number, decisions in zip(sorted(ft.seed.unique()), matrix):
                        first, second, p = mcnemar_exact(panel.true_label, panel[f'predicted_label_{point}'], decisions)
                        mcnemar.append({'fixed': model, 'fine_tuning': ft_name, 'seed': seed_number, 'panel': sensitivity,
                                        'operating_point': point, 'fixed_only_correct': first, 'ft_only_correct': second, 'p_value': p})
    table(pd.DataFrame(summaries), output, 'seed_mean_metrics')
    table(pd.DataFrame(confidence), output, 'cluster_bootstrap_wilson')
    table(pd.DataFrame(paired), output, 'substitution_analysis')
    table(pd.DataFrame(ensemble_rows), output, 'ensemble_sensitivity')
    mcnemar_frame = pd.DataFrame(mcnemar)
    for _, index in mcnemar_frame.groupby(['panel', 'operating_point']).groups.items():
        mcnemar_frame.loc[index, 'p_holm'] = holm_adjust(mcnemar_frame.loc[index, 'p_value'].to_numpy())
    table(mcnemar_frame, output, 'mcnemar_exact_holm')
    reliability_table = pd.concat(calibration)
    table(reliability_table, output, 'calibration_reliability')
    obstacle = predictions[predictions.true_label.eq(0)].copy()
    obstacle['false_clear'] = obstacle.predicted_label_default.eq(1)
    table(obstacle.groupby(['model', 'seed', 'false_clear']).score_clear.describe().reset_index(), output, 'obstructed_score_distributions')
    if not smoke:
        config = json.loads((ROOT/'configs/external_test.json').read_text())
        article_design = config.get('fixed_oof', {}).get('design') == 'article_original_10fold'
        nested = pd.read_csv(results/'fixed_oof_predictions.csv' if article_design else ROOT/config['nested_results']/'outer_predictions.csv', float_precision='round_trip')
        via_calibrated = pd.read_csv(results/'calibrated_via_oof.csv', float_precision='round_trip')
        ft = pd.read_csv(ROOT/config['fine_tuning_results']/'per_image_predictions.csv', float_precision='round_trip')
        full_metrics = pd.read_csv(ROOT/config['fixed_search_results']/'per_fold_metrics.csv')
        ft_metrics = pd.read_csv(ROOT/config['fine_tuning_results']/'per_fold_metrics.csv')
        bias_path = output/'tibshirani_bias.csv'
        bias = pd.read_csv(bias_path).set_index('approach') if bias_path.exists() else None
        domain, comparison, domain_calibration = [], [], []
        for model, panel in predictions[~predictions.scene_overlap].groupby('model'):
            filenames = sorted(panel.filename.unique())
            external_truth = panel.groupby('filename').true_label.first().loc[filenames].to_numpy()
            groups = panel.groupby('filename').group_id.first().loc[filenames].to_numpy()
            decisions = panel.pivot(index='filename', columns='seed', values='predicted_label_default').loc[filenames].to_numpy().T
            if panel.family.iloc[0] == 'fixed':
                source = nested[nested.procedure.eq(panel.procedure.iloc[0])].sort_values('filename')
                via_decisions = source.predicted_label_default.to_numpy()
                calibration_source = via_calibrated[via_calibrated.procedure.eq(panel.procedure.iloc[0])].sort_values('filename')
                via_probabilities = calibration_source.score_clear.to_numpy()[None, :]
                original = full_metrics[full_metrics.experiment.eq(model)]
                corrected = bias.loc[panel.procedure.iloc[0]] if bias is not None else None
            else:
                architecture = model.removeprefix('ft_')
                source = ft[ft.architecture.eq(architecture) & ft.family.eq('fine_tuning')]
                via_decisions = source.pivot(index='filename', columns='seed', values='predicted_label').sort_index().to_numpy().T
                via_probabilities = source.pivot(index='filename', columns='seed', values='score_clear').sort_index().to_numpy().T
                source = source.sort_values('filename').drop_duplicates('filename')
                original = ft_metrics[ft_metrics.architecture.eq(architecture) & ft_metrics.family.eq('fine_tuning')]
                corrected = None
            for row in domain_difference(source.true_label.to_numpy(), via_decisions, external_truth, decisions, groups,
                                         repetitions=repetitions, seed=seed):
                domain.append({'model': model, **row})
            original_accuracy = float(original.accuracy.mean())
            original_recall = float(original.obstructed_recall.mean())
            external_scores = panel.pivot(index='filename', columns='seed', values='score_clear').loc[filenames].to_numpy().T
            external_metrics = [prediction_metrics(external_truth, d, s) for d,s in zip(decisions, external_scores)]
            comparison.append({'model': model, 'via_original_accuracy': original_accuracy,
                'via_original_obstructed_recall': original_recall,
                'via_tibshirani_accuracy': None if corrected is None else float(corrected.accuracy_corrected),
                'via_tibshirani_obstructed_recall': None if corrected is None else float(corrected.obstructed_recall_corrected),
                **{f'external_{endpoint}': float(np.mean([m[endpoint] for m in external_metrics])) for endpoint in ENDPOINTS},
                'external_minus_original_accuracy': float(np.mean([m['accuracy'] for m in external_metrics]))-original_accuracy,
                'external_minus_tibshirani_accuracy': None if corrected is None else float(np.mean([m['accuracy'] for m in external_metrics]))-float(corrected.accuracy_corrected),
                'notice': 'Original VIA native-default mean fold metrics; external default uses frozen probabilities. Corrected recall is a conservative secondary adaptation.'})
            for domain_name, domain_truth, probabilities in [('via_oof', source.true_label.to_numpy(), via_probabilities),
                                                            ('external', external_truth, external_scores)]:
                for seed_index, probability in enumerate(probabilities):
                    rel = reliability(domain_truth, probability)
                    rel['model'], rel['domain'], rel['seed_index'] = model, domain_name, seed_index
                    rel['brier'] = brier_score_loss(domain_truth, probability)
                    rel['calibration_notice'] = 'Fixed VIA probabilities fitted on the same selected-finalist OOF; optimistic calibration reference' if domain_name == 'via_oof' and panel.family.iloc[0] == 'fixed' else 'Frozen probability model'
                    domain_calibration.append(rel)
        table(pd.DataFrame(domain), output, 'domain_shift')
        table(pd.DataFrame(comparison), output, 'domain_comparison')
        table(pd.concat(domain_calibration), output, 'domain_calibration')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(7, 5)); axis.plot([0, 1], [0, 1], color='gray', linestyle='--')
    for model, rows in reliability_table[reliability_table.panel.eq('principal')].groupby('model'):
        rows = rows.groupby('bin')[['mean_probability', 'observed_clear_fraction']].mean()
        axis.plot(rows.mean_probability, rows.observed_clear_fraction, marker='.', label=model)
    axis.set(xlabel='Clear probability', ylabel='Observed clear fraction', xlim=(0, 1), ylim=(0, 1))
    axis.legend(fontsize=6); fig.tight_layout(); fig.savefig(output/'reliability.png', dpi=160); fig.savefig(output/'reliability.pdf'); plt.close(fig)
    write_json(output/'analysis_manifest.json', {'bootstrap_iterations': repetitions, 'seed': seed, 'smoke': smoke,
               'source_predictions_sha256': sha256(results/'predictions.csv'),
               'cluster_strata': ['obstructed-only', 'clear-only', 'mixed'], 'primary': 'mean of individual seed metrics'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/external_test'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bootstrap-iterations', type=int, default=10000)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    analyze(args.results, output=args.output, repetitions=40 if args.smoke else args.bootstrap_iterations, smoke=args.smoke)


if __name__ == '__main__': main()
