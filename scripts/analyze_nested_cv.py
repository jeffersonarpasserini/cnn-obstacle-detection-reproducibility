#!/usr/bin/env python3
"""Nested OOF intervals, selection optimism and paired FT/grouping comparisons."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from src.artifact import prediction_metrics
from src.study_io import sha256, atomic_csv, write_json, guard_output
from src.study_statistics import bootstrap_intervals, paired_bootstrap, wilson
from analyze_fine_tuning import paired_wilcoxon, mcnemar_exact
from build_manuscript_tables import holm_adjust, latex_escape


def table(frame, output, name):
    atomic_csv(frame, output/f'{name}.csv')
    lines = []
    for row in frame.itertuples(index=False, name=None):
        cells = [f'{value:.4f}' if isinstance(value, (float, np.floating)) and np.isfinite(value)
                 else '--' if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
                 else latex_escape(str(value)) for value in row]
        lines.append(' & '.join(cells) + r' \\')
    (output/f'{name}_rows.tex').write_text('\n'.join(lines)+'\n')


def verify_metrics(predictions, records):
    for row in records.to_dict('records'):
        panel = predictions[predictions.fold.eq(row['fold']) & predictions.procedure.eq(row['procedure'])]
        metric = prediction_metrics(panel.true_label, panel[f"predicted_label_{row['operating_point']}"], panel.score_clear)
        for key, value in metric.items():
            if not np.isclose(value, row[key], equal_nan=True, atol=1e-12):
                raise ValueError(f'Recomputed nested metric differs: {key}')


def analyze(results, *, output=None, repetitions=10000, seed=1980, smoke=False, compare_ungrouped=None):
    results = Path(results)
    metadata = json.loads((results/'run_metadata.json').read_text())
    if not smoke and (not metadata.get('completed') or metadata.get('smoke')):
        raise ValueError('Scientific nested analysis requires all outer folds, not a smoke/partial run')
    for name, checksum in metadata['outputs'].items():
        if sha256(results/name) != checksum:
            raise ValueError(f'Nested source changed: {name}')
    predictions = pd.read_csv(results/'outer_predictions.csv')
    metrics = pd.read_csv(results/'outer_metrics.csv')
    if predictions.duplicated(['filename', 'procedure']).any():
        raise ValueError('Duplicate nested OOF predictions')
    verify_metrics(predictions, metrics)
    output = guard_output(output or results/'manuscript_tables'); output.mkdir(parents=True, exist_ok=True)
    summary = []
    for (procedure, point), folds in metrics.groupby(['procedure', 'operating_point']):
        panel = predictions[predictions.procedure.eq(procedure)].sort_values('filename')
        decisions = panel[f'predicted_label_{point}']
        pooled = prediction_metrics(panel.true_label, decisions, panel.score_clear)
        # Native margins/probabilities may differ across selected configurations. AUC is descriptive.
        intervals = bootstrap_intervals(panel.true_label, decisions, iterations=repetitions, seed=seed)
        n_obstructed = int(panel.true_label.eq(0).sum())
        recall = wilson(pooled['tn_obstructed'], n_obstructed)
        fnr = wilson(pooled['fp_obstructed_as_clear'], n_obstructed)
        for endpoint, value in pooled.items():
            if endpoint not in folds:
                continue
            row = {'procedure': procedure, 'operating_point': point, 'endpoint': endpoint,
                   'fold_mean': folds[endpoint].mean(), 'fold_std': folds[endpoint].std(),
                   'fold_median': folds[endpoint].median(),
                   'fold_iqr': folds[endpoint].quantile(.75)-folds[endpoint].quantile(.25), 'pooled': value,
                   'ci_low': np.nan, 'ci_high': np.nan, 'diagnostic': smoke}
            if endpoint in intervals:
                row['ci_low'], row['ci_high'] = intervals[endpoint]
            if endpoint == 'obstructed_recall': row['ci_low'], row['ci_high'] = recall
            if endpoint == 'obstructed_false_negative_rate': row['ci_low'], row['ci_high'] = fnr
            summary.append(row)
    table(pd.DataFrame(summary), output, 'nested_summary')
    choices = pd.read_csv(results/'selected_configurations.csv')
    table(choices, output, 'selection_optimism')
    table(choices.groupby(['procedure', 'selected_configuration']).size().reset_index(name='outer_fold_count'), output, 'selection_frequency')
    obstruction = predictions[predictions.true_label.eq(0)].copy()
    obstruction['false_clear'] = obstruction.predicted_label_default.eq(1)
    table(obstruction.groupby(['procedure', 'false_clear']).score_clear.describe().reset_index(), output, 'obstructed_score_distributions')
    if compare_ungrouped is not None:
        if not metadata.get('groups_sha256'):
            raise ValueError('Grouped comparison must start from grouped results')
        original_meta = json.loads((Path(compare_ungrouped)/'run_metadata.json').read_text())
        if not original_meta.get('completed') or original_meta.get('groups_sha256'):
            raise ValueError('Grouped sensitivity needs complete original ungrouped results')
        other = pd.read_csv(Path(compare_ungrouped)/'outer_predictions.csv')
        rows = []
        for procedure, panel in predictions.groupby('procedure'):
            aligned = panel.merge(other[other.procedure.eq(procedure)], on='filename', suffixes=('_grouped', '_original'), validate='one_to_one')
            if len(aligned) != len(panel) or not aligned.true_label_grouped.eq(aligned.true_label_original).all():
                raise ValueError('Grouped/non-grouped image pairing differs')
            for endpoint, interval in paired_bootstrap(aligned.true_label_grouped, aligned.predicted_label_default_grouped,
                    aligned.predicted_label_default_original, iterations=repetitions, seed=seed).items():
                rows.append({'procedure': procedure, 'endpoint': endpoint, **interval, 'direction': 'grouped minus original'})
        table(pd.DataFrame(rows), output, 'grouped_vs_ungrouped')
    # FT had only original ungrouped folds; primary = average individual seed metrics.
    if not metadata.get('groups_sha256') and not smoke:
        config = metadata['identity']['config']
        ft_path = ROOT/config['fine_tuning_results']/'per_image_predictions.csv'
        ft = pd.read_csv(ft_path); ft = ft[ft.family.eq('fine_tuning')]
        pairs, bootstrap, mcnemar, ensemble_rows = [], [], [], []
        for procedure, panel in predictions.groupby('procedure'):
            panel = panel.sort_values('filename')
            for architecture, neural in ft.groupby('architecture'):
                matrix = neural.pivot(index='filename', columns='seed', values='predicted_label').loc[panel.filename].to_numpy().T
                probability_matrix = neural.pivot(index='filename', columns='seed', values='score_clear').loc[panel.filename].to_numpy().T
                by_fold = []
                for fold, outer in panel.groupby('fold'):
                    values = []
                    for _, seed_rows in neural[neural.fold.eq(fold)].groupby('seed'):
                        if set(seed_rows.filename) != set(outer.filename):
                            raise ValueError('FT folds differ from original nested outer folds')
                        values.append(prediction_metrics(seed_rows.true_label, seed_rows.predicted_label)['accuracy'])
                    by_fold.append({'procedure': procedure, 'architecture': architecture, 'fold': fold,
                                    'difference': prediction_metrics(outer.true_label, outer.predicted_label_default)['accuracy']-np.mean(values)})
                statistic, p_value = paired_wilcoxon([row['difference'] for row in by_fold])
                pairs.append({'procedure': procedure, 'architecture': architecture, 'wilcoxon': statistic,
                              'p_value': p_value, 'mean_difference': np.mean([row['difference'] for row in by_fold])})
                for endpoint, interval in paired_bootstrap(panel.true_label, panel.predicted_label_default, matrix,
                        iterations=repetitions, seed=seed).items():
                    bootstrap.append({'procedure': procedure, 'architecture': architecture, 'endpoint': endpoint,
                                      'comparison': 'mean individual seed metrics', **interval})
                for seed_number, decisions in zip(sorted(neural.seed.unique()), matrix):
                    first, second, p = mcnemar_exact(panel.true_label, panel.predicted_label_default, decisions)
                    mcnemar.append({'procedure': procedure, 'architecture': architecture, 'seed': seed_number,
                                    'fixed_only_correct': first, 'ft_only_correct': second, 'p_value': p})
                ensemble = (probability_matrix.mean(axis=0) > .5).astype(int)
                ensemble_rows.append({'procedure': procedure, 'architecture': architecture, 'sensitivity': '3-seed ensemble',
                                      **prediction_metrics(panel.true_label, ensemble, probability_matrix.mean(axis=0))})
        for rows, name in [(pairs, 'wilcoxon_holm'), (mcnemar, 'mcnemar_exact_holm')]:
            frame = pd.DataFrame(rows); frame['p_holm'] = holm_adjust(frame.p_value.to_numpy()); table(frame, output, name)
        table(pd.DataFrame(bootstrap), output, 'paired_image_bootstrap')
        table(pd.DataFrame(ensemble_rows), output, 'fine_tuning_ensemble_sensitivity')
    write_json(output/'analysis_manifest.json', {'source': str(results), 'metadata_sha256': sha256(results/'run_metadata.json'),
               'bootstrap_iterations': repetitions, 'seed': seed, 'diagnostic': smoke,
               'estimand': 'configuration-selection procedure',
               'ft_comparison': 'not performed for grouped folds' if metadata.get('groups_sha256') else 'mean individual seed metrics; ensemble sensitivity'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/nested_cv'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare-ungrouped', type=Path)
    parser.add_argument('--bootstrap-iterations', type=int, default=10000)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    analyze(args.results, output=args.output, repetitions=100 if args.smoke else args.bootstrap_iterations,
            smoke=args.smoke, compare_ungrouped=args.compare_ungrouped)


if __name__ == '__main__': main()
