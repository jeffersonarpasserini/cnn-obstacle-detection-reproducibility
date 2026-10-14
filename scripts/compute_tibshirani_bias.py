#!/usr/bin/env python3
"""Author-specified fold-wise optimism correction, using archived metrics only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.selection import select_procedures
from src.study_io import sha256, write_json, guard_output
from scripts.analyze_nested_cv import table


def tibshirani_tables(records, selected, winner, experiments):
    if records.duplicated(['experiment', 'fold']).any():
        raise ValueError('Duplicate experiment/fold in the bias input')
    if set(records.experiment) != set(experiments):
        raise ValueError('Bias correction requires the entire original search grid')
    folds = sorted(records.fold.unique())
    if any(sorted(panel.fold.tolist()) != folds for _, panel in records.groupby('experiment')):
        raise ValueError('Bias correction requires complete common folds')
    rows, differences, diagnostics = [], [], []
    for approach, name in {**selected, 'winner': winner}.items():
        names = list(experiments) if approach == 'winner' else [n for n,e in experiments.items() if e['approach'] == approach]
        candidates = records[records.experiment.isin(names)]
        finalist = records[records.experiment.eq(name)].set_index('fold')
        row = {'approach': approach, 'experiment': name, 'configurations': len(names), 'folds': len(folds)}
        for endpoint in ('accuracy', 'obstructed_recall'):
            best = candidates.groupby('fold')[endpoint].max().reindex(finalist.index)
            bias = float((best-finalist[endpoint]).mean())
            estimate = float(finalist[endpoint].mean())
            row.update({f'{endpoint}_original': estimate, f'{endpoint}_bias': bias,
                        f'{endpoint}_corrected': estimate-bias})
            diagnostics.extend({'approach': approach, 'experiment': name, 'fold': int(fold),
                                'endpoint': endpoint, 'fold_maximum': float(best.loc[fold]),
                                'finalist': float(finalist.loc[fold, endpoint]),
                                'optimism': float(best.loc[fold]-finalist.loc[fold, endpoint])} for fold in finalist.index)
        rows.append(row)
        if approach != 'winner':
            relief_names = [n for n in names if experiments[n].get('reduction') == 'relieff']
            if relief_names:
                means = candidates[candidates.experiment.isin(relief_names)].groupby('experiment').accuracy.mean()
                best_name = means.sort_values(ascending=False, kind='stable').index[0]
                differences.append({'approach': approach, 'pca_finalist': name, 'best_relieff': best_name,
                                    'pca_accuracy_mean': row['accuracy_original'], 'relieff_accuracy_mean': float(means[best_name]),
                                    'pca_minus_relieff': row['accuracy_original']-float(means[best_name]),
                                    'definition': 'Highest mean-fold accuracy among all Relief-F configurations; name breaks ties'})
    return pd.DataFrame(rows), pd.DataFrame(differences), pd.DataFrame(diagnostics)


def compute(output):
    output = guard_output(output); output.mkdir(parents=True, exist_ok=True)
    path = ROOT/'results/full_search/per_fold_metrics.csv'
    records = pd.read_csv(path)
    config = json.loads((ROOT/'configs/full_search.json').read_text())
    experiments = {e['name']: e for e in config['experiments']}
    selection = select_procedures(records, config)
    bias, relief, diagnostics = tibshirani_tables(records, selection['selected'], selection['winner'], experiments)
    table(bias, output, 'tibshirani_bias')
    table(relief, output, 'relieff_vs_pca')
    table(diagnostics, output, 'tibshirani_fold_diagnostics')
    write_json(output/'tibshirani_manifest.json', {'per_fold_metrics_sha256': sha256(path),
        'configurations': len(experiments), 'new_fits': 0, 'selection': selection['selected'], 'winner': selection['winner'],
        'definition': 'Mean fold(maximum candidate endpoint minus selected-finalist endpoint); corrected=original-bias.',
        'notice': 'Conservative author-specified adaptation of Tibshirani & Tibshirani (2009); recall is a secondary extension, not an unbiased guarantee.',
        'reference': 'https://doi.org/10.1214/08-AOAS224'})
    print(bias.to_string(index=False), flush=True)
    print(relief.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/external_test/manuscript_tables'))
    args = parser.parse_args(); compute(args.output)


if __name__ == '__main__': main()
