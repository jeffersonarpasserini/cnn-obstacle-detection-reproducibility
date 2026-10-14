#!/usr/bin/env python3
"""Read-only stage-one audit; emits the author's review report without approving it."""
import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.study_io import sha256, write_json, atomic_csv
from src.artifact import scan_dataset, dataset_content_fingerprint
from src.selection import select_procedures
from scripts.analyze_nested_cv import verify_metrics


def main():
    output = ROOT/'results/stage1_verification'
    checks = []
    def check(name, expected, obtained):
        if expected != obtained:
            raise ValueError(f'{name}: {expected} != {obtained}')
        checks.append({'verification': name, 'expected': expected, 'obtained': obtained, 'status': 'passed'})
    # Reference is the pull immediately preceding this implementation; six functions were moved verbatim.
    old = subprocess.check_output(['git', 'show', '4a5a1a5:cnn-obstacle-detection-reproducibility/scripts/build_manuscript_tables.py'], cwd=ROOT, text=True)
    original = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(old).body if isinstance(node, ast.FunctionDef)}
    current = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse((ROOT/'src/selection.py').read_text()).body if isinstance(node, ast.FunctionDef)}
    names = ['average_friedman_ranks', 'pooled_metrics', 'experiment_label', 'enrich_table', 'choose_winner', 'select_hierarchically']
    check('Original selection functions have identical AST', 6, sum(original[name] == current[name] for name in names))
    search = json.loads((ROOT/'configs/full_search.json').read_text())
    finalists = json.loads((ROOT/'configs/article_selected_models_10fold.json').read_text())
    selection = select_procedures(pd.read_csv(ROOT/'results/full_search/per_fold_metrics.csv'), search)
    for experiment in finalists['experiments']:
        approach = experiment['approach']
        check(f'Article finalist {approach}', experiment['name'], selection['selected'][approach])
    winner = pd.read_csv(ROOT/'results/full_search/manuscript_tables/selected_approaches.csv').iloc[0].experiment
    check('Article global winner', winner, selection['winner'])
    protected = json.loads((output/'protected_inputs_verification.json').read_text())
    check('Protected files changed', 0, len(protected['changed']))
    check('Training package versions unchanged', True, protected['packages_unchanged'])
    paths, _ = scan_dataset(ROOT/'via-dataset')
    reference = json.loads((ROOT/'results/full_search/run_metadata.json').read_text())
    check('VIA content fingerprint', reference['dataset_content_sha256'], dataset_content_fingerprint(paths))
    report = json.loads((ROOT/'external-data/cleaning_report.json').read_text())
    check('Raw images', {'total': 1929, 'clear': 1026, 'nonclear': 903}, report['raw_counts'])
    check('Novel exact-clean images', {'total': 1576, 'clear': 844, 'nonclear': 732}, report['exact_clean_counts'])
    check('VIA exact matches excluded', 350, report['via_exact_matches'])
    check('Label-swapped VIA copies', 4, report['via_label_conflicts'])
    check('Internal exact duplicates excluded', 3, report['internal_duplicates'])
    raw = pd.read_csv(ROOT/'external-data/external_manifest_raw.csv')
    via_hashes = {sha256(path) for path in paths}
    novel = pd.read_csv(ROOT/'external-data/external_manifest_clean.csv')
    check('Novel SHA intersection with VIA', 0, len(set(novel.sha256) & via_hashes))
    check('Novel duplicate SHA', 0, int(novel.sha256.duplicated().sum()))
    overlaps = pd.read_csv(ROOT/'external-data/external_excluded_scene_overlap.csv')
    check('Principal excludes every accepted automatic scene link', 0,
          len(set(overlaps.filename) & set(novel[~novel.excluded].filename)))
    check('Author approval remains pending', None, report['approved_by'])
    check('Audit sample size', 100, report['audit_images'])
    sample = pd.read_csv(ROOT/'external-data/label_audit_sample.csv')
    check('Audit class split', [54, 46], [int(sample.original_label.eq(1).sum()), int(sample.original_label.eq(0).sum())])
    for stem, count in [('scene', 2), ('audit', 5)]:
        check(f'{stem} mosaic files exist', count, sum((ROOT/path).is_file() for path in report[f'{stem}_mosaics']))
    for directory, image_count in [('nested_cpu', 35), ('nested_grouped_cpu', 34)]:
        location = output/directory
        metadata = json.loads((location/'run_metadata.json').read_text())
        check(f'{directory} diagnostic completion', True, metadata['requested_run_completed'] and metadata['smoke'] and not metadata['completed'])
        predictions = pd.read_csv(location/'outer_predictions.csv')
        check(f'{directory} outer prediction rows', image_count*5, len(predictions))
        check(f'{directory} duplicate image/procedure', 0, int(predictions.duplicated(['filename', 'procedure']).sum()))
        records = pd.read_csv(location/'outer_metrics.csv'); verify_metrics(predictions, records)
        check(f'{directory} recomputed outer metric rows', 10, len(records))
        inner = pd.read_csv(location/'inner_results/outer_01/inner_metrics.csv.gz')
        check(f'{directory} inner attempts', 40, len(inner))
        check(f'{directory} inner failed attempts', 0, int(inner.status.ne('valid').sum()))
    models = output/'models_cpu'
    freeze = json.loads((models/'final_models.json').read_text())
    check('CPU model smoke never freezes production models', False, freeze['frozen'])
    check('Fixed CPU reload panels', 4, sum(row['family'] == 'fixed' for row in freeze['models']))
    check('Neural CPU reload panels', 5, sum(row['family'] == 'fine_tuning' for row in freeze['models']))
    costs = pd.read_csv(models/'computational_costs.csv')
    neural = costs[costs.seed.notna()]
    check('Neural smoke epochs', [1], sorted(neural.effective_epochs.unique().astype(int).tolist()))
    check('Neural smoke seeds', [1980], sorted(neural.seed.unique().astype(int).tolist()))
    check('Real external classifications', 0, report['external_classification_predictions']+freeze['external_classification_predictions'])
    check('Production external predictions exist', False, (ROOT/'results/external_test/predictions.csv').exists())
    check('Production nested ungrouped exists', False, (ROOT/'results/nested_cv/outer_predictions.csv').exists())
    check('Production nested grouped exists', False, (ROOT/'results/nested_cv_grouped/outer_predictions.csv').exists())
    # Raw files removed from the index, never from disk.
    tracked = subprocess.check_output(['git', 'ls-files', '--', 'external-data/via-dataset-extended'], cwd=ROOT, text=True).splitlines()
    check('Raw images still tracked in current index', 0, len(tracked))
    check('Raw images still local', 1929, len(list((ROOT/'external-data/via-dataset-extended').glob('*.jpg'))))
    write_json(output/'verification_report.json', {'checks': checks, 'cleaning': report,
               'protected': protected, 'pytest_log_sha256': sha256(output/'final_pytest.log'),
               'estimated_stages': {name: json.loads((output/f'dry_{name}.json').read_text()) for name in
                                    ('nested', 'nested_grouped', 'models', 'external', 'learning')},
               'source_reference_commit': '4a5a1a5', 'author_review_required': True})
    atomic_csv(pd.DataFrame(checks), output/'expected_vs_obtained.csv')
    print(f'{len(checks)} stage-one verification rows passed; author approval remains pending')


if __name__ == '__main__': main()
