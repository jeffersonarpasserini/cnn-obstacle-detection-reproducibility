"""Read-only validation of the author's exploratory/post hoc review."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .study_io import sha256
from .stage3_io import table


# Closeout checks specified by the author, not a replacement category vocabulary.
EXPECTED_COUNTS = {
    'clear_to_nonclear': {
        'bueiro/grelha': 24, 'mudança de piso/textura': 24,
        'cena aberta/horizonte': 21, 'sem causa aparente': 16,
        'grama/vegetação': 13, 'baixa luz': 10, 'rótulo questionável': 8,
        'junta ou rachadura': 8, 'piso tátil': 4, 'corredor estreito': 4,
        'faixa ou pintura': 3, 'sombra': 2,
    },
    'nonclear_to_clear': {
        'borda de piscina': 4, 'rótulo questionável': 3, 'baixa luz': 2,
        'desfoque': 1, 'obstáculo pequeno/distante': 1,
    },
}


def load_error_categories(config_path: Path, addendum_path: Path) -> list[str]:
    """Read the frozen vocabulary and the delimited complete list in the addendum."""
    config = json.loads(config_path.read_text(encoding='utf-8'))
    text = addendum_path.read_text(encoding='utf-8')
    match = re.search(
        r'^##\s+Lista completa usada na anotação\s*\n(.*?)(?=^#{1,6}\s|\Z)',
        text, flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError('Category addendum has no complete annotation list')
    section = match.group(1).strip().rstrip('.')
    added = [item.strip().strip('`') for item in section.split(',')]
    if not section or any(not item for item in added):
        raise ValueError('Category addendum has an empty category list/item')
    return list(dict.fromkeys([*config['error_categories'], *added]))


def validate_author_review(root: Path, mode: str = 'final'):
    """Validate copies read from disk; never write names, annotations or categories."""
    if mode not in {'handoff', 'final'}:
        raise ValueError('Unknown stage-three report mode')
    folder = root/'results/stage3/error_analysis'
    clusters = pd.read_csv(folder/'cluster_names.csv', keep_default_na=False, encoding='utf-8')
    errors = pd.read_csv(folder/'error_annotations.csv', keep_default_na=False, encoding='utf-8')
    if mode == 'handoff':
        if not clusters.name.eq('').all() or not errors.category.eq('').all():
            raise ValueError('Author-only cluster names/error categories must stay empty at this handoff')
        return clusters, errors
    for frame, columns, title in [
        (clusters, ['cluster', 'name', 'reviewer', 'date'], 'Cluster review'),
        (errors, ['filename', 'error_type', 'category', 'reviewer', 'date'], 'Error annotations'),
    ]:
        if not set(columns) <= set(frame.columns):
            raise ValueError(f'{title} is missing required columns')
        if frame[columns].astype(str).apply(lambda col: col.str.strip().eq('')).any().any():
            raise ValueError(f'{title} is incomplete')
    cluster_ids = pd.to_numeric(clusters.cluster, errors='raise')
    if len(clusters) != 12 or set(cluster_ids) != set(range(12)):
        raise ValueError('Cluster review must cover clusters 0–11 exactly once')
    consensus = pd.read_csv(folder/'manuscript_tables/consensus_errors.csv',
                            keep_default_na=False, encoding='utf-8')
    keys = ['filename', 'error_type']
    if (len(errors) != 148 or errors.filename.duplicated().any()
            or errors[keys].to_dict('records') != consensus[keys].to_dict('records')):
        raise ValueError('Error annotations must match all 148 consensus errors in order')
    categories = load_error_categories(root/'configs/stage3.json', folder/'CATEGORY_ADDENDUM.md')
    invalid = sorted(set(errors.category) - set(categories))
    if invalid:
        raise ValueError(f'Error category outside the configured/addendum list: {invalid}')
    actual = errors.groupby(['error_type', 'category']).size().to_dict()
    expected = {(kind, category): n for kind, counts in EXPECTED_COUNTS.items()
                for category, n in counts.items()}
    if actual != expected:
        raise ValueError(f'Author category counts differ from the closeout checks: {actual}')
    return clusters, errors


def validate_blind_analysis(root: Path, mode: str = 'final'):
    """Check the recorded, nonsynthetic analysis without rerunning its estimator."""
    folder = root/'results/stage3/blind_relabel'
    status = json.loads((folder/'status.json').read_text(encoding='utf-8'))
    if mode == 'handoff':
        if status['analysis_executed'] or status['responses_received']:
            raise ValueError('Blind study has no real responses yet')
        return status
    if mode != 'final':
        raise ValueError('Unknown stage-three report mode')
    if status.get('analysis_executed') is not True or status.get('responses_received') is not True:
        raise ValueError('Real blind responses must have been received and analysed')
    if any(not str(status.get(field, '')).strip()
           for field in ['rater', 'responses', 'responses_received_date']):
        raise ValueError('Blind response provenance is incomplete')
    if status.get('bootstrap_iterations') != 10000 or status.get('bootstrap_seed') != 1980:
        raise ValueError('Blind analysis must use 10000 iterations and seed 1980')
    tables = folder/'manuscript_tables'
    for field, path in [('responses_sha256', root/status['responses']),
                        ('agreement_sha256', tables/'agreement.csv')]:
        if status.get(field) != sha256(path):
            raise ValueError(f'Blind analysis hash mismatch: {field}')
    agreement = pd.read_csv(tables/'agreement.csv')
    if len(agreement) != 1:
        raise ValueError('Blind agreement must contain one analysis')
    row = agreement.iloc[0]
    expected = {
        'agreement': .98, 'agreement_wilson_low': .929988,
        'agreement_wilson_high': .994498, 'agreement_cluster_ci_low': .949495,
        'agreement_cluster_ci_high': 1., 'cohen_kappa': .959612,
        'kappa_ci_low': .897748, 'kappa_ci_high': 1.,
    }
    if (row.synthetic != False or row.bootstrap_iterations != 10000
            or any(not np.isfinite(row[field]) or abs(row[field] - value) > 1e-6
                   for field, value in expected.items())):
        raise ValueError('Blind agreement differs from the official closeout checks')
    matrix = pd.read_csv(tables/'confusion_matrix.csv')
    if (matrix.original_label.tolist() != [0, 1]
            or not np.array_equal(matrix[['0', '1']].to_numpy(), [[44, 2], [0, 54]])):
        raise ValueError('Blind confusion matrix differs from the official checks')
    discordances = pd.read_csv(tables/'discordances.csv')
    expected_discordances = [
        {'item': 'item_042.jpg', 'filename': 'nonclear.0256.jpg', 'group_id': 'scene_0831',
         'original_label': 0, 'blind_label': 1},
        {'item': 'item_056.jpg', 'filename': 'nonclear.0258.jpg', 'group_id': 'scene_0833',
         'original_label': 0, 'blind_label': 1},
    ]
    if discordances[list(expected_discordances[0])].to_dict('records') != expected_discordances:
        raise ValueError('Blind discordances differ from the official checks')
    return status


def build_review_tables(root: Path, clusters: pd.DataFrame, errors: pd.DataFrame):
    """Join the author's review to frozen assignments/metrics; no inference or refits."""
    folder = root/'results/stage3/error_analysis'
    tables = folder/'manuscript_tables'
    categories = load_error_categories(root/'configs/stage3.json', folder/'CATEGORY_ADDENDUM.md')
    index = pd.MultiIndex.from_product(
        [categories, ['clear_to_nonclear', 'nonclear_to_clear']], names=['category', 'error_type'])
    counts = errors.groupby(['category', 'error_type']).size().reindex(index, fill_value=0)
    category_counts = counts.rename('n').reset_index()
    totals = errors.error_type.value_counts()
    category_counts['percent_within_type'] = (
        100 * category_counts.n / category_counts.error_type.map(totals))
    category_counts = category_counts.sort_values(
        ['error_type', 'n', 'category'], ascending=[True, False, True]).reset_index(drop=True)

    assignments = pd.read_csv(tables/'cluster_assignments.csv')
    external = assignments.loc[assignments.domain.eq('external'), ['filename', 'cluster']]
    joined = errors.merge(external, on='filename', how='left', validate='one_to_one')
    joined = joined.merge(clusters[['cluster', 'name']], on='cluster', how='left', validate='many_to_one')
    if joined[['cluster', 'name']].isna().any().any():
        raise ValueError('A consensus error has no external cluster assignment/name')
    category_by_cluster = joined.groupby(
        ['category', 'error_type', 'cluster', 'name']).size().rename('n').reset_index()

    cluster_counts = pd.read_csv(tables/'cluster_counts.csv').drop(columns='analysis_scope')
    named = cluster_counts.merge(clusters[['cluster', 'name']], on='cluster',
                                 how='outer', validate='one_to_one')
    if len(named) != 12 or named.isna().any().any():
        raise ValueError('Named summary must cover the 12 recorded cluster counts')
    named['external_clear_percent'] = 100 * named.external_clear_fraction
    named['via_clear_percent'] = 100 * named.via_clear_fraction
    metric_names = {
        ('b_efficientnetb3_pca_300_linear_svm', 'accuracy'): 'fixed_b_accuracy',
        ('b_efficientnetb3_pca_300_linear_svm', 'obstructed_recall'): 'fixed_b_obstructed_recall',
        ('ft_EfficientNetB3', 'accuracy'): 'ft_efficientnetb3_accuracy',
        ('ft_EfficientNetB3', 'obstructed_recall'): 'ft_efficientnetb3_obstructed_recall',
    }
    metrics = pd.read_csv(tables/'cluster_metrics.csv')
    metrics = metrics.loc[
        metrics.model.isin({model for model, _ in metric_names}) &
        metrics.endpoint.isin({'accuracy', 'obstructed_recall'})]
    if len(metrics) != 48 or set(metrics.cluster) != set(range(12)):
        raise ValueError('Recorded B/FT metrics do not cover the 12 clusters')
    wide = metrics.pivot(index='cluster', columns=['model', 'endpoint'], values='estimate')
    wide = wide[list(metric_names)]
    wide.columns = list(metric_names.values())
    named = named.merge(wide.reset_index(), on='cluster', how='left', validate='one_to_one')
    if named[list(metric_names.values())].isna().any().any():
        raise ValueError('Named cluster summary is missing a recorded model metric')
    named = named.sort_values('cluster').reset_index(drop=True)
    named['ft_metric_aggregation'] = 'mean of 3 individual-seed metrics'
    result = {
        'error_category_counts': category_counts,
        'error_category_by_cluster': category_by_cluster,
        'cluster_summary_named': named,
    }
    return {name: table(frame, folder, name) for name, frame in result.items()}
