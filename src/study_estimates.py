"""Read-only estimates from recorded timing/dimension measurements."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from .artifact import preprocessing_key
from .study_io import ROOT, sha256


def nested_estimate(config, *, grouped=False):
    reference = ROOT / 'results/full_search'
    records = pd.read_csv(reference / 'per_fold_metrics.csv')
    search = json.loads((ROOT / config['search_config']).read_text())
    experiments = {e['name']: e for e in search['experiments']}
    records = records[records.experiment.isin(experiments)].copy()
    records["preparation_key"] = records.experiment.map(lambda name: str(preprocessing_key(experiments[name])))
    feasible = [name for name,e in experiments.items() if not (e.get("reduction") in ("pca","umap") and int(e.get("components") or 0)>=250)]
    eligible = records[records.experiment.isin(feasible)].copy()
    relief_names = [name for name,e in experiments.items() if e.get("reduction")=="relieff" and e["approach"] in ("B","C")]
    relief = eligible[eligible.experiment.isin(relief_names)].copy()
    relief["ranking_key"] = relief.experiment.map(lambda name: str((experiments[name]["approach"],experiments[name]["extractors"])))
    rank_seconds = float(relief.groupby(["fold", "ranking_key"]).reduction_seconds.max().sum())
    reference_jobs = json.loads((reference/'run_metadata.json').read_text())['relieff_n_jobs']
    # Parallel speed-up is uncertain; do not silently reuse8-worker wall time for1-worker execution.
    relief_thread_scale = max(1., reference_jobs/max(1, config['relieff_n_jobs']))
    other = eligible[~eligible.experiment.map(lambda name: experiments[name].get("reduction")=="relieff")]
    other_seconds = float(other.groupby(["fold","preparation_key"]).reduction_seconds.max().sum())
    classifier = float(eligible.training_seconds.sum()+eligible.prediction_seconds.sum())
    inner = config["outer_folds"]*config["inner_folds"]
    prep = (rank_seconds*.64*relief_thread_scale+other_seconds*.8)*inner/10
    classify = classifier*.8*inner/10
    serial = prep+classify
    feature_time = pd.read_csv(reference / 'feature_extraction_times.csv')
    numeric = feature_time.select_dtypes('number')
    extraction = float(numeric.filter(regex='second').sum().sum())
    dimensions = pd.read_csv(reference / 'feature_dimensions.csv')
    dimension_col = next(c for c in dimensions if c in ('features', 'feature_dimension', 'dimension', 'feature_count', 'features_per_image'))
    feature_bytes = float(dimensions[dimension_col].sum()*342*4)
    wall = (prep + classify/max(1, config['n_jobs']))/3600
    return {'stage': 'nested_cv_grouped' if grouped else 'nested_cv',
            'configurations_per_outer_fold': len(experiments), 'inner_configuration_attempts': len(experiments)*inner,
            'outer_fits_upper_bound': config['outer_folds']*4, 'serial_compute_hours': serial/3600, 'preprocessing_hours': prep/3600,
            'classifier_hours': classify/3600,
            'relief_cache_policy': 'B single-CNN/C paired ranking once per fold; D shares B rankings; exclude infeasible PCA/UMAP250/300 from timing',
            'relieff_reference_jobs': reference_jobs, 'relieff_requested_jobs': config['relieff_n_jobs'],
            'time_hours_range': [wall*.6, wall*1.8], 'feature_extraction_hours_if_missing': extraction/3600,
            'additional_disk_gib_range': [(feature_bytes+inner*len(experiments)*3*4096)/2**30,
                                          (feature_bytes*1.5+inner*len(experiments)*3*8192)/2**30],
            'small_file_count_upper_bound': inner*len(experiments)*3,
            'memory_budget_gib': config['memory_budget_gib'],
            'basis': 'Recorded full_search timing; shared Relief rankings once per fold, 0.64 quadratic Relief and0.8 other inner sample scale. Preprocessing serial, classifier jobs parallel. Disk includes4–8KiB allocation per atomic job file; caches/UMAP/group folds add uncertainty.',
            'input_hashes': {'per_fold_metrics': sha256(reference/'per_fold_metrics.csv'),
                             'feature_dimensions': sha256(reference/'feature_dimensions.csv')}}


def external_estimate(config, stage):
    costs_path = ROOT / config['fine_tuning_results'] / 'computational_costs.csv'
    costs = pd.read_csv(costs_path)
    ft = costs[costs.family.eq('fine_tuning')]
    duration = 'training_seconds' if 'training_seconds' in ft else 'train_seconds'
    # Final model train ~80% of342 versus the matched80% of outer train (~308).
    base_seconds = float(ft.groupby('architecture')[duration].mean().sum())*3*(342/308)
    fixed = costs[costs.family.eq('fixed') | costs.family.eq('fixed_features')]
    fixed_seconds = float(fixed[duration].sum()/10) if len(fixed) else 0.
    n = config['expected_exact_clean']['total']
    report = ROOT / config['cleaning_directory'] / 'cleaning_report.json'
    if report.exists():
        review = json.loads(report.read_text())
        n = review.get('reviewed_principal_images', review.get('main_external_images_pending_approval', n))
    latency_cols = [c for c in costs if 'gpu_latency_median_ms' in c]
    latency = float(costs[latency_cols[0]].dropna().mean())/1000 if latency_cols else .05
    if stage == 'final-models':
        seconds, disk, jobs = base_seconds+fixed_seconds, [3., 12.], 15+4
        basis = 'Measured matched training costs scaled from outer train308 to VIA342; 15 FT jobs +4 fixed, checkpoint/optimizer disk included.'
    elif stage == 'evaluate':
        seconds, disk, jobs = n*19*latency+600, [.02, .15], 19
        basis = 'Recorded end-to-end GPU latency extrapolated to19 model/seed panels; CSV+10k scene-bootstrap/analysis overhead. CPU will take longer.'
    else:
        pool = n/2
        multiplier = sum((342+pool*f)/342 for f in config['learning_curve']['fractions'] if f > 0)
        estimate_all = base_seconds*multiplier/3600
        reduced = estimate_all > config['learning_curve']['reduce_ft_if_hours_above']
        architectures = config['learning_curve']['reduced_architectures'] if reduced else config['architectures']
        if reduced:
            base_seconds = float(ft[ft.architecture.isin(architectures)].groupby('architecture')[duration].mean().sum())*3*(342/308)
        seconds, disk, jobs = (base_seconds+fixed_seconds)*multiplier, [8., 35.], len(architectures)*3*3+4*3
        basis = 'Measured epochs/train cost, linear sample scale, three new sizes; 0% reuses final models. Early stopping uncertainty is material.'
    result = {'stage': stage, 'jobs': jobs, 'external_images_upper_bound': n, 'serial_compute_hours': seconds/3600,
              'time_hours_range': [seconds/3600*.65, seconds/3600*1.8], 'additional_disk_gib_range': disk,
              'basis': basis, 'source_sha256': sha256(costs_path), 'does_not_fit_or_predict': True}
    if stage == 'learning-curve':
        result.update({'ft_architectures': architectures, 'ft_reduced_by_48h_rule': reduced,
                       'unrestricted_ft_estimate_hours': estimate_all})
    return result
