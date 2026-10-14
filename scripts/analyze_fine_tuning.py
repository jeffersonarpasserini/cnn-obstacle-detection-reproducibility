#!/usr/bin/env python3
"""Analyze matched VIA fine-tuning runs without changing canonical full-search data."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_article_followup import correct_mask, read_predictions
from build_manuscript_tables import holm_adjust, latex_escape, sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artifact import prediction_metrics
from fine_tuning import inner_split

ENDPOINTS = ("accuracy", "obstructed_recall")
BOOTSTRAP_ENDPOINTS = ("accuracy", "mcc", "obstructed_recall")
CAVEAT = (
    "Exploratory: fixed finalists were selected on the same folds. Neither these "
    "paired tests nor an image bootstrap provide independent confirmation; "
    "overlapping training sets and model selection limit inferential interpretation."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/fine_tuning"))
    parser.add_argument("--fixed-results", type=Path, default=Path("results/full_search"))
    parser.add_argument("--fixed-config", type=Path,
                        default=Path("configs/article_selected_models_10fold.json"))
    parser.add_argument("--fixed-predictions", type=Path,
                        default=Path("results/article_selected_10fold"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None,
                        help="Prespecified fine config; defaults to run_metadata config_path")
    parser.add_argument("--nested-predictions", type=Path, default=None,
                        help="External nested CSV: filename, fold, procedure (A/B/C/D/winner), "
                             "selected_configuration, true_label, score_clear, predicted_label "
                             "(or predicted_label_default); optional predicted_label_safety")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1980)
    return parser.parse_args()


def wilson_interval(successes: int, total: int, confidence: float = 0.95):
    """Wilson score interval, including deterministic empty-population handling."""
    if total < 0 or successes < 0 or successes > total or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts or confidence")
    if total == 0:
        return np.nan, np.nan
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / total
                         + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def pooled_safety_metrics(truth, predicted):
    values = prediction_metrics(truth, predicted)
    obstructed = values["tn_obstructed"] + values["fp_obstructed_as_clear"]
    recall_low, recall_high = wilson_interval(values["tn_obstructed"], obstructed)
    fnr_low, fnr_high = wilson_interval(values["fp_obstructed_as_clear"], obstructed)
    return {
        **{key: values[key] for key in [
            "accuracy", "mcc", "obstructed_recall", "obstructed_false_negative_rate",
            "tn_obstructed", "fp_obstructed_as_clear", "fn_clear_as_obstructed", "tp_clear",
        ]},
        "obstructed_recall_wilson_low": recall_low,
        "obstructed_recall_wilson_high": recall_high,
        "obstructed_fnr_wilson_low": fnr_low, "obstructed_fnr_wilson_high": fnr_high,
    }


def paired_wilcoxon(differences):
    """Two-sided signed-rank test with Pratt zero handling; all-zero p=1."""
    differences = np.round(np.asarray(differences, dtype=float), decimals=12)
    if differences.ndim != 1 or not len(differences) or not np.isfinite(differences).all():
        raise ValueError("Expected a nonempty finite paired difference vector")
    if not np.any(differences):
        return 0.0, 1.0
    result = wilcoxon(differences, zero_method="pratt", alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def mcnemar_exact(truth, first, second):
    """Exact two-sided binomial McNemar test on paired image correctness."""
    truth, first, second = map(np.asarray, (truth, first, second))
    if truth.ndim != 1 or first.shape != truth.shape or second.shape != truth.shape:
        raise ValueError("McNemar inputs must be aligned one-dimensional vectors")
    if not all(np.isin(values, [0, 1]).all() for values in (truth, first, second)):
        raise ValueError("McNemar inputs must contain binary labels")
    first_only = int(np.sum((first == truth) & (second != truth)))
    second_only = int(np.sum((first != truth) & (second == truth)))
    discordant = first_only + second_only
    p_value = float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
    return first_only, second_only, p_value


def _array_metrics(truth, predicted):
    """Vectorized accuracy/MCC/obstructed recall for image bootstrap batches."""
    tn = np.sum((truth == 0) & (predicted == 0), axis=-1)
    fp = np.sum((truth == 0) & (predicted == 1), axis=-1)
    fn = np.sum((truth == 1) & (predicted == 0), axis=-1)
    tp = np.sum((truth == 1) & (predicted == 1), axis=-1)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), dtype=float)
    mcc = np.divide(tp * tn - fp * fn, denominator,
                    out=np.zeros_like(denominator, dtype=float), where=denominator != 0)
    return np.stack(((tn + tp) / truth.shape[-1], mcc, tn / (tn + fp)), axis=-1)


def bootstrap(truth, predictions: dict, pairs=(), repetitions=10000, seed=1980):
    """Stratified IMAGE bootstrap; identical sampled indices for every paired model.

    Returns model and paired percentile 95% interval tables. Input may be one
    label vector or a seed-by-image matrix per model. For seed matrices, each
    replicate averages the per-seed statistics, not predictions or observations.
    """
    truth = np.asarray(truth)
    names = list(predictions)
    if truth.ndim != 1 or not len(truth) or not np.isin(truth, [0, 1]).all():
        raise ValueError("Bootstrap truth must be a nonempty binary image vector")
    if repetitions < 1 or not names:
        raise ValueError("Bootstrap needs models and at least one replicate")
    arrays = [np.atleast_2d(np.asarray(predictions[name])) for name in names]
    if any(values.ndim != 2 or not values.shape[0] or values.shape[1] != len(truth)
           or not np.isin(values, [0, 1]).all() for values in arrays):
        raise ValueError("Bootstrap predictions must be aligned binary image vectors or seed matrices")
    offsets = np.cumsum([0] + [len(values) for values in arrays])
    seed_slices = [slice(start, stop) for start, stop in zip(offsets[:-1], offsets[1:])]
    matrix = np.concatenate(arrays, axis=0)
    strata = [np.flatnonzero(truth == label) for label in (0, 1)]
    if any(not len(indices) for indices in strata):
        raise ValueError("Stratified bootstrap requires both classes")
    pairs = list(pairs)
    if any(first not in names or second not in names for first, second in pairs):
        raise ValueError("Bootstrap pair contains an unknown model")
    rng = np.random.default_rng(seed)
    samples = np.empty((repetitions, len(names), len(BOOTSTRAP_ENDPOINTS)))
    for start in range(0, repetitions, 200):
        count = min(200, repetitions - start)
        indices = np.concatenate([rng.choice(group, (count, len(group)), replace=True)
                                  for group in strata], axis=1)
        values = _array_metrics(truth[indices][None, :, :], matrix[:, indices])
        values = np.stack([values[indices].mean(axis=0) for indices in seed_slices])
        samples[start:start + count] = values.transpose(1, 0, 2)
    point = _array_metrics(truth[None, :], matrix)
    point = np.stack([point[indices].mean(axis=0) for indices in seed_slices])
    rows, pair_rows = [], []
    for index, name in enumerate(names):
        for endpoint_index, endpoint in enumerate(BOOTSTRAP_ENDPOINTS):
            low, high = np.quantile(samples[:, index, endpoint_index], [0.025, 0.975])
            rows.append(dict(experiment=name, endpoint=endpoint,
                             estimate=float(point[index, endpoint_index]),
                             ci_low=float(low), ci_high=float(high)))
    for first, second in pairs:
        i, j = names.index(first), names.index(second)
        for endpoint_index, endpoint in enumerate(BOOTSTRAP_ENDPOINTS):
            low, high = np.quantile(samples[:, i, endpoint_index]
                                    - samples[:, j, endpoint_index], [0.025, 0.975])
            pair_rows.append(dict(fine_experiment=first, fixed_experiment=second,
                                  endpoint=endpoint,
                                  difference=float(point[i, endpoint_index] - point[j, endpoint_index]),
                                  ci_low=float(low), ci_high=float(high)))
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def validate_substitution_config(metadata, config_path=None):
    """Refuse post-start config drift; return the prespecified margin in pp."""
    path = Path(config_path or metadata.get("config_path", ""))
    if not path.is_file() or sha256(path) != metadata.get("config_sha256"):
        raise ValueError("Prespecified substitution config hash mismatch or missing config")
    config = json.loads(path.read_text(encoding="utf-8"))
    effective = metadata.get("config", {})
    margin = config.get("substitution_margin_pp")
    metrics = config.get("substitution_metrics")
    if (isinstance(margin, bool) or not isinstance(margin, (int, float))
            or not np.isfinite(margin) or not 0 < margin < 100
            or not isinstance(metrics, list) or len(metrics) != 2
            or set(metrics) != set(ENDPOINTS)):
        raise ValueError("Invalid prespecified substitution margin or metrics")
    for field in ("substitution_margin_pp", "substitution_metrics"):
        if metadata.get(field) != config[field] or effective.get(field) != config[field]:
            raise ValueError(f"Prespecified substitution {field} drift")
    timestamp = metadata.get("substitution_prespecified_utc")
    if not timestamp or timestamp != metadata.get("started_utc"):
        raise ValueError("Missing or changed substitution prespecification timestamp")
    return float(margin)


def substitution_analysis(paired_intervals, fine_means, fixed_records, margin_pp):
    """Transform the shared image bootstrap and add paired-fold t sensitivity."""
    if not np.isfinite(margin_pp) or not 0 < margin_pp < 100:
        raise ValueError("Substitution margin must be finite and in (0,100) pp")
    margin = margin_pp / 100
    rows = []
    for record in paired_intervals.itertuples(index=False):
        if record.endpoint not in ENDPOINTS:
            continue
        fine = fine_means[fine_means["experiment"] == record.fine_experiment].set_index("fold")
        fixed = fixed_records[fixed_records["experiment"] == record.fixed_experiment].set_index("fold")
        fine, fixed = fine.sort_index(), fixed.sort_index()
        if (len(fine) != 10 or fine.index.has_duplicates or fixed.index.has_duplicates
                or not fine.index.equals(fixed.index) or set(fine.index) != set(range(1, 11))):
            raise ValueError("Substitution sensitivity requires ten identical complete folds")
        delta = (fixed[record.endpoint] - fine[record.endpoint]).to_numpy(dtype=float)
        if not np.isfinite(delta).all():
            raise ValueError("Substitution fold metrics must be finite")
        mean = float(delta.mean())
        radius = float(t.ppf(0.975, 9) * delta.std(ddof=1) / np.sqrt(10))
        endpoints = [record.endpoint]
        if record.endpoint == "obstructed_recall":
            endpoints.append("obstructed_false_negative_rate")
        for endpoint in endpoints:
            fnr = endpoint == "obstructed_false_negative_rate"
            difference, low, high = -record.difference, -record.ci_high, -record.ci_low
            fold_mean, fold_low, fold_high = mean, mean - radius, mean + radius
            if fnr:
                difference, low, high = -difference, -high, -low
                fold_mean, fold_low, fold_high = -fold_mean, -fold_high, -fold_low

            def decision(lower, upper):
                if fnr:
                    return "supported" if upper < margin else "not_supported" if lower > margin else "inconclusive"
                return "supported" if lower > -margin else "not_supported" if upper < -margin else "inconclusive"

            row = dict(fine_experiment=record.fine_experiment, fixed_experiment=record.fixed_experiment,
                       endpoint=endpoint, difference=difference, ci_low=low, ci_high=high,
                       margin_pp=margin_pp, boundary=margin if fnr else -margin,
                       decision=decision(low, high), fold_difference=fold_mean,
                       fold_ci_low=fold_low, fold_ci_high=fold_high,
                       fold_decision=decision(fold_low, fold_high), folds=10, fold_df=9,
                       units="proportion; *_pp columns in percentage points",
                       image_aggregation="mean per-seed pooled OOF metrics; paired class-stratified images",
                       fold_aggregation="fixed minus within-fold mean single-seed metric; paired t sensitivity",
                       interval="two-sided 95%; image percentile and fold paired t",
                       caveat="Conditional on fitted OOF predictions; overlapping training folds limit inference")
            for field in ("difference", "ci_low", "ci_high", "fold_difference", "fold_ci_low", "fold_ci_high"):
                row[field + "_pp"] = row[field] * 100
            rows.append(row)
    return pd.DataFrame(rows)


def validate_nested_predictions(predictions, assignments):
    """Normalize external performance only; never infer procedure training costs."""
    validate_assignments(assignments)
    frame = predictions.copy()
    source = "nested external predictions"
    _require_columns(frame, ["filename", "fold", "procedure", "selected_configuration",
                             "true_label", "score_clear"], source)
    label_column = "predicted_label" if "predicted_label" in frame else "predicted_label_default"
    _require_columns(frame, [label_column], source)
    if (frame[["filename", "procedure", "selected_configuration"]].isna().any().any()
            or frame["selected_configuration"].astype(str).str.strip().eq("").any()
            or not set(frame["procedure"]).issubset({"A", "B", "C", "D", "winner"})
            or frame.empty):
        raise ValueError(f"{source}: invalid procedure or selected configuration")
    columns = ["fold", "true_label", label_column]
    if label_column == "predicted_label" and "predicted_label_default" in frame:
        columns.append("predicted_label_default")
    if "predicted_label_safety" in frame:
        columns.append("predicted_label_safety")
    _integer_columns(frame, columns + (["sample_index"] if "sample_index" in frame else []), source)
    if ("predicted_label" in frame and "predicted_label_default" in frame
            and not frame["predicted_label"].eq(frame["predicted_label_default"]).all()):
        raise ValueError(f"{source}: conflicting default label columns")
    identity = assignments.set_index("filename")["sample_index"]
    expected_index = frame["filename"].map(identity)
    if expected_index.isna().any():
        raise ValueError(f"{source}: foreign image filename")
    if "sample_index" in frame and not frame["sample_index"].eq(expected_index).all():
        raise ValueError(f"{source}: sample_index does not match canonical filename")
    frame["sample_index"] = expected_index.astype(np.int64)
    if frame.duplicated(["procedure", "filename"]).any():
        raise ValueError(f"{source}: duplicate procedure/image")
    if not frame.groupby(["procedure", "fold"])["selected_configuration"].nunique().eq(1).all():
        raise ValueError(f"{source}: selected configuration changes within procedure/fold")
    variants = []
    for variant, column in [("default", label_column), ("safety", "predicted_label_safety")]:
        if column not in frame:
            continue
        selected = frame.copy()
        selected["experiment"] = "nested_" + selected["procedure"] + "_" + variant
        selected["predicted_label"] = selected[column]
        selected["correct"] = selected["predicted_label"] == selected["true_label"]
        variants.append(selected)
    normalized = pd.concat(variants, ignore_index=True)
    names = sorted(normalized["experiment"].unique())
    validate_predictions(normalized, assignments, names, source=source)
    rows = []
    for (name, fold), group in normalized.groupby(["experiment", "fold"]):
        rows.append(dict(experiment=name, fold=int(fold), train_size=len(assignments) - len(group),
                         test_size=len(group), selected_configuration=group["selected_configuration"].iloc[0],
                         **prediction_metrics(group["true_label"], group["predicted_label"], group["score_clear"])))
    return normalized, pd.DataFrame(rows)


def training_cost_table(costs, fields):
    """Mean seeds within each fold, then median/IQR of actual recorded costs."""
    rows = []
    for name, group in costs.groupby("experiment", sort=True):
        fold_means = group.groupby("fold")[fields].mean()
        row = dict(experiment=name, family=group["family"].iloc[0], folds=len(fold_means),
                   seeds=group["seed"].nunique(),
                   source="measured current fixed finalist" if group["family"].iloc[0] == "fixed_features"
                   else "measured prescribed neural training",
                   aggregation="seed means within fold; median/Q1/Q3/IQR across ten folds",
                   scope="training excludes download/save; fixed adaptation cached outer-train PCA/classifier refit; neural adaptation full fresh prescribed inner-fit retraining",
                   adaptation_timing_source=group["update_timing_source"].iloc[0])
        for field in fields:
            q1, median, q3 = fold_means[field].quantile([0.25, 0.5, 0.75])
            row.update({f"{field}_median": median, f"{field}_q1": q1,
                        f"{field}_q3": q3, f"{field}_iqr": q3 - q1,
                        f"{field}_measured_folds": int(fold_means[field].notna().sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def _require_columns(frame, columns, source):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")


def _integer_columns(frame, columns, source):
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{source}: nonintegral {column}")
        frame[column] = values.astype(np.int64)


def validate_assignments(assignments):
    _require_columns(assignments, ["sample_index", "filename", "label", "test_fold"], "fold assignments")
    _integer_columns(assignments, ["sample_index", "label", "test_fold"], "fold assignments")
    if (len(assignments) != 342 or assignments["sample_index"].duplicated().any()
            or assignments["filename"].isna().any() or assignments["filename"].duplicated().any()
            or set(assignments["sample_index"]) != set(range(342))
            or assignments["label"].value_counts().to_dict() != {0: 167, 1: 175}
            or set(assignments["test_fold"]) != set(range(1, 11))):
        raise ValueError("Expected VIA's 342 unique images, 167 obstructed/175 clear, and ten CSV folds")


def validate_predictions(predictions, assignments, experiments, seeds=None, source="predictions"):
    keys = ["experiment"] + (["seed"] if seeds is not None else [])
    _require_columns(predictions, keys + ["sample_index", "filename", "fold", "true_label",
                                         "predicted_label", "score_clear", "correct"], source)
    _integer_columns(predictions, ["sample_index", "fold", "true_label", "predicted_label"]
                     + (["seed"] if seeds is not None else []), source)
    expected_groups = {(name, seed) for name in experiments for seed in seeds[name]} if seeds is not None else {(name,) for name in experiments}
    groups = predictions.groupby(keys[0] if len(keys) == 1 else keys, sort=False, dropna=False)
    actual_groups = {key if isinstance(key, tuple) else (key,) for key in groups.groups}
    if actual_groups != expected_groups:
        raise ValueError(f"{source}: missing/foreign model-seed groups; missing={sorted(expected_groups - actual_groups)}, foreign={sorted(actual_groups - expected_groups)}")
    if predictions.duplicated(keys + ["sample_index"]).any():
        raise ValueError(f"{source}: duplicate model/seed/image predictions")
    identity = assignments.set_index("sample_index").sort_index()
    for key, group in groups:
        if set(group["sample_index"]) != set(identity.index):
            raise ValueError(f"{source}: incomplete image coverage for {key}")
        group = group.set_index("sample_index").sort_index()
        for column, expected in [("filename", "filename"), ("true_label", "label"), ("fold", "test_fold")]:
            mismatches = group[column].to_numpy() != identity[expected].to_numpy()
            if mismatches.any():
                bad = identity.index[mismatches].tolist()[:5]
                raise ValueError(f"{source}: {key} {column} mismatch at sample indices {bad}")
    if not predictions["predicted_label"].isin([0, 1]).all():
        raise ValueError(f"{source}: predictions are not binary")
    if not np.array_equal(correct_mask(predictions).to_numpy(),
                          (predictions["true_label"] == predictions["predicted_label"]).to_numpy()):
        raise ValueError(f"{source}: incorrect correctness flags")
    predictions["score_clear"] = pd.to_numeric(predictions["score_clear"], errors="raise")
    if seeds is not None:
        scores = predictions["score_clear"].to_numpy()
        if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise ValueError(f"{source}: fine scores must be finite probabilities in [0,1]")
        if not np.array_equal((scores > 0.5).astype(int), predictions["predicted_label"].to_numpy()):
            raise ValueError(f"{source}: fine label must be argmax (clear only if p>0.5)")
    elif not np.isfinite(predictions["score_clear"]).all():
        raise ValueError(f"{source}: fixed finalist scores must be finite, untransformed native scores")


def validate_fold_metrics(records, predictions, experiments, folds, seeds=None,
                          source="metrics", validation_fraction=0.2):
    keys = ["experiment", "fold"] + (["seed"] if seeds is not None else [])
    metric_names = list(prediction_metrics([0, 1], [0, 1], [0.0, 1.0]))
    # Historical full_search predates the explicit FNR field; it is redundant with recall.
    required_metrics = metric_names if seeds is not None else [name for name in metric_names if name != "obstructed_false_negative_rate"]
    size_fields = ["train_size", "test_size"]
    if seeds is not None:
        if not 0 < validation_fraction < 1:
            raise ValueError("Fine validation fraction must be in (0,1)")
        size_fields += ["outer_train_size", "validation_size"]
    _require_columns(records, keys + size_fields + required_metrics, source)
    _integer_columns(records, ["fold"] + size_fields + (["seed"] if seeds is not None else []), source)
    expected = {(name, fold, seed) for name in experiments for fold in folds for seed in seeds[name]} if seeds is not None else set(product(experiments, folds))
    actual = set(records[keys].itertuples(index=False, name=None))
    if records.duplicated(keys).any() or actual != expected:
        raise ValueError(f"{source}: missing/duplicate/foreign fold records; missing={sorted(expected - actual)}, foreign={sorted(actual - expected)}")
    indexed = records.set_index(keys)
    discrepancies = []
    for key, group in predictions.groupby(keys, sort=False):
        expected_metrics = prediction_metrics(group["true_label"], group["predicted_label"], group["score_clear"])
        row = indexed.loc[key]
        for metric in required_metrics:
            if not np.isclose(float(row[metric]), expected_metrics[metric], atol=1e-10, rtol=1e-8, equal_nan=True):
                discrepancies.append(f"{key}: {metric} canonical/stored={row[metric]!r}, recalculated={expected_metrics[metric]!r}")
        outer_size = 342 - len(group)
        if seeds is None:
            sizes_match = row["test_size"] == len(group) and row["train_size"] == outer_size
        else:
            validation_size = int(np.ceil(outer_size * validation_fraction))
            sizes_match = (row["test_size"] == len(group)
                           and row["outer_train_size"] == outer_size
                           and row["validation_size"] == validation_size
                           and row["train_size"] == outer_size - validation_size)
        if not sizes_match:
            discrepancies.append(f"{key}: train/validation/test sizes disagree with assigned split")
    if discrepancies:
        raise ValueError(f"{source}: prediction metrics disagree; refusing source substitution. " + "; ".join(discrepancies[:20]))


def paired_tests(fine_means, fixed_records, fine_names, fixed_names):
    differences, tests = [], []
    for endpoint in ENDPOINTS:
        fine = fine_means.pivot(index="fold", columns="experiment", values=endpoint)
        fixed = fixed_records.pivot(index="fold", columns="experiment", values=endpoint)
        if not fine.index.equals(fixed.index):
            raise ValueError("Paired tests require identical complete fold sets")
        for first, second in product(fine_names, fixed_names):
            delta = fine[first] - fixed[second]
            statistic, p_value = paired_wilcoxon(delta)
            for fold, value in delta.items():
                differences.append(dict(fine_experiment=first, fixed_experiment=second,
                                        endpoint=endpoint, fold=int(fold), difference=float(value)))
            tests.append(dict(fine_experiment=first, fixed_experiment=second, endpoint=endpoint,
                              folds=len(delta), mean_difference=float(delta.mean()),
                              median_difference=float(delta.median()), wilcoxon_statistic=statistic,
                              p_value=p_value))
    table = pd.DataFrame(tests)
    table["p_holm"] = holm_adjust(table["p_value"].to_numpy())
    table["significant_0_05"] = table["p_holm"] < 0.05
    return pd.DataFrame(differences), table


def aggregate_predictions(predictions):
    keys = ["experiment", "sample_index", "filename", "fold", "true_label"]
    aggregate = predictions.groupby(keys, as_index=False, sort=True).agg(
        score_clear=("score_clear", "mean"), seeds=("seed", "nunique"))
    aggregate["predicted_label"] = (aggregate["score_clear"] > 0.5).astype(int)
    aggregate["correct"] = aggregate["predicted_label"] == aggregate["true_label"]
    aggregate["aggregation"] = "pooled out-of-fold mean-probability seed ensemble"
    return aggregate


def validate_costs(costs, fine_records, fine_names, fixed_names, seeds, folds, fixed_seed, metadata):
    fields = ["training_seconds", "effective_epochs", "batch_size", "peak_gpu_memory_bytes",
              "total_parameters", "trainable_parameters", "model_size_mb",
              "cpu_latency_median_ms", "cpu_latency_p95_ms", "gpu_latency_median_ms", "gpu_latency_p95_ms",
              "feature_forward_training_seconds", "feature_extraction_training_seconds",
              "reduction_seconds", "classifier_training_seconds", "adaptation_seconds",
              "adaptation_reduction_seconds", "adaptation_classifier_seconds", "peak_host_memory_bytes"]
    update_fields = ["update_image_filename", "update_previous_labeled_size", "update_labeled_size",
                     "update_timing_source"]
    _require_columns(costs, ["experiment", "fold", "seed", "family", "architecture", "configuration"]
                     + fields + update_fields, "computational costs")
    _integer_columns(costs, ["fold", "seed"], "computational costs")
    expected = {(name, fold, seed) for name in fine_names for fold in folds for seed in seeds[name]}
    expected |= {(name, fold, fixed_seed) for name in fixed_names for fold in folds}
    actual = set(costs[["experiment", "fold", "seed"]].itertuples(index=False, name=None))
    if costs.duplicated(["experiment", "fold", "seed"]).any() or actual != expected:
        raise ValueError(f"Missing/duplicate/unmatched measured costs; missing={sorted(expected - actual)}, foreign={sorted(actual - expected)}")
    if costs[["family", "architecture", "configuration"]].isna().any().any():
        raise ValueError("Cost model identity fields may not be missing")
    for field in fields:
        costs[field] = pd.to_numeric(costs[field], errors="raise")
        separate = field in {"feature_forward_training_seconds", "feature_extraction_training_seconds",
                             "reduction_seconds", "classifier_training_seconds",
                             "adaptation_reduction_seconds", "adaptation_classifier_seconds"}
        if separate:
            fixed = costs["experiment"].isin(fixed_names)
            valid = ((fixed & np.isfinite(costs[field]) & (costs[field] >= 0))
                     | (~fixed & costs[field].isna()))
        elif field.startswith("gpu_") or field == "peak_gpu_memory_bytes":
            valid = costs[field].isna() | (np.isfinite(costs[field]) & (costs[field] >= 0))
        else:
            valid = np.isfinite(costs[field]) & (costs[field] >= 0)
        if not valid.all():
            raise ValueError(f"Missing or invalid measured cost: {field}")
    if (costs["cpu_latency_p95_ms"] < costs["cpu_latency_median_ms"]).any():
        raise ValueError("CPU latency p95 must not be below median")
    gpu_fields = ["gpu_latency_median_ms", "gpu_latency_p95_ms"]
    if (costs[gpu_fields].isna().any(axis=1) != costs[gpu_fields].isna().all(axis=1)).any():
        raise ValueError("GPU median/p95 availability must match")
    gpu_missing = costs[gpu_fields].isna().all(axis=1)
    if gpu_missing.any() and not gpu_missing.all():
        raise ValueError("GPU costs are missing for some matched jobs; refusing partial-job averages")
    devices = metadata.get("benchmark_devices")
    if (not isinstance(devices, list) or len(set(devices)) != len(devices)
            or set(devices) not in ({"cpu"}, {"cpu", "gpu"})):
        raise ValueError("Metadata must record benchmark_devices as cpu or cpu/gpu")
    if ("gpu" in devices) == bool(gpu_missing.all()):
        raise ValueError("GPU latency availability disagrees with recorded benchmark devices")
    if (costs["gpu_latency_p95_ms"] < costs["gpu_latency_median_ms"]).any():
        raise ValueError("GPU latency p95 must not be below median")
    if ((costs["trainable_parameters"] > costs["total_parameters"]).any()
            or (costs["total_parameters"] <= 0).any() or (costs["model_size_mb"] <= 0).any()
            or (costs["batch_size"] <= 0).any()):
        raise ValueError("Invalid model parameter/size/batch costs")
    if not costs.loc[costs["experiment"].isin(fixed_names), "family"].eq("fixed_features").all():
        raise ValueError("Fixed finalist costs must be labeled fixed_features")
    identities = fine_records.set_index(["experiment", "fold", "seed"])
    for row in costs[costs["experiment"].isin(fine_names)].itertuples(index=False):
        identity = identities.loc[(row.experiment, row.fold, row.seed)]
        for field in ["family", "architecture", "configuration"]:
            if str(getattr(row, field)) != str(identity[field]):
                raise ValueError(f"Unmatched cost identity: {row.experiment}/{row.fold}/{row.seed} {field}")
        if not np.isclose(row.adaptation_seconds, row.training_seconds, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Neural adaptation must be the measured prescribed full retraining: {row.experiment}")
    for name, group in costs.groupby("experiment"):
        if any(group[field].nunique() != 1 for field in ["family", "architecture", "configuration"]):
            raise ValueError(f"Cost identity changes across jobs for {name}")
    _integer_columns(costs, ["update_previous_labeled_size", "update_labeled_size"], "cost update sizes")
    if (costs["update_image_filename"].isna().any()
            or not costs["update_previous_labeled_size"].eq(costs["update_labeled_size"] - 1).all()
            or (costs["update_previous_labeled_size"] < 1).any()):
        raise ValueError("Invalid measured newly labeled image/update sizes")
    for name, group in costs.groupby("experiment"):
        expected_source = ("measured_cached_pca_classifier_refit" if name in fixed_names
                           else "prescribed_full_retraining")
        if not group["update_timing_source"].eq(expected_source).all():
            raise ValueError(f"Invalid adaptation timing source for {name}")
    return fields


def write_table(table, output, name, caveat=CAVEAT):
    table.to_csv(output / f"{name}.csv", index=False)
    lines = ["% Generated automatically from validated matched fine-tuning results.",
             "% " + caveat, "% Columns: " + ", ".join(table.columns) + "."]
    for row in table.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("--")
            elif isinstance(value, (bool, np.bool_)):
                cells.append("yes" if value else "no")
            elif isinstance(value, (int, np.integer)):
                cells.append(str(value))
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.4f}")
            else:
                cells.append(latex_escape(str(value)))
        lines.append(" & ".join(cells) + r" \\")
    (output / f"{name}_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_outputs(aggregate, output, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    figure, axes = plt.subplots(len(names), 1, figsize=(8, 2.6 * len(names)), squeeze=False)
    bins = np.linspace(0, 1, 21)
    groups = [("obstructed predicted clear", 1), ("correct obstructed", 0)]
    for index, name in enumerate(names):
        ax = axes[index, 0]
        obstructed = aggregate[(aggregate["experiment"] == name) & (aggregate["true_label"] == 0)]
        for label, predicted in groups:
            values = obstructed.loc[obstructed["predicted_label"] == predicted, "score_clear"].to_numpy()
            quantiles = np.quantile(values, [0.25, 0.5, 0.75]) if len(values) else [np.nan] * 3
            rows.append(dict(experiment=name, group=label, images=len(values),
                             score_mean=float(np.mean(values)) if len(values) else np.nan,
                             score_q1=quantiles[0], score_median=quantiles[1], score_q3=quantiles[2],
                             aggregation="mean-probability seed ensemble; P(clear)"))
            ax.hist(values, bins=bins, histtype="step", linewidth=1.5,
                    label=f"{label} (n={len(values)})")
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set(xlim=(0, 1), ylabel="Obstructed images", title=name)
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Mean seed P(clear), pooled out-of-fold ensemble")
    figure.tight_layout()
    figure.savefig(output / "obstructed_score_distributions.png", dpi=180)
    figure.savefig(output / "obstructed_score_distributions.pdf")
    plt.close(figure)
    write_table(pd.DataFrame(rows), output, "obstructed_score_distributions")


def _expected_plan(metadata):
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("Fine run metadata must contain its effective configuration object")
    architectures = config.get("architectures")
    configurations = config.get("configurations")
    defaults = config.get("initialization_seeds")
    if (not isinstance(architectures, list) or not architectures
            or not isinstance(configurations, list) or not configurations):
        raise ValueError("Effective configuration must list architectures/configurations")
    names = [f"ft_{configuration.lower()}_{architecture.lower()}"
             for configuration, architecture in product(configurations, architectures)]
    if (len(set(names)) != len(names)
            or len(metadata.get("experiments", [])) != len(names)
            or set(metadata.get("experiments", [])) != set(names)):
        raise ValueError("Metadata experiment plan disagrees with configuration Cartesian product")
    if (not isinstance(defaults, list) or not defaults
            or len(set(defaults)) != len(defaults)
            or any(not isinstance(value, int) for value in defaults)
            or metadata.get("seeds") != defaults):
        raise ValueError("Metadata initialization seeds disagree with effective configuration")
    seeds = {name: defaults for name in names}
    return names, seeds


def main():
    args = parse_args()
    output = args.output or args.results / "manuscript_tables"
    canonical_results = Path(__file__).resolve().parents[1] / "results/full_search"
    for protected in (canonical_results, args.fixed_results, args.fixed_predictions):
        for destination in (output, args.results):
            if destination.resolve() == protected.resolve() or protected.resolve() in destination.resolve().parents:
                raise ValueError("Analysis output may not modify canonical/fixed archived results")
    if args.bootstrap_repetitions != 10000 or args.seed != 1980:
        raise ValueError("Prespecified substitution analysis requires 10000 image replicates and seed 1980")
    inputs = []

    def read_csv(path):
        inputs.append(path)
        return pd.read_csv(path)

    def read_json(path):
        inputs.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    metadata = read_json(args.results / "run_metadata.json")
    margin_pp = validate_substitution_config(metadata, args.config)
    inputs.append(Path(args.config or metadata["config_path"]))
    if metadata.get("completed") is not True or metadata.get("smoke") is not False:
        raise ValueError("Manuscript analysis requires a complete non-smoke fine-tuning run")
    names, seeds = _expected_plan(metadata)
    assignments = read_csv(args.fixed_results / "fold_assignments.csv")
    validate_assignments(assignments)
    folds = sorted(assignments["test_fold"].unique().tolist())
    if metadata.get("folds") != folds or metadata.get("images") != len(assignments):
        raise ValueError("Fine run metadata must match all ten canonical VIA folds and 342 images")
    fixed_config = read_json(args.fixed_config)
    fixed_names = [item["name"] for item in fixed_config["experiments"]]
    if len(fixed_names) != 4 or len(set(fixed_names)) != 4:
        raise ValueError("Expected the four selected fixed-feature finalists")
    measured_fixed_names = fixed_names.copy()
    if args.nested_predictions is not None:
        fixed_predictions, canonical = validate_nested_predictions(read_csv(args.nested_predictions), assignments)
        fixed_names = sorted(fixed_predictions["experiment"].unique())
        fixed_source = "nested external predictions; selected configuration per procedure/fold"
        analysis_caveat = ("Nested external performance, conditional on fitted OOF predictions; "
                           "overlapping training folds limit inference. Current finalist costs "
                           "are not costs of nested-selected procedures.")
    else:
        canonical = read_csv(args.fixed_results / "per_fold_metrics.csv")
        canonical = canonical[canonical["experiment"].isin(fixed_names)].copy()
        archive_metadata = read_json(args.fixed_predictions / "run_metadata.json")
        if archive_metadata.get("completed") is not True or archive_metadata.get("prediction_mode") != "all":
            raise ValueError("Fixed archived predictions must come from a completed all-image run")
        shard_index = read_csv(args.fixed_predictions / "prediction_shard_index.csv")
        inputs.extend(args.fixed_predictions / relative for relative in shard_index["file"])
        fixed_predictions = read_predictions(args.fixed_predictions, archive_metadata)
        validate_predictions(fixed_predictions, assignments, fixed_names, source="fixed archived predictions")
        validate_fold_metrics(canonical, fixed_predictions, fixed_names, folds, source="canonical full_search vs fixed archive")
        fixed_source = "canonical full_search per-fold metrics and matched archived predictions"
        analysis_caveat = CAVEAT
    records = read_csv(args.results / "per_fold_metrics.csv")
    predictions = read_csv(args.results / "per_image_predictions.csv")
    _require_columns(records, ["family", "architecture", "configuration"], "fine fold metrics")
    validate_predictions(predictions, assignments, names, seeds, "fine predictions")
    validate_fold_metrics(records, predictions, names, folds, seeds, "fine fold metrics",
                          float(metadata["config"]["validation_fraction"]))
    for configuration, architecture in product(metadata["config"]["configurations"],
                                               metadata["config"]["architectures"]):
        name = f"ft_{configuration.lower()}_{architecture.lower()}"
        expected_identity = {"architecture": architecture, "configuration": configuration,
                             "family": "frozen_head" if configuration == "A" else "fine_tuning"}
        for frame, source in [(records, "fine metrics"), (predictions, "fine predictions")]:
            _require_columns(frame, expected_identity, source)
            selected = frame[frame["experiment"] == name]
            for field, value in expected_identity.items():
                if not selected[field].eq(value).all():
                    raise ValueError(f"{source}: {name} {field} disagrees with effective configuration")
    costs = read_csv(args.results / "computational_costs.csv")
    cost_fields = validate_costs(costs, records, names, measured_fixed_names, seeds, folds,
                                int(fixed_config.get("seed", 1980)), metadata)
    for experiment in fixed_config["experiments"]:
        selected = costs[costs["experiment"] == experiment["name"]]
        if (not selected["configuration"].eq("fixed").all()
                or not selected["architecture"].eq("+".join(experiment["extractors"])).all()):
            raise ValueError(f"Fixed cost architecture/configuration mismatch: {experiment['name']}")
    image_fold = assignments.set_index("filename")["test_fold"]
    if (costs["update_image_filename"].map(image_fold).isna().any()
            or costs["update_image_filename"].map(image_fold).eq(costs["fold"]).any()
            or not costs.groupby("fold")["update_image_filename"].nunique().eq(1).all()):
        raise ValueError("Adaptation image must be shared across models and outside the outer test fold")
    for fold in folds:
        outer_train = np.sort(assignments.loc[assignments["test_fold"] != fold, "sample_index"].to_numpy())
        fit_train, _ = inner_split(outer_train, assignments.sort_values("sample_index")["label"].to_numpy(),
                                 metadata["config"]["validation_fraction"], metadata["config"]["validation_seed"])
        allowed = set(assignments.loc[assignments["sample_index"].isin(fit_train), "filename"])
        if not costs.loc[costs["fold"] == fold, "update_image_filename"].isin(allowed).all():
            raise ValueError("Newly labeled adaptation image may not come from internal validation")
    fine_sizes = records.set_index(["experiment", "fold", "seed"])["train_size"]
    for row in costs.itertuples(index=False):
        expected_size = (342 - int(assignments["test_fold"].eq(row.fold).sum())
                         if row.experiment in measured_fixed_names
                         else int(fine_sizes.loc[(row.experiment, row.fold, row.seed)]))
        if row.update_labeled_size != expected_size:
            raise ValueError("Adaptation label count disagrees with measured training population")
    metric_names = list(prediction_metrics([0, 1], [0, 1], [0., 1.]))
    means = records.groupby(["experiment", "fold"], as_index=False)[metric_names].mean()
    means["seeds"] = means["experiment"].map({name: len(values) for name, values in seeds.items()})
    means["aggregation"] = "arithmetic mean single-seed metrics; not a seed ensemble"
    aggregate = aggregate_predictions(predictions)
    differences, paired = paired_tests(means, canonical, names, fixed_names)
    for table in (differences, paired):
        table["fine_aggregation"] = "arithmetic mean single-seed metrics per fold"
        table["fixed_source"] = fixed_source
    truth = assignments.sort_values("sample_index")["label"].to_numpy()
    model_predictions = {}
    for frame in (aggregate, fixed_predictions):
        for name, group in frame.groupby("experiment"):
            model_predictions[name] = group.sort_values("sample_index")["predicted_label"].to_numpy()
    pairs = list(product(names, fixed_names))
    mcnemar_rows = []
    for first, second in pairs:
        b, c, p_value = mcnemar_exact(truth, model_predictions[first], model_predictions[second])
        mcnemar_rows.append(dict(fine_experiment=first, fixed_experiment=second,
                                 fine_correct_fixed_wrong=b, fine_wrong_fixed_correct=c,
                                 discordant_images=b + c, p_value=p_value,
                                 fine_aggregation="mean-probability seed ensemble"))
    mcnemar = pd.DataFrame(mcnemar_rows)
    mcnemar["p_holm"] = holm_adjust(mcnemar["p_value"].to_numpy())
    mcnemar["significant_0_05"] = mcnemar["p_holm"] < 0.05
    bootstrap_predictions = {name: model_predictions[name] for name in fixed_names}
    for name in names:
        bootstrap_predictions[name] = np.stack([
            predictions[(predictions["experiment"] == name) & (predictions["seed"] == seed)]
            .sort_values("sample_index")["predicted_label"].to_numpy()
            for seed in seeds[name]
        ])
    intervals, paired_intervals = bootstrap(truth, bootstrap_predictions, pairs,
                                            args.bootstrap_repetitions, args.seed)
    intervals["aggregation"] = intervals["experiment"].map(
        lambda name: "arithmetic mean pooled single-seed OOF metrics" if name in names else fixed_source)
    paired_intervals["fine_aggregation"] = "arithmetic mean pooled single-seed OOF metrics"
    paired_intervals["fixed_source"] = fixed_source
    substitution = substitution_analysis(paired_intervals, means, canonical, margin_pp)
    substitution["fixed_source"] = fixed_source
    substitution["caveat"] = analysis_caveat
    substitution["bootstrap_repetitions"] = 10000
    substitution["bootstrap_seed"] = 1980
    training_cost = training_cost_table(costs, cost_fields)
    pooled_rows = [
        dict(experiment=name, images=len(truth),
             aggregation="mean-probability seed ensemble" if name in names else fixed_source,
             **pooled_safety_metrics(truth, predicted))
        for name, predicted in model_predictions.items()
    ]
    pooled = pd.DataFrame(pooled_rows)
    seed_wilson_rows = []
    for (name, seed), group in predictions.groupby(["experiment", "seed"]):
        group = group.sort_values("sample_index")
        seed_wilson_rows.append(dict(
            experiment=name, seed=int(seed), images=len(group),
            aggregation="pooled OOF metrics for one initialization seed; 167 obstructed images",
            **pooled_safety_metrics(group["true_label"], group["predicted_label"]),
        ))
    seed_wilson = pd.DataFrame(seed_wilson_rows)
    mean_rows = []
    all_folds = pd.concat([means, canonical], ignore_index=True)
    for name, group in all_folds.groupby("experiment", sort=True):
        row = dict(experiment=name, folds=len(group),
                   aggregation="arithmetic mean single-seed metrics per fold" if name in names else fixed_source)
        for endpoint in BOOTSTRAP_ENDPOINTS:
            q1, median, q3 = group[endpoint].quantile([0.25, 0.5, 0.75])
            row.update({f"{endpoint}_mean": group[endpoint].mean(), f"{endpoint}_std": group[endpoint].std(ddof=1),
                        f"{endpoint}_q1": q1, f"{endpoint}_median": median, f"{endpoint}_q3": q3})
        mean_rows.append(row)
    fold_summary = pd.DataFrame(mean_rows)
    seed_summary = records.groupby(["experiment", "seed"], as_index=False)[list(BOOTSTRAP_ENDPOINTS)].mean()
    seed_summary["aggregation"] = "arithmetic mean across ten folds for one initialization seed"
    cost_rows = []
    for name, group in costs.groupby("experiment", sort=True):
        row = dict(experiment=name, family=group.iloc[0]["family"], architecture=group.iloc[0]["architecture"],
                   configuration=group.iloc[0]["configuration"], measured_jobs=len(group),
                   folds=group["fold"].nunique(), seeds=group["seed"].nunique(),
                   training_seconds_total=group["training_seconds"].sum())
        for field in cost_fields:
            # No CPU fallback: unavailable GPU measurements remain NaN.
            row[f"{field}_job_mean"] = group[field].mean()
        cost_rows.append(row)
    cost_summary = pd.DataFrame(cost_rows)
    tradeoff = fold_summary[["experiment", "aggregation", "accuracy_mean", "obstructed_recall_mean"]].merge(
        cost_summary, on="experiment", how="left", validate="one_to_one")
    tradeoff["cost_source"] = tradeoff["experiment"].map(
        lambda name: "unavailable for nested-selected procedure" if name.startswith("nested_")
        else "measured current finalist/neural jobs")
    # All scientific validation precedes writes, especially historical archive disagreements.
    output.mkdir(parents=True, exist_ok=True)
    write_table(substitution, args.results, "substitution_analysis", analysis_caveat)
    write_table(training_cost, args.results, "training_cost",
                "Measured current finalists and neural jobs only; no nested procedure cost attribution.")
    tables = dict(per_fold_seed_mean=means, per_image_seed_aggregate=aggregate,
                  paired_fold_differences=differences, wilcoxon_holm=paired,
                  mcnemar_exact_holm=mcnemar, pooled_metrics_wilson=pooled,
                  initialization_seed_metrics_wilson=seed_wilson,
                  stratified_image_bootstrap=intervals, paired_image_bootstrap=paired_intervals,
                  fold_metric_summary=fold_summary, initialization_seed_summary=seed_summary,
                  computational_cost_summary=cost_summary, accuracy_recall_cost_tradeoff=tradeoff,
                  substitution_analysis=substitution, training_cost=training_cost)
    if args.nested_predictions is not None:
        tables["nested_external_predictions"] = fixed_predictions
    for name, table in tables.items():
        write_table(table, output, name, analysis_caveat)
    score_outputs(aggregate, output, names)
    manifest = dict(
        inputs={str(path): sha256(path) for path in inputs},
        analysis_script_sha256=sha256(Path(__file__)),
        results=str(args.results), fixed_results=str(args.fixed_results), fixed_predictions=str(args.fixed_predictions),
        experiments=names, fixed_experiments=fixed_names, folds=folds, seeds=seeds,
        fine_fixed_pairs=len(pairs),
        performance_source=fixed_source,
        nested_predictions=str(args.nested_predictions) if args.nested_predictions is not None else None,
        measured_fixed_cost_experiments=measured_fixed_names,
        substitution=dict(margin_pp=margin_pp, metrics=list(ENDPOINTS),
                          prespecified_utc=metadata["substitution_prespecified_utc"],
                          difference="fixed minus fine; FNR is negative recall difference",
                          boundary_equality="inconclusive", fold_sensitivity="paired t 95%, df=9"),
        bootstrap=dict(repetitions=args.bootstrap_repetitions, seed=args.seed,
                       unit="image, never initialization seed", strata={"obstructed": 167, "clear": 175},
                       same_sampled_indices_all_models=True, interval="percentile 95%, conditional on fitted OOF predictions",
                       fine_estimand="arithmetic mean of per-seed pooled OOF statistics within every image resample",
                       paired_endpoints=list(BOOTSTRAP_ENDPOINTS)),
        multiplicity=dict(wilcoxon={"family": "all requested fine/control vs fixed pairs, both endpoints together", "tests": len(paired), "method": "Holm"},
                          mcnemar={"family": "all requested fine/control ensemble vs fixed image-correctness pairs", "tests": len(mcnemar), "method": "Holm"},
                          bootstrap="descriptive marginal intervals, not multiplicity-adjusted"),
        aggregation=dict(fold="arithmetic mean of single-initialization metrics within each fold; not an ensemble",
                         training="neural fit train reserves the configured stratified inner-validation fraction of outer train; fixed finalists fit the entire outer train",
                         pooled="mean P(clear) across seeds for each OOF image; clear iff mean>0.5; descriptive ensemble",
                         image_bootstrap="average per-seed pooled OOF metrics; images resampled jointly across seeds/models, never seed/image pseudoreplication",
                         fixed=fixed_source,
                         costs="arithmetic mean of measured per-job costs across folds/seeds, plus total training time; latency quantiles are means of per-job quantiles, not pooled timing quantiles; size is decimal MB; unavailable GPU stays NaN",
                         training_cost="seed means within folds, then median/Q1/Q3/IQR over ten folds; actually benchmarked current finalists only; no nested cost reuse; unavailable components/GPU stay NaN",
                         scores="fine mean-probability ensemble only; fixed SVM margins never converted to probabilities"),
        wilcoxon=dict(alternative="two-sided", zero_method="pratt", all_zero_p=1,
                      difference_rounding_decimals=12),
        wilson=dict(confidence=0.95, population="167 truly obstructed images", obstacle_false_negative="true label 0 predicted label 1"),
        hardware=metadata.get("hardware"), dependencies=metadata.get("dependencies"),
        benchmark_devices=metadata.get("benchmark_devices"),
        exploratory_warning=analysis_caveat,
    )
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(names)} neural models vs {len(fixed_names)} fixed procedures; wrote analysis to {output}")


if __name__ == "__main__":
    main()
