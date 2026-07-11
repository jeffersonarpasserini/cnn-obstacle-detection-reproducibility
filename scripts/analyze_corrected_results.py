#!/usr/bin/env python3
"""Generate manuscript-oriented analyses from a completed corrected run."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Completed output directory")
    parser.add_argument(
        "--output", default=None, help="Analysis directory (default: RESULTS/article_analysis)"
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Top median-accuracy configurations used in exploratory tests",
    )
    return parser.parse_args()


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def pooled_metrics(records):
    rows = []
    for experiment, group in records.groupby("experiment", sort=False):
        tn = int(group["tn_obstructed"].sum())
        fp = int(group["fp_obstructed_as_clear"].sum())
        fn = int(group["fn_clear_as_obstructed"].sum())
        tp = int(group["tp_clear"].sum())
        total = tn + fp + fn + tp
        rows.append(
            {
                "experiment": experiment,
                "tn_obstructed": tn,
                "fp_obstructed_as_clear": fp,
                "fn_clear_as_obstructed": fn,
                "tp_clear": tp,
                "pooled_accuracy": safe_ratio(tn + tp, total),
                "pooled_obstructed_recall": safe_ratio(tn, tn + fp),
                "pooled_clear_recall": safe_ratio(tp, tp + fn),
                "pooled_obstructed_precision": safe_ratio(tn, tn + fn),
                "pooled_clear_precision": safe_ratio(tp, tp + fp),
            }
        )
    return pd.DataFrame(rows)


def holm_adjust(p_values):
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for position, original_index in enumerate(order):
        candidate = min(1.0, (count - position) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def paired_tests(records, selected_names):
    accuracy = records.pivot(index="fold", columns="experiment", values="accuracy")
    accuracy = accuracy[selected_names].dropna()
    friedman = {
        "exploratory_warning": (
            "Models were selected by performance on these folds; inferential "
            "p-values are descriptive and require independent confirmation."
        ),
        "models": list(accuracy.columns),
        "folds": len(accuracy),
        "statistic": None,
        "p_value": None,
    }
    if accuracy.shape[1] >= 3 and len(accuracy) >= 2:
        result = friedmanchisquare(
            *(accuracy[column].to_numpy() for column in accuracy.columns)
        )
        friedman["statistic"] = float(result.statistic)
        friedman["p_value"] = float(result.pvalue)

    pair_rows = []
    for first, second in combinations(accuracy.columns, 2):
        differences = accuracy[first] - accuracy[second]
        try:
            result = wilcoxon(
                accuracy[first], accuracy[second], zero_method="pratt",
                alternative="two-sided",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        except ValueError:
            statistic = 0.0
            p_value = 1.0
        pair_rows.append(
            {
                "model_1": first,
                "model_2": second,
                "median_accuracy_difference": float(np.median(differences)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs["p_holm"] = holm_adjust(pairs["p_value"].to_numpy())
        pairs["significant_0_05"] = pairs["p_holm"] < 0.05
    return friedman, pairs


def main():
    args = parse_args()
    results_dir = Path(args.results)
    records_path = results_dir / "per_fold_metrics.csv"
    summary_path = results_dir / "summary_metrics.csv"
    if not records_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Completed per-fold and summary files are required")

    records = pd.read_csv(records_path)
    summary = pd.read_csv(summary_path)
    required = {
        "experiment", "fold", "accuracy", "balanced_accuracy",
        "obstructed_recall", "clear_recall", "tn_obstructed",
        "fp_obstructed_as_clear", "fn_clear_as_obstructed", "tp_clear",
    }
    missing = required - set(records.columns)
    if missing:
        raise ValueError(
            "Results predate the extended scientific metrics; rerun with the "
            f"current pipeline. Missing columns: {sorted(missing)}"
        )

    output_dir = Path(args.output) if args.output else results_dir / "article_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled = pooled_metrics(records)
    table = summary.merge(pooled, on="experiment", how="left")
    table = table.sort_values(
        ["accuracy_median", "balanced_accuracy_median", "obstructed_recall_median"],
        ascending=False,
    ).reset_index(drop=True)
    table.insert(0, "accuracy_rank", np.arange(1, len(table) + 1))
    table.to_csv(output_dir / "article_model_summary.csv", index=False)

    selected_names = table.head(min(args.top, len(table)))["experiment"].tolist()
    friedman, pairs = paired_tests(records, selected_names)
    (output_dir / "friedman_top_models.json").write_text(
        json.dumps(friedman, indent=2) + "\n", encoding="utf-8"
    )
    pairs.to_csv(output_dir / "pairwise_wilcoxon_holm.csv", index=False)

    print(f"Analyzed {len(table)} configurations across {records['fold'].nunique()} folds")
    print(f"Outputs written to {output_dir}")
    print(
        table[
            [
                "accuracy_rank", "experiment", "accuracy_median",
                "pooled_accuracy", "pooled_obstructed_recall",
                "pooled_clear_recall",
            ]
        ].head(args.top).to_string(index=False)
    )


if __name__ == "__main__":
    main()
