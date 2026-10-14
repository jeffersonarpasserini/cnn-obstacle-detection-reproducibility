#!/usr/bin/env python3
"""Build manuscript tables from a completed full-search run.

The selection procedure mirrors the hierarchy described in the manuscript:

* Approach A: rank all configurations across the ten paired folds.
* Approach B: select one configuration per CNN, then rank the CNN finalists.
* Approaches C/D: select one configuration per CNN pair, then rank the pair
  finalists.

Accuracy is the endpoint used for Friedman ranks. Quartiles and the additional
safety-oriented metrics are descriptive. Because selection and evaluation use
the same folds, every inferential output produced here remains exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


APPROACH_TABLE_SIZES = {"A": 20, "B": 10, "C": 9, "D": 9}
NEMENYI_Q_ALPHA_005 = {4: 2.569}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Completed full-search directory")
    parser.add_argument("--config", required=True, help="Full-search JSON configuration")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: RESULTS/manuscript_tables)",
    )
    parser.add_argument(
        "--selected-config-dir",
        default=None,
        help="Directory for generated selected-model configurations",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.selection import (average_friedman_ranks, pooled_metrics, experiment_label,
                           enrich_table, choose_winner, select_hierarchically, select_procedures)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for position, original_index in enumerate(order):
        candidate = min(1.0, (count - position) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def selected_comparison(records: pd.DataFrame, names: list[str]):
    matrix = records[records["experiment"].isin(names)].pivot(
        index="fold", columns="experiment", values="accuracy"
    )[names]
    ranks = matrix.rank(axis=1, method="average", ascending=True).mean(axis=0)
    friedman = friedmanchisquare(*(matrix[name].to_numpy() for name in names))

    pairs = []
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            try:
                test = wilcoxon(
                    matrix[first], matrix[second], zero_method="pratt", method="auto"
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            except ValueError:
                statistic = 0.0
                p_value = 1.0
            pairs.append(
                {
                    "model_1": first,
                    "model_2": second,
                    "median_accuracy_difference": float(
                        np.median(matrix[first] - matrix[second])
                    ),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    pair_table = pd.DataFrame(pairs)
    pair_table["p_holm"] = holm_adjust(pair_table["p_value"].to_numpy())
    pair_table["significant_0_05"] = pair_table["p_holm"] < 0.05

    k = len(names)
    n = len(matrix)
    q = NEMENYI_Q_ALPHA_005.get(k)
    cd = float(q * np.sqrt(k * (k + 1) / (6 * n))) if q else None
    result = {
        "scope": "four fixed-feature approach finalists only",
        "exploratory_warning": (
            "The finalists were selected on these same folds. P-values and the "
            "Nemenyi critical difference are descriptive, not confirmatory."
        ),
        "models": names,
        "folds": n,
        "friedman_statistic": float(friedman.statistic),
        "friedman_p_value": float(friedman.pvalue),
        "average_ranks": {name: float(ranks[name]) for name in names},
        "nemenyi_q_alpha_0_05": q,
        "nemenyi_critical_difference": cd,
    }
    return result, pair_table


def latex_escape(text: str) -> str:
    return (
        text.replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def write_latex_rows(table: pd.DataFrame, destination: Path, limit: int):
    lines = [
        "% Generated automatically from the full-search results.",
        "% Columns: model, average rank, Q1, median, Q3, IQR, obstructed recall.",
    ]
    for _, row in table.head(limit).iterrows():
        lines.append(
            f"{latex_escape(row['model'])} & {row['average_friedman_rank']:.2f} "
            f"& {row['accuracy_q1']:.4f} & {row['accuracy_median']:.4f} "
            f"& {row['accuracy_q3']:.4f} & {row['accuracy_iqr']:.4f} "
            f"& {row['pooled_obstructed_recall']:.4f} \\\\" 
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selected_config(
    destination: Path,
    base_config: dict,
    selected_names: list[str],
    protocol: str,
):
    selected = [
        item for item in base_config["experiments"] if item["name"] in selected_names
    ]
    by_name = {item["name"]: item for item in selected}
    payload = {
        "seed": base_config["seed"],
        "n_splits": base_config.get("n_splits", 10),
        "protocol": protocol,
        "selection_notice": (
            "Models selected hierarchically from the completed 10-fold "
            "search. LOOCV on the same dataset assesses stability but is not an "
            "independent confirmation of model selection."
        ),
        "experiments": [by_name[name] for name in selected_names],
    }
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    results_dir = Path(args.results)
    config_path = Path(args.config)
    output_dir = Path(args.output) if args.output else results_dir / "manuscript_tables"
    config_dir = (
        Path(args.selected_config_dir)
        if args.selected_config_dir
        else config_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((results_dir / "run_metadata.json").read_text())
    if not metadata.get("completed"):
        raise RuntimeError("Refusing to tabulate an incomplete run")
    records = pd.read_csv(results_dir / "per_fold_metrics.csv")
    summaries = pd.read_csv(results_dir / "summary_metrics.csv")
    base_config = json.loads(config_path.read_text())
    experiment_map = {item["name"]: item for item in base_config["experiments"]}
    expected = len(experiment_map) * int(metadata["actual_folds"])
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} per-fold rows, found {len(records)}")
    if records.duplicated(["experiment", "fold"]).any():
        raise RuntimeError("Duplicate experiment/fold rows detected")

    selection = select_procedures(records, base_config, summaries)
    final_tables, stage_one_tables = selection["final_tables"], selection["stage_one"]
    selected_names = list(selection["selected"].values())
    for approach in "ABCD":
        final_table, stage_one = final_tables[approach], stage_one_tables[approach]
        final_table.to_csv(output_dir / f"approach_{approach.lower()}_ranking.csv", index=False)
        stage_one.to_csv(output_dir / f"approach_{approach.lower()}_stage_one_winners.csv", index=False)
        write_latex_rows(final_table, output_dir / f"approach_{approach.lower()}_rows.tex", APPROACH_TABLE_SIZES[approach])
    selected_table = selection["selected_table"]
    selected_table.to_csv(output_dir / "selected_approaches.csv", index=False)
    write_latex_rows(
        selected_table, output_dir / "selected_approaches_rows.tex", len(selected_table)
    )

    comparison, pairs = selected_comparison(records, selected_names)
    (output_dir / "selected_approaches_statistics.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pairs.to_csv(output_dir / "selected_approaches_wilcoxon_holm.csv", index=False)

    selected_records = records[records["experiment"].isin(selected_names)].copy()
    selected_records.to_csv(output_dir / "selected_approaches_per_fold.csv", index=False)

    write_selected_config(
        config_dir / "article_selected_models_10fold.json",
        base_config,
        selected_names,
        "stratified_kfold",
    )
    write_selected_config(
        config_dir / "article_selected_models_loocv.json",
        base_config,
        selected_names,
        "loocv",
    )

    manifest = {
        "source_results": str(results_dir),
        "source_per_fold_sha256": sha256(results_dir / "per_fold_metrics.csv"),
        "source_config": str(config_path),
        "source_config_sha256": sha256(config_path),
        "records": len(records),
        "folds": int(records["fold"].nunique()),
        "selection_endpoint": "per-fold accuracy",
        "selection_method": "hierarchical average Friedman rank",
        "tie_breakers": [
            "median accuracy",
            "pooled out-of-fold accuracy",
            "pooled obstructed-path recall",
            "experiment identifier",
        ],
        "selected_models_by_approach": {
            approach: name for approach, name in zip("ABCD", selected_names)
        },
        "reference_baseline_included": False,
        "reference_baseline_notice": (
            "The reference model requires evaluation with results/full_search/fold_assignments.csv before inclusion in a paired Friedman comparison with the four approaches."
        ),
        "selection_scope_notice": (
            "The exhaustive search and model selection use the same ten folds; "
            "the resulting comparisons are exploratory."
        ),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Selected models:")
    for approach, name in zip("ABCD", selected_names):
        row = final_tables[approach].iloc[0]
        print(
            f"  {approach}: {name} | rank={row['average_friedman_rank']:.2f} "
            f"| median={row['accuracy_median']:.4f} "
            f"| pooled obstacle recall={row['pooled_obstructed_recall']:.4f}"
        )
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
