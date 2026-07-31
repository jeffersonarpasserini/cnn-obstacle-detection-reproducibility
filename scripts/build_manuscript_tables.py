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


def average_friedman_ranks(records: pd.DataFrame, names: list[str]) -> pd.Series:
    matrix = records[records["experiment"].isin(names)].pivot(
        index="fold", columns="experiment", values="accuracy"
    )
    matrix = matrix.reindex(columns=names)
    if matrix.isna().any().any():
        raise ValueError("Incomplete experiment/fold matrix during ranking")
    # Larger accuracy receives a larger rank, matching the manuscript tables.
    return matrix.rank(axis=1, method="average", ascending=True).mean(axis=0)


def pooled_metrics(records: pd.DataFrame) -> pd.DataFrame:
    grouped = records.groupby("experiment", as_index=False).agg(
        tn_obstructed=("tn_obstructed", "sum"),
        fp_obstructed_as_clear=("fp_obstructed_as_clear", "sum"),
        fn_clear_as_obstructed=("fn_clear_as_obstructed", "sum"),
        tp_clear=("tp_clear", "sum"),
        training_seconds_total=("training_seconds", "sum"),
        prediction_seconds_total=("prediction_seconds", "sum"),
    )
    total = (
        grouped["tn_obstructed"]
        + grouped["fp_obstructed_as_clear"]
        + grouped["fn_clear_as_obstructed"]
        + grouped["tp_clear"]
    )
    grouped["pooled_accuracy"] = (
        grouped["tn_obstructed"] + grouped["tp_clear"]
    ) / total
    grouped["pooled_obstructed_recall"] = grouped["tn_obstructed"] / (
        grouped["tn_obstructed"] + grouped["fp_obstructed_as_clear"]
    )
    grouped["pooled_clear_recall"] = grouped["tp_clear"] / (
        grouped["tp_clear"] + grouped["fn_clear_as_obstructed"]
    )
    return grouped


def experiment_label(item: dict) -> str:
    extractors = "+".join(item["extractors"])
    reduction = item.get("reduction", "full").lower()
    if reduction == "full":
        processing = "full features"
    else:
        method = {"pca": "PCA", "umap": "UMAP", "relieff": "Relief-F"}[reduction]
        suffix = " per CNN" if item["approach"] == "D" else ""
        processing = f"{method}-{item['components']}{suffix}"
    classifier = {
        "decision_tree": "decision tree",
        "rbf_svm": "RBF SVM",
        "linear_svm": "linear SVM",
        "mlp": "MLP",
        "logistic": "logistic regression",
        "random_forest": "random forest",
        "adaboost": "AdaBoost",
        "gaussian_nb": "Gaussian naive Bayes",
    }[item["classifier"]]
    return f"{extractors}; {processing}; {classifier}"


def enrich_table(
    names: list[str],
    ranks: pd.Series,
    experiment_map: dict[str, dict],
    summaries: pd.DataFrame,
    pooled: pd.DataFrame,
) -> pd.DataFrame:
    metadata = []
    for name in names:
        item = experiment_map[name]
        metadata.append(
            {
                "experiment": name,
                "approach": item["approach"],
                "model": experiment_label(item),
                "extractors": "+".join(item["extractors"]),
                "reduction": item.get("reduction", "full"),
                "components": item.get("components"),
                "classifier": item["classifier"],
                "average_friedman_rank": float(ranks[name]),
            }
        )
    table = pd.DataFrame(metadata).merge(summaries, on="experiment", how="left")
    table = table.merge(pooled, on="experiment", how="left")
    table["accuracy_iqr"] = table["accuracy_q3"] - table["accuracy_q1"]
    table = table.sort_values(
        [
            "average_friedman_rank",
            "accuracy_median",
            "pooled_accuracy",
            "pooled_obstructed_recall",
            "experiment",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    table.insert(0, "rank_position", np.arange(1, len(table) + 1))
    return table


def choose_winner(table: pd.DataFrame) -> str:
    return str(table.iloc[0]["experiment"])


def select_hierarchically(
    approach: str,
    records: pd.DataFrame,
    summaries: pd.DataFrame,
    pooled: pd.DataFrame,
    experiment_map: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    approach_names = [
        name for name, item in experiment_map.items() if item["approach"] == approach
    ]
    if approach == "A":
        ranks = average_friedman_ranks(records, approach_names)
        table = enrich_table(
            approach_names, ranks, experiment_map, summaries, pooled
        )
        return table, table.copy()

    groups: dict[tuple[str, ...], list[str]] = {}
    for name in approach_names:
        key = tuple(experiment_map[name]["extractors"])
        groups.setdefault(key, []).append(name)

    stage_one_rows = []
    finalists = []
    for key, names in sorted(groups.items()):
        ranks = average_friedman_ranks(records, names)
        table = enrich_table(names, ranks, experiment_map, summaries, pooled)
        winner = choose_winner(table)
        finalists.append(winner)
        row = table.iloc[0].copy()
        row["selection_group"] = "+".join(key)
        stage_one_rows.append(row)

    finalist_ranks = average_friedman_ranks(records, finalists)
    final_table = enrich_table(
        finalists, finalist_ranks, experiment_map, summaries, pooled
    )
    return final_table, pd.DataFrame(stage_one_rows)


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

    pooled = pooled_metrics(records)
    final_tables = {}
    stage_one_tables = {}
    selected_names = []
    for approach in "ABCD":
        final_table, stage_one = select_hierarchically(
            approach, records, summaries, pooled, experiment_map
        )
        final_tables[approach] = final_table
        stage_one_tables[approach] = stage_one
        selected_names.append(choose_winner(final_table))
        final_table.to_csv(output_dir / f"approach_{approach.lower()}_ranking.csv", index=False)
        stage_one.to_csv(
            output_dir / f"approach_{approach.lower()}_stage_one_winners.csv",
            index=False,
        )
        write_latex_rows(
            final_table,
            output_dir / f"approach_{approach.lower()}_rows.tex",
            APPROACH_TABLE_SIZES[approach],
        )

    selected_ranks = average_friedman_ranks(records, selected_names)
    selected_table = enrich_table(
        selected_names, selected_ranks, experiment_map, summaries, pooled
    )
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
