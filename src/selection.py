"""Pure article selection shared by manuscript tables, nested CV and final refits."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .artifact import summarise

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



def select_procedures(records, config, summaries=None):
    """Rank a complete eligible search; never reads files or accesses held-out data."""
    experiments = {item["name"]: item for item in config["experiments"]}
    if records.duplicated(["experiment", "fold"]).any():
        raise ValueError("Duplicate experiment/fold rows")
    summaries = summarise(records) if summaries is None else summaries
    pooled = pooled_metrics(records)
    final_tables, stage_one, selected = {}, {}, {}
    for approach in "ABCD":
        final_tables[approach], stage_one[approach] = select_hierarchically(approach, records, summaries, pooled, experiments)
        selected[approach] = choose_winner(final_tables[approach])
    names = list(selected.values())
    table = enrich_table(names, average_friedman_ranks(records, names), experiments, summaries, pooled)
    return {"selected": selected, "winner": choose_winner(table), "final_tables": final_tables, "stage_one": stage_one, "selected_table": table}
