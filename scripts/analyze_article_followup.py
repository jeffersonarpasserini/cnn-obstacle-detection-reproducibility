#!/usr/bin/env python3
"""Analyze selected-model 10-fold and LOOCV runs for the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenfold-results", required=True)
    parser.add_argument("--loocv-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ground-signage-manifest",
        default=None,
        help="Optional CSV containing a filename column for the predefined subset",
    )
    return parser.parse_args()


def read_completed(directory: Path, expected_protocol: str):
    metadata_path = directory / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("completed"):
        raise RuntimeError(f"Incomplete run: {directory}")
    if metadata.get("protocol") != expected_protocol:
        raise ValueError(
            f"Expected protocol {expected_protocol}, found {metadata.get('protocol')}"
        )
    records = pd.read_csv(directory / "per_fold_metrics.csv")
    return metadata, records


def pooled_metrics(records: pd.DataFrame, protocol: str) -> pd.DataFrame:
    grouped = records.groupby(["experiment", "approach"], as_index=False).agg(
        folds=("fold", "nunique"),
        tn_obstructed=("tn_obstructed", "sum"),
        fp_obstructed_as_clear=("fp_obstructed_as_clear", "sum"),
        fn_clear_as_obstructed=("fn_clear_as_obstructed", "sum"),
        tp_clear=("tp_clear", "sum"),
        training_seconds_total=("training_seconds", "sum"),
        prediction_seconds_total=("prediction_seconds", "sum"),
    )
    grouped.insert(2, "protocol", protocol)
    total = (
        grouped["tn_obstructed"]
        + grouped["fp_obstructed_as_clear"]
        + grouped["fn_clear_as_obstructed"]
        + grouped["tp_clear"]
    )
    grouped["samples"] = total
    grouped["accuracy"] = (
        grouped["tn_obstructed"] + grouped["tp_clear"]
    ) / total
    grouped["obstructed_recall"] = grouped["tn_obstructed"] / (
        grouped["tn_obstructed"] + grouped["fp_obstructed_as_clear"]
    )
    grouped["clear_recall"] = grouped["tp_clear"] / (
        grouped["tp_clear"] + grouped["fn_clear_as_obstructed"]
    )
    grouped["balanced_accuracy"] = (
        grouped["obstructed_recall"] + grouped["clear_recall"]
    ) / 2
    grouped["obstructed_precision"] = grouped["tn_obstructed"] / (
        grouped["tn_obstructed"] + grouped["fn_clear_as_obstructed"]
    )
    grouped["clear_precision"] = grouped["tp_clear"] / (
        grouped["tp_clear"] + grouped["fp_obstructed_as_clear"]
    )
    return grouped


def read_predictions(directory: Path, metadata: dict) -> pd.DataFrame:
    index_path = directory / "prediction_shard_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Prediction index missing in {directory}; rerun with --prediction-mode all"
        )
    shards = pd.read_csv(index_path)
    frames = []
    for relative in shards["file"]:
        path = directory / relative
        if not path.exists():
            raise FileNotFoundError(
                f"Prediction shard missing: {path}. Ensure per_sample_predictions/ "
                "was copied from the execution machine."
            )
        frames.append(pd.read_csv(path))
    predictions = pd.concat(frames, ignore_index=True)
    keys = ["experiment", "sample_index"]
    if predictions.duplicated(keys).any():
        raise RuntimeError(f"Duplicate prediction rows in {directory}")

    if metadata.get("prediction_mode") == "all":
        expected = int(metadata["experiments"]) * int(metadata["images"])
        if len(predictions) != expected:
            raise RuntimeError(
                f"Expected {expected} predictions in {directory}, found {len(predictions)}"
            )
    return predictions


def correct_mask(predictions: pd.DataFrame) -> pd.Series:
    values = predictions["correct"]
    if pd.api.types.is_bool_dtype(values):
        return values
    normalized = values.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    invalid = sorted(set(normalized) - allowed)
    if invalid:
        raise ValueError(f"Invalid values in correct column: {invalid}")
    return normalized.isin({"true", "1"})


def error_outputs(predictions: pd.DataFrame, output: Path, prefix: str):
    mask = correct_mask(predictions)
    errors = predictions[~mask].copy()
    errors.to_csv(output / f"{prefix}_error_predictions.csv", index=False)
    per_sample = (
        errors.groupby(["sample_index", "filename", "true_label"], as_index=False)
        .agg(
            approaches_with_error=("approach", "nunique"),
            models_with_error=("experiment", "nunique"),
        )
        .sort_values(["models_with_error", "filename"], ascending=[False, True])
    )
    per_sample.to_csv(output / f"{prefix}_error_samples.csv", index=False)
    distribution = (
        per_sample.groupby("models_with_error", as_index=False)
        .size()
        .rename(columns={"size": "images"})
        .sort_values("models_with_error")
    )
    distribution.to_csv(output / f"{prefix}_error_distribution.csv", index=False)
    class_summary = (
        per_sample.groupby("true_label", as_index=False)
        .size()
        .rename(columns={"size": "images_with_at_least_one_error"})
    )
    class_summary.to_csv(output / f"{prefix}_error_class_summary.csv", index=False)


def subset_metrics(predictions: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    if "filename" not in manifest.columns:
        raise ValueError("Ground-signage manifest requires a filename column")
    subset = predictions[predictions["filename"].isin(manifest["filename"])].copy()
    missing = sorted(set(manifest["filename"]) - set(subset["filename"]))
    if missing:
        raise ValueError(f"Ground-signage filenames absent from predictions: {missing}")
    rows = []
    for experiment, group in subset.groupby("experiment"):
        rows.append(
            {
                "experiment": experiment,
                "approach": group["approach"].iloc[0],
                "images": int(group["filename"].nunique()),
                "accuracy": float(correct_mask(group).mean()),
                "errors": int((~correct_mask(group)).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("approach")


def write_latex_comparison(table: pd.DataFrame, destination: Path):
    pivot = table.pivot(index=["experiment", "approach"], columns="protocol")
    lines = [
        "% Generated from selected-model 10-fold and LOOCV runs.",
        "% model, approach, LOOCV accuracy, 10-fold pooled accuracy,",
        "% LOOCV obstacle recall, 10-fold obstacle recall",
    ]
    for (experiment, approach), row in pivot.iterrows():
        latex_experiment = experiment.replace("_", "\\_")
        lines.append(
            f"{latex_experiment} & {approach} "
            f"& {row[('accuracy', 'loocv')]:.4f} "
            f"& {row[('accuracy', 'stratified_kfold')]:.4f} "
            f"& {row[('obstructed_recall', 'loocv')]:.4f} "
            f"& {row[('obstructed_recall', 'stratified_kfold')]:.4f} \\\\"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    tenfold_dir = Path(args.tenfold_results)
    loocv_dir = Path(args.loocv_results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    ten_meta, ten_records = read_completed(tenfold_dir, "stratified_kfold")
    loo_meta, loo_records = read_completed(loocv_dir, "loocv")
    ten_names = set(ten_records["experiment"])
    loo_names = set(loo_records["experiment"])
    if ten_names != loo_names:
        raise ValueError("10-fold and LOOCV runs contain different model sets")

    comparison = pd.concat(
        [
            pooled_metrics(ten_records, "stratified_kfold"),
            pooled_metrics(loo_records, "loocv"),
        ],
        ignore_index=True,
    ).sort_values(["approach", "protocol"])
    comparison.to_csv(output / "validation_protocol_comparison.csv", index=False)
    write_latex_comparison(comparison, output / "validation_protocol_rows.tex")

    ten_predictions = read_predictions(tenfold_dir, ten_meta)
    loo_predictions = read_predictions(loocv_dir, loo_meta)
    error_outputs(ten_predictions, output, "tenfold")
    error_outputs(loo_predictions, output, "loocv")

    if args.ground_signage_manifest:
        manifest = Path(args.ground_signage_manifest)
        subset_metrics(ten_predictions, manifest).to_csv(
            output / "ground_signage_tenfold.csv", index=False
        )
        subset_metrics(loo_predictions, manifest).to_csv(
            output / "ground_signage_loocv.csv", index=False
        )

    manifest = {
        "models": sorted(ten_names),
        "images": int(ten_meta["images"]),
        "tenfold_results": str(tenfold_dir),
        "loocv_results": str(loocv_dir),
        "ground_signage_manifest": args.ground_signage_manifest,
        "interpretation_notice": (
            "LOOCV uses the same dataset used for model selection. It measures "
            "resampling stability and is not independent external validation."
        ),
    }
    (output / "followup_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Follow-up analysis written to {output}")


if __name__ == "__main__":
    main()
