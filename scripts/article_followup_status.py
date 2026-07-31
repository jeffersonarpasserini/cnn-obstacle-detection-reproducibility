#!/usr/bin/env python3
"""Report progress of the selected-model 10-fold and LOOCV runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


RUNS = (
    ("10-fold", Path("results/article_selected_10fold"), 10),
    ("LOOCV", Path("results/article_selected_loocv"), 342),
)


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def status(label: str, directory: Path, expected_folds: int) -> str:
    metadata_path = directory / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    experiments = int(metadata.get("experiments", 4))
    expected = experiments * expected_folds
    final_path = directory / "per_fold_metrics.csv"
    partial_path = directory / "per_fold_metrics.partial.csv"
    observed = row_count(final_path if final_path.exists() else partial_path)
    percent = 100.0 * min(observed, expected) / expected
    shards = len(list((directory / "per_sample_predictions").glob("group_*.csv.gz")))
    complete = bool(metadata.get("completed")) and final_path.exists()
    state = "complete" if complete else ("running/resumable" if metadata else "not started")
    return (
        f"{label:7s} | {state:17s} | {observed:4d}/{expected:4d} fold records "
        f"({percent:6.2f}%) | prediction shards: {shards}/{experiments}"
    )


def main():
    for item in RUNS:
        print(status(*item))
    followup = Path("results/article_followup/validation_protocol_comparison.csv")
    print(f"tables  | {'complete' if followup.exists() else 'pending'}")


if __name__ == "__main__":
    main()
