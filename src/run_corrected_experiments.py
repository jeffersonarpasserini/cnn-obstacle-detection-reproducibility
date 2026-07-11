#!/usr/bin/env python3
"""Run selected leakage-free experiments from a JSON configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from artifact import (
    evaluate_experiment,
    extract_or_load_features,
    load_config,
    make_splits,
    scan_dataset,
    set_global_seed,
    summarise,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="via-dataset", help="Directory containing VIA images"
    )
    parser.add_argument(
        "--config", default="configs/selected_models.json", help="Experiment JSON"
    )
    parser.add_argument("--cache-dir", default="cache/features")
    parser.add_argument("--output-dir", default="corrected_results")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 1980))
    n_splits = int(config.get("n_splits", 10))
    protocol = config.get("protocol", "stratified_kfold")
    set_global_seed(seed)

    image_paths, labels = scan_dataset(args.dataset)
    required_extractors = sorted(
        {name for experiment in config["experiments"] for name in experiment["extractors"]}
    )

    feature_map = {}
    extraction_times = []
    for model_name in required_extractors:
        features, elapsed = extract_or_load_features(
            model_name, image_paths, args.cache_dir, args.batch_size
        )
        if len(features) != len(labels):
            raise ValueError(f"Feature count mismatch for {model_name}")
        feature_map[model_name] = features
        extraction_times.append({"extractor": model_name, "seconds": elapsed})

    splits = make_splits(labels, protocol, n_splits, seed)
    all_records = []
    for experiment in config["experiments"]:
        all_records.extend(
            evaluate_experiment(experiment, feature_map, labels, splits, seed)
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame(all_records)
    records.to_csv(output_dir / "per_fold_metrics.csv", index=False)
    summarise(records).to_csv(output_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(extraction_times).to_csv(
        output_dir / "feature_extraction_times.csv", index=False
    )
    pd.DataFrame(
        {"filename": [path.name for path in image_paths], "label": labels}
    ).to_csv(output_dir / "dataset_index.csv", index=False)
    print(summarise(records).to_string(index=False))


if __name__ == "__main__":
    main()
