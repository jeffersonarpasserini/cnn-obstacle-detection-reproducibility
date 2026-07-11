#!/usr/bin/env python3
"""Run leakage-free experiments with shared preprocessing and checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd

try:
    from artifact import (
        dataset_content_fingerprint,
        dataset_fingerprint,
        evaluate_experiment_group,
        extract_or_load_features,
        load_config,
        make_splits,
        preprocessing_key,
        ReliefRankingCache,
        scan_dataset,
        set_global_seed,
        summarise,
    )
except ModuleNotFoundError:
    # Support importing the runner as ``src.run_corrected_experiments`` in
    # protocol tests while preserving direct script execution.
    from src.artifact import (
        dataset_content_fingerprint,
        dataset_fingerprint,
        evaluate_experiment_group,
        extract_or_load_features,
        load_config,
        make_splits,
        preprocessing_key,
        ReliefRankingCache,
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
    parser.add_argument(
        "--max-loaded-extractors",
        type=int,
        default=2,
        help="Maximum CNN feature arrays retained in RAM (default: 2)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard an existing partial checkpoint and start again",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=("none", "errors", "all"),
        default="errors",
        help="Per-sample prediction records to retain (default: errors)",
    )
    parser.add_argument(
        "--relieff-n-jobs",
        type=int,
        default=2,
        help="CPU workers used inside Relief-F (default: 2; monitor RAM)",
    )
    parser.add_argument(
        "--relieff-cache-dir",
        default=None,
        help="Persistent ranking cache (default: OUTPUT_DIR/relieff_rankings)",
    )
    return parser.parse_args()


def group_experiments(experiments):
    """Preserve config order while grouping classifier-independent work."""
    grouped = OrderedDict()
    for experiment in experiments:
        grouped.setdefault(preprocessing_key(experiment), []).append(experiment)
    return list(grouped.values())


class FeatureStore:
    """Lazily load CNN caches with a small least-recently-used RAM window."""

    def __init__(
        self, paths, labels, cache_dir, batch_size, max_loaded,
        existing_extraction_times=None, existing_dimensions=None,
    ):
        if max_loaded < 1:
            raise ValueError("--max-loaded-extractors must be at least 1")
        self.paths = paths
        self.labels = labels
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_loaded = max_loaded
        self.loaded = OrderedDict()
        self.extraction_times = OrderedDict(existing_extraction_times or {})
        self.feature_dimensions = OrderedDict(existing_dimensions or {})

    def get(self, model_name):
        if model_name in self.loaded:
            features = self.loaded.pop(model_name)
            self.loaded[model_name] = features
            return features

        features, elapsed = extract_or_load_features(
            model_name, self.paths, self.cache_dir, self.batch_size
        )
        if len(features) != len(self.labels):
            raise ValueError(f"Feature count mismatch for {model_name}")
        self.extraction_times.setdefault(model_name, elapsed)
        self.feature_dimensions[model_name] = {
            "extractor": model_name,
            "samples": int(features.shape[0]),
            "features_per_image": int(features.shape[1]),
            "array_mebibytes": float(features.nbytes / (1024 ** 2)),
        }
        self.loaded[model_name] = features
        while len(self.loaded) > self.max_loaded:
            self.loaded.popitem(last=False)

        # TensorFlow may retain a completed application model even though only
        # its NumPy feature matrix is needed from this point onward.
        if elapsed > 0:
            try:
                import tensorflow as tf

                tf.keras.backend.clear_session()
            except ImportError:
                pass
            gc.collect()
        return features

    def feature_map(self, experiments):
        names = experiments[0]["extractors"]
        return {name: self.get(name) for name in names}

    def times_frame(self):
        return pd.DataFrame(
            [
                {"extractor": name, "seconds": elapsed}
                for name, elapsed in self.extraction_times.items()
            ]
        )

    def dimensions_frame(self):
        return pd.DataFrame(self.feature_dimensions.values())


def config_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pipeline_digest():
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("artifact.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dependency_versions():
    packages = [
        "numpy", "pandas", "scipy", "scikit-learn", "tensorflow",
        "umap-learn", "skrebate", "Pillow",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_metadata():
    def run(*arguments):
        try:
            return subprocess.run(
                ["git", *arguments], check=True, capture_output=True, text=True
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def hardware_metadata():
    result = {
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "physical_memory_bytes": None,
        "gpus": [],
    }
    try:
        result["physical_memory_bytes"] = (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        )
    except (AttributeError, ValueError):
        pass
    try:
        import tensorflow as tf

        result["gpus"] = []
        for device in tf.config.list_physical_devices("GPU"):
            details = tf.config.experimental.get_device_details(device)
            result["gpus"].append(
                {"logical_name": device.name, "device_name": details.get("device_name")}
            )
        result["tensorflow_build"] = tf.sysconfig.get_build_info()
    except ImportError:
        pass
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def expected_pairs(experiments, n_splits):
    return {
        (experiment["name"], fold)
        for experiment in experiments
        for fold in range(1, n_splits + 1)
    }


def read_checkpoint(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    records = pd.read_csv(path, on_bad_lines="skip")
    required = {"experiment", "fold"}
    if not required.issubset(records.columns):
        raise ValueError(f"Invalid checkpoint columns in {path}")
    records["fold"] = pd.to_numeric(records["fold"], errors="coerce")
    records = records.dropna(subset=["experiment", "fold"]).copy()
    records["fold"] = records["fold"].astype(int)
    return records


def clean_incomplete_groups(records, groups, n_splits):
    """Remove any partly appended group before resuming it from scratch."""
    if records.empty:
        return records, set()
    allowed_names = {
        experiment["name"] for group in groups for experiment in group
    }
    records = records[
        records["experiment"].isin(allowed_names)
        & records["fold"].between(1, n_splits)
    ].drop_duplicates(subset=["experiment", "fold"], keep="last").copy()
    observed = set(zip(records["experiment"], records["fold"].astype(int)))
    completed = set()
    incomplete_names = set()
    for index, group in enumerate(groups):
        expected = expected_pairs(group, n_splits)
        present = expected & observed
        if present == expected:
            completed.add(index)
        elif present:
            incomplete_names.update(experiment["name"] for experiment in group)
    if incomplete_names:
        records = records[~records["experiment"].isin(incomplete_names)].copy()
    return records, completed


def write_checkpoint_rows(path, records):
    frame = pd.DataFrame(records)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


PREDICTION_COLUMNS = [
    "experiment", "approach", "classifier", "fold", "sample_index",
    "filename", "true_label", "predicted_label", "score_clear", "correct",
]


def prediction_shard_path(directory, group_index):
    return directory / f"group_{group_index:04d}.csv.gz"


def write_prediction_shard(directory, group_index, records):
    directory.mkdir(parents=True, exist_ok=True)
    destination = prediction_shard_path(directory, group_index)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pd.DataFrame(records, columns=PREDICTION_COLUMNS).to_csv(
        temporary, index=False, compression="gzip"
    )
    temporary.replace(destination)


def write_fold_assignments(path, image_paths, labels, splits):
    rows = []
    for fold, (_, test) in enumerate(splits, start=1):
        for sample_index in test:
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "filename": image_paths[sample_index].name,
                    "label": int(labels[sample_index]),
                    "test_fold": fold,
                }
            )
    pd.DataFrame(rows).sort_values("sample_index").to_csv(path, index=False)


def write_prediction_analysis(
    output_dir, prediction_dir, groups, image_paths, labels, experiment_count
):
    shard_rows = []
    error_counts = {path.name: 0 for path in image_paths}
    for group_index, group in enumerate(groups):
        shard = prediction_shard_path(prediction_dir, group_index)
        frame = pd.read_csv(shard)
        errors = frame[frame["correct"].astype(str).str.lower().isin(["false", "0"])]
        for filename, count in errors["filename"].value_counts().items():
            error_counts[filename] += int(count)
        shard_rows.append(
            {
                "group": group_index,
                "file": str(shard.relative_to(output_dir)),
                "experiments": len(group),
                "prediction_rows": len(frame),
            }
        )
    pd.DataFrame(shard_rows).to_csv(
        output_dir / "prediction_shard_index.csv", index=False
    )
    label_by_name = {
        path.name: int(label) for path, label in zip(image_paths, labels)
    }
    error_rows = [
        {
            "filename": filename,
            "true_label": label_by_name[filename],
            "error_count": count,
            "evaluated_experiments": experiment_count,
            "error_rate": count / experiment_count,
        }
        for filename, count in error_counts.items()
    ]
    pd.DataFrame(error_rows).sort_values(
        ["error_count", "filename"], ascending=[False, True]
    ).to_csv(output_dir / "sample_error_summary.csv", index=False)


def main():
    args = parse_args()
    if args.relieff_n_jobs == 0 or args.relieff_n_jobs < -1:
        raise ValueError("--relieff-n-jobs must be -1 or a positive integer")
    config = load_config(args.config)
    seed = int(config.get("seed", 1980))
    n_splits = int(config.get("n_splits", 10))
    protocol = config.get("protocol", "stratified_kfold")
    set_global_seed(seed)

    image_paths, labels = scan_dataset(args.dataset)
    splits = make_splits(labels, protocol, n_splits, seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    relieff_cache_dir = Path(
        args.relieff_cache_dir or output_dir / "relieff_rankings"
    )
    relieff_cache = ReliefRankingCache(
        relieff_cache_dir, n_jobs=args.relieff_n_jobs
    )
    relieff_components = [
        int(experiment["components"])
        for experiment in config["experiments"]
        if experiment.get("reduction", "full").lower() == "relieff"
    ]
    relieff_max_features = max(relieff_components, default=0)
    checkpoint_path = output_dir / "per_fold_metrics.partial.csv"
    final_path = output_dir / "per_fold_metrics.csv"
    metadata_path = output_dir / "run_metadata.json"
    prediction_dir = output_dir / "per_sample_predictions"
    groups = group_experiments(config["experiments"])
    fold_count = len(splits)
    all_expected = expected_pairs(config["experiments"], fold_count)

    metadata = {
        "config": str(Path(args.config)),
        "config_sha256": config_digest(args.config),
        "pipeline_sha256": pipeline_digest(),
        "dataset": str(Path(args.dataset)),
        "dataset_filename_fingerprint": dataset_fingerprint(image_paths),
        "dataset_content_sha256": dataset_content_fingerprint(image_paths),
        "images": len(image_paths),
        "seed": seed,
        "protocol": protocol,
        "n_splits": n_splits,
        "actual_folds": fold_count,
        "experiments": len(config["experiments"]),
        "preprocessing_groups": len(groups),
        "prediction_mode": args.prediction_mode,
        "relieff_n_jobs": args.relieff_n_jobs,
        "relieff_cache_dir": str(relieff_cache_dir),
        "relieff_max_features": relieff_max_features,
        "command": sys.argv,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
        "hardware": hardware_metadata(),
        "git": git_metadata(),
    }

    if metadata_path.exists() and not args.no_resume:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        identity_fields = [
            "config_sha256", "pipeline_sha256", "dataset_filename_fingerprint",
            "dataset_content_sha256", "seed",
            "protocol", "n_splits", "prediction_mode",
            "relieff_n_jobs",
        ]
        if any(previous.get(field) != metadata.get(field) for field in identity_fields):
            raise ValueError(
                "Output directory contains a checkpoint from another run; "
                "choose a new --output-dir or pass --no-resume"
            )
        # Preserve an auditable history created by an explicit checkpoint
        # migration.  The freshly collected runtime metadata above remains
        # authoritative for the resumed segment.
        if previous.get("migrations"):
            metadata["migrations"] = previous["migrations"]

    if args.no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()
    if args.no_resume:
        relieff_cache.clear()
        for stale_path in (
            final_path,
            output_dir / "summary_metrics.csv",
            output_dir / "relieff_cache_status.json",
        ):
            if stale_path.exists():
                stale_path.unlink()
    write_json(metadata_path, metadata)

    pd.DataFrame(
        {"filename": [path.name for path in image_paths], "label": labels}
    ).to_csv(output_dir / "dataset_index.csv", index=False)
    write_fold_assignments(
        output_dir / "fold_assignments.csv", image_paths, labels, splits
    )

    # A completed, identity-matched run needs no work.
    if final_path.exists() and not checkpoint_path.exists() and not args.no_resume:
        final_records = pd.read_csv(final_path)
        observed = set(
            zip(final_records["experiment"], final_records["fold"].astype(int))
        )
        predictions_complete = args.prediction_mode == "none" or all(
            prediction_shard_path(prediction_dir, index).exists()
            for index in range(len(groups))
        )
        if observed == all_expected and predictions_complete:
            print("Run already complete; no configurations were recomputed.")
            print(summarise(final_records).to_string(index=False))
            return
        if observed == all_expected:
            final_records.to_csv(checkpoint_path, index=False)

    checkpoint = read_checkpoint(checkpoint_path)
    checkpoint, completed_groups = clean_incomplete_groups(
        checkpoint, groups, fold_count
    )
    if not checkpoint.empty or checkpoint_path.exists():
        checkpoint.to_csv(checkpoint_path, index=False)

    # A result group is complete only when both its aggregate checkpoint and,
    # when requested, its atomic per-sample prediction shard are present.
    if args.prediction_mode != "none" and completed_groups:
        missing_shards = {
            index
            for index in completed_groups
            if not prediction_shard_path(prediction_dir, index).exists()
        }
        if missing_shards:
            incomplete_names = {
                experiment["name"]
                for index in missing_shards
                for experiment in groups[index]
            }
            checkpoint = checkpoint[
                ~checkpoint["experiment"].isin(incomplete_names)
            ].copy()
            checkpoint.to_csv(checkpoint_path, index=False)
            completed_groups.difference_update(missing_shards)

    extraction_times_path = output_dir / "feature_extraction_times.csv"
    feature_dimensions_path = output_dir / "feature_dimensions.csv"
    existing_extraction_times = {}
    existing_dimensions = {}
    if extraction_times_path.exists() and not args.no_resume:
        previous_times = pd.read_csv(extraction_times_path)
        existing_extraction_times = dict(
            zip(previous_times["extractor"], previous_times["seconds"])
        )
    if feature_dimensions_path.exists() and not args.no_resume:
        previous_dimensions = pd.read_csv(feature_dimensions_path)
        existing_dimensions = {
            row["extractor"]: row.to_dict()
            for _, row in previous_dimensions.iterrows()
        }
    store = FeatureStore(
        image_paths,
        labels,
        args.cache_dir,
        args.batch_size,
        args.max_loaded_extractors,
        existing_extraction_times,
        existing_dimensions,
    )
    started = perf_counter()
    initially_completed = len(completed_groups)
    for index, group in enumerate(groups):
        if index in completed_groups:
            continue
        group_started = perf_counter()
        feature_map = store.feature_map(group)
        # Persist extraction timing before the long reduction/classifier stage,
        # so it survives an interruption during the first group for a CNN.
        store.times_frame().to_csv(extraction_times_path, index=False)
        store.dimensions_frame().to_csv(feature_dimensions_path, index=False)
        evaluated = evaluate_experiment_group(
            group,
            feature_map,
            labels,
            splits,
            seed,
            sample_names=[path.name for path in image_paths],
            prediction_mode=args.prediction_mode,
            relieff_cache=relieff_cache,
            relieff_max_features=relieff_max_features,
            relieff_n_jobs=args.relieff_n_jobs,
        )
        if args.prediction_mode == "none":
            group_records = evaluated
        else:
            group_records, prediction_records = evaluated
            # The shard is replaced atomically before the aggregate group is
            # marked complete in the append-only checkpoint.
            write_prediction_shard(
                prediction_dir, index, prediction_records
            )
        write_checkpoint_rows(checkpoint_path, group_records)
        completed_groups.add(index)
        write_json(
            output_dir / "relieff_cache_status.json",
            {
                "ranking_cache": str(relieff_cache_dir),
                "ranking_size": relieff_max_features,
                "n_jobs": args.relieff_n_jobs,
                "hits": relieff_cache.hits,
                "misses": relieff_cache.misses,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

        completed_now = len(completed_groups) - initially_completed
        remaining = len(groups) - len(completed_groups)
        mean_group_seconds = (perf_counter() - started) / completed_now
        eta_hours = remaining * mean_group_seconds / 3600
        print(
            f"[{len(completed_groups)}/{len(groups)}] "
            f"{group[0]['approach']} {'+'.join(group[0]['extractors'])} "
            f"{group[0].get('reduction', 'full')} "
            f"{group[0].get('components')} | "
            f"{perf_counter() - group_started:.1f}s | ETA {eta_hours:.1f}h",
            flush=True,
        )
        if args.prediction_mode == "none":
            del feature_map, group_records
        else:
            del feature_map, group_records, prediction_records
        gc.collect()

    records = pd.read_csv(checkpoint_path)
    observed = set(zip(records["experiment"], records["fold"].astype(int)))
    if observed != all_expected or len(records) != len(all_expected):
        raise RuntimeError(
            f"Incomplete final checkpoint: {len(observed)} of "
            f"{len(all_expected)} experiment-fold records"
        )

    experiment_order = {
        experiment["name"]: index
        for index, experiment in enumerate(config["experiments"])
    }
    records["_experiment_order"] = records["experiment"].map(experiment_order)
    records = records.sort_values(["_experiment_order", "fold"]).drop(
        columns="_experiment_order"
    )
    records.to_csv(final_path, index=False)
    summarise(records).to_csv(output_dir / "summary_metrics.csv", index=False)
    if args.prediction_mode != "none":
        write_prediction_analysis(
            output_dir,
            prediction_dir,
            groups,
            image_paths,
            labels,
            len(config["experiments"]),
        )
    checkpoint_path.unlink()
    metadata["completed"] = True
    metadata["records"] = len(records)
    metadata["last_segment_seconds"] = perf_counter() - started
    write_json(metadata_path, metadata)
    print(summarise(records).to_string(index=False))


if __name__ == "__main__":
    main()
