#!/usr/bin/env python3
"""Migrate a checkpoint to persistent, shared Relief-F fold rankings."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_corrected_experiments import (
    config_digest,
    group_experiments,
    pipeline_digest,
    prediction_shard_path,
    read_checkpoint,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full_search.json")
    parser.add_argument("--output-dir", default="corrected_results/full_search")
    parser.add_argument("--relieff-n-jobs", type=int, default=2)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration. Without this flag, only report the plan.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.relieff_n_jobs == 0 or args.relieff_n_jobs < -1:
        raise ValueError("--relieff-n-jobs must be -1 or a positive integer")

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "per_fold_metrics.partial.csv"
    metadata_path = output_dir / "run_metadata.json"
    final_path = output_dir / "per_fold_metrics.csv"
    prediction_dir = output_dir / "per_sample_predictions"
    ranking_dir = output_dir / "relieff_rankings"

    if final_path.exists():
        raise RuntimeError("Refusing to migrate a completed run")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("config_sha256") != config_digest(config_path):
        raise ValueError("Configuration hash does not match run metadata")

    new_pipeline_hash = pipeline_digest()
    already_applied = any(
        item.get("type") == "shared_relieff_fold_ranking"
        and item.get("new_pipeline_sha256") == new_pipeline_hash
        and item.get("relieff_n_jobs") == args.relieff_n_jobs
        for item in metadata.get("migrations", [])
    )
    if already_applied:
        print("Shared Relief-F migration is already applied.")
        return

    groups = group_experiments(config["experiments"])
    relieff_group_indexes = {
        index
        for index, group in enumerate(groups)
        if group[0].get("reduction", "full").lower() == "relieff"
    }
    relieff_names = {
        experiment["name"]
        for index in relieff_group_indexes
        for experiment in groups[index]
    }
    relieff_max_features = max(
        int(experiment["components"])
        for experiment in config["experiments"]
        if experiment.get("reduction", "full").lower() == "relieff"
    )

    records = read_checkpoint(checkpoint_path)
    obsolete = records[records["experiment"].isin(relieff_names)].copy()
    retained = records[~records["experiment"].isin(relieff_names)].copy()
    shard_paths = [
        prediction_shard_path(prediction_dir, index)
        for index in sorted(relieff_group_indexes)
        if prediction_shard_path(prediction_dir, index).exists()
    ]

    print(f"Checkpoint rows: {len(records)}")
    print(f"Existing Relief-F rows to remove: {len(obsolete)}")
    print(f"Rows to preserve: {len(retained)}")
    print(f"Relief-F prediction shards to remove: {len(shard_paths)}")
    print(f"Persistent ranking size: {relieff_max_features}")
    print(f"Relief-F workers: {args.relieff_n_jobs}")
    print(f"New pipeline hash: {new_pipeline_hash}")
    if not args.apply:
        print("Dry run only. Re-run with --apply after stopping the experiment.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = output_dir / "checkpoint_migrations" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint_path, backup_dir / checkpoint_path.name)
    shutil.copy2(metadata_path, backup_dir / metadata_path.name)
    if shard_paths:
        shard_backup_dir = backup_dir / prediction_dir.name
        shard_backup_dir.mkdir()
        for shard in shard_paths:
            shutil.copy2(shard, shard_backup_dir / shard.name)

    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    retained.to_csv(temporary, index=False)
    temporary.replace(checkpoint_path)
    for shard in shard_paths:
        shard.unlink()

    migrations = list(metadata.get("migrations", []))
    migrations.append(
        {
            "applied_utc": datetime.now(timezone.utc).isoformat(),
            "type": "shared_relieff_fold_ranking",
            "reason": (
                "Reuse one training-fold Relief-F ranking for all nested "
                "feature-count cutoffs without changing held-out predictions."
            ),
            "previous_pipeline_sha256": metadata.get("pipeline_sha256"),
            "new_pipeline_sha256": new_pipeline_hash,
            "removed_checkpoint_rows": int(len(obsolete)),
            "removed_prediction_shards": int(len(shard_paths)),
            "preserved_checkpoint_rows": int(len(retained)),
            "relieff_ranking_size": relieff_max_features,
            "relieff_n_jobs": args.relieff_n_jobs,
            "backup_directory": str(backup_dir),
        }
    )
    metadata["pipeline_sha256"] = new_pipeline_hash
    metadata["relieff_n_jobs"] = args.relieff_n_jobs
    metadata["relieff_cache_dir"] = str(ranking_dir)
    metadata["relieff_max_features"] = relieff_max_features
    metadata["migrations"] = migrations
    write_json(metadata_path, metadata)

    print(f"Migration applied. Backup: {backup_dir}")
    print("Resume the original command; do not use --no-resume.")


if __name__ == "__main__":
    main()
