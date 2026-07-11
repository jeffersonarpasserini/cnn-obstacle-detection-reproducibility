#!/usr/bin/env python3
"""Remove supervised-UMAP rows and migrate a resumable full-search checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration. Without this flag, only report the planned changes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "per_fold_metrics.partial.csv"
    metadata_path = output_dir / "run_metadata.json"
    final_path = output_dir / "per_fold_metrics.csv"
    prediction_dir = output_dir / "per_sample_predictions"

    if final_path.exists():
        raise RuntimeError("Refusing to migrate a completed run")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_config_hash = config_digest(config_path)
    if metadata.get("config_sha256") != expected_config_hash:
        raise ValueError("Configuration hash does not match run metadata")

    groups = group_experiments(config["experiments"])
    umap_group_indexes = {
        index
        for index, group in enumerate(groups)
        if group[0].get("reduction", "full").lower() == "umap"
    }
    umap_names = {
        experiment["name"]
        for index in umap_group_indexes
        for experiment in groups[index]
    }

    records = read_checkpoint(checkpoint_path)
    invalid = records[records["experiment"].isin(umap_names)].copy()
    retained = records[~records["experiment"].isin(umap_names)].copy()
    shard_paths = [
        prediction_shard_path(prediction_dir, index)
        for index in sorted(umap_group_indexes)
        if prediction_shard_path(prediction_dir, index).exists()
    ]

    print(f"Checkpoint rows: {len(records)}")
    print(f"Supervised-UMAP rows to remove: {len(invalid)}")
    print(f"Rows to preserve: {len(retained)}")
    print(f"UMAP prediction shards to remove: {len(shard_paths)}")
    print(f"New pipeline hash: {pipeline_digest()}")
    if not args.apply:
        print("Dry run only. Re-run with --apply after stopping the experiment process.")
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

    old_pipeline_hash = metadata.get("pipeline_sha256")
    new_pipeline_hash = pipeline_digest()
    migrations = list(metadata.get("migrations", []))
    migrations.append(
        {
            "applied_utc": datetime.now(timezone.utc).isoformat(),
            "type": "unsupervised_umap_protocol_correction",
            "reason": (
                "UMAP had received y_train and therefore performed supervised "
                "dimension reduction; the protocol requires unsupervised UMAP."
            ),
            "previous_pipeline_sha256": old_pipeline_hash,
            "new_pipeline_sha256": new_pipeline_hash,
            "removed_checkpoint_rows": int(len(invalid)),
            "removed_prediction_shards": int(len(shard_paths)),
            "preserved_checkpoint_rows": int(len(retained)),
            "backup_directory": str(backup_dir),
        }
    )
    metadata["pipeline_sha256"] = new_pipeline_hash
    metadata["migrations"] = migrations
    write_json(metadata_path, metadata)

    print(f"Migration applied. Backup: {backup_dir}")
    print("Resume with the original command, without --no-resume.")


if __name__ == "__main__":
    main()
