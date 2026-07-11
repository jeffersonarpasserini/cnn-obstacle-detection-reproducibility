#!/usr/bin/env python3
"""Validate the VIA dataset and optionally write its integrity manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(dataset: Path):
    images = sorted(
        (p for p in dataset.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES),
        key=lambda p: p.name.lower(),
    )
    for path in images:
        name = path.name.lower()
        if name.startswith("clear."):
            label = "clear"
        elif name.startswith("nonclear.") or name.startswith("non-clear."):
            label = "obstructed"
        else:
            raise ValueError(f"Unknown filename convention: {path.name}")
        yield {
            "filename": path.name,
            "label": label,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="via-dataset")
    parser.add_argument("--manifest", default="data/dataset_manifest.csv")
    parser.add_argument("--write", action="store_true", help="Write or replace manifest")
    args = parser.parse_args()

    rows = list(records(Path(args.dataset)))
    counts = {
        "clear": sum(row["label"] == "clear" for row in rows),
        "obstructed": sum(row["label"] == "obstructed" for row in rows),
    }
    if len(rows) != 342 or counts != {"clear": 175, "obstructed": 167}:
        raise AssertionError(f"Unexpected dataset distribution: {len(rows)} images, {counts}")

    manifest = Path(args.manifest)
    if args.write:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "label", "bytes", "sha256"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote manifest: {manifest}")
    else:
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}; run with --write")
        with manifest.open(newline="", encoding="utf-8") as handle:
            expected = list(csv.DictReader(handle))
        actual = [{key: str(value) for key, value in row.items()} for row in rows]
        if actual != expected:
            raise AssertionError("Dataset differs from data/dataset_manifest.csv")
    print(f"Dataset OK: {len(rows)} images ({counts['clear']} clear, {counts['obstructed']} obstructed)")


if __name__ == "__main__":
    main()

