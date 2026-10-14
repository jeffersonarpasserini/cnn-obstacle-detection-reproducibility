#!/usr/bin/env python3
"""Verify released external images against the authenticated release manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = '02eb271bd17d9a63d8d4edf7af11bd782b3cc4f165c2516175e39478c94cdd48'
EXPECTED = {'images': 1576, 'clear': 844, 'obstructed': 732, 'groups': 1378,
            'excluded': 353, 'anonymized': 16, 'faces': 1, 'plates': 19}
FIELDS = {'filename', 'final_label', 'group_id', 'width', 'height',
          'sha256_evaluated', 'sha256_released', 'anonymized', 'n_blurred_faces', 'n_blurred_plates'}


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def verify_release(root: Path = ROOT, *, manifest_sha256: str = MANIFEST_SHA256,
                   expected: dict[str, int] = EXPECTED) -> dict[str, int | str]:
    root = Path(root)
    directory = root / 'external-data'
    manifest = directory / 'external_release_manifest.csv'
    require(sha256(manifest) == manifest_sha256, 'Release manifest SHA-256 mismatch')
    rows = read_csv(manifest)
    require(len(rows) == expected['images'], 'Incorrect released image count')
    require(bool(rows) and FIELDS <= rows[0].keys(), 'Release manifest columns missing')
    names = [row['filename'] for row in rows]
    require(len(set(names)) == len(names), 'Duplicate released filename')
    require(all(Path(name).name == name and name.lower().endswith('.jpg') and '\\' not in name
                for name in names), 'Unsafe or unsupported image filename')
    labels = Counter(int(row['final_label']) for row in rows)
    require(labels == {1: expected['clear'], 0: expected['obstructed']}, 'Incorrect class counts')
    require(all(row['group_id'] for row in rows), 'Missing scene group')
    groups = {row['group_id'] for row in rows}
    require(len(groups) == expected['groups'], 'Incorrect scene group count')

    clean = read_csv(directory / 'external_manifest_clean.csv')
    kept = {row['filename']: row for row in clean if row['excluded'].lower() == 'false'}
    require(len(kept) == len(rows) and set(kept) == set(names), 'Release differs from kept manifest')
    raw = read_csv(directory / 'external_manifest_raw.csv')
    raw_names = {row['filename'] for row in raw}
    require(len(raw_names) == len(raw), 'Duplicate raw manifest filename')
    require(set(names) <= raw_names, 'Released filename absent from raw manifest')
    excluded = raw_names - set(kept)
    require(len(excluded) == expected['excluded'], 'Incorrect excluded population count')

    images = directory / 'via-dataset-extended'
    entries = list(images.iterdir())
    actual = {entry.name for entry in entries if entry.name != 'README.md'}
    require(not actual & excluded, 'Excluded image present in released directory')
    require(actual == set(names), 'Missing or extra released files')
    require(all(entry.is_file() and not entry.is_symlink() for entry in entries),
            'Released directory contains a subdirectory or symlink')
    anonymized = faces = plates = 0
    for row in rows:
        name = row['filename']
        original = kept[name]
        require(row['sha256_evaluated'] == original['sha256'], f'Evaluated hash differs: {name}')
        for field in ('final_label', 'group_id', 'width', 'height'):
            require(row[field] == original[field], f'Frozen {field} differs: {name}')
        require(int(row['width']) > 0 and int(row['height']) > 0, f'Invalid dimensions: {name}')
        require(row['anonymized'] in {'True', 'False'}, f'Invalid anonymized flag: {name}')
        face_count, plate_count = int(row['n_blurred_faces']), int(row['n_blurred_plates'])
        require(face_count >= 0 and plate_count >= 0, f'Negative blur count: {name}')
        changed = row['anonymized'] == 'True'
        require(changed == (face_count + plate_count > 0), f'Blur count/flag mismatch: {name}')
        require(changed == (row['sha256_evaluated'] != row['sha256_released']),
                f'Hash/flag mismatch: {name}')
        require(sha256(images / name) == row['sha256_released'], f'Released SHA-256 mismatch: {name}')
        anonymized += int(changed); faces += face_count; plates += plate_count
    require((anonymized, faces, plates) == (expected['anonymized'], expected['faces'], expected['plates']),
            'Incorrect anonymization counts')
    return {'images': len(rows), 'clear': labels[1], 'obstructed': labels[0], 'groups': len(groups),
            'excluded_absent': len(excluded), 'anonymized': anonymized, 'unchanged': len(rows)-anonymized,
            'blurred_faces': faces, 'blurred_plates': plates, 'manifest_sha256': manifest_sha256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    arguments = parser.parse_args()
    try:
        result = verify_release(arguments.root)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'External release verification failed: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
