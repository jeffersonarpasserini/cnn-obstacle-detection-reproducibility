#!/usr/bin/env python3
"""Fetch and authenticate v2.0.0 model assets without loading executable models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'results/external_test/'
DOWNLOAD = ('https://github.com/jeffersonarpasserini/'
            'cnn-obstacle-detection-reproducibility/releases/download/v2.0.0/')
FIELDS = ['package', 'path_in_repo', 'bytes', 'sha256', 'model_id', 'seed', 'role']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or not name or path.is_absolute()
            or any(part in {'', '.', '..'} for part in name.split('/'))
            or '\\' in name or '\x00' in name or '\n' in name or '\r' in name
            or str(path) != name):
        raise ValueError(f'Unsafe asset path: {name!r}')
    return path


def expected_records(freeze):
    if freeze.get('frozen') is not True or freeze.get('smoke') is not False:
        raise ValueError('A frozen non-smoke final_models.json is required')
    models = {m.get('model_path', m.get('pipeline')): m for m in freeze['models']}
    encoders = {path: name for name, path in freeze['encoders'].items()}
    expected = {}
    for relative, digest in freeze['artifacts'].items():
        if PurePosixPath(relative).suffix not in {'.keras', '.joblib'}:
            continue
        safe_name(relative)
        if relative in encoders:
            package = 'models_encoders_v2.0.0.tar.gz'
            model_id, seed, role = encoders[relative], '', 'encoder'
        elif relative in models:
            model = models[relative]
            model_id, seed = model['model'], str(model['seed'])
            if model['family'] == 'fine_tuning':
                package = f'models_ft_{model["architecture"]}_v2.0.0.tar.gz'
                role = 'fine_tuned_model'
            elif model['family'] == 'fixed':
                package, role = 'models_fixed_pipelines_v2.0.0.tar.gz', 'fixed_pipeline'
            else:
                raise ValueError(f'Unknown frozen model family: {relative}')
        elif re.fullmatch(r'adapter_(A|B|C|D|winner)\.joblib', relative):
            procedure = PurePosixPath(relative).stem.removeprefix('adapter_')
            model_id = (freeze['global_winner'] if procedure == 'winner'
                        else freeze['article_finalists'][procedure])
            seed, role = '', 'probability_adapter'
            package = 'models_fixed_pipelines_v2.0.0.tar.gz'
        else:
            raise ValueError(f'Unclassified frozen binary: {relative}')
        expected[BASE + relative] = dict(package=package, sha256=digest,
                                        model_id=model_id, seed=seed, role=role)
    return expected


def load_manifest(root=ROOT):
    root = Path(root)
    freeze = json.loads((root / BASE / 'final_models.json').read_text())
    expected = expected_records(freeze)
    with (root / 'models_manifest.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError('Unexpected models_manifest.csv columns')
        rows = list(reader)
    seen = set()
    for row in rows:
        name = row['path_in_repo']
        safe_name(name)
        if name in seen or name not in expected:
            raise ValueError(f'Duplicate or unexpected model: {name}')
        seen.add(name)
        if any(row[key] != value for key, value in expected[name].items()):
            raise ValueError(f'Model manifest differs from frozen record: {name}')
        row['bytes'] = int(row['bytes'])
        if row['bytes'] <= 0 or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError(f'Invalid size or SHA-256: {name}')
    if seen != set(expected):
        raise ValueError('Model manifest does not cover every frozen binary')
    packages = json.loads((root / 'models_packages.json').read_text())
    by_package = {}
    for record in packages:
        name = record['package']
        safe_name(name)
        if (name in by_package or '/' in name or not name.endswith('.tar.gz')
                or not 0 < int(record['bytes']) < 2 * 1024 ** 3
                or not re.fullmatch('[0-9a-f]{64}', record['sha256'])
                or record['url'] != DOWNLOAD + name):
            raise ValueError(f'Invalid package metadata: {name}')
        by_package[name] = record
    if set(by_package) != {r['package'] for r in rows}:
        raise ValueError('Package list does not match model manifest')
    for name, record in by_package.items():
        if record['files'] != sum(r['package'] == name for r in rows):
            raise ValueError(f'Wrong package member count: {name}')
    return rows, by_package


def destination(root, name):
    safe_name(name)
    root = Path(root).resolve()
    target = root / name
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f'Refusing symlink destination: {name}')
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Asset escapes repository: {name}')
    return target


def verify_installed(root, rows):
    for row in rows:
        path = destination(root, row['path_in_repo'])
        if (not path.is_file() or path.stat().st_size != row['bytes']
                or sha256(path) != row['sha256']):
            raise ValueError(f'Missing or changed installed model: {row["path_in_repo"]}')
    print(f'Installed files verified: {len(rows)}')


def stage_package(path, record, rows, staging):
    if path.stat().st_size != int(record['bytes']) or sha256(path) != record['sha256']:
        raise ValueError(f'Package size or SHA-256 mismatch: {record["package"]}')
    expected = {r['path_in_repo']: r for r in rows}
    seen = set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = member.name
            safe_name(name)
            if not member.isfile() or name in seen or name not in expected:
                raise ValueError(f'Unexpected, duplicate or non-regular member: {name}')
            row = expected[name]
            if member.size != row['bytes']:
                raise ValueError(f'Wrong extracted size: {name}')
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with archive.extractfile(member) as source, target.open('xb') as output:
                while chunk := source.read(1024 * 1024):
                    count += len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
            if count != row['bytes'] or digest.hexdigest() != row['sha256']:
                raise ValueError(f'Extracted SHA-256 mismatch: {name}')
            seen.add(name)
    if seen != set(expected):
        raise ValueError(f'Package missing members: {record["package"]}')
    print(f'Package and {len(seen)} members verified: {record["package"]}')


def fetch(root=ROOT, from_dir=None, verify_only=False):
    root = Path(root).resolve()
    rows, packages = load_manifest(root)
    # --verify-only alone authenticates an existing installation offline.
    # With --from-dir, also authenticate archives, without installing anything.
    if verify_only and from_dir is None:
        verify_installed(root, rows)
        return
    with tempfile.TemporaryDirectory(prefix='release-models-') as folder:
        temporary = Path(folder)
        staging = temporary / 'staged'
        staging.mkdir()
        for name, record in sorted(packages.items()):
            if from_dir is None:
                path = temporary / name
                request = urllib.request.Request(record['url'], headers={'User-Agent': 'release-v2-model-fetcher'})
                with urllib.request.urlopen(request, timeout=120) as response, path.open('xb') as output:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > int(record['bytes']):
                            raise ValueError(f'Download exceeds recorded size: {name}')
                        output.write(chunk)
            else:
                path = Path(from_dir) / name
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f'Missing regular local package: {name}')
            stage_package(path, record, [r for r in rows if r['package'] == name], staging)
        if verify_only:
            print(f'Archives verified without installing: {len(packages)} packages / {len(rows)} files')
            return
        # Validate every destination before installing any file; never replace a
        # different existing model. No pickle/Keras deserialization takes place.
        targets = [(row, destination(root, row['path_in_repo'])) for row in rows]
        for row, target in targets:
            if target.exists() and (not target.is_file() or target.stat().st_size != row['bytes']
                                    or sha256(target) != row['sha256']):
                raise ValueError(f'Refusing to overwrite a different existing file: {row["path_in_repo"]}')
        for row, target in targets:
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, pending = tempfile.mkstemp(prefix='.model-', dir=target.parent)
            try:
                with os.fdopen(handle, 'wb') as output, (staging / row['path_in_repo']).open('rb') as source:
                    shutil.copyfileobj(source, output)
                os.chmod(pending, 0o644)
                os.replace(pending, target)
            finally:
                if os.path.exists(pending):
                    os.unlink(pending)
        verify_installed(root, rows)
        print(f'Model installation complete: {len(packages)} packages / {len(rows)} files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-dir', type=Path, help='Use local release packages without downloading')
    parser.add_argument('--verify-only', action='store_true',
                        help='Verify installed files; with --from-dir verify archives without installing')
    args = parser.parse_args()
    try:
        fetch(from_dir=args.from_dir, verify_only=args.verify_only)
    except (ValueError, OSError, KeyError, TypeError, tarfile.TarError) as error:
        parser.exit(1, f'Model verification failed: {error}\n')


if __name__ == '__main__':
    main()
