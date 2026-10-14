"""Offline contract and integrity tests for release model assets."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts import fetch_models as models


def test_release_model_manifest_matches_frozen_models():
    rows, packages = models.load_manifest()
    assert len(rows) == 28 and len(packages) == 7
    assert sum(r['role'] == 'fine_tuned_model' for r in rows) == 15
    assert sum(r['role'] == 'fixed_pipeline' for r in rows) == 4
    assert sum(r['role'] == 'encoder' for r in rows) == 4
    assert sum(r['role'] == 'probability_adapter' for r in rows) == 5
    assert {r['seed'] for r in rows if r['role'] == 'fine_tuned_model'} == {'1980', '1981', '1982'}
    assert all(int(p['bytes']) < 2 * 1024 ** 3 for p in packages.values())


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('Offline model tests must not download')
    monkeypatch.setattr(models.urllib.request, 'urlopen', no_network)
    root = tmp_path / 'repo'; root.mkdir()
    assets = tmp_path / 'assets'; assets.mkdir()
    name = 'results/external_test/models/ft_MobileNet__seed_1980/model.keras'
    package = 'models_ft_MobileNet_v2.0.0.tar.gz'
    payload = b'fixture model bytes, never deserialized'
    digest = hashlib.sha256(payload).hexdigest()
    freeze = {'frozen': True, 'smoke': False, 'models': [{'model': 'ft_MobileNet',
              'seed': 1980, 'family': 'fine_tuning', 'architecture': 'MobileNet',
              'model_path': name.removeprefix(models.BASE)}], 'encoders': {},
              'artifacts': {name.removeprefix(models.BASE): digest}}
    f = root / models.BASE / 'final_models.json'; f.parent.mkdir(parents=True)
    f.write_text(json.dumps(freeze))
    row = dict(package=package, path_in_repo=name, bytes=len(payload), sha256=digest,
               model_id='ft_MobileNet', seed='1980', role='fine_tuned_model')
    with (root / 'models_manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=models.FIELDS)
        writer.writeheader(); writer.writerow(row)
    def archive(members):
        path = assets / package
        with tarfile.open(path, 'w:gz') as tar:
            for member_name, data in members:
                info = tarfile.TarInfo(member_name); info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        meta = dict(package=package, bytes=path.stat().st_size,
                    sha256=models.sha256(path), url=models.DOWNLOAD + package, files=1)
        (root / 'models_packages.json').write_text(json.dumps([meta]))
    archive([(name, payload)])
    return root, assets, name, package, payload, archive


def test_local_install_and_offline_verify(bundle):
    root, assets, name, _, payload, _ = bundle
    models.fetch(root, assets)
    assert (root / name).read_bytes() == payload
    models.fetch(root, verify_only=True)
    (root / name).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed installed'):
        models.fetch(root, verify_only=True)


def test_verify_archives_does_not_install(bundle):
    root, assets, name, *_ = bundle
    models.fetch(root, assets, verify_only=True)
    assert not (root / name).exists()


def test_manifest_tamper_rejected(bundle):
    root, *_ = bundle
    path = root / 'models_manifest.csv'
    text = path.read_text().replace('ft_MobileNet,1980', 'wrong_model,1980')
    path.write_text(text)
    with pytest.raises(ValueError, match='differs from frozen'):
        models.load_manifest(root)


def test_incomplete_manifest_rejected(bundle):
    root, *_ = bundle
    (root / 'models_manifest.csv').write_text(','.join(models.FIELDS) + '\n')
    with pytest.raises(ValueError, match='every frozen binary'):
        models.load_manifest(root)


def test_package_tamper_rejected_without_install(bundle):
    root, assets, name, package, *_ = bundle
    with (assets / package).open('ab') as stream:stream.write(b'tampered')
    with pytest.raises(ValueError, match='Package size or SHA-256'):
        models.fetch(root, assets)
    assert not (root / name).exists()


def test_payload_tamper_rejected_even_with_valid_package_hash(bundle):
    root, assets, name, _, payload, archive = bundle
    archive([(name, b'x' * len(payload))])
    with pytest.raises(ValueError, match='Extracted SHA-256'):
        models.fetch(root, assets)
    assert not (root / name).exists()


@pytest.mark.parametrize('member', ['../escape', '/absolute/escape', 'unexpected.keras'])
def test_unsafe_or_unexpected_member_rejected(bundle, member):
    root, assets, name, _, payload, archive = bundle
    archive([(member, payload)])
    with pytest.raises(ValueError, match='Unsafe asset path|Unexpected'):
        models.fetch(root, assets)
    assert not (root / name).exists()


def test_symlink_destination_rejected(bundle, tmp_path):
    root, assets, name, *_ = bundle
    outside = tmp_path / 'outside'; outside.mkdir()
    (root / 'results/external_test/models').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink destination'):
        models.fetch(root, assets)
    assert not list(outside.iterdir())


def test_existing_different_file_not_overwritten(bundle):
    root, assets, name, *_ = bundle
    target = root / name; target.parent.mkdir(parents=True);target.write_bytes(b'keep existing')
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        models.fetch(root, assets)
    assert target.read_bytes() == b'keep existing'
