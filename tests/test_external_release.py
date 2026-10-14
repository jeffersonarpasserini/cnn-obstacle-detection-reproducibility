"""Public-release integrity checks, including tampering and excluded images."""
import csv
import hashlib
from pathlib import Path

import pytest

from scripts.verify_external_release import verify_release

ROOT = Path(__file__).resolve().parents[1]


def test_actual_public_external_release():
    result = verify_release(ROOT)
    assert result['images'] == 1576
    assert result['unchanged'] == 1560
    assert result['anonymized'] == 16
    assert result['excluded_absent'] == 353


@pytest.fixture
def small_release(tmp_path):
    directory = tmp_path / 'external-data'
    images = directory / 'via-dataset-extended'
    images.mkdir(parents=True)
    evaluated = [b'original clear', b'original obstructed']
    released = [evaluated[0], b'anonymized obstructed']
    names = ['clear.1.jpg', 'nonclear.2.jpg']
    clean = []
    rows = []
    for i, name in enumerate(names):
        original_hash = hashlib.sha256(evaluated[i]).hexdigest()
        output_hash = hashlib.sha256(released[i]).hexdigest()
        (images/name).write_bytes(released[i])
        clean.append({'filename': name, 'final_label': 1-i, 'group_id': f'scene_{i}',
                      'width': 750, 'height': 1000, 'sha256': original_hash, 'excluded': False})
        rows.append({'filename': name, 'final_label': 1-i, 'group_id': f'scene_{i}',
                     'width': 750, 'height': 1000, 'sha256_evaluated': original_hash,
                     'sha256_released': output_hash, 'anonymized': bool(i),
                     'n_blurred_faces': 0, 'n_blurred_plates': i})
    def write(path, records):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    write(directory/'external_manifest_clean.csv', clean)
    write(directory/'external_manifest_raw.csv', [{'filename': name} for name in names+['excluded.jpg']])
    manifest = directory/'external_release_manifest.csv'
    write(manifest, rows)
    expected = {'images': 2, 'clear': 1, 'obstructed': 1, 'groups': 2, 'excluded': 1,
                'anonymized': 1, 'faces': 0, 'plates': 1}
    return tmp_path, hashlib.sha256(manifest.read_bytes()).hexdigest(), expected


def test_unchanged_and_anonymized_hashes_are_verified(small_release):
    root, checksum, expected = small_release
    assert verify_release(root, manifest_sha256=checksum, expected=expected)['unchanged'] == 1


def test_tampered_image_is_rejected(small_release):
    root, checksum, expected = small_release
    (root/'external-data/via-dataset-extended/clear.1.jpg').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Released SHA-256 mismatch'):
        verify_release(root, manifest_sha256=checksum, expected=expected)


def test_tampered_manifest_is_rejected(small_release):
    root, checksum, expected = small_release
    path = root/'external-data/external_release_manifest.csv'
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='manifest SHA-256 mismatch'):
        verify_release(root, manifest_sha256=checksum, expected=expected)


def test_excluded_image_is_rejected(small_release):
    root, checksum, expected = small_release
    (root/'external-data/via-dataset-extended/excluded.jpg').write_bytes(b'excluded')
    with pytest.raises(ValueError, match='Excluded image present'):
        verify_release(root, manifest_sha256=checksum, expected=expected)


def test_unlisted_image_is_rejected(small_release):
    root, checksum, expected = small_release
    (root/'external-data/via-dataset-extended/extra.jpg').write_bytes(b'extra')
    with pytest.raises(ValueError, match='Missing or extra'):
        verify_release(root, manifest_sha256=checksum, expected=expected)


def test_original_manifest_label_change_is_rejected(small_release):
    root, checksum, expected = small_release
    path = root/'external-data/external_manifest_clean.csv'
    text = path.read_text().replace('clear.1.jpg,1,', 'clear.1.jpg,0,')
    path.write_text(text)
    with pytest.raises(ValueError, match='Frozen final_label differs'):
        verify_release(root, manifest_sha256=checksum, expected=expected)
