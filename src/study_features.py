"""Bounded, checksum-checked globally batched fixed ImageNet features."""
from __future__ import annotations
import gc
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from .artifact import _tensorflow_model, dataset_content_fingerprint
from .fixed_refit import canonical_fixed_features
from .study_io import ROOT, sha256


class CanonicalFeatureStore:
    def __init__(self, paths, *, cache=None, max_loaded=2, device='cpu', threads=2):
        self.paths = paths
        self.cache = Path(cache or ROOT / 'cache/fixed_canonical_features')
        self.max_loaded, self.device, self.threads = max_loaded, device, threads
        self.content = dataset_content_fingerprint(paths)
        self.loaded, self.provenance = OrderedDict(), {}
        self.tf = None
        reference = json.loads((ROOT / 'results/full_search/run_metadata.json').read_text())
        if self.content != reference['dataset_content_sha256']:
            raise ValueError('Only the unchanged canonical VIA images may enter this feature store')

    def get(self, architecture):
        if architecture in self.loaded:
            self.loaded.move_to_end(architecture)
            return self.loaded[architecture]
        matches = []
        for path in sorted(self.cache.glob('*/metadata.json')):
            meta = json.loads(path.read_text())
            identity = meta['identity']
            if (identity['architecture'] == architecture and identity['dataset_sha256'] == self.content
                    and identity['batch_size'] == 4 and identity['source_sha256'] == sha256(ROOT / 'src/fixed_refit.py')):
                matches.append((path, meta))
        if matches:
            checksums = {meta['features_sha256'] for _, meta in matches}
            if len(checksums) != 1:
                raise ValueError('Multiple numerical canonical caches exist; choose an explicit cache directory')
            path, meta = matches[0]
            reference = json.loads((ROOT/'results/fine_tuning_recovered/run_metadata.json').read_text())
            weight_reference = reference.get('imagenet_weights', {}).get(architecture)
            if weight_reference and (meta['identity']['weights_sha256'] != weight_reference['sha256']
                                     or sha256(weight_reference['path']) != weight_reference['sha256']):
                raise ValueError('Canonical cache/native ImageNet weight identities differ')
            array_path = path.parent / 'features.npy'
            if sha256(array_path) != meta['features_sha256']:
                raise ValueError('Canonical feature checksum changed')
            matrix = np.load(array_path, mmap_mode='r', allow_pickle=False)
            if list(matrix.shape) != meta['shape'] or len(matrix) != len(self.paths) or not np.isfinite(matrix).all():
                raise ValueError('Canonical feature coverage changed')
            # Downstream CPU smoke reuses original GPU forwards; origin is explicitly preserved.
            self.provenance[architecture] = {'metadata_path': str(path.relative_to(ROOT)), **meta, 'reused': True}
        else:
            from .fine_tuning import configure_tensorflow, imagenet_weight_manifest
            if self.tf is None:
                self.tf = configure_tensorflow(1980, self.device, self.threads)
            model, processor, size = _tensorflow_model(architecture)
            model.trainable = False
            if architecture in {'MobileNet', 'ResNet50', 'ResNet101V2', 'EfficientNetB3', 'EfficientNetB4'}:
                weights = imagenet_weight_manifest([architecture])[architecture]
            else:
                # The canonical helper also serves the other 20 ImageNet extractors.
                import hashlib
                checksum = hashlib.sha256()
                for weight in model.get_weights():
                    checksum.update(str((weight.dtype.str, weight.shape)).encode())
                    checksum.update(weight.tobytes())
                weights = {'sha256': checksum.hexdigest(), 'scope': 'native ImageNet tensors'}
            request = {'extractors': [architecture], 'imagenet_weights': {architecture: weights},
                       'fold_assignments': str(ROOT / 'results/full_search/fold_assignments.csv')}
            values, provenance = canonical_fixed_features(request, self.paths, [model], [(processor, size)],
                                                         self.tf, self.cache)
            matrix = values[0]
            self.provenance[architecture] = provenance
            del model
            self.tf.keras.backend.clear_session()
            gc.collect()
        self.loaded[architecture] = matrix
        while len(self.loaded) > self.max_loaded:
            self.loaded.popitem(last=False)
        return matrix

    def feature_map(self, names):
        return {name: self.get(name) for name in names}
