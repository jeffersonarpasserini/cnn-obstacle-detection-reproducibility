"""Canonical frozen features, independent of an outer fold's batch membership."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np


POLICY = "globally_ordered_frozen_features_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_batch_size(metadata):
    """Recover the canonical extractor's recorded CLI setting (default four)."""
    command = metadata["command"]
    if command and command[-1] == "--batch-size":
        raise ValueError("Missing canonical feature extraction batch size")
    matches = [int(command[i + 1]) for i, value in enumerate(command[:-1]) if value == "--batch-size"]
    matches += [int(value.split("=", 1)[1]) for value in command if value.startswith("--batch-size=")]
    if len(matches) > 1 or (matches and matches[0] < 1):
        raise ValueError("Invalid canonical feature extraction batch size")
    return matches[0] if matches else 4


def canonical_fixed_features(request, paths, models, processors, tf, cache_root=None):
    """Load/extract image-only features; no fold, label or fitted reducer is used.

    The global setup is reported separately from timed training-image forwards.
    PCA/classifiers still fit exclusively on the outer training indices.
    """
    try:
        from .artifact import dataset_content_fingerprint
        from .fine_tuning import image_batch, runtime_metadata
    except ImportError:
        from artifact import dataset_content_fingerprint
        from fine_tuning import image_batch, runtime_metadata

    reference = json.loads((Path(request["fold_assignments"]).parent / "run_metadata.json").read_text())
    batch_size = reference_batch_size(reference)
    fingerprint = dataset_content_fingerprint(paths)
    if reference["dataset_content_sha256"] != fingerprint:
        raise ValueError("Canonical features require the unchanged reference images")
    if [p.name for p in paths] != sorted(p.name for p in paths):
        raise ValueError("Canonical features require global sorted image order")
    architectures = request["extractors"]
    if not len(architectures) == len(models) == len(processors):
        raise ValueError("Extractor/model/preprocessor counts differ")
    cache_root = Path(cache_root or Path(__file__).resolve().parents[1] / "cache/fixed_canonical_features")
    cache_root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    matrices, provenance = [], []
    for architecture, model, (processor, size) in zip(architectures, models, processors):
        if model.trainable:
            raise ValueError("Canonical extraction requires a frozen ImageNet model")
        identity = {"policy": POLICY, "dataset_sha256": fingerprint, "architecture": architecture,
                    "weights_sha256": request["imagenet_weights"][architecture]["sha256"],
                    "batch_size": batch_size, "runtime": runtime_metadata(tf), "source_sha256": sha256(__file__)}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = cache_root / key
        directory.mkdir(exist_ok=True)
        array_path, metadata_path = directory / "features.npy", directory / "metadata.json"
        shape = (len(paths), int(np.prod(model.output_shape[1:])))
        reused = metadata_path.exists()
        if reused:
            metadata = json.loads(metadata_path.read_text())
            if metadata["identity"] != identity or not array_path.is_file() or sha256(array_path) != metadata["features_sha256"]:
                raise ValueError(f"Canonical feature cache changed: {architecture}")
        else:
            temporary = directory / "features.tmp.npy"
            if array_path.exists() or temporary.exists():
                raise ValueError(f"Incomplete canonical feature cache needs inspection: {directory}")
            values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
            extraction_started = perf_counter()
            for offset in range(0, len(paths), batch_size):
                batch_paths = paths[offset:offset + batch_size]
                batch = image_batch(batch_paths, processor, size, tf)
                outputs = np.asarray(model.predict(batch, verbose=0), dtype=np.float32).reshape(len(batch_paths), -1)
                if outputs.shape != (len(batch_paths), shape[1]) or not np.isfinite(outputs).all():
                    raise ValueError(f"Invalid canonical activations: {architecture}")
                values[offset:offset + len(batch_paths)] = outputs
            values.flush()
            del values
            temporary.replace(array_path)
            metadata = {"identity": identity, "features_sha256": sha256(array_path),
                        "extraction_seconds": perf_counter() - extraction_started, "shape": list(shape)}
            temporary_meta = directory / "metadata.tmp.json"
            temporary_meta.write_text(json.dumps(metadata, indent=2) + "\n")
            temporary_meta.replace(metadata_path)
        matrix = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if matrix.shape != shape or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid canonical feature cache shape/type: {architecture}")
        matrices.append(matrix)
        provenance.append({"architecture": architecture, "cache_key": key,
                           "features_sha256": metadata["features_sha256"], "reused": reused,
                           "original_extraction_seconds": metadata["extraction_seconds"]})
    return matrices, {"policy": POLICY, "batch_size": batch_size, "setup_seconds": perf_counter() - started,
                      "cache": provenance, "scope": "fixed ImageNet forwards only; no labels, PCA or classifier fitting"}
