#!/usr/bin/env python3
"""Matched ImageNet fine-tuning and end-to-end fixed-feature cost measurements."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
import resource
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

try:
    from artifact import (
        MODEL_SPECS, SEED, _tensorflow_model, build_classifier, continuous_scores,
        dataset_content_fingerprint, dataset_fingerprint, prediction_metrics,
        scan_dataset, summarise,
    )
    from run_experiments import dependency_versions, hardware_metadata, write_json
except ModuleNotFoundError:
    from src.artifact import (
        MODEL_SPECS, SEED, _tensorflow_model, build_classifier, continuous_scores,
        dataset_content_fingerprint, dataset_fingerprint, prediction_metrics,
        scan_dataset, summarise,
    )
    from src.run_experiments import dependency_versions, hardware_metadata, write_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_outer_folds(path, image_paths, labels):
    """Consume the recorded index, never regenerate an outer split."""
    assignments = pd.read_csv(path).sort_values("sample_index")
    required = {"sample_index", "filename", "label", "test_fold"}
    if not required.issubset(assignments):
        raise ValueError("Invalid fold assignment columns")
    expected_indices = np.arange(len(image_paths))
    if not np.array_equal(assignments["sample_index"], expected_indices):
        raise ValueError("Fold assignments must cover the dataset index exactly once")
    if assignments["filename"].tolist() != [p.name for p in image_paths]:
        raise ValueError("Fold assignments do not match sorted dataset filenames")
    if not np.array_equal(assignments["label"], labels):
        raise ValueError("Fold assignment labels differ from dataset labels")
    if set(assignments["test_fold"]) != set(range(1, 11)):
        raise ValueError("The matched protocol requires the ten recorded outer folds")
    splits = []
    for fold in range(1, 11):
        mask = assignments["test_fold"].to_numpy() == fold
        train, test = expected_indices[~mask], expected_indices[mask]
        if set(labels[train]) != {0, 1} or set(labels[test]) != {0, 1}:
            raise ValueError(f"Both classes are required in outer fold {fold}")
        splits.append((train, test))
    return splits


def inner_split(outer_train, labels, fraction=0.2, seed=SEED):
    """One fixed stratified validation partition shared by all initialization seeds."""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    train, validation = train_test_split(
        np.asarray(outer_train), test_size=fraction, random_state=seed,
        shuffle=True, stratify=labels[outer_train],
    )
    return np.sort(train), np.sort(validation)


def label_update_indices(outer_train, fit_train):
    """The deterministic new label is in internal fit, never validation/test."""
    outer_train, fit_train = np.asarray(outer_train), np.asarray(fit_train)
    if (outer_train.ndim != 1 or fit_train.ndim != 1 or len(fit_train) == 0
            or len(np.unique(outer_train)) != len(outer_train)
            or not np.isin(fit_train, outer_train).all()):
        raise ValueError("Internal fit must be a nonempty subset of unique outer training indices")
    image = int(fit_train[-1])
    return outer_train[outer_train != image], image


def check_reload_values(expected, actual, filenames, kind="probabilities"):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if (expected.shape != actual.shape or expected.ndim == 0 or expected.size == 0
            or expected.shape[0] != len(filenames)
            or not np.isfinite(expected).all() or not np.isfinite(actual).all()
            or not np.allclose(expected, actual, atol=1e-5, rtol=0)):
        raise ValueError(f"Saved-model reload changed held-out {kind}")
    return {"status": "passed", "image_count": len(filenames), "filenames": list(filenames),
            "max_abs_difference": float(np.max(np.abs(expected - actual))),
            "atol": 1e-5, "kind": kind}


def validate_results(records, predictions, assignments, experiments, seeds, folds=None):
    """Reject incomplete coverage, identity mismatches and inconsistent metrics."""
    folds = list(folds if folds is not None else sorted(assignments["test_fold"].unique()))
    keys = ["experiment", "seed", "fold"]
    expected_jobs = {(e, s, f) for e in experiments for s in seeds for f in folds}
    observed_jobs = set(records[keys].itertuples(index=False, name=None))
    if records.duplicated(keys).any() or observed_jobs != expected_jobs:
        raise ValueError("Incomplete or duplicate experiment/seed/fold metrics")
    if predictions.duplicated(["experiment", "seed", "sample_index"]).any():
        raise ValueError("Duplicate image predictions")
    expected_images = assignments[assignments["test_fold"].isin(folds)].sort_values("sample_index")
    if set(predictions["experiment"]) != set(experiments) or set(predictions["seed"]) != set(seeds):
        raise ValueError("Unexpected prediction experiment or seed")
    for experiment in experiments:
        for seed in seeds:
            group = predictions[(predictions["experiment"] == experiment) & (predictions["seed"] == seed)]
            group = group.sort_values("sample_index")
            for actual, expected in [("sample_index", "sample_index"), ("filename", "filename"),
                                     ("true_label", "label"), ("fold", "test_fold")]:
                if not np.array_equal(group[actual].to_numpy(), expected_images[expected].to_numpy()):
                    raise ValueError(f"Image coverage/identity mismatch: {experiment}/{seed}/{actual}")
            scores = group["score_clear"].to_numpy(dtype=float)
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError("Fine-tuning scores must be finite clear-class probabilities")
            if not np.array_equal(group["predicted_label"], (scores > 0.5).astype(int)):
                raise ValueError("Predictions must use softmax argmax, ties assigned to obstructed")
    for _, record in records.iterrows():
        group = predictions[(predictions["experiment"] == record["experiment"])
                            & (predictions["seed"] == record["seed"])
                            & (predictions["fold"] == record["fold"])]
        metrics = prediction_metrics(group["true_label"], group["predicted_label"], group["score_clear"])
        for name, value in metrics.items():
            if name not in records or not np.isclose(record[name], value, atol=1e-10, rtol=1e-10, equal_nan=True):
                raise ValueError(f"Recalculated metric mismatch: {record['experiment']}/{record['seed']}/{record['fold']}/{name}")
        if "test_size" in records and record["test_size"] != len(group):
            raise ValueError("Incorrect test_size in per-fold metrics")


def configure_tensorflow(seed, device, threads):
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    import tensorflow as tf

    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise RuntimeError("--device gpu requested, but TensorFlow detects no GPU")
    if gpus:
        tf.config.set_visible_devices(gpus[:1], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def runtime_metadata(tf):
    hardware = hardware_metadata()
    build = tf.sysconfig.get_build_info()
    return {
        "python": sys.version, "platform": platform.platform(),
        "dependencies": dependency_versions(), "hardware": hardware,
        "tensorflow_version": tf.__version__, "cuda_version": build.get("cuda_version"),
        "cudnn_version": build.get("cudnn_version"),
        "gpu_models": [g["device_name"] for g in hardware["gpus"]],
        "training_device": "GPU:0" if tf.config.list_logical_devices("GPU") else "CPU:0",
        "benchmark_devices": ["cpu", "gpu"] if tf.config.list_logical_devices("GPU") else ["cpu"],
        "TF_DETERMINISTIC_OPS": os.environ["TF_DETERMINISTIC_OPS"],
        "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
    }


def preprocessing(architecture):
    _, module, image_size = MODEL_SPECS[architecture]
    return importlib.import_module(f"tensorflow.keras.applications.{module}").preprocess_input, image_size


def imagenet_weight_manifest(architectures, *, download=False):
    """Check or prefetch native no-top weights without constructing models."""
    import os
    import tempfile
    from keras.src.applications import efficientnet, mobilenet, resnet
    from keras.src.backend import config as keras_config
    from keras.src.utils import file_utils

    supported = {"ResNet101V2", "EfficientNetB3", "MobileNet", "ResNet50", "EfficientNetB4"}
    architectures = list(architectures)
    unsupported = set(architectures) - supported
    if unsupported:
        raise ValueError(f"Unsupported ImageNet weight architecture(s): {sorted(unsupported)}")

    # Match get_file's cache resolution, including its unwritable-home fallback.
    cache_home = os.path.expanduser(keras_config.keras_home())
    if not os.access(cache_home, os.W_OK):
        cache_home = os.path.join(
            "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir(), ".keras"
        )
    cache = Path(cache_home) / "models"
    weights = {}
    for architecture in architectures:
        if architecture in {"ResNet101V2", "ResNet50"}:
            name = architecture.lower()
            filename = f"{name}_weights_tf_dim_ordering_tf_kernels_notop.h5"
            source_url = resnet.BASE_WEIGHTS_PATH + filename
            file_hash = resnet.WEIGHTS_HASHES[name][1]
        elif architecture in {"EfficientNetB3", "EfficientNetB4"}:
            name = architecture.lower()
            filename = f"{name}_notop.h5"
            source_url = efficientnet.BASE_WEIGHTS_PATH + filename
            file_hash = efficientnet.WEIGHTS_HASHES[name[-2:]][1]
        else:
            filename = "mobilenet_1_0_224_tf_no_top.h5"
            source_url = mobilenet.BASE_WEIGHT_PATH + filename
            # Installed Keras publishes no reference checksum for MobileNet.
            file_hash = None
        weights[architecture] = (filename, source_url, file_hash)

    if not download:
        missing = [
            str(cache / filename)
            for filename, _, _ in weights.values()
            if not (cache / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "ImageNet weights are missing: " + ", ".join(missing)
                + ". Run with --prefetch-weights before training."
            )

    manifest = {}
    for architecture, (filename, source_url, file_hash) in weights.items():
        if download:
            path = Path(file_utils.get_file(
                filename, source_url, cache_subdir="models", file_hash=file_hash
            ))
        else:
            path = cache / filename
            if file_hash is not None and not file_utils.validate_file(path, file_hash):
                raise ValueError(
                    f"ImageNet weights checksum mismatch: {path}. "
                    "Run with --prefetch-weights to repair the cache before training."
                )
        manifest[architecture] = {
            "filename": filename, "path": str(path.resolve()),
            "sha256": sha256(path), "source_url": source_url,
            "keras_file_hash": file_hash,
            "keras_hash_algorithm": "md5" if file_hash is not None else None,
        }
    return manifest


def image_batch(paths, preprocess_input, image_size, tf):
    arrays = []
    for path in paths:
        image = tf.keras.utils.load_img(path, target_size=image_size)
        arrays.append(tf.keras.utils.img_to_array(image))
        image.close()
    return preprocess_input(np.asarray(arrays, dtype=np.float32))


def batch_dataset(paths, labels, indices, architecture, batch_size, tf):
    preprocess_input, image_size = preprocessing(architecture)

    def generate():
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset:offset + batch_size]
            yield image_batch([paths[i] for i in batch], preprocess_input, image_size, tf), labels[batch]

    dataset = tf.data.Dataset.from_generator(
        generate, output_signature=(tf.TensorSpec((None, *image_size, 3), tf.float32),
                                    tf.TensorSpec((None,), tf.int64)),
    )
    dataset = dataset.apply(tf.data.experimental.assert_cardinality((len(indices) + batch_size - 1) // batch_size))
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    return dataset.with_options(options).prefetch(1)


def build_fine_tuning_model(architecture, configuration, config, tf):
    backbone, _, _ = _tensorflow_model(architecture)
    spec = config["configuration_specs"][configuration]
    backbone.trainable = spec["backbone_trainable"]
    features = backbone.output if spec["backbone_trainable"] else backbone(backbone.input, training=False)
    output = tf.keras.layers.GlobalAveragePooling2D()(features)
    output = tf.keras.layers.Dense(**config["head"])(output)
    output = tf.keras.layers.Dense(2, activation="softmax")(output)
    model = tf.keras.Model(backbone.input, output)
    optimizer = {"adam": tf.keras.optimizers.Adam, "rmsprop": tf.keras.optimizers.RMSprop}[spec["optimizer"]]
    model.compile(optimizer=optimizer(learning_rate=spec["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def gpu_peak(tf, reset=False):
    if not tf.config.list_logical_devices("GPU"):
        return None
    try:
        if reset:
            tf.config.experimental.reset_memory_stats("GPU:0")
        return int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except (ValueError, NotImplementedError):
        return None


def host_peak_bytes():
    """Fresh-process high-water RSS; macOS reports bytes, Linux reports KiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def save_deployment(model, path):
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.keras")
    optimizer, compiled = getattr(model, "optimizer", None), model.compiled
    try:
        # Native Keras ignores include_optimizer=False; omit it without copying weights.
        if optimizer is not None:
            model.optimizer = None
        model.compiled = False
        model.save(temporary)
        temporary.replace(path)
    finally:
        if optimizer is not None:
            model.optimizer = optimizer
        model.compiled = compiled


def callback_state(callback, fields):
    return {field: (None if getattr(callback, field) is None else float(getattr(callback, field)))
            for field in fields}


def train_neural(request, directory, manifest, paths, labels, train, validation, test, tf):
    config, seed = request["config"], request["seed"]
    model_path = directory / "model.keras"
    if manifest.get("state") == "trained":
        return tf.keras.models.load_model(model_path, compile=False), manifest["training_costs"]
    batch_size = int(manifest.get("batch_size", config["batch_size"]))
    while True:
        tf.keras.backend.clear_session()
        gc.collect()
        tf.keras.utils.set_random_seed(seed)
        checkpoint = directory / manifest["checkpoint"] if manifest.get("checkpoint") else None
        state = json.loads((checkpoint / "state.json").read_text()) if checkpoint else {
            "epoch": 0, "history": [], "training_seconds": 0.0, "peak_gpu_memory_bytes": None,
            "peak_host_memory_bytes": 0, "stopped": False,
        }
        model = (tf.keras.models.load_model(checkpoint / "last.keras") if checkpoint
                 else build_fine_tuning_model(request["architecture"], request["configuration"], config, tf))
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(**config["reduce_lr"])
        early_stop = tf.keras.callbacks.EarlyStopping(**config["early_stopping"])
        for callback in (reduce_lr, early_stop):
            callback.set_model(model)
            callback.on_train_begin()
        if checkpoint:
            for callback, key in ((reduce_lr, "reduce_lr"), (early_stop, "early_stopping")):
                for field, value in state[key].items():
                    setattr(callback, field, int(value) if field in {"wait", "cooldown_counter", "best_epoch", "stopped_epoch"} else value)
            model.load_weights(checkpoint / "best.weights.h5")
            early_stop.best_weights = model.get_weights()
            model.load_weights(checkpoint / "last.keras")
        gpu_peak(tf, reset=True)
        try:
            for epoch in range(state["epoch"], config["epochs"]):
                if state["stopped"]:
                    break
                # Epoch-keyed order and seed do not depend on held-out predictions.
                tf.keras.utils.set_random_seed(seed + epoch)
                order = np.random.default_rng(seed + epoch).permutation(train)
                training_data = batch_dataset(paths, labels, order, request["architecture"], batch_size, tf)
                validation_data = batch_dataset(paths, labels, validation, request["architecture"], batch_size, tf)
                started = perf_counter()
                history = model.fit(training_data, validation_data=validation_data,
                                    epochs=1, verbose=config["verbose"])
                epoch_seconds = perf_counter() - started
                state["training_seconds"] += epoch_seconds
                logs = {key: float(values[-1]) for key, values in history.history.items()}
                logs["epoch_seconds"] = epoch_seconds
                del history  # History retains its model; release it before an OOM retry.
                logs["learning_rate_before_schedule"] = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
                reduce_lr.on_epoch_end(epoch, logs)
                early_stop.on_epoch_end(epoch, logs)
                logs["learning_rate"] = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
                state["history"].append({"epoch": epoch + 1, **logs})
                state["epoch"] = epoch + 1
                state["stopped"] = bool(model.stop_training)
                state["peak_host_memory_bytes"] = max(state["peak_host_memory_bytes"], host_peak_bytes())
                peak = gpu_peak(tf)
                if peak is not None:
                    state["peak_gpu_memory_bytes"] = max(peak, state["peak_gpu_memory_bytes"] or 0)
                state["reduce_lr"] = callback_state(reduce_lr, ["wait", "best", "cooldown_counter"])
                state["early_stopping"] = callback_state(early_stop, ["wait", "best", "best_epoch", "stopped_epoch"])
                destination = directory / f"checkpoint_{epoch + 1:03d}"
                staging = directory / f"checkpoint_{epoch + 1:03d}.tmp"
                if staging.exists():
                    shutil.rmtree(staging)
                staging.mkdir()
                model.save(staging / "last.keras")
                if early_stop.best_epoch == epoch:
                    model.save_weights(staging / "best.weights.h5")
                else:
                    shutil.copyfile(checkpoint / "best.weights.h5", staging / "best.weights.h5")
                write_json(staging / "state.json", state)
                if destination.exists():
                    shutil.rmtree(destination)  # An uncommitted generation from an interrupted save.
                staging.replace(destination)
                previous = checkpoint
                checkpoint = destination
                manifest.update(checkpoint=destination.name, batch_size=batch_size, state="running")
                write_json(directory / "manifest.json", manifest)
                if previous is not None and previous != destination:
                    shutil.rmtree(previous)
                atomic_csv(pd.DataFrame(state["history"]), directory / "history.csv")
            peak_host = max(state["peak_host_memory_bytes"], host_peak_bytes())
            early_stop.on_train_end()
            check_paths = [paths[int(index)] for index in test[:8]]
            processor, image_size = preprocessing(request["architecture"])
            check_batch = image_batch(check_paths, processor, image_size, tf)
            expected = model(check_batch, training=False).numpy()
            save_deployment(model, model_path)
            restored = tf.keras.models.load_model(model_path, compile=False)
            actual = restored(check_batch, training=False).numpy()
            reload_check = check_reload_values(expected, actual, [path.name for path in check_paths])
            if not np.array_equal(expected[:, 1] > 0.5, actual[:, 1] > 0.5):
                raise ValueError("Saved-model reload changed held-out neural decisions")
            reload_check["decisions_identical"] = True
            del restored, check_batch
            peak = state["peak_gpu_memory_bytes"]
            if manifest.get("oom_peak_gpu_memory_bytes") is not None:
                peak = max(peak or 0, manifest["oom_peak_gpu_memory_bytes"])
            costs = {
                "training_seconds": state["training_seconds"] + manifest.get("oom_training_seconds", 0.0),
                "effective_epochs": state["epoch"],
                "best_epoch": int(early_stop.best_epoch) + 1, "batch_size": batch_size,
                "peak_gpu_memory_bytes": peak,
                "peak_gpu_memory_status": "measured" if peak is not None else "unavailable",
                "peak_host_memory_bytes": peak_host,
                "total_parameters": model.count_params(),
                "trainable_parameters": sum(int(np.prod(v.shape)) for v in model.trainable_weights),
                "model_size_mb": model_path.stat().st_size / 1e6,
                "reload_check": reload_check,
            }
            manifest.update(state="trained", training_costs=costs)
            write_json(directory / "manifest.json", manifest)
            return model, costs
        except tf.errors.ResourceExhaustedError:
            if batch_size == 1:
                raise
            old_batch = batch_size
            batch_size = max(1, batch_size // 2)
            manifest["oom_training_seconds"] = manifest.get("oom_training_seconds", 0.0) + perf_counter() - started
            peak = gpu_peak(tf)
            if peak is not None:
                manifest["oom_peak_gpu_memory_bytes"] = max(peak, manifest.get("oom_peak_gpu_memory_bytes", 0))
            manifest.setdefault("oom_reductions", []).append({"from": old_batch, "to": batch_size, "epoch": state["epoch"] + 1})
            manifest["batch_size"] = batch_size
            # Only the committed checkpoint is reusable; discard a partially updated epoch.
            write_json(directory / "manifest.json", manifest)
            print(f"OOM: {request['experiment']} fold {request['fold']} seed {seed}; batch {old_batch} -> {batch_size}", flush=True)
            del model, early_stop, reduce_lr


def job_fields(request):
    return {key: request[key] for key in ["experiment", "architecture", "configuration", "family", "seed", "fold", "approach", "classifier"]}


def prediction_rows(request, paths, labels, test, predicted, scores):
    return [{**job_fields(request), "sample_index": int(index), "filename": paths[index].name,
             "true_label": int(labels[index]), "predicted_label": int(predicted[position]),
             "score_clear": float(scores[position]), "correct": bool(labels[index] == predicted[position])}
            for position, index in enumerate(test)]


def transform_fixed(pipeline, parts):
    reducers = pipeline["reducers"]
    if not reducers:
        return np.hstack(parts)
    if pipeline["approach"] == "D":
        return np.hstack([reducer.transform(part) for reducer, part in zip(reducers, parts)])
    return reducers[0].transform(np.hstack(parts))


def fit_fixed_pipeline(experiment, matrices, truth, seed):
    """Retain fitted PCA objects so benchmarking includes the original transformation."""
    reducers = []
    started = perf_counter()
    if experiment["approach"] == "A":
        prepared = np.hstack(matrices)
    elif experiment["approach"] == "D":
        parts = []
        for matrix in matrices:
            reducer = PCA(n_components=experiment["components"], random_state=seed)
            parts.append(reducer.fit_transform(matrix))
            reducers.append(reducer)
        prepared = np.hstack(parts)
    else:
        reducer = PCA(n_components=experiment["components"], random_state=seed)
        prepared = reducer.fit_transform(np.hstack(matrices))
        reducers = [reducer]
    reduction_seconds = perf_counter() - started
    classifier = build_classifier(experiment["classifier"], seed)
    started = perf_counter()
    classifier.fit(prepared, truth)
    classifier_seconds = perf_counter() - started
    pipeline = {"approach": experiment["approach"], "reducers": reducers, "classifier": classifier}
    parameters = sum(reducer.components_.size + reducer.mean_.size for reducer in reducers)
    attributes = ["support_vectors_", "dual_coef_", "intercept_"] if hasattr(classifier, "support_vectors_") else ["coef_", "intercept_"]
    parameters += sum(np.asarray(getattr(classifier, name)).size for name in attributes)
    return pipeline, prepared.shape[1], reduction_seconds, classifier_seconds, parameters


def benchmark_saved(request, directory, model_files, tf, pipeline=None):
    """Batch-one JPEG-to-decision latency, including host transforms and synchronization."""
    paths, labels = scan_dataset(request["dataset"])
    outer_train, _ = load_outer_folds(request["fold_assignments"], paths, labels)[request["fold"] - 1]
    image_path = paths[int(outer_train[0])]
    result = {"latency_filename": image_path.name, "latency_batch_size": 1,
              "latency_repetitions": request["config"]["benchmark"]["repetitions"],
              "latency_warmup": request["config"]["benchmark"]["warmup"]}
    architectures = request.get("extractors", [request["architecture"]])
    for device in ("cpu", "gpu"):
        if device == "gpu" and not tf.config.list_logical_devices("GPU"):
            result.update(gpu_latency_median_ms=None, gpu_latency_p95_ms=None, gpu_latency_status="not_available")
            continue
        tf.keras.backend.clear_session()
        gc.collect()
        with tf.device(f"/{device.upper()}:0"):
            models = [tf.keras.models.load_model(directory / name, compile=False) for name in model_files]
            processors = [preprocessing(name) for name in architectures]

            def make_forward(model):
                @tf.function(reduce_retracing=True)
                def forward(batch):
                    return model(batch, training=False)
                return forward

            forwards = [make_forward(model) for model in models]

            def infer():
                parts = [forward(image_batch([image_path], processor, size, tf)).numpy().reshape(1, -1)
                         for forward, (processor, size) in zip(forwards, processors)]
                if pipeline is not None:
                    return pipeline["classifier"].predict(transform_fixed(pipeline, parts))
                return np.argmax(parts[0], axis=1)

            for _ in range(result["latency_warmup"]):
                infer()
            timings = []
            for _ in range(result["latency_repetitions"]):
                started = perf_counter()
                infer()
                timings.append((perf_counter() - started) * 1000)
        result[f"{device}_latency_median_ms"] = float(np.median(timings))
        result[f"{device}_latency_p95_ms"] = float(np.percentile(timings, 95))
        result[f"{device}_latency_status"] = "measured"
        del models, forwards, infer
    tf.keras.backend.clear_session()
    return result


def run_neural_job(request, directory, manifest, paths, labels, outer_train, test, tf):
    train, validation = inner_split(outer_train, labels, request["config"]["validation_fraction"], request["config"]["validation_seed"])
    write_json(directory / "split_indices.json", {"outer_train": outer_train.tolist(), "fit_train": train.tolist(),
               "validation": validation.tolist(), "test": test.tolist(), "validation_seed": request["config"]["validation_seed"]})
    model, costs = train_neural(request, directory, manifest, paths, labels, train, validation, test, tf)
    costs.update(
        adaptation_seconds=costs["training_seconds"], update_timing_source="prescribed_full_retraining",
        update_image_filename=paths[int(train[-1])].name,
        update_previous_labeled_size=len(train) - 1, update_labeled_size=len(train),
        feature_forward_training_seconds=None, feature_extraction_training_seconds=None,
        reduction_seconds=None, classifier_training_seconds=None,
        adaptation_reduction_seconds=None, adaptation_classifier_seconds=None,
    )
    started = perf_counter()
    probabilities = np.concatenate([model(batch, training=False).numpy()
        for batch, _ in batch_dataset(paths, labels, test, request["architecture"], costs["batch_size"], tf)])
    scores = probabilities[:, 1]
    predicted = (scores > 0.5).astype(int)
    record = {**job_fields(request), "train_size": len(train), "outer_train_size": len(outer_train),
              "validation_size": len(validation), "test_size": len(test), "prepared_features": request["config"]["head"]["units"],
              **prediction_metrics(labels[test], predicted, scores), "reduction_seconds": 0.0,
              "training_seconds": costs["training_seconds"], "prediction_seconds": perf_counter() - started}
    atomic_csv(pd.DataFrame(prediction_rows(request, paths, labels, test, predicted, scores)), directory / "predictions.csv")
    write_json(directory / "metrics.json", record)
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    costs.update(benchmark_saved(request, directory, ["model.keras"], tf))
    return {**job_fields(request), **costs}


def run_fixed_job(request, directory, manifest, paths, labels, outer_train, test, tf):
    try:
        from .fixed_refit import canonical_fixed_features
    except ImportError:
        from fixed_refit import canonical_fixed_features

    experiment = request["fixed_experiment"]
    model_files = [f"{name}.keras" for name in experiment["extractors"]]
    if manifest.get("state") != "trained":
        models = [_tensorflow_model(name)[0] for name in experiment["extractors"]]
        for model in models:
            model.trainable = False
        processors = [preprocessing(name) for name in experiment["extractors"]]
        feature_map, canonical = canonical_fixed_features(request, paths, models, processors, tf)
        write_json(directory / "canonical_features.json", canonical)
        batch_size = request["config"]["batch_size"]
        gpu_peak(tf, reset=True)

        def measure_forward(indices):
            nonlocal batch_size
            forward_seconds = 0.0
            offset = 0
            while offset < len(indices):
                indices_batch = indices[offset:offset + batch_size]
                try:
                    for model, (processor, size) in zip(models, processors):
                        batch = image_batch([paths[i] for i in indices_batch], processor, size, tf)
                        forward_started = perf_counter()
                        try:
                            model(batch, training=False).numpy()
                        finally:
                            forward_seconds += perf_counter() - forward_started
                    offset += len(indices_batch)
                except tf.errors.ResourceExhaustedError:
                    if batch_size == 1:
                        raise
                    previous = batch_size
                    batch_size = max(1, batch_size // 2)
                    manifest.setdefault("oom_reductions", []).append({"from": previous, "to": batch_size, "stage": "feature_extraction"})
                    write_json(directory / "manifest.json", manifest)
            return forward_seconds

        started = perf_counter()
        forward_seconds = measure_forward(outer_train)
        extraction_seconds = perf_counter() - started
        fit_train, _ = inner_split(outer_train, labels, request["config"]["validation_fraction"], request["config"]["validation_seed"])
        previous_train, update_image = label_update_indices(outer_train, fit_train)
        write_json(directory / "split_indices.json", {"outer_train": outer_train.tolist(),
            "fit_train": fit_train.tolist(), "update_previous_train": previous_train.tolist(),
            "update_train": outer_train.tolist(), "update_image_index": update_image, "test": test.tolist()})
        # Unmeasured N-1 setup; only the cached full-N PCA/classifier refit is timed.
        baseline, _, _, _, _ = fit_fixed_pipeline(
            experiment, [matrix[previous_train] for matrix in feature_map], labels[previous_train], request["seed"])
        del baseline
        training_features = [matrix[outer_train] for matrix in feature_map]
        update_started = perf_counter()
        pipeline, width, reduction_seconds, classifier_seconds, downstream_parameters = fit_fixed_pipeline(
            experiment, training_features, labels[outer_train], request["seed"])
        adaptation_seconds = perf_counter() - update_started
        update_reduction, update_classifier = reduction_seconds, classifier_seconds
        del training_features
        peak_host = host_peak_bytes()
        peak = gpu_peak(tf)
        prediction_started = perf_counter()
        measure_forward(test)
        x_test = transform_fixed(pipeline, [matrix[test] for matrix in feature_map])
        predicted = pipeline["classifier"].predict(x_test)
        scores = continuous_scores(pipeline["classifier"], x_test)
        record = {**job_fields(request), "train_size": len(outer_train), "test_size": len(test), "prepared_features": width,
                  **prediction_metrics(labels[test], predicted, scores), "reduction_seconds": reduction_seconds,
                  "training_seconds": classifier_seconds, "prediction_seconds": perf_counter() - prediction_started}
        for key, expected in request["reference_metrics"].items():
            if not np.isclose(record[key], expected, atol=1e-8, rtol=1e-8, equal_nan=True):
                raise ValueError(f"Fixed refit differs from canonical full_search: {request['experiment']}/fold {request['fold']}/{key}: {record[key]} != {expected}")
        atomic_csv(pd.DataFrame(prediction_rows(request, paths, labels, test, predicted, scores)), directory / "predictions.csv")
        write_json(directory / "metrics.json", record)
        temporary = directory / "pipeline.joblib.tmp"
        joblib.dump(pipeline, temporary)
        temporary.replace(directory / "pipeline.joblib")
        for model, filename in zip(models, model_files):
            save_deployment(model, directory / filename)
        check_paths = [paths[int(index)] for index in test[:8]]
        check_batches = [image_batch(check_paths, processor, size, tf) for processor, size in processors]
        parts = [model(batch, training=False).numpy().reshape(len(check_paths), -1)
                 for model, batch in zip(models, check_batches)]
        check_x = transform_fixed(pipeline, parts)
        expected_scores = continuous_scores(pipeline["classifier"], check_x)
        expected_decisions = pipeline["classifier"].predict(check_x)
        restored_models = [tf.keras.models.load_model(directory / name, compile=False) for name in model_files]
        restored_pipeline = joblib.load(directory / "pipeline.joblib")
        parts = [model(batch, training=False).numpy().reshape(len(check_paths), -1)
                 for model, batch in zip(restored_models, check_batches)]
        check_x = transform_fixed(restored_pipeline, parts)
        actual_scores = continuous_scores(restored_pipeline["classifier"], check_x)
        reload_check = check_reload_values(expected_scores, actual_scores,
                                           [path.name for path in check_paths], "native_classifier_scores")
        if not np.array_equal(expected_decisions, restored_pipeline["classifier"].predict(check_x)):
            raise ValueError("Saved-model reload changed held-out fixed-pipeline decisions")
        reload_check["decisions_identical"] = True
        del restored_models, restored_pipeline, check_batches, parts, check_x
        costs = {"training_seconds": extraction_seconds + reduction_seconds + classifier_seconds,
                 "feature_extraction_training_seconds": extraction_seconds, "reduction_seconds": reduction_seconds,
                 "feature_forward_training_seconds": forward_seconds,
                 "adaptation_seconds": adaptation_seconds, "adaptation_reduction_seconds": update_reduction,
                 "adaptation_classifier_seconds": update_classifier,
                 "update_timing_source": "measured_cached_pca_classifier_refit",
                 "update_image_filename": paths[update_image].name,
                 "update_previous_labeled_size": len(previous_train), "update_labeled_size": len(outer_train),
                 "classifier_training_seconds": classifier_seconds, "effective_epochs": 0, "batch_size": batch_size,
                 "canonical_feature_policy": canonical["policy"],
                 "canonical_feature_batch_size": canonical["batch_size"],
                 "canonical_feature_setup_seconds": canonical["setup_seconds"],
                 "canonical_feature_setup_in_training_seconds": False,
                 "peak_gpu_memory_bytes": peak, "peak_gpu_memory_status": "measured" if peak is not None else "unavailable",
                 "peak_host_memory_bytes": peak_host,
                 "cnn_parameters": sum(model.count_params() for model in models), "downstream_parameters": downstream_parameters,
                 "total_parameters": sum(model.count_params() for model in models) + downstream_parameters,
                 "trainable_parameters": downstream_parameters,
                 "reload_check": reload_check,
                 "model_size_mb": sum((directory / filename).stat().st_size for filename in model_files + ["pipeline.joblib"]) / 1e6}
        manifest.update(state="trained", training_costs=costs)
        write_json(directory / "manifest.json", manifest)
        del models, feature_map, pipeline
        tf.keras.backend.clear_session()
        gc.collect()
    costs = dict(manifest["training_costs"])
    pipeline = joblib.load(directory / "pipeline.joblib")
    costs.update(benchmark_saved(request, directory, model_files, tf, pipeline))
    return {**job_fields(request), **costs}


def complete_job(directory, manifest, costs):
    write_json(directory / "costs.json", costs)
    artifacts = ["metrics.json", "predictions.csv", "costs.json", "split_indices.json"]
    artifacts += [path.name for path in directory.glob("*.keras")]
    if (directory / "pipeline.joblib").exists():
        artifacts.append("pipeline.joblib")
    if (directory / "canonical_features.json").exists():
        artifacts.append("canonical_features.json")
    manifest.update(state="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                    artifacts={name: sha256(directory / name) for name in artifacts})
    write_json(directory / "manifest.json", manifest)


def completed_job(directory, request):
    path = directory / "manifest.json"
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    if manifest["request"] != request:
        raise ValueError(f"Job identity differs: {directory}")
    if manifest.get("state") != "completed":
        return False
    required = {"metrics.json", "predictions.csv", "costs.json"}
    if not required.issubset(manifest.get("artifacts", {})):
        raise ValueError(f"Completed job has no complete artifact manifest: {directory}")
    for name, digest in manifest["artifacts"].items():
        if not (directory / name).is_file() or sha256(directory / name) != digest:
            raise ValueError(f"Completed job artifact is missing or changed: {directory / name}")
    return True


def worker(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    request = manifest["request"]
    determinism_enabled = False
    try:
        tf = configure_tensorflow(request["seed"], request["device"], request["config"]["cpu_threads"])
        determinism_enabled = True
        for architecture in request.get("extractors", [request["architecture"]]):
            weight = request["imagenet_weights"][architecture]
            if sha256(weight["path"]) != weight["sha256"]:
                raise ValueError(f"ImageNet weights changed before training: {architecture}")
        manifest["runtime"] = runtime_metadata(tf)
        manifest["state"] = manifest.get("state", "running")
        write_json(directory / "manifest.json", manifest)
        paths, labels = scan_dataset(request["dataset"])
        outer_train, test = load_outer_folds(request["fold_assignments"], paths, labels)[request["fold"] - 1]
        run_job = run_fixed_job if request["family"] == "fixed_features" else run_neural_job
        costs = run_job(request, directory, manifest, paths, labels, outer_train, test, tf)
        complete_job(directory, manifest, costs)
    except Exception as error:
        node = getattr(error, "node_def", None)
        operation = getattr(node, "op", None)
        if operation is None:
            match = re.search(r"\[Op:([^\]]+)\]|\{\{node ([^}]+)\}\}", str(error))
            operation = (match.group(1) or match.group(2)) if match else None
        manifest.update(state="failed", failure={"architecture": request["architecture"],
            "exception": type(error).__name__, "message": str(error), "operation": operation,
            "traceback": traceback.format_exc(),
            "determinism_enabled": determinism_enabled, "determinism_disabled": False,
            "determinism_error": "determin" in str(error).lower()})
        write_json(directory / "manifest.json", manifest)
        raise


def validate_config(config):
    margin = config.get("substitution_margin_pp")
    if (isinstance(margin, bool) or not isinstance(margin, (int, float))
            or not np.isfinite(margin) or not 0 < margin < 100):
        raise ValueError("substitution_margin_pp must be finite and between 0 and 100")
    if sorted(config.get("substitution_metrics", [])) != ["accuracy", "obstructed_recall"]:
        raise ValueError("substitution_metrics must be accuracy and obstructed_recall")
    if not config["architectures"] or len(set(config["architectures"])) != len(config["architectures"]):
        raise ValueError("Architectures must be nonempty and unique")
    if set(config["architectures"]) - MODEL_SPECS.keys():
        raise ValueError("Unsupported architecture")
    if not config["configurations"] or set(config["configurations"]) - {"A", "D"} or len(set(config["configurations"])) != len(config["configurations"]):
        raise ValueError("Select unique Breve configurations A and/or D")
    seeds = config["initialization_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Initialization seeds must be unique nonnegative integers")
    for key in ("epochs", "batch_size", "cpu_threads"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("Invalid internal validation fraction")
    if config["reduce_lr"]["monitor"] not in {"val_accuracy", "val_loss"} or config["early_stopping"]["monitor"] not in {"val_accuracy", "val_loss"}:
        raise ValueError("Callbacks may monitor only internal-validation metrics")
    if not config["early_stopping"]["restore_best_weights"]:
        raise ValueError("restore_best_weights is required")
    if any(config["benchmark"][key] < 1 for key in ("warmup", "repetitions")):
        raise ValueError("Benchmark warmup/repetitions must be positive")
    for name in config["configurations"]:
        spec = config["configuration_specs"][name]
        if spec["optimizer"] not in {"adam", "rmsprop"} or spec["learning_rate"] <= 0:
            raise ValueError("Unsupported optimizer or invalid learning rate")
        if bool(spec["backbone_trainable"]) != (name == "D"):
            raise ValueError("D must train the backbone; A must freeze it")


def aggregate_jobs(output, jobs, assignments, experiments, seeds, folds):
    records, predictions, costs, fixed_records, fixed_predictions = [], [], [], [], []
    for directory, request in jobs:
        if not completed_job(directory, request):
            raise ValueError(f"Cannot aggregate incomplete job: {directory}")
        record = json.loads((directory / "metrics.json").read_text())
        prediction = pd.read_csv(directory / "predictions.csv")
        costs.append(json.loads((directory / "costs.json").read_text()))
        if request["family"] == "fixed_features":
            fixed_records.append(record)
            fixed_predictions.append(prediction)
        else:
            records.append(record)
            predictions.append(prediction)
    records, predictions = pd.DataFrame(records), pd.concat(predictions, ignore_index=True)
    validate_results(records, predictions, assignments, experiments, seeds, folds)
    records = records.sort_values(["experiment", "seed", "fold"])
    predictions = predictions.sort_values(["experiment", "seed", "sample_index"])
    atomic_csv(records, output / "per_fold_metrics.csv")
    atomic_csv(predictions, output / "per_image_predictions.csv")
    metrics = list(prediction_metrics(np.array([0, 1]), np.array([0, 1]), np.array([0.0, 1.0])))
    means = records.groupby(["experiment", "architecture", "configuration", "family", "fold"], as_index=False)[metrics].mean()
    means["seeds"] = len(seeds)
    atomic_csv(means, output / "per_fold_seed_mean.csv")
    image_means = predictions.groupby(["experiment", "architecture", "configuration", "family", "fold", "sample_index", "filename", "true_label"], as_index=False).agg(score_clear=("score_clear", "mean"), seeds=("seed", "nunique"))
    image_means["predicted_label"] = (image_means["score_clear"] > 0.5).astype(int)
    image_means["correct"] = image_means["predicted_label"] == image_means["true_label"]
    atomic_csv(image_means, output / "per_image_seed_aggregate.csv")
    atomic_csv(summarise(means), output / "summary_metrics.csv")
    atomic_csv(pd.DataFrame(costs).sort_values(["family", "experiment", "seed", "fold"]), output / "computational_costs.csv")
    if fixed_records:
        atomic_csv(pd.DataFrame(fixed_records), output / "fixed_per_fold_metrics.csv")
        atomic_csv(pd.concat(fixed_predictions, ignore_index=True), output / "fixed_per_image_predictions.csv")


def compare_smoke_runs(current_dir, reference_dir):
    """Exact independent-run prediction equality, not a resumed-job comparison."""
    current_dir, reference_dir = Path(current_dir).resolve(), Path(reference_dir).resolve()
    if current_dir == reference_dir:
        raise ValueError("Smoke repeats require two independent output directories")
    reports, metadata, frames = [], [], []
    columns = ["experiment", "sample_index", "filename", "fold", "seed",
               "true_label", "predicted_label", "score_clear"]
    for directory in (current_dir, reference_dir):
        report = json.loads((directory / "smoke_report.json").read_text())
        meta = json.loads((directory / "run_metadata.json").read_text())
        architectures = report["requested_architectures"]
        if (not report.get("completed") or not report.get("smoke_all")
                or not meta.get("completed") or not meta.get("smoke_all")
                or architectures != meta["config"].get("architectures")
                or set(report["architectures"]) != set(architectures)
                or any(row["status"] != "passed" for row in report["architectures"].values())):
            raise ValueError("Smoke comparison requires both complete five-architecture runs")
        frame = pd.read_csv(directory / "per_image_predictions.csv")
        expected = {f"ft_d_{architecture.lower()}" for architecture in architectures}
        if (not set(columns).issubset(frame) or set(frame["experiment"]) != expected
                or frame.duplicated(["experiment", "sample_index"]).any()
                or not np.isfinite(frame["score_clear"]).all()):
            raise ValueError("Smoke prediction coverage/identity is incomplete")
        reports.append(report)
        metadata.append(meta)
        frames.append(frame.sort_values(["experiment", "sample_index"])[columns].reset_index(drop=True))
    if reports[0]["requested_architectures"] != reports[1]["requested_architectures"]:
        raise ValueError("Smoke architectures differ between runs")
    for key in ("identity", "config", "runtime", "dataset_content_sha256", "source_hashes"):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f"Smoke protocol/data/source/runtime differs: {key}")
    if not frames[0].equals(frames[1]):
        raise ValueError("Smoke native scores, decisions or image identities differ between runs")
    return {"status": "passed", "identical_predictions": True,
            "reference_path": str(reference_dir),
            "architecture_count": len(reports[0]["requested_architectures"]),
            "prediction_rows": len(frames[0])}


def update_smoke_report(report, directory, architecture):
    manifest = json.loads((directory / "manifest.json").read_text())
    row = report["architectures"][architecture]
    if manifest["state"] == "failed":
        row.update(status="failed", determinism={"enabled": manifest["failure"]["determinism_enabled"],
                   "status": "failed_operation" if manifest["failure"]["determinism_error"] else "not_completed"},
                   failure=manifest["failure"], oom_reductions=manifest.get("oom_reductions", []))
        report["completed"] = False
        if manifest.get("checkpoint"):
            state = json.loads((directory / manifest["checkpoint"] / "state.json").read_text())
            row.update(epoch_seconds=state["history"][0]["epoch_seconds"],
                training_seconds=state["training_seconds"], batch_size=manifest["batch_size"],
                peak_gpu_memory_bytes=state["peak_gpu_memory_bytes"],
                peak_host_memory_bytes=state["peak_host_memory_bytes"])
    elif manifest["state"] == "completed":
        cost = json.loads((directory / "costs.json").read_text())
        history = pd.read_csv(directory / "history.csv")
        row.update(status="passed", epoch_seconds=float(history["epoch_seconds"].iloc[0]),
            training_seconds=cost["training_seconds"], batch_size=cost["batch_size"],
            peak_gpu_memory_bytes=cost["peak_gpu_memory_bytes"],
            peak_gpu_memory_status=cost["peak_gpu_memory_status"],
            peak_host_memory_bytes=cost["peak_host_memory_bytes"],
            model_size_mb=cost["model_size_mb"], oom_reductions=manifest.get("oom_reductions", []),
            reload_check=cost["reload_check"],
            determinism={"enabled": True, "status": "operations_completed_repeat_not_checked"},
            latency={key: value for key, value in cost.items() if "latency" in key})


def run(args):
    config_path = Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    smoke = args.smoke or args.smoke_all
    if smoke:
        config.update(configurations=["D"], initialization_seeds=[SEED], epochs=1)
        if args.smoke:
            config["architectures"] = ["MobileNet"]
        config["fixed_features"]["enabled"] = False
    validate_config(config)
    output = Path(args.output_dir).resolve()
    folds_path = Path(config["fold_assignments"]).resolve()
    fixed_results = Path(config["fixed_features"]["results"]).resolve()
    canonical_results = (Path(__file__).resolve().parents[1] / "results/full_search").resolve()
    for protected in (canonical_results, fixed_results):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Output must not overwrite full_search or its ancestors")
    if args.compare_smoke and output == Path(args.compare_smoke).resolve():
        raise ValueError("Smoke repeats require two independent output directories")
    weight_architectures = list(config["architectures"])
    if config["fixed_features"]["enabled"]:
        fixed_config = json.loads(Path(config["fixed_features"]["config"]).read_text())
        for experiment in fixed_config["experiments"]:
            weight_architectures.extend(name for name in experiment["extractors"] if name not in weight_architectures)
    metadata_path = output / "run_metadata.json"
    if args.prefetch_weights:
        previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        if previous and previous.get("stage") != "imagenet_prefetch":
            raise ValueError("Prefetch must not overwrite training metadata; choose a separate output directory")
        weights = imagenet_weight_manifest(weight_architectures, download=True)
        output.mkdir(parents=True, exist_ok=True)
        write_json(metadata_path, {"stage": "imagenet_prefetch", "completed": False,
            "prefetch_complete": True, "config_path": str(config_path), "config_sha256": config_hash,
            "imagenet_weights": weights, "command": sys.argv,
            "completed_utc": datetime.now(timezone.utc).isoformat()})
        print(f"Prefetched {len(weights)} ImageNet weight files; no training started: {metadata_path}")
        return
    paths, labels = scan_dataset(args.dataset)
    splits = load_outer_folds(folds_path, paths, labels)
    assignments = pd.read_csv(folds_path)
    canonical_assignments = pd.read_csv(canonical_results / "fold_assignments.csv")
    if not np.array_equal(assignments.sort_values("sample_index")["test_fold"].to_numpy(),
                          canonical_assignments.sort_values("sample_index")["test_fold"].to_numpy()):
        raise ValueError("Outer assignments must match the canonical full_search folds")
    folds = [1] if smoke else list(range(1, 11))
    content_hash = dataset_content_fingerprint(paths)
    canonical_metadata = json.loads((canonical_results / "run_metadata.json").read_text())
    if canonical_metadata.get("dataset_content_sha256") != content_hash:
        raise ValueError("Images differ from the canonical full_search dataset")
    weights = imagenet_weight_manifest(weight_architectures)
    source_hashes = {"fold_assignments": sha256(folds_path), "fine_tuning": sha256(__file__),
                     "fixed_refit": sha256(Path(__file__).with_name("fixed_refit.py")),
                     "fine_config": config_hash,
                     "artifact": sha256(Path(__file__).with_name("artifact.py")),
                     "run_experiments": sha256(Path(__file__).with_name("run_experiments.py")),
                     "fixed_config": sha256(config["fixed_features"]["config"]),
                     "fixed_metrics": sha256(fixed_results / "per_fold_metrics.csv")}
    identity = hashlib.sha256(json.dumps({"config": config, "sources": source_hashes,
        "dataset": content_hash, "folds": folds, "device": args.device,
        "imagenet_weights": weights}, sort_keys=True).encode()).hexdigest()
    runtime = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--runtime-info", "--device", args.device,
                              "--threads", str(config["cpu_threads"])], capture_output=True, text=True)
    if runtime.returncode:
        if args.smoke_all:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "smoke_report.json", {"completed": False, "smoke_all": True,
                "requested_architectures": config["architectures"], "architectures": {},
                "failure": {"stage": "runtime_setup", "architecture": None,
                            "traceback": runtime.stderr, "determinism_disabled": False}})
        raise RuntimeError(f"TensorFlow runtime setup failed without disabling determinism:\n{runtime.stderr}")
    hardware = json.loads(runtime.stdout)
    output.mkdir(parents=True, exist_ok=True)
    if runtime.stderr:
        print(runtime.stderr, file=sys.stderr, end="")
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if previous and previous.get("stage") == "imagenet_prefetch":
        if previous["config_sha256"] != config_hash or previous["imagenet_weights"] != weights:
            raise ValueError("Prefetched config/weights changed; use a new output directory")
        previous = None
    if args.compare_smoke and previous:
        raise ValueError("Independent repeat requires a fresh output directory, not reused/resumed jobs")
    if previous and previous["identity"] != identity:
        raise ValueError("Output belongs to another protocol/config/dataset; choose a new --output-dir")
    if previous and previous["runtime"] != hardware:
        raise ValueError("Runtime/hardware changed; do not mix measurements in one output directory")
    experiments = [f"ft_{configuration.lower()}_{architecture.lower()}" for configuration in config["configurations"] for architecture in config["architectures"]]
    started_utc = previous["started_utc"] if previous else datetime.now(timezone.utc).isoformat()
    metadata = {"identity": identity, "completed": False, "smoke": smoke, "smoke_all": args.smoke_all,
                "config": config, "imagenet_weights": weights,
                "config_path": str(config_path), "config_sha256": config_hash,
                "substitution_margin_pp": config["substitution_margin_pp"],
                "substitution_metrics": config["substitution_metrics"],
                "substitution_prespecified_utc": started_utc,
                "initialization": "imagenet", "seeds": config["initialization_seeds"], "folds": folds,
                "experiments": experiments, "images": len(paths), "source_hashes": source_hashes,
                "dataset_filename_fingerprint": dataset_fingerprint(paths), "dataset_content_sha256": content_hash,
                "protocol": "recorded_stratified_kfold", "outer_folds": 10, "runtime": hardware,
                **hardware, "command": sys.argv,
                "started_utc": started_utc,
                "decision_rule": "softmax argmax; equal probabilities predict obstructed (class 0)",
                "cost_scope": "Training includes validation, not model saving; fixed training includes CNN forward on outer train + PCA + classifier. Latency includes JPEG/native resizing/preprocess/CNN/host transforms/decision and synchronization.",
                "parameter_definition": "Keras parameters including nontrainable BN state; fixed adds PCA components/mean and classifier coefficients or support vectors/dual coefficients/intercept. Fixed trainable count is downstream fitted scalars.",
                "adaptation_scope": "One outer-training image is newly labeled; fixed refits PCA/classifier on cached features, neural uses its measured prescribed fresh retraining. N-1 to N label sets differ only by that image; no test image is used.",
                "host_memory_scope": "Fresh-process high-water RSS captured at training end; includes framework/model and checkpoint overhead, not a resettable allocator-only peak.",
                "resume_notice": "Committed epoch checkpoints preserve optimizer, callback counters and best weights. Uncommitted epochs restart. Seed/order is epoch keyed; bitwise equality across process restarts is not guaranteed for backbone-internal stochastic RNG state."}
    write_json(metadata_path, metadata)
    smoke_report = None
    if args.smoke_all:
        smoke_report = {key: metadata[key] for key in ("identity", "config", "runtime", "protocol",
            "folds", "seeds", "source_hashes", "dataset_content_sha256")}
        smoke_report.update(completed=False, smoke_all=True, requested_architectures=config["architectures"],
            architectures={name: {"status": "pending"} for name in config["architectures"]},
            repeat_check={"status": "not_checked"})
        write_json(output / "smoke_report.json", smoke_report)
    atomic_csv(assignments, output / "fold_assignments.csv")
    validation_rows = []
    for fold in folds:
        train, _ = splits[fold - 1]
        _, validation = inner_split(train, labels, config["validation_fraction"], config["validation_seed"])
        validation_rows.extend({"fold": fold, "sample_index": int(i), "filename": paths[i].name,
                                "label": int(labels[i]), "validation_seed": config["validation_seed"]} for i in validation)
    atomic_csv(pd.DataFrame(validation_rows), output / "inner_validation_indices.csv")
    common = {"dataset": str(Path(args.dataset).resolve()), "fold_assignments": str(folds_path),
              "identity": identity, "config": config, "device": args.device, "imagenet_weights": weights}
    requests = []
    for configuration in config["configurations"]:
        for architecture in config["architectures"]:
            for fold in folds:
                for seed in config["initialization_seeds"]:
                    requests.append({**common, "experiment": f"ft_{configuration.lower()}_{architecture.lower()}",
                        "architecture": architecture, "configuration": configuration, "approach": configuration,
                        "classifier": config["configuration_specs"][configuration]["optimizer"] + "_dense",
                        "family": "fine_tuning" if configuration == "D" else "frozen_head", "fold": fold, "seed": seed})
    if config["fixed_features"]["enabled"]:
        fixed_config = json.loads(Path(config["fixed_features"]["config"]).read_text())
        fixed_names = [item["name"] for item in fixed_config["experiments"]]
        reference = pd.read_csv(fixed_results / "per_fold_metrics.csv")
        reference = reference[reference["experiment"].isin(fixed_names)]
        metric_names = [name for name in prediction_metrics([0, 1], [0, 1], [0., 1.]) if name in reference]
        for experiment in fixed_config["experiments"]:
            if experiment.get("scale", False) or experiment["reduction"] not in {"full", "pca"}:
                raise ValueError("Cost benchmark supports the four unscaled full/PCA finalists")
            for fold in folds:
                row = reference[(reference["experiment"] == experiment["name"]) & (reference["fold"] == fold)]
                if len(row) != 1:
                    raise ValueError("Missing/duplicate canonical fixed-feature fold")
                requests.append({**common, "experiment": experiment["name"], "extractors": experiment["extractors"],
                    "architecture": "+".join(experiment["extractors"]), "configuration": "fixed", "family": "fixed_features",
                    "approach": experiment["approach"], "classifier": experiment["classifier"], "seed": int(fixed_config["seed"]),
                    "fold": fold, "fixed_experiment": experiment,
                    "reference_metrics": {name: float(row.iloc[0][name]) for name in metric_names}})
    jobs = []
    started = perf_counter()
    for index, request in enumerate(requests, 1):
        directory = output / "jobs" / f"{request['experiment']}__fold_{request['fold']:02d}__seed_{request['seed']}"
        jobs.append((directory, request))
        if completed_job(directory, request):
            print(f"[{index}/{len(requests)}] Reusing completed {directory.name}", flush=True)
            if smoke_report is not None:
                update_smoke_report(smoke_report, directory, request["architecture"])
                write_json(output / "smoke_report.json", smoke_report)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "manifest.json").exists():
            write_json(directory / "manifest.json", {"request": request, "state": "running", "created_utc": datetime.now(timezone.utc).isoformat()})
        if smoke_report is not None:
            smoke_report["architectures"][request["architecture"]]["status"] = "running"
            write_json(output / "smoke_report.json", smoke_report)
        environment = {**os.environ, "PYTHONHASHSEED": str(request["seed"]), "TF_DETERMINISTIC_OPS": "1",
                       "OMP_NUM_THREADS": str(config["cpu_threads"]), "OPENBLAS_NUM_THREADS": str(config["cpu_threads"]),
                       "MKL_NUM_THREADS": str(config["cpu_threads"])}
        if args.device == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(directory)], env=environment, check=True)
        except subprocess.CalledProcessError:
            if smoke_report is not None:
                update_smoke_report(smoke_report, directory, request["architecture"])
                write_json(output / "smoke_report.json", smoke_report)
            raise
        if not completed_job(directory, request):
            raise RuntimeError(f"Worker did not commit job: {directory}")
        if smoke_report is not None:
            update_smoke_report(smoke_report, directory, request["architecture"])
            write_json(output / "smoke_report.json", smoke_report)
        print(f"[{index}/{len(requests)}] Completed {directory.name}", flush=True)
    aggregate_jobs(output, jobs, assignments, experiments, config["initialization_seeds"], folds)
    metadata.update(completed=True, jobs=len(jobs), last_segment_seconds=perf_counter() - started,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(metadata_path, metadata)
    if smoke_report is not None:
        smoke_report["completed"] = True
        write_json(output / "smoke_report.json", smoke_report)
        if args.compare_smoke:
            try:
                smoke_report["repeat_check"] = compare_smoke_runs(output, args.compare_smoke)
                for row in smoke_report["architectures"].values():
                    row["determinism"]["status"] = "passed_exact_independent_repeat"
            except ValueError as error:
                smoke_report["completed"] = False
                smoke_report["repeat_check"] = {"status": "failed", "message": str(error)}
                write_json(output / "smoke_report.json", smoke_report)
                raise
            write_json(output / "smoke_report.json", smoke_report)
    print(f"Completed {'five-architecture smoke' if args.smoke_all else 'smoke' if smoke else 'matched experiment'}: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fine_tuning.json")
    parser.add_argument("--dataset", default="via-dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true", help="One recorded fold, one epoch, MobileNet, seed 1980; no fixed-cost jobs")
    modes.add_argument("--smoke-all", action="store_true", help="Fold 1, one epoch, seed 1980, all configured architectures; no fixed-cost jobs")
    modes.add_argument("--prefetch-weights", action="store_true", help="Download/check all configured ImageNet weights and record SHA-256; do not train")
    parser.add_argument("--compare-smoke", type=Path, help="Compare this independent --smoke-all run with a completed reference directory")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-info", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.compare_smoke and not args.smoke_all:
        parser.error("--compare-smoke requires --smoke-all")
    if args.output_dir is None:
        args.output_dir = ("results/fine_tuning_smoke_all" if args.smoke_all else
                          "results/fine_tuning_prefetch" if args.prefetch_weights else
                          "results/fine_tuning_smoke" if args.smoke else "results/fine_tuning")
    seed = (json.loads((Path(args.worker) / "manifest.json").read_text())["request"]["seed"] if args.worker
            else SEED if args.runtime_info else json.loads(Path(args.config).read_text())["global_seed"])
    if os.environ.get("PYTHONHASHSEED") != str(seed) or os.environ.get("TF_DETERMINISTIC_OPS") != "1":
        environment = {**os.environ, "PYTHONHASHSEED": str(seed), "TF_DETERMINISTIC_OPS": "1"}
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)
    if args.runtime_info:
        tf = configure_tensorflow(seed, args.device, args.threads)
        print(json.dumps(runtime_metadata(tf)))
    elif args.worker:
        worker(args.worker)
    else:
        run(args)


if __name__ == "__main__":
    main()
