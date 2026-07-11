"""Leakage-free utilities for the CNN feature-extraction experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SEED = 1980
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

MODEL_SPECS = {
    "VGG16": ("VGG16", "vgg16", (224, 224)),
    "VGG19": ("VGG19", "vgg19", (224, 224)),
    "Xception": ("Xception", "xception", (299, 299)),
    "ResNet50": ("ResNet50", "resnet", (224, 224)),
    "ResNet101": ("ResNet101", "resnet", (224, 224)),
    "ResNet152": ("ResNet152", "resnet", (224, 224)),
    "ResNet50V2": ("ResNet50V2", "resnet_v2", (224, 224)),
    "ResNet101V2": ("ResNet101V2", "resnet_v2", (224, 224)),
    "ResNet152V2": ("ResNet152V2", "resnet_v2", (224, 224)),
    "InceptionV3": ("InceptionV3", "inception_v3", (299, 299)),
    "InceptionResNetV2": ("InceptionResNetV2", "inception_resnet_v2", (299, 299)),
    "MobileNet": ("MobileNet", "mobilenet", (224, 224)),
    "MobileNetV2": ("MobileNetV2", "mobilenet_v2", (224, 224)),
    "DenseNet121": ("DenseNet121", "densenet", (224, 224)),
    "DenseNet169": ("DenseNet169", "densenet", (224, 224)),
    "DenseNet201": ("DenseNet201", "densenet", (224, 224)),
    "NASNetMobile": ("NASNetMobile", "nasnet", (224, 224)),
    "EfficientNetB0": ("EfficientNetB0", "efficientnet", (224, 224)),
    "EfficientNetB1": ("EfficientNetB1", "efficientnet", (240, 240)),
    "EfficientNetB2": ("EfficientNetB2", "efficientnet", (260, 260)),
    "EfficientNetB3": ("EfficientNetB3", "efficientnet", (300, 300)),
    "EfficientNetB4": ("EfficientNetB4", "efficientnet", (380, 380)),
    "EfficientNetB5": ("EfficientNetB5", "efficientnet", (456, 456)),
    "EfficientNetB6": ("EfficientNetB6", "efficientnet", (528, 528)),
    "EfficientNetB7": ("EfficientNetB7", "efficientnet", (600, 600)),
}


def set_global_seed(seed: int = SEED) -> None:
    """Configure deterministic seeds used by Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def scan_dataset(image_dir: str | Path) -> tuple[list[Path], np.ndarray]:
    """Return deterministically ordered image paths and binary labels."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {image_dir}")

    paths = sorted(
        (p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name.lower(),
    )
    if not paths:
        raise ValueError(f"No supported images found in: {image_dir}")

    labels = []
    for path in paths:
        lower = path.name.lower()
        if lower.startswith("clear."):
            labels.append(1)
        elif lower.startswith("nonclear.") or lower.startswith("non-clear."):
            labels.append(0)
        else:
            raise ValueError(f"Cannot infer class from filename: {path.name}")
    return paths, np.asarray(labels, dtype=np.int64)


def dataset_fingerprint(paths: Iterable[Path]) -> str:
    """Hash the ordered filenames to bind a feature cache to a dataset index."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _tensorflow_model(model_name: str):
    import importlib
    import tensorflow as tf

    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unsupported CNN: {model_name}")
    constructor_name, module_name, image_size = MODEL_SPECS[model_name]
    constructor = getattr(tf.keras.applications, constructor_name)
    preprocessing_module = importlib.import_module(
        f"tensorflow.keras.applications.{module_name}"
    )
    preprocess_input = preprocessing_module.preprocess_input
    model = constructor(
        weights="imagenet",
        include_top=False,
        pooling=None,
        input_shape=image_size + (3,),
    )
    return model, preprocess_input, image_size


def extract_or_load_features(
    model_name: str,
    paths: list[Path],
    cache_dir: str | Path,
    batch_size: int = 4,
) -> tuple[np.ndarray, float]:
    """Load a validated feature cache or extract flattened CNN activations."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{model_name}.npz"
    fingerprint = dataset_fingerprint(paths)

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_fingerprint = str(cached["fingerprint"].item())
        if cached_fingerprint != fingerprint:
            raise ValueError(
                f"Feature cache {cache_path} belongs to a different dataset order"
            )
        return np.asarray(cached["features"], dtype=np.float32), 0.0

    import tensorflow as tf

    model, preprocess_input, image_size = _tensorflow_model(model_name)
    batches = []
    started = perf_counter()
    for offset in range(0, len(paths), batch_size):
        arrays = []
        for path in paths[offset : offset + batch_size]:
            image = tf.keras.utils.load_img(path, target_size=image_size)
            arrays.append(tf.keras.utils.img_to_array(image))
        batch = preprocess_input(np.asarray(arrays, dtype=np.float32))
        output = model.predict(batch, verbose=0)
        batches.append(np.asarray(output, dtype=np.float32).reshape(len(arrays), -1))
    elapsed = perf_counter() - started
    features = np.concatenate(batches, axis=0)
    np.savez_compressed(
        cache_path,
        features=features,
        fingerprint=np.asarray(fingerprint),
    )
    return features, elapsed


def make_splits(labels: np.ndarray, protocol: str, n_splits: int, seed: int):
    """Create deterministic, leakage-free validation splits."""
    indices = np.arange(len(labels))
    if protocol == "stratified_kfold":
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(indices, labels))
    if protocol == "loocv":
        return list(LeaveOneOut().split(indices, labels))
    raise ValueError(f"Unknown protocol: {protocol}")


def _scale_train_test(
    x_train: np.ndarray, x_test: np.ndarray, enabled: bool
) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return x_train, x_test
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def reduce_train_test(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    method: str,
    components: int | None,
    seed: int,
    scale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a transformation on training data only, then transform held-out data."""
    method = method.lower()
    x_train, x_test = _scale_train_test(x_train, x_test, scale)
    if method == "full":
        return x_train, x_test
    if not components or components <= 0:
        raise ValueError(f"Positive components required for {method}")

    if method == "pca":
        reducer = PCA(n_components=components, random_state=seed)
    elif method == "umap":
        import umap

        reducer = umap.UMAP(
            n_neighbors=20,
            min_dist=0.1,
            n_components=components,
            metric="euclidean",
            random_state=seed,
        )
    elif method == "relieff":
        try:
            from ReliefF import ReliefF
        except ImportError:
            from skrebate import ReliefF
        reducer = ReliefF(n_features_to_keep=components)
    else:
        raise ValueError(f"Unknown reduction: {method}")

    transformed_train = reducer.fit_transform(x_train, y_train)
    transformed_test = reducer.transform(x_test)
    return transformed_train, transformed_test


def prepare_fold_features(
    experiment: dict,
    feature_map: dict[str, np.ndarray],
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Implement feature-processing Approaches A--D without test-fold fitting."""
    approach = experiment["approach"].upper()
    extractors = experiment["extractors"]
    method = experiment.get("reduction", "full")
    components = experiment.get("components")
    scale = bool(experiment.get("scale", False))
    matrices = [feature_map[name] for name in extractors]

    if approach == "A":
        return np.hstack([m[train] for m in matrices]), np.hstack(
            [m[test] for m in matrices]
        )
    if approach == "B":
        if len(matrices) != 1:
            raise ValueError("Approach B requires one feature extractor")
        return reduce_train_test(
            matrices[0][train], labels[train], matrices[0][test], method,
            components, seed, scale
        )
    if approach == "C":
        x_train = np.hstack([m[train] for m in matrices])
        x_test = np.hstack([m[test] for m in matrices])
        return reduce_train_test(
            x_train, labels[train], x_test, method, components, seed, scale
        )
    if approach == "D":
        if not components:
            raise ValueError("Approach D requires a per-extractor component count")
        # In Approach D, dimensionality reduction is applied independently to
        # every CNN output before concatenation.  Therefore, ``components`` is
        # the number retained from EACH extractor, not the size of the final
        # concatenated vector.  For example, two extractors with
        # ``components=100`` produce a final vector with 200 components.
        components_per_extractor = int(components)
        train_parts = []
        test_parts = []
        for matrix in matrices:
            part_train, part_test = reduce_train_test(
                matrix[train], labels[train], matrix[test], method,
                components_per_extractor, seed, scale
            )
            train_parts.append(part_train)
            test_parts.append(part_test)
        return np.hstack(train_parts), np.hstack(test_parts)
    raise ValueError(f"Unknown approach: {approach}")


def build_classifier(name: str, seed: int):
    """Build the classifier settings documented in the manuscript."""
    name = name.lower()
    classifiers = {
        "decision_tree": lambda: DecisionTreeClassifier(random_state=seed),
        "rbf_svm": lambda: SVC(kernel="rbf", probability=False),
        "linear_svm": lambda: SVC(kernel="linear", C=0.025, probability=False),
        "mlp": lambda: MLPClassifier(random_state=1, max_iter=1000),
        "logistic": lambda: LogisticRegression(max_iter=500, random_state=seed),
        "random_forest": lambda: RandomForestClassifier(random_state=seed),
        "adaboost": lambda: AdaBoostClassifier(random_state=seed),
        "gaussian_nb": GaussianNB,
    }
    if name not in classifiers:
        raise ValueError(f"Unknown classifier: {name}")
    return classifiers[name]()


def continuous_scores(classifier, x_test: np.ndarray) -> np.ndarray | None:
    """Return continuous scores for a conventional ROC-AUC calculation."""
    if hasattr(classifier, "decision_function"):
        score = classifier.decision_function(x_test)
        return np.asarray(score).reshape(-1)
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(x_test)
        return np.asarray(probabilities)[:, 1]
    return None


def evaluate_experiment(
    experiment: dict,
    feature_map: dict[str, np.ndarray],
    labels: np.ndarray,
    splits,
    seed: int,
) -> list[dict]:
    """Evaluate one configuration and return one record per fold."""
    records = []
    for fold, (train, test) in enumerate(splits, start=1):
        reduction_started = perf_counter()
        x_train, x_test = prepare_fold_features(
            experiment, feature_map, labels, train, test, seed
        )
        reduction_time = perf_counter() - reduction_started

        classifier = build_classifier(experiment["classifier"], seed)
        train_started = perf_counter()
        classifier.fit(x_train, labels[train])
        training_time = perf_counter() - train_started

        prediction_started = perf_counter()
        predicted = classifier.predict(x_test)
        scores = continuous_scores(classifier, x_test)
        prediction_time = perf_counter() - prediction_started

        record = {
            "experiment": experiment["name"],
            "approach": experiment["approach"],
            "fold": fold,
            "train_size": len(train),
            "test_size": len(test),
            "accuracy": accuracy_score(labels[test], predicted),
            "f1": f1_score(labels[test], predicted, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(labels[test], predicted),
            "roc_auc": np.nan,
            "reduction_seconds": reduction_time,
            "training_seconds": training_time,
            "prediction_seconds": prediction_time,
        }
        if scores is not None and len(np.unique(labels[test])) == 2:
            record["roc_auc"] = roc_auc_score(labels[test], scores)
        records.append(record)
    return records


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if "experiments" not in config or not config["experiments"]:
        raise ValueError("Configuration must contain at least one experiment")
    return config


def summarise(records: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "f1", "balanced_accuracy", "roc_auc"]
    rows = []
    for experiment, group in records.groupby("experiment", sort=False):
        row = {"experiment": experiment, "folds": len(group)}
        for metric in metrics:
            row[f"{metric}_q1"] = group[metric].quantile(0.25)
            row[f"{metric}_median"] = group[metric].median()
            row[f"{metric}_q3"] = group[metric].quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows)
