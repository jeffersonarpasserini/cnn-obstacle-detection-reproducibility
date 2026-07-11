"""Leakage-free utilities for the CNN feature-extraction experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
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
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
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


def dataset_content_fingerprint(paths: Iterable[Path]) -> str:
    """Hash ordered filenames and file contents for run-level provenance."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
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


def _make_relieff(n_features: int, n_jobs: int = 1):
    """Build the supported Relief-F implementation with explicit parallelism."""
    try:
        from skrebate import ReliefF

        return ReliefF(n_features_to_select=n_features, n_jobs=n_jobs)
    except ImportError:
        # Compatibility fallback for the older ``ReliefF`` package.  The
        # reproducibility environment pins scikit-rebate and uses the branch
        # above; the fallback does not expose controlled parallelism.
        from ReliefF import ReliefF

        return ReliefF(n_features_to_keep=n_features)


@dataclass
class ReliefRankingCache:
    """Persist fold-isolated Relief-F rankings for reuse across cutoffs.

    Only the indices of the highest-ranked features are cached.  The training
    and held-out matrices are scaled again for each experiment group, while
    the expensive supervised ranking is fitted once per fold and feature
    family.  Cache files are written atomically so an interrupted fit cannot
    be mistaken for a complete ranking.
    """

    directory: Path
    n_jobs: int = 1

    def __post_init__(self):
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.reported_seconds = 0.0
        self.hits = 0
        self.misses = 0

    def reset_reported_seconds(self) -> None:
        self.reported_seconds = 0.0

    def clear(self) -> None:
        """Discard rankings when an output directory is explicitly restarted."""
        for path in self.directory.glob("*.npz"):
            path.unlink()
        status = self.directory / "status.json"
        if status.exists():
            status.unlink()
        self.reported_seconds = 0.0
        self.hits = 0
        self.misses = 0

    def _write_status(self, state: str, key: tuple, **details) -> None:
        destination = self.directory / "status.json"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        payload = {
            "state": state,
            "key": key,
            "n_jobs": self.n_jobs,
            "session_hits": self.hits,
            "session_misses": self.misses,
            "cached_rankings": len(list(self.directory.glob("*.npz"))),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def _path(self, key: tuple, feature_count: int, ranking_size: int) -> Path:
        payload = json.dumps(
            {
                "key": key,
                "feature_count": int(feature_count),
                "ranking_size": int(ranking_size),
                "algorithm": "skrebate.ReliefF",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return self.directory / f"{hashlib.sha256(payload).hexdigest()}.npz"

    def get_or_fit(
        self,
        key: tuple,
        x_train: np.ndarray,
        y_train: np.ndarray,
        ranking_size: int,
    ) -> np.ndarray:
        ranking_size = min(int(ranking_size), int(x_train.shape[1]))
        path = self._path(key, x_train.shape[1], ranking_size)
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                ranking = np.asarray(cached["ranking"], dtype=np.int64)
                fit_seconds = float(cached["fit_seconds"])
            if (
                len(ranking) != ranking_size
                or np.any(ranking < 0)
                or np.any(ranking >= x_train.shape[1])
                or len(np.unique(ranking)) != len(ranking)
            ):
                raise ValueError(f"Invalid Relief-F ranking cache: {path}")
            self.hits += 1
            self.reported_seconds += fit_seconds
            self._write_status(
                "cache_hit", key, fit_seconds=fit_seconds,
                ranking_size=ranking_size,
            )
            return ranking

        selector = _make_relieff(ranking_size, self.n_jobs)
        self._write_status("fitting", key, ranking_size=ranking_size)
        started = perf_counter()
        selector.fit(x_train, y_train)
        fit_seconds = perf_counter() - started
        ranking = np.asarray(selector.top_features_[:ranking_size], dtype=np.int64)
        if len(ranking) != ranking_size:
            raise RuntimeError(
                f"Relief-F returned {len(ranking)} ranked features; "
                f"expected {ranking_size}"
            )

        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                ranking=ranking,
                fit_seconds=np.asarray(fit_seconds),
            )
        temporary.replace(path)
        self.misses += 1
        self.reported_seconds += fit_seconds
        self._write_status(
            "fit_complete", key, fit_seconds=fit_seconds,
            ranking_size=ranking_size,
        )
        return ranking


def reduce_train_test(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    method: str,
    components: int | None,
    seed: int,
    scale: bool = False,
    relieff_n_jobs: int = 1,
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
        reducer = _make_relieff(components, relieff_n_jobs)
    else:
        raise ValueError(f"Unknown reduction: {method}")

    # PCA and UMAP are unsupervised in the experimental protocol.  Passing
    # ``y_train`` to umap-learn changes UMAP into supervised dimensionality
    # reduction, which is a different method.  Relief-F, in contrast, is a
    # supervised feature selector and must receive the training labels.
    if method == "relieff":
        transformed_train = reducer.fit_transform(x_train, y_train)
    else:
        transformed_train = reducer.fit_transform(x_train)
    transformed_test = reducer.transform(x_test)
    return transformed_train, transformed_test


def prepare_fold_features(
    experiment: dict,
    feature_map: dict[str, np.ndarray],
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
    fold: int | None = None,
    relieff_cache: ReliefRankingCache | None = None,
    relieff_max_features: int | None = None,
    relieff_n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Implement feature-processing Approaches A--D without test-fold fitting."""
    approach = experiment["approach"].upper()
    extractors = experiment["extractors"]
    method = experiment.get("reduction", "full")
    components = experiment.get("components")
    scale = bool(experiment.get("scale", False))
    matrices = [feature_map[name] for name in extractors]

    def reduce_arrays(x_train, x_test, part_key):
        if method.lower() != "relieff" or relieff_cache is None:
            return reduce_train_test(
                x_train, labels[train], x_test, method, components, seed,
                scale, relieff_n_jobs
            )

        scaled_train, scaled_test = _scale_train_test(x_train, x_test, scale)
        ranking_size = max(int(components), int(relieff_max_features or components))
        cache_key = (
            approach,
            tuple(extractors),
            bool(scale),
            int(fold) if fold is not None else -1,
            str(part_key),
        )
        ranking = relieff_cache.get_or_fit(
            cache_key, scaled_train, labels[train], ranking_size
        )
        selected = ranking[: int(components)]
        return scaled_train[:, selected], scaled_test[:, selected]

    def reduce_matrix(matrix, part_key):
        return reduce_arrays(matrix[train], matrix[test], part_key)

    if approach == "A":
        return np.hstack([m[train] for m in matrices]), np.hstack(
            [m[test] for m in matrices]
        )
    if approach == "B":
        if len(matrices) != 1:
            raise ValueError("Approach B requires one feature extractor")
        return reduce_matrix(matrices[0], extractors[0])
    if approach == "C":
        return reduce_arrays(
            np.hstack([matrix[train] for matrix in matrices]),
            np.hstack([matrix[test] for matrix in matrices]),
            "combined",
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
        for extractor, matrix in zip(extractors, matrices):
            part_train, part_test = reduce_matrix(matrix, extractor)
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


def preprocessing_key(experiment: dict) -> tuple:
    """Return the classifier-independent feature-processing signature."""
    components = experiment.get("components")
    return (
        experiment["approach"].upper(),
        tuple(experiment["extractors"]),
        experiment.get("reduction", "full").lower(),
        None if components is None else int(components),
        bool(experiment.get("scale", False)),
    )


def evaluate_prepared_fold(
    experiment: dict,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    fold: int,
    x_train: np.ndarray,
    x_test: np.ndarray,
    reduction_time: float,
    seed: int,
    shared_preprocessing_size: int = 1,
    sample_names: list[str] | None = None,
    prediction_mode: str = "none",
) -> tuple[dict, list[dict]]:
    """Train and evaluate one classifier using already prepared fold features."""
    classifier = build_classifier(experiment["classifier"], seed)
    train_started = perf_counter()
    classifier.fit(x_train, labels[train])
    training_time = perf_counter() - train_started

    prediction_started = perf_counter()
    predicted = classifier.predict(x_test)
    scores = continuous_scores(classifier, x_test)
    prediction_time = perf_counter() - prediction_started

    truth = labels[test]
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()

    def safe_ratio(numerator, denominator):
        return numerator / denominator if denominator else np.nan

    record = {
        "experiment": experiment["name"],
        "approach": experiment["approach"],
        "classifier": experiment["classifier"],
        "fold": fold,
        "train_size": len(train),
        "test_size": len(test),
        "prepared_features": x_train.shape[1],
        "shared_preprocessing_size": shared_preprocessing_size,
        "accuracy": accuracy_score(truth, predicted),
        "f1": f1_score(truth, predicted, zero_division=0),
        "f1_clear": f1_score(truth, predicted, pos_label=1, zero_division=0),
        "f1_obstructed": f1_score(
            truth, predicted, pos_label=0, zero_division=0
        ),
        "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "mcc": matthews_corrcoef(truth, predicted),
        "tn_obstructed": int(tn),
        "fp_obstructed_as_clear": int(fp),
        "fn_clear_as_obstructed": int(fn),
        "tp_clear": int(tp),
        "obstructed_recall": safe_ratio(tn, tn + fp),
        "clear_recall": safe_ratio(tp, tp + fn),
        "obstructed_precision": safe_ratio(tn, tn + fn),
        "clear_precision": safe_ratio(tp, tp + fp),
        "roc_auc": np.nan,
        # This is the wall time of the shared transformation. It is repeated
        # in all classifier rows that reused the transformation and therefore
        # must not be summed across those rows to estimate total wall time.
        "reduction_seconds": reduction_time,
        "training_seconds": training_time,
        "prediction_seconds": prediction_time,
    }
    if scores is not None and len(np.unique(truth)) == 2:
        record["roc_auc"] = roc_auc_score(truth, scores)

    prediction_records = []
    if prediction_mode not in {"none", "errors", "all"}:
        raise ValueError(f"Unknown prediction mode: {prediction_mode}")
    if prediction_mode != "none":
        score_values = (
            np.asarray(scores).reshape(-1)
            if scores is not None
            else np.full(len(test), np.nan)
        )
        for local_index, sample_index in enumerate(test):
            is_correct = int(truth[local_index]) == int(predicted[local_index])
            if prediction_mode == "errors" and is_correct:
                continue
            prediction_records.append(
                {
                    "experiment": experiment["name"],
                    "approach": experiment["approach"],
                    "classifier": experiment["classifier"],
                    "fold": fold,
                    "sample_index": int(sample_index),
                    "filename": (
                        sample_names[sample_index]
                        if sample_names is not None
                        else str(sample_index)
                    ),
                    "true_label": int(truth[local_index]),
                    "predicted_label": int(predicted[local_index]),
                    "score_clear": float(score_values[local_index]),
                    "correct": bool(is_correct),
                }
            )
    return record, prediction_records


def evaluate_experiment_group(
    experiments: list[dict],
    feature_map: dict[str, np.ndarray],
    labels: np.ndarray,
    splits,
    seed: int,
    sample_names: list[str] | None = None,
    prediction_mode: str = "none",
    relieff_cache: ReliefRankingCache | None = None,
    relieff_max_features: int | None = None,
    relieff_n_jobs: int = 1,
):
    """Evaluate classifiers sharing one fold transformation.

    The dimensionality reduction is fitted once per fold and reused by every
    classifier in ``experiments``. This removes the eightfold duplicate PCA,
    UMAP, or Relief-F work in the complete search.
    """
    if not experiments:
        return []
    expected_key = preprocessing_key(experiments[0])
    if any(preprocessing_key(item) != expected_key for item in experiments[1:]):
        raise ValueError("Experiment group contains incompatible preprocessing")

    records = []
    prediction_records = []
    for fold, (train, test) in enumerate(splits, start=1):
        if relieff_cache is not None:
            relieff_cache.reset_reported_seconds()
        reduction_started = perf_counter()
        x_train, x_test = prepare_fold_features(
            experiments[0], feature_map, labels, train, test, seed,
            fold=fold,
            relieff_cache=relieff_cache,
            relieff_max_features=relieff_max_features,
            relieff_n_jobs=relieff_n_jobs,
        )
        reduction_time = perf_counter() - reduction_started
        if (
            relieff_cache is not None
            and experiments[0].get("reduction", "full").lower() == "relieff"
        ):
            # Report the original shared ranking cost even on a cache hit, so
            # timing remains comparable across different component cutoffs.
            reduction_time = relieff_cache.reported_seconds
        for experiment in experiments:
            record, fold_predictions = evaluate_prepared_fold(
                experiment,
                labels,
                train,
                test,
                fold,
                x_train,
                x_test,
                reduction_time,
                seed,
                shared_preprocessing_size=len(experiments),
                sample_names=sample_names,
                prediction_mode=prediction_mode,
            )
            records.append(record)
            prediction_records.extend(fold_predictions)
    if prediction_mode == "none":
        return records
    return records, prediction_records


def evaluate_experiment(
    experiment: dict,
    feature_map: dict[str, np.ndarray],
    labels: np.ndarray,
    splits,
    seed: int,
) -> list[dict]:
    """Evaluate one configuration and return one record per fold."""
    return evaluate_experiment_group(
        [experiment], feature_map, labels, splits, seed
    )


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if "experiments" not in config or not config["experiments"]:
        raise ValueError("Configuration must contain at least one experiment")
    return config


def summarise(records: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy", "f1", "f1_clear", "f1_obstructed", "macro_f1",
        "balanced_accuracy", "mcc", "obstructed_recall", "clear_recall",
        "obstructed_precision", "clear_precision", "roc_auc",
    ]
    rows = []
    for experiment, group in records.groupby("experiment", sort=False):
        row = {"experiment": experiment, "folds": len(group)}
        for metric in metrics:
            if metric not in group:
                continue
            row[f"{metric}_q1"] = group[metric].quantile(0.25)
            row[f"{metric}_median"] = group[metric].median()
            row[f"{metric}_q3"] = group[metric].quantile(0.75)
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)
