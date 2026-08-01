#!/usr/bin/env python3
"""Generate exact spatial decision-contribution maps for the four finalists.

The selected downstream models are linear.  Their coefficients can therefore
be projected through PCA (when present) back to the flattened final
convolutional activations.  Summing activation * effective coefficient across
channels yields a spatial decomposition of the class decision, apart from the
intercept.  This is an exact explanation of the evaluated fixed-feature
pipeline, not Grad-CAM for a separately trained end-to-end network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifact import (  # noqa: E402
    build_classifier,
    dataset_fingerprint,
    extract_or_load_features,
    load_config,
    scan_dataset,
    set_global_seed,
)


FINAL_CONV_SHAPES = {
    "ResNet101V2": (7, 7, 2048),
    "EfficientNetB3": (10, 10, 1536),
    "MobileNet": (7, 7, 1024),
    "ResNet50": (7, 7, 2048),
}

DISPLAY_NAMES = {
    "A": "A: ResNet101V2 + LR",
    "B": "B: EfficientNetB3 + PCA + SVM",
    "C": "C: MobileNet+ResNet50 + joint PCA + SVM",
    "D": "D: separate PCA + MobileNet+ResNet50 + SVM",
}

STORED_SCORE_TOLERANCE = 5e-4
RECONSTRUCTION_TOLERANCE = 1e-5


@dataclass
class LinearExplanation:
    direct_score: float
    reconstructed_score: float
    predicted_label: int
    effective_weights: dict[str, np.ndarray]
    effective_intercept: float


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="via-dataset")
    parser.add_argument(
        "--config", default="configs/article_selected_models_loocv.json"
    )
    parser.add_argument(
        "--predictions", default="results/article_selected_loocv"
    )
    parser.add_argument("--cache-dir", default="cache/features")
    parser.add_argument(
        "--output", default="results/decision_attribution"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--samples",
        nargs="*",
        help=(
            "Optional filenames. By default, explain every LOOCV image "
            "misclassified by all four finalists."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_predictions(directory: Path) -> pd.DataFrame:
    index = pd.read_csv(directory / "prediction_shard_index.csv")
    frames = []
    for relative in index["file"]:
        path = directory / relative
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.duplicated(["experiment", "sample_index"]).any():
        raise ValueError("Duplicate experiment/sample prediction rows")
    return predictions


def input_signature(
    config_path: Path, prediction_directory: Path, image_paths: list[Path]
) -> str:
    """Bind resumable records to the exact config, predictions, and index."""
    digest = hashlib.sha256()
    digest.update(sha256(config_path).encode("ascii"))
    digest.update(dataset_fingerprint(image_paths).encode("ascii"))
    index_path = prediction_directory / "prediction_shard_index.csv"
    digest.update(sha256(index_path).encode("ascii"))
    index = pd.read_csv(index_path)
    for relative in index["file"]:
        digest.update(str(relative).encode("utf-8"))
        digest.update(sha256(prediction_directory / relative).encode("ascii"))
    return digest.hexdigest()


def correctness_mask(frame: pd.DataFrame) -> pd.Series:
    values = frame["correct"]
    if pd.api.types.is_bool_dtype(values):
        return values
    normalized = values.astype(str).str.strip().str.lower()
    if not set(normalized).issubset({"true", "false", "1", "0"}):
        raise ValueError("Invalid values in prediction column 'correct'")
    return normalized.isin({"true", "1"})


def choose_samples(
    predictions: pd.DataFrame, requested: list[str] | None
) -> pd.DataFrame:
    metadata = predictions[
        ["sample_index", "filename", "true_label"]
    ].drop_duplicates()
    if requested:
        selected = metadata[metadata["filename"].isin(requested)].copy()
        missing = sorted(set(requested) - set(selected["filename"]))
        if missing:
            raise ValueError(f"Requested samples absent from predictions: {missing}")
        selected["selection_rule"] = "explicit"
        return selected.sort_values(["true_label", "filename"], ascending=[False, True])

    errors = predictions[~correctness_mask(predictions)]
    model_count = predictions["experiment"].nunique()
    consensus = (
        errors.groupby("filename")["experiment"].nunique()
        .loc[lambda value: value == model_count]
        .index
    )
    selected = metadata[metadata["filename"].isin(consensus)].copy()
    selected["selection_rule"] = "misclassified_by_all_finalists_under_loocv"
    if selected.empty:
        raise RuntimeError("No images were misclassified by every finalist")
    return selected.sort_values(["true_label", "filename"], ascending=[False, True])


def classifier_coefficient(classifier) -> tuple[np.ndarray, float]:
    if not hasattr(classifier, "coef_"):
        raise TypeError("Decision attribution requires a linear classifier")
    classes = list(classifier.classes_)
    if classes != [0, 1]:
        raise ValueError(f"Expected binary classes [0, 1], found {classes}")
    coefficient = np.asarray(classifier.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(classifier.intercept_).reshape(-1)[0])
    return coefficient, intercept


def _fit_pca(
    train_features: np.ndarray, test_features: np.ndarray, components: int, seed: int
):
    reducer = PCA(n_components=components, random_state=seed)
    transformed_train = reducer.fit_transform(train_features)
    transformed_test = reducer.transform(test_features)
    return reducer, transformed_train, transformed_test


def fit_linear_explanation(
    experiment: dict,
    feature_map: dict[str, np.ndarray],
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> LinearExplanation:
    """Fit one held-out model and project its linear score to raw activations."""
    if bool(experiment.get("scale", False)):
        raise NotImplementedError("Selected finalists are expected to use scale=false")

    approach = experiment["approach"].upper()
    method = experiment.get("reduction", "full").lower()
    extractors = experiment["extractors"]
    components = experiment.get("components")
    matrices = [feature_map[name] for name in extractors]
    reducers = []

    if approach == "A" and method == "full":
        transformed_train = np.hstack([matrix[train] for matrix in matrices])
        transformed_test = np.hstack([matrix[test] for matrix in matrices])
    elif approach == "B" and method == "pca":
        if len(matrices) != 1:
            raise ValueError("Approach B requires one extractor")
        reducer, transformed_train, transformed_test = _fit_pca(
            matrices[0][train], matrices[0][test], int(components), seed
        )
        reducers = [reducer]
    elif approach == "C" and method == "pca":
        raw_train = np.hstack([matrix[train] for matrix in matrices])
        raw_test = np.hstack([matrix[test] for matrix in matrices])
        reducer, transformed_train, transformed_test = _fit_pca(
            raw_train, raw_test, int(components), seed
        )
        reducers = [reducer]
    elif approach == "D" and method == "pca":
        train_parts = []
        test_parts = []
        for matrix in matrices:
            reducer, part_train, part_test = _fit_pca(
                matrix[train], matrix[test], int(components), seed
            )
            reducers.append(reducer)
            train_parts.append(part_train)
            test_parts.append(part_test)
        transformed_train = np.hstack(train_parts)
        transformed_test = np.hstack(test_parts)
    else:
        raise NotImplementedError(
            f"Unsupported explanation pipeline: approach={approach}, method={method}"
        )

    classifier = build_classifier(experiment["classifier"], seed)
    classifier.fit(transformed_train, labels[train])
    direct_score = float(np.asarray(classifier.decision_function(transformed_test)).item())
    predicted_label = int(np.asarray(classifier.predict(transformed_test)).item())
    coefficient, effective_intercept = classifier_coefficient(classifier)
    effective_weights = {}

    if approach == "A":
        offset = 0
        for extractor, matrix in zip(extractors, matrices):
            width = matrix.shape[1]
            effective_weights[extractor] = coefficient[offset : offset + width]
            offset += width
    elif approach == "B":
        effective = reducers[0].components_.T @ coefficient
        effective_weights[extractors[0]] = effective
        effective_intercept -= float(reducers[0].mean_ @ effective)
    elif approach == "C":
        effective = reducers[0].components_.T @ coefficient
        effective_intercept -= float(reducers[0].mean_ @ effective)
        offset = 0
        for extractor, matrix in zip(extractors, matrices):
            width = matrix.shape[1]
            effective_weights[extractor] = effective[offset : offset + width]
            offset += width
    else:  # Approach D
        coefficient_offset = 0
        for extractor, reducer in zip(extractors, reducers):
            width = reducer.n_components_
            reduced_coefficient = coefficient[
                coefficient_offset : coefficient_offset + width
            ]
            effective = reducer.components_.T @ reduced_coefficient
            effective_weights[extractor] = effective
            effective_intercept -= float(reducer.mean_ @ effective)
            coefficient_offset += width

    reconstructed_score = effective_intercept
    sample_index = int(test[0])
    for extractor in extractors:
        reconstructed_score += float(
            feature_map[extractor][sample_index] @ effective_weights[extractor]
        )

    return LinearExplanation(
        direct_score=direct_score,
        reconstructed_score=float(reconstructed_score),
        predicted_label=predicted_label,
        effective_weights=effective_weights,
        effective_intercept=float(effective_intercept),
    )


def contribution_map(
    experiment: dict,
    explanation: LinearExplanation,
    feature_map: dict[str, np.ndarray],
    sample_index: int,
) -> np.ndarray:
    """Return positive spatial evidence for the model's predicted class."""
    direction = 1.0 if explanation.predicted_label == 1 else -1.0
    maps = []
    for extractor in experiment["extractors"]:
        if extractor not in FINAL_CONV_SHAPES:
            raise KeyError(f"Missing final-convolution shape for {extractor}")
        shape = FINAL_CONV_SHAPES[extractor]
        feature = feature_map[extractor][sample_index]
        weight = explanation.effective_weights[extractor]
        if feature.size != int(np.prod(shape)) or weight.size != feature.size:
            raise ValueError(f"Unexpected feature shape for {extractor}")
        activation = feature.reshape(shape)
        coefficients = weight.reshape(shape)
        maps.append(np.sum(activation * coefficients * direction, axis=2))

    if any(item.shape != maps[0].shape for item in maps[1:]):
        raise ValueError("Selected paired extractors must share a spatial grid")
    combined = np.sum(maps, axis=0)
    return np.maximum(combined, 0.0)


def normalize_map(values: np.ndarray) -> np.ndarray:
    positive = np.asarray(values, dtype=np.float64)
    scale = float(np.percentile(positive, 99))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(positive))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(positive, dtype=np.float64)
    return np.clip(positive / scale, 0.0, 1.0)


def overlay_map(image_path: Path, heatmap: np.ndarray) -> Image.Image:
    original = Image.open(image_path).convert("RGB")
    normalized = normalize_map(heatmap)
    heat = Image.fromarray(np.uint8(normalized * 255), mode="L").resize(
        original.size, Image.Resampling.BILINEAR
    )
    intensity = np.asarray(heat, dtype=np.float32) / 255.0
    colors = np.zeros((*intensity.shape, 3), dtype=np.float32)
    colors[..., 0] = 255.0
    colors[..., 1] = 210.0 * np.sqrt(intensity)
    alpha = (0.58 * intensity)[..., None]
    base = np.asarray(original, dtype=np.float32)
    blended = base * (1.0 - alpha) + colors * alpha
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)), mode="RGB")


def _font(size: int, bold: bool = False):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_panel(
    representative: pd.DataFrame,
    image_by_sample: dict[str, Path],
    overlays: dict[tuple[str, str], Image.Image],
    predictions: pd.DataFrame,
    destination: Path,
):
    approaches = ["A", "B", "C", "D"]
    cell_width, cell_height = 330, 420
    header_height, label_height = 72, 52
    canvas = Image.new(
        "RGB",
        (cell_width * 5, header_height + (cell_height + label_height) * len(representative)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(25, bold=True)
    label_font = _font(23, bold=True)
    small_font = _font(19)
    headers = ["Original", *[DISPLAY_NAMES[item] for item in approaches]]
    for column, header in enumerate(headers):
        draw.multiline_text(
            (column * cell_width + cell_width / 2, 12),
            header,
            fill="black",
            font=small_font,
            anchor="ma",
            align="center",
        )

    for row_index, sample in enumerate(representative.itertuples(index=False)):
        y = header_height + row_index * (cell_height + label_height)
        original = Image.open(image_by_sample[sample.filename]).convert("RGB")
        images = [original, *[overlays[(sample.filename, item)] for item in approaches]]
        for column, image in enumerate(images):
            fitted = image.copy()
            fitted.thumbnail((cell_width - 12, cell_height - 12), Image.Resampling.LANCZOS)
            x = column * cell_width + (cell_width - fitted.width) // 2
            yy = y + (cell_height - fitted.height) // 2
            canvas.paste(fitted, (x, yy))
        truth = "clear" if sample.true_label == 1 else "obstructed"
        prediction = predictions[predictions["filename"].eq(sample.filename)][
            "predicted_label"
        ].mode().iloc[0]
        predicted = "clear" if int(prediction) == 1 else "obstructed"
        draw.text(
            (12, y + cell_height + 8),
            f"{sample.filename}: true={truth}; unanimous prediction={predicted}",
            fill="black",
            font=label_font,
        )
        draw.line(
            (0, y + cell_height + label_height - 1, canvas.width, y + cell_height + label_height - 1),
            fill=(170, 170, 170),
            width=2,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, dpi=(300, 300))


def main():
    args = parse_args()
    dataset = Path(args.dataset)
    config_path = Path(args.config)
    prediction_dir = Path(args.predictions)
    output = Path(args.output)
    config = load_config(config_path)
    seed = int(config.get("seed", 1980))
    set_global_seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    overlay_dir = output / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    record_dir = output / "records"
    record_dir.mkdir(parents=True, exist_ok=True)

    image_paths, labels = scan_dataset(dataset)
    by_name = {path.name: path for path in image_paths}
    predictions = read_predictions(prediction_dir)
    prediction_metadata = predictions[
        ["sample_index", "filename", "true_label"]
    ].drop_duplicates().sort_values("sample_index")
    expected_metadata = pd.DataFrame(
        {
            "sample_index": np.arange(len(image_paths)),
            "filename": [path.name for path in image_paths],
            "true_label": labels,
        }
    )
    pd.testing.assert_frame_equal(
        prediction_metadata.reset_index(drop=True),
        expected_metadata,
        check_dtype=False,
    )
    run_signature = input_signature(config_path, prediction_dir, image_paths)
    selected = choose_samples(predictions, args.samples)
    selected.to_csv(output / "selected_samples.csv", index=False)

    extractor_names = []
    for experiment in config["experiments"]:
        for extractor in experiment["extractors"]:
            if extractor not in extractor_names:
                extractor_names.append(extractor)
    features = {}
    extraction_times = {}
    for extractor in extractor_names:
        matrix, elapsed = extract_or_load_features(
            extractor, image_paths, args.cache_dir, args.batch_size
        )
        features[extractor] = matrix
        extraction_times[extractor] = float(elapsed)

    validation_rows = []
    overlays = {}
    all_indices = np.arange(len(image_paths))
    for sample in selected.itertuples(index=False):
        test = np.asarray([int(sample.sample_index)])
        train = all_indices[all_indices != int(sample.sample_index)]
        for experiment in config["experiments"]:
            approach = experiment["approach"].upper()
            overlay_path = overlay_dir / f"{Path(sample.filename).stem}_{approach}.png"
            record_path = record_dir / f"{Path(sample.filename).stem}_{approach}.json"
            if overlay_path.exists() and record_path.exists():
                cached = json.loads(record_path.read_text(encoding="utf-8"))
                if cached.get("run_signature") == run_signature:
                    with Image.open(overlay_path) as cached_image:
                        overlays[(sample.filename, approach)] = cached_image.convert(
                            "RGB"
                        ).copy()
                    validation_rows.append(cached["validation"])
                    print(f"Reused {sample.filename} / Approach {approach}")
                    continue

            print(f"Fitting {sample.filename} / Approach {approach}...")
            explanation = fit_linear_explanation(
                experiment, features, labels, train, test, seed
            )
            stored = predictions[
                predictions["experiment"].eq(experiment["name"])
                & predictions["sample_index"].eq(int(sample.sample_index))
            ]
            if len(stored) != 1:
                raise ValueError("Expected one stored prediction per sample/model")
            stored = stored.iloc[0]
            score_difference = abs(explanation.direct_score - float(stored["score_clear"]))
            reconstruction_error = abs(
                explanation.direct_score - explanation.reconstructed_score
            )
            if explanation.predicted_label != int(stored["predicted_label"]):
                raise RuntimeError("Regenerated prediction differs from stored LOOCV output")
            if (
                score_difference > STORED_SCORE_TOLERANCE
                or reconstruction_error > RECONSTRUCTION_TOLERANCE
            ):
                raise RuntimeError(
                    f"Decision reconstruction failed for {experiment['name']} / "
                    f"{sample.filename}: stored={score_difference}, "
                    f"reconstructed={reconstruction_error}"
                )

            heatmap = contribution_map(
                experiment, explanation, features, int(sample.sample_index)
            )
            overlay = overlay_map(by_name[sample.filename], heatmap)
            overlays[(sample.filename, approach)] = overlay
            overlay.save(overlay_path, dpi=(300, 300))
            validation_row = {
                "sample_index": int(sample.sample_index),
                "filename": sample.filename,
                "true_label": int(sample.true_label),
                "experiment": experiment["name"],
                "approach": approach,
                "predicted_label": explanation.predicted_label,
                "stored_score_clear": float(stored["score_clear"]),
                "regenerated_score_clear": explanation.direct_score,
                "reconstructed_score_clear": explanation.reconstructed_score,
                "stored_score_absolute_difference": score_difference,
                "reconstruction_absolute_error": reconstruction_error,
                "positive_map_cells": int((heatmap > 0).sum()),
                "map_cells": int(heatmap.size),
            }
            validation_rows.append(validation_row)
            temporary_record = record_path.with_suffix(".json.tmp")
            temporary_record.write_text(
                json.dumps(
                    {
                        "run_signature": run_signature,
                        "validation": validation_row,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary_record.replace(record_path)
            print(f"Completed {sample.filename} / Approach {approach}")

    validation = pd.DataFrame(validation_rows).sort_values(
        ["sample_index", "approach"]
    )
    validation.to_csv(output / "attribution_validation.csv", index=False)

    representatives = []
    for label in (1, 0):
        group = selected[selected["true_label"].eq(label)]
        if not group.empty:
            representatives.append(group.iloc[0])
    representative = pd.DataFrame(representatives)
    representative.to_csv(output / "representative_samples.csv", index=False)
    make_panel(
        representative,
        by_name,
        overlays,
        predictions,
        output / "decision_attribution_representative.png",
    )

    manifest = {
        "method": (
            "Exact decomposition of each linear class score into final-convolution "
            "activation contributions. PCA-space coefficients are projected to raw "
            "features with components_.T. Positive evidence for the predicted class "
            "is summed across channels and normalized independently for display."
        ),
        "selection_rule": (
            "All LOOCV images misclassified by all four fixed finalists; the article "
            "panel uses the lexicographically first clear and obstructed examples."
        ),
        "interpretation_limit": (
            "Maps are qualitative because the dataset has no obstacle segmentation "
            "or localization ground truth. They must not be interpreted as measured "
            "localization accuracy or causal attention."
        ),
        "dataset": str(dataset),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "prediction_directory": str(prediction_dir),
        "images_explained": int(len(selected)),
        "models": int(len(config["experiments"])),
        "feature_extraction_seconds": extraction_times,
        "script_sha256": sha256(Path(__file__)),
        "run_signature": run_signature,
        "stored_score_tolerance": STORED_SCORE_TOLERANCE,
        "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
    }
    (output / "attribution_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Decision-attribution outputs written to {output}")
    print(validation.groupby("approach")["reconstruction_absolute_error"].max())


if __name__ == "__main__":
    main()
