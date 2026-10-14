"""Blind image-only review artifacts and explicitly post-freeze CV diagnostics."""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.model_selection import StratifiedGroupKFold


def _font(size: int, bold: bool = False):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _matrix(values, count: int, name: str, *, optional: bool = False):
    if optional and values is None:
        return np.full((count, count), np.nan)
    result = np.asarray(values, dtype=float)
    if result.shape != (count, count):
        raise ValueError(f"{name} must have shape ({count}, {count})")
    if np.isinf(result).any() or (not optional and not np.isfinite(result).all()):
        raise ValueError(f"{name} contains invalid scores")
    if not np.allclose(result, result.T, equal_nan=True):
        raise ValueError(f"{name} must be symmetric")
    return result


def _inputs(paths, image_ids):
    paths = [Path(path) for path in paths]
    image_ids = [str(value) for value in image_ids]
    if len(paths) != len(image_ids):
        raise ValueError("Each image must have one blind image ID")
    return paths, image_ids


def _assignments(paths, groups):
    """Align frozen assignments by file identity, never interpreting filenames."""
    paths = [Path(path) for path in paths]
    names = [path.name for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("Image basenames must be unique")
    if not {"filename", "group_id"}.issubset(groups.columns):
        raise ValueError("Groups must contain filename and group_id")
    if groups[["filename", "group_id"]].isna().any().any():
        raise ValueError("Frozen assignments cannot contain missing values")
    if groups["filename"].duplicated().any() or set(groups["filename"]) != set(names):
        raise ValueError("Frozen groups must cover each image exactly once")
    assignments = groups.set_index("filename").loc[names, "group_id"].astype(str).to_numpy()
    if any(not value.strip() for value in assignments):
        raise ValueError("Group IDs cannot be empty")
    return paths, assignments


def _paste_image(canvas, path, box):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
        canvas.paste(image, (box[0] + (box[2] - box[0] - image.width) // 2,
                             box[1] + (box[3] - box[1] - image.height) // 2))


def _draw_id(draw, image_id, x, y, width):
    """Wrap rather than truncate SHA-based identifiers."""
    font = _font(17)
    lines = []
    remaining = image_id
    while remaining:
        length = len(remaining)
        while length > 1 and draw.textlength(remaining[:length], font=font) > width:
            length -= 1
        lines.append(remaining[:length])
        remaining = remaining[length:]
    draw.multiline_text((x, y), "\n".join(lines), fill="black", font=font, spacing=2)


def write_threshold_reviews(paths, image_ids, phash_distances, dhash_distances,
                            whash_distances, similarities, inliers, output_dir,
                            seed=1980, samples_per_bin=6):
    """Sample unordered pairs uniformly within fixed cosine bins, without labels."""
    paths, image_ids = _inputs(paths, image_ids)
    if not isinstance(samples_per_bin, (int, np.integer)) or samples_per_bin < 0:
        raise ValueError("samples_per_bin must be a nonnegative integer")
    count = len(paths)
    cosine = _matrix(similarities, count, "similarities")
    hashes = [_matrix(values, count, name) for name, values in (
        ("phash_distances", phash_distances), ("dhash_distances", dhash_distances),
        ("whash_distances", whash_distances))]
    structural = _matrix(inliers, count, "inliers", optional=True)
    destination = Path(output_dir) / "threshold_review"
    destination.mkdir(parents=True, exist_ok=True)
    first, second = np.triu_indices(count, k=1)
    scores = cosine[first, second]
    rng = np.random.default_rng(seed)
    rows, bins, mosaics = [], [], []
    edges = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        final = bin_index == len(edges) - 2
        eligible = np.flatnonzero((scores >= low) & (scores <= high if final else scores < high))
        selected = np.sort(rng.choice(eligible, min(samples_per_bin, len(eligible)), replace=False))
        bin_name = f"{low:.2f}_{high:.2f}"
        bins.append({"bin": bin_name, "lower": low, "upper": high,
                     "upper_inclusive": final, "eligible_pairs": int(len(eligible)),
                     "sampled_pairs": int(len(selected))})
        for pair_number, pair_index in enumerate(selected, start=1):
            left, right = int(first[pair_index]), int(second[pair_index])
            measured = float(structural[left, right])
            structural_score = measured if np.isfinite(measured) else None
            artifact = destination / f"cosine_{bin_name}_pair_{pair_number:02d}.png"
            canvas = Image.new("RGB", (720, 650), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 12), f"Cosine bin {low:.2f}-{high:.2f} | pair {pair_number}",
                      fill="black", font=_font(24, bold=True))
            structural_text = "n/a" if structural_score is None else f"{structural_score:g}"
            draw.text((16, 48), f"cosine {cosine[left, right]:.4f} | pHash {hashes[0][left, right]:g} | "
                      f"dHash {hashes[1][left, right]:g} | wHash {hashes[2][left, right]:g}",
                      fill="black", font=_font(19))
            draw.text((16, 76), f"RANSAC inliers: {structural_text}", fill="black", font=_font(19))
            for column, index in enumerate((left, right)):
                x = column * 360
                _paste_image(canvas, paths[index], (x + 12, 112, x + 348, 554))
                _draw_id(draw, image_ids[index], x + 12, 560, 336)
            draw.line((360, 112, 360, canvas.height), fill=(220, 220, 220), width=1)
            canvas.save(artifact, dpi=(300, 300))
            mosaics.append(str(artifact))
            rows.append({"bin": bin_name, "left_index": left, "right_index": right,
                         "left_image_id": image_ids[left], "right_image_id": image_ids[right],
                         "cosine": float(cosine[left, right]), "phash_distance": float(hashes[0][left, right]),
                         "dhash_distance": float(hashes[1][left, right]),
                         "whash_distance": float(hashes[2][left, right]),
                         "ransac_inliers": structural_score, "mosaic_path": str(artifact)})
    csv_path = destination / "sampled_pairs.csv"
    pd.DataFrame(rows, columns=["bin", "left_index", "right_index", "left_image_id", "right_image_id",
                               "cosine", "phash_distance", "dhash_distance", "whash_distance",
                               "ransac_inliers", "mosaic_path"]).to_csv(csv_path, index=False)
    return {"seed": int(seed), "samples_per_bin": int(samples_per_bin), "bins": bins,
            "sampled_pair_count": len(rows), "csv_path": str(csv_path), "mosaic_paths": mosaics,
            "artifacts": [str(csv_path), *mosaics]}


def write_group_reviews(paths, image_ids, groups, similarities, output_dir):
    """Render every member of every frozen group, in pages of at most 24."""
    paths, image_ids = _inputs(paths, image_ids)
    paths, assignments = _assignments(paths, groups)
    cosine = _matrix(similarities, len(paths), "similarities")
    destination = Path(output_dir) / "group_review"
    destination.mkdir(parents=True, exist_ok=True)
    rows, mosaics = [], []
    for group_number, group_id in enumerate(sorted(set(assignments)), start=1):
        indices = np.flatnonzero(assignments == group_id)
        internal = cosine[np.ix_(indices, indices)][np.triu_indices(len(indices), k=1)]
        minimum = float(internal.min()) if len(internal) else None
        group_mosaics = []
        pages = math.ceil(len(indices) / 24)
        for page in range(pages):
            members = indices[page * 24:(page + 1) * 24]
            columns = min(6, len(members))
            cell_width, cell_height, header_height = 300, 460, 90
            canvas = Image.new("RGB", (columns * cell_width,
                                       header_height + math.ceil(len(members) / columns) * cell_height), "white")
            draw = ImageDraw.Draw(canvas)
            # Display a neutral review index; manual group IDs need not themselves be blind.
            draw.text((12, 12), f"Group {group_number:04d} | page {page + 1}/{pages}",
                      fill="black", font=_font(20, bold=True))
            minimum_text = "n/a" if minimum is None else f"{minimum:.4f}"
            draw.text((12, 46), f"{len(indices)} members | min cosine: {minimum_text}",
                      fill="black", font=_font(16))
            for offset, index in enumerate(members):
                x = (offset % columns) * cell_width
                y = header_height + (offset // columns) * cell_height
                _paste_image(canvas, paths[index], (x + 10, y + 8, x + 290, y + 394))
                _draw_id(draw, image_ids[index], x + 10, y + 402, 280)
                draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1),
                               outline=(220, 220, 220), width=1)
            artifact = destination / f"group_{group_number:04d}_page_{page + 1:03d}.png"
            canvas.save(artifact, dpi=(300, 300))
            group_mosaics.append(str(artifact))
            mosaics.append(str(artifact))
        rows.append({"group_id": str(group_id), "review_group_id": f"{group_number:04d}",
                     "filenames": json.dumps([paths[index].name for index in indices]), "size": int(len(indices)),
                     "min_internal_cosine": minimum,
                     "median_internal_cosine": float(np.median(internal)) if len(internal) else None,
                     "p05_internal_cosine": float(np.quantile(internal, 0.05)) if len(internal) else None,
                     "mosaic_paths": json.dumps(group_mosaics), "review_status": "pending"})
    sheet = Path(output_dir) / "review_sheet.csv"
    pd.DataFrame(rows, columns=["group_id", "review_group_id", "filenames", "size", "min_internal_cosine",
                               "median_internal_cosine", "p05_internal_cosine", "mosaic_paths",
                               "review_status"]).to_csv(sheet, index=False)
    return {"group_count": len(rows), "image_count": len(paths), "review_sheet_path": str(sheet),
            "mosaic_count": len(mosaics), "mosaic_paths": mosaics, "artifacts": [str(sheet), *mosaics]}


def _labels_after_freeze(paths):
    """Post-freeze diagnostic only: match VIA's documented filename convention."""
    labels = []
    for path in paths:
        name = path.name.lower()
        if name.startswith("clear."):
            labels.append(1)
        elif name.startswith(("nonclear.", "non-clear.")):
            labels.append(0)
        else:
            raise ValueError(f"Unknown filename convention for post-freeze diagnostics: {path.name}")
    return np.asarray(labels, dtype=np.int64)


def summarize_groups(paths, groups):
    """Report frozen group sizes and mixed-class counts; never adjust grouping."""
    paths, assignments = _assignments(paths, groups)
    labels = _labels_after_freeze(paths)
    group_ids, sizes = np.unique(assignments, return_counts=True)
    unique_sizes, frequencies = np.unique(sizes, return_counts=True)
    return {"image_count": len(paths), "group_count": int(len(group_ids)),
            "size_distribution": {str(int(size)): int(frequency) for size, frequency in zip(unique_sizes, frequencies)},
            "singleton_count": int(np.count_nonzero(sizes == 1)),
            "largest_group_size": int(sizes.max()) if len(sizes) else 0,
            "mixed_class_group_count": sum(int(len(np.unique(labels[assignments == group_id])) == 2)
                                           for group_id in group_ids)}


def grouped_cv_diagnostics(paths, groups, n_splits=10, seed=1980):
    """Check deterministic grouped-fold feasibility after assignments are frozen."""
    if not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two")
    paths, assignments = _assignments(paths, groups)
    labels = _labels_after_freeze(paths)
    group_count = len(set(assignments))
    class_group_counts = {name: len(set(assignments[labels == value]))
                          for name, value in (("nonclear", 0), ("clear", 1))}

    def evaluate(splits):
        result = {"n_splits": int(splits), "feasible": False, "folds": [], "reason": None}
        if splits > group_count:
            result["reason"] = f"Only {group_count} unique groups are available"
            return result
        if min(class_group_counts.values()) < splits:
            result["reason"] = "At least one class occurs in fewer groups than requested folds"
            return result
        test_coverage = np.zeros(len(paths), dtype=int)
        try:
            splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", UserWarning)
                for fold, (train, test) in enumerate(splitter.split(np.zeros(len(paths)), labels, assignments), start=1):
                    train_groups, test_groups = set(assignments[train]), set(assignments[test])
                    counts = {part: {name: int(np.count_nonzero(labels[indices] == value))
                                     for name, value in (("nonclear", 0), ("clear", 1))}
                              for part, indices in (("train", train), ("test", test))}
                    test_coverage[test] += 1
                    result["folds"].append({"fold": fold, "train_image_count": int(len(train)),
                                            "test_image_count": int(len(test)), "class_counts": counts,
                                            "train_group_count": len(train_groups), "test_group_count": len(test_groups),
                                            "groups_disjoint": train_groups.isdisjoint(test_groups),
                                            "both_classes_train": all(counts["train"].values()),
                                            "both_classes_test": all(counts["test"].values())})
            result["warnings"] = [str(item.message) for item in recorded]
        except ValueError as error:
            result["reason"] = str(error)
            return result
        result["test_coverage_exactly_once"] = bool(np.all(test_coverage == 1))
        result["feasible"] = bool(len(result["folds"]) == splits and result["test_coverage_exactly_once"]
                                  and all(fold["groups_disjoint"] and fold["both_classes_train"]
                                          and fold["both_classes_test"] for fold in result["folds"]))
        if not result["feasible"]:
            result["reason"] = "The deterministic split does not provide disjoint groups and both classes in every train/test fold"
        return result

    requested = evaluate(int(n_splits))
    alternatives = []
    suggested = None
    if not requested["feasible"]:
        for splits in range(int(n_splits) - 1, 1, -1):
            alternative = evaluate(splits)
            alternatives.append(alternative)
            if suggested is None and alternative["feasible"]:
                suggested = splits
    return {"n_splits": int(n_splits), "seed": int(seed), "feasible": requested["feasible"],
            "group_count": group_count, "class_group_counts": class_group_counts,
            "folds": requested["folds"], "requested": requested,
            "suggested_n_splits": suggested, "alternatives": alternatives,
            "diagnostic_only": True}
