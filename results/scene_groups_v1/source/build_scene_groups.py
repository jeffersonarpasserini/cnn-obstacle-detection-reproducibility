#!/usr/bin/env python3
"""Infer VIA scene groups from images only; labels enter post-freeze diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
from time import perf_counter

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fine_tuning import atomic_csv
from run_experiments import write_json
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_dataset import sha256


def perceptual_hashes(paths):
    """Three 64-bit hashes for original and horizontal mirror; never parse names."""
    result = {name: np.empty((len(paths), 2), dtype=np.uint64)
              for name in ("phash", "dhash", "whash")}
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        for orientation, variant in enumerate((image, ImageOps.mirror(image))):
            for name in result:
                result[name][index, orientation] = int(str(getattr(imagehash, name)(variant, hash_size=8)), 16)
    return result


def minimum_hamming(hashes):
    values = np.asarray(hashes, dtype=np.uint64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected original/mirror uint64 hashes per image")
    distances = np.full((len(values), len(values)), 64, dtype=np.uint8)
    for first in range(2):
        for second in range(2):
            candidate = np.bitwise_count(values[:, first, None] ^ values[None, :, second])
            np.minimum(distances, candidate, out=distances)
    return distances


def cosine_similarities(embeddings):
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite image embedding rows")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Image embeddings may not have zero norm")
    normalized = values / norms
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def scene_components(phash_distances, similarities, *, phash_threshold=10,
                     semantic_threshold=0.95, inliers=None, ransac_threshold=30):
    """Union-find scene graph, with no access to labels, folds or predictions."""
    distances, cosine = np.asarray(phash_distances), np.asarray(similarities)
    if (distances.ndim != 2 or not len(distances) or distances.shape[0] != distances.shape[1]
            or cosine.shape != distances.shape or not np.isfinite(cosine).all()
            or not np.isfinite(distances).all() or np.any((distances < 0) | (distances > 64))
            or not np.allclose(cosine, cosine.T) or not np.array_equal(distances, distances.T)
            or not 0 <= phash_threshold <= 64 or not -1 <= semantic_threshold <= 1
            or ransac_threshold < 1):
        raise ValueError("Invalid symmetric image distances or thresholds")
    edges = (distances <= phash_threshold) | (cosine >= semantic_threshold)
    if inliers is not None:
        counts = np.asarray(inliers)
        if counts.shape != distances.shape or not np.allclose(counts, counts.T, equal_nan=True):
            raise ValueError("Inliers must be a symmetric image-pair matrix")
        edges |= counts >= ransac_threshold
    np.fill_diagonal(edges, False)
    parents = np.arange(len(distances))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for first, second in zip(*np.where(np.triu(edges, 1))):
        a, b = root(first), root(second)
        if a != b:
            parents[max(a, b)] = min(a, b)
    roots = [root(index) for index in range(len(parents))]
    identifiers = {value: f"scene_{index + 1:04d}" for index, value in enumerate(sorted(set(roots)))}
    return np.asarray([identifiers[value] for value in roots]), edges


def calibrate_threshold(similarities):
    """Knee: maximum distance above the diagonal of sorted nearest-neighbor cosines."""
    cosine = np.asarray(similarities, dtype=float)
    if cosine.ndim != 2 or len(cosine) < 2 or cosine.shape[0] != cosine.shape[1] or not np.isfinite(cosine).all():
        raise ValueError("Calibration requires at least two finite image embeddings")
    candidate = cosine.copy()
    np.fill_diagonal(candidate, -np.inf)
    nearest = np.sort(candidate.max(axis=1))
    span = nearest[-1] - nearest[0]
    if span <= 1e-12:
        threshold = min(1.0, float(nearest[-1]) + 1e-6)
        knee = len(nearest) - 1
    else:
        x = np.linspace(0, 1, len(nearest))
        knee = int(np.argmax((nearest - nearest[0]) / span - x))
        threshold = float(nearest[knee])
    return {"threshold": max(0.70, threshold), "method": "sorted_nearest_neighbor_diagonal_knee",
            "knee_index": knee, "nearest_neighbors": nearest.tolist()}


def guard_components(phash_distances, similarities, config, inliers=None):
    semantic = config.get("semantic_threshold")
    if semantic is None:
        semantic = calibrate_threshold(similarities)["threshold"]
    thresholds = {"phash_threshold": config["phash_threshold"],
                  "semantic_threshold": float(semantic), "ransac_threshold": config["ransac_threshold"]}
    fraction = config["max_component_fraction"]
    if not 0 < fraction <= 1 or config["semantic_guard_step"] <= 0 or config["phash_guard_step"] < 1:
        raise ValueError("Invalid giant-component guard settings")
    history = []
    while True:
        groups, edges = scene_components(phash_distances, similarities, inliers=inliers, **thresholds)
        _, sizes = np.unique(groups, return_counts=True)
        largest = int(sizes.max())
        history.append({**thresholds, "groups": len(sizes), "largest_component": largest,
                        "giant": largest > len(groups) * fraction})
        if largest <= len(groups) * fraction:
            break
        if thresholds["semantic_threshold"] < 1.0:
            thresholds["semantic_threshold"] = min(1.0, round(thresholds["semantic_threshold"] + config["semantic_guard_step"], 12))
        elif thresholds["phash_threshold"] > 0:
            thresholds["phash_threshold"] = max(0, thresholds["phash_threshold"] - config["phash_guard_step"])
        else:
            # Genuine exact duplicates can remain large; never split them to improve CV.
            break
    return groups, edges, thresholds, history


def validate_groups(groups, filenames):
    if (not {"filename", "group_id"}.issubset(groups.columns)
            or groups[["filename", "group_id"]].isna().any().any()
            or groups["filename"].duplicated().any()
            or set(groups["filename"]) != set(filenames) or len(groups) != len(filenames)
            or groups["group_id"].astype(str).str.strip().eq("").any()):
        raise ValueError("Groups must assign every image exactly once with a nonempty group ID")


def apply_overrides(groups, overrides, reason=None):
    """Apply explicitly justified filename/group assignments without changing other rows."""
    validate_groups(groups, groups["filename"].tolist())
    if (not {"filename", "group_id"}.issubset(overrides.columns)
            or overrides[["filename", "group_id"]].isna().any().any()
            or overrides["filename"].duplicated().any()
            or not set(overrides["filename"]).issubset(set(groups["filename"]))
            or overrides["group_id"].astype(str).str.strip().eq("").any()):
        raise ValueError("Invalid, duplicate or unknown override image/group ID")
    result = groups.copy()
    result["group_id"] = result["group_id"].astype(str)
    audit = []
    for row in overrides.to_dict("records"):
        justification = row.get("reason", reason)
        if not isinstance(justification, str) or not justification.strip():
            raise ValueError("Every override needs a reason column or --override-reason")
        index = result.index[result["filename"] == row["filename"]][0]
        previous = result.loc[index, "group_id"]
        group_id = str(row["group_id"])
        result.loc[index, "group_id"] = group_id
        audit.append({"filename": row["filename"], "previous_group_id": previous,
                      "group_id": group_id, "reason": justification.strip()})
    return result, audit


def semantic_embeddings(paths, config, device="cpu", threads=2, batch_size=8):
    """Real pinned DINO CLS embeddings, or the explicitly specified CLIP fallback."""
    os.environ["USE_TF"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch detects no CUDA GPU")
    torch.set_num_threads(threads)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    selected, unavailable = config["semantic_model"], None

    def load(spec):
        weight_path = Path(hf_hub_download(spec["id"], spec["weights_file"], revision=spec["revision"]))
        digest = sha256(weight_path)
        if spec.get("weights_sha256") and digest != spec["weights_sha256"]:
            raise ValueError("Downloaded semantic weights differ from the pinned SHA-256")
        options = dict(revision=spec["revision"], trust_remote_code=False)
        if spec["id"] == "facebook/dinov2-small":
            processor = AutoImageProcessor.from_pretrained(spec["id"], use_fast=False, **options)
            model = AutoModel.from_pretrained(spec["id"], use_safetensors=True, attn_implementation="eager", **options)
        elif spec["id"] == "openai/clip-vit-base-patch32":
            processor = CLIPImageProcessor.from_pretrained(spec["id"], **options)
            model = CLIPVisionModelWithProjection.from_pretrained(spec["id"], use_safetensors=False,
                        weights_only=True, attn_implementation="eager", **options)
        else:
            raise ValueError("Only the prespecified DINOv2-small and CLIP ViT-B/32 are supported")
        return processor, model.eval().to(device), weight_path, digest

    started = perf_counter()
    try:
        processor, model, weight_path, digest = load(selected)
    except OSError as error:
        fallback = config.get("fallback_model")
        if fallback is None:
            raise
        unavailable = f"{type(error).__name__}: {error}"
        selected = fallback
        print("DINO unavailable; loading the pinned CLIP fallback", flush=True)
        processor, model, weight_path, digest = load(selected)
    loading_seconds = perf_counter() - started
    started = perf_counter()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as source:
                    images.append(ImageOps.exif_transpose(source).convert("RGB"))
            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            values = (outputs.last_hidden_state[:, 0] if selected["id"] == "facebook/dinov2-small"
                      else outputs.image_embeds)
            rows.append(torch.nn.functional.normalize(values, p=2, dim=1).cpu().numpy())
            print(f"Embedding images {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    embeddings = np.concatenate(rows)
    metadata = {**selected, "actual_weights_sha256": digest, "weights_path": str(weight_path),
                "preprocessing": processor.to_dict(), "embedding_dimension": embeddings.shape[1],
                "device": device, "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
                "fallback_reason": unavailable, "loading_seconds": loading_seconds,
                "embedding_seconds": perf_counter() - started}
    return embeddings, metadata


def structural_scores(paths, distances, similarities, config, seed, threads):
    """Optional ORB/RANSAC and grayscale SSIM for prespecified image-only candidates."""
    if not config["enabled"]:
        return None, None, {"enabled": False, "candidate_count": 0}
    import cv2
    from skimage.metrics import structural_similarity

    cv2.setNumThreads(threads)
    cv2.setRNGSeed(seed)
    count = len(paths)
    candidates = distances <= config["candidate_max_phash"]
    cosine = similarities.copy()
    np.fill_diagonal(cosine, -np.inf)
    for index in range(count):
        nearest = np.argsort(cosine[index])[-min(config["top_neighbors"], count - 1):]
        candidates[index, nearest[cosine[index, nearest] >= config["candidate_min_cosine"]]] = True
    candidates |= candidates.T
    pairs = list(zip(*np.where(np.triu(candidates, 1))))
    grayscale, features = [], []
    detector = cv2.ORB_create(nfeatures=config["orb_features"])
    for path in paths:
        with Image.open(path) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert("L").resize(tuple(config["image_size"]), Image.Resampling.LANCZOS))
        grayscale.append(image)
        features.append(detector.detectAndCompute(image, None))
    inliers = np.full((count, count), np.nan)
    ssim = np.full_like(inliers, np.nan)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    for first, second in pairs:
        cv2.setRNGSeed((seed + int(first) * count + int(second)) % (2 ** 31 - 1))
        ssim[first, second] = ssim[second, first] = structural_similarity(
            grayscale[first], grayscale[second], data_range=255)
        keypoints_a, descriptors_a = features[first]
        keypoints_b, descriptors_b = features[second]
        valid = []
        if descriptors_a is not None and descriptors_b is not None and len(descriptors_b) >= 2:
            valid = [matches[0] for matches in matcher.knnMatch(descriptors_a, descriptors_b, k=2)
                     if len(matches) == 2 and matches[0].distance < config["match_ratio"] * matches[1].distance]
        value = 0
        if len(valid) >= 4:
            a = np.float32([keypoints_a[match.queryIdx].pt for match in valid])
            b = np.float32([keypoints_b[match.trainIdx].pt for match in valid])
            _, mask = cv2.findHomography(a, b, cv2.RANSAC, config["ransac_reprojection_px"])
            if mask is not None:
                value = int(mask.sum())
        inliers[first, second] = inliers[second, first] = value
    return inliers, ssim, {"enabled": True, "method": "ORB+ratio_test+homography_RANSAC",
                          "candidate_count": len(pairs), "candidate_policy": config}


def dependencies():
    packages = ("numpy", "pandas", "Pillow", "ImageHash", "PyWavelets", "torch", "transformers",
                "huggingface-hub", "opencv-python-headless", "scikit-image", "scikit-learn")
    return {name: importlib.metadata.version(name) for name in packages}


def write_calibration_plot(knee, thresholds, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nearest = np.asarray(knee["nearest_neighbors"])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(nearest, bins=30, color="steelblue", edgecolor="white")
    axes[0].set(xlabel="Nearest-neighbor cosine", ylabel="Images")
    axes[1].plot(np.arange(len(nearest)), nearest, color="steelblue")
    axes[1].scatter([knee["knee_index"]], [nearest[knee["knee_index"]]], color="black")
    axes[1].set(xlabel="Sorted image rank", ylabel="Nearest-neighbor cosine")
    for axis, line in ((axes[0], "axvline"), (axes[1], "axhline")):
        getattr(axis, line)(knee["threshold"], color="black", linestyle="--", label="Image-only knee")
        getattr(axis, line)(thresholds["semantic_threshold"], color="darkorange", label="Final threshold")
        axis.legend(fontsize=8)
    figure.tight_layout()
    destination = output / "threshold_review/nearest_neighbor_distribution.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("via-dataset"))
    parser.add_argument("--config", type=Path, default=Path("configs/scene_groups.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/scene_groups"))
    parser.add_argument("--groups-dir", type=Path, default=Path("data"))
    parser.add_argument("--overrides", type=Path, default=Path("data/groups_overrides.csv"))
    parser.add_argument("--override-reason")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--refresh-embeddings", action="store_true")
    args = parser.parse_args()
    from scene_groups_review import (write_threshold_reviews, write_group_reviews,
                                     summarize_groups, grouped_cv_diagnostics)

    output = args.output_dir.resolve()
    protected = [ROOT / "results" / name for name in ("full_search", "article_selected_10fold",
                 "article_selected_loocv", "nested_cv", "nested_cv_grouped", "fine_tuning")]
    for destination in (output, args.groups_dir.resolve()):
        if destination in (ROOT, ROOT / "results", args.dataset.resolve()):
            raise ValueError("Output may not overwrite the repository, dataset or results root")
        if any(destination == path or path in destination.parents or destination in path.parents for path in protected):
            raise ValueError("Scene groups may not overwrite existing experiment results")
    config = json.loads(args.config.read_text())
    config_at_start_sha = sha256(args.config)
    if config["batch_size"] < 1 or config["cpu_threads"] < 1:
        raise ValueError("Batch size and CPU threads must be positive")
    started = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    paths = sorted([path for path in args.dataset.iterdir() if path.is_file()
                    and path.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=lambda path: path.name)
    if len(paths) < 2:
        raise ValueError("Scene calibration requires at least two images")
    source_images = [{"filename": path.name, "sha256": sha256(path)} for path in paths]
    dataset_hash = hashlib.sha256(json.dumps(source_images, sort_keys=True).encode()).hexdigest()
    versions = dependencies()
    model_identity = {"dataset_sha256": dataset_hash, "semantic_model": config["semantic_model"],
                      "fallback_model": config.get("fallback_model"), "device": args.device,
                      "torch": versions["torch"], "transformers": versions["transformers"]}
    output.mkdir(parents=True, exist_ok=True)
    args.groups_dir.mkdir(parents=True, exist_ok=True)
    cache_file, cache_metadata = output / "embeddings.npz", output / "embedding_metadata.json"
    cached = json.loads(cache_metadata.read_text()) if cache_metadata.exists() else None
    embedding_cache_reused = bool(not args.refresh_embeddings and cached
                                  and cached["identity"] == model_identity and cache_file.exists())
    if embedding_cache_reused:
        if sha256(cache_file) != cached["cache_sha256"]:
            raise ValueError("Image embedding cache checksum differs")
        with np.load(cache_file, allow_pickle=False) as cache:
            embeddings = cache["embeddings"]
        embedding_metadata = cached["model"]
        print("Reusing checksum-validated image embeddings", flush=True)
    else:
        embeddings, embedding_metadata = semantic_embeddings(paths, config, args.device,
            config["cpu_threads"], config["batch_size"])
        temporary = output / "embeddings.tmp.npz"
        np.savez_compressed(temporary, embeddings=embeddings)
        temporary.replace(cache_file)
        write_json(cache_metadata, {"identity": model_identity, "model": embedding_metadata,
                                  "cache_sha256": sha256(cache_file)})
    if len(embeddings) != len(paths):
        raise ValueError("Embedding cache image coverage differs")
    previous_calibration = config.get("calibration")
    if previous_calibration and (previous_calibration["dataset_sha256"] != dataset_hash
            or previous_calibration["used_model"] != embedding_metadata["id"]
            or previous_calibration["used_model_revision"] != embedding_metadata["revision"]):
        print("Images or semantic weights changed; recalibrating without labels", flush=True)
        del config["calibration"]
        config["semantic_threshold"] = None
    hashes = perceptual_hashes(paths)
    distances = {name: minimum_hamming(values) for name, values in hashes.items()}
    similarities = cosine_similarities(embeddings)
    knee = calibrate_threshold(similarities)
    inliers, ssim, structural = structural_scores(paths, distances["phash"], similarities,
        config["structural"], config["seed"], config["cpu_threads"])
    group_ids, edges, thresholds, history = guard_components(distances["phash"], similarities, config, inliers)
    automatic = pd.DataFrame({"filename": [path.name for path in paths], "group_id": group_ids})
    final, override_audit = automatic.copy(), []
    if args.overrides.exists():
        final, override_audit = apply_overrides(automatic, pd.read_csv(args.overrides), args.override_reason)
    validate_groups(automatic, [path.name for path in paths])
    validate_groups(final, [path.name for path in paths])
    atomic_csv(automatic, args.groups_dir / "groups_auto.csv")
    atomic_csv(final, args.groups_dir / "groups.csv")
    # Freeze assignments BEFORE any class or CV diagnostics. Never feed them back.
    config.update(thresholds)
    if "calibration" not in config:
        config["calibration"] = {"dataset_sha256": dataset_hash, "knee": knee,
                                 "guard_history": history, "used_model": embedding_metadata["id"],
                                 "used_model_revision": embedding_metadata["revision"],
                                 "calibrated_utc": started_utc, "author_review": "pending"}
    write_json(args.config, config)
    image_ids = ["img_" + row["sha256"][:12] for row in source_images]
    review = write_threshold_reviews(paths, image_ids, distances["phash"], distances["dhash"],
        distances["whash"], similarities, inliers, output, config["seed"], config["review_samples_per_bin"])
    calibration_plot = write_calibration_plot(knee, thresholds, output)
    group_review = write_group_reviews(paths, image_ids, final, similarities, output)
    pair_rows = []
    for first, second in zip(*np.where(np.triu(edges, 1))):
        pair_rows.append({"filename_a": paths[first].name, "filename_b": paths[second].name,
            "image_id_a": image_ids[first], "image_id_b": image_ids[second],
            "phash_distance": int(distances["phash"][first, second]),
            "dhash_distance": int(distances["dhash"][first, second]),
            "whash_distance": int(distances["whash"][first, second]),
            "semantic_cosine": float(similarities[first, second]),
            "ransac_inliers": None if inliers is None or np.isnan(inliers[first, second]) else int(inliers[first, second]),
            "ssim": None if ssim is None or np.isnan(ssim[first, second]) else float(ssim[first, second]),
            "phash_edge": bool(distances["phash"][first, second] <= thresholds["phash_threshold"]),
            "semantic_edge": bool(similarities[first, second] >= thresholds["semantic_threshold"]),
            "ransac_edge": bool(inliers is not None and inliers[first, second] >= thresholds["ransac_threshold"])})
    pair_columns = ["filename_a", "filename_b", "image_id_a", "image_id_b", "phash_distance", "dhash_distance",
                    "whash_distance", "semantic_cosine", "ransac_inliers", "ssim", "phash_edge", "semantic_edge", "ransac_edge"]
    atomic_csv(pd.DataFrame(pair_rows, columns=pair_columns), output / "pairs.csv")
    summary = summarize_groups(paths, final)
    feasibility = grouped_cv_diagnostics(paths, final, 10, config["seed"])
    summary.update(thresholds=thresholds, versions=versions, seed=config["seed"],
        automatic=summarize_groups(paths, automatic), giant_guard=history,
        unresolved_giant=summary["largest_group_size"] > len(paths) * config["max_component_fraction"],
        override_count=len(override_audit), grouped_cv=feasibility,
        review_status="pending_author_validation", elapsed_seconds=perf_counter() - started)
    write_json(output / "summary.json", summary)
    write_json(output / "run_metadata.json", {"started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "completed": True,
        "command": sys.argv, "platform": platform.platform(), "python": sys.version,
        "dataset_sha256": dataset_hash, "images": source_images, "seed": config["seed"],
        "config_sha256_at_start": config_at_start_sha, "config_sha256": sha256(args.config),
        "config": config, "dependencies": versions, "semantic": embedding_metadata,
        "embedding_cache_reused": embedding_cache_reused,
        "structural": structural, "calibration": knee, "giant_guard": history,
        "source_sha256": {name: sha256(Path(__file__).with_name(name))
                          for name in ("build_scene_groups.py", "scene_groups_review.py")},
        "pairs_scope": "automatic graph edges at final automatic thresholds; overrides do not fabricate image-pair scores",
        "no_labels_folds_predictions_or_metrics_in_grouping": True,
        "class_diagnostics_only_after_group_csv_freeze": True,
        "review_status": "pending_author_validation", "threshold_review": review,
        "group_review": group_review, "overrides_path": str(args.overrides) if args.overrides.exists() else None,
        "overrides_sha256": sha256(args.overrides) if args.overrides.exists() else None,
        "override_audit": override_audit,
        "artifacts": {str(path): sha256(Path(path)) for path in [args.groups_dir / "groups_auto.csv",
            args.groups_dir / "groups.csv", output / "pairs.csv", output / "review_sheet.csv",
            output / "summary.json", calibration_plot, *review["artifacts"], *group_review["artifacts"]]},
        "nested_integration": "not performed: executor absent; grouping-only scope explicitly approved",
        "elapsed_seconds": perf_counter() - started})
    print(f"Scene groups: {summary['group_count']} groups; sizes={summary['size_distribution']}; "
          f"singletons={summary['singleton_count']}; mixed={summary['mixed_class_group_count']}")
    print(f"CPU/GPU elapsed: {perf_counter() - started:.1f}s; human-author review remains pending")


if __name__ == "__main__":
    main()
