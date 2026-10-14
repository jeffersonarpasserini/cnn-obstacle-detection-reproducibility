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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_dataset import sha256
from scene_groups_graph import (atomic_csv, write_json, archive_v1, groups_from_edges,
    validate_groups, apply_overrides, apply_edge_rejections, validate_accepted_edges, graph_bridges)
from scene_groups_geometry import structural_scores, geometry_qualifies


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
                     semantic_threshold=0.795, structural=None, ransac_threshold=30,
                     min_inlier_ratio=0.25, min_quadrants=3, min_inliers_per_quadrant=3):
    """V2 connected components; semantic similarity alone never creates an edge."""
    distances, cosine = np.asarray(phash_distances), np.asarray(similarities)
    if (distances.ndim != 2 or not len(distances) or distances.shape[0] != distances.shape[1]
            or cosine.shape != distances.shape or not np.isfinite(cosine).all()
            or not np.isfinite(distances).all() or np.any((distances < 0) | (distances > 64))
            or not np.allclose(cosine, cosine.T) or not np.array_equal(distances, distances.T)
            or not 0 <= phash_threshold <= 64 or not -1 <= semantic_threshold <= 1
            or ransac_threshold < 1):
        raise ValueError("Invalid symmetric image distances or thresholds")
    edges = distances <= phash_threshold
    for (first, second), record in (structural or {}).items():
        if (record.get("candidate", False) and cosine[first, second] >= semantic_threshold
                and geometry_qualifies(record, ransac_threshold, min_inlier_ratio,
                                       min_quadrants, min_inliers_per_quadrant)):
            edges[first, second] = edges[second, first] = True
    np.fill_diagonal(edges, False)
    return groups_from_edges(edges), edges


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


def guard_components(phash_distances, similarities, config, structural=None):
    """Report size alerts once; never tune thresholds or divide valid components."""
    thresholds = {"phash_threshold": config["phash_threshold"],
                  "semantic_threshold": float(config["semantic_threshold"]),
                  "ransac_threshold": config["ransac_threshold"]}
    fraction = config["max_component_fraction"]
    if not 0 < fraction <= 1:
        raise ValueError("Invalid component alert fraction")
    settings = config["structural"]
    groups, edges = scene_components(phash_distances, similarities, structural=structural,
        min_inlier_ratio=settings["min_inlier_ratio"], min_quadrants=settings["min_quadrants"],
        min_inliers_per_quadrant=settings["min_inliers_per_quadrant"], **thresholds)
    _, sizes = np.unique(groups, return_counts=True)
    largest = int(sizes.max())
    history = [{**thresholds, "groups": len(sizes), "largest_component": largest,
                "giant": largest > len(groups) * fraction, "action": "report_only"}]
    return groups, edges, thresholds, history


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


def dependencies():
    packages = ("numpy", "pandas", "Pillow", "ImageHash", "PyWavelets", "torch", "transformers",
                "huggingface-hub", "opencv-python-headless", "scikit-image", "scikit-learn")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_embedding_cache(directory, dataset_hash, config, count):
    """Reuse pinned pixels/encoder outputs, retaining original execution provenance."""
    directory = Path(directory)
    cache_file, metadata_file = directory / "embeddings.npz", directory / "embedding_metadata.json"
    if not cache_file.exists() or not metadata_file.exists():
        return None
    metadata = json.loads(metadata_file.read_text())
    identity = metadata["identity"]
    expected = {"dataset_sha256": dataset_hash, "semantic_model": config["semantic_model"],
                "fallback_model": config.get("fallback_model")}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Embedding cache pixels or prespecified encoder changed")
    model = metadata["model"]
    specs = [config["semantic_model"], config.get("fallback_model")]
    selected = next((spec for spec in specs if spec and spec["id"] == model["id"]), None)
    if (selected is None or selected["revision"] != model["revision"]
            or selected["weights_sha256"] != model["actual_weights_sha256"]
            or sha256(cache_file) != metadata["cache_sha256"]):
        raise ValueError("Embedding cache model or checksum differs")
    with np.load(cache_file, allow_pickle=False) as cache:
        embeddings = cache["embeddings"]
    if embeddings.shape != (count, model["embedding_dimension"]) or not np.isfinite(embeddings).all():
        raise ValueError("Embedding cache image coverage or values differ")
    return embeddings, model, metadata


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
    parser.add_argument("--edge-rejections", type=Path, default=Path("data/edge_rejections.csv"))
    parser.add_argument("--override-reason")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--refresh-embeddings", action="store_true")
    parser.add_argument("--embedding-cache-source", type=Path)
    args = parser.parse_args()
    from scene_groups_review import (write_threshold_reviews, write_group_reviews,
        write_bridge_reviews, write_geometry_reviews, summarize_groups, grouped_cv_diagnostics)

    output = args.output_dir.resolve()
    protected = [ROOT / "results" / name for name in ("full_search", "article_selected_10fold",
                 "article_selected_loocv", "nested_cv", "nested_cv_grouped", "fine_tuning")]
    protected.append(ROOT / ".venv")
    for destination in (output, args.groups_dir.resolve()):
        if destination in (ROOT, ROOT / "results", args.dataset.resolve()):
            raise ValueError("Output may not overwrite the repository, dataset or results root")
        if any(destination == path or path in destination.parents or destination in path.parents for path in protected):
            raise ValueError("Scene groups may not overwrite experiment results or the training environment")
    config = json.loads(args.config.read_text())
    config_at_start_sha = sha256(args.config)
    if (config.get("protocol_version") != 2 or config["batch_size"] < 1 or config["cpu_threads"] < 1
            or not config["structural"]["enabled"]
            or config["structural"]["candidate_max_phash"] < config["phash_threshold"]):
        raise ValueError("A valid V2 config with enabled structural verification is required")
    archive = archive_v1(ROOT)
    started = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    paths = sorted([path for path in args.dataset.iterdir() if path.is_file()
                    and path.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=lambda path: path.name)
    if len(paths) < 2:
        raise ValueError("Scene grouping requires at least two images")
    filenames = [path.name for path in paths]
    source_images = [{"filename": path.name, "sha256": sha256(path)} for path in paths]
    dataset_hash = hashlib.sha256(json.dumps(source_images, sort_keys=True).encode()).hexdigest()
    versions = dependencies()
    output.mkdir(parents=True, exist_ok=True)
    args.groups_dir.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_metadata.json", {"protocol_version": 2, "completed": False,
        "started_utc": started_utc, "config_sha256": config_at_start_sha,
        "review_status": "pending_author_validation"})
    cache_source = args.embedding_cache_source or output
    cached = None if args.refresh_embeddings else load_embedding_cache(cache_source, dataset_hash, config, len(paths))
    if args.embedding_cache_source and cached is None and not args.refresh_embeddings:
        raise ValueError("The explicit embedding cache source is missing")
    cache_file, cache_metadata = output / "embeddings.npz", output / "embedding_metadata.json"
    if cached is not None:
        import shutil
        embeddings, embedding_metadata, original_cache_metadata = cached
        if cache_source.resolve() != output:
            shutil.copy2(cache_source / "embeddings.npz", cache_file)
            shutil.copy2(cache_source / "embedding_metadata.json", cache_metadata)
        print("Reusing pinned, checksum-validated image embeddings; no encoder execution", flush=True)
    else:
        embeddings, embedding_metadata = semantic_embeddings(paths, config, args.device,
            config["cpu_threads"], config["batch_size"])
        temporary = output / "embeddings.tmp.npz"
        np.savez_compressed(temporary, embeddings=embeddings)
        temporary.replace(cache_file)
        original_cache_metadata = {"identity": {"dataset_sha256": dataset_hash,
            "semantic_model": config["semantic_model"], "fallback_model": config.get("fallback_model"),
            "device": args.device, "torch": versions["torch"], "transformers": versions["transformers"]},
            "model": embedding_metadata, "cache_sha256": sha256(cache_file)}
        write_json(cache_metadata, original_cache_metadata)
    hashes = perceptual_hashes(paths)
    distances = {name: minimum_hamming(values) for name, values in hashes.items()}
    similarities = cosine_similarities(embeddings)
    structural = structural_scores(paths, distances["phash"], similarities,
        config["structural"], config["seed"], config["cpu_threads"], config["ransac_threshold"])
    raw_group_ids, confirmed_edges, thresholds, history = guard_components(
        distances["phash"], similarities, config, structural["records"])
    unreviewed = pd.DataFrame({"filename": filenames, "group_id": raw_group_ids})
    rejections = (pd.read_csv(args.edge_rejections) if args.edge_rejections.exists()
                  else pd.DataFrame(columns=["filename_a", "filename_b", "reason", "reviewer", "date"]))
    accepted_edges, rejection_audit = apply_edge_rejections(confirmed_edges, filenames, rejections)
    automatic = pd.DataFrame({"filename": filenames, "group_id": groups_from_edges(accepted_edges)})
    final, override_audit = automatic.copy(), []
    if args.overrides.exists():
        final, override_audit = apply_overrides(automatic, pd.read_csv(args.overrides), args.override_reason)
    validate_accepted_edges(final, filenames, accepted_edges)
    atomic_csv(unreviewed, args.groups_dir / "groups_unreviewed.csv")
    atomic_csv(automatic, args.groups_dir / "groups_auto.csv")
    atomic_csv(final, args.groups_dir / "groups.csv")
    # Freeze before ANY label or fold diagnostics; these never feed back to the graph.
    frozen_sha = sha256(args.groups_dir / "groups.csv")
    image_ids = ["img_" + row["sha256"][:12] for row in source_images]
    bridges = set(graph_bridges(accepted_edges))
    assignment_map = final.set_index("filename")["group_id"].to_dict()
    group_sizes = final["group_id"].value_counts().to_dict()
    rejection_reasons = {(row["filename_a"], row["filename_b"]): row["reason"] for row in rejection_audit}
    rejection_records = {(row["filename_a"], row["filename_b"]): row for row in rejection_audit}
    pair_rows = []
    for first, second in zip(*np.triu_indices(len(paths), k=1)):
        first, second = int(first), int(second)
        record = structural["records"].get((first, second), {})
        measured = bool(record)
        phash_pass = bool(distances["phash"][first, second] <= thresholds["phash_threshold"])
        semantic_pass = bool(similarities[first, second] >= thresholds["semantic_threshold"])
        geometric_edge = bool(measured and record["geometry_qualified"] and semantic_pass)
        criteria = "|".join(name for name, valid in (("phash", phash_pass),
                                                     ("semantic_geometry", geometric_edge)) if valid)
        candidate_reasons = "|".join(name for name, valid in (
            ("phash_16", distances["phash"][first, second] <= config["structural"]["candidate_max_phash"]),
            ("mutual_knn", structural["mutual_neighbors"][first, second])) if valid)
        pair_rows.append({"filename_a": filenames[first], "filename_b": filenames[second],
            "image_id_a": image_ids[first], "image_id_b": image_ids[second],
            "phash_distance": int(distances["phash"][first, second]),
            "dhash_distance": int(distances["dhash"][first, second]),
            "whash_distance": int(distances["whash"][first, second]),
            "semantic_cosine": float(similarities[first, second]),
            "candidate": measured, "candidate_reason": candidate_reasons,
            "evaluation_status": record.get("evaluation_status", "not_evaluated"),
            **{name: record.get(name) for name in ("valid_match_count", "ransac_inliers", "inlier_ratio",
               "supported_quadrants_a", "supported_quadrants_b", "homography_valid", "homography_error")},
            "valid_matches": record.get("valid_match_count"),
            "quadrant_inliers_a": json.dumps(record["quadrant_inliers_a"]) if measured else None,
            "quadrant_inliers_b": json.dumps(record["quadrant_inliers_b"]) if measured else None,
            "quadrants_a": json.dumps(record["quadrant_inliers_a"]) if measured else None,
            "quadrants_b": json.dumps(record["quadrant_inliers_b"]) if measured else None,
            "homography": json.dumps(record["homography"]) if measured else None,
            "ssim": float(structural["ssim"][first, second]) if measured else None,
            "semantic_threshold_pass": semantic_pass, "phash_edge": phash_pass,
            "semantic_edge": geometric_edge, "ransac_edge": geometric_edge,
            "geometry_qualified": bool(record.get("geometry_qualified", False)),
            "edge_criteria": criteria, "confirmed_edge": bool(confirmed_edges[first, second]),
            "edge_reason": criteria,
            "review_rejected": (filenames[first], filenames[second]) in rejection_reasons,
            "rejection_reason": rejection_reasons.get((filenames[first], filenames[second])),
            "rejection_reviewer": rejection_records.get((filenames[first], filenames[second]), {}).get("reviewer"),
            "rejection_date": rejection_records.get((filenames[first], filenames[second]), {}).get("date"),
            "accepted_edge": bool(accepted_edges[first, second]),
            "is_bridge": (first, second) in bridges,
            "same_final_group": assignment_map[filenames[first]] == assignment_map[filenames[second]]})
    pairs = pd.DataFrame(pair_rows)
    atomic_csv(pairs, output / "pairs.csv")
    write_json(output / "group_freeze.json", {"groups_sha256": frozen_sha,
        "pairs_sha256": sha256(output / "pairs.csv"), "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "image-only grouping snapshot frozen before diagnostic labels/folds",
        "author_validation": "pending", "accepted_edges_cross_groups": 0})
    review = write_threshold_reviews(paths, image_ids, distances["phash"], distances["dhash"],
        distances["whash"], similarities, structural["inliers"], output, config["seed"], config["review_samples_per_bin"])
    knee = calibrate_threshold(similarities)
    calibration_plot = write_calibration_plot(knee, thresholds, output)
    group_review = write_group_reviews(paths, image_ids, final, similarities, output, pairs=pairs)
    bridge_review = write_bridge_reviews(paths, image_ids, final, bridges, structural, pairs, output)
    geometry_review = write_geometry_reviews(paths, image_ids, final, structural, pairs, output,
                                             thresholds["ransac_threshold"])
    summary = summarize_groups(paths, final)
    feasibility = grouped_cv_diagnostics(paths, final, 10, config["seed"])
    folds = [{"fold": f["fold"], "test_image_count": f["test_image_count"],
              "clear": f["class_counts"]["test"]["clear"], "nonclear": f["class_counts"]["test"]["nonclear"],
              "clear_fraction": f["class_counts"]["test"]["clear"] / f["test_image_count"],
              "groups_disjoint": f["groups_disjoint"]} for f in feasibility["folds"]]
    atomic_csv(pd.DataFrame(folds, columns=["fold", "test_image_count", "clear", "nonclear",
        "clear_fraction", "groups_disjoint"]), output / "fold_diagnostics.csv")
    size_alerts = [{"group_id": str(group), "size": int(size)} for group, size in sorted(group_sizes.items())
                   if size > len(paths) * config["max_component_fraction"]]
    summary.update(protocol_version=2, thresholds=thresholds, versions=versions, seed=config["seed"],
        automatic=summarize_groups(paths, automatic), size_alerts=size_alerts,
        size_alert_limit=int(np.floor(len(paths) * config["max_component_fraction"])),
        size_policy="report_only_no_threshold_tuning_or_splitting",
        candidate_pair_count=int(pairs["candidate"].sum()), pair_count=len(pairs),
        confirmed_edge_count=int(pairs["confirmed_edge"].sum()), accepted_edge_count=int(pairs["accepted_edge"].sum()),
        geometry_qualified_pair_count=int(pairs["geometry_qualified"].sum()),
        semantic_geometry_pair_count=int(pairs["semantic_edge"].sum()),
        geometric_review_pair_count=geometry_review["review_count"],
        rejection_count=len(rejection_audit), override_count=len(override_audit),
        accepted_edges_cross_groups=0, bridge_count=len(bridges), grouped_cv=feasibility,
        review_status="pending_author_validation", elapsed_seconds=perf_counter() - started)
    write_json(output / "summary.json", summary)
    if sha256(args.groups_dir / "groups.csv") != frozen_sha or sha256(args.config) != config_at_start_sha:
        raise ValueError("Frozen groups or fixed config changed during diagnostics")
    artifacts = [args.groups_dir / name for name in ("groups_unreviewed.csv", "groups_auto.csv", "groups.csv")]
    artifacts += [output / name for name in ("pairs.csv", "group_freeze.json", "summary.json", "fold_diagnostics.csv")]
    artifacts += [calibration_plot, *review["artifacts"], *group_review["artifacts"],
                  *bridge_review["artifacts"], *geometry_review["artifacts"]]
    write_json(output / "run_metadata.json", {"protocol_version": 2, "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "completed": True,
        "command": sys.argv, "platform": platform.platform(), "python": sys.version,
        "dataset_sha256": dataset_hash, "images": source_images, "seed": config["seed"],
        "config_sha256": config_at_start_sha, "config": config, "dependencies": versions,
        "semantic": embedding_metadata, "embedding_cache_reused": cached is not None,
        "embedding_cache_source": str(cache_source.resolve()) if cached else None,
        "embedding_cache_origin_identity": original_cache_metadata["identity"],
        "structural": structural["metadata"], "v1_archive": str(archive),
        "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in (
            "build_scene_groups.py", "scene_groups_graph.py", "scene_groups_geometry.py", "scene_groups_review.py")},
        "pairs_scope": "all unordered image pairs; unevaluated geometry is explicitly not_evaluated",
        "no_labels_folds_predictions_or_metrics_in_grouping": True,
        "class_diagnostics_only_after_group_csv_freeze": True, "frozen_groups_sha256": frozen_sha,
        "review_status": "pending_author_validation", "threshold_review": review,
        "group_review": group_review, "bridge_review": bridge_review, "geometry_review": geometry_review,
        "edge_rejections_path": str(args.edge_rejections) if args.edge_rejections.exists() else None,
        "edge_rejections_sha256": sha256(args.edge_rejections) if args.edge_rejections.exists() else None,
        "rejection_audit": rejection_audit,
        "overrides_path": str(args.overrides) if args.overrides.exists() else None,
        "overrides_sha256": sha256(args.overrides) if args.overrides.exists() else None,
        "override_audit": override_audit, "accepted_edges_cross_groups": 0,
        "artifacts": {str(path): sha256(Path(path)) for path in artifacts},
        "nested_integration": "not performed; pending human-author scene validation",
        "elapsed_seconds": perf_counter() - started})
    print(f"V2: {summary['group_count']} groups; sizes={summary['size_distribution']}; "
          f"edges={summary['accepted_edge_count']}; size alerts={len(size_alerts)}")
    print(f"Elapsed: {perf_counter() - started:.1f}s; human-author review remains pending")


if __name__ == "__main__":
    main()
