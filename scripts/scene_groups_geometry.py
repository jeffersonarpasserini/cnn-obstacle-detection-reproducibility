"""Reciprocal Lowe matches, homography RANSAC and measured quadrant coverage."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

try:
    from scene_groups_graph import candidate_pairs
except ModuleNotFoundError:
    from scripts.scene_groups_graph import candidate_pairs


def reciprocal_ratio_matches(descriptors_a, descriptors_b, ratio=0.75):
    import cv2

    if descriptors_a is None or descriptors_b is None or min(len(descriptors_a), len(descriptors_b)) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def good(source, target):
        return [pair[0] for pair in matcher.knnMatch(source, target, k=2)
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]

    forward, reverse = good(descriptors_a, descriptors_b), good(descriptors_b, descriptors_a)
    reciprocal = {(match.trainIdx, match.queryIdx) for match in reverse}
    return sorted([match for match in forward if (match.queryIdx, match.trainIdx) in reciprocal],
                  key=lambda match: (match.queryIdx, match.trainIdx))


def quadrant_counts(points, image_size):
    """TL, TR, BL, BR counts in the resized image's own coordinate system."""
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    width, height = image_size
    quadrants = (values[:, 0] >= width / 2).astype(int) + 2 * (values[:, 1] >= height / 2).astype(int)
    return np.bincount(quadrants, minlength=4).astype(int).tolist()


def geometry_qualifies(record, ransac_threshold=30, min_ratio=0.25,
                       min_quadrants=3, min_per_quadrant=3):
    if record.get("evaluation_status") != "evaluated" or not record.get("homography_valid", False):
        return False
    valid, inliers = record["valid_match_count"], record["ransac_inliers"]
    return bool(valid > 0 and inliers >= ransac_threshold and inliers / valid >= min_ratio
                and sum(n >= min_per_quadrant for n in record["quadrant_inliers_a"]) >= min_quadrants
                and sum(n >= min_per_quadrant for n in record["quadrant_inliers_b"]) >= min_quadrants)


def structural_scores(paths, distances, similarities, config, seed, threads, ransac_threshold=30):
    import cv2
    from skimage.metrics import structural_similarity

    if not config["enabled"]:
        raise ValueError("V2 requires structural.enabled=true")
    if (not 0 < config["match_ratio"] < 1 or config["orb_features"] < 1
            or config["ransac_reprojection_px"] <= 0
            or not 0 <= config["min_inlier_ratio"] <= 1
            or not 1 <= config["min_quadrants"] <= 4 or config["min_inliers_per_quadrant"] < 1):
        raise ValueError("Invalid structural verification settings")
    cv2.setNumThreads(threads)
    cv2.setRNGSeed(seed)
    candidates, mutual = candidate_pairs(distances, similarities,
                                         config["candidate_max_phash"], config["top_neighbors"])
    count = len(paths)
    image_size = tuple(config["image_size"])
    grayscale, features = [], []
    detector = cv2.ORB_create(nfeatures=config["orb_features"])
    for path in paths:
        with Image.open(path) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert("L").resize(image_size, Image.Resampling.LANCZOS))
        grayscale.append(image)
        features.append(detector.detectAndCompute(image, None))
    inliers = np.full((count, count), np.nan)
    ssim = np.full_like(inliers, np.nan)
    records = {}
    pairs = list(zip(*np.where(np.triu(candidates, 1))))
    for position, (first, second) in enumerate(pairs, start=1):
        first, second = int(first), int(second)
        cv2.setRNGSeed((seed + first * count + second) % (2 ** 31 - 1))
        keypoints_a, descriptors_a = features[first]
        keypoints_b, descriptors_b = features[second]
        matches = reciprocal_ratio_matches(descriptors_a, descriptors_b, config["match_ratio"])
        a = np.float32([keypoints_a[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        b = np.float32([keypoints_b[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        homography, mask, error = None, np.zeros(len(matches), dtype=bool), None
        if len(matches) >= 4:
            try:
                homography, found = cv2.findHomography(a, b, cv2.RANSAC, config["ransac_reprojection_px"])
                if found is not None and homography is not None and np.isfinite(homography).all():
                    mask = found.ravel().astype(bool)
                else:
                    homography = None
            except cv2.error as exception:
                error = str(exception)
        value = int(mask.sum())
        quadrants_a, quadrants_b = quadrant_counts(a[mask], image_size), quadrant_counts(b[mask], image_size)
        record = {
            "candidate": True, "evaluation_status": "evaluated",
            "valid_match_count": len(matches), "ransac_inliers": value,
            "inlier_ratio": value / len(matches) if matches else 0.0,
            "quadrant_inliers_a": quadrants_a, "quadrant_inliers_b": quadrants_b,
            "supported_quadrants_a": sum(n >= config["min_inliers_per_quadrant"] for n in quadrants_a),
            "supported_quadrants_b": sum(n >= config["min_inliers_per_quadrant"] for n in quadrants_b),
            "homography_valid": homography is not None, "homography_error": error,
            "homography": homography.tolist() if homography is not None else None,
            "_points_a": a, "_points_b": b, "_inlier_mask": mask,
        }
        record["geometry_qualified"] = geometry_qualifies(record, ransac_threshold,
            config["min_inlier_ratio"], config["min_quadrants"], config["min_inliers_per_quadrant"])
        inliers[first, second] = inliers[second, first] = value
        ssim[first, second] = ssim[second, first] = structural_similarity(grayscale[first], grayscale[second], data_range=255)
        records[(first, second)] = record
        if position % 100 == 0 or position == len(pairs):
            print(f"Structural candidates {position}/{len(pairs)}", flush=True)
    return {"records": records, "candidates": candidates, "mutual_neighbors": mutual,
            "inliers": inliers, "ssim": ssim, "image_size": image_size,
            "metadata": {"enabled": True, "method": "ORB+reciprocal_Lowe_ratio+homography_RANSAC",
                         "candidate_count": len(pairs), "candidate_policy": config,
                         "quadrant_order": ["top_left", "top_right", "bottom_left", "bottom_right"],
                         "inlier_ratio_denominator": "reciprocal matches passing Lowe ratio in both directions"}}
