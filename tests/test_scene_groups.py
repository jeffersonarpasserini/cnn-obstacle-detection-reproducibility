import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from scripts.build_scene_groups import (
    apply_overrides,
    cosine_similarities,
    guard_components,
    minimum_hamming,
    perceptual_hashes,
    scene_components,
)
from scripts.scene_groups_review import summarize_groups


ROOT = Path(__file__).resolve().parents[1]


class SceneGroupsTests(unittest.TestCase):
    def test_grouping_signature_has_no_label_parameter(self):
        parameters = inspect.signature(scene_components).parameters
        for name in parameters:
            self.assertFalse("label" in name.lower() or "target" in name.lower())
            self.assertNotEqual(name, "y")

    def test_exact_and_mirrored_duplicates_co_group_with_real_hashes(self):
        rng = np.random.default_rng(1980)
        image = Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / f"image_{i}.png" for i in range(3)]
            image.save(paths[0])
            image.save(paths[1])
            ImageOps.mirror(image).save(paths[2])
            hashes = perceptual_hashes(paths)
            for kind in ("phash", "dhash", "whash"):
                np.testing.assert_array_equal(minimum_hamming(hashes[kind]), np.zeros((3, 3)))
            groups, _ = scene_components(
                minimum_hamming(hashes["phash"]), cosine_similarities(np.eye(3))
            )
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[0], groups[2])

    def test_distinct_random_images_remain_separate(self):
        rng = np.random.default_rng(1980)
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / f"image_{i}.png" for i in range(4)]
            for path in paths:
                Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(path)
            distances = minimum_hamming(perceptual_hashes(paths)["phash"])
            groups, edges = scene_components(distances, cosine_similarities(np.eye(4)))
        self.assertEqual(len(set(groups)), 4)
        self.assertFalse(edges.any())

    def test_semantic_edges_require_geometry_and_confirmed_links_are_transitive(self):
        distances = np.full((4, 4), 64, dtype=np.uint8)
        np.fill_diagonal(distances, 0)
        similarities = np.eye(4)
        distances[0, 1] = distances[1, 0] = 10
        similarities[1, 2] = similarities[2, 1] = 0.95
        record = {"candidate": True, "evaluation_status": "evaluated", "homography_valid": True,
                  "valid_match_count": 60, "ransac_inliers": 30,
                  "quadrant_inliers_a": [10, 10, 10, 0], "quadrant_inliers_b": [10, 10, 10, 0]}
        groups, edges = scene_components(distances, similarities, structural={(1, 2): record, (2, 3): record})
        self.assertEqual(len(set(groups)), 2)
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertNotEqual(groups[2], groups[3])
        expected = np.zeros((4, 4), dtype=bool)
        for first, second in ((0, 1), (1, 2)):
            expected[first, second] = expected[second, first] = True
        np.testing.assert_array_equal(edges, expected)

    def test_size_alert_preserves_links_without_tuning_thresholds(self):
        config = {
            "phash_threshold": 10, "semantic_threshold": 0.7, "ransac_threshold": 30,
            "max_component_fraction": 0.5,
            "structural": {"min_inlier_ratio": 0.25, "min_quadrants": 3, "min_inliers_per_quadrant": 3},
        }
        distances = np.full((4, 4), 64, dtype=np.uint8)
        np.fill_diagonal(distances, 0)
        similarities = np.eye(4)
        for first, second, cosine in ((0, 1, 0.8), (1, 2, 0.75), (2, 3, 0.8)):
            similarities[first, second] = similarities[second, first] = cosine
            distances[first, second] = distances[second, first] = 10
        groups, _, thresholds, history = guard_components(distances, similarities, config)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertEqual(groups[0], groups[2])
        self.assertEqual(thresholds["semantic_threshold"], 0.7)
        self.assertEqual([row["largest_component"] for row in history], [4])
        self.assertTrue(history[0]["giant"])
        self.assertEqual(history[0]["action"], "report_only")
        distances[:] = 64
        np.fill_diagonal(distances, 0)
        distances[:3, :3] = 0
        groups, _, thresholds, history = guard_components(distances, np.eye(4), config)
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertNotEqual(groups[0], groups[3])
        self.assertEqual(thresholds["phash_threshold"], 10)
        self.assertTrue(history[-1]["giant"])

    def test_real_groups_csv_covers_all_342_images_once(self):
        csv_path = ROOT / "data" / "groups.csv"
        if not csv_path.exists():
            self.skipTest("Run build_scene_groups.py to generate the real groups.csv artifact")
        filenames = {path.name for path in (ROOT / "via-dataset").glob("*.jpg")}
        groups = pd.read_csv(csv_path)
        self.assertEqual(len(filenames), 342)
        self.assertEqual(len(groups), 342)
        self.assertTrue(groups["filename"].is_unique)
        self.assertEqual(set(groups["filename"]), filenames)
        self.assertTrue(groups["group_id"].notna().all())
        self.assertTrue(groups["group_id"].astype(str).str.strip().ne("").all())

    def test_justified_overrides_apply_and_log_previous_assignment(self):
        original = pd.DataFrame({
            "filename": ["first.png", "second.png", "third.png"],
            "group_id": ["scene_0001", "scene_0001", "scene_0002"],
        })
        snapshot = original.copy(deep=True)
        overrides = pd.DataFrame({
            "filename": ["second.png"],
            "group_id": ["scene_0002"],
            "reason": ["Visual review confirms the same corridor"],
        })
        result, audit = apply_overrides(original, overrides)
        self.assertEqual(result.set_index("filename")["group_id"].to_dict(), {
            "first.png": "scene_0002", "second.png": "scene_0002", "third.png": "scene_0002",
        })
        self.assertEqual(audit, [{
            "filename": "second.png",
            "previous_group_id": "scene_0001",
            "group_id": "scene_0002",
            "reason": "Visual review confirms the same corridor",
            "affected_filenames": ["first.png", "second.png"], "operation": "merge_entire_component",
        }])
        pd.testing.assert_frame_equal(original, snapshot)
        globally_justified, global_audit = apply_overrides(
            original, overrides.drop(columns="reason"), reason="Documented author review"
        )
        pd.testing.assert_frame_equal(globally_justified, result)
        self.assertEqual(global_audit[0]["reason"], "Documented author review")
        with self.assertRaises(ValueError):
            apply_overrides(original, pd.DataFrame({"filename": ["second.png"], "group_id": [42]}),
                            reason="Splitting into a new group is forbidden")

    def test_invalid_overrides_refuse_without_mutating_assignments(self):
        original = pd.DataFrame({"filename": ["first.png"], "group_id": ["scene_0001"]})
        snapshot = original.copy(deep=True)
        invalid = [
            {"filename": ["first.png", "first.png"], "group_id": ["a", "b"], "reason": ["review", "review"]},
            {"filename": ["unknown.png"], "group_id": ["a"], "reason": ["review"]},
            {"filename": ["first.png"], "group_id": [""], "reason": ["review"]},
            {"filename": ["first.png"], "group_id": ["   "], "reason": ["review"]},
            {"filename": ["first.png"], "group_id": [None], "reason": ["review"]},
            {"filename": ["first.png"], "group_id": ["a"]},
            {"filename": ["first.png"], "group_id": ["a"], "reason": ["   "]},
        ]
        for values in invalid:
            with self.subTest(overrides=values):
                with self.assertRaises(ValueError):
                    apply_overrides(original, pd.DataFrame(values))
                pd.testing.assert_frame_equal(original, snapshot)

    def test_post_freeze_summary_counts_mixed_classes_without_changing_groups(self):
        paths = [Path(name) for name in (
            "clear.1.jpg", "nonclear.1.jpg", "clear.2.jpg", "clear.3.jpg", "nonclear.2.jpg"
        )]
        groups = pd.DataFrame({
            "filename": [path.name for path in paths],
            "group_id": ["scene_0001", "scene_0001", "scene_0002", "scene_0002", "scene_0003"],
        })
        snapshot = groups.copy(deep=True)
        summary = summarize_groups(paths, groups)
        self.assertEqual(summary["image_count"], 5)
        self.assertEqual(summary["group_count"], 3)
        self.assertEqual(summary["size_distribution"], {"1": 1, "2": 2})
        self.assertEqual(summary["singleton_count"], 1)
        self.assertEqual(summary["largest_group_size"], 2)
        self.assertEqual(summary["mixed_class_group_count"], 1)
        pd.testing.assert_frame_equal(groups, snapshot)


if __name__ == "__main__":
    unittest.main()
