import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.build_scene_groups import load_embedding_cache, scene_components
from scripts.scene_groups_geometry import geometry_qualifies, quadrant_counts, structural_scores
from scripts.scene_groups_graph import (archive_v1, apply_edge_rejections, apply_overrides,
    candidate_pairs, graph_bridges, groups_from_edges, validate_accepted_edges)


ROOT = Path(__file__).resolve().parents[1]


def geometry_record(**changes):
    record = {"candidate": True, "evaluation_status": "evaluated", "homography_valid": True,
              "valid_match_count": 120, "ransac_inliers": 30,
              "quadrant_inliers_a": [10, 10, 10, 0], "quadrant_inliers_b": [10, 10, 10, 0]}
    record.update(changes)
    return record


def test_cosine_alone_never_creates_an_edge():
    groups, edges = scene_components(np.array([[0, 32], [32, 0]]), np.ones((2, 2)))
    assert not edges.any()
    assert groups[0] != groups[1]


@pytest.mark.parametrize("changes", [
    {"valid_match_count": 121},
    {"ransac_inliers": 29, "quadrant_inliers_a": [10, 10, 9, 0], "quadrant_inliers_b": [10, 10, 9, 0]},
    {"quadrant_inliers_a": [15, 15, 0, 0]},
    {"quadrant_inliers_b": [15, 15, 0, 0]},
    {"quadrant_inliers_a": [26, 2, 2, 0]},
    {"homography_valid": False},
    {"evaluation_status": "not_evaluated"},
])
def test_each_geometric_requirement_is_required(changes):
    valid = geometry_record()
    assert geometry_qualifies(valid)
    invalid = geometry_record(**changes)
    assert not geometry_qualifies(invalid)
    _, edges = scene_components(np.array([[0, 32], [32, 0]]), np.ones((2, 2)), structural={(0, 1): invalid})
    assert not edges.any()


def test_candidate_neighbors_are_mutual_and_hash_branch_is_independent():
    cosine = np.array([[1, .9, .2], [.9, 1, .8], [.2, .8, 1]])
    distances = np.full((3, 3), 64)
    np.fill_diagonal(distances, 0)
    candidates, mutual = candidate_pairs(distances, cosine, k=1)
    assert candidates[0, 1] and mutual[0, 1]
    assert not candidates[1, 2]  # 2 chooses 1, but 1 chooses 0.
    distances[0, 2] = distances[2, 0] = 16
    candidates, mutual = candidate_pairs(distances, cosine, k=1)
    assert candidates[0, 2] and not mutual[0, 2]
    assert not np.diag(candidates).any()
    np.testing.assert_array_equal(candidates, candidates.T)


def test_candidate_ties_have_stable_index_order():
    cosine, distances = np.ones((4, 4)), np.full((4, 4), 64)
    np.fill_diagonal(distances, 0)
    first, _ = candidate_pairs(distances, cosine, k=1)
    second, _ = candidate_pairs(distances, cosine, k=1)
    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first) == 2 and first[0, 1]


def test_rejection_removes_bridge_then_recomputes_components_with_audit():
    edges = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    original = edges.copy()
    names = ["a.jpg", "b.jpg", "c.jpg"]
    assert graph_bridges(edges) == [(0, 1), (1, 2)]
    accepted, audit = apply_edge_rejections(edges, names, pd.DataFrame([
        {"filename_a": "c.jpg", "filename_b": "b.jpg", "reason": "Repeated texture; different scene",
         "reviewer": "fixture author", "date": "2026-10-05"}]))
    assignments = groups_from_edges(accepted)
    assert assignments[0] == assignments[1] != assignments[2]
    assert audit[0] == {"filename_a": "b.jpg", "filename_b": "c.jpg", "reason": "Repeated texture; different scene",
                        "reviewer": "fixture author", "date": "2026-10-05"}
    np.testing.assert_array_equal(edges, original)
    groups = pd.DataFrame({"filename": names, "group_id": assignments})
    validate_accepted_edges(groups, names, accepted)
    with pytest.raises(ValueError, match="crosses"):
        validate_accepted_edges(groups, names, edges)


@pytest.mark.parametrize("rows", [
    [{"filename_a": "a", "filename_b": "a", "reason": "review"}],
    [{"filename_a": "a", "filename_b": "missing", "reason": "review"}],
    [{"filename_a": "a", "filename_b": "b", "reason": " "}],
    [{"filename_a": "a", "filename_b": "c", "reason": "not an edge"}],
    [{"filename_a": "a", "filename_b": "b", "reason": "one"},
     {"filename_a": "b", "filename_b": "a", "reason": "duplicate"}],
])
def test_invalid_rejections_do_not_mutate_original_edges(rows):
    edges = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    snapshot = edges.copy()
    with pytest.raises(ValueError):
        apply_edge_rejections(edges, ["a", "b", "c"], pd.DataFrame([
            {**row, "reviewer": "fixture author", "date": "2026-10-05"} for row in rows]))
    np.testing.assert_array_equal(edges, snapshot)


@pytest.mark.parametrize("changes", [
    {"reviewer": ""}, {"reviewer": None}, {"date": ""}, {"date": None},
    {"date": "2026-13-05"}, {"missing_column": "reviewer"}, {"missing_column": "date"},
])
def test_rejection_requires_reviewer_and_valid_date_without_mutating_edges(changes):
    edges = np.array([[0, 1], [1, 0]], dtype=bool)
    snapshot = edges.copy()
    row = {"filename_a": "a", "filename_b": "b", "reason": "Different scenes",
           "reviewer": "fixture author", "date": "2026-10-05"}
    if "missing_column" in changes:
        row.pop(changes["missing_column"])
    else:
        row.update(changes)
    with pytest.raises(ValueError, match="reviewer|date"):
        apply_edge_rejections(edges, ["a", "b"], pd.DataFrame([row]))
    np.testing.assert_array_equal(edges, snapshot)


def test_cycle_edges_are_not_bridges():
    edges = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(edges, False)
    assert graph_bridges(edges) == []


def test_merge_override_preserves_all_accepted_pairs():
    names = ["a", "b", "c"]
    edges = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    original = pd.DataFrame({"filename": names, "group_id": groups_from_edges(edges)})
    merged, audit = apply_overrides(original, pd.DataFrame([
        {"filename": "b", "group_id": original.loc[2, "group_id"], "reason": "Same scene confirmed visually"}]))
    assert merged["group_id"].nunique() == 1
    assert audit[0]["affected_filenames"] == ["a", "b"]
    validate_accepted_edges(merged, names, edges)


def test_quadrants_use_each_images_dimensions():
    assert quadrant_counts([[1, 1], [90, 1], [1, 60], [90, 60]], (100, 80)) == [1, 1, 1, 1]
    assert quadrant_counts([], (100, 80)) == [0, 0, 0, 0]


def test_real_orb_reciprocal_ransac_and_quadrant_counts(tmp_path):
    rng = np.random.default_rng(1980)
    texture = rng.integers(0, 256, (341, 256), dtype=np.uint8)
    paths = [tmp_path / "first.png", tmp_path / "shifted.png"]
    Image.fromarray(texture).save(paths[0])
    Image.fromarray(np.roll(texture, (7, 11), axis=(0, 1))).save(paths[1])
    config = json.loads((ROOT / "configs/scene_groups.json").read_text())["structural"]
    scored = structural_scores(paths, np.array([[0, 32], [32, 0]]), np.ones((2, 2)), config, 1980, 1)
    record = scored["records"][(0, 1)]
    assert record["geometry_qualified"]
    assert record["inlier_ratio"] == record["ransac_inliers"] / record["valid_match_count"]
    assert sum(record["quadrant_inliers_a"]) == sum(record["quadrant_inliers_b"]) == record["ransac_inliers"]
    assert record["supported_quadrants_a"] >= 3 and record["supported_quadrants_b"] >= 3
    _, edges = scene_components(np.array([[0, 32], [32, 0]]), np.ones((2, 2)), structural=scored["records"])
    assert edges[0, 1]


def test_v1_archive_is_immutable_and_surface_groups_are_original(tmp_path):
    source = tmp_path / "results/scene_groups"
    source.mkdir(parents=True)
    (source / "summary.json").write_text('{"group_count": 2}')
    (source / "mosaic.png").write_bytes(b"original mosaic")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/groups.csv").write_text("filename,group_id\na,x\nb,y\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/scene_groups.json").write_text('{"semantic_threshold": 0.795}')
    (tmp_path / "scripts").mkdir()
    for name in ("build_scene_groups.py", "scene_groups_review.py"):
        (tmp_path / "scripts" / name).write_text("original implementation")
    archive = archive_v1(tmp_path)
    before = {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    (source / "summary.json").write_text('{"protocol_version": 2}')
    (tmp_path / "data/groups.csv").write_text("new groups")
    assert archive_v1(tmp_path) == archive
    assert (tmp_path / "data/surface_groups.csv").read_bytes() == before["groups.csv"]
    assert before == {p.name: p.read_bytes() for p in archive.iterdir() if p.is_file()}
    (archive / "mosaic.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        archive_v1(tmp_path)


def test_pinned_embedding_cache_reuse_and_tampering_rejection(tmp_path):
    config = {"semantic_model": {"id": "encoder", "revision": "fixed", "weights_sha256": "weights"}}
    np.savez_compressed(tmp_path / "embeddings.npz", embeddings=np.eye(3))
    checksum = hashlib.sha256((tmp_path / "embeddings.npz").read_bytes()).hexdigest()
    metadata = {"identity": {"dataset_sha256": "pixels", **config, "fallback_model": None},
        "model": {"id": "encoder", "revision": "fixed", "actual_weights_sha256": "weights", "embedding_dimension": 3},
        "cache_sha256": checksum}
    (tmp_path / "embedding_metadata.json").write_text(json.dumps(metadata))
    cached = load_embedding_cache(tmp_path, "pixels", config, 3)
    np.testing.assert_array_equal(cached[0], np.eye(3))
    with pytest.raises(ValueError, match="pixels"):
        load_embedding_cache(tmp_path, "changed pixels", config, 3)
    (tmp_path / "embeddings.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_embedding_cache(tmp_path, "pixels", config, 3)


def test_complete_cli_freezes_groups_and_hashes_every_artifact(tmp_path, monkeypatch):
    import scripts.build_scene_groups as runner

    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "data").mkdir()
    original = tmp_path / "results/scene_groups"
    original.mkdir(parents=True)
    (original / "summary.json").write_text('{"group_count": 4}')
    for name in ("build_scene_groups.py", "scene_groups_review.py"):
        (tmp_path / "scripts" / name).write_text("original code")
    config = json.loads((ROOT / "configs/scene_groups.json").read_text())
    config["semantic_model"] = {"id": "fixture-encoder", "revision": "fixed", "weights_sha256": "weights"}
    config["fallback_model"] = None
    config_path = tmp_path / "configs/scene_groups.json"
    config_path.write_text(json.dumps(config))
    config_before = config_path.read_bytes()
    dataset = tmp_path / "images"
    dataset.mkdir()
    rng = np.random.default_rng(1980)
    names = ["clear.0.png", "clear.1.png", "nonclear.0.png", "nonclear.1.png"]
    for name in names:
        Image.fromarray(rng.integers(0, 256, (341, 256), dtype=np.uint8)).save(dataset / name)
    pd.DataFrame({"filename": names, "group_id": names}).to_csv(tmp_path / "data/groups.csv", index=False)
    images = [{"filename": name, "sha256": hashlib.sha256((dataset / name).read_bytes()).hexdigest()} for name in names]
    dataset_hash = hashlib.sha256(json.dumps(images, sort_keys=True).encode()).hexdigest()
    np.savez_compressed(original / "embeddings.npz", embeddings=np.eye(4))
    cache_hash = hashlib.sha256((original / "embeddings.npz").read_bytes()).hexdigest()
    cache_metadata = {"identity": {"dataset_sha256": dataset_hash, "semantic_model": config["semantic_model"], "fallback_model": None},
        "model": {"id": "fixture-encoder", "revision": "fixed", "actual_weights_sha256": "weights", "embedding_dimension": 4},
        "cache_sha256": cache_hash}
    (original / "embedding_metadata.json").write_text(json.dumps(cache_metadata))
    output = tmp_path / "results/scene_groups_v2"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["build_scene_groups.py", "--dataset", str(dataset), "--config", str(config_path),
        "--output-dir", str(output), "--groups-dir", str(tmp_path / "data"),
        "--edge-rejections", str(tmp_path / "data/edge_rejections.csv"),
        "--overrides", str(tmp_path / "data/groups_overrides.csv"), "--embedding-cache-source", str(original)])
    runner.main()
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["completed"] and metadata["protocol_version"] == 2
    assert metadata["class_diagnostics_only_after_group_csv_freeze"]
    assert metadata["accepted_edges_cross_groups"] == 0
    assert config_path.read_bytes() == config_before
    for path, digest in metadata["artifacts"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    pairs = pd.read_csv(output / "pairs.csv", low_memory=False)
    assert len(pairs) == 6 and pairs["evaluation_status"].eq("evaluated").all()
    assert {"valid_matches", "ransac_inliers", "inlier_ratio", "quadrants_a", "quadrants_b", "edge_reason"}.issubset(pairs)
    for required, existing in (("valid_matches", "valid_match_count"), ("quadrants_a", "quadrant_inliers_a"),
                               ("quadrants_b", "quadrant_inliers_b"), ("edge_reason", "edge_criteria")):
        pd.testing.assert_series_equal(pairs[required], pairs[existing], check_names=False)
    assert list(pd.read_csv(output / "fold_diagnostics.csv").columns) == [
        "fold", "test_image_count", "clear", "nonclear", "clear_fraction", "groups_disjoint"]
    assert json.loads((output / "group_freeze.json").read_text())["groups_sha256"] == metadata["frozen_groups_sha256"]


def test_bridge_correspondence_mosaics_cover_both_links_in_a_chain(tmp_path):
    from scripts.scene_groups_review import write_bridge_reviews, write_group_reviews

    paths = [tmp_path / f"member{i}.png" for i in range(3)]
    for path in paths:
        Image.new("RGB", (256, 341), "gray").save(path)
    names = [path.name for path in paths]
    groups = pd.DataFrame({"filename": names, "group_id": ["scene_0001"] * 3})
    record = geometry_record()
    record.update(inlier_ratio=.25, _points_a=np.array([[50, 50]] * 30, dtype=float),
                  _points_b=np.array([[55, 55]] * 30, dtype=float), _inlier_mask=np.ones(30, dtype=bool))
    structural = {"image_size": (256, 341), "records": {(0, 1): record, (1, 2): record}}
    pairs = pd.DataFrame([{"filename_a": names[a], "filename_b": names[b], "edge_criteria": "semantic_geometry",
        "semantic_cosine": .9, "phash_distance": 32, "evaluation_status": "evaluated", "ransac_inliers": 30,
        "accepted_edge": True, "is_bridge": True} for a, b in ((0, 1), (1, 2))])
    pairs = pd.concat([pairs, pd.DataFrame([{"filename_a": names[0], "filename_b": names[2],
        "edge_criteria": "", "semantic_cosine": .4, "phash_distance": 32,
        "evaluation_status": "not_evaluated", "ransac_inliers": np.nan,
        "accepted_edge": False, "is_bridge": False}])], ignore_index=True)
    review = write_bridge_reviews(paths, ["img_a", "img_b", "img_c"], groups, [(0, 1), (1, 2)], structural, pairs, tmp_path)
    assert review["review_count"] == 2
    for artifact in review["artifacts"]:
        assert Path(artifact).exists()
    sheet = pd.read_csv(review["csv_path"])
    assert sheet["group_size"].eq(3).all()
    cosine = np.array([[1, .9, .4], [.9, 1, .9], [.4, .9, 1]])
    group_review = write_group_reviews(paths, ["img_a", "img_b", "img_c"], groups, cosine, tmp_path, pairs=pairs)
    row = pd.read_csv(group_review["review_sheet_path"]).iloc[0]
    assert row["accepted_edge_count"] == 2 and row["bridge_count"] == 2
    assert row["not_evaluated_internal_pairs"] == 1
    assert row["min_internal_cosine"] == .4
    assert row["min_internal_ransac_inliers_evaluated"] == 30
