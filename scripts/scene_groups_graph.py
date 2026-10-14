"""Image-only graph rules and audited review operations for scene grouping V2."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def archive_v1(root):
    """Keep an immutable, checksummed V1 copy before replacing any live artifact."""
    root = Path(root)
    archive = root / "results/scene_groups_v1"
    surface = root / "data/surface_groups.csv"
    manifest_path = archive / "archive_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest["artifacts"].items():
            if not (archive / name).is_file() or file_sha256(archive / name) != digest:
                raise ValueError(f"V1 archive changed or is incomplete: {name}")
        if not surface.exists() or file_sha256(surface) != manifest["artifacts"]["groups.csv"]:
            raise ValueError("surface_groups.csv differs from the preserved V1")
        return archive
    if archive.exists():
        raise ValueError("Refusing to overwrite an incomplete V1 archive")
    source = root / "results/scene_groups"
    if not source.exists():
        raise ValueError("The original scene_groups results are required to archive V1")
    summary = json.loads((source / "summary.json").read_text())
    if summary.get("protocol_version", 1) != 1:
        raise ValueError("Cannot archive V2 results as V1")
    original = root / "data/groups.csv"
    if surface.exists() and file_sha256(surface) != file_sha256(original):
        raise ValueError("Refusing to overwrite an existing surface_groups.csv")
    staging = archive.with_name(archive.name + ".tmp")
    if staging.exists():
        raise ValueError("An unfinished V1 archive needs inspection before retrying")
    shutil.copytree(source, staging)
    for name in ("groups.csv", "groups_auto.csv", "groups_overrides.csv"):
        path = root / "data" / name
        if path.exists():
            shutil.copy2(path, staging / name)
    shutil.copy2(root / "configs/scene_groups.json", staging / "config.json")
    (staging / "source").mkdir()
    for name in ("build_scene_groups.py", "scene_groups_review.py"):
        shutil.copy2(root / "scripts" / name, staging / "source" / name)
    artifacts = {str(p.relative_to(staging)): file_sha256(p)
                 for p in sorted(staging.rglob("*")) if p.is_file()}
    write_json(staging / "archive_manifest.json", {
        "protocol_version": 1, "purpose": "Preserved appearance groups; not validated surface labels",
        "artifacts": artifacts,
    })
    staging.replace(archive)
    if not surface.exists():
        shutil.copy2(archive / "groups.csv", surface)
    return archive


def candidate_pairs(distances, similarities, max_phash=16, k=10):
    """pHash candidates OR mutual k-nearest neighbors; stable index tie breaking."""
    distances, cosine = np.asarray(distances), np.asarray(similarities)
    count = len(cosine)
    if (cosine.shape != (count, count) or distances.shape != cosine.shape
            or not np.isfinite(cosine).all() or not np.allclose(cosine, cosine.T)
            or not np.array_equal(distances, distances.T) or not 0 <= max_phash <= 64
            or not isinstance(k, (int, np.integer)) or k < 1):
        raise ValueError("Invalid candidate scores or settings")
    nearest = np.zeros_like(cosine, dtype=bool)
    for index in range(count):
        scores = cosine[index].copy()
        scores[index] = -np.inf
        neighbors = np.argsort(-scores, kind="stable")[:min(k, count - 1)]
        nearest[index, neighbors] = True
    mutual = nearest & nearest.T
    candidates = (distances <= max_phash) | mutual
    np.fill_diagonal(candidates, False)
    return candidates, mutual


def groups_from_edges(edges):
    edges = np.asarray(edges, dtype=bool)
    count = len(edges)
    if edges.shape != (count, count) or not np.array_equal(edges, edges.T):
        raise ValueError("Edges must be a symmetric square matrix")
    parents = np.arange(count)

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for first, second in zip(*np.where(np.triu(edges, 1))):
        a, b = root(first), root(second)
        parents[max(a, b)] = min(a, b)
    roots = [root(i) for i in range(count)]
    identifiers = {value: f"scene_{index + 1:04d}" for index, value in enumerate(sorted(set(roots)))}
    return np.asarray([identifiers[value] for value in roots])


def validate_groups(groups, filenames):
    if (not {"filename", "group_id"}.issubset(groups.columns)
            or groups[["filename", "group_id"]].isna().any().any()
            or groups["filename"].duplicated().any()
            or set(groups["filename"]) != set(filenames) or len(groups) != len(filenames)
            or groups["group_id"].astype(str).str.strip().eq("").any()):
        raise ValueError("Groups must assign every image exactly once with a nonempty group ID")


def apply_edge_rejections(edges, filenames, rejections):
    """Remove only reviewed graph edges, recording reasons; never edit group IDs."""
    names = {name: i for i, name in enumerate(filenames)}
    if not {"filename_a", "filename_b", "reason", "reviewer", "date"}.issubset(rejections.columns):
        raise ValueError("Edge rejections require filename_a, filename_b, reason, reviewer and date")
    result = np.asarray(edges, dtype=bool).copy()
    audit, seen = [], set()
    for row in rejections.to_dict("records"):
        a, b, reason = row["filename_a"], row["filename_b"], row["reason"]
        if a not in names or b not in names or a == b or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Invalid edge rejection or missing justification")
        reviewer, reviewed_date = row["reviewer"], row["date"]
        if (not isinstance(reviewer, str) or not reviewer.strip()
                or not isinstance(reviewed_date, str) or not reviewed_date.strip()):
            raise ValueError("Edge rejections require a reviewer and an ISO date")
        try:
            reviewed_date = date.fromisoformat(reviewed_date.strip()).isoformat()
        except ValueError as error:
            raise ValueError("Edge rejection date must be an ISO calendar date") from error
        pair = tuple(sorted((names[a], names[b])))
        if pair in seen or not result[pair]:
            raise ValueError("Duplicate rejection or pair that is not a confirmed edge")
        seen.add(pair)
        result[pair] = result[pair[::-1]] = False
        audit.append({"filename_a": filenames[pair[0]], "filename_b": filenames[pair[1]],
                      "reason": reason.strip(), "reviewer": reviewer.strip(), "date": reviewed_date})
    return result, audit


def apply_overrides(groups, overrides, reason=None):
    """Merge whole components into existing target groups; never split a component."""
    validate_groups(groups, groups["filename"].tolist())
    if (not {"filename", "group_id"}.issubset(overrides.columns)
            or overrides[["filename", "group_id"]].isna().any().any()
            or overrides["filename"].duplicated().any()
            or not set(overrides["filename"]).issubset(set(groups["filename"]))):
        raise ValueError("Invalid, duplicate or unknown override image/group ID")
    result = groups.copy()
    result["group_id"] = result["group_id"].astype(str)
    aliases = {value: value for value in result["group_id"].unique()}
    audit = []
    for row in overrides.to_dict("records"):
        justification = row.get("reason", reason)
        target = str(row["group_id"])
        if target not in aliases or not isinstance(justification, str) or not justification.strip():
            raise ValueError("Merge overrides require an existing target group and a reason")
        previous = result.loc[result["filename"] == row["filename"], "group_id"].iloc[0]
        target = aliases[target]
        affected = result.loc[result["group_id"] == previous, "filename"].tolist()
        result.loc[result["group_id"] == previous, "group_id"] = target
        aliases = {key: target if value == previous else value for key, value in aliases.items()}
        audit.append({"filename": row["filename"], "previous_group_id": previous,
                      "group_id": target, "reason": justification.strip(),
                      "affected_filenames": affected, "operation": "merge_entire_component"})
    return result, audit


def validate_accepted_edges(groups, filenames, edges):
    validate_groups(groups, filenames)
    assignments = groups.set_index("filename").loc[filenames, "group_id"].to_numpy()
    for a, b in zip(*np.where(np.triu(edges, 1))):
        if assignments[a] != assignments[b]:
            raise ValueError(f"Accepted edge crosses final groups: {filenames[a]}, {filenames[b]}")


def graph_bridges(edges):
    """Tarjan bridges on accepted image-pair links, without labels or group splitting."""
    graph = np.asarray(edges, dtype=bool)
    discovered = np.full(len(graph), -1, dtype=int)
    low = discovered.copy()
    bridges = []
    clock = 0

    def visit(node, parent=-1):
        nonlocal clock
        discovered[node] = low[node] = clock
        clock += 1
        for neighbor in np.flatnonzero(graph[node]):
            if neighbor == parent:
                continue
            if discovered[neighbor] == -1:
                visit(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovered[node]:
                    bridges.append(tuple(sorted((int(node), int(neighbor)))))
            else:
                low[node] = min(low[node], discovered[neighbor])

    for node in range(len(graph)):
        if discovered[node] == -1:
            visit(node)
    return sorted(bridges)
