#!/usr/bin/env python3
"""Reuse verified neural jobs after the narrowly scoped fixed-refit repair."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fine_tuning import (completed_job, dataset_content_fingerprint, scan_dataset,
                             sha256, write_json)


def validate_source_compatibility(previous_source, current_source):
    """All neural/protocol code must be unchanged, including module globals."""
    previous, current = ast.parse(previous_source), ast.parse(current_source)
    allowed = "run_fixed_job"
    archive_statement = ast.parse('if (directory / "canonical_features.json").exists():\n    artifacts.append("canonical_features.json")').body[0]
    before, after = {}, {}
    for tree, mapping, normalize in ((previous, before, False), (current, after, True)):
        for position, node in enumerate(tree.body):
            node = copy.deepcopy(node)
            name = getattr(node, "name", f"module_node_{position}")
            if name == allowed:
                continue
            if normalize and name == "complete_job":
                node.body = [statement for statement in node.body
                             if ast.dump(statement) != ast.dump(archive_statement)]
            if normalize and name == "run":
                for item in ast.walk(node):
                    if isinstance(item, ast.Dict):
                        pairs = [(key, value) for key, value in zip(item.keys, item.values)
                                 if not isinstance(key, ast.Constant) or key.value != "fixed_refit"]
                        item.keys = [key for key, _ in pairs]
                        item.values = [value for _, value in pairs]
            mapping[name] = ast.dump(node)
    if before != after:
        changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        raise ValueError(f"Recovery refuses changes outside the fixed-refit repair: {changed}")


def historical_source(expected, supplied_path=None):
    if supplied_path:
        contents = Path(supplied_path).read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError("Supplied legacy source does not match the original run")
        return contents
    relative = "src/fine_tuning.py"
    repository = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip()
    relative = (ROOT / relative).relative_to(repository).as_posix()
    revisions = ["HEAD"] + subprocess.check_output(
        ["git", "log", "--format=%H", "-100", "--", relative], cwd=repository, text=True).splitlines()
    for revision in dict.fromkeys(revisions):
        result = subprocess.run(["git", "show", f"{revision}:{relative}"], cwd=repository, capture_output=True)
        if result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == expected:
            return result.stdout
    raise ValueError("Original executor source unavailable; supply --legacy-source matching its recorded hash")


def current_sources(metadata):
    paths = {"fine_tuning": ROOT / "src/fine_tuning.py", "artifact": ROOT / "src/artifact.py",
             "run_experiments": ROOT / "src/run_experiments.py",
             "fine_config": Path(metadata["config_path"]),
             "fixed_config": Path(metadata["config"]["fixed_features"]["config"]),
             "fold_assignments": Path(metadata["config"]["fold_assignments"]),
             "fixed_metrics": Path(metadata["config"]["fixed_features"]["results"]) / "per_fold_metrics.csv"}
    result = {name: sha256(path) for name, path in paths.items()}
    for name, expected in metadata["source_hashes"].items():
        if name != "fine_tuning" and result.get(name) != expected:
            raise ValueError(f"Recovery refuses a changed protocol/config/reference: {name}")
    result["fixed_refit"] = sha256(ROOT / "src/fixed_refit.py")
    return result


def run_identity(config, sources, dataset, folds, device, weights):
    return hashlib.sha256(json.dumps({"config": config, "sources": sources, "dataset": dataset,
        "folds": folds, "device": device, "imagenet_weights": weights}, sort_keys=True).encode()).hexdigest()


def prepare_recovery(source, output, runtime, legacy_source=None, dry_run=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Recovery output must be separate from the original run")
    metadata_path = source / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("smoke") or not metadata["config"]["fixed_features"]["enabled"]:
        raise ValueError("Recovery applies to a full run with fixed-feature benchmarks")
    original_metadata_sha = sha256(metadata_path)
    if metadata["runtime"] != runtime:
        raise ValueError("Recovery requires the original hardware/software runtime")
    previous_source = historical_source(metadata["source_hashes"]["fine_tuning"], legacy_source)
    validate_source_compatibility(previous_source, (ROOT / "src/fine_tuning.py").read_text())
    sources = current_sources(metadata)
    existing = output / "recovery_manifest.json"
    existing_provenance = None
    if output.exists():
        if not existing.exists():
            raise ValueError("Refusing an existing output without a recovery manifest")
        provenance = json.loads(existing.read_text())
        if (provenance["source_run_metadata_sha256"] != original_metadata_sha
                or provenance["current_source_hashes"] != sources
                or provenance["recovery_script_sha256"] != sha256(__file__)):
            raise ValueError("Recovery output belongs to a changed source run or repair")
        existing_provenance = provenance
    manifests = sorted((source / "jobs").glob("*/manifest.json"))
    neural = []
    devices, datasets = set(), set()
    for path in manifests:
        manifest = json.loads(path.read_text())
        request = manifest["request"]
        devices.add(request["device"])
        datasets.add(request["dataset"])
        if request["identity"] != metadata["identity"] or request["config"] != metadata["config"]:
            raise ValueError(f"Original job identity/config differs: {path.parent.name}")
        if request["family"] == "fixed_features":
            continue
        if not completed_job(path.parent, request):
            raise ValueError(f"Recovery requires completed neural training: {path.parent.name}")
        neural.append((path, manifest))
        if len(neural) % 25 == 0:
            print(f"Verified {len(neural)} completed neural jobs", flush=True)
    expected = {(experiment, seed, fold) for experiment in metadata["experiments"]
                for seed in metadata["seeds"] for fold in metadata["folds"]}
    observed = [(m["request"]["experiment"], m["request"]["seed"], m["request"]["fold"]) for _, m in neural]
    if len(observed) != len(expected) or set(observed) != expected or len(devices) != 1 or len(datasets) != 1:
        raise ValueError("Incomplete or duplicate neural job coverage")
    device, dataset = next(iter(devices)), next(iter(datasets))
    paths, _ = scan_dataset(dataset)
    if dataset_content_fingerprint(paths) != metadata["dataset_content_sha256"]:
        raise ValueError("Recovery refuses changed dataset contents")
    for weights in metadata["imagenet_weights"].values():
        if sha256(weights["path"]) != weights["sha256"]:
            raise ValueError("Recovery refuses changed ImageNet weights")
    old_identity = run_identity(metadata["config"], metadata["source_hashes"], metadata["dataset_content_sha256"],
        metadata["folds"], device, metadata["imagenet_weights"])
    if old_identity != metadata["identity"]:
        raise ValueError("Original run identity does not match its recorded inputs")
    identity = run_identity(metadata["config"], sources, metadata["dataset_content_sha256"],
        metadata["folds"], device, metadata["imagenet_weights"])
    if existing_provenance is not None:
        recovered_metadata = json.loads((output / "run_metadata.json").read_text())
        if recovered_metadata["identity"] != identity or recovered_metadata["source_hashes"] != sources:
            raise ValueError("Recovery metadata differs from verified repair inputs")
        return existing_provenance
    fixed_config = json.loads(Path(metadata["config"]["fixed_features"]["config"]).read_text())
    provenance = {"prepared_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(source),
        "source_run_metadata_sha256": original_metadata_sha, "original_identity": old_identity,
        "identity": identity, "original_source_hashes": metadata["source_hashes"], "current_source_hashes": sources,
        "recovery_script_sha256": sha256(__file__), "neural_source_compatibility": "all neural/protocol AST unchanged",
        "neural_jobs_reused": len(neural), "fixed_jobs_reused": 0,
        "fixed_jobs_to_recompute": len(fixed_config["experiments"]) * len(metadata["folds"]),
        "original_run_modified": False, "reuse_method": "hard-linked immutable artifacts; copied manifest with audited identity",
        "device": device, "dataset": dataset, "config_path": metadata["config_path"],
        "neural_jobs": [{"job": path.parent.name, "original_manifest_sha256": sha256(path)} for path, _ in neural]}
    if dry_run:
        return provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != output.parent.stat().st_dev:
        raise ValueError("Choose a recovery directory on the same filesystem to reuse models without copying them")
    staging = output.with_name(output.name + ".preparing")
    if staging.exists():
        raise ValueError("Unfinished recovery staging directory needs inspection")
    (staging / "jobs").mkdir(parents=True)
    for path, original in neural:
        destination = staging / "jobs" / path.parent.name
        shutil.copytree(path.parent, destination, copy_function=os.link)
        manifest = copy.deepcopy(original)
        manifest["request"]["identity"] = identity
        manifest["reused_from"] = {"directory": str(path.parent), "manifest_sha256": sha256(path),
                                   "identity": old_identity, "source_hashes": metadata["source_hashes"]}
        write_json(destination / "manifest.json", manifest)
    recovered_metadata = copy.deepcopy(metadata)
    recovered_metadata.update(identity=identity, source_hashes=sources, completed=False)
    write_json(staging / "run_metadata.json", recovered_metadata)
    write_json(staging / "recovery_manifest.json", provenance)
    for filename in ("fold_assignments.csv", "inner_validation_indices.csv"):
        shutil.copy2(source / filename, staging / filename)
    staging.replace(output)
    if sha256(metadata_path) != original_metadata_sha:
        raise ValueError("Original metadata changed during recovery")
    for item in provenance["neural_jobs"]:
        if sha256(source / "jobs" / item["job"] / "manifest.json") != item["original_manifest_sha256"]:
            raise ValueError("Original job manifest changed during recovery")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("results/fine_tuning"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fine_tuning_recovered"))
    parser.add_argument("--legacy-source", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true", help="Prepare safely, then run only pending fixed benchmarks")
    args = parser.parse_args()
    metadata = json.loads((args.source_dir / "run_metadata.json").read_text())
    device = json.loads(next((args.source_dir / "jobs").glob("*/manifest.json")).read_text())["request"]["device"]
    command = [sys.executable, str(ROOT / "src/fine_tuning.py"), "--runtime-info", "--device", device,
               "--threads", str(metadata["config"]["cpu_threads"])]
    runtime = subprocess.run(command, capture_output=True, text=True)
    if runtime.returncode:
        raise RuntimeError(f"Runtime check failed:\n{runtime.stderr}")
    plan = prepare_recovery(args.source_dir, args.output_dir, json.loads(runtime.stdout), args.legacy_source, args.dry_run)
    print(f"Verified neural jobs reused: {plan['neural_jobs_reused']}; fixed jobs to recompute: {plan['fixed_jobs_to_recompute']}")
    if args.run:
        command = [sys.executable, str(ROOT / "src/fine_tuning.py"), "--dataset", plan["dataset"],
                   "--config", plan["config_path"], "--output-dir", str(args.output_dir.resolve()), "--device", plan["device"]]
        subprocess.run(command, check=True)
    else:
        print("Dry-run only" if args.dry_run else f"Recovery prepared: {args.output_dir}")


if __name__ == "__main__":
    main()
