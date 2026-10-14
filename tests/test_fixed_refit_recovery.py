"""Cache and recovery contracts; actual fold reproduction is checked on GPU."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.artifact import dataset_content_fingerprint
from src.fixed_refit import canonical_fixed_features, reference_batch_size
from scripts import resume_fine_tuning as recovery


@pytest.mark.parametrize("command,expected", [
    (["run_experiments.py"], 4), (["runner", "--batch-size", "16"], 16),
    (["runner", "--batch-size=8"], 8),
])
def test_reference_batch_size_uses_recorded_canonical_setting(command, expected):
    assert reference_batch_size({"command": command}) == expected


@pytest.mark.parametrize("command", [["runner", "--batch-size", "0"],
    ["runner", "--batch-size"], ["runner", "--batch-size", "4", "--batch-size=8"]])
def test_invalid_reference_batch_size_is_rejected(command):
    with pytest.raises(ValueError):
        reference_batch_size({"command": command})


def test_canonical_cache_uses_global_batches_and_rejects_tampering(tmp_path, monkeypatch):
    import src.fine_tuning as fine
    paths = [tmp_path / f"image_{index}.jpg" for index in range(5)]
    for path in paths:
        path.write_bytes(path.name.encode())
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "run_metadata.json").write_text(json.dumps({"command": ["runner", "--batch-size", "2"],
        "dataset_content_sha256": dataset_content_fingerprint(paths)}))
    monkeypatch.setattr(fine, "runtime_metadata", lambda tf: {"runtime": "fixture"})
    monkeypatch.setattr(fine, "image_batch", lambda batch, processor, size, tf:
        np.asarray([[int(path.stem.split("_")[-1]), 1] for path in batch], dtype=np.float32))

    class Model:
        trainable = False
        output_shape = (None, 2)

        def __init__(self):
            self.batches = []

        def predict(self, batch, verbose=0):
            self.batches.append(batch[:, 0].tolist())
            return batch

    model = Model()
    request = {"fold_assignments": str(reference / "fold_assignments.csv"), "extractors": ["fixture"],
               "imagenet_weights": {"fixture": {"sha256": "weights"}}}
    matrices, first = canonical_fixed_features(request, paths, [model], [(None, (1, 1))], object(), tmp_path / "cache")
    assert model.batches == [[0, 1], [2, 3], [4]]
    np.testing.assert_array_equal(matrices[0][:, 0], np.arange(5))
    _, reused = canonical_fixed_features(request, paths, [model], [(None, (1, 1))], object(), tmp_path / "cache")
    assert reused["cache"][0]["reused"] and len(model.batches) == 3
    cache = next((tmp_path / "cache").glob("*/features.npy"))
    del matrices
    cache.write_bytes(b"corrupted cache")
    with pytest.raises(ValueError, match="cache changed"):
        canonical_fixed_features(request, paths, [model], [(None, (1, 1))], object(), tmp_path / "cache")


OLD = '''def train_neural():
    return "unchanged"
def run_fixed_job():
    return "old batching"
def complete_job(directory, artifacts):
    return artifacts
def run():
    source_hashes = {"fine_tuning": "old"}
    return source_hashes
'''
NEW = '''def train_neural():
    return "unchanged"
def run_fixed_job():
    return "canonical batching"
def complete_job(directory, artifacts):
    if (directory / "canonical_features.json").exists():
        artifacts.append("canonical_features.json")
    return artifacts
def run():
    source_hashes = {"fine_tuning": "old", "fixed_refit": "new"}
    return source_hashes
'''


def test_source_compatibility_accepts_only_the_fixed_repair():
    recovery.validate_source_compatibility(OLD, NEW)
    with pytest.raises(ValueError, match="train_neural"):
        recovery.validate_source_compatibility(OLD, NEW.replace('return "unchanged"', 'return "new neural protocol"'))
    with pytest.raises(ValueError, match="run"):
        recovery.validate_source_compatibility(OLD, NEW.replace('return source_hashes', 'return "changed protocol"'))
    with pytest.raises(ValueError, match="complete_job"):
        recovery.validate_source_compatibility(OLD, NEW.replace('return artifacts', 'return []'))


@pytest.fixture
def miniature_run(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/fine_tuning.py").write_text(NEW)
    legacy = tmp_path / "legacy.py"
    legacy.write_text(OLD)
    for name in ("artifact.py", "run_experiments.py", "fixed_refit.py"):
        (tmp_path / "src" / name).write_text(name)
    dataset = tmp_path / "images"
    dataset.mkdir()
    for name in ("clear.000.jpg", "nonclear.000.jpg"):
        (dataset / name).write_bytes(name.encode())
    folds, metrics, fixed_config, config_path, weights = [tmp_path / name for name in
        ("folds.csv", "reference_metrics.csv", "fixed.json", "fine.json", "weights.h5")]
    folds.write_text("original folds")
    metrics.write_text("original metrics")
    fixed_config.write_text(json.dumps({"experiments": [{"name": "fixed"}]}))
    weights.write_bytes(b"original weights")
    config = {"architectures": ["MobileNet"], "initialization_seeds": [1980],
        "fixed_features": {"enabled": True, "config": str(fixed_config), "results": str(tmp_path)},
        "fold_assignments": str(folds)}
    (tmp_path / "per_fold_metrics.csv").write_bytes(metrics.read_bytes())
    config_path.write_text(json.dumps(config))
    sources = {"fine_tuning": recovery.sha256(legacy), "artifact": recovery.sha256(tmp_path / "src/artifact.py"),
        "run_experiments": recovery.sha256(tmp_path / "src/run_experiments.py"), "fine_config": recovery.sha256(config_path),
        "fixed_config": recovery.sha256(fixed_config), "fold_assignments": recovery.sha256(folds),
        "fixed_metrics": recovery.sha256(metrics)}
    imagenet = {"MobileNet": {"path": str(weights), "sha256": recovery.sha256(weights)}}
    paths = sorted(dataset.iterdir())
    content = dataset_content_fingerprint(paths)
    identity = recovery.run_identity(config, sources, content, [1], "gpu", imagenet)
    source = tmp_path / "original"
    job = source / "jobs/ft_d_mobilenet__fold_01__seed_1980"
    job.mkdir(parents=True)
    request = {"experiment": "ft_d_mobilenet", "seed": 1980, "fold": 1,
               "config": config, "identity": identity, "device": "gpu", "dataset": str(dataset), "family": "fine_tuning"}
    artifacts = {}
    for name in ("metrics.json", "predictions.csv", "costs.json", "model.keras"):
        (job / name).write_bytes(name.encode())
        artifacts[name] = recovery.sha256(job / name)
    (job / "manifest.json").write_text(json.dumps({"request": request, "state": "completed", "artifacts": artifacts}))
    metadata = {"identity": identity, "runtime": {"runtime": "fixture"}, "source_hashes": sources,
        "config": config, "config_path": str(config_path), "experiments": ["ft_d_mobilenet"],
        "seeds": [1980], "folds": [1], "dataset_content_sha256": content, "imagenet_weights": imagenet,
        "smoke": False, "completed": False}
    (source / "run_metadata.json").write_text(json.dumps(metadata))
    for name in ("fold_assignments.csv", "inner_validation_indices.csv"):
        (source / name).write_text(name)
    return source, tmp_path / "recovered", job, legacy, metadata["runtime"]


def test_recovery_preserves_original_and_links_verified_models(miniature_run):
    source, output, job, legacy, runtime = miniature_run
    original = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    plan = recovery.prepare_recovery(source, output, runtime, legacy, dry_run=True)
    assert not output.exists() and plan["neural_jobs_reused"] == 1
    plan = recovery.prepare_recovery(source, output, runtime, legacy)
    recovered_job = output / "jobs" / job.name
    assert recovered_job.joinpath("model.keras").stat().st_ino == job.joinpath("model.keras").stat().st_ino
    manifest = json.loads(recovered_job.joinpath("manifest.json").read_text())
    assert manifest["request"]["identity"] == plan["identity"] != plan["original_identity"]
    assert manifest["reused_from"]["manifest_sha256"] == recovery.sha256(job / "manifest.json")
    assert recovery.prepare_recovery(source, output, runtime, legacy) == plan
    assert all(path.read_bytes() == contents for path, contents in original.items())


@pytest.mark.parametrize("corruption", ["model", "weights", "config", "dataset", "runtime"])
def test_recovery_rejects_changed_inputs_and_corrupt_models(miniature_run, corruption):
    source, output, job, legacy, runtime = miniature_run
    if corruption == "model":
        (job / "model.keras").write_bytes(b"changed model")
    elif corruption == "weights":
        (source.parent / "weights.h5").write_bytes(b"changed weights")
    elif corruption == "config":
        (source.parent / "fine.json").write_text("{}")
    elif corruption == "dataset":
        (source.parent / "images/clear.000.jpg").write_bytes(b"changed pixels")
    else:
        runtime = {"runtime": "different"}
    with pytest.raises(ValueError):
        recovery.prepare_recovery(source, output, runtime, legacy)
    assert not output.exists()
