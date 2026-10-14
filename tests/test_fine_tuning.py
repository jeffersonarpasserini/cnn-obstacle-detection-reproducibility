"""Matched-fold checks use archived predictions, not simulated training results."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.artifact import prediction_metrics
from src.fine_tuning import inner_split, load_outer_folds, validate_results

ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENTS = ROOT / "results/full_search/fold_assignments.csv"
MODEL = "a_resnet101v2_logistic"
SEEDS = [1980, 2025]


def independent_metrics(rows):
    truth, predicted = rows.true_label, rows.predicted_label
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(truth, predicted),
        "f1": f1_score(truth, predicted, zero_division=0),
        "f1_clear": f1_score(truth, predicted, pos_label=1, zero_division=0),
        "f1_obstructed": f1_score(truth, predicted, pos_label=0, zero_division=0),
        "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "mcc": matthews_corrcoef(truth, predicted),
        "tn_obstructed": tn, "fp_obstructed_as_clear": fp,
        "fn_clear_as_obstructed": fn, "tp_clear": tp,
        "obstructed_recall": recall_score(truth, predicted, pos_label=0),
        "clear_recall": recall_score(truth, predicted, pos_label=1),
        "obstructed_precision": precision_score(truth, predicted, pos_label=0),
        "clear_precision": precision_score(truth, predicted, pos_label=1),
        "obstructed_false_negative_rate": fp / (tn + fp),
        "roc_auc": roc_auc_score(truth, rows.score_clear),
    }


@pytest.fixture(scope="module")
def archived():
    assignments = pd.read_csv(ASSIGNMENTS)
    paths = [ROOT / "via-dataset" / name for name in assignments.filename]
    labels = assignments.label.to_numpy()
    predictions = pd.read_csv(
        ROOT / "results/article_selected_10fold/per_sample_predictions/group_0000.csv.gz"
    )
    assert set(predictions.experiment) == {MODEL}
    # This archive stores logistic decision margins. Sigmoid changes no decisions.
    predictions["score_clear"] = expit(predictions.score_clear.to_numpy(dtype=float))
    np.testing.assert_array_equal(
        predictions.predicted_label, (predictions.score_clear > 0.5).astype(int)
    )
    records = pd.read_csv(ROOT / "results/full_search/per_fold_metrics.csv")
    records = records.loc[records.experiment == MODEL].copy()
    records["obstructed_false_negative_rate"] = (
        records.fp_obstructed_as_clear /
        (records.tn_obstructed + records.fp_obstructed_as_clear)
    )
    # Replication exercises seed accounting only; these are archived fixed-model rows.
    predictions = pd.concat([predictions.assign(seed=seed) for seed in SEEDS], ignore_index=True)
    records = pd.concat([records.assign(seed=seed) for seed in SEEDS], ignore_index=True)
    return assignments, paths, labels, records, predictions


def test_outer_folds_match_csv_and_original_stratified_kfold(archived, tmp_path):
    assignments, paths, labels, _, _ = archived
    folds = load_outer_folds(ASSIGNMENTS, paths, labels)
    original = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=1980).split(paths, labels))
    assert len(folds) == 10
    for number, ((train, test), (original_train, original_test)) in enumerate(zip(folds, original), 1):
        expected_test = assignments.loc[assignments.test_fold == number, "sample_index"].to_numpy()
        np.testing.assert_array_equal(test, expected_test)
        np.testing.assert_array_equal(test, original_test)
        np.testing.assert_array_equal(train, original_train)
    # A different valid CSV must be honored, rather than regenerating the split.
    changed = assignments.copy()
    first = changed.index[(changed.test_fold == 1) & (changed.label == 1)][0]
    second = changed.index[(changed.test_fold == 2) & (changed.label == 1)][0]
    changed.loc[[first, second], "test_fold"] = [2, 1]
    path = tmp_path / "assignments.csv"
    changed.to_csv(path, index=False)
    changed_folds = load_outer_folds(path, paths, labels)
    assert first not in changed_folds[0][1] and second in changed_folds[0][1]


def test_runner_rejects_changed_folds_and_protects_canonical_results(archived, tmp_path):
    from argparse import Namespace
    from src.fine_tuning import run

    assignments, _, _, _, _ = archived
    changed = assignments.copy()
    first = changed.index[(changed.test_fold == 1) & (changed.label == 1)][0]
    second = changed.index[(changed.test_fold == 2) & (changed.label == 1)][0]
    changed.loc[[first, second], "test_fold"] = [2, 1]
    folds_path = tmp_path / "changed_folds.csv"
    changed.to_csv(folds_path, index=False)
    config = json.loads((ROOT / "configs/fine_tuning.json").read_text())
    config["fold_assignments"] = str(folds_path)
    config["fixed_features"]["results"] = str(tmp_path / "alternate_reference")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    args = Namespace(config=str(config_path), dataset=str(ROOT / "via-dataset"),
                     output_dir=str(tmp_path / "output"), device="cpu", smoke=True,
                     smoke_all=False, prefetch_weights=False, compare_smoke=None)
    with pytest.raises(ValueError, match="canonical full_search folds"):
        run(args)
    config["fold_assignments"] = str(ASSIGNMENTS)
    config_path.write_text(json.dumps(config))
    args.output_dir = str(ROOT / "results/full_search")
    with pytest.raises(ValueError, match="must not overwrite full_search"):
        run(args)


def test_internal_validation_is_stratified_reproducible_and_outer_train_only(archived):
    _, paths, labels, _, _ = archived
    for outer_train, test in load_outer_folds(ASSIGNMENTS, paths, labels):
        train, validation = inner_split(outer_train, labels)
        expected_train, expected_validation = train_test_split(
            outer_train, test_size=0.2, stratify=labels[outer_train], random_state=1980
        )
        np.testing.assert_array_equal(train, np.sort(expected_train))
        np.testing.assert_array_equal(validation, np.sort(expected_validation))
        assert len(validation) == int(np.ceil(0.2 * len(outer_train)))
        assert set(train) | set(validation) == set(outer_train)
        assert not (set(train) & set(validation))
        assert not (set(validation) & set(test))
        assert set(labels[train]) == set(labels[validation]) == {0, 1}
        for _ in SEEDS:
            repeated = inner_split(outer_train, labels, fraction=0.2, seed=1980)
            np.testing.assert_array_equal(repeated[0], train)
            np.testing.assert_array_equal(repeated[1], validation)


def test_label_update_uses_one_internal_fit_image_in_each_canonical_fold(archived):
    from src.fine_tuning import label_update_indices

    _, paths, labels, _, _ = archived
    for outer_train, test in load_outer_folds(ASSIGNMENTS, paths, labels):
        fit_train, validation = inner_split(outer_train, labels)
        previous, new_image = label_update_indices(outer_train, fit_train)
        assert len(previous) == len(outer_train) - 1
        assert set(outer_train) - set(previous) == {new_image}
        np.testing.assert_array_equal(previous, outer_train[outer_train != new_image])
        assert new_image == fit_train[-1]
        assert new_image not in set(test) | set(validation)


def test_native_deployment_preserves_probabilities_without_live_optimizer_changes(tmp_path, monkeypatch):
    import io
    import zipfile
    # Serialization does not need GPU memory. Keep the pytest parent on CPU
    # so opt-in GPU integration subprocesses can use the board independently.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    tf = pytest.importorskip("tensorflow")
    tf.config.set_visible_devices([], "GPU")
    import h5py
    from src.fine_tuning import check_reload_values, save_deployment

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(3,)),
        tf.keras.layers.Rescaling(1 / 255),
        tf.keras.layers.Normalization(mean=[0.1, 0.2, 0.3], variance=[1.0, 2.0, 3.0]),
        tf.keras.layers.Dense(
            1, activation="sigmoid", kernel_initializer=tf.keras.initializers.Constant(0.2),
            bias_initializer="zeros",
        ),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="binary_crossentropy")
    images = np.arange(24, dtype=np.float32).reshape(8, 3)
    labels = np.array([0, 1] * 4, dtype=np.float32).reshape(8, 1)
    model.train_on_batch(images, labels)
    expected = model(images, training=False).numpy()
    optimizer, compiled = model.optimizer, model.compiled
    compile_config = model.get_compile_config()
    optimizer_values = [value.numpy().copy() for value in optimizer.variables]
    path = tmp_path / "deployment.keras"
    save_deployment(model, path)

    assert model.optimizer is optimizer and model.compiled is compiled
    assert model.get_compile_config() == compile_config
    for variable, expected_value in zip(optimizer.variables, optimizer_values, strict=True):
        np.testing.assert_array_equal(variable.numpy(), expected_value)
    with zipfile.ZipFile(path) as archive:
        assert not json.loads(archive.read("config.json")).get("compile_config")
        with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as weights:
            assert "optimizer" not in weights
    restored = tf.keras.models.load_model(path)
    assert getattr(restored, "optimizer", None) is None
    actual = restored(images, training=False).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    result = check_reload_values(expected, actual, [f"heldout_{i}.jpg" for i in range(8)])
    assert result["status"] == "passed" and result["image_count"] == 8
    assert result["max_abs_difference"] <= 1e-5
    iteration = int(optimizer.iterations.numpy())
    model.train_on_batch(images, labels)
    assert int(optimizer.iterations.numpy()) == iteration + 1


@pytest.mark.parametrize("corruption", [
    "difference", "relative_tolerance", "shape", "expected_nan", "actual_nan",
    "expected_inf", "actual_inf",
])
def test_reload_values_reject_changed_nonfinite_or_reshaped_outputs(corruption):
    from src.fine_tuning import check_reload_values

    expected = np.linspace(0.1, 0.8, 8).reshape(8, 1)
    actual = expected.copy()
    filenames = [f"heldout_{i}.jpg" for i in range(8)]
    close = check_reload_values(expected, expected + 9e-6, filenames)
    assert close["status"] == "passed" and close["image_count"] == 8
    if corruption == "difference":
        actual[0, 0] += 2e-5
    elif corruption == "relative_tolerance":
        expected[:] = 100
        actual[:] = 100.0001
    elif corruption == "shape":
        actual = actual.reshape(8)
    elif corruption.startswith("expected"):
        expected[0, 0] = np.nan if corruption.endswith("nan") else np.inf
    else:
        actual[0, 0] = np.nan if corruption.endswith("nan") else np.inf
    with pytest.raises(ValueError):
        check_reload_values(expected, actual, filenames)


@pytest.fixture
def miniature_smoke_runs(tmp_path):
    from sklearn.linear_model import LogisticRegression

    architectures = json.loads((ROOT / "configs/fine_tuning.json").read_text())["architectures"]
    features = np.random.default_rng(1980).normal(size=(24, 3))
    labels = (features[:, 0] > 0).astype(int)
    rows = []
    for number, architecture in enumerate(architectures, 1):
        model = LogisticRegression(C=number, random_state=1980).fit(features[:16], labels[:16])
        scores = model.predict_proba(features[16:])[:, 1]
        for index, (score, decision) in enumerate(zip(scores, model.predict(features[16:]))):
            rows.append({
                "experiment": f"ft_d_{architecture.lower()}", "architecture": architecture,
                "configuration": "D", "family": "fine_tuning", "fold": 1, "seed": 1980,
                "sample_index": index, "filename": f"heldout_{index}.jpg",
                "true_label": int(labels[16 + index]), "score_clear": score,
                "predicted_label": int(decision),
            })
    predictions = pd.DataFrame(rows)
    report = {
        "completed": True, "smoke": True, "smoke_all": True,
        "identity": "matched-miniature-protocol",
        "protocol": "recorded_stratified_kfold", "folds": [1], "seeds": [1980],
        "config": {"architectures": architectures, "configurations": ["D"], "epochs": 1,
                   "initialization_seeds": [1980], "validation_fraction": 0.2},
        "requested_architectures": architectures,
        "architectures": {name: {"status": "passed"} for name in architectures},
        "runtime": {"platform": sys.platform, "python": sys.version, "device": "cpu"},
        "dataset_content_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
        "source_hashes": {"test_fine_tuning": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    current, reference = tmp_path / "current", tmp_path / "reference"
    for directory in (current, reference):
        directory.mkdir()
        for filename in ("smoke_report.json", "run_metadata.json"):
            (directory / filename).write_text(json.dumps(report))
        predictions.to_csv(directory / "per_image_predictions.csv", index=False)
    return current, reference, report, predictions


def test_compare_smoke_runs_accepts_identical_native_predictions(miniature_smoke_runs):
    from src.fine_tuning import compare_smoke_runs

    current, reference, report, _ = miniature_smoke_runs
    result = compare_smoke_runs(current, reference)
    assert result["status"] == "passed" and result["identical_predictions"] is True
    assert result["architecture_count"] == len(report["requested_architectures"])


@pytest.mark.parametrize("corruption", [
    "score", "decision", "filename", "fold", "seed", "missing_predictions",
    "missing_architecture", "identity", "config", "runtime",
    "dataset_content_sha256", "source_hashes", "incomplete",
])
def test_compare_smoke_runs_rejects_prediction_and_protocol_drift(miniature_smoke_runs, corruption):
    from src.fine_tuning import compare_smoke_runs

    current, reference, report, predictions = miniature_smoke_runs
    if corruption == "score":
        # Below deployment reload tolerance, but repeat runs require exact native scores.
        predictions.loc[0, "score_clear"] += 1e-8
    elif corruption == "decision":
        predictions.loc[0, "predicted_label"] = 1 - predictions.loc[0, "predicted_label"]
    elif corruption == "filename":
        predictions.loc[0, "filename"] = "different_image.jpg"
    elif corruption in {"fold", "seed"}:
        predictions.loc[0, corruption] += 1
    elif corruption == "missing_predictions":
        predictions = predictions[predictions.architecture != report["requested_architectures"][-1]]
    elif corruption == "missing_architecture":
        del report["architectures"][report["requested_architectures"][-1]]
    elif corruption == "incomplete":
        report["completed"] = False
    elif corruption in {"config", "runtime", "source_hashes"}:
        report[corruption] = {"changed": True}
    else:
        report[corruption] = "different"
    predictions.to_csv(current / "per_image_predictions.csv", index=False)
    for filename in ("smoke_report.json", "run_metadata.json"):
        (current / filename).write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compare_smoke_runs(current, reference)


def test_prediction_coverage_and_corruption_detection(archived):
    assignments, _, _, records, predictions = archived
    for _, group in predictions.groupby(["experiment", "seed"]):
        assert len(group) == group.filename.nunique() == 342
        assert set(group.fold) == set(range(1, 11))
    validate_results(records, predictions, assignments, [MODEL], SEEDS)
    corruptions = [predictions.iloc[1:].copy(), pd.concat([predictions, predictions.iloc[[0]]])]
    for column, value in [("fold", 2), ("filename", "foreign.jpg"), ("true_label", 0), ("seed", 999)]:
        changed = predictions.copy()
        changed.loc[0, column] = value
        corruptions.append(changed)
    for changed in corruptions:
        with pytest.raises(ValueError):
            validate_results(records, changed, assignments, [MODEL], SEEDS)


def test_metrics_match_real_predictions_and_detect_corruption(archived):
    assignments, _, _, records, predictions = archived
    for fold in [1, 5, 10]:
        rows = predictions.loc[(predictions.fold == fold) & (predictions.seed == SEEDS[0])]
        expected = independent_metrics(rows)
        actual = prediction_metrics(rows.true_label, rows.predicted_label, rows.score_clear)
        recorded = records.loc[(records.fold == fold) & (records.seed == SEEDS[0])].iloc[0]
        for name, value in expected.items():
            assert actual[name] == pytest.approx(value)
            assert recorded[name] == pytest.approx(value)
    # Missing an obstacle is class 0 predicted as 1, not the opposite error.
    safety = prediction_metrics([0, 0, 0, 1, 1], [0, 1, 1, 0, 1])
    assert safety["obstructed_false_negative_rate"] == pytest.approx(2 / 3)
    assert safety["fn_clear_as_obstructed"] == 1
    for name in ["accuracy", "mcc", "fp_obstructed_as_clear", "obstructed_false_negative_rate", "roc_auc"]:
        changed = records.copy()
        changed.loc[changed.index[0], name] += 0.1
        with pytest.raises(ValueError):
            validate_results(changed, predictions, assignments, [MODEL], SEEDS)


def test_real_cpu_imagenet_mobilenet_smoke(tmp_path):
    pytest.importorskip("tensorflow")
    subprocess.run(
        [sys.executable, "src/fine_tuning.py", "--config", "configs/fine_tuning.json",
         "--dataset", "via-dataset", "--output-dir", str(tmp_path), "--device", "cpu", "--smoke"],
        cwd=ROOT, check=True,
    )
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["completed"] is True and metadata["smoke"] is True
    assert metadata["folds"] == [1] and metadata["seeds"] == [1980]
    assert metadata["experiments"] == ["ft_d_mobilenet"]
    config_path = ROOT / "configs/fine_tuning.json"
    config = json.loads(config_path.read_text())
    assert Path(metadata["config_path"]) == config_path
    assert metadata["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert metadata["substitution_margin_pp"] == config["substitution_margin_pp"]
    assert metadata["substitution_metrics"] == ["accuracy", "obstructed_recall"]
    assert metadata["substitution_prespecified_utc"] == metadata["started_utc"]
    assert datetime.fromisoformat(metadata["substitution_prespecified_utc"]).utcoffset() is not None
    predictions = pd.read_csv(tmp_path / "per_image_predictions.csv")
    assert metadata["initialization"] == "imagenet"
    assert metadata["config"]["configuration_specs"]["D"]["optimizer"] == "adam"
    assert metadata["config"]["architectures"] == ["MobileNet"]
    assert metadata["config"]["configurations"] == ["D"]
    records = pd.read_csv(tmp_path / "per_fold_metrics.csv")
    costs = pd.read_csv(tmp_path / "computational_costs.csv")
    assignments = pd.read_csv(ASSIGNMENTS)
    expected_names = set(assignments.loc[assignments.test_fold == 1, "filename"])
    assert len(predictions) == predictions.filename.nunique() == 35
    assert set(predictions.filename) == expected_names
    assert set(predictions.fold) == {1}
    assert np.isfinite(predictions.score_clear).all()
    assert predictions.score_clear.between(0, 1).all()
    np.testing.assert_array_equal(predictions.predicted_label, (predictions.score_clear > 0.5).astype(int))
    for name, value in independent_metrics(predictions).items():
        assert records.iloc[0][name] == pytest.approx(value)
    validate_results(records, predictions, assignments, ["ft_d_mobilenet"], [1980], folds=[1])
    assert len(costs) == 1 and costs.iloc[0].effective_epochs == 1
    assert costs.iloc[0].trainable_parameters > 0
    assert np.isnan(costs.iloc[0].gpu_latency_median_ms)
    cost = costs.iloc[0]
    assert cost.peak_host_memory_bytes > 0
    assert cost.training_seconds > 0
    assert cost.adaptation_seconds == cost.training_seconds == records.iloc[0].training_seconds
    assert cost.update_timing_source == "prescribed_full_retraining"
    assert cost.update_labeled_size == records.iloc[0].train_size
    assert cost.update_previous_labeled_size == cost.update_labeled_size - 1
    outer_train = np.flatnonzero(assignments.test_fold.to_numpy() != 1)
    fit_train, validation = inner_split(
        outer_train, assignments.label.to_numpy(),
        metadata["config"]["validation_fraction"], metadata["config"]["validation_seed"],
    )
    assert cost.update_image_filename in set(assignments.iloc[fit_train].filename)
    assert cost.update_image_filename not in set(assignments.iloc[validation].filename) | expected_names
    for field in ("feature_forward_training_seconds", "feature_extraction_training_seconds",
                  "reduction_seconds", "classifier_training_seconds",
                  "adaptation_reduction_seconds", "adaptation_classifier_seconds"):
        assert np.isnan(cost[field])


def test_stratified_bootstrap_and_paired_statistical_boundaries():
    from scripts.analyze_fine_tuning import (
        bootstrap, mcnemar_exact, paired_wilcoxon, wilson_interval,
    )

    truth = np.asarray([0] * 167 + [1] * 175)
    perfect, all_clear = truth.copy(), np.ones(342, dtype=int)
    intervals, paired = bootstrap(
        truth, {"perfect": perfect, "all_clear": all_clear},
        pairs=[("perfect", "perfect"), ("perfect", "all_clear")],
        repetitions=10000, seed=1980,
    )
    identical = paired[paired.fixed_experiment == "perfect"]
    np.testing.assert_allclose(identical[["difference", "ci_low", "ci_high"]], 0)
    different = paired[paired.fixed_experiment == "all_clear"].set_index("endpoint")
    for endpoint, expected in [("accuracy", 167 / 342), ("mcc", 1), ("obstructed_recall", 1)]:
        np.testing.assert_allclose(different.loc[endpoint, ["difference", "ci_low", "ci_high"]].to_numpy(dtype=float), expected)
    assert intervals.set_index(["experiment", "endpoint"]).loc[("all_clear", "mcc"), "estimate"] == 0
    seed_means, seed_differences = bootstrap(
        truth, {"seeds": np.stack([perfect, all_clear]), "fixed": perfect},
        pairs=[("seeds", "fixed")], repetitions=10000, seed=1980,
    )
    mean_mcc = seed_means.set_index(["experiment", "endpoint"]).loc[("seeds", "mcc")]
    np.testing.assert_allclose(mean_mcc[["estimate", "ci_low", "ci_high"]].to_numpy(dtype=float), 0.5)
    difference_mcc = seed_differences.set_index("endpoint").loc["mcc"]
    np.testing.assert_allclose(difference_mcc[["difference", "ci_low", "ci_high"]].to_numpy(dtype=float), -0.5)
    assert mcnemar_exact(truth, perfect, all_clear) == pytest.approx((167, 0, 2.0 ** (1 - 167)))
    assert paired_wilcoxon(np.zeros(10)) == (0, 1)
    low, high = wilson_interval(0, 100)
    assert low == pytest.approx(0, abs=1e-15)
    assert high == pytest.approx(0.03699349820698568)
    assert wilson_interval(100, 100) == pytest.approx((1 - high, 1))


@pytest.mark.parametrize("drift", [
    "file_margin", "file_other_setting", "recorded_margin", "recorded_metrics", "missing_timestamp",
])
def test_substitution_rejects_config_and_prespecification_drift(tmp_path, drift):
    from scripts.analyze_fine_tuning import validate_substitution_config

    config = json.loads((ROOT / "configs/fine_tuning.json").read_text())
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    metadata = {
        "config_path": str(path),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "config": json.loads(path.read_text()),
        "substitution_margin_pp": config["substitution_margin_pp"],
        "substitution_metrics": config["substitution_metrics"],
        "started_utc": "2026-10-04T20:00:00+00:00",
        "substitution_prespecified_utc": "2026-10-04T20:00:00+00:00",
    }
    assert validate_substitution_config(metadata) == config["substitution_margin_pp"]
    assert validate_substitution_config(metadata, path) == config["substitution_margin_pp"]
    if drift == "file_margin":
        config["substitution_margin_pp"] += 1
        path.write_text(json.dumps(config))
    elif drift == "file_other_setting":
        config["batch_size"] += 1
        path.write_text(json.dumps(config))
    elif drift == "recorded_margin":
        metadata["substitution_margin_pp"] += 1
    elif drift == "recorded_metrics":
        metadata["substitution_metrics"] = ["accuracy"]
    else:
        del metadata["substitution_prespecified_utc"]
    with pytest.raises(ValueError):
        validate_substitution_config(metadata)


def test_bootstrap_reuses_joint_class_stratified_image_samples():
    from scripts.analyze_fine_tuning import bootstrap

    truth = np.array([0] * 8 + [1] * 8)
    fine, fixed = truth.copy(), truth.copy()
    fine[[0, 1, 8, 9, 10]] = 1 - fine[[0, 1, 8, 9, 10]]
    fixed[[1, 2, 3, 9, 11]] = 1 - fixed[[1, 2, 3, 9, 11]]
    repetitions, seed = 161, 1980
    _, paired = bootstrap(
        truth, {"fine": fine, "fixed": fixed}, pairs=[("fine", "fixed")],
        repetitions=repetitions, seed=seed,
    )
    rng = np.random.default_rng(seed)
    indices = np.concatenate([
        rng.choice(np.flatnonzero(truth == label), (repetitions, 8), replace=True)
        for label in (0, 1)
    ], axis=1)
    correctness_difference = (fine == truth).astype(int) - (fixed == truth).astype(int)
    for endpoint, selected in (("accuracy", indices), ("obstructed_recall", indices[:, :8])):
        expected = np.quantile(correctness_difference[selected].mean(axis=1), [0.025, 0.975])
        actual = paired.set_index("endpoint").loc[endpoint]
        np.testing.assert_allclose(actual[["ci_low", "ci_high"]].to_numpy(dtype=float), expected)


def test_identical_imperfect_predictions_have_zero_substitution_difference():
    from scripts.analyze_fine_tuning import bootstrap, substitution_analysis

    truth = np.array([0] * 8 + [1] * 8)
    predicted = truth.copy()
    predicted[[0, 2, 8, 11]] = 1 - predicted[[0, 2, 8, 11]]
    _, paired = bootstrap(
        truth, {"fine": np.stack([predicted, predicted]), "fixed": predicted},
        pairs=[("fine", "fixed")], repetitions=10000, seed=1980,
    )
    fine = pd.DataFrame({
        "experiment": "fine", "fold": np.arange(1, 11),
        "accuracy": 0.75, "obstructed_recall": 0.75,
    })
    fixed = fine.assign(experiment="fixed")
    result = substitution_analysis(paired, fine, fixed, 3.0)
    assert set(result.endpoint) == {"accuracy", "obstructed_recall", "obstructed_false_negative_rate"}
    np.testing.assert_allclose(result[
        ["difference", "ci_low", "ci_high", "fold_difference", "fold_ci_low", "fold_ci_high"]
    ], 0)
    assert (result.ci_low <= 0).all() and (result.ci_high >= 0).all()
    assert set(result.decision) == set(result.fold_decision) == {"supported"}


@pytest.mark.parametrize("low,high,decision", [
    (-0.20, -0.10, "supported"),
    (-0.40, -0.30, "not_supported"),
    (-0.25, -0.10, "inconclusive"),
    (-0.40, -0.25, "inconclusive"),
    (-0.25, -0.25, "inconclusive"),
    (-0.40, -0.10, "inconclusive"),
])
def test_substitution_sign_strict_margin_boundaries_and_fnr_reversal(low, high, decision):
    from scripts.analyze_fine_tuning import substitution_analysis

    difference = (low + high) / 2
    paired = pd.DataFrame([
        {"fine_experiment": "fine", "fixed_experiment": "fixed", "endpoint": endpoint,
         "difference": -difference, "ci_low": -high, "ci_high": -low}
        for endpoint in ("accuracy", "obstructed_recall")
    ])
    fine = pd.DataFrame({
        "experiment": "fine", "fold": np.arange(1, 11), "accuracy": 0.75, "obstructed_recall": 0.75,
    })
    # Exactly representable -25pp tests the zero-variance fold equality boundary.
    fixed = fine.assign(experiment="fixed", accuracy=0.5, obstructed_recall=0.5)
    result = substitution_analysis(paired, fine, fixed, 25.0).set_index("endpoint")
    for endpoint in ("accuracy", "obstructed_recall"):
        row = result.loc[endpoint]
        np.testing.assert_allclose([row.difference, row.ci_low, row.ci_high], [difference, low, high])
        assert row.decision == decision
        assert row.fold_decision == "inconclusive"
        np.testing.assert_allclose([row.fold_difference, row.fold_ci_low, row.fold_ci_high], -0.25)
    fnr = result.loc["obstructed_false_negative_rate"]
    np.testing.assert_allclose([fnr.difference, fnr.ci_low, fnr.ci_high], [-difference, -high, -low])
    assert fnr.decision == decision and fnr.fold_decision == "inconclusive"


@pytest.mark.parametrize("label_column", ["predicted_label", "predicted_label_default"])
def test_nested_predictions_preserve_canonical_identity_and_fold_metrics(archived, label_column):
    from scripts.analyze_fine_tuning import validate_nested_predictions

    assignments, _, _, _, predictions = archived
    nested = predictions.loc[predictions.seed == SEEDS[0], [
        "filename", "fold", "true_label", "score_clear", "predicted_label",
    ]].rename(columns={"predicted_label": label_column})
    nested = nested.assign(procedure="A", selected_configuration="selected_A")
    nested["predicted_label_safety"] = nested[label_column]
    normalized, records = validate_nested_predictions(nested, assignments)
    assert set(normalized.experiment) == {"nested_A_default", "nested_A_safety"}
    canonical = assignments.set_index("filename")
    for _, group in normalized.groupby("experiment"):
        actual = group.set_index("filename").reindex(canonical.index)
        np.testing.assert_array_equal(actual.sample_index, canonical.sample_index)
        np.testing.assert_array_equal(actual.fold, canonical.test_fold)
        np.testing.assert_array_equal(actual.true_label, canonical.label)
    for row in records.itertuples(index=False):
        group = normalized[(normalized.experiment == row.experiment) & (normalized.fold == row.fold)]
        for metric, expected in independent_metrics(group).items():
            assert getattr(row, metric) == pytest.approx(expected)


@pytest.mark.parametrize("corruption", [
    "missing_image", "duplicate_image", "foreign_image", "fold", "sample_index",
    "true_label", "selected_configuration", "nonfinite_score",
])
def test_nested_predictions_reject_image_fold_and_selection_corruption(archived, corruption):
    from scripts.analyze_fine_tuning import validate_nested_predictions

    assignments, _, _, _, predictions = archived
    nested = predictions.loc[predictions.seed == SEEDS[0], [
        "filename", "fold", "true_label", "score_clear", "predicted_label",
    ]].reset_index(drop=True).assign(procedure="A", selected_configuration="selected_A")
    if corruption == "missing_image":
        nested = nested.iloc[1:]
    elif corruption == "duplicate_image":
        nested = pd.concat([nested, nested.iloc[[0]]], ignore_index=True)
    elif corruption == "foreign_image":
        nested.loc[0, "filename"] = "not_a_canonical_image.jpg"
    elif corruption == "fold":
        nested.loc[0, "fold"] = nested.loc[0, "fold"] % 10 + 1
    elif corruption == "sample_index":
        nested["sample_index"] = nested.filename.map(assignments.set_index("filename").sample_index)
        nested.loc[0, "sample_index"] += 1
    elif corruption == "true_label":
        nested.loc[0, "true_label"] = 1 - nested.loc[0, "true_label"]
    elif corruption == "selected_configuration":
        nested.loc[0, "selected_configuration"] = "different_selection_in_same_fold"
    else:
        nested.loc[0, "score_clear"] = np.inf
    with pytest.raises(ValueError):
        validate_nested_predictions(nested, assignments)


def test_substitution_fold_sensitivity_pairs_by_fold_and_uses_t_interval():
    from scipy.stats import t
    from scripts.analyze_fine_tuning import substitution_analysis

    folds = np.arange(1, 11)
    fine = pd.DataFrame({
        "experiment": "fine", "fold": folds,
        "accuracy": np.linspace(0.6, 0.9, 10),
        "obstructed_recall": np.linspace(0.5, 0.8, 10),
    })
    delta = np.linspace(-0.08, 0.04, 10)
    fixed = fine.assign(
        experiment="fixed", accuracy=fine.accuracy + delta,
        obstructed_recall=fine.obstructed_recall + delta,
    ).iloc[::-1]
    paired = pd.DataFrame([
        {"fine_experiment": "fine", "fixed_experiment": "fixed", "endpoint": endpoint,
         "difference": 0.02, "ci_low": -0.02, "ci_high": 0.06}
        for endpoint in ("accuracy", "obstructed_recall")
    ])
    result = substitution_analysis(paired, fine, fixed, 3.0).set_index("endpoint")
    mean = delta.mean()
    radius = t.ppf(0.975, 9) * delta.std(ddof=1) / np.sqrt(10)
    for endpoint in ("accuracy", "obstructed_recall"):
        row = result.loc[endpoint]
        np.testing.assert_allclose(
            [row.fold_difference, row.fold_ci_low, row.fold_ci_high],
            [mean, mean - radius, mean + radius],
        )
        assert row.fold_decision == "inconclusive"
    fnr = result.loc["obstructed_false_negative_rate"]
    np.testing.assert_allclose(
        [fnr.fold_difference, fnr.fold_ci_low, fnr.fold_ci_high],
        [-mean, -mean - radius, -mean + radius],
    )
    assert fnr.fold_decision == "inconclusive"
