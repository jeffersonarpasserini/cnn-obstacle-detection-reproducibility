import unittest

import numpy as np

try:
    import sklearn  # noqa: F401
except ImportError:
    SKLEARN_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = True
    import pandas as pd
    from unittest.mock import patch

    from src.artifact import (
        evaluate_experiment_group,
        make_splits,
        prepare_fold_features,
        reduce_train_test,
    )
    from src.run_corrected_experiments import (
        clean_incomplete_groups,
        group_experiments,
    )


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
class ProtocolTests(unittest.TestCase):
    def test_stratified_folds_contain_both_classes(self):
        labels = np.asarray([0] * 20 + [1] * 20)
        splits = make_splits(labels, "stratified_kfold", 10, 1980)
        self.assertEqual(len(splits), 10)
        for _, test in splits:
            self.assertEqual(set(labels[test]), {0, 1})

    def test_pca_training_projection_does_not_depend_on_test_values(self):
        rng = np.random.default_rng(1980)
        train = rng.normal(size=(20, 6))
        test_a = rng.normal(size=(4, 6))
        test_b = test_a + 10000
        train_a, _ = reduce_train_test(train, np.arange(20) % 2, test_a, "pca", 3, 1980)
        train_b, _ = reduce_train_test(train, np.arange(20) % 2, test_b, "pca", 3, 1980)
        np.testing.assert_allclose(train_a, train_b)

    def test_relieff_uses_skrebate_api_and_requested_components(self):
        try:
            import skrebate  # noqa: F401
        except ImportError:
            self.skipTest("skrebate is not installed")

        rng = np.random.default_rng(1980)
        train = rng.normal(size=(200, 8))
        labels = np.asarray([0, 1] * 100)
        test = rng.normal(size=(20, 8))
        reduced_train, reduced_test = reduce_train_test(
            train, labels, test, "relieff", 3, 1980
        )
        self.assertEqual(reduced_train.shape, (200, 3))
        self.assertEqual(reduced_test.shape, (20, 3))

    def test_approach_d_uses_requested_components_per_extractor(self):
        rng = np.random.default_rng(1980)
        labels = np.asarray([0, 1] * 15)
        feature_map = {
            "A": rng.normal(size=(30, 20)),
            "B": rng.normal(size=(30, 20)),
        }
        experiment = {
            "approach": "D", "extractors": ["A", "B"],
            "reduction": "pca", "components": 10, "scale": False,
        }
        train = np.arange(20)
        test = np.arange(20, 30)
        x_train, x_test = prepare_fold_features(
            experiment, feature_map, labels, train, test, 1980
        )
        # Approach D retains 10 components from each of the two extractors,
        # then concatenates them into a 20-component classification vector.
        self.assertEqual(x_train.shape, (20, 20))
        self.assertEqual(x_test.shape, (10, 20))

    def test_group_reuses_one_preparation_per_fold(self):
        labels = np.asarray([0, 1] * 10)
        splits = [
            (np.arange(4, 20), np.arange(0, 4)),
            (np.arange(0, 16), np.arange(16, 20)),
        ]
        experiments = [
            {
                "name": "gaussian", "approach": "B", "extractors": ["A"],
                "reduction": "pca", "components": 2,
                "classifier": "gaussian_nb", "scale": False,
            },
            {
                "name": "logistic", "approach": "B", "extractors": ["A"],
                "reduction": "pca", "components": 2,
                "classifier": "logistic", "scale": False,
            },
        ]
        prepared_train = np.asarray([[index, index % 2] for index in range(16)])
        prepared_test = np.asarray([[index, index % 2] for index in range(4)])
        with patch(
            "src.artifact.prepare_fold_features",
            return_value=(prepared_train, prepared_test),
        ) as prepare:
            records, predictions = evaluate_experiment_group(
                experiments,
                {"A": np.zeros((20, 3))},
                labels,
                splits,
                1980,
                sample_names=[f"sample_{index}.jpg" for index in range(20)],
                prediction_mode="all",
            )
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(len(records), 4)
        self.assertEqual(len(predictions), 16)
        self.assertTrue(all(row["shared_preprocessing_size"] == 2 for row in records))
        self.assertTrue(
            all(
                row["tn_obstructed"]
                + row["fp_obstructed_as_clear"]
                + row["fn_clear_as_obstructed"]
                + row["tp_clear"]
                == row["test_size"]
                for row in records
            )
        )
        self.assertTrue(all("score_clear" in row for row in predictions))

    def test_classifier_variants_form_one_preprocessing_group(self):
        common = {
            "approach": "B", "extractors": ["MobileNet"],
            "reduction": "pca", "components": 40, "scale": False,
        }
        experiments = [
            {**common, "name": "rbf", "classifier": "rbf_svm"},
            {**common, "name": "linear", "classifier": "linear_svm"},
        ]
        groups = group_experiments(experiments)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)

    def test_partial_group_is_removed_before_resume(self):
        common = {
            "approach": "B", "extractors": ["MobileNet"],
            "reduction": "pca", "components": 40, "scale": False,
        }
        group = [
            {**common, "name": "rbf", "classifier": "rbf_svm"},
            {**common, "name": "linear", "classifier": "linear_svm"},
        ]
        partial = pd.DataFrame(
            [
                {"experiment": "rbf", "fold": 1},
                {"experiment": "linear", "fold": 1},
            ]
        )
        cleaned, completed = clean_incomplete_groups(partial, [group], 2)
        self.assertTrue(cleaned.empty)
        self.assertEqual(completed, set())


if __name__ == "__main__":
    unittest.main()
