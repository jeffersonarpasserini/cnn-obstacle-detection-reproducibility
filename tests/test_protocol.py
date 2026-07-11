import unittest

import numpy as np

try:
    import sklearn  # noqa: F401
except ImportError:
    SKLEARN_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = True
    from src.artifact import make_splits, prepare_fold_features, reduce_train_test


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

    def test_approach_d_uses_requested_total_components(self):
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
        self.assertEqual(x_train.shape, (20, 10))
        self.assertEqual(x_test.shape, (10, 10))


if __name__ == "__main__":
    unittest.main()
