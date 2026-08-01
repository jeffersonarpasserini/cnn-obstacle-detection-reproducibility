import unittest

import numpy as np

from scripts.build_decision_attribution import (
    contribution_map,
    fit_linear_explanation,
    normalize_map,
)


class DecisionAttributionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1980)
        self.labels = np.asarray([0, 1] * 20)
        self.train = np.arange(39)
        self.test = np.asarray([39])
        self.first = rng.normal(size=(40, 12))
        self.second = rng.normal(size=(40, 8))

    def test_pca_linear_score_is_exactly_reconstructed(self):
        experiment = {
            "approach": "B",
            "extractors": ["first"],
            "reduction": "pca",
            "components": 5,
            "classifier": "linear_svm",
            "scale": False,
        }
        explanation = fit_linear_explanation(
            experiment,
            {"first": self.first},
            self.labels,
            self.train,
            self.test,
            1980,
        )
        self.assertAlmostEqual(
            explanation.direct_score, explanation.reconstructed_score, places=10
        )

    def test_separate_pca_score_is_exactly_reconstructed(self):
        experiment = {
            "approach": "D",
            "extractors": ["first", "second"],
            "reduction": "pca",
            "components": 4,
            "classifier": "linear_svm",
            "scale": False,
        }
        explanation = fit_linear_explanation(
            experiment,
            {"first": self.first, "second": self.second},
            self.labels,
            self.train,
            self.test,
            1980,
        )
        self.assertAlmostEqual(
            explanation.direct_score, explanation.reconstructed_score, places=10
        )

    def test_contribution_map_is_nonnegative_and_spatial(self):
        experiment = {
            "approach": "A",
            "extractors": ["MobileNet"],
            "reduction": "full",
            "components": None,
            "classifier": "logistic",
            "scale": False,
        }
        rng = np.random.default_rng(7)
        feature_map = {"MobileNet": rng.normal(size=(40, 7 * 7 * 1024))}
        explanation = fit_linear_explanation(
            experiment,
            feature_map,
            self.labels,
            self.train,
            self.test,
            1980,
        )
        heatmap = contribution_map(
            experiment, explanation, feature_map, int(self.test[0])
        )
        self.assertEqual(heatmap.shape, (7, 7))
        self.assertTrue(np.all(heatmap >= 0))
        normalized = normalize_map(heatmap)
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
