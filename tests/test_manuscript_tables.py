import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_manuscript_tables.py"
SPEC = importlib.util.spec_from_file_location("build_manuscript_tables", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ManuscriptTableTests(unittest.TestCase):
    def test_average_friedman_ranks_use_larger_rank_for_better_accuracy(self):
        records = pd.DataFrame(
            {
                "fold": [0, 0, 0, 1, 1, 1],
                "experiment": ["a", "b", "c", "a", "b", "c"],
                "accuracy": [0.9, 0.8, 0.8, 0.7, 0.9, 0.8],
            }
        )
        ranks = MODULE.average_friedman_ranks(records, ["a", "b", "c"])
        self.assertAlmostEqual(ranks["a"], 2.0)
        self.assertAlmostEqual(ranks["b"], 2.25)
        self.assertAlmostEqual(ranks["c"], 1.75)

    def test_holm_adjustment_is_monotone_in_ordered_p_values(self):
        p_values = np.array([0.04, 0.001, 0.02])
        adjusted = MODULE.holm_adjust(p_values)
        ordered = adjusted[np.argsort(p_values)]
        self.assertTrue(np.all(np.diff(ordered) >= 0))
        np.testing.assert_allclose(adjusted, [0.04, 0.003, 0.04])

    def test_tie_breaker_prefers_obstructed_recall_after_accuracy(self):
        table = pd.DataFrame(
            {
                "average_friedman_rank": [2.0, 2.0],
                "accuracy_median": [0.9, 0.9],
                "pooled_accuracy": [0.9, 0.9],
                "pooled_obstructed_recall": [0.8, 0.9],
                "experiment": ["first", "second"],
            }
        ).sort_values(
            [
                "average_friedman_rank",
                "accuracy_median",
                "pooled_accuracy",
                "pooled_obstructed_recall",
                "experiment",
            ],
            ascending=[False, False, False, False, True],
        )
        self.assertEqual(MODULE.choose_winner(table), "second")


if __name__ == "__main__":
    unittest.main()
