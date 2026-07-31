import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_article_followup.py"
SPEC = importlib.util.spec_from_file_location("analyze_article_followup", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ArticleFollowupTests(unittest.TestCase):
    def test_correct_mask_accepts_csv_boolean_forms(self):
        frame = pd.DataFrame({"correct": ["True", "false", "1", "0"]})
        self.assertEqual(MODULE.correct_mask(frame).tolist(), [True, False, True, False])

    def test_correct_mask_rejects_unknown_values(self):
        frame = pd.DataFrame({"correct": ["yes"]})
        with self.assertRaises(ValueError):
            MODULE.correct_mask(frame)


if __name__ == "__main__":
    unittest.main()
