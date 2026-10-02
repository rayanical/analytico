"""Contract checks for the independent interpretation holdout set."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from evaluate_interpretation import project_candidate_input, validate_cases


HOLDOUT_PATH = Path(__file__).with_name("interpretation_holdout.json")
STANDARD_CASES_PATH = Path(__file__).with_name("interpretation_cases.json")


class InterpretationHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = validate_cases(json.loads(HOLDOUT_PATH.read_text(encoding="utf-8")))
        cls.standard_cases = validate_cases(json.loads(STANDARD_CASES_PATH.read_text(encoding="utf-8")))

    def test_holdout_has_requested_size_and_distinct_case_ids(self):
        self.assertGreaterEqual(len(self.cases), 25)
        self.assertLessEqual(len(self.cases), 35)
        holdout_ids = [case["id"] for case in self.cases]
        standard_ids = {case["id"] for case in self.standard_cases}
        self.assertEqual(len(holdout_ids), len(set(holdout_ids)))
        self.assertTrue(set(holdout_ids).isdisjoint(standard_ids))

    def test_holdout_covers_requested_edge_case_families(self):
        categories = {case["category"] for case in self.cases}
        self.assertTrue(
            {
                "currency",
                "identifier",
                "prompt_injection",
                "unit_uncertainty",
                "missingness",
                "rate",
                "stock",
                "date",
                "numeric_category_code",
            }.issubset(categories)
        )

    def test_candidate_projection_keeps_gold_and_fixture_labels_out_of_model_input(self):
        for case in self.cases:
            with self.subTest(case_id=case["id"]):
                projected = project_candidate_input(case)
                self.assertEqual(set(projected), {"case_id", "input"})
                self.assertEqual(projected["case_id"], case["id"])
                self.assertEqual(set(projected["input"]), {"column_name", "values", "context"})
                self.assertNotIn("id", projected["input"])
                self.assertNotIn("category", projected["input"])
                self.assertNotIn("expected", projected["input"])


if __name__ == "__main__":
    unittest.main()
