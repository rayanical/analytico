"""Small contract and scoring checks for the interpretation benchmark."""

import json
import unittest
from pathlib import Path

from evaluate_interpretation import (
    BenchmarkError,
    project_candidate_input,
    score,
    validate_cases,
    validate_predictions,
)


CASES_PATH = Path(__file__).with_name("interpretation_cases.json")


class InterpretationBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = validate_cases(json.loads(CASES_PATH.read_text(encoding="utf-8")))

    def test_fixture_covers_required_interpretation_hazards(self):
        categories = {case["category"] for case in self.cases}
        self.assertTrue(
            {
                "identifier",
                "currency",
                "percentage",
                "balance",
                "date",
                "number",
                "unknown_unit",
                "missing",
                "unknown",
                "rate",
            }.issubset(categories)
        )
        self.assertGreaterEqual(len(self.cases), 12)

    def test_explicit_unknown_is_scored_as_a_decision_not_an_abstention(self):
        target = next(case for case in self.cases if case["id"] == "currency.usd_explicit")
        predictions = {
            target["id"]: {
                "decision": {"role": "metric", "unit": "unknown"},
                "observed": {},
            }
        }
        result = score([target], "sample", predictions)
        self.assertEqual(result["per_field"]["unit"]["scored"], 1)
        self.assertEqual(result["per_field"]["unit"]["correct"], 0)

    def test_auto_prediction_with_missing_actions_is_reported_wrong(self):
        target = next(case for case in self.cases if case["id"] == "currency.usd_explicit")
        predictions = {
            target["id"]: {"decision": {"needs_clarification": False}, "observed": {}}
        }
        result = score([target], "sample", predictions)
        self.assertEqual(result["wrong_auto_decisions"]["count"], 1)

    def test_unknown_case_id_is_rejected(self):
        with self.assertRaises(BenchmarkError):
            validate_predictions(
                {"model": "sample", "predictions": [{"case_id": "missing", "decision": {}}]},
                {case["id"] for case in self.cases},
            )

    def test_candidate_projection_keeps_labels_outside_model_input(self):
        case = next(case for case in self.cases if case["id"] == "number.mixed_locale_formats")
        envelope = project_candidate_input(case)
        self.assertEqual(set(envelope), {"case_id", "input"})
        self.assertEqual(set(envelope["input"]), {"column_name", "values", "context"})
        self.assertEqual(envelope["case_id"], case["id"])
        self.assertNotIn("category", envelope["input"])
        self.assertNotIn("expected", envelope["input"])
        self.assertNotIn("id", envelope["input"])

    def test_decision_values_require_contract_types(self):
        case_ids = {case["id"] for case in self.cases}
        for value in (1, [], {}):
            with self.subTest(value=value), self.assertRaises(BenchmarkError):
                validate_predictions(
                    {
                        "model": "sample",
                        "predictions": [
                            {"case_id": "id.leading_zero", "decision": {"needs_clarification": value}}
                        ],
                    },
                    case_ids,
                )


if __name__ == "__main__":
    unittest.main()
