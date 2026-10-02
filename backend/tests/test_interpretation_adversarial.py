"""Adversarial safety checks for validated column decisions in the real clean path."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules.column_interpretation import InterpretationDecision, InterpretationResult
from modules.data_janitor import clean_dataframe


def _decision(
    *,
    role="metric",
    unit="none",
    parsing_policy="parse_decimal",
    aggregation="sum",
    needs_clarification=False,
):
    return {
        "role": role,
        "unit": unit,
        "parsing_policy": parsing_policy,
        "recommended_aggregation": aggregation,
        "needs_clarification": needs_clarification,
    }


def _provider_result(value):
    """Build the same validated result object the provider adapter returns."""
    decision = InterpretationDecision.model_validate(value, strict=True)
    return InterpretationResult(
        decision=decision,
        status="uncertain" if decision.needs_clarification else "ok",
        provider="openai",
        model="gpt-6-luna",
        prompt_version="adversarial-test",
    )


class InterpretationAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {"COLUMN_INTERPRETER": "luna", "OPENAI_API_KEY": "mock-key"},
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()

    def _clean(self, frame, decisions, *, capture=None):
        original = frame.copy(deep=True)

        def respond(request):
            if capture is not None:
                capture[request["column_name"]] = request
            proposal = decisions[request["column_name"]]
            return _provider_result(proposal)

        with patch("modules.column_interpretation.interpret_column", side_effect=respond), patch(
            "builtins.print"
        ):
            cleaned, actions, missing, formats, semantic_types = clean_dataframe(frame)

        pd.testing.assert_frame_equal(frame, original)
        return cleaned, actions, missing, formats, semantic_types

    def _assert_source_preserved(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for index, (source_value, cleaned_value) in enumerate(zip(expected, actual)):
            if pd.isna(source_value):
                self.assertTrue(pd.isna(cleaned_value), f"row {index} changed a source null")
            else:
                self.assertIs(type(cleaned_value), type(source_value), f"row {index} changed value type")
                self.assertEqual(cleaned_value, source_value, f"row {index} changed a source value")

    def _assert_reviewed(self, cleaned, semantic_types, name, values):
        self._assert_source_preserved(cleaned[name].tolist(), values)
        self.assertEqual(semantic_types.get(name), "unknown")
        metadata = cleaned.attrs["column_interpretations"][name]
        self.assertEqual(metadata["runtime_status"], "clarification")

    def test_wrong_or_ambiguous_proposals_preserve_entire_columns(self):
        numeric = _decision()
        cases = [
            (
                "customer_id",
                [101, 102, 103],
                numeric,
                "an explicit identifier token cannot become a metric",
            ),
            (
                "Customer-ID",
                [101, 102, 103],
                numeric,
                "header normalization must retain the identifier signal",
            ),
            (
                "customer_i_d",
                [101, 102, 103],
                numeric,
                "split identifier tokens must not be hidden by punctuation",
            ),
            *[(name, [101, 102, 103], numeric, "concatenated entity ID cannot become a metric")
              for name in ("customerid", "accountid", "personid")],
            (
                "postal_code",
                ["02108", "10001", "94105"],
                numeric,
                "leading zero codes cannot become numeric measurements",
            ),
            (
                "value",
                ["001", "002", "003"],
                numeric,
                "leading zero lexemes stay intact without an identifier-like name",
            ),
            (
                "amount",
                ["$10.00", "$20.00"],
                _decision(unit="USD", parsing_policy="parse_currency_decimal"),
                "a currency symbol alone does not establish USD identity",
            ),
            (
                "discount_pct",
                ["10%", "20%"],
                _decision(unit="USD", parsing_policy="parse_currency_decimal"),
                "percentage evidence contradicts a USD proposal",
            ),
            (
                "discount_pct",
                ["10", "20"],
                _decision(unit="ratio", parsing_policy="parse_percent_to_ratio"),
                "a percent policy requires percent-marked values throughout",
            ),
            (
                "amount_usd",
                ["$10.00", "EUR 20.00"],
                _decision(unit="USD", parsing_policy="parse_currency_decimal"),
                "a late EUR value contradicts an otherwise USD column",
            ),
            (
                "amount_eur",
                ["EUR 10.00", "$20.00"],
                _decision(unit="EUR", parsing_policy="parse_currency_decimal"),
                "a late dollar symbol contradicts an otherwise EUR column",
            ),
            (
                "sales_usd",
                ["USD 10.00"] * 18 + ["EUR 20.00", "USD 30.00"],
                _decision(unit="USD", parsing_policy="parse_currency_decimal"),
                "the complete column must be checked past the model sample",
            ),
            (
                "conversion_pct",
                ["10%"] * 18 + ["0.15", "20%"],
                _decision(unit="ratio", parsing_policy="parse_percent_to_ratio"),
                "a decimal suffix contradicts the percentage sample",
            ),
            (
                "amount_pct",
                ["10", "20"],
                _decision(unit="USD", parsing_policy="parse_currency_decimal"),
                "a numeric percent-like header cannot make plain values USD",
            ),
            (
                "rate",
                ["$1.25", "$2.50"],
                _decision(unit="ratio", parsing_policy="parse_percent_to_ratio"),
                "currency symbols contradict a ratio interpretation",
            ),
            (
                "measurement",
                ["1,25", "2,50"],
                _decision(unit="ratio", parsing_policy="parse_decimal", aggregation="mean"),
                "comma decimal locale must not be guessed",
            ),
            (
                "measurement",
                ["1.234,56", "2,345.67"],
                _decision(unit="none", parsing_policy="parse_decimal"),
                "mixed numeric locales must not be normalized into one scale",
            ),
            (
                "measurement",
                ["10"] * 12 + ["not-a-number"],
                numeric,
                "one invalid suffix blocks parsing the entire numeric column",
            ),
            (
                "event_date",
                ["01/02/2024", "03/04/2024"],
                _decision(
                    role="temporal", unit="calendar_date",
                    parsing_policy="parse_unambiguous_date", aggregation="none",
                ),
                "ambiguous numeric dates cannot establish a locale",
            ),
            (
                "event_date",
                ["2024-01-01", "04/05/2024"],
                _decision(
                    role="temporal", unit="calendar_date",
                    parsing_policy="parse_unambiguous_date", aggregation="none",
                ),
                "a contradictory date-order suffix must remain visible",
            ),
            (
                "measurement",
                [None, None, None],
                numeric,
                "all-null columns cannot be promoted to metrics",
            ),
            (
                "measurement",
                [1.0, float("inf"), 2.0],
                _decision(parsing_policy="preserve_numeric_value"),
                "infinity cannot be treated as a finite numeric metric",
            ),
            (
                "measurement",
                [1.0, float("-inf"), 2.0],
                _decision(parsing_policy="preserve_numeric_value"),
                "negative infinity cannot be treated as a finite numeric metric",
            ),
            (
                "flag",
                [True, False, True],
                numeric,
                "booleans cannot be aggregated as a numeric measure",
            ),
            (
                "event_year",
                [2020.0, 2021.0, 2022.0],
                _decision(
                    role="temporal", unit="calendar_date",
                    parsing_policy="parse_unambiguous_date", aggregation="none",
                ),
                "native numeric years must not be silently rewritten as dates",
            ),
            (
                "event_date",
                ["2024-01-01", "2024-01-02"],
                _decision(
                    role="metric", unit="none", parsing_policy="parse_decimal", aggregation="sum"
                ),
                "date values cannot be accepted as a numeric metric proposal",
            ),
            (
                "measurement",
                ["1", "2"],
                _decision(unit="ratio", aggregation="ratio_of_sums"),
                "ratio_of_sums needs a safe numerator and denominator computation",
            ),
            (
                "measurement",
                ["USD 10.00", "USD 20.00"],
                _decision(unit="USD", parsing_policy="parse_currency_decimal", aggregation="last_by_entity"),
                "last_by_entity is unsupported without validated entity and time semantics",
            ),
        ]

        for name, values, proposal, reason in cases:
            with self.subTest(column=name, reason=reason):
                source = pd.DataFrame({name: values})
                cleaned, _, _, _, types = self._clean(source, {name: proposal})
                clean_name = "event_year" if name == "event_year" else name.lower().replace("-", "_")
                self._assert_reviewed(cleaned, types, clean_name, values)

    def test_context_exposes_generic_identifier_evidence_but_runtime_guard_ignores_it(self):
        source = pd.DataFrame(
            {
                "record_id": ["10001", "10002", "10003"],
                "sequence": [10001, 10002, 10003],
            }
        )
        captured = {}
        decisions = {
            "record_id": _decision(
                role="identifier", unit="none", parsing_policy="preserve_lexeme", aggregation="count"
            ),
            "sequence": _decision(),
        }
        cleaned, _, _, _, types = self._clean(source, decisions, capture=captured)

        self.assertIn("record_id", captured["sequence"]["context"]["other_column_names"])
        self._assert_source_preserved(source["sequence"].tolist(), cleaned["sequence"].tolist())
        self.assertEqual(types["sequence"], "unknown")

    def test_large_integer_with_missing_value_never_loses_precision(self):
        source_values = ["9007199254740993", None]
        source = pd.DataFrame({"measurement": pd.Series(source_values, dtype=object)})
        cleaned, _, _, _, types = self._clean(source, {"measurement": _decision()})

        self._assert_source_preserved(cleaned["measurement"].tolist(), source_values)
        self.assertEqual(types.get("measurement"), "unknown")

    def test_unsampled_suffix_currency_conflict_blocks_interpretation(self):
        values = ["USD 10.00"] * 18 + ["EUR 20.00", "USD 30.00"]
        source = pd.DataFrame({"sales_usd": values})
        captured = {}
        cleaned, _, _, _, types = self._clean(
            source,
            {"sales_usd": _decision(unit="USD", parsing_policy="parse_currency_decimal")},
            capture=captured,
        )

        self.assertNotIn("EUR 20.00", captured["sales_usd"]["values"])
        self._assert_reviewed(cleaned, types, "sales_usd", values)

    def test_native_numeric_policy_preserves_floats_and_nan_as_null(self):
        source = pd.DataFrame({"measurement": [0.1, float("nan"), 0.3]})
        cleaned, _, missing, _, types = self._clean(
            source, {"measurement": _decision(parsing_policy="preserve_numeric_value", aggregation="mean")}
        )

        self.assertEqual(cleaned["measurement"].iloc[0], 0.1)
        self.assertTrue(pd.isna(cleaned["measurement"].iloc[1]))
        self.assertEqual(cleaned["measurement"].iloc[2], 0.3)
        self.assertEqual(missing["measurement"], 1)
        self.assertEqual(types["measurement"], "metric")

    def test_unambiguous_iso_dates_are_accepted_and_converted_to_temporal(self):
        source = pd.DataFrame({"event_date": ["2024-01-01", "2024-01-02"]})
        proposal = _decision(
            role="temporal", unit="calendar_date",
            parsing_policy="parse_unambiguous_date", aggregation="none",
        )
        cleaned, _, _, formats, types = self._clean(source, {"event_date": proposal})

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["event_date"]))
        self.assertEqual(types.get("event_date"), "temporal")
        self.assertEqual(formats.get("event_date"), "date")
        self.assertEqual(cleaned["event_date"].dt.strftime("%Y-%m-%d").tolist(), [
            "2024-01-01", "2024-01-02"
        ])

    def test_valid_explicit_units_and_percentages_apply_without_changing_source(self):
        source = pd.DataFrame(
            {
                "sales_usd": ["$1,250.00", "$80.00", None],
                "conversion_rate": ["10%", "25%", None],
                "segment_code": ["001", "002", None],
            }
        )
        decisions = {
            "sales_usd": _decision(unit="USD", parsing_policy="parse_currency_decimal"),
            "conversion_rate": _decision(
                unit="ratio", parsing_policy="parse_percent_to_ratio", aggregation="mean"
            ),
            "segment_code": _decision(
                role="categorical", unit="none", parsing_policy="preserve_source", aggregation="count"
            ),
        }
        cleaned, _, missing, formats, types = self._clean(source, decisions)

        self.assertEqual(cleaned["sales_usd"].iloc[:2].tolist(), [1250.0, 80.0])
        self.assertTrue(pd.isna(cleaned["sales_usd"].iloc[2]))
        self.assertEqual(cleaned["conversion_rate"].iloc[:2].tolist(), [0.1, 0.25])
        self.assertEqual(cleaned["segment_code"].iloc[:2].tolist(), ["001", "002"])
        self.assertEqual(types["sales_usd"], "metric")
        self.assertEqual(types["conversion_rate"], "metric")
        self.assertEqual(types["segment_code"], "categorical")
        self.assertEqual(formats["sales_usd"], "currency")
        self.assertEqual(formats["conversion_rate"], "percentage")
        self.assertEqual(formats["segment_code"], "general")
        self.assertEqual(missing["sales_usd"], 1)
        self.assertTrue(pd.isna(cleaned["segment_code"].iloc[2]))

    def test_numeric_codes_and_explicit_generic_identifier_remain_unchanged(self):
        source = pd.DataFrame(
            {
                "segment_code": [1.0, 2.0, 1.0],
                "value": [9007199254740993, 9007199254740995, 9007199254740997],
            }
        )
        decisions = {
            "segment_code": _decision(
                role="categorical", unit="none", parsing_policy="preserve_source", aggregation="count"
            ),
            "value": _decision(
                role="identifier", unit="none", parsing_policy="preserve_lexeme", aggregation="count"
            ),
        }
        cleaned, _, _, _, types = self._clean(source, decisions)

        self.assertEqual(cleaned["segment_code"].tolist(), [1.0, 2.0, 1.0])
        self.assertEqual(cleaned["value"].tolist(), [9007199254740993, 9007199254740995, 9007199254740997])
        self.assertEqual(types["segment_code"], "categorical")
        self.assertEqual(types["value"], "identifier")

    def test_valid_datetime_and_large_integer_requires_review_for_browser_precision(self):
        large_values = ["9007199254740993", "9007199254740995"]
        source = pd.DataFrame(
            {
                "event_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "measurement": large_values,
            }
        )
        decisions = {
            "event_date": _decision(
                role="temporal", unit="calendar_date",
                parsing_policy="parse_unambiguous_date", aggregation="none",
            ),
            "measurement": _decision(),
        }
        cleaned, _, _, _, types = self._clean(source, decisions)

        self.assertEqual(types["event_date"], "temporal")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["event_date"]))
        # Exact Python integers still cannot be represented exactly by browser numbers.
        self.assertEqual(cleaned["measurement"].tolist(), large_values)
        self.assertEqual(types["measurement"], "unknown")


if __name__ == "__main__":
    unittest.main()
