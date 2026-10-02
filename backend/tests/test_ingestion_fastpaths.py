"""Regression coverage for the plain-number ingestion fast path."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import data_janitor
from modules.data_janitor import clean_dataframe


class IngestionFastPathTests(unittest.TestCase):
    def test_distinct_text_mapping_calls_mapper_once_per_lexeme(self):
        values = pd.Series(["a", "b", "a", "c", "b"])
        calls = []

        def uppercase(value):
            calls.append(value)
            return value.upper()

        mapped = data_janitor._map_distinct_text(values, uppercase)

        self.assertEqual(calls, ["a", "b", "c"])
        self.assertEqual(mapped.tolist(), ["A", "B", "A", "C", "B"])
        self.assertEqual(mapped.index.tolist(), values.index.tolist())

    def test_distinct_text_mapping_falls_back_after_cardinality_cap(self):
        values = pd.Series(["a", "b", "c", "a"])
        calls = []

        def uppercase(value):
            calls.append(value)
            return value.upper()

        with patch.object(data_janitor, "_MAX_DISTINCT_TEXT_VALUES", 2):
            mapped = data_janitor._map_distinct_text(values, uppercase)

        self.assertEqual(calls, values.tolist())
        self.assertEqual(mapped.tolist(), ["A", "B", "C", "A"])

    def test_plain_numbers_skip_unicode_currency_scans(self):
        values = pd.Series(["1", "-2.5", ".75", "+4."])

        with patch.object(data_janitor, "_contains_currency_symbol",
                          side_effect=AssertionError("plain numbers must skip symbol detection")), \
             patch.object(data_janitor, "_strip_currency_symbols",
                          side_effect=AssertionError("plain numbers must skip symbol stripping")):
            parsed, fmt, action = data_janitor._parse_numeric_text(values, "measurement")

        pd.testing.assert_series_equal(parsed, pd.Series([1.0, -2.5, 0.75, 4.0]))
        self.assertIsNone(fmt)
        self.assertEqual(action, "Converted 'measurement' from numeric text to numeric")

    def test_currency_symbol_checks_run_once_per_repeated_lexeme(self):
        values = pd.Series(["₹10.5"] * 100 + ["₹20.0"] * 50)
        with patch.object(data_janitor, "_contains_currency_symbol",
                          wraps=data_janitor._contains_currency_symbol) as detect, \
             patch.object(data_janitor, "_strip_currency_symbols",
                          wraps=data_janitor._strip_currency_symbols) as strip:
            parsed, fmt, _ = data_janitor._parse_numeric_text(values, "fare_amount")

        self.assertEqual(parsed.iloc[[0, 100]].tolist(), [10.5, 20.0])
        self.assertEqual(fmt, "currency")
        self.assertEqual(detect.call_count, 2)
        self.assertEqual(strip.call_count, 2)

    def test_numeric_like_threshold_uses_row_frequency(self):
        values = pd.Series(["12,34"] * 9 + ["foo", "bar"])

        parsed, fmt, action = data_janitor._parse_numeric_text(values, "measurement")

        self.assertIsNone(parsed)
        self.assertIsNone(fmt)
        self.assertEqual(
            action,
            "Could not safely parse 'measurement' numeric text; retained source values",
        )

    def test_high_cardinality_numeric_text_still_checks_and_parses_all_values(self):
        values = pd.Series([str(value) for value in range(1, 10_002)])

        parsed, fmt, action = data_janitor._parse_numeric_text(values, "measurement")

        self.assertEqual(len(parsed), len(values))
        self.assertEqual(parsed.iloc[[0, -1]].tolist(), [1, 10001])
        self.assertIsNone(fmt)
        self.assertEqual(action, "Converted 'measurement' from numeric text to numeric")

        values.iloc[-1] = "invalid"
        parsed, _, action = data_janitor._parse_numeric_text(values, "measurement")
        self.assertIsNone(parsed)
        self.assertIn("retained source values", action)

    def test_numeric_cleaning_keeps_nulls_and_source_index(self):
        source = pd.DataFrame(
            {"measurement": pd.Series(["1", None, "-2.5", ".75"], index=[10, 10, 30, 40])}
        )

        cleaned, actions, missing, _, _ = clean_dataframe(source)

        self.assertEqual(cleaned.index.tolist(), [10, 10, 30, 40])
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["measurement"]))
        self.assertEqual(cleaned["measurement"].iloc[[0, 2, 3]].tolist(), [1.0, -2.5, 0.75])
        self.assertTrue(pd.isna(cleaned["measurement"].iloc[1]))
        self.assertEqual(missing, {"measurement": 1})
        self.assertIn("Converted 'measurement' from numeric text to numeric", actions)

    def test_integer_reinsertion_preserves_exact_signed_and_unsigned_values(self):
        cases = [
            (["9007199254740992", None, "9007199254740993"], "Int64",
             [9007199254740992, 9007199254740993]),
            (["18446744073709551614", None, "18446744073709551615"], "UInt64",
             [18446744073709551614, 18446744073709551615]),
        ]
        for values, expected_dtype, expected_integers in cases:
            with self.subTest(dtype=expected_dtype), patch.dict(
                "os.environ", {"COLUMN_INTERPRETER": "off", "OPENAI_API_KEY": ""}
            ):
                source = pd.DataFrame({"measurement": values})
                cleaned, _, missing, _, _ = clean_dataframe(source)

            result = cleaned["measurement"]
            self.assertEqual(str(result.dtype), expected_dtype)
            self.assertEqual(result.iloc[0], expected_integers[0])
            self.assertTrue(pd.isna(result.iloc[1]))
            self.assertEqual(result.iloc[2], expected_integers[1])
            self.assertEqual(missing, {"measurement": 1})

    def test_compact_date_lexemes_in_identifier_columns_remain_text(self):
        source = pd.DataFrame({"customer_id": ["20240101", "20240102"]})

        cleaned, actions, _, formats, _ = clean_dataframe(
            source, interpret_columns=False
        )

        self.assertEqual(cleaned["customer_id"].tolist(), ["20240101", "20240102"])
        self.assertTrue(pd.api.types.is_object_dtype(cleaned["customer_id"]))
        self.assertEqual(formats["customer_id"], "identifier")
        self.assertNotIn("Parsed 'customer_id' as date", actions)

    def test_unrepresentable_numeric_lexemes_are_retained_without_raising(self):
        cases = [
            ["18446744073709551616", "18446744073709551617"],
            ["9" * 400 + ".1", "9" * 400 + ".2"],
        ]

        for values in cases:
            with self.subTest(lexeme_length=len(values[0])):
                source = pd.DataFrame({"amount": values})
                cleaned, actions, _, _, _ = clean_dataframe(
                    source, interpret_columns=False
                )

                self.assertEqual(cleaned["amount"].tolist(), values)
                self.assertTrue(pd.api.types.is_object_dtype(cleaned["amount"]))
                self.assertEqual(
                    actions[-1],
                    "Could not safely parse 'amount' as representable numbers; retained source values",
                )

    def test_mixed_currency_symbols_across_rows_are_preserved_for_review(self):
        values = ["$10", "€20"]
        source = pd.DataFrame({"amount": values})

        cleaned, actions, _, _, _ = clean_dataframe(source, interpret_columns=False)

        self.assertEqual(cleaned["amount"].tolist(), values)
        self.assertTrue(pd.api.types.is_object_dtype(cleaned["amount"]))
        self.assertIn("multiple currency symbols", actions[-1])
        self.assertIn("review", actions[-1])

    def test_multiple_currency_symbols_in_one_lexeme_are_preserved_for_review(self):
        values = ["$€10", "$€10"]
        source = pd.DataFrame({"amount": values})

        cleaned, actions, _, _, _ = clean_dataframe(source, interpret_columns=False)

        self.assertEqual(cleaned["amount"].tolist(), values)
        self.assertTrue(pd.api.types.is_object_dtype(cleaned["amount"]))
        self.assertIn("multiple currency symbols", actions[-1])
        self.assertIn("review", actions[-1])

    def test_interpret_columns_false_skips_provider_and_runs_deterministic_parse(self):
        source = pd.DataFrame({"Measurement?": ["10", "20"]})
        with patch.dict(
            "os.environ", {"COLUMN_INTERPRETER": "jev", "OPENAI_API_KEY": "configured"}
        ), patch.object(
            data_janitor, "llm_enrich_columns",
            side_effect=AssertionError("interpretation must be skipped"),
        ):
            cleaned, actions, _, _, semantic_types = clean_dataframe(
                source, interpret_columns=False
            )

        self.assertEqual(cleaned.columns.tolist(), ["measurement"])
        self.assertEqual(cleaned["measurement"].tolist(), [10, 20])
        self.assertIn("Converted 'measurement' from numeric text to numeric", actions)
        self.assertEqual(semantic_types, {})

    def test_currency_text_still_uses_currency_symbol_parsing(self):
        values = pd.Series(["₹10.5", "₹20.0"])
        with patch.object(data_janitor, "_contains_currency_symbol", wraps=data_janitor._contains_currency_symbol) as detect, \
             patch.object(data_janitor, "_strip_currency_symbols", wraps=data_janitor._strip_currency_symbols) as strip:
            parsed, fmt, action = data_janitor._parse_numeric_text(values, "fare_amount")

        self.assertEqual(parsed.tolist(), [10.5, 20.0])
        self.assertEqual(fmt, "currency")
        self.assertIn("currency text", action)
        self.assertGreater(detect.call_count, 0)
        self.assertGreater(strip.call_count, 0)


if __name__ == "__main__":
    unittest.main()
