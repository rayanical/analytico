"""Provider decisions remain subordinate to full-column parsing and capabilities."""

import os
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.data_janitor import clean_dataframe, _interpretation_input
from modules.intelligence import auto_profile, generate_default_chart
from services.ingestion_service import ingest_dataframe
from storage import DATASETS
from models import QueryRequest
from services.query_service import run_query


def interpreted_row(name, **overrides):
    decision = dict(role="metric", unit="none", parsing_policy="parse_decimal",
                    recommended_aggregation="sum", needs_clarification=False)
    decision.update(overrides)
    return dict(original=name, clean="provider_cannot_rename", format=None,
                semantic_type=decision["role"], interpretation=dict(status="ok",
                runtime_status="clarification", provider="jev", model="typesafe-ai/jev",
                decision=decision))


class InterpretationIngestionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"COLUMN_INTERPRETER": "off", "OPENAI_API_KEY": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        DATASETS.clear()

    def clean(self, frame, row):
        with patch("modules.data_janitor.llm_enrich_columns", return_value=[row]):
            return clean_dataframe(frame)

    def test_representative_sample_covers_tail_and_preserves_nulls(self):
        values = pd.Series([str(i) for i in range(200)] + [None, "EUR 17"])
        candidate = _interpretation_input("amount", values, ["amount", "region"])
        self.assertEqual(set(candidate), {"column_name", "values", "context"})
        self.assertEqual(candidate["values"][-1], "EUR 17")
        self.assertEqual(len(candidate["values"]), 12)
        self.assertEqual(candidate["context"]["missing_count"], 1)
        self.assertFalse(candidate["context"]["sample_is_complete"])

    def test_verified_currency_balance_overrides_account_false_positive(self):
        source = pd.DataFrame({"account_balance_usd": ["$1,240.50", "$950.00", None]})
        row = interpreted_row("account_balance_usd", unit="USD", parsing_policy="parse_currency_decimal")
        cleaned, _, missing, formats, types = self.clean(source, row)
        self.assertEqual(cleaned.account_balance_usd.iloc[:2].tolist(), [1240.5, 950.0])
        self.assertTrue(pd.isna(cleaned.account_balance_usd.iloc[2]))
        self.assertEqual(missing, {"account_balance_usd": 1})
        self.assertEqual(types["account_balance_usd"], "metric")
        self.assertEqual(formats["account_balance_usd"], "currency")
        self.assertEqual(source.account_balance_usd.iloc[0], "$1,240.50")

    def test_unestablished_or_mixed_currency_is_preserved_even_if_model_says_usd(self):
        for name, values in [("amount", ["$10", "$20"]), ("amount_usd", ["$10", "€20"])]:
            with self.subTest(name=name):
                row = interpreted_row(name, unit="USD", parsing_policy="parse_currency_decimal")
                cleaned, actions, _, formats, types = self.clean(pd.DataFrame({name: values}), row)
                self.assertEqual(cleaned[name].tolist(), values)
                self.assertEqual(types[name], "unknown")
                self.assertEqual(formats[name], "general")
                self.assertTrue(any("review" in action for action in actions))

    def test_model_cannot_turn_identifiers_into_measures(self):
        for name, values in [("customer_id", ["123", "456"]), ("field", ["00123", "00456"])]:
            with self.subTest(name=name):
                cleaned, _, _, _, types = self.clean(pd.DataFrame({name: values}), interpreted_row(name))
                self.assertEqual(cleaned[name].tolist(), values)
                self.assertEqual(types[name], "unknown")

    def test_late_invalid_value_blocks_sample_based_metric_claim(self):
        values = ["10"] * 100 + ["bad"]
        cleaned, _, _, _, types = self.clean(pd.DataFrame({"measurement": values}), interpreted_row("measurement"))
        self.assertEqual(cleaned.measurement.tolist(), values)
        self.assertEqual(types["measurement"], "unknown")
        self.assertEqual(cleaned.attrs["column_interpretations"]["measurement"]["runtime_status"], "clarification")

    def test_unsupported_aggregation_does_not_generate_automatic_summary(self):
        row = interpreted_row("rate", unit="ratio", parsing_policy="parse_percent_to_ratio",
                              recommended_aggregation="ratio_of_sums")
        cleaned, _, _, formats, types = self.clean(pd.DataFrame({"rate": ["10%", "50%"]}), row)
        self.assertEqual(cleaned.rate.tolist(), ["10%", "50%"])
        self.assertEqual(auto_profile(cleaned, types, formats)["top_metrics"], [])
        self.assertIsNone(generate_default_chart(cleaned, types))

    def test_profile_uses_accepted_mean_instead_of_header_heuristic(self):
        row = interpreted_row("measurement", recommended_aggregation="mean")
        cleaned, _, _, formats, types = self.clean(pd.DataFrame({"measurement": ["10", "20"]}), row)
        self.assertEqual(auto_profile(cleaned, types, formats)["top_metrics"][0]["aggregation"], "mean")

    def test_upload_exposes_provenance_and_preserves_raw_source(self):
        frame = pd.DataFrame({"Measurement?": ["1", "2", None]})
        row = interpreted_row("Measurement?")
        with patch("modules.data_janitor.llm_enrich_columns", return_value=[row]):
            response = ingest_dataframe(frame, "source.csv", "/upload")
        column = response.columns[0]
        self.assertEqual(column.name, "measurement")
        self.assertEqual(column.interpretation.runtime_status, "applied")
        self.assertEqual(column.interpretation.decision["recommended_aggregation"], "sum")
        pd.testing.assert_frame_equal(DATASETS[response.dataset_id].raw_df, frame)

    def test_date_sample_cannot_establish_locale_for_later_rows(self):
        values = ["01/01/2024"] * 100 + ["02/03/2024"]
        row = interpreted_row("event_date", role="temporal", unit="calendar_date",
                              parsing_policy="parse_unambiguous_date", recommended_aggregation="none")
        cleaned, _, _, _, types = self.clean(pd.DataFrame({"event_date": values}), row)
        self.assertEqual(cleaned.event_date.tolist(), values)
        self.assertEqual(types["event_date"], "unknown")

    def test_unrepresentable_metric_retains_source_without_failing_upload(self):
        values = ["9" * 400, "12"]
        cleaned, _, _, _, types = self.clean(pd.DataFrame({"measurement": values}), interpreted_row("measurement"))
        self.assertEqual(cleaned.measurement.tolist(), values)
        self.assertEqual(types["measurement"], "unknown")

    def test_model_cannot_assign_currency_to_unlabeled_plain_numbers(self):
        row = interpreted_row("measurement", unit="USD")
        cleaned, _, _, formats, types = self.clean(pd.DataFrame({"measurement": ["10", "20"]}), row)
        self.assertEqual(cleaned.measurement.tolist(), ["10", "20"])
        self.assertEqual(types["measurement"], "unknown")
        self.assertEqual(formats["measurement"], "general")

    def test_percent_scaling_avoids_intermediate_binary_rounding(self):
        values = ["0.07%", "0.7%", "1.15%", None]
        row = interpreted_row("share", unit="ratio", parsing_policy="parse_percent_to_ratio", recommended_aggregation="mean")
        cleaned, _, _, _, types = self.clean(pd.DataFrame({"share": values}), row)
        self.assertEqual(cleaned.share.iloc[:3].tolist(), [0.0007, 0.007, 0.0115])
        self.assertEqual(types["share"], "metric")
        self.assertTrue(pd.isna(cleaned.share.iloc[3]))

    def test_high_precision_currency_and_conflicting_native_unit_require_review(self):
        cases = [("amount_usd", ["$0.123456789012345678", "$0.2"],
                  interpreted_row("amount_usd", unit="USD", parsing_policy="parse_currency_decimal")),
                 ("amount_usd", [10.0, 20.0],
                  interpreted_row("amount_usd", unit="none", parsing_policy="preserve_numeric_value")),
                 ("measurement", [-9223372036854775808, -12],
                  interpreted_row("measurement", parsing_policy="preserve_numeric_value"))]
        for name, values, row in cases:
            with self.subTest(name=name, values=values):
                cleaned, _, _, _, types = self.clean(pd.DataFrame({name: values}), row)
                self.assertEqual(cleaned[name].tolist(), values)
                self.assertEqual(types[name], "unknown")

    def test_large_integer_totals_require_review_before_profiles_or_charts(self):
        # Each observation is exact, but the total cannot be represented exactly by JS.
        for values, policy in [(["4503599627370496", "4503599627370496"], "parse_decimal"),
                               ([4503599627370496, 4503599627370496], "preserve_numeric_value")]:
            with self.subTest(policy=policy):
                row = interpreted_row("measurement", parsing_policy=policy)
                cleaned, _, _, formats, types = self.clean(pd.DataFrame({"measurement": values}), row)
                self.assertEqual(cleaned.measurement.tolist(), values)
                self.assertEqual(types["measurement"], "unknown")
                self.assertEqual(auto_profile(cleaned, types, formats)["top_metrics"], [])

    def test_ai_plan_cannot_use_unreviewed_column(self):
        frame = pd.DataFrame({"group": ["A", "B"], "amount": ["$10", "$20"]})
        row = interpreted_row("amount", unit="unknown", needs_clarification=True)
        with patch("modules.data_janitor.llm_enrich_columns", return_value=[
            dict(original="group", semantic_type=None), row
        ]):
            upload = ingest_dataframe(frame, "source.csv", "/upload")
        plan = dict(kind="chart", x_axis_key="group", y_axis_keys=["amount"],
                    aggregation="sum", chart_type="bar", title="Amount",
                    x_axis_label="Group", y_axis_label="Amount", filters=[], clarification=None)
        message = SimpleNamespace(content=json.dumps(plan), tool_calls=None)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=message)]))))
        with patch("services.query_service.get_openai_client", return_value=client), patch(
            "services.query_service.run_aggregate", side_effect=AssertionError("must not execute")
        ):
            response = run_query(QueryRequest(dataset_id=upload.dataset_id, user_prompt="sum amount"))
        self.assertEqual(response.data, [])
        self.assertIn("interpretation review", response.analysis)

    def test_provider_failure_stops_calls_and_preserves_all_columns(self):
        frame = pd.DataFrame({"amount_usd": ["$10", "$20"], "value": ["1", "2"]})
        failure = SimpleNamespace(decision=None, status="unavailable", provider="openai",
                                  model="gpt-6-luna", prompt_version="test", latency_ms=8000,
                                  usage={}, confidence=None, error_code="request_timeout")
        with patch.dict(os.environ, {"COLUMN_INTERPRETER": "luna"}), patch(
            "modules.column_interpretation.interpret_column", return_value=failure
        ) as call:
            cleaned, _, _, _, types = clean_dataframe(frame)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(cleaned.amount_usd.tolist(), frame.amount_usd.tolist())
        self.assertEqual(cleaned.value.tolist(), frame.value.tolist())
        self.assertEqual(types, {"amount_usd": "unknown", "value": "unknown"})
        self.assertEqual(cleaned.attrs["column_interpretations"]["value"]["status"], "skipped")

    def test_invalid_provider_configuration_does_not_break_upload(self):
        with patch.dict(os.environ, {"COLUMN_INTERPRETER": "typo"}):
            response = ingest_dataframe(pd.DataFrame({"value": ["1", "2"]}), "source.csv", "/upload")
        self.assertEqual(response.columns[0].interpretation.status, "error")
        self.assertEqual(response.columns[0].interpretation.runtime_status, "clarification")

    def test_off_never_constructs_a_provider_with_existing_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "configured", "COLUMN_INTERPRETER": "off"}), patch(
            "core.config.get_openai_client", side_effect=AssertionError("must remain offline")
        ):
            _, _, _, _, types = clean_dataframe(pd.DataFrame({"value": ["1", "2"]}))
        self.assertEqual(types, {})


if __name__ == "__main__":
    unittest.main()
