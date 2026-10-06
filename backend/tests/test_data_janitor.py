import sys
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules.data_janitor import clean_dataframe, detect_column_format
from modules.intelligence import SemanticType, auto_profile, generate_default_chart
from utils.dataframe_utils import read_csv_fast


class DataJanitorGeneralizationTests(unittest.TestCase):
    def setUp(self):
        self.no_ai = patch.dict("os.environ", {"OPENAI_API_KEY": ""})
        self.no_ai.start()

    def tearDown(self):
        self.no_ai.stop()

    def test_header_normalization_is_conservative(self):
        df = pd.DataFrame(
            {
                "  What is your annual salary?  ": [100, 200],
                "Highest level of education!": ["BS", "MS"],
            }
        )
        cleaned, actions, _, _, _ = clean_dataframe(df)
        self.assertIn("Normalized 2 column headers", actions)
        self.assertIn("what_is_your_annual_salary", cleaned.columns.tolist())
        self.assertIn("highest_level_of_education", cleaned.columns.tolist())

    def test_taxi_like_numeric_and_currency_detection(self):
        df = pd.DataFrame(
            {
                "trip_distance": ["1.2", "2.0", "3.5", None],
                "fare_amount": ["₹10.5", "₹20.0", "₹5.0", None],
                "total_amount": ["$12.0", "$22.0", "$6.5", None],
            }
        )
        _, actions, _, column_formats, _ = clean_dataframe(df)
        self.assertEqual(column_formats.get("trip_distance"), "number")
        self.assertEqual(column_formats.get("fare_amount"), "currency")
        self.assertEqual(column_formats.get("total_amount"), "currency")
        self.assertIn("Converted 'trip_distance' from numeric text to numeric", actions)
        self.assertNotIn("Converted 'trip_distance' from currency text to numeric", actions)

    def test_non_taxi_vendor_string_not_forced_to_coded_numeric(self):
        df = pd.DataFrame({"vendor": ["Acme", None, "Globex", "Acme"]})
        cleaned, actions, _, _, _ = clean_dataframe(df)
        # Categorical string remains categorical and not coerced to numeric sentinel.
        self.assertTrue(cleaned["vendor"].dtype == object)
        self.assertTrue(all("sentinel (-1)" not in action for action in actions))

    def test_missing_observations_are_preserved_by_default(self):
        df = pd.DataFrame({"sensor_value": [1.1, 2.4, 3.8, None, 5.2, 6.1, None, 8.0]})
        cleaned, actions, missing, _, _ = clean_dataframe(df)
        self.assertEqual(int(cleaned["sensor_value"].isna().sum()), 2)
        self.assertEqual(missing, {"sensor_value": 2})
        self.assertFalse(any("Filled" in action for action in actions))

    def test_uncertain_numeric_defaults_to_leave_null(self):
        # Integer-like low-density column without identifier hints should remain null.
        df = pd.DataFrame({"x_metric": [1, None, 2, None]})
        cleaned, actions, _, _, _ = clean_dataframe(df)
        self.assertTrue(any("missing" in action.lower() for action in actions))
        self.assertEqual(int(cleaned["x_metric"].isna().sum()), 2)

    def test_ambiguous_locale_numbers_remain_source_strings(self):
        df = pd.DataFrame({"amount": ["1,234", "2,345", "3,456"]})
        cleaned, actions, _, _, _ = clean_dataframe(df)
        self.assertEqual(cleaned["amount"].tolist(), ["1,234", "2,345", "3,456"])
        self.assertEqual(cleaned["amount"].dtype, object)
        self.assertTrue(any("ambiguous" in action.lower() for action in actions))

    def test_fractional_values_alone_do_not_imply_percentage_format(self):
        self.assertEqual(detect_column_format(pd.Series([0.1, 0.25, 0.5]), "conversion"), "number")

    def test_iso_dates_parse_but_ambiguous_date_order_stays_text(self):
        parsed, _, _, _, _ = clean_dataframe(pd.DataFrame({"order_date": ["2024-01-02", "2024-03-04"]}))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(parsed["order_date"]))

        ambiguous, actions, _, _, _ = clean_dataframe(
            pd.DataFrame({"order_date": ["01/02/2024", "02/03/2024"]})
        )
        self.assertEqual(ambiguous["order_date"].tolist(), ["01/02/2024", "02/03/2024"])
        self.assertTrue(any("ambiguous date order" in action.lower() for action in actions))

    def test_identifier_leading_zeroes_survive_csv_read_and_cleaning(self):
        raw = read_csv_fast(io.StringIO("postal_code\n00123\n00456\n"))
        cleaned, _, _, _, _ = clean_dataframe(raw)
        self.assertEqual(raw["postal_code"].tolist(), ["00123", "00456"])
        self.assertEqual(cleaned["postal_code"].tolist(), ["00123", "00456"])

    def test_header_collisions_are_globally_unique(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["A!", "A?", "a_2"])
        cleaned, _, _, _, _ = clean_dataframe(df)
        self.assertEqual(len(cleaned.columns), len(set(cleaned.columns)))
        self.assertEqual(cleaned.columns.tolist(), ["a", "a_3", "a_2"])

    def test_llm_metric_alias_cannot_make_text_a_metric_or_rename_schema_key(self):
        df = pd.DataFrame({"Revenue?": ["10", "bad", "30"]})
        llm_rows = [{"original": "Revenue?", "clean": "fake_metric", "format": None, "semantic_type": "metric"}]
        with patch("modules.data_janitor.llm_enrich_columns", return_value=llm_rows):
            cleaned, actions, _, _, semantic = clean_dataframe(df)
        self.assertEqual(cleaned.columns.tolist(), ["revenue"])
        self.assertEqual(cleaned["revenue"].tolist(), ["10", "bad", "30"])
        self.assertNotEqual(semantic.get("revenue"), "metric")
        self.assertTrue(any("parse" in action.lower() for action in actions))
        profile = auto_profile(cleaned, {"revenue": SemanticType.METRIC})
        self.assertEqual(profile["top_metrics"], [])

    def test_default_chart_uses_one_mean_measure_without_claiming_trend(self):
        df = pd.DataFrame({"year": [2000, 2005, 2010], "life_expectancy": [60.0, 62.0, 64.0]})
        types = {"year": SemanticType.TEMPORAL, "life_expectancy": SemanticType.METRIC}
        chart = generate_default_chart(df, types)
        self.assertEqual(chart["aggregation"], "mean")
        self.assertEqual(chart["y_axis_keys"], ["life_expectancy"])
        self.assertNotIn("trend", chart["analysis"].lower())

    def test_profile_keeps_true_sum_and_recommends_mean_for_nonadditive_metric(self):
        df = pd.DataFrame({"life_expectancy": [60.0, 70.0]})
        profile = auto_profile(
            df,
            {"life_expectancy": SemanticType.METRIC},
            {"life_expectancy": "number"},
        )
        metric = profile["top_metrics"][0]
        self.assertEqual(metric["total"], 130.0)
        self.assertEqual(metric["average"], 65.0)
        self.assertEqual(metric["aggregation"], "mean")

if __name__ == "__main__":
    unittest.main()
