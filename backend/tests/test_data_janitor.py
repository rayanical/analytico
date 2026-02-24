import sys
import unittest
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules.data_janitor import clean_dataframe


class DataJanitorGeneralizationTests(unittest.TestCase):
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

    def test_iot_continuous_metric_prefers_statistical_imputation_when_confident(self):
        # 75% density, continuous values -> high-confidence metric path.
        df = pd.DataFrame({"sensor_value": [1.1, 2.4, 3.8, None, 5.2, 6.1, None, 8.0]})
        cleaned, actions, _, _, _ = clean_dataframe(df)
        self.assertTrue(any("high-confidence metric" in action for action in actions))
        self.assertEqual(int(cleaned["sensor_value"].isna().sum()), 0)

    def test_uncertain_numeric_defaults_to_leave_null(self):
        # Integer-like low-density column without identifier hints should remain null.
        df = pd.DataFrame({"x_metric": [1, None, 2, None]})
        cleaned, actions, _, _, _ = clean_dataframe(df)
        self.assertTrue(any("low-confidence imputation" in action for action in actions))
        self.assertEqual(int(cleaned["x_metric"].isna().sum()), 2)


if __name__ == "__main__":
    unittest.main()
