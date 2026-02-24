import unittest
from pathlib import Path
import sys

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules.data_janitor import clean_dataframe


class DataJanitorFormatAndImputationTests(unittest.TestCase):
    def test_trip_distance_numeric_text_is_not_currency(self):
        df = pd.DataFrame(
            {
                "trip_distance": ["1.2", "2.0", "3.5", None],
                "fare_amount": ["$10.5", "$20.0", "$5.0", None],
                "total_amount": ["$12.0", "$22.0", "$6.5", None],
                "vendorid": [1, None, 2, None],
                "payment_type": [1, None, 2, None],
                "ratecodeid": [1, None, 2, None],
                "trip_type": [1, None, 2, None],
                "passenger_count": [1, None, 2, None],
                "congestion_surcharge": [2.5, None, 0.0, None],
                "metric_value": [10.0, None, 30.0, 50.0],
            }
        )

        cleaned, actions, missing_counts, column_formats, _ = clean_dataframe(df)

        self.assertEqual(column_formats.get("trip_distance"), "number")
        self.assertEqual(column_formats.get("fare_amount"), "currency")
        self.assertEqual(column_formats.get("total_amount"), "currency")

        self.assertIn("Converted 'trip_distance' from numeric text to numeric", actions)
        self.assertIn("Converted 'fare_amount' from currency text to numeric", actions)
        self.assertNotIn("Converted 'trip_distance' from currency text to numeric", actions)

        self.assertIn(
            "Filled 2 missing 'vendorid' values with sentinel (-1) for coded field",
            actions,
        )
        self.assertIn(
            "Filled 2 missing 'payment_type' values with sentinel (-1) for coded field",
            actions,
        )
        self.assertIn(
            "Filled 2 missing 'ratecodeid' values with sentinel (-1) for coded field",
            actions,
        )
        self.assertIn(
            "Filled 2 missing 'trip_type' values with sentinel (-1) for coded field",
            actions,
        )
        self.assertNotIn(
            "Filled 2 missing 'passenger_count' values with sentinel (-1) for coded field",
            actions,
        )
        self.assertNotIn(
            "Filled 2 missing 'congestion_surcharge' values with sentinel (-1) for coded field",
            actions,
        )
        self.assertIn(
            "Filled 2 missing 'passenger_count' values with mode (1) for count metric",
            actions,
        )
        self.assertIn(
            "Filled 2 missing 'congestion_surcharge' values with 0 for surcharge/fee metric",
            actions,
        )

        # True metric still mean-imputed.
        self.assertIn("Filled 1 missing 'metric_value' values with mean", actions)

        # Ensure sentinel applied in data.
        self.assertTrue((cleaned["vendorid"] == -1).sum() >= 2)
        self.assertTrue((cleaned["payment_type"] == -1).sum() >= 2)
        self.assertTrue((cleaned["ratecodeid"] == -1).sum() >= 2)
        self.assertTrue((cleaned["trip_type"] == -1).sum() >= 2)
        self.assertTrue((cleaned["passenger_count"] == -1).sum() == 0)
        self.assertTrue((cleaned["congestion_surcharge"] == -1).sum() == 0)

        # Core response structures remain present.
        self.assertIsInstance(column_formats, dict)
        self.assertIsInstance(missing_counts, dict)
        self.assertIsInstance(actions, list)


if __name__ == "__main__":
    unittest.main()
