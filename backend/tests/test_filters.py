import os
import sys
import unittest
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENAI_API_KEY", "test-key")
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import apply_filters
from models import FilterConfig


class FilterBehaviorTests(unittest.TestCase):
    def test_numeric_equality_filter_handles_numeric_values(self):
        df = pd.DataFrame({"amount": [1, 2, 3]})
        filtered, applied = apply_filters(
            df,
            [FilterConfig(column="amount", operator="eq", value=2)],
        )

        self.assertEqual(filtered["amount"].tolist(), [2])
        self.assertIn("amount == 2", applied)

    def test_numeric_equality_filter_handles_string_number(self):
        df = pd.DataFrame({"amount": [1, 2, 3]})
        filtered, _ = apply_filters(
            df,
            [FilterConfig(column="amount", operator="eq", value="2.0")],
        )

        self.assertEqual(filtered["amount"].tolist(), [2])

    def test_datetime_equality_filter_matches_timestamp_value(self):
        df = pd.DataFrame(
            {"created_at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])}
        )
        filtered, _ = apply_filters(
            df,
            [FilterConfig(column="created_at", operator="eq", value="2024-01-02")],
        )

        self.assertEqual(filtered["created_at"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-02"])


if __name__ == "__main__":
    unittest.main()
