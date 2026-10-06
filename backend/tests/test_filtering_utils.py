import sys
import unittest
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import FilterConfig
from utils.filtering import apply_filters, resolve_effective_limit


class FilteringUtilsTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "name": ["Alpha", "Beta", "Gamma"],
                "value": [10, 20, 30],
                "date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-20"]),
            }
        )

    def test_apply_filters_operator_variants(self):
        filtered_eq, _ = apply_filters(self.df, [FilterConfig(column="name", operator="eq", value="Beta")])
        self.assertEqual(len(filtered_eq), 1)
        self.assertEqual(filtered_eq.iloc[0]["name"], "Beta")

        filtered_contains, _ = apply_filters(
            self.df,
            [FilterConfig(column="name", operator="contains", value="a")],
        )
        self.assertEqual(len(filtered_contains), 3)

        filtered_gt, _ = apply_filters(self.df, [FilterConfig(column="value", operator="gt", value=15)])
        self.assertEqual(len(filtered_gt), 2)

        filtered_date, _ = apply_filters(
            self.df,
            [FilterConfig(column="date", operator="gte", value="2024-01-10")],
        )
        self.assertEqual(len(filtered_date), 2)

    def test_apply_filters_legacy_values_min_max(self):
        filtered_values, _ = apply_filters(
            self.df,
            [FilterConfig(column="name", values=["Alpha", "Gamma"])],
        )
        self.assertEqual(len(filtered_values), 2)

        filtered_bounds, _ = apply_filters(
            self.df,
            [FilterConfig(column="value", min_val=15, max_val=25)],
        )
        self.assertEqual(len(filtered_bounds), 1)
        self.assertEqual(int(filtered_bounds.iloc[0]["value"]), 20)

    def test_resolve_effective_limit(self):
        capped_zero, warning_zero = resolve_effective_limit(0, default_limit=20)
        self.assertEqual(capped_zero, 500)
        self.assertIsNotNone(warning_zero)

        capped_large, warning_large = resolve_effective_limit(1000, default_limit=20)
        self.assertEqual(capped_large, 500)
        self.assertIn("capped", warning_large)

        normal, warning_normal = resolve_effective_limit(25, default_limit=20)
        self.assertEqual(normal, 25)
        self.assertIsNone(warning_normal)


if __name__ == "__main__":
    unittest.main()
