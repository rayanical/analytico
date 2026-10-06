"""Exact integer axes survive browser JSON and round-trip drilldown filters."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import app
from models import FilterConfig
from services.aggregation_service import records_to_json
from utils.filtering import apply_filters, FilterValidationError


class IntegerRoundTripTests(unittest.TestCase):
    def test_json_keeps_unsafe_integer_keys_as_exact_strings(self):
        values = [9007199254740992, 9007199254740993, -9007199254740993]
        records = json.loads(json.dumps(records_to_json(pd.DataFrame({"customer_id": values}))))
        self.assertEqual([row["customer_id"] for row in records], [str(v) for v in values])
        self.assertEqual(records_to_json(pd.DataFrame({"n": [2**53-1, True]})),
                         [{"n": 2**53-1}, {"n": True}])

    def test_exact_membership_does_not_merge_adjacent_large_ids(self):
        for dtype in ("int64", "Int64", "uint64"):
            with self.subTest(dtype=dtype):
                frame = pd.DataFrame({"customer_id": pd.Series([9007199254740992, 9007199254740993], dtype=dtype)})
                for operator in ("eq", "values"):
                    spec = {"column": "customer_id", "operator": "eq", "value": "9007199254740993"} if operator == "eq" else {
                        "column": "customer_id", "values": ["9007199254740993"]}
                    filtered, _ = apply_filters(frame, [FilterConfig(**spec)])
                    self.assertEqual(filtered.customer_id.tolist(), [9007199254740993])

    def test_fractional_integer_boundary_is_not_rounded(self):
        frame = pd.DataFrame({"value": [9007199254740992, 9007199254740993]})
        filtered, _ = apply_filters(frame, [FilterConfig(column="value", operator="gt", value="9007199254740992.5")])
        self.assertEqual(filtered.value.tolist(), [9007199254740993])

    def test_integer_filter_rejects_nonfinite_and_excessive_operands(self):
        frame = pd.DataFrame({"value": [1, 2]})
        for value in ("NaN", "Infinity", "1e999999", "1"*513, True):
            with self.subTest(value=value), self.assertRaises(FilterValidationError):
                apply_filters(frame, [FilterConfig(column="value", operator="eq", value=value)])

    def test_drilldown_preserves_and_matches_the_exact_integer_key(self):
        frame = pd.DataFrame({"customer_id": [9007199254740992, 9007199254740993], "units": [1, 2]})
        ds = SimpleNamespace(df=frame)
        with patch("routers.analytics.get_dataset", return_value=ds):
            response = TestClient(app).post("/drilldown", json={"dataset_id": "fixture",
                "filters": [{"column": "customer_id", "values": ["9007199254740993"]}]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["data"], [{"customer_id": "9007199254740993", "units": 2}])


if __name__ == "__main__":
    unittest.main()
