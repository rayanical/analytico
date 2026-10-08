"""A query plan must stay bound to the dataset snapshot sent to the provider."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import QueryPlan, QueryRequest
from services.query_service import run_query


class QueryDatasetVersionTests(unittest.TestCase):
    def test_replacement_during_planning_returns_conflict_before_aggregation(self):
        def make_dataset(version, values):
            return SimpleNamespace(
                id="fixture",
                cache_version=version,
                df=pd.DataFrame({"group": ["A", "B"], "value": values}),
                column_types={"group": "categorical", "value": "metric"},
                column_formats={},
                column_schema=[{"column": "value", "aggregation": "sum"}],
                summary="A small test dataset.",
            )

        original = make_dataset("version-1", [1, 2])
        replacement = make_dataset("version-2", [100, 200])
        registry = {"current": original}

        def load_current(_dataset_id):
            return registry["current"]

        def replace_during_planning(_context):
            # Simulates a same-ID upload completing while the provider is deciding.
            registry["current"] = replacement
            return QueryPlan(
                kind="chart",
                x_axis_key="group",
                y_axis_keys=["value"],
                aggregation="sum",
                chart_type="bar",
                title="Value by group",
                x_axis_label="group",
                y_axis_label="value",
                filters=[],
                clarification=None,
            )

        with (
            patch("services.query_service.get_dataset", side_effect=load_current),
            patch("services.query_service.plan_decisions", side_effect=replace_during_planning),
            patch("services.query_service.run_aggregate") as aggregate,
        ):
            with self.assertRaises(HTTPException) as raised:
                run_query(QueryRequest(dataset_id="fixture", user_prompt="sum value by group"))

        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("changed while the chart was being planned", raised.exception.detail)
        aggregate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
