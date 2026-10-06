"""Regression coverage for the analytics execution seam."""

import os
import sys
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import AggregateRequest, FilterConfig, QueryRequest
from main import app
from services.aggregation_service import run_aggregate
from services.query_service import run_query
from utils.filtering import apply_filters
from core.config import get_openai_client


def dataset(df: pd.DataFrame, column_types: dict[str, str] | None = None):
    return SimpleNamespace(
        id="fixture",
        df=df,
        column_types=column_types or {column: "metric" for column in df.columns},
        column_formats={},
    )


class AnalyticsServiceTests(unittest.TestCase):
    def test_count_top_n_aggregates_rows_once(self):
        ds = dataset(
            pd.DataFrame({"group": ["A"] * 3 + ["B"] * 2 + ["C"], "value": [10, 20, 30, 40, 50, 60]}),
            {"group": "categorical", "value": "metric"},
        )
        request = AggregateRequest(
            dataset_id="fixture",
            x_axis_key="group",
            y_axis_keys=["value"],
            aggregation="count",
            limit=2,
        )
        with patch("services.aggregation_service.get_dataset", return_value=ds):
            result = run_aggregate(request)

        self.assertEqual([row["value"] for row in result.data], [3, 3])
        self.assertEqual(sum(row["value"] for row in result.data), 6)

    def test_synthetic_others_label_never_collides_with_a_real_category(self):
        ds = dataset(
            pd.DataFrame({"group": ["Others"] * 10 + ["A", "B"], "value": [1] * 12}),
            {"group": "categorical", "value": "metric"},
        )
        request = AggregateRequest(
            dataset_id="fixture", x_axis_key="group", y_axis_keys=["value"], limit=2
        )
        with patch("services.aggregation_service.get_dataset", return_value=ds):
            result = run_aggregate(request)
        self.assertEqual(result.others_label, "Others (2)")
        self.assertIn("Others", [row["group"] for row in result.data])
        self.assertIn("Others (2)", [row["group"] for row in result.data])

    def test_date_count_resampling_preserves_raw_rows_until_aggregation(self):
        ds = dataset(
            pd.DataFrame({"date": pd.date_range("2024-01-01", periods=120), "value": [1] * 120}),
            {"date": "temporal", "value": "metric"},
        )
        request = AggregateRequest(
            dataset_id="fixture",
            x_axis_key="date",
            y_axis_keys=["value"],
            aggregation="count",
            limit=500,
        )
        with patch("services.aggregation_service.get_dataset", return_value=ds):
            result = run_aggregate(request)

        self.assertEqual(sum(row["value"] for row in result.data), 120)
        self.assertEqual(result.source_x_axis_key, "date")
        self.assertEqual(result.time_bucket, "week")
        self.assertEqual(result.x_axis_key, "date")

    def test_explicit_time_bucket_is_replayed_when_filtered_cardinality_is_small(self):
        ds = dataset(
            pd.DataFrame({"date": pd.to_datetime(["2024-01-15", "2024-12-15"]), "value": [1, 2]}),
            {"date": "temporal", "value": "metric"},
        )
        request = AggregateRequest(
            dataset_id="fixture",
            x_axis_key="date",
            y_axis_keys=["value"],
            aggregation="sum",
            time_bucket="year",
        )
        with patch("services.aggregation_service.get_dataset", return_value=ds):
            result = run_aggregate(request)
        self.assertEqual(result.time_bucket, "year")
        self.assertEqual(result.data, [{"date": "2024-01-01T00:00:00", "value": 3}])

    def test_mean_top_n_ranks_by_mean_and_does_not_double_aggregate(self):
        ds = dataset(
            pd.DataFrame({"group": ["A"] * 10 + ["B", "C"], "value": [1] * 10 + [9, 8]}),
            {"group": "categorical", "value": "metric"},
        )
        request = AggregateRequest(
            dataset_id="fixture",
            x_axis_key="group",
            y_axis_keys=["value"],
            aggregation="mean",
            limit=2,
            group_others=True,
        )
        with patch("services.aggregation_service.get_dataset", return_value=ds):
            result = run_aggregate(request)

        self.assertEqual(result.data[0], {"group": "B", "value": 9.0})
        self.assertEqual(result.data[1]["group"], "Others")
        self.assertAlmostEqual(result.data[1]["value"], 18 / 11)

    def test_null_groups_and_all_null_sums_are_preserved_as_null(self):
        ds = dataset(
            pd.DataFrame({"group": ["A", None, "B"], "value": [None, 5.0, 2.0]}),
            {"group": "categorical", "value": "metric"},
        )
        request = AggregateRequest(dataset_id="fixture", x_axis_key="group", y_axis_keys=["value"])
        with patch("services.aggregation_service.get_dataset", return_value=ds):
            result = run_aggregate(request)

        self.assertEqual(len(result.data), 3)
        self.assertIn({"group": "A", "value": None}, result.data)
        self.assertIn({"group": None, "value": 5.0}, result.data)

    def test_numeric_equality_compares_numeric_values(self):
        filtered, _ = apply_filters(
            pd.DataFrame({"value": [1.0, 2.0]}),
            [FilterConfig(column="value", operator="eq", value=1)],
        )
        self.assertEqual(filtered["value"].tolist(), [1.0])

    def test_value_membership_can_select_null_alone_or_with_other_values(self):
        source = pd.DataFrame({"group": ["A", None, "B"]})
        only_null, _ = apply_filters(source, [FilterConfig(column="group", values=[None])])
        mixed, _ = apply_filters(source, [FilterConfig(column="group", values=["A", None])])
        self.assertEqual(len(only_null), 1)
        self.assertTrue(only_null.iloc[0]["group"] is None or pd.isna(only_null.iloc[0]["group"]))
        self.assertEqual(len(mixed), 2)

    def test_timezone_aware_filter_boundaries_match_naive_source_dates(self):
        source = pd.DataFrame({"date": pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02"])})
        filtered, _ = apply_filters(
            source,
            [FilterConfig(column="date", operator="gte", value="2024-02-01T00:00:00Z")],
        )
        self.assertEqual(filtered["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-02-01", "2024-02-02"])

    def test_unknown_columns_and_invalid_filter_values_are_rejected(self):
        df = pd.DataFrame({"value": [1, 2]})
        with self.assertRaises(ValueError):
            apply_filters(df, [FilterConfig(column="missing", operator="eq", value=1)])
        with self.assertRaises(ValueError):
            apply_filters(df, [FilterConfig(column="value", operator="gt", value="not-a-number")])
        with self.assertRaises(ValidationError):
            FilterConfig(column="value", operator="contains", value={"nested": True})

    def test_request_enums_and_limits_are_validated(self):
        base = {"dataset_id": "fixture", "x_axis_key": "group", "y_axis_keys": ["value"]}
        with self.assertRaises(ValidationError):
            AggregateRequest(**{**base, "aggregation": "median-ish"})
        with self.assertRaises(ValidationError):
            AggregateRequest(**{**base, "sort_by": "random"})
        with self.assertRaises(ValidationError):
            AggregateRequest(**{**base, "limit": -1})
        with self.assertRaises(ValidationError):
            AggregateRequest(**{**base, "y_axis_keys": []})

    def test_ai_client_is_optional_until_an_ai_feature_is_called(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch("core.config._openai_client", None), patch(
            "core.config._openai_client_key", None
        ):
            with self.assertRaises(HTTPException) as context:
                get_openai_client()
        self.assertEqual(context.exception.status_code, 503)

    def test_api_rejects_negative_drilldown_limit(self):
        response = TestClient(app).post("/drilldown", json={"dataset_id": "fixture", "limit": -1})
        self.assertEqual(response.status_code, 422)

    def test_explicit_null_sort_by_uses_a_supported_default(self):
        ds = dataset(
            pd.DataFrame({"group": ["A", "B"], "value": [1, 10]}),
            {"group": "categorical", "value": "metric"},
        )
        request = AggregateRequest(
            dataset_id="fixture", x_axis_key="group", y_axis_keys=["value"], sort_by=None
        )
        with patch("services.aggregation_service.get_dataset", return_value=ds):
            result = run_aggregate(request)
        self.assertEqual(result.data[0]["group"], "B")

    def test_numeric_axis_label_sort_stays_numeric(self):
        ds = dataset(
            pd.DataFrame({"period": [1, 10, 2], "value": [1, 1, 1]}),
            {"period": "metric", "value": "metric"},
        )
        request = AggregateRequest(
            dataset_id="fixture", x_axis_key="period", y_axis_keys=["value"], sort_by="label"
        )
        with patch("services.aggregation_service.get_dataset", return_value=ds):
            result = run_aggregate(request)
        self.assertEqual([row["period"] for row in result.data], [1, 2, 10])

    def test_unexpected_provider_tool_call_is_never_executed(self):
        fake_message = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments='{"code":"raise RuntimeError()"}'))],
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=fake_message)]))
            )
        )
        ds = dataset(pd.DataFrame({"group": ["A"], "value": [1]}), {"group": "categorical", "value": "metric"})
        request = QueryRequest(dataset_id="fixture", user_prompt="show total by group")

        with patch("services.query_service.get_dataset", return_value=ds), patch(
            "services.query_service.get_openai_client", return_value=fake_client
        ):
            result = run_query(request)

        self.assertEqual(result.chart_type, "empty")
        self.assertIn("unsupported action", (result.analysis or "").lower())

    def test_query_plan_filters_are_ai_only_and_computed_result_drives_chart(self):
        plan = {
            "kind": "chart",
            "x_axis_key": "group",
            "y_axis_keys": ["value"],
            "aggregation": "sum",
            "chart_type": "bar",
            "title": "Value by group",
            "x_axis_label": "Group",
            "y_axis_label": "Total",
            "filters": [{"column": "group", "operator": "eq", "value": "B"}],
            "clarification": None,
        }
        fake_message = SimpleNamespace(content=json.dumps(plan), tool_calls=None)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=fake_message)]))
            )
        )
        ds = dataset(
            pd.DataFrame({"group": ["A", "B"], "value": [1, 2]}),
            {"group": "categorical", "value": "metric"},
        )
        request = QueryRequest(
            dataset_id="fixture",
            user_prompt="show B",
            filters=[FilterConfig(column="value", operator="gte", value=1)],
        )

        with patch("services.query_service.get_dataset", return_value=ds), patch(
            "services.aggregation_service.get_dataset", return_value=ds
        ), patch("services.query_service.get_openai_client", return_value=fake_client):
            result = run_query(request)

        self.assertEqual(result.data, [{"group": "B", "value": 2}])
        self.assertEqual(result.title, "Value by group")
        self.assertEqual(result.filters, [
            FilterConfig(column="value", operator="gte", value=1),
            FilterConfig(column="group", operator="eq", value="B"),
        ])
        self.assertEqual(result.llm_filters, [FilterConfig(column="group", operator="eq", value="B")])
        self.assertIsNone(result.analysis)

    def test_unsupported_advanced_question_returns_clarification(self):
        plan = {
            "kind": "clarification",
            "x_axis_key": None,
            "y_axis_keys": [],
            "aggregation": "sum",
            "chart_type": "bar",
            "title": "",
            "x_axis_label": None,
            "y_axis_label": None,
            "filters": [],
            "clarification": "I cannot forecast this data, but I can chart historical totals.",
        }
        fake_message = SimpleNamespace(content=json.dumps(plan), tool_calls=None)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=fake_message)]))
            )
        )
        ds = dataset(pd.DataFrame({"group": ["A"], "value": [1]}), {"group": "categorical", "value": "metric"})
        with patch("services.query_service.get_dataset", return_value=ds), patch(
            "services.query_service.get_openai_client", return_value=fake_client
        ):
            result = run_query(QueryRequest(dataset_id="fixture", user_prompt="forecast next year"))
        self.assertEqual(result.chart_type, "empty")
        self.assertIn("cannot forecast", result.analysis)

    def test_invalid_ai_filters_reject_the_whole_plan(self):
        plan = {
            "kind": "chart",
            "x_axis_key": "group",
            "y_axis_keys": ["value"],
            "aggregation": "sum",
            "chart_type": "bar",
            "title": "Value by group",
            "x_axis_label": None,
            "y_axis_label": None,
            "filters": [{"column": "unknown", "operator": "eq", "value": "B"}],
            "clarification": None,
        }
        fake_message = SimpleNamespace(content=json.dumps(plan), tool_calls=None)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=fake_message)]))
            )
        )
        ds = dataset(pd.DataFrame({"group": ["A"], "value": [1]}), {"group": "categorical", "value": "metric"})
        with patch("services.query_service.get_dataset", return_value=ds), patch(
            "services.query_service.get_openai_client", return_value=fake_client
        ):
            result = run_query(QueryRequest(dataset_id="fixture", user_prompt="show value"))
        self.assertEqual(result.chart_type, "empty")
        self.assertIn("column", result.analysis.lower())

    def test_drilldown_serializes_missing_float_and_datetime_values_as_null(self):
        ds = dataset(
            pd.DataFrame(
                {
                    "group": ["A", None],
                    "value": [float("nan"), float("inf")],
                    "date": pd.to_datetime(["2024-01-01", None]),
                }
            )
        )
        client = TestClient(app)
        with patch("routers.analytics.get_dataset", return_value=ds):
            response = client.post("/drilldown", json={"dataset_id": "fixture", "limit": 10})
            null_group = client.post(
                "/drilldown",
                json={
                    "dataset_id": "fixture",
                    "filters": [{"column": "group", "values": [None]}],
                    "limit": 10,
                },
            )
            mixed_groups = client.post(
                "/drilldown",
                json={
                    "dataset_id": "fixture",
                    "filters": [{"column": "group", "values": ["A", None]}],
                    "limit": 10,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["data"][0]["value"])
        self.assertEqual(response.json()["data"][0]["date"], "2024-01-01T00:00:00")
        self.assertIsNone(response.json()["data"][1]["value"])
        self.assertIsNone(response.json()["data"][1]["date"])
        self.assertEqual(null_group.status_code, 200)
        self.assertEqual(len(null_group.json()["data"]), 1)
        self.assertIsNone(null_group.json()["data"][0]["group"])
        self.assertEqual(mixed_groups.status_code, 200)
        self.assertEqual(len(mixed_groups.json()["data"]), 2)

    def test_identifier_measure_rejects_misleading_sum_label(self):
        ds = dataset(
            pd.DataFrame({"group": ["A"], "record_id": [12]}),
            {"group": "categorical", "record_id": "identifier"},
        )
        request = AggregateRequest(
            dataset_id="fixture", x_axis_key="group", y_axis_keys=["record_id"], aggregation="sum"
        )
        with patch("services.aggregation_service.get_dataset", return_value=ds):
            with self.assertRaises(HTTPException) as context:
                run_aggregate(request)
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("require count", context.exception.detail)


if __name__ == "__main__":
    unittest.main()
