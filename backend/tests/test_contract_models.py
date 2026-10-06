import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import ChartResponse, ColumnSummary, DefaultChart, FilterConfig, QueryRequest


class ContractModelTests(unittest.TestCase):
    def test_upload_metadata_uses_frontend_enum_values(self):
        column = ColumnSummary(
            name="year",
            dtype="int64",
            is_numeric=True,
            is_datetime=False,
            semantic_type="unknown",
            format="general",
            unique_count=2,
            sample_values=["2020", "2021"],
        )
        self.assertEqual(column.format, "general")

        with self.assertRaises(ValidationError):
            ColumnSummary(
                name="year",
                dtype="int64",
                is_numeric=True,
                is_datetime=False,
                semantic_type="made-up",
                format="number",
                unique_count=2,
                sample_values=["2020", "2021"],
            )

        with self.assertRaises(ValidationError):
            DefaultChart(
                x_axis_key="year",
                y_axis_keys=["revenue"],
                chart_type="empty",
                aggregation="sum",
                title="Revenue by year",
                analysis="Summary",
            )

    def test_query_request_exposes_supported_sort_order(self):
        request = QueryRequest(dataset_id="dataset", user_prompt="Show revenue", sort_by="label")
        self.assertEqual(request.sort_by, "label")

    def test_chart_response_nulls_are_explicit_in_serialized_contract(self):
        response = ChartResponse(
            data=[],
            x_axis_key="",
            y_axis_keys=[],
            chart_type="empty",
            title="Clarification Needed",
            row_count=0,
            filters=[FilterConfig(column="region", values=["West", None])],
        ).model_dump(mode="json")

        for field in (
            "aggregation",
            "analysis",
            "warnings",
            "applied_filters",
            "llm_filters",
            "source_x_axis_key",
            "time_bucket",
            "others_label",
            "answer",
        ):
            with self.subTest(field=field):
                self.assertIn(field, response)
                self.assertIsNone(response[field])

        self.assertEqual(response["filters"][0], {
            "column": "region",
            "operator": None,
            "value": None,
            "values": ["West", None],
            "min_val": None,
            "max_val": None,
        })


if __name__ == "__main__":
    unittest.main()
