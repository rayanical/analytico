import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import app
from models import UploadResponse


class ApiSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @staticmethod
    def _fake_upload_response(filename: str = "demo.csv") -> UploadResponse:
        return UploadResponse.model_validate(
            {
                "dataset_id": "demo-id",
                "filename": filename,
                "row_count": 2,
                "columns": [
                    {
                        "name": "value",
                        "dtype": "int64",
                        "is_numeric": True,
                        "is_datetime": False,
                        "semantic_type": "metric",
                        "format": "number",
                        "unique_count": 2,
                        "sample_values": [1, 2],
                    }
                ],
                "column_formats": {"value": "number"},
                "data_health": {
                    "missing_values": {"value": 0},
                    "cleaning_actions": [],
                    "quality_score": 100,
                },
                "profile": {
                    "top_metrics": [
                        {"name": "value", "total": 3, "average": 1.5, "min": 1, "max": 2}
                    ],
                    "time_range": None,
                    "row_count": 2,
                    "column_count": 1,
                },
                "default_chart": {
                    "x_axis_key": "value",
                    "y_axis_keys": ["value"],
                    "chart_type": "bar",
                    "aggregation": "sum",
                    "title": "Value",
                    "analysis": "Simple chart.",
                },
                "suggestions": ["a", "b", "c"],
                "summary": "summary",
            }
        )

    def test_root_and_validate_reachable(self):
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        self.assertEqual(root.json()["status"], "ok")

        validate = self.client.get("/validate/missing-id")
        self.assertEqual(validate.status_code, 200)
        self.assertFalse(validate.json()["valid"])

    def test_upload_smoke(self):
        fake_df = pd.DataFrame({"value": [1, 2]})
        with patch("routers.ingestion.read_csv_fast", return_value=fake_df), patch(
            "routers.ingestion.ingest_dataframe", return_value=self._fake_upload_response("upload.csv")
        ):
            response = self.client.post(
                "/upload",
                files={"file": ("upload.csv", io.BytesIO(b"value\n1\n2\n"), "text/csv")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["filename"], "upload.csv")

    def test_load_demo_taxi_gapminder_and_unknown(self):
        fake_df = pd.DataFrame({"value": [1, 2]})
        with patch("routers.ingestion.read_csv_fast", return_value=fake_df), patch(
            "routers.ingestion.ingest_dataframe", return_value=self._fake_upload_response("demo.csv")
        ):
            taxi = self.client.post("/load-demo", params={"dataset": "taxi"})
            gapminder = self.client.post("/load-demo", params={"dataset": "gapminder"})

        self.assertEqual(taxi.status_code, 200)
        self.assertEqual(gapminder.status_code, 200)

        unknown = self.client.post("/load-demo", params={"dataset": "unknown"})
        self.assertEqual(unknown.status_code, 400)

    def test_analytics_routes_reachable(self):
        aggregate = self.client.post(
            "/aggregate",
            json={
                "dataset_id": "missing",
                "x_axis_key": "x",
                "y_axis_keys": ["y"],
            },
        )
        self.assertEqual(aggregate.status_code, 404)

        drilldown = self.client.post("/drilldown", json={"dataset_id": "missing", "limit": 10})
        self.assertEqual(drilldown.status_code, 404)

        query = self.client.post(
            "/query",
            json={
                "dataset_id": "missing",
                "user_prompt": "show total",
            },
        )
        self.assertIn(query.status_code, {500, 404})


if __name__ == "__main__":
    unittest.main()
