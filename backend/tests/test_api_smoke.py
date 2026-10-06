import io
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import app
from models import UploadResponse
from storage import DATASETS


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
        with patch(
            "routers.ingestion.ingest_csv",
            return_value=self._fake_upload_response("upload.csv"),
        ):
            response = self.client.post(
                "/upload",
                files={"file": ("upload.csv", io.BytesIO(b"value\n1\n2\n"), "text/csv")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["filename"], "upload.csv")

    def test_upload_and_manual_aggregate_work_without_ai_key(self):
        DATASETS.clear()
        self.addCleanup(DATASETS.clear)
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            response = self.client.post(
                "/upload",
                files={
                    "file": (
                        "manual.csv",
                        io.BytesIO(b"category,amount\nA,10\nB,20\nC,30\n"),
                        "text/csv",
                    )
                },
            )
            self.assertEqual(response.status_code, 200)
            uploaded = response.json()
            self.assertIsNone(uploaded["summary"])
            self.assertEqual(uploaded["enrichment_status"], "disabled")
            chart = self.client.post(
                "/aggregate",
                json={
                    "dataset_id": uploaded["dataset_id"],
                    "x_axis_key": "category",
                    "y_axis_keys": ["amount"],
                    "aggregation": "sum",
                    "sort_by": "label",
                    "group_others": False,
                },
            )
        self.assertEqual(chart.status_code, 200)
        self.assertEqual([row["amount"] for row in chart.json()["data"]], [10, 20, 30])

    def test_upload_preserves_domain_http_status(self):
        with patch(
            "routers.ingestion.ingest_csv",
            side_effect=HTTPException(status_code=422, detail="invalid ingestion request"),
        ):
            response = self.client.post(
                "/upload",
                files={"file": ("upload.csv", io.BytesIO(b"value\n1\n"), "text/csv")},
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "invalid ingestion request")

    def test_malformed_csv_is_a_client_error_and_internal_errors_remain_server_errors(self):
        malformed = self.client.post(
            "/upload",
            files={"file": ("bad.csv", io.BytesIO(b"value,other\n1\n2,3,4\n"), "text/csv")},
        )
        self.assertEqual(malformed.status_code, 400)

        with patch(
            "routers.ingestion.ingest_csv",
            side_effect=RuntimeError("internal failure"),
        ):
            internal = self.client.post(
                "/upload",
                files={"file": ("upload.csv", io.BytesIO(b"value\n1\n"), "text/csv")},
            )
        self.assertEqual(internal.status_code, 500)

    def test_load_demo_taxi_gapminder_and_unknown(self):
        demo_fixtures = {
            "taxi": {"filename": "taxi.csv", "path": BACKEND_DIR / "main.py"},
            "gapminder": {"filename": "gapminder.csv", "path": BACKEND_DIR / "main.py"},
        }
        with patch("routers.ingestion.DEMO_DATASETS", demo_fixtures), patch(
            "routers.ingestion.ingest_csv",
            return_value=self._fake_upload_response("demo.csv"),
        ):
            taxi = self.client.post("/load-demo", params={"dataset": "taxi"})
            gapminder = self.client.post("/load-demo", params={"dataset": "gapminder"})

        self.assertEqual(taxi.status_code, 200)
        self.assertEqual(gapminder.status_code, 200)

        unknown = self.client.post("/load-demo", params={"dataset": "unknown"})
        self.assertEqual(unknown.status_code, 400)

    def test_small_upload_enrichment_runs_after_response_and_keeps_data_unchanged(self):
        DATASETS.clear()
        self.addCleanup(DATASETS.clear)
        summary_started = threading.Event()
        release_summary = threading.Event()

        def slow_summary(_filename, _frame):
            summary_started.set()
            release_summary.wait(timeout=3)
            return "Prepared after upload."

        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "COLUMN_INTERPRETER": "off"}), patch(
                "services.ingestion_service._generate_business_summary",
                side_effect=slow_summary,
            ):
                response = self.client.post(
                    "/upload",
                    files={
                        "file": (
                            "deferred.csv",
                            io.BytesIO(b"category,amount\nA,10\nB,20\n"),
                            "text/csv",
                        )
                    },
                )
                self.assertEqual(response.status_code, 200)
                uploaded = response.json()
                dataset_id = uploaded["dataset_id"]
                self.assertIn(uploaded["enrichment_status"], {"pending", "running"})
                self.assertTrue(summary_started.wait(timeout=1))

                initial_status = self.client.get(f"/enrichment/{dataset_id}")
                self.assertEqual(initial_status.status_code, 200)
                self.assertIn(initial_status.json()["status"], {"pending", "running"})
                stored_dataset = DATASETS[dataset_id]
                original_frame = stored_dataset.df.copy(deep=True)
                original_column_types = dict(stored_dataset.column_types)

                release_summary.set()
                deadline = time.monotonic() + 2
                final_status = initial_status
                while time.monotonic() < deadline:
                    final_status = self.client.get(f"/enrichment/{dataset_id}")
                    if final_status.status_code == 200 and final_status.json()["status"] == "done":
                        break
                    time.sleep(0.01)

                self.assertEqual(final_status.status_code, 200)
                self.assertEqual(final_status.json()["status"], "done")
                self.assertEqual(final_status.json()["summary"], "Prepared after upload.")
                pd.testing.assert_frame_equal(stored_dataset.df, original_frame)
                self.assertEqual(stored_dataset.column_types, original_column_types)
        finally:
            release_summary.set()

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
