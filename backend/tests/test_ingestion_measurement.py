"""Timing includes CSV parsing; telemetry never implies request-local RSS."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import pandas as pd
from fastapi import HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routers.ingestion import upload_csv
from services.ingestion_service import ingest_dataframe
from storage import DATASETS
from utils.pipeline_logging import process_peak_rss_bytes


class IngestionMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.ai = patch.dict(os.environ, {"OPENAI_API_KEY": "", "COLUMN_INTERPRETER": "off"})
        self.ai.start()
        self.addCleanup(self.ai.stop)
        self.addCleanup(DATASETS.clear)

    def test_route_total_includes_csv_parse_and_defers_summary(self):
        clock = [0.0]
        def parse(source):
            clock[0] += 7
            return pd.DataFrame({"amount": [10, 20], "group": ["a", "b"]})
        output = io.StringIO()
        with patch.dict(os.environ, {"ANALYTICO_INGESTION_ENGINE": "pandas"}), \
             patch("utils.pipeline_logging.perf_counter", side_effect=lambda: clock[0]), \
             patch("services.csv_ingestion.read_csv_fast", side_effect=parse), \
             patch("services.ingestion_service._generate_business_summary",
                   side_effect=AssertionError("summary must run after the response")) as summary, \
             patch("utils.pipeline_logging.process_peak_rss_bytes", side_effect=[1000, 2000]), \
             contextlib.redirect_stdout(output):
            response = upload_csv(UploadFile(filename="private-name.csv", file=io.BytesIO(b"ignored")))
        summary.assert_not_called()
        lines = [line for line in output.getvalue().splitlines() if line.startswith('{')]
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["phase_seconds"]["csv_parse"], 7)
        self.assertEqual(record["phase_seconds"].get("llm_summary", 0), 0)
        self.assertEqual(record["total_seconds"], 7)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(response.enrichment_status, "disabled")
        self.assertIsNone(response.summary)
        self.assertEqual(record["timing_scope"], "handler_parse_through_storage_excludes_transfer_and_serialization")
        self.assertEqual((record["rows"], record["columns"]), (2, 2))
        self.assertEqual(record["process_peak_rss_bytes"], 2000)
        self.assertEqual(record["memory_scope"], "process_lifetime_high_water")
        self.assertNotIn("private-name", output.getvalue())

    def test_failed_parse_has_sanitized_error_record(self):
        output = io.StringIO()
        with patch("modules.disk_dataset.DiskDataset.from_csv", side_effect=ValueError("secret input")), \
             contextlib.redirect_stdout(output):
            with self.assertRaises(HTTPException):
                upload_csv(UploadFile(filename="private-name.csv", file=io.BytesIO(b"ignored")))
        record = json.loads(next(line for line in output.getvalue().splitlines() if line.startswith('{')))
        self.assertEqual(record["status"], "error")
        self.assertIn("disk_ingestion", record["phase_seconds"])
        self.assertNotIn("secret", output.getvalue())

    def test_direct_dataframe_ingestion_does_not_claim_csv_parse(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ingest_dataframe(pd.DataFrame({"amount": [10, 20]}), "private.csv", "/test")
        record = json.loads(next(line for line in output.getvalue().splitlines() if line.startswith('{')))
        self.assertNotIn("csv_parse", record["phase_seconds"])
        self.assertEqual(record["timing_scope"],
                         "dataframe_ingestion_through_storage_excludes_csv_parse_transfer_and_serialization")
        self.assertIn("column_statistics", record["phase_seconds"])
        self.assertGreaterEqual(record["total_seconds"], sum(record["phase_seconds"].values()))

    def test_rss_units_and_unavailable_value(self):
        try:
            import resource
        except ImportError:
            self.assertIsNone(process_peak_rss_bytes())
            return
        with patch.object(resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=123)):
            with patch("utils.pipeline_logging.sys.platform", "darwin"):
                self.assertEqual(process_peak_rss_bytes(), 123)
            with patch("utils.pipeline_logging.sys.platform", "linux"):
                self.assertEqual(process_peak_rss_bytes(), 123 * 1024)
        with patch.object(resource, "getrusage", side_effect=OSError("unavailable")):
            self.assertIsNone(process_peak_rss_bytes())
