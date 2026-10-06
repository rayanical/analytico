"""Differential integration tests for pandas and disk-backed CSV datasets."""

import io
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import storage
from main import app
from models import DrillDownRequest
from modules.disk_dataset import DiskDataset
from services.csv_ingestion import ingest_csv
from storage import DATASETS, DiskDatasetInfo, cleanup_expired, lease_dataset, store_dataset


class CsvDiskIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.ai_environment = patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "COLUMN_INTERPRETER": "off"},
        )
        self.ai_environment.start()
        self.addCleanup(self.ai_environment.stop)
        self.addCleanup(self._close_datasets)
        self._close_datasets()

    @staticmethod
    def _close_datasets():
        for dataset in list(DATASETS.values()):
            dataset.close()
        DATASETS.clear()

    def upload(self, contents: str, engine: str, **extra_environment):
        environment = {
            "ANALYTICO_INGESTION_ENGINE": engine,
            "OPENAI_API_KEY": "",
            "COLUMN_INTERPRETER": "off",
            **extra_environment,
        }
        with patch.dict(os.environ, environment):
            response = self.client.post(
                "/upload",
                files={
                    "file": (
                        "fixture.csv",
                        io.BytesIO(contents.encode("utf-8")),
                        "text/csv",
                    )
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["enrichment_status"], "disabled")
        return payload

    def assert_uploads_match(self, pandas_upload, disk_upload):
        self.assertEqual(pandas_upload["row_count"], disk_upload["row_count"])
        self.assertEqual(pandas_upload["profile"]["row_count"], disk_upload["profile"]["row_count"])
        self.assertEqual(
            pandas_upload["data_health"]["missing_values"],
            disk_upload["data_health"]["missing_values"],
        )
        self.assertEqual(
            [column["name"] for column in pandas_upload["columns"]],
            [column["name"] for column in disk_upload["columns"]],
        )

    def aggregate(self, dataset_id, **overrides):
        request = {
            "dataset_id": dataset_id,
            "x_axis_key": "segment",
            "y_axis_keys": ["amount"],
            "aggregation": "sum",
            "chart_type": "bar",
            "sort_by": "label",
            "group_others": False,
            "limit": 500,
            **overrides,
        }
        response = self.client.post("/aggregate", json=request)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def drilldown(self, dataset_id, **overrides):
        response = self.client.post(
            "/drilldown",
            json={"dataset_id": dataset_id, "limit": 100, **overrides},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    @staticmethod
    def values_by_group(payload, group_key="segment", value_key="amount"):
        return {row[group_key]: row[value_key] for row in payload["data"]}

    def assert_values_match(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            if left[key] is None:
                self.assertIsNone(right[key], key)
            else:
                self.assertAlmostEqual(float(left[key]), float(right[key]), places=9, msg=str(key))

    def test_forced_engines_match_rows_missing_values_and_all_aggregations(self):
        contents = (
            "segment,amount\n"
            "A,1\nA,3\n"
            "B,5\nB,\n"
            ",7\n,9\n"
            "C,\n"
        )
        pandas_upload = self.upload(contents, "pandas")
        disk_upload = self.upload(contents, "disk")
        self.assert_uploads_match(pandas_upload, disk_upload)

        expected = {
            "sum": {"A": 4, "B": 5, None: 16, "C": None},
            "mean": {"A": 2, "B": 5, None: 8, "C": None},
            "median": {"A": 2, "B": 5, None: 8, "C": None},
            "count": {"A": 2, "B": 1, None: 2, "C": 0},
            "min": {"A": 1, "B": 5, None: 7, "C": None},
            "max": {"A": 3, "B": 5, None: 9, "C": None},
        }
        for aggregation, wanted in expected.items():
            with self.subTest(aggregation=aggregation):
                pandas_chart = self.aggregate(
                    pandas_upload["dataset_id"], aggregation=aggregation,
                )
                disk_chart = self.aggregate(
                    disk_upload["dataset_id"], aggregation=aggregation,
                )
                self.assertEqual(pandas_chart["row_count"], 4)
                self.assertEqual(disk_chart["row_count"], 4)
                pandas_values = self.values_by_group(pandas_chart)
                disk_values = self.values_by_group(disk_chart)
                self.assert_values_match(wanted, pandas_values)
                self.assert_values_match(pandas_values, disk_values)

        pandas_all_null = self.upload("segment,amount\n,1\n,2\n,3\n", "pandas")
        disk_all_null = self.upload("segment,amount\n,1\n,2\n,3\n", "disk")
        for uploaded in (pandas_all_null, disk_all_null):
            chart = self.aggregate(uploaded["dataset_id"], aggregation="sum")
            self.assertEqual(chart["row_count"], 1)
            self.assertEqual(chart["data"], [{"segment": None, "amount": 6}])

    def test_weighted_mean_others_preserves_a_real_others_label_and_repeats(self):
        rows = ["segment,amount"]
        rows.extend(["A,1"] * 10)
        rows.extend(["B,9", "C,8", "Others,7", "D,6"])
        contents = "\n".join(rows) + "\n"
        uploads = [self.upload(contents, engine) for engine in ("pandas", "disk")]

        charts = []
        for uploaded in uploads:
            request = {
                "aggregation": "mean",
                "group_others": True,
                "limit": 3,
                "sort_by": "value",
            }
            first = self.aggregate(uploaded["dataset_id"], **request)
            second = self.aggregate(uploaded["dataset_id"], **request)
            self.assertEqual(first["data"], second["data"])
            self.assertEqual(first["others_label"], "Others (2)")
            other = next(row for row in first["data"] if row["segment"] == "Others (2)")
            self.assertAlmostEqual(other["amount"], 23 / 12, places=9)
            charts.append(first)

        self.assertEqual(charts[0]["data"], charts[1]["data"])
        self.assertEqual(charts[0]["others_label"], charts[1]["others_label"])

    def test_large_integer_records_and_decimal_filters_are_exact(self):
        contents = (
            "amount,measure\n"
            "9007199254740992,0.1\n"
            "9007199254740993,0.2\n"
            "9007199254740994,0.3\n"
        )
        uploads = [self.upload(contents, engine) for engine in ("pandas", "disk")]
        for uploaded in uploads:
            dataset_id = uploaded["dataset_id"]
            exact = self.drilldown(
                dataset_id,
                filters=[{"column": "amount", "operator": "eq", "value": "9007199254740993"}],
            )
            self.assertEqual(exact["total_rows"], 1)
            self.assertEqual(exact["data"][0]["amount"], "9007199254740993")
            self.assertEqual(exact["data"][0]["measure"], 0.2)

            decimal_boundary = self.drilldown(
                dataset_id,
                filters=[{"column": "amount", "operator": "gt", "value": "9007199254740992.5"}],
            )
            self.assertEqual(decimal_boundary["total_rows"], 2)
            self.assertEqual(
                [row["amount"] for row in decimal_boundary["data"]],
                ["9007199254740993", "9007199254740994"],
            )

            decimal_filter = self.drilldown(
                dataset_id,
                filters=[{"column": "measure", "operator": "gt", "value": "0.15"}],
            )
            self.assertEqual(decimal_filter["total_rows"], 2)

        pandas_chart = self.client.post(
            "/aggregate",
            json={
                "dataset_id": uploads[0]["dataset_id"],
                "x_axis_key": "amount",
                "y_axis_keys": ["measure"],
                "aggregation": "sum",
                "sort_by": "label",
                "group_others": False,
            },
        )
        disk_chart = self.client.post(
            "/aggregate",
            json={
                "dataset_id": uploads[1]["dataset_id"],
                "x_axis_key": "amount",
                "y_axis_keys": ["measure"],
                "aggregation": "sum",
                "sort_by": "label",
                "group_others": False,
            },
        )
        self.assertEqual(pandas_chart.status_code, 200, pandas_chart.text)
        self.assertEqual(disk_chart.status_code, 200, disk_chart.text)
        wanted_keys = [str(value) for value in (9007199254740992, 9007199254740993, 9007199254740994)]
        self.assertEqual([row["amount"] for row in pandas_chart.json()["data"]], wanted_keys)
        self.assertEqual([row["amount"] for row in disk_chart.json()["data"]], wanted_keys)

    def test_invalid_columns_filters_and_no_match_responses_are_consistent(self):
        contents = "segment,amount\nA,1\nB,2\n"
        uploads = [self.upload(contents, engine) for engine in ("pandas", "disk")]
        for uploaded in uploads:
            dataset_id = uploaded["dataset_id"]
            same_axis = self.client.post(
                "/aggregate",
                json={
                    "dataset_id": dataset_id,
                    "x_axis_key": "amount",
                    "y_axis_keys": ["amount"],
                },
            )
            self.assertEqual(same_axis.status_code, 400)

            missing_axis = self.client.post(
                "/aggregate",
                json={
                    "dataset_id": dataset_id,
                    "x_axis_key": "missing",
                    "y_axis_keys": ["amount"],
                },
            )
            self.assertEqual(missing_axis.status_code, 400)

            invalid_filter = self.client.post(
                "/aggregate",
                json={
                    "dataset_id": dataset_id,
                    "x_axis_key": "segment",
                    "y_axis_keys": ["amount"],
                    "filters": [{"column": "amount", "operator": "gt", "value": "not-a-number"}],
                },
            )
            self.assertEqual(invalid_filter.status_code, 400)

            missing_filter_column = self.client.post(
                "/drilldown",
                json={"dataset_id": dataset_id, "filters": [{"column": "missing", "values": ["x"]}]},
            )
            self.assertEqual(missing_filter_column.status_code, 400)

            no_match = self.client.post(
                "/aggregate",
                json={
                    "dataset_id": dataset_id,
                    "x_axis_key": "segment",
                    "y_axis_keys": ["amount"],
                    "filters": [{"column": "amount", "operator": "gt", "value": 999}],
                },
            )
            self.assertEqual(no_match.status_code, 400)
            self.assertIn("no data matches filters", no_match.json()["detail"].lower())
            no_match_rows = self.drilldown(
                dataset_id,
                filters=[{"column": "amount", "operator": "gt", "value": 999}],
            )
            self.assertEqual(no_match_rows["total_rows"], 0)
            self.assertEqual(no_match_rows["data"], [])

    def test_leased_eviction_keeps_disk_readable_until_lease_exits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original = Path(temp_dir) / "source.csv"
            source_contents = "segment,amount\nA,11\n"
            original.write_text(source_contents, encoding="utf-8")

            with patch.object(storage, "MAX_DATASETS", 1):
                first_disk = DiskDataset.from_csv(original, filename=original.name)
                first = DiskDatasetInfo(first_disk, original.name)
                store_dataset(first)
                managed_source = first_disk.raw_source_path
                self.assertNotEqual(managed_source, original)

                with lease_dataset(first.id) as leased:
                    second = DiskDatasetInfo(
                        DiskDataset.from_csv(b"segment,amount\nB,22\n"),
                        "second.csv",
                    )
                    store_dataset(second)

                    self.assertNotIn(first.id, DATASETS)
                    self.assertFalse(first_disk._closed)
                    self.assertTrue(managed_source.exists())
                    result = leased.disk.drilldown(
                        DrillDownRequest(dataset_id=first.id, limit=10)
                    )
                    self.assertEqual(result["total_rows"], 1)
                    self.assertEqual(result["data"], [{"segment": "A", "amount": 11}])

                self.assertTrue(first_disk._closed)
                self.assertFalse(managed_source.exists())
                self.assertEqual(original.read_text(encoding="utf-8"), source_contents)

    def test_exception_inside_lease_still_closes_retired_disk_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original = Path(temp_dir) / "source.csv"
            source_contents = "segment,amount\nA,11\n"
            original.write_text(source_contents, encoding="utf-8")

            with patch.object(storage, "MAX_DATASETS", 1):
                first_disk = DiskDataset.from_csv(original, filename=original.name)
                first = DiskDatasetInfo(first_disk, original.name)
                store_dataset(first)
                managed_source = first_disk.raw_source_path

                with self.assertRaisesRegex(RuntimeError, "simulated reader failure"):
                    with lease_dataset(first.id):
                        second = DiskDatasetInfo(
                            DiskDataset.from_csv(b"segment,amount\nB,22\n"),
                            "second.csv",
                        )
                        store_dataset(second)
                        self.assertNotIn(first.id, DATASETS)
                        self.assertFalse(first_disk._closed)
                        self.assertTrue(managed_source.exists())
                        raise RuntimeError("simulated reader failure")

                self.assertTrue(first_disk._closed)
                self.assertFalse(managed_source.exists())
                self.assertEqual(original.read_text(encoding="utf-8"), source_contents)

    def test_evicted_disk_dataset_finishes_an_in_flight_aggregate(self):
        entered_aggregate = threading.Event()
        release_aggregate = threading.Event()
        result = {}
        errors = []

        def paused_aggregate(disk, request):
            entered_aggregate.set()
            if not release_aggregate.wait(timeout=2):
                raise TimeoutError("test did not release the paused aggregate")
            return original_aggregate(disk, request)

        with patch.object(storage, "MAX_DATASETS", 1):
            first_upload = self.upload("segment,amount\nA,11\n", "disk")
            first = DATASETS[first_upload["dataset_id"]]
            first_disk = first.disk
            managed_source = first_disk.raw_source_path
            original_aggregate = DiskDataset.aggregate

            def request_aggregate():
                try:
                    result["response"] = self.client.post(
                        "/aggregate",
                        json={
                            "dataset_id": first.id,
                            "x_axis_key": "segment",
                            "y_axis_keys": ["amount"],
                            "aggregation": "sum",
                            "group_others": False,
                        },
                    )
                except Exception as error:  # surfaced after joining the worker thread
                    errors.append(error)

            with patch.object(DiskDataset, "aggregate", new=paused_aggregate):
                worker = threading.Thread(target=request_aggregate)
                worker.start()
                try:
                    self.assertTrue(entered_aggregate.wait(timeout=2))
                    second_client = TestClient(app)
                    second_response = second_client.post(
                        "/upload",
                        files={
                            "file": (
                                "second.csv",
                                io.BytesIO(b"segment,amount\nB,22\n"),
                                "text/csv",
                            )
                        },
                    )
                    self.assertEqual(second_response.status_code, 200, second_response.text)
                    self.assertNotIn(first.id, DATASETS)
                    self.assertFalse(first_disk._closed)
                    self.assertTrue(managed_source.exists())
                finally:
                    release_aggregate.set()
                    worker.join(timeout=2)

            self.assertFalse(worker.is_alive(), "aggregate request should finish after release")
            self.assertEqual(errors, [])
            self.assertIn("response", result)
            self.assertEqual(result["response"].status_code, 200, result["response"].text)
            self.assertEqual(
                result["response"].json()["data"],
                [{"segment": "A", "amount": 11}],
            )
            self.assertTrue(first_disk._closed)
            self.assertFalse(managed_source.exists())

    def test_month_and_week_time_buckets_match(self):
        dates = pd.date_range("2024-01-15", periods=70, freq="D")
        contents = "date,amount\n" + "".join(
            f"{value:%Y-%m-%d},1\n" for value in dates
        )
        uploads = [self.upload(contents, engine) for engine in ("pandas", "disk")]
        self.assert_uploads_match(*uploads)

        for bucket, expected_groups in (("month", 3), ("week", 10)):
            charts = []
            for uploaded in uploads:
                response = self.client.post(
                    "/aggregate",
                    json={
                        "dataset_id": uploaded["dataset_id"],
                        "x_axis_key": "date",
                        "y_axis_keys": ["amount"],
                        "aggregation": "sum",
                        "chart_type": "line",
                        "time_bucket": bucket,
                        "sort_by": "label",
                        "group_others": False,
                        "limit": 500,
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                charts.append(response.json())
            self.assertEqual(charts[0]["time_bucket"], bucket)
            self.assertEqual(charts[1]["time_bucket"], bucket)
            self.assertEqual(charts[0]["row_count"], expected_groups)
            self.assertEqual(charts[1]["row_count"], expected_groups)
            self.assertEqual(sum(row["amount"] for row in charts[0]["data"]), 70)
            self.assertEqual(sum(row["amount"] for row in charts[1]["data"]), 70)
            self.assertEqual(charts[0]["data"], charts[1]["data"])

    def test_auto_engine_uses_pandas_for_small_and_disk_for_large_sources(self):
        with patch.dict(os.environ, {"ANALYTICO_DISK_THRESHOLD_BYTES": "100"}):
            small = self.upload("segment,amount\nA,1\n", "auto")
            large_contents = "segment,amount\n" + "".join("A,1\n" for _ in range(100))
            large = self.upload(large_contents, "auto")

        self.assertFalse(hasattr(DATASETS[small["dataset_id"]], "disk"))
        self.assertTrue(hasattr(DATASETS[small["dataset_id"]], "df"))
        self.assertTrue(hasattr(DATASETS[large["dataset_id"]], "disk"))
        self.assertFalse(hasattr(DATASETS[large["dataset_id"]], "df"))

    def test_disk_cleanup_on_expiration_eviction_and_replacement_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original = Path(temp_dir) / "original.csv"
            original_contents = "segment,amount\nA,1\n"
            original.write_text(original_contents, encoding="utf-8")
            with patch.dict(os.environ, {"ANALYTICO_INGESTION_ENGINE": "disk"}):
                uploaded = ingest_csv(original, "original.csv", "/integration", defer_enrichment=False)
            expired = DATASETS[uploaded.dataset_id]
            expired_path = expired.disk.path
            expired.last_accessed = datetime.now() - timedelta(hours=2)
            cleanup_expired()
            self.assertTrue(expired.disk._closed)
            self.assertFalse(expired_path.exists())
            self.assertEqual(original.read_text(encoding="utf-8"), original_contents)

        with patch.object(storage, "MAX_DATASETS", 1):
            first_disk = DiskDataset.from_csv(b"segment,amount\nA,1\n")
            first = DiskDatasetInfo(first_disk, "first.csv")
            store_dataset(first)
            first_path = first_disk.path

            replacement_disk = DiskDataset.from_csv(b"segment,amount\nB,2\n")
            replacement = DiskDatasetInfo(replacement_disk, "replacement.csv")
            replacement.id = first.id
            store_dataset(replacement)
            self.assertTrue(first_disk._closed)
            self.assertFalse(first_path.exists())
            self.assertIs(DATASETS[first.id], replacement)

            replacement_path = replacement_disk.path
            evicting_disk = DiskDataset.from_csv(b"segment,amount\nC,3\n")
            evicting = DiskDatasetInfo(evicting_disk, "evicting.csv")
            store_dataset(evicting)
            self.assertTrue(replacement_disk._closed)
            self.assertFalse(replacement_path.exists())
            self.assertIs(DATASETS[evicting.id], evicting)

    def test_configured_upload_bytes_and_two_ingestion_slots_are_enforced(self):
        with patch.dict(os.environ, {"ANALYTICO_MAX_UPLOAD_BYTES": "8"}):
            too_large = self.client.post(
                "/upload",
                files={"file": ("large.csv", io.BytesIO(b"segment,amount\nA,100\n"), "text/csv")},
            )
        self.assertEqual(too_large.status_code, 413)

        both_started = threading.Event()
        release = threading.Event()
        entered = 0
        entry_lock = threading.Lock()
        failures = []

        def blocking_parse(_source):
            nonlocal entered
            with entry_lock:
                entered += 1
                if entered == 2:
                    both_started.set()
            release.wait(timeout=3)
            return pd.DataFrame({"segment": ["A"], "amount": [1]})

        def ingest_in_thread(name):
            try:
                ingest_csv(
                    io.BytesIO(b"segment,amount\nA,1\n"),
                    name,
                    "/integration",
                    engine="pandas",
                    defer_enrichment=False,
                )
            except Exception as error:
                failures.append(error)

        with patch("services.csv_ingestion.read_csv_fast", side_effect=blocking_parse):
            threads = [
                threading.Thread(target=ingest_in_thread, args=(f"slot-{index}.csv",))
                for index in range(2)
            ]
            for thread in threads:
                thread.start()
            try:
                self.assertTrue(both_started.wait(timeout=2))
                with self.assertRaises(HTTPException) as error:
                    ingest_csv(
                        io.BytesIO(b"segment,amount\nA,1\n"),
                        "overflow.csv",
                        "/integration",
                        engine="pandas",
                        defer_enrichment=False,
                    )
                self.assertEqual(error.exception.status_code, 429)
            finally:
                release.set()
                for thread in threads:
                    thread.join(timeout=5)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
