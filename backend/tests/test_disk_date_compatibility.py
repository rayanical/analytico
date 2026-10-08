"""Date parsing compatibility checks across the pandas and DuckDB loaders."""
import io
import os
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException

from models import AggregateRequest, FilterConfig
from modules.import_policy import ColumnOverride, ImportSettings
from services.aggregation_service import run_aggregate
from services.csv_ingestion import ingest_csv
from storage import DATASETS


class DiskDateCompatibilityTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            "OPENAI_API_KEY": "",
            "AI_GATEWAY_API_KEY": "",
            "COLUMN_INTERPRETER": "off",
            "ANALYTICO_NATIVE_VALIDATION": "0",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def ingest(self, source, engine, *, settings=None, overrides=None, column="event_date"):
        dataset = None
        try:
            response = ingest_csv(
                io.BytesIO(source), "date-bounds.csv", "/test", engine=engine,
                enqueue_enrichment=False, import_settings=settings,
                column_overrides=overrides,
            )
            dataset = DATASETS[response.dataset_id]
            source_path = dataset.disk.source_path if hasattr(dataset, "disk") else dataset.source_path
            self.assertEqual(source_path.read_bytes(), source)
            frame = (dataset.sample_frame(dataset.row_count) if hasattr(dataset, "disk")
                     else dataset.df)
            values = [None if pd.isna(value) else str(value) for value in frame[column]]
            return dataset.column_types[column], dataset.column_formats[column], values
        finally:
            if dataset is not None:
                DATASETS.pop(dataset.id, None)
                dataset.close()

    def assert_out_of_bounds_dates_remain_text(self, out_of_bounds):
        rows = pd.date_range("2024-01-01", periods=101).strftime("%Y-%m-%d").tolist()
        rows.append(out_of_bounds)
        source = ("event_date\n" + "\n".join(rows) + "\n").encode("ascii")

        pandas_type, pandas_format, pandas_values = self.ingest(source, "pandas")
        disk_type, disk_format, disk_values = self.ingest(source, "disk")

        self.assertEqual(pandas_type, "categorical")
        self.assertEqual(disk_type, pandas_type)
        self.assertEqual(disk_format, pandas_format)
        self.assertEqual(disk_values, pandas_values)
        self.assertEqual(disk_values[-1], out_of_bounds)

    def test_late_value_above_pandas_timestamp_max_disables_date_inference(self):
        self.assert_out_of_bounds_dates_remain_text("2263-01-01")

    def test_late_value_below_pandas_timestamp_min_disables_date_inference(self):
        self.assert_out_of_bounds_dates_remain_text("1677-09-20")

    def test_fractional_values_outside_pandas_timestamp_bounds_remain_text(self):
        for value in (
            "1677-09-21 00:12:43.145224",
            "2262-04-11 23:47:16.854776",
        ):
            with self.subTest(value=value):
                self.assert_out_of_bounds_dates_remain_text(value)

    def test_microsecond_aligned_pandas_timestamp_bounds_remain_temporal(self):
        values = [
            "1677-09-21 00:12:43.145225",
            "2262-04-11 23:47:16.854775",
        ]
        source = ("event_date\n" + "\n".join(values) + "\n").encode("ascii")
        pandas_result = self.ingest(source, "pandas")
        disk_result = self.ingest(source, "disk")

        self.assertEqual(disk_result, pandas_result)
        self.assertEqual(pandas_result[0], "temporal")
        self.assertEqual(pandas_result[1], "date")

    def test_timezone_and_submicrosecond_values_remain_exact_text_in_both_loaders(self):
        values = [
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00+02:00",
            "2024-01-03T00:00:00-05:00",
            "2024-01-04T00:00:00.1234567",
            "2024-01-05T00:00:00.12345678",
            "2024-01-06T00:00:00.123456789",
        ]
        source = ("event_date\n" + "\n".join(values) + "\n").encode("ascii")

        pandas_type, pandas_format, pandas_values = self.ingest(source, "pandas")
        disk_type, disk_format, disk_values = self.ingest(source, "disk")

        self.assertEqual(pandas_type, "categorical")
        self.assertEqual(disk_type, pandas_type)
        self.assertEqual(disk_format, pandas_format)
        self.assertEqual(disk_values, values)
        self.assertEqual(pandas_values, values)

    def test_forced_date_parse_rejects_timezone_and_submicrosecond_values(self):
        for value in ("2024-01-01T00:00:00Z", "2024-01-01T00:00:00.123456789"):
            source = f"event_date\n{value}\n".encode("ascii")
            for engine in ("pandas", "disk"):
                with self.subTest(value=value, engine=engine):
                    with self.assertRaises((HTTPException, ValueError)):
                        self.ingest(
                            source, engine,
                            overrides=[ColumnOverride(column="event_date", parse_as="date")],
                        )

    def test_ambiguous_dates_stay_text_until_order_is_selected(self):
        source = b"event_date\n01/02/2024\n02/03/2024\n"
        outcomes = {}
        for order in ("auto", "mdy", "dmy"):
            settings = ImportSettings(date_order=order)
            outcomes[order] = [
                self.ingest(source, engine, settings=settings)
                for engine in ("pandas", "disk")
            ]

        self.assertEqual(outcomes["auto"][0][0], "categorical")
        self.assertEqual(outcomes["auto"][0], outcomes["auto"][1])
        for order, expected in (("mdy", ["2024-01-02 00:00:00", "2024-02-03 00:00:00"]),
                                ("dmy", ["2024-02-01 00:00:00", "2024-03-02 00:00:00"])):
            with self.subTest(order=order):
                self.assertEqual(outcomes[order][0], outcomes[order][1])
                self.assertEqual(outcomes[order][0][0], "temporal")
                self.assertEqual(outcomes[order][0][1], "date")
                self.assertEqual(outcomes[order][0][2], expected)

    def test_leading_nulls_with_custom_tokens_do_not_hide_dates(self):
        source = b"event_date\nNA\nNULL\n2024-01-02\n"
        settings = ImportSettings(null_values=["", "NA", "NULL"])
        overrides = [ColumnOverride(column="event_date", parse_as="date")]
        results = [
            self.ingest(source, engine, settings=settings, overrides=overrides)
            for engine in ("pandas", "disk")
        ]

        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][0], "temporal")
        self.assertEqual(results[0][2], [None, None, "2024-01-02 00:00:00"])

    def test_encoding_leading_blank_and_quoted_newline_preserve_date_values(self):
        cases = [
            (
                b"\nevent_date,note\n2024-03-01,plain\n",
                None,
            ),
            (
                "event_date,note\n2024-03-02,\"first line\nsecond line\"\n".encode("utf-16"),
                ImportSettings(encoding="utf-16"),
            ),
            (
                "event_date,note\n2024-03-03,caf\u00e9\n".encode("cp1252"),
                ImportSettings(encoding="cp1252"),
            ),
        ]
        for source, settings in cases:
            with self.subTest(settings=settings):
                pandas_result = self.ingest(source, "pandas", settings=settings)
                disk_result = self.ingest(source, "disk", settings=settings)
                self.assertEqual(pandas_result, disk_result)
                self.assertEqual(pandas_result[0], "temporal")

    def test_date_filtering_and_month_buckets_match_serialized_results(self):
        source = (
            b"event_date,amount\n"
            b"2024-03-01,1\n2024-03-15,2\n2024-03-31,3\n2024-04-01,4\n"
        )
        charts = []
        for engine in ("pandas", "disk"):
            response = ingest_csv(
                io.BytesIO(source), "date-filter.csv", "/test", engine=engine,
                enqueue_enrichment=False,
            )
            dataset = DATASETS[response.dataset_id]
            try:
                chart = run_aggregate(AggregateRequest(
                    dataset_id=dataset.id,
                    x_axis_key="event_date",
                    y_axis_keys=["amount"],
                    aggregation="sum",
                    filters=[FilterConfig(column="event_date", operator="gte", value="2024-03-15")],
                    time_bucket="month",
                    group_others=False,
                ))
                charts.append(chart.model_dump(mode="json"))
            finally:
                DATASETS.pop(dataset.id, None)
                dataset.close()

        self.assertEqual(charts[0], charts[1])
        self.assertEqual(charts[0]["data"], [
            {"event_date": "2024-03-01T00:00:00", "amount": 5},
            {"event_date": "2024-04-01T00:00:00", "amount": 4},
        ])


if __name__ == "__main__":
    unittest.main()
