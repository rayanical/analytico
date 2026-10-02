"""Behavioral contract for the disk-backed CSV dataset module."""

import io
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import duckdb  # noqa: F401
except ImportError:
    duckdb = None

from models import AggregateRequest, DrillDownRequest, FilterConfig
from modules.disk_dataset import DiskDataset


class _ExecuteCounter:
    def __init__(self, connection):
        self.connection = connection
        self.count = 0

    def execute(self, *args, **kwargs):
        self.count += 1
        return self.connection.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.connection, name)


@unittest.skipIf(duckdb is None, "DuckDB is an application dependency.")
class DiskDatasetTests(unittest.TestCase):
    def create_dataset(self, text: str, **kwargs) -> DiskDataset:
        chunk_size = kwargs.pop("chunk_size", 2)
        dataset = DiskDataset.from_csv(
            io.BytesIO(text.encode("utf-8")),
            filename="fixture.csv",
            chunk_size=chunk_size,
            **kwargs,
        )
        self.addCleanup(dataset.close)
        return dataset

    def test_preserves_csv_lexemes_duplicate_headers_nulls_and_source_bytes(self):
        source = "Code,Code,Label\nNA,001,NA\nNULL,002,NULL\nN/A,003,N/A\n,004,\n"
        dataset = self.create_dataset(source)

        self.assertEqual(dataset.original_headers, ["Code", "Code", "Label"])
        self.assertEqual(dataset.parsed_headers, ["Code", "Code.1", "Label"])
        self.assertEqual(dataset.columns, ["code", "code_1", "label"])
        self.assertEqual(dataset.source_path.read_bytes(), source.encode("utf-8"))
        frame = dataset.sample_frame(10)
        self.assertEqual(frame["code"].tolist()[:3], ["NA", "NULL", "N/A"])
        self.assertEqual(frame["code_1"].tolist(), ["001", "002", "003", "004"])
        self.assertIsNone(frame["label"].iloc[3])
        self.assertEqual(dataset.raw_missing_counts, {"code": 1, "label": 1})

    def test_late_bad_lexeme_prevents_chunk_local_numeric_conversion(self):
        source = "amount\n" + "\n".join(["1", "2", "3", "4", "bad"]) + "\n"
        dataset = self.create_dataset(source)

        self.assertEqual(dataset.column_dtypes["amount"], "object")
        self.assertEqual(dataset.sample_frame(10)["amount"].tolist(), ["1", "2", "3", "4", "bad"])
        self.assertFalse(dataset.column_summaries[0].is_numeric)

    def test_high_cardinality_integer_column_remains_numeric(self):
        source = "group,amount\n" + "".join(f"all,{value}\n" for value in range(1, 2_501))
        dataset = self.create_dataset(source)

        self.assertEqual(dataset.column_dtypes["amount"], "int64")
        self.assertEqual(dataset.unique_counts["amount"], 2_500)
        chart = dataset.aggregate(AggregateRequest(
            dataset_id=dataset.id,
            x_axis_key="group",
            y_axis_keys=["amount"],
            aggregation="sum",
            group_others=False,
        ))
        self.assertEqual(chart.data, [{"group": "all", "amount": 3_126_250}])

    def test_currency_integers_keep_precision_above_float_exact_range(self):
        dataset = self.create_dataset(
            "group,amount\nA,$9007199254740992\nA,$9007199254740993\n"
        )

        self.assertEqual(dataset.column_dtypes["amount"], "int64")
        self.assertEqual(dataset.unique_counts["amount"], 2)
        self.assertEqual(
            dataset.sample_frame(2)["amount"].tolist(),
            [9007199254740992, 9007199254740993],
        )
        chart = dataset.aggregate(AggregateRequest(
            dataset_id=dataset.id,
            x_axis_key="group",
            y_axis_keys=["amount"],
            aggregation="sum",
            group_others=False,
        ))
        self.assertEqual(chart.data, [{"group": "A", "amount": "18014398509481985"}])

    def test_mixed_currency_symbols_preserve_source_text(self):
        for source in ("amount\n$10\n€20\n", "amount\n$€10\n"):
            with self.subTest(source=source):
                dataset = self.create_dataset(source)

                self.assertEqual(dataset.column_dtypes["amount"], "object")
                self.assertEqual(
                    dataset.sample_frame(10)["amount"].tolist(),
                    [line for line in source.splitlines()[1:]],
                )
                self.assertTrue(any(
                    "multiple currency symbols" in action
                    for action in dataset.cleaning_actions
                ))

    def test_nullable_integer_preview_does_not_round_adjacent_large_values(self):
        dataset = self.create_dataset(
            "group,measurement\nA,9007199254740992\nA,\nA,9007199254740993\n"
        )

        values = dataset.sample_frame(3)["measurement"].tolist()
        self.assertEqual(values, [9007199254740992, None, 9007199254740993])

    def test_identifier_compact_dates_remain_source_text(self):
        dataset = self.create_dataset("customer_id\n20240101\n20240102\n")

        self.assertEqual(dataset.column_dtypes["customer_id"], "object")
        self.assertFalse(dataset.column_summaries[0].is_datetime)
        self.assertEqual(dataset.sample_frame(2)["customer_id"].tolist(), ["20240101", "20240102"])

    def test_wide_schema_uses_bounded_statistics_batches(self):
        headers = [f"value_{index}" for index in range(50)]
        rows = [",".join(str(row % 20) for _ in headers) for row in range(10_000)]
        source = ",".join(headers) + "\n" + "\n".join(rows) + "\n"
        dataset = self.create_dataset(source, chunk_size=5_000)

        self.assertEqual(dataset.row_count, 10_000)
        self.assertEqual(len(dataset.columns), 50)
        self.assertEqual(dataset.column_dtypes["value_0"], "int64")
        self.assertEqual(dataset.unique_counts["value_0"], 20)

    def test_aggregate_uses_parameterized_filters_and_drilldown_is_bounded(self):
        dataset = self.create_dataset("group,amount\nA,1\nB,2\nA,3\n")
        request = AggregateRequest(
            dataset_id=dataset.id,
            x_axis_key="group",
            y_axis_keys=["amount"],
            aggregation="sum",
            filters=[FilterConfig(column="group", values=["A"])],
            group_others=False,
        )
        chart = dataset.aggregate(request)
        self.assertEqual(chart.data, [{"group": "A", "amount": 4}])
        detail = dataset.drilldown(DrillDownRequest(
            dataset_id=dataset.id,
            filters=[FilterConfig(column="group", values=["A"])],
            limit=1,
        ))
        self.assertEqual(detail["total_rows"], 2)
        self.assertEqual(detail["data"], [{"group": "A", "amount": 1}])

        injected = request.model_copy(update={
            "filters": [FilterConfig(column="group", operator="eq", value="A' OR TRUE --")]
        })
        with self.assertRaisesRegex(ValueError, "No data matches filters"):
            dataset.aggregate(injected)

    def test_others_combines_tail_rows_and_uses_collision_free_label(self):
        source = "group,amount\n" + "".join(["Others,1\n"] * 10) + "A,2\nB,3\n"
        dataset = self.create_dataset(source)
        chart = dataset.aggregate(AggregateRequest(
            dataset_id=dataset.id,
            x_axis_key="group",
            y_axis_keys=["amount"],
            aggregation="sum",
            limit=2,
            group_others=True,
        ))

        self.assertEqual(chart.others_label, "Others (2)")
        self.assertEqual(chart.data, [
            {"group": "Others", "amount": 10},
            {"group": "Others (2)", "amount": 5},
        ])

    def test_others_collision_search_is_bounded_for_many_occupied_suffixes(self):
        rows = ["Others,1"]
        rows.extend(f"Others ({suffix}),{suffix}" for suffix in range(2, 1_002))
        rows.append("A,0")
        dataset = self.create_dataset("group,amount\n" + "\n".join(rows) + "\n", chunk_size=1_002)
        counter = _ExecuteCounter(dataset._connection)
        dataset._connection = counter

        chart = dataset.aggregate(AggregateRequest(
            dataset_id=dataset.id,
            x_axis_key="group",
            y_axis_keys=["amount"],
            aggregation="sum",
            limit=2,
            group_others=True,
        ))

        self.assertEqual(chart.others_label, "Others (1002)")
        self.assertLessEqual(counter.count, 8)

    def test_others_collision_ignores_noncanonical_suffixes_and_handles_uint64_max(self):
        source = (
            "group,amount\n"
            "Others,1\n"
            "Others (02),2\n"
            "Others (18446744073709551615),3\n"
            "A,4\n"
            "B,5\n"
        )
        dataset = self.create_dataset(source)
        chart = dataset.aggregate(AggregateRequest(
            dataset_id=dataset.id,
            x_axis_key="group",
            y_axis_keys=["amount"],
            aggregation="sum",
            limit=2,
            group_others=True,
        ))

        self.assertEqual(chart.others_label, "Others (2)")

    def test_disk_aggregate_rejects_axis_reused_as_measure(self):
        dataset = self.create_dataset("group,amount\nA,1\nB,2\n")
        request = AggregateRequest(
            dataset_id=dataset.id,
            x_axis_key="amount",
            y_axis_keys=["amount"],
            aggregation="sum",
            group_others=False,
        )

        with self.assertRaisesRegex(ValueError, "x-axis column"):
            dataset.aggregate(request)

    def test_date_columns_use_global_validation_and_time_buckets(self):
        dataset = self.create_dataset(
            "date,amount\n2024-01-15,1\n2024-04-15,2\n2024-12-15,3\n"
        )
        self.assertTrue(dataset.column_summaries[0].is_datetime)
        chart = dataset.aggregate(AggregateRequest(
            dataset_id=dataset.id,
            x_axis_key="date",
            y_axis_keys=["amount"],
            aggregation="sum",
            time_bucket="year",
            group_others=False,
        ))
        self.assertEqual(chart.time_bucket, "year")
        self.assertEqual(chart.data, [{"date": "2024-01-01T00:00:00", "amount": 6}])

    def test_timezone_and_submicrosecond_timestamps_are_preserved_as_text(self):
        source = (
            "event_time\n"
            "2024-01-01T00:00:00+02:00\n"
            "2024-01-02T00:00:00.123456789\n"
        )
        dataset = self.create_dataset(source)

        self.assertEqual(dataset.column_dtypes["event_time"], "object")
        self.assertFalse(dataset.column_summaries[0].is_datetime)
        self.assertEqual(
            dataset.sample_frame(2)["event_time"].tolist(),
            ["2024-01-01T00:00:00+02:00", "2024-01-02T00:00:00.123456789"],
        )

    def test_numeric_filters_reject_values_that_overflow_float(self):
        dataset = self.create_dataset("group,amount\nA,1.25\n")
        request = AggregateRequest(
            dataset_id=dataset.id,
            x_axis_key="group",
            y_axis_keys=["amount"],
            aggregation="sum",
            filters=[FilterConfig(column="amount", operator="gt", value="1e400")],
            group_others=False,
        )

        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            dataset.aggregate(request)

    def test_interpretation_inputs_are_evenly_spaced_and_bounded(self):
        source = "code,amount\n" + "".join(f"00{index},{index}\n" for index in range(24))
        dataset = self.create_dataset(source)
        inputs = dataset.interpretation_inputs()

        self.assertEqual(len(inputs), 2)
        column, payload = inputs[0]
        self.assertEqual(column, "code")
        self.assertEqual(payload["column_name"], "code")
        self.assertEqual(len(payload["values"]), 12)
        self.assertEqual(payload["context"]["row_count"], 24)
        self.assertFalse(payload["context"]["sample_is_complete"])
        self.assertEqual(payload["values"][0], "000")
        self.assertEqual(payload["values"][-1], "0023")


if __name__ == "__main__":
    unittest.main()
