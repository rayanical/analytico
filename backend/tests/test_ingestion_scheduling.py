"""Regression tests for overlapped CSV validation and loader scheduling."""

from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import modules.disk_dataset as disk_dataset
from modules.disk_dataset import DiskDataset
from modules.import_policy import (
    CSVStructureError,
    ImportSettings,
    validate_csv_structure as canonical_validate_csv_structure,
)


class IngestionSchedulingTests(unittest.TestCase):
    WAIT_SECONDS = 4

    def run_in_thread(self, source, *, callback=None, temp_root=None):
        result = {}

        def load():
            try:
                result["dataset"] = DiskDataset.from_csv(
                    source,
                    on_source_loaded=callback,
                    temp_root=temp_root,
                )
            except BaseException as error:
                result["error"] = error

        thread = threading.Thread(target=load, name="csv-import-test")
        thread.start()
        return result, thread

    def finish_thread(self, result, thread):
        thread.join(self.WAIT_SECONDS)
        self.assertFalse(thread.is_alive(), "CSV import worker did not finish")
        dataset = result.get("dataset")
        if dataset is not None:
            dataset.close()

    def test_native_load_overlaps_validation_and_callback_waits_for_validation(self):
        source = b"code,label\n001,A\n002,B\n"
        validator_started = threading.Event()
        release_validation = threading.Event()
        validation_finished = threading.Event()
        loader_started = threading.Event()
        callback_validation_state = []
        callback_called = threading.Event()
        original_ingest = DiskDataset._ingest_csv

        def blocking_validator(source_path, settings=None, **kwargs):
            validator_started.set()
            if not release_validation.wait(self.WAIT_SECONDS):
                raise TimeoutError("test did not release CSV validation")
            try:
                return canonical_validate_csv_structure(source_path, settings, **kwargs)
            finally:
                validation_finished.set()

        def observe_loader(dataset, *args, **kwargs):
            if not validator_started.wait(self.WAIT_SECONDS):
                raise AssertionError("validation was not submitted before native load")
            loader_started.set()
            return original_ingest(dataset, *args, **kwargs)

        def callback(_dataset):
            callback_validation_state.append(validation_finished.is_set())
            callback_called.set()

        result = {}
        with (
            patch.object(disk_dataset, "validate_csv_structure", blocking_validator),
            patch.object(DiskDataset, "_ingest_csv", observe_loader),
        ):
            result, thread = self.run_in_thread(source, callback=callback)
            try:
                self.assertTrue(
                    loader_started.wait(self.WAIT_SECONDS),
                    "native loading did not start while strict validation was pending",
                )
                self.assertFalse(callback_called.is_set())
            finally:
                release_validation.set()
                self.finish_thread(result, thread)

        self.assertNotIn("error", result)
        self.assertEqual(callback_validation_state, [True])

    def test_late_structural_error_skips_callback_and_cleans_temporary_source(self):
        source = b"code,label\n001,A\n002,B,extra\n"
        callback_called = threading.Event()
        with tempfile.TemporaryDirectory(prefix="analytico-scheduling-test-") as temp_root:
            with self.assertRaises(CSVStructureError):
                DiskDataset.from_csv(
                    source,
                    temp_root=temp_root,
                    on_source_loaded=lambda _dataset: callback_called.set(),
                )
            self.assertFalse(callback_called.is_set())
            self.assertEqual(list(Path(temp_root).iterdir()), [])

    def test_loader_failure_waits_for_validation_and_validation_error_wins(self):
        source = b"code,label\n001,A\n002,B,extra\n"
        validator_started = threading.Event()
        release_validation = threading.Event()
        validation_finished = threading.Event()
        loader_failed = threading.Event()
        premature_close = threading.Event()
        original_close = DiskDataset.close

        def blocking_validator(source_path, settings=None, **kwargs):
            validator_started.set()
            if not release_validation.wait(self.WAIT_SECONDS):
                raise TimeoutError("test did not release CSV validation")
            try:
                return canonical_validate_csv_structure(source_path, settings, **kwargs)
            finally:
                validation_finished.set()

        def failing_loader(dataset, *_args, **_kwargs):
            if not validator_started.wait(self.WAIT_SECONDS):
                raise AssertionError("validation was not submitted before native load")
            loader_failed.set()
            raise RuntimeError("forced native-loader failure")

        def track_close(dataset):
            if not validation_finished.is_set():
                premature_close.set()
            return original_close(dataset)

        with tempfile.TemporaryDirectory(prefix="analytico-scheduling-test-") as temp_root:
            with (
                patch.object(disk_dataset, "validate_csv_structure", blocking_validator),
                patch.object(DiskDataset, "_ingest_csv", failing_loader),
                patch.object(DiskDataset, "close", track_close),
            ):
                result, thread = self.run_in_thread(source, temp_root=temp_root)
                try:
                    self.assertTrue(loader_failed.wait(self.WAIT_SECONDS))
                finally:
                    release_validation.set()
                    self.finish_thread(result, thread)

            self.assertIsInstance(result.get("error"), CSVStructureError)
            self.assertFalse(premature_close.is_set())
            self.assertEqual(list(Path(temp_root).iterdir()), [])

    def test_executor_submission_failure_never_returns_unvalidated_rows(self):
        source = b"code,label\n001,A\n002,B,extra\n"
        with tempfile.TemporaryDirectory(prefix="analytico-scheduling-test-") as temp_root:
            with patch(
                "concurrent.futures.ThreadPoolExecutor.submit",
                side_effect=RuntimeError("validator worker unavailable"),
            ):
                with self.assertRaises((RuntimeError, CSVStructureError)):
                    DiskDataset.from_csv(source, temp_root=temp_root)
            self.assertEqual(list(Path(temp_root).iterdir()), [])

    def test_reader_fallbacks_preserve_headers_rows_and_source_values(self):
        cases = [
            (
                b"\ncode,label\n001,A\n",
                None,
                ["code", "label"],
                [["001", "A"]],
            ),
            (
                b"label\nA\n\n   \nB\n",
                None,
                ["label"],
                [["A"], ["B"]],
            ),
            (
                "code,label\n001,café\n".encode("cp1252"),
                ImportSettings(encoding="cp1252"),
                ["code", "label"],
                [["001", "café"]],
            ),
        ]
        for source, settings, expected_headers, expected_rows in cases:
            with self.subTest(settings=settings, size=len(source)):
                dataset = DiskDataset.from_csv(source, import_settings=settings)
                try:
                    headers, rows, row_count = dataset.raw_source_snapshot(
                        limit=1_000,
                        evenly_spaced=False,
                    )
                finally:
                    dataset.close()
                self.assertEqual(headers, expected_headers)
                self.assertEqual(rows, expected_rows)
                self.assertEqual(row_count, len(expected_rows))

    def test_parallel_column_statistics_match_single_thread_results_and_budget(self):
        source = (
            b"record_id,amount,ratio,year,label\n"
            b"0001,10,5%,2024,alpha\n"
            b"0002,20,10%,2025,beta\n"
            b"0003,30,15%,2026,alpha\n"
            b"0004,40,20%,2027,gamma\n"
        )

        def loaded_with_threads(count):
            with (
                patch.dict("os.environ", {"ANALYTICO_DUCKDB_THREADS": str(count)}),
                patch("modules.disk_dataset.os.cpu_count", return_value=8),
            ):
                dataset = DiskDataset.from_csv(source)
            self.addCleanup(dataset.close)
            before = dataset._connection.execute(
                "SELECT current_setting('threads'), current_setting('memory_limit')"
            ).fetchone()
            counts = dataset._collect_column_statistics()
            after = dataset._connection.execute(
                "SELECT current_setting('threads'), current_setting('memory_limit')"
            ).fetchone()
            return dataset, counts, before, after

        serial, serial_counts, serial_before, serial_after = loaded_with_threads(1)
        parallel, parallel_counts, parallel_before, parallel_after = loaded_with_threads(4)

        self.assertEqual(serial._ingestion_threads, 1)
        self.assertEqual(parallel._ingestion_threads, 4)
        self.assertEqual(serial_before[0], 1)
        self.assertEqual(parallel_before[0], 4)
        self.assertEqual(serial_before, serial_after)
        self.assertEqual(parallel_before, parallel_after)
        self.assertEqual(serial_before[1], parallel_before[1])
        self.assertEqual(serial_counts, parallel_counts)
        self.assertEqual(serial.column_schema, parallel.column_schema)
        self.assertEqual(serial.column_stats, parallel.column_stats)


if __name__ == "__main__":
    unittest.main()
