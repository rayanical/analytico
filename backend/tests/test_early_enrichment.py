"""The opt-in overlap path submits bounded work and adopts it without blocking workers."""

import io
import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.early_enrichment import prepare_early_enrichment
from services.parallel_enrichment import run_parallel_enrichment, submit_early_requests


class FakeDisk:
    def __init__(self, columns=21, rows=40):
        self.parsed_headers = [f"header_{index}" for index in range(columns)]
        self.columns = [f"column_{index}" for index in range(columns)]
        self.row_count = rows
        self._values = [[f"r{row}_c{column}" for column in range(columns)] for row in range(rows)]

    def raw_source_snapshot(self, limit=12, *, evenly_spaced=True):
        if evenly_spaced:
            positions = sorted({round(index * (self.row_count - 1) / 11) for index in range(12)})
        else:
            positions = list(range(min(limit, self.row_count)))
        positions = positions[:limit]
        return self.parsed_headers, [self._values[index] for index in positions], self.row_count


class EarlyEnrichmentTests(unittest.TestCase):
    def test_raw_summary_starts_from_first_three_rows_and_labels_keep_even_sample(self):
        disk = FakeDisk()
        summary_seen = []
        labels_seen = []
        summary_started = threading.Event()

        def summary(filename, frame):
            summary_seen.append((filename, frame.copy()))
            summary_started.set()
            return "summary"

        def labels(snapshot):
            labels_seen.append(snapshot)
            return {"interpretation_proposals": {}, "column_labels": {}}

        with patch.dict(os.environ, {"ANALYTICO_EARLY_AI": "1"}):
            early = prepare_early_enrichment(
                disk, "dataset.csv", summary_generator=summary, label_generator=labels,
            )
        self.assertIsNotNone(early)
        self.assertTrue(summary_started.wait(1))
        self.assertEqual(early.futures["summary"].result(2), "summary")
        early.futures["labels"].result(2)

        filename, frame = summary_seen[0]
        self.assertEqual(filename, "dataset.csv")
        self.assertEqual(frame.shape, (3, 20))
        self.assertEqual(frame.iloc[:, 0].tolist(), ["r0_c0", "r1_c0", "r2_c0"])
        label_snapshot = labels_seen[0]
        self.assertEqual(label_snapshot["rows"], disk.row_count)
        self.assertEqual(label_snapshot["columns"][0]["original_name"], "header_0")
        self.assertEqual(label_snapshot["columns"][0]["samples"],
                         [disk._values[index][0] for index in sorted({round(i * 39 / 11) for i in range(12)})])
        self.assertEqual(label_snapshot["columns"][0]["physical_type"], "unknown")
        self.assertEqual(label_snapshot["columns"][0]["statistics"], {})

    def test_experiment_is_off_by_default_and_identity_is_sample_scoped(self):
        disk = FakeDisk(columns=2)
        with patch.dict(os.environ, {"ANALYTICO_EARLY_AI": "0"}):
            self.assertIsNone(prepare_early_enrichment(
                disk, "dataset.csv", summary_generator=lambda *_: None,
                label_generator=None,
            ))

        with patch.dict(os.environ, {"ANALYTICO_EARLY_AI": "1"}):
            early = prepare_early_enrichment(
                disk, "dataset.csv", summary_generator=None,
                label_generator=lambda _: {"interpretation_proposals": {}},
            )
        self.assertIsNotNone(early)
        dataset = type("Dataset", (), {"disk": disk, "id": "id", "cache_version": "version"})()
        self.assertTrue(early.bind_after_store(dataset))
        self.assertTrue(early.matches(dataset))
        disk._values[0][0] = "changed-source"
        self.assertFalse(early.matches(dataset))

    def test_early_slots_bound_submissions_and_released_after_completion(self):
        started = threading.Event()
        release = threading.Event()
        lock = threading.Lock()
        active = 0
        peak = 0

        def call():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 4:
                    started.set()
            release.wait(2)
            with lock:
                active -= 1
            return threading.current_thread().name

        first = submit_early_requests({"summary": call, "labels": call})
        second = submit_early_requests({"summary": call, "labels": call})
        third = submit_early_requests({"summary": call, "labels": call})
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertEqual(third, {})
        try:
            self.assertTrue(started.wait(1))
            self.assertEqual(peak, 4)
        finally:
            release.set()
        names = [future.result(2) for future in (*first.values(), *second.values())]
        self.assertTrue(all(name.startswith("ai-request") for name in names))
        self.assertEqual(peak, 4)

    def test_adopted_futures_are_pending_tasks_without_duplicate_requests_or_pool_deadlock(self):
        started = threading.Event()
        release = threading.Event()
        start_lock = threading.Lock()
        count = 0

        def request(kind):
            nonlocal count
            with start_lock:
                count += 1
                if count == 4:
                    started.set()
            release.wait(2)
            if kind == "summary":
                return "prestarted summary"
            return {"interpretation_proposals": {}, "column_labels": {}}

        batches = [
            submit_early_requests({"summary": lambda: request("summary"),
                                   "labels": lambda: request("labels")})
            for _ in range(2)
        ]
        self.assertTrue(all(len(batch) == 2 for batch in batches))
        handled = []

        def no_duplicate(*_args):
            raise AssertionError("an adopted task was submitted again")

        def run(batch):
            return run_parallel_enrichment(
                no_duplicate, [], total_columns=0, publish=lambda _: None,
                is_current=lambda: True, labels=no_duplicate,
                early_futures=batch,
                early_handlers={"labels": lambda result: handled.append(result) or result},
            )

        with ThreadPoolExecutor(max_workers=2) as drivers:
            jobs = [drivers.submit(run, batch) for batch in batches]
            try:
                self.assertTrue(started.wait(1))
                self.assertTrue(all(not job.done() for job in jobs))
            finally:
                release.set()
            results = [job.result(2) for job in jobs]

        self.assertTrue(all(result["summary"] == "prestarted summary" for result in results))
        self.assertEqual(len(handled), 2)
        self.assertEqual(count, 4)

    def test_stale_version_does_not_publish_or_apply_a_completed_early_label_future(self):
        future = submit_early_requests({
            "labels": lambda: {"interpretation_proposals": {"value": {"decision": {}}}},
        })["labels"]
        self.assertIsNotNone(future.result(1))
        published = []
        applied = []

        result = run_parallel_enrichment(
            None, [], total_columns=1, publish=published.append,
            is_current=lambda: False, labels=lambda: None,
            early_futures={"labels": future},
            early_handlers={"labels": lambda value: applied.append(value) or value},
        )

        self.assertEqual(result["coverage"]["stop_reason"], "stale_version")
        self.assertEqual(published, [])
        self.assertEqual(applied, [])

    def test_failed_source_acceptance_cancels_queued_early_work(self):
        from fastapi import HTTPException
        from modules.disk_dataset import DiskDataset
        from services.csv_ingestion import ingest_csv

        fake_disk = FakeDisk(columns=2)
        early = Mock()
        started = []

        def trigger_then_fail(_source, **kwargs):
            kwargs["on_source_loaded"](fake_disk)
            raise ValueError("CSV validation failed")

        with patch.dict(os.environ, {
            "ANALYTICO_INGESTION_ENGINE": "disk",
            "ANALYTICO_EARLY_AI": "1",
            "OPENAI_API_KEY": "local-test-key",
            "COLUMN_INTERPRETER": "off",
        }), patch("services.csv_ingestion.prepare_early_enrichment", return_value=early) as prepare, \
                patch.object(DiskDataset, "from_csv", side_effect=trigger_then_fail):
            with self.assertRaises(HTTPException):
                ingest_csv(io.BytesIO(b"a,b\n1,2\n"), "bad.csv", "/test", engine="disk")
        self.assertTrue(prepare.called)
        early.cancel.assert_called_once_with()

    def test_ingestion_adopts_summary_after_store_without_reissuing_it(self):
        from modules.disk_dataset import DiskDataset
        from services.csv_ingestion import ingest_csv
        from services.enrichment_service import EnrichmentManager
        from storage import DATASETS, store_dataset

        manager = EnrichmentManager(max_workers=1, max_pending=0, max_records=2)
        summary_started = threading.Event()
        release_summary = threading.Event()
        stored = threading.Event()
        started_before_store = []
        updates_before_store = []
        summary_calls = []
        response = None
        original_from_csv = DiskDataset.from_csv

        def summary(_filename, _frame):
            summary_calls.append(True)
            started_before_store.append(not stored.is_set())
            summary_started.set()
            release_summary.wait(2)
            return "prestarted summary"

        def store_and_mark(dataset):
            result = store_dataset(dataset)
            stored.set()
            return result

        def delayed_parser(source, **kwargs):
            callback = kwargs.get("on_source_loaded")

            def start_then_validate(disk):
                callback(disk)
                if not summary_started.wait(1):
                    raise AssertionError("early summary did not start before full validation")

            kwargs["on_source_loaded"] = start_then_validate
            return original_from_csv(source, **kwargs)

        original_update = manager.update

        def update_after_store(dataset_id, version, result):
            if not stored.is_set():
                updates_before_store.append(True)
            original_update(dataset_id, version, result)

        try:
            with patch.dict(os.environ, {
                "ANALYTICO_INGESTION_ENGINE": "disk",
                "ANALYTICO_EARLY_AI": "1",
                "OPENAI_API_KEY": "local-test-key",
                "COLUMN_INTERPRETER": "off",
            }), patch("services.enrichment_service.manager", manager), \
                    patch("services.ingestion_service._generate_business_summary", side_effect=summary), \
                    patch("services.csv_ingestion.store_dataset", side_effect=store_and_mark), \
                    patch.object(manager, "update", side_effect=update_after_store), \
                    patch.object(DiskDataset, "from_csv", side_effect=delayed_parser):
                response = ingest_csv(io.BytesIO(b"group,value\na,1\nb,2\nc,3\n"),
                                      "sample.csv", "/test", engine="disk")
                self.assertTrue(stored.is_set())
                self.assertTrue(summary_started.is_set())
                self.assertTrue(started_before_store[0])
                release_summary.set()
                import time
                end = time.monotonic() + 2
                while manager.get_status(response.dataset_id)["status"] != "done":
                    self.assertLess(time.monotonic(), end)
                    time.sleep(.005)
                status = manager.get_status(response.dataset_id)
            self.assertEqual(status["summary"], "prestarted summary")
            self.assertEqual(len(summary_calls), 1)
            self.assertEqual(updates_before_store, [])
        finally:
            release_summary.set()
            manager.shutdown()
            if response is not None and response.dataset_id in DATASETS:
                DATASETS.pop(response.dataset_id).close()


if __name__ == "__main__":
    unittest.main()
