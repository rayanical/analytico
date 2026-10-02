"""Tests for bounded, version-scoped background enrichment."""

import sys
import threading
import time
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.enrichment_service import EnrichmentManager


def wait_for_status(manager: EnrichmentManager, dataset_id: str, status: str, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.get_status(dataset_id)
        if snapshot["status"] == status:
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"Timed out waiting for enrichment status {status!r}")


class EnrichmentManagerTests(unittest.TestCase):
    def test_enqueue_returns_while_slow_callback_runs_and_records_result(self):
        manager = EnrichmentManager(max_workers=1, max_pending=1, max_records=4)
        started = threading.Event()
        release = threading.Event()
        calls = []

        def slow_work():
            calls.append("called")
            started.set()
            release.wait(timeout=2)
            return {
                "summary": "A concise summary.",
                "interpretation_proposals": {"amount": {"role": "metric"}},
            }

        try:
            before = time.monotonic()
            accepted = manager.enqueue("dataset-a", "version-a", slow_work)
            elapsed = time.monotonic() - before

            self.assertLess(elapsed, 0.2)
            self.assertIn(accepted["status"], {"pending", "running"})
            self.assertTrue(started.wait(timeout=1))
            self.assertEqual(calls, ["called"])
            self.assertEqual(manager.get_status("dataset-a")["status"], "running")
        finally:
            release.set()

        completed = wait_for_status(manager, "dataset-a", "done")
        self.assertEqual(completed["progress"], 100)
        self.assertEqual(completed["summary"], "A concise summary.")
        self.assertEqual(completed["interpretation_proposals"], {"amount": {"role": "metric"}})
        manager.shutdown()

    def test_active_workers_and_queued_jobs_are_bounded(self):
        manager = EnrichmentManager(max_workers=1, max_pending=1, max_records=4)
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        active = 0
        max_active = 0
        count_lock = threading.Lock()

        def work(started=None, block=False):
            def callback():
                nonlocal active, max_active
                with count_lock:
                    active += 1
                    max_active = max(max_active, active)
                if started is not None:
                    started.set()
                if block:
                    release_first.wait(timeout=2)
                with count_lock:
                    active -= 1
                return {"summary": "done"}

            return callback

        try:
            manager.enqueue("first", "v1", work(first_started, block=True))
            self.assertTrue(first_started.wait(timeout=1))
            manager.enqueue("second", "v1", work(second_started))
            overflow = manager.enqueue("third", "v1", work())
            self.assertEqual(overflow["status"], "disabled")
            self.assertEqual(overflow["reason"], "queue_full")
            self.assertFalse(second_started.is_set())
        finally:
            release_first.set()

        wait_for_status(manager, "first", "done")
        wait_for_status(manager, "second", "done")
        self.assertTrue(second_started.is_set())
        self.assertEqual(max_active, 1)
        manager.shutdown()

    def test_cancelled_stale_version_does_not_run_or_replace_latest_result(self):
        manager = EnrichmentManager(max_workers=1, max_pending=2, max_records=5)
        first_started = threading.Event()
        release_first = threading.Event()
        stale_calls = []

        def blocker():
            first_started.set()
            release_first.wait(timeout=2)
            return {"summary": "first"}

        try:
            manager.enqueue("blocker", "v1", blocker)
            self.assertTrue(first_started.wait(timeout=1))

            manager.enqueue("replaceable", "old", lambda: stale_calls.append("old") or {"summary": "old"})
            manager.enqueue("replaceable", "new", lambda: {"summary": "new"})
        finally:
            release_first.set()

        wait_for_status(manager, "replaceable", "done")
        final = manager.get_status("replaceable", "new")
        self.assertEqual(final["summary"], "new")
        self.assertEqual(stale_calls, [])
        self.assertEqual(manager.get_status("replaceable", "old")["status"], "disabled")
        manager.shutdown()

    def test_callback_error_is_safe_and_does_not_expose_exception_text(self):
        manager = EnrichmentManager(max_workers=1, max_pending=0, max_records=2)
        try:
            manager.enqueue(
                "dataset-error",
                "version-error",
                lambda: (_ for _ in ()).throw(RuntimeError("provider secret payload")),
            )
            failed = wait_for_status(manager, "dataset-error", "error")
        finally:
            manager.shutdown()

        self.assertEqual(failed["error"], "Enrichment could not be completed.")
        self.assertNotIn("secret", failed["error"])

    def test_proposal_results_and_status_snapshots_are_copied(self):
        manager = EnrichmentManager(max_workers=1, max_pending=0, max_records=2)
        proposals = {"amount": {"role": "metric"}}
        try:
            manager.enqueue(
                "dataset-copy",
                "version-copy",
                lambda: {"summary": "Copied", "interpretation_proposals": proposals},
            )
            wait_for_status(manager, "dataset-copy", "done")
            proposals["amount"]["role"] = "identifier"
            snapshot = manager.get_status("dataset-copy")
            snapshot["interpretation_proposals"]["amount"]["role"] = "unknown"
            stored = manager.get_status("dataset-copy")
        finally:
            manager.shutdown()

        self.assertEqual(stored["interpretation_proposals"]["amount"]["role"], "metric")


if __name__ == "__main__":
    unittest.main()
