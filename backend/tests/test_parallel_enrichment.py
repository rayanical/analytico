"""Concurrency, partial results and stopping use the real shared request pool."""
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.parallel_enrichment import run_parallel_enrichment
from services.enrichment_service import EnrichmentManager


class ParallelEnrichmentTests(unittest.TestCase):
    def test_ingestion_worker_reviews_columns_beyond_old_twelve_column_cap(self):
        import os
        import pandas as pd
        from modules.column_interpretation import InterpretationResult
        from services.ingestion_service import ingest_dataframe
        from storage import DATASETS
        manager = EnrichmentManager(max_workers=1, max_pending=0, max_records=2)
        frame = pd.DataFrame({f'measure_{i}': ['1', '2', '3'] for i in range(20)})
        result = InterpretationResult(decision=None, status='uncertain', model='gpt-6-luna')
        response = None
        try:
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test', 'COLUMN_INTERPRETER': 'luna'}), \
                 patch('services.enrichment_service.manager', manager), \
                 patch('services.ingestion_service._generate_business_summary', return_value='Summary'), \
                 patch('modules.schema_interpretation.analyze_schema', return_value={'interpretation_proposals': {f'measure_{i}': {'status': 'uncertain', 'decision': None} for i in range(20)}, 'coverage': {'complete': True}}) as interpret:
                response = ingest_dataframe(frame, 'wide.csv', '/test', defer_enrichment=True)
                schema = DATASETS[response.dataset_id].column_schema
                deadline = time.monotonic() + 2
                while manager.get_status(response.dataset_id)['status'] != 'done':
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.005)
                status = manager.get_status(response.dataset_id)
                self.assertEqual(interpret.call_count, 1)
                self.assertEqual(len(status['interpretation_proposals']), 20)
                self.assertTrue(status['coverage']['complete'])
                self.assertEqual(DATASETS[response.dataset_id].column_schema, schema)
        finally:
            manager.shutdown()
            if response is not None:
                DATASETS.pop(response.dataset_id).close()

    def test_summary_overlaps_columns_and_partial_results_survive_failure(self):
        summary_started = threading.Event()
        release_summary = threading.Event()
        partial = threading.Event()
        snapshots = []

        def summary():
            summary_started.set()
            release_summary.wait(2)
            return 'Summary'

        def column():
            self.assertTrue(summary_started.wait(1))
            return {'status': 'ok'}

        def failed():
            raise RuntimeError('provider secret')

        def publish(result):
            snapshots.append(result)
            if result['interpretation_proposals']:
                partial.set()

        with ThreadPoolExecutor(max_workers=1) as driver:
            future = driver.submit(run_parallel_enrichment, summary,
                [('good', column), ('bad', failed)], total_columns=2,
                publish=publish, is_current=lambda: True)
            try:
                self.assertTrue(partial.wait(1), 'Columns waited for the summary')
                self.assertFalse(future.done())
            finally:
                release_summary.set()
            result = future.result(2)
        self.assertEqual(result['summary'], 'Summary')
        self.assertEqual(result['coverage']['completed_columns'], 2)
        self.assertEqual(result['coverage']['failed_columns'], 1)
        self.assertFalse(result['coverage']['complete'])
        self.assertNotIn('secret', str(snapshots))

    def test_two_jobs_share_at_most_four_active_requests(self):
        lock = threading.Lock()
        release = threading.Event()
        four_active = threading.Event()
        active = peak = 0

        def call():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 4:
                    four_active.set()
            release.wait(2)
            with lock:
                active -= 1
            return {'status': 'ok'}

        with ThreadPoolExecutor(max_workers=2) as drivers:
            jobs = [drivers.submit(run_parallel_enrichment, None,
                    [(str(i), call) for i in range(8)], total_columns=8,
                    publish=lambda _: None, is_current=lambda: True) for _ in range(2)]
            try:
                self.assertTrue(four_active.wait(1))
                with lock:
                    self.assertEqual(active, 4)
            finally:
                release.set()
            results = [job.result(2) for job in jobs]
        self.assertEqual(peak, 4)
        self.assertTrue(all(result['coverage']['complete'] for result in results))

    def test_time_budget_keeps_partial_results_and_does_not_wait_for_late_calls(self):
        release = threading.Event()
        snapshots = []

        def slow():
            release.wait(2)
            return {'status': 'ok'}

        start = time.monotonic()
        try:
            result = run_parallel_enrichment(None,
                [('fast', lambda: {'status': 'ok'}), ('slow', slow)],
                total_columns=2, publish=snapshots.append, is_current=lambda: True,
                timeout_seconds=.1)
            self.assertLess(time.monotonic() - start, .5)
            self.assertEqual(result['coverage']['stop_reason'], 'time_budget')
            self.assertEqual(set(result['interpretation_proposals']), {'fast'})
            count = len(snapshots)
        finally:
            release.set()
        self.assertEqual(len(snapshots), count)

    def test_stale_job_does_not_submit_calls(self):
        called = []
        result = run_parallel_enrichment(lambda: called.append('summary'),
            [('column', lambda: called.append('column'))], total_columns=1,
            publish=lambda _: None, is_current=lambda: False)
        self.assertEqual(called, [])
        self.assertEqual(result['coverage']['stop_reason'], 'stale_version')

    def test_provider_outage_stops_new_requests_and_reports_incomplete_coverage(self):
        called = []

        def unavailable():
            called.append(True)
            return {'status': 'unavailable', 'error_code': 'provider_unavailable'}

        result = run_parallel_enrichment(None,
            [(str(i), unavailable) for i in range(20)], total_columns=20,
            publish=lambda _: None, is_current=lambda: True)
        self.assertLessEqual(len(called), 4)
        self.assertFalse(result['coverage']['complete'])
        self.assertEqual(result['coverage']['stop_reason'], 'provider_unavailable')

    def test_manager_ignores_stale_partial_and_terminal_updates(self):
        manager = EnrichmentManager(max_workers=1, max_pending=1, max_records=3)
        started = threading.Event()
        release = threading.Event()

        def old():
            started.set()
            release.wait(2)
            manager.update('dataset', 'old', {'summary': 'obsolete'})
            return {'summary': 'obsolete'}

        try:
            manager.enqueue('dataset', 'old', old)
            self.assertTrue(started.wait(1))
            manager.enqueue('dataset', 'new', lambda: {'summary': 'current'})
            release.set()
            deadline = time.monotonic() + 2
            while manager.get_status('dataset')['status'] != 'done':
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.005)
            manager.update('dataset', 'new', {'summary': 'late'})
            self.assertEqual(manager.get_status('dataset')['summary'], 'current')
        finally:
            release.set()
            manager.shutdown()
