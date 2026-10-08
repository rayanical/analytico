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
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test', 'COLUMN_INTERPRETER': 'luna', 'COLUMN_ROLE_BACKEND': 'decisions', 'COLUMN_ROLE_CONTEXT': 'summary'}), \
                 patch('services.enrichment_service.manager', manager), \
                 patch('services.ingestion_service._generate_business_summary', return_value='Summary'), \
                 patch('modules.schema_decisions.analyze_schema_decisions', return_value={'interpretation_proposals': {f'measure_{i}': {'status': 'uncertain', 'decision': None} for i in range(20)}, 'coverage': {'complete': True}}) as interpret, \
                 patch('modules.schema_interpretation.analyze_schema_labels', return_value={'interpretation_proposals': {}, 'coverage': {'complete': True}}):
                response = ingest_dataframe(frame, 'wide.csv', '/test', defer_enrichment=True)
                schema = DATASETS[response.dataset_id].column_schema
                deadline = time.monotonic() + 2
                while manager.get_status(response.dataset_id)['status'] != 'done':
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.005)
                status = manager.get_status(response.dataset_id)
                self.assertEqual(interpret.call_count, 1)
                self.assertEqual(interpret.call_args.args[0]['dataset_description'], 'Summary')
                self.assertTrue(interpret.call_args.kwargs['enriched_context'])
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

    def test_schema_waits_for_summary_while_labels_continue_independently(self):
        summary_started = threading.Event()
        release_summary = threading.Event()
        labels_finished = threading.Event()
        schema_called = threading.Event()
        schema_descriptions = []

        def summary():
            summary_started.set()
            release_summary.wait(2)
            return 'Dataset summary'

        def schema(description):
            schema_descriptions.append(description)
            schema_called.set()
            return {'interpretation_proposals': {}, 'coverage': {'complete': True}}

        def labels():
            self.assertTrue(summary_started.wait(1))
            labels_finished.set()
            return {'interpretation_proposals': {}, 'column_labels': {}}

        with ThreadPoolExecutor(max_workers=1) as driver:
            future = driver.submit(run_parallel_enrichment, summary, [], total_columns=0,
                publish=lambda _: None, is_current=lambda: True,
                schema=schema, labels=labels, schema_after_summary=True)
            try:
                self.assertTrue(labels_finished.wait(1), 'Labels waited for the summary')
                self.assertFalse(schema_called.is_set(), 'Schema ran before the summary completed')
                self.assertFalse(future.done())
            finally:
                release_summary.set()
            result = future.result(2)

        self.assertEqual(result['summary'], 'Dataset summary')
        self.assertEqual(schema_descriptions, ['Dataset summary'])

    def test_summary_failure_submits_schema_without_description(self):
        descriptions = []

        def failed_summary():
            raise RuntimeError('provider details stay private')

        def schema(description):
            descriptions.append(description)
            return {'interpretation_proposals': {}, 'coverage': {'complete': True}}

        result = run_parallel_enrichment(failed_summary, [], total_columns=0,
            publish=lambda _: None, is_current=lambda: True,
            schema=schema, schema_after_summary=True)

        self.assertEqual(result['summary'], None)
        self.assertEqual(descriptions, [None])

    def test_schema_without_summary_receives_none(self):
        descriptions = []

        def schema(description):
            descriptions.append(description)
            return {'interpretation_proposals': {}, 'coverage': {'complete': True}}

        run_parallel_enrichment(None, [], total_columns=0,
            publish=lambda _: None, is_current=lambda: True,
            schema=schema, schema_after_summary=True)

        self.assertEqual(descriptions, [None])

    def test_stale_summary_does_not_submit_dependent_schema(self):
        summary_started = threading.Event()
        release_summary = threading.Event()
        current = {'value': True}
        schema_calls = []

        def summary():
            summary_started.set()
            release_summary.wait(2)
            current['value'] = False
            return 'outdated summary'

        def schema(description):
            schema_calls.append(description)
            return {'interpretation_proposals': {}, 'coverage': {'complete': True}}

        with ThreadPoolExecutor(max_workers=1) as driver:
            future = driver.submit(run_parallel_enrichment, summary, [], total_columns=0,
                publish=lambda _: None, is_current=lambda: current['value'],
                schema=schema, schema_after_summary=True)
            self.assertTrue(summary_started.wait(1))
            release_summary.set()
            result = future.result(2)

        self.assertEqual(result['coverage']['stop_reason'], 'stale_version')
        self.assertEqual(schema_calls, [])

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

    def test_schema_roles_publish_while_labels_are_pending_and_merge_independently(self):
        labels_started = threading.Event()
        release_labels = threading.Event()
        roles_published = threading.Event()
        snapshots = []

        def schema():
            return {
                'interpretation_proposals': {'sales': {
                    'status': 'ok', 'confidence': 0.97,
                    'decision': {'role': 'metric', 'evidence_strength': 'strong'},
                }},
                'coverage': {'total_columns': 1, 'selected_columns': 1,
                    'completed_columns': 1, 'failed_columns': 0,
                    'skipped_columns': 0, 'complete': True, 'stop_reason': None},
                'column_roles': {'sales': 'metric'},
                'column_labels': {'sales': 'Bundled label'},
                'semantic_revision': 11,
            }

        def labels():
            labels_started.set()
            release_labels.wait(2)
            return {
                'interpretation_proposals': {'sales': {
                    'status': 'uncertain', 'confidence': 0.12,
                    'decision': {'role': 'unknown', 'display_name': 'Net Sales',
                        'label_evidence_strength': 'strong'},
                    'label_runtime_status': 'applied',
                }},
                'column_labels': {'sales': 'Net Sales'},
                'semantic_revision': 4,
            }

        def publish(snapshot):
            snapshots.append(snapshot)
            if snapshot['column_roles'].get('sales') == 'metric':
                roles_published.set()

        with ThreadPoolExecutor(max_workers=1) as driver:
            future = driver.submit(run_parallel_enrichment, None, [], total_columns=1,
                publish=publish, is_current=lambda: True, schema=schema, labels=labels)
            try:
                self.assertTrue(labels_started.wait(1))
                self.assertTrue(roles_published.wait(1), 'Roles waited for the labels request')
                early = next(item for item in snapshots if item['column_roles'].get('sales'))
                self.assertEqual(early['interpretation_proposals']['sales']['decision']['role'], 'metric')
                self.assertEqual(early['coverage']['completed_columns'], 1)
            finally:
                release_labels.set()
            result = future.result(2)

        proposal = result['interpretation_proposals']['sales']
        self.assertEqual(proposal['status'], 'ok')
        self.assertEqual(proposal['confidence'], 0.97)
        self.assertEqual(proposal['decision']['role'], 'metric')
        self.assertEqual(proposal['decision']['evidence_strength'], 'strong')
        self.assertEqual(proposal['decision']['display_name'], 'Net Sales')
        self.assertEqual(proposal['decision']['label_evidence_strength'], 'strong')
        self.assertEqual(proposal['label_runtime_status'], 'applied')
        self.assertEqual(result['column_roles'], {'sales': 'metric'})
        self.assertEqual(result['column_labels'], {'sales': 'Net Sales'})
        self.assertEqual(result['semantic_revision'], 11)
        self.assertTrue(result['coverage']['complete'])
        revisions = [item['semantic_revision'] for item in snapshots]
        self.assertEqual(revisions, sorted(revisions))

    def test_label_failure_before_schema_does_not_replace_role_proposal(self):
        labels_failed = threading.Event()

        def labels():
            labels_failed.set()
            raise RuntimeError('private label provider details')

        def schema():
            self.assertTrue(labels_failed.wait(1))
            return {
                'interpretation_proposals': {'account_id': {
                    'status': 'ok', 'confidence': 0.91,
                    'decision': {'role': 'identifier', 'evidence_strength': 'strong'},
                }},
                'coverage': {'total_columns': 1, 'selected_columns': 1,
                    'completed_columns': 1, 'failed_columns': 0,
                    'skipped_columns': 0, 'complete': True, 'stop_reason': None},
                'column_roles': {'account_id': 'identifier'},
                'semantic_revision': 3,
            }

        result = run_parallel_enrichment(None, [], total_columns=1,
            publish=lambda _: None, is_current=lambda: True, schema=schema, labels=labels)
        self.assertEqual(result['interpretation_proposals']['account_id']['status'], 'ok')
        self.assertEqual(result['interpretation_proposals']['account_id']['confidence'], 0.91)
        self.assertEqual(result['interpretation_proposals']['account_id']['decision']['role'], 'identifier')
        self.assertNotIn(None, result['interpretation_proposals'])
        self.assertEqual(result['column_roles'], {'account_id': 'identifier'})
        self.assertTrue(result['coverage']['complete'])

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


class FailedRoleLabelMergeTests(unittest.TestCase):
    def test_successful_labels_do_not_turn_a_failed_role_into_a_partial_decision(self):
        roles = {'interpretation_proposals': {'value': {'status': 'unavailable',
                    'decision': None, 'error_code': 'provider_unavailable'}},
                 'coverage': {'complete': False}}
        labels = {'interpretation_proposals': {'value': {'status': 'uncertain',
                    'decision': {'scope': 'role_only', 'role': 'unknown',
                        'recommended_aggregation': 'unknown', 'display_name': 'Value',
                        'label_evidence_strength': 'strong'}}}, 'column_labels': {'value': 'Value'}}
        result = run_parallel_enrichment(None, [], total_columns=1,
            schema=lambda: roles, labels=lambda: labels, is_current=lambda: True, publish=lambda _: None)
        self.assertIsNone(result['interpretation_proposals']['value']['decision'])
        self.assertEqual(result['interpretation_proposals']['value']['status'], 'unavailable')
        self.assertEqual(result['column_labels'], {'value': 'Value'})
