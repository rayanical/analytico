"""Background semantic updates do not gate readiness or change source values."""
import io
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.csv_ingestion import ingest_csv
from services.enrichment_service import EnrichmentManager
from services.aggregation_service import _chart_cache_key
from models import AggregateRequest
from modules.schema_interpretation import apply_schema_roles
from storage import DATASETS


def result_for(snapshot, **options):
    return {'interpretation_proposals': {c['column']: {'status': 'ok', 'runtime_status': 'clarification',
        'decision': {'scope': 'role_only', 'role': 'metric', 'unit': 'unknown',
                     'recommended_aggregation': 'unknown', 'parsing_policy': 'preserve_source',
                     'needs_clarification': False, 'evidence_strength': 'strong', 'temporal_kind': 'none'}}
        for c in snapshot['columns'] if c['column']=='observations'},
        'coverage': {'complete': True}}


class AutomaticSchemaRoleTests(unittest.TestCase):
    def test_both_engines_are_ready_before_ai_then_publish_roles_without_reparse(self):
        for engine in ('pandas', 'disk'):
            with self.subTest(engine=engine):
                manager = EnrichmentManager(max_workers=1, max_pending=0, max_records=2)
                entered = threading.Event(); release = threading.Event(); response = None
                def analyze(snapshot):
                    entered.set(); release.wait(2)
                    return result_for(snapshot)
                try:
                    with patch.dict(os.environ, {'OPENAI_API_KEY':'offline-test','COLUMN_INTERPRETER':'off', 'COLUMN_ROLE_BACKEND':'responses',
                                                 'ANALYTICO_INGESTION_ENGINE':engine}), \
                         patch('services.enrichment_service.manager',manager), \
                         patch('services.ingestion_service._generate_business_summary',return_value=None), \
                         patch('modules.schema_interpretation.analyze_schema',side_effect=analyze):
                        response = ingest_csv(io.BytesIO(b'group,observations,account_code\nA,3,001\nB,8,002\nA,4,003\n'),
                                              'real-shape.csv','/test',ai_column_analysis=True)
                        dataset = DATASETS[response.dataset_id]
                        self.assertTrue(entered.wait(1))
                        self.assertEqual(dataset.column_types['observations'],'unknown')
                        source_before = dataset.sample_frame(3).to_dict()
                        version = dataset.cache_version
                        request = AggregateRequest(dataset_id=dataset.id,x_axis_key='group',y_axis_keys=['observations'],aggregation='sum')
                        old_key = _chart_cache_key(dataset,request)
                        release.set()
                        deadline=time.monotonic()+2
                        while manager.get_status(dataset.id)['status']!='done':
                            self.assertLess(time.monotonic(),deadline); time.sleep(.005)
                        status=manager.get_status(dataset.id)
                        self.assertEqual(dataset.column_types['observations'],'metric')
                        self.assertEqual(dataset.column_types['account_code'],'identifier')
                        self.assertEqual(dataset.sample_frame(3).to_dict(),source_before)
                        self.assertEqual(dataset.cache_version,version)
                        self.assertNotEqual(_chart_cache_key(dataset,request),old_key)
                        self.assertEqual(status['column_roles']['observations'],'metric')
                        self.assertEqual(status['interpretation_proposals']['observations']['runtime_status'],'applied')
                        self.assertEqual(next(c for c in dataset.column_schema if c['column']=='observations')['aggregation'],None)
                finally:
                    release.set(); manager.shutdown()
                    if response: DATASETS.pop(response.dataset_id).close()

    def test_labels_are_independent_of_roles_preserve_keys_and_survive_review(self):
        from services.schema_review import apply_schema, get_schema
        for engine in ('pandas', 'disk'):
            with self.subTest(engine=engine), patch.dict(os.environ, {'ANALYTICO_INGESTION_ENGINE':engine}):
                response=ingest_csv(io.BytesIO(b'account_id,observations,opaque\n001,3,x\n002,8,y\n003,4,z\n'),
                                    'labels.csv','/test',enqueue_enrichment=False)
                dataset=DATASETS[response.dataset_id]
                replacement=None
                try:
                    version=dataset.cache_version
                    values=dataset.sample_frame(3).to_dict()
                    role=dataset.column_types['account_id']
                    proposals={name:{'status':'uncertain','decision':{'scope':'role_only','role':'unknown',
                        'needs_clarification':True,'evidence_strength':'unknown','display_name':label,
                        'label_evidence_strength':'strong'}}
                        for name,label in [('account_id','Account ID'),('observations','Observations'),('opaque','Opaque')]}
                    result=apply_schema_roles(dataset.id,version,{'interpretation_proposals':proposals})
                    self.assertEqual(result['column_labels']['account_id'],'Account ID')
                    self.assertEqual(dataset.column_types['account_id'],role)
                    self.assertEqual(dataset.sample_frame(3).to_dict(),values)
                    self.assertEqual(dataset.cache_version,version)
                    self.assertEqual(get_schema(dataset.id)['columns'][0]['display_name'],'Account ID')
                    original_handle=dataset
                    dataset.column_types['observations']='metric'
                    next(c for c in dataset.column_schema if c['column']=='observations')['role']='metric'
                    with patch('services.schema_review.ingest_csv', side_effect=AssertionError('Label edits must not re-ingest')):
                        replacement=apply_schema(dataset.id,version,[{'column':'observations','display_name':'Observed count'}])
                    self.assertIs(DATASETS[dataset.id],original_handle)
                    self.assertEqual(original_handle.column_types['observations'],'metric')
                    self.assertNotEqual(replacement.version,version)
                    from fastapi import HTTPException
                    with self.assertRaises(HTTPException) as stale:
                        apply_schema(dataset.id,version,[{'column':'observations','display_name':'Stale'}])
                    self.assertEqual(stale.exception.status_code,409)
                    current=DATASETS[dataset.id]
                    self.assertEqual(next(c for c in current.column_schema if c['column']=='observations')['display_name'],'Observed count')
                    self.assertEqual(next(c for c in current.column_schema if c['column']=='account_id')['display_name'],'Account ID')
                    apply_schema_roles(current.id,current.cache_version,{'interpretation_proposals':proposals})
                    self.assertEqual(next(c for c in current.column_schema if c['column']=='observations')['display_name'],'Observed count')
                finally:
                    DATASETS.pop(response.dataset_id).close()

    def test_label_collisions_and_unsafe_labels_fail_without_changing_values(self):
        response=ingest_csv(io.BytesIO(b'first,second,third,fourth\n3,8,4,5\n4,9,5,6\n'),
                            'labels.csv','/test',enqueue_enrichment=False)
        dataset=DATASETS[response.dataset_id]
        try:
            proposals={name:{'status':'ok','decision':{'scope':'role_only','display_name':label,
                'label_evidence_strength':'strong'}} for name,label in [('first','Shared'),('second','shared'),
                    ('third','fourth'),('fourth','bad\nlabel')]}
            result=apply_schema_roles(dataset.id,dataset.cache_version,{'interpretation_proposals':proposals})
            self.assertEqual(result['column_labels'],{})
            self.assertTrue(all(p['label_runtime_status']=='clarification' for p in result['interpretation_proposals'].values()))
        finally:
            DATASETS.pop(response.dataset_id).close()

    def test_decisions_roles_publish_while_independent_labels_are_pending(self):
        for engine in ('pandas', 'disk'):
            with self.subTest(engine=engine):
                manager = EnrichmentManager(max_workers=1, max_pending=0, max_records=2)
                labels_started = threading.Event()
                labels_release = threading.Event()
                response = None
                def labels(snapshot):
                    labels_started.set()
                    labels_release.wait(3)
                    return {'interpretation_proposals': {'observations': {'status': 'uncertain',
                        'decision': {'scope': 'role_only', 'role': 'unknown',
                            'evidence_strength': 'unknown', 'needs_clarification': True,
                            'display_name': 'Observation count', 'label_evidence_strength': 'strong'}}},
                        'coverage': {'complete': True}}
                try:
                    with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test',
                            'COLUMN_ROLE_BACKEND': 'decisions', 'COLUMN_INTERPRETER': 'off',
                            'COLUMN_ROLE_CONTEXT': 'original',
                            'ANALYTICO_INGESTION_ENGINE': engine}), \
                         patch('services.enrichment_service.manager', manager), \
                         patch('services.ingestion_service._generate_business_summary', return_value=None), \
                         patch('modules.schema_decisions.analyze_schema_decisions', side_effect=result_for) as role_request, \
                         patch('modules.schema_interpretation.analyze_schema_labels', side_effect=labels), \
                         patch('modules.schema_interpretation.analyze_schema', side_effect=AssertionError('No legacy role request')):
                        response = ingest_csv(io.BytesIO(b'group,observations,account_code\nA,3,001\nB,8,002\nA,4,003\n'),
                            'roles.csv', '/test', ai_column_analysis=True)
                        dataset = DATASETS[response.dataset_id]
                        original = dataset.sample_frame(3).to_dict()
                        version = dataset.cache_version
                        self.assertTrue(labels_started.wait(1))
                        deadline = time.monotonic() + 2
                        while manager.get_status(dataset.id).get('column_roles', {}).get('observations') != 'metric':
                            self.assertLess(time.monotonic(), deadline)
                            time.sleep(.005)
                        self.assertNotEqual(manager.get_status(dataset.id)['status'], 'done')
                        self.assertEqual(dataset.column_types['observations'], 'metric')
                        self.assertFalse(role_request.call_args.kwargs['enriched_context'])
                        labels_release.set()
                        while manager.get_status(dataset.id)['status'] != 'done':
                            self.assertLess(time.monotonic(), deadline)
                            time.sleep(.005)
                        status = manager.get_status(dataset.id)
                        self.assertEqual(status['column_roles']['observations'], 'metric')
                        self.assertEqual(status['column_labels']['observations'], 'Observation count')
                        self.assertEqual(status['interpretation_proposals']['observations']['decision']['role'], 'metric')
                        self.assertEqual(dataset.column_types['account_code'], 'identifier')
                        self.assertEqual(dataset.sample_frame(3).to_dict(), original)
                        self.assertEqual(dataset.cache_version, version)
                finally:
                    labels_release.set()
                    manager.shutdown()
                    if response:
                        DATASETS.pop(response.dataset_id).close()

    def test_stale_job_cannot_change_replaced_dataset(self):
        result={'interpretation_proposals':{}}
        self.assertIs(apply_schema_roles('missing','old',result),result)

    def test_application_rechecks_protected_and_unsupported_roles(self):
        from types import SimpleNamespace
        from modules.column_statistics import ColumnStatistics
        from modules.column_usage import infer_column_usage
        stats = ColumnStatistics(unique_count=3, sample_values=('1','2','3'), numeric_year=False,
                                 row_position_sequence=False, has_fractional_values=False,
                                 numeric_min=0, numeric_max=6)
        entries = [
            {'column':'location_id','role':'identifier','usage':infer_column_usage('location_id','number',stats).metadata(),'provenance':'inference','aggregation':None},
            {'column':'weekday','role':'categorical','usage':infer_column_usage('weekday','number',stats).metadata(),'provenance':'inference','aggregation':None},
            {'column':'reviewed','role':'categorical','usage':infer_column_usage('reviewed','number',stats,role_override='categorical').metadata(),'provenance':'override','aggregation':None},
        ]
        dataset=SimpleNamespace(cache_version='current',column_schema=entries,
            column_types={e['column']:e['role'] for e in entries}, column_stats={e['column']:stats for e in entries})
        DATASETS['guard-test']=dataset
        try:
            result={'interpretation_proposals':{e['column']:{'status':'ok','decision':{'scope':'role_only','role':role,
                'temporal_kind':'weekday','needs_clarification':False,'evidence_strength':'strong'}}
                for e,role in zip(entries,('categorical','temporal','metric'))}}
            apply_schema_roles('guard-test','current',result)
            self.assertEqual(dataset.column_types,{'location_id':'identifier','weekday':'categorical','reviewed':'categorical'})
            self.assertFalse(hasattr(dataset,'semantic_revision'))
            self.assertIs(apply_schema_roles('guard-test','obsolete',result),result)
        finally:
            DATASETS.pop('guard-test')
