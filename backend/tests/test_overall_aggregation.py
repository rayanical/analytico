"""Overall charts must compute full filtered data and retain null/count semantics."""
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError
from models import AggregateRequest, FilterConfig, QueryRequest, DrillDownRequest
from modules.disk_dataset import DiskDataset
from services.aggregation_service import run_aggregate
from services.query_service import run_query


class OverallTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'amount': [10., None, -5., 25.]})
        self.memory = SimpleNamespace(id='fixture', df=self.frame, column_types={'category': 'categorical', 'amount': 'metric'}, column_formats={})
        self.disk = DiskDataset.from_csv(io.BytesIO(b'category,amount\nA,10\nA,\nB,-5\nB,25\n'), filename='overall.csv')
        self.addCleanup(self.disk.close)
        self.adapters = [self.memory, SimpleNamespace(id='disk', disk=self.disk, column_types=self.disk.column_types, column_formats=self.disk.column_formats, cache_version='overall-test', semantic_revision=0)]

    def test_both_engines_all_operations_and_counts(self):
        for ds in self.adapters:
            for operation, expected in [('sum', 30), ('mean', 10), ('median', 10), ('min', -5), ('max', 25), ('count', 3)]:
                with self.subTest(engine=ds.id, operation=operation), patch('services.aggregation_service.get_dataset', return_value=ds):
                    result = run_aggregate(AggregateRequest(dataset_id=ds.id, x_axis_key=None, y_axis_keys=['amount'], aggregation=operation, limit=1))
                    self.assertEqual(result.data[0]['amount'], expected)
                    self.assertEqual(result.aggregation_scope, 'overall')
                    self.assertEqual(result.chart_type, 'bar')
                    self.assertIsNone(result.source_x_axis_key)
            with patch('services.aggregation_service.get_dataset', return_value=ds):
                result = run_aggregate(AggregateRequest(dataset_id=ds.id, y_axis_keys=[], aggregation='count'))
                self.assertTrue(result.count_rows)
                self.assertEqual(result.data[0][result.y_axis_keys[0]], 4)

    def test_filters_all_null_and_empty_result(self):
        for ds in self.adapters:
            with self.subTest(engine=ds.id), patch('services.aggregation_service.get_dataset', return_value=ds):
                filters = [FilterConfig(column='category', values=['A'])]
                result = run_aggregate(AggregateRequest(dataset_id=ds.id, y_axis_keys=['amount'], aggregation='mean', filters=filters))
                self.assertEqual(result.data[0]['amount'], 10)
                self.assertEqual(result.filters, filters)
                self.assertTrue(result.applied_filters)
                with self.assertRaises(HTTPException):
                    run_aggregate(AggregateRequest(dataset_id=ds.id, y_axis_keys=['amount'], filters=[FilterConfig(column='category', values=['missing'])]))
        null_ds = SimpleNamespace(id='null', df=pd.DataFrame({'amount': [float('nan')]}), column_types={'amount': 'metric'})
        with patch('services.aggregation_service.get_dataset', return_value=null_ds):
            self.assertIsNone(run_aggregate(AggregateRequest(dataset_id='null', y_axis_keys=['amount'])).data[0]['amount'])
        all_null_disk = DiskDataset.from_csv(io.BytesIO(b'category,amount\nA,\n'), filename='null.csv')
        self.addCleanup(all_null_disk.close)
        self.assertEqual(all_null_disk.aggregate(AggregateRequest(dataset_id=all_null_disk.id, y_axis_keys=[], aggregation="count")).data[0]["__row_count__"], 1)

    def test_plan_overall_and_row_count_use_engine_and_filters(self):
        for measures, aggregation, expected in [(['amount'], 'sum', 20), ([], 'count', 2)]:
            plan = dict(kind='chart', x_axis_key=None, y_axis_keys=measures, aggregation=aggregation, chart_type='line', title='Overall result', x_axis_label=None, y_axis_label=None, filters=[], clarification=None)
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(plan), tool_calls=None))]))))
            with patch('services.query_service.get_dataset', return_value=self.memory), patch('services.aggregation_service.get_dataset', return_value=self.memory), patch('services.query_service.get_openai_client', return_value=client):
                result = run_query(QueryRequest(dataset_id='fixture', user_prompt='overall', filters=[FilterConfig(column='category', values=['B'])]))
            self.assertEqual(result.data[0][result.y_axis_keys[0]], expected)
            self.assertEqual(result.chart_type, 'bar')
            self.assertEqual(self.disk.drilldown(DrillDownRequest(dataset_id='disk', filters=result.filters))['total_rows'], 2)

    def test_invalid_requests_and_non_numeric_measures(self):
        for kwargs in [dict(y_axis_keys=[], aggregation='sum'), dict(x_axis_key='category', y_axis_keys=[], aggregation='count'), dict(y_axis_keys=['amount'], time_bucket='month')]:
            with self.assertRaises(ValidationError):
                AggregateRequest(dataset_id='fixture', **kwargs)
        for ds in self.adapters:
            with patch('services.aggregation_service.get_dataset', return_value=ds), self.assertRaises(HTTPException):
                run_aggregate(AggregateRequest(dataset_id=ds.id, y_axis_keys=['category'], aggregation='mean'))

    def test_synthetic_axis_never_overwrites_source_measure(self):
        ds = SimpleNamespace(id='collision', df=pd.DataFrame({'__overall__': [2, 3]}), column_types={'__overall__': 'metric'})
        with patch('services.aggregation_service.get_dataset', return_value=ds):
            result = run_aggregate(AggregateRequest(dataset_id='collision', y_axis_keys=['__overall__']))
        self.assertNotEqual(result.x_axis_key, '__overall__')
        self.assertEqual(result.data[0]['__overall__'], 5)

    def test_filtered_all_missing_numeric_values_stay_missing_in_both_engines(self):
        disk = DiskDataset.from_csv(io.BytesIO(b'category,amount\nA,\nB,10\n'), filename='missing.csv')
        self.addCleanup(disk.close)
        memory = SimpleNamespace(id='missing', df=pd.DataFrame({'category': ['A', 'B'], 'amount': [float('nan'), 10.]}), column_types={'amount': 'metric'})
        adapters = [memory, SimpleNamespace(id=disk.id, disk=disk, cache_version=disk.cache_version, column_types=disk.column_types)]
        for ds in adapters:
            with patch('services.aggregation_service.get_dataset', return_value=ds):
                for operation in ['sum', 'mean', 'median', 'min', 'max', 'count']:
                    result = run_aggregate(AggregateRequest(dataset_id=ds.id, y_axis_keys=['amount'], aggregation=operation, filters=[FilterConfig(column='category', values=['A'])]))
                    self.assertEqual(result.data[0]['amount'], 0 if operation == 'count' else None)
                with self.assertRaises(HTTPException):
                    run_aggregate(AggregateRequest(dataset_id=ds.id, y_axis_keys=['missing_column']))
