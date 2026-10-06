"""Usage decisions stay stable across cardinality, storage engines and reviews."""
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.column_statistics import compute_column_statistics
from modules.column_usage import infer_column_usage
from modules.intelligence import detect_semantic_type
from modules.import_policy import ColumnOverride
from models import AggregateRequest
from services.aggregation_service import run_aggregate
from services.csv_ingestion import ingest_csv
from storage import DATASETS


class ColumnUsageTests(unittest.TestCase):
    def test_numeric_cardinality_does_not_define_role_or_remove_operations(self):
        for name, expected in [('passenger_count', 'metric'), ('opaque', 'unknown')]:
            for values in ([1, 2, 3], [1, 2, 3] * 100, list(range(1, 100))):
                values = list(reversed(values))
                frame = pd.DataFrame({name: values})
                stat = compute_column_statistics(frame)[name]
                usage = infer_column_usage(name, 'number', stat)
                self.assertEqual(usage.role, expected)
                self.assertTrue(usage.can_group)
                self.assertTrue(usage.can_measure)
                self.assertEqual(usage.automatic_measure, expected == 'metric')

    def test_many_text_values_are_not_identifier_evidence(self):
        for size in (3, 49, 50, 1000):
            frame = pd.DataFrame({'description': [f'Product {i}' for i in range(size)]})
            self.assertEqual(detect_semantic_type(frame, 'description'), 'categorical')
        ids = pd.DataFrame({'customer_id': ['001', '002', '003']})
        self.assertEqual(detect_semantic_type(ids, 'customer_id'), 'identifier')

    def test_years_are_not_limited_by_an_arbitrary_future_cutoff(self):
        frame = pd.DataFrame({'year': [2300, 2301, 2302], 'age_year': [2300, 2301, 2302]})
        self.assertEqual(detect_semantic_type(frame, 'year'), 'temporal')
        # Units in a quantity name do not make the quantity a calendar axis.
        self.assertNotEqual(detect_semantic_type(frame, 'age_year'), 'temporal')

    def test_calendar_domain_requires_every_value_and_explicit_aggregation_wins(self):
        valid = compute_column_statistics(pd.DataFrame({'hour': [0, 1, 2, 23]}))['hour']
        invalid = compute_column_statistics(pd.DataFrame({'hour': [0, 1, 2, 50]}))['hour']
        self.assertEqual(infer_column_usage('hour', 'number', valid).role, 'categorical')
        self.assertEqual(infer_column_usage('hour', 'number', invalid).role, 'unknown')
        usage = infer_column_usage('hour', 'number', valid, aggregation_override='sum')
        self.assertTrue(usage.automatic_measure)
        self.assertEqual(usage.preferred_aggregation, 'sum')
        self.assertFalse(infer_column_usage('hour', 'number', valid, aggregation_override='none').automatic_measure)

    def test_engines_share_roles_formats_default_and_explicit_numeric_access(self):
        source = 'group,opaque,passenger_count,description,total,rate,customer_id\n' + ''.join(
            f'G{i%3},{(i%4)+10},{(i%2)+1},Product {i},{i%7},{(i%3)+1},00{i}\n' for i in range(80))
        snapshots = []
        with patch.dict(os.environ, {'OPENAI_API_KEY': '', 'COLUMN_INTERPRETER': 'off'}):
            for engine in ('pandas', 'disk'):
                response = ingest_csv(io.BytesIO(source.encode()), 'usage.csv', '/test', engine=engine, enqueue_enrichment=False)
                dataset = DATASETS[response.dataset_id]
                try:
                    self.assertEqual(dataset.column_types['opaque'], 'unknown')
                    self.assertEqual(dataset.column_formats['total'], 'number')
                    self.assertEqual(dataset.column_formats['rate'], 'number')
                    self.assertEqual(dataset.column_formats['customer_id'], 'identifier')
                    chart = run_aggregate(AggregateRequest(dataset_id=dataset.id, x_axis_key='group', y_axis_keys=['opaque'],
                        aggregation='sum', limit=0, group_others=False))
                    self.assertEqual(sum(r['opaque'] for r in chart.data), 920)
                    self.assertEqual(dataset.sample_frame()['customer_id'].iloc[0], '000')
                    snapshots.append((dataset.column_types, dataset.column_formats, response.default_chart.model_dump(),
                        [item['usage'] for item in response.column_schema]))
                finally:
                    DATASETS.pop(dataset.id).close()
        self.assertEqual(snapshots[0], snapshots[1])

    def test_frequency_fallback_and_reviewed_aggregation(self):
        source = b'group,opaque\nA,10\nA,11\nB,10\nB,11\n'
        with patch.dict(os.environ, {'OPENAI_API_KEY': '', 'COLUMN_INTERPRETER': 'off'}):
            for engine in ('pandas', 'disk'):
                for overrides in ([], [ColumnOverride(column='opaque', parse_as='number', role='metric', aggregation='sum')]):
                    response = ingest_csv(io.BytesIO(source), 'frequency.csv', '/test', engine=engine,
                        enqueue_enrichment=False, column_overrides=overrides)
                    dataset = DATASETS[response.dataset_id]
                    try:
                        plan = response.default_chart
                        self.assertEqual(plan.aggregation, 'sum' if overrides else 'count')
                        chart = run_aggregate(AggregateRequest(dataset_id=dataset.id, x_axis_key=plan.x_axis_key,
                            y_axis_keys=plan.y_axis_keys, aggregation=plan.aggregation, limit=0, group_others=False))
                        self.assertEqual(sum(row[plan.y_axis_keys[0]] for row in chart.data), 42 if overrides else 4)
                    finally:
                        DATASETS.pop(dataset.id).close()

if __name__ == '__main__':
    unittest.main()
