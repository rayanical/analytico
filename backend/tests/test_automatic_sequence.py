"""Positional sequences stay queryable without becoming automatic measures."""
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.column_statistics import is_row_position_sequence
from modules.import_policy import ColumnOverride
from models import AggregateRequest
from services.aggregation_service import run_aggregate
from services.csv_ingestion import ingest_csv
from storage import DATASETS


class AutomaticSequenceTests(unittest.TestCase):
    def test_sequence_requires_full_integer_positional_evidence(self):
        self.assertTrue(is_row_position_sequence(pd.Series(range(40))))
        self.assertTrue(is_row_position_sequence(pd.Series(range(1, 41))))
        self.assertFalse(is_row_position_sequence(pd.Series([0, 1, 3, 2])))
        self.assertFalse(is_row_position_sequence(pd.Series([0, 1, 1, 3])))
        self.assertFalse(is_row_position_sequence(pd.Series([0., .5, 2.])))
        self.assertFalse(is_row_position_sequence(pd.Series([0, 1, None, 3], dtype='Int64')))

    def test_both_engines_omit_index_from_automatic_insights_but_allow_explicit_query(self):
        source = 'position,group,amount\n' + ''.join(f'{i},G{i % 2},{(i+2)**2}\n' for i in range(40))
        for engine in ('pandas', 'disk'):
            for reviewed in (False, True):
                with self.subTest(engine=engine, reviewed=reviewed), patch.dict(os.environ, {'OPENAI_API_KEY': '', 'COLUMN_INTERPRETER': 'off'}):
                    response = ingest_csv(io.BytesIO(source.encode()), 'sequence.csv', '/test', engine=engine,
                        enqueue_enrichment=False, column_overrides=[ColumnOverride(column='position', role='metric')] if reviewed else [])
                    dataset = DATASETS[response.dataset_id]
                    try:
                        expected = 'position' if reviewed else 'amount'
                        self.assertEqual(response.default_chart.y_axis_keys, [expected])
                        self.assertEqual(response.profile.top_metrics[0].name, expected)
                        chart = run_aggregate(AggregateRequest(dataset_id=dataset.id, x_axis_key='group',
                                              y_axis_keys=['position'], aggregation='sum', limit=0, group_others=False))
                        self.assertEqual(sum(row['position'] for row in chart.data), sum(range(40)))
                        if hasattr(dataset, 'disk'):
                            payload = dataset.disk.interpretation_inputs(256)[0][1]
                            self.assertTrue(payload['context']['row_position_sequence'])
                            self.assertEqual(payload['context']['column_unique_count'], 40)
                    finally:
                        DATASETS.pop(dataset.id).close()

if __name__ == '__main__':
    unittest.main()
