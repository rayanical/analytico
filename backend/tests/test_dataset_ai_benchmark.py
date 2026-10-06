"""Benchmark routing and source sampling stay bounded and provider-independent."""
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.dataset_ai_latency import all_inputs, needs_ai, needs_ai_cautious
from services.csv_ingestion import ingest_csv
from storage import DATASETS, get_dataset


class DatasetAiBenchmarkTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'OPENAI_API_KEY': '', 'COLUMN_INTERPRETER': 'off'})
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for dataset in list(DATASETS.values()):
            dataset.close()
        DATASETS.clear()

    def test_extended_disk_inputs_match_production_bounds_and_original_values(self):
        header = ['Code', *[f'value_{index}' for index in range(19)]]
        source = ','.join(header) + '\n' + ''.join(','.join([f'{row:05d}', *[str(row + index) for index in range(19)]]) + '\n' for row in range(30))
        uploaded = ingest_csv(io.BytesIO(source.encode()), 'wide.csv', '/benchmark-test',
                              engine='disk', defer_enrichment=True, enqueue_enrichment=False)
        dataset = get_dataset(uploaded.dataset_id)
        extended = all_inputs(dataset, None)
        original = dataset.disk.interpretation_inputs(limit=12)
        self.assertEqual(extended[:12], original)
        self.assertEqual(extended, dataset.disk.interpretation_inputs(limit=256))
        self.assertEqual(len(extended), 20)
        for _, payload in extended:
            self.assertEqual(len(payload['values']), 12)
            self.assertLessEqual(len(payload['context']['other_column_names']), 20)
        self.assertEqual(extended[0][1]['values'][-1], '00029')

    def test_triage_is_explicit_about_skipping_all_missing_and_known_numeric_roles(self):
        self.assertFalse(needs_ai('unknown', 0))
        self.assertFalse(needs_ai('metric', 100))
        self.assertFalse(needs_ai('temporal', 100))
        for role in ['identifier', 'categorical', 'unknown', None]:
            self.assertTrue(needs_ai(role, 100))

    def test_cautious_gate_routes_numeric_ambiguity_without_dataset_names(self):
        for values, unique in [(['1', '2'], 2), (['10001', '10002'], 10000),
                               (['00123', '456'], 1001), (['1,23', '4.56'], 1001),
                               (['$10', '$20'], 1001), (['Infinity'], 1001)]:
            self.assertTrue(needs_ai_cautious('metric', 10000, unique, values))
        self.assertFalse(needs_ai_cautious('metric', 10000, 1001, ['1.23', '4.56']))
        self.assertFalse(needs_ai_cautious('unknown', 0, 0, [None]))
        self.assertTrue(needs_ai_cautious('metric', 10000, None, ['1']))
