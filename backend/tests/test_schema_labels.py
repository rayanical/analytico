"""Independent readable labels preserve role semantics and fail closed."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from modules.schema_interpretation import analyze_schema_labels, apply_schema_roles
from schema_label_readability import run_live, score_label
from test_schema_interpretation import column


class SchemaLabelTests(unittest.TestCase):
    def run_labels(self, predictions, status='completed'):
        client = MagicMock()
        client.__enter__.return_value.post.return_value.json.return_value = {
            'status': status, 'output': [{'content': [{'type': 'output_text',
                'text': json.dumps({'columns': predictions})}]}]}
        snapshot = {'rows': 10, 'columns': [column('observations')]}
        original = json.dumps(snapshot, sort_keys=True)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test'}), \
             patch('modules.schema_interpretation.httpx.Client', return_value=client):
            result = analyze_schema_labels(snapshot)
        self.assertEqual(original, json.dumps(snapshot, sort_keys=True))
        return result, client.__enter__.return_value.post

    def test_labels_do_not_propose_roles_or_conversions(self):
        result, post = self.run_labels([{'index': 0, 'display_name': 'Observation count',
                                       'label_evidence_strength': 'strong'}])
        decision = result['interpretation_proposals']['observations']['decision']
        self.assertEqual(decision['role'], 'unknown')
        self.assertTrue(decision['needs_clarification'])
        self.assertEqual(decision['display_name'], 'Observation count')
        request = post.call_args.kwargs['json']
        self.assertEqual(request['model'], 'gpt-6-luna')
        self.assertNotIn('role', request['text']['format']['schema']['$defs']['LabelProposal']['properties'])
        prompt = request['input'][0]['content']
        self.assertIn('pickup_datetime', prompt)
        self.assertIn('Pickup Date and Time', prompt)
        self.assertIn('already readable', prompt)

    def test_unchanged_technical_header_is_not_published_as_an_ai_label(self):
        from storage import DATASETS

        dataset_id = 'schema-label-unchanged-header'
        dataset = SimpleNamespace(
            cache_version='current',
            column_schema=[{'column': 'pickup_datetime', 'original_name': 'pickup_datetime',
                'display_name': None, 'display_name_provenance': None,
                'usage': {'role': 'unknown', 'evidence': 'ambiguous', 'physical_type': 'text'}}],
            column_types={'pickup_datetime': 'unknown'},
            column_stats={},
            df=SimpleNamespace(attrs={}),
        )
        proposal = {'status': 'uncertain', 'decision': {'scope': 'role_only',
            'role': 'unknown', 'needs_clarification': True, 'evidence_strength': 'unknown',
            'display_name': 'pickup_datetime', 'label_evidence_strength': 'strong'}}
        with patch.dict(DATASETS, {dataset_id: dataset}), \
             patch('modules.intelligence.dataframe_column_usages', return_value={}):
            result = apply_schema_roles(dataset_id, 'current', {
                'interpretation_proposals': {'pickup_datetime': proposal}})

        self.assertEqual(result['column_labels'], {})
        self.assertIsNone(dataset.column_schema[0]['display_name'])
        self.assertEqual(result['interpretation_proposals']['pickup_datetime']['label_runtime_status'], 'clarification')

    def test_missing_duplicate_invented_or_incomplete_labels_fail_closed(self):
        label = {'index': 0, 'display_name': 'Observations', 'label_evidence_strength': 'strong'}
        for predictions, status in [([], 'completed'), ([label, label], 'completed'),
                ([{**label, 'index': 1}], 'completed'), ([label], 'incomplete'),
                ([{**label, 'display_name': 9}], 'completed')]:
            with self.subTest(predictions=predictions, status=status):
                result, _ = self.run_labels(predictions, status)
                self.assertFalse(result['coverage']['complete'])
                self.assertIsNone(result['interpretation_proposals']['observations']['decision'])

    def test_missing_credentials_skips_remote_request(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}), \
             patch('modules.schema_interpretation.httpx.Client') as client:
            result = analyze_schema_labels({'rows': 1, 'columns': [column('observations')]})
        client.assert_not_called()
        self.assertEqual(result['coverage']['stop_reason'], 'not_configured')


class SchemaLabelBenchmarkTests(unittest.TestCase):
    def test_scorer_separates_cosmetic_improvements_echoes_and_invention_signals(self):
        reformatted = score_label('trip_distance', 'Trip Distance', False, {'trip', 'distance'}, set())
        echoed = score_label('GrossAmount', 'GrossAmount', False, {'grossamount'}, set())
        opaque = score_label('var_17', None, True, {'var', '17'}, set())
        renamed_opaque = score_label('var_17', 'Variable 17', True, {'var', '17'}, set())
        unsupported_currency = score_label('trip_distance', '$Revenue', False, {'trip', 'distance'}, set())
        supported_currency = score_label('trip_distance', '$Revenue', False, {'trip', 'distance'}, {'$'})

        self.assertEqual(reformatted['cosmetic_outcome'], 'readable_label')
        self.assertTrue(reformatted['formatting_only_label'])
        self.assertEqual(echoed['cosmetic_outcome'], 'technical_name_echo')
        self.assertFalse(echoed['cosmetic_readable'])
        self.assertTrue(opaque['opaque_retained'])
        self.assertTrue(renamed_opaque['opaque_renamed_review'])
        self.assertEqual(unsupported_currency['unsupported_unit_or_currency_symbols'], ['$'])
        self.assertEqual(supported_currency['unsupported_unit_or_currency_symbols'], [])

    def test_run_live_keeps_eight_datasets_and_98_columns_once_per_arm(self):
        import modules.schema_interpretation as schema_interpretation

        counts = [12] * 7 + [14]
        datasets = []
        snapshots = []
        for dataset_index, count in enumerate(counts):
            dataset_id = f'fixture_{dataset_index}'
            datasets.append({
                'id': dataset_id, 'source': 'public test fixture',
                'resolved_path': f'/fixtures/{dataset_id}.csv',
                'observed_sha256': f'{dataset_index:064x}', 'opaque_columns': [],
            })
            columns = []
            for column_index in range(count):
                header = f'fixture{dataset_index}Field{column_index}'
                columns.append({
                    'column': header, 'original_name': header, 'samples': ['sample'],
                })
            snapshots.append({'rows': 10, 'columns': columns})

        def mock_analysis(snapshot):
            proposals = {}
            for item in snapshot['columns']:
                proposals[item['column']] = {
                    'status': 'uncertain', 'label_status': 'done', 'label_error_code': None,
                    'decision': {
                        'display_name': f"Readable {item['original_name']}",
                        'label_evidence_strength': 'strong',
                    },
                }
            return {
                'interpretation_proposals': proposals,
                'coverage': {'complete': True, 'stop_reason': None},
            }

        with patch.object(schema_interpretation, 'analyze_schema_labels', side_effect=mock_analysis) as analyze:
            result = run_live(datasets, snapshots)

        self.assertEqual(analyze.call_count, 16)
        self.assertEqual(len(result['datasets']), 8)
        self.assertEqual(len(result['columns']), 98)
        identities = [(item['dataset'], item['source_header']) for item in result['columns']]
        self.assertEqual(len(set(identities)), 98)
        self.assertEqual(len({item['source_header'] for item in result['columns']}), 98)
        for arm in ('before', 'after'):
            self.assertEqual(result['overall'][arm]['columns'], 98)
            self.assertEqual(result['overall'][arm]['labels_proposed'], 98)
            self.assertEqual(result['overall'][arm]['provider_failures'], 0)
        self.assertEqual(sum(item['columns'] for item in result['datasets']), 98)
        self.assertTrue(all(item['before']['columns'] == item['after']['columns'] for item in result['datasets']))
