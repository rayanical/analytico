"""Schema suggestions remain advisory, bounded and fail closed."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.schema_interpretation import analyze_schema


def column(name, role='unknown', evidence='ambiguous_integer', physical='number', high=10):
    return {'column': name, 'original_name': name, 'physical_type': physical,
            'samples': ['1', '2'], 'local_usage': {'role': role, 'evidence': evidence},
            'statistics': {'unique_count': 2, 'has_fractional_values': False,
                           'numeric_year': False, 'numeric_min': 0, 'numeric_max': high}}


def proposal(index, role='metric', temporal_kind='none', strength='strong'):
    return {'index': index, 'role': role, 'temporal_kind': temporal_kind, 'evidence_strength': strength}


class SchemaInterpretationTests(unittest.TestCase):
    def run_prediction(self, columns, predictions, status='completed'):
        client = MagicMock()
        response = client.__enter__.return_value.post.return_value
        response.json.return_value = {'status': status, 'output': [{'content': [
            {'type': 'output_text', 'text': json.dumps({'columns': predictions})}]}]}
        snapshot = {'rows': 10, 'columns': columns}
        before = json.dumps(snapshot, sort_keys=True)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test'}), patch('modules.schema_interpretation.httpx.Client', return_value=client):
            result = analyze_schema(snapshot)
        self.assertEqual(before, json.dumps(snapshot, sort_keys=True), 'AI must not mutate local metadata')
        return result, client.__enter__.return_value.post

    def test_one_request_uses_whole_context_for_labels_and_protects_roles(self):
        columns = [column('count'), column('date', role='temporal', evidence='validated_datetime', physical='datetime')]
        result, post = self.run_prediction(columns, [proposal(0), proposal(1, 'temporal', 'date_time')])
        post.assert_called_once()
        request = post.call_args.kwargs['json']
        self.assertEqual(request['model'], 'gpt-6-luna')
        self.assertFalse(request['store'])
        proposal_schema = request['text']['format']['schema']['$defs']['Proposal']
        self.assertEqual(set(proposal_schema['required']), set(proposal_schema['properties']))
        self.assertNotIn('default', proposal_schema['properties']['display_name'])
        state = json.loads(request['input'][1]['content'])
        self.assertEqual(len(state['columns']), 2)
        self.assertEqual(state['requested_indexes'], [0, 1])
        self.assertEqual(result['interpretation_proposals']['count']['decision']['scope'], 'role_only')
        self.assertEqual(result['interpretation_proposals']['count']['status'], 'ok')
        self.assertTrue(result['coverage']['complete'])
        self.assertEqual(result['coverage']['skipped_columns'], 0)

    def test_conflicts_and_unsupported_conversions_remain_uncertain(self):
        columns = [column('location_id', role='identifier', evidence='key_name_hint'),
                   column('weekday', role='categorical', evidence='validated_calendar_component', high=6),
                   column('yr', high=1), column('label', physical='string')]
        result, _ = self.run_prediction(columns, [proposal(0, 'categorical'), proposal(1, 'temporal', 'weekday'),
                                                    proposal(2, 'temporal', 'year'), proposal(3)])
        self.assertTrue(all(p['status']=='uncertain' for p in result['interpretation_proposals'].values()))
        self.assertEqual(result['interpretation_proposals']['location_id']['error_code'], 'role_conflict')

    def test_partial_duplicate_and_incomplete_provider_responses_are_rejected(self):
        for predictions, status in (([], 'completed'), ([proposal(0), proposal(0)], 'completed'), ([proposal(0)], 'incomplete')):
            result, _ = self.run_prediction([column('count')], predictions, status)
            self.assertIsNone(result['interpretation_proposals']['count']['decision'])
            self.assertFalse(result['coverage']['complete'])

    def test_missing_key_and_input_budget_never_call_provider(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}), patch('modules.schema_interpretation.httpx.Client') as client:
            result = analyze_schema({'rows': 2, 'columns': [column('count')]})
            client.assert_not_called()
            self.assertEqual(result['coverage']['stop_reason'], 'not_configured')
        cols = [column('x' * 256 + str(i)) for i in range(65)]
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test'}), patch('modules.schema_interpretation.httpx.Client') as client:
            result = analyze_schema({'rows': 2, 'columns': cols})
            client.assert_not_called()
            self.assertEqual(result['coverage']['stop_reason'], 'input_limit')

    def test_provider_failure_does_not_publish_credentials(self):
        import httpx
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test'}), patch('modules.schema_interpretation.httpx.Client', side_effect=httpx.ConnectError('secret sample')):
            result = analyze_schema({'rows': 2, 'columns': [column('count')]})
        self.assertNotIn('secret', json.dumps(result))
        self.assertEqual(result['coverage']['stop_reason'], 'provider_unavailable')
