"""Guard experiment output integrity before scoring or merging role proposals."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.semantic_hybrid import compatible, model_state, selected, validate_proposals


class SemanticHybridBenchmarkTests(unittest.TestCase):
    def test_provider_must_return_exact_requested_columns(self):
        proposal = {'index': 2, 'role': 'metric', 'evidence_strength': 'strong', 'temporal_kind': 'none'}
        for items in ([], [proposal, proposal], [{**proposal, 'index': 3}]):
            with self.assertRaises(ValueError):
                validate_proposals(json.dumps({'columns': items}), [2])
        self.assertEqual(validate_proposals(json.dumps({'columns': [proposal]}), [2])[2], {**proposal, 'display_name': None, 'label_evidence_strength': 'unknown'})

    def test_role_checks_reject_impossible_numeric_or_calendar_use(self):
        column = {'physical_type': 'string', 'statistics': {'unique_count': 5}}
        self.assertFalse(compatible(column, {'role': 'metric'}))
        column = {'physical_type': 'number', 'statistics': {'unique_count': 5, 'has_fractional_values': False,
                  'numeric_year': False, 'numeric_min': 0, 'numeric_max': 50}}
        self.assertFalse(compatible(column, {'role': 'temporal', 'temporal_kind': 'hour'}))
        column['statistics']['numeric_max'] = 23
        self.assertTrue(compatible(column, {'role': 'temporal', 'temporal_kind': 'hour'}))
        column['statistics']['has_fractional_values'] = True
        self.assertFalse(compatible(column, {'role': 'temporal', 'temporal_kind': 'hour'}))

    def test_prompt_does_not_send_local_roles_or_gold_labels(self):
        column = {'column': 'x', 'original_name': 'x', 'physical_type': 'number', 'samples': ['1','2'],
                  'statistics': {'unique_count': 2, 'sample_values': ['1','2']},
                  'local_usage': {'role': 'unknown', 'evidence': 'ambiguous_integer'}}
        state = model_state({'rows': 2, 'columns': [column]}, 1)
        self.assertTrue(selected(column))
        self.assertEqual(state['requested_indexes'], [0])
        self.assertNotIn('local_usage', state['columns'][0])
        self.assertNotIn('role', state['columns'][0])
        column['local_usage'] = {'role': 'temporal', 'evidence': 'validated_datetime'}
        self.assertFalse(selected(column))


if __name__ == '__main__':
    unittest.main()
