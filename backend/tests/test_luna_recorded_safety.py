"""Replay frozen live predictions through ingestion without provider calls."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from modules.column_interpretation import InterpretationDecision, InterpretationResult
from modules.data_janitor import clean_dataframe
from modules.intelligence import auto_profile


class RecordedLunaSafetyTests(unittest.TestCase):
    def test_holdout_clarifications_and_wrong_auto_decisions_cannot_apply(self):
        cases = json.loads((BACKEND / 'evals/interpretation_holdout.json').read_text())['cases']
        report = json.loads((BACKEND / 'evals/results/luna-improvement-2026-10-01/luna-v3-holdout-none-1.json').read_text())
        results = {row['case_id']: row for row in report['results']}
        rejected = 0
        for case in cases:
            row = results[case['id']]
            actual = InterpretationDecision.model_validate(row['decision'])
            result = InterpretationResult(decision=actual, status=row['status'], provider='openai', model='gpt-6-luna')
            source = pd.DataFrame({case['column_name']: case['values']})
            with self.subTest(case=case['id']), patch.dict(os.environ, {'COLUMN_INTERPRETER': 'luna'}), patch(
                'modules.column_interpretation.interpret_column', return_value=result
            ), patch('builtins.print'):
                cleaned, _, _, formats, types = clean_dataframe(source)
                metadata = next(iter(cleaned.attrs['column_interpretations'].values()))
                if actual.needs_clarification or (not actual.needs_clarification and row['decision'] != case['expected']):
                    self.assertEqual(metadata['runtime_status'], 'clarification')
                    pd.testing.assert_series_equal(cleaned.iloc[:, 0], source.iloc[:, 0], check_names=False)
                    self.assertEqual(auto_profile(cleaned, types, formats)['top_metrics'], [])
                    rejected += 1
        self.assertGreater(rejected, 10)


if __name__ == '__main__':
    unittest.main()
