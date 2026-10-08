"""Decision transport and filter contracts, independent of live provider behavior."""
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import httpx
from openai import OpenAI

from models import QueryRequest
from modules.query_decisions import filter_values, plan_decisions, query_context, questions_for


class QueryDecisionsTests(unittest.TestCase):
    def context(self, prompt='Show average value by category where value is greater than 40'):
        dataset = SimpleNamespace(id='unit', column_types={'category': 'categorical', 'value': 'unknown'},
                                  column_formats={}, summary='An inferred description.', row_count=100,
                                  column_schema=[{'column': 'value', 'original_name': 'Value', 'display_name': 'Measurement'}])
        frame = pd.DataFrame({'category': ['A', 'B'], 'value': [3, 9]})
        return query_context(dataset, frame, QueryRequest(dataset_id='unit', user_prompt=prompt))

    def answer(self, questions, selected):
        answers = []
        for q in questions:
            options = [v['value'] for v in q['choices']]
            value = selected.get(q['name'], options[0])
            answers.append({'name': q['name'], 'type': 'choice', 'choice': value,
                            'confidence': 1.0, 'probabilities': [
                                {'value': v, 'probability': float(v == value)} for v in options]})
        return {'model': 'gpt-6-luna', 'answers': answers}

    def test_context_and_literal_filter_boundaries_are_preserved(self):
        context = self.context()
        self.assertEqual(context['dataset_summary'], 'An inferred description.')
        self.assertEqual(context['columns'][1]['role'], 'unknown')
        self.assertEqual(context['columns'][1]['display_name'], 'Measurement')
        self.assertIn('40', filter_values(context))  # not present in observed rows
        questions, _ = questions_for(context)
        self.assertEqual(len(questions), 18)
        self.assertTrue(all(2 <= len(q['choices']) <= 255 for q in questions))

    def test_one_decision_call_builds_exact_filter_and_stable_keys(self):
        context = self.context()
        questions, values = questions_for(context)
        selected = {'kind': 'chart', 'filter_count': '1', 'x': 'c0', 'y': 'c1', 'aggregation': 'mean',
                    'filter0_column': 'c1', 'filter0_operator': 'gt',
                    'filter0_value': 'v' + str(values.index('40'))}
        body = self.answer(questions, selected)
        client = SimpleNamespace(post=lambda *a, **kw: body)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), patch('modules.query_decisions.get_openai_client', return_value=client):
            plan = plan_decisions(context)
        self.assertEqual(plan.x_axis_key, 'category')
        self.assertEqual(plan.y_axis_keys, ['value'])
        self.assertEqual(plan.filters[0].model_dump(exclude_none=True), {'column': 'value', 'operator': 'gt', 'value': '40'})

    def test_refusal_missing_answers_and_incoherent_row_count_delegate(self):
        context = self.context()
        questions, _ = questions_for(context)
        bodies = [self.answer(questions, {'kind': 'fallback'}),
                  self.answer(questions, {'kind': 'chart', 'x': 'c0', 'y': 'rows', 'aggregation': 'sum'}),
                  {'model': 'gpt-6-luna', 'answers': []}]
        for body in bodies:
            client = SimpleNamespace(post=lambda *a, **kw: body)
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), patch('modules.query_decisions.get_openai_client', return_value=client):
                    self.assertIsNone(plan_decisions(context))

    def test_duplicate_filters_delegate_instead_of_silently_dropping(self):
        context = self.context()
        questions, values = questions_for(context)
        selected = {'kind': 'chart', 'filter_count': '2', 'x': 'c0', 'y': 'c1', 'aggregation': 'mean'}
        for i in range(2):
            selected.update({f'filter{i}_column': 'c1', f'filter{i}_operator': 'gt',
                             f'filter{i}_value': 'v' + str(values.index('40'))})
        body = self.answer(questions, selected)
        client = SimpleNamespace(post=lambda *a, **kw: body)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), patch('modules.query_decisions.get_openai_client', return_value=client):
            self.assertIsNone(plan_decisions(context))

    def test_sdk_decodes_actual_json_before_strict_validation(self):
        context = self.context()
        questions, _ = questions_for(context)
        body = self.answer(questions, {'kind': 'chart', 'x': 'c0', 'y': 'c1', 'aggregation': 'mean'})
        client = OpenAI(api_key='test-key', http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))))
        try:
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), patch('modules.query_decisions.get_openai_client', return_value=client):
                plan = plan_decisions(context)
            self.assertIsNotNone(plan)
            self.assertEqual(plan.y_axis_keys, ['value'])
        finally:
            client.close()

    def test_missing_ranking_objective_is_clarification_without_another_call(self):
        context = self.context('Which category is best?')
        questions, _ = questions_for(context)
        body = self.answer(questions, {'objective': 'needs_definition'})
        client = SimpleNamespace(post=lambda *a, **kw: body)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), patch('modules.query_decisions.get_openai_client', return_value=client):
            plan = plan_decisions(context)
        self.assertEqual(plan.kind, 'clarification')
        self.assertIn('statistic', plan.clarification)

    def test_leading_zero_and_quoted_unknown_value_remain_literal(self):
        values = filter_values(self.context('Filter code "001" and country "New Place"'))
        self.assertIn('001', values)
        self.assertIn('New Place', values)
        self.assertFalse(any(type(v) is int and v == 1 for v in values))
