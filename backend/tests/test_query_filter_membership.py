"""Null/membership query plans use the existing complete-data filtering engine."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from pydantic import ValidationError

from models import QueryFilterPlan, QueryRequest
from services.query_service import run_query


class QueryFilterMembershipTests(unittest.TestCase):
    def test_membership_and_missing_rows_are_computed(self):
        dataset = SimpleNamespace(id='unit', df=pd.DataFrame({'group': ['A', 'B', None], 'amount': [2, 3, 5]}),
                                  column_types={'group': 'categorical', 'amount': 'metric'}, column_formats={})
        for values, expected in [(['A', 'B'], 5), ([None], 5)]:
            plan = dict(kind='chart', x_axis_key=None, y_axis_keys=['amount'], aggregation='sum',
                        chart_type='bar', title='Total', x_axis_label=None, y_axis_label=None,
                        filters=[dict(column='group', operator=None, value=None, values=values)], clarification=None)
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(plan), tool_calls=None))]))))
            with patch('services.query_service.get_dataset', return_value=dataset), patch('services.aggregation_service.get_dataset', return_value=dataset), patch('services.query_service.plan_decisions', return_value=None), patch('services.query_service.get_openai_client', return_value=client):
                result = run_query(QueryRequest(dataset_id='unit', user_prompt='sum matching rows'))
            self.assertEqual(result.data[0]['amount'], expected)
            self.assertEqual(result.llm_filters[0].values, values)

    def test_conflicting_empty_and_nonfinite_memberships_are_rejected(self):
        for kwargs in [dict(values=[]), dict(operator='eq', value='A', values=['A']),
                       dict(values=[float('inf')]), dict(operator='eq', value=None)]:
            with self.assertRaises(ValidationError):
                QueryFilterPlan(column='group', **kwargs)
