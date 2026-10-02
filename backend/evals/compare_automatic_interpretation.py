#!/usr/bin/env python3
"""Compare saved model decisions with local inference; never make provider calls."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ['OPENAI_API_KEY'] = ''
os.environ['COLUMN_INTERPRETER'] = 'off'
import pandas as pd
from modules.data_janitor import clean_dataframe
from modules.intelligence import auto_profile, detect_semantic_type
from evaluate_interpretation import score


def inspect(frame, provider_row=None):
    with contextlib.redirect_stdout(io.StringIO()):
        if provider_row is None:
            cleaned, actions, missing, formats, roles = clean_dataframe(frame, interpret_columns=False)
        else:
            decision = provider_row.get('decision')
            row = {'original': frame.columns[0], 'clean': frame.columns[0], 'format': None,
                   'semantic_type': decision.get('role') if decision else None,
                   'interpretation': {'status': provider_row['status'], 'runtime_status': 'clarification', 'decision': decision}}
            with patch('modules.data_janitor.llm_enrich_columns', return_value=[row]):
                cleaned, actions, missing, formats, roles = clean_dataframe(frame)
    column = cleaned.columns[0]
    role = roles.get(column) or detect_semantic_type(cleaned, column)
    profile = auto_profile(cleaned, {column: role}, formats)
    metric = next((item for item in profile['top_metrics'] if item['name'] == column), None)
    decision = {'role': role}
    if metric:
        decision['recommended_aggregation'] = metric['aggregation']
    serial_values = [None if pd.isna(value) else value.isoformat() if isinstance(value, pd.Timestamp)
                     else value.item() if hasattr(value, 'item') else value for value in cleaned[column]]
    return {'decision': decision, 'dtype': str(cleaned[column].dtype), 'values': serial_values,
            'missing': missing, 'format': formats.get(column), 'actions': actions,
            'runtime_status': cleaned.attrs.get('column_interpretations', {}).get(column, {}).get('runtime_status')}


def compare(cases_path, runs):
    cases = json.loads(cases_path.read_text())['cases']
    observations, timings = {}, []
    for case in cases:
        frame = pd.DataFrame({case['column_name']: case['values']})
        case_times = []
        for _ in range(5):
            tick = perf_counter()
            observation = inspect(frame)
            case_times.append((perf_counter() - tick) * 1000)
        timings.append(statistics.median(case_times))
        observations[case['id']] = {'case_id': case['id'], **observation}
    result = {'cases': len(cases), 'automatic': {
        'local_parse_profile_p50_ms': statistics.median(timings),
        'scoring': score(cases, 'automatic', observations), 'observations': list(observations.values())}, 'ai': {}}
    for path in runs:
        saved = json.loads(path.read_text())
        predictions = {row['case_id']: row for row in saved['results']}
        rows = []
        for case in cases:
            provider = predictions.get(case['id'])
            if provider is None:
                continue
            observation = inspect(pd.DataFrame({case['column_name']: case['values']}), provider)
            rows.append({'case_id': case['id'], **observation})
        result['ai'][path.stem] = {'model_decision_scoring': saved['scoring'], 'provider_metrics': saved['metrics'],
            'failure_count': saved['failure_count'],
            'validated_runtime_scoring': score(cases, path.stem, {row['case_id']: row for row in rows}),
            'applied_columns': sum(row['runtime_status'] == 'applied' for row in rows),
            'review_columns': sum(row['runtime_status'] != 'applied' for row in rows),
            'observations': rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.cases.resolve(), *(path.resolve() for path in args.runs)}:
        parser.error("Output must not overwrite cases or saved predictions.")
    output = compare(args.cases, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(f"Compared {output['cases']} cases; results saved to {args.output}.")


if __name__ == '__main__':
    main()
