"""Paired real-file chart accuracy/latency benchmark; gold is never in prompts."""
from __future__ import annotations
import argparse
import contextlib
import csv
from decimal import Decimal
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / 'benchmarks'))


def cases_for(spec):
    cases = [dict(c) for c in spec['queries']]
    extra = {
        'bike': [dict(prompt='Show total cnt by weekday for workingday equal to 1', x='weekday', y='cnt', aggregation='sum', filters=[('workingday', 'eq', '1')])],
        'bank': [dict(prompt='Show average age by job for married people older than 40', x='job', y='age', aggregation='mean', filters=[('marital', 'eq', 'married'), ('age', 'gt', '40')]),
                 dict(prompt='Show average balance by job', x='job', y='balance', aggregation='mean', active_filters=[('marital', 'eq', 'married')]),
                 dict(prompt='Show average age by marital status for people working in management or services', x='marital', y='age', aggregation='mean', filters=[('job', 'values', ['management', 'services'])])],
        'retail': [dict(prompt='Show total quantity by country for United Kingdom', x='country', y='quantity', aggregation='sum', filters=[('country', 'eq', 'United Kingdom')]),
                   dict(prompt='Show total quantity by country where unit_price is greater than 10', x='country', y='quantity', aggregation='sum', filters=[('unit_price', 'gt', '10')])],
        'taxi': [dict(prompt='Which payment types have the highest average trip distance?', x='payment_type', y='trip_distance', aggregation='mean'),
                 dict(prompt='Show average trip distance by payment type where fare_amount is greater than 10 and trip_distance is less than 20', x='payment_type', y='trip_distance', aggregation='mean', filters=[('fare_amount', 'gt', '10'), ('trip_distance', 'lt', '20')]),
                 dict(prompt='Count all trip records', x=None, y=None, aggregation='count')],
        'wine': [dict(prompt='Show average alcohol by quality where quality is at least 7', x='quality', y='alcohol', aggregation='mean', filters=[('quality', 'gte', '7')])],
        'diamonds': [dict(prompt='Show average price by cut where carat is greater than 1', x='cut', y='price', aggregation='mean', filters=[('carat', 'gt', '1')]),
                     dict(prompt='Compare price by cut', x='cut', y='price', aggregation='mean')],
        'penguins': [dict(prompt='Show average body mass by species on Biscoe island', x='species', y='body_mass_g', aggregation='mean', filters=[('island', 'eq', 'Biscoe')]),
                     dict(prompt='Show average body mass by species only where sex is missing', x='species', y='body_mass_g', aggregation='mean', filters=[('sex', 'values', [None])])],
        'tips': [dict(prompt='Show average tip by day for females who do not smoke', x='day', y='tip', aggregation='mean', filters=[('sex', 'eq', 'Female'), ('smoker', 'eq', 'No')]),
                 dict(prompt='Show total bill by time', x='time', y='total_bill', aggregation='sum')],
    }
    cases += [{'kind': 'chart', **c} for c in extra[spec['id']]]
    for i, c in enumerate(cases):
        c['case_id'] = f"{spec['id']}-{i}"
    return cases


def oracle(path, spec, case):
    """Independent full-source CSV/Decimal aggregate, including filters."""
    from modules.data_janitor import legacy_normalize_header
    groups = {}
    numeric = {'gt', 'gte', 'lt', 'lte'}
    filters = [*case.get('active_filters', []), *case.get('filters', [])]
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=spec.get('delimiter', ','))
        names = {legacy_normalize_header(h): h for h in reader.fieldnames}
        for row in reader:
            accept = True
            for column, op, value in filters:
                actual = row[names[column]]
                if op in numeric:
                    if not actual:
                        accept = False
                        break
                    a, b = Decimal(actual.replace(',', '')), Decimal(value)
                    accept &= {'gt': a > b, 'gte': a >= b, 'lt': a < b, 'lte': a <= b}[op]
                elif op == 'values':
                    accept &= (None if actual == '' else actual) in value
                else:
                    accept &= actual == value
                if not accept:
                    break
            if not accept:
                continue
            key = row[names[case['x']]] if case['x'] else '__overall__'
            y = case['y']
            text = row[names[y]] if y else '1'
            if not text:
                continue
            number = Decimal(1) if case['aggregation'] == 'count' else Decimal(text.replace(',', ''))
            total, count = groups.setdefault(key, [Decimal(0), 0])
            groups[key] = [total + number, count + 1]
    return {k: float(total / count if case['aggregation'] == 'mean' else total)
            for k, (total, count) in groups.items()}


def score(result, case, expected):
    kind = 'clarification' if result.chart_type == 'empty' else 'chart'
    if case['kind'] == 'clarification':
        return kind == 'clarification', {'kind': kind}
    if kind != 'chart':
        return False, {'kind': kind, 'false_clarification': True}
    x = case['x']
    y = case['y'] or result.y_axis_keys[0]
    values = {('' if row.get(x) is None else str(row[x])) if x else '__overall__': row.get(y) for row in result.data}
    shape = result.source_x_axis_key == x and result.y_axis_keys == [y] and result.aggregation == case['aggregation']
    # source_x_axis_key may be None for ordinary non-resampled grouped charts.
    shape = ((result.source_x_axis_key or result.x_axis_key) == x if x else result.aggregation_scope == 'overall') and result.y_axis_keys == [y] and result.aggregation == case['aggregation']
    numbers = values.keys() == expected.keys() and all(
        isinstance(values[k], (int, float)) and math.isclose(values[k], v, rel_tol=1e-9, abs_tol=1e-6)
        for k, v in expected.items())
    def filter_key(column, operator, value):
        if operator == 'values' and len(value) == 1 and value[0] is not None:
            operator, value = 'eq', value[0]
        if operator == 'values':
            token = json.dumps(sorted(value, key=lambda v: json.dumps(v)), ensure_ascii=False)
        elif operator in {'gt', 'gte', 'lt', 'lte'}:
            token = str(Decimal(str(value)).normalize())
        else:
            token = str(value)
        return (column, operator, token)
    actual_filters = sorted(filter_key(f.column, f.operator or 'values', f.value if f.operator else f.values) for f in result.llm_filters or [])
    expected_filters = sorted(filter_key(*f) for f in case.get('filters', []))
    filters = actual_filters == expected_filters
    return shape and numbers and filters, {'kind': kind, 'shape': shape, 'values': numbers,
                                          'filters': filters, 'actual_filters': actual_filters,
                                          'expected_groups': len(expected), 'actual_groups': len(values),
                                          'missing_group_keys': sorted(expected.keys() - values.keys()),
                                          'extra_group_keys': sorted(values.keys() - expected.keys()),
                                          'actual_x': result.x_axis_key, 'actual_y': result.y_axis_keys,
                                          'actual_aggregation': result.aggregation}


def run(args):
    from core import config  # loads the configured key without printing it
    from ingestion_batching import gold_specs
    from ingestion_parallelism import spec_path
    from modules.import_policy import ImportSettings
    from services.csv_ingestion import ingest_csv
    from services import query_service as current
    from storage import DATASETS, get_dataset
    from models import QueryRequest, FilterConfig
    from modules.schema_interpretation import dataset_snapshot, analyze_schema_labels, apply_schema_roles
    from modules.schema_decisions import analyze_schema_decisions
    from services.ingestion_service import _generate_business_summary

    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('Configured provider key required')
    spec_module = importlib.util.spec_from_file_location('frozen_query_baseline', args.baseline)
    baseline = importlib.util.module_from_spec(spec_module)
    spec_module.loader.exec_module(baseline)
    baseline.SYSTEM_PROMPT = json.loads(args.baseline_prompt.read_text())
    # Keep the old provider contract despite new optional membership support.
    import copy
    baseline._QUERY_PLAN_SCHEMA = copy.deepcopy(baseline._QUERY_PLAN_SCHEMA)
    old_filter = baseline._QUERY_PLAN_SCHEMA['$defs']['QueryFilterPlan']
    old_filter['properties'].pop('values', None)
    old_filter['properties']['operator'] = {'type': 'string', 'enum': ['eq', 'gt', 'lt', 'gte', 'lte', 'contains']}
    old_filter['properties']['value'] = {'anyOf': [{'type': 'string'}, {'type': 'integer'}, {'type': 'number'}, {'type': 'boolean'}]}
    old_filter['required'] = ['column', 'operator', 'value']
    specs = [s for s in gold_specs() if not args.datasets or s['id'] in args.datasets]
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    old_decision = current.plan_decisions
    total = sum(len(cases_for(s)) for s in specs) * args.repeats * len(args.arms)
    for spec in specs:
        path = spec_path(spec)
        with contextlib.redirect_stdout(io.StringIO()):
            upload = ingest_csv(path, spec['filename'], '/benchmark', enqueue_enrichment=False,
                                import_settings=ImportSettings(delimiter=spec.get('delimiter', ',')))
        ds = get_dataset(upload.dataset_id)
        try:
            if args.enrich:
                snap = dataset_snapshot(ds)
                ds.summary = _generate_business_summary(spec['filename'], ds.sample_frame(3))
                roles = analyze_schema_decisions(snap)
                apply_schema_roles(ds.id, ds.cache_version, roles)
                labels = analyze_schema_labels(snap)
                apply_schema_roles(ds.id, ds.cache_version, labels)
            cases = cases_for(spec)
            expectations = {c['case_id']: oracle(path, spec, c) for c in cases if c['kind'] == 'chart'}
            jobs = [(c, repeat, arm) for c in cases for repeat in range(args.repeats) for arm in args.arms]
            random.Random(20261007).shuffle(jobs)
            for case, repeat, arm in jobs:
                route = {'decisions_called': False, 'decisions_plan': False}
                def traced(context):
                    route['decisions_called'] = True
                    start = perf_counter()
                    p = old_decision(context, diagnostics=route)
                    route['decisions_seconds'] = perf_counter() - start
                    route['decisions_plan'] = p is not None
                    return p
                request = QueryRequest(dataset_id=ds.id, user_prompt=case['prompt'],
                    filters=[FilterConfig(column=c, operator=o, value=v) for c, o, v in case.get('active_filters', [])] or None,
                    limit=10000, sort_by='label', group_others=False)
                start = perf_counter()
                try:
                    with patch.dict(os.environ, {'QUERY_PLANNER_BACKEND': 'generative' if arm == 'context' else 'decisions'}), patch.object(current, 'plan_decisions', traced):
                        result = (baseline if arm == 'before' else current).run_query(request)
                    seconds = perf_counter() - start
                    correct, details = score(result, case, expectations.get(case['case_id']))
                    record = {'dataset': spec['id'], 'case_id': case['case_id'], 'repeat': repeat,
                              'arm': arm, 'seconds': seconds, 'correct': correct, 'expected_kind': case['kind'],
                              **details, **route}
                except Exception as error:
                    record = {'dataset': spec['id'], 'case_id': case['case_id'], 'repeat': repeat,
                              'arm': arm, 'seconds': perf_counter() - start, 'correct': False,
                              'expected_kind': case['kind'], 'error_type': type(error).__name__, **route}
                records.append(record)
                with (args.output / 'records.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print(json.dumps({'progress': f'{len(records)}/{total}', **record}), flush=True)
        finally:
            DATASETS.pop(ds.id, None)
            ds.close()
    summary = []
    for arm in args.arms:
        rows = [r for r in records if r['arm'] == arm]
        charts = [r for r in rows if r['expected_kind'] == 'chart']
        timings = sorted(r['seconds'] for r in rows)
        summary.append({'arm': arm, 'cases': len(rows), 'correct': sum(r['correct'] for r in rows),
                        'chart_cases': len(charts), 'false_clarifications': sum(r.get('false_clarification', False) for r in rows),
                        'unsafe_charts': sum(r.get('kind') == 'chart' and r['expected_kind'] == 'clarification' for r in rows),
                        'median_seconds': statistics.median(timings), 'p95_seconds': timings[max(0, math.ceil(.95 * len(timings)) - 1)],
                        'median_chart_seconds': statistics.median(r['seconds'] for r in charts),
                        'decisions_plan_count': sum(r['decisions_plan'] for r in rows),
                        'errors': sum('error_type' in r for r in rows)})
    payload = {'date': '2026-10-07', 'repeats': args.repeats, 'datasets': [s['id'] for s in specs],
               'scope': 'Live question to computed chart/clarification; excludes ingestion, transfer and browser. Real full files; independent CSV/Decimal numerical oracle.',
               'enriched': args.enrich, 'summary': summary, 'failures': [r for r in records if not r['correct']]}
    (args.output / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'summary': summary}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--arms', nargs='+', choices=['before', 'context', 'decisions'], default=['before', 'decisions'])
    parser.add_argument('--repeats', type=int, choices=[1, 3], default=3)
    parser.add_argument('--enrich', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('/private/tmp/analytico-query-before-2026-10-07.py'))
    parser.add_argument('--baseline-prompt', type=Path, default=Path('/private/tmp/analytico-query-before-prompt-2026-10-07.json'))
    run(parser.parse_args())
