"""Real-data service acceptance benchmark. --live explicitly enables paid Luna calls."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
MANIFEST = BACKEND / 'evals/real_dataset_manifest.json'


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(path, delimiter, x, y, aggregation):
    """Independent streaming csv/decimal oracle; preserve negative and missing values."""
    from decimal import Decimal
    from modules.data_janitor import legacy_normalize_header
    groups = {}
    rows = 0
    missing = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        mapping = {legacy_normalize_header(h): h for h in reader.fieldnames}
        missing = {legacy_normalize_header(h): 0 for h in reader.fieldnames}
        for row in reader:
            rows += 1
            for name, original in mapping.items():
                missing[name] += row[original] == ''
            value, key = row[mapping[y]], row[mapping[x]]
            # These reference columns contain plain numeric or correctly grouped decimal text.
            if value != '':
                total, count = groups.get(key, (Decimal(0), 0))
                groups[key] = (total + Decimal(value.replace(',', '')), count + 1)
    return rows, missing, {key: float(total / count if aggregation == 'mean' else total)
                           for key, (total, count) in groups.items()}


def matches(result, expected, x, y):
    actual = {('' if row[x] is None else str(row[x])): row[y] for row in result.get('data', [])}
    return actual.keys() == expected.keys() and all(
        isinstance(actual[key], (int, float)) and math.isclose(actual[key], value, rel_tol=1e-9, abs_tol=1e-6)
        for key, value in expected.items())


def worker(spec, path, live, repeats, queries_only=False):
    import core.config as config
    from fastapi.testclient import TestClient
    from main import app
    from modules.data_janitor import legacy_normalize_header
    from modules.column_interpretation import interpret_column
    from modules.data_janitor import _interpretation_input
    from services.import_preview import stage_import, confirm_import
    from storage import get_dataset, DATASETS
    if live:
        assert config.OPENAI_MODEL == 'gpt-6-luna'
        os.environ['COLUMN_INTERPRETER_MODEL'] = 'gpt-6-luna'
        if not os.getenv('OPENAI_API_KEY', '').strip():
            raise ValueError('A configured OpenAI key is required.')
    else:
        os.environ['OPENAI_API_KEY'] = ''
    records = []
    telemetry = io.StringIO()
    # No dataset summary or interpreter job is scheduled inside timed local readiness.
    for trial in range(repeats):
        with contextlib.redirect_stdout(telemetry):
            start = perf_counter()
            preview = stage_import(path, path.name)
            staged = perf_counter()
            if not preview['can_confirm']:
                raise AssertionError('Real source did not pass preflight')
            from unittest.mock import patch
            with patch('services.ingestion_service.queue_dataset_enrichment', return_value='disabled'), \
                 patch('services.csv_ingestion.queue_dataset_enrichment', return_value='disabled'):
                response = confirm_import(preview['import_id'], ai_column_analysis=False)
            ready = perf_counter()
            ds = get_dataset(response.dataset_id)
            client = TestClient(app)
            request = {'dataset_id': ds.id, 'x_axis_key': spec['x'], 'y_axis_keys': [spec['y']],
                       'aggregation': spec['aggregation'], 'limit': 0, 'group_others': False, 'include_analysis': False}
            tick = perf_counter()
            chart_response = client.post('/aggregate', json=request)
            cold = perf_counter() - tick
            assert chart_response.status_code == 200
            chart = chart_response.json()
            tick = perf_counter()
            warm_response = client.post('/aggregate', json=request)
            warm = perf_counter() - tick
            assert warm_response.status_code == 200
            assert warm_response.json() == chart
        # Oracle after readiness to avoid distorting the pipeline timing.
        oracle_rows, oracle_missing, expected = reference(path, spec['delimiter'], spec['x'], spec['y'], spec['aggregation'])
        assert matches(chart, expected, spec['x'], spec['y']), 'Aggregate disagrees with streaming oracle'
        assert response.row_count == oracle_rows
        assert {k: v for k, v in response.data_health.missing_values.items() if v} == {k: v for k, v in oracle_missing.items() if v}
        retained_source = ds.disk.source_path if hasattr(ds, 'disk') else Path(ds.source_path)
        assert file_hash(retained_source) == file_hash(path), 'Retained CSV bytes changed'
        record = {'trial': trial + 1, 'stage_seconds': staged - start, 'confirm_seconds': ready - staged,
                  'local_ready_seconds': ready - start, 'cold_aggregate_api_seconds': cold,
                  'warm_aggregate_api_seconds': warm, 'rows': ds.row_count, 'columns': len(ds.column_names),
                  'engine': 'disk' if hasattr(ds, 'disk') else 'pandas', 'aggregate_matches_oracle': True,
                  'source_missing_counts_match': True, 'local_roles': ds.column_types, 'default_chart': response.default_chart.model_dump() if response.default_chart else None,
                  'source_bytes_preserved': True,
                  'top_metrics': [m.name for m in response.profile.top_metrics],
                  'column_formats': ds.column_formats,
                  'column_usages': {item['column']: item.get('usage') for item in response.column_schema}}
        if live and trial == 0:
            if not queries_only:
                payloads = (ds.disk.interpretation_inputs(limit=256) if hasattr(ds, 'disk') else
                            [(ds.column_names[i], _interpretation_input(header, ds.raw_df.iloc[:, i], list(ds.raw_df.columns)))
                             for i, header in enumerate(ds.raw_df.columns)])
                if not hasattr(ds, 'disk'):
                    for name, payload in payloads:
                        stat = ds.column_stats[name]
                        payload['context'].update(column_unique_count=stat.unique_count,
                                                  row_position_sequence=stat.row_position_sequence)
                def classify(item):
                    name, payload = item
                    result = interpret_column(payload, provider='luna', use_cache=False)
                    original = payload['column_name']
                    expected_role = spec['roles'].get(original)
                    valid_roles = [expected_role, *spec['role_alternatives'].get(original, [])]
                    decision = result.decision.model_dump() if result.decision else None
                    return {'column': name, 'original_name': original, 'expected_role': expected_role,
                            'accepted_roles': valid_roles, 'opaque': original in spec['opaque_columns'],
                            'decision': decision, 'status': result.status, 'latency_ms': result.latency_ms,
                            'model': result.model, 'prompt_version': result.prompt_version, 'error_code': result.error_code,
                            'role_correct': bool(decision and decision['role'] in valid_roles)}
                tick = perf_counter()
                with ThreadPoolExecutor(max_workers=4) as pool:
                    record['interpretations'] = list(pool.map(classify, payloads))
                record['column_analysis_seconds'] = perf_counter() - tick
            record['queries'] = []
            for case in spec['queries']:
                tick = perf_counter()
                answer = client.post('/query', json={'dataset_id': ds.id, 'user_prompt': case['prompt'],
                                                    'limit': 0, 'group_others': False})
                elapsed = perf_counter() - tick
                if answer.status_code != 200:
                    record['queries'].append({'prompt': case['prompt'], 'seconds': elapsed,
                                              'status_code': answer.status_code, 'correct': False})
                    continue
                result = answer.json()
                kind = 'clarification' if result['chart_type'] == 'empty' else 'chart'
                correct = kind == case['kind']
                if correct and kind == 'chart':
                    correct = (result['x_axis_key'] == case['x'] and result['y_axis_keys'] == [case['y']]
                               and result['aggregation'] == case['aggregation'])
                    if correct:
                        _, _, expected_query = reference(path, spec['delimiter'], case['x'], case['y'], case['aggregation'])
                        correct = matches(result, expected_query, case['x'], case['y'])
                record['queries'].append({'prompt': case['prompt'], 'seconds': elapsed, 'correct': correct,
                                          'expected_kind': case['kind'], 'actual_kind': kind,
                                          'x': result['x_axis_key'], 'ys': result['y_axis_keys'],
                                          'aggregation': result.get('aggregation'), 'analysis': result.get('analysis')})
        records.append(record)
        ds.close()
        DATASETS.pop(ds.id, None)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'dataset': spec['id'], 'file_bytes': path.stat().st_size,
            'sha256': file_hash(path), 'records': records,
            'process_peak_rss_bytes': rss if sys.platform == 'darwin' else rss * 1024}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/private/tmp/analytico-real-fixtures'))
    parser.add_argument('--datasets', nargs='+', default=['bike', 'bank', 'retail', 'taxi'])
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--queries-only', action='store_true', help='With --live, skip paid column interpretation calls')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    specs = {x['id']: x for x in json.loads(MANIFEST.read_text())['datasets']}
    if args.worker:
        spec = specs[args.datasets[0]]
        path = BACKEND.parent / spec['local_path'] if 'local_path' in spec else args.root / spec['filename']
        try:
            result = worker(spec, path, args.live, args.repeats, args.queries_only)
        except Exception as error:
            # Never output model/provider exception content or credentials.
            result = {'dataset': spec['id'], 'failure_type': type(error).__name__}
        args.output.write_text(json.dumps(result, indent=2))
        return
    report = {'manifest_version': 1, 'live': args.live, 'model': 'gpt-6-luna' if args.live else None,
              'code_sha256': {str(path.relative_to(BACKEND)): file_hash(path) for path in
                              [BACKEND / ('modules/' + name + '.py') for name in
                               ('column_usage', 'column_statistics', 'intelligence', 'data_janitor', 'disk_dataset')]},
              'queries_only': args.queries_only, 'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              'timing_scope': 'local stage/confirm and in-process aggregate HTTP API; excludes transfer/browser render',
              'hardware': {'machine': platform.machine(), 'platform': platform.platform()},
              'datasets': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        part = args.output.with_name(args.output.stem + '-' + name + '.json')
        command = [sys.executable, __file__, '--worker', '--root', str(args.root), '--datasets', name,
                   '--repeats', str(args.repeats), '--output', str(part)] + (['--live'] if args.live else []) + (['--queries-only'] if args.queries_only else [])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        report['datasets'].append(json.loads(part.read_text()))
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({'dataset': name, 'saved': True, 'failed': 'failure_type' in report['datasets'][-1]}), flush=True)
    # A persisted failure is evidence, not a successful acceptance run.
    if any('failure_type' in dataset or any(not query['correct'] for record in dataset.get('records', [])
           for query in record.get('queries', [])) for dataset in report['datasets']):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
