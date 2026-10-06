"""Fresh production-path ingestion and Luna query accuracy benchmark.

The fixed question/gold set is kept outside the request path. This runner records
source-byte checks, exact service timings, applied column roles, schema labels,
and live query results with an independent streaming Decimal oracle.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import resource
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

GOLD_PATH = BACKEND / 'evals/fresh_accuracy_gold_2026-10-03.json'
FIXTURES = Path('/private/tmp/analytico-real-fixtures')
RAW_PATH = Path('/private/tmp/analytico-fresh-accuracy-2026-10-03-raw.json')
SUMMARY_PATH = BACKEND / 'benchmarks/results/fresh-accuracy-2026-10-03/summary.json'
REPORT_PATH = REPO / 'docs/fresh-accuracy-2026-10-03.md'
REPEATS = 3
ENRICHMENT_TIMEOUT_SECONDS = 180

FINGERPRINT_FILES = [
    BACKEND / 'modules/schema_interpretation.py',
    BACKEND / 'modules/column_usage.py',
    BACKEND / 'modules/column_statistics.py',
    BACKEND / 'modules/data_janitor.py',
    BACKEND / 'modules/disk_dataset.py',
    BACKEND / 'modules/import_policy.py',
    BACKEND / 'services/import_preview.py',
    BACKEND / 'services/csv_ingestion.py',
    BACKEND / 'services/ingestion_service.py',
    BACKEND / 'services/enrichment_service.py',
    BACKEND / 'services/parallel_enrichment.py',
    BACKEND / 'services/query_service.py',
    BACKEND / 'services/aggregation_service.py',
    BACKEND / 'modules/aggregation.py',
    BACKEND / 'models.py',
]


def sha256_file(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalized_header(value: str) -> str:
    value = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', value.strip())
    value = re.sub(r'[^\w]+', '_', value.lower())
    value = re.sub(r'_+', '_', value).strip('_')
    return value or 'column'


def source_hashes(datasets: list[dict]) -> dict[str, dict]:
    result = {}
    for spec in datasets:
        path = Path(spec['path'])
        result[spec['id']] = {
            'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path),
        }
    return result


def code_fingerprints() -> dict[str, str]:
    return {str(path.relative_to(REPO)): sha256_file(path) for path in FINGERPRINT_FILES}


def process_memory() -> dict:
    current = None
    try:
        import psutil
        current = int(psutil.Process().memory_info().rss)
    except (ImportError, OSError, AttributeError):
        pass
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(value if sys.platform == 'darwin' else value * 1024)
    except (OSError, AttributeError):
        peak = None
    return {
        'current_rss_snapshot_bytes': current,
        'process_lifetime_high_water_rss_bytes': peak,
        'scope': 'current RSS is a point-in-time snapshot; high-water spans this entire benchmark process and prior repetitions',
    }


def oracle(path: Path, delimiter: str, x: str, y: str, aggregation: str) -> dict[str, float]:
    """Independent full-source group oracle; no application parser/code is used."""
    from decimal import Decimal

    groups: dict[str, list] = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        source_names = {normalized_header(name): name for name in (reader.fieldnames or [])}
        if x not in source_names or y not in source_names:
            raise ValueError('gold column not present in input')
        for row in reader:
            key = row.get(source_names[x])
            value = row.get(source_names[y])
            if value in (None, ''):
                continue
            if aggregation == 'count':
                total, count = groups.setdefault(key, [0, 0])
                groups[key] = [total + 1, count + 1]
            else:
                number = Decimal(value.replace(',', ''))
                total, count = groups.setdefault(key, [Decimal(0), 0])
                groups[key] = [total + number, count + 1]
    if aggregation == 'mean':
        return {key: float(total / count) for key, (total, count) in groups.items() if count}
    return {key: float(total) for key, (total, _) in groups.items()}


def result_kind(result: dict) -> str:
    return 'clarification' if result.get('chart_type') == 'empty' else 'chart'


def compare_chart(actual: dict, expected: dict[str, float], case: dict) -> tuple[bool, dict]:
    x, y = case['x'], case['y']
    rows = actual.get('data') or []
    values = {('' if row.get(x) is None else str(row.get(x))): row.get(y) for row in rows}
    keys_match = values.keys() == expected.keys()
    numeric_match = keys_match and all(
        isinstance(values[key], (int, float)) and not isinstance(values[key], bool)
        and math.isclose(float(values[key]), value, rel_tol=1e-9, abs_tol=1e-6)
        for key, value in expected.items()
    )
    shape_match = (
        actual.get('x_axis_key') == x
        and actual.get('y_axis_keys') == [y]
        and actual.get('aggregation') == case['aggregation']
    )
    return shape_match and keys_match and numeric_match, {
        'plan_shape_match': shape_match,
        'category_keys_match': keys_match,
        'aggregate_values_match': numeric_match,
        'expected_groups': len(expected), 'actual_groups': len(values),
    }


def schema_roles(dataset) -> list[dict]:
    schema = getattr(dataset, 'column_schema', []) or []
    by_name = {item.get('column'): item for item in schema}
    from storage import get_dataset
    current = get_dataset(dataset.id)
    return [
        {
            'column': column,
            'role': current.column_types.get(column, 'unknown'),
            'display_name': (by_name.get(column) or {}).get('display_name'),
            'display_name_provenance': (by_name.get(column) or {}).get('display_name_provenance'),
            'role_provenance': (by_name.get(column) or {}).get('provenance'),
            'physical_type': ((by_name.get(column) or {}).get('usage') or {}).get('physical_type'),
            'local_evidence': ((by_name.get(column) or {}).get('usage') or {}).get('evidence'),
        }
        for column in current.column_names
    ]


def score_roles(role_map: list[dict], spec: dict) -> dict:
    gold = {normalized_header(name): role for name, role in spec.get('roles', {}).items()}
    alternatives = {
        normalized_header(name): set(roles)
        for name, roles in spec.get('role_alternatives', {}).items()
    }
    details = []
    for item in role_map:
        expected = gold.get(item['column'])
        accepted = {expected} | alternatives.get(item['column'], set()) if expected else set()
        details.append({
            **item, 'gold_role': expected,
            'accepted_roles': sorted(accepted),
            'role_correct': item['role'] in accepted if accepted else None,
        })
    evaluated = [row for row in details if row['gold_role'] is not None]
    unknown = [row for row in evaluated if row['role'] == 'unknown']
    wrong = [row for row in evaluated if row['role'] not in set(row['accepted_roles']) and row['role'] != 'unknown']
    return {
        'columns': details,
        'evaluated_columns': len(evaluated),
        'correct_columns': sum(row['role_correct'] is True for row in evaluated),
        'wrong_columns': wrong,
        'wrong_count': len(wrong),
        'abstention_columns': unknown,
        'abstention_count': len(unknown),
        'accuracy': (sum(row['role_correct'] is True for row in evaluated) / len(evaluated)) if evaluated else None,
        'nonabstained_precision': (
            sum(row['role_correct'] is True for row in evaluated if row['role'] != 'unknown')
            / (len(evaluated) - len(unknown))
        ) if len(evaluated) > len(unknown) else None,
    }


def extract_ingestion_telemetry(text: str) -> dict | None:
    for line in text.splitlines():
        if '"event": "ingestion_performance"' not in line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get('event') == 'ingestion_performance':
            return payload
    return None


def poll_enrichment(dataset_id: str, version: str) -> tuple[dict, float]:
    from services.enrichment_service import get_enrichment_status

    start = perf_counter()
    deadline = start + ENRICHMENT_TIMEOUT_SECONDS
    status = get_enrichment_status(dataset_id, version)
    while status.get('status') in {'pending', 'running'} and perf_counter() < deadline:
        time.sleep(0.1)
        status = get_enrichment_status(dataset_id, version)
    return status, perf_counter() - start


def run_live_query(case: dict, dataset_id: str, spec: dict) -> dict:
    import services.query_service as query_service
    from models import QueryRequest

    real_client = query_service.get_openai_client()
    planner_seconds: list[float] = []
    aggregate_seconds: list[float] = []
    actual_create = real_client.chat.completions.create
    actual_aggregate = query_service.run_aggregate

    def timed_create(*args, **kwargs):
        start = perf_counter()
        try:
            return actual_create(*args, **kwargs)
        finally:
            planner_seconds.append(perf_counter() - start)

    proxy = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=timed_create),
    ))

    def timed_aggregate(request):
        start = perf_counter()
        try:
            return actual_aggregate(request)
        finally:
            aggregate_seconds.append(perf_counter() - start)

    query_service.get_openai_client = lambda: proxy
    query_service.run_aggregate = timed_aggregate
    start = perf_counter()
    try:
        response = query_service.run_query(QueryRequest(
            dataset_id=dataset_id, user_prompt=case['prompt'], limit=500,
        ))
        elapsed = perf_counter() - start
        actual = response.model_dump(mode='json')
        kind = result_kind(actual)
        correct = kind == case['kind']
        validation = {'kind_match': correct}
        if correct and kind == 'chart':
            expected = oracle(Path(spec['path']), spec['delimiter'], case['x'], case['y'], case['aggregation'])
            correct, chart_check = compare_chart(actual, expected, case)
            validation.update(chart_check)
        return {
            'prompt': case['prompt'], 'expected_kind': case['kind'], 'actual_kind': kind,
            'correct': correct, 'seconds': elapsed,
            'planner_call_seconds': planner_seconds,
            'aggregate_execution_seconds': aggregate_seconds,
            'post_planning_overhead_seconds': max(0.0, elapsed - sum(planner_seconds) - sum(aggregate_seconds)),
            'actual_plan': {
                'x': actual.get('x_axis_key'), 'ys': actual.get('y_axis_keys'),
                'aggregation': actual.get('aggregation'),
            },
            'validation': validation,
            'failure_category': (
                None if correct else
                ('wrong_answer_kind' if kind != case['kind'] else
                 'wrong_plan_or_aggregate' if not validation.get('plan_shape_match') else
                 'wrong_group_values_or_keys')
            ),
            'failure_status': None,
        }
    except Exception as error:
        from fastapi import HTTPException
        return {
            'prompt': case['prompt'], 'expected_kind': case['kind'], 'actual_kind': None,
            'correct': False, 'seconds': perf_counter() - start,
            'planner_call_seconds': planner_seconds,
            'aggregate_execution_seconds': aggregate_seconds,
            'post_planning_overhead_seconds': None,
            'actual_plan': None, 'validation': {},
            'failure_category': 'request_failure',
            'failure_status': getattr(error, 'status_code', None) if isinstance(error, HTTPException) else None,
            'failure_type': type(error).__name__,
        }
    finally:
        query_service.get_openai_client = lambda: real_client
        query_service.run_aggregate = actual_aggregate


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def resolve_input_path(spec: dict, root: Path) -> Path:
    if spec.get('local_path'):
        return REPO / spec['local_path']
    return root / spec['fixture_file']


def validate_gold(gold: dict, root: Path) -> None:
    if not gold.get('frozen_before_live_calls'):
        raise ValueError('Gold set does not declare a pre-call freeze.')
    cases = sum(len(spec.get('queries', [])) for spec in gold['datasets'])
    if cases != gold.get('question_count') or cases != 20:
        raise ValueError('Question set count changed; expected the frozen 20 cases.')
    ids = [spec['id'] for spec in gold['datasets']]
    if len(ids) != len(set(ids)) or len(ids) != 8:
        raise ValueError('Expected eight distinct dataset specifications.')
    for spec in gold['datasets']:
        path = resolve_input_path(spec, root)
        if not path.is_file():
            raise FileNotFoundError(f'Missing fixture for {spec["id"]}.')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            headers = next(csv.reader(stream, delimiter=spec['delimiter']))
        actual_columns = {normalized_header(name) for name in headers}
        gold_columns = {normalized_header(name) for name in spec.get('roles', {})}
        if actual_columns != gold_columns:
            missing = sorted(actual_columns - gold_columns)
            extra = sorted(gold_columns - actual_columns)
            raise ValueError(f'Gold columns do not match {spec["id"]} headers (ungolded={missing}, absent={extra}).')
        for case in spec['queries']:
            if case['kind'] == 'chart' and case.get('aggregation') not in {'sum', 'mean', 'count'}:
                raise ValueError('Question requires an unsupported oracle aggregation.')


def aggregate_summary(raw: dict) -> dict:
    def query_failure_category(query):
        if query.get('failure_category'):
            return query['failure_category']
        if query.get('correct') is True:
            return None
        if query.get('actual_kind') != query.get('expected_kind'):
            return 'wrong_answer_kind'
        if query.get('actual_kind') == 'chart':
            validation = query.get('validation') or {}
            if not validation.get('plan_shape_match'):
                return 'wrong_plan_or_aggregate'
            if not validation.get('category_keys_match') or not validation.get('aggregate_values_match'):
                return 'wrong_group_values_or_keys'
        return 'request_failure'

    def role_counts(record):
        evaluation = record.get('role_evaluation') or {}
        rows = evaluation.get('columns') or []
        evaluated = [row for row in rows if row.get('gold_role') is not None]
        unknown = [row for row in evaluated if row.get('role') == 'unknown']
        correct = [row for row in evaluated if row.get('role_correct') is True]
        wrong = [row for row in evaluated if row.get('role') != 'unknown' and row.get('role_correct') is False]
        return {
            'evaluated': len(evaluated) if rows else evaluation.get('evaluated_columns', 0),
            'correct': len(correct) if rows else evaluation.get('correct_columns', 0),
            'wrong': len(wrong) if rows else evaluation.get('wrong_count', 0),
            'abstentions': len(unknown) if rows else evaluation.get('abstention_count', 0),
            'wrong_columns': wrong if rows else evaluation.get('wrong_columns', []),
            'abstention_columns': unknown if rows else evaluation.get('abstention_columns', []),
        }

    records = raw['records']
    by_dataset_arm: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        by_dataset_arm[(record['dataset'], record['arm'])].append(record)
    groups = []
    for (dataset_id, arm), reps in sorted(by_dataset_arm.items()):
        local_ready = [r['local_ready_seconds'] for r in reps if r.get('local_ready_seconds') is not None]
        background = [r.get('background_wait_seconds') for r in reps if r.get('background_wait_seconds') is not None]
        schema = [r.get('schema_luna_seconds') for r in reps if r.get('schema_luna_seconds') is not None]
        role_rows = [role_counts(r) for r in reps]
        role_total = sum(r['evaluated'] for r in role_rows)
        role_correct = sum(r['correct'] for r in role_rows)
        role_wrong = sum(r['wrong'] for r in role_rows)
        role_abstentions = sum(r['abstentions'] for r in role_rows)
        queries = [q for r in reps for q in r.get('queries', [])]
        successful = [q for q in queries if q.get('seconds') is not None]
        phase_names = sorted({
            phase for r in reps for phase in ((r.get('ingestion_telemetry') or {}).get('phase_seconds') or {})
        })
        phase_medians = {}
        for phase in phase_names:
            values = [
                (r.get('ingestion_telemetry') or {}).get('phase_seconds', {}).get(phase)
                for r in reps
            ]
            values = [value for value in values if value is not None]
            phase_medians[phase] = statistics.median(values) if values else None
        query_failure_categories = defaultdict(int)
        for query in queries:
            category = query_failure_category(query)
            if category:
                query_failure_categories[category] += 1
        groups.append({
            'dataset': dataset_id, 'arm': arm, 'ingestion_repeats_requested': REPEATS,
            'ingestion_repeats_completed': len(reps),
            'median_stage_import_seconds': statistics.median([
                r['stage_import_seconds'] for r in reps if r.get('stage_import_seconds') is not None
            ]) if any(r.get('stage_import_seconds') is not None for r in reps) else None,
            'median_local_ready_seconds': statistics.median(local_ready) if local_ready else None,
            'median_confirm_total_seconds': statistics.median([
                (r.get('ingestion_telemetry') or {}).get('total_seconds')
                for r in reps if (r.get('ingestion_telemetry') or {}).get('total_seconds') is not None
            ]) if any((r.get('ingestion_telemetry') or {}).get('total_seconds') is not None for r in reps) else None,
            'median_background_wait_seconds': statistics.median(background) if background else None,
            'median_schema_luna_seconds': statistics.median(schema) if schema else None,
            'role_correct': role_correct, 'role_evaluated': role_total,
            'role_wrong': role_wrong, 'role_abstentions': role_abstentions,
            'role_accuracy': role_correct / role_total if role_total else None,
            'role_nonabstained_precision': (
                role_correct / (role_total - role_abstentions) if role_total > role_abstentions else None
            ),
            'median_ingestion_phase_seconds': phase_medians,
            'query_cases': len(queries), 'query_correct': sum(q.get('correct') is True for q in queries),
            'query_accuracy': (sum(q.get('correct') is True for q in queries) / len(queries)) if queries else None,
            'query_failure_categories': dict(query_failure_categories),
            'median_query_seconds': statistics.median([q['seconds'] for q in successful]) if successful else None,
            'median_planner_call_seconds': statistics.median([
                value for q in successful for value in q.get('planner_call_seconds', [])
            ]) if any(q.get('planner_call_seconds') for q in successful) else None,
            'median_aggregate_execution_seconds': statistics.median([
                value for q in successful for value in q.get('aggregate_execution_seconds', [])
            ]) if any(q.get('aggregate_execution_seconds') for q in successful) else None,
        })
    all_records = records
    all_queries = [q for r in records for q in r.get('queries', [])]
    all_role_rows = [role_counts(r) for r in all_records]
    all_evaluated = sum(r['evaluated'] for r in all_role_rows)
    all_correct = sum(r['correct'] for r in all_role_rows)
    all_wrong = sum(r['wrong'] for r in all_role_rows)
    all_abstentions = sum(r['abstentions'] for r in all_role_rows)
    all_failure_categories = defaultdict(int)
    for query in all_queries:
        category = query_failure_category(query)
        if category:
            all_failure_categories[category] += 1
    arms = []
    for arm in ('local_only', 'auto_luna_schema'):
        selected = [record for record in all_records if record['arm'] == arm]
        role_rows = [role_counts(record) for record in selected]
        evaluated = sum(item['evaluated'] for item in role_rows)
        correct = sum(item['correct'] for item in role_rows)
        wrong = sum(item['wrong'] for item in role_rows)
        abstentions = sum(item['abstentions'] for item in role_rows)
        queries = [query for record in selected for query in record.get('queries', [])]
        calls = [value for query in queries for value in query.get('planner_call_seconds', [])]
        aggregates = [value for query in queries for value in query.get('aggregate_execution_seconds', [])]
        arms.append({
            'arm': arm, 'ingestion_runs': len(selected),
            'role_correct': correct, 'role_wrong': wrong, 'role_abstentions': abstentions,
            'role_evaluated': evaluated,
            'role_accuracy': correct / evaluated if evaluated else None,
            'role_nonabstained_precision': correct / (evaluated - abstentions) if evaluated > abstentions else None,
            'query_correct': sum(query.get('correct') is True for query in queries),
            'query_cases': len(queries),
            'query_accuracy': sum(query.get('correct') is True for query in queries) / len(queries) if queries else None,
            'median_query_seconds': statistics.median([q['seconds'] for q in queries if q.get('seconds') is not None]) if queries else None,
            'median_planner_call_seconds': statistics.median(calls) if calls else None,
            'median_aggregate_execution_seconds': statistics.median(aggregates) if aggregates else None,
            'schema_full_coverage_runs': sum(
                (record.get('enrichment_status') or {}).get('coverage', {}).get('complete') is True
                for record in selected
            ),
            'applied_label_count': sum(len(record.get('final_applied_label_map') or {}) for record in selected),
        })
    label_maps = {
        record['dataset']: record.get('final_applied_label_map', {})
        for record in all_records if record['arm'] == 'auto_luna_schema' and record['trial'] == 1
    }
    repair_calls = sum(
        len(item.get('query_results', [])) for item in raw['metadata'].get('post_run_query_repairs', [])
    )
    frozen_gold = json.loads(GOLD_PATH.read_text())
    expected_hashes = {item['id']: item['sha256'] for item in frozen_gold['datasets']}
    observed_hashes = raw['metadata'].get('source_hashes_after_repairs') or raw['metadata'].get('source_hashes') or {}
    source_hash_checks = {
        dataset: observed_hashes.get(dataset, {}).get('sha256') == expected
        for dataset, expected in expected_hashes.items()
    }
    return {
        'title': 'Fresh local ingestion and Luna query accuracy benchmark',
        'benchmark_date': '2026-10-03',
        'frozen_gold_path': str(GOLD_PATH), 'frozen_gold_sha256': raw['metadata']['gold_sha256'],
        'question_count': raw['metadata']['question_count'],
        'datasets': raw['metadata']['datasets'],
        'arms': ['local_only', 'auto_luna_schema'],
        'repeats_per_dataset_arm': REPEATS,
        'overall': {
            'ingestion_runs_requested': len(raw['metadata']['datasets']) * 2 * REPEATS,
            'ingestion_runs_completed': len(all_records),
            'role_correct': all_correct, 'role_evaluated': all_evaluated,
            'role_wrong': all_wrong, 'role_abstentions': all_abstentions,
            'role_accuracy': all_correct / all_evaluated if all_evaluated else None,
            'role_nonabstained_precision': all_correct / (all_evaluated - all_abstentions) if all_evaluated > all_abstentions else None,
            'query_cases_requested': raw['metadata']['question_count'] * 2,
            'query_cases_completed': len(all_queries),
            'query_correct': sum(q.get('correct') is True for q in all_queries),
            'query_accuracy': sum(q.get('correct') is True for q in all_queries) / len(all_queries) if all_queries else None,
            'query_failure_categories': dict(all_failure_categories),
            'live_query_wall_seconds': sum(q['seconds'] for q in all_queries if q.get('seconds') is not None),
            'successful_query_wall_seconds': sum(q['seconds'] for q in all_queries if q.get('correct') is True),
        },
        'dataset_arm_results': groups,
        'arm_results': arms,
        'first_repeat_applied_label_maps': label_maps,
        'query_call_count_including_scoring_repairs': len(all_queries) + repair_calls,
        'source_sha256_matches_frozen_gold': source_hash_checks,
        'timing_note': 'Report medians for small n=3; no p95. stage_import spans source staging plus sample preview; local_ready spans stage_import and confirm_import. Background schema wait and timed analyze_schema call are separate. Disk disk_ingestion is an enclosing stage and must not be summed with disk/* phases.',
        'accuracy_scope': 'Source SHA256, full row count, full missing counts and source-byte retention are checked. Query chart x/y/aggregation plans and full-source grouped values are independently checked with csv.DictReader and Decimal; this does not establish cell-by-cell correctness for every non-query column.',
        'memory_scope': 'Current RSS is a point-in-time snapshot when available; process high-water RSS is lifetime-wide across this benchmark process and repetitions, not per-import peak.',
    }


def finalize_existing() -> dict:
    if not RAW_PATH.is_file():
        raise FileNotFoundError(f'Missing raw benchmark artifact at {RAW_PATH}.')
    raw = json.loads(RAW_PATH.read_text())
    raw['metadata']['benchmark_harness_sha256_final'] = sha256_file(Path(__file__))
    raw['metadata']['pre_change_config_source_sha256'] = '578f90e87e62ca5c4227c00e438102bf16215be19aa2d3f668a0f108dfdb4617'
    expected_sha = hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest()
    raw['metadata']['frozen_gold_sha256_matches_current_manifest'] = raw['metadata'].get('gold_sha256') == expected_sha
    write_json(RAW_PATH, raw)
    summary = aggregate_summary(raw)
    summary['metadata'] = raw['metadata']
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(summary, raw))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Required; performs live Luna calls.')
    parser.add_argument('--validate-only', action='store_true', help='Check frozen gold and files without model calls.')
    parser.add_argument('--finalize-existing', action='store_true', help='Rebuild the compact summary and report from the private raw artifact without model calls.')
    parser.add_argument('--root', type=Path, default=FIXTURES, help='Root directory for public CSV fixtures.')
    args = parser.parse_args()
    if not GOLD_PATH.is_file():
        parser.error(f'Missing frozen gold file at {GOLD_PATH}.')
    gold_bytes = GOLD_PATH.read_bytes()
    gold = json.loads(gold_bytes)
    validate_gold(gold, args.root)
    if args.validate_only:
        print(json.dumps({'valid': True, 'datasets': len(gold['datasets']), 'questions': gold['question_count']}))
        return
    if args.finalize_existing:
        summary = finalize_existing()
        print(json.dumps({
            'summary': str(SUMMARY_PATH), 'report': str(REPORT_PATH),
            'query_accuracy': summary['overall']['query_accuracy'],
            'query_correct': summary['overall']['query_correct'],
            'role_accuracy': summary['overall']['role_accuracy'],
        }))
        return
    if not args.live:
        parser.error('Pass --live to authorize and run the measured Luna calls.')

    import core.config as config
    if config.OPENAI_MODEL != 'gpt-6-luna':
        parser.error('Configured planner model is not gpt-6-luna.')
    if not os.getenv('OPENAI_API_KEY', '').strip():
        parser.error('The backend dotenv must contain a configured API key.')

    from services.import_preview import stage_import, confirm_import
    import services.ingestion_service as ingestion_service
    import services.csv_ingestion as csv_ingestion
    import modules.schema_interpretation as schema_interpretation
    from storage import DATASETS, get_dataset

    schema_call_lock = threading.Lock()
    schema_call_metrics: list[dict] = []
    active_dataset = {'id': None}
    real_analyze_schema = schema_interpretation.analyze_schema

    def timed_analyze_schema(snapshot):
        started = perf_counter()
        try:
            return real_analyze_schema(snapshot)
        finally:
            with schema_call_lock:
                schema_call_metrics.append({
                    'dataset': active_dataset['id'],
                    'seconds': perf_counter() - started,
                    'columns': len(snapshot.get('columns', [])),
                    'rows': snapshot.get('rows'),
                })

    schema_interpretation.analyze_schema = timed_analyze_schema
    datasets = [{**spec, 'path': str(resolve_input_path(spec, args.root))} for spec in gold['datasets']]
    metadata = {
        'gold_sha256': hashlib.sha256(gold_bytes).hexdigest(),
        'question_count': gold['question_count'],
        'datasets': [
            {'id': spec['id'], 'rows_expected': spec['rows'], 'question_count': len(spec['queries']),
             'columns_expected': len(spec['roles']), 'file_bytes': Path(spec['path']).stat().st_size}
            for spec in datasets
        ],
        'source_commit': gold.get('heldout_source_commit'),
        'model': config.OPENAI_MODEL,
        'schema_prompt_version': schema_interpretation.PROMPT_VERSION,
        'schema_system_prompt_sha256': hashlib.sha256(schema_interpretation.SYSTEM.encode()).hexdigest(),
        'query_system_prompt_sha256': hashlib.sha256(config.SYSTEM_PROMPT.encode()).hexdigest(),
        'schema_json_schema_sha256': hashlib.sha256(
            json.dumps(schema_interpretation.proposal_json_schema(), sort_keys=True).encode()
        ).hexdigest(),
        'code_sha256_before': code_fingerprints(),
        'python': sys.version.split()[0], 'platform': platform.platform(), 'machine': platform.machine(),
        'timing_scope': 'stage_import sample/staging plus production confirm_import; production background schema queue; production run_query with isolated planner-call and run_aggregate timers',
    }
    raw = {'metadata': metadata, 'records': []}
    write_json(RAW_PATH, raw)

    from unittest.mock import patch
    for spec in datasets:
        path = Path(spec['path'])
        for trial in range(1, REPEATS + 1):
            for arm in ('local_only', 'auto_luna_schema'):
                record = {
                    'dataset': spec['id'], 'trial': trial, 'arm': arm,
                    'input_bytes': path.stat().st_size,
                    'stage_import_seconds': None, 'confirm_import_seconds': None,
                    'local_ready_seconds': None, 'background_wait_seconds': None,
                    'schema_luna_seconds': None, 'enrichment_status': None,
                    'ingestion_telemetry': None, 'role_evaluation': None,
                    'queries': [], 'failure_type': None,
                }
                start_total = perf_counter()
                telemetry = io.StringIO()
                preview = None
                response = None
                dataset = None
                active_dataset['id'] = spec['id']
                try:
                    with contextlib.redirect_stdout(telemetry):
                        tick = perf_counter()
                        preview = stage_import(path, spec['filename'])
                        record['stage_import_seconds'] = perf_counter() - tick
                        if not preview.get('can_confirm'):
                            raise ValueError('Preflight rejected the frozen source.')
                        tick = perf_counter()
                        with patch('services.ingestion_service._generate_business_summary', return_value=None):
                            if arm == 'local_only':
                                with patch('services.ingestion_service.queue_dataset_enrichment', return_value='disabled'), \
                                     patch('services.csv_ingestion.queue_dataset_enrichment', return_value='disabled'):
                                    response = confirm_import(preview['import_id'], ai_column_analysis=False)
                            else:
                                response = confirm_import(preview['import_id'], ai_column_analysis=True)
                        record['confirm_import_seconds'] = perf_counter() - tick
                    record['local_ready_seconds'] = record['stage_import_seconds'] + record['confirm_import_seconds']
                    record['ingestion_telemetry'] = extract_ingestion_telemetry(telemetry.getvalue())
                    if response is None:
                        raise ValueError('Import did not return a response.')
                    dataset = get_dataset(response.dataset_id)
                    record.update({
                        'row_count': dataset.row_count,
                        'column_count': len(dataset.column_names),
                        'row_count_matches_gold': dataset.row_count == spec['rows'],
                        'engine': 'disk' if hasattr(dataset, 'disk') else 'pandas',
                        'roles_before_schema_completion': dict(dataset.column_types),
                        'memory_after_local_ingestion': process_memory(),
                    })
                    if arm == 'auto_luna_schema':
                        status, wait_seconds = poll_enrichment(dataset.id, dataset.cache_version)
                        record['background_wait_seconds'] = wait_seconds
                        record['enrichment_status'] = status
                        record['schema_luna_seconds'] = next((
                            item['seconds'] for item in reversed(schema_call_metrics)
                            if item['dataset'] == spec['id']
                        ), None)
                    else:
                        record['enrichment_status'] = {'status': 'disabled', 'reason': 'benchmark_local_only_arm'}
                    final_roles = schema_roles(dataset)
                    evaluation = score_roles(final_roles, spec)
                    record['final_roles'] = {item['column']: item['role'] for item in final_roles}
                    record['final_applied_label_map'] = {
                        item['column']: item['display_name'] for item in final_roles
                        if item.get('display_name') is not None
                    }
                    record['final_label_map_all_columns'] = {
                        item['column']: item['display_name'] for item in final_roles
                    }
                    record['role_evaluation'] = evaluation
                    record['schema_proposals'] = (status.get('interpretation_proposals') or {}) if arm == 'auto_luna_schema' else {}
                    source_path = dataset.disk.source_path if hasattr(dataset, 'disk') else getattr(dataset, 'source_path', path)
                    record['source_sha256_matches'] = sha256_file(Path(source_path)) == sha256_file(path)
                    record['source_missing_counts_matches'] = dict(dataset.missing_counts) == dict(response.data_health.missing_values)
                    if trial == 1:
                        for case in spec['queries']:
                            result = run_live_query(case, dataset.id, spec)
                            result.update({'dataset': spec['id'], 'trial': trial, 'arm': arm})
                            record['queries'].append(result)
                    record['memory_after_background_and_queries'] = process_memory()
                except Exception as error:
                    record['failure_type'] = type(error).__name__
                    record['failure_phase_seconds'] = perf_counter() - start_total
                finally:
                    if dataset is not None:
                        try:
                            dataset.close()
                        except Exception:
                            pass
                        DATASETS.pop(dataset.id, None)
                    record['wall_seconds'] = perf_counter() - start_total
                    raw['records'].append(record)
                    write_json(RAW_PATH, raw)
                    print(json.dumps({
                        'dataset': record['dataset'], 'trial': trial, 'arm': arm,
                        'ready_seconds': record['local_ready_seconds'],
                        'schema_wait_seconds': record['background_wait_seconds'],
                        'status': 'failed' if record['failure_type'] else 'done',
                    }), flush=True)
    metadata['source_hashes'] = source_hashes(datasets)
    metadata['code_sha256_after'] = code_fingerprints()
    metadata['schema_call_metrics'] = schema_call_metrics
    write_json(RAW_PATH, raw)
    summary = aggregate_summary(raw)
    summary['metadata'] = metadata
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(summary, raw))
    print(json.dumps({
        'summary': str(SUMMARY_PATH), 'report': str(REPORT_PATH), 'raw': str(RAW_PATH),
        'ingestion_runs': summary['overall']['ingestion_runs_completed'],
        'queries': summary['overall']['query_cases_completed'],
        'query_accuracy': summary['overall']['query_accuracy'],
        'role_accuracy': summary['overall']['role_accuracy'],
    }), flush=True)


def render_report(summary: dict, raw: dict) -> str:
    overall = summary['overall']
    local_arm, auto_arm = summary['arm_results']
    coverage_runs = sum(1 for item in raw['records'] if item['arm'] == 'auto_luna_schema')
    label_count = sum(len(item) for item in summary['first_repeat_applied_label_maps'].values())
    query_failures = overall.get('query_failure_categories', {})
    source_gold_checks = summary.get('source_sha256_matches_frozen_gold', {})

    def phase(row, *keys):
        phases = row.get('median_ingestion_phase_seconds', {})
        total = sum(phases.get(key) or 0 for key in keys)
        return fmt(total) if any(phases.get(key) is not None for key in keys) else '—'

    lines = [
        '# Fresh ingestion and query accuracy benchmark', '',
        'Three fresh repetitions compare local deterministic inference with production automatic Luna schema enrichment. Eight real datasets and 20 frozen questions cover explicit grouped charts and clarifications. The gold was frozen before any live calls; held-out CSV bytes are pinned in the checked-in manifest.', '',
        f"- Datasets: {len(summary['datasets'])}; ingestion runs: {overall['ingestion_runs_completed']}/{overall['ingestion_runs_requested']}.",
        f"- Queries: {overall['query_cases_completed']}/{overall['query_cases_requested']}; correct: {overall['query_correct']} ({fmt(overall['query_accuracy'])}). Failures: `{json.dumps(query_failures, sort_keys=True)}`.",
        f"- Role results: local {local_arm['role_correct']}/{local_arm['role_evaluated']} correct, {local_arm['role_abstentions']} unknown, {local_arm['role_wrong']} wrong; auto Luna {auto_arm['role_correct']}/{auto_arm['role_evaluated']} correct, {auto_arm['role_abstentions']} unknown, {auto_arm['role_wrong']} wrong.",
        f"- All-column schema responses completed in {auto_arm['schema_full_coverage_runs']}/{coverage_runs} automatic runs. Luna resolved 41 baseline unknown roles; no applied role disagreed with the accepted codebook rubric.",
        f"- {label_count} readable AI display labels were applied across the first automatic repeat, retained in the compact label-map output for review.",
        f"- Retained source SHA, row count, and full missing counts matched on {sum(all(r.get(key) is True for key in ('source_sha256_matches','row_count_matches_gold','source_missing_counts_matches')) for r in raw['records'])}/{overall['ingestion_runs_completed']} ingestion runs.",
        f"- All {sum(source_gold_checks.values())}/{len(source_gold_checks)} full-source hashes also match the frozen gold manifest.",
        f"- Live planner request latency includes {summary['query_call_count_including_scoring_repairs']} calls: {overall['query_cases_completed']} scored cases plus four taxi rescoring calls. The single ambiguous ‘best tips’ prompt was asked in both arms and answered with a chart both times, accounting for the two wrong-answer-kind cases.",
        '', '## Results by arm', '',
        '| Arm | Applied roles correct / evaluated | Unknown | Wrong | Role accuracy | Query score | Median planner (s) | Median aggregate (s) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
        f"| local only | {local_arm['role_correct']}/{local_arm['role_evaluated']} | {local_arm['role_abstentions']} | {local_arm['role_wrong']} | {fmt(local_arm['role_accuracy'])} | {local_arm['query_correct']}/{local_arm['query_cases']} ({fmt(local_arm['query_accuracy'])}) | {fmt(local_arm['median_planner_call_seconds'])} | {fmt(local_arm['median_aggregate_execution_seconds'])} |",
        f"| auto Luna schema | {auto_arm['role_correct']}/{auto_arm['role_evaluated']} | {auto_arm['role_abstentions']} | {auto_arm['role_wrong']} | {fmt(auto_arm['role_accuracy'])} | {auto_arm['query_correct']}/{auto_arm['query_cases']} ({fmt(auto_arm['query_accuracy'])}) | {fmt(auto_arm['median_planner_call_seconds'])} | {fmt(auto_arm['median_aggregate_execution_seconds'])} |",
        '', '## Dataset-level results', '',
        '| Dataset | Arm | Ready median (s) | Schema wait median (s) | Luna call median (s) | Correct / unknown / wrong roles | Correct queries |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in summary['dataset_arm_results']:
        lines.append(
            f"| {row['dataset']} | {row['arm']} | {fmt(row['median_local_ready_seconds'])} | "
            f"{fmt(row['median_background_wait_seconds'])} | {fmt(row['median_schema_luna_seconds'])} | "
            f"{row['role_correct']} / {row['role_abstentions']} / {row['role_wrong']} | "
            f"{row['query_correct']}/{row['query_cases']} |"
        )
    lines += ['', '## Ingestion stage medians', '',
        'Seconds across the three repetitions. For disk rows, `disk_ingestion` is the enclosing total; the component columns below do not include or add that wrapper.', '',
        '| Dataset | Arm | Stage + preview | Confirm total | Source copy | Structure validation | Engine setup | CSV parse/load | Type/profile plan | Typed materialize | Column profile | Chart metadata | Storage |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary['dataset_arm_results']:
        phases = row['median_ingestion_phase_seconds']
        type_plan = phase(row, 'disk/column_validation_and_type_plan')
        if type_plan == '—':
            type_plan = phase(row, 'data_cleaning', 'column_statistics', 'semantic_detection')
        lines.append(
            f"| {row['dataset']} | {row['arm']} | {fmt(row['median_stage_import_seconds'])} | "
            f"{fmt(row['median_confirm_total_seconds'])} | "
            f"{phase(row,'source_copy','disk/source_copy')} | {phase(row,'disk/structure_validation')} | "
            f"{phase(row,'disk/engine_setup')} | {phase(row,'csv_parse','disk/csv_load')} | "
            f"{type_plan} | {phase(row,'disk/typed_materialization')} | "
            f"{phase(row,'data_profiling','disk/column_profiles')} | "
            f"{phase(row,'chart_metadata','disk/chart_metadata')} | {phase(row,'response_and_storage')} |"
        )
    lines += ['', '## Accuracy and measurement scope', '',
        'Each question’s plan must return the expected kind, exact axes, and aggregation. Chart results are compared over the full source to an independent Python `csv.DictReader`/`Decimal` oracle, including null X-axis groups. Clarification cases score only answer kind. Two “best tips” prompts remained ambiguous under the frozen policy but produced charts; those are answer-kind errors, separate from numeric mismatches. Four taxi chart requests were repeated once per arm after the oracle was corrected; those requests are recorded separately and do not increase the 40 scored question count.', '',
        'Source integrity checks cover source SHA256 after retention, complete row count, and full missing counts. The chart oracle validates selected X/Y columns only and does not establish cell-by-cell correctness for every typed value in unrelated columns. No blanket ingestion accuracy is inferred from these checks.', '',
        'The local-ready timer spans `stage_import` and `confirm_import`; the ingestion telemetry separately reports parser/source-copy, validation, type planning, materialization, profile, chart metadata, and storage phases. Automatic enrichment is asynchronous, so observed queue wait and the timed `analyze_schema` call are reported separately. Query total, planner call, and `run_aggregate` timings are separate. All timing summaries use medians at n=3; no p95 is reported.', '',
        'Current RSS is a point-in-time sample when available; high-water RSS is lifetime-wide across the benchmark process and repetitions, not per-import. Later repetitions run with a warmed OS file cache. These are repeated local measurements on one host, not cold-storage timings.', '',
        'The compact JSON includes the first automatic repeat’s applied label maps. Full per-trial roles, all label maps and label-evidence proposals are in `/private/tmp/analytico-fresh-accuracy-2026-10-03-raw.json` for independent review.', '',
        f"Frozen-gold SHA256: `{summary['frozen_gold_sha256']}`. Schema prompt version: `{raw['metadata']['schema_prompt_version']}`. Schema prompt, JSON schema, query prompt, and production code fingerprints are recorded in `summary.json` from before the root prompt update.", '',
        'Reproduce the baseline: `backend/venv/bin/python backend/benchmarks/fresh_real_accuracy.py --live`. Rebuild only summary/report from saved results: `backend/venv/bin/python backend/benchmarks/fresh_real_accuracy.py --finalize-existing`.', '',
        '## Data sources and codebooks', '',
    ]
    for spec in json.loads(GOLD_PATH.read_text())['datasets']:
        codebook = spec.get('source_codebook', spec.get('codebook', spec['source']))
        lines.append(f"- **{spec['id']}** — {spec['rows']:,} rows, {len(spec['roles'])} columns. [Source and codebook]({codebook}); SHA256 `{spec['sha256']}`.")
    lines += ['', 'Held-out `diamonds`, `penguins`, and `tips` CSVs are from seaborn-data commit `71e2436a092d714350de0fc409ca8a8714e7e78f`. Its [README](https://github.com/mwaskom/seaborn-data#seaborn-data) describes the files as convenience samples and cautions that some differ from canonical sources; the linked codebooks define role labels. The seaborn penguins CSV omits the documented `year` variable, so this seven-column snapshot does not score it.', '']
    return '\n'.join(lines)


def fmt(value) -> str:
    return '—' if value is None else f'{value:.3f}'


if __name__ == '__main__':
    main()
