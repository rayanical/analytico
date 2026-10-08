"""Paired live and offline calibration benchmark for schema role proposals."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import platform
import random
import statistics
import sys
from pathlib import Path
from time import perf_counter
from uuid import uuid4

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
GOLD = BACKEND / 'evals/fresh_accuracy_gold_2026-10-03.json'
MANIFEST = BACKEND / 'evals/real_dataset_manifest.json'
LIVE_OUTPUT = Path('/private/tmp/analytico-decisions-roles-2026-10-06.json')
REPLAY_OUTPUT = Path('/private/tmp/analytico-decisions-calibrated-2026-10-06.json')
ARMS = ('responses', 'decisions')
GRID = (0.8, 0.85, 0.9, 0.95)
CALIBRATION = ('bike', 'bank', 'retail', 'taxi', 'wine')


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def value_hash(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      default=str, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def load_specs() -> tuple[list[dict], dict]:
    gold = json.loads(GOLD.read_text())
    manifest = json.loads(MANIFEST.read_text())
    by_id = {row['id']: row for row in manifest['datasets']}
    if len(gold['datasets']) != 8:
        raise ValueError('Expected eight frozen gold datasets.')
    specs = []
    for row in gold['datasets']:
        book = by_id.get(row['id'], row)
        specs.append({
            **row,
            'roles': book.get('roles', row.get('roles', {})),
            'role_alternatives': book.get('role_alternatives', row.get('role_alternatives', {})),
            'opaque_columns': book.get('opaque_columns', row.get('opaque_columns', [])),
            'gold_source': ('real_dataset_manifest' if row['id'] in by_id and book.get('roles')
                            else 'fresh_accuracy_gold'),
        })
    return specs, {'fresh_accuracy_gold_sha256': file_hash(GOLD),
                   'real_dataset_manifest_sha256': file_hash(MANIFEST)}


def rubric(spec: dict) -> tuple[dict, dict, set]:
    from modules.data_janitor import legacy_normalize_header

    gold = {legacy_normalize_header(k): v for k, v in spec['roles'].items()}
    alternatives = {legacy_normalize_header(k): set(v)
                    for k, v in spec.get('role_alternatives', {}).items()}
    opaque = {legacy_normalize_header(k) for k in spec.get('opaque_columns', [])}
    return gold, alternatives, opaque


def score(roles: dict, spec: dict) -> dict:
    gold, alternatives, opaque = rubric(spec)
    correct = unknown = wrong = unavailable = opaque_unknown = nonabstained = nonabstained_correct = 0
    for column, expected in gold.items():
        role = roles.get(column)
        if role in ({expected} | alternatives.get(column, set())):
            correct += 1
            nonabstained_correct += 1
        elif role == 'unknown':
            unknown += 1
            opaque_unknown += column in opaque
        elif role is None:
            unavailable += 1
        else:
            wrong += 1
        nonabstained += role not in (None, 'unknown')
    total = len(gold)
    return {
        'evaluated_columns': total, 'correct_columns': correct, 'wrong_count': wrong,
        'unknown_count': unknown, 'unavailable_count': unavailable,
        'accuracy_including_abstentions': correct / total if total else None,
        'coverage': nonabstained / total if total else None,
        'nonabstained_precision': nonabstained_correct / nonabstained if nonabstained else None,
        'opaque_column_count': len(opaque & gold.keys()), 'opaque_unknown_count': opaque_unknown,
    }


def proposal_roles(proposals: dict, names: list[str]) -> dict:
    return {name: ((proposals.get(name) or {}).get('decision') or {}).get('role') for name in names}


def source_schema(dataset):
    if hasattr(dataset, 'disk'):
        return dataset.disk.column_schema, dataset.disk.column_types, dataset.disk.column_stats
    return dataset.column_schema, dataset.column_types, dataset.column_stats


def fake_dataset(dataset, snapshot):
    """Clone source metadata into the production apply guard without disk IO."""
    import pandas as pd
    from types import SimpleNamespace

    schema, types, stats = source_schema(dataset)
    names = list(dataset.column_names)
    if names != [item['column'] for item in snapshot['columns']]:
        raise ValueError('Column order changed between ingestion and snapshot.')
    dtypes = {'number': 'float64', 'datetime': 'datetime64[ns]',
              'boolean': 'bool', 'text': 'object'}
    physical = {item['column']: item['usage']['physical_type'] for item in schema}
    frame = pd.DataFrame({name: pd.Series(dtype=dtypes.get(physical.get(name), 'object'))
                          for name in names})
    frame.attrs['column_schema'] = copy.deepcopy(schema)
    frame.attrs['column_usages'] = {}
    return SimpleNamespace(
        id='role-benchmark-' + uuid4().hex, cache_version='benchmark-version',
        semantic_revision=0, disk=None, df=frame,
        column_schema=copy.deepcopy(schema), column_types=copy.deepcopy(types),
        column_stats=copy.deepcopy(dict(stats)),
    )


def apply_roles(dataset, snapshot, result) -> tuple[dict, float]:
    import modules.schema_interpretation as semantic
    from storage import DATASETS

    fake = fake_dataset(dataset, snapshot)
    DATASETS[fake.id] = fake
    started = perf_counter()
    try:
        semantic.apply_schema_roles(fake.id, fake.cache_version, result)
        return dict(fake.column_types), perf_counter() - started
    finally:
        DATASETS.pop(fake.id, None)


def role_changes(applied: dict, original: dict, spec: dict) -> tuple[list, list]:
    gold, alternatives, _ = rubric(spec)
    changed, wrong = [], []
    for name, role in applied.items():
        previous = original.get(name)
        if role == previous:
            continue
        changed.append({'column': name, 'from': previous, 'to': role})
        if name in gold and role not in ({gold[name]} | alternatives.get(name, set())):
            wrong.append({'column': name, 'from': previous, 'to': role})
    return changed, wrong


def call_health(call: dict) -> dict:
    coverage = call.get('coverage')
    proposals = call.get('raw_proposals') or {}
    failed_status = {'error', 'unavailable', 'disabled'}
    failed_proposals = sum(p.get('status') in failed_status for p in proposals.values())
    reported = (coverage.get('failed_columns', 0) if isinstance(coverage, dict)
                and isinstance(coverage.get('failed_columns'), int) else 0)
    incomplete = isinstance(coverage, dict) and coverage.get('complete') is False
    return {
        'request_failed': bool(call.get('classifier_error_type') or failed_proposals or reported or incomplete),
        'request_failed_columns': max(failed_proposals, reported),
        'coverage_complete': coverage.get('complete') if isinstance(coverage, dict) else None,
        'coverage_failed_columns_reported': reported,
        'model_refusal_columns': sum(p.get('error_code') == 'model_refusal' for p in proposals.values()),
    }


def live(args) -> dict:
    """Ingest each fixture once and run randomized paired live classifier calls."""
    from core import config
    import modules.schema_decisions as decisions
    from modules import schema_interpretation as responses
    from modules.schema_interpretation import dataset_snapshot
    from modules.import_policy import ImportSettings
    from services.csv_ingestion import ingest_csv
    from storage import DATASETS, get_dataset
    from ingestion_parallelism import spec_path

    if not os.getenv('OPENAI_API_KEY', '').strip():
        raise ValueError('--live requires OPENAI_API_KEY.')
    specs, manifests = load_specs()
    if args.datasets:
        chosen = set(args.datasets)
        unknown = chosen - {s['id'] for s in specs}
        if unknown:
            raise ValueError('Unknown dataset IDs: ' + ', '.join(sorted(unknown)))
        specs = [s for s in specs if s['id'] in chosen]
    if not specs:
        raise ValueError('Select at least one dataset.')

    output = {
        'status': 'running', 'methodology': {
            'min_confidence': args.min_confidence,
            'design': 'randomized paired calls over one shared complete snapshot per dataset',
            'apply': 'production apply_schema_roles on cloned fake dataset',
        },
        'runtime': {'python': sys.version.split()[0], 'platform': platform.platform(),
                    'configured_model': config.OPENAI_MODEL},
        'manifest_hashes': manifests,
        'parameters': {'repeats': args.repeats, 'min_confidence': args.min_confidence,
                       'label_repeats': args.label_repeats if args.labels else 0,
                       'seed': args.seed, 'datasets': [s['id'] for s in specs]},
        'datasets': [], 'summary': None,
    }
    write_json(args.output, output)
    rng = random.Random(args.seed)

    for spec in specs:
        path = spec_path(spec)
        digest = file_hash(path)
        record = {
            'dataset': spec['id'], 'fixture': spec.get('local_path') or spec.get('fixture_file'),
            'gold_source': spec['gold_source'], 'source_sha256': digest,
            'source_bytes': path.stat().st_size, 'expected_source_sha256': spec.get('sha256'),
            'source_sha256_matches_gold': digest == spec.get('sha256'),
            'expected_rows': spec.get('rows'), 'row_count': None, 'column_count': None,
            'ingestion_seconds': None, 'original_roles': {}, 'snapshot_sha256': None,
            'snapshot_unchanged_after_all_calls': None, 'local_schema_unchanged': None,
            'calls': [], 'labels': [], 'status': 'running',
        }
        output['datasets'].append(record)
        write_json(args.output, output)
        dataset = None
        try:
            started = perf_counter()
            with path.open('rb') as source, contextlib.redirect_stdout(io.StringIO()):
                upload = ingest_csv(source, spec['filename'], '/decisions-roles-benchmark',
                                    import_settings=ImportSettings(delimiter=spec['delimiter']),
                                    enqueue_enrichment=False, ai_column_analysis=False)
            dataset = get_dataset(upload.dataset_id)
            record.update(ingestion_seconds=perf_counter() - started,
                          row_count=dataset.row_count, column_count=len(dataset.column_names))
            if dataset.row_count != spec.get('rows') or digest != spec.get('sha256'):
                raise ValueError('Fixture hash or row count differs from frozen gold.')
            snapshot = dataset_snapshot(dataset, None if hasattr(dataset, 'disk') else dataset.raw_df)
            snapshot_hash = value_hash(snapshot)
            schema, original, _ = source_schema(dataset)
            schema_hash = value_hash({'schema': schema, 'roles': original})
            original = dict(original)
            record.update(snapshot_sha256=snapshot_hash, original_roles=original)
            names = list(dataset.column_names)

            for repeat in range(1, args.repeats + 1):
                order = list(ARMS)
                rng.shuffle(order)
                for arm in order:
                    view = copy.deepcopy(snapshot)
                    before = value_hash(view)
                    started = perf_counter()
                    error = None
                    try:
                        result = (responses.analyze_schema(view) if arm == 'responses'
                                  else decisions.analyze_schema_decisions(
                                      view, min_confidence=args.min_confidence))
                    except Exception as exc:
                        result, error = None, type(exc).__name__
                    elapsed = perf_counter() - started
                    proposals = (result or {}).get('interpretation_proposals', {})
                    raw_score = score(proposal_roles(proposals, names), spec)
                    applied, apply_seconds, apply_error = dict(original), None, None
                    if result is not None:
                        try:
                            applied, apply_seconds = apply_roles(dataset, snapshot, result)
                        except Exception as exc:
                            apply_error = type(exc).__name__
                    applied_score = score(applied, spec)
                    changed, wrong_new = role_changes(applied, original, spec)
                    call = {
                        'repetition': repeat, 'randomized_pair_order': order, 'arm': arm,
                        'classifier_wall_seconds': elapsed, 'role_application_wall_seconds': apply_seconds,
                        'classifier_error_type': error, 'role_application_error_type': apply_error,
                        'coverage': (result or {}).get('coverage'), 'usage': (result or {}).get('usage'),
                        'snapshot_unchanged_by_call': before == value_hash(view) == snapshot_hash,
                        'raw_role_score': raw_score, 'applied_role_score': applied_score,
                        'applied_roles': applied, 'changed_roles_vs_original': changed,
                        'wrong_new_changes_vs_original': wrong_new, 'raw_proposals': proposals,
                    }
                    call.update(call_health(call))
                    record['calls'].append(call)
                    write_json(args.output, output)
                    print(json.dumps({'dataset': spec['id'], 'arm': arm, 'repetition': repeat,
                                      'seconds': round(elapsed, 3), 'correct': raw_score['correct_columns'],
                                      'unknown': raw_score['unknown_count'], 'wrong': raw_score['wrong_count'],
                                      'errors': error or apply_error}), flush=True)

            if args.labels:
                for repeat in range(1, args.label_repeats + 1):
                    view, started = copy.deepcopy(snapshot), perf_counter()
                    try:
                        result = responses.analyze_schema_labels(view)
                        error = None
                    except Exception as exc:
                        result, error = None, type(exc).__name__
                    label = {'repetition': repeat, 'classifier_wall_seconds': perf_counter() - started,
                             'classifier_error_type': error, 'coverage': (result or {}).get('coverage'),
                             'raw_label_proposals': (result or {}).get('interpretation_proposals', {})}
                    record['labels'].append(label)
                    write_json(args.output, output)
                    print(json.dumps({'dataset': spec['id'], 'arm': 'labels', 'repetition': repeat,
                                      'seconds': round(label['classifier_wall_seconds'], 3),
                                      'errors': error}), flush=True)

            after_schema, after_roles, _ = source_schema(dataset)
            record['snapshot_unchanged_after_all_calls'] = (
                value_hash(dataset_snapshot(dataset, None if hasattr(dataset, 'disk') else dataset.raw_df))
                == snapshot_hash)
            record['local_schema_unchanged'] = (
                value_hash({'schema': after_schema, 'roles': after_roles}) == schema_hash)
            record['status'] = 'ok'
        except Exception as exc:
            record.update(status='error', error_type=type(exc).__name__)
            print(json.dumps({'dataset': spec['id'], 'errors': type(exc).__name__}), flush=True)
        finally:
            if dataset is not None:
                DATASETS.pop(dataset.id, None)
                dataset.close()
        write_json(args.output, output)

    output['summary'] = summarize(output['datasets'])
    output['status'] = 'complete' if all(d['status'] == 'ok' for d in output['datasets']) else 'partial'
    write_json(args.output, output)
    print(json.dumps({'status': output['status'], 'output': str(args.output),
                      'summary': output['summary']}), flush=True)
    return output


def summarize(datasets: list[dict]) -> dict:
    per_dataset, all_calls = [], {arm: [] for arm in ARMS}
    for dataset in datasets:
        arms = {}
        for arm in ARMS:
            calls = [c for c in dataset['calls'] if c['arm'] == arm]
            all_calls[arm].extend(calls)
            arms[arm] = {
                'calls': len(calls),
                'median_classifier_wall_seconds': statistics.median(
                    [c['classifier_wall_seconds'] for c in calls]) if calls else None,
                'median_role_application_wall_seconds': statistics.median(
                    [c['role_application_wall_seconds'] for c in calls
                     if c['role_application_wall_seconds'] is not None]) if any(
                    c['role_application_wall_seconds'] is not None for c in calls) else None,
                'raw_correct': sum(c['raw_role_score']['correct_columns'] for c in calls),
                'raw_wrong': sum(c['raw_role_score']['wrong_count'] for c in calls),
                'raw_unknown': sum(c['raw_role_score']['unknown_count'] for c in calls),
                'applied_correct': sum(c['applied_role_score']['correct_columns'] for c in calls),
                'applied_wrong': sum(c['applied_role_score']['wrong_count'] for c in calls),
                'applied_unknown': sum(c['applied_role_score']['unknown_count'] for c in calls),
                'wrong_new_changes': sum(len(c['wrong_new_changes_vs_original']) for c in calls),
                'request_failed_calls': sum(call_health(c)['request_failed'] for c in calls),
                'coverage_incomplete_calls': sum(call_health(c)['coverage_complete'] is False for c in calls),
            }
        base, candidate = (arms[a]['median_classifier_wall_seconds'] for a in ARMS)
        per_dataset.append({
            'dataset': dataset['dataset'], 'status': dataset['status'], 'arms': arms,
            'decisions_over_responses_latency_ratio': candidate / base if base else None,
        })
    totals_by_arm = {}
    for arm, calls in all_calls.items():
        totals_by_arm[arm] = {
            'calls': len(calls),
            'raw_correct': sum(c['raw_role_score']['correct_columns'] for c in calls),
            'raw_wrong': sum(c['raw_role_score']['wrong_count'] for c in calls),
            'raw_unknown': sum(c['raw_role_score']['unknown_count'] for c in calls),
            'applied_correct': sum(c['applied_role_score']['correct_columns'] for c in calls),
            'applied_wrong': sum(c['applied_role_score']['wrong_count'] for c in calls),
            'applied_unknown': sum(c['applied_role_score']['unknown_count'] for c in calls),
            'wrong_new_changes': sum(len(c['wrong_new_changes_vs_original']) for c in calls),
            'request_failed_calls': sum(call_health(c)['request_failed'] for c in calls),
            'coverage_incomplete_calls': sum(call_health(c)['coverage_complete'] is False for c in calls),
            'median_classifier_wall_seconds': statistics.median(
                [c['classifier_wall_seconds'] for c in calls]) if calls else None,
        }
    ratios = [r['decisions_over_responses_latency_ratio'] for r in per_dataset
              if r['decisions_over_responses_latency_ratio'] is not None]
    return {'per_dataset': per_dataset, 'totals': totals_by_arm,
            'median_decisions_over_responses_latency_ratio': statistics.median(ratios) if ratios else None}


def threshold_result(snapshot: dict, proposals: dict, threshold: float, coverage=None, usage=None) -> dict:
    import modules.schema_decisions as decisions
    from modules.schema_interpretation import compatible

    columns = {item['column']: item for item in snapshot['columns']}
    proposals = copy.deepcopy(proposals)
    for name, item in proposals.items():
        decision = item.get('decision')
        if not isinstance(decision, dict):
            continue
        role = decision.get('role')
        prediction = {'role': role, 'temporal_kind': decision.get('temporal_kind', 'none')}
        column = columns.get(name)
        conflict = column is not None and decisions._conflict(column, prediction)
        confidence, probability = item.get('decision_confidence'), item.get('choice_probability')
        numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in (confidence, probability))
        accepted = bool(column and role != 'unknown' and numbers
                        and confidence >= threshold and probability >= threshold
                        and compatible(column, prediction) and not conflict)
        decision.update(scope='role_only', needs_clarification=not accepted,
                        evidence_strength='strong' if accepted else
                        'unknown' if role == 'unknown' else 'tentative')
        item.update(status='ok' if accepted else 'uncertain',
                    error_code='role_conflict' if conflict else None)
    return {'interpretation_proposals': proposals, 'coverage': coverage, 'usage': usage}


def replay_case(dataset, snapshot, saved_call, spec, threshold) -> dict:
    proposals = saved_call.get('raw_proposals') or {}
    health = call_health(saved_call)
    result = threshold_result(snapshot, proposals, threshold, saved_call.get('coverage'),
                              saved_call.get('usage'))
    raw = score(proposal_roles(proposals, dataset.column_names), spec)
    applied, apply_seconds = apply_roles(dataset, snapshot, result)
    applied_score = score(applied, spec)
    changed, wrong = role_changes(applied, dict(dataset.column_types), spec)
    return {
        'repetition': saved_call.get('repetition'), 'classifier_error_type': saved_call.get('classifier_error_type'),
        **health, 'coverage': saved_call.get('coverage'),
        'role_application_wall_seconds': apply_seconds, 'raw_role_score': raw,
        'applied_role_score': applied_score, 'applied_roles': applied,
        'changed_roles_vs_original': changed, 'wrong_new_changes_vs_original': wrong,
    }


def total(records: list[dict]) -> dict:
    scores = [r['applied_role_score'] for r in records]
    evaluated = sum(s['evaluated_columns'] for s in scores)
    correct = sum(s['correct_columns'] for s in scores)
    return {
        'calls': len(records), 'evaluated_columns': evaluated, 'correct_columns': correct,
        'correct_coverage': correct / evaluated if evaluated else None,
        'unknown_columns': sum(s['unknown_count'] for s in scores),
        'wrong_classifications': sum(s['wrong_count'] for s in scores),
        'unavailable_columns': sum(s['unavailable_count'] for s in scores),
        'opaque_columns': sum(s['opaque_column_count'] for s in scores),
        'opaque_unknown_abstentions': sum(s['opaque_unknown_count'] for s in scores),
        'changed_roles_vs_original': sum(len(r['changed_roles_vs_original']) for r in records),
        'wrong_new_changes_vs_original': sum(len(r['wrong_new_changes_vs_original']) for r in records),
        'request_failed_calls': sum(r['request_failed'] for r in records),
        'request_failed_columns': sum(r['request_failed_columns'] for r in records),
        'coverage_incomplete_calls': sum(r['coverage_complete'] is False for r in records),
        'model_refusal_columns': sum(r['model_refusal_columns'] for r in records),
    }


def curve_row(threshold, per_dataset) -> dict:
    rows = list(per_dataset.values())
    correct = sum(r['correct_columns'] for r in rows)
    evaluated = sum(r['evaluated_columns'] for r in rows)
    baseline_correct = sum(r['responses_correct'] for r in rows)
    baseline_total = sum(r['responses_total'] for r in rows)
    wrong_new = sum(r['wrong_new_changes_vs_original'] for r in rows)
    coverage_ok = bool(evaluated and baseline_total
                       and correct * baseline_total >= baseline_correct * evaluated)
    return {
        'threshold': threshold, 'training_datasets_scored': list(per_dataset),
        'calls': sum(r['calls'] for r in rows), 'evaluated_columns': evaluated,
        'correct_columns': correct, 'correct_coverage': correct / evaluated if evaluated else None,
        'unknown_columns': sum(r['unknown_columns'] for r in rows),
        'wrong_classifications': sum(r['wrong_classifications'] for r in rows),
        'unavailable_columns': sum(r['unavailable_columns'] for r in rows),
        'changed_roles_vs_original': sum(r['changed_roles_vs_original'] for r in rows),
        'wrong_new_changes_vs_original': wrong_new,
        'responses_baseline_correct_columns': baseline_correct,
        'responses_baseline_evaluated_columns': baseline_total,
        'responses_baseline_correct_coverage': baseline_correct / baseline_total if baseline_total else None,
        'coverage_at_least_responses': coverage_ok, 'zero_wrong_new_changes': wrong_new == 0,
        'meets_selection_constraints': coverage_ok and wrong_new == 0, 'per_dataset': per_dataset,
    }


def offline_replay(args) -> dict:
    """Select on the frozen first five, then evaluate the three heldout sets."""
    from modules.import_policy import ImportSettings
    from modules.schema_interpretation import dataset_snapshot
    from services.csv_ingestion import ingest_csv
    from storage import DATASETS, get_dataset
    from ingestion_parallelism import spec_path

    saved = json.loads(args.replay.read_text())
    threshold = saved.get('parameters', {}).get(
        'min_confidence', saved.get('methodology', {}).get('min_confidence'))
    if saved.get('status') != 'complete' or threshold != 0.9:
        raise ValueError('Replay requires a completed paired run frozen at confidence .9.')
    specs, manifests = load_specs()
    by_id = {s['id']: s for s in specs}
    source = saved.get('datasets', [])
    training = {name: next(r for r in source if r.get('dataset') == name) for name in CALIBRATION}
    if any(name not in by_id or training[name].get('status') != 'ok' for name in CALIBRATION):
        raise ValueError('A first-five calibration dataset is missing or incomplete.')

    def ingest(spec, record):
        path = spec_path(spec)
        if file_hash(path) != spec.get('sha256') or file_hash(path) != record.get('source_sha256'):
            raise ValueError(f'Fixture hash mismatch: {spec["id"]}')
        with path.open('rb') as stream, contextlib.redirect_stdout(io.StringIO()):
            upload = ingest_csv(stream, spec['filename'], '/decisions-roles-offline-replay',
                                import_settings=ImportSettings(delimiter=spec['delimiter']),
                                enqueue_enrichment=False, ai_column_analysis=False)
        ds = get_dataset(upload.dataset_id)
        snapshot = dataset_snapshot(ds, None if hasattr(ds, 'disk') else ds.raw_df)
        if ds.row_count != spec.get('rows') or ds.row_count != record.get('row_count'):
            raise ValueError(f'Row count mismatch: {spec["id"]}')
        if value_hash(snapshot) != record.get('snapshot_sha256'):
            raise ValueError(f'Snapshot mismatch: {spec["id"]}')
        return ds, snapshot, path

    output = {
        'status': 'calibrating', 'source_run': str(args.replay),
        'source_run_manifest_hashes': saved.get('manifest_hashes'), 'manifest_hashes': manifests,
        'methodology': {
            'selection_datasets': list(CALIBRATION),
            'heldout_datasets': [s['id'] for s in specs if s['id'] not in CALIBRATION],
            'threshold_grid': list(GRID),
            'selection_rule': 'highest threshold with zero wrong new changes and correct coverage at least the Responses baseline',
            'heldout_predictions_read_before_selection': False,
            'replay': 'offline only; production apply_schema_roles guards are reused',
        },
        'threshold_selection': None, 'calibration_curve': [],
        'selected_applied_scores': [], 'selected_summary': None,
    }
    write_json(args.output, output)
    opened, prepared = {}, {}
    points = {t: {} for t in GRID}
    try:
        for name in CALIBRATION:
            spec, original = by_id[name], training[name]
            ds, snapshot, path = ingest(spec, original)
            opened[name] = ds
            calls = [c for c in original['calls'] if c['arm'] == 'decisions']
            baseline = [c for c in original['calls'] if c['arm'] == 'responses']
            if len(calls) != 5 or len(baseline) != 5:
                raise ValueError(f'Expected five paired calls for {name}.')
            for call in calls:
                if replay_case(ds, snapshot, call, spec, 0.9)['applied_roles'] != call.get('applied_roles'):
                    raise ValueError(f'.9 replay mismatch: {name}')
            base_scores = [c['applied_role_score'] for c in baseline]
            base_correct = sum(s['correct_columns'] for s in base_scores)
            base_total = sum(s['evaluated_columns'] for s in base_scores)
            prepared[name] = (spec, ds, snapshot, calls, path)
            for t in GRID:
                cases = [replay_case(ds, snapshot, call, spec, t) for call in calls]
                scored = total(cases)
                points[t][name] = {
                    **{k: scored[k] for k in (
                        'calls', 'evaluated_columns', 'correct_columns', 'unknown_columns',
                        'wrong_classifications', 'unavailable_columns', 'changed_roles_vs_original',
                        'wrong_new_changes_vs_original')},
                    'responses_correct': base_correct, 'responses_total': base_total,
                }
            output['calibration_curve'] = [curve_row(t, points[t]) for t in GRID]
            write_json(args.output, output)
            print(json.dumps({'phase': 'calibration', 'dataset': name, 'snapshot_match': True,
                              'original_threshold_replays': 5}), flush=True)

        curve = [curve_row(t, points[t]) for t in GRID]
        feasible = [row for row in curve if row['meets_selection_constraints']]
        selected = max((row['threshold'] for row in feasible), default=None)
        baseline = points[GRID[0]].values()
        base_correct = sum(row['responses_correct'] for row in baseline)
        base_total = sum(row['responses_total'] for row in baseline)
        output['calibration_curve'] = curve
        output['threshold_selection'] = {
            'selected_threshold': selected, 'training_datasets': list(CALIBRATION),
            'responses_baseline_correct_columns': base_correct,
            'responses_baseline_evaluated_columns': base_total,
            'responses_baseline_correct_coverage': base_correct / base_total if base_total else None,
            'feasible_thresholds': [row['threshold'] for row in feasible],
            'selection_rule': 'highest feasible threshold',
            'heldout_predictions_read_before_selection': False,
        }
        if selected is None:
            output['status'] = 'no_feasible_threshold'
            write_json(args.output, output)
            return output

        records = []
        def save_selected(name, spec, ds, snapshot, path, calls):
            for call in calls:
                result = replay_case(ds, snapshot, call, spec, selected)
                records.append({
                    'dataset': name, 'threshold': selected,
                    'fixture': spec.get('local_path') or spec.get('fixture_file'),
                    'source_sha256': file_hash(path), 'source_bytes': path.stat().st_size,
                    'row_count': ds.row_count, 'snapshot_sha256': value_hash(snapshot), **result,
                })

        for name in CALIBRATION:
            spec, ds, snapshot, calls, path = prepared[name]
            save_selected(name, spec, ds, snapshot, path, calls)
        heldout_summary = []
        for spec in specs:
            name = spec['id']
            if name in CALIBRATION:
                continue
            # This is the first access to heldout proposal records.
            original = next(r for r in source if r.get('dataset') == name)
            if original.get('status') != 'ok':
                raise ValueError(f'Incomplete heldout dataset: {name}')
            ds, snapshot, path = ingest(spec, original)
            opened[name] = ds
            calls = [c for c in original['calls'] if c['arm'] == 'decisions']
            if len(calls) != 5:
                raise ValueError(f'Expected five heldout calls for {name}')
            before = len(records)
            save_selected(name, spec, ds, snapshot, path, calls)
            heldout_summary.append({'dataset': name, **total(records[before:])})

        train = [r for r in records if r['dataset'] in CALIBRATION]
        heldout = [r for r in records if r['dataset'] not in CALIBRATION]
        output.update(status='complete', selected_applied_scores=records)
        output['selected_summary'] = {
            'threshold': selected, 'calibration_first_five': total(train),
            'heldout_last_three': {'datasets': heldout_summary, **total(heldout)},
            'all_eight': total(records),
        }
        write_json(args.output, output)
        print(json.dumps({'status': 'complete', 'selected_threshold': selected,
                          'calibration': output['selected_summary']['calibration_first_five'],
                          'heldout': output['selected_summary']['heldout_last_three'],
                          'all_eight': output['selected_summary']['all_eight'],
                          'output': str(args.output)}), flush=True)
        return output
    finally:
        for ds in opened.values():
            DATASETS.pop(ds.id, None)
            ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Enable live model requests.')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--labels', action='store_true')
    parser.add_argument('--label-repeats', type=int, default=2)
    parser.add_argument('--min-confidence', type=float, default=0.85)
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--output', type=Path, default=LIVE_OUTPUT)
    parser.add_argument('--replay', type=Path, help='Offline calibration input JSON.')
    parser.add_argument('--calibration-output', type=Path, default=REPLAY_OUTPUT)
    args = parser.parse_args()
    if args.replay:
        if args.live:
            parser.error('--replay cannot be combined with --live')
        args.output = args.calibration_output
        offline_replay(args)
    elif not args.live:
        parser.error('Pass --live for paid calls, or --replay for offline calibration.')
    elif args.repeats < 1 or args.label_repeats < 1:
        parser.error('--repeats and --label-repeats must be positive')
    else:
        live(args)


if __name__ == '__main__':
    main()
