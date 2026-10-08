"""Paired context ablation using real fixtures and production role guards."""
import argparse
import contextlib
import copy
import io
import json
import os
import random
import statistics
from pathlib import Path
from time import perf_counter

from decisions_roles import (apply_roles, file_hash, load_specs, proposal_roles,
                             score, source_schema, value_hash, write_json)

ARMS = ('original', 'local_context', 'local_context_summary')


def summarize(records):
    result = {}
    for arm in ARMS:
        calls = [c for r in records for c in r['calls'] if c['arm'] == arm]
        total = sum(c['score']['evaluated_columns'] for c in calls)
        correct = sum(c['score']['correct_columns'] for c in calls)
        token_counts = [c['input_tokens'] for c in calls if c['input_tokens'] is not None]
        result[arm] = {
            'calls': len(calls), 'evaluated': total, 'correct': correct,
            'accuracy_including_unknowns': correct / total if total else None,
            'unknown': sum(c['score']['unknown_count'] for c in calls),
            'wrong': sum(c['score']['wrong_count'] for c in calls),
            'failed_calls': sum(c['failed'] for c in calls),
            'median_classifier_seconds': statistics.median(c['classifier_seconds'] for c in calls),
            'median_role_ready_after_local_seconds': statistics.median(c['role_ready_seconds'] for c in calls),
            'median_input_tokens': statistics.median(token_counts) if token_counts else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('/private/tmp/analytico-decisions-context-2026-10-06.json'))
    args = parser.parse_args()
    if not args.live or not 1 <= args.repeats <= 10:
        parser.error('Explicit --live and 1–10 repeats required')
    from core import config
    from ingestion_parallelism import spec_path
    from modules.import_policy import ImportSettings
    from modules.schema_decisions import analyze_schema_decisions
    from modules.schema_interpretation import dataset_snapshot
    from services.csv_ingestion import ingest_csv
    from services.ingestion_service import _generate_business_summary
    from storage import DATASETS, get_dataset
    if config.OPENAI_MODEL != 'gpt-6-luna' or not os.getenv('OPENAI_API_KEY', '').strip():
        raise ValueError('Configured GPT-6 Luna and API key required')
    specs, hashes = load_specs()
    rng = random.Random(20261006)
    output = {'status': 'running', 'model': 'gpt-6-luna', 'threshold': .85,
              'repeats': args.repeats, 'manifest_hashes': hashes,
              'methodology': 'randomized paired three-arm context ablation; full real ingestion; same snapshot; production apply guards on isolated clones; no gold in model input; summary wait included in third arm',
              'datasets': []}
    for spec in specs:
        path = spec_path(spec)
        if file_hash(path) != spec['sha256']:
            raise ValueError('Fixture hash mismatch: ' + spec['id'])
        ds = None
        try:
            started = perf_counter()
            with path.open('rb') as stream, contextlib.redirect_stdout(io.StringIO()):
                upload = ingest_csv(stream, spec['filename'], '/context-benchmark',
                                    import_settings=ImportSettings(delimiter=spec['delimiter']),
                                    enqueue_enrichment=False, ai_column_analysis=False)
            ds = get_dataset(upload.dataset_id)
            local_seconds = perf_counter() - started
            if ds.row_count != spec['rows']:
                raise ValueError('Fixture row mismatch')
            snapshot = dataset_snapshot(ds, None if hasattr(ds, 'disk') else ds.raw_df)
            initial_snapshot = value_hash(snapshot)
            initial_schema = value_hash(source_schema(ds)[0])
            summary_frame = ds.sample_frame(3).iloc[:, :20].copy(deep=True)
            record = {'dataset': spec['id'], 'rows': ds.row_count,
                      'columns': len(ds.column_names), 'source_sha256': file_hash(path),
                      'local_preparation_seconds': local_seconds, 'calls': [], 'summaries': []}
            output['datasets'].append(record)
            for repetition in range(1, args.repeats + 1):
                arms = list(ARMS)
                rng.shuffle(arms)
                for arm in arms:
                    view = copy.deepcopy(snapshot)
                    summary_seconds = 0.0
                    if arm == 'local_context_summary':
                        started = perf_counter()
                        description = _generate_business_summary(ds.filename, summary_frame)
                        summary_seconds = perf_counter() - started
                        view['dataset_description'] = description
                        record['summaries'].append({'repetition': repetition,
                                                    'seconds': summary_seconds, 'description': description})
                    started = perf_counter()
                    result = analyze_schema_decisions(view, min_confidence=.85,
                                                      enriched_context=arm != 'original')
                    classifier_seconds = perf_counter() - started
                    roles, apply_seconds = apply_roles(ds, view, result)
                    proposals = result['interpretation_proposals']
                    call = {'arm': arm, 'repetition': repetition,
                            'classifier_seconds': classifier_seconds, 'summary_seconds': summary_seconds,
                            'role_ready_seconds': summary_seconds + classifier_seconds + apply_seconds,
                            'input_tokens': (result.get('usage') or {}).get('input_tokens'),
                            'score': score(roles, spec), 'raw_score': score(proposal_roles(proposals, ds.column_names), spec),
                            'applied_roles': roles, 'proposals': proposals,
                            'failed': bool(result['coverage']['failed_columns'] or not result['coverage']['complete'])}
                    record['calls'].append(call)
                    write_json(args.output, output)
                    print(json.dumps({'dataset': spec['id'], 'arm': arm, 'repetition': repetition,
                                      'correct': call['score']['correct_columns'],
                                      'unknown': call['score']['unknown_count'], 'wrong': call['score']['wrong_count'],
                                      'role_ready_seconds': round(call['role_ready_seconds'], 3)}), flush=True)
            record['integrity'] = {
                'source_unchanged': file_hash(path) == spec['sha256'],
                'snapshot_unchanged': value_hash(snapshot) == initial_snapshot,
                'original_schema_unchanged': value_hash(source_schema(ds)[0]) == initial_schema}
            if not all(record['integrity'].values()):
                raise ValueError('Source/schema changed')
            write_json(args.output, output)
        finally:
            if ds is not None:
                DATASETS.pop(ds.id, None)
                ds.close()
    output['summary'] = summarize(output['datasets'])
    output['status'] = 'complete'
    write_json(args.output, output)
    print(json.dumps(output['summary']), flush=True)


if __name__ == '__main__':
    main()
