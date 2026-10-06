"""Compare current full-profile Luna analysis with provisional early-sample overlap.

The early response is replayed through the production validator against the
completed local schema. No early proposal is applied before local validation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from core import config  # loads the existing BYOK configuration without printing it
from modules import schema_interpretation as semantic
from modules.import_policy import ImportSettings
from services.csv_ingestion import ingest_csv
from storage import get_dataset, DATASETS
from fresh_real_accuracy import score_roles, schema_roles
from ingestion_parallelism import spec_path

OUTPUT = Path('/private/tmp/analytico-ai-overlap-2026-10-03.json')


def early_state(spec):
    with spec_path(spec).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream, delimiter=spec['delimiter'])
        headers = next(reader)
        rows = []
        for index, row in enumerate(reader):
            if index == 100:
                break
            rows.append(row)
    positions = sorted({round(i * (len(rows) - 1) / 11) for i in range(12)}) if rows else []
    return {
        'row_count': None, 'schema_is_provisional': True,
        'sampling_scope': 'first_100_source_rows_only',
        'full_column_validation': 'pending; do not assume physical type or full-file distribution',
        'columns': [{'index': i, 'name': header[:256], 'physical_type': 'unvalidated_source_text',
                     'values': [rows[p][i][:160] if i < len(rows[p]) else None for p in positions],
                     'sample_statistics': {'unique_count': len({r[i] for r in rows if i < len(r) and r[i]})},
                     'complete_column_statistics': None} for i, header in enumerate(headers)],
        'requested_indexes': list(range(len(headers))),
    }


def request_early(state):
    import httpx
    started = time.perf_counter()
    request = {'model': 'gpt-6-luna', 'store': False, 'reasoning': {'effort': 'none'},
               'input': [{'role': 'system', 'content': semantic.SYSTEM},
                         {'role': 'user', 'content': json.dumps(state, ensure_ascii=False)}],
               'text': {'format': {'type': 'json_schema', 'name': 'schema_roles', 'strict': True,
                                  'schema': semantic.proposal_json_schema()}},
               'max_output_tokens': 8192}
    with httpx.Client(timeout=15) as client:
        response = client.post('https://api.openai.com/v1/responses', json=request,
                               headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY']})
    response.raise_for_status()
    return response, time.perf_counter() - started, time.perf_counter()


class ReplayClient:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def post(self, *_args, **_kwargs):
        return self.response


def run(repeats):
    specs = json.loads((BACKEND / 'evals/fresh_accuracy_gold_2026-10-03.json').read_text())['datasets']
    jobs = [(spec, mode, repeat) for spec in specs for mode in ['current_after_local', 'early_sample_overlap'] for repeat in range(repeats)]
    random.Random(60704).shuffle(jobs)
    records = []
    for spec, mode, repeat in jobs:
        start = time.perf_counter()
        record = {'dataset': spec['id'], 'mode': mode, 'repeat': repeat, 'model': config.OPENAI_MODEL}
        future, dataset = None, None
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                if mode == 'early_sample_overlap':
                    state = early_state(spec)
                    record['sample_ready_seconds'] = time.perf_counter() - start
                    future = pool.submit(request_early, state)
                with spec_path(spec).open('rb') as stream, contextlib.redirect_stdout(io.StringIO()):
                    upload = ingest_csv(stream, spec['filename'], '/ai-overlap',
                                        import_settings=ImportSettings(delimiter=spec['delimiter']),
                                        enqueue_enrichment=False, ai_column_analysis=False)
                dataset = get_dataset(upload.dataset_id)
                record['local_ready_seconds'] = time.perf_counter() - start
                snapshot = semantic.dataset_snapshot(dataset, None if hasattr(dataset, 'disk') else dataset.raw_df)
                model_started = time.perf_counter()
                if future:
                    response, latency, finished = future.result()
                    with patch.object(semantic.httpx, 'Client', return_value=ReplayClient(response)):
                        result = semantic.analyze_schema(snapshot)
                    record['model_seconds'] = latency
                    record['model_finished_seconds'] = finished - start
                else:
                    result = semantic.analyze_schema(snapshot)
                    record['model_seconds'] = time.perf_counter() - model_started
                semantic.apply_schema_roles(dataset.id, dataset.cache_version, result)
                record['ai_applied_seconds'] = time.perf_counter() - start
                record['wait_after_local_seconds'] = record['ai_applied_seconds'] - record['local_ready_seconds']
                record['coverage_complete'] = result['coverage']['complete']
                record['role_scores'] = score_roles(schema_roles(dataset), spec)
                record['applied_labels'] = sum(bool(item.get('display_name')) for item in dataset.column_schema)
                record['status'] = 'ok' if result['coverage']['complete'] else 'provider_or_validation_failure'
            except Exception as error:
                record.update(status='error', error_type=type(error).__name__)
            finally:
                if dataset is not None:
                    DATASETS.pop(dataset.id, None)
                    dataset.close()
        records.append(record)
        OUTPUT.write_text(json.dumps(records, default=str, indent=2))
        print(json.dumps({key: value for key, value in record.items() if key != 'role_scores'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Required authorization flag for paid live Luna calls.')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if not args.live:
        parser.error('--live is required')
    run(args.repeats)
