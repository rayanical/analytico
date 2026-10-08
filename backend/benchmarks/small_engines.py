"""Current pandas versus disk ingestion on frozen real small files, no AI.

Each fresh worker measures first and subsequent stage/confirm uploads. Full
ordered-value verification follows timing; exact and 12-significant-digit
numeric fingerprints are separate so floating-point differences remain visible.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from decimal import Decimal
import hashlib
import json
import numbers
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time

from ingestion_batching import gold_specs
from ingestion_parallelism import spec_path


def scalar(value, rounded=False):
    import pandas as pd
    if value is None or pd.isna(value):
        return ['null']
    if isinstance(value, (bool,)) or type(value).__name__ == 'bool':
        return ['bool', bool(value)]
    if isinstance(value, (datetime, date)):
        return ['date', value.isoformat()]
    if isinstance(value, numbers.Number):
        if rounded and isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
            value = format(float(value), '.12g')
        return ['number', str(Decimal(str(value)).normalize())]
    return ['text', str(value)]


def fingerprint(ds):
    columns = ds.column_names
    exact = [hashlib.sha256() for _ in columns]
    rounded = [hashlib.sha256() for _ in columns]
    nulls = [0] * len(columns)
    if hasattr(ds, 'disk'):
        from modules.disk_dataset import _q
        cursor = ds.disk._connection.execute(
            f"SELECT {', '.join(_q(c) for c in columns)} FROM typed_data ORDER BY _row_ordinal"
        )
        def rows():
            while batch := cursor.fetchmany(4096):
                yield from batch
        iterator = rows()
    else:
        iterator = ds.df.itertuples(index=False, name=None)
    for row in iterator:
        for index, value in enumerate(row):
            item = scalar(value)
            nulls[index] += item == ['null']
            exact[index].update((json.dumps(item, ensure_ascii=False) + '\n').encode())
            rounded[index].update((json.dumps(scalar(value, True), ensure_ascii=False) + '\n').encode())
    return {'columns': columns, 'rows': ds.row_count,
            'exact': {c: h.hexdigest() for c, h in zip(columns, exact)},
            'rounded_numeric_12_digits': {c: h.hexdigest() for c, h in zip(columns, rounded)},
            'nulls': dict(zip(columns, nulls)), 'roles': ds.column_types,
            'formats': ds.column_formats}


def score_roles(spec, observed):
    from modules.data_janitor import _unique_normalized_headers
    names = _unique_normalized_headers(list(spec['roles']))
    correct = unknown = wrong = 0
    for normalized, (original, expected) in zip(names, spec['roles'].items()):
        actual = observed.get(normalized)
        accepted = [expected, *spec.get('role_alternatives', {}).get(original, [])]
        if actual in accepted:
            correct += 1
        elif actual == 'unknown':
            unknown += 1
        else:
            wrong += 1
    return {'role_correct':correct, 'role_unknown':unknown,
            'role_incorrect_nonunknown':wrong, 'role_total':len(names)}


def worker(spec, engine, output):
    os.environ.update(OPENAI_API_KEY='', AI_GATEWAY_API_KEY='', COLUMN_INTERPRETER='off',
                      ANALYTICO_NATIVE_VALIDATION='0', ANALYTICO_DUCKDB_THREADS='4',
                      ANALYTICO_INGESTION_ENGINE='disk' if engine == 'disk_preloaded' else engine)
    from services import import_preview as preview
    from storage import DATASETS
    from modules.import_policy import ImportSettings
    preload_seconds = 0.0
    if engine == 'disk_preloaded':
        preload_started = time.perf_counter()
        import duckdb
        from modules.disk_dataset import DiskDataset
        preload_seconds = time.perf_counter() - preload_started
    records = []
    for state in ['first_upload', 'subsequent_upload']:
        ds = item = None
        try:
            started = time.perf_counter()
            response = preview.stage_import(spec_path(spec), spec['filename'],
                                            ImportSettings(delimiter=spec['delimiter']))
            staged_at = time.perf_counter()
            item = preview._imports[response['import_id']]
            response = preview.confirm_import(item.id, item.settings, ai_column_analysis=False)
            finished = time.perf_counter()
            ds = DATASETS[response.dataset_id]
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            record = {'dataset': spec['id'], 'engine': engine, 'state': state, 'status': 'ok',
                      'ready_seconds': finished-started, 'stage_seconds': staged_at-started,
                      'confirm_seconds': finished-staged_at,
                      'peak_mib': peak/(2**20) if sys.platform == 'darwin' else peak/1024,
                      'actual_engine': 'disk' if hasattr(ds, 'disk') else 'pandas'}
            record['preload_seconds'] = preload_seconds
            # Verification is outside the captured timer and memory high-water.
            record['fingerprint'] = fingerprint(ds)
            source = ds.disk.source_path if hasattr(ds, 'disk') else ds.source_path
            with source.open('rb') as stream:
                record['source_matches'] = hashlib.file_digest(stream, 'sha256').hexdigest() == spec['sha256']
            record['rows_match'] = ds.row_count == spec['rows']
            record.update(score_roles(spec, ds.column_types))
            records.append(record)
        except Exception as error:
            records.append({'dataset': spec['id'], 'engine': engine, 'state': state,
                            'status': 'error', 'error': str(error)[:300]})
        finally:
            if ds is not None:
                DATASETS.pop(ds.id, None)
                ds.close()
            if item is not None and item.id in preview._imports:
                preview.cancel_import(item.id)
    output.write_text(json.dumps(records, default=str))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker')
    p.add_argument('--engine', choices=['pandas', 'disk', 'disk_preloaded'])
    p.add_argument('--engines', nargs='+', choices=['pandas', 'disk', 'disk_preloaded'], default=['pandas','disk'])
    p.add_argument('--output', type=Path)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--reference-root', type=Path)
    p.add_argument('--output-root', type=Path, default=Path('/private/tmp/analytico-small-engines-2026-10-07'))
    a = p.parse_args()
    specs = [s for s in gold_specs() if spec_path(s).stat().st_size < 8*1024*1024]
    if a.worker:
        return worker(next(s for s in specs if s['id'] == a.worker), a.engine, a.output)
    a.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(s, e, r) for s in specs for e in a.engines for r in range(a.repeats)]
    random.Random(20261007).shuffle(jobs)
    records = []
    for index, (spec, engine, repeat) in enumerate(jobs, 1):
        output = a.output_root / f"{spec['id']}-{engine}-{repeat}.json"
        output.unlink(missing_ok=True)
        with (a.output_root/'worker.log').open('a') as log:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', spec['id'],
                                        '--engine', engine, '--output', str(output)], stdout=log, stderr=log)
        batch = json.loads(output.read_text()) if output.exists() else [{'dataset':spec['id'], 'engine':engine,
                                                                       'status':'crash','exit_code':completed.returncode}]
        for record in batch:
            record['repeat'] = repeat
            records.append(record)
        (a.output_root/'records.json').write_text(json.dumps(records, default=str))
        print(json.dumps({'progress':f'{index}/{len(jobs)}','dataset':spec['id'],'engine':engine,
                          'results':[(r.get('state'), r['status'], round(r.get('ready_seconds',0),4)) for r in batch]}), flush=True)
    results = []
    references = json.loads((a.reference_root/'records.json').read_text()) if a.reference_root else []
    for spec in specs:
        reference = next((r['fingerprint'] for r in [*records, *references] if r['dataset']==spec['id'] and r['engine']=='pandas'
                          and r['status']=='ok'), None)
        for engine in a.engines:
            for state in ['first_upload', 'subsequent_upload']:
                rs = [r for r in records if r['dataset']==spec['id'] and r['engine']==engine and r.get('state')==state]
                valid = [r for r in rs if r['status']=='ok']
                differences = {}
                if reference:
                    for r in valid:
                        for key in ['exact','rounded_numeric_12_digits','nulls','roles','formats']:
                            for c in reference['columns']:
                                if r['fingerprint'][key].get(c) != reference[key].get(c):
                                    differences.setdefault(key, set()).add(c)
                results.append({'dataset':spec['id'],'bytes':spec_path(spec).stat().st_size,'rows':spec['rows'],
                                'engine':engine,'state':state,'runs':len(rs),'successes':len(valid),
                                'median_seconds':statistics.median(r['ready_seconds'] for r in valid) if valid else None,
                                'median_peak_mib':statistics.median(r['peak_mib'] for r in valid) if valid else None,
                                'median_preload_seconds':statistics.median(r['preload_seconds'] for r in valid) if valid else None,
                                'source_matches':sum(r.get('source_matches',False) for r in rs),
                                'row_matches':sum(r.get('rows_match',False) for r in rs),
                                'role_correct':valid[0]['role_correct'] if valid else None,
                                'role_unknown':valid[0]['role_unknown'] if valid else None,
                                'role_incorrect_nonunknown':valid[0]['role_incorrect_nonunknown'] if valid else None,
                                'role_total':len(spec['roles']),
                                'pandas_reference_available':reference is not None,
                                'differences_vs_pandas':{k:sorted(v) for k,v in differences.items()},
                                'errors':[r.get('error','worker crash') for r in rs if r['status']!='ok']})
    summary = {'date':'2026-10-07','ai_calls':0,'repeats':a.repeats,'datasets':[s['id'] for s in specs],
               'scope':'Backend stage and confirm; includes lazy engine imports; excludes transfer, rendering and verification',
               'resources':{'duckdb_threads':4,'duckdb_memory_limit':'64MB'},
               'memory_note':'Process lifetime high-water captured before verification. Subsequent-upload RSS can include first verification; use first-upload RSS for engine comparison.',
               'results':results}
    (a.output_root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'summary':str(a.output_root/'summary.json')}),flush=True)


if __name__ == '__main__':
    main()
