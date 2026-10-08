"""Live paired import benchmark: late versus overlapped background enrichment."""
import argparse
import hashlib
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

from ingestion_parallelism import spec_path
from ingestion_batching import exact_typed_fingerprint
from decisions_roles import load_specs, score, write_json


def worker(spec, mode, output, combined_size=8):
    from core import config
    from services.csv_ingestion import ingest_csv
    from modules.import_policy import ImportSettings
    from services.enrichment_service import manager
    from storage import DATASETS
    from ingestion_batching import install_experiment
    install_experiment('combined' if mode == 'combined' else 'baseline', combined_size)
    os.environ['ANALYTICO_EARLY_AI'] = '1' if mode in {'early', 'combined'} else '0'
    os.environ['ANALYTICO_NATIVE_VALIDATION'] = '1' if mode == 'combined' else '0'
    os.environ['COLUMN_ROLE_BACKEND'] = 'decisions'
    os.environ['COLUMN_ROLE_CONTEXT'] = 'original'
    if config.OPENAI_MODEL != 'gpt-6-luna':
        raise ValueError('Benchmark requires GPT-6 Luna')
    ds = None
    started = time.perf_counter()
    try:
        path = spec_path(spec)
        response = ingest_csv(path, spec['filename'], '/overlap-benchmark', ai_column_analysis=True,
                              import_settings=ImportSettings(delimiter=spec['delimiter']))
        chart_ready = time.perf_counter() - started
        ds = DATASETS[response.dataset_id]
        physical_version = ds.cache_version
        first = {}
        final = None
        while time.perf_counter() - started < 90:
            status = manager.get_status(ds.id)
            for field in ('summary', 'column_roles', 'column_labels'):
                if field not in first and status.get(field):
                    first[field] = time.perf_counter() - started
            if status['status'] in {'done', 'error', 'disabled'}:
                final = status
                break
            time.sleep(.01)
        complete = time.perf_counter() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        Path(str(output) + '.ready').write_text('ready')
        record = {'dataset': spec['id'], 'mode': mode,
                  'status': (final or {}).get('status', 'timeout'),
                  'chart_ready_seconds': chart_ready, 'complete_seconds': complete,
                  'first': first, 'rows': ds.row_count, 'role_score': score(ds.column_types, spec),
                  'roles': ds.column_types, 'labels': (final or {}).get('column_labels'),
                  'summary_present': bool((final or {}).get('summary')),
                  'coverage': (final or {}).get('coverage'),
                  'physical_version_unchanged': ds.cache_version == physical_version}
        # AI can change roles/labels; compare physical values and parser metadata.
        if hasattr(ds, 'disk'):
            # Verification is outside the timed/memory window and may need
            # more workspace than ingestion's streaming queries.
            ds.disk._connection.execute("SET threads=1")
            ds.disk._connection.execute("SET memory_limit='1GB'")
            data = exact_typed_fingerprint(ds.disk)
            record['physical_fingerprint'] = {k: data[k] for k in
                ('typed_checksums', 'ordered_typed_rows_sha256', 'dtypes', 'formats', 'missing', 'unique', 'headers')}
            source = ds.disk.source_path
            record['phases'] = ds.disk.ingestion_timings
        else:
            import pandas as pd
            record['physical_fingerprint'] = {
                'hash': str(int(pd.util.hash_pandas_object(ds.df, index=True).sum())),
                'dtypes': {k: str(v) for k, v in ds.df.dtypes.items()},
                'missing': ds.raw_missing_counts, 'rows': ds.row_count}
            source = ds.source_path
        with Path(source).open('rb') as stream:
            record['source_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        record['source_matches'] = record['source_sha256'] == spec['sha256']
        record['rows_match'] = ds.row_count == spec['rows']
        record['self_peak_bytes'] = peak if sys.platform == 'darwin' else peak * 1024
        write_json(output, record)
    finally:
        if ds is not None:
            DATASETS.pop(ds.id, None)
            ds.close()
        manager.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--datasets', nargs='+', default=['taxi', 'retail', 'wine'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-root', type=Path, default=Path('/private/tmp/analytico-ingestion-overlap-2026-10-06'))
    parser.add_argument('--worker')
    parser.add_argument('--mode', choices=['late', 'early', 'combined'])
    parser.add_argument('--modes', nargs='+', choices=['late', 'early', 'combined'], default=['late', 'early'])
    parser.add_argument('--combined-size', type=int, choices=[4, 8, 16], default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    specs, hashes = load_specs()
    if args.worker:
        return worker(next(s for s in specs if s['id'] == args.worker), args.mode, args.output, args.combined_size)
    if not args.live or not 1 <= args.repeats <= 5:
        parser.error('--live and 1–5 repeats required')
    if set(args.datasets) - {s['id'] for s in specs}:
        parser.error('Unknown dataset')
    selected = [s for s in specs if s['id'] in args.datasets]
    jobs = [(s, mode, repeat) for s in selected for mode in args.modes
            for repeat in range(args.repeats)]
    random.Random(20261006).shuffle(jobs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for spec, mode, repeat in jobs:
        output = args.output_root / f'{spec["id"]}-{mode}-{repeat}.json'
        output.unlink(missing_ok=True)
        ready = Path(str(output) + '.ready')
        ready.unlink(missing_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', spec['id'],
                   '--mode', mode, '--combined-size', str(args.combined_size), '--output', str(output)]
        with (args.output_root / 'worker.log').open('a') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            while process.poll() is None:
                time.sleep(.15)
        record = json.loads(output.read_text()) if output.exists() else {
            'dataset': spec['id'], 'mode': mode, 'status': 'crash'}
        record.update(repeat=repeat)
        records.append(record)
        write_json(args.output_root / 'records.json', {'records': records})
        print(json.dumps({k: record.get(k) for k in
            ('dataset', 'mode', 'status', 'chart_ready_seconds', 'complete_seconds', 'self_peak_bytes')}), flush=True)
    summary = []
    for spec in selected:
        reference = next(r['physical_fingerprint'] for r in records
                         if r['dataset'] == spec['id'] and r['mode'] == 'late' and r['status'] == 'done')
        for mode in args.modes:
            cases = [r for r in records if r['dataset'] == spec['id'] and r['mode'] == mode]
            valid = [r for r in cases if r['status'] == 'done']
            summary.append({'dataset': spec['id'], 'mode': mode, 'runs': len(cases),
                'complete_runs': len(valid),
                'summary_present_runs': sum(r.get('summary_present', False) for r in valid),
                'labels_present_runs': sum(bool(r.get('labels')) for r in valid),
                'chart_ready_median': statistics.median(r['chart_ready_seconds'] for r in valid) if valid else None,
                'complete_median': statistics.median(r['complete_seconds'] for r in valid) if valid else None,
                'memory_median_mib': statistics.median(r['self_peak_bytes'] / 2**20 for r in valid) if valid else None,
                'physical_matches': sum(r.get('physical_fingerprint') == reference for r in cases),
                'source_matches': sum(r.get('source_matches', False) for r in cases),
                'row_matches': sum(r.get('rows_match', False) for r in cases),
                'physical_version_unchanged_runs': sum(r.get('physical_version_unchanged', False) for r in cases),
                'role_correct': sum(r['role_score']['correct_columns'] for r in valid),
                'role_evaluated': sum(r['role_score']['evaluated_columns'] for r in valid),
                'role_wrong': sum(r['role_score']['wrong_count'] for r in valid),
                'role_unknown': sum(r['role_score']['unknown_count'] for r in valid),
                'provider_failed_columns': sum((r.get('coverage') or {}).get('failed_columns', 0) for r in valid)})
    write_json(args.output_root / 'summary.json', {'manifest_hashes': hashes, 'summary': summary})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
