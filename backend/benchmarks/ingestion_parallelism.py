"""Isolated experiments; production ingestion is never changed by this runner.

Every case runs in a fresh process. Concurrent statistics use identical SQL;
process workers read a checkpoint while the parent connection is closed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import contextlib
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
GOLD = BACKEND / 'evals/fresh_accuracy_gold_2026-10-03.json'
OUTPUT = Path('/private/tmp/analytico-parallelism-2026-10-03')
MODES = ['threads2', 'threads4', 'threads8', 'batches_threads2', 'batches_processes2', 'proven_plain_pruning']


def process_query(database, sql):
    import duckdb
    with duckdb.connect(database, read_only=True, config={'threads': '1', 'memory_limit': '64MB'}) as connection:
        return connection.execute(sql).fetchone()


class CachedConnection:
    def __init__(self, connection, results):
        self.connection, self.results = connection, results

    def execute(self, sql, *args, **kwargs):
        if sql in self.results:
            return CachedRow(self.results[sql])
        return self.connection.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.connection, name)


class CachedRow:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


def install_experiment(mode):
    import modules.disk_dataset as disk
    # A nominated benchmark configuration overrides the local user preference.
    os.environ.pop("ANALYTICO_DUCKDB_THREADS", None)
    disk.DUCKDB_THREADS = {'threads4': 4, 'threads8': 8}.get(mode, 2)
    if mode == 'proven_plain_pruning':
        original_analyze = disk.DiskDataset._analyze_and_build
        original_projection = disk.DiskDataset._stats_projection

        def projection(dataset, columns=None):
            expressions, slots = original_projection(dataset, columns)
            proofs = getattr(dataset, '_benchmark_plain_proofs', {})
            zeros = {'percent', 'currency_symbol', 'currency_grouped', 'comma',
                     'ambiguous_grouped', 'grouped_integer', 'grouped_decimal'}
            equal_nonnull = {'nonnull', 'plain', 'percent_inner_plain', 'currency_plain', 'numeric_like'}
            for index, (raw, name) in enumerate(slots):
                if raw in proofs and name in zeros | equal_nonnull:
                    alias = expressions[index].rsplit(' AS ', 1)[1]
                    value = 0 if name in zeros else proofs[raw]
                    expressions[index] = f'{value} AS {alias}'
            return expressions, slots

        def analyze_plain(dataset):
            started = time.perf_counter()
            proofs = {}
            for start in range(0, len(dataset._raw_columns), 4):
                columns = dataset._raw_columns[start:start + 4]
                expressions = []
                for raw in columns:
                    q = disk._q(raw)
                    expressions.extend([f'COUNT({q})', f'COUNT(*) FILTER (WHERE {q} IS NOT NULL AND REGEXP_FULL_MATCH(TRIM({q}), {disk._literal(disk._PLAIN_NUMBER.pattern)}))'])
                counts = dataset._connection.execute(f"SELECT {', '.join(expressions)} FROM source_data").fetchone()
                for index, raw in enumerate(columns):
                    if counts[index * 2] and counts[index * 2] == counts[index * 2 + 1]:
                        proofs[raw] = counts[index * 2]
            dataset._benchmark_plain_proofs = proofs
            proof_seconds = time.perf_counter() - started
            original_analyze(dataset)
            dataset.ingestion_timings['column_validation_and_type_plan'] += proof_seconds
            dataset.ingestion_timings['plain_proof_included_in_validation'] = proof_seconds
        disk.DiskDataset._stats_projection = projection
        disk.DiskDataset._analyze_and_build = analyze_plain
        return
    if not mode.startswith('batches_'):

        return
    original = disk.DiskDataset._analyze_and_build

    def analyze(dataset):
        started = time.perf_counter()
        queries = []
        for start in range(0, len(dataset._raw_columns), 4):
            projections, _ = dataset._stats_projection(dataset._raw_columns[start:start + 4])
            queries.append(f"SELECT {', '.join(projections)} FROM source_data")
        if mode == 'batches_threads2':
            def query(sql):
                cursor = dataset._connection.cursor()
                try:
                    return cursor.execute(sql).fetchone()
                finally:
                    cursor.close()
            with ThreadPoolExecutor(max_workers=2) as pool:
                rows = list(pool.map(query, queries))
        else:
            import duckdb
            dataset._connection.execute('CHECKPOINT')
            dataset._connection.close()
            try:
                with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
                    futures = [pool.submit(process_query, str(dataset.database_path), sql) for sql in queries]
                    rows = [future.result() for future in futures]
            finally:
                dataset._connection = duckdb.connect(str(dataset.database_path), config={
                    'threads': '2', 'memory_limit': disk.DUCKDB_MEMORY_LIMIT,
                    'temp_directory': str(dataset.path / 'spill'),
                    'max_temp_directory_size': disk.DUCKDB_TEMP_LIMIT,
                    'preserve_insertion_order': 'true',
                })
        precompute = time.perf_counter() - started
        connection = dataset._connection
        dataset._connection = CachedConnection(connection, dict(zip(queries, rows)))
        try:
            original(dataset)
        finally:
            dataset._connection = connection
        dataset.ingestion_timings['column_validation_and_type_plan'] += precompute
        dataset.ingestion_timings['parallel_precompute_included_in_validation'] = precompute
    disk.DiskDataset._analyze_and_build = analyze


def spec_path(spec):
    return REPO / spec['local_path'] if spec.get('local_path') else Path('/private/tmp/analytico-real-fixtures') / spec['fixture_file']


def fingerprint(dataset):
    """Compare every typed value and row position, plus deterministic metadata."""
    from modules.disk_dataset import _q
    columns = ', '.join(_q(column) for column in dataset.columns)
    ordered = f'{_q("_row_ordinal")}, {columns}'
    hashes = dataset._connection.execute(
        f'SELECT COUNT(*), SUM(CAST(HASH({columns}) AS HUGEINT)), '
        f'SUM(CAST(HASH({ordered}) AS HUGEINT)) FROM typed_data'
    ).fetchone()
    return {
        'typed_checksums': [str(value) for value in hashes],
        'types': dataset.column_types, 'dtypes': dataset.column_dtypes,
        'formats': dataset.column_formats, 'missing': dataset.raw_missing_counts,
        'unique': dataset.unique_counts,
        'samples': {key: [str(v) for v in values] for key, values in dataset.sample_values.items()},
        'headers': dataset.original_headers, 'schema': dataset.column_schema,
    }


def worker(spec, mode, result_path):
    from modules.disk_dataset import DiskDataset
    from modules.import_policy import ImportSettings
    install_experiment(mode)
    started = time.perf_counter()
    record = {'dataset': spec['id'], 'mode': mode, 'status': 'error'}
    dataset = None
    try:
        dataset = DiskDataset.from_csv(spec_path(spec), import_settings=ImportSettings(delimiter=spec['delimiter']))
        record.update(status='ok', preparation_seconds=time.perf_counter() - started,
                      phases=dataset.ingestion_timings, rows=dataset.row_count)
        # Signals the end of ingestion for the independent RSS sampler.
        Path(str(result_path) + '.ready').write_text('ready')
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record['self_peak_rss_bytes'] = int(peak if sys.platform == 'darwin' else peak * 1024)
        record['fingerprint'] = fingerprint(dataset)
        with dataset.source_path.open('rb') as stream:
            record['source_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        record['source_matches_gold'] = record['source_sha256'] == spec['sha256']
        record['rows_match_gold'] = dataset.row_count == spec['rows']
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error)[:350], preparation_seconds=time.perf_counter() - started)
    finally:
        if dataset is not None:
            dataset.close()
    result_path.write_text(json.dumps(record, default=str))


def tree_rss(pid):
    lines = subprocess.check_output(['/bin/ps', '-axo', 'pid=,ppid=,rss='], text=True).splitlines()
    processes = [tuple(map(int, line.split())) for line in lines if len(line.split()) == 3]
    children = {pid}
    while True:
        new = {p for p, parent, _ in processes if parent in children}
        if new <= children:
            break
        children |= new
    return sum(rss * 1024 for p, _, rss in processes if p in children)


def run_matrix(repeats, modes):
    specs = json.loads(GOLD.read_text())['datasets']
    jobs = [(spec, mode, repeat) for spec in specs for mode in modes for repeat in range(repeats)]
    random.Random(20261003).shuffle(jobs)
    OUTPUT.mkdir(exist_ok=True)
    records = []
    for index, (spec, mode, repeat) in enumerate(jobs):
        result_path = OUTPUT / f'{spec["id"]}-{mode}-{repeat}.json'
        ready = Path(str(result_path) + '.ready')
        ready.unlink(missing_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', spec['id'], '--mode', mode, '--output', str(result_path)]
        started = time.perf_counter()
        with (OUTPUT / 'worker.log').open('a') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            peak, samples = 0, 0
            while process.poll() is None:
                if not ready.exists():
                    peak = max(peak, tree_rss(process.pid))
                    samples += 1
                time.sleep(.15)
        record = json.loads(result_path.read_text()) if result_path.exists() else {'dataset': spec['id'], 'mode': mode, 'status': 'crash'}
        record.update(repeat=repeat, sampled_tree_peak_rss_bytes=peak, rss_samples=samples,
                      process_wall_seconds=time.perf_counter() - started)
        records.append(record)
        (OUTPUT / 'records.json').write_text(json.dumps(records, default=str))
        print(json.dumps({'progress': f'{index + 1}/{len(jobs)}', 'dataset': spec['id'], 'mode': mode,
                          'status': record['status'], 'seconds': round(record.get('preparation_seconds', 0), 3),
                          'memory_mib': round(peak / 2**20, 1)}), flush=True)
    baseline = {spec['id']: next((r for r in records if r['dataset'] == spec['id'] and r['mode'] == 'threads2' and r['status'] == 'ok'), None) for spec in specs}
    for record in records:
        reference = baseline[record['dataset']]
        record['identical_to_baseline'] = record.get('fingerprint') == reference['fingerprint'] if reference else None
    (OUTPUT / 'records.json').write_text(json.dumps(records, default=str))
    summary = []
    for spec in specs:
        for mode in modes:
            selected = [r for r in records if r['dataset'] == spec['id'] and r['mode'] == mode]
            valid = [r for r in selected if r['status'] == 'ok']
            summary.append({'dataset': spec['id'], 'mode': mode, 'runs': len(selected), 'successes': len(valid),
                'identical': sum(r['identical_to_baseline'] is True for r in selected),
                'source_checks': sum(r.get('source_matches_gold', False) for r in selected),
                'row_checks': sum(r.get('rows_match_gold', False) for r in selected),
                'median_seconds': statistics.median(r['preparation_seconds'] for r in valid) if valid else None,
                'median_tree_peak_mib': statistics.median(r['sampled_tree_peak_rss_bytes'] / 2**20 for r in valid) if valid else None,
                'median_self_peak_mib': statistics.median(r['self_peak_rss_bytes'] / 2**20 for r in valid) if valid else None,
                'median_phases': {key: statistics.median(r['phases'].get(key, 0) for r in valid) for key in valid[0]['phases']} if valid else {},
                'errors': sorted({r.get('error', r['status']) for r in selected if r['status'] != 'ok'})})
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--mode', choices=MODES)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--only-mode', choices=MODES)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    global OUTPUT
    if args.output_root:
        OUTPUT = args.output_root
    if args.worker:
        spec = next(s for s in json.loads(GOLD.read_text())['datasets'] if s['id'] == args.worker)
        worker(spec, args.mode, args.output)
    else:
        run_matrix(args.repeats, [args.only_mode] if args.only_mode else MODES)


if __name__ == '__main__':
    main()
