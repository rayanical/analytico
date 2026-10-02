"""Equal-work native-loader experiment with isolated processes and optional Luna."""
import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from time import perf_counter, sleep
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


def worker(source, label, strategy, live):
    import core.config as config
    from modules.disk_dataset import DiskDataset
    from native_csv import native_loader, ORIGINAL_LOADER
    from models import AggregateRequest
    from services.csv_ingestion import ingest_csv
    from services.aggregation_service import run_aggregate
    from services.enrichment_service import get_enrichment_status
    from storage import get_dataset, DATASETS
    if live:
        if not os.environ.get('OPENAI_API_KEY', '').strip():
            raise ValueError('Live benchmark requires a configured key.')
        assert config.OPENAI_MODEL == 'gpt-6-luna'
    else:
        os.environ['OPENAI_API_KEY'] = ''
    os.environ['COLUMN_INTERPRETER'] = 'luna' if live else 'off'
    os.environ['COLUMN_INTERPRETER_MODEL'] = 'gpt-6-luna'
    loader_times = []
    def loader(self, chunk_size):
        tick = perf_counter()
        (native_loader if strategy == 'native' else ORIGINAL_LOADER)(self, chunk_size)
        loader_times.append(perf_counter() - tick)
    telemetry = io.StringIO()
    with patch.object(DiskDataset, '_ingest_csv_chunks', loader), contextlib.redirect_stdout(telemetry):
        start = perf_counter()
        # Force disk for the wide workload to exercise both transport paths.
        response = ingest_csv(source, f'{label}.csv', '/pattern-benchmark',
                              engine='disk' if label == 'wide' else 'auto', defer_enrichment=True)
        ready = perf_counter() - start
        dataset = get_dataset(response.dataset_id)
        try:
            schema = json.dumps(dataset.column_schema, sort_keys=True)
            x = next((c for c in dataset.column_names if dataset.column_types[c] == 'categorical'), dataset.column_names[0])
            y = next((c for c in dataset.column_names if dataset.column_types[c] == 'metric' and c != x), None)
            if y is None:
                y = next(c for c in dataset.column_names if c != x)
            chart_start = perf_counter()
            chart = run_aggregate(AggregateRequest(dataset_id=dataset.id, x_axis_key=x,
                y_axis_keys=[y], aggregation='sum' if dataset.column_types[y] == 'metric' else 'count', limit=50))
            chart_seconds = perf_counter() - chart_start
            while True:
                status = get_enrichment_status(dataset.id, dataset.cache_version)
                if status['status'] in {'done', 'error', 'disabled'}:
                    break
                if perf_counter() - start > 180:
                    raise RuntimeError('Enrichment timeout.')
                sleep(.02)
            complete = perf_counter() - start
            assert schema == json.dumps(dataset.column_schema, sort_keys=True), 'AI mutated schema.'
            output = {'dataset': label, 'strategy': strategy, 'luna': live,
                'rows': response.row_count, 'columns': len(response.columns),
                'engine': 'disk' if hasattr(dataset, 'disk') else 'pandas',
                'source_bytes': Path(source).stat().st_size, 'ready_seconds': ready,
                'loader_seconds': sum(loader_times), 'chart_seconds': chart_seconds,
                'complete_seconds': complete, 'ai_remaining_seconds': max(0, complete-ready),
                'peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024),
                'schema': dataset.column_schema, 'profile': response.profile.model_dump(),
                'chart': chart.model_dump(), 'ai_status': status['status'], 'coverage': status.get('coverage'),
                'proposals': status.get('interpretation_proposals', {})}
        finally:
            DATASETS.pop(dataset.id, None)
            dataset.close()
    return output


def flow_worker(source, label, strategy, live):
    """Actual staging, automatic confirmation and immediate Luna chart request."""
    if not live:
        raise ValueError('Natural-language flow requires --live.')
    import core.config as config
    from modules.disk_dataset import DiskDataset
    from native_csv import native_loader, ORIGINAL_LOADER
    from services.import_preview import stage_import, confirm_import
    from services.query_service import run_query
    from services.enrichment_service import manager
    from storage import DATASETS
    from models import QueryRequest
    assert config.OPENAI_MODEL == 'gpt-6-luna'
    os.environ['COLUMN_INTERPRETER'] = 'luna'
    os.environ['COLUMN_INTERPRETER_MODEL'] = 'gpt-6-luna'
    prompts = {'taxi': 'Show a bar chart of total tip_amount by payment_type.',
               'gapminder': 'Show a bar chart of average lifeExp by continent.'}
    try:
        with patch.object(DiskDataset, '_ingest_csv_chunks', native_loader if strategy == 'native' else ORIGINAL_LOADER), contextlib.redirect_stdout(io.StringIO()):
            start = perf_counter()
            preview = stage_import(source, f'{label}.csv')
            staged = perf_counter()
            response = confirm_import(preview['import_id'])
            ready = perf_counter()
            chart = run_query(QueryRequest(dataset_id=response.dataset_id, user_prompt=prompts[label]))
            done = perf_counter()
        if not chart.data:
            raise AssertionError('Expected supported chart, received clarification.')
        return {'dataset': label, 'strategy': strategy, 'luna': True, 'rows': response.row_count,
                'model': config.OPENAI_MODEL, 'preview_seconds': staged-start,
                'ready_seconds': ready-staged, 'question_seconds': done-ready,
                'total_processing_seconds': done-start, 'result': 'chart',
                'axes': [chart.x_axis_key, chart.y_axis_keys], 'data': chart.data,
                'aggregation': chart.aggregation}
    finally:
        manager.shutdown()
        for dataset in list(DATASETS.values()):
            dataset.close()
        DATASETS.clear()


def equal_payload(left, right):
    """Parallel floating-point sums need tolerance; strings/integers stay exact."""
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal_payload(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equal_payload(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--flow', action='store_true')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--label')
    parser.add_argument('--strategy', choices=['current', 'native'])
    args = parser.parse_args()
    if args.worker:
        print(json.dumps((flow_worker if args.flow else worker)(args.worker, args.label, args.strategy, args.live), allow_nan=False))
        return
    if not 1 <= args.repeats <= 3:
        parser.error('Use one to three repetitions.')
    from core.config import DEMO_DATASETS
    results = []
    with tempfile.TemporaryDirectory(prefix='analytico-patterns-') as directory:
        sources = {label: item['path'] for label, item in DEMO_DATASETS.items()}
        for label, rows, columns in ([] if args.flow else [('million', 1_000_000, 5), ('wide', 10_000, 50)]):
            source = Path(directory) / f'{label}.csv'
            with source.open('w') as stream:
                stream.write(','.join(['category', 'amount_usd', 'units', 'score', 'label', *[f'measure_{i}' for i in range(columns-5)]])+'\n')
                for row in range(rows):
                    stream.write(','.join([f'group_{row%10}', str(row%1001), str(row%71), str(row%97), f'label_{row%23}', *[str((row+i)%1001) for i in range(columns-5)]])+'\n')
            sources[label] = source
        for label, source in sources.items():
            arms = [('current', args.flow), ('native', args.flow)]
            if args.live and not args.flow and label in {'taxi', 'gapminder', 'wide'}:
                arms.append(('native', True))
            for strategy, live in arms:
                for trial in range(args.repeats):
                    command = [sys.executable, __file__, '--worker', str(source), '--label', label,
                               '--strategy', strategy, '--output', str(args.output)]
                    if live:
                        command.append('--live')
                    if args.flow:
                        command.append('--flow')
                    child = subprocess.run(command, capture_output=True, text=True, timeout=240)
                    if child.returncode:
                        raise RuntimeError(f'{label}/{strategy} failed; raw source/provider errors withheld.')
                    result = json.loads(child.stdout)
                    result['trial'] = trial + 1
                    previous = next((r for r in results if r['dataset'] == label), None)
                    if previous:
                        # UUID/version identifiers are excluded; payloads are identical.
                        for field in (['axes', 'data', 'aggregation'] if args.flow else ['schema', 'profile', 'chart']):
                            if field == 'chart':
                                left, right = dict(previous[field]), dict(result[field])
                                left.pop('dataset_id', None); right.pop('dataset_id', None)
                                assert equal_payload(left, right), f'{label}: chart parity failed'
                            else:
                                assert equal_payload(previous[field], result[field]), f'{label}: {field} parity failed'
                    results.append(result)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps({'results': results}, indent=2, allow_nan=False)+'\n')
                    print(f'{label} {strategy} luna={live} trial={trial+1}: ready={result["ready_seconds"]:.3f}s total={result.get("total_processing_seconds", result.get("complete_seconds", 0)):.3f}s', flush=True)


if __name__ == '__main__':
    main()
