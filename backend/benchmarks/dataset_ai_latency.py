"""Explicit live dataset benchmark; experimental strategies never alter production."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import contextlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from threading import Lock
from time import perf_counter, sleep
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
STRATEGIES = ('off', 'current', 'parallel_all', 'hybrid', 'hybrid_cautious')


def needs_ai(role, nonnull):
    """Experimental semantic triage, not a guarantee of correct meaning or units."""
    return nonnull > 0 and role not in {'metric', 'temporal'}


def needs_ai_cautious(role, nonnull, unique_count, values):
    """Experimental gate: prefer extra calls over silently accepting codes.

    These cardinality thresholds are heuristics, not calibrated confidence.
    No column names or dataset-specific patterns enter the decision.
    """
    if not nonnull:
        return False
    if role == 'temporal':
        return False
    if role != 'metric' or unique_count is None:
        return True
    if unique_count <= 32 or unique_count / nonnull >= .9:
        return True
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        unsigned = text.lstrip('+-')
        if len(unsigned) > 1 and unsigned[0] == '0' and unsigned[1].isdigit():
            return True
        try:
            if not Decimal(text).is_finite():
                return True
        except InvalidOperation:
            return True
    return False


def all_inputs(dataset, source_frame):
    from modules.data_janitor import _interpretation_input
    if source_frame is not None:
        headers = [str(column) for column in source_frame.columns]
        return [(dataset.column_names[index], _interpretation_input(header, source_frame.iloc[:, index], headers))
                for index, header in enumerate(headers)]
    return dataset.disk.interpretation_inputs(limit=256)


def worker(path, label, strategy, live, parallelism=4):
    import core.config as config
    from services import ingestion_service, csv_ingestion
    from services.enrichment_service import get_enrichment_status, enqueue_enrichment
    from modules import column_interpretation
    from storage import get_dataset, DATASETS
    if not live and strategy != 'off':
        raise ValueError('Live strategies require --live.')
    if strategy == 'off':
        os.environ['OPENAI_API_KEY'] = ''
    elif not os.environ.get('OPENAI_API_KEY', '').strip():
        raise ValueError('A configured OpenAI key is required; credentials are never output.')
    os.environ['COLUMN_INTERPRETER'] = 'luna' if strategy != 'off' else 'off'
    os.environ['COLUMN_INTERPRETER_MODEL'] = 'gpt-6-luna'
    assert config.OPENAI_MODEL == 'gpt-6-luna', 'Benchmark summary model must be GPT-6 Luna.'
    original_interpret = column_interpretation.interpret_column
    original_summary = ingestion_service._generate_business_summary
    observations = []
    lock = Lock()
    sampling_times = []
    selected_columns = []

    def interpret(payload, **kwargs):
        kwargs['use_cache'] = False
        result = original_interpret(payload, **kwargs)
        with lock:
            observations.append({'column': payload['column_name'], 'status': result.status,
                'model': result.model, 'latency_ms': result.latency_ms, 'usage': result.usage,
                'cache_hit': result.cache_hit, 'error_code': result.error_code,
                'decision': result.decision.model_dump() if result.decision else None})
        return result

    def summary(*args):
        tick = perf_counter()
        result = original_summary(*args)
        with lock:
            observations.append({'kind': 'summary', 'model': config.OPENAI_MODEL,
                'seconds': perf_counter() - tick, 'available': bool(result)})
        return result

    def queue(dataset, source_frame=None):
        tick = perf_counter()
        payloads = all_inputs(dataset, source_frame)
        if strategy == 'hybrid':
            payloads = [(column, payload) for column, payload in payloads
                        if needs_ai(dataset.column_types.get(column), dataset.row_count - dataset.missing_counts.get(column, 0))]
        elif strategy == 'hybrid_cautious':
            counts = (dataset.disk.column_stats if hasattr(dataset, 'disk') else None)
            payloads = [(column, payload) for column, payload in payloads
                        if needs_ai_cautious(dataset.column_types.get(column),
                            dataset.row_count - dataset.missing_counts.get(column, 0),
                            counts[column].unique_count if counts is not None else int(dataset.df[column].nunique()),
                            payload['values'])]
        selected_columns.extend(column for column, _ in payloads)
        frame = dataset.sample_frame(3).iloc[:, :20].copy(deep=True)
        sampling_times.append(perf_counter() - tick)
        def work():
            # Summary and independent column classifications overlap. Keep a
            # fixed request pool; no unbounded per-column concurrency.
            proposals = {}
            with ThreadPoolExecutor(max_workers=parallelism) as pool:
                description = pool.submit(summary, dataset.filename, frame)
                futures = [(column, pool.submit(interpret, payload, provider='luna')) for column, payload in payloads]
                for column, future in futures:
                    result = future.result()
                    proposals[column] = {'status': result.status, 'decision': result.decision.model_dump() if result.decision else None}
                text = description.result()
            return {'summary': text, 'interpretation_proposals': proposals}
        return enqueue_enrichment(dataset.id, dataset.cache_version, work)['status']

    patches = [patch('modules.column_interpretation.interpret_column', side_effect=interpret),
               patch('services.ingestion_service._generate_business_summary', side_effect=summary)]
    if strategy in {'parallel_all', 'hybrid', 'hybrid_cautious'}:
        patches += [patch('services.ingestion_service.queue_dataset_enrichment', side_effect=queue),
                    patch('services.csv_ingestion.queue_dataset_enrichment', side_effect=queue)]
    start = perf_counter()
    with contextlib.ExitStack() as stack:
        for item in patches: stack.enter_context(item)
        telemetry = io.StringIO()
        with contextlib.redirect_stdout(telemetry):
            response = csv_ingestion.ingest_csv(path, f'{label}.csv', '/dataset-benchmark', defer_enrichment=True)
            ready = perf_counter() - start
            dataset = get_dataset(response.dataset_id)
            ready_schema = json.dumps(dataset.column_schema, sort_keys=True)
            deadline = perf_counter() + 180
            while True:
                status = get_enrichment_status(dataset.id, dataset.cache_version)
                if status['status'] in {'done', 'error', 'disabled'}: break
                if perf_counter() > deadline: raise RuntimeError('Enrichment benchmark timed out.')
                sleep(.01)
            complete = perf_counter() - start
            assert json.dumps(dataset.column_schema, sort_keys=True) == ready_schema, 'Background job mutated schema.'
            measurement = next((json.loads(line) for line in telemetry.getvalue().splitlines()
                                if line.startswith('{') and '"ingestion_performance"' in line), {})
            output = {'dataset': label, 'strategy': strategy, 'rows': response.row_count,
                'columns': len(response.columns), 'source_bytes': Path(path).stat().st_size,
                'engine': 'disk' if hasattr(dataset, 'disk') else 'pandas',
                'ready_seconds': ready, 'complete_seconds': complete,
                'remaining_ai_after_ready_seconds': max(0, complete - ready), 'status': status['status'],
                'coverage': status.get('coverage'),
                'ingestion_phase_seconds': measurement.get('phase_seconds'),
                'sample_preparation_seconds': sum(sampling_times) if sampling_times else None,
                'selected_columns': selected_columns if strategy in {'parallel_all', 'hybrid', 'hybrid_cautious'} else None,
                'interpretation_requests': sum('kind' not in item for item in observations),
                'summary_requests': sum(item.get('kind') == 'summary' for item in observations),
                'peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024),
                'proposals': len(status['interpretation_proposals']), 'calls': observations,
                'initial_profile': response.profile.model_dump(), 'column_roles': dataset.column_types}
            DATASETS.pop(dataset.id)
            dataset.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=('taxi', 'gapminder', 'wide'), default=['taxi', 'gapminder', 'wide'])
    parser.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--label', help=argparse.SUPPRESS)
    parser.add_argument('--strategy', choices=STRATEGIES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3: parser.error('Repetitions must be between one and three.')
    if args.worker:
        print(json.dumps(worker(args.worker, args.label, args.strategy, args.live), allow_nan=False))
        return
    if any(strategy != 'off' for strategy in args.strategies) and not args.live:
        parser.error('Explicit --live is required for provider calls.')
    from core.config import DEMO_DATASETS
    with tempfile.TemporaryDirectory(prefix='analytico-ai-benchmark-') as directory:
        wide = Path(directory) / 'wide.csv'
        with wide.open('w') as stream:
            stream.write(','.join(['category', 'label', *[f'measure_{i}' for i in range(48)]]) + '\n')
            for row in range(10000):
                stream.write(','.join([f'group_{row % 10}', f'label_{row % 23}', *[str((row + i) % 1001) for i in range(48)]]) + '\n')
        results = []
        for label in args.datasets:
            path = wide if label == 'wide' else DEMO_DATASETS[label]['path']
            if not path.is_file(): parser.error(f'Local dataset {label} is unavailable.')
            for strategy in args.strategies:
                for trial in range(args.repeats):
                    command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(path), '--label', label,
                               '--strategy', strategy, '--output', str(args.output)]
                    if args.live: command.append('--live')
                    child = subprocess.run(command, capture_output=True, text=True, timeout=240)
                    if child.returncode: raise RuntimeError(f'{label}/{strategy} worker failed; no provider error content is exposed.')
                    result = json.loads(child.stdout)
                    result['trial'] = trial + 1
                    results.append(result)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps({'live': args.live, 'application_cache': 'bypassed',
                        'parallel_requests': 4, 'results': results}, indent=2) + '\n')
                    print(f"{label}/{strategy}/{trial + 1}: ready {result['ready_seconds']:.3f}s, complete {result['complete_seconds']:.3f}s, {result['interpretation_requests']} column calls", flush=True)


if __name__ == '__main__': main()
