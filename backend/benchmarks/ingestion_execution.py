"""Screen parser/parallel candidates, then repeat promising modes on real inputs.

No AI calls. Reuses the exact full-source oracle, isolated process and separate
verification budget from ingestion_batching. Production defaults are untouched.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
import sys

import ingestion_batching as base

PARSER_MODES = {'overlap_validation', 'native_only', 'single_pass'}
PARALLEL_MODES = {'parallel_stats', 'parallel_profiles', 'fused_profiles', 'plain_pruning4'}
MODES = ['baseline', *sorted(PARSER_MODES), *sorted(PARALLEL_MODES), 'overlap_parallel', 'production_final']
BASE_INSTALL = base.install_experiment
CONTEXT = ExitStack()


def install(mode, _combined_size=4):
    if mode == 'production_final':
        BASE_INSTALL('production_final', 4)
        return
    BASE_INSTALL('baseline', 4)
    if mode == 'overlap_parallel':
        from parallel_candidates import install as parallel_install
        from parser_candidates import install as parser_install
        parallel_install('parallel_stats')
        CONTEXT.enter_context(parser_install('overlap_validation'))
    elif mode in PARSER_MODES:
        from parser_candidates import install as parser_install
        CONTEXT.enter_context(parser_install(mode))
    elif mode in PARALLEL_MODES:
        from parallel_candidates import install as parallel_install
        parallel_install(mode)
    elif mode != 'baseline':
        raise ValueError(f'Unknown candidate: {mode}')


def run(args):
    specs = base.gold_specs()
    if args.datasets:
        unknown = set(args.datasets) - {s['id'] for s in specs}
        if unknown:
            raise ValueError(f'Unknown datasets: {sorted(unknown)}')
        specs = [s for s in specs if s['id'] in args.datasets]
    args.output_root.mkdir(parents=True, exist_ok=True)
    stress = base.stress_spec(args.output_root)
    jobs = [(s, mode, repeat) for s in specs for mode in args.modes
            for repeat in range(args.start_repeat, args.start_repeat + args.repeats)]
    random.Random(20261007 + args.start_repeat).shuffle(jobs)
    records = []
    raw = args.output_root / 'records.jsonl'
    raw.write_text('')
    for index, (spec, mode, repeat) in enumerate(jobs, 1):
        record = base.run_case(spec, mode, repeat, args.output_root, 4)
        records.append(record)
        with raw.open('a') as stream:
            stream.write(json.dumps(record, default=str) + '\n')
        print(json.dumps({'progress': f'{index}/{len(jobs)}',
            'dataset': spec['id'], 'mode': mode,
            'ingestion': record.get('ingestion_status', record['status']),
            'verification': record.get('verification_status'),
            'seconds': round(record.get('preparation_seconds', 0), 3),
            'peak_mib': round(record.get('ingestion_peak_rss_bytes', 0) / 2**20, 1)}), flush=True)
    for mode in args.modes:
        record = base.run_case(stress, mode, args.start_repeat, args.output_root, 4)
        records.append(record)
        with raw.open('a') as stream:
            stream.write(json.dumps(record, default=str) + '\n')
    references = []
    if args.reference_root:
        references = [json.loads(line) for line in (args.reference_root / 'records.jsonl').read_text().splitlines()
                      if json.loads(line)['mode'] == 'baseline']
    payload = base.summarize([*records, *references], [*specs, stress], args.modes, stress['id'])
    payload['methodology'].update(repeats=args.repeats,
        start_repeat=args.start_repeat, random_seed=20261007 + args.start_repeat,
        purpose='Exploratory screen' if args.repeats == 1 else 'Fresh randomized repetitions',
        ai_calls=0,
        parser_modes={'overlap_validation': 'Strict validation overlaps raw load; join before analysis/publication',
                      'native_only': 'Strict native parser and full-source value checks; allow benign trailing-quote whitespace',
                      'single_pass': 'Canonical row validation and loading share one streaming parser'},
        exact_values_required=True)
    payload['results'] = [r for r in payload['results'] if r['dataset'] != stress['id']]
    (args.output_root / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'summary': str(args.output_root / 'summary.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--mode', choices=MODES)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--spec-json')
    parser.add_argument('--combined-size', type=int, default=4)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--repeats', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--start-repeat', type=int, default=0)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('/private/tmp/analytico-execution-screen-2026-10-07'))
    args = parser.parse_args()
    # Child cases must re-enter this runner, not the original batching CLI.
    base.__file__ = str(Path(__file__).resolve())
    base.install_experiment = install
    if args.worker:
        spec = json.loads(args.spec_json)
        try:
            return base.worker(spec, args.mode, 4, args.output)
        finally:
            CONTEXT.close()
    return run(args)


if __name__ == '__main__':
    main()
