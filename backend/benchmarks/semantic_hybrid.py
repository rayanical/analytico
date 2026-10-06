"""Opt-in role-only schema enrichment experiment; never alters production data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

import httpx

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import core.config as config

from modules.schema_interpretation import (PROMPT_VERSION, SchemaProposals, SYSTEM,
    selected, validate_proposals, compatible, model_state, proposal_json_schema)

def call(dataset, trial):
    state = model_state(dataset, trial)
    encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode()) > 65536:
        raise ValueError('Schema state exceeds bounded request')
    request = {'model': 'gpt-6-luna', 'store': False, 'reasoning': {'effort': 'none'},
        'input': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': encoded}],
        'text': {'format': {'type': 'json_schema', 'name': 'schema_roles', 'strict': True, 'schema': proposal_json_schema()}},
        'max_output_tokens': 4096}
    tick = perf_counter()
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post('https://api.openai.com/v1/responses', json=request,
                headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY']})
        response.raise_for_status()
        body = response.json()
        if body.get('status') != 'completed':
            raise ValueError('Incomplete response')
        texts = [content['text'] for item in body.get('output', []) for content in item.get('content', []) if content.get('type') == 'output_text']
        predictions = validate_proposals(''.join(texts), state['requested_indexes'])
        roles = {c['column']: c['local_usage']['role'] for c in dataset['columns']}
        proposed_roles = dict(roles)
        decisions = []
        for index, prediction in predictions.items():
            column = dataset['columns'][index]
            proposed_roles[column['column']] = prediction['role']
            accepted = prediction['evidence_strength'] == 'strong' and compatible(column, prediction)
            # A rejected proposal preserves the existing inference, including unknown.
            if accepted:
                roles[column['column']] = prediction['role']
            decisions.append({'column': column['column'], 'local_role': column['local_usage']['role'],
                'proposal': prediction, 'accepted_for_experiment': accepted})
        return {'dataset': dataset['dataset'], 'trial': trial, 'status': 'ok', 'seconds': perf_counter() - tick,
                'requests': 1, 'selected_columns': len(state['requested_indexes']), 'total_columns': len(dataset['columns']),
                'roles': roles, 'proposed_roles': proposed_roles, 'decisions': decisions,
                'usage': body.get('usage'), 'returned_model': body.get('model'), 'input_bytes': len(encoded.encode())}
    except Exception as error:
        # Exception/provider response text can contain credentials or internal URLs.
        return {'dataset': dataset['dataset'], 'trial': trial, 'status': 'failed', 'failure_type': type(error).__name__,
                'seconds': perf_counter() - tick, 'requests': 1, 'selected_columns': len(state['requested_indexes'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not args.live or not os.environ.get('OPENAI_API_KEY', '').strip():
        parser.error('Explicit --live and an existing configured key are required.')
    assert config.OPENAI_MODEL == 'gpt-6-luna'
    source = args.input.read_bytes()
    inputs = json.loads(source)
    assert len(inputs['datasets']) == 5
    report = {'arm': 'current_plus_luna_roles', 'model': 'gpt-6-luna', 'prompt_version': PROMPT_VERSION,
        'input_sha256': hashlib.sha256(source).hexdigest(),
        'selector': 'Unknown or evidence other than explicit_role/validated_datetime/validated_calendar_year; qualitative weak evidence, not calibrated confidence.',
        'promotion_rule': 'Experimental role-only merge: strong self-reported evidence plus physical/profile compatibility. No units/casts/aggregations applied; no business meaning guarantee.',
        'scope': 'One request per schema per repeat. AI elapsed excludes source ingestion and browser; use cached local-ready measurements separately.', 'records': []}
    for trial in range(1, args.repeats + 1):
        for dataset in inputs['datasets']:
            result = call(dataset, trial)
            report['records'].append(result)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report['process_peak_rss_bytes'] = rss if sys.platform == 'darwin' else rss * 1024
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({k: result[k] for k in ['dataset', 'trial', 'status', 'seconds', 'selected_columns']}), flush=True)
    if any(r['status'] != 'ok' for r in report['records']):
        raise SystemExit(1)
