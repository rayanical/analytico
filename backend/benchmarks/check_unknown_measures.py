"""Explicitly opt-in Luna query regression for safely unclassified real quantities."""
import argparse
import json
from pathlib import Path
from real_acceptance import BACKEND, MANIFEST, worker

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--live', action='store_true')
parser.add_argument('--root', type=Path, default=Path('/private/tmp/analytico-real-fixtures'))
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if not args.live:
    parser.error('--live is required; this check makes paid model requests')
specs = {spec['id']: spec for spec in json.loads(MANIFEST.read_text())['datasets']}
cases = {
    'bike': {'prompt': 'Show sum of registered by weekday', 'kind': 'chart', 'x': 'weekday', 'y': 'registered', 'aggregation': 'sum'},
    'bank': {'prompt': 'Show sum of campaign by job', 'kind': 'chart', 'x': 'job', 'y': 'campaign', 'aggregation': 'sum'},
}
records = []
args.output.parent.mkdir(parents=True, exist_ok=True)
for name, case in cases.items():
    spec = {**specs[name], 'queries': [case]}
    records.append(worker(spec, args.root / spec['filename'], True, 1, queries_only=True))
args.output.write_text(json.dumps({'model':'gpt-6-luna','datasets':records},indent=2))
if not all(query['correct'] for record in records for query in record['records'][0]['queries']):
    raise SystemExit(1)
print('Both real-data unknown-measure queries matched the independent oracle.')
