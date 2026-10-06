"""Compare recorded deterministic roles without making model requests."""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.data_janitor import legacy_normalize_header


def score(path, specs):
    rows = []
    for dataset in json.loads(path.read_text())['datasets']:
        spec = specs[dataset['dataset']]
        records = dataset['records']
        first = records[0]
        roles = first['local_roles']
        gold = {
            legacy_normalize_header(column): [role, *spec.get('role_alternatives', {}).get(column, [])]
            for column, role in spec['roles'].items()
        }
        rows.append({
            'dataset': dataset['dataset'],
            'correct': sum(roles[column] in allowed for column, allowed in gold.items()),
            'total': len(gold),
            'unknown': sum(role == 'unknown' for role in roles.values()),
            'wrong': [column for column, allowed in gold.items()
                      if roles[column] != 'unknown' and roles[column] not in allowed],
            'ready_seconds': statistics.median(r['local_ready_seconds'] for r in records),
            'peak_rss_mb': dataset['process_peak_rss_bytes'] / 1e6,
            'cold_aggregate_ms': 1000 * statistics.median(r['cold_aggregate_api_seconds'] for r in records),
            'warm_aggregate_ms': 1000 * statistics.median(r['warm_aggregate_api_seconds'] for r in records),
            'default_chart': first['default_chart'],
            'full_value_checks': all(r['aggregate_matches_oracle'] and r['source_missing_counts_match'] for r in records),
            # Historical baseline did not record a byte hash. Missing is not failure.
            'source_bytes_preserved': all(r['source_bytes_preserved'] for r in records)
                if all('source_bytes_preserved' in r for r in records) else None,
        })
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / 'evals/real_dataset_manifest.json'
    specs = {s['id']: s for s in json.loads(manifest.read_text())['datasets']}
    result = {'before': score(args.before, specs), 'after': score(args.after, specs)}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for label, rows in result.items():
        for row in rows:
            print(label, row['dataset'], row['correct'], row['unknown'], len(row['wrong']),
                  round(row['ready_seconds'], 3), round(row['peak_rss_mb'], 1))
