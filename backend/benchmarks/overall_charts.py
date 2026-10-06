"""Compare overall charts on complete public files against a streaming Decimal oracle."""
import sys, json, csv, time, math
from pathlib import Path
import argparse
from decimal import Decimal
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'backend'))
from services.csv_ingestion import ingest_csv
from services.query_service import run_query
from services.aggregation_service import run_aggregate
from models import QueryRequest, AggregateRequest, FilterConfig
from core.config import OPENAI_MODEL

def main():
    parser = argparse.ArgumentParser(description='Full-source overall chart correctness checks; --live adds paid Luna calls.')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    specs = [('tips', 'total_bill', Path('/private/tmp/analytico-real-fixtures/tips.csv')), ('diamonds', 'price', Path('/private/tmp/analytico-real-fixtures/diamonds.csv')), ('taxi', 'fare_amount', REPO / 'backend/datasets/2021_Green_Taxi_Trip_Data_20260221.csv')]
    records = []
    for name, col, path in specs:
        total = Decimal(0)
        count = row_count = 0
        with path.open() as f:
            for row in csv.DictReader(f):
                row_count += 1
                if row[col].strip():
                    total += Decimal(row[col].replace(',', ''))
                    count += 1
        gold = {'sum': float(total), 'mean': float(total / count), 'count': count, 'rows': row_count}
        with path.open('rb') as f:
            response = ingest_csv(f, path.name, '/overall-eval', enqueue_enrichment=False, ai_column_analysis=False)
        for operation, expected in gold.items():
            start = time.perf_counter()
            chart = run_aggregate(AggregateRequest(dataset_id=response.dataset_id, y_axis_keys=[] if operation == 'rows' else [col], aggregation='count' if operation == 'rows' else operation))
            value = chart.data[0][chart.y_axis_keys[0]]
            passed = math.isclose(float(value), float(expected), rel_tol=1e-09, abs_tol=1e-06)
            records.append(dict(dataset=name, mode='engine', operation=operation, passed=passed, seconds=time.perf_counter() - start, value=value, expected=expected))
        for prompt, operation in [] if not args.live else [(f'What is the total {col} across all rows?', 'sum'), (f'What is the average {col} across all rows?', 'mean'), ('How many rows are in this dataset?', 'rows')]:
            start = time.perf_counter()
            chart = run_query(QueryRequest(dataset_id=response.dataset_id, user_prompt=prompt))
            value = chart.data[0][chart.y_axis_keys[0]] if chart.data else None
            passed = chart.aggregation_scope == 'overall' and chart.chart_type == 'bar' and (value is not None) and math.isclose(float(value), float(gold[operation]), rel_tol=1e-09, abs_tol=1e-06)
            records.append(dict(dataset=name, mode='live_luna', prompt=prompt, passed=passed, seconds=time.perf_counter() - start, value=value, expected=gold[operation]))
            print(json.dumps(records[-1]), flush=True)
    result = dict(model=OPENAI_MODEL, passed=sum((r['passed'] for r in records)), total=len(records), records=records)
    print(json.dumps(result, indent=2))
    assert all((r['passed'] for r in records)), result
if __name__ == '__main__':
    main()
