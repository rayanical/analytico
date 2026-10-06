"""Offline ingestion/chart benchmark; isolate each workload's process peak RSS."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
WORKLOADS = {"small": (10_000, 5), "wide": (10_000, 50), "million": (1_000_000, 5), "unique": (1_000_000, 5)}


def worker(source, name, engine="dataframe"):
    # Set again after dotenv initialization; this benchmark never calls providers.
    import core.config  # noqa: F401
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["COLUMN_INTERPRETER"] = "off"
    from models import AggregateRequest
    from services.aggregation_service import run_aggregate
    from services.ingestion_service import ingest_dataframe
    from storage import get_dataset
    from utils.dataframe_utils import read_csv_fast

    start = perf_counter()
    telemetry = io.StringIO()
    with contextlib.redirect_stdout(telemetry):
        if engine == "dataframe":
            frame = read_csv_fast(source)
            parsed = perf_counter()
            uploaded = ingest_dataframe(frame, "synthetic.csv", "/benchmark")
        else:
            from services.csv_ingestion import ingest_csv
            parsed = start
            uploaded = ingest_csv(source, "synthetic.csv", "/benchmark", engine=engine, defer_enrichment=False)
    ingested = perf_counter()
    measurement = next((json.loads(line) for line in telemetry.getvalue().splitlines()
                        if line.startswith('{') and '"ingestion_performance"' in line), {})
    request = AggregateRequest(dataset_id=uploaded.dataset_id, x_axis_key="category",
                               y_axis_keys=["amount_usd"], aggregation="sum", limit=50)
    cold = perf_counter()
    result = run_aggregate(request)
    cold_seconds = perf_counter() - cold
    warm_times = []
    for _ in range(3):
        tick = perf_counter()
        repeated = run_aggregate(request)
        warm_times.append(perf_counter() - tick)
        if repeated.model_dump() != result.model_dump():
            raise AssertionError("Repeated chart results changed")
    dataset = get_dataset(uploaded.dataset_id)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
    except ImportError:
        peak_bytes = None
    retained = None if hasattr(dataset, "disk") else int(dataset.df.memory_usage(deep=True).sum()
                                                        + dataset.raw_df.memory_usage(deep=True).sum())
    disk_bytes = sum(path.stat().st_size for path in dataset.disk.path.rglob("*") if path.is_file()) if hasattr(dataset, "disk") else 0
    selected_engine = "disk" if hasattr(dataset, "disk") else "pandas"
    parse_seconds = (parsed - start) if engine == "dataframe" else measurement.get("phase_seconds", {}).get("csv_parse")
    dataset.close()
    return {"workload": name, "engine": engine, "rows": uploaded.row_count, "columns": len(uploaded.columns),
            "selected_engine": selected_engine,
            "csv_parse_seconds": parse_seconds,
            "ingestion_seconds": ingested - start - parse_seconds if parse_seconds is not None else None,
            "parse_and_ingestion_seconds": ingested - start,
            "ingestion_phase_seconds": measurement.get("phase_seconds"),
            "chart_cold_seconds": cold_seconds, "chart_warm_median_seconds": statistics.median(warm_times),
            "process_peak_rss_bytes": peak_bytes,
            "retained_frame_bytes": retained, "retained_disk_bytes": disk_bytes,
            "result": result.model_dump()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=["small", "wide", "million"])
    parser.add_argument("--engine", choices=["dataframe", "pandas", "disk", "auto"], default="dataframe")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--name", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 10:
        parser.error("--repeats must be between 1 and 10")
    if args.worker:
        print(json.dumps(worker(args.worker, args.name, args.engine), allow_nan=False))
        return
    results = []
    with tempfile.TemporaryDirectory(prefix="analytico-benchmark-") as directory:
        for name in args.workloads:
            rows, columns = WORKLOADS[name]
            source = Path(directory) / f"{name}.csv"
            headers = ["category", "amount_usd", "units", "score", "label"]
            headers += [f"measure_{i}" for i in range(columns - 5)]
            with source.open("w", encoding="utf-8") as stream:
                stream.write(",".join(headers) + "\n")
                for index in range(rows):
                    values = [f"group_{index % 10}", str(index if name == "unique" else index % 1001), str(index if name == "unique" else index % 71),
                              str(index % 97), f"label_{index % 23}"]
                    values += [str((index + i) % 1001) for i in range(columns - 5)]
                    stream.write(",".join(values) + "\n")
            for trial in range(args.repeats):
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        "--worker", str(source), "--name", name, "--engine", args.engine],
                                       capture_output=True, text=True)
                if child.returncode:
                    raise RuntimeError(f"Offline benchmark {name} failed:\n{child.stderr}")
                result = json.loads(child.stdout)
                result["trial"] = trial + 1
                results.append(result)
    payload = {"ai_enabled": False, "interpretation_enabled": False,
               "memory_scope": "isolated worker lifetime peak, including imports, parse, ingestion and charts",
               "results": results}
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
