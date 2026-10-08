"""Compare isolated DiskDataset statistics batch sizes on frozen real inputs.

The worker executes a source copy of _analyze_and_build with only the local
stats_batch_size literal changed. Production files and defaults are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import resource
import statistics
import subprocess
import sys
import textwrap
import time

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND / "benchmarks"))
sys.path.insert(0, str(BACKEND))

from ingestion_parallelism import GOLD, fingerprint as base_fingerprint, spec_path

OUTPUT = Path("/private/tmp/analytico-ingestion-batching-2026-10-06")
MODES = ("baseline", "batch8", "batch16", "adaptive", "native", "combined", "production_final")
DEFAULT_MODES = ("baseline", "batch8", "batch16", "adaptive")
THREADS = 4
MEMORY_LIMIT = "64MB"
SEED = 20261006
REAL_THREAD_POOL_EXECUTOR = ThreadPoolExecutor


class _ImmediateFutureExecutor:
    """Run submitted work inline while preserving the Future interface."""
    validation_seconds = 0.0
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, function, /, *args, **kwargs):
        future = Future()
        try:
            result = function(*args, **kwargs)
            if function.__name__ == 'validate_source':
                # Runtime adds this duration after loading. Inline submission
                # already included it in the preflight wall-clock measurement.
                type(self).validation_seconds += result
            future.set_result(result)
        except BaseException as error:
            future.set_exception(error)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def batch_expression(mode: str, combined_size: int) -> str | None:
    return {
        "baseline": None,
        "native": None,
        "production_final": None,
        "batch8": "8",
        "batch16": "16",
        # Keep at most roughly 160 raw columns in one statistics query.
        "adaptive": "max(4, min(16, 160 // max(1, len(self._raw_columns))))",
        "combined": str(combined_size),
    }[mode]


def install_experiment(mode: str, combined_size: int) -> None:
    _ImmediateFutureExecutor.validation_seconds = 0.0
    # Set both knobs inside each fresh child, overriding inherited shell state.
    os.environ["ANALYTICO_DUCKDB_THREADS"] = str(THREADS)
    os.environ["ANALYTICO_NATIVE_VALIDATION"] = "0"
    if mode in {"native", "combined"}:
        os.environ["ANALYTICO_NATIVE_VALIDATION"] = "1"

    import modules.disk_dataset as disk

    disk.DUCKDB_THREADS = THREADS
    if hasattr(disk.DiskDataset, "_collect_column_statistics"):
        # Keep benchmark baselines serial despite the promoted runtime overlap.
        # Production-final explicitly restores the real executor for both
        # validation and statistics collection.
        disk.ThreadPoolExecutor = (
            REAL_THREAD_POOL_EXECUTOR if mode == "production_final"
            else _ImmediateFutureExecutor
        )
    expression = batch_expression(mode, combined_size)
    if expression is None:
        return

    if hasattr(disk.DiskDataset, "_collect_column_statistics"):
        original = disk.DiskDataset._collect_column_statistics
        source = textwrap.dedent(inspect.getsource(original))
        source, range_count = re.subn(
            r"(?m)^(\s*)for start in range\(0, len\(self\._raw_columns\), 4\):$",
            lambda match: f"{match.group(1)}for start in range(0, len(self._raw_columns), {expression}):",
            source,
        )
        source, slice_count = re.subn(
            r"(?m)^(\s*)projections, slots = self\._stats_projection\(self\._raw_columns\[start:start \+ 4\]\)$",
            lambda match: f"{match.group(1)}projections, slots = self._stats_projection(self._raw_columns[start:start + {expression}])",
            source,
        )
        if range_count != 1 or slice_count != 1:
            raise RuntimeError("Could not locate the statistics batch boundaries in collector method.")
        namespace = original.__globals__
        exec(compile(source, disk.__file__, "exec"), namespace)
        disk.DiskDataset._collect_column_statistics = namespace["_collect_column_statistics"]
        return

    source = inspect.getsource(disk.DiskDataset._analyze_and_build)
    patched, count = re.subn(
        r"(?m)^(\s*)stats_batch_size = 4\s*$",
        lambda match: f"{match.group(1)}stats_batch_size = {expression}",
        textwrap.dedent(source),
    )
    if count != 1:
        raise RuntimeError("Expected one stats_batch_size = 4 assignment in source method.")
    namespace = disk.__dict__
    exec(compile(patched, disk.__file__, "exec"), namespace)
    disk.DiskDataset._analyze_and_build = namespace["_analyze_and_build"]


def exact_typed_fingerprint(dataset) -> dict:
    """Retain the shared harness metadata and hash every ordered typed row."""
    from modules.disk_dataset import _q

    result = base_fingerprint(dataset)
    result["roles"] = dict(dataset.column_types)
    columns = ", ".join(_q(column) for column in dataset.columns)
    cursor = dataset._connection.execute(
        f"SELECT {_q('_row_ordinal')}, {columns} FROM typed_data ORDER BY {_q('_row_ordinal')}"
    )
    digest = hashlib.sha256()
    while rows := cursor.fetchmany(4096):
        for row in rows:
            # repr preserves Python scalar types, nulls, and source row position.
            digest.update(repr(tuple(row)).encode("utf-8"))
            digest.update(b"\n")
    result["ordered_typed_rows_sha256"] = digest.hexdigest()
    return result


def is_oom(error: BaseException | str) -> bool:
    message = str(error).casefold()
    return any(token in message for token in (
        "out of memory", "memory limit", "failed to allocate", "cannot allocate memory",
    ))


def worker(spec: dict, mode: str, combined_size: int, result_path: Path) -> None:
    import modules.disk_dataset as disk
    from modules.import_policy import ImportSettings

    install_experiment(mode, combined_size)
    started = time.perf_counter()
    record = {
        "dataset": spec["id"], "mode": mode, "status": "error",
        "disk_dataset_sha256": hashlib.sha256(Path(disk.__file__).read_bytes()).hexdigest(),
    }
    dataset = None
    try:
        dataset = disk.DiskDataset.from_csv(
            spec_path(spec), import_settings=ImportSettings(delimiter=spec["delimiter"])
        )
        # The ingestion result and its peak are fixed before any validation scans.
        record.update(status="ok", ingestion_status="ok",
                      preparation_seconds=time.perf_counter() - started,
                      phases=dict(dataset.ingestion_timings), rows=dataset.row_count)
        if getattr(disk, 'ThreadPoolExecutor', None) is _ImmediateFutureExecutor:
            record['phases']['structure_validation'] -= _ImmediateFutureExecutor.validation_seconds
        Path(str(result_path) + ".ready").write_text("ready")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["ingestion_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        try:
            # Verification is excluded from ingestion timing and uses one
            # thread plus a larger budget so the oracle itself does not fail at
            # the 64 MB ingestion cap. This does not alter ingestion settings.
            dataset._connection.execute("SET threads = 1")
            dataset._connection.execute("SET memory_limit = '1GB'")
            record["fingerprint"] = exact_typed_fingerprint(dataset)
            with dataset.source_path.open("rb") as stream:
                record["source_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            record["source_matches_gold"] = record["source_sha256"] == spec["sha256"]
            record["rows_match_gold"] = dataset.row_count == spec["rows"]
            record["verification_status"] = "ok"
        except Exception as error:
            record.update(
                verification_status="oom" if is_oom(error) or isinstance(error, MemoryError) else "error",
                verification_error_type=type(error).__name__, verification_error=str(error)[:500],
            )
    except Exception as error:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record.update(
            status="oom" if is_oom(error) or isinstance(error, MemoryError) else "error",
            ingestion_status="oom" if is_oom(error) or isinstance(error, MemoryError) else "error",
            ingestion_peak_rss_bytes=int(peak if sys.platform == "darwin" else peak * 1024),
            error_type=type(error).__name__, error=str(error)[:500],
            preparation_seconds=time.perf_counter() - started,
        )
    finally:
        if dataset is not None:
            dataset.close()
    result_path.write_text(json.dumps(record, default=str))


def gold_specs() -> list[dict]:
    gold_bytes = GOLD.read_bytes()
    gold = json.loads(gold_bytes)
    if hashlib.sha256(gold_bytes).hexdigest() != "7d52429b8b5c3e1d9517ed03ce292ec34600dd1ca26a8be659064d51420f2b60":
        raise RuntimeError("Frozen eight-dataset gold manifest hash changed.")
    specs = gold["datasets"]
    if len(specs) != 8:
        raise RuntimeError(f"Expected eight frozen datasets, found {len(specs)}.")
    for spec in specs:
        path = spec_path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen fixture for {spec['id']}: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != spec["sha256"]:
            raise RuntimeError(f"Frozen source hash mismatch for {spec['id']}.")
    return specs


def stress_spec(root: Path) -> dict:
    """Recreate the existing test_disk_dataset 10k x 50 stress fixture."""
    source = root / "wide-schema-stress.csv"
    headers = [f"value_{index}" for index in range(50)]
    with source.open("w", encoding="utf-8", newline="") as stream:
        stream.write(",".join(headers) + "\n")
        for row in range(10_000):
            stream.write(",".join([str(row % 20)] * len(headers)) + "\n")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "id": "wide_schema_stress", "local_path": str(source), "delimiter": ",",
        "rows": 10_000, "sha256": digest, "stress_only": True,
    }


def make_record(spec: dict, mode: str, repeat: int, result_path: Path,
                log_path: Path, started: float,
                exit_code: int | None, combined_size: int) -> dict:
    record = json.loads(result_path.read_text()) if result_path.exists() else None
    if record is None:
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        suspected_oom = is_oom(log) or exit_code == -9
        record = {
            "dataset": spec["id"], "mode": mode,
            "status": "oom_suspected" if suspected_oom else "crash",
            "error": (log.strip()[-500:] or f"worker exited {exit_code}"),
        }
    record.update(
        repeat=repeat, combined_batch_size=combined_size,
        process_wall_seconds=time.perf_counter() - started, exit_code=exit_code,
    )
    if record.get("status") == "ok":
        record["source_matches_gold"] = record.get("source_matches_gold") is True
        record["rows_match_gold"] = record.get("rows_match_gold") is True
    return record


def run_case(spec: dict, mode: str, repeat: int, output: Path,
             combined_size: int) -> dict:
    stem = f"{spec['id']}-{mode}-{repeat}"
    result_path = output / f"{stem}.json"
    ready = Path(str(result_path) + ".ready")
    log_path = output / f"{stem}.log"
    result_path.unlink(missing_ok=True)
    ready.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["ANALYTICO_DUCKDB_THREADS"] = str(THREADS)
    environment["ANALYTICO_NATIVE_VALIDATION"] = "0"
    if mode in {"native", "combined"}:
        environment["ANALYTICO_NATIVE_VALIDATION"] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               spec["id"], "--mode", mode, "--combined-size", str(combined_size),
               "--output", str(result_path), "--spec-json", json.dumps(spec)]
    started = time.perf_counter()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=environment)
        while process.poll() is None:
            time.sleep(0.1)
    return make_record(spec, mode, repeat, result_path, log_path, started,
                       process.returncode, combined_size)


def summarize(records: list[dict], specs: list[dict], modes: list[str], stress_id: str) -> dict:
    baseline: dict[str, dict] = {}
    for spec in specs:
        candidates = [r for r in records if r["dataset"] == spec["id"]
                      and r["mode"] == "baseline" and r["status"] == "ok"
                      and r.get("verification_status") == "ok"]
        if candidates:
            baseline[spec["id"]] = candidates[0]["fingerprint"]
    for record in records:
        reference = baseline.get(record["dataset"])
        record["identical_to_baseline"] = (
            record.get("fingerprint") == reference
            if reference and record.get("status") == "ok" and record.get("verification_status") == "ok"
            else None
        )

    results = []
    for spec in specs:
        for mode in modes:
            selected = [r for r in records if r["dataset"] == spec["id"] and r["mode"] == mode]
            ingested = [r for r in selected if r.get("ingestion_status") == "ok"]
            successful = [r for r in ingested if r.get("verification_status") == "ok"]
            results.append({
                "dataset": spec["id"], "mode": mode,
                "runs": len(selected), "ingestion_successes": len(ingested),
                "fully_checked_successes": len(successful),
                "ingestion_oom_or_error": sum(r.get("ingestion_status") != "ok" for r in selected),
                "verification_oom_or_error": sum(r.get("verification_status") != "ok" for r in ingested),
                "identical_to_baseline": sum(r["identical_to_baseline"] is True for r in selected),
                "source_hash_matches": sum(r.get("source_matches_gold") is True for r in selected),
                "rows_match_gold": sum(r.get("rows_match_gold") is True for r in selected),
                "median_ingestion_seconds": statistics.median(r["preparation_seconds"] for r in ingested) if ingested else None,
                "median_ingestion_peak_mib": statistics.median(r["ingestion_peak_rss_bytes"] / 2**20 for r in ingested) if ingested else None,
                "median_phases": {
                    key: statistics.median(r["phases"].get(key, 0) for r in ingested)
                    for key in ingested[0]["phases"]
                } if ingested else {},
                "errors": sorted({
                    r.get("error", r["status"]) if r.get("ingestion_status") != "ok"
                    else r.get("verification_error", r.get("verification_status", "verification failed"))
                    for r in selected
                    if r.get("ingestion_status") != "ok" or r.get("verification_status") != "ok"
                }),
            })
    return {
        "methodology": {
            "gold_manifest_sha256": hashlib.sha256(GOLD.read_bytes()).hexdigest(),
            "dataset_count": len(specs) - 1, "real_datasets": [s["id"] for s in specs if not s.get("stress_only")],
            "modes": modes, "repeats": 3, "random_seed": SEED,
            "duckdb_threads": THREADS, "duckdb_memory_limit": MEMORY_LIMIT,
            "adaptive_expression": batch_expression("adaptive", 8),
            "combined_batch_size": 8,
            "fingerprint": "shared typed/metadata fingerprint plus SHA256 over every typed row ordered by source row ordinal",
            "stress_fixture": "existing test_disk_dataset 10k-row, 50-column bounded-batch fixture; safety gate only, excluded from performance claims",
            "measurement": "fresh child process per case; child records resource high-water RSS immediately after ingestion, before verification",
            "verification": "full typed/metadata fingerprint and source hash run after timing with one thread and a separate 1 GB query budget; ingestion remains at 64 MB and four threads",
            "native_validation": "ANALYTICO_NATIVE_VALIDATION=1 only for native/combined; reset to unset for every other arm",
        },
        "results": results,
        "stress_gate": [r for r in results if r["dataset"] == stress_id],
    }


def run_matrix(repeats: int, modes: list[str], selected_ids: list[str] | None,
               output: Path, combined_size: int) -> dict:
    if repeats != 3:
        raise ValueError("Use exactly three randomized repetitions for this comparison.")
    specs = gold_specs()
    if selected_ids:
        unknown = set(selected_ids) - {s["id"] for s in specs}
        if unknown:
            raise ValueError(f"Unknown frozen dataset ids: {sorted(unknown)}")
        specs = [s for s in specs if s["id"] in selected_ids]
    output.mkdir(parents=True, exist_ok=True)
    stress = stress_spec(output)
    all_specs = [*specs, stress]
    jobs = [(spec, mode, repeat) for spec in specs for mode in modes for repeat in range(repeats)]
    random.Random(SEED).shuffle(jobs)
    records = []
    raw_path = output / "records.jsonl"
    raw_path.write_text("")
    for index, (spec, mode, repeat) in enumerate(jobs, start=1):
        record = run_case(spec, mode, repeat, output, combined_size)
        records.append(record)
        with raw_path.open("a") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
        print(json.dumps({
            "progress": f"{index}/{len(jobs)}", "dataset": spec["id"], "mode": mode,
            "ingestion": record.get("ingestion_status", record["status"]),
            "verification": record.get("verification_status"),
            "seconds": round(record.get("preparation_seconds", 0), 3),
            "memory_mib": round(record.get("ingestion_peak_rss_bytes", 0) / 2**20, 1),
        }), flush=True)

    # The previously existing wide-schema test fixture is a stress gate only.
    for mode in modes:
        record = run_case(stress, mode, 0, output, combined_size)
        records.append(record)
        with raw_path.open("a") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
        print(json.dumps({"stress_gate": stress["id"], "mode": mode,
                          "status": record["status"]}), flush=True)

    payload = summarize(records, all_specs, modes, stress["id"])
    # Exclude stress-only runs from the real dataset performance table.
    payload["results"] = [r for r in payload["results"] if r["dataset"] != stress["id"]]
    payload["stress_gate"] = [r for r in payload["results"] if r["dataset"] == stress["id"]]
    # Keep the full stress outcomes, including fingerprints, in raw JSONL only.
    payload["stress_gate"] = [
        {"mode": r["mode"], "status": r["status"],
         "identical_to_baseline": r["identical_to_baseline"],
         "source_hash_matches": r.get("source_matches_gold"),
         "rows_match": r.get("rows_match_gold"),
         "ingestion_peak_rss_mib": round(r.get("ingestion_peak_rss_bytes", 0) / 2**20, 2),
         "verification_status": r.get("verification_status")}
        for r in records if r["dataset"] == stress["id"]
    ]
    payload["methodology"]["repeats"] = repeats
    payload["methodology"]["dataset_count"] = len(specs)
    payload["methodology"]["combined_batch_size"] = combined_size
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"summary": str(summary_path), "raw_results": str(raw_path)}), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--spec-json", help=argparse.SUPPRESS)
    parser.add_argument("--combined-size", type=int, choices=(8, 16), default=8, help=argparse.SUPPRESS)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(DEFAULT_MODES))
    parser.add_argument("--datasets", nargs="+", help="Optional frozen dataset ids for a focused run.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.worker:
        if not args.mode or not args.output:
            parser.error("worker requires --mode and --output")
        spec = json.loads(args.spec_json) if args.spec_json else next(
            s for s in json.loads(GOLD.read_text())["datasets"] if s["id"] == args.worker
        )
        worker(spec, args.mode, args.combined_size, args.output)
        return
    run_matrix(args.repeats, args.modes, args.datasets, args.output_root, args.combined_size)


if __name__ == "__main__":
    main()
