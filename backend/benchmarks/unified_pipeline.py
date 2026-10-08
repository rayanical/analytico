"""Measure the warmed, staged user path through DuckDB and default charting.

Offline mode performs no AI calls. ``--live-ai`` is a separate, explicit run for
the existing configured Luna account; it records durations and status only, never
credentials, prompts, source cells, provider content, or chart rows.

The readiness clock covers stage preview, confirm, and the first default-chart
aggregate. It excludes browser upload transfer, HTTP serialization, rendering,
verification, and asynchronous enrichment. Instrumented phase durations overlap
and are not additive; executor events retain start/end offsets for overlap math.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "benchmarks"))
sys.path.insert(0, str(BACKEND))

from ingestion_batching import exact_typed_fingerprint, gold_specs
from ingestion_parallelism import spec_path

THREADS = 4
OUTPUT = Path("/private/tmp/analytico-unified-pipeline-2026-10-07")
REFERENCE_ROOT = Path("/private/tmp/analytico-execution-final-2026-10-07")


class Trace:
    """Thread-safe interval collector with offsets from the readiness start."""

    def __init__(self) -> None:
        self.origin: float | None = None
        self.enabled = False
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        self.origin = time.perf_counter()
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False

    def begin(self, name: str, **details: Any) -> float | None:
        if not self.enabled or self.origin is None:
            return None
        return time.perf_counter()

    def end(self, name: str, started: float | None, **details: Any) -> None:
        if started is None or self.origin is None:
            return
        finished = time.perf_counter()
        item = {"name": name, "start_seconds": started - self.origin,
                "end_seconds": finished - self.origin,
                "duration_seconds": finished - started, **details}
        with self._lock:
            self.events.append(item)


@contextmanager
def interval(trace: Trace, name: str, **details: Any):
    started = trace.begin(name)
    try:
        yield
    except BaseException:
        trace.end(name, started, failed=True, **details)
        raise
    else:
        trace.end(name, started, failed=False, **details)


def _traced_executor(trace: Trace, executor_type):
    class TracedExecutor(executor_type):
        def submit(self, function, /, *args, **kwargs):
            task_name = getattr(function, "__name__", "worker")
            pool_name = getattr(self, "_thread_name_prefix", "executor")

            @wraps(function)
            def observed(*inner_args, **inner_kwargs):
                started = trace.begin(task_name)
                try:
                    result = function(*inner_args, **inner_kwargs)
                except BaseException:
                    trace.end(task_name, started, pool=pool_name, failed=True)
                    raise
                else:
                    trace.end(task_name, started, pool=pool_name, failed=False)
                    return result

            return super().submit(observed, *args, **kwargs)

    return TracedExecutor


def _wrap_method(cls, method_name: str, trace: Trace, label: str) -> None:
    original = getattr(cls, method_name, None)
    if original is None:
        return

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        with interval(trace, label):
            return original(self, *args, **kwargs)

    setattr(cls, method_name, wrapped)


def _wrap_function(owner, function_name: str, trace: Trace, label: str) -> None:
    original = getattr(owner, function_name)

    @wraps(original)
    def wrapped(*args, **kwargs):
        with interval(trace, label):
            return original(*args, **kwargs)

    setattr(owner, function_name, wrapped)


def _sanitize_fingerprint(result: dict[str, Any]) -> dict[str, Any]:
    """Keep deterministic sample metadata comparable without storing source values."""
    result = dict(result)
    samples = result.pop("samples", {})
    result["samples_sha256"] = hashlib.sha256(
        json.dumps(samples, sort_keys=True, ensure_ascii=False, default=str,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def _metadata_digest(response) -> str:
    body = {
        "columns": [item.model_dump(mode="json") for item in response.columns],
        "column_formats": response.column_formats,
        "data_health": response.data_health.model_dump(mode="json"),
        "profile": response.profile.model_dump(mode="json"),
        "default_chart": response.default_chart.model_dump(mode="json") if response.default_chart else None,
        "suggestions": response.suggestions,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _chart_digest(chart) -> str:
    return hashlib.sha256(json.dumps(chart.model_dump(mode="json"), sort_keys=True,
                                    ensure_ascii=False, separators=(",", ":"),
                                    default=str).encode("utf-8")).hexdigest()


def _peak_rss_bytes() -> int | None:
    try:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(raw if sys.platform == "darwin" else raw * 1024)
    except (ImportError, OSError, AttributeError):
        return None


def _sanitize_reference_fingerprint(value: dict[str, Any]) -> dict[str, Any]:
    return _sanitize_fingerprint(value)


def _install_ai_timers(trace: Trace, calls: list[dict[str, Any]]) -> None:
    """Wrap only known AI call boundaries; record wall duration and outcome."""
    from services import ingestion_service
    from modules import schema_decisions, schema_interpretation

    def wrap_callable(owner, attribute: str, label: str) -> None:
        original = getattr(owner, attribute)

        @wraps(original)
        def observed(*args, **kwargs):
            started = trace.begin(label)
            failed = False
            try:
                return original(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                ended = time.perf_counter()
                if started is not None and trace.origin is not None:
                    item = {"name": label, "start_seconds": started - trace.origin,
                            "end_seconds": ended - trace.origin,
                            "duration_seconds": ended - started, "failed": failed}
                    calls.append(item)

        setattr(owner, attribute, observed)

    wrap_callable(ingestion_service, "_generate_business_summary", "ai_summary")
    wrap_callable(schema_interpretation, "analyze_schema_labels", "ai_labels")
    wrap_callable(schema_decisions, "analyze_schema_decisions", "ai_decisions")


def _wrap_openai_calls(query_service, aggregation_service, trace: Trace, calls: list[dict[str, Any]]) -> None:
    """Record the planner and chart-explanation provider calls without payloads."""
    from types import SimpleNamespace

    def instrument(owner, label: str) -> None:
        original_get_client = owner.get_openai_client

        def get_traced_client():
            client = original_get_client()
            create = client.chat.completions.create

            @wraps(create)
            def observed(*args, **kwargs):
                started = trace.begin(label)
                failed = False
                try:
                    return create(*args, **kwargs)
                except BaseException:
                    failed = True
                    raise
                finally:
                    ended = time.perf_counter()
                    if started is not None and trace.origin is not None:
                        calls.append({"name": label,
                            "start_seconds": started - trace.origin,
                            "end_seconds": ended - trace.origin,
                            "duration_seconds": ended - started, "failed": failed})

            return SimpleNamespace(chat=SimpleNamespace(
                completions=SimpleNamespace(create=observed)))

        owner.get_openai_client = get_traced_client

    instrument(query_service, "ai_query_planner")
    instrument(aggregation_service, "ai_chart_analysis")


def _reference_map(root: Path, datasets: set[str]) -> dict[str, dict[str, Any]]:
    path = root / "records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Expected exact reference records at {path}.")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (record.get("dataset") in datasets and record.get("mode") == "baseline"
                and record.get("verification_status") == "ok"
                and record.get("fingerprint")):
            grouped.setdefault(record["dataset"], []).append(record["fingerprint"])
    result = {}
    for dataset in datasets:
        fingerprints = grouped.get(dataset, [])
        if not fingerprints:
            raise ValueError(f"No verified previous baseline exists for {dataset}.")
        normalized = [_sanitize_reference_fingerprint(item) for item in fingerprints]
        if any(item != normalized[0] for item in normalized[1:]):
            raise ValueError(f"Previous baseline fingerprints disagree for {dataset}.")
        result[dataset] = normalized[0]
    return result


def _gold_query(dataset_id: str) -> dict[str, Any] | None:
    gold_path = BACKEND / "evals/fresh_accuracy_gold_2026-10-03.json"
    gold = json.loads(gold_path.read_text())
    spec = next((item for item in gold["datasets"] if item["id"] == dataset_id), None)
    if not spec:
        return None
    case = next((item for item in spec.get("queries", []) if item.get("kind") == "chart"), None)
    return case


def worker(spec: dict[str, Any], output: Path, *, live_ai: bool) -> None:
    # Set deterministic engine/resource knobs before importing backend modules.
    os.environ["ANALYTICO_INGESTION_ENGINE"] = "auto"
    os.environ["ANALYTICO_DUCKDB_THREADS"] = str(THREADS)
    os.environ["ANALYTICO_NATIVE_VALIDATION"] = "0"
    if live_ai:
        os.environ["ANALYTICO_EARLY_AI"] = "1"
        os.environ["COLUMN_INTERPRETER"] = "luna"
        os.environ["COLUMN_INTERPRETER_MODEL"] = "gpt-6-luna"
        os.environ["COLUMN_ROLE_BACKEND"] = "decisions"
    else:
        # Empty child-only overrides disable accidental API use without exposing
        # or changing credentials in the parent process.
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["AI_GATEWAY_API_KEY"] = ""
        os.environ["ANALYTICO_EARLY_AI"] = "0"
        os.environ["COLUMN_INTERPRETER"] = "off"

    trace = Trace()
    ai_calls: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "dataset": spec["id"], "live_ai": live_ai, "status": "error",
        "thread_budget": THREADS, "duckdb_memory_limit": "64MB",
    }
    dataset = import_item = None
    manager = None
    DATASETS = None
    import_preview = None
    confirm_started = None
    enrichment_terminal_times: dict[tuple[str, str], float] = {}
    try:
        startup_started = time.perf_counter()
        from core import config
        from models import AggregateRequest, QueryRequest
        from services import csv_ingestion, import_preview, query_service, aggregation_service
        from services.enrichment_service import manager as enrichment_manager
        from storage import DATASETS

        manager = enrichment_manager
        if live_ai:
            if not os.getenv("OPENAI_API_KEY", "").strip():
                record.update(status="skipped", skip_reason="no_configured_openai_key")
                output.write_text(json.dumps(record))
                return
            if getattr(config, "OPENAI_MODEL", None) != "gpt-6-luna":
                record.update(status="skipped", skip_reason="configured_model_not_gpt_6_luna")
                output.write_text(json.dumps(record))
                return

            original_manager_run = manager._run

            @wraps(original_manager_run)
            def tracked_manager_run(dataset_id, version, work):
                try:
                    return original_manager_run(dataset_id, version, work)
                finally:
                    status = manager.get_status(dataset_id, version)
                    if status.get("status") in {"done", "error", "disabled"}:
                        enrichment_terminal_times[(dataset_id, version)] = time.perf_counter()

            manager._run = tracked_manager_run

        disk_started = time.perf_counter()
        # Root owns this production preload seam. It warms the backend engine
        # once before each worker's request clock starts.
        csv_ingestion.preload_ingestion_engine()
        preload_seconds = time.perf_counter() - disk_started
        from modules import disk_dataset as disk

        real_executor = disk.ThreadPoolExecutor
        disk.ThreadPoolExecutor = _traced_executor(trace, real_executor)
        for method_name, label in (
            ("_ingest_csv", "csv_load"),
            ("_collect_column_statistics", "column_statistics"),
            ("_analyze_and_build", "type_plan_and_materialize"),
            ("_build_profile_and_suggestions", "profile_and_suggestions"),
            ("to_upload_response", "upload_response_build"),
            ("aggregate", "default_chart_sql_and_postprocess"),
        ):
            _wrap_method(disk.DiskDataset, method_name, trace, label)

        if live_ai:
            from fresh_real_accuracy import compare_chart, oracle
            from modules import schema_decisions, schema_interpretation
            _install_ai_timers(trace, ai_calls)
            _wrap_openai_calls(query_service, aggregation_service, trace, ai_calls)
            from services import early_enrichment
            _wrap_function(early_enrichment, "prepare_early_enrichment", trace,
                           "early_enrichment_prepare")

            original_query_aggregate = query_service.run_aggregate

            @wraps(original_query_aggregate)
            def traced_query_aggregate(request):
                with interval(trace, "nl_query_sql_aggregate"):
                    return original_query_aggregate(request)

            query_service.run_aggregate = traced_query_aggregate

        original_measurement = csv_ingestion.IngestionMeasurement
        measurements: list[dict[str, Any]] = []

        class RecordingMeasurement(original_measurement):
            def __exit__(self, error_type, error, traceback):
                measurements.append({
                    "total_seconds": time.perf_counter() - self.started,
                    "phase_seconds": dict(self.phases),
                    "status": "error" if error_type else "ok",
                })
                return super().__exit__(error_type, error, traceback)

        csv_ingestion.IngestionMeasurement = RecordingMeasurement
        startup_seconds = time.perf_counter() - startup_started
        record.update(startup_seconds=startup_seconds, engine_preload_seconds=preload_seconds,
                      disk_dataset_source_sha256=hashlib.sha256(Path(disk.__file__).read_bytes()).hexdigest())

        trace.start()
        started = time.perf_counter()
        with open(os.devnull, "w") as null_output, redirect_stdout(null_output):
            preview_started = time.perf_counter()
            preview = import_preview.stage_import(spec_path(spec), spec["filename"])
            preview_finished = time.perf_counter()
            import_item = import_preview._imports[preview["import_id"]]
            confirm_started = time.perf_counter()
            response = import_preview.confirm_import(
                import_item.id, import_item.settings, ai_column_analysis=live_ai,
            )
            confirm_finished = time.perf_counter()
            dataset = DATASETS[response.dataset_id]
            actual_engine = "disk" if hasattr(dataset, "disk") else "pandas"
            if actual_engine != "disk":
                raise AssertionError("Unified pipeline run resolved to a non-DuckDB engine.")

            chart_started = time.perf_counter()
            if response.default_chart is not None:
                plan = response.default_chart
                chart = aggregation_service.run_aggregate(AggregateRequest(
                    dataset_id=response.dataset_id,
                    x_axis_key=plan.x_axis_key,
                    y_axis_keys=plan.y_axis_keys,
                    aggregation=plan.aggregation,
                    chart_type=plan.chart_type,
                ))
                chart_hash = _chart_digest(chart)
                chart_row_count = len(chart.data)
            else:
                chart_hash = None
                chart_row_count = 0
            chart_finished = time.perf_counter()
        if not live_ai:
            trace.stop()

        readiness_seconds = chart_finished - started
        record.update(status="ok", actual_engine=actual_engine,
            stage_preview_seconds=preview_finished - preview_started,
            confirm_seconds=confirm_finished - confirm_started,
            default_chart_seconds=chart_finished - chart_started,
            stage_to_chart_seconds=readiness_seconds,
            stage_to_confirm_seconds=confirm_finished - started,
            row_count=response.row_count, column_count=len(response.columns),
            default_chart_present=response.default_chart is not None,
            default_chart_rows=chart_row_count, default_chart_sha256=chart_hash,
            upload_metadata_sha256=_metadata_digest(response),
            source_bytes=spec_path(spec).stat().st_size,
            ingestion_measurements=measurements,
            disk_phase_seconds=dict(dataset.disk.ingestion_timings),
            ingestion_peak_rss_bytes=_peak_rss_bytes())

        if live_ai:
            # Capture the manager's terminal time in its worker thread. Waiting
            # before verification keeps the independent full-file scan from
            # delaying background work or inflating its reported completion.
            enrich_start = time.perf_counter()
            while time.perf_counter() - enrich_start < 180:
                status = manager.get_status(dataset.id)
                if status.get("status") in {"done", "error", "disabled"}:
                    break
                time.sleep(0.05)
            else:
                status = {"status": "timeout"}
            record["enrichment_status"] = status.get("status")
            terminal_at = enrichment_terminal_times.get((dataset.id, dataset.cache_version))
            record["enrichment_done_seconds_from_confirm"] = (
                terminal_at - confirm_started if terminal_at is not None else None)
            record["summary_ready"] = bool(status.get("summary"))
            record["label_count"] = len(status.get("column_labels") or {})
            record["decision_role_count"] = len(status.get("column_roles") or {})
            coverage = status.get("coverage")
            if isinstance(coverage, dict):
                record["enrichment_coverage"] = {
                    key: coverage.get(key) for key in (
                        "total_columns", "selected_columns", "completed_columns",
                        "failed_columns", "skipped_columns", "complete", "stop_reason")
                }
            proposals = status.get("interpretation_proposals") or {}
            proposal_statuses: dict[str, int] = {}
            label_statuses: dict[str, int] = {}
            proposal_errors: dict[str, int] = {}
            label_errors: dict[str, int] = {}
            if isinstance(proposals, dict):
                for proposal in proposals.values():
                    if not isinstance(proposal, dict):
                        proposal_statuses["invalid"] = proposal_statuses.get("invalid", 0) + 1
                        continue
                    label = str(proposal.get("status", "missing"))[:40]
                    proposal_statuses[label] = proposal_statuses.get(label, 0) + 1
                    label_status = proposal.get("label_status")
                    if label_status is not None:
                        label_status = str(label_status)[:40]
                        label_statuses[label_status] = label_statuses.get(label_status, 0) + 1
                    code = proposal.get("error_code")
                    if code is not None:
                        code = str(code)[:80]
                        proposal_errors[code] = proposal_errors.get(code, 0) + 1
                    label_code = proposal.get("label_error_code")
                    if label_code is not None:
                        label_code = str(label_code)[:80]
                        label_errors[label_code] = label_errors.get(label_code, 0) + 1
            record["decision_proposal_status_counts"] = proposal_statuses
            record["label_status_counts"] = label_statuses
            record["decision_error_code_counts"] = proposal_errors
            record["label_error_code_counts"] = label_errors
            record["ai_call_intervals"] = ai_calls

        if live_ai:
            case = _gold_query(spec["id"])
            query = None
            if case is not None:
                query_started = time.perf_counter()
                try:
                    # Keep every gold group in the result so the independent
                    # streaming oracle can verify category keys and values.
                    query = query_service.run_query(QueryRequest(
                        dataset_id=dataset.id, user_prompt=case["prompt"],
                        limit=1000, group_others=False,
                    ))
                except Exception as error:
                    record["query_error_type"] = type(error).__name__
                query_finished = time.perf_counter()
                record["query_seconds"] = query_finished - query_started
                if query is not None:
                    ready = query.chart_type != "empty" and bool(query.data)
                    record.update(query_chart_ready=ready,
                        query_axes_match=(query.x_axis_key == case["x"]
                            and query.y_axis_keys == [case["y"]]
                            and query.aggregation == case["aggregation"]),
                        query_chart_sha256=_chart_digest(query),
                        query_row_count=len(query.data))
                    provider_seconds = sum(item["duration_seconds"] for item in ai_calls
                                           if item["name"] == "ai_query_planner")
                    sql_seconds = sum(item["duration_seconds"] for item in trace.events
                                      if item["name"] == "nl_query_sql_aggregate")
                    record.update(query_planner_provider_seconds=provider_seconds,
                        query_sql_seconds=sql_seconds,
                        query_local_overhead_seconds=max(0.0, query_finished-query_started
                                                         - provider_seconds - sql_seconds))

                    if ready:
                        # The NL endpoint returns a computed chart without AI
                        # analysis. Measure explanation separately on this chart.
                        analysis_request = AggregateRequest(
                            dataset_id=dataset.id, x_axis_key=query.x_axis_key,
                            y_axis_keys=query.y_axis_keys, aggregation=query.aggregation or "sum",
                            chart_type=query.chart_type, limit=1000, sort_by=None,
                            group_others=False,
                            include_analysis=True,
                        )
                        analysis_started = time.perf_counter()
                        explained = aggregation_service.run_aggregate(analysis_request)
                        analysis_finished = time.perf_counter()
                        record.update(chart_analysis_seconds=analysis_finished-analysis_started,
                            chart_analysis_present=bool(explained.analysis),
                            chart_analysis_sha256=hashlib.sha256(
                                (explained.analysis or "").encode("utf-8")).hexdigest())
                        record["chart_analysis_provider_seconds"] = sum(
                            item["duration_seconds"] for item in ai_calls
                            if item["name"] == "ai_chart_analysis")
                    else:
                        record["chart_analysis_skipped_reason"] = "query_did_not_return_chart"
            trace.stop()
            record["ai_call_intervals"] = ai_calls

        # Exact validation uses a separate budget and starts only after all
        # readiness, enrichment, NL-query and chart-analysis timings, and RSS
        # capture are complete. This scan never competes with measured work.
        try:
            dataset.disk._connection.execute("SET threads = 1")
            dataset.disk._connection.execute("SET memory_limit = '1GB'")
            exact = exact_typed_fingerprint(dataset.disk)
            if live_ai:
                # AI may revise semantic roles/labels after the local response.
                record["physical_values_sha256"] = exact["ordered_typed_rows_sha256"]
            else:
                record["fingerprint"] = _sanitize_fingerprint(exact)
            with dataset.disk.source_path.open("rb") as stream:
                source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            record["source_sha256"] = source_hash
            record["source_matches_gold"] = source_hash == spec["sha256"]
            record["rows_match_gold"] = dataset.row_count == spec["rows"]
            record["verification_status"] = (
                "ok" if record["source_matches_gold"] and record["rows_match_gold"] else "mismatch")
        except Exception as error:
            record["verification_status"] = "error"
            record["verification_error_type"] = type(error).__name__

        if live_ai and case is not None and query is not None:
            # The independent Decimal oracle is excluded from all timings.
            try:
                expected = oracle(spec_path(spec), spec["delimiter"], case["x"],
                                  case["y"], case["aggregation"])
                expected = {"" if key is None else str(key): value
                            for key, value in expected.items()}
                correct, checks = compare_chart(query.model_dump(mode="json"), expected, case)
                record["query_gold_correct"] = correct
                record["query_gold_checks"] = checks
            except Exception as error:
                record["query_oracle_error_type"] = type(error).__name__

        record["executor_intervals"] = sorted(trace.events, key=lambda item: item["start_seconds"])

        output.write_text(json.dumps(record, default=str))
    except Exception as error:
        record.update(error_type=type(error).__name__,
                      error_module=getattr(error, "name", None)
                      if isinstance(error, ModuleNotFoundError) else None)
        output.write_text(json.dumps(record, default=str))
    finally:
        if dataset is not None and DATASETS is not None:
            DATASETS.pop(dataset.id, None)
            dataset.close()
        if import_item is not None and import_preview is not None and import_item.id in import_preview._imports:
            import_preview.cancel_import(import_item.id)
        if manager is not None and live_ai:
            manager.shutdown()


def _overlap_seconds(events: list[dict[str, Any]], first: str, second: str) -> float:
    a = [e for e in events if e.get("name") == first]
    b = [e for e in events if e.get("name") == second]
    total = 0.0
    for left in a:
        for right in b:
            total += max(0.0, min(left["end_seconds"], right["end_seconds"])
                         - max(left["start_seconds"], right["start_seconds"]))
    return total


def _median(records: list[dict[str, Any]], key: str):
    values = [record[key] for record in records if isinstance(record.get(key), (int, float))]
    return statistics.median(values) if values else None


def _summarize(records: list[dict[str, Any]], specs: list[dict[str, Any]],
               references: dict[str, dict[str, Any]], repeats: int,
               live_ai: bool) -> dict[str, Any]:
    summary = []
    for spec in specs:
        cases = [item for item in records if item["dataset"] == spec["id"]]
        valid = [item for item in cases if item.get("status") == "ok"]
        verified = [item for item in valid if item.get("verification_status") == "ok"]
        expected = references.get(spec["id"])
        physical_value_matches = (
            sum(item.get("physical_values_sha256") == expected.get("ordered_typed_rows_sha256")
                for item in verified)
            if expected and live_ai else None)
        chart_hashes = {item.get("default_chart_sha256") for item in valid}
        metadata_hashes = {item.get("upload_metadata_sha256") for item in valid}
        phase_names = sorted({name for item in valid
                              for run in item.get("ingestion_measurements", [])
                              for name in run.get("phase_seconds", {})})
        phase_medians = {}
        for name in phase_names:
            values = [run["phase_seconds"][name] for item in valid
                      for run in item.get("ingestion_measurements", [])
                      if name in run.get("phase_seconds", {})]
            if values:
                phase_medians[name] = statistics.median(values)
        summary.append({
            "dataset": spec["id"], "bytes": spec_path(spec).stat().st_size,
            "runs": len(cases), "successes": len(valid), "verification_successes": len(verified),
            "median_stage_preview_seconds": _median(valid, "stage_preview_seconds"),
            "median_confirm_seconds": _median(valid, "confirm_seconds"),
            "median_default_chart_seconds": _median(valid, "default_chart_seconds"),
            "median_stage_to_chart_seconds": _median(valid, "stage_to_chart_seconds"),
            "median_engine_preload_seconds": _median(valid, "engine_preload_seconds"),
            "median_ingestion_peak_mib": (
                statistics.median(item["ingestion_peak_rss_bytes"] / 2**20 for item in valid
                                  if item.get("ingestion_peak_rss_bytes") is not None)
                if any(item.get("ingestion_peak_rss_bytes") is not None for item in valid) else None),
            "source_matches_gold": sum(item.get("source_matches_gold") is True for item in verified),
            "rows_match_gold": sum(item.get("rows_match_gold") is True for item in verified),
            "exact_previous_fingerprint_matches": (
                sum(item.get("fingerprint") == expected for item in verified)
                if expected and not live_ai else None),
            "physical_values_reference_matches": physical_value_matches,
            "reference_available": expected is not None,
            "default_chart_hashes_identical": len(chart_hashes) <= 1,
            "upload_metadata_hashes_identical": len(metadata_hashes) <= 1,
            "phase_medians_seconds_non_additive": phase_medians,
            "median_validator_load_overlap_seconds": (
                statistics.median(_overlap_seconds(item.get("executor_intervals", []),
                                                   "validate_source", "csv_load") for item in valid)
                if valid else None),
            "median_enrichment_done_seconds": _median(valid, "enrichment_done_seconds_from_confirm") if live_ai else None,
            "median_nl_query_seconds": _median(valid, "query_seconds") if live_ai else None,
            "median_chart_analysis_seconds": _median(valid, "chart_analysis_seconds") if live_ai else None,
            "nl_query_chart_ready_runs": sum(item.get("query_chart_ready") is True for item in valid) if live_ai else None,
            "nl_query_gold_plan_matches": sum(item.get("query_axes_match") is True for item in valid) if live_ai else None,
            "nl_query_gold_value_matches": sum(item.get("query_gold_correct") is True for item in verified) if live_ai else None,
            "enrichment_successes": sum(
                item.get("enrichment_status") == "done" and item.get("summary_ready") is True
                and item.get("label_count", 0) > 0
                and not item.get("label_error_code_counts")
                and item.get("enrichment_coverage", {}).get("complete") is True
                and item.get("enrichment_coverage", {}).get("failed_columns") == 0
                for item in valid) if live_ai else None,
            "chart_analysis_successes": sum(item.get("chart_analysis_present") is True for item in valid) if live_ai else None,
            "ai_calls": [call for item in valid for call in item.get("ai_call_intervals", [])] if live_ai else [],
            "ai_component_medians_seconds": {
                name: statistics.median(call["duration_seconds"] for item in valid
                    for call in item.get("ai_call_intervals", []) if call.get("name") == name)
                for name in sorted({call.get("name") for item in valid
                    for call in item.get("ai_call_intervals", [])})
            } if live_ai else {},
            "median_query_planner_provider_seconds": _median(valid, "query_planner_provider_seconds") if live_ai else None,
            "median_query_sql_seconds": _median(valid, "query_sql_seconds") if live_ai else None,
            "median_query_local_overhead_seconds": _median(valid, "query_local_overhead_seconds") if live_ai else None,
            "median_chart_analysis_provider_seconds": _median(valid, "chart_analysis_provider_seconds") if live_ai else None,
            "errors": [{"status": item.get("status"), "type": item.get("error_type"),
                        "skip_reason": item.get("skip_reason")}
                       for item in cases if item.get("status") != "ok"],
        })
    return {
        "date": "2026-10-07", "live_ai": live_ai, "repeats": repeats,
        "datasets": [spec["id"] for spec in specs],
        "resources": {"duckdb_threads": THREADS, "duckdb_memory_limit": "64MB",
                       "engine_selection": "auto; each worker asserts DuckDB resolved"},
        "scope": "Warm imports/engine; stage preview -> confirm/store -> first default-chart aggregation. Excludes HTTP transfer/serialization, browser rendering and verification.",
        "phase_note": "Phase intervals can nest or overlap. Use executor start/end events for overlap; never sum phase durations as wall time.",
        "references": {"root": str(REFERENCE_ROOT),
                       "description": "Three verified direct-DiskDataset baseline repetitions; exact ordered typed rows and full deterministic schema/statistics metadata. Stage preview/chart were not in reference timing."},
        "summary": summary,
    }


def run(args) -> None:
    all_specs = gold_specs()
    if args.live_ai:
        default_datasets = ["taxi", "retail"]
        if args.live_control:
            default_datasets.append("tips")
    else:
        default_datasets = [spec["id"] for spec in all_specs]
    selected_ids = args.datasets or default_datasets
    unknown = set(selected_ids) - {spec["id"] for spec in all_specs}
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    if args.live_ai and args.repeats != 3:
        raise ValueError("Live AI mode uses exactly three repetitions.")
    specs = [spec for spec in all_specs if spec["id"] in selected_ids]
    reference = _reference_map(args.reference_root, {spec["id"] for spec in specs})
    args.output_root.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_root / ("live-records.jsonl" if args.live_ai else "records.jsonl")
    raw_path.write_text("")
    jobs = [(spec, repeat) for spec in specs for repeat in range(args.repeats)]
    random.Random(20261007 + args.repeats + (1 if args.live_ai else 0)).shuffle(jobs)
    records = []
    for index, (spec, repeat) in enumerate(jobs, start=1):
        output = args.output_root / f"{spec['id']}-{'live' if args.live_ai else 'offline'}-{repeat}.json"
        output.unlink(missing_ok=True)
        environment = os.environ.copy()
        environment["ANALYTICO_INGESTION_ENGINE"] = "auto"
        environment["ANALYTICO_DUCKDB_THREADS"] = str(THREADS)
        environment["ANALYTICO_NATIVE_VALIDATION"] = "0"
        if not args.live_ai:
            environment["OPENAI_API_KEY"] = ""
            environment["AI_GATEWAY_API_KEY"] = ""
            environment["COLUMN_INTERPRETER"] = "off"
            environment["ANALYTICO_EARLY_AI"] = "0"
        else:
            environment["ANALYTICO_EARLY_AI"] = "1"
            environment["COLUMN_INTERPRETER"] = "luna"
            environment["COLUMN_INTERPRETER_MODEL"] = "gpt-6-luna"
        environment["COLUMN_ROLE_BACKEND"] = "decisions"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--spec-json", json.dumps(spec), "--output", str(output)]
        if args.live_ai:
            command.append("--live-ai")
        started = time.perf_counter()
        with (args.output_root / "worker.log").open("a") as log:
            completed = subprocess.run(command, env=environment, stdout=log, stderr=log, check=False)
        record = json.loads(output.read_text()) if output.exists() else {
            "dataset": spec["id"], "status": "crash", "exit_code": completed.returncode}
        record.update(repeat=repeat, process_wall_seconds=time.perf_counter() - started,
                      exit_code=completed.returncode)
        records.append(record)
        with raw_path.open("a") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
        print(json.dumps({"progress": f"{index}/{len(jobs)}", "dataset": spec["id"],
                          "repeat": repeat, "status": record.get("status"),
                          "stage_to_chart_seconds": record.get("stage_to_chart_seconds"),
                          "enrichment_status": record.get("enrichment_status")}), flush=True)
    payload = _summarize(records, specs, reference, args.repeats, args.live_ai)
    (args.output_root / ("live-summary.json" if args.live_ai else "summary.json")).write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"summary": str(args.output_root / ("live-summary.json" if args.live_ai else "summary.json"))}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--spec-json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live-ai", action="store_true",
                        help="Separate explicit run using the existing configured Luna key; no secrets or source rows are emitted.")
    parser.add_argument("--live-control", action="store_true",
                        help="Include one small `tips` control in the live three-repeat run.")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--repeats", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.worker:
        if not args.spec_json or not args.output:
            parser.error("--worker requires --spec-json and --output")
        worker(json.loads(args.spec_json), args.output, live_ai=args.live_ai)
        return
    run(args)


if __name__ == "__main__":
    main()
