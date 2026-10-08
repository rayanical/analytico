"""Isolated ingestion experiments and a frozen pre-promotion comparison.

All timings follow staged preview -> confirm -> first chart, with exact whole-file
verification afterwards. Uses complete existing public fixtures and no AI calls.
"""
from __future__ import annotations

import argparse
from functools import wraps
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Optional

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "benchmarks"))
import unified_pipeline as pipeline

MODES = ("baseline", "memory128", "memory256", "guarded_stats", "uncompressed", "production")


def legacy_stats_projection(
    self, raw_columns: Optional[list[str]] = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    # Frozen pre-promotion oracle for reproducible before/after experiments.
    from modules.disk_dataset import (
        _literal, _q, _CURRENCY_CLASS, _PLAIN_NUMBER, _INTEGER_TEXT,
        _US_GROUPED_NUMBER, _AMBIGUOUS_GROUPED, _NUMERIC_LIKE,
    )
    projections: list[str] = []
    slots: list[tuple[str, str]] = []
    currency = _literal(_CURRENCY_CLASS)
    selected_columns = raw_columns or self._raw_columns
    for raw_column in selected_columns:
        index = self._raw_columns.index(raw_column)
        column = _q(raw_column)
        trimmed = f"TRIM({column})"
        currency_trim = f"TRIM(REGEXP_REPLACE({trimmed}, {currency}, '', 'g'))"
        expressions = {
            "nonnull": f"COUNT({column})",
            "plain": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, {_literal(_PLAIN_NUMBER.pattern)}))",
            "integer": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, {_literal(_INTEGER_TEXT.pattern)}))",
            "bigint": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, {_literal(_INTEGER_TEXT.pattern)}) AND TRY_CAST({trimmed} AS BIGINT) IS NOT NULL)",
            "uint64": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, {_literal(_INTEGER_TEXT.pattern)}) AND TRY_CAST({trimmed} AS UBIGINT) IS NOT NULL)",
            "percent": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, '.*%$'))",
            "percent_inner_plain": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH(TRIM(REGEXP_REPLACE({trimmed}, '%$', '')), {_literal(_PLAIN_NUMBER.pattern)}))",
            "currency_symbol": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_MATCHES({column}, {currency}))",
            "currency_plain": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({currency_trim}, {_literal(_PLAIN_NUMBER.pattern)}))",
            "currency_grouped": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({currency_trim}, {_literal(_US_GROUPED_NUMBER.pattern)}))",
            "comma": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND STRPOS({trimmed}, ',') > 0)",
            "ambiguous_grouped": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, {_literal(_AMBIGUOUS_GROUPED.pattern)}))",
            "grouped_integer": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, '^[+-]?[1-9][0-9]{{0,2}}(,[0-9]{{3}})+$'))",
            "grouped_decimal": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, {_literal(_US_GROUPED_NUMBER.pattern)}))",
            "numeric_like": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, {_literal(_NUMERIC_LIKE.pattern)}))",
            "leading_zero": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, '^[+-]?0[0-9]+$'))",
        }
        for name, expression in expressions.items():
            alias = f"s{index}_{name}"
            projections.append(f"{expression} AS {_q(alias)}")
            slots.append((raw_column, name))
    return projections, slots


def guarded_projection(original):
    """Short-circuit expensive predicates only after an exact per-value proof."""
    @wraps(original)
    def project(self, raw_columns=None):
        import modules.disk_dataset as disk
        projections, slots = original(self, raw_columns)
        result = []
        for expression, (raw, name) in zip(projections, slots):
            column = disk._q(raw)
            trimmed = f"TRIM({column})"
            plain = f"REGEXP_FULL_MATCH({trimmed}, {disk._literal(disk._PLAIN_NUMBER.pattern)})"
            integer = f"REGEXP_FULL_MATCH({trimmed}, {disk._literal(disk._INTEGER_TEXT.pattern)})"
            prefix = f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND "
            aggregate, alias = expression.rsplit(" AS ", 1)
            if not aggregate.startswith(prefix):
                result.append(expression)
                continue
            predicate = aggregate[len(prefix):-1]
            if name in {"currency_plain", "percent_inner_plain", "numeric_like"}:
                predicate = f"CASE WHEN {plain} THEN TRUE ELSE ({predicate}) END"
            elif name in {"currency_symbol", "currency_grouped"}:
                predicate = f"CASE WHEN {plain} THEN FALSE ELSE ({predicate}) END"
            elif name in {"ambiguous_grouped", "grouped_integer", "grouped_decimal"}:
                predicate = f"CASE WHEN STRPOS({trimmed}, ',') > 0 THEN ({predicate}) ELSE FALSE END"
            elif name in {"bigint", "uint64"}:
                cast = "BIGINT" if name == "bigint" else "UBIGINT"
                predicate = f"CASE WHEN {integer} THEN TRY_CAST({trimmed} AS {cast}) IS NOT NULL ELSE FALSE END"
            result.append(f"{prefix}{predicate}) AS {alias}")
        return result, slots
    return project


def install(mode):
    import duckdb
    import modules.disk_dataset as disk
    if mode not in MODES:
        raise ValueError("Unknown experiment")
    disk.DUCKDB_MEMORY_LIMIT = {"memory128": "128MB", "memory256": "256MB"}.get(mode, "64MB")
    if mode != "production":
        disk.DiskDataset._stats_projection = legacy_stats_projection
    if mode == "guarded_stats":
        disk.DiskDataset._stats_projection = guarded_projection(legacy_stats_projection)
    if mode == "uncompressed":
        connect = duckdb.connect

        @wraps(connect)
        def uncompressed(*args, **kwargs):
            connection = connect(*args, **kwargs)
            connection.execute("SET force_compression = 'uncompressed'")
            return connection

        duckdb.connect = uncompressed
    return disk


def worker(spec, output, mode):
    for key, value in {"OPENAI_API_KEY": "", "AI_GATEWAY_API_KEY": "",
                       "COLUMN_INTERPRETER": "off", "ANALYTICO_EARLY_AI": "0",
                       "ANALYTICO_INGESTION_ENGINE": "auto",
                       "ANALYTICO_DUCKDB_THREADS": "4"}.items():
        os.environ[key] = value
    disk = install(mode)
    storage = {}
    fingerprint = pipeline.exact_typed_fingerprint

    def measured_fingerprint(dataset):
        # Called after all readiness clocks and RSS capture, before verification.
        files = [p for p in dataset.path.rglob("*") if p.is_file() and p != dataset.source_path]
        storage.update(database_and_spill_bytes=sum(p.stat().st_size for p in files),
                       database_and_spill_allocated_bytes=sum(p.stat().st_blocks * 512 for p in files),
                       source_bytes=dataset.source_path.stat().st_size)
        return fingerprint(dataset)

    pipeline.exact_typed_fingerprint = measured_fingerprint
    pipeline.worker(spec, output, live_ai=False)
    record = json.loads(output.read_text())
    record.update(mode=mode, configured_memory_limit=disk.DUCKDB_MEMORY_LIMIT,
                  duckdb_memory_limit=disk.DUCKDB_MEMORY_LIMIT,
                  compression="uncompressed" if mode == "uncompressed" else "auto",
                  storage_after_readiness=storage)
    output.write_text(json.dumps(record))


def summarize(records, references):
    rows = []
    for dataset in sorted({r["dataset"] for r in records}):
        for mode in MODES:
            cases = [r for r in records if r["dataset"] == dataset and r["mode"] == mode]
            if not cases:
                continue
            valid = [r for r in cases if r.get("status") == "ok" and r.get("verification_status") == "ok"
                     and r.get("fingerprint") == references[dataset]]
            row = {"dataset": dataset, "mode": mode, "runs": len(cases), "exact_matches": len(valid)}
            for key in ("stage_to_chart_seconds", "stage_preview_seconds", "confirm_seconds", "default_chart_seconds"):
                values = [r[key] for r in valid]
                row["median_" + key] = statistics.median(values) if values else None
            if valid:
                row["median_peak_mib"] = statistics.median(r["ingestion_peak_rss_bytes"] / 2**20 for r in valid)
                row["median_database_and_spill_mib"] = statistics.median(
                    r["storage_after_readiness"]["database_and_spill_bytes"] / 2**20 for r in valid)
                row["median_database_and_spill_allocated_mib"] = statistics.median(
                    r["storage_after_readiness"]["database_and_spill_allocated_bytes"] / 2**20 for r in valid)
                row["phase_medians_non_additive"] = {
                    key: statistics.median(r["disk_phase_seconds"][key] for r in valid)
                    for key in valid[0]["disk_phase_seconds"]}
            row["errors"] = [{"status": r.get("status"), "error_type": r.get("error_type"),
                              "verification": r.get("verification_status"),
                              "verification_error_type": r.get("verification_error_type")}
                             for r in cases if r not in valid]
            rows.append(row)
    for row in rows:
        baseline = next((r for r in rows if r["dataset"] == row["dataset"] and r["mode"] == "baseline"), None)
        if baseline and baseline.get("median_stage_to_chart_seconds") and row.get("median_stage_to_chart_seconds"):
            row["reduction_percent_vs_same_batch_baseline"] = 100 * (
                1 - row["median_stage_to_chart_seconds"] / baseline["median_stage_to_chart_seconds"])
    return {"scope": "Preloaded local stage-preview -> confirm -> default chart; excludes HTTP/browser/AI and verification.",
            "ai_calls": 0, "production_arm": "Current production implementation; baseline freezes pre-guard statistics", "summary": rows,
            "resources": "4 DuckDB threads; memory varied only in named memory arms; stats uses existing 2-worker scheduler",
            "storage_scope": "Database, WAL and spill logical/allocated bytes after readiness, excluding retained source; not peak disk use.",
            "verification": "Exact old full-source typed/schema/statistics fingerprints; verifier runs after clocks/RSS with 1 thread/1GB."}


def run(args):
    specs = [s for s in pipeline.gold_specs() if not args.datasets or s["id"] in args.datasets]
    if args.datasets and set(args.datasets) != {s["id"] for s in specs}:
        raise ValueError("Unknown dataset")
    references = pipeline._reference_map(args.reference_root, {s["id"] for s in specs})
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(s, mode, repeat) for s in specs for mode in args.modes for repeat in range(args.repeats)]
    random.Random(20261007 + args.repeats).shuffle(jobs)
    records = []
    raw = args.output_root / "records.jsonl"
    raw.write_text("")
    for number, (spec, mode, repeat) in enumerate(jobs, 1):
        output = args.output_root / f"{spec['id']}-{mode}-{repeat}.json"
        output.unlink(missing_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--mode", mode,
                   "--spec-json", json.dumps(spec), "--output", str(output)]
        with (args.output_root / "worker.log").open("a") as log:
            result = subprocess.run(command, stdout=log, stderr=log, timeout=240)
        record = json.loads(output.read_text()) if output.exists() else {
            "dataset": spec["id"], "mode": mode, "status": "worker_failed", "exit_code": result.returncode}
        records.append(record)
        with raw.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({"progress": f"{number}/{len(jobs)}", "dataset": spec["id"], "mode": mode,
                          "seconds": record.get("stage_to_chart_seconds"),
                          "exact_match": record.get("fingerprint") == references[spec["id"]]}), flush=True)
    (args.output_root / "summary.json").write_text(json.dumps(summarize(records, references), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--spec-json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--repeats", type=int, choices=(1, 3), default=3)
    parser.add_argument("--reference-root", type=Path, default=pipeline.REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("/private/tmp/analytico-last-three-2026-10-07"))
    options = parser.parse_args()
    if options.worker:
        worker(json.loads(options.spec_json), options.output, options.mode)
    else:
        run(options)
