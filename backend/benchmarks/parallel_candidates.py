"""Isolated query-scheduling candidates for DiskDataset ingestion benchmarks.

Import this module in a fresh worker and call ``install(mode)`` before
``DiskDataset.from_csv``. Production files and defaults are not modified.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import os
import textwrap
from time import perf_counter
from typing import Any


MODES = ("baseline", "parallel_stats", "parallel_profiles", "fused_profiles", "plain_pruning4")
SUPPORTED_MODES = MODES
WORKERS = 2
THREADS = 4
MEMORY_LIMIT = "64MB"
_ORIGINALS: dict[int, tuple[Any, Any]] = {}


def budgets() -> dict[str, str]:
    """Describe candidate resource settings for the parent scorecard."""
    return {
        "baseline": "One DuckDB connection, four threads, unchanged 64 MB limit.",
        "parallel_stats": (
            "Two query cursors share the same DuckDB database, four-thread scheduler, and "
            "64 MB memory limit; worker calls do not change connection settings."
        ),
        "parallel_profiles": (
            "Two query cursors share the same DuckDB database, four-thread scheduler, and "
            "64 MB memory limit; worker calls do not change connection settings."
        ),
        "fused_profiles": (
            "One existing DuckDB connection, four threads and unchanged 64 MB limit; "
            "one grouped profile query per column."
        ),
        "plain_pruning4": (
            "One DuckDB connection, four threads and unchanged 64 MB limit; a full-column "
            "plain-number proof allows only impossible percent/currency/grouping predicates "
            "to be replaced with exact counts."
        ),
    }


def _stat_queries(dataset) -> list[tuple[str, list[tuple[str, str]]]]:
    queries = []
    for start in range(0, len(dataset._raw_columns), 4):
        selected = dataset._raw_columns[start:start + 4]
        projections, slots = dataset._stats_projection(selected)
        queries.append((f"SELECT {', '.join(projections)} FROM source_data", slots))
    return queries


def _execute_groups(dataset, groups: list[list[tuple[str, Any]]]) -> dict[str, Any]:
    """Run bounded query groups through separate DuckDB cursor connections."""
    def run_group(group):
        cursor = dataset._connection.cursor()
        results = {}
        try:
            for sql, value_kind in group:
                cursor.execute(sql)
                if value_kind == "one":
                    results[sql] = ("one", cursor.fetchone())
                else:
                    results[sql] = ("all", cursor.fetchall())
        finally:
            cursor.close()
        return results

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_group, group) for group in groups if group]
        combined = {}
        for future in futures:
            combined.update(future.result())
    return combined


def _parallel_stats(dataset) -> dict[str, dict[str, int]]:
    queries = _stat_queries(dataset)
    groups: list[list[tuple[str, Any]]] = [[] for _ in range(WORKERS)]
    for index, (sql, slots) in enumerate(queries):
        groups[index % WORKERS].append((sql, "one"))
    results = _execute_groups(dataset, groups)
    stats: dict[str, dict[str, int]] = {}
    for sql, slots in queries:
        row = results[sql][1]
        for index, (raw, name) in enumerate(slots):
            stats.setdefault(raw, {})[name] = int(row[index] or 0)
    return stats


class _CachedResult:
    def __init__(self, kind: str, rows):
        self.kind, self.rows = kind, rows

    def fetchone(self):
        return self.rows if self.kind == "one" else (self.rows[0] if self.rows else None)

    def fetchall(self):
        return [self.rows] if self.kind == "one" and self.rows is not None else self.rows


class _CachedConnection:
    """Serve precomputed profile query results while delegating other calls."""
    def __init__(self, connection, cache):
        self._connection = connection
        self._cache = cache

    def execute(self, sql, *args, **kwargs):
        cached = self._cache.get(sql) if not args and not kwargs else None
        if cached is not None:
            return _CachedResult(*cached)
        return self._connection.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._connection, name)


def _profile_stat_sql(dataset, clean: str) -> str:
    import modules.disk_dataset as disk

    column = disk._q(clean)
    numeric = dataset._plans[clean].kind in {
        "integer", "unsigned", "number", "percentage", "currency",
    }
    fractional = (
        f"BOOL_OR({column} != TRUNC({column}))"
        if dataset._plans[clean].kind not in {"integer", "unsigned"} else "FALSE"
    )
    extra = f", MIN({column}), MAX({column}), {fractional}" if numeric else ""
    return f"SELECT COUNT(DISTINCT {column}){extra} FROM typed_data"


def _profile_sample_sql(clean: str) -> str:
    import modules.disk_dataset as disk

    column = disk._q(clean)
    return (
        f"SELECT {column} FROM typed_data WHERE {column} IS NOT NULL "
        f"GROUP BY {column} ORDER BY MIN(_row_ordinal) LIMIT 20"
    )


def _profile_year_sql(clean: str) -> str:
    import modules.disk_dataset as disk

    column = disk._q(clean)
    return (
        f"SELECT COUNT({column}), COUNT(*) FILTER (WHERE {column} BETWEEN "
        f"{disk.MIN_CALENDAR_YEAR} AND {disk.MAX_CALENDAR_YEAR} "
        f"AND {column} = FLOOR({column})) FROM typed_data"
    )


def _profile_sequence_sql(clean: str) -> str:
    import modules.disk_dataset as disk

    column = disk._q(clean)
    return (
        f"SELECT BOOL_AND({column} = _row_ordinal) OR "
        f"BOOL_AND({column} = _row_ordinal + 1) FROM typed_data"
    )


def _profile_queries(dataset):
    """Return exact baseline profile queries and their fetch methods."""
    queries = []
    for clean in dataset.columns:
        kind = dataset._plans[clean].kind
        numeric = kind in {"integer", "unsigned", "number", "percentage", "currency"}
        queries.append((_profile_stat_sql(dataset, clean), "one"))
        queries.append((_profile_sample_sql(clean), "all"))
        if "year" in set(clean.lower().split("_")) and numeric:
            queries.append((_profile_year_sql(clean), "one"))
    return queries


def _parallel_profile_cache(dataset) -> dict[str, tuple[str, Any]]:
    queries = _profile_queries(dataset)
    groups: list[list[tuple[str, Any]]] = [[] for _ in range(WORKERS)]
    for index, query in enumerate(queries):
        groups[index % WORKERS].append(query)
    cache = _execute_groups(dataset, groups)
    # Sequence checks are only needed when every source row has one distinct
    # integer value. Run those conditionally after unique counts are known.
    sequence_queries = []
    for clean in dataset.columns:
        kind = dataset._plans[clean].kind
        if kind not in {"integer", "unsigned"} or dataset.row_count < 3:
            continue
        stats_row = cache[_profile_stat_sql(dataset, clean)][1]
        unique_count = int(stats_row[0] or 0)
        if unique_count == dataset.row_count:
            sequence_queries.append((_profile_sequence_sql(clean), "one"))
    groups = [[] for _ in range(WORKERS)]
    for index, query in enumerate(sequence_queries):
        groups[index % WORKERS].append(query)
    cache.update(_execute_groups(dataset, groups))
    return cache


def _fused_profile_query(dataset, clean: str) -> dict[str, Any]:
    """Compute profile metadata and first-seen samples in one grouped scan."""
    import modules.disk_dataset as disk

    column = disk._q(clean)
    kind = dataset._plans[clean].kind
    numeric = kind in {"integer", "unsigned", "number", "percentage", "currency"}
    is_year = "year" in set(clean.lower().split("_")) and numeric
    is_sequence = dataset.row_count >= 3 and kind in {"integer", "unsigned"}
    aggregates = ["COUNT(*)", "ARG_MIN(value, first_row, 20)"]
    if numeric:
        fractional = "FALSE" if kind in {"integer", "unsigned"} else "BOOL_OR(value != TRUNC(value))"
        aggregates.extend(["MIN(value)", "MAX(value)", fractional])
    if is_year:
        aggregates.extend([
            "SUM(occurrences)",
            f"SUM(occurrences) FILTER (WHERE value BETWEEN {disk.MIN_CALENDAR_YEAR} "
            f"AND {disk.MAX_CALENDAR_YEAR} AND value = FLOOR(value))",
        ])
    if is_sequence:
        aggregates.append(
            "BOOL_AND(value = first_row) OR BOOL_AND(value = first_row + 1)"
        )
    sql = (
        f"WITH distinct_values AS (SELECT {column} AS value, "
        f"MIN(_row_ordinal) AS first_row, COUNT(*) AS occurrences "
        f"FROM typed_data WHERE {column} IS NOT NULL GROUP BY {column}) "
        f"SELECT {', '.join(aggregates)} FROM distinct_values"
    )
    row = dataset._connection.execute(sql).fetchone()
    index = 0
    result = {"unique_count": int(row[index] or 0)}
    index += 1
    result["samples"] = list(row[index] or [])
    index += 1
    if numeric:
        result["numeric_stats"] = (row[index], row[index + 1], row[index + 2])
        index += 3
    if is_year:
        result["numeric_year_counts"] = (int(row[index] or 0), int(row[index + 1] or 0))
        index += 2
    if is_sequence:
        result["sequence"] = bool(row[index])
    return result


def _fused_profile_cache(dataset) -> dict[str, tuple[str, Any]]:
    cache = {}
    for clean in dataset.columns:
        profile = _fused_profile_query(dataset, clean)
        numeric = "numeric_stats" in profile
        stats_row = (profile["unique_count"], *profile["numeric_stats"]) if numeric else (profile["unique_count"],)
        cache[_profile_stat_sql(dataset, clean)] = ("one", stats_row)
        cache[_profile_sample_sql(clean)] = ("all", [(value,) for value in profile["samples"]])

        if "numeric_year_counts" in profile:
            cache[_profile_year_sql(clean)] = ("one", profile["numeric_year_counts"])
        if "sequence" in profile:
            cache[_profile_sequence_sql(clean)] = ("one", (profile["sequence"],))
    return cache


def _profile_patch_source(original, cache_function) -> str:
    source = textwrap.dedent(inspect.getsource(original))
    loop_start = "    for clean in self.columns:\n        column = _q(clean)\n"
    loop_end = '    self.ingestion_timings["column_profiles"] = perf_counter() - stage_started\n'
    if source.count(loop_start) != 1 or source.count(loop_end) != 1:
        raise RuntimeError("Could not locate the column profile loop in source method.")
    start = source.index(loop_start)
    end = source.index(loop_end, start)
    body = source[start:end]
    wrapped = (
        f"    _benchmark_original_connection = self._connection\n"
        f"    self._connection = _BenchmarkCachedConnection(\n"
        f"        _benchmark_original_connection, _benchmark_profile_cache(self)\n"
        f"    )\n"
        f"    try:\n{textwrap.indent(body, '    ')}"
        f"    finally:\n"
        f"        self._connection = _benchmark_original_connection\n"
    )
    namespace = original.__globals__
    namespace["_BenchmarkCachedConnection"] = _CachedConnection
    namespace["_benchmark_profile_cache"] = cache_function
    return source[:start] + wrapped + source[end:]


def _install_plain_pruning(disk, original_analyze, original_projection) -> None:
    zeros = {
        "percent", "currency_symbol", "currency_grouped", "comma",
        "ambiguous_grouped", "grouped_integer", "grouped_decimal",
    }
    equal_nonnull = {
        "nonnull", "plain", "percent_inner_plain", "currency_plain", "numeric_like",
    }

    def projection(dataset, raw_columns=None):
        expressions, slots = original_projection(dataset, raw_columns)
        proofs = getattr(dataset, "_benchmark_plain_proofs", {})
        for index, (raw, name) in enumerate(slots):
            if raw not in proofs or name not in zeros | equal_nonnull:
                continue
            alias = expressions[index].rsplit(" AS ", 1)[1]
            value = 0 if name in zeros else proofs[raw]
            expressions[index] = f"{value} AS {alias}"
        return expressions, slots

    def analyze(dataset):
        started = perf_counter()
        proofs = {}
        for start in range(0, len(dataset._raw_columns), 4):
            columns = dataset._raw_columns[start:start + 4]
            expressions = []
            for raw in columns:
                column = disk._q(raw)
                trimmed = f"TRIM({column})"
                expressions.extend((
                    f"COUNT({column})",
                    f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND "
                    f"REGEXP_FULL_MATCH({trimmed}, {disk._literal(disk._PLAIN_NUMBER.pattern)}))",
                ))
            counts = dataset._connection.execute(
                f"SELECT {', '.join(expressions)} FROM source_data"
            ).fetchone()
            for index, raw in enumerate(columns):
                nonnull, plain = int(counts[index * 2] or 0), int(counts[index * 2 + 1] or 0)
                if nonnull > 0 and plain == nonnull:
                    proofs[raw] = nonnull
        dataset._benchmark_plain_proofs = proofs
        proof_seconds = perf_counter() - started
        original_analyze(dataset)
        dataset.ingestion_timings["column_validation_and_type_plan"] += proof_seconds
        dataset.ingestion_timings["plain_proof_included_in_validation"] = proof_seconds

    disk.DiskDataset._stats_projection = projection
    disk.DiskDataset._analyze_and_build = analyze


def install(mode: str) -> None:
    """Install one benchmark-only adapter in the current isolated worker."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    # Every benchmark arm is isolated from shell settings and native validation.
    os.environ["ANALYTICO_DUCKDB_THREADS"] = str(THREADS)
    os.environ["ANALYTICO_NATIVE_VALIDATION"] = "0"
    import modules.disk_dataset as disk

    disk.DUCKDB_THREADS = THREADS
    if disk.DUCKDB_MEMORY_LIMIT != MEMORY_LIMIT:
        raise RuntimeError(
            f"Expected the frozen {MEMORY_LIMIT} DuckDB memory cap, got "
            f"{disk.DUCKDB_MEMORY_LIMIT}."
        )
    key = id(disk.DiskDataset)
    if key not in _ORIGINALS:
        _ORIGINALS[key] = (
            disk.DiskDataset._analyze_and_build,
            disk.DiskDataset._stats_projection,
        )
    original_analyze, original_projection = _ORIGINALS[key]
    # Cases normally use fresh workers; restoration also makes repeated smoke
    # installs in one process deterministic.
    disk.DiskDataset._analyze_and_build = original_analyze
    disk.DiskDataset._stats_projection = original_projection
    if mode == "baseline":
        return
    if mode == "plain_pruning4":
        _install_plain_pruning(disk, original_analyze, original_projection)
        return

    source = textwrap.dedent(inspect.getsource(original_analyze))
    if mode == "parallel_stats":
        namespace = original_analyze.__globals__
        namespace["_benchmark_parallel_stats"] = _parallel_stats
        collector_call = "    stats = self._collect_column_statistics()\n"
        if source.count(collector_call) == 1:
            source = source.replace(
                collector_call,
                "    stats = _benchmark_parallel_stats(self)\n",
            )
        else:
            start_marker = "    stats: dict[str, dict[str, int]] = {}\n"
            end_marker = "    self.row_count = int(self._connection.execute(\"SELECT COUNT(*) FROM source_data\").fetchone()[0])\n"
            if source.count(start_marker) != 1 or source.count(end_marker) != 1:
                raise RuntimeError("Could not locate the statistics collector call or legacy batch loop.")
            start = source.index(start_marker)
            end = source.index(end_marker, start)
            source = source[:start] + "    stats = _benchmark_parallel_stats(self)\n" + source[end:]
    else:
        cache_function = _parallel_profile_cache if mode == "parallel_profiles" else _fused_profile_cache
        source = _profile_patch_source(original_analyze, cache_function)
        namespace = original_analyze.__globals__

    exec(compile(source, disk.__file__, "exec"), namespace)
    disk.DiskDataset._analyze_and_build = namespace["_analyze_and_build"]
