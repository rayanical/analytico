"""Disk-backed CSV datasets with bounded-memory ingestion and analytics."""

from __future__ import annotations

import csv
import difflib
import io
import math
import os
import re
import shutil
import tempfile
import threading
import unicodedata
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO, Optional, TextIO

import numpy as np
import pandas as pd

from core.config import MAX_CHART_POINTS
from models import (
    AggregateRequest, ChartResponse, ColumnSummary, DataHealth, DataProfile,
    DefaultChart, DrillDownRequest, MetricSummary, TimeRange, UploadResponse,
)
from modules.column_statistics import ColumnStatistics
from modules.data_janitor import (
    DATE_FORMAT_CANDIDATES, UNIVERSAL_CURRENCY_KEYWORDS, _column_tokens,
    _is_date_name, _is_identifier_name, _unique_normalized_headers,
)
from utils.filtering import FilterValidationError, resolve_effective_limit


DEFAULT_CHUNK_SIZE = 50_000
MAX_CHUNK_SIZE = 100_000
MAX_COLUMNS = 256
MAX_COLUMN_NAME_BYTES = 256
MAX_PREVIEW_ROWS = 1_000
DEFAULT_MAX_FILE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_ROWS = 5_000_000
DUCKDB_MEMORY_LIMIT = "64MB"
DUCKDB_THREADS = 2
DUCKDB_TEMP_LIMIT = "1GB"

_PLAIN_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_US_GROUPED_NUMBER = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+\.\d+$")
_INTEGER_TEXT = re.compile(r"^[+-]?\d+$")
_AMBIGUOUS_GROUPED = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+$")
_NUMERIC_LIKE = re.compile(r"^[+-]?[\d.,\s]+$")
_CURRENCY_CHARS = "".join(
    chr(codepoint)
    for codepoint in range(0x110000)
    if unicodedata.category(chr(codepoint)) == "Sc"
)
_CURRENCY_CLASS = "[" + re.escape(_CURRENCY_CHARS) + "]"


class _Plan:
    __slots__ = ("kind", "expression", "dtype", "format", "action", "attempted")

    def __init__(
        self, kind: str, expression: str, dtype: str, format: Optional[str] = None,
        action: Optional[str] = None, attempted: bool = False,
    ) -> None:
        self.kind = kind
        self.expression = expression
        self.dtype = dtype
        self.format = format
        self.action = action
        self.attempted = attempted


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _raw_name(index: int) -> str:
    return f"c{index}"


def _copy_source(
    source: str | Path | bytes | bytearray | BinaryIO | TextIO,
    destination: Path,
    max_bytes: int,
) -> int:
    """Persist source bytes in bounded pieces, returning the copied byte count."""
    total = 0

    def write_chunks(input_file: Any, output_file: Any) -> None:
        nonlocal total
        while True:
            chunk = input_file.read(1024 * 1024)
            if not chunk:
                return
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"CSV file exceeds the {max_bytes}-byte upload limit.")
            output_file.write(chunk)

    if isinstance(source, (str, Path)):
        with Path(source).open("rb") as input_file, destination.open("wb") as output_file:
            write_chunks(input_file, output_file)
        return total
    if isinstance(source, (bytes, bytearray)):
        if len(source) > max_bytes:
            raise ValueError(f"CSV file exceeds the {max_bytes}-byte upload limit.")
        with destination.open("wb") as output_file:
            view = memoryview(source)
            for start in range(0, len(view), 1024 * 1024):
                output_file.write(view[start:start + 1024 * 1024])
        return len(source)
    if not hasattr(source, "read"):
        raise TypeError("CSV source must be a path, bytes, or readable file object.")

    original_position: Optional[int] = None
    try:
        if hasattr(source, "tell") and hasattr(source, "seek"):
            original_position = source.tell()
            source.seek(0)
    except (OSError, ValueError):
        original_position = None
    try:
        with destination.open("wb") as output_file:
            write_chunks(source, output_file)
    finally:
        if original_position is not None:
            try:
                source.seek(original_position)
            except (OSError, ValueError):
                pass
    return total


def _source_headers(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as source_file:
        try:
            return next(csv.reader(source_file))
        except StopIteration:
            return []


def _date_expr(column: str, fmt: str) -> str:
    if fmt == "ISO8601":
        return f"TRY_CAST({column} AS TIMESTAMP)"
    return f"TRY_STRPTIME({column}, {_literal(fmt)})"


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.isoformat()
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        return str(value) if abs(value) > 2**53 - 1 else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if type(value) is int and abs(value) > 2**53 - 1:
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _filter_descriptions(filter_config: Any) -> list[str]:
    column = filter_config.column
    descriptions: list[str] = []
    if filter_config.operator is not None:
        descriptions.append(f"{column} {filter_config.operator} {filter_config.value}")
    if filter_config.values is not None:
        preview = ", ".join(str(value) for value in filter_config.values[:3])
        descriptions.append(f"{column} in ({preview})")
    if filter_config.min_val is not None:
        descriptions.append(f"{column} >= {filter_config.min_val}")
    if filter_config.max_val is not None:
        descriptions.append(f"{column} <= {filter_config.max_val}")
    return descriptions


class DiskDataset:
    """A temporary, disk-backed dataset with bounded previews and SQL analytics.

    Raw source lexemes remain in VARCHAR columns. Public column names match the
    existing deterministic header cleaning. Every conversion decision is based
    on the complete column, and source bytes remain available at source_path.
    """

    def __init__(self) -> None:
        self.id = str(uuid.uuid4())
        self.cache_version = str(uuid.uuid4())
        self.filename = "dataset.csv"
        self.path: Optional[Path] = None
        self.raw_source_path: Optional[Path] = None
        self.database_path: Optional[Path] = None
        self.original_headers: list[str] = []
        self.parsed_headers: list[str] = []
        self.columns: list[str] = []
        self.row_count = 0
        self.column_types: dict[str, str] = {}
        self.column_formats: dict[str, str] = {}
        self.column_dtypes: dict[str, str] = {}
        self.cleaning_actions: list[str] = []
        self.missing_counts: dict[str, int] = {}
        self.raw_missing_counts: dict[str, int] = {}
        self.unique_counts: dict[str, int] = {}
        self.sample_values: dict[str, tuple[Any, ...]] = {}
        self.column_summaries: list[ColumnSummary] = []
        self.column_stats: dict[str, ColumnStatistics] = {}
        self.column_interpretations: dict[str, Any] = {}
        self.profile: dict[str, Any] = {
            "top_metrics": [], "time_range": None, "row_count": 0, "column_count": 0,
        }
        self.default_chart: Optional[dict[str, Any]] = None
        self.suggestions: list[str] = []
        self._temporary_directory: Optional[tempfile.TemporaryDirectory[str]] = None
        self._connection = None
        self._closed = False
        self._lock = threading.RLock()
        self._raw_columns: list[str] = []
        self._plans: dict[str, _Plan] = {}
        self._clean_to_raw: dict[str, str] = {}
        self._max_rows = DEFAULT_MAX_ROWS

    @classmethod
    def from_csv(
        cls,
        source: str | Path | bytes | bytearray | BinaryIO | TextIO,
        *,
        filename: str = "dataset.csv",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        temp_root: str | Path | None = None,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> "DiskDataset":
        """Copy and ingest a CSV using the current reader's exact parser options."""
        if type(chunk_size) is not int or not 1 <= chunk_size <= MAX_CHUNK_SIZE:
            raise ValueError(f"chunk_size must be between 1 and {MAX_CHUNK_SIZE}.")
        if type(max_file_bytes) is not int or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer.")
        if type(max_rows) is not int or max_rows <= 0:
            raise ValueError("max_rows must be a positive integer.")
        try:
            import duckdb
        except ImportError as error:
            raise RuntimeError("DuckDB is required for disk-backed ingestion.") from error

        dataset = cls()
        dataset.filename = Path(filename).name or "dataset.csv"
        dataset._max_rows = max_rows
        try:
            dataset._temporary_directory = tempfile.TemporaryDirectory(
                prefix="analytico-dataset-", dir=str(temp_root) if temp_root else None,
            )
            managed = Path(dataset._temporary_directory.name)
            dataset.path = managed
            dataset.raw_source_path = managed / "source.csv"
            dataset.database_path = managed / "dataset.duckdb"
            spill_path = managed / "spill"
            spill_path.mkdir(mode=0o700)
            _copy_source(source, dataset.raw_source_path, max_file_bytes)
            try:
                dataset.original_headers = _source_headers(dataset.raw_source_path)
            except (UnicodeDecodeError, csv.Error):
                dataset.original_headers = []
            os.chmod(dataset.raw_source_path, 0o400)
            dataset._connection = duckdb.connect(
                str(dataset.database_path),
                config={
                    "memory_limit": DUCKDB_MEMORY_LIMIT,
                    "threads": str(DUCKDB_THREADS),
                    "temp_directory": str(spill_path),
                    "max_temp_directory_size": DUCKDB_TEMP_LIMIT,
                    "preserve_insertion_order": "true",
                },
            )
            dataset._ingest_csv_chunks(chunk_size)
            dataset._analyze_and_build()
            if dataset.row_count > max_rows:
                raise ValueError(f"CSV file exceeds the {max_rows}-row ingestion limit.")
            return dataset
        except Exception:
            dataset.close()
            raise

    @property
    def source_path(self) -> Path:
        if self.raw_source_path is None:
            raise RuntimeError("Dataset source is not available.")
        return self.raw_source_path

    def _ensure_open(self) -> None:
        if self._closed or self._connection is None:
            raise RuntimeError("Disk dataset is closed.")

    def _ingest_csv_chunks(self, requested_chunk_size: int) -> None:
        self._ensure_open()
        try:
            header = pd.read_csv(
                self.source_path, engine="c", dtype=str, keep_default_na=False,
                na_values=[""], low_memory=False, nrows=0,
            )
            headers = header.columns.tolist()
        except pd.errors.EmptyDataError as error:
            raise ValueError("The CSV file is empty.") from error
        self._initialize_source_table(headers)
        chunk_size = min(requested_chunk_size, max(1, 250_000 // len(headers)))
        try:
            chunks = pd.read_csv(
                self.source_path, engine="c", dtype=str, keep_default_na=False,
                na_values=[""], low_memory=False, chunksize=chunk_size,
            )
            for chunk in chunks:
                self._append_chunk(chunk)
        except pd.errors.EmptyDataError as error:
            raise ValueError("The CSV file is empty.") from error

    def _initialize_source_table(self, headers: list[str]) -> None:
        if not headers:
            raise ValueError("The CSV file must contain at least one column.")
        if len(headers) > MAX_COLUMNS:
            raise ValueError(f"CSV files may contain at most {MAX_COLUMNS} columns.")
        if any(len(str(header).encode("utf-8")) > MAX_COLUMN_NAME_BYTES for header in headers):
            raise ValueError("A CSV column name exceeds the supported length.")
        self.parsed_headers = [str(header) for header in headers]
        self._raw_columns = [f"c{index}" for index in range(len(headers))]
        self.columns = _unique_normalized_headers(self.parsed_headers)
        self._clean_to_raw = dict(zip(self.columns, self._raw_columns))
        definitions = [f"{_q('_row_ordinal')} BIGINT"]
        definitions.extend(f"{_q(column)} VARCHAR" for column in self._raw_columns)
        self._connection.execute(f"CREATE TABLE source_data ({', '.join(definitions)})")

    def _append_chunk(self, chunk: pd.DataFrame) -> None:
        if chunk.shape[1] != len(self._raw_columns):
            raise ValueError("CSV columns changed while reading the source.")
        if chunk.empty:
            return
        if self.row_count + len(chunk) > self._max_rows:
            raise ValueError(f"CSV file exceeds the {self._max_rows}-row ingestion limit.")
        values = chunk.to_numpy(dtype=object, copy=True)
        values[pd.isna(values)] = None
        normalized = pd.DataFrame(values, columns=self._raw_columns)
        normalized.insert(
            0, "_row_ordinal",
            np.arange(self.row_count, self.row_count + len(chunk), dtype=np.int64),
        )
        self._connection.register("_incoming_chunk", normalized)
        try:
            columns = ["_row_ordinal", *self._raw_columns]
            quoted = ", ".join(_q(column) for column in columns)
            self._connection.execute(
                f"INSERT INTO source_data ({quoted}) SELECT {quoted} FROM _incoming_chunk"
            )
        finally:
            self._connection.unregister("_incoming_chunk")
        self.row_count += len(chunk)

    def _stats_projection(
        self, raw_columns: Optional[list[str]] = None,
    ) -> tuple[list[str], list[tuple[str, str]]]:
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
                "grouped_decimal": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, {_literal(_US_GROUPED_NUMBER.pattern)}))",
                "numeric_like": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, {_literal(_NUMERIC_LIKE.pattern)}))",
                "leading_zero": f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND REGEXP_FULL_MATCH({trimmed}, '^[+-]?0[0-9]+$'))",
            }
            for name, expression in expressions.items():
                alias = f"s{index}_{name}"
                projections.append(f"{expression} AS {_q(alias)}")
                slots.append((raw_column, name))
        return projections, slots

    def _precision_guard_counts(
        self, raw: str, conditions: dict[str, str],
    ) -> dict[str, int]:
        """Run expensive float-safety predicates only for a chosen parse plan."""
        column = _q(raw)
        projections = [
            f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND ({condition})) AS {_q(name)}"
            for name, condition in conditions.items()
        ]
        row = self._connection.execute(
            f"SELECT {', '.join(projections)} FROM source_data"
        ).fetchone()
        return {name: int(row[index] or 0) for index, name in enumerate(conditions)}

    def _currency_symbol_count(self, raw: str) -> int:
        """Count distinct Unicode currency symbols across the full source column."""
        row = self._connection.execute(
            f"SELECT COUNT(DISTINCT symbol) FROM source_data, "
            f"UNNEST(REGEXP_EXTRACT_ALL({_q(raw)}, {_literal(_CURRENCY_CLASS)})) "
            "AS symbols(symbol)"
        ).fetchone()
        return int(row[0] or 0)

    def _numeric_plan(self, raw: str, clean: str, counts: dict[str, int]) -> Optional[_Plan]:
        nonnull = counts["nonnull"]
        if nonnull == 0 or _is_identifier_name(clean) or counts["leading_zero"]:
            return None

        column = _q(raw)
        trimmed = f"TRIM({column})"
        currency_trim = f"TRIM(REGEXP_REPLACE({trimmed}, {_literal(_CURRENCY_CLASS)}, '', 'g'))"
        safe_integer = "9007199254740991"
        kind: Optional[str] = None
        number_source = trimmed
        display_format: Optional[str] = None
        action: Optional[str] = None
        attempted = False

        if counts["percent"] == nonnull:
            attempted = True
            safe = False
            if counts["percent_inner_plain"] == nonnull:
                percent_number = f"TRIM(REGEXP_REPLACE({trimmed}, '%$', ''))"
                guard = self._precision_guard_counts(raw, {
                    "safe": (
                        f"ISFINITE(TRY_CAST({percent_number} AS DOUBLE)) AND "
                        f"ABS(TRY_CAST({percent_number} AS DOUBLE)) <= {safe_integer}"
                    ),
                })
                safe = guard["safe"] == nonnull
            if counts["percent_inner_plain"] == nonnull and safe:
                kind, display_format = "percentage", "percentage"
                number_source = f"TRIM(REGEXP_REPLACE({trimmed}, '%$', ''))"
                action = f"Converted '{clean}' from percentage text to decimal"
            else:
                action = f"Could not safely parse '{clean}' percentage text without precision loss; retained source values"
        elif counts["currency_symbol"]:
            attempted = True
            currency_symbols = self._currency_symbol_count(raw)
            if currency_symbols > 1:
                action = (
                    f"Could not safely parse '{clean}' mixed currency text: multiple currency "
                    "symbols; retained source values for review"
                )
            elif counts["currency_plain"] == nonnull:
                display_format = "currency"
                number_source = currency_trim
                currency_integer = f"REGEXP_FULL_MATCH({currency_trim}, {_literal(_INTEGER_TEXT.pattern)})"
                currency_guards = self._precision_guard_counts(raw, {
                    "integer": currency_integer,
                    "bigint": (
                        f"{currency_integer} AND TRY_CAST({currency_trim} AS BIGINT) IS NOT NULL"
                    ),
                    "uint64": (
                        f"{currency_integer} AND TRY_CAST({currency_trim} AS UBIGINT) IS NOT NULL"
                    ),
                    "double_safe": (
                        f"ISFINITE(TRY_CAST({currency_trim} AS DOUBLE)) AND "
                        f"ABS(TRY_CAST({currency_trim} AS DOUBLE)) <= {safe_integer}"
                    ),
                })
                if currency_guards["integer"] == nonnull:
                    if currency_guards["bigint"] == nonnull:
                        kind = "integer"
                    elif currency_guards["uint64"] == nonnull:
                        kind = "unsigned"
                    else:
                        action = (
                            f"Could not safely parse '{clean}' as representable currency numbers; "
                            "retained source values"
                        )
                elif currency_guards["double_safe"] == nonnull:
                    kind = "currency"
                else:
                    action = (
                        f"Could not safely parse '{clean}' currency text without precision loss; "
                        "retained source values"
                    )
                if kind is not None:
                    action = f"Converted '{clean}' from currency text to numeric"
            elif counts["currency_grouped"] == nonnull:
                number_source = f"REPLACE({currency_trim}, ',', '')"
                display_format = "currency"
                grouped_currency_number = f"REPLACE({currency_trim}, ',', '')"
                guard = self._precision_guard_counts(raw, {
                    "safe": (
                        f"ISFINITE(TRY_CAST({grouped_currency_number} AS DOUBLE)) AND "
                        f"ABS(TRY_CAST({grouped_currency_number} AS DOUBLE)) <= {safe_integer}"
                    ),
                })
                if guard["safe"] == nonnull:
                    kind = "currency"
                    action = f"Converted '{clean}' from currency text to numeric"
                else:
                    action = (
                        f"Could not safely parse '{clean}' currency text without precision loss; "
                        "retained source values"
                    )
            else:
                action = f"Could not safely parse '{clean}' currency text; retained source values"
        elif counts["plain"] == nonnull:
            if counts["integer"] == nonnull:
                if counts["bigint"] == nonnull:
                    kind = "integer"
                elif counts["uint64"] == nonnull:
                    kind = "unsigned"
                else:
                    action = (
                        f"Could not safely parse '{clean}' as representable numbers; "
                        "retained source values"
                    )
            else:
                guard = self._precision_guard_counts(raw, {
                    "safe": (
                        f"ISFINITE(TRY_CAST({trimmed} AS DOUBLE)) AND "
                        f"ABS(TRY_CAST({trimmed} AS DOUBLE)) <= {safe_integer}"
                    ),
                })
                if guard["safe"] == nonnull:
                    kind = "number"
                else:
                    action = (
                        f"Could not safely parse '{clean}' as numeric values without precision loss; "
                        "retained source values"
                    )
            if kind is not None:
                action = f"Converted '{clean}' from numeric text to numeric"
            attempted = True
        elif counts["comma"]:
            if counts["ambiguous_grouped"] == nonnull:
                action = (
                    f"Could not safely parse '{clean}' numeric text: ambiguous locale separators; "
                    "retained source values"
                )
                attempted = True
            elif counts["grouped_decimal"] == nonnull:
                number_source = f"REPLACE({trimmed}, ',', '')"
                grouped_number = f"REPLACE({trimmed}, ',', '')"
                guard = self._precision_guard_counts(raw, {
                    "safe": (
                        f"ISFINITE(TRY_CAST({grouped_number} AS DOUBLE)) AND "
                        f"ABS(TRY_CAST({grouped_number} AS DOUBLE)) <= {safe_integer}"
                    ),
                })
                if guard["safe"] == nonnull:
                    kind = "number"
                    action = f"Converted '{clean}' from grouped numeric text to numeric"
                else:
                    action = (
                        f"Could not safely parse '{clean}' as numeric values without precision loss; "
                        "retained source values"
                    )
                attempted = True

        # The numeric-like fallback is row weighted, including repeated lexemes.
        if kind is None and action is None and counts["numeric_like"] * 2 >= nonnull:
            action = f"Could not safely parse '{clean}' numeric text; retained source values"
            attempted = True
        if (
            kind is None
            and action is None
            and _column_tokens(clean)
            & (UNIVERSAL_CURRENCY_KEYWORDS | {"percent", "percentage", "pct", "rate", "ratio"})
        ):
            action = f"Could not safely parse '{clean}' numeric text; retained source values"
            attempted = True
        if kind is None:
            return _Plan(
                "text", column, "object", action=action, attempted=attempted,
            )

        if kind == "integer":
            expression, dtype = f"TRY_CAST({number_source} AS BIGINT)", "int64"
        elif kind == "unsigned":
            expression, dtype = f"TRY_CAST({number_source} AS UBIGINT)", "uint64"
        else:
            expression, dtype = f"TRY_CAST({number_source} AS DOUBLE)", "float64"
            if kind == "percentage":
                expression = f"({expression} / 100.0)"
        return _Plan(
            kind, expression, dtype, display_format, action=action, attempted=True,
        )

    def _date_plan(
        self, raw: str, clean: str, first_values: list[Optional[str]], nonnull: int,
    ) -> tuple[Optional[_Plan], Optional[str]]:
        if nonnull == 0:
            return None, None
        sample = pd.Series(first_values[:100], dtype=object)
        if sample.empty:
            return None, None
        candidates: list[str] = []
        for fmt in DATE_FORMAT_CANDIDATES:
            try:
                parsed_sample = pd.to_datetime(sample, format=fmt, errors="coerce")
            except (TypeError, ValueError, OverflowError):
                continue
            if parsed_sample.notna().all():
                candidates.append(fmt)
        if not candidates:
            if _is_date_name(clean):
                return None, f"Could not safely parse non-null values in '{clean}' as date; retained source values"
            return None, None

        column = _q(raw)
        precision_loss = int(self._connection.execute(
            f"SELECT COUNT(*) FILTER (WHERE {column} IS NOT NULL AND ("
            f"REGEXP_MATCHES(TRIM({column}), '[Zz]$|[+-][0-9]{{2}}:?[0-9]{{2}}$') OR "
            f"REGEXP_MATCHES(TRIM({column}), '\\\\.[0-9]{{7,}}'))) FROM source_data"
        ).fetchone()[0] or 0)
        if precision_loss:
            return None, (
                f"Could not safely parse '{clean}' as date without losing timezone or "
                "submicrosecond precision; retained source values"
            )

        parts = [f"COUNT({column})"]
        for index, fmt in enumerate(candidates):
            parts.append(
                f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND {_date_expr(column, fmt)} IS NOT NULL) AS {_q(f'valid_{index}')}"
            )
        pairs: list[tuple[int, int]] = []
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                left_expr = _date_expr(column, candidates[left])
                right_expr = _date_expr(column, candidates[right])
                parts.append(
                    f"COUNT(*) FILTER (WHERE {column} IS NOT NULL AND "
                    f"{left_expr} IS DISTINCT FROM {right_expr}) AS {_q(f'diff_{left}_{right}')}"
                )
                pairs.append((left, right))
        row = self._connection.execute(f"SELECT {', '.join(parts)} FROM source_data").fetchone()
        total = int(row[0] or 0)
        valid = [
            index for index in range(len(candidates))
            if int(row[1 + index] or 0) == total
        ]
        if not valid:
            return None, f"Could not safely parse all non-null values in '{clean}' as date; retained source values"
        for pair_index, (left, right) in enumerate(pairs):
            if left in valid and right in valid and int(row[1 + len(candidates) + pair_index] or 0):
                return None, f"Could not safely parse '{clean}' as date: ambiguous date order; retained source values"
        selected = valid[0]
        return _Plan(
            "date", _date_expr(column, candidates[selected]), "datetime64[ns]",
            "date", action=f"Parsed '{clean}' as date",
        ), None

    def _analyze_and_build(self) -> None:
        self._ensure_open()
        first_values = {column: [] for column in self._raw_columns}
        selection = ", ".join([_q("_row_ordinal"), *(_q(column) for column in self._raw_columns)])
        for row in self._connection.execute(
            f"SELECT {selection} FROM source_data ORDER BY _row_ordinal LIMIT 100"
        ).fetchall():
            for index, raw in enumerate(self._raw_columns, start=1):
                if row[index] is not None and len(first_values[raw]) < 100:
                    first_values[raw].append(row[index])

        stats: dict[str, dict[str, int]] = {}
        # DuckDB keeps aggregate state for every projected column. Small
        # batches avoid exhausting the fixed 64 MB connection cap on wide
        # uploads while retaining SQL's exact full-column consensus checks.
        stats_batch_size = 4
        for start in range(0, len(self._raw_columns), stats_batch_size):
            selected = self._raw_columns[start:start + stats_batch_size]
            projections, slots = self._stats_projection(selected)
            raw_result = self._connection.execute(
                f"SELECT {', '.join(projections)} FROM source_data"
            ).fetchone()
            for offset, (raw, name) in enumerate(slots):
                stats.setdefault(raw, {})[name] = int(raw_result[offset] or 0)
        self.row_count = int(self._connection.execute("SELECT COUNT(*) FROM source_data").fetchone()[0])

        renamed_count = sum(a != b for a, b in zip(self.parsed_headers, self.columns))
        if renamed_count:
            self.cleaning_actions.append(f"Normalized {renamed_count} column headers")

        typed_select = [_q("_row_ordinal")]
        for raw, clean in zip(self._raw_columns, self.columns):
            counts = stats[raw]
            missing = self.row_count - counts["nonnull"]
            if missing:
                self.missing_counts[clean] = missing
                self.raw_missing_counts[clean] = missing
                self.cleaning_actions.append(
                    f"Preserved {missing} source missing values as null in '{clean}'"
                )
            plan = self._numeric_plan(raw, clean, counts)
            if plan is not None and plan.kind == "text" and plan.action is None and not plan.attempted:
                if not _is_identifier_name(clean):
                    date_plan, date_action = self._date_plan(raw, clean, first_values[raw], counts["nonnull"])
                    if date_plan is not None:
                        plan = date_plan
                    elif date_action:
                        plan.action = date_action
            if plan is None:
                # Identifier labels commonly contain compact YYYYMMDD-like
                # values; preserve them unless their name explicitly means time.
                if _is_identifier_name(clean) and not _is_date_name(clean):
                    date_plan, date_action = None, None
                else:
                    date_plan, date_action = self._date_plan(
                        raw, clean, first_values[raw], counts["nonnull"]
                    )
                plan = date_plan or _Plan("text", _q(raw), "object", action=date_action)
            if plan.action:
                self.cleaning_actions.append(plan.action)
            self._plans[clean] = plan
            self.column_dtypes[clean] = plan.dtype
            self.column_formats[clean] = self._format_for_column(clean, plan)
            typed_select.append(f"{plan.expression} AS {_q(clean)}")

        self._connection.execute(
            f"CREATE TABLE typed_data AS SELECT {', '.join(typed_select)} FROM source_data"
        )
        for clean in self.columns:
            column = _q(clean)
            unique_count = int(self._connection.execute(
                f"SELECT COUNT(DISTINCT {column}) FROM typed_data"
            ).fetchone()[0] or 0)
            samples = self._connection.execute(
                f"SELECT {column} FROM typed_data WHERE {column} IS NOT NULL "
                f"GROUP BY {column} ORDER BY MIN(_row_ordinal) LIMIT 20"
            ).fetchall()
            sample_values = tuple(row[0] for row in samples)
            self.unique_counts[clean] = unique_count
            self.sample_values[clean] = sample_values
            numeric_year = self._numeric_year(clean)
            self.column_stats[clean] = ColumnStatistics(unique_count, sample_values, numeric_year)
            self.column_types[clean] = self._semantic_type(clean)
        self._build_profile_and_suggestions()

    def _format_for_column(self, column: str, plan: _Plan) -> str:
        if plan.kind == "date":
            return "date"
        if plan.format:
            return plan.format
        lower = column.lower()
        if _is_identifier_name(column) and not any(word in lower for word in UNIVERSAL_CURRENCY_KEYWORDS):
            return "identifier"
        if any(word in lower for word in UNIVERSAL_CURRENCY_KEYWORDS):
            return "currency"
        if any(word in lower for word in ("percent", "pct", "rate", "ratio")):
            return "percentage"
        return "number" if plan.kind in {"integer", "unsigned", "number"} else "general"

    def _numeric_year(self, column: str) -> Optional[bool]:
        if "year" not in set(column.lower().split("_")):
            return None
        if self._plans[column].kind not in {"integer", "unsigned", "number", "percentage", "currency"}:
            return None
        qcolumn = _q(column)
        total, valid = self._connection.execute(
            f"SELECT COUNT({qcolumn}), COUNT(*) FILTER (WHERE {qcolumn} BETWEEN 1000 AND 2200 "
            f"AND {qcolumn} = FLOOR({qcolumn})) FROM typed_data"
        ).fetchone()
        return bool(total and total == valid)

    def _semantic_type(self, column: str) -> str:
        plan = self._plans[column]
        tokens = set(column.lower().split("_"))
        if plan.kind == "date":
            return "temporal"
        if tokens & {
            "id", "code", "key", "name", "email", "phone", "address",
            "zip", "postal", "ssn", "account", "serial", "ref",
        }:
            return "identifier"
        if plan.kind in {"integer", "unsigned", "number", "percentage", "currency"}:
            if "year" in tokens and self.column_stats[column].numeric_year:
                return "temporal"
            if self.unique_counts[column] / max(self.row_count, 1) > 0.5:
                return "metric"
            return "categorical" if self.unique_counts[column] < 20 else "metric"
        return "categorical" if self.unique_counts[column] < 50 else "identifier"

    @staticmethod
    def _non_additive_metric(column: str, column_format: Optional[str]) -> bool:
        tokens = set(str(column).lower().split("_"))
        name = str(column).lower()
        return column_format == "percentage" or "life_exp" in name or bool(
            tokens & {
                "rate", "ratio", "pct", "percent", "percentage", "lifeexp",
                "expectancy", "score", "index", "average", "mean", "age",
                "temperature",
            }
        )

    def _build_profile_and_suggestions(self) -> None:
        metric_columns = [
            column for column in self.columns
            if self.column_types[column] == "metric"
            and self._plans[column].kind in {"integer", "unsigned", "number", "percentage", "currency"}
        ]
        temporal_columns = [
            column for column in self.columns
            if self.column_types[column] == "temporal"
            and (self._plans[column].kind == "date" or self.column_stats[column].numeric_year)
        ]
        categorical_columns = [
            column for column in self.columns if self.column_types[column] == "categorical"
        ]
        top_metrics: list[dict[str, Any]] = []
        for column in metric_columns[:3]:
            qcolumn = _q(column)
            total, average, minimum, maximum = self._connection.execute(
                f"SELECT SUM({qcolumn}), AVG({qcolumn}), MIN({qcolumn}), MAX({qcolumn}) FROM typed_data"
            ).fetchone()
            if average is None:
                continue
            aggregation = (
                "mean"
                if self._non_additive_metric(column, self.column_formats[column])
                else "sum"
            )
            top_metrics.append({
                "name": column,
                "total": float(total) if total is not None else 0.0,
                "average": float(average),
                "min": float(minimum),
                "max": float(maximum),
                "aggregation": aggregation,
            })

        time_range = None
        if temporal_columns:
            column = temporal_columns[0]
            start, end = self._connection.execute(
                f"SELECT MIN({_q(column)}), MAX({_q(column)}) FROM typed_data"
            ).fetchone()
            if start is not None:
                time_range = {"column": column, "start": str(start), "end": str(end)}
        self.profile = {
            "top_metrics": top_metrics,
            "time_range": time_range,
            "row_count": self.row_count,
            "column_count": len(self.columns),
        }

        default_chart = None
        if metric_columns:
            metric = metric_columns[0]
            aggregation = (
                "mean"
                if self._non_additive_metric(metric, self.column_formats[metric])
                else "sum"
            )
            axis = temporal_columns[0] if temporal_columns else None
            chart_type = "line"
            if axis is None and categorical_columns:
                axis = min(categorical_columns, key=lambda item: abs(self.unique_counts[item] - 10))
                chart_type = "bar"
            if axis is not None:
                default_chart = {
                    "x_axis_key": axis,
                    "y_axis_keys": [metric],
                    "chart_type": chart_type,
                    "aggregation": aggregation,
                    "title": f"{aggregation.title()} {metric} by {axis}".replace("_", " ").title(),
                    "analysis": f"Shows the {aggregation} of {metric} for each observed {axis} value.",
                }
        self.default_chart = default_chart
        self.suggestions = self._build_suggestions(metric_columns, temporal_columns, categorical_columns)

        summaries: list[ColumnSummary] = []
        for column in self.columns:
            plan = self._plans[column]
            sample_values = sorted(
                (str(value) for value in self.sample_values[column]),
                key=str.lower,
            )
            summaries.append(ColumnSummary(
                name=column,
                dtype=plan.dtype,
                is_numeric=plan.kind in {"integer", "unsigned", "number", "percentage", "currency"},
                is_datetime=plan.kind == "date",
                semantic_type=self.column_types[column],
                format=self.column_formats[column],
                unique_count=self.unique_counts[column],
                sample_values=sample_values,
            ))
        self.column_summaries = summaries

    @staticmethod
    def _build_suggestions(
        metric_columns: list[str],
        temporal_columns: list[str],
        categorical_columns: list[str],
    ) -> list[str]:
        def formatted(column: str) -> str:
            return column.replace("_", " ")

        def normalize(prompt: str, max_len: int = 72) -> str:
            compact = " ".join(prompt.split())
            return compact if len(compact) <= max_len else compact[:max_len - 1].rstrip() + "…"

        suggestions: list[str] = []
        if categorical_columns and metric_columns:
            suggestions.append(normalize(
                f"Show {formatted(metric_columns[0])} by categories in {formatted(categorical_columns[0])}"
            ))
        if temporal_columns and metric_columns:
            suggestions.append(normalize(
                f"How does {formatted(metric_columns[0])} trend over {formatted(temporal_columns[0])}?"
            ))
        if categorical_columns and metric_columns:
            suggestions.append(normalize(
                f"Which {formatted(categorical_columns[0])} categories have the highest {formatted(metric_columns[0])}?"
            ))
        elif metric_columns:
            suggestions.append(normalize(f"What is the distribution of {formatted(metric_columns[0])}?"))
        if len(metric_columns) >= 2:
            suggestions.append(normalize(
                f"Compare {formatted(metric_columns[0])} vs {formatted(metric_columns[1])}"
            ))
        if metric_columns:
            suggestions.append(normalize(f"What is the average {formatted(metric_columns[0])}?"))
        suggestions.append("Give me a quick summary of this dataset")
        return suggestions

    def to_upload_response(
        self, dataset_id: Optional[str] = None, filename: Optional[str] = None,
    ) -> UploadResponse:
        """Build the existing upload response from bounded metadata."""
        cell_count = self.row_count * len(self.columns)
        missing_count = sum(self.missing_counts.values())
        quality = max(0.0, 100.0 - missing_count / max(cell_count, 1) * 100.0)
        profile = DataProfile(
            top_metrics=[MetricSummary(**item) for item in self.profile["top_metrics"]],
            time_range=TimeRange(**self.profile["time_range"]) if self.profile["time_range"] else None,
            row_count=self.row_count,
            column_count=len(self.columns),
        )
        return UploadResponse(
            dataset_id=dataset_id or self.id,
            filename=filename or self.filename,
            row_count=self.row_count,
            columns=self.column_summaries,
            column_formats=self.column_formats,
            data_health=DataHealth(
                missing_values=self.missing_counts,
                cleaning_actions=self.cleaning_actions,
                quality_score=round(quality, 1),
            ),
            profile=profile,
            default_chart=DefaultChart(**self.default_chart) if self.default_chart else None,
            suggestions=self.suggestions,
            summary=None,
        )

    def sample_frame(self, limit: int = 5) -> pd.DataFrame:
        """Return a small parsed preview, never the complete table."""
        self._ensure_open()
        if type(limit) is not int or not 0 <= limit <= MAX_PREVIEW_ROWS:
            raise ValueError(f"Preview limit must be between 0 and {MAX_PREVIEW_ROWS}.")
        projection = ", ".join(_q(column) for column in self.columns)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT {projection} FROM typed_data ORDER BY _row_ordinal LIMIT ?",
                [limit],
            ).fetchall()
        values_by_column = {
            column: [row[index] for row in rows]
            for index, column in enumerate(self.columns)
        }
        # Pandas' default null inference upcasts nullable integers to float64,
        # which rounds adjacent values above 2**53. Keep exact integer samples
        # as object values while allowing ordinary columns to infer naturally.
        frame_columns: dict[str, pd.Series] = {}
        for column, values in values_by_column.items():
            if self._plans[column].kind in {"integer", "unsigned"}:
                frame_columns[column] = pd.Series(values, dtype=object)
            else:
                frame_columns[column] = pd.Series(values)
        return pd.DataFrame(frame_columns, columns=self.columns)

    def interpretation_inputs(self, limit: int = 12) -> list[tuple[str, dict[str, Any]]]:
        """Build bounded, source-grounded interpretation proposals for a worker."""
        self._ensure_open()
        if type(limit) is not int or not 0 <= limit <= 12:
            raise ValueError("Interpretation limit must be between 0 and 12.")
        if not limit or not self.columns:
            return []
        positions = sorted({
            round(index * (self.row_count - 1) / 11)
            for index in range(12)
        }) if self.row_count else []
        if positions:
            placeholders = ", ".join("?" for _ in positions)
            projection = ", ".join([_q("_row_ordinal"), *(_q(column) for column in self._raw_columns)])
            rows = self._connection.execute(
                f"SELECT {projection} FROM source_data WHERE _row_ordinal IN ({placeholders}) "
                "ORDER BY _row_ordinal",
                positions,
            ).fetchall()
        else:
            rows = []
        by_position = {int(row[0]): row for row in rows}
        headers = self.parsed_headers
        output: list[tuple[str, dict[str, Any]]] = []
        for column_index, clean_column in enumerate(self.columns[:limit]):
            values: list[Any] = []
            for position in positions:
                row = by_position.get(position)
                value = row[column_index + 1] if row is not None else None
                values.append(None if value is None else str(value)[:160])
            non_null = [value for value in values if value is not None]
            output.append((clean_column, {
                "column_name": headers[column_index][:256],
                "values": values,
                "context": {
                    "purpose": "Conservative dataset ingestion; no requested calculation or external unit metadata.",
                    "row_count": self.row_count,
                    "missing_count": self.missing_counts.get(clean_column, 0),
                    "sample_unique_count": len({str(value) for value in non_null}),
                    "sample_is_complete": self.row_count <= 12,
                    "other_column_names": [
                        header[:128] for index, header in enumerate(headers)
                        if index != column_index
                    ][:20],
                },
            }))
        return output

    def _resolve_column(self, name: str) -> str:
        if name in self._clean_to_raw:
            return _q(name)
        close = difflib.get_close_matches(name, self.columns, n=3, cutoff=0.4)
        raise ValueError(f"'{name}' not found, try: {close}")

    def _is_numeric(self, column: str) -> bool:
        return self._plans[column].kind in {"integer", "unsigned", "number", "percentage", "currency"}

    def _is_datetime(self, column: str) -> bool:
        return self._plans[column].kind == "date"

    def _filter_operand(self, column: str, value: Any) -> Any:
        if self._is_datetime(column):
            try:
                stamp = pd.Timestamp(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise FilterValidationError(
                    f"Filter for '{column}' requires a valid date or timestamp."
                ) from error
            if pd.isna(stamp):
                raise FilterValidationError(
                    f"Filter for '{column}' requires a valid date or timestamp."
                )
            # Pandas localizes a naive source time to its source timezone. The
            # disk parser currently creates timezone-naive timestamps.
            if stamp.tzinfo is not None:
                stamp = stamp.tz_localize(None)
            return stamp.to_pydatetime()
        if self._is_numeric(column):
            if isinstance(value, bool):
                raise FilterValidationError(f"Filter for '{column}' requires a numeric value.")
            try:
                token = str(value)
                if len(token) > 512:
                    raise ValueError("Numeric operand too long")
                number = Decimal(token)
                if not number.is_finite() or abs(number.adjusted()) > 512:
                    raise ValueError("Numeric operand out of range")
            except (ValueError, TypeError, ArithmeticError) as error:
                raise FilterValidationError(
                    f"Filter for '{column}' requires a finite numeric value."
                ) from error
            if self._plans[column].kind in {"integer", "unsigned"}:
                return int(number) if number == number.to_integral_value() else number
            converted = float(number)
            if not math.isfinite(converted):
                raise FilterValidationError(
                    f"Filter for '{column}' requires a finite numeric value."
                )
            return converted
        if value is None or isinstance(value, (dict, list, tuple, set)):
            raise FilterValidationError(f"Filter for '{column}' requires a scalar value.")
        return str(value)

    def _compile_filters(
        self, filters: Optional[list[Any]],
    ) -> tuple[list[str], list[Any], list[str]]:
        predicates: list[str] = []
        parameters: list[Any] = []
        applied: list[str] = []
        sql_operator = {"eq": "=", "gt": ">", "lt": "<", "gte": ">=", "lte": "<="}

        for item in filters or []:
            column = item.column
            qcolumn = self._resolve_column(column)
            if item.operator is not None:
                operator = item.operator
                if operator == "contains":
                    if self._is_numeric(column) or self._is_datetime(column):
                        raise FilterValidationError(
                            f"The contains operator is not supported for '{column}'."
                        )
                    predicates.append(f"CONTAINS(LOWER(CAST({qcolumn} AS VARCHAR)), LOWER(?))")
                    parameters.append(item.value)
                elif operator != "eq" and not (
                    self._is_numeric(column) or self._is_datetime(column)
                ):
                    raise FilterValidationError(
                        f"Operator '{operator}' requires a numeric or date column ('{column}')."
                    )
                elif self._is_numeric(column) or self._is_datetime(column):
                    predicates.append(f"{qcolumn} {sql_operator[operator]} ?")
                    parameters.append(self._filter_operand(column, item.value))
                elif type(item.value) is str:
                    predicates.append(f"{qcolumn} = ?")
                    parameters.append(item.value)
                else:
                    # String Series comparisons against numeric/bool operands
                    # are false in the DataFrame implementation.
                    predicates.append("FALSE")
                applied.extend(_filter_descriptions(item)[:1])

            if item.values is not None:
                includes_null = any(value is None for value in item.values)
                if self._is_numeric(column) or self._is_datetime(column):
                    values = [
                        self._filter_operand(column, value)
                        for value in item.values if value is not None
                    ]
                else:
                    values = [value for value in item.values if type(value) is str]
                options: list[str] = []
                if values:
                    options.append(f"{qcolumn} IN ({', '.join('?' for _ in values)})")
                    parameters.extend(values)
                if includes_null:
                    options.append(f"{qcolumn} IS NULL")
                predicates.append("(" + " OR ".join(options or ["FALSE"]) + ")")
                descriptions = _filter_descriptions(item)
                if len(descriptions) > 1:
                    applied.append(descriptions[1])

            if item.min_val is not None and item.max_val is not None:
                lower = self._filter_operand(column, item.min_val)
                upper = self._filter_operand(column, item.max_val)
                if lower > upper:
                    raise FilterValidationError(
                        f"Minimum filter value exceeds maximum for '{column}'."
                    )
            if item.min_val is not None:
                if not (self._is_numeric(column) or self._is_datetime(column)):
                    raise FilterValidationError(
                        f"Operator 'gte' requires a numeric or date column ('{column}')."
                    )
                predicates.append(f"{qcolumn} >= ?")
                parameters.append(self._filter_operand(column, item.min_val))
                applied.append(f"{column} >= {item.min_val}")
            if item.max_val is not None:
                if not (self._is_numeric(column) or self._is_datetime(column)):
                    raise FilterValidationError(
                        f"Operator 'lte' requires a numeric or date column ('{column}')."
                    )
                predicates.append(f"{qcolumn} <= ?")
                parameters.append(self._filter_operand(column, item.max_val))
                applied.append(f"{column} <= {item.max_val}")
        return predicates, parameters, applied

    @staticmethod
    def _aggregate_expression(aggregation: str, column: str) -> str:
        allowed = {
            "sum": f"SUM({column})",
            "mean": f"AVG({column})",
            "median": f"MEDIAN({column})",
            "count": f"COUNT({column})",
            "min": f"MIN({column})",
            "max": f"MAX({column})",
        }
        if aggregation not in allowed:
            raise ValueError(f"Unsupported aggregation '{aggregation}'.")
        return allowed[aggregation]

    def aggregate(self, request: AggregateRequest) -> ChartResponse:
        """Execute a closed AggregateRequest plan directly in DuckDB."""
        self._ensure_open()
        self._resolve_column(request.x_axis_key)
        for column in request.y_axis_keys:
            self._resolve_column(column)
        if request.x_axis_key in request.y_axis_keys:
            raise ValueError("The x-axis column cannot also be a measure.")
        for column in request.y_axis_keys:
            if not self._is_numeric(column) and request.aggregation != "count":
                raise ValueError(
                    f"Measure '{column}' is not numeric; choose count or a numeric measure."
                )
        identifier_measures = [
            column for column in request.y_axis_keys
            if self.column_types.get(column) == "identifier"
        ]
        if identifier_measures and request.aggregation != "count":
            raise ValueError(
                f"Identifier measures require count aggregation: {identifier_measures}"
            )

        limit, cap_warning = resolve_effective_limit(request.limit, default_limit=50)
        predicates, parameters, applied_filters = self._compile_filters(request.filters)
        where = " WHERE " + " AND ".join(predicates) if predicates else ""
        x_column = _q(request.x_axis_key)
        y_columns = [_q(column) for column in request.y_axis_keys]
        is_datetime = self._is_datetime(request.x_axis_key)
        if request.time_bucket is not None and not is_datetime:
            raise ValueError(
                f"Time bucketing requires a datetime column ('{request.x_axis_key}')."
            )

        time_bucket: Optional[str] = None
        x_expression = x_column
        ctes = f"filtered AS (SELECT * FROM typed_data{where})"
        if is_datetime:
            with self._lock:
                cardinality, minimum, maximum = self._connection.execute(
                    f"SELECT COUNT(DISTINCT {x_column}), MIN({x_column}), MAX({x_column}) "
                    f"FROM typed_data{where}",
                    parameters,
                ).fetchone()
            selected_bucket = request.time_bucket
            if selected_bucket is None and cardinality and cardinality > 100:
                days = (maximum - minimum).days if maximum is not None and minimum is not None else 0
                selected_bucket = "year" if days > 730 else "month" if days > 180 else "week"
            if selected_bucket is not None:
                time_bucket = selected_bucket
                if selected_bucket not in {"year", "month", "week"}:
                    raise ValueError(f"Unsupported time bucket '{selected_bucket}'.")
                x_expression = f"DATE_TRUNC('{selected_bucket}', {x_column})"

        with self._lock:
            matched_rows = self._connection.execute(
                f"SELECT COUNT(*) FROM typed_data{where}", parameters,
            ).fetchone()[0]
        if not matched_rows:
            raise ValueError("No data matches filters.")

        use_others = False
        others_label: Optional[str] = None
        group_others = request.group_others if request.group_others is not None else True
        if group_others and not is_datetime:
            with self._lock:
                group_count = self._connection.execute(
                    f"SELECT COUNT(*) FROM (SELECT {x_column} FROM typed_data{where} GROUP BY {x_column}) groups",
                    parameters,
                ).fetchone()[0]
                use_others = int(group_count or 0) > limit
            if use_others:
                label_pattern = r"^Others \(([1-9][0-9]*)\)$"
                label_condition = (" AND " if predicates else " WHERE ") + f"{x_column} IS NOT NULL"
                with self._lock:
                    base_collision, first_suffix = self._connection.execute(
                        f"WITH filtered_labels AS ("
                        f"SELECT CAST({x_column} AS VARCHAR) AS label FROM typed_data{where}{label_condition}"
                        f"), occupied AS ("
                        f"SELECT DISTINCT TRY_CAST(REGEXP_EXTRACT(label, {_literal(label_pattern)}, 1) AS HUGEINT) AS suffix "
                        f"FROM filtered_labels WHERE REGEXP_FULL_MATCH(label, {_literal(label_pattern)})"
                        f"), ordered AS ("
                        "SELECT suffix, LAG(suffix) OVER (ORDER BY suffix) AS previous "
                        "FROM occupied WHERE suffix >= 2"
                        ") SELECT "
                        "EXISTS (SELECT 1 FROM filtered_labels WHERE label = ?), "
                        "COALESCE((SELECT MIN(CASE "
                        "WHEN previous IS NULL AND suffix > 2 THEN 2 "
                        "WHEN previous IS NOT NULL AND suffix > previous + 1 THEN previous + 1 "
                        "END) FROM ordered), "
                        "COALESCE((SELECT MAX(suffix) + 1 FROM occupied WHERE suffix >= 2), 2))",
                        [*parameters, "Others"],
                    ).fetchone()
                others_label = (
                    f"Others ({int(first_suffix)})" if base_collision else "Others"
                )

        source = "filtered"
        output_axis = "__chart_x"
        group_by = "__chart_x"
        if use_others:
            first_measure = y_columns[0]
            rank_score = self._aggregate_expression(request.aggregation, first_measure)
            top_count = max(limit - 1, 0)
            ctes += (
                f", ranked AS (SELECT {x_expression} AS __rank_x, {rank_score} AS __score, "
                f"MIN(_row_ordinal) AS __rank_first FROM filtered GROUP BY __rank_x), "
                f"top_groups AS (SELECT __rank_x FROM ranked "
                f"ORDER BY __score DESC NULLS LAST, __rank_first ASC "
                f"LIMIT {top_count}), "
                f"mapped AS (SELECT f.*, "
                f"NOT EXISTS (SELECT 1 FROM top_groups g "
                f"WHERE g.__rank_x IS NOT DISTINCT FROM {x_expression}) AS __is_other, "
                f"CASE WHEN EXISTS (SELECT 1 FROM top_groups g "
                f"WHERE g.__rank_x IS NOT DISTINCT FROM {x_expression}) "
                f"THEN {x_expression} ELSE NULL END AS __group_x "
                f"FROM filtered f)"
            )
            source = "mapped"
            output_axis = f"CASE WHEN __is_other THEN NULL ELSE __group_x END"
            group_by = "__is_other, __group_x"
        else:
            ctes += f", bucketed AS (SELECT *, {x_expression} AS __chart_x FROM filtered)"
            source = "bucketed"

        aggregate_fields = [
            f"{self._aggregate_expression(request.aggregation, column)} "
            f"AS {_q(original)}"
            for original, column in zip(request.y_axis_keys, y_columns)
        ]
        axis_fields = f"{output_axis} AS {_q(request.x_axis_key)}"
        if use_others:
            axis_fields += ", __is_other, __group_x"
        sort_by = request.sort_by
        if sort_by is None:
            sort_by = (
                "label"
                if self.column_types.get(request.x_axis_key) in {"metric", "temporal"}
                else "value"
            )
        if sort_by == "value":
            order_by = f"{_q(request.y_axis_keys[0])} DESC NULLS LAST, __first_row ASC"
        elif use_others:
            label_value = (
                f"CASE WHEN __is_other THEN {_literal(others_label or 'Others')} "
                "WHEN __group_x IS NULL THEN '' ELSE CAST(__group_x AS VARCHAR) END"
            )
            order_by = f"{label_value} ASC, __first_row ASC"
        elif is_datetime or self._is_numeric(request.x_axis_key):
            order_by = f"{x_expression} ASC NULLS LAST, __first_row ASC"
        else:
            order_by = f"COALESCE(CAST({x_expression} AS VARCHAR), '') ASC, __first_row ASC"
        statement = (
            f"WITH {ctes}, aggregated AS (SELECT {axis_fields}, "
            f"{', '.join(aggregate_fields)}, MIN(_row_ordinal) AS __first_row "
            f"FROM {source} GROUP BY {group_by}) SELECT * FROM aggregated "
            f"ORDER BY {order_by} LIMIT ?"
        )
        with self._lock:
            try:
                rows = self._connection.execute(
                    statement, [*parameters, limit + 1],
                ).fetchall()
                names = [column[0] for column in self._connection.description]
            except Exception as error:
                raise ValueError("The aggregate request could not be executed.") from error

        was_capped = len(rows) > limit
        axis_index = names.index(request.x_axis_key)
        other_index = names.index("__is_other") if use_others else None
        data: list[dict[str, Any]] = []
        for row in rows[:limit]:
            record: dict[str, Any] = {}
            for index, name in enumerate(names):
                if name in {"__is_other", "__group_x", "__first_row"}:
                    continue
                value = row[index]
                if index == axis_index and other_index is not None and row[other_index]:
                    value = others_label
                record[name] = _json_value(value)
            data.append(record)
        warnings: list[str] = []
        if cap_warning and (
            was_capped
            or request.limit == 0
            or (request.limit is not None and request.limit > MAX_CHART_POINTS)
        ):
            warnings.append(cap_warning)
        return ChartResponse(
            data=data,
            x_axis_key=request.x_axis_key,
            y_axis_keys=request.y_axis_keys,
            chart_type=request.chart_type,
            title=f"{', '.join(request.y_axis_keys)} by {request.x_axis_key}".replace("_", " ").title(),
            aggregation=request.aggregation,
            y_axis_label="Count of Records" if request.aggregation == "count" else None,
            row_count=len(data),
            analysis=None,
            warnings=warnings or None,
            applied_filters=applied_filters or None,
            filters=request.filters or None,
            source_x_axis_key=request.x_axis_key,
            time_bucket=time_bucket,
            others_label=others_label,
        )

    def drilldown(self, request: DrillDownRequest) -> dict[str, Any]:
        """Return matching rows up to the requested limit and exact row count."""
        self._ensure_open()
        predicates, parameters, _ = self._compile_filters(request.filters)
        where = " WHERE " + " AND ".join(predicates) if predicates else ""
        projection = ", ".join(_q(column) for column in self.columns)
        with self._lock:
            total = self._connection.execute(
                f"SELECT COUNT(*) FROM typed_data{where}", parameters,
            ).fetchone()[0]
            rows = self._connection.execute(
                f"SELECT {projection} FROM typed_data{where} "
                "ORDER BY _row_ordinal LIMIT ?",
                [*parameters, request.limit],
            ).fetchall()
        data = [
            {column: _json_value(value) for column, value in zip(self.columns, row)}
            for row in rows
        ]
        return {"data": data, "total_rows": int(total), "limit": request.limit}

    def close(self) -> None:
        """Close the database and remove its managed temporary files."""
        if self._closed:
            return
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._temporary_directory is not None:
                self._temporary_directory.cleanup()
                self._temporary_directory = None
            self._closed = True

    def __enter__(self) -> "DiskDataset":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["DiskDataset", "DEFAULT_CHUNK_SIZE", "MAX_CHUNK_SIZE"]
