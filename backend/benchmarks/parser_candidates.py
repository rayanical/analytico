"""Private CSV parser experiments for benchmark-only installation.

Each candidate is installed for one benchmark run with ``with install(mode):``.
The module does not make network calls or change the production parser module.
"""

from __future__ import annotations

import csv
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, BinaryIO, Iterator, Optional, TextIO
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import pandas as pd

from modules import disk_dataset
from modules.disk_dataset import DiskDataset, MAX_COLUMNS
from modules.import_policy import (
    CSVStructureError,
    ImportSettings,
    read_csv_headers,
    reader_options,
    validate_csv_headers,
    validate_csv_structure,
)


_QUOTE_WHITESPACE = re.compile(rb'"(?:[^"]|"")*"[ \t]+(?:,|\r|\n|$)')
_VALID_MODES = {"baseline", "overlap_validation", "native_only", "single_pass"}


@dataclass(frozen=True)
class ParserOutcome:
    """Small parser result for differential probes and benchmark diagnostics."""

    status: str
    original_headers: tuple[str, ...] = ()
    parsed_headers: tuple[str, ...] = ()
    rows: tuple[tuple[Optional[str], ...], ...] = ()
    row_count: int = 0
    loader: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class _ValidationOverlap:
    """Submit canonical validation, then join after the native raw load."""

    def __init__(self, validator, ingest) -> None:
        self._validator = validator
        self._ingest_csv = ingest
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="csv-validate")
        self.local = threading.local()

    def validate(self, source, settings=None, **kwargs):
        if getattr(self.local, "future", None) is not None:
            return self._validator(source, settings, **kwargs)
        future = self.executor.submit(self._validator, source, settings, **kwargs)
        self.local.future = future
        # from_csv ignores the validation tuple; joining after _ingest_csv makes
        # any validator error visible before its callback or analysis can run.
        return 0, 0

    def ingest(self, dataset, requested_chunk_size: int):
        future: Optional[Future] = getattr(self.local, "future", None)
        try:
            result = self._ingest_csv(dataset, requested_chunk_size)
        except BaseException as load_error:
            if future is not None:
                try:
                    future.result()
                except BaseException as validation_error:
                    raise validation_error from load_error
            raise
        else:
            if future is not None:
                future.result()
            return result
        finally:
            self.local.future = None


def _header_only_preflight(source, settings=None, **kwargs) -> tuple[int, int]:
    """Validate only the header; the single-pass producer validates each row."""
    try:
        headers = read_csv_headers(source, settings)
    except CSVStructureError as header_error:
        # Reuse canonical messages for an empty or malformed header. This path
        # is exceptional and may scan the source because no rows will be loaded.
        try:
            validate_csv_structure(source, settings, **kwargs)
        except CSVStructureError as canonical_error:
            raise canonical_error
        raise header_error

    validate_csv_headers(
        headers,
        max_columns=kwargs.get("max_columns", MAX_COLUMNS),
        max_header_bytes=kwargs.get("max_header_bytes", 256),
    )
    if len(headers) == 1:
        # The existing pandas engine skips whitespace-only physical lines in a
        # single-column file, while csv.reader exposes them as a field.
        return validate_csv_structure(source, settings, **kwargs)
    return 0, len(headers)


def _single_pass_ingest(
    dataset: DiskDataset,
    requested_chunk_size: int,
    *,
    canonical_validator,
    fallback_ingest,
):
    """Stream strict CSV rows directly into bounded DuckDB DataFrame batches."""
    dataset._ensure_open()
    if len(dataset.original_headers) == 1:
        return fallback_ingest(dataset, requested_chunk_size)

    try:
        headers = pd.read_csv(
            dataset.source_path, nrows=0, **reader_options(dataset.import_settings),
        ).columns.tolist()
    except pd.errors.EmptyDataError as error:
        raise ValueError("The CSV file is empty.") from error
    except pd.errors.ParserError:
        # pandas' header parser can inspect malformed quoted data beyond row 0.
        # Preserve canonical errors on rejected inputs, then retain pandas' own
        # error if the canonical parser considers the source structurally valid.
        canonical_validator(
            dataset.source_path, dataset.import_settings,
            max_rows=dataset._max_rows, max_columns=MAX_COLUMNS,
        )
        raise

    dataset._initialize_source_table(headers)
    dataset.csv_loader = "python_stream"
    dataset.csv_fallback_reason = None
    chunk_size = min(requested_chunk_size, max(1, 250_000 // len(headers)))
    null_values = set(dataset.import_settings.null_values)
    batch: list[list[Optional[str]]] = []
    row_count = 0
    header_seen = False
    stream = None
    reader = None

    def append_batch() -> None:
        if not batch:
            return
        frame = pd.DataFrame(batch, columns=dataset._raw_columns, dtype=object)
        dataset._append_chunk(frame)
        batch.clear()

    try:
        stream = dataset.source_path.open(
            "r", encoding=dataset.import_settings.encoding, newline="",
        )
        reader = csv.reader(stream, delimiter=dataset.import_settings.delimiter, strict=True)
        source_headers = next(reader, None)
        while source_headers == []:
            source_headers = next(reader, None)
        if source_headers is None:
            raise CSVStructureError("CSV header has no columns.", row_number=1)
        validate_csv_headers(source_headers, max_columns=MAX_COLUMNS)
        header_seen = True
        if source_headers != dataset.original_headers:
            raise CSVStructureError("CSV header changed while reading the source.", row_number=1)

        for row in reader:
            if not row:
                continue
            row_count += 1
            record_number = row_count + 1
            if row_count > dataset._max_rows:
                raise CSVStructureError(
                    f"CSV exceeds the {dataset._max_rows}-row limit.",
                    row_number=record_number, limit_exceeded=True,
                )
            if any("\x00" in value for value in row):
                raise CSVStructureError(
                    f"CSV contains a NUL byte near row {record_number}.",
                    row_number=record_number,
                )
            if len(row) != len(source_headers):
                detail = "extra fields" if len(row) > len(source_headers) else "missing fields"
                raise CSVStructureError(
                    f"CSV row {record_number} has {detail}; expected {len(source_headers)} columns.",
                    row_number=record_number,
                )

            batch.append([None if value in null_values else value for value in row])
            if len(batch) >= chunk_size:
                append_batch()
        append_batch()
    except csv.Error as error:
        row_number = row_count + 2
        detail = (
            "exceeds the 262144-character field limit"
            if "field larger than field limit" in str(error)
            else "is malformed"
        )
        if not header_seen:
            raise CSVStructureError(
                f"Malformed CSV near line {max(1, reader.line_num if reader else 1)}.",
                row_number=1,
            ) from error
        raise CSVStructureError(
            f"CSV {detail} near line {max(1, reader.line_num)} (row {row_number}).",
            row_number=row_number,
            limit_exceeded="field larger than field limit" in str(error),
        ) from error
    except UnicodeError as error:
        raise CSVStructureError(
            f"CSV encoding could not be decoded as {dataset.import_settings.encoding}."
        ) from error
    finally:
        if stream is not None:
            stream.close()

    dataset.row_count = row_count
    return None


@contextmanager
def install(mode: str) -> Iterator[str]:
    """Install one benchmark-only parser candidate for the current process.

    Modes are ``baseline``, ``overlap_validation``, ``native_only``, and
    ``single_pass``. Install one mode at a time; monkeypatches are process-wide.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown parser candidate {mode!r}; choose from {sorted(_VALID_MODES)}.")

    with ExitStack() as stack:
        validation_mode = "1" if mode == "native_only" else "0"
        stack.enter_context(patch.dict(os.environ, {"ANALYTICO_NATIVE_VALIDATION": validation_mode}))

        if mode == "overlap_validation":
            coordinator = _ValidationOverlap(
                disk_dataset.validate_csv_structure,
                DiskDataset._ingest_csv,
            )
            stack.callback(coordinator.executor.shutdown, wait=True, cancel_futures=False)
            stack.enter_context(patch.object(disk_dataset, "validate_csv_structure", coordinator.validate))
            stack.enter_context(patch.object(
                DiskDataset, "_ingest_csv",
                lambda dataset, chunk_size: coordinator.ingest(dataset, chunk_size),
            ))

        elif mode == "native_only":
            # The production candidate already keeps header/NUL/field/row checks
            # and canonical pandas fallback. Only bypass the conservative quote gate.
            stack.enter_context(patch.object(disk_dataset, "_source_contains_quote_byte", lambda _path: False))

        elif mode == "single_pass":
            canonical_validator = disk_dataset.validate_csv_structure
            fallback_ingest = DiskDataset._ingest_csv
            stack.enter_context(patch.object(
                disk_dataset, "validate_csv_structure", _header_only_preflight,
            ))
            stack.enter_context(patch.object(
                DiskDataset, "_ingest_csv",
                lambda dataset, chunk_size: _single_pass_ingest(
                    dataset, chunk_size, canonical_validator=canonical_validator,
                    fallback_ingest=fallback_ingest,
                ),
            ))

        yield mode


def install_parser_candidate(mode: str):
    """Alias retained for benchmark callers that prefer an explicit name."""
    return install(mode)


def capture_outcome(
    source: str | Path | bytes | bytearray | BinaryIO | TextIO,
    *,
    mode: str = "baseline",
    settings: ImportSettings | dict[str, Any] | None = None,
    **kwargs,
) -> ParserOutcome:
    """Load one source without AI services and return bounded parser evidence."""
    dataset = None
    try:
        with install(mode):
            dataset = DiskDataset.from_csv(source, import_settings=settings, **kwargs)
        headers, rows, row_count = dataset.raw_source_snapshot(
            limit=1_000, evenly_spaced=False,
        )
        return ParserOutcome(
            status="accepted",
            original_headers=tuple(dataset.original_headers),
            parsed_headers=tuple(headers),
            rows=tuple(tuple(row) for row in rows),
            row_count=row_count,
            loader=dataset.csv_loader,
        )
    except Exception as error:
        return ParserOutcome(
            status="rejected",
            error_type=type(error).__name__,
            error_message=str(error),
        )
    finally:
        if dataset is not None:
            dataset.close()


def classify_outcomes(source: bytes, reference: ParserOutcome, candidate: ParserOutcome) -> str:
    """Classify parser deltas, keeping quote-whitespace acceptance explicit."""
    if reference.status == "rejected" and candidate.status == "rejected":
        return "same_rejection" if reference.error_type == candidate.error_type else "rejection_class_change"
    if reference.status == "rejected" and candidate.status == "accepted":
        if _QUOTE_WHITESPACE.search(source):
            return "quote_whitespace_acceptance"
        return "unexpected_acceptance"
    if reference.status == "accepted" and candidate.status == "rejected":
        return "genuinely_broken_rows"
    if candidate.row_count < reference.row_count:
        return "data_loss"
    if candidate.row_count > reference.row_count:
        return "row_count_change"
    if (candidate.original_headers, candidate.parsed_headers, candidate.rows) != (
        reference.original_headers, reference.parsed_headers, reference.rows,
    ):
        return "value_change"
    return "same"
