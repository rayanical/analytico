"""Locally staged imports: bounded preview, explicit confirmation, owned cleanup."""

from dataclasses import dataclass, field
import csv
import codecs
import io
from pathlib import Path
from threading import RLock
from time import monotonic
import uuid

import pandas as pd

from fastapi import HTTPException

from modules.import_policy import ImportSettings, CSVStructureError
from modules.data_janitor import clean_dataframe
from modules.column_statistics import compute_column_statistics
from modules.intelligence import detect_semantic_type
from services.csv_ingestion import ingest_csv, _config_integer, _source_size
from services.response_builders import get_column_summary
from utils.dataframe_utils import read_csv_fast
from utils.source_files import retain_source

MAX_STAGED_IMPORTS = 4
MAX_STAGED_BYTES = 512 * 1024 * 1024
PREVIEW_BYTES = 256 * 1024
PREVIEW_ROWS = 20
IMPORT_TTL_SECONDS = 30 * 60


@dataclass
class StagedImport:
    id: str
    filename: str
    owner: object
    path: Path
    size: int
    settings: ImportSettings
    touched: float = field(default_factory=monotonic)
    busy: bool = False
    removed: bool = False
    preview_valid: bool = False


_imports: dict[str, StagedImport] = {}
_lock = RLock()


def _cleanup():
    for item in list(_imports.values()):
        if not item.busy and (item.removed or monotonic() - item.touched > IMPORT_TTL_SECONDS):
            item.owner.cleanup()
            _imports.pop(item.id, None)


def _acquire(import_id):
    with _lock:
        _cleanup()
        item = _imports.get(import_id)
        if item is None or item.removed:
            raise HTTPException(404, "Import preview expired or was cancelled. Please choose the file again.")
        if item.busy:
            raise HTTPException(409, "This import is already being processed.")
        item.busy = True
        item.touched = monotonic()
        return item


def _release(item):
    with _lock:
        item.busy = False
        _cleanup()


def _detect_settings(path):
    with path.open("rb") as handle:
        prefix = handle.read(PREVIEW_BYTES)
    encoding = "utf-16" if prefix.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    try:
        text = codecs.getincrementaldecoder(encoding)(errors="strict").decode(prefix, final=False)
    except UnicodeDecodeError:
        encoding, text = "cp1252", prefix.decode("cp1252", errors="replace")
    try:
        delimiter = csv.Sniffer().sniff(text, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    return ImportSettings(delimiter=delimiter, encoding=encoding)


def stage_import(source, filename, settings=None):
    size = _source_size(source)
    max_bytes = _config_integer("ANALYTICO_MAX_UPLOAD_BYTES", 256 * 1024 * 1024, 1)
    if size > max_bytes:
        raise HTTPException(413, "CSV exceeds the configured upload size limit.")
    with _lock:
        _cleanup()
        if len(_imports) >= MAX_STAGED_IMPORTS or sum(item.size for item in _imports.values()) + size > MAX_STAGED_BYTES:
            raise HTTPException(429, "Close another import preview before choosing another file.")
        owner, path, actual_size = retain_source(source, max_bytes)
        try:
            selected = settings or _detect_settings(path)
            item = StagedImport(str(uuid.uuid4()), Path(filename).name, owner, path, actual_size, selected)
            _imports[item.id] = item
        except Exception:
            owner.cleanup()
            raise
    try:
        return preview_import(item.id)
    except Exception:
        cancel_import(item.id)
        raise


def _sample(item):
    with item.path.open("rb") as handle:
        prefix = handle.read(PREVIEW_BYTES + 1)
    complete = len(prefix) <= PREVIEW_BYTES
    prefix = prefix[:PREVIEW_BYTES]
    # The incremental decoder tolerates only an incomplete final code unit.
    text = codecs.getincrementaldecoder(item.settings.encoding)(errors="strict").decode(prefix, final=complete)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=item.settings.delimiter, strict=True)
    rows = []
    try:
        for row in reader:
            if not row:
                continue
            rows.append(row)
            if len(rows) > PREVIEW_ROWS:
                complete = complete and reader.line_num >= len(text.splitlines())
                break
    except csv.Error:
        if complete:
            raise CSVStructureError("CSV quoting or field size is invalid in the preview.")
        # Truncation can split a quoted record. Fully decoded earlier records remain useful.
    if not rows:
        raise CSVStructureError("CSV has no readable header in the preview.")
    header, data = rows[0], rows[1:]
    if not 1 <= len(header) <= 256 or any(len(value.encode("utf-8")) > 256 for value in header):
        raise CSVStructureError("CSV header exceeds the supported column or name limits.")
    for index, row in enumerate(data, start=2):
        if len(row) != len(header):
            raise CSVStructureError(f"CSV record {index} has a different number of fields than its header. Check the delimiter.")
    if not complete and data and not text.endswith(("\n", "\r")) and len(rows) <= PREVIEW_ROWS:
        data = data[:-1]
    if not data:
        raise CSVStructureError("No complete data rows fit in the preview. Check the file format or field sizes.")
    return header, data, complete


def preview_import(import_id, settings=None):
    item = _acquire(import_id)
    try:
        if settings is not None:
            item.settings = settings
        item.preview_valid = False
        try:
            header, rows, complete = _sample(item)
            sample_csv = io.StringIO(newline="")
            writer = csv.writer(sample_csv, delimiter=item.settings.delimiter)
            writer.writerow(header)
            writer.writerows(rows)
            sample_csv.seek(0)
            raw = read_csv_fast(sample_csv, settings=item.settings)
            frame, actions, _, formats, roles = clean_dataframe(
                raw, interpret_columns=False, import_settings=item.settings,
            )
            stats = compute_column_statistics(frame)
            columns = [get_column_summary(frame, column, roles.get(column) or detect_semantic_type(frame, column, stats),
                                          formats.get(column, "general"), stats).model_dump() for column in frame.columns]
            for index, column in enumerate(columns):
                column["original_name"] = header[index][:256]
            # Source and parsed samples are intentionally capped; entire-file validity is not claimed.
            parsed = [{column: None if pd.isna(value) else str(value)[:160]
                       for column, value in record.items()} for record in frame.head(PREVIEW_ROWS).to_dict("records")]
            warnings = ["Preview covers a sample. The entire file will be validated after confirmation.", *actions[:20]]
            if len(header) == 1:
                warnings.append("One column detected. Confirm this is intentional or change the delimiter.")
            if len(set(header)) != len(header):
                warnings.append("Duplicate column names will receive distinct normalized names.")
            item.preview_valid = True
            return dict(import_id=item.id, filename=item.filename, settings=item.settings.model_dump(), columns=columns,
                        raw_rows=[[value[:160] for value in row] for row in rows], parsed_rows=parsed,
                        warnings=warnings, sample_row_count=len(rows), sample_complete=complete, can_confirm=True)
        except (CSVStructureError, UnicodeDecodeError, ValueError) as error:
            message = str(error) if isinstance(error, CSVStructureError) else "The sample cannot be parsed with these settings. Check encoding and number/date conventions."
            return dict(import_id=item.id, filename=item.filename, settings=item.settings.model_dump(), columns=[],
                        raw_rows=[], parsed_rows=[], warnings=[message], sample_row_count=0,
                        sample_complete=False, can_confirm=False)
    finally:
        _release(item)


def confirm_import(import_id, settings=None, column_overrides=None):
    item = _acquire(import_id)
    try:
        if not item.preview_valid or (settings is not None and settings != item.settings):
            raise HTTPException(409, "Refresh the preview with these settings before confirming.")
        response = ingest_csv(item.path, item.filename, "/imports/confirm", import_settings=item.settings,
                              column_overrides=column_overrides or [])
        item.removed = True
        return response
    except CSVStructureError as error:
        raise HTTPException(413 if error.limit_exceeded else 400, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, "The full file does not satisfy the confirmed parsing policy. Review its formats and column settings.") from error
    finally:
        _release(item)


def cancel_import(import_id):
    with _lock:
        item = _imports.get(import_id)
        if item is not None:
            item.removed = True
        _cleanup()
    return {"cancelled": True}
