"""Shared, deterministic policy for reading and interpreting imported CSV data."""

from __future__ import annotations

import csv
import io
import math
import re
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Literal, Optional, TextIO

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Delimiter = Literal[",", ";", "\t", "|"]
Encoding = Literal["utf-8-sig", "utf-8", "cp1252", "utf-16"]
DecimalSeparator = Literal["auto", ".", ","]
GroupingSeparator = Literal[",", ".", " "]
DateOrder = Literal["auto", "ymd", "dmy", "mdy"]
ParseAs = Literal["auto", "text", "number", "date"]
ColumnRole = Literal["metric", "identifier", "temporal", "categorical", "unknown"]
Aggregation = Literal["sum", "mean", "count", "none"]
ColumnFormat = Literal["currency", "percentage", "number", "date", "identifier", "general"]


class ImportSettings(BaseModel):
    """User-selected CSV syntax and locale settings shared by import engines."""

    model_config = ConfigDict(extra="forbid")

    delimiter: Delimiter = ","
    encoding: Encoding = "utf-8-sig"
    decimal_separator: DecimalSeparator = "auto"
    grouping_separator: Optional[GroupingSeparator] = None
    date_order: DateOrder = "auto"
    null_values: list[str] = Field(default_factory=lambda: [""], max_length=32)

    @field_validator("null_values")
    @classmethod
    def validate_null_values(cls, values: list[str]) -> list[str]:
        if any(not isinstance(value, str) or len(value) > 128 for value in values):
            raise ValueError("Each null token must be a string of at most 128 characters.")
        return values

    @model_validator(mode="after")
    def validate_separators(self) -> "ImportSettings":
        if self.grouping_separator is not None and self.decimal_separator == "auto":
            raise ValueError("Choose an explicit decimal separator when specifying a grouping separator.")
        if (
            self.decimal_separator != "auto"
            and self.grouping_separator is not None
            and self.decimal_separator == self.grouping_separator
        ):
            raise ValueError("Decimal and grouping separators must be different.")
        return self


def normalize_column_name(value: str) -> str:
    """Normalize a header to the same deterministic key used for override lookup."""
    text = str(value).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^\w]+", "_", text.lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "column"


class ColumnOverride(BaseModel):
    """Explicit parsing and semantic metadata for one normalized column key."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: str = Field(min_length=1, max_length=256)
    parse_as: ParseAs = "auto"
    role: Optional[ColumnRole] = None
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    unit: Optional[str] = Field(default=None, max_length=64)
    aggregation: Optional[Aggregation] = None
    format: Optional[ColumnFormat] = None

    @field_validator("column")
    @classmethod
    def normalize_column(cls, value: str) -> str:
        return normalize_column_name(value)

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and any(unicodedata.category(c).startswith("C") for c in value):
            raise ValueError("Display names cannot contain control characters.")
        return value

    @field_validator("unit")
    @classmethod
    def normalize_unit(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Unit must not be empty when supplied.")
        return normalized


class CSVStructureError(ValueError):
    """A safe structural CSV error with row context and no source cell contents."""

    def __init__(
        self,
        message: str,
        *,
        row_number: Optional[int] = None,
        limit_exceeded: bool = False,
    ) -> None:
        super().__init__(message)
        self.row_number = row_number
        self.limit_exceeded = limit_exceeded


MAX_CSV_FIELD_CHARS = 256 * 1024
MAX_HEADER_BYTES = 256
# The standard library defaults to 128 Ki characters. Set a documented cap
# once so valid long text fields have room while pathological fields are bounded.
csv.field_size_limit(MAX_CSV_FIELD_CHARS)


def _coerce_settings(settings: Optional[ImportSettings]) -> ImportSettings:
    if settings is None:
        return ImportSettings()
    if isinstance(settings, ImportSettings):
        return settings
    return ImportSettings.model_validate(settings)


def reader_options(settings: Optional[ImportSettings] = None) -> dict:
    """Return pandas reader kwargs that preserve source lexemes and configured nulls."""
    policy = _coerce_settings(settings)
    return {
        "sep": policy.delimiter,
        "encoding": policy.encoding,
        "encoding_errors": "strict",
        "engine": "c",
        "dtype": str,
        "keep_default_na": False,
        "na_values": list(policy.null_values),
        "low_memory": False,
    }


@contextmanager
def _text_csv_source(source, encoding: str):
    """Yield a CSV text stream from paths, content, or seekable file objects."""
    if isinstance(source, Path):
        try:
            with source.open("r", encoding=encoding, newline="") as stream:
                yield stream
        except UnicodeError as error:
            raise CSVStructureError(f"CSV encoding could not be decoded as {encoding}.") from error
        return

    if isinstance(source, str):
        path = None
        if "\n" not in source and "\r" not in source and len(source) < 4096:
            try:
                candidate = Path(source)
                if candidate.is_file():
                    path = candidate
            except OSError:
                path = None
        if path is not None:
            with _text_csv_source(path, encoding) as stream:
                yield stream
            return
        if "\n" in source or "\r" in source or "," in source or "\t" in source or ";" in source or "|" in source:
            yield io.StringIO(source, newline="")
            return
        raise CSVStructureError("CSV source path could not be read.")

    if isinstance(source, (bytes, bytearray)):
        try:
            with io.TextIOWrapper(io.BytesIO(source), encoding=encoding, newline="") as stream:
                yield stream
        except UnicodeError as error:
            raise CSVStructureError(f"CSV encoding could not be decoded as {encoding}.") from error
        return

    if not hasattr(source, "read"):
        raise TypeError("CSV source must be a path, bytes, or readable file object.")

    try:
        original_position = source.tell()
        source.seek(0)
    except (AttributeError, OSError, ValueError) as error:
        raise CSVStructureError("CSV input must support bounded file access.") from error

    wrapped = None
    try:
        if isinstance(source, (io.TextIOBase, io.StringIO)):
            stream = source
        else:
            wrapped = io.TextIOWrapper(source, encoding=encoding, newline="")
            stream = wrapped
        yield stream
    except UnicodeError as error:
        raise CSVStructureError(f"CSV encoding could not be decoded as {encoding}.") from error
    finally:
        if wrapped is not None:
            try:
                wrapped.detach()
            except (OSError, ValueError):
                pass
        try:
            source.seek(original_position)
        except (OSError, ValueError):
            pass


def validate_csv_structure(
    source,
    settings: Optional[ImportSettings] = None,
    *,
    max_rows: int = 5_000_000,
    max_columns: int = 256,
    max_header_bytes: int = MAX_HEADER_BYTES,
) -> tuple[int, int]:
    """Stream-validate the complete CSV and return ``(data_rows, columns)``.

    Blank physical records are ignored like pandas' default ``skip_blank_lines``.
    Errors report only a record or physical line number, never cell contents.
    Seekable input streams are read from byte/text position zero and restored.
    """
    policy = _coerce_settings(settings)
    if max_rows < 0 or max_columns < 1 or max_header_bytes < 1:
        raise ValueError("CSV row and column limits must be nonnegative and positive.")

    rows = 0
    with _text_csv_source(source, policy.encoding) as stream:
        reader = csv.reader(stream, delimiter=policy.delimiter, strict=True)
        try:
            headers = next(reader, None)
            while headers == []:
                headers = next(reader, None)
            if headers is None:
                headers = []
        except csv.Error as error:
            raise CSVStructureError(f"Malformed CSV near line {max(1, reader.line_num)}.",
                                    row_number=1) from error
        except UnicodeError as error:
            raise CSVStructureError(f"CSV encoding could not be decoded as {policy.encoding}.") from error

        width = len(headers)
        if width == 0:
            raise CSVStructureError("CSV header has no columns.", row_number=1)
        if width > max_columns:
            raise CSVStructureError(f"CSV exceeds the {max_columns}-column limit.", row_number=1,
                                    limit_exceeded=True)
        if any(len(header.encode("utf-8")) > max_header_bytes for header in headers):
            raise CSVStructureError(f"CSV header names exceed the {max_header_bytes}-byte limit.",
                                    row_number=1, limit_exceeded=True)
        if any("\x00" in header for header in headers):
            raise CSVStructureError("CSV contains a NUL byte in the header.", row_number=1)

        try:
            for row in reader:
                if not row:
                    continue
                rows += 1
                record_number = rows + 1
                if rows > max_rows:
                    raise CSVStructureError(f"CSV exceeds the {max_rows}-row limit.",
                                            row_number=record_number, limit_exceeded=True)
                if any("\x00" in value for value in row):
                    raise CSVStructureError(f"CSV contains a NUL byte near row {record_number}.",
                                            row_number=record_number)
                if len(row) != width:
                    detail = "extra fields" if len(row) > width else "missing fields"
                    raise CSVStructureError(
                        f"CSV row {record_number} has {detail}; expected {width} columns.",
                        row_number=record_number,
                    )
        except csv.Error as error:
            row_number = rows + 2
            detail = "exceeds the 262144-character field limit" if "field larger than field limit" in str(error) else "is malformed"
            raise CSVStructureError(
                f"CSV {detail} near line {max(1, reader.line_num)} (row {row_number}).",
                row_number=row_number,
                limit_exceeded="field larger than field limit" in str(error),
            ) from error
        except UnicodeError as error:
            raise CSVStructureError(f"CSV encoding could not be decoded as {policy.encoding}.") from error

    return rows, width


def read_csv_headers(source, settings: Optional[ImportSettings] = None) -> list[str]:
    """Read the source header while preserving a seekable stream's position."""
    policy = _coerce_settings(settings)
    with _text_csv_source(source, policy.encoding) as stream:
        reader = csv.reader(stream, delimiter=policy.delimiter, strict=True)
        try:
            headers = next(reader, None)
            while headers == []:
                headers = next(reader, None)
            if headers is None:
                raise CSVStructureError("CSV is empty.")
            return headers
        except csv.Error as error:
            raise CSVStructureError("Malformed CSV header.", row_number=1) from error
        except UnicodeError as error:
            raise CSVStructureError(f"CSV encoding could not be decoded as {policy.encoding}.") from error


def _overrides_by_column(
    column_overrides: Optional[list[ColumnOverride] | tuple[ColumnOverride, ...]],
) -> dict[str, ColumnOverride]:
    result: dict[str, ColumnOverride] = {}
    for raw_override in column_overrides or ():
        override = raw_override if isinstance(raw_override, ColumnOverride) else ColumnOverride.model_validate(raw_override)
        if override.column in result:
            raise ValueError(f"More than one override targets column '{override.column}'.")
        result[override.column] = override
    return result


def resolve_column_policy(
    column: str,
    settings: Optional[ImportSettings] = None,
    column_overrides: Optional[list[ColumnOverride] | tuple[ColumnOverride, ...]] = None,
) -> dict:
    """Resolve settings and one exact normalized-column override into a small policy."""
    policy = _coerce_settings(settings)
    override = _overrides_by_column(column_overrides).get(normalize_column_name(column))
    values = {
        "column": normalize_column_name(column),
        "parse_as": override.parse_as if override else "auto",
        "role": override.role if override else None,
        "display_name": override.display_name if override else None,
        "unit": override.unit if override else None,
        "aggregation": override.aggregation if override else None,
        "format": override.format if override else None,
        "delimiter": policy.delimiter,
        "encoding": policy.encoding,
        "decimal_separator": policy.decimal_separator,
        "grouping_separator": policy.grouping_separator,
        "date_order": policy.date_order,
        "null_values": list(policy.null_values),
    }
    values["provenance"] = {
        key: "user" if override is not None and getattr(override, key) is not None else "default"
        for key in ("parse_as", "role", "unit", "aggregation", "format")
    }
    return values


_PLAIN_DOT_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_INTEGER_NUMBER = re.compile(r"^[+-]?\d+$")
_GROUPED_NUMBER = re.compile(r"^[+-]?\d{1,3}(?:GROUP\d{3})+(?:DECIMAL\d*)?$")
_TIMEZONE_SUFFIX = re.compile(r"[Tt ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})$")
_SUBMICROSECOND = re.compile(r"[Tt ]\d{2}:\d{2}:\d{2}\.\d{7,}")
_LEADING_ZERO_IDENTIFIER = re.compile(r"^[+-]?0\d+$")
_IDENTIFIER_TOKENS = {
    "id", "code", "zip", "zipcode", "postal", "phone", "key", "identifier",
    "ssn", "account", "serial", "reference", "ref", "sequence",
}


def _looks_like_identifier(series: pd.Series, column: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", column.lower()))
    pieces = re.findall(r"[a-z0-9]+", column.lower())
    if tokens & _IDENTIFIER_TOKENS or any(left == "i" and right == "d" for left, right in zip(pieces, pieces[1:])):
        return True
    return bool(series.dropna().map(lambda value: bool(_LEADING_ZERO_IDENTIFIER.fullmatch(str(value).strip()))).any())


def _integer_series(values: list[Optional[int]], index: pd.Index) -> pd.Series:
    numbers = [value for value in values if value is not None]
    minimum = min(numbers) if numbers else 0
    maximum = max(numbers) if numbers else 0
    if minimum < -(2**63) or maximum > 2**64 - 1:
        raise ValueError("integer is outside pandas' lossless 64-bit range")
    unsigned = minimum >= 0 and maximum > 2**63 - 1
    kind = "UInt64" if unsigned else "Int64"
    if any(value is None for value in values):
        return pd.Series(pd.array(values, dtype=kind), index=index)
    dtype = np.uint64 if unsigned else np.int64
    return pd.Series(np.asarray(numbers, dtype=dtype), index=index)


def _normalized_numeric_token(
    token: str,
    decimal_separator: str,
    grouping_separator: Optional[str],
) -> Optional[str]:
    if not token or token.strip() != token:
        return None
    sign = ""
    body = token
    if body[:1] in {"+", "-"}:
        sign, body = body[0], body[1:]
    if not body:
        return None

    if decimal_separator == "auto":
        if _PLAIN_DOT_NUMBER.fullmatch(token):
            return token
        if grouping_separator and grouping_separator in body:
            pattern = re.escape(grouping_separator)
            integer_part, separator, fraction_part = body.partition(".")
            if separator and body.count(".") > 1:
                return None
            if not re.fullmatch(rf"\d{{1,3}}(?:{pattern}\d{{3}})+", integer_part):
                return None
            if separator and not fraction_part.isdigit():
                return None
            return sign + integer_part.replace(grouping_separator, "") + ("." + fraction_part if separator else "")
        # Automatic settings retain the familiar unambiguous US grouped form.
        if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+\.\d+", token):
            return token.replace(",", "")
        return None

    if body.count(decimal_separator) > 1:
        return None
    integer_part, separator, fraction_part = body.partition(decimal_separator)
    if separator and (not fraction_part or not fraction_part.isdigit()):
        return None
    if not integer_part:
        return None
    if grouping_separator is not None and grouping_separator in integer_part:
        group_pattern = re.escape(grouping_separator)
        if not re.fullmatch(rf"\d{{1,3}}(?:{group_pattern}\d{{3}})+", integer_part):
            return None
        integer_part = integer_part.replace(grouping_separator, "")
    elif not integer_part.isdigit():
        return None
    if not integer_part.isdigit():
        return None
    normalized = sign + integer_part
    if separator:
        normalized += "." + fraction_part
    return normalized


def parse_locale_numbers(
    series: pd.Series,
    decimal_separator: DecimalSeparator = "auto",
    grouping_separator: Optional[GroupingSeparator] = None,
) -> Optional[pd.Series]:
    """Parse one complete numeric column under strict separator rules.

    Returns ``None`` unless every nonnull lexeme is valid. Source null positions
    and the original index are preserved. Integer-only columns keep exact 64-bit
    integer values, including nullable signed and unsigned values.
    """
    present = series.notna().to_numpy()
    if not present.any():
        return None
    source_values = series.iloc[np.flatnonzero(present)].map(lambda value: str(value).strip())
    if source_values.empty or source_values.eq("").any():
        return None

    # Auto is deliberately conservative: unlike explicit locale settings, it
    # accepts plain decimals and US grouping only when both marks disambiguate it.
    normalized: list[str] = []
    for value in source_values.tolist():
        token = _normalized_numeric_token(value, decimal_separator, grouping_separator)
        if token is None:
            return None
        normalized.append(token)

    if all(_INTEGER_NUMBER.fullmatch(token) for token in normalized):
        try:
            parsed_ints = [int(token) for token in normalized]
            full: list[Optional[int]] = [None] * len(series)
            for index, value in zip(np.flatnonzero(present), parsed_ints):
                full[index] = value
            return _integer_series(full, series.index)
        except (ValueError, OverflowError):
            return None

    values: list[float] = []
    try:
        for token in normalized:
            if not _PLAIN_DOT_NUMBER.fullmatch(token):
                return None
            parsed = float(token)
            if not math.isfinite(parsed):
                return None
            # Decimal string round-tripping rejects values whose significant
            # decimal digits cannot survive pandas' float representation.
            from decimal import Decimal, InvalidOperation

            if Decimal(str(parsed)) != Decimal(token):
                return None
            values.append(parsed)
    except (ValueError, OverflowError, InvalidOperation):
        return None

    result = np.full(len(series), np.nan, dtype=np.float64)
    result[np.flatnonzero(present)] = values
    return pd.Series(result, index=series.index)


_AUTO_DATE_FORMATS = (
    "ISO8601", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d", "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y", "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f", "%Y %b %d %I:%M:%S %p", "%d %b %Y %H:%M:%S",
    "%d %b %Y", "%b %d, %Y", "%b %d %Y %H:%M:%S", "%Y%m%d",
)
_ORDER_DATE_FORMATS = {
    "ymd": ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"),
    "dmy": ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S"),
    "mdy": ("%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%m/%d/%Y %H:%M:%S", "%m-%d-%Y %H:%M:%S"),
}


def _has_unsupported_timestamp_precision(values: pd.Series) -> bool:
    return bool(values.map(lambda value: bool(_SUBMICROSECOND.search(value))).any())


def parse_policy_dates(
    series: pd.Series,
    date_order: DateOrder = "auto",
    *,
    force: bool = False,
    column: str = "column",
) -> Optional[pd.Series]:
    """Parse dates only under a unique or user-selected date order.

    Automatic parsing preserves timezone-aware and sub-microsecond timestamp
    lexemes as text. Forced parsing raises a safe, column-scoped ``ValueError``.
    """
    present = series.notna().to_numpy()
    if not present.any():
        if force:
            raise ValueError(f"Column '{column}' has no non-null values to parse as dates.")
        return None
    values = series.iloc[np.flatnonzero(present)].map(lambda value: str(value).strip())
    unsupported = values.map(
        lambda value: bool(_TIMEZONE_SUFFIX.search(value) or _SUBMICROSECOND.search(value))
    )
    if unsupported.any():
        if force:
            reason = "timezone-aware timestamps" if values[unsupported].map(
                lambda value: bool(_TIMEZONE_SUFFIX.search(value))
            ).any() else "timestamps with more than six fractional digits"
            raise ValueError(f"Column '{column}' contains {reason} that cannot be represented safely as pandas dates.")
        return None

    formats = policy_date_formats(date_order)
    candidates: list[pd.Series] = []
    for fmt in formats:
        try:
            parsed = pd.to_datetime(values, format=fmt, errors="coerce")
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed.notna().all():
            if not any(parsed.equals(candidate) for candidate in candidates):
                candidates.append(parsed)
    if len(candidates) != 1:
        if force:
            reason = "ambiguous date order" if len(candidates) > 1 else "values that do not match the selected date order"
            raise ValueError(f"Column '{column}' contains {reason}; choose a compatible date order or retain the source values.")
        return None

    parsed_values = candidates[0]
    output = np.full(len(series), np.datetime64("NaT"), dtype="datetime64[ns]")
    output[np.flatnonzero(present)] = parsed_values.to_numpy(dtype="datetime64[ns]")
    return pd.Series(output, index=series.index)


def policy_date_formats(date_order: DateOrder = "auto") -> tuple[str, ...]:
    """Return the deterministic formats considered by the shared date parser."""
    return _ORDER_DATE_FORMATS.get(date_order, _AUTO_DATE_FORMATS)


def unsupported_date_reason(series: pd.Series) -> Optional[str]:
    """Return why timestamps should remain text under the pandas safety policy."""
    values = series.dropna().map(lambda value: str(value).strip())
    if values.map(lambda value: bool(_TIMEZONE_SUFFIX.search(value))).any():
        return "timezone-aware timestamps"
    if values.map(lambda value: bool(_SUBMICROSECOND.search(value))).any():
        return "timestamps with more than six fractional digits"
    return None


def _as_text(series: pd.Series) -> pd.Series:
    values = [pd.NA if pd.isna(value) else str(value) for value in series.tolist()]
    return pd.Series(pd.array(values, dtype="string"), index=series.index)


def prepare_numeric_source(series: pd.Series, column_format=None, unit=None, *, markers=None):
    """Normalize confirmed currency/percent syntax, retaining full-column checks.

    A disk caller retains ``markers`` across chunks to detect mixed currencies.
    This never infers a currency unit or changes the selected number locale.
    """
    seen = markers if markers is not None else set()
    percent_modes = set()
    values = []
    for raw in series.tolist():
        if pd.isna(raw):
            values.append(pd.NA)
            continue
        token = str(raw).strip()
        if column_format == "percentage":
            percent = token.endswith("%")
            percent_modes.add(percent)
            if percent:
                token = token[:-1].strip()
        if column_format == "currency":
            symbols = {char for char in token if unicodedata.category(char) == "Sc"}
            seen.update(("symbol", symbol) for symbol in symbols)
            code_match = re.match(r"^([A-Z]{3})\s+", token)
            if code_match:
                code = code_match.group(1)
                seen.add(("code", code))
                if unit and unit.upper() != code:
                    raise ValueError("Currency code conflicts with the confirmed unit.")
                token = token[code_match.end():]
            if (unit and unit.upper() == "USD" and symbols - {"$"}) or (unit and unit.upper() == "EUR" and symbols - {"€"}):
                raise ValueError("Currency symbol conflicts with the confirmed unit.")
            if any(len({marker for kind, marker in seen if kind == category}) > 1 for category in ("symbol", "code")):
                raise ValueError("Column contains mixed currency identities.")
            codes = {marker for kind, marker in seen if kind == "code"}
            symbols_seen = {marker for kind, marker in seen if kind == "symbol"}
            known_symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
            if codes and symbols_seen and (len(codes) != 1 or known_symbols.get(next(iter(codes))) not in symbols_seen):
                raise ValueError("Column contains conflicting currency codes and symbols.")
            token = "".join(char for char in token if unicodedata.category(char) != "Sc").strip()
        values.append(token)
    if len(percent_modes) > 1:
        raise ValueError("Column mixes percent suffixes and plain numbers.")
    return pd.Series(values, index=series.index, dtype=object), next(iter(percent_modes), False)


def apply_column_policy(
    df: pd.DataFrame,
    settings: Optional[ImportSettings] = None,
    column_overrides: Optional[list[ColumnOverride] | tuple[ColumnOverride, ...]] = None,
    *,
    original_names: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply explicit import choices and attach shared per-column schema metadata.

    Columns must already have their final normalized, collision-free names.
    Overrides match those complete names exactly after normalization. Explicit
    metric and temporal roles are checked against the resulting dtype.
    """
    policy = _coerce_settings(settings)
    custom_settings = settings is not None and policy != ImportSettings()
    overrides = _overrides_by_column(column_overrides)
    normalized_columns = {normalize_column_name(str(column)) for column in df.columns}
    missing = sorted(set(overrides) - normalized_columns)
    if missing:
        raise ValueError(f"Column override does not match a column: '{missing[0]}'.")
    if len(normalized_columns) != len(df.columns):
        raise ValueError("Column policy requires unique normalized column names.")

    result = df.copy(deep=False)
    schema: list[dict] = []
    for column in result.columns:
        column_name = str(column)
        resolved = resolve_column_policy(column_name, policy, column_overrides)
        parse_as = resolved["parse_as"]
        role = resolved["role"]
        column_format = resolved["format"]
        source = result[column_name]

        if role == "identifier" and parse_as in {"number", "date"}:
            raise ValueError(f"Column '{column_name}' has conflicting identifier and {parse_as} policies.")

        if parse_as == "text" or column_format == "identifier" or (
            parse_as == "auto" and role in {"identifier", "categorical"}
        ):
            result[column_name] = _as_text(source)
        elif parse_as == "date" or column_format == "date" or role == "temporal":
            if pd.api.types.is_datetime64_any_dtype(source):
                if isinstance(source.dtype, pd.DatetimeTZDtype):
                    raise ValueError(
                        f"Column '{column_name}' contains timezone-aware timestamps that cannot be represented safely as pandas dates."
                    )
                parsed = source
            else:
                parsed = parse_policy_dates(source, policy.date_order, force=True, column=column_name)
            result[column_name] = parsed
        elif parse_as == "number" or role == "metric" or column_format in {"number", "currency", "percentage"}:
            if pd.api.types.is_numeric_dtype(source):
                finite = source.dropna().map(lambda value: math.isfinite(float(value)))
                if not finite.all():
                    raise ValueError(f"Column '{column_name}' contains non-finite numbers.")
                parsed = source
            else:
                numeric_source, is_percent_text = prepare_numeric_source(
                    source, column_format, resolved["unit"],
                )
                parsed = parse_locale_numbers(
                    numeric_source, policy.decimal_separator, policy.grouping_separator
                )
                if parsed is None:
                    raise ValueError(f"Column '{column_name}' contains values that cannot be parsed safely as numbers.")
                if column_format == "percentage" and is_percent_text:
                    if pd.api.types.is_integer_dtype(parsed.dtype) and any(abs(int(value)) > 2**53 - 1 for value in parsed.dropna()):
                        raise ValueError(f"Column '{column_name}' would lose integer precision when converted to percentages.")
                    parsed = parsed / 100
            if parsed is None:
                raise ValueError(f"Column '{column_name}' contains values that cannot be parsed safely as numbers.")
            result[column_name] = parsed
        elif (
            policy.decimal_separator != "auto"
            and parse_as == "auto"
            and role is None
            and column_format is None
            and _looks_like_identifier(source, column_name)
        ):
            # Global locale choices do not erase a likely identifier's source lexeme.
            result[column_name] = _as_text(source)
        else:
            is_text = pd.api.types.is_object_dtype(source) or pd.api.types.is_string_dtype(source)
            if is_text and policy.decimal_separator != "auto":
                parsed = parse_locale_numbers(source, policy.decimal_separator, policy.grouping_separator)
                if parsed is not None:
                    result[column_name] = parsed
            if is_text and policy.date_order != "auto":
                parsed = parse_policy_dates(source, policy.date_order, force=False, column=column_name)
                if parsed is not None:
                    result[column_name] = parsed

        if role == "metric" and not pd.api.types.is_numeric_dtype(result[column_name]):
            raise ValueError(f"Column '{column_name}' has role 'metric' but does not contain safely parsed numbers.")
        if role == "temporal" and not pd.api.types.is_datetime64_any_dtype(result[column_name]):
            raise ValueError(f"Column '{column_name}' has role 'temporal' but does not contain safely parsed dates.")
        if column_format in {"number", "currency", "percentage"} and not pd.api.types.is_numeric_dtype(result[column_name]):
            raise ValueError(f"Column '{column_name}' has format '{column_format}' but does not contain safely parsed numbers.")
        if column_format == "date" and not pd.api.types.is_datetime64_any_dtype(result[column_name]):
            raise ValueError(f"Column '{column_name}' has format 'date' but does not contain safely parsed dates.")

        has_override = any(resolved[key] is not None for key in ("role", "unit", "aggregation", "format")) or resolved["provenance"]["parse_as"] == "user"
        provenance = "override" if has_override else "import_settings" if custom_settings else "inference"
        item = {
            "column": column_name,
            "original_name": (original_names or {}).get(column_name, column_name),
            "display_name": resolved["display_name"],
            "display_name_provenance": "user" if resolved["display_name"] else None,
            "parse_as": parse_as,
            "role": role,
            "format": column_format,
            "unit": resolved["unit"],
            "aggregation": resolved["aggregation"],
            "provenance": provenance,
            "status": "confirmed" if has_override else "suggested",
        }
        schema.append(item)

    result.attrs["column_schema"] = schema
    return result, schema
