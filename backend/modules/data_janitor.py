"""
Analytico Backend - Data Janitor Module
Smart ingestion, header normalization, type repair, and data cleaning
"""

from modules.column_statistics import MIN_CALENDAR_YEAR, MAX_CALENDAR_YEAR

import os
import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation, localcontext
from time import perf_counter
from typing import Callable, Optional

import numpy as np
import pandas as pd

from modules.import_policy import (
    ColumnOverride,
    ImportSettings,
    apply_column_policy,
    unsupported_date_reason,
)

# Master list of candidate date formats for fast C-vectorized parsing.
# Order matters: most common/high-signal formats are first.
DATE_FORMAT_CANDIDATES = [
    "ISO8601",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y %b %d %I:%M:%S %p",
    "%d %b %Y %H:%M:%S",
    "%d %b %Y",
    "%b %d, %Y",
    "%b %d %Y %H:%M:%S",
    "%Y%m%d",
]

UNIVERSAL_CURRENCY_KEYWORDS = {
    "amount", "price", "cost", "revenue", "income", "salary", "expense",
    "value", "total", "balance", "payment",
}
IDENTIFIER_HINT_KEYWORDS = {
    "id", "code", "zip", "zipcode", "postal", "phone", "key", "identifier",
    "ssn", "account", "serial", "reference", "ref", "sequence",
}
_MAX_DISTINCT_TEXT_VALUES = 10_000
_DISTINCT_TEXT_CHUNK_SIZE = 16_384
_LEADING_ZERO_IDENTIFIER = re.compile(r"^[+-]?0\d+$")


def _distinct_text_values(values: pd.Series) -> Optional[dict[str, None]]:
    """Collect distinct lexemes up to a fixed cap, or signal high cardinality."""
    unique: dict[str, None] = {}
    for start in range(0, len(values), _DISTINCT_TEXT_CHUNK_SIZE):
        for value in values.iloc[start:start + _DISTINCT_TEXT_CHUNK_SIZE].unique():
            unique.setdefault(value, None)
            if len(unique) > _MAX_DISTINCT_TEXT_VALUES:
                return None
    return unique


def _map_distinct_text(values: pd.Series, mapper: Callable[[str], object]) -> pd.Series:
    """Map repeated text once per lexeme, falling back for high-cardinality data."""
    unique = _distinct_text_values(values)
    if unique is None:
        return values.map(mapper)
    mapping = {value: mapper(value) for value in unique}
    return values.map(mapping)


def legacy_normalize_header(header: str) -> str:
    """Conservative, deterministic snake_case sanitizer."""
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", header.strip())
    clean = re.sub(r"[^\w]+", "_", value.lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "column"


def _unique_normalized_headers(headers: list[str]) -> list[str]:
    """Normalize headers with collision-free suffixes, independent of AI output."""
    bases = [legacy_normalize_header(str(header)) for header in headers]
    reserved = set(bases)
    used: set[str] = set()
    result: list[str] = []

    for base in bases:
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
            while candidate in used or candidate in reserved:
                candidate = f"{base}_{suffix}"
                suffix += 1
        used.add(candidate)
        result.append(candidate)
    return result


def _column_tokens(col_name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(col_name).lower()))


def _has_split_id_token(col_name: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", str(col_name).lower())
    entity_names = {"customer", "account", "person", "user", "record", "order", "invoice",
                    "product", "employee", "transaction", "item", "row", "entity"}
    if any(token == entity + "id" for token in tokens for entity in entity_names):
        return True
    return any(left == "i" and right == "d" for left, right in zip(tokens, tokens[1:]))


def _is_identifier_name(col_name: str) -> bool:
    return bool(_column_tokens(col_name) & IDENTIFIER_HINT_KEYWORDS) or _has_split_id_token(col_name)


def _is_date_name(col_name: str) -> bool:
    return bool(_column_tokens(col_name) & {"date", "time", "datetime", "timestamp", "year"})


def _has_leading_zero_identifiers(series: pd.Series) -> bool:
    values = _map_distinct_text(series.dropna().astype(str), lambda value: value.strip())
    return bool(_map_distinct_text(
        values, lambda value: bool(_LEADING_ZERO_IDENTIFIER.match(value))
    ).any())


def _normalize_llm_format(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    fmt = str(value).strip().lower()
    allowed = {"currency", "percentage", "number", "date"}
    return fmt if fmt in allowed else None


def _normalize_llm_semantic(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    sem = str(value).strip().lower()
    allowed = {"metric", "identifier", "temporal", "categorical"}
    return sem if sem in allowed else None


def _interpretation_input(header: str, series: pd.Series, headers: list[str]) -> dict:
    """Sample throughout the column, preserving nulls and bounding outbound data."""
    count = len(series)
    positions = sorted({round(i * (count - 1) / 11) for i in range(12)}) if count else []
    values = []
    for position in positions:
        value = series.iloc[position]
        if pd.isna(value):
            values.append(None)
        elif isinstance(value, (int, float, bool)):
            values.append(value if not isinstance(value, float) or math.isfinite(value) else None)
        else:
            values.append(str(value)[:160])
    return {
        "column_name": header[:256],
        "values": values,
        "context": {
            "purpose": "Conservative dataset ingestion; no requested calculation or external unit metadata.",
            "row_count": count,
            "missing_count": int(series.isna().sum()),
            "sample_unique_count": len({str(v) for v in values if v is not None}),
            "sample_is_complete": count <= 12,
            "other_column_names": [name[:128] for name in headers if name != header][:20],
        },
    }


def llm_enrich_columns(headers: list[str], df: pd.DataFrame) -> list[dict]:
    """Optional bounded interpretation; source keys never depend on model output."""
    rows = [{"original": h, "clean": legacy_normalize_header(h), "format": None,
             "semantic_type": None} for h in headers]
    provider = os.getenv("COLUMN_INTERPRETER", "off").strip().lower()
    if provider == "off":
        return rows
    from modules.column_interpretation import interpret_column

    start = perf_counter()
    unavailable = False
    for index, (header, row) in enumerate(zip(headers, rows)):
        if index >= 12 or perf_counter() - start >= 12 or unavailable:
            row["interpretation"] = {"status": "skipped", "runtime_status": "clarification",
                                     "provider": provider, "decision": None}
            continue
        result = interpret_column(_interpretation_input(header, df.iloc[:, index], headers), use_cache=True)
        decision = result.decision.model_dump() if result.decision is not None else None
        row["interpretation"] = {
            "status": result.status, "runtime_status": "clarification",
            "provider": result.provider or provider, "model": result.model,
            "prompt_version": result.prompt_version, "latency_ms": result.latency_ms,
            "usage": result.usage, "confidence": result.confidence,
            "error_code": result.error_code, "cache_hit": result.cache_hit, "decision": decision,
        }
        unavailable = result.status in {"unavailable", "disabled", "error"}
        if decision is not None:
            row["semantic_type"] = decision["role"]
    return rows


def llm_clean_headers(headers: list[str], df: pd.DataFrame) -> list[str]:
    """Smart header normalization using LLM with sample data context"""
    return _unique_normalized_headers(headers)


def detect_column_format(series: pd.Series, col_name: str) -> str:
    """Physical display format; units require source syntax or explicit metadata."""
    if _is_identifier_name(col_name):
        return "identifier"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    return "number" if pd.api.types.is_numeric_dtype(series) else "general"


def _contains_currency_symbol(value: str) -> bool:
    return any(unicodedata.category(ch) == "Sc" for ch in value)


def _currency_symbol_set(value: str) -> frozenset[str]:
    return frozenset(ch for ch in value if unicodedata.category(ch) == "Sc")


def _currency_symbols_in_values(values: pd.Series) -> set[str]:
    """Collect symbols across the full source, reusing repeated lexemes."""
    distinct = _distinct_text_values(values)
    lexemes = distinct if distinct is not None else values
    symbols: set[str] = set()
    for value in lexemes:
        symbols.update(_currency_symbol_set(value))
        if len(symbols) > 1:
            break
    return symbols


def _strip_currency_symbols(value: str) -> str:
    return "".join(ch for ch in value if unicodedata.category(ch) != "Sc").strip()



def llm_fix_data_issues(df: pd.DataFrame, column_types: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Use LLM guidance to intelligently fix data issues.
    For numeric columns: fill with mean instead of 0.
    For categorical: fill with mode or 'Unknown'.
    Returns: (fixed_df, actions_taken)
    """
    actions = []
    df = df.copy()
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count == 0:
            continue
            
        col_type = column_types.get(col, 'unknown')
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # For metrics/numeric: fill with mean
            mean_val = df[col].mean()
            if pd.notna(mean_val):
                df[col] = df[col].fillna(round(mean_val, 2))
                actions.append(f"Filled {missing_count} missing '{col}' values with mean ({mean_val:.2f})")
            else:
                df[col] = df[col].fillna(0)
                actions.append(f"Filled {missing_count} missing '{col}' values with 0")
        else:
            # For categorical/identifier: fill with mode or 'Unknown'
            mode = df[col].mode()
            if len(mode) > 0:
                df[col] = df[col].fillna(mode[0])
                actions.append(f"Filled {missing_count} missing '{col}' values with mode ('{mode[0]}')")
            else:
                df[col] = df[col].fillna('Unknown')
                actions.append(f"Filled {missing_count} missing '{col}' values with 'Unknown'")
    
    return df, actions


_PLAIN_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_US_GROUPED_NUMBER = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+\.\d+$")


def _parse_numeric_text(series: pd.Series, col: str, *, allow_metric_name: bool = False) -> tuple[Optional[pd.Series], Optional[str], Optional[str]]:
    """Parse only uniformly valid, unambiguous numeric text; otherwise retain it."""
    values = _map_distinct_text(series.dropna().astype(str), lambda value: value.strip())
    if values.empty:
        return None, None, None

    if (_is_identifier_name(col) and not allow_metric_name) or _has_leading_zero_identifiers(values):
        return None, None, None

    percent_values = _map_distinct_text(values, lambda value: value.endswith("%"))
    if percent_values.all():
        normalized = _map_distinct_text(values, lambda value: value[:-1].strip())
        if _map_distinct_text(
            normalized, lambda value: bool(_PLAIN_NUMBER.fullmatch(value))
        ).all():
            parsed = pd.to_numeric(normalized, errors="raise") / 100
            return parsed, "percentage", f"Converted '{col}' from percentage text to decimal"
        return None, None, f"Could not safely parse '{col}' percentage text; retained source values"

    # Plain numeric tokens cannot contain currency symbols. Validate them before
    # the per-character Unicode currency scan used for decorated values.
    if _map_distinct_text(
        values, lambda value: bool(_PLAIN_NUMBER.fullmatch(value))
    ).all():
        parsed = pd.to_numeric(values, errors="raise")
        return parsed, None, f"Converted '{col}' from numeric text to numeric"

    has_currency_symbol = _map_distinct_text(values, _contains_currency_symbol)
    normalized = _map_distinct_text(values, _strip_currency_symbols)
    if has_currency_symbol.any():
        if len(_currency_symbols_in_values(values)) > 1:
            return None, None, (
                f"Could not safely parse '{col}' mixed currency text: multiple currency "
                "symbols; retained source values for review"
            )
        if _PLAIN_NUMBER.fullmatch(normalized.iloc[0]) and _map_distinct_text(
            normalized, lambda value: bool(_PLAIN_NUMBER.fullmatch(value))
        ).all():
            parsed = pd.to_numeric(normalized, errors="raise")
            return parsed, "currency", f"Converted '{col}' from currency text to numeric"
        if _map_distinct_text(
            normalized, lambda value: bool(_US_GROUPED_NUMBER.fullmatch(value))
        ).all():
            normalized = _map_distinct_text(normalized, lambda value: value.replace(",", ""))
            parsed = pd.to_numeric(normalized, errors="raise")
            return parsed, "currency", f"Converted '{col}' from currency text to numeric"
        return None, None, f"Could not safely parse '{col}' currency text; retained source values"

    if _map_distinct_text(
        normalized, lambda value: bool(_PLAIN_NUMBER.fullmatch(value))
    ).all():
        parsed = pd.to_numeric(normalized, errors="raise")
        return parsed, None, f"Converted '{col}' from numeric text to numeric"

    has_comma = _map_distinct_text(normalized, lambda value: "," in value)
    if has_comma.any():
        # A comma without a decimal point can mean grouping or decimal notation.
        if _map_distinct_text(
            normalized,
            lambda value: bool(re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", value)),
        ).all():
            return None, None, f"Could not safely parse '{col}' numeric text: ambiguous locale separators; retained source values"
        # Mixed plain/grouped values are safe only with decimal-point evidence
        # and valid three-digit grouping throughout the complete column.
        valid = _map_distinct_text(normalized, lambda value: bool(
            _PLAIN_NUMBER.fullmatch(value) or _US_GROUPED_NUMBER.fullmatch(value)
            or re.fullmatch(r"[+-]?[1-9][0-9]{0,2}(?:,[0-9]{3})+", value)
        ))
        if valid.all() and normalized.map(lambda value: "." in value).any():
            normalized = _map_distinct_text(normalized, lambda value: value.replace(",", ""))
            if normalized.map(lambda value: value.lstrip("+-").isdigit() and abs(int(value)) > 2**53 - 1).any():
                return None, None, f"Could not safely parse '{col}' grouped values without integer precision loss; retained source values"
            parsed = pd.to_numeric(normalized, errors="raise")
            return parsed, None, f"Converted '{col}' from grouped numeric text to numeric"

    numeric_like = _map_distinct_text(
        normalized,
        lambda value: bool(re.fullmatch(r"[+-]?[\d.,\s]+", value))
    )
    if numeric_like.any():
        return None, None, f"Could not safely parse '{col}' numeric text; retained source values"
    if not _is_date_name(col) and _column_tokens(col) & (UNIVERSAL_CURRENCY_KEYWORDS | {"percent", "percentage", "pct", "rate", "ratio"}):
        return None, None, f"Could not safely parse '{col}' numeric text; retained source values"
    return None, None, None


def _parse_date_text(series: pd.Series, col: str) -> tuple[Optional[pd.Series], Optional[str]]:
    """Parse dates only when a deterministic format is supported by the sample."""
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return None, None

    unsupported = unsupported_date_reason(values)
    if unsupported:
        return None, f"Could not safely parse '{col}' as date: {unsupported} retained as text"

    sample = values.head(100)
    candidates: list[tuple[str, pd.Series]] = []
    for fmt in DATE_FORMAT_CANDIDATES:
        try:
            parsed_sample = pd.to_datetime(sample, format=fmt, errors="coerce")
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed_sample.notna().all():
            candidates.append((fmt, parsed_sample))

    if not candidates:
        if _is_date_name(col):
            return None, f"Could not safely parse non-null values in '{col}' as date; retained source values"
        return None, None

    # The sample only narrows formats. Compare their interpretations across every
    # source value: equal day/month values in the sample cannot establish locale.
    parsed = None
    for fmt, _ in candidates:
        try:
            candidate = pd.to_datetime(values, format=fmt, errors="coerce")
        except (TypeError, ValueError, OverflowError):
            continue
        if not candidate.notna().all():
            continue
        if parsed is not None and not candidate.equals(parsed):
            return None, f"Could not safely parse '{col}' as date: ambiguous date order; retained source values"
        parsed = candidate
    if parsed is None:
        return None, f"Could not safely parse all non-null values in '{col}' as date; retained source values"
    return parsed, f"Parsed '{col}' as date"


def _validated_llm_semantic(series: pd.Series, col: str, value: Optional[str]) -> Optional[str]:
    semantic = _normalize_llm_semantic(value)
    tokens = _column_tokens(col)
    if "year" in tokens and pd.api.types.is_numeric_dtype(series):
        values = series.dropna()
        if len(values) and values.map(lambda value: float(value).is_integer() and MIN_CALENDAR_YEAR <= value <= MAX_CALENDAR_YEAR).all():
            return "temporal" if semantic == "temporal" else None
    if semantic == "metric" and (
        not pd.api.types.is_numeric_dtype(series) or _is_identifier_name(col)
    ):
        return None
    if semantic == "temporal":
        if pd.api.types.is_datetime64_any_dtype(series):
            return semantic
        if "year" in tokens and pd.api.types.is_numeric_dtype(series):
            values = series.dropna()
            if len(values) and values.map(lambda value: float(value).is_integer() and MIN_CALENDAR_YEAR <= value <= MAX_CALENDAR_YEAR).all():
                return semantic
        return None
    if semantic == "categorical" and pd.api.types.is_datetime64_any_dtype(series):
        return None
    return semantic


def _numeric_values_are_finite(values: pd.Series) -> bool:
    """Check parsed values without converting exact integer arrays through float."""
    if pd.api.types.is_integer_dtype(values.dtype):
        return True
    if pd.api.types.is_numeric_dtype(values.dtype):
        return bool(np.isfinite(values.to_numpy()).all())
    return bool(values.map(math.isfinite).all())


def _can_apply_interpretation(series: pd.Series, col: str, metadata: dict) -> bool:
    """Reject unsafe combinations before any parsed view is changed."""
    decision = metadata.get("decision")
    if metadata.get("status") != "ok" or not decision or decision["needs_clarification"]:
        return False
    role, policy, unit = decision["role"], decision["parsing_policy"], decision["unit"]
    aggregation = decision["recommended_aggregation"]
    if role == "unknown" or unit in {"unknown", "mixed_currency"}:
        return False
    if aggregation not in {"sum", "mean", "count", "none"}:
        return False
    if role in {"identifier", "categorical"}:
        return policy in {"preserve_lexeme", "preserve_source"} and aggregation in {"count", "none"}
    if role == "temporal":
        return (not _is_identifier_name(col) and policy == "parse_unambiguous_date"
                and unit == "calendar_date" and aggregation == "none")
    if role != "metric" or aggregation not in {"sum", "mean"}:
        return False
    tokens = _column_tokens(col)
    if tokens & (IDENTIFIER_HINT_KEYWORDS - {"account"}) or _has_split_id_token(col):
        return False
    if "account" in tokens and ("balance" not in tokens or unit not in {"USD", "EUR"}):
        return False
    if series.dropna().empty or _has_leading_zero_identifiers(series):
        return False
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        observed = series.dropna()
        if not _numeric_values_are_finite(observed) or observed.map(lambda value: abs(float(value)) > 2**53 - 1).any():
            return False
        if math.fsum(abs(float(value)) for value in observed) > 2**53 - 1:
            return False
    values = _map_distinct_text(series.dropna().astype(str), lambda value: value.strip())
    header_currencies = tokens & {"usd", "eur"}
    if header_currencies and header_currencies != {unit.lower()}:
        return False
    if unit in {"USD", "EUR"}:
        explicit_units = _map_distinct_text(
            values, lambda value: re.match(r"^(?:USD|EUR)\s+", value) is not None
        )
        if unit.lower() not in tokens and not explicit_units.all():
            return False
    if policy == "parse_currency_decimal":
        if unit not in {"USD", "EUR"}:
            return False
        # A symbol alone does not establish currency identity. Require source evidence.
        header_unit = unit.lower() in _column_tokens(col)
        explicit_units = _map_distinct_text(
            values, lambda value: re.match(r"^(?:USD|EUR)\s+", value) is not None
        )
        if not header_unit and not explicit_units.all():
            return False
        other = "EUR" if unit == "USD" else "USD"
        if _map_distinct_text(values, lambda value: other in value).any():
            return False
        unique_values = _distinct_text_values(values)
        symbol_values = values.array if unique_values is None else unique_values
        symbols = {
            ch for value in symbol_values for ch in value
            if unicodedata.category(ch) == "Sc"
        }
        if not symbols.issubset({"$"} if unit == "USD" else {"€"}):
            return False
        return True
    if policy == "parse_percent_to_ratio":
        return unit == "ratio" and _map_distinct_text(
            values, lambda value: value.endswith("%")
        ).all()
    if policy == "preserve_numeric_value":
        return pd.api.types.is_numeric_dtype(series)
    if policy in {"parse_decimal", "preserve_nulls_parse_numeric"}:
        return not _map_distinct_text(
            values, _contains_currency_symbol
        ).any() and not _map_distinct_text(values, lambda value: value.endswith("%")).any()
    return False


def _parse_interpreted_numeric(series: pd.Series, col: str, decision: dict):
    if decision["parsing_policy"] == "parse_percent_to_ratio":
        values = _map_distinct_text(series.dropna().astype(str), lambda value: value.strip())
        values = _map_distinct_text(values, lambda value: value[:-1].strip())
        if not _map_distinct_text(
            values, lambda value: bool(_PLAIN_NUMBER.fullmatch(value))
        ).all():
            return None, None, f"Could not safely parse '{col}' percentage text; retained source values"
        def ratio(token):
            with localcontext() as context:
                context.prec = max(28, len(token) + 2)
                return float(Decimal(token) / 100)
        return _map_distinct_text(values, ratio), "percentage", f"Converted '{col}' from percentage text to decimal"
    if decision["parsing_policy"] != "parse_currency_decimal":
        return _parse_numeric_text(series, col, allow_metric_name=True)
    values = _map_distinct_text(series.dropna().astype(str), lambda value: value.strip())
    normalized = _map_distinct_text(
        values,
        lambda value: _strip_currency_symbols(re.sub(r"^(?:USD|EUR)\s+", "", value)),
    )
    if not _map_distinct_text(
        normalized,
        lambda value: bool(_PLAIN_NUMBER.fullmatch(value) or _US_GROUPED_NUMBER.fullmatch(value)),
    ).all():
        return None, None, f"Could not safely parse '{col}' currency text; retained source values"
    normalized = _map_distinct_text(normalized, lambda value: value.replace(",", ""))
    parsed = pd.to_numeric(normalized, errors="raise")
    return parsed, "currency", f"Converted '{col}' from currency text to numeric with verified currency identity"


def _interpreted_numbers_are_lossless(original: pd.Series, converted: pd.Series, decision: dict) -> bool:
    """Reject rounding and quantities outside JavaScript's exact integer range."""
    magnitude_sum = Decimal(0)
    policy = decision["parsing_policy"]

    def normalize_token(token: str) -> str:
        if policy == "parse_currency_decimal":
            token = re.sub(r"^(?:USD|EUR)\s+", "", token)
            token = _strip_currency_symbols(token)
        if policy == "parse_percent_to_ratio":
            token = token[:-1].strip()
        return token.replace(",", "")

    tokens = iter(_map_distinct_text(
        _map_distinct_text(original.dropna().astype(str), lambda value: value.strip()),
        normalize_token,
    ).tolist())
    for source, value in zip(original.tolist(), converted.tolist()):
        if pd.isna(source):
            if not pd.isna(value):
                return False
            continue
        token = next(tokens)
        percent = policy == "parse_percent_to_ratio"
        try:
            with localcontext() as context:
                context.prec = max(28, len(token) + 2)
                expected = Decimal(token) / (100 if percent else 1)
            actual = Decimal(str(value))
            if (not actual.is_finite() or abs(actual) > 2**53 - 1 or actual != expected):
                return False
            magnitude_sum += abs(actual)
            if magnitude_sum > 2**53 - 1:
                return False
        except (InvalidOperation, ValueError, OverflowError):
            return False
    return True


def _expand_parsed_values(original: pd.Series, parsed: pd.Series, missing_value) -> pd.Series:
    """Reinsert parsed non-null values by position while preserving source nulls."""
    present = original.notna().to_numpy()
    positions = np.flatnonzero(present)
    if len(positions) == len(original):
        values = parsed.array.copy()
    elif pd.api.types.is_integer_dtype(parsed.dtype):
        # A floating NaN would coerce large integers to float and lose precision.
        bits = parsed.dtype.itemsize * 8
        integer_type = "UInt" if pd.api.types.is_unsigned_integer_dtype(parsed.dtype) else "Int"
        values = pd.array([pd.NA] * len(original), dtype=f"{integer_type}{bits}")
        values[positions] = parsed.array
    else:
        values = pd.array([missing_value] * len(original), dtype=parsed.dtype)
        values[positions] = parsed.array
    return pd.Series(values, index=original.index)


def clean_dataframe(
    df: pd.DataFrame,
    interpret_columns: bool = True,
    *,
    import_settings: Optional[ImportSettings] = None,
    column_overrides: Optional[list[ColumnOverride]] = None,
) -> tuple[pd.DataFrame, list[str], dict[str, int], dict[str, str], dict[str, str]]:
    """
    Apply loss-aware type normalization and extract metadata without imputing.
    Returns: (cleaned_df, cleaning_actions, missing_counts, column_formats, semantic_types)
    """
    if import_settings is not None and not isinstance(import_settings, ImportSettings):
        import_settings = ImportSettings.model_validate(import_settings)
    original_headers = list(df.attrs.get("original_headers", []))
    df = df.copy(deep=False)
    cleaning_actions = []
    missing_counts = {}
    column_formats = {}
    semantic_types = {}

    # Count source nulls before parsing, using stable normalized schema keys.
    original_cols = df.columns.tolist()
    new_cols = _unique_normalized_headers([str(col) for col in original_cols])
    source_names = original_headers if len(original_headers) == len(original_cols) else [str(col) for col in original_cols]
    original_names = {
        normalized: str(original)
        for normalized, original in zip(new_cols, source_names)
    }
    source_missing = [int(df.iloc[:, index].isna().sum()) for index in range(len(original_cols))]

    # Optional semantic suggestions cannot control the deterministic column keys.
    headers = [str(col) for col in original_cols]
    llm_start = perf_counter()
    if interpret_columns:
        llm_columns = llm_enrich_columns(headers, df)
    else:
        llm_columns = [
            {"original": header, "clean": legacy_normalize_header(header),
             "format": None, "semantic_type": None}
            for header in headers
        ]
    llm_end = perf_counter()
    print(f"LLM Schema Mapping Time: {llm_end - llm_start:.2f}s")
    renamed = [(o, n) for o, n in zip(original_cols, new_cols) if o != n]
    if renamed:
        cleaning_actions.append(f"Normalized {len(renamed)} column headers")
    df.columns = new_cols

    df, column_schema = apply_column_policy(
        df,
        import_settings,
        column_overrides,
        original_names=original_names,
    )
    schema_by_column = {item["column"]: item for item in column_schema}

    for col, count in zip(new_cols, source_missing):
        if count:
            missing_counts[col] = count
            cleaning_actions.append(f"Preserved {count} source missing values as null in '{col}'")

    llm_semantic_candidates = {
        final_col: row.get("semantic_type")
        for final_col, row in zip(new_cols, llm_columns)
    }

    interpretations = {
        final_col: row["interpretation"] for final_col, row in zip(new_cols, llm_columns)
        if "interpretation" in row
    }
    df.attrs["column_interpretations"] = interpretations

    # Parse complete columns only when every non-null value has one safe interpretation.
    pandas_processing_start = perf_counter()
    for col in df.columns:
        policy = schema_by_column[col]
        manual_role = policy["role"]
        parse_as = policy["parse_as"]
        manual_format = policy["format"]
        if (
            parse_as in {"text", "number", "date"}
            or manual_role is not None
            or manual_format is not None
        ):
            if manual_role is not None:
                semantic_types[col] = manual_role
            if parse_as == "date" or manual_role == "temporal" or manual_format == "date":
                column_formats[col] = "date"
            elif parse_as == "number" or manual_role == "metric" or manual_format in {"number", "currency", "percentage"}:
                column_formats[col] = manual_format or detect_column_format(df[col], col)
            elif manual_format:
                column_formats[col] = manual_format
            continue

        metadata = interpretations.get(col)
        decision = metadata.get("decision") if metadata else None
        if metadata and not _can_apply_interpretation(df[col], col, metadata):
            semantic_types[col] = "unknown"
            cleaning_actions.append(f"Column '{col}' needs interpretation review; source values preserved")
            continue
        if decision and decision["role"] in {"identifier", "categorical"}:
            semantic_types[col] = decision["role"]
            metadata["runtime_status"] = "applied"
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            original = df[col]
            if decision and decision["role"] == "temporal":
                parsed, fmt, action = None, None, None
            else:
                try:
                    if decision:
                        parsed, fmt, action = _parse_interpreted_numeric(original, col, decision)
                    elif not import_settings or import_settings.decimal_separator == "auto":
                        parsed, fmt, action = _parse_numeric_text(original, col)
                    else:
                        parsed, fmt, action = None, None, None
                    if parsed is not None and not _numeric_values_are_finite(parsed):
                        parsed, fmt, action = None, None, f"Could not safely parse '{col}' as finite numbers; retained source values"
                except (ValueError, TypeError, OverflowError):
                    parsed, fmt, action = None, None, f"Could not safely parse '{col}' as representable numbers; retained source values"
            if action:
                cleaning_actions.append(action)
            if parsed is not None:
                converted = _expand_parsed_values(original, parsed, float("nan"))
                if decision and not _interpreted_numbers_are_lossless(original, converted, decision):
                    semantic_types[col] = "unknown"
                    cleaning_actions.append(f"Column '{col}' needs interpretation review; numeric precision would be lost")
                    continue
                df[col] = converted
                if fmt:
                    column_formats[col] = fmt
                continue
            if (
                fmt is None
                and not action
                and (not decision or decision["role"] == "temporal")
                and not _is_identifier_name(col)
                and (not import_settings or import_settings.date_order == "auto")
            ):
                parsed_dates, date_action = _parse_date_text(original, col)
                if date_action:
                    cleaning_actions.append(date_action)
                if parsed_dates is not None:
                    converted = _expand_parsed_values(original, parsed_dates, pd.NaT)
                    df[col] = pd.to_datetime(converted, errors="raise")
                    column_formats[col] = "date"

    pandas_processing_end = perf_counter()
    print(f"Pandas Vectorization & Date Parsing Time: {pandas_processing_end - pandas_processing_start:.2f}s")

    for col in df.columns:
        series = df[col]
        column_policy = schema_by_column[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            column_formats[col] = "date"
        elif column_policy["role"] == "metric" or column_policy["parse_as"] == "number":
            inferred = detect_column_format(series, col)
            column_formats[col] = "number" if inferred == "identifier" else inferred
        elif column_policy["role"] == "identifier":
            column_formats[col] = "identifier"
        elif column_policy["role"] == "unknown":
            column_formats[col] = "general"
        elif _is_identifier_name(col):
            column_formats[col] = "identifier"
        elif col not in column_formats:
            inferred = detect_column_format(series, col)
            if inferred != "number" or pd.api.types.is_numeric_dtype(series):
                column_formats[col] = inferred
            values = series.dropna().astype(str)
            if col not in column_formats and len(values) and _map_distinct_text(
                values, _contains_currency_symbol
            ).any():
                column_formats[col] = "currency"

    for col, candidate in llm_semantic_candidates.items():
        if col in interpretations:
            metadata = interpretations[col]
            decision = metadata.get("decision")
            if metadata["runtime_status"] == "applied":
                continue
            if semantic_types.get(col) == "unknown":
                continue
            role = decision["role"] if decision else "unknown"
            if (role == "metric" and not pd.api.types.is_numeric_dtype(df[col])) or (
                role == "temporal" and not pd.api.types.is_datetime64_any_dtype(df[col])
            ):
                semantic_types[col] = "unknown"
                cleaning_actions.append(f"Column '{col}' needs interpretation review; proposed role was not supported by parsed values")
            else:
                semantic_types[col] = role
                metadata["runtime_status"] = "applied"
            continue
        llm_sem = _validated_llm_semantic(df[col], col, candidate)
        if llm_sem:
            semantic_types[col] = llm_sem

    for col, metadata in interpretations.items():
        decision = metadata.get("decision")
        if metadata["runtime_status"] != "applied":
            column_formats[col] = "general"
        elif decision["role"] == "identifier":
            column_formats[col] = "identifier"
        elif decision["role"] == "temporal":
            column_formats[col] = "date"
        elif decision["role"] == "metric":
            column_formats[col] = {"USD": "currency", "EUR": "currency", "ratio": "percentage"}.get(decision["unit"], "number")
        else:
            column_formats[col] = "general"

    # Explicit formats override name-based and interpretation suggestions.
    for col, item in schema_by_column.items():
        if item["format"] is not None:
            column_formats[col] = item["format"]

    # User roles take precedence over interpretation suggestions and name hints.
    for col, item in schema_by_column.items():
        if item["role"] is not None:
            semantic_types[col] = item["role"]
        if item["format"] is not None:
            column_formats[col] = item["format"]

    # Complete one shared schema record per final, normalized column.
    for item in column_schema:
        col = item["column"]
        if item["role"] is None:
            item["role"] = semantic_types.get(col)
        if item["format"] is None:
            item["format"] = column_formats.get(col)
        if item["provenance"] == "inference" and (
            item["role"] is not None or item["format"] is not None
        ):
            item["status"] = "suggested"
    df.attrs["column_schema"] = column_schema

    return df, cleaning_actions, missing_counts, column_formats, semantic_types
