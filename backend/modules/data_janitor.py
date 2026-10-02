"""
Analytico Backend - Data Janitor Module
Smart ingestion, header normalization, type repair, and data cleaning
"""

import os
import math
import re
import unicodedata
from time import perf_counter
from typing import Optional

import pandas as pd

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
    "ssn", "account", "serial", "reference", "ref",
}


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


def _is_identifier_name(col_name: str) -> bool:
    return bool(_column_tokens(col_name) & IDENTIFIER_HINT_KEYWORDS)


def _is_date_name(col_name: str) -> bool:
    return bool(_column_tokens(col_name) & {"date", "time", "datetime", "timestamp", "year"})


def _has_leading_zero_identifiers(series: pd.Series) -> bool:
    values = series.dropna().astype(str).str.strip()
    return bool(values.str.match(r"^[+-]?0\d+$").any())


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
        result = interpret_column(_interpretation_input(header, df.iloc[:, index], headers))
        decision = result.decision.model_dump() if result.decision is not None else None
        row["interpretation"] = {
            "status": result.status, "runtime_status": "clarification",
            "provider": result.provider or provider, "model": result.model,
            "prompt_version": result.prompt_version, "latency_ms": result.latency_ms,
            "usage": result.usage, "confidence": result.confidence,
            "error_code": result.error_code, "decision": decision,
        }
        unavailable = result.status in {"unavailable", "disabled", "error"}
        if decision is not None:
            row["semantic_type"] = decision["role"]
    return rows


def llm_clean_headers(headers: list[str], df: pd.DataFrame) -> list[str]:
    """Smart header normalization using LLM with sample data context"""
    return _unique_normalized_headers(headers)


def detect_column_format(series: pd.Series, col_name: str) -> str:
    """Detect the display format for a column"""
    col_lower = col_name.lower()
    
    if _is_identifier_name(col_name) and not any(kw in col_lower for kw in UNIVERSAL_CURRENCY_KEYWORDS):
        return 'identifier'
    
    # Currency keywords (domain-agnostic set)
    if any(kw in col_lower for kw in UNIVERSAL_CURRENCY_KEYWORDS):
        return 'currency'
    
    # Percentage keywords
    if any(kw in col_lower for kw in ['percent', 'pct', 'rate', 'ratio']):
        return 'percentage'
    
    return 'number'


def _contains_currency_symbol(value: str) -> bool:
    return any(unicodedata.category(ch) == "Sc" for ch in value)


def _strip_currency_symbols(value: str) -> str:
    return "".join(ch for ch in value if unicodedata.category(ch) != "Sc").strip()


def _is_coded_numeric_column(
    series: pd.Series,
    col_name: str,
    semantic_types: dict[str, str],
    column_formats: dict[str, str],
) -> bool:
    """Detect numeric columns that are categorical/identifier-like codes using generic signals."""
    sem = semantic_types.get(col_name)
    col_lower = col_name.lower()
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    if column_formats.get(col_name) in {"currency", "percentage"}:
        return False

    if sem == "identifier":
        return True

    keyword_hint = any(kw in col_lower for kw in IDENTIFIER_HINT_KEYWORDS)
    numeric_vals = pd.to_numeric(non_null, errors="coerce").dropna()
    if len(numeric_vals) == 0:
        return False

    integer_like = (numeric_vals.round() == numeric_vals).mean() > 0.95
    unique_count = int(numeric_vals.nunique())
    unique_ratio = unique_count / max(len(numeric_vals), 1)
    repeat_ratio = 1.0 - unique_ratio
    low_cardinality = unique_count <= max(20, int(len(numeric_vals) * 0.05))

    if sem == "categorical" and integer_like and low_cardinality and repeat_ratio >= 0.5:
        return True
    if keyword_hint and integer_like and low_cardinality:
        return True
    return False


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
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return None, None, None

    if (_is_identifier_name(col) and not allow_metric_name) or _has_leading_zero_identifiers(values):
        return None, None, None

    percent_values = values.str.endswith("%")
    if percent_values.all():
        normalized = values.str[:-1].str.strip()
        if normalized.map(lambda value: bool(_PLAIN_NUMBER.fullmatch(value))).all():
            parsed = pd.to_numeric(normalized, errors="raise") / 100
            return parsed, "percentage", f"Converted '{col}' from percentage text to decimal"
        return None, None, f"Could not safely parse '{col}' percentage text; retained source values"

    has_currency_symbol = values.map(_contains_currency_symbol)
    normalized = values.map(_strip_currency_symbols)
    if has_currency_symbol.any():
        if _PLAIN_NUMBER.fullmatch(normalized.iloc[0]) and normalized.map(
            lambda value: bool(_PLAIN_NUMBER.fullmatch(value))
        ).all():
            parsed = pd.to_numeric(normalized, errors="raise")
            return parsed, "currency", f"Converted '{col}' from currency text to numeric"
        if normalized.map(lambda value: bool(_US_GROUPED_NUMBER.fullmatch(value))).all():
            parsed = pd.to_numeric(normalized.str.replace(",", "", regex=False), errors="raise")
            return parsed, "currency", f"Converted '{col}' from currency text to numeric"
        return None, None, f"Could not safely parse '{col}' currency text; retained source values"

    if normalized.map(lambda value: bool(_PLAIN_NUMBER.fullmatch(value))).all():
        parsed = pd.to_numeric(normalized, errors="raise")
        return parsed, None, f"Converted '{col}' from numeric text to numeric"

    has_comma = normalized.str.contains(",", regex=False)
    if has_comma.any():
        # A comma without a decimal point can mean grouping or decimal notation.
        if normalized.map(lambda value: bool(re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", value))).all():
            return None, None, f"Could not safely parse '{col}' numeric text: ambiguous locale separators; retained source values"
        if normalized.map(lambda value: bool(_US_GROUPED_NUMBER.fullmatch(value))).all():
            parsed = pd.to_numeric(normalized.str.replace(",", "", regex=False), errors="raise")
            return parsed, None, f"Converted '{col}' from grouped numeric text to numeric"

    numeric_like = normalized.map(
        lambda value: bool(re.fullmatch(r"[+-]?[\d.,\s]+", value))
    )
    if numeric_like.mean() >= 0.5:
        return None, None, f"Could not safely parse '{col}' numeric text; retained source values"
    if _column_tokens(col) & (UNIVERSAL_CURRENCY_KEYWORDS | {"percent", "percentage", "pct", "rate", "ratio"}):
        return None, None, f"Could not safely parse '{col}' numeric text; retained source values"
    return None, None, None


def _parse_date_text(series: pd.Series, col: str) -> tuple[Optional[pd.Series], Optional[str]]:
    """Parse dates only when a deterministic format is supported by the sample."""
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return None, None

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
        if len(values) and values.map(lambda value: float(value).is_integer() and 1000 <= value <= 2200).all():
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
            if len(values) and values.map(lambda value: float(value).is_integer() and 1000 <= value <= 2200).all():
                return semantic
        return None
    if semantic == "categorical" and pd.api.types.is_datetime64_any_dtype(series):
        return None
    return semantic


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
        return policy == "parse_unambiguous_date" and unit == "calendar_date" and aggregation == "none"
    if role != "metric" or aggregation not in {"sum", "mean"}:
        return False
    tokens = _column_tokens(col)
    if tokens & (IDENTIFIER_HINT_KEYWORDS - {"account"}):
        return False
    if "account" in tokens and ("balance" not in tokens or unit not in {"USD", "EUR"}):
        return False
    if series.dropna().empty or _has_leading_zero_identifiers(series):
        return False
    if pd.api.types.is_numeric_dtype(series) and not series.dropna().map(math.isfinite).all():
        return False
    values = series.dropna().astype(str).str.strip()
    if unit in {"USD", "EUR"}:
        explicit_units = values.str.match(r"^(?:USD|EUR)\s+")
        if unit.lower() not in tokens and not explicit_units.all():
            return False
    if policy == "parse_currency_decimal":
        if unit not in {"USD", "EUR"}:
            return False
        # A symbol alone does not establish currency identity. Require source evidence.
        header_unit = unit.lower() in _column_tokens(col)
        explicit_units = values.str.match(r"^(?:USD|EUR)\s+")
        if not header_unit and not explicit_units.all():
            return False
        other = "EUR" if unit == "USD" else "USD"
        if values.str.contains(other, regex=False).any():
            return False
        symbols = {ch for value in values for ch in value if unicodedata.category(ch) == "Sc"}
        if not symbols.issubset({"$"} if unit == "USD" else {"€"}):
            return False
        return True
    if policy == "parse_percent_to_ratio":
        return unit == "ratio" and values.str.endswith("%").all()
    if policy == "preserve_numeric_value":
        return pd.api.types.is_numeric_dtype(series)
    if policy in {"parse_decimal", "preserve_nulls_parse_numeric"}:
        return not values.map(_contains_currency_symbol).any() and not values.str.endswith("%").any()
    return False


def _parse_interpreted_numeric(series: pd.Series, col: str, decision: dict):
    if decision["parsing_policy"] != "parse_currency_decimal":
        return _parse_numeric_text(series, col, allow_metric_name=True)
    values = series.dropna().astype(str).str.strip()
    normalized = values.str.replace(r"^(?:USD|EUR)\s+", "", regex=True).map(_strip_currency_symbols)
    if not normalized.map(lambda value: bool(_PLAIN_NUMBER.fullmatch(value) or _US_GROUPED_NUMBER.fullmatch(value))).all():
        return None, None, f"Could not safely parse '{col}' currency text; retained source values"
    parsed = pd.to_numeric(normalized.str.replace(",", "", regex=False), errors="raise")
    return parsed, "currency", f"Converted '{col}' from currency text to numeric with verified currency identity"


def _expand_parsed_values(original: pd.Series, parsed: pd.Series, missing_value) -> pd.Series:
    """Reinsert parsed non-null values by position while preserving source nulls."""
    positions = [position for position, present in enumerate(original.notna().to_numpy()) if present]
    expanded = [missing_value] * len(original)
    for position, value in zip(positions, parsed.to_list()):
        expanded[position] = value
    return pd.Series(expanded, index=original.index)


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, int], dict[str, str], dict[str, str]]:
    """
    Apply loss-aware type normalization and extract metadata without imputing.
    Returns: (cleaned_df, cleaning_actions, missing_counts, column_formats, semantic_types)
    """
    df = df.copy(deep=True)
    cleaning_actions = []
    missing_counts = {}
    column_formats = {}
    semantic_types = {}

    # Count source nulls before parsing, using stable normalized schema keys.
    original_cols = df.columns.tolist()
    new_cols = _unique_normalized_headers([str(col) for col in original_cols])
    source_missing = [int(df.iloc[:, index].isna().sum()) for index in range(len(original_cols))]

    # Optional semantic suggestions cannot control the deterministic column keys.
    llm_start = perf_counter()
    llm_columns = llm_enrich_columns([str(col) for col in original_cols], df)
    llm_end = perf_counter()
    print(f"LLM Schema Mapping Time: {llm_end - llm_start:.2f}s")
    renamed = [(o, n) for o, n in zip(original_cols, new_cols) if o != n]
    if renamed:
        cleaning_actions.append(f"Normalized {len(renamed)} column headers")
    df.columns = new_cols

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
            elif decision:
                try:
                    parsed, fmt, action = _parse_interpreted_numeric(original, col, decision)
                    if parsed is not None and not parsed.map(math.isfinite).all():
                        parsed, fmt, action = None, None, f"Could not safely parse '{col}' as finite numbers; retained source values"
                except (ValueError, TypeError, OverflowError):
                    parsed, fmt, action = None, None, f"Could not safely parse '{col}' as representable numbers; retained source values"
            else:
                parsed, fmt, action = _parse_numeric_text(original, col)
            if action:
                cleaning_actions.append(action)
            if parsed is not None:
                df[col] = _expand_parsed_values(original, parsed, float("nan"))
                if fmt:
                    column_formats[col] = fmt
                continue
            if fmt is None and not action:
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
        if pd.api.types.is_datetime64_any_dtype(series):
            column_formats[col] = "date"
        elif _is_identifier_name(col) and not any(kw in col.lower() for kw in UNIVERSAL_CURRENCY_KEYWORDS):
            column_formats[col] = "identifier"
        elif col not in column_formats:
            inferred = detect_column_format(series, col)
            if inferred != "number" or pd.api.types.is_numeric_dtype(series):
                column_formats[col] = inferred
            values = series.dropna().astype(str)
            if col not in column_formats and len(values) and values.map(_contains_currency_symbol).any():
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

    return df, cleaning_actions, missing_counts, column_formats, semantic_types
