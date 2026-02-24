"""
Analytico Backend - Data Janitor Module
Smart ingestion, header normalization, type repair, and data cleaning
"""

import json
import os
import re
import unicodedata
import warnings
from time import perf_counter
from typing import Optional

import pandas as pd
from openai import OpenAI

# Client initialized lazily to avoid import-time side effects
_client: Optional[OpenAI] = None

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

# Skip LLM schema mapping for very wide datasets to cap latency/cost.
LLM_SCHEMA_MAX_COLUMNS = 120
UNIVERSAL_CURRENCY_KEYWORDS = {
    "amount", "price", "cost", "revenue", "income", "salary", "expense",
    "value", "total", "balance", "payment",
}
IDENTIFIER_HINT_KEYWORDS = {"id", "code", "key", "identifier"}

def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def legacy_normalize_header(header: str) -> str:
    """Conservative, deterministic snake_case sanitizer."""
    clean = re.sub(r"[^\w]+", "_", header.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "column"


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


def llm_enrich_columns(headers: list[str], df: pd.DataFrame) -> list[dict[str, Optional[str]]]:
    """
    Use one LLM pass to normalize header names and classify each column.
    Returns rows with: original, clean, format, semantic_type.
    """
    fallback = [
        {
            "original": h,
            "clean": legacy_normalize_header(h),
            "format": None,
            "semantic_type": None,
        }
        for h in headers
    ]

    if not os.getenv("OPENAI_API_KEY") or len(headers) > LLM_SCHEMA_MAX_COLUMNS:
        return fallback

    try:
        client = get_openai_client()
        # Compact payload: per-column samples reduce tokens vs full row objects.
        sample_values = {}
        for col in headers:
            if col not in df.columns:
                sample_values[col] = []
                continue
            values = df[col].dropna().head(3).tolist()
            sample_values[col] = [str(v) for v in values]

        system_prompt = (
            "Respond ONLY with a JSON object where each key is the original column name and each value is an object "
            "with: semantic_type and clean_name. "
            "Schema: {\"Original Column Name\": {\"semantic_type\": \"metric|identifier|temporal|categorical\", "
            "\"clean_name\": \"short_snake_case_name\"}}. "
            "clean_name must be short, human-readable snake_case (example: lpep_pickup_datetime -> pickup_date). "
            "Do not include explanations, formatting, or markdown. Return valid JSON only."
        )
        user_prompt = f"""Headers:
{json.dumps(headers)}

Per-column sample values (up to 3 non-null each):
{json.dumps(sample_values, separators=(",", ":"))}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        by_original: dict[str, dict[str, Optional[str]]] = {}

        # Fast path: mapping "original_header" -> {"semantic_type", "clean_name"}.
        # Backward compatible with legacy flat semantic values.
        for original, payload in parsed.items():
            original_key = str(original).strip()
            if not original_key:
                continue
            semantic_value = payload.get("semantic_type") if isinstance(payload, dict) else payload
            clean_name = payload.get("clean_name") if isinstance(payload, dict) else None
            normalized_clean = legacy_normalize_header(str(clean_name).strip() if clean_name else original_key)
            by_original[original_key] = {
                "original": original_key,
                "clean": normalized_clean,
                "format": None,
                "semantic_type": _normalize_llm_semantic(semantic_value),
            }

        # Backward-compatible parser for older structured payloads.
        raw_columns = parsed.get("columns", []) if isinstance(parsed, dict) else []
        for row in raw_columns:
            if not isinstance(row, dict):
                continue
            original = str(row.get("original", "")).strip()
            if not original:
                continue
            raw_clean = str(row.get("clean_name") or row.get("clean") or "").strip()
            by_original[original] = {
                "original": original,
                "clean": legacy_normalize_header(raw_clean or original),
                "format": None,
                "semantic_type": _normalize_llm_semantic(row.get("semantic_type")),
            }

        return [by_original.get(h, f) for h, f in zip(headers, fallback)]
    except Exception as e:
        print(f"LLM Column Enrichment failed: {e}")
        return fallback


def llm_clean_headers(headers: list[str], df: pd.DataFrame) -> list[str]:
    """Smart header normalization using LLM with sample data context"""
    enriched = llm_enrich_columns(headers, df)
    return [re.sub(r"[^\w]", "_", str(row.get("clean", "") or "")).lower() for row in enriched]


def detect_column_format(series: pd.Series, col_name: str) -> str:
    """Detect the display format for a column"""
    col_lower = col_name.lower()
    
    # Identifier columns (IDs, codes, serial numbers)
    if any(kw in col_lower for kw in ['id', 'code', 'serial', 'key', 'number', 'ref', 'identifier']):
        # But not if it contains currency keywords
        if not any(kw in col_lower for kw in UNIVERSAL_CURRENCY_KEYWORDS):
            return 'identifier'
    
    # Currency keywords (domain-agnostic set)
    if any(kw in col_lower for kw in UNIVERSAL_CURRENCY_KEYWORDS):
        return 'currency'
    
    # Percentage keywords
    if any(kw in col_lower for kw in ['percent', 'pct', 'rate', 'ratio']):
        return 'percentage'
    
    # Check data patterns (more conservative)
    if pd.api.types.is_numeric_dtype(series):
        sample = series.dropna().head(100)
        if len(sample) > 0:
            # Values between 0-1 might be percentages
            if sample.between(0, 1).all() and sample.max() < 1:
                return 'percentage'
    
    return 'number'


def _contains_currency_symbol(value: str) -> bool:
    return any(unicodedata.category(ch) == "Sc" for ch in value)


def _strip_numeric_text(value: str) -> str:
    # Remove unicode currency symbols + common separators.
    no_currency = "".join(ch for ch in value if unicodedata.category(ch) != "Sc")
    return re.sub(r"[,\s]", "", no_currency)


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


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, int], dict[str, str], dict[str, str]]:
    """
    Clean DataFrame and extract metadata.
    Returns: (cleaned_df, cleaning_actions, missing_counts, column_formats, semantic_types)
    """
    cleaning_actions = []
    missing_counts = {}
    column_formats = {}
    semantic_types = {}
    
    # 1. Header Normalization (LLM Powered with Sample Data)
    original_cols = df.columns.tolist()
    llm_start = perf_counter()
    llm_columns = llm_enrich_columns(original_cols, df)
    llm_end = perf_counter()
    print(f"LLM Schema Mapping Time: {llm_end - llm_start:.2f}s")
    new_cols = [str(row.get("clean", "") or legacy_normalize_header(h)) for h, row in zip(original_cols, llm_columns)]
    
    # Handle duplicate column names by appending _2, _3, etc.
    seen = {}
    unique_cols = []
    for col in new_cols:
        if col in seen:
            seen[col] += 1
            unique_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 1
            unique_cols.append(col)
    new_cols = unique_cols
    
    renamed = [(o, n) for o, n in zip(original_cols, new_cols) if o != n]
    if renamed:
        cleaning_actions.append(f"Normalized {len(renamed)} column headers")
    df.columns = new_cols

    # Capture LLM-provided semantic_type on the final deduped names.
    for final_col, row in zip(new_cols, llm_columns):
        llm_sem = _normalize_llm_semantic(row.get("semantic_type"))
        if llm_sem:
            semantic_types[final_col] = llm_sem
    
    # 2. Type Repair and Format Detection
    pandas_processing_start = perf_counter()
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100).astype(str)
            if len(sample) == 0:
                continue
            
            normalized_numeric = sample.apply(_strip_numeric_text).str.strip()
            numeric_like_matches = normalized_numeric.str.match(r'^[+-]?\d+(\.\d+)?$').sum()
            currency_symbol_matches = sample.apply(_contains_currency_symbol).sum()
            has_currency_name = any(kw in col.lower() for kw in UNIVERSAL_CURRENCY_KEYWORDS)

            # Numeric-like text columns should be converted to numeric deterministically.
            if numeric_like_matches > len(sample) * 0.5:
                try:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .apply(_strip_numeric_text)
                        .str.strip()
                    )
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    has_currency_evidence = has_currency_name or currency_symbol_matches > len(sample) * 0.1
                    if has_currency_evidence:
                        column_formats[col] = 'currency'
                        cleaning_actions.append(f"Converted '{col}' from currency text to numeric")
                    else:
                        cleaning_actions.append(f"Converted '{col}' from numeric text to numeric")
                except Exception:
                    pass
                continue
            
            # Percentage pattern
            pct_matches = sample.str.match(r'^\d+\.?\d*%$').sum()
            if pct_matches > len(sample) * 0.5:
                try:
                    df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100
                    column_formats[col] = 'percentage'
                    cleaning_actions.append(f"Converted '{col}' from percentage text to decimal")
                except Exception:
                    pass
                continue
            
            # Try date parsing
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sample_parsed = pd.to_datetime(sample, errors='coerce')
                if sample_parsed.notna().sum() > len(sample) * 0.7:
                    sample_threshold = len(sample) * 0.7
                    full_threshold = len(df) * 0.7
                    parsed = None
                    success = False
                    winning_format = None

                    # Infer the fastest viable format on the sample first.
                    for fmt in DATE_FORMAT_CANDIDATES:
                        try:
                            sample_fmt_parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                            if sample_fmt_parsed.notna().sum() > sample_threshold:
                                winning_format = fmt
                                break
                        except Exception:
                            continue

                    # Apply winning format to full column for fast vectorized parsing.
                    if winning_format:
                        try:
                            parsed_candidate = pd.to_datetime(df[col], format=winning_format, errors='coerce')
                            if parsed_candidate.notna().sum() > full_threshold:
                                parsed = parsed_candidate
                                success = True
                        except Exception:
                            pass

                    # Final fallback for obscure/mixed formats.
                    if not success:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            parsed_fallback = pd.to_datetime(df[col], errors='coerce')
                        if parsed_fallback.notna().sum() > full_threshold:
                            parsed = parsed_fallback
                            success = True

                    if success and parsed is not None:
                        df[col] = parsed
                        column_formats[col] = 'date'
                        cleaning_actions.append(f"Parsed '{col}' as date")
            except Exception:
                pass
    pandas_processing_end = perf_counter()
    print(f"Pandas Vectorization & Date Parsing Time: {pandas_processing_end - pandas_processing_start:.2f}s")
    
    # Heuristic-first format detection (authoritative): always recompute for numeric columns.
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        inferred = detect_column_format(df[col], col)
        # Preserve strong parsing evidence from earlier conversion.
        if column_formats.get(col) in {"currency", "percentage"} and inferred == "number":
            continue
        column_formats[col] = inferred
    
    # 3. Record missing counts BEFORE filling
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_counts[col] = int(missing_count)
    
    # 4. Auto-Imputation using smart logic
    for col in numeric_cols:
        if col in missing_counts:
            if _is_coded_numeric_column(df[col], col, semantic_types, column_formats):
                df[col] = df[col].fillna(-1)
                cleaning_actions.append(
                    f"Filled {missing_counts[col]} missing '{col}' values with sentinel (-1) for coded field"
                )
                continue

            non_null = df[col].dropna()
            if len(non_null) == 0:
                cleaning_actions.append(
                    f"Left {missing_counts[col]} missing '{col}' values as null (low-confidence imputation)"
                )
                continue

            numeric_vals = pd.to_numeric(non_null, errors="coerce").dropna()
            coverage = len(non_null) / max(len(df[col]), 1)
            integer_like = (numeric_vals.round() == numeric_vals).mean() > 0.95 if len(numeric_vals) else False
            unique_ratio = numeric_vals.nunique() / max(len(numeric_vals), 1) if len(numeric_vals) else 0.0
            sem = semantic_types.get(col)

            # High-confidence bounded binary/ratio fields.
            if len(numeric_vals) and numeric_vals.between(0, 1).all():
                if (numeric_vals == 0).mean() >= 0.8:
                    df[col] = df[col].fillna(0)
                    cleaning_actions.append(
                        f"Filled {missing_counts[col]} missing '{col}' values with 0 for bounded ratio field"
                    )
                    continue
                mode = numeric_vals.mode(dropna=True)
                if len(mode) > 0:
                    mode_val = float(mode.iloc[0])
                    df[col] = df[col].fillna(mode_val)
                    cleaning_actions.append(
                        f"Filled {missing_counts[col]} missing '{col}' values with mode ({mode_val}) for bounded ratio field"
                    )
                    continue

            metric_confident = (
                sem == "metric"
                or (coverage >= 0.7 and not integer_like and unique_ratio > 0.2)
            )
            if metric_confident:
                skew = abs(float(numeric_vals.skew())) if len(numeric_vals) > 2 and pd.notna(numeric_vals.skew()) else 0.0
                if skew > 1.0:
                    median_val = float(numeric_vals.median())
                    df[col] = df[col].fillna(round(median_val, 4))
                    cleaning_actions.append(
                        f"Filled {missing_counts[col]} missing '{col}' values with median (high-confidence metric)"
                    )
                else:
                    mean_val = float(numeric_vals.mean())
                    df[col] = df[col].fillna(round(mean_val, 4))
                    cleaning_actions.append(
                        f"Filled {missing_counts[col]} missing '{col}' values with mean (high-confidence metric)"
                    )
                continue

            cleaning_actions.append(
                f"Left {missing_counts[col]} missing '{col}' values as null (low-confidence imputation)"
            )
    
    return df, cleaning_actions, missing_counts, column_formats, semantic_types
