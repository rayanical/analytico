"""
Analytico Backend - Data Janitor Module
Smart ingestion, header normalization, type repair, and data cleaning
"""

import json
import os
import re
from typing import Optional

import pandas as pd
from openai import OpenAI

# Client initialized lazily to avoid import-time side effects
_client: Optional[OpenAI] = None

def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def legacy_normalize_header(header: str) -> str:
    """Convert header to clean, short snake_case names"""
    # First, basic cleanup
    clean = re.sub(r'[^\w\s]', '', header.strip())
    clean = re.sub(r'\s+', '_', clean).lower()
    
    # Phase 1: Full replacement patterns (match entire column to known pattern)
    full_replacements = [
        (r'^(how_old_are_you|what_is_your_age).*', 'age'),
        (r'^(what_industry_do_you_work_in|industry_you_work_in).*', 'industry'),
        (r'^(what_is_your_annual_salary|annual_salary).*', 'annual_salary'),
        (r'^(what_is_your_job_title|job_title).*', 'job_title'),
        (r'^(what_is_your_gender|gender_identity).*', 'gender'),
        (r'^(what_country_do_you_work_in|country_you_work_in).*', 'country'),
        (r'^(what_state_do_you_work_in|state_you_work_in).*', 'state'),
        (r'^(what_city_do_you_work_in|city_you_work_in).*', 'city'),
        (r'^(how_many_years_of|years_of_professional|years_of_experience).*', 'years_experience'),
        (r'^(highest_level_of_education|education_level).*', 'education'),
        (r'^(what_is_your_race|race).*', 'race'),
        (r'^(please_indicate_the_currency|currency).*', 'currency'),
        (r'^(how_much_additional_monetary).*', 'additional_comp'),
        (r'^(if_your_job_title_needs).*', 'job_context'),
        (r'^(if_your_income_needs).*', 'income_context'),
        (r'^(if_youre_in_the_us).*', 'us_state'),
        (r'^(if_other_please).*', 'other_info'),
    ]
    
    for pattern, replacement in full_replacements:
        if re.match(pattern, clean):
            clean = replacement
            break
    
    # Phase 2: Prefix removals (apply all that match)
    prefix_removals = [
        r'^(what_is_your_|whats_your_|what_is_the_|what_are_your_)',
        r'^(please_indicate_the_|please_select_the_|please_enter_the_|please_specify_the_)',
        r'^(please_indicate_your_|please_select_your_|please_enter_your_)',
    ]
    for pattern in prefix_removals:
        clean = re.sub(pattern, '', clean)
    
    # Phase 3: Suffix removals (apply all that match)
    suffix_removals = [
        r'_youll_indicate.*$',
        r'_you_would_earn.*$',
        r'_please_only_include.*$',
        r'_choose_all_that_apply$',
        r'_or_prefer_not_to_answer$',
        r'_if_any$',
        r'_optional$',
        r'_please$',
    ]
    for pattern in suffix_removals:
        clean = re.sub(pattern, '', clean)
    
    # Final cleanup: remove leading/trailing underscores
    clean = clean.strip('_')
    
    # Truncate overly long names (max 20 chars for cleaner display)
    if len(clean) > 20:
        clean = clean[:20].rstrip('_')
    
    return clean


def llm_clean_headers(headers: list[str], df: pd.DataFrame) -> list[str]:
    """Smart header normalization using LLM with sample data context"""
    # Quick fallback if no API key
    if not os.getenv("OPENAI_API_KEY"):
        return [legacy_normalize_header(h) for h in headers]
        
    try:
        client = get_openai_client()
        
        # Build sample rows for context (much better than individual columns)
        # Get up to 3 sample rows to show relationships between columns
        sample_rows = df.head(3).to_dict('records')
        # Convert all values to strings for JSON serialization
        sample_rows = [{k: str(v) for k, v in row.items()} for row in sample_rows]
        
        prompt = f"""Normalize these column headers to clean snake_case variable names.
        
        IMPORTANT: You are provided with sample rows showing ALL columns together.
        Use this context to infer the meaning of abbreviated or unclear headers.
        
        For example:
        - If you see headers ["emp_id", "nm", "dept", "sal"] with sample rows like:
          {{"emp_id": "101", "nm": "John Doe", "dept": "Engineering", "sal": "75000"}}
          You can infer: "nm" -> "name", "sal" -> "salary"
        
        - If you see "wt" alongside "product_name" and "price", it's likely "weight"
        - If you see "wt" alongside "employee_name" and "department", it might be "work_type"
        
        Rules:
        1. "How old are you?" -> "age"
        2. "What is your annual salary?" -> "annual_salary"
        3. "DoB" -> "date_of_birth"
        4. Remove verbose prefixes: "what_is_your", "please_indicate", etc.
        5. Use context from OTHER columns to disambiguate abbreviations
        6. Keep names concise but descriptive (max 25 characters)
        7. Return JSON: {{"mapping": {{"original": "clean", ...}}}}

        Original headers:
        {json.dumps(headers)}
        
        Sample rows for context:
        {json.dumps(sample_rows, indent=2)}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        mapping = json.loads(content).get("mapping", {})
        
        results = []
        for h in headers:
            # Prefer LLM result, fallback to legacy
            cleaned = mapping.get(h)
            if not cleaned:
                cleaned = legacy_normalize_header(h)
            # Ensure safe char set even if LLM is creative
            cleaned = re.sub(r'[^\w]', '_', cleaned).lower()
            results.append(cleaned)
        return results
            
    except Exception as e:
        print(f"LLM Header Clean failed: {e}")
        return [legacy_normalize_header(h) for h in headers]


def detect_column_format(series: pd.Series, col_name: str) -> str:
    """Detect the display format for a column"""
    col_lower = col_name.lower()
    
    # Currency keywords
    if any(kw in col_lower for kw in ['salary', 'price', 'cost', 'revenue', 'amount', 'income', 'wage', 'comp', 'pay']):
        return 'currency'
    
    # Percentage keywords
    if any(kw in col_lower for kw in ['percent', 'pct', 'rate', 'ratio']):
        return 'percentage'
    
    # Check data patterns
    if pd.api.types.is_numeric_dtype(series):
        sample = series.dropna().head(100)
        if len(sample) > 0:
            # Large numbers often indicate currency
            if sample.abs().mean() > 1000:
                return 'currency'
            # Values between 0-1 might be percentages
            if sample.between(0, 1).all() and sample.max() < 1:
                return 'percentage'
    
    return 'number'


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


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, int], dict[str, str]]:
    """
    Clean DataFrame and extract metadata.
    Returns: (cleaned_df, cleaning_actions, missing_counts, column_formats)
    """
    cleaning_actions = []
    missing_counts = {}
    column_formats = {}
    
    # 1. Header Normalization (LLM Powered with Sample Data)
    original_cols = df.columns.tolist()
    new_cols = llm_clean_headers(original_cols, df)
    
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
    
    # 2. Type Repair and Format Detection
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100).astype(str)
            if len(sample) == 0:
                continue
            
            # Currency pattern
            currency_matches = sample.str.replace(',', '').str.match(r'^\$?[\d,]+\.?\d*$').sum()
            if currency_matches > len(sample) * 0.5:
                try:
                    df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    column_formats[col] = 'currency'
                    cleaning_actions.append(f"Converted '{col}' from currency text to numeric")
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
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() > len(df) * 0.7:
                    df[col] = parsed
                    column_formats[col] = 'date'
                    cleaning_actions.append(f"Parsed '{col}' as date")
            except Exception:
                pass
    
    # Detect formats for remaining numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col not in column_formats:
            column_formats[col] = detect_column_format(df[col], col)
    
    # 3. Record missing counts BEFORE filling
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_counts[col] = int(missing_count)
    
    # 4. Auto-Imputation using smart logic
    for col in numeric_cols:
        if col in missing_counts:
            mean_val = df[col].mean()
            if pd.notna(mean_val):
                df[col] = df[col].fillna(round(mean_val, 2))
                cleaning_actions.append(f"Filled {missing_counts[col]} missing '{col}' values with mean")
            else:
                df[col] = df[col].fillna(0)
    
    return df, cleaning_actions, missing_counts, column_formats
