"""
Analytico Backend Modules
"""

from .data_janitor import (
    legacy_normalize_header,
    llm_clean_headers,
    detect_column_format,
    clean_dataframe,
    llm_fix_data_issues,
)

from .intelligence import (
    SemanticType,
    detect_semantic_type,
    generate_default_chart,
    auto_profile,
    generate_dynamic_suggestions,
)

from .aggregation import (
    smart_group_top_n,
    smart_resample_dates,
    enforce_semantic_rules,
    aggregate_data,
)

__all__ = [
    # Data Janitor
    'legacy_normalize_header',
    'llm_clean_headers',
    'detect_column_format',
    'clean_dataframe',
    'llm_fix_data_issues',
    # Intelligence
    'SemanticType',
    'detect_semantic_type',
    'generate_default_chart',
    'auto_profile',
    'generate_dynamic_suggestions',
    # Aggregation
    'smart_group_top_n',
    'smart_resample_dates',
    'enforce_semantic_rules',
    'aggregate_data',
]
