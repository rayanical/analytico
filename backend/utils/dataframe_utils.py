"""DataFrame and CSV helper utilities."""

from pathlib import Path
from typing import TextIO

import pandas as pd


def df_to_markdown(df: pd.DataFrame, n: int = 5) -> str:
    """Convert dataframe head to markdown table."""
    sample = df.head(n)
    header = "| " + " | ".join(str(c) for c in sample.columns) + " |"
    sep = "| " + " | ".join("---" for _ in sample.columns) + " |"
    rows = ["| " + " | ".join(str(v)[:25] for v in row) + " |" for _, row in sample.iterrows()]
    return "\n".join([header, sep] + rows)


def read_csv_fast(source: str | Path | TextIO) -> pd.DataFrame:
    """Read CSV using pyarrow when available, with safe fallback."""
    try:
        return pd.read_csv(source, engine="pyarrow")
    except Exception:
        if hasattr(source, "seek"):
            source.seek(0)
            return pd.read_csv(source, low_memory=False)
        return pd.read_csv(source, low_memory=False)
