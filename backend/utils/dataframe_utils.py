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
    """Read source lexemes before any analytics type inference.

    Pandas' PyArrow engine can infer numeric values before applying dtype=str,
    stripping identifier zeros. The C engine applies string typing at read time.
    """
    return pd.read_csv(
        source,
        engine="c",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
