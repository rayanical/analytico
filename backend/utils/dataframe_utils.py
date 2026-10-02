"""DataFrame and CSV helper utilities."""

from pathlib import Path
from typing import BinaryIO, TextIO

import pandas as pd

from modules.import_policy import (
    ImportSettings,
    read_csv_headers,
    reader_options,
    validate_csv_structure,
)


def df_to_markdown(df: pd.DataFrame, n: int = 5) -> str:
    """Convert dataframe head to markdown table."""
    sample = df.head(n)
    header = "| " + " | ".join(str(c) for c in sample.columns) + " |"
    sep = "| " + " | ".join("---" for _ in sample.columns) + " |"
    rows = ["| " + " | ".join(str(v)[:25] for v in row) + " |" for _, row in sample.iterrows()]
    return "\n".join([header, sep] + rows)


def read_csv_fast(
    source: str | Path | TextIO | BinaryIO,
    settings: ImportSettings | None = None,
) -> pd.DataFrame:
    """Read source lexemes before any analytics type inference.

    Pandas' PyArrow engine can infer numeric values before applying dtype=str,
    stripping identifier zeros. The C engine applies string typing at read time.
    """
    validate_csv_structure(source, settings)
    headers = read_csv_headers(source, settings)

    original_position = None
    if not isinstance(source, (str, Path)):
        try:
            original_position = source.tell()
            source.seek(0)
        except (AttributeError, OSError, ValueError):
            original_position = None
    try:
        frame = pd.read_csv(source, **reader_options(settings))
    finally:
        if original_position is not None:
            try:
                source.seek(original_position)
            except (OSError, ValueError):
                pass
    # Pandas mangles duplicate names (for example, ``code`` and ``code.1``).
    # Keep the actual source header by position so later schema metadata can
    # report the original name without trying to reverse that mangling.
    frame.attrs["original_headers"] = headers
    return frame
