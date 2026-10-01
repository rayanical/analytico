"""User-facing error mapping helpers."""

import csv

import pandas as pd
from fastapi import HTTPException


def friendly_ingestion_error_message(error: Exception) -> tuple[int, str]:
    """Map internal ingestion exceptions to concise user-facing errors."""
    if isinstance(error, HTTPException):
        return error.status_code, str(error.detail)
    if isinstance(error, (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, csv.Error)):
        return 400, "The uploaded file is not a valid CSV. Please check its delimiter, quoting, and encoding."

    error_text = str(error)
    if "could not convert string to float" in error_text:
        return (
            400,
            "We couldn't auto-parse one of the metric columns because it contains text values "
            "(for example, unit-suffixed values like '142 min'). "
            "Try changing that column to plain numeric values before upload.",
        )
    return (500, "Dataset processing failed. Please check the file format and try again.")
