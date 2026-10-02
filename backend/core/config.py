"""
Application configuration and shared clients/constants.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from fastapi import HTTPException

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

APP_TITLE = "Analytico API V5"
APP_DESCRIPTION = "Local-first CSV analytics API"
APP_VERSION = "5.0.0"

CORS_ALLOW_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS = 10.0
OPENAI_MAX_RETRIES = 1
_openai_client: Optional[OpenAI] = None
_openai_client_key: Optional[str] = None


def get_openai_client() -> OpenAI:
    """Create the optional provider client only when an AI feature is used."""
    global _openai_client, _openai_client_key

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="AI features require OPENAI_API_KEY.")
    if _openai_client is None or _openai_client_key != api_key:
        _openai_client = OpenAI(
            api_key=api_key,
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=OPENAI_MAX_RETRIES,
        )
        _openai_client_key = api_key
    return _openai_client

DEMO_DATASETS = {
    "taxi": {
        "filename": "taxi_1m_rows.csv",
        "path": Path(__file__).resolve().parents[1] / "datasets" / "2021_Green_Taxi_Trip_Data_20260221.csv",
    },
    "gapminder": {
        "filename": "gapminderDataFiveYear.csv",
        "path": Path(__file__).resolve().parents[1] / "datasets" / "gapminderDataFiveYear.csv",
    },
}

MAX_CHART_POINTS = 500

SYSTEM_PROMPT = """You plan local CSV charts from the supplied dataset columns and a user question.

Return one strict JSON object matching the provided schema. For a supported grouped
sum, mean, median, count, minimum, or maximum, use kind=chart and choose exact
column names. Use kind=clarification for calculations that need unsupported
statistics, predictions, derived fields, or multi-step transformations. Explain
what the user can ask instead in clarification. Do not claim to have computed an
insight; the application computes the chart after your plan is validated.

Rules:
- Use only exact dataset column names for axes and filters.
- Prefer numeric metric columns for measures and temporal columns for time trends.
- Use count for identifier fields and non-numeric measures.
- Add filters only when the question clearly asks for them.
- Filter values must be scalar strings, numbers, or booleans.
- Keep chart titles and labels short and factual.
- A chart plan must include an X-axis and at least one measure.
- A clarification plan must include a concise user-facing clarification.
"""
