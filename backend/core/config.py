"""
Application configuration and shared clients/constants.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

APP_TITLE = "Analytico API V5"
APP_DESCRIPTION = "Zero Friction Enterprise Analytics - Modular Architecture"
APP_VERSION = "5.0.0"

CORS_ALLOW_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

ALLOWED_FILTER_OPERATORS = {"eq", "gt", "lt", "gte", "lte", "contains"}
MAX_CHART_POINTS = 500

SYSTEM_PROMPT = """You are a data visualization assistant. Given a user question and dataset metadata, return JSON with:

1. xAxisKey: Column for X-axis
2. yAxisKeys: Numeric columns for Y-axis
3. chartType: "bar", "line", "area", "pie", or "composed"
4. aggregation: "sum", "mean", "count", "min", "max"
5. title: Chart title
6. xAxisLabel: Human-readable X-axis label (e.g., "Department")
7. yAxisLabel: Human-readable Y-axis label (e.g., "Total Revenue ($)")
8. analysis: A 2-sentence business insight. First sentence summarizes what is shown, second sentence highlights the key trend or outlier.
9. calculated_field: Optional {name, expression} for derived metrics

Rules:
- IDENTIFIER columns: use COUNT only
- TEMPORAL columns: prefer as X-axis
- Format labels based on column format (currency -> include $)

ROUTING DECISION:
- Primary goal: produce a JSON chart configuration that visualizes the answer.
- If the user asks for basic aggregations (sum, average/mean, min, max, count), grouped by category or time, you MUST return JSON config and MUST NOT use the Python tool.
- Infer the best chart type even if the user asks for a single number; default to a chartable aggregation JSON response whenever possible.
- Use the Python tool only as a fallback for advanced statistics or multi-step transformations that cannot be represented as a standard aggregated chart.

Return ONLY raw JSON, no markdown."""
