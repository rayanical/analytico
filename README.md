<div align="center">

# Analytico

**Turn CSV files into interactive charts and exportable dashboards.**

A local-first analytics app combining natural-language exploration, deterministic calculations, and optional AI-powered column understanding.

[Demo](#demo) · [Features](#features) · [Architecture](#architecture) · [Run locally](#run-locally)

</div>

## Demo

https://github.com/user-attachments/assets/4c3df7ed-f755-4f2b-91f6-d5cb18f45186

Upload a dataset, ask a question, explore the underlying records, and assemble a dashboard without writing analysis code.

## Features

| Capability | What you can do |
| --- | --- |
| **Natural-language charts** | Ask for comparisons, trends, totals, averages, or counts. Supported questions become validated chart plans; ambiguous requests receive clarification. |
| **Automatic CSV preparation** | Detect file settings, validate the full file, profile missing values, and recognize supported numeric, currency, percentage, and date formats. |
| **AI column understanding** | Enable GPT-6 Luna before upload to infer column roles and readable labels in the background. Start charting after local preparation; review or edit the schema whenever needed. |
| **Interactive exploration** | Build charts manually, apply structured filters, and drill into matching source rows. Overall answers appear as a single bar with its value. |
| **Dashboard builder** | Pin charts, drag and resize widgets, and save per-dataset layouts in browser storage. |
| **Exportable reports** | Download PNG/SVG chart assets or export a multi-page PDF dashboard. |

## Engineering highlights

- **AI plans; the engine calculates.** Model output passes through structured validation before execution. Pandas and DuckDB perform the aggregations; the app does not execute generated Python or model-written SQL.
- **A hybrid ingestion engine.** Small CSVs use pandas; files of at least 8 MiB use temporary disk-backed DuckDB by default. Native CSV loading, bounded previews, and a CPU-capped worker budget support larger datasets without putting every row into the browser.
- **Source-preserving interpretation.** Original CSV bytes are retained for the session, missing observations remain null, and conversions require full-column validation. AI display labels do not rename source keys or rewrite values.
- **Non-blocking enrichment.** A bounded background queue runs optional schema analysis. Compatibility checks, user-override protection, and schema revisions prevent stale enrichment from overwriting reviewed choices.
- **One calculation path across interactions.** AI questions and manual charts share aggregation logic. Filters apply before calculation, and drill-down uses the same source filters as the chart.

## Architecture

```mermaid
flowchart TD
    CSV[CSV file] --> API[FastAPI ingestion]
    API --> Validate[Full-file validation and type planning]
    Validate --> Engine[Dataset engine: pandas or DuckDB]
    Engine --> Profile[Column profiles and chart metadata]
    Profile --> UI[Next.js / React workspace]

    Profile -. Optional background analysis .-> Luna[GPT-6 Luna: roles and labels]
    Luna --> Guard[Compatibility and revision checks]
    Guard --> UI

    UI --> Question[Natural-language question]
    Question --> Plan[GPT-6 Luna: structured chart plan]
    Plan --> Check[Plan validation]
    UI --> Manual[Manual chart configuration]
    Check --> Aggregate[Deterministic aggregation and filters]
    Manual --> Aggregate
    Engine --> Aggregate
    Aggregate --> Chart[Recharts visualization and source-row drill-down]
    Chart --> Dashboard[Dashboard layout]
    Dashboard --> Export[PNG / SVG / PDF]
```

The backend separates ingestion, schema review, query planning, aggregation, and enrichment into modules with validated request/response contracts. The frontend keeps chart interactions, dashboard layout, and export rendering in the local workspace.

## Measured performance & validation

Benchmarks use complete public datasets rather than generated toy inputs. With the same validation behavior, increasing DuckDB's default worker budget from two to four produced:

| Dataset | Rows | Two threads | Four threads | Preparation time reduction |
| --- | ---: | ---: | ---: | ---: |
| NYC green taxi | 1,068,755 | 13.03 s | 10.84 s | 16.8% |
| Online retail | 541,909 | 3.36 s | 2.82 s | 16.0% |

Three-run medians on one development machine, measuring local engine preparation only; upload transfer, AI requests, and browser rendering are excluded. More threads increase peak memory. The default is capped to available CPUs and can be lowered with `ANALYTICO_DUCKDB_THREADS=1` or `2`.

- **245 backend tests** passed, covering parsing, validation, filters, aggregation, schema updates, and background enrichment.
- **144 experimental imports** matched baseline full-source fingerprints across eight real datasets.
- **12 full-file calculation checks** passed against an independent Decimal-based reference across tips, diamonds, and taxi; **nine live Luna chart-planning checks** also passed.
- Frontend TypeScript and helper/rendering checks passed. These checks establish behavior on the evaluated cases, not universal AI accuracy.

Reproducible runners live in [`backend/benchmarks`](backend/benchmarks); compact results are retained alongside them. Real-data runners require their external fixtures; live AI runs consume provider credits.

## Tech stack

| Layer | Technologies |
| --- | --- |
| Frontend | TypeScript, React 19, Next.js 16, Tailwind CSS |
| Visualization & interaction | Recharts, Framer Motion, react-grid-layout |
| Backend | Python, FastAPI, Pydantic |
| Data engine | DuckDB, pandas |
| AI | OpenAI GPT-6 Luna; structured output and local validation |
| Export | html-to-image, jsPDF |
| Verification | Python unittest, TypeScript, frontend helper checks |

## Run locally

Use Python 3.11+ and a Node.js version compatible with Next.js 16. The tested backend environment is Python 3.14 on macOS. Python dependencies are pinned with hashes; frontend dependencies use `package-lock.json`.

**Backend**

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install --require-hashes --only-binary=:all: -r requirements.txt
cp -n .env.example .env
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

On Windows, activate the virtual environment with `venv\Scripts\activate` and copy `.env.example` to `.env` using your shell.

**Frontend** — in a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open [localhost:3000](http://localhost:3000). The included Gapminder dataset lets you explore immediately; the taxi demo requires a separately installed CSV.

For AI features, add your own `OPENAI_API_KEY` to `backend/.env`, which is ignored by Git. Chart planning and descriptions default to `gpt-6-luna`. The upload-time **AI column semantic analysis** toggle enables the batched Luna role/label request; manual charts and local profiling work without a key.

Data preparation and calculations run locally. Enabled AI features send bounded dataset context to OpenAI; an API key also enables background summary generation. Dataset sessions use memory or temporary disk storage and cannot be reopened after a backend restart. Run the backend on localhost.

## Development checks

```bash
# Backend: from backend/ with the virtual environment active
pip install --require-hashes --only-binary=:all: -r requirements-dev.txt
python -m unittest discover -s tests -v

# Frontend: from frontend/
npx tsc --noEmit
npm run test:helpers
npm run lint
npm run build
```
