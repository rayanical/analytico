# Analytico

Analytico is a local-first analytics application that takes users from CSV exploration to exportable reporting. Manual analytics work without an AI key; optional AI features send selected dataset context to the configured provider.

- **Explore Mode:** upload data, ask questions in natural language, or build charts manually.
- **Dashboard Mode:** pin charts, drag/resize widgets on a snap grid, and assemble a report layout.
- **Export Workflow:** download chart assets (PNG/SVG) or export a multi-page PDF dashboard report.

## The Stack

- **Backend:** Python, FastAPI, Pandas
- **Frontend:** TypeScript, Next.js, Tailwind CSS
- **Visualization:** Recharts, Framer Motion, react-grid-layout
- **AI:** optional OpenAI integration (default model: GPT-4o-mini)
- **Export:** html-to-image + jsPDF

---

## Core Product Capabilities

### 1) Data Ingestion

- CSV upload with conservative parsing and profiling.
- Included Gapminder demo; optional taxi demo when its CSV is installed.
- Stable column names and conservative format detection for numeric, currency, percentage, and date fields.
- Missing observations remain null; ingestion retains the original data separately from the parsed view.

### 2) AI + Manual Charting

- Chat-to-chart flow with validated chart configs and filters.
- Both AI and manual charts use the same deterministic aggregation path. Generated Python execution is not supported.
- Manual chart builder with aggregation support:
  - `sum`, `mean`, `median`, `count`, `min`, `max`
- On-demand AI chart analysis for current view.

### 3) Filtering + Drilldown

- Structured filter operators:
  - `eq`, `gt`, `lt`, `gte`, `lte`, `contains`
- Drilldown uses full structured filter state, including AI-generated filters.
- Graceful handling of invalid AI chart configs.

### 4) Data Quality + Cleaning Transparency

- Dataset quality score with clear completeness formula.
- Missing-value breakdown and profile summaries.
- Cleaning Report view showing what was changed during ingestion.

### 5) Dashboard Command Center

- Pin charts directly from Explore.
- Drag, resize, and snap widgets on a responsive grid.
- Per-dataset dashboard persistence in local storage.
- Collapsible data summary panel to maximize dashboard space.

### 6) Export

- Chart-level exports: PNG, SVG.
- Dashboard-level export: multi-page A4 landscape PDF.
- Export capture scoped to dashboard surface to avoid UI overlay artifacts.

---

## Typical Workflow

1. Upload a CSV or load the included Gapminder demo.
2. Ask a question in chat or build a chart manually.
3. Refine with filters and drilldown.
4. Pin charts to Dashboard and arrange layout.
5. Export the dashboard as a PDF report.

---

## Project Structure

```text
analytico/
├── backend/
│   ├── main.py
│   ├── modules/
│   ├── models.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── context/
│   │   ├── lib/
│   │   └── types/
│   └── package.json
└── README.md
```

---

## Running Locally

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install --require-hashes --only-binary=:all: -r requirements.txt
uvicorn main:app --host 127.0.0.1 --reload
```

The backend lock targets Python 3.11 or newer; the tested environment is Python 3.14 on macOS. Runtime and development requirements pin all resolved packages and verify download hashes. See [dependency maintenance](docs/dependency-locking.md) for controlled upgrades and platform limitations.

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

Set `OPENAI_API_KEY` in the ignored `backend/.env` for chart planning and descriptions. Column interpretation is separately opt-in: set `COLUMN_INTERPRETER=luna` to use OpenAI, or `COLUMN_INTERPRETER=jev` with `AI_GATEWAY_API_KEY` to use Jev through Vercel Gateway. Its default is `off`. See [column interpretation](docs/column-interpretation.md) for sampling limits and safety checks. With an OpenAI key configured, ingestion can still send summary context automatically; an explicit per-dataset privacy switch is pending. Upload, profiling, filtering, and manual charts remain available without it. Unsupported advanced questions return a clarification rather than running generated code.

The backend is intended to run on localhost. It has no account/authentication system; do not expose it as a public server. Dataset storage is still in memory, expires after inactivity, and is lost on restart. History and dashboard snapshots are stored in the browser, scoped to the dataset. Retaining raw data is not yet a durable workspace or backup mechanism.

Gapminder is included in the repository. The taxi demo requires the separate CSV named in `backend/core/config.py`; it is not included in a fresh download.

## Validation

Backend checks are deterministic and mock AI calls:

```bash
cd backend
pip install --require-hashes --only-binary=:all: -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Frontend checks:

```bash
cd frontend
npm run lint
npx tsc --noEmit
npm run test:helpers
npm run build
```

## Before changing AI interpretation

The synthetic interpretation benchmark defines expected column roles, units, parsing policies, aggregation recommendations, and clarification decisions. It includes ambiguous cases where guessing would be wrong. Its richer interpretation vocabulary does not add new runtime operations such as weighted rates or snapshot aggregation.

```bash
python3 backend/evals/evaluate_interpretation.py --self-check
python3 backend/evals/evaluate_interpretation.py --predictions backend/evals/baseline_current_repo.json
```

See [benchmark instructions](docs/interpretation-benchmark.md) for scoring future Jev/Luna predictions and regenerating the offline baseline. See [dependency audit](docs/dependency-audit-2026-10-01.md) for package changes and residual findings. The [original repository audit](docs/production-readiness-audit-2026-10-01.md) is a historical pre-fix document; [completed fixes](docs/local-safety-fixes-2026-10-01.md) records the verified implementation.
